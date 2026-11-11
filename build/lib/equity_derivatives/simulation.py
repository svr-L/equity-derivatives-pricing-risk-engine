"""Joint Heston simulation: Full Truncation Euler and Andersen QE-M.

QE-M matches the first two CIR conditional variance moments and uses
Andersen's conditional-stock martingale correction. For xi=0 the QE-M
branch uses exact deterministic integrated variance. Both schemes remain
time-discretized for nonconstant stochastic variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr
from numpy.typing import NDArray

from .models import EquityMarket, HestonParams, MCConfig


@dataclass(frozen=True)
class NormalShocks:
    z_variance: NDArray[np.float64]
    z_independent: NDArray[np.float64]


@dataclass(frozen=True)
class HestonPaths:
    spot: NDArray[np.float64]  # shape: (n_steps+1, n_paths)
    variance: NDArray[np.float64]  # positive part of raw Euler variances
    antithetic: bool


def sample_shocks(config: MCConfig) -> NormalShocks:
    """The reflected paths are adjacent: (0,1), (2,3), etc."""
    rng = np.random.default_rng(config.seed)
    if config.antithetic:
        shape = (config.n_steps, config.n_paths // 2)
        zv = rng.standard_normal(shape)
        zi = rng.standard_normal(shape)
        variance = np.empty((config.n_steps, config.n_paths))
        independent = np.empty_like(variance)
        variance[:, 0::2] = zv
        variance[:, 1::2] = -zv
        independent[:, 0::2] = zi
        independent[:, 1::2] = -zi
        return NormalShocks(variance, independent)
    shape = (config.n_steps, config.n_paths)
    return NormalShocks(rng.standard_normal(shape), rng.standard_normal(shape))


def _qe_m_step(v: NDArray[np.float64], zv: NDArray[np.float64],
               zi: NDArray[np.float64], params: HestonParams, dt: float,
               drift: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (next nonnegative variance, log-spot increment), Andersen QE-M.

    gamma_1 = gamma_2 = 1/2. The distribution of v_{t+dt} matches the
    CIR conditional mean and variance. The log-stock correction subtracts
    the conditional log-moment-generating function so E[S_next|v,S] =
    S exp((r-q)*dt) at each step, if the relevant exponential moment exists.
    """
    k, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    decay = np.exp(-k * dt)
    one_minus_decay = -np.expm1(-k * dt)
    m = theta + (v - theta) * decay
    s2 = (v * xi**2 * decay * one_minus_decay / k
          + theta * xi**2 * one_minus_decay**2 / (2 * k))
    psi = np.divide(s2, m*m, out=np.zeros_like(v), where=m > 0)
    out = np.zeros_like(v)
    log_mgf = np.zeros_like(v)
    k1 = .5 * dt * (k * rho / xi - .5) - rho / xi
    k2 = .5 * dt * (k * rho / xi - .5) + rho / xi
    k3 = .5 * dt * (1. - rho**2)
    k4 = .5 * dt * (1. - rho**2)
    A = k2 + .5 * k4

    # Degenerate/zero CIR variance and deterministic conditional moments.
    deterministic = psi < 1e-12
    out[deterministic] = m[deterministic]
    log_mgf[deterministic] = A * m[deterministic]

    quadratic = (~deterministic) & (psi <= 1.5)
    if np.any(quadratic):
        ps = psi[quadratic]
        inv = 2. / ps
        b2 = inv - 1. + np.sqrt(inv * (inv - 1.))
        a = m[quadratic] / (1. + b2)
        out[quadratic] = a * (np.sqrt(b2) + zv[quadratic])**2
        denom = 1. - 2. * A * a
        if np.any(denom <= 0):
            raise ValueError("QE-M nonfinite conditional exponential moment (quadratic)")
        log_mgf[quadratic] = (-.5 * np.log(denom)
                              + A * a * b2 / denom)

    exponential = (~deterministic) & (~quadratic)
    if np.any(exponential):
        ps = psi[exponential]
        p = (ps - 1.) / (ps + 1.)
        beta = (1. - p) / m[exponential]
        if np.any(beta <= A):
            raise ValueError("QE-M nonfinite conditional exponential moment (exponential)")
        # Gaussian-to-uniform preserves deterministic replay and reflected
        # antithetic shocks. The exponential branch has an atom at zero.
        u = np.clip(ndtr(zv[exponential]),
                    np.finfo(float).tiny, np.nextafter(1., 0.))
        positive = u > p
        vals = np.zeros_like(u)
        vals[positive] = (np.log1p(-p[positive]) - np.log1p(-u[positive])) / beta[positive]
        out[exponential] = vals
        log_mgf[exponential] = np.log(p + (1. - p)*beta/(beta-A))

    # Conditional martingale correction (unlike uncorrected QE discretization).
    var_conditional = np.maximum(k3*v + k4*out, 0.)
    increment = (drift*dt + k1*v + k2*out - (k1+.5*k3)*v
                 - log_mgf + np.sqrt(var_conditional)*zi)
    return out, increment


def simulate_heston(market: EquityMarket, params: HestonParams, config: MCConfig,
                    shocks: NormalShocks | None = None) -> HestonPaths:
    """Simulate Heston grid under config.scheme using compatible shock shapes.

    Full Truncation Euler is the v0.1 default and remains unchanged. QE-M uses
    z_variance to sample its CIR moment-matched transition and z_independent
    for the conditional stock Gaussian. Parameters/shocks can be reused with
    common-random-number bump estimates.
    """
    if shocks is None:
        shocks = sample_shocks(config)
    shape = (config.n_steps, config.n_paths)
    if shocks.z_variance.shape != shape or shocks.z_independent.shape != shape:
        raise ValueError(f"shocks must both have shape {shape}")
    if not np.isfinite(shocks.z_variance).all() or not np.isfinite(shocks.z_independent).all():
        raise ValueError("shocks must be finite")
    dt = config.maturity / config.n_steps
    root_dt = np.sqrt(dt)
    spots = np.empty((config.n_steps + 1, config.n_paths), dtype=float)
    variances = np.empty_like(spots)
    spots[0, :] = market.spot
    variances[0, :] = params.v0
    raw_v = np.full(config.n_paths, params.v0)
    log_s = np.full(config.n_paths, np.log(market.spot))
    root_one_minus_rho2 = np.sqrt(max(0., 1. - params.rho**2))
    drift = market.rate - market.dividend_yield
    for j in range(config.n_steps):
        z_v = shocks.z_variance[j]
        z_i = shocks.z_independent[j]
        if config.scheme == "full_truncation_euler":
            vp = np.maximum(raw_v, 0.)
            z_s = params.rho * z_v + root_one_minus_rho2 * z_i
            log_s += (drift - .5*vp)*dt + np.sqrt(vp)*root_dt*z_s
            raw_v += (params.kappa*(params.theta-vp)*dt
                      + params.xi*np.sqrt(vp)*root_dt*z_v)
            v_out = np.maximum(raw_v, 0.)
        elif params.xi == 0:
            # Exact deterministic mean-reverting variance and integrated v.
            vi = raw_v
            v_out = params.theta + (vi - params.theta)*np.exp(-params.kappa*dt)
            integrated = (params.theta*dt + (vi-params.theta)
                          * (-np.expm1(-params.kappa*dt))/params.kappa)
            z_s = params.rho*z_v + root_one_minus_rho2*z_i
            log_s += drift*dt - .5*integrated + np.sqrt(np.maximum(integrated, 0.))*z_s
            raw_v = v_out
        else:
            v_out, log_increment = _qe_m_step(raw_v, z_v, z_i, params, dt, drift)
            log_s += log_increment
            raw_v = v_out
        spots[j+1] = np.exp(log_s)
        variances[j+1] = v_out
    if not np.isfinite(spots).all() or not np.isfinite(variances).all() or np.any(spots<=0):
        raise FloatingPointError("nonfinite simulated path: consider smaller time steps")
    return HestonPaths(spots, variances, config.antithetic)

"""Correlated Heston full-truncation Euler, vectorized across paths.

The scheme is a numerical approximation, not an exact Heston simulation.
Variance may be negative internally; *positive part* enters its drift,
volatility and spot dynamics. The returned variance path reports positive
parts, whereas the internal raw variance is retained for time stepping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def simulate_heston(market: EquityMarket, params: HestonParams, config: MCConfig,
                    shocks: NormalShocks | None = None) -> HestonPaths:
    """Simulate joint Q-dynamics with *pre-step* variance and correlation rho.

    Supplying the same shocks allows common-random-number bumps for Greeks.
    All monitoring times are uniform and include t=0 and maturity.
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
        vp = np.maximum(raw_v, 0.)
        z_v = shocks.z_variance[j]
        z_s = params.rho * z_v + root_one_minus_rho2 * shocks.z_independent[j]
        log_s = log_s + (drift - 0.5 * vp) * dt + np.sqrt(vp) * root_dt * z_s
        raw_v = (raw_v + params.kappa * (params.theta - vp) * dt
                 + params.xi * np.sqrt(vp) * root_dt * z_v)
        spots[j + 1] = np.exp(log_s)
        variances[j + 1] = np.maximum(raw_v, 0.)
    return HestonPaths(spots, variances, config.antithetic)

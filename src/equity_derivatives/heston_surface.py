"""Batch Heston European vanilla pricing using fixed Gauss-Legendre quadrature.

Same Lewis representation as :mod:`heston_fourier`, but shared characteristic
function evaluations across all strikes at each maturity. This is suitable for
calibration; numerical truncation/integration parameters require convergence
checks, especially for very short expiries or extreme strikes.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from .models import EquityMarket, HestonParams
from .vanilla import black_scholes


@lru_cache(maxsize=16)
def _quadrature(n_nodes: int, upper: float) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes < 16:
        raise ValueError("n_nodes must be an integer >= 16")
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError("integration_upper must be positive and finite")
    x, w = leggauss(n_nodes)
    u = (x + 1.0) * upper / 2.0
    return u, w * upper / 2.0


def _heston_cf_array(u: np.ndarray, params: HestonParams, t: float,
                     r: float, q: float) -> np.ndarray:
    """Vectorized little-trap CF, independently checked against scalar CF."""
    k, theta, xi, rho, v0 = (
        params.kappa, params.theta, params.xi, params.rho, params.v0)
    if xi == 0:
        var_integral = theta * t + (v0-theta)*(-math.expm1(-k*t))/k
        return np.exp(1j*u*(r-q)*t - 0.5*var_integral*(u*u+1j*u))
    a = k-rho*xi*1j*u
    d = np.sqrt(a*a + xi*xi*(u*u+1j*u))
    d = np.where(d.real < 0, -d, d)
    g = (a-d)/(a+d)
    e = np.exp(-d*t)
    c = (1j*u*(r-q)*t + k*theta/(xi*xi)
         * ((a-d)*t - 2*np.log((1-g*e)/(1-g))))
    dd = (a-d)/(xi*xi) * (1-e)/(1-g*e)
    return np.exp(c+dd*v0)


def heston_surface_prices(market: EquityMarket, params: HestonParams,
                          strikes: np.ndarray, maturities: np.ndarray,
                          is_calls: np.ndarray | None = None, *,
                          n_nodes: int = 96, integration_upper: float = 180.) -> np.ndarray:
    """Batch discounted European calls/puts under flat rates/dividends.

    Inputs broadcast to a common shape, outputs preserve it. The Fourier
    integral is shared among all strikes of a given maturity.  No silent
    clipping of negative prices: callers must check no-arbitrage constraints.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    if is_calls is None:
        is_calls = np.ones_like(strikes, dtype=bool)
    else:
        is_calls = np.asarray(is_calls, dtype=bool)
    strikes, maturities, is_calls = np.broadcast_arrays(strikes, maturities, is_calls)
    if np.any(~np.isfinite(strikes)) or np.any(strikes <= 0):
        raise ValueError("strikes must be positive and finite")
    if np.any(~np.isfinite(maturities)) or np.any(maturities < 0):
        raise ValueError("maturities must be finite and nonnegative")
    u, weights = _quadrature(n_nodes, float(integration_upper))
    result = np.empty(strikes.size, dtype=float)
    kflat, tflat, calls = strikes.ravel(), maturities.ravel(), is_calls.ravel()
    for t in np.unique(tflat):
        idx = np.flatnonzero(tflat == t)
        k = kflat[idx]
        if t == 0:
            result[idx] = np.where(calls[idx], np.maximum(market.spot-k, 0),
                                   np.maximum(k-market.spot, 0))
            continue
        if params.xi == 0:
            intvar = params.theta*t + (params.v0-params.theta)*(
                -math.expm1(-params.kappa*t))/params.kappa
            result[idx] = [black_scholes(market, float(s), float(t),
                                         math.sqrt(max(0., intvar)/t), bool(ic))
                           for s, ic in zip(k, calls[idx])]
            continue
        cf = _heston_cf_array(u - 0.5j, params, float(t), market.rate, market.dividend_yield)
        base = cf*weights/(u*u + 0.25)
        lm = np.log(market.spot/k)
        integral = (np.exp(1j*lm[:, None]*u[None, :]) * base[None, :]).real.sum(axis=1)
        call = (market.spot*math.exp(-market.dividend_yield*t)
                - math.exp(-market.rate*t)*np.sqrt(market.spot*k)*integral/math.pi)
        put = call - market.spot*math.exp(-market.dividend_yield*t) + k*math.exp(-market.rate*t)
        result[idx] = np.where(calls[idx], call, put)
    return result.reshape(strikes.shape)

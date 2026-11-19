"""Batch Heston European vanilla pricing for calibration.

Same Lewis representation and numerics as :mod:`heston_fourier` (stable
characteristic function, composite Gauss-Legendre panels, automatic
truncation, :class:`~equity_derivatives.heston_fourier.FourierAccuracyWarning`).
The quadrature grid and characteristic-function evaluations are shared by all
strikes of a given maturity.
"""
from __future__ import annotations

import math

import numpy as np

from .carry import EquityCarryCurves
from .heston_fourier import (  # noqa: F401  (_heston_cf_array re-exported)
    _heston_cf_array,
    _lewis_call_prices,
    _validate_numerics,
    _warn_if_inaccurate,
)
from .models import EquityMarket, HestonParams
from .vanilla import black_scholes


def heston_surface_prices(market: EquityMarket, params: HestonParams,
                          strikes: np.ndarray, maturities: np.ndarray,
                          is_calls: np.ndarray | None = None, *,
                          n_nodes: int = 16, integration_upper: float | None = None,
                          carry_curves: EquityCarryCurves | None = None,
                          rel_tol: float = 1e-10) -> np.ndarray:
    """Batch discounted European calls/puts under flat rates/dividends or dated curves.

    Inputs broadcast to a common shape, outputs preserve it. ``n_nodes`` is the
    number of Gauss-Legendre nodes per quadrature panel; ``integration_upper``
    None (default) selects the truncation per maturity from the decay of the
    characteristic function (target error ``rel_tol * spot``), a float fixes it.
    No silent clipping of negative prices: suspicious results raise a
    ``FourierAccuracyWarning`` instead.
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
    _validate_numerics(n_nodes, integration_upper, rel_tol)
    if carry_curves is not None:
        carry_curves.validate_market(market)
    result = np.empty(strikes.size, dtype=float)
    kflat, tflat, calls = strikes.ravel(), maturities.ravel(), is_calls.ravel()
    fourier = []  # (index array, maturity, effective market) for xi > 0 and t > 0
    for t in np.unique(tflat):
        idx = np.flatnonzero(tflat == t)
        k = kflat[idx]
        mkt = (carry_curves.effective_market(float(t)) if carry_curves is not None and t > 0
               else market)
        if t == 0:
            result[idx] = np.where(calls[idx], np.maximum(market.spot-k, 0),
                                   np.maximum(k-market.spot, 0))
            continue
        if params.xi == 0:
            intvar = params.theta*t + (params.v0-params.theta)*(
                -math.expm1(-params.kappa*t))/params.kappa
            result[idx] = [black_scholes(mkt, float(s), float(t),
                                         math.sqrt(max(0., intvar)/t), bool(ic))
                           for s, ic in zip(k, calls[idx], strict=True)]
            continue
        fourier.append((idx, float(t), mkt))
    if fourier:
        call_groups, diagnostics = _lewis_call_prices(
            [mkt for _, _, mkt in fourier], [t for _, t, _ in fourier],
            [kflat[idx] for idx, _, _ in fourier], params,
            n_nodes=n_nodes, integration_upper=integration_upper, rel_tol=rel_tol)
        for (idx, t, mkt), call in zip(fourier, call_groups, strict=True):
            k = kflat[idx]
            put = call - mkt.spot*math.exp(-mkt.dividend_yield*t) + k*math.exp(-mkt.rate*t)
            result[idx] = np.where(calls[idx], call, put)
        _warn_if_inaccurate(diagnostics, params, stacklevel=3)
    return result.reshape(strikes.shape)

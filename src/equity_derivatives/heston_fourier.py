"""Semi-analytic Heston European vanilla prices by the Lewis Fourier integral.

Numerics (v0.4.1)
-----------------
* Characteristic function in the "little trap" orientation, rewritten so that
  ``(a-d)/xi**2`` and ``log((1-g*exp(-d*T))/(1-g))/xi**2`` are evaluated
  without cancellation. It is accurate for small vol-of-vol and continuous
  with the exact deterministic-variance case ``xi == 0``.
* Composite Gauss-Legendre quadrature on panels that start small at ``u = 0``
  (resolving the ``1/(u**2 + 1/4)`` peak of the Lewis integrand) and whose
  width is capped by the phase rate of the integrand.
* Automatic truncation from the observed decay of ``|phi(u - i/2)|``.
  A :class:`FourierAccuracyWarning` is issued when the estimated truncation
  error, or a violation of the static no-arbitrage bounds, exceeds
  ``1000 * rel_tol * spot``. Non-finite values raise ``FloatingPointError``.

Flat ``r`` and ``q`` are used directly; dated carry curves enter through their
maturity-equivalent flat rates, which is exact for European payoffs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss

from .carry import EquityCarryCurves
from .models import EquityMarket, HestonParams
from .vanilla import black_scholes

#: Largest automatically chosen Fourier truncation point.
MAX_AUTO_INTEGRATION_UPPER = 2.0**16
#: Width of the first quadrature panel [0, FIRST_PANEL_WIDTH].
FIRST_PANEL_WIDTH = 0.5
#: Maximum integrand phase change (radians) allowed across one panel.
PANEL_PHASE_BUDGET = 10.0
_TAIL_GRID = 2.0 ** (np.arange(0, 2 * 16 + 1) / 2.0)  # 1 .. 2**16, step sqrt(2)
_MIN_AUTO_UPPER = 8.0
_WARN_FACTOR = 1e3


class FourierAccuracyWarning(RuntimeWarning):
    """A Heston Fourier price may be inaccurate (truncation or bound check)."""


def _log1p_complex(w: np.ndarray) -> np.ndarray:
    """Principal ``log(1 + w)``, accurate for small ``|w|``.

    ``numpy.log1p`` is not accurate for small complex arguments.
    """
    x, y = w.real, w.imag
    with np.errstate(invalid="ignore", divide="ignore"):
        accurate = 0.5 * np.log1p(x * (2.0 + x) + y * y) + 1j * np.arctan2(y, 1.0 + x)
        direct = np.log(1.0 + w)
    return np.where(np.abs(w) < 0.5, accurate, direct)


def _log1p_over_w(w: np.ndarray) -> np.ndarray:
    """``log(1 + w) / w`` with the removable singularity at ``w = 0``."""
    tiny = np.abs(w) < 1e-4
    with np.errstate(invalid="ignore", divide="ignore"):
        direct = _log1p_complex(w) / np.where(tiny, 1.0, w)
    series = 1.0 - w * (0.5 - w * (1.0 / 3.0 - 0.25 * w))
    return np.where(tiny, series, direct)


def _heston_log_cf(z, params: HestonParams, t, r, q) -> np.ndarray:
    """``log E_Q[exp(i*z*log(S_t/S_0))]``; ``z``, ``t``, ``r``, ``q`` broadcast.

    Same function as the classic little-trap formula, but ``s = a + d`` and
    ``m = a - d`` are taken from the product ``s*m = -xi**2*(z**2 + i*z)``
    whichever of the two would cancel, and the logarithm is evaluated as a
    ``log1p``. No division by ``xi**2`` remains in terms that vanish as
    ``xi -> 0``. ``t`` must be > 0.
    """
    z = np.asarray(z, dtype=complex)
    t = np.asarray(t, dtype=float)
    k, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0
    zz = z * z + 1j * z
    drift_term = 1j * z * (np.asarray(r, dtype=float) - np.asarray(q, dtype=float)) * t
    if xi == 0:
        int_var = theta * t + (v0 - theta) * (-np.expm1(-k * t)) / k
        return drift_term - 0.5 * int_var * zz
    a = k - rho * xi * 1j * z
    d = np.sqrt(a * a + xi * xi * zz)  # principal branch: Re(d) >= 0
    s0, m0 = a + d, a - d
    use_s = np.abs(s0) >= np.abs(m0)
    e = np.exp(-d * t)
    one_minus_e = -np.expm1(-d * t)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(use_s, s0, -xi * xi * zz / m0)
        m_over_xi2 = np.where(use_s, -zz / s0, m0 / (xi * xi))
        g = m_over_xi2 * (xi * xi) / s
        dd = m_over_xi2 * one_minus_e / (1.0 - g * e)
        ratio_term = one_minus_e / (1.0 - g)
        w = g * ratio_term  # (1 - g*e)/(1 - g) = 1 + w
        w_over_xi2 = (m_over_xi2 / s) * ratio_term
        c = drift_term + k * theta * (m_over_xi2 * t - 2.0 * _log1p_over_w(w) * w_over_xi2)
        log_cf = c + dd * v0
    # z**2 + i*z = 0 (z = 0 or z = -i): the Riccati forcing vanishes, D = 0 and
    # log(phi) = i*z*(r-q)*t exactly. Handled explicitly because s = a + d is 0
    # there when kappa < rho*xi (0/0 in the formulas above).
    return np.where(zz == 0, drift_term, log_cf)


def _heston_cf_array(u: np.ndarray, params: HestonParams, t, r, q) -> np.ndarray:
    """Vectorized Heston log-return characteristic function."""
    return np.exp(_heston_log_cf(u, params, t, r, q))


def heston_log_return_cf(
    u: complex,
    params: HestonParams,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> complex:
    """E_Q[exp(i*u*log(S_T/S_0))], complex u (strip including Im(u)=-1/2)."""
    if maturity < 0 or not math.isfinite(maturity):
        raise ValueError("maturity must be finite and nonnegative")
    if not all(math.isfinite(z) for z in (rate, dividend_yield)):
        raise ValueError("rates must be finite")
    if maturity == 0:
        return 1.0 + 0j
    return complex(
        _heston_cf_array(
            np.asarray([u], dtype=complex), params, float(maturity), rate, dividend_yield
        )[0]
    )


@lru_cache(maxsize=16)
def _gauss_legendre(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return leggauss(n_nodes)


def _validate_numerics(n_nodes: int, integration_upper: float | None, rel_tol: float) -> None:
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes < 16:
        raise ValueError("n_nodes (Gauss-Legendre nodes per panel) must be an integer >= 16")
    if integration_upper is not None and (
        not math.isfinite(integration_upper) or integration_upper <= 0
    ):
        raise ValueError("integration_upper must be None (automatic) or positive and finite")
    if not math.isfinite(rel_tol) or not 0 < rel_tol <= 1e-3:
        raise ValueError("rel_tol must lie in (0, 1e-3]")


def _panel_nodes(upper: float, max_width: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes/weights on [0, upper].

    Panels double in width from FIRST_PANEL_WIDTH until they reach
    ``max_width``; the rest of the interval uses panels of at most ``max_width``.
    """
    edges = [0.0]
    width = FIRST_PANEL_WIDTH
    while edges[-1] < upper and width <= max_width:
        edges.append(min(edges[-1] + width, upper))
        width = edges[-1]
    if edges[-1] < upper:
        n_uniform = math.ceil((upper - edges[-1]) / max_width)
        edges.extend(np.linspace(edges[-1], upper, n_uniform + 1)[1:].tolist())
    edges_arr = np.asarray(edges)
    x, w = _gauss_legendre(n_nodes)
    lo, hi = edges_arr[:-1, None], edges_arr[1:, None]
    nodes = (lo + (hi - lo) * (x[None, :] + 1.0) / 2.0).ravel()
    weights = ((hi - lo) / 2.0 * w[None, :]).ravel()
    return nodes, weights


@dataclass(frozen=True)
class _LewisDiagnostics:
    maturity: float
    integration_upper: float
    n_nodes_total: int
    truncation_error_estimate: float
    max_bound_violation: float
    warn_tolerance: float

    @property
    def inaccurate(self) -> bool:
        return (
            self.truncation_error_estimate > self.warn_tolerance
            or self.max_bound_violation > self.warn_tolerance
        )


def _lewis_call_prices(
    markets: Sequence[EquityMarket],
    maturities: Sequence[float],
    strike_groups: Sequence[np.ndarray],
    params: HestonParams,
    *,
    n_nodes: int,
    integration_upper: float | None,
    rel_tol: float,
) -> tuple[list[np.ndarray], list[_LewisDiagnostics]]:
    """Discounted European calls, one strike group per maturity (> 0, xi > 0).

    ``markets[j]`` is the flat (maturity-equivalent) market of ``maturities[j]``;
    all share the same spot. Characteristic-function evaluations are batched
    across maturities.
    """
    spot = markets[0].spot
    t = np.asarray(maturities, dtype=float)
    r = np.asarray([mk.rate for mk in markets], dtype=float)
    q = np.asarray([mk.dividend_yield for mk in markets], dtype=float)
    n_grid = _TAIL_GRID.size
    grid = (
        _TAIL_GRID if integration_upper is None else np.append(_TAIL_GRID, float(integration_upper))
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_cf = _heston_log_cf(grid[None, :] - 0.5j, params, t[:, None], r[:, None], q[:, None])
        tail_ratio = np.exp(log_cf.real) / grid[None, :]
    if not np.all(np.isfinite(tail_ratio)):
        raise FloatingPointError("non-finite Heston characteristic function on the Lewis contour")
    grid0 = np.concatenate(([0.0], _TAIL_GRID))
    phase_rates = (
        np.abs(
            np.diff(
                np.concatenate((np.zeros((t.size, 1)), log_cf.imag[:, :n_grid]), axis=1), axis=1
            )
        )
        / np.diff(grid0)[None, :]
    )
    plans = []
    for j in range(t.size):
        lm = np.log(spot / strike_groups[j])
        df_r = math.exp(-r[j] * t[j])
        price_scale = df_r * math.sqrt(spot * float(np.max(strike_groups[j]))) / math.pi
        ratio_j = tail_ratio[j, :n_grid]
        # |integrand| <= |phi(u - i/2)|/u**2, so the tail beyond U is bounded by
        # sup_{u >= U} |phi(u - i/2)| / U (|phi| assumed eventually decreasing).
        if integration_upper is None:
            ok_from_here = np.flip(
                np.logical_and.accumulate(np.flip(ratio_j <= rel_tol * spot / price_scale))
            )
            upper = (
                max(_MIN_AUTO_UPPER, float(_TAIL_GRID[np.argmax(ok_from_here)]))
                if ok_from_here.any()
                else MAX_AUTO_INTEGRATION_UPPER
            )
            tail_sup = float(np.max(ratio_j[_TAIL_GRID >= upper]))
        else:
            upper = float(integration_upper)
            tail_sup = max(
                float(tail_ratio[j, -1]), float(np.max(ratio_j[_TAIL_GRID >= upper], initial=0.0))
            )
        # Integrand phase rate on [0, U]: strike term plus the CF phase measured
        # on the grid from the continuous imaginary part of log(phi).
        omega = float(np.max(np.abs(lm))) + 1.5 * float(np.max(phase_rates[j][grid0[:-1] < upper]))
        nodes, weights = _panel_nodes(upper, PANEL_PHASE_BUDGET / max(omega, 1e-12), n_nodes)
        plans.append((lm, df_r, price_scale * tail_sup, upper, nodes, weights))
    sizes = [p[4].size for p in plans]
    all_nodes = np.concatenate([p[4] for p in plans])
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        base_all = (
            _heston_cf_array(
                all_nodes - 0.5j,
                params,
                np.repeat(t, sizes),
                np.repeat(r, sizes),
                np.repeat(q, sizes),
            )
            * np.concatenate([p[5] for p in plans])
            / (all_nodes * all_nodes + 0.25)
        )
    calls_out, diagnostics = [], []
    start = 0
    for j, (lm, df_r, truncation_error, upper, nodes, _) in enumerate(plans):
        base = base_all[start : start + nodes.size]
        start += nodes.size
        strikes = strike_groups[j]
        integral = (np.exp(1j * lm[:, None] * nodes[None, :]) * base[None, :]).real.sum(axis=1)
        df_q = math.exp(-q[j] * t[j])
        calls = spot * df_q - df_r * np.sqrt(spot * strikes) * integral / math.pi
        if not np.all(np.isfinite(calls)):
            raise FloatingPointError("non-finite Heston Fourier price")
        lower_bound = np.maximum(spot * df_q - strikes * df_r, 0.0)
        violation = float(np.max(np.maximum(lower_bound - calls, calls - spot * df_q), initial=0.0))
        calls_out.append(calls)
        diagnostics.append(
            _LewisDiagnostics(
                float(t[j]),
                upper,
                int(nodes.size),
                truncation_error,
                violation,
                _WARN_FACTOR * rel_tol * spot,
            )
        )
    return calls_out, diagnostics


def _lewis_calls(
    market: EquityMarket,
    params: HestonParams,
    maturity: float,
    strikes: np.ndarray,
    *,
    n_nodes: int,
    integration_upper: float | None,
    rel_tol: float,
) -> tuple[np.ndarray, _LewisDiagnostics]:
    """Single-maturity convenience wrapper around :func:`_lewis_call_prices`."""
    calls, diags = _lewis_call_prices(
        [market],
        [float(maturity)],
        [np.asarray(strikes, dtype=float)],
        params,
        n_nodes=n_nodes,
        integration_upper=integration_upper,
        rel_tol=rel_tol,
    )
    return calls[0], diags[0]


def _warn_if_inaccurate(
    diagnostics: Sequence[_LewisDiagnostics], params: HestonParams, stacklevel: int
) -> None:
    bad = [d for d in diagnostics if d.inaccurate]
    if not bad:
        return
    worst = max(bad, key=lambda d: max(d.truncation_error_estimate, d.max_bound_violation))
    warnings.warn(
        f"Heston Fourier prices may be inaccurate for {params} at {len(bad)} maturity(ies); "
        f"worst at T={worst.maturity:.6g}: truncation at U={worst.integration_upper:.6g} "
        f"(estimated error {worst.truncation_error_estimate:.2e}), no-arbitrage bound "
        f"violation {worst.max_bound_violation:.2e}, warning tolerance "
        f"{worst.warn_tolerance:.2e}. Very slowly decaying characteristic function "
        f"(tiny v0 with large vol-of-vol) or a too small explicit integration_upper.",
        FourierAccuracyWarning,
        stacklevel=stacklevel,
    )


def heston_european(
    market: EquityMarket,
    params: HestonParams,
    strike: float,
    maturity: float,
    is_call: bool = True,
    *,
    integration_upper: float | None = None,
    epsabs: float | None = None,
    carry_curves: EquityCarryCurves | None = None,
    n_nodes: int = 16,
    rel_tol: float = 1e-10,
) -> float:
    """Price a European option by the Lewis integral; put from put-call parity.

    ``integration_upper=None`` (default) chooses the Fourier truncation from
    the decay of the characteristic function so that the estimated truncation
    error is below ``rel_tol * spot``; a float fixes it. ``n_nodes`` is the
    number of Gauss-Legendre nodes per panel. ``epsabs`` belonged to the
    pre-v0.4.1 adaptive quadrature and is ignored (DeprecationWarning).
    The xi=0 case uses a *closed-form* deterministic-integrated-variance BS
    benchmark instead of performing unnecessary quadrature.
    """
    if not math.isfinite(strike) or strike <= 0 or not math.isfinite(maturity) or maturity < 0:
        raise ValueError("require strike>0 and maturity>=0")
    _validate_numerics(n_nodes, integration_upper, rel_tol)
    if epsabs is not None:
        warnings.warn(
            "epsabs is ignored since v0.4.1 (composite Gauss-Legendre quadrature); use rel_tol",
            DeprecationWarning,
            stacklevel=2,
        )
    if carry_curves is not None:
        carry_curves.validate_market(market)
        if maturity > 0:
            market = carry_curves.effective_market(maturity)
    if maturity == 0:
        return max((1 if is_call else -1) * (market.spot - strike), 0.0)
    if params.xi == 0:
        int_var = (
            params.theta * maturity
            + (params.v0 - params.theta) * (-math.expm1(-params.kappa * maturity)) / params.kappa
        )
        return black_scholes(
            market, strike, maturity, math.sqrt(max(int_var, 0.0) / maturity), is_call
        )
    calls, diag = _lewis_calls(
        market,
        params,
        maturity,
        np.asarray([strike], dtype=float),
        n_nodes=n_nodes,
        integration_upper=integration_upper,
        rel_tol=rel_tol,
    )
    _warn_if_inaccurate([diag], params, stacklevel=3)
    call = float(calls[0])
    if is_call:
        return call
    return float(
        call
        - market.spot * math.exp(-market.dividend_yield * maturity)
        + strike * math.exp(-market.rate * maturity)
    )

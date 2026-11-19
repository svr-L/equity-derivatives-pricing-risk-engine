"""Robustness of the Heston Fourier pricers (v0.4.1 fixes).

Covers: stability for vanishing vol-of-vol, automatic truncation for short
expiries / large vol-of-vol, the accuracy warning, quadrature self-convergence
and scalar/batch consistency. FourierAccuracyWarning is an error in the test
suite (pyproject), so every test also asserts that none is raised unless
the test expects it.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from equity_derivatives import (
    DiscountCurve,
    EquityCarryCurves,
    EquityMarket,
    FourierAccuracyWarning,
    HestonParams,
    heston_european,
    heston_log_return_cf,
    heston_surface_prices,
)

M = EquityMarket(100.0, 0.03, 0.01)


@pytest.mark.parametrize("v0", [0.05, 0.02])
@pytest.mark.parametrize("strike,maturity", [(80.0, 0.5), (100.0, 1.0), (130.0, 2.0)])
def test_tiny_xi_converges_linearly_to_deterministic_variance_price(v0, strike, maturity):
    """v0.4.0 returned 0.84 instead of 7.36 for xi=1e-9 (cancellation in k*theta/xi**2)."""
    base = HestonParams(1.5, 0.05, 0.0, -0.7, v0)
    exact = heston_european(M, base, strike, maturity)  # closed-form xi=0 benchmark
    for xi in (1e-12, 1e-9, 1e-7, 1e-5):
        p = replace(base, xi=xi)
        scalar = heston_european(M, p, strike, maturity)
        batch = heston_surface_prices(M, p, [strike], [maturity])[0]
        # observed |dP/dxi| at xi->0 is below 5 for these cases
        assert abs(scalar - exact) <= 10 * xi + 1e-10
        assert abs(batch - exact) <= 10 * xi + 1e-10


def test_characteristic_function_continuous_at_xi_zero():
    base = HestonParams(2.0, 0.05, 0.0, -0.6, 0.03)
    for u in (0.3, 2.0 - 0.5j, 10.0 - 0.5j, 40.0 - 0.5j, -1j):
        ref = heston_log_return_cf(u, base, 1.3, 0.02, 0.01)
        for xi in (1e-10, 1e-7):
            val = heston_log_return_cf(u, replace(base, xi=xi), 1.3, 0.02, 0.01)
            assert abs(val - ref) <= 1e3 * xi * max(1.0, abs(ref))


@pytest.mark.parametrize(
    "params",
    [
        HestonParams(2.0, 0.04, 0.5, -0.7, 0.04),
        HestonParams(0.05, 0.0004, 3.0, -0.98, 0.0004),
        HestonParams(15.0, 0.5, 3.0, 0.98, 0.5),
        HestonParams(1.5, 0.06, 2.5, 0.7, 0.02),
        HestonParams(3.0, 0.04, 1e-9, -0.5, 0.09),
    ],
)
@pytest.mark.parametrize("maturity", [1 / 365, 0.5, 10.0])
def test_characteristic_function_martingale_extreme_parameters(params, maturity):
    value = heston_log_return_cf(-1j, params, maturity, 0.03, 0.01)
    assert value == pytest.approx(math.exp(0.02 * maturity), rel=1e-12)


HARD_BUT_SUPPORTED = [
    HestonParams(1.0, 0.09, 2.0, -0.8, 0.01),  # large vol-of-vol, low v0
    HestonParams(3.3418, 0.0550, 1.3483, -0.4889, 0.02323),  # SPY-like fit
    HestonParams(0.5, 0.04, 1.0, -0.9, 0.04),  # Andersen case I
    HestonParams(1.5, 0.06, 2.5, 0.7, 0.02),  # rho > 0, large vol-of-vol
    HestonParams(15.0, 0.0004, 0.01, 0.98, 0.5),  # tiny vol-of-vol, fast reversion
    HestonParams(0.05, 0.5, 3.0, -0.98, 0.5),  # slow reversion, extreme xi/rho
    HestonParams(15.0, 0.5, 0.01, -0.98, 0.0004),
]


@pytest.mark.parametrize("params", HARD_BUT_SUPPORTED)
def test_no_arbitrage_bounds_and_no_warning_from_one_day_to_ten_years(params):
    days = np.array([1, 3, 7, 21, 91, 365, 1825, 3650])
    t = np.repeat(days / 365, 9)
    fwd = M.spot * np.exp((M.rate - M.dividend_yield) * t)
    k = fwd * np.exp(np.tile(np.linspace(-0.5, 0.5, 9), days.size))
    calls = heston_surface_prices(M, params, k, t, True)
    puts = heston_surface_prices(M, params, k, t, False)
    df_r, df_q = np.exp(-M.rate * t), np.exp(-M.dividend_yield * t)
    tol = 1e-9 * M.spot
    assert np.all(calls >= np.maximum(M.spot * df_q - k * df_r, 0.0) - tol)
    assert np.all(calls <= M.spot * df_q + tol)
    assert np.all(puts >= -tol)
    np.testing.assert_allclose(calls - puts, M.spot * df_q - k * df_r, atol=1e-11 * M.spot)


def test_too_small_explicit_truncation_warns():
    """The v0.4.0 default cut-off (200) is insufficient here and is now flagged."""
    m = EquityMarket(100.0, 0.03, 0.0)
    p = HestonParams(1.0, 0.09, 2.0, -0.8, 0.01)
    with pytest.warns(FourierAccuracyWarning, match="truncation"):
        heston_european(m, p, 105.0, 7 / 365, integration_upper=200.0)
    with pytest.warns(FourierAccuracyWarning):
        heston_surface_prices(m, p, [95.0, 105.0], [7 / 365, 7 / 365], integration_upper=180.0)


def test_unreachable_tail_warns_instead_of_failing_silently():
    """Tiny v0 with xi=3: |phi| decays too slowly for the automatic cut-off cap."""
    p = HestonParams(0.05, 0.0004, 3.0, -0.98, 0.0004)
    with pytest.warns(FourierAccuracyWarning, match="U=65536"):
        heston_european(M, p, 100.0, 1 / 365)


def test_explicit_truncation_large_enough_matches_automatic():
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    auto = heston_european(M, p, 100.0, 1.0)
    fixed = heston_european(M, p, 100.0, 1.0, integration_upper=200.0)
    assert fixed == pytest.approx(auto, abs=1e-10)


@pytest.mark.parametrize("params", HARD_BUT_SUPPORTED[:5])
def test_quadrature_self_convergence(params):
    k = np.array([70.0, 95.0, 100.0, 105.0, 140.0])
    for t in (3 / 365, 0.25, 2.0):
        base = heston_surface_prices(M, params, k, t)
        fine = heston_surface_prices(M, params, k, t, n_nodes=48, rel_tol=1e-13)
        np.testing.assert_allclose(base, fine, rtol=0, atol=1e-9 * M.spot)


def test_batch_equals_scalar_with_mixed_inputs_and_curves():
    rng = np.random.default_rng(3)
    curves = EquityCarryCurves(
        100.0,
        DiscountCurve((0.25, 1.0, 3.0), (0.995, 0.97, 0.9)),
        DiscountCurve((0.5, 3.0), (0.995, 0.97)),
        asof="2026-11-19",
    )
    p = HestonParams(2.5, 0.05, 0.9, -0.6, 0.03)
    k = rng.uniform(70.0, 140.0, 30)
    t = rng.choice([7 / 365, 0.1, 0.5, 1.0, 2.5], 30)
    calls = rng.uniform(size=30) < 0.5
    batch = heston_surface_prices(M, p, k, t, calls, carry_curves=curves)
    scalar = np.array(
        [
            heston_european(M, p, float(a), float(b), bool(c), carry_curves=curves)
            for a, b, c in zip(k, t, calls, strict=True)
        ]
    )
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-10 * M.spot)


def test_epsabs_is_deprecated_and_ignored():
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    with pytest.warns(DeprecationWarning, match="epsabs"):
        value = heston_european(M, p, 100.0, 1.0, epsabs=1e-8)
    assert value == heston_european(M, p, 100.0, 1.0)


def test_invalid_numerical_settings():
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    for kwargs in (
        dict(n_nodes=8),
        dict(n_nodes=16.0),
        dict(integration_upper=-1.0),
        dict(integration_upper=math.inf),
        dict(rel_tol=0.0),
        dict(rel_tol=0.1),
    ):
        with pytest.raises(ValueError):
            heston_european(M, p, 100.0, 1.0, **kwargs)
        with pytest.raises(ValueError):
            heston_surface_prices(M, p, [100.0], [1.0], **kwargs)

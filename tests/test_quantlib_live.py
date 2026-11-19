"""Live cross-checks against QuantLib (skipped when QuantLib is not installed).

Heston prices are compared with ``AnalyticHestonEngine`` using the
AngledContour contour and exp-sinh integration. QuantLib's default
configuration (Gatheral formula + Gauss-Lobatto) is avoided on purpose: it
loses accuracy for small vol-of-vol and large kappa*T (errors up to ~2e-5 on
a 100 spot were observed against 40-digit references).
"""

import math

import numpy as np
import pytest
from equity_derivatives import (
    Barrier,
    DiscountCurve,
    EquityCarryCurves,
    EquityMarket,
    HestonParams,
    black_scholes,
    black_scholes_barrier,
    heston_european,
    heston_surface_prices,
)

ql = pytest.importorskip("QuantLib")
if not (hasattr(ql, "FourierIntegration") and hasattr(ql.AnalyticHestonEngine, "AngledContour")):
    pytest.skip(
        "QuantLib too old: no AngledContour/exp-sinh Heston engine", allow_module_level=True
    )

TODAY = ql.Date(19, 11, 2026)
DC = ql.Actual365Fixed()
CALIBRATION_BOX = (
    np.array([0.05, 0.0004, 0.01, -0.98, 0.0004]),
    np.array([15.0, 0.50, 3.0, 0.98, 0.50]),
)


@pytest.fixture(autouse=True)
def _evaluation_date():
    ql.Settings.instance().evaluationDate = TODAY


def _flat(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DC, ql.Continuous))


def _heston_engine(spot, rts, qts, p):
    process = ql.HestonProcess(
        rts, qts, ql.QuoteHandle(ql.SimpleQuote(spot)), p.v0, p.kappa, p.theta, p.xi, p.rho
    )
    return ql.AnalyticHestonEngine(
        ql.HestonModel(process),
        ql.AnalyticHestonEngine.AngledContour,
        ql.FourierIntegration.expSinh(1e-13),
    )


def _ql_vanilla(engine, strike, days, is_call):
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        ql.EuropeanExercise(TODAY + int(days)),
    )
    option.setPricingEngine(engine)
    try:
        return option.NPV()
    except RuntimeError:  # QuantLib's exp-sinh occasionally fails (NaN integrand)
        return math.nan


def test_heston_random_sweep_over_calibration_box():
    rng = np.random.default_rng(20261119)
    spot, r, q = 100.0, 0.03, 0.01
    days = np.array([1, 7, 21, 91, 365, 1825])
    lms = np.array([-0.4, -0.15, -0.05, 0.0, 0.05, 0.15, 0.4])
    compared = 0
    worst = 0.0
    for _ in range(25):
        low, high = CALIBRATION_BOX
        p = HestonParams(*(low + rng.uniform(size=5) * (high - low)))
        engine = _heston_engine(spot, _flat(r), _flat(q), p)
        t = np.repeat(days / 365, lms.size)
        k = spot * np.exp((r - q) * t) * np.exp(np.tile(lms, days.size))
        is_call = np.tile(lms >= 0, days.size)
        ours = heston_surface_prices(EquityMarket(spot, r, q), p, k, t, is_call)
        ref = np.array(
            [
                _ql_vanilla(engine, float(a), round(b * 365), bool(c))
                for a, b, c in zip(k, t, is_call, strict=True)
            ]
        )
        ok = np.isfinite(ref)
        compared += ok.sum()
        worst = max(worst, float(np.max(np.abs(ours[ok] - ref[ok]))))
    assert compared > 0.95 * 25 * days.size * lms.size
    assert worst < 1e-10 * spot


@pytest.mark.parametrize(
    "p",
    [
        HestonParams(2.0, 0.04, 0.5, -0.7, 0.04),
        HestonParams(1.0, 0.09, 2.0, -0.8, 0.01),
        HestonParams(15.0, 0.0004, 0.01, 0.98, 0.5),
    ],
)
def test_heston_scalar_pricer_vs_quantlib(p):
    spot, r, q = 100.0, 0.02, 0.03
    engine = _heston_engine(spot, _flat(r), _flat(q), p)
    for days in (3, 30, 400):
        for strike in (80.0, 100.0, 125.0):
            for is_call in (True, False):
                ref = _ql_vanilla(engine, strike, days, is_call)
                ours = heston_european(EquityMarket(spot, r, q), p, strike, days / 365, is_call)
                assert ours == pytest.approx(ref, abs=1e-10 * spot)


def test_heston_with_dated_curves_vs_quantlib_term_structures():
    """Log-linear discount-factor curves on both sides; maturities between nodes."""
    spot = 100.0
    nodes = (30, 91, 182, 365, 730, 1095)
    dfs = (0.997, 0.99, 0.981, 0.963, 0.925, 0.89)
    qfs = (0.999, 0.997, 0.994, 0.988, 0.976, 0.965)
    curves = EquityCarryCurves(
        spot,
        DiscountCurve(tuple(d / 365 for d in nodes), dfs),
        DiscountCurve(tuple(d / 365 for d in nodes), qfs),
        asof="2026-11-19",
    )
    dates = [TODAY] + [TODAY + d for d in nodes]
    rts = ql.YieldTermStructureHandle(ql.DiscountCurve(dates, [1.0, *dfs], DC))
    qts = ql.YieldTermStructureHandle(ql.DiscountCurve(dates, [1.0, *qfs], DC))
    p = HestonParams(2.2, 0.05, 0.8, -0.65, 0.035)
    engine = _heston_engine(spot, rts, qts, p)
    days = np.array([45, 200, 500, 1000])
    k = np.array([85.0, 100.0, 115.0])
    t = np.repeat(days / 365, k.size)
    kk = np.tile(k, days.size)
    ours = heston_surface_prices(EquityMarket(spot), p, kk, t, carry_curves=curves)
    ref = np.array(
        [_ql_vanilla(engine, float(a), round(b * 365), True) for a, b in zip(kk, t, strict=True)]
    )
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-10 * spot)


def test_black_scholes_vanilla_and_barrier_random_sweep():
    rng = np.random.default_rng(7)
    types = {
        ("up", "out"): ql.Barrier.UpOut,
        ("up", "in"): ql.Barrier.UpIn,
        ("down", "out"): ql.Barrier.DownOut,
        ("down", "in"): ql.Barrier.DownIn,
    }
    for _ in range(120):
        spot = 100.0
        r, q = rng.uniform(-0.01, 0.06), rng.uniform(0.0, 0.04)
        vol = rng.uniform(0.08, 0.6)
        days = int(rng.integers(10, 1500))
        strike = float(rng.uniform(60.0, 150.0))
        direction = "up" if rng.uniform() < 0.5 else "down"
        knock = "out" if rng.uniform() < 0.5 else "in"
        barrier = float(rng.uniform(102.0, 160.0) if direction == "up" else rng.uniform(50.0, 98.0))
        is_call = bool(rng.uniform() < 0.5)
        vts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), vol, DC))
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)), _flat(q), _flat(r), vts
        )
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike)
        exercise = ql.EuropeanExercise(TODAY + days)
        option = ql.BarrierOption(types[(direction, knock)], barrier, 0.0, payoff, exercise)
        option.setPricingEngine(ql.AnalyticBarrierEngine(process))
        market = EquityMarket(spot, r, q)
        ours = black_scholes_barrier(
            market, Barrier(strike, barrier, direction, knock, is_call), days / 365, vol
        )
        assert ours == pytest.approx(option.NPV(), abs=1e-10 * spot)
        vanilla = ql.VanillaOption(payoff, exercise)
        vanilla.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        assert black_scholes(market, strike, days / 365, vol, is_call) == pytest.approx(
            vanilla.NPV(), abs=1e-10 * spot
        )

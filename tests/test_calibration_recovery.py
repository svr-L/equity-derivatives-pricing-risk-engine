"""Synthetic recovery tests that do NOT start the optimizer at the true
parameters (some v0.3/v0.4 tests passed ``initial=truth``, which makes
recovery trivial).

Quotes are generated with the scalar pricer at tight tolerance (validated
against mpmath/QuantLib, see test_reference_values.py) and fitted with the
batch pricer at default settings. FourierAccuracyWarning is an error in the
test suite, so these runs also check that no inaccurate price is used.
"""

import numpy as np
import pytest
from equity_derivatives import (
    CalibrationConfig,
    DatedCalibrationSlice,
    DiscountCurve,
    EquityCarryCurves,
    EquityMarket,
    HestonParams,
    VanillaQuote,
    calibrate_heston,
    calibrate_history,
    heston_european,
)

DEFAULT_START = HestonParams(2.0, 0.04, 0.5, -0.6, 0.04)  # calibrate_heston's own default


def _quotes(market, truth, maturities, log_moneyness, carry_curves=None):
    quotes = []
    for t in maturities:
        fwd = carry_curves.forward(t) if carry_curves is not None else market.forward(t)
        for x in log_moneyness:
            k = float(fwd * np.exp(x))
            is_call = k >= fwd
            px = heston_european(
                market, truth, k, t, is_call, carry_curves=carry_curves, n_nodes=32, rel_tol=1e-13
            )
            quotes.append(VanillaQuote(k, t, px, is_call, bid=0.995 * px, ask=1.005 * px))
    return quotes


def _assert_close(fit, truth, tol):
    for name, err in tol.items():
        assert abs(getattr(fit, name) - getattr(truth, name)) < err, name


def test_recovers_distant_truth_from_default_start():
    market = EquityMarket(100.0, 0.03, 0.01)
    truth = HestonParams(4.0, 0.09, 0.9, -0.45, 0.06)
    quotes = _quotes(market, truth, (0.1, 0.25, 0.5, 1.0, 2.0), (-0.2, -0.1, 0.0, 0.1, 0.2))
    res = calibrate_heston(market, quotes, config=CalibrationConfig(n_starts=2))
    assert res.starts[0].start == tuple(vars(DEFAULT_START).values())
    assert res.success
    assert res.market_rmse_iv < 1e-6
    assert res.bid_ask_coverage == 1.0
    _assert_close(res.params, truth, dict(kappa=0.02, theta=2e-4, xi=0.005, rho=0.002, v0=1e-4))


def test_recovers_with_dated_curves_from_default_start():
    market = EquityMarket(100.0)
    curves = EquityCarryCurves(
        100.0,
        DiscountCurve((0.25, 1.0, 2.5), (0.992, 0.965, 0.91)),
        DiscountCurve((0.5, 2.5), (0.993, 0.962)),
        asof="2026-11-19",
    )
    truth = HestonParams(1.2, 0.06, 0.7, -0.75, 0.03)
    quotes = _quotes(
        market, truth, (0.1, 0.4, 1.0, 2.0), (-0.15, -0.05, 0.05, 0.15), carry_curves=curves
    )
    res = calibrate_heston(
        market, quotes, config=CalibrationConfig(n_starts=2), carry_curves=curves
    )
    assert res.market_rmse_iv < 1e-6
    assert "dated user-supplied" in res.assumptions
    _assert_close(res.params, truth, dict(kappa=0.02, theta=3e-4, xi=0.005, rho=0.002, v0=1e-4))


def test_history_warm_start_tracks_changing_parameters_without_truth_start():
    days = ("2026-11-16", "2026-11-17", "2026-11-18")
    truths = (
        HestonParams(3.0, 0.05, 0.8, -0.6, 0.03),
        HestonParams(3.2, 0.052, 0.82, -0.62, 0.036),
        HestonParams(3.1, 0.055, 0.78, -0.58, 0.041),
    )
    slices = []
    for day, truth, spot in zip(days, truths, (100.0, 101.5, 99.0), strict=True):
        curves = EquityCarryCurves(
            spot, DiscountCurve.flat(0.03, 2.5), DiscountCurve.flat(0.012, 2.5), asof=day
        )
        market = EquityMarket(spot)
        quotes = tuple(
            _quotes(
                market,
                truth,
                (0.15, 0.5, 1.0, 2.0),
                (-0.15, -0.05, 0.05, 0.15),
                carry_curves=curves,
            )
        )
        slices.append(DatedCalibrationSlice(day, "SYN", market, curves, quotes))
    history = calibrate_history(slices, config=CalibrationConfig(n_starts=1))
    assert [h.asof for h in history] == list(days)
    for h, truth in zip(history, truths, strict=True):
        assert h.calibration.market_rmse_iv < 1e-6
        _assert_close(
            h.calibration.params, truth, dict(kappa=0.03, theta=3e-4, xi=0.005, rho=0.002, v0=1e-4)
        )
    # parameters really changed, so the next-date transfer error is not ~0
    assert history[0].next_date_cross_section.rmse_iv > 1e-4


@pytest.mark.parametrize("objective", ["price", "iv"])
def test_other_objectives_recover_from_default_start(objective):
    market = EquityMarket(100.0, 0.02, 0.0)
    truth = HestonParams(2.8, 0.07, 0.65, -0.7, 0.05)
    quotes = _quotes(market, truth, (0.25, 0.75, 1.5), (-0.15, -0.05, 0.05, 0.15))
    res = calibrate_heston(
        market, quotes, config=CalibrationConfig(objective=objective, n_starts=1)
    )
    assert res.market_rmse_iv < 1e-5
    _assert_close(res.params, truth, dict(kappa=0.05, theta=5e-4, xi=0.01, rho=0.005, v0=2e-4))

"""Calibrated Heston regression suite, including synthetic parameter recovery."""
import numpy as np
import pytest

from equity_derivatives import (
    CalibrationConfig, EquityMarket, HestonParams, VanillaQuote,
    calibrate_heston, evaluate_heston_fit, heston_european,
    heston_surface_prices,
)


@pytest.fixture(scope="module")
def synthetic_surface():
    market = EquityMarket(100., 0.02, 0.01)
    p = HestonParams(2.5, 0.05, 0.6, -0.65, 0.045)
    k = np.tile(np.array([75., 85., 95., 105., 115., 125.]), 4)
    t = np.repeat([0.25, 0.5, 1., 2.], 6)
    calls = np.array([(i % 2) == 0 for i in range(len(k))])
    prices = heston_surface_prices(market, p, k, t, calls)
    quotes = [VanillaQuote(float(a), float(b), float(c), bool(d), identifier=str(i))
              for i, (a, b, c, d) in enumerate(zip(k, t, prices, calls))]
    return market, p, k, t, calls, prices, quotes


def test_vectorized_fourier_matches_scalar(synthetic_surface):
    m, p, k, t, calls, prices, _ = synthetic_surface
    slow = np.array([heston_european(m, p, float(a), float(b), bool(c))
                     for a, b, c in zip(k, t, calls)])
    assert np.max(np.abs(prices - slow)) < 0.002


def test_input_broadcast_and_deterministic_variance():
    m = EquityMarket(100., 0.01, 0.0)
    p = HestonParams(1., .04, 0., -.5, .04)
    out = heston_surface_prices(m, p, [90., 100., 110.], .5)
    assert out.shape == (3,)
    for k, x in zip([90.,100.,110.],out):
        assert x == pytest.approx(heston_european(m,p,k,.5),abs=1e-12)
    put = heston_surface_prices(m,p,np.array([100.]),np.array([.5]),np.array([False]))
    assert put[0] == pytest.approx(heston_european(m,p,100.,.5,False),abs=1e-12)


def test_bad_batch_inputs():
    m=EquityMarket(100.)
    p=HestonParams(2.,.04,.5,-.7,.04)
    with pytest.raises(ValueError):
        heston_surface_prices(m,p,[-10.],[1.])
    with pytest.raises(ValueError):
        heston_surface_prices(m,p,[100.],[-1.])
    with pytest.raises(ValueError):
        heston_surface_prices(m,p,[100.],[1.],n_nodes=8)


def test_recover_synthetic_parameters_multistart(synthetic_surface):
    m, true, _, _, _, _, quotes = synthetic_surface
    result = calibrate_heston(m, quotes,
                              config=CalibrationConfig(n_starts=3,max_nfev=80),
                              initial=HestonParams(1.7,.045,.45,-.5,.04))
    assert result.success
    assert result.n_quotes == 24
    assert result.n_expiries == 4
    assert result.n_starts == 3
    assert result.market_rmse_price < 1e-5
    assert result.market_rmse_iv < 2e-6
    assert len(result.starts) == 3
    assert abs(result.params.kappa-true.kappa) < .04
    assert abs(result.params.rho-true.rho) < .005
    assert result.jacobian_condition_number > 1
    holdout = evaluate_heston_fit(m,result.params,quotes[1::2])
    assert holdout.rmse_iv < .0002  # separate, higher-accuracy quadrature
    assert holdout.bid_ask_coverage is None


def test_calibration_price_objective(synthetic_surface):
    m, true, _, _, _, _, quotes = synthetic_surface
    r = calibrate_heston(m, quotes,
                         config=CalibrationConfig(objective='price', n_starts=1,max_nfev=80),
                         initial=HestonParams(2.2,.05,.59,-.6,.045))
    assert r.market_rmse_price < 1e-5


def test_calibration_iv_objective(synthetic_surface):
    m, true, _, _, _, _, quotes = synthetic_surface
    r = calibrate_heston(m, quotes,
                         config=CalibrationConfig(objective='iv', n_starts=1,max_nfev=80),
                         initial=HestonParams(2.3,.05,.58,-.6,.04))
    assert r.market_rmse_iv < 1e-5


def test_american_exercise_requires_explicit_override(synthetic_surface):
    m, _, _, _, _, _, qs = synthetic_surface
    with pytest.raises(ValueError, match='American'):
        calibrate_heston(m, qs, exercise_style='american')
    res=calibrate_heston(m, qs, exercise_style='american', allow_american_proxy=True,
                         config=CalibrationConfig(n_starts=1,max_nfev=15))
    assert 'AMERICAN QUOTES' in res.assumptions


def test_bad_quote_and_bounds(synthetic_surface):
    m, _, _, _, _, _, q = synthetic_surface
    with pytest.raises(ValueError):
        VanillaQuote(100.,1.,-1.)
    with pytest.raises(ValueError):
        VanillaQuote(100.,1.,10.,bid=11.,ask=12.)
    with pytest.raises(ValueError, match='duplicate'):
        calibrate_heston(m, [q[0]]*8)
    with pytest.raises(ValueError):
        CalibrationConfig(objective='nonsense')
    with pytest.raises(ValueError):
        CalibrationConfig(lower=(0.,.04,.01,-.98,.01))
    with pytest.raises(ValueError,match='inside'):
        calibrate_heston(m, q, initial=HestonParams(20.,.04,.5,-.5,.04),
                         config=CalibrationConfig(n_starts=1))


def test_no_arbitrage_iv_bounds_rejected(synthetic_surface):
    m, _, _, _, _, _, quotes = synthetic_surface
    bad = [VanillaQuote(100., 0.5, 1000.)] + list(quotes[:10])
    with pytest.raises(ValueError,match='arbitrage'):
        calibrate_heston(m,bad)


def test_holdout_bidask_coverage(synthetic_surface):
    m,p,k,t,cs,px,_ = synthetic_surface
    quotes = [VanillaQuote(float(a),float(b),float(c),bool(d),bid=float(c)-.05,
                           ask=float(c)+.05) for a,b,c,d in zip(k,t,px,cs)]
    out=evaluate_heston_fit(m,p,quotes)
    assert out.bid_ask_coverage == 1.0

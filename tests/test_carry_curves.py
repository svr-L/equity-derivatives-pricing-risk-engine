"""Independent release v0.4 tests: term-structure no-arbitrage and time-path carry."""
import math
from dataclasses import replace

import numpy as np
import pytest

from equity_derivatives import (
    DiscountCurve, EquityCarryCurves, EquityMarket, HestonParams, MCConfig,
    ArithmeticAsian, European, calibrate_heston, CalibrationConfig, VanillaQuote,
    heston_european, heston_surface_prices, simulate_heston, price_mc,
    bump_greeks,
)
from equity_derivatives.vanilla import black_scholes, implied_vol


def curves(r=0.025, q=0.015):
    return EquityCarryCurves(100, DiscountCurve((.4, 1., 2.),
                    tuple(math.exp(-r*t) for t in (.4, 1., 2.))),
                    DiscountCurve((.4, 1., 2.),
                    tuple(math.exp(-q*t) for t in (.4, 1., 2.))),
                    asof='2026-10-27')


def test_discount_interpolation_and_no_extrapolation():
    d = DiscountCurve((.25, 1.), (math.exp(-.01*.25), math.exp(-.03)))
    assert d(0) == 1.
    assert d(.25) == pytest.approx(math.exp(-.0025))
    assert d(.625) == pytest.approx(math.sqrt(d(.25)*d(1.)))
    assert list(d(np.array([0., .25, 1.]))) == pytest.approx([1., d(.25), d(1.)])
    for t in [-.001, 1.001, float('nan')]:
        with pytest.raises(ValueError): d(t)
    with pytest.raises(ValueError): DiscountCurve((1., .5), (.9, .8))
    with pytest.raises(ValueError): DiscountCurve((.1,), (0.,))
    assert DiscountCurve.flat(-.02, 1.)(1.) > 1


def test_forward_curve_from_explicit_observed_forwards():
    d=DiscountCurve((.5, 1.),(math.exp(-.02*.5),math.exp(-.03)))
    c=EquityCarryCurves.from_forwards(100,d,(.5,1.),(101.,102.),asof='2026-10-27')
    assert c.forward(.5)==pytest.approx(101.)
    assert c.forward(1.)==pytest.approx(102.)
    assert c.log_forward_increment(.5,1.)==pytest.approx(math.log(102./101.))
    assert c.discount(1.)==pytest.approx(math.exp(-.03))
    assert c.effective_market(1.).forward(1.) == pytest.approx(102.)
    with pytest.raises(ValueError, match='asof'): c.validate_snapshot('2026-10-28T01:00:00Z')
    c.validate_snapshot('2026-10-27T23:59:59Z')
    with pytest.raises(ValueError, match='spot'):c.validate_market(EquityMarket(101))
    with pytest.raises(ValueError):EquityCarryCurves(100,d,c.dividend_curve,'2026-10-27T13:00:00')


def test_flat_legacy_fourier_and_mc_path_equivalence():
    m=EquityMarket(100,.025,.015)
    c=curves()
    p=HestonParams(2.,.05,.7,-.6,.04)
    strikes=np.array([80.,100.,120.,95.]); maturities=np.array([.4,1.,2.,.8])
    a=heston_surface_prices(m,p,strikes,maturities,n_nodes=72)
    b=heston_surface_prices(m,p,strikes,maturities,n_nodes=72,carry_curves=c)
    assert np.allclose(a,b,rtol=1e-10,atol=1e-9)
    assert heston_european(m,p,100,1.,carry_curves=c)==pytest.approx(
        heston_european(m,p,100,1.), abs=1e-8)
    for scheme in ('full_truncation_euler','qe_m'):
        cfg=MCConfig(1.,16,200,seed=17,scheme=scheme)
        pa=simulate_heston(m,p,cfg)
        pb=simulate_heston(m,p,cfg,carry_curves=c)
        assert np.allclose(pa.spot,pb.spot,rtol=1e-11,atol=1e-10)
        a=price_mc(m,p,European(100),cfg)
        b=price_mc(m,p,European(100),cfg,carry_curves=c)
        assert a.price==pytest.approx(b.price,abs=1e-10)


def test_nonflat_carry_applies_at_each_step_to_path_dependents():
    # Zero variance => each path is deterministic S(t)=F(0,t); a crisp
    # test that catches using only the terminal effective yield for Asians.
    d=DiscountCurve((.5,1.),(math.exp(-.01*.5),math.exp(-.03)))
    q=DiscountCurve((.5,1.),(math.exp(-.02*.5),math.exp(-.01)))
    c=EquityCarryCurves(100,d,q,asof='2026-10-27')
    m=EquityMarket(100,.0,.0)
    p=HestonParams(2.,0.,0.,0.,0.)
    cfg=MCConfig(1.,2,100,seed=1,scheme='qe_m')
    paths=simulate_heston(m,p,cfg,carry_curves=c)
    assert paths.spot[1,0]==pytest.approx(c.forward(.5))
    assert paths.spot[2,0]==pytest.approx(c.forward(1.))
    payoff=ArithmeticAsian(100, is_call=True)
    value=price_mc(m,p,payoff,cfg,carry_curves=c)
    expected=c.discount(1.)*max(.5*(c.forward(.5)+c.forward(1.))-100,0.)
    assert value.price==pytest.approx(expected,abs=1e-11)
    assert value.stderr==pytest.approx(0.,abs=1e-11)


def test_deterministic_carry_same_flat_discounted_option_and_iv():
    m=EquityMarket(100,.025,.015)
    c=curves()
    p=HestonParams(2.,.04,0.,-.5,.04)
    for k,t in [(100,.4),(110,1.),(90,2.)]:
        eff=c.effective_market(t)
        assert implied_vol(black_scholes(eff,k,t,.2),eff,k,t)==pytest.approx(.2,abs=1e-10)
        assert heston_european(m,p,k,t,carry_curves=c)==pytest.approx(black_scholes(m,k,t,.2))


def test_calibration_with_explicit_curves_and_synthetic_quotes():
    m=EquityMarket(100,0.,0.)
    c=curves()
    truth=HestonParams(2.5,.05,.6,-.65,.045)
    ks=np.tile([90.,100.,110.],3)
    ts=np.repeat([.4,1.,2.],3)
    prices=heston_surface_prices(m,truth,ks,ts,carry_curves=c)
    quotes=[VanillaQuote(float(k),float(t),float(v)) for k,t,v in zip(ks,ts,prices)]
    fit=calibrate_heston(m,quotes,initial=truth,
              config=CalibrationConfig(n_starts=1,max_nfev=10),carry_curves=c)
    assert fit.market_rmse_price<1e-7
    assert 'dated user-supplied' in fit.assumptions


def test_greeks_with_curves_spot_bump_preserves_yield():
    m=EquityMarket(100,0.,0.)
    c=curves()
    p=HestonParams(2.,.04,0.,0.,.04)
    cfg=MCConfig(1.,5,120,scheme='qe_m')
    g=bump_greeks(m,p,European(100),cfg,carry_curves=c)
    assert np.isfinite(g.delta) and np.isfinite(g.gamma)
    assert g.price>=0
    with pytest.raises(ValueError,match='spot'):
        price_mc(replace(m,spot=101),p,European(100),cfg,carry_curves=c)


def test_no_silent_curve_extrapolation():
    m=EquityMarket(100)
    c=curves()
    p=HestonParams(2.,.04,.4,-.6,.04)
    with pytest.raises(ValueError,match='extrapolation'):
        heston_surface_prices(m,p,np.array([100.]),np.array([2.1]),carry_curves=c)
    with pytest.raises(ValueError,match='extrapolation'):
        simulate_heston(m,p,MCConfig(2.1,8,100),carry_curves=c)

import numpy as np
import pytest

from equity_derivatives import (Barrier, EquityMarket, European, HestonParams,
                                MCConfig, black_scholes_barrier, price_mc,
                                barrier_survival_probabilities, simulate_heston)


@pytest.mark.parametrize('direction,knock,is_call,h', [
    ('up','out',True,125), ('up','in',True,125),
    ('down','out',False,80), ('down','in',False,80),
])
def test_constant_vol_bridge_vs_independent_analytic_formula(direction,knock,is_call,h):
    m = EquityMarket(100, .01, .02)
    p = HestonParams(2., .04, 0., -.4, .04)
    c = MCConfig(1., 1, 80000, 26, True, 'qe_m')
    payoff = Barrier(100, h, direction, knock, is_call, 'continuous_bridge')
    result = price_mc(m,p,payoff,c)
    analytic = black_scholes_barrier(m,payoff,1.,.2)
    assert abs(result.price-analytic) < 4*result.stderr+.015


@pytest.mark.parametrize('scheme', ['qe_m','full_truncation_euler'])
def test_bridge_out_not_higher_than_discrete_and_in_out_parity(scheme):
    m = EquityMarket(100, .01, .02)
    p = HestonParams(2, .04, .5, -.7, .04)
    c = MCConfig(1,60,6000,2026,True,scheme)
    d = price_mc(m,p,Barrier(100,125),c)
    ko = price_mc(m,p,Barrier(100,125,monitoring='continuous_bridge'),c)
    ki = price_mc(m,p,Barrier(100,125,knock='in',monitoring='continuous_bridge'),c)
    vanilla = price_mc(m,p,European(100),c)
    assert ko.price <= d.price
    assert ko.price + ki.price == pytest.approx(vanilla.price, abs=1e-12)


def test_survival_weights_are_probabilities_and_start_at_barrier_is_knocked_out():
    m = EquityMarket(125)
    p = HestonParams(1.,.04,.5,-.5,.04)
    c = MCConfig(1,12,100,5,True,'qe_m')
    o = Barrier(100,125,monitoring='continuous_bridge')
    w = barrier_survival_probabilities(simulate_heston(m,p,c),p,c,o)
    assert np.all(w == 0)
    assert price_mc(m,p,o,c).price == 0.


def test_constant_vol_bridge_one_step_matches_analytic_regardless_grid_for_given_seed():
    # Exact conditional survival weights are path-dependent; prices converge
    # in expectation to the analytical benchmark irrespective of node count.
    m = EquityMarket(100,.015,0.)
    p = HestonParams(2.,.09,0.,-.5,.09)
    o = Barrier(100,125,monitoring='continuous_bridge')
    target = black_scholes_barrier(m,o,1,.3)
    for n in (1,4,16):
        c = MCConfig(1,n,60000,70,True,'qe_m')
        x = price_mc(m,p,o,c)
        assert abs(x.price-target) < 4*x.stderr + .02


def test_bridge_wrong_shapes_and_unknown_monitoring():
    with pytest.raises(ValueError,match='monitoring'):
        Barrier(100,125,monitoring='intraday')
    m=EquityMarket(100)
    p=HestonParams(2.,.04,0.,0.,.04)
    c=MCConfig(1,2,100)
    x=simulate_heston(m,p,c)
    with pytest.raises(ValueError,match='shape'):
        barrier_survival_probabilities(x,p,MCConfig(1,3,100),Barrier(100,125))

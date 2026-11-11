"""Second scheme, moments and martingale tests; separate from v0.1 regression tests."""
import math

import numpy as np
import pytest

from equity_derivatives import (EquityMarket, European, HestonParams, MCConfig,
                                heston_european, price_mc, sample_shocks,
                                simulate_heston, black_scholes, bump_greeks)


def cfg(n_steps=30, n_paths=8000, seed=1234):
    return MCConfig(1., n_steps, n_paths, seed, True, 'qe_m')


def test_scheme_validation_and_full_truncation_still_default():
    assert cfg().scheme == 'qe_m'
    assert MCConfig(1.).scheme == 'full_truncation_euler'
    with pytest.raises(ValueError, match='scheme'):
        MCConfig(1., scheme='q_e')


def test_qe_m_seed_replay_and_nonnegative_variance_even_when_feller_violated():
    m = EquityMarket(100, -.01, .03)
    p = HestonParams(1.2, .018, .9, -.85, .012)
    shocks = sample_shocks(cfg(24, 3000))
    x = simulate_heston(m, p, cfg(24, 3000), shocks)
    y = simulate_heston(m, p, cfg(24, 3000), shocks)
    np.testing.assert_array_equal(x.spot, y.spot)
    np.testing.assert_array_equal(x.variance, y.variance)
    assert np.all(x.spot > 0)
    assert np.all(x.variance >= 0)
    assert (x.variance[1:] == 0).any()


@pytest.mark.parametrize('xi', [0., .3, .7])
def test_qe_m_martingale_forward(xi):
    m = EquityMarket(100, .02, .01)
    p = HestonParams(2., .04, xi, -.7, .04)
    x = simulate_heston(m, p, cfg(32, 60000, seed=123))
    # Tolerate finite-sample noise (antithetic average of terminal prices).
    discounted = x.spot[-1]*math.exp(-m.rate)
    units = discounted.reshape(-1,2).mean(axis=1)
    se = units.std(ddof=1)/math.sqrt(len(units))
    assert abs(units.mean() - m.spot*math.exp(-m.dividend_yield)) < 4*se + .025


def test_qe_m_one_step_conditional_variance_moments():
    m = EquityMarket(100, .0)
    p = HestonParams(1.3, .04, .6, -.6, .06)
    t = 0.45
    x = simulate_heston(m, p, MCConfig(t, 1, 180000, 2026, True, 'qe_m'))
    observed = x.variance[-1]
    decay = math.exp(-p.kappa*t)
    mean = p.theta + (p.v0-p.theta)*decay
    var = (p.v0*p.xi**2*decay*(1-decay)/p.kappa
           + p.theta*p.xi**2*(1-decay)**2/(2*p.kappa))
    assert observed.mean() == pytest.approx(mean, abs=4e-4)
    assert observed.var() == pytest.approx(var, abs=6e-4)


def test_qe_m_heston_vanilla_fourier_consistency():
    m = EquityMarket(100, .01, .02)
    p = HestonParams(2, .04, .5, -.7, .04)
    result = price_mc(m, p, European(100), cfg(96, 28000, seed=2026))
    target = heston_european(m, p, 100, 1)
    assert result.scheme == 'qe_m'
    assert abs(result.price-target) <= 4*result.stderr + .12


def test_qe_m_xi_zero_nonconstant_variance_exact_node_values():
    m = EquityMarket(100, .03)
    p = HestonParams(2., .09, 0., -.5, .04)
    c = cfg(7, 1000)
    x = simulate_heston(m, p, c)
    time = np.linspace(0, 1, c.n_steps+1)
    theoretical = p.theta+(p.v0-p.theta)*np.exp(-p.kappa*time)
    np.testing.assert_allclose(x.variance[:, 0], theoretical, atol=1e-15)
    vbar = p.theta+(p.v0-p.theta)*(-math.expm1(-p.kappa))/(p.kappa)
    result = price_mc(m, p, European(100), cfg(7, 28000, 21))
    assert abs(result.price-black_scholes(m,100,1,math.sqrt(vbar))) < 4*result.stderr


def test_qe_m_crn_bump_greeks_and_statistical_se():
    m = EquityMarket(100, .01)
    p = HestonParams(2, .04, .4, -.6, .04)
    c = cfg(50, 6000)
    g1 = bump_greeks(m,p,European(100),c)
    g2 = bump_greeks(m,p,European(100),c)
    assert g1 == g2
    assert 0. < g1.delta < 1.
    assert g1.v0_sensitivity > 0.


def test_qe_m_feller_boundary_absorbing_zero():
    m = EquityMarket(100)
    p = HestonParams(2., 0., .9, -.5, 0.)
    x = simulate_heston(m,p,cfg(12,2000))
    assert np.all(x.variance == 0)
    np.testing.assert_allclose(x.spot, 100, rtol=0, atol=1e-12)


def test_qe_m_quadratic_branch_cir_first_two_moments():
    m = EquityMarket(100)
    p = HestonParams(2., .04, .5, -.5, .04)
    t = .05  # psi ~= 0.28 -> quadratic branch
    x = simulate_heston(m, p, MCConfig(t, 1, 160000, 2026, True, 'qe_m'))
    v = x.variance[-1]
    e = math.exp(-p.kappa*t)
    target_mean = p.theta + (p.v0-p.theta)*e
    target_var = (p.v0*p.xi**2*e*(1-e)/p.kappa
                  + p.theta*p.xi**2*(1-e)**2/(2*p.kappa))
    assert v.mean() == pytest.approx(target_mean, abs=1.5e-4)
    assert v.var() == pytest.approx(target_var, abs=3e-5)

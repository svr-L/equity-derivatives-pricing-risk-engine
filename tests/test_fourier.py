import cmath
import math

import pytest

from equity_derivatives import (EquityMarket, European, HestonParams, MCConfig,
                                black_scholes, heston_european, heston_log_return_cf,
                                price_mc)


def test_zero_vol_of_vol_exact_constant_variance_black_scholes():
    m = EquityMarket(100, .02, .01)
    p = HestonParams(2, .04, 0, -.6, .04)
    for strike in (70, 90, 100, 115, 135):
        assert heston_european(m, p, strike, 1.2) == pytest.approx(
            black_scholes(m, strike, 1.2, .2), abs=1e-10)


def test_zero_xi_nonconstant_variance_integral():
    m = EquityMarket(100, -.01, .02)
    p = HestonParams(2.5, .09, 0, 0., .04)
    t = 1.4
    vbar = p.theta + (p.v0-p.theta)*(-math.expm1(-p.kappa*t))/(p.kappa*t)
    assert heston_european(m, p, 110, t) == pytest.approx(
        black_scholes(m, 110, t, math.sqrt(vbar)), abs=1e-10)


def test_characteristic_function_martingale():
    p = HestonParams(2, .04, .5, -.7, .04)
    assert heston_log_return_cf(-1j, p, 1, .025, .01) == pytest.approx(
        cmath.exp(.015), abs=1e-12)
    assert heston_log_return_cf(0, p, 1) == pytest.approx(1.+0j, abs=1e-12)


def test_heston_fourier_prices_put_call_parity_positive():
    m = EquityMarket(100, .015, .02)
    p = HestonParams(2, .04, .5, -.7, .04)
    for k in (80, 100, 120):
        c = heston_european(m, p, k, 1)
        put = heston_european(m, p, k, 1, is_call=False)
        assert c >= 0 and put >= 0
        assert c - put == pytest.approx(100*math.exp(-.02) - k*math.exp(-.015), abs=1e-9)


def test_heston_mc_vs_heston_fourier():
    m = EquityMarket(100, .01, .02)
    p = HestonParams(2, .04, .5, -.7, .04)
    mc = price_mc(m, p, European(100), MCConfig(1, 150, 12000, 2026))
    exact = heston_european(m, p, 100, 1)
    # A Monte Carlo statistical comparison, with tolerance for FT Euler time bias.
    assert abs(mc.price - exact) <= 4 * mc.stderr + .2


def test_fourier_integration_boundary_stability_for_standard_case():
    m = EquityMarket(100, .01)
    p = HestonParams(1.7, .04, .4, -.6, .05)
    assert heston_european(m,p,100,1,integration_upper=150) == pytest.approx(
        heston_european(m,p,100,1,integration_upper=220), abs=5e-5)

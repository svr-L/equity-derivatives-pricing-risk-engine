import math

import numpy as np
import pytest

from equity_derivatives import (ArithmeticAsian, Barrier, EquityMarket, European,
                                FixedStrikeLookback, HestonParams, MCConfig, black_scholes,
                                bump_greeks, price_mc)
from equity_derivatives.simulation import NormalShocks


def setup():
    return EquityMarket(100, .01), HestonParams(2., .04, 0., -.7, .04)


def test_constant_variance_mc_matches_black_scholes_within_ci():
    m, p = setup()
    cfg = MCConfig(1, 120, 12000, 42)
    result = price_mc(m, p, European(100), cfg)
    target = black_scholes(m, 100, 1, .2)
    assert abs(result.price-target) < 3*result.stderr
    assert result.n_independent == 6000
    assert result.n_paths == 12000
    assert result.ci_low < result.price < result.ci_high


def test_antithetic_mc_standard_error_uses_independent_pairs():
    m, p = setup()
    cfg = MCConfig(1, 1, 4, 9)
    shocks = NormalShocks(np.array([[1., -1., 2., -2.]]),
                           np.array([[1., -1., 2., -2.]]))
    out = price_mc(m, p, European(100), cfg, shocks=shocks)
    # The two independent sample units are pair-averages, not four paths.
    from equity_derivatives import simulate_heston
    terminals = simulate_heston(m, p, cfg, shocks).spot[-1]
    vals = m.discount(1)*np.maximum(terminals-100,0)
    pairs = vals.reshape(2,2).mean(axis=1)
    assert out.stderr == pytest.approx(pairs.std(ddof=1)/math.sqrt(2))


def test_pathwise_payoff_orderings():
    m, p = setup()
    cfg = MCConfig(1, 70, 3000)
    vanilla = price_mc(m, p, European(100), cfg).price
    asian = price_mc(m, p, ArithmeticAsian(100), cfg).price
    lookback = price_mc(m, p, FixedStrikeLookback(100), cfg).price
    barrier = price_mc(m, p, Barrier(100, 125), cfg).price
    assert 0 <= barrier <= vanilla
    assert asian < lookback
    assert vanilla <= lookback


def test_knock_in_out_mc_parity_exact_same_shocks():
    m, p = setup()
    cfg = MCConfig(1, 80, 3000)
    c = price_mc(m, p, European(100), cfg).price
    ki = price_mc(m, p, Barrier(100, 125, knock="in"), cfg).price
    ko = price_mc(m, p, Barrier(100, 125, knock="out"), cfg).price
    assert ki + ko == pytest.approx(c, abs=1e-11)


def test_crn_delta_reasonable_and_positive_gamma():
    m, p = setup()
    out = bump_greeks(m, p, European(100), MCConfig(1, 100, 14000))
    assert .48 < out.delta < .67
    assert 0 < out.gamma < .04
    assert out.v0_sensitivity > 0


def test_invalid_confidence_and_bumps():
    m, p = setup()
    cfg = MCConfig(1, 3, 100)
    with pytest.raises(ValueError):
        price_mc(m, p, European(100), cfg, confidence_level=1)
    with pytest.raises(ValueError):
        bump_greeks(m, p, European(100), cfg, absolute_v0_bump=.05)

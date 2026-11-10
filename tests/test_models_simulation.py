import numpy as np
import pytest

from equity_derivatives import (EquityMarket, HestonParams, MCConfig,
                                sample_shocks, simulate_heston)


def test_validation():
    with pytest.raises(ValueError):
        EquityMarket(0)
    with pytest.raises(ValueError):
        HestonParams(2, .04, .5, 1.1, .04)
    with pytest.raises(ValueError):
        HestonParams(0, .04, .5, .2, .04)
    with pytest.raises(ValueError):
        MCConfig(1., 10, 999, antithetic=True)
    with pytest.raises(ValueError):
        MCConfig(0.)


def test_shocks_reproducible_and_antithetic():
    cfg = MCConfig(1, 10, 10, 12)
    a, b = sample_shocks(cfg), sample_shocks(cfg)
    np.testing.assert_array_equal(a.z_variance, b.z_variance)
    np.testing.assert_array_equal(a.z_variance[:, ::2], -a.z_variance[:, 1::2])
    np.testing.assert_array_equal(a.z_independent[:, ::2], -a.z_independent[:, 1::2])


def test_paths_are_positive_and_martingale_in_expectation():
    market = EquityMarket(100, .02, .01)
    pars = HestonParams(2., .04, .4, -.65, .04)
    cfg = MCConfig(1, 125, 12000, 17)
    paths = simulate_heston(market, pars, cfg)
    assert paths.spot.shape == (126, 12000)
    assert paths.variance.shape == (126, 12000)
    assert np.all(paths.spot > 0)
    assert np.all(paths.variance >= 0)
    assert np.all(paths.spot[0] == 100)
    assert np.all(paths.variance[0] == .04)
    assert paths.spot[-1].mean() == pytest.approx(market.forward(1), rel=.01)


def test_common_random_numbers_exact_replay():
    m, p = EquityMarket(100, 0), HestonParams(2, .04, .5, -.7, .04)
    cfg = MCConfig(1, 40, 200, 77)
    shocks = sample_shocks(cfg)
    paths1 = simulate_heston(m, p, cfg, shocks)
    paths2 = simulate_heston(m, p, cfg, shocks)
    np.testing.assert_array_equal(paths1.spot, paths2.spot)
    with pytest.raises(ValueError, match="shape"):
        simulate_heston(m, p, cfg, sample_shocks(MCConfig(1, 30, 200)))


def test_zero_vol_of_vol_exact_gbm_distribution_without_grid_bias():
    m = EquityMarket(100, .03, .01)
    p = HestonParams(1.8, .09, 0, -.5, .09)
    paths = simulate_heston(m, p, MCConfig(1, 30, 5000))
    assert np.allclose(paths.variance, .09)
    assert np.all(paths.spot > 0)


def test_minimum_antithetic_paths_and_seed_validation():
    with pytest.raises(ValueError, match=">= 4"):
        MCConfig(1, 10, 2)
    with pytest.raises(ValueError, match="seed"):
        MCConfig(1, 10, 100, -1)

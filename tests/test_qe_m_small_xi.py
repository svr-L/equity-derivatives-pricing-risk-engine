"""QE-M with vanishing vol-of-vol (v0.4.1 fix of the degenerate branch).

v0.4.0 set v_next = m when psi < 1e-12 but kept the general log-stock update,
whose rho-correlated part comes only from (rho/xi)*(v_next - m) = 0. For tiny
xi > 0 the stock therefore lost the rho**2 share of its variance (European
ATM 5.15 instead of 7.36 at rho=-0.7).
"""

import math

import numpy as np
import pytest
from equity_derivatives import (
    Barrier,
    EquityMarket,
    European,
    HestonParams,
    MCConfig,
    black_scholes,
    black_scholes_barrier,
    price_mc,
    sample_shocks,
    simulate_heston,
)

M = EquityMarket(100.0, 0.01, 0.02)


@pytest.mark.parametrize("xi", [1e-12, 1e-9, 1e-7])
def test_one_step_log_return_has_full_variance(xi):
    p = HestonParams(2.0, 0.04, xi, -0.7, 0.04)
    x = np.log(simulate_heston(M, p, MCConfig(1.0, 1, 200_000, 3, True, "qe_m")).spot[-1] / M.spot)
    # full integrated variance 0.04; v0.4.0 gave (1 - rho**2)*0.04 = 0.0204
    assert x.var() == pytest.approx(0.04, abs=1e-3)


@pytest.mark.parametrize("xi", [1e-9, 1e-7])
def test_tiny_xi_reproduces_xi_zero_branch_with_common_shocks(xi):
    cfg = MCConfig(1.0, 24, 20_000, 5, True, "qe_m")
    shocks = sample_shocks(cfg)
    exact = price_mc(
        M, HestonParams(2.0, 0.04, 0.0, -0.7, 0.04), European(100.0), cfg, shocks=shocks
    ).price
    tiny = price_mc(
        M, HestonParams(2.0, 0.04, xi, -0.7, 0.04), European(100.0), cfg, shocks=shocks
    ).price
    assert tiny == pytest.approx(exact, abs=1e-6)


def test_tiny_xi_nonconstant_variance_close_to_xi_zero_branch():
    # xi=0 uses exact integrated variance per step, xi>0 QE-M the trapezoid rule
    cfg = MCConfig(1.0, 24, 20_000, 5, True, "qe_m")
    shocks = sample_shocks(cfg)
    exact = price_mc(
        M, HestonParams(2.0, 0.04, 0.0, -0.7, 0.09), European(100.0), cfg, shocks=shocks
    ).price
    tiny = price_mc(
        M, HestonParams(2.0, 0.04, 1e-9, -0.7, 0.09), European(100.0), cfg, shocks=shocks
    ).price
    assert tiny == pytest.approx(exact, abs=3e-3)


def test_price_continuous_across_degenerate_branch_threshold():
    # psi ~ xi**2*dt/v: with daily steps the degenerate branch applies below xi ~ 1.4e-6
    cfg = MCConfig(1.0, 252, 10_000, 9, True, "qe_m")
    shocks = sample_shocks(cfg)
    prices = [
        price_mc(
            M, HestonParams(2.0, 0.04, xi, -0.7, 0.04), European(100.0), cfg, shocks=shocks
        ).price
        for xi in (5e-7, 1e-6, 2e-6, 5e-6)
    ]
    assert max(prices) - min(prices) < 1e-4


def test_tiny_xi_european_matches_black_scholes():
    res = price_mc(
        M,
        HestonParams(2.0, 0.04, 1e-8, -0.7, 0.04),
        European(100.0),
        MCConfig(1.0, 12, 100_000, 21, True, "qe_m"),
    )
    assert abs(res.price - black_scholes(M, 100.0, 1.0, 0.2)) < 3.5 * res.stderr


def test_tiny_xi_bridge_barrier_matches_continuous_formula():
    prod = Barrier(100.0, 125.0, "up", "out", True, "continuous_bridge")
    res = price_mc(
        M,
        HestonParams(2.0, 0.04, 1e-9, -0.7, 0.04),
        prod,
        MCConfig(1.0, 16, 40_000, 8, True, "qe_m"),
    )
    assert abs(res.price - black_scholes_barrier(M, prod, 1.0, 0.2)) < 3.5 * res.stderr


def test_tiny_xi_martingale_forward():
    paths = simulate_heston(
        M, HestonParams(2.0, 0.04, 1e-9, 0.9, 0.04), MCConfig(2.0, 8, 100_000, 13, True, "qe_m")
    )
    terminal = paths.spot[-1]
    pair_means = 0.5 * (terminal[0::2] + terminal[1::2])
    se = pair_means.std(ddof=1) / math.sqrt(pair_means.size)
    assert abs(terminal.mean() - M.forward(2.0)) < 4 * se

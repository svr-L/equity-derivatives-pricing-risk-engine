"""Monte Carlo accuracy against independent references, with statistical
tolerances (several seeds, pooled standard errors) instead of loose absolute
fudge terms.

References (see tests/test_reference_values.py): Heston European price from
mpmath/QuantLib; European Greeks from QuantLib AnalyticHestonEngine finite
differences with the same bumps as ``bump_greeks``; continuous Heston barrier
from QuantLib FdHestonBarrierEngine.
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
    black_scholes_barrier,
    bump_greeks,
    price_mc,
)

M = EquityMarket(100.0, 0.01, 0.02)
P = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
EUROPEAN_ATM_REF = 6.51384202112733
# QE-M time-discretization allowance for this case: multi-seed studies give
# |bias| <~ 0.005 already at 12-24 steps per year (~0.08% of the price).
QE_M_BIAS_ALLOWANCE = 0.005


def _pooled(results):
    prices = np.array([r.price for r in results])
    return prices.mean(), math.sqrt(sum(r.stderr**2 for r in results)) / len(results)


@pytest.mark.parametrize("steps,seed0", [(24, 100), (12, 200)])
def test_qe_m_european_matches_reference_at_coarse_steps(steps, seed0):
    runs = [
        price_mc(M, P, European(100.0), MCConfig(1.0, steps, 40_000, seed0 + s, True, "qe_m"))
        for s in range(8)
    ]
    mean, se = _pooled(runs)
    assert abs(mean - EUROPEAN_ATM_REF) < 4 * se + QE_M_BIAS_ALLOWANCE


def test_constant_volatility_bridge_is_unbiased():
    prod = Barrier(100.0, 125.0, "up", "out", True, "continuous_bridge")
    exact = black_scholes_barrier(M, prod, 1.0, 0.2)
    runs = [
        price_mc(
            M,
            HestonParams(2.0, 0.04, 0.0, -0.7, 0.04),
            prod,
            MCConfig(1.0, 16, 40_000, 500 + s, True, "qe_m"),
        )
        for s in range(8)
    ]
    mean, se = _pooled(runs)
    assert abs(mean - exact) < 4 * se


def test_bump_greeks_match_quantlib_reference():
    ref = dict(
        price=6.513842021127332,
        delta=0.5925844609750781,
        gamma=0.022944937237106444,
        v0_sensitivity=40.986146303781275,
    )
    greeks = [
        bump_greeks(M, P, European(100.0), MCConfig(1.0, 24, 20_000, 700 + s, True, "qe_m"))
        for s in range(6)
    ]
    for name in ("delta", "gamma", "v0_sensitivity"):
        x = np.array([getattr(g, name) for g in greeks])
        se = x.std(ddof=1) / math.sqrt(x.size)
        assert abs(x.mean() - ref[name]) < 4 * se, name


def test_heston_bridge_barrier_close_to_quantlib_pde():
    """Bridge weights are approximate under stochastic volatility (documented).
    Against QuantLib FD (finest grid 4.3886; grid sensitivity ~0.005) the bridge
    is about 0.5-1% low at 126-252 steps and about 2% low at 52 steps."""
    fd_reference = 4.3886
    res = price_mc(
        M,
        P,
        Barrier(100.0, 125.0, "up", "out", True, "continuous_bridge"),
        MCConfig(1.0, 126, 30_000, 31, True, "qe_m"),
    )
    assert abs(res.price - fd_reference) < 0.02 * fd_reference + 3.5 * res.stderr

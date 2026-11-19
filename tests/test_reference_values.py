"""Prices against independent reference values (no QuantLib needed at test time).

Heston European references: 40-significant-digit mpmath evaluation of the
Lewis integral with the classic little-trap characteristic function, cross-
checked against QuantLib 1.44 ``AnalyticHestonEngine`` (AngledContour
contour + exp-sinh integration). The two agree to <= 1.1e-13 on every row.
Barrier references: QuantLib ``AnalyticBarrierEngine``.
Regenerate with ``python tests/reference_generator.py`` (needs mpmath, QuantLib).
pyproject's pytest settings turn FourierAccuracyWarning into an error, so
every comparison below also asserts that no accuracy warning is raised.

Several rows are the v0.4.0 failure cases (short expiries with large
vol-of-vol, where the old fixed truncation returned negative prices; tiny
vol-of-vol; long maturities with large kappa*T).
"""

import math
from collections import defaultdict

import numpy as np
import pytest
from equity_derivatives import (
    Barrier,
    EquityMarket,
    HestonParams,
    black_scholes_barrier,
    heston_european,
    heston_surface_prices,
)

DOC = (2.0, 0.04, 0.5, -0.7, 0.04)
SYN = (2.5, 0.05, 0.6, -0.65, 0.045)
SPY = (3.3418, 0.0550, 1.3483, -0.4889, 0.02323)
HXI = (1.0, 0.09, 2.0, -0.8, 0.01)
AND1 = (0.5, 0.04, 1.0, -0.9, 0.04)
RPOS = (1.5, 0.06, 2.5, 0.7, 0.02)
SMXI = (15.0, 0.0004, 0.01, 0.98, 0.5)
BIGK = (
    11.8024237292485,
    0.00935575556100876,
    0.17910576521853594,
    0.19318827677060546,
    0.11342933335011533,
)

# (label, spot, r, q, (kappa, theta, xi, rho, v0), days (ACT/365), strike, is_call, price)
HESTON_REFERENCE = [
    ("doc 1y put 80", 100.0, 0.01, 0.02, DOC, 365, 80.0, False, 1.88988132271041),
    ("doc 1y call 100", 100.0, 0.01, 0.02, DOC, 365, 100.0, True, 6.51384202112733),
    ("doc 1y call 120", 100.0, 0.01, 0.02, DOC, 365, 120.0, True, 0.659663836090283),
    ("doc 1d call 104", 100.0, 0.01, 0.02, DOC, 1, 104.0, True, 4.29335382957443e-6),
    ("syn 3m put 90", 100.0, 0.02, 0.01, SYN, 91, 90.0, False, 1.16025578872829),
    ("syn 2y call 110", 100.0, 0.02, 0.01, SYN, 730, 110.0, True, 7.69115677104627),
    ("spy 7d put 0.95", 769.66, 0.04, 0.01, SPY, 7, 731.177, False, 0.272185526253622),
    ("spy 7d call 1.05", 769.66, 0.04, 0.01, SPY, 7, 808.143, True, 0.0348517275135442),
    ("spy 21d call atm", 769.66, 0.04, 0.01, SPY, 21, 769.66, True, 11.1989190687931),
    ("spy 1y put 0.8", 769.66, 0.04, 0.01, SPY, 365, 615.728, False, 12.7741975798741),
    ("hxi 7d put 95", 100.0, 0.03, 0.0, HXI, 7, 95.0, False, 0.0292588881094509),
    ("hxi 7d call 105", 100.0, 0.03, 0.0, HXI, 7, 105.0, True, 2.24082511293205e-5),
    ("hxi 21d call 105", 100.0, 0.03, 0.0, HXI, 21, 105.0, True, 0.00675980024854475),
    ("andersen-I 10y call 100", 100.0, 0.0, 0.0, AND1, 3650, 100.0, True, 13.0846701369924),
    ("andersen-I 7d call 100", 100.0, 0.0, 0.0, AND1, 7, 100.0, True, 1.08458870088646),
    ("rho>0 30d call 125", 100.0, 0.02, 0.0, RPOS, 30, 125.0, True, 0.0797794021037648),
    ("small xi 1y call atm", 100.0, 0.03, 0.01, SMXI, 365, 102.0201, True, 7.24245102856912),
    ("big kappa*T 10y call", 100.0, 0.03, 0.01, BIGK, 3650, 156.5734, True, 4.11703753416211),
]

# BS continuous barriers, S=100 r=.01 q=.02 vol=.2 T=1:
# (strike, barrier, direction, knock, is_call, QuantLib AnalyticBarrierEngine price)
BARRIER_REFERENCE = [
    (90.0, 125.0, "up", "out", True, 5.136342507581385),
    (90.0, 125.0, "up", "out", False, 3.8058043260793943),
    (110.0, 125.0, "up", "out", True, 0.3959229796768291),
    (110.0, 125.0, "up", "out", False, 14.463255837037359),
    (90.0, 125.0, "up", "in", True, 7.602022314723297),
    (90.0, 125.0, "up", "in", False, 0.017178202974884982),
    (110.0, 125.0, "up", "in", True, 3.514464372521946),
    (110.0, 125.0, "up", "in", False, 0.33274589689435885),
    (90.0, 80.0, "down", "out", True, 12.405497608146232),
    (90.0, 80.0, "down", "out", False, 0.3054467891530588),
    (110.0, 80.0, "down", "out", True, 3.8902418937902516),
    (110.0, 80.0, "down", "out", False, 5.4376326023815675),
    (90.0, 80.0, "down", "in", True, 0.3328672141584499),
    (90.0, 80.0, "down", "in", False, 3.5175357399012204),
    (110.0, 80.0, "down", "in", True, 0.020145458408523498),
    (110.0, 80.0, "down", "in", False, 9.358369131550152),
]


@pytest.mark.parametrize("row", HESTON_REFERENCE, ids=[r[0] for r in HESTON_REFERENCE])
def test_scalar_heston_matches_reference(row):
    _, spot, r, q, p, days, strike, is_call, ref = row
    value = heston_european(EquityMarket(spot, r, q), HestonParams(*p), strike, days / 365, is_call)
    assert value == pytest.approx(ref, abs=1e-10 * spot)


def test_batch_heston_matches_reference_grouped_by_market_and_params():
    groups = defaultdict(list)
    for row in HESTON_REFERENCE:
        groups[(row[1], row[2], row[3], row[4])].append(row)
    for (spot, r, q, p), rows in groups.items():
        out = heston_surface_prices(
            EquityMarket(spot, r, q),
            HestonParams(*p),
            [x[6] for x in rows],
            [x[5] / 365 for x in rows],
            [x[7] for x in rows],
        )
        np.testing.assert_allclose(out, [x[8] for x in rows], rtol=0, atol=1e-10 * spot)


def test_andersen_2008_case_i_literature_value():
    """Andersen (2008), test case I: S=100, K=100, T=10, r=0, kappa=0.5,
    theta=v0=0.04, eps(xi)=1, rho=-0.9 has reference price 13.0847."""
    value = heston_european(
        EquityMarket(100.0), HestonParams(0.5, 0.04, 1.0, -0.9, 0.04), 100.0, 10.0
    )
    assert value == pytest.approx(13.0847, abs=5e-5)


def test_v040_failure_cases_are_now_positive_and_accurate():
    m = EquityMarket(100.0, 0.03, 0.0)
    p = HestonParams(*HXI)
    call = heston_european(m, p, 105.0, 7 / 365)  # v0.4.0: -0.00715
    batch = heston_surface_prices(m, p, [105.0], [7 / 365])[0]  # v0.4.0: -0.0109
    assert call > 0 and batch > 0
    assert call == pytest.approx(2.24082511293205e-5, abs=1e-9)
    tiny = heston_european(
        EquityMarket(100.0, 0.01, 0.02), HestonParams(2.0, 0.04, 1e-9, -0.7, 0.04), 100.0, 1.0
    )
    assert tiny == pytest.approx(7.364289, abs=1e-6)  # v0.4.0: 0.84111 (BS 20%: 7.36429)


@pytest.mark.parametrize(
    "row",
    BARRIER_REFERENCE,
    ids=[f"{r[2]}-{r[3]}-{'C' if r[4] else 'P'}-K{r[0]:.0f}" for r in BARRIER_REFERENCE],
)
def test_black_scholes_barrier_matches_quantlib_reference(row):
    strike, barrier, direction, knock, is_call, ref = row
    value = black_scholes_barrier(
        EquityMarket(100.0, 0.01, 0.02),
        Barrier(strike, barrier, direction, knock, is_call),
        1.0,
        0.2,
    )
    assert value == pytest.approx(ref, abs=1e-11)


def test_barrier_reference_in_out_parity_matches_vanilla():
    for strike in (90.0, 110.0):
        for is_call in (True, False):
            rows = [r for r in BARRIER_REFERENCE if r[0] == strike and r[4] == is_call]
            up = sum(r[5] for r in rows if r[2] == "up")
            down = sum(r[5] for r in rows if r[2] == "down")
            assert up == pytest.approx(down, abs=1e-12)  # both equal the vanilla price
    assert math.isfinite(up)

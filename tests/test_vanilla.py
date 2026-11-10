import math

import pytest

from equity_derivatives import (EquityMarket, black76_forward, black_scholes,
                                black_vega, implied_vol)


@pytest.mark.parametrize("s,k,r,q,t,v", [
    (100, 100, .01, .0, 1, .2), (95, 105, -.01, .02, 2, .35),
    (110, 90, .07, .03, .5, .14),
])
def test_black_put_call_parity_and_implied_vol(s, k, r, q, t, v):
    m = EquityMarket(s, r, q)
    c = black_scholes(m, k, t, v, True)
    p = black_scholes(m, k, t, v, False)
    assert c - p == pytest.approx(s * math.exp(-q*t) - k * math.exp(-r*t), abs=1e-11)
    assert implied_vol(c, m, k, t, True) == pytest.approx(v, abs=1e-9)
    assert implied_vol(p, m, k, t, False) == pytest.approx(v, abs=1e-9)
    assert c == pytest.approx(m.discount(t) * black76_forward(m.forward(t), k, t, v))


def test_black_vega_vs_bump():
    m = EquityMarket(100, .02, .01)
    h = 1e-4
    numerical = (black_scholes(m, 105, 1.5, .21+h) - black_scholes(m, 105, 1.5, .21-h))/(2*h)
    assert numerical == pytest.approx(black_vega(m, 105, 1.5, .21), rel=1e-7)


def test_implied_vol_rejects_infeasible_prices():
    m = EquityMarket(100, 0, 0)
    with pytest.raises(ValueError, match="bounds"):
        implied_vol(101, m, 100, 1)
    with pytest.raises(ValueError, match="bounds"):
        implied_vol(-.1, m, 100, 1)
    assert implied_vol(0, m, 150, 1) == 0
    with pytest.raises(ValueError, match="maturity"):
        implied_vol(5, m, 100, 0)

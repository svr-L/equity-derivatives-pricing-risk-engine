import math

import pytest

from equity_derivatives import (Barrier, EquityMarket, black_scholes,
                                black_scholes_barrier)


@pytest.mark.parametrize('direction,h', [('up',125),('down',80)])
@pytest.mark.parametrize('is_call', [True,False])
def test_continuous_in_out_equals_vanilla(direction,h,is_call):
    m = EquityMarket(100,.02,.01)
    ko = black_scholes_barrier(m,Barrier(100,h,direction,'out',is_call),1,.25)
    ki = black_scholes_barrier(m,Barrier(100,h,direction,'in',is_call),1,.25)
    assert ko >= 0
    assert ko + ki == pytest.approx(black_scholes(m,100,1,.25,is_call),abs=1e-12)


def test_initial_barrier_hit_touches_immediately():
    m = EquityMarket(100)
    assert black_scholes_barrier(m,Barrier(100,100,knock='out'),1,.2) == 0.
    assert black_scholes_barrier(m,Barrier(100,100,knock='in'),1,.2) == pytest.approx(
        black_scholes(m,100,1,.2))


def test_zero_vol_deterministic_path_and_expiry():
    m = EquityMarket(100, math.log(1.1))
    assert black_scholes_barrier(m,Barrier(90,105),1,0) == 0.
    assert black_scholes_barrier(m,Barrier(90,125),1,0) > 0.
    assert black_scholes_barrier(m,Barrier(100,125),0,.2) == 0.


def test_up_out_call_strike_at_or_above_barrier_zero():
    m = EquityMarket(100,.01)
    assert black_scholes_barrier(m,Barrier(135,125),1,.2) == 0.


def test_black_barrier_price_low_for_barrier_close_to_initial_spot():
    m = EquityMarket(100)
    a = black_scholes_barrier(m,Barrier(100,101),1,.2)
    b = black_scholes_barrier(m,Barrier(100,130),1,.2)
    assert 0. <= a < b < black_scholes(m,100,1,.2)

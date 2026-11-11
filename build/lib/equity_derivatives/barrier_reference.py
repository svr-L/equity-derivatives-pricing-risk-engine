"""Continuous-monitoring zero-rebate barrier benchmarks under constant vol.

Independent closed-form reflection-principle implementation for log GBM.
For monitored Heston barriers, see barrier_bridge.py (approximate).
"""
from __future__ import annotations

import math

from scipy.special import ndtr

from .models import EquityMarket
from .payoffs import Barrier
from .vanilla import black_scholes


def _truncated_lognormal_moment(lower: float, upper: float, mean: float,
                                sd: float, power: int) -> float:
    """E[e^(power X) 1_{lower < X < upper}], X~N(mean,sd²)."""
    shift = mean + power*sd*sd
    return math.exp(power*mean+.5*power*power*sd*sd) * (
        ndtr((upper-shift)/sd)-ndtr((lower-shift)/sd))


def black_scholes_barrier(market: EquityMarket, option: Barrier, maturity: float,
                          vol: float) -> float:
    """Exact continuously monitored European single-barrier price for GBM.

    Uses reflection-principle killed log-price density, with zero rebates.
    Touch at time 0 (including equality) activates a knock-in or kills a
    knock-out immediately. Allows calls/puts, up/down, in/out.
    Flat rate/dividend, nonnegative vol; no discrete-dividend jumps.
    """
    if not math.isfinite(maturity) or maturity < 0:
        raise ValueError("maturity must be finite and nonnegative")
    if not math.isfinite(vol) or vol < 0:
        raise ValueError("vol must be finite and nonnegative")
    vanilla = black_scholes(market, option.strike, maturity, vol, option.is_call)
    spot, h = market.spot, option.barrier
    already_hit = (spot >= h if option.direction == "up" else spot <= h)
    if already_hit:
        return float(vanilla if option.knock == "in" else 0.)
    if maturity == 0:
        return float(0. if option.knock == "in" else vanilla)
    if vol == 0:
        terminal = market.forward(maturity)
        hit = (terminal >= h if option.direction == "up" else terminal <= h)
        ko = 0. if hit else vanilla
        return float(vanilla-ko if option.knock == "in" else ko)

    sd = vol*math.sqrt(maturity)
    mu = market.rate-market.dividend_yield-.5*vol*vol
    mean = mu*maturity
    b = math.log(h/spot)
    k = math.log(option.strike/spot)
    if option.direction == "up":
        lower, upper = (k,b) if option.is_call else (-math.inf,min(b,k))
    else:
        lower, upper = (max(b,k),math.inf) if option.is_call else (b,k)
    if lower >= upper:
        ko = 0.
    else:
        # Killed Gaussian density via method of images for drifted log GBM.
        image_mean = 2*b + mean
        weight = math.exp(2*mu*b/(vol*vol))
        mass = (_truncated_lognormal_moment(lower,upper,mean,sd,0)
                - weight*_truncated_lognormal_moment(lower,upper,image_mean,sd,0))
        exp_mass = (_truncated_lognormal_moment(lower,upper,mean,sd,1)
                    - weight*_truncated_lognormal_moment(lower,upper,image_mean,sd,1))
        phi = 1 if option.is_call else -1
        ko = phi*(spot*exp_mass-option.strike*mass)*market.discount(maturity)
        # Tolerate cancellation at extreme near-zero prices, not large violations.
        if ko < -1e-7 or ko > vanilla + 1e-7:
            raise ArithmeticError("barrier reflection evaluation violates no-arbitrage bounds")
        ko = max(0.,min(float(ko),vanilla))
    return float(vanilla-ko if option.knock == "in" else ko)

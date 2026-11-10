"""Equity Black-Scholes / forward Black-76 and Black implied volatility.

Reimplemented independently from analytic conventions reviewed in the source
FX v0.2.0 (black.py) and FI v0.13.0 (core.py, volatility.py). Their repositories
are not modified and neither is a runtime dependency. See docs/PROVENANCE.md.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.special import ndtr

from .models import EquityMarket


def _validate(strike: float, maturity: float, vol: float) -> None:
    if not math.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be strictly positive")
    if not math.isfinite(maturity) or maturity < 0:
        raise ValueError("maturity must be nonnegative")
    if not math.isfinite(vol) or vol < 0:
        raise ValueError("vol must be nonnegative")


def black76_forward(forward: float, strike: float, maturity: float, vol: float,
                    is_call: bool = True) -> float:
    """Undiscounted European Black-76 payoff in forward units."""
    _validate(strike, maturity, vol)
    if not math.isfinite(forward) or forward <= 0:
        raise ValueError("forward must be strictly positive")
    phi = 1 if is_call else -1
    if vol == 0 or maturity == 0:
        return max(phi * (forward - strike), 0.0)
    sd = vol * math.sqrt(maturity)
    d1 = math.log(forward / strike) / sd + 0.5 * sd
    d2 = d1 - sd
    return float(phi * (forward * ndtr(phi * d1) - strike * ndtr(phi * d2)))


def black_scholes(market: EquityMarket, strike: float, maturity: float,
                  vol: float, is_call: bool = True) -> float:
    """European option with continuous dividend yield, time-0 discounted price."""
    return market.discount(maturity) * black76_forward(
        market.forward(maturity), strike, maturity, vol, is_call)


def black_vega(market: EquityMarket, strike: float, maturity: float, vol: float) -> float:
    """Derivative of discounted European price wrt Black volatility (per vol=1)."""
    _validate(strike, maturity, vol)
    if maturity == 0 or vol == 0:
        return 0.0
    sd = vol * math.sqrt(maturity)
    d1 = (math.log(market.forward(maturity) / strike) + 0.5 * sd * sd) / sd
    return float(market.spot * math.exp(-market.dividend_yield * maturity)
                 * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
                 * math.sqrt(maturity))


def implied_vol(price: float, market: EquityMarket, strike: float,
                maturity: float, is_call: bool = True) -> float:
    """Invert a discounted European Black price; return 0 for intrinsic boundary.

    Prices outside closed no-arbitrage bounds raise ValueError. At the upper
    bound implied volatility is unbounded and therefore undefined.
    """
    _validate(strike, maturity, 0.2)
    if maturity == 0:
        raise ValueError("implied volatility is not identified at maturity=0")
    if not math.isfinite(price):
        raise ValueError("price must be finite")
    disc = market.discount(maturity)
    fwd = market.forward(maturity)
    undiscounted = price / disc
    phi = 1 if is_call else -1
    lo = max(phi * (fwd - strike), 0.0)
    hi = fwd if is_call else strike
    tol = 1e-10 * max(1., fwd, strike)
    if undiscounted < lo - tol or undiscounted > hi + tol:
        raise ValueError("price violates European no-arbitrage bounds")
    if abs(undiscounted - lo) <= tol:
        return 0.0
    if undiscounted >= hi - tol:
        raise ValueError("Black implied volatility is unbounded at the upper bound")

    def error(vol: float) -> float:
        return black76_forward(fwd, strike, maturity, vol, is_call) - undiscounted

    upper = 0.5
    while error(upper) < 0 and upper < 128:
        upper *= 2
    if error(upper) < 0:
        raise ValueError("could not bracket implied volatility")
    return float(brentq(error, 0., upper, xtol=1e-12))

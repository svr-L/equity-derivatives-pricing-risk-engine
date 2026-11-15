"""Deterministic discount and *proportional* equity carry term structures.

No bootstrapping, dividend forecasts, or market-history inference is performed.
Cash/ex-date dividends are NOT supported. Snapshots and curves must share as-of.
Log-linear interpolation in discount factors => piecewise-constant instant forwards.
Outside the last quoted tenor, extrapolation is forbidden by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Sequence

import numpy as np

from .models import EquityMarket


@dataclass(frozen=True)
class DiscountCurve:
    """DF(t), anchored at DF(0)=1; time in ACT/365F-style year fractions.

    Input tenors must start strictly after zero; the origin is injected.
    Discount factors can increase (negative rates are supported).
    """
    tenors: tuple[float, ...]
    dfs: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.dfs) or not self.tenors:
            raise ValueError("curve requires nonempty equal-length tenors and dfs")
        if (any(not math.isfinite(t) or t <= 0 for t in self.tenors)
                or any(a >= b for a,b in zip(self.tenors,self.tenors[1:]))):
            raise ValueError("curve tenors must be finite, positive and strictly increasing")
        if any(not math.isfinite(d) or d <= 0 for d in self.dfs):
            raise ValueError("discount factors must be finite and positive")

    def __call__(self, maturity: float | np.ndarray) -> float | np.ndarray:
        a=np.asarray(maturity, dtype=float)
        if np.any(~np.isfinite(a)) or np.any(a < 0) or np.any(a > self.tenors[-1] + 1e-12):
            raise ValueError("maturity outside [0, last curve tenor]: no extrapolation")
        x=np.array((0., *self.tenors))
        y=np.log(np.array((1., *self.dfs)))
        value=np.exp(np.interp(a, x, y))
        return float(value) if a.ndim == 0 else value

    @classmethod
    def flat(cls, rate: float, maturity: float) -> "DiscountCurve":
        if not math.isfinite(rate) or not math.isfinite(maturity) or maturity <= 0:
            raise ValueError("require finite rate and positive maturity")
        return cls((float(maturity),), (math.exp(-rate*maturity),))


@dataclass(frozen=True)
class EquityCarryCurves:
    """Risk-free DF D(t) and proportional-dividend factor Q(t).

    F(0,t)=S0*Q(t)/D(t). This parameterization represents deterministic
    proportional carry, NOT scheduled discrete cash dividends or exercise value.
    """
    spot: float
    discount_curve: DiscountCurve
    dividend_curve: DiscountCurve
    asof: str  # YYYY-MM-DD; explicitly supplied valuation date

    def __post_init__(self) -> None:
        if not math.isfinite(self.spot) or self.spot <= 0:
            raise ValueError("spot must be finite and positive")
        try:
            parsed=date.fromisoformat(self.asof)
        except (TypeError, ValueError) as exc:
            raise ValueError("asof must be an ISO YYYY-MM-DD valuation date") from exc
        if parsed.isoformat() != self.asof:
            raise ValueError("asof must be exactly YYYY-MM-DD")

    def discount(self,t: float)->float:
        return float(self.discount_curve(t))

    def forward(self,t: float)->float:
        return self.spot*float(self.dividend_curve(t))/self.discount(t)

    def effective_market(self,t: float)->EquityMarket:
        if not math.isfinite(t) or t <= 0:
            raise ValueError("effective flat rate requires t>0")
        r=-math.log(self.discount(t))/t
        q=-math.log(float(self.dividend_curve(t)))/t
        return EquityMarket(self.spot,r,q)

    def log_forward_increment(self,t0: float,t1:float)->float:
        """Integral of (r-q) on [t0,t1] with exact piecewise-DF carry."""
        if t0 < 0 or t1 < t0:
            raise ValueError("require 0<=t0<=t1")
        return math.log(self.forward(t1)/self.forward(t0))

    def validate_market(self, market: EquityMarket) -> None:
        if not math.isclose(self.spot, market.spot, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("term structure spot differs from market spot")

    def validate_snapshot(self, timestamp: str) -> None:
        from datetime import datetime
        try:
            ts=datetime.fromisoformat(str(timestamp).replace("Z","+00:00"))
        except ValueError as exc:
            raise ValueError("unparseable snapshot timestamp") from exc
        if ts.date().isoformat()!=self.asof:
            raise ValueError("curve asof differs from snapshot UTC date")

    @classmethod
    def from_forwards(cls, spot: float, discount_curve: DiscountCurve,
                      maturities: Sequence[float], forwards: Sequence[float], *,
                      asof: str) -> "EquityCarryCurves":
        """Derive proportional dividend factors from supplied dated forwards.

        Does NOT infer forwards from an American option chain or history.
        """
        tenors=tuple(float(x) for x in maturities)
        fs=tuple(float(f) for f in forwards)
        if len(tenors)!=len(fs) or any(not math.isfinite(f) or f<=0 for f in fs):
            raise ValueError("forwards must be finite positive and match maturities")
        if not math.isfinite(spot) or spot<=0:
            raise ValueError("spot must be positive")
        q=tuple(f*float(discount_curve(t))/spot for t,f in zip(tenors,fs))
        return cls(spot,discount_curve,DiscountCurve(tenors,q),asof)

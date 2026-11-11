"""Common-random-number bump-and-revalue Monte Carlo sensitivities.

Spot delta/gamma hold Heston parameters, r and q constant. 'v0_sensitivity'
is wrt instantaneous *variance* v0, not an implied-vol market vega.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .models import EquityMarket, HestonParams, MCConfig
from .payoffs import Payoff
from .pricing import price_mc
from .simulation import sample_shocks


@dataclass(frozen=True)
class BumpGreeks:
    price: float
    delta: float
    gamma: float
    v0_sensitivity: float
    spot_bump: float
    v0_bump: float


def bump_greeks(market: EquityMarket, params: HestonParams, payoff: Payoff,
                config: MCConfig, *, relative_spot_bump: float = 0.01,
                absolute_v0_bump: float = 0.001) -> BumpGreeks:
    if not 0 < relative_spot_bump < 1 or absolute_v0_bump <= 0:
        raise ValueError("bumps must satisfy 0<spot bump<1 and v0 bump>0")
    if params.v0 - absolute_v0_bump < 0:
        raise ValueError("central v0 bump requires v0 >= absolute_v0_bump")
    shocks = sample_shocks(config)
    def value(mkt: EquityMarket, par: HestonParams) -> float:
        return price_mc(mkt, par, payoff, config, shocks=shocks).price

    p0 = value(market, params)
    ds = market.spot * relative_spot_bump
    p_sp = value(replace(market, spot=market.spot + ds), params)
    p_sm = value(replace(market, spot=market.spot - ds), params)
    p_vp = value(market, replace(params, v0=params.v0 + absolute_v0_bump))
    p_vm = value(market, replace(params, v0=params.v0 - absolute_v0_bump))
    return BumpGreeks(p0, (p_sp-p_sm)/(2*ds), (p_sp-2*p0+p_sm)/ds**2,
                      (p_vp-p_vm)/(2*absolute_v0_bump), ds, absolute_v0_bump)

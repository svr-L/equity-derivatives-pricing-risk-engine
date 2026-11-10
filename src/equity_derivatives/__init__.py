"""Equity derivatives pricing engine, version 0.1.0."""
from .heston_fourier import heston_european, heston_log_return_cf
from .greeks import BumpGreeks, bump_greeks
from .models import EquityMarket, HestonParams, MCConfig
from .payoffs import ArithmeticAsian, Barrier, European, FixedStrikeLookback
from .pricing import PriceResult, price_mc
from .simulation import HestonPaths, NormalShocks, sample_shocks, simulate_heston
from .vanilla import black76_forward, black_scholes, black_vega, implied_vol

__all__ = [
    "EquityMarket", "HestonParams", "MCConfig", "European", "ArithmeticAsian",
    "FixedStrikeLookback", "Barrier", "HestonPaths", "NormalShocks", "sample_shocks",
    "simulate_heston", "price_mc", "PriceResult", "bump_greeks", "BumpGreeks",
    "heston_european", "heston_log_return_cf", "black76_forward", "black_scholes", "black_vega", "implied_vol",
]

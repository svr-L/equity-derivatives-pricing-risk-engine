"""Equity derivatives pricing & risk engine, version 0.3.0."""
from .heston_fourier import heston_european, heston_log_return_cf
from .heston_surface import heston_surface_prices
from .barrier_bridge import barrier_survival_probabilities
from .barrier_reference import black_scholes_barrier
from .greeks import BumpGreeks, bump_greeks
from .models import EquityMarket, HestonParams, MCConfig
from .payoffs import ArithmeticAsian, Barrier, European, FixedStrikeLookback
from .pricing import PriceResult, price_mc
from .simulation import HestonPaths, NormalShocks, sample_shocks, simulate_heston
from .vanilla import black76_forward, black_scholes, black_vega, implied_vol
from .calibration import (
    VanillaQuote, CalibrationConfig, CalibrationResult, StartDiagnostic,
    FitEvaluation, calibrate_heston, evaluate_heston_fit,
)

__all__ = [
    "black_scholes_barrier", "barrier_survival_probabilities",
    "EquityMarket", "HestonParams", "MCConfig", "European", "ArithmeticAsian",
    "FixedStrikeLookback", "Barrier", "HestonPaths", "NormalShocks", "sample_shocks",
    "simulate_heston", "price_mc", "PriceResult", "bump_greeks", "BumpGreeks",
    "heston_european", "heston_log_return_cf", "heston_surface_prices",
    "black76_forward", "black_scholes", "black_vega", "implied_vol",
    "VanillaQuote", "CalibrationConfig", "CalibrationResult", "StartDiagnostic",
    "FitEvaluation", "calibrate_heston", "evaluate_heston_fit",
]

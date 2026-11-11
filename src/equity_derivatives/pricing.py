"""Monte Carlo estimator with independent-unit error bars for antithetic pairs."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .models import EquityMarket, HestonParams, MCConfig
from .payoffs import Barrier, Payoff
from .barrier_bridge import barrier_survival_probabilities
from .simulation import NormalShocks, simulate_heston


@dataclass(frozen=True)
class PriceResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_independent: int
    n_steps: int
    confidence_level: float
    scheme: str = "full_truncation_euler"


def price_mc(market: EquityMarket, params: HestonParams, payoff: Payoff,
             config: MCConfig, *, confidence_level: float = 0.95,
             shocks: NormalShocks | None = None) -> PriceResult:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    paths = simulate_heston(market, params, config, shocks)
    if isinstance(payoff, Barrier) and payoff.monitoring == "continuous_bridge":
        survival = barrier_survival_probabilities(paths, params, config, payoff)
        phi = 1 if payoff.is_call else -1
        vanilla = np.maximum(phi*(paths.spot[-1]-payoff.strike), 0.)
        samples = vanilla*(1.-survival if payoff.knock == "in" else survival)
    else:
        samples = np.asarray(payoff.evaluate(paths.spot), dtype=float)
    if samples.shape != (config.n_paths,):
        raise ValueError("payoff must return one value per path")
    discounted = market.discount(config.maturity) * samples
    # Each antithetic pair has one independent Gaussian seed. Using all paths
    # as iid incorrectly understates or overstates the Monte Carlo SE.
    independent = (discounted.reshape(-1, 2).mean(axis=1) if config.antithetic
                   else discounted)
    mean = float(independent.mean())
    stderr = float(independent.std(ddof=1) / math.sqrt(len(independent)))
    hw = float(norm.ppf(0.5 + confidence_level / 2) * stderr)
    return PriceResult(mean, stderr, mean - hw, mean + hw, config.n_paths,
                       len(independent), config.n_steps, confidence_level, config.scheme)

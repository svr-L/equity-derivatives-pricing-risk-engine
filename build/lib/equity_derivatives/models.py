"""Model inputs. Annualized continuous-time parameters under risk-neutral Q."""
from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(**kwargs: float) -> None:
    for key, x in kwargs.items():
        if not math.isfinite(x):
            raise ValueError(f"{key} must be finite")


@dataclass(frozen=True)
class EquityMarket:
    spot: float
    rate: float = 0.0  # continuously compounded, annualized risk-free rate
    dividend_yield: float = 0.0  # continuous proportional dividend yield

    def __post_init__(self) -> None:
        _finite(spot=self.spot, rate=self.rate, dividend_yield=self.dividend_yield)
        if self.spot <= 0:
            raise ValueError("spot must be strictly positive")

    def forward(self, maturity: float) -> float:
        return self.spot * math.exp((self.rate - self.dividend_yield) * maturity)

    def discount(self, maturity: float) -> float:
        return math.exp(-self.rate * maturity)


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float  # volatility of instantaneous variance (not market implied-vol vega)
    rho: float
    v0: float  # instantaneous variance, not volatility

    def __post_init__(self) -> None:
        _finite(kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho, v0=self.v0)
        if self.kappa <= 0 or self.theta < 0 or self.xi < 0 or self.v0 < 0:
            raise ValueError("require kappa > 0 and theta, xi, v0 >= 0")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be between -1 and 1")

    @property
    def feller_ratio(self) -> float:
        return math.inf if self.xi == 0 else 2 * self.kappa * self.theta / self.xi**2


@dataclass(frozen=True)
class MCConfig:
    maturity: float
    n_steps: int = 252
    n_paths: int = 20_000
    seed: int = 42
    antithetic: bool = True
    scheme: str = "full_truncation_euler"  # or "qe_m" (Andersen QE martingale correction)

    def __post_init__(self) -> None:
        _finite(maturity=self.maturity)
        if self.maturity <= 0:
            raise ValueError("MC maturity must be strictly positive")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if isinstance(self.n_paths, bool) or not isinstance(self.n_paths, int) or self.n_paths < 2:
            raise ValueError("n_paths must be an integer >= 2")
        if self.antithetic and (self.n_paths % 2 or self.n_paths < 4):
            raise ValueError("antithetic requires an even n_paths >= 4 for a defined SE")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.scheme not in ("full_truncation_euler", "qe_m"):
            raise ValueError("scheme must be full_truncation_euler or qe_m")

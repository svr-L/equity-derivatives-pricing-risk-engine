"""European, path-dependent cash-settled terminal payoffs; no rebates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Payoff(Protocol):
    def evaluate(self, spot_paths: NDArray[np.float64]) -> NDArray[np.float64]: ...


def _assert_paths(x: NDArray[np.float64]) -> None:
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("spot_paths must have shape (n_steps+1, n_paths), n_steps >= 1")
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("spot_paths must be finite and strictly positive")


def _validate_strike(strike: float) -> None:
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be strictly positive")


@dataclass(frozen=True)
class European:
    strike: float
    is_call: bool = True

    def __post_init__(self) -> None:
        _validate_strike(self.strike)

    def evaluate(self, spot_paths: NDArray[np.float64]) -> NDArray[np.float64]:
        _assert_paths(spot_paths)
        phi = 1 if self.is_call else -1
        return np.maximum(phi * (spot_paths[-1] - self.strike), 0.)


@dataclass(frozen=True)
class ArithmeticAsian:
    strike: float
    is_call: bool = True
    include_initial: bool = False

    def __post_init__(self) -> None:
        _validate_strike(self.strike)

    def evaluate(self, spot_paths: NDArray[np.float64]) -> NDArray[np.float64]:
        _assert_paths(spot_paths)
        average = np.mean(spot_paths if self.include_initial else spot_paths[1:], axis=0)
        phi = 1 if self.is_call else -1
        return np.maximum(phi * (average - self.strike), 0.)


@dataclass(frozen=True)
class FixedStrikeLookback:
    strike: float
    is_call: bool = True

    def __post_init__(self) -> None:
        _validate_strike(self.strike)

    def evaluate(self, spot_paths: NDArray[np.float64]) -> NDArray[np.float64]:
        _assert_paths(spot_paths)
        extremum = np.max(spot_paths, axis=0) if self.is_call else np.min(spot_paths, axis=0)
        phi = 1 if self.is_call else -1
        return np.maximum(phi * (extremum - self.strike), 0.)


@dataclass(frozen=True)
class Barrier:
    strike: float
    barrier: float
    direction: str = "up"  # "up" or "down"
    knock: str = "out"  # "out" or "in"
    is_call: bool = True
    monitoring: str = "discrete"  # or "continuous_bridge" (conditional approximation)

    def __post_init__(self) -> None:
        _validate_strike(self.strike)
        if self.monitoring not in ("discrete", "continuous_bridge"):
            raise ValueError("monitoring must be discrete or continuous_bridge")
        if not np.isfinite(self.barrier) or self.barrier <= 0:
            raise ValueError("barrier must be strictly positive")
        if self.direction not in ("up", "down"):
            raise ValueError("direction must be up/down")
        if self.knock not in ("in", "out"):
            raise ValueError("knock must be in/out")

    def evaluate(self, spot_paths: NDArray[np.float64]) -> NDArray[np.float64]:
        """Payoff observed on the simulation grid (including t=0).

        For monitoring="continuous_bridge", this *standalone* evaluate method
        still returns the grid-monitored payoff. Use price_mc() to apply the
        conditional-crossing bridge approximation with HestonPaths variance.
        """
        _assert_paths(spot_paths)
        hit = np.any(spot_paths >= self.barrier, axis=0) if self.direction == "up" else (
            np.any(spot_paths <= self.barrier, axis=0))
        active = hit if self.knock == "in" else ~hit
        phi = 1 if self.is_call else -1
        vanilla = np.maximum(phi * (spot_paths[-1] - self.strike), 0.)
        return np.where(active, vanilla, 0.)

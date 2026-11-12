"""Calibrate the risk-neutral Heston model against *European* vanilla quotes.

Rates, dividend yields, exercise style and quote filtration are user-supplied;
this module never infers curves, silently assumes real market conventions, or
represents American ETF options as European observations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .heston_surface import heston_surface_prices
from .models import EquityMarket, HestonParams
from .vanilla import black_vega, implied_vol


@dataclass(frozen=True)
class VanillaQuote:
    strike: float
    maturity: float
    mid: float
    is_call: bool = True
    bid: float | None = None
    ask: float | None = None
    identifier: str = ""

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.strike, self.maturity, self.mid)):
            raise ValueError("strike, maturity, mid must be finite")
        if self.strike <= 0 or self.maturity <= 0 or self.mid <= 0:
            raise ValueError("require strike, maturity and mid > 0")
        if (self.bid is None) != (self.ask is None):
            raise ValueError("bid and ask must be given together")
        if self.bid is not None:
            if not all(math.isfinite(x) for x in (self.bid, self.ask)) or not (0 <= self.bid <= self.mid <= self.ask):
                raise ValueError("require finite 0 <= bid <= mid <= ask")


@dataclass(frozen=True)
class CalibrationConfig:
    objective: str = "vega_scaled_price"  # 'price' or 'iv'
    n_starts: int = 4
    seed: int = 2026
    max_nfev: int = 100
    n_nodes: int = 96
    integration_upper: float = 180.
    vega_floor_frac_spot: float = 0.001
    # kappa, theta, vol-of-vol xi, rho, v0; these are priors, NOT Feller restrictions
    lower: tuple[float, ...] = (0.05, 0.0004, 0.01, -0.98, 0.0004)
    upper: tuple[float, ...] = (15., 0.50, 3.0, 0.98, 0.50)

    def __post_init__(self) -> None:
        if self.objective not in ("price", "vega_scaled_price", "iv"):
            raise ValueError("objective must be price, vega_scaled_price, or iv")
        if isinstance(self.n_starts, bool) or not isinstance(self.n_starts, int) or self.n_starts < 1:
            raise ValueError("n_starts must be a positive integer")
        if isinstance(self.max_nfev, bool) or not isinstance(self.max_nfev, int) or self.max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer")
        if len(self.lower) != 5 or len(self.upper) != 5:
            raise ValueError("bounds require five components")
        if any(not math.isfinite(x) for x in (*self.lower, *self.upper)):
            raise ValueError("finite bounds required")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("each lower bound must be less than upper bound")
        if self.lower[0] <= 0 or min(self.lower[1], self.lower[2], self.lower[4]) < 0:
            raise ValueError("invalid Heston parameter lower bounds")
        if not (-1 < self.lower[3] < self.upper[3] < 1):
            raise ValueError("rho bounds must lie strictly inside (-1,1)")
        if self.vega_floor_frac_spot <= 0 or not math.isfinite(self.vega_floor_frac_spot):
            raise ValueError("vega_floor_frac_spot must be positive")


@dataclass(frozen=True)
class StartDiagnostic:
    start: tuple[float, ...]
    fit: tuple[float, ...]
    cost: float
    success: bool
    message: str
    evaluations: int


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    objective: str
    success: bool
    optimizer_message: str
    market_rmse_price: float
    market_rmse_iv: float
    price_bias: float
    iv_bias: float
    objective_rmse: float
    jacobian_condition_number: float
    n_quotes: int
    n_expiries: int
    n_starts: int
    market_prices: np.ndarray = field(repr=False)
    model_prices: np.ndarray = field(repr=False)
    market_ivs: np.ndarray = field(repr=False)
    model_ivs: np.ndarray = field(repr=False)
    identifiers: tuple[str, ...] = ()
    starts: tuple[StartDiagnostic, ...] = ()
    assumptions: str = "European exercise; flat continuously compounded rate and dividend yield"

    @property
    def params_dict(self) -> dict[str, float]:
        p = self.params
        return dict(kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho, v0=p.v0)


def _as_parameters(x: np.ndarray) -> HestonParams:
    return HestonParams(*[float(i) for i in x])


def _initial_points(config: CalibrationConfig, initial: HestonParams | None) -> list[np.ndarray]:
    low = np.asarray(config.lower, dtype=float)
    high = np.asarray(config.upper, dtype=float)
    default = HestonParams(kappa=2., theta=0.04, xi=0.5, rho=-0.6, v0=0.04)
    x0 = np.array(list(vars(initial or default).values()), dtype=float)
    if np.any(x0 <= low) or np.any(x0 >= high):
        raise ValueError("initial parameters must be strictly inside calibration bounds")
    result = [x0]
    rng = np.random.default_rng(config.seed)
    # Start points span the admissible space. Reproducible and explicit, not
    # claimed to prove global optimality or parameter identification.
    for _ in range(config.n_starts - 1):
        x = low + rng.uniform(0.08, 0.92, size=5)*(high-low)
        result.append(x)
    return result


def calibrate_heston(market: EquityMarket, quotes: Sequence[VanillaQuote], *,
                     config: CalibrationConfig | None = None,
                     initial: HestonParams | None = None,
                     exercise_style: str = "european",
                     allow_american_proxy: bool = False) -> CalibrationResult:
    """Multi-start nonlinear least-squares calibration.

    For American-style quotes (common for ETF options) caller must explicitly
    set ``exercise_style='american'`` *and* ``allow_american_proxy=True``.
    Such output is a **European proxy fit**, not an exercise-adjusted valuation.
    Optimizer success does not imply unique identification or good out-of-sample fit.
    """
    if exercise_style not in ("european", "american"):
        raise ValueError("exercise_style must be european or american")
    if exercise_style == "american" and not allow_american_proxy:
        raise ValueError("American option quotes need an exercise-aware pricer; "
                         "set allow_american_proxy=True only for explicitly labelled exploration")
    config = config or CalibrationConfig()
    quotes = tuple(quotes)
    if len(quotes) < 6:
        raise ValueError("calibration requires >= 6 distinct liquid quotes for 5 Heston parameters")
    if len({(q.strike, q.maturity, q.is_call) for q in quotes}) != len(quotes):
        raise ValueError("duplicate strike/maturity/type quotes")
    k = np.asarray([q.strike for q in quotes], float)
    t = np.asarray([q.maturity for q in quotes], float)
    ic = np.asarray([q.is_call for q in quotes], bool)
    price = np.asarray([q.mid for q in quotes], float)
    market_iv = np.array([implied_vol(q.mid, market, q.strike, q.maturity, q.is_call)
                          for q in quotes])
    if np.any((market_iv < 0.015) | (market_iv > 3)):
        raise ValueError("IV outside calibration range [1.5%, 300%]; filter quotes first")
    market_vega = np.asarray([black_vega(market, q.strike, q.maturity, iv)
                              for q, iv in zip(quotes, market_iv)])
    vega_denom = np.maximum(market_vega, config.vega_floor_frac_spot*market.spot)
    # Fix quote weights outside optimization: model cannot adjust its own weights.
    spreads = np.array([max(1e-4, q.ask-q.bid) if q.bid is not None else np.nan
                        for q in quotes])
    if np.isfinite(spreads).any():
        median_spread = float(np.nanmedian(spreads))
        spread_weight = np.where(np.isfinite(spreads),
                                 np.sqrt(median_spread/np.maximum(spreads, median_spread*0.25)), 1.0)
        spread_weight = np.clip(spread_weight, 0.25, 2.)
    else:
        spread_weight = np.ones_like(price)

    def predictions(x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return heston_surface_prices(market, _as_parameters(x), k, t, ic,
                                         n_nodes=config.n_nodes,
                                         integration_upper=config.integration_upper)

    def model_ivs(pred: np.ndarray) -> np.ndarray:
        out = np.empty(len(pred))
        for j, q in enumerate(quotes):
            try:
                out[j] = implied_vol(float(pred[j]), market, q.strike, q.maturity, q.is_call)
            except ValueError:
                out[j] = np.nan
        return out

    def residuals(x: np.ndarray) -> np.ndarray:
        try:
            pred = predictions(x)
            if config.objective == "price":
                err = (pred-price)/market.spot
            elif config.objective == "vega_scaled_price":
                err = (pred-price)/vega_denom
            else:
                ivs = model_ivs(pred)
                err = ivs-market_iv
            if not np.all(np.isfinite(err)):
                return np.full(len(price), 10.)
            return spread_weight*err
        except (FloatingPointError, ArithmeticError, ValueError):
            return np.full(len(price), 10.)

    solutions = []
    starts = []
    for x0 in _initial_points(config, initial):
        fit = least_squares(residuals, x0, bounds=(config.lower, config.upper),
                            method="trf", x_scale="jac", max_nfev=config.max_nfev,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        solutions.append(fit)
        starts.append(StartDiagnostic(tuple(x0), tuple(fit.x), float(fit.cost),
                                      bool(fit.success), str(fit.message), int(fit.nfev)))
    # Always select smallest finite cost, even if run terminated at max_nfev.
    fit = min(solutions, key=lambda f: float(f.cost) if math.isfinite(float(f.cost)) else math.inf)
    pred = predictions(fit.x)
    iv_pred = model_ivs(pred)
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(iv_pred)):
        raise RuntimeError("best fitted parameters yield invalid European prices/IVs")
    s = np.linalg.svd(fit.jac, compute_uv=False)
    cond = (float(s[0]/s[-1]) if len(s) == 5 and s[-1] > 1e-12*s[0]
            else math.inf)
    assumptions = ("European exercise" if exercise_style == "european" else
                   "AMERICAN QUOTES FIT WITH EUROPEAN HESTON PROXY; NO EXERCISE PREMIA")
    assumptions += "; flat user-supplied rate and continuous dividend yield"
    return CalibrationResult(
        params=_as_parameters(fit.x), objective=config.objective,
        success=bool(fit.success), optimizer_message=str(fit.message),
        market_rmse_price=float(np.sqrt(np.mean((pred-price)**2))),
        market_rmse_iv=float(np.sqrt(np.mean((iv_pred-market_iv)**2))),
        price_bias=float(np.mean(pred-price)), iv_bias=float(np.mean(iv_pred-market_iv)),
        objective_rmse=float(np.sqrt(np.mean(residuals(fit.x)**2))),
        jacobian_condition_number=cond, n_quotes=len(quotes),
        n_expiries=len(set(t)), n_starts=len(starts),
        market_prices=price, model_prices=pred, market_ivs=market_iv,
        model_ivs=iv_pred, identifiers=tuple(q.identifier for q in quotes),
        starts=tuple(starts), assumptions=assumptions)


@dataclass(frozen=True)
class FitEvaluation:
    n_quotes: int
    rmse_price: float
    rmse_iv: float
    mean_error_price: float
    mean_error_iv: float
    bid_ask_coverage: float | None


def evaluate_heston_fit(market: EquityMarket, params: HestonParams,
                        quotes: Sequence[VanillaQuote], *,
                        n_nodes: int = 128, integration_upper: float = 200.) -> FitEvaluation:
    """Evaluate any nonempty holdout quote set without refitting parameters."""
    quotes = tuple(quotes)
    if not quotes:
        raise ValueError("nonempty holdout quote set required")
    p = heston_surface_prices(market, params, [x.strike for x in quotes],
                              [x.maturity for x in quotes], [x.is_call for x in quotes],
                              n_nodes=n_nodes, integration_upper=integration_upper)
    y = np.array([x.mid for x in quotes])
    iv_model = np.array([implied_vol(float(px), market, q.strike, q.maturity, q.is_call)
                         for px, q in zip(p, quotes)])
    iv_obs = np.array([implied_vol(q.mid, market, q.strike, q.maturity, q.is_call)
                       for q in quotes])
    quoted = [(float(px), q.bid, q.ask) for px, q in zip(p, quotes) if q.bid is not None]
    coverage = (float(np.mean([bid <= px <= ask for px, bid, ask in quoted]))
                if quoted else None)
    return FitEvaluation(n_quotes=len(quotes),
                         rmse_price=float(np.sqrt(np.mean((p-y)**2))),
                         rmse_iv=float(np.sqrt(np.mean((iv_model-iv_obs)**2))),
                         mean_error_price=float(np.mean(p-y)),
                         mean_error_iv=float(np.mean(iv_model-iv_obs)),
                         bid_ask_coverage=coverage)

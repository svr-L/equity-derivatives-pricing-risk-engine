"""Point-in-time Heston calibration panels from caller-prepared daily slices.

No data are downloaded or internally imputed. All dates, quotes and curves
are supplied by caller. No backfill using later snapshots is performed.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Sequence
import math

import numpy as np

from .carry import EquityCarryCurves
from .calibration import (VanillaQuote, CalibrationConfig, CalibrationResult,
                          FitEvaluation, calibrate_heston, evaluate_heston_fit)
from .models import EquityMarket, HestonParams


@dataclass(frozen=True)
class DatedCalibrationSlice:
    """One valuation date, with quotes and same-day market/curve inputs.

    European exercise is required unless American proxy is explicitly enabled.
    """
    asof: str
    ticker: str
    market: EquityMarket
    curves: EquityCarryCurves
    quotes: tuple[VanillaQuote, ...]
    exercise_style: str = "european"
    allow_american_proxy: bool = False

    def __post_init__(self) -> None:
        try:
            parsed=date.fromisoformat(self.asof)
        except ValueError as exc:
            raise ValueError("invalid slice asof date") from exc
        if parsed.isoformat()!=self.asof or self.curves.asof!=self.asof:
            raise ValueError("slice and curve valuation dates must match")
        if not self.ticker.strip():
            raise ValueError("ticker is required")
        self.curves.validate_market(self.market)
        if len(self.quotes)<6:
            raise ValueError("at least six quotes per slice are required")
        if self.exercise_style not in ("european", "american"):
            raise ValueError("exercise_style must be european or american")
        if self.exercise_style=="american" and not self.allow_american_proxy:
            raise ValueError("American quote calibration requires explicit proxy opt-in")


@dataclass(frozen=True)
class DatedCalibrationResult:
    asof: str
    ticker: str
    calibration: CalibrationResult
    next_date_cross_section: FitEvaluation | None = None
    next_asof: str | None = None


def calibrate_history(slices: Sequence[DatedCalibrationSlice], *,
                      config: CalibrationConfig | None = None,
                      initial: HestonParams | None = None,
                      warm_start: bool = True,
                      evaluate_next_date: bool = True) -> tuple[DatedCalibrationResult, ...]:
    """Calibrate each date using its own quotes and its own dated curves.

    Optional next-date evaluation transfers PREVIOUS fitted *parameters* to
    NEXT date's observed spot/curves and quotes. This is a cross-sectional
    parameter-transfer diagnostic, NOT a tradable price forecast or P&L.
    """
    observations=tuple(slices)
    if not observations:
        raise ValueError("nonempty historical panel required")
    for a,b in zip(observations, observations[1:]):
        if a.asof >= b.asof:
            raise ValueError("dates must be unique and strictly ascending; no future leakage")
    if len({a.ticker for a in observations})!=1:
        raise ValueError("each historical panel must contain exactly one ticker")
    cfg=config or CalibrationConfig()
    output=[]
    current=initial
    for i,sl in enumerate(observations):
        result=calibrate_heston(sl.market,sl.quotes,config=cfg,initial=current,
                                exercise_style=sl.exercise_style,
                                allow_american_proxy=sl.allow_american_proxy,
                                carry_curves=sl.curves)
        next_eval=None
        next_asof=None
        if evaluate_next_date and i+1<len(observations):
            nx=observations[i+1]
            next_eval=evaluate_heston_fit(nx.market,result.params,nx.quotes,
                                           carry_curves=nx.curves,
                                           n_nodes=cfg.n_nodes,
                                           integration_upper=cfg.integration_upper)
            next_asof=nx.asof
        output.append(DatedCalibrationResult(sl.asof,sl.ticker,result,next_eval,next_asof))
        if warm_start:
            if result.success:
                current=result.params
        else:
            current=initial
    return tuple(output)


def history_summary(results: Sequence[DatedCalibrationResult]) -> list[dict]:
    """JSON-safe flat per-date diagnostics; no arbitrary threshold for stability."""
    rows=[]
    for x in results:
        c=x.calibration
        e=x.next_date_cross_section
        rows.append(dict(asof=x.asof,ticker=x.ticker,success=c.success,
                         n_quotes=c.n_quotes,iv_rmse=c.market_rmse_iv,
                         price_rmse=c.market_rmse_price,
                         jacobian_condition_number=(c.jacobian_condition_number
                           if math.isfinite(c.jacobian_condition_number) else None),
                         params=c.params_dict,assumptions=c.assumptions,
                         next_asof=x.next_asof,
                         next_date_iv_rmse=(e.rmse_iv if e is not None else None),
                         next_date_bid_ask_coverage=(e.bid_ask_coverage if e is not None else None)))
    return rows

"""Adapter for the user's daily Yahoo option-surface collector (schema v1).

Pandas and (for Parquet) pyarrow/fastparquet are *optional*. Historical files
are selected by embedded timestamp, not stale 'latest.csv.gz' aliases. All
curve assumptions are supplied explicitly by the caller.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import VanillaQuote
from .models import EquityMarket
from .vanilla import implied_vol


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("option snapshot adapter requires pip install '.[marketdata]'") from exc
    return pd


@dataclass(frozen=True)
class QuoteFilterConfig:
    min_dte: int = 21
    max_dte: int = 365
    min_bid: float = 0.05
    min_mid: float = 0.10
    min_open_interest: int = 100
    max_relative_spread: float = 0.25
    max_abs_log_moneyness: float = 0.25
    min_iv: float = 0.04
    max_iv: float = 2.0
    max_expiries: int = 8
    max_quotes_per_expiry_per_side: int = 6


@dataclass(frozen=True)
class QuoteFilterReport:
    ticker: str
    snapshot_timestamp: str
    original_rows: int
    after_basic_filters: int
    after_european_iv_checks: int
    selected_quotes: int
    selected_expiries: int
    rejected_iv: int
    exercise_style: str
    caveat: str


def _read_bytes(content: bytes, name: str):
    pd = _pandas()
    if name.lower().endswith(".parquet"):
        try:
            return pd.read_parquet(io.BytesIO(content))
        except ImportError as exc:
            raise ImportError("Parquet requires 'pyarrow' or 'fastparquet'. Install '.[marketdata]' ") from exc
    if name.lower().endswith(".csv.gz"):
        return pd.read_csv(io.BytesIO(content), compression="gzip")
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    raise ValueError("snapshot must be .parquet, .csv.gz, or .csv")


def load_snapshot(path: str | Path):
    """Read one timestamped collector snapshot, CSV or Parquet."""
    p = Path(path)
    return _read_bytes(p.read_bytes(), p.name)


def list_archive_snapshots(archive: str | Path, ticker: str) -> tuple[str, ...]:
    """List timestamped snapshots in ascending order; never use latest aliases."""
    code = ticker.replace("^", "INDEX_").upper()
    pat = re.compile(rf"/(?:{re.escape(code)})_(\d{{8}}T\d{{6}}Z)\.(?:parquet|csv\.gz|csv)$", re.I)
    with zipfile.ZipFile(archive) as z:
        found = [n for n in z.namelist() if pat.search('/'+n)]
    return tuple(sorted(found, key=lambda n: re.search(r'(\d{8}T\d{6}Z)', n).group(1)))


def load_archive_snapshot(archive: str | Path, ticker: str, *,
                          date: str | None = None, timestamp: str | None = None):
    """Read the newest dated snapshot for ticker (or YYYY-MM-DD / exact stamp).

    Does not silently fall back to an older CSV if the latest Parquet cannot
    be opened: that would introduce accidental look-ahead / staleness errors.
    """
    found = list_archive_snapshots(archive, ticker)
    if date is not None:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
            raise ValueError("date must be YYYY-MM-DD")
        found = tuple(n for n in found if re.search(r'(\d{8})T', n).group(1) == date.replace('-', ''))
    if timestamp is not None:
        found = tuple(n for n in found if f'_{timestamp}.' in n)
    if not found:
        raise FileNotFoundError(f"no snapshot for {ticker=} {date=} {timestamp=}")
    name = found[-1]
    with zipfile.ZipFile(archive) as z:
        return _read_bytes(z.read(name), name), name


def filter_option_quotes(frame, market: EquityMarket, *,
                         config: QuoteFilterConfig | None = None,
                         exercise_style: str = "american",
                         allow_american_proxy: bool = False
                         ) -> tuple[list[VanillaQuote], QuoteFilterReport]:
    """Quality screen and stratified OTM selection for one dated collector snapshot.

    ``exercise_style='american'`` is the safe default for Yahoo ETF quotes.
    Requiring explicit allow_american_proxy prevents treating their implied
    volatilities as precise European-market quotes by accident.
    """
    pd = _pandas()
    config = config or QuoteFilterConfig()
    if exercise_style not in ("european", "american"):
        raise ValueError("exercise_style must be 'european' or 'american'")
    if exercise_style == "american" and not allow_american_proxy:
        raise ValueError("American ETF option quotes are not European. Set "
                         "allow_american_proxy=True for an exploratory proxy calibration")
    required = ("underlying_ticker", "snapshot_ts_utc", "underlying_spot", "option_type",
                "strike", "expiry", "dte_calendar", "bid", "ask", "openInterest")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"collector schema missing: {missing}")
    if frame.empty:
        raise ValueError("empty option snapshot")
    x = frame.copy()
    tickers = x.underlying_ticker.dropna().astype(str).unique()
    stamps = pd.to_datetime(x.snapshot_ts_utc, utc=True, errors="coerce").dropna().unique()
    if len(tickers) != 1 or len(stamps) != 1:
        raise ValueError("filter expects exactly one ticker and one timestamp")
    ticker = tickers[0]
    stamp = str(stamps[0])
    spot = pd.to_numeric(x.underlying_spot, errors="coerce")
    if not (np.isfinite(spot).all() and np.allclose(spot, market.spot, rtol=1e-7, atol=1e-7)):
        raise ValueError("provided market spot must match the archived snapshot spot")
    for col in ("bid", "ask", "strike", "dte_calendar", "openInterest"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    original_n = len(x)
    t = x.dte_calendar/365.0  # ACT/365F calendar-days convention, not session time
    fwd = market.spot*np.exp((market.rate-market.dividend_yield)*t)
    lm = np.log(x.strike/fwd)
    is_call = x.option_type.eq("call")
    is_put = x.option_type.eq("put")
    # Use *forward* moneyness for OTM filtering, not spot moneyness.
    otm = (is_call & (lm >= 0)) | (is_put & (lm < 0))
    mid = (x.bid+x.ask)/2.0
    relative_spread = (x.ask-x.bid)/mid
    good = (x.strike.gt(0) & x.dte_calendar.between(config.min_dte, config.max_dte)
            & x.bid.ge(config.min_bid) & x.ask.ge(x.bid)
            & mid.ge(config.min_mid) & x.openInterest.ge(config.min_open_interest)
            & relative_spread.le(config.max_relative_spread)
            & lm.abs().le(config.max_abs_log_moneyness) & otm)
    x = x.loc[good].copy()
    x["mid_recomputed"] = mid.loc[x.index]
    x["rel_spread"] = relative_spread.loc[x.index]
    x["abs_log_forward_moneyness"] = lm.loc[x.index].abs()
    x["maturity"] = t.loc[x.index]
    # Guard against duplicate rows for the same contract (raise rather than
    # silently average duplicates from disparate snapshots).
    if x.duplicated(subset=["expiry", "strike", "option_type"]).any():
        raise ValueError("duplicate option quote in snapshot")
    basic = len(x)
    # IV reconstructed internally, never Yahoo impliedVolatility; European
    # bounds are a *proxy* test on American quotes, not exercise-price adjustment.
    ivs = []
    for row in x.itertuples():
        try:
            iv = implied_vol(float(row.mid_recomputed), market, float(row.strike),
                             float(row.maturity), row.option_type == "call")
        except (ValueError, OverflowError):
            iv = math.nan
        ivs.append(iv)
    x["recomputed_iv"] = ivs
    x = x.loc[x.recomputed_iv.between(config.min_iv, config.max_iv)].copy()
    after_iv = len(x)
    if x.empty:
        raise ValueError("no quote survived quality/European proxy IV filters")
    # Preserve maturity coverage rather than allowing very short dated quotes
    # to crowd the entire fit. Each expiry gets both wings where available.
    expiries = sorted(x.expiry.unique())
    if len(expiries) > config.max_expiries:
        idx = np.unique(np.round(np.linspace(0, len(expiries)-1, config.max_expiries)).astype(int))
        selected_expiries = [expiries[i] for i in idx]
    else:
        selected_expiries = expiries
    x = x.loc[x.expiry.isin(selected_expiries)]
    selected = []
    for (expiry, right), group in x.groupby(["expiry", "option_type"], sort=True):
        # Rank by relative spread then proximity to forward ATM.
        group = group.sort_values(["rel_spread", "abs_log_forward_moneyness"])
        selected.append(group.head(config.max_quotes_per_expiry_per_side))
    chosen = pd.concat(selected, ignore_index=True).sort_values(["expiry", "option_type", "strike"])
    quotes = [VanillaQuote(strike=float(row.strike), maturity=float(row.maturity),
                           mid=float(row.mid_recomputed), is_call=row.option_type == "call",
                           bid=float(row.bid), ask=float(row.ask),
                           identifier=str(getattr(row, "contractSymbol", "")))
              for row in chosen.itertuples()]
    caveat = ("AMERICAN ETF OPTIONS: EUROPEAN PROXY ONLY; NO EARLY-EXERCISE ADJUSTMENT"
              if exercise_style == "american" else
              "European exercise assumed; curve inputs and quote timestamps must be validated")
    return quotes, QuoteFilterReport(ticker=ticker, snapshot_timestamp=stamp,
                                    original_rows=original_n, after_basic_filters=basic,
                                    after_european_iv_checks=after_iv,
                                    selected_quotes=len(quotes),
                                    selected_expiries=len(set(q.maturity for q in quotes)),
                                    rejected_iv=basic-after_iv,
                                    exercise_style=exercise_style, caveat=caveat)


@dataclass(frozen=True)
class StaticArbitrageReport:
    n_slices: int
    n_adjacent_pairs: int
    n_butterflies: int
    mid_monotonicity_violations: int
    executable_monotonicity_violations: int
    mid_convexity_violations: int
    executable_convexity_violations: int


def static_arbitrage_report(frame, *, tolerance: float = 1e-8) -> StaticArbitrageReport:
    """Strike monotonicity / convexity diagnostics on bid/ask chains.

    The executable checks use conservative *crossed* bid/ask inequalities,
    not midpoint deviations alone. These are isolated strike checks; no claim
    of exhaustive cross-maturity or dynamic arbitrage validation is made.
    """
    pd = _pandas()
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    for c in ("expiry", "option_type", "strike", "bid", "ask"):
        if c not in frame.columns:
            raise ValueError(f"missing column: {c}")
    x = frame.copy()
    for c in ("strike", "bid", "ask"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.loc[x.strike.gt(0) & x.bid.ge(0) & x.ask.ge(x.bid)]
    x = x.drop_duplicates(["expiry", "option_type", "strike"], keep=False)
    slices = pairs = butterflies = mm = em = mc = ec = 0
    for (expiry, typ), g in x.groupby(["expiry", "option_type"]):
        if typ not in ("call", "put"):
            continue
        g = g.sort_values("strike")
        if len(g) < 2:
            continue
        slices += 1
        k = g.strike.to_numpy(float)
        b = g.bid.to_numpy(float)
        a = g.ask.to_numpy(float)
        mid = (b+a)/2
        iscall = typ == "call"
        pairs += len(g)-1
        if iscall:
            mm += int(np.count_nonzero(np.diff(mid) > tolerance))
            em += int(np.count_nonzero(b[1:]-a[:-1] > tolerance))
        else:
            mm += int(np.count_nonzero(np.diff(mid) < -tolerance))
            em += int(np.count_nonzero(b[:-1]-a[1:] > tolerance))
        if len(g) >= 3:
            butterflies += len(g)-2
            w = (k[2:]-k[1:-1])/(k[2:]-k[:-2])
            # A convex price curve lies below the interpolated outer strikes.
            middle_mid = mid[1:-1]
            outer_mid = w*mid[:-2]+(1-w)*mid[2:]
            outer_ask = w*a[:-2]+(1-w)*a[2:]
            mc += int(np.count_nonzero(middle_mid-outer_mid > tolerance))
            ec += int(np.count_nonzero(b[1:-1]-outer_ask > tolerance))
    return StaticArbitrageReport(slices, pairs, butterflies, mm, em, mc, ec)

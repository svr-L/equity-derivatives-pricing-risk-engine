# Market-data snapshot adapter — v0.3.0

Supports the provided `option_surface_collector.py` **schema version 1**, with
columns `underlying_ticker`, `snapshot_ts_utc`, `underlying_spot`, `expiry`,
`option_type`, `strike`, `dte_calendar`, `bid`, `ask`, `openInterest`, and
optionally `contractSymbol`.

## Loading

- `load_snapshot(path)` reads timestamped `.csv`, `.csv.gz` or `.parquet`.
- `list_archive_snapshots(zip_path, ticker)` enumerates timestamped snapshots.
- `load_archive_snapshot(zip_path, ticker, date='2026-10-27')` selects the
  **latest timestamp for that calendar date**, ignoring stale root-level
  `latest.csv.gz`/`latest.parquet` aliases.
- `load_archive_snapshot(zip_path, ticker)` selects the latest timestamp in
  the archive, independently of file format. If it is Parquet and the
  optional Parquet engine is unavailable, it raises: **never silently**
  substitutes an older CSV.

`pandas` and `pyarrow` are optional market-data dependencies. The core pricers
and Heston calibrator have no pandas/pyarrow dependency.

## Default quote selection

1. One ticker and timestamp; spot must match `EquityMarket.spot`.
2. DTE between 21 and 365, positive bid and coherent bid <= ask, midpoint
   >= 0.10, open interest >= 100, relative spread <= 25%.
3. Forward OTM options, `abs(log(K/F)) <= 0.25`, with the **user-supplied**
   forward `F = S exp((r-q)T)` and ACT/365F calendar-day maturity.
4. Recalculate European Black IV from the midpoint (ignore Yahoo's supplied
   `impliedVolatility`) and keep 4%-200% quotes passing inversion.
5. Preserve expiry coverage (default up to 8 evenly spaced available
   expiries), choosing up to six calls and six puts per expiry by spread
   quality and nearness to ATM-forward.

These defaults are **illustrative and configurable**, not a statistically
optimal liquidity filter. No interpolation/smoothing or full no-arbitrage
surface repair is performed. The `static_arbitrage_report` function checks
strike monotonicity and convexity on midpoints, plus stricter crossed
bid/ask inequalities consistent with potentially executable violations.
It does not validate all possible butterfly structures, calendar arbitrage,
or dynamic consistency; raw Yahoo quotes can be sparse/stale.

## Licensing/recordkeeping

This repository does **not** redistribute the user's raw Yahoo price history.
The empirical demonstration reads the separate supplied ZIP locally and logs
ticker, dated snapshot, filtration, assumed rates and optimization settings.
Treat the output as exploratory analysis, not a recommendation to trade.

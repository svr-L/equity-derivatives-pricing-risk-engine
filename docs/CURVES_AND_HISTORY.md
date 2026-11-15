# v0.4.0 — Dated discount/forward curves and chronological Heston diagnostics

## Why this matters

The v0.3.0 European calibration assumed user-supplied **flat** continuously
compounded `r` and proportional dividend yield `q`. This is still available;
**it has not been replaced**. The v0.4.0 **adds** term-structured inputs where
quotes and curves share an explicit valuation date. The new objects do not
infer curves from Yahoo or reinterpret US ETF options as European contracts.

## Curve conventions

- `DiscountCurve(tenors, dfs)`: strictly increasing years `t>0`, positive
  discount factors `D(t)`; injects `D(0)=1`; logarithmic interpolation of DFs.
  Negative interest rates are permitted. No extrapolation past final tenor.
- `EquityCarryCurves(spot, discount_curve, dividend_curve, asof)`: second curve
  is a **proportional dividend discount factor**, `Q(t)=exp(-integral q(s) ds)`.
  At each tenor, `F(0,t)=S0 Q(t)/D(t)`.
- `EquityCarryCurves.from_forwards(spot, discount_curve, maturities, forwards,
  asof=...)`: derive implied `Q(t)=F(0,t)D(t)/S0` from **supplied** forwards.
  The caller must substantiate the curve data, their quotes and their dates.
- Time inputs use year fractions (ACT/365F convention in the existing Yahoo
  adapter). The `asof` field is a `YYYY-MM-DD` valuation date. The adapter
  compares it to the recorded snapshot **UTC date**. A US trading date can
  differ from UTC late at night; in that case a deliberate mapping policy is
  needed rather than silently accepting a mismatch.
- European vanilla Heston Fourier uses an effective integrated r/q for each
  maturity, mathematically equivalent under deterministic carry. Monte Carlo
  applies interval-specific `log(F(t1)/F(t0))` at each step, avoiding an
  inappropriate constant terminal carry for path-dependent payoffs.
- **Cash dividends are not implemented.** This is a continuous proportional
  carry model; it is not an ETF discrete ex-dividend stochastic-price model,
  and does not price American early exercise.
- `carry_curves=None` preserves all v0.3 numerical behavior and APIs.

## Sample curve JSON (SYTHETIC, not actual observed market curves)

```json
{
  "asof": "2026-10-27",
  "discount": {
    "tenors": [0.25, 0.5, 1.0, 2.0],
    "dfs": [0.98955, 0.97922, 0.95887, 0.92000]
  },
  "forwards": {
    "tenors": [0.25, 0.5, 1.0, 2.0],
    "values": [100.5, 101.0, 102.1, 104.0]
  }
}
```

`values` in this example assume S0=100 and are **illustrative only**. The
same fields can be used for SPY only after replacing them with authentic
same-date market discount factors and forwards, for that SPY spot. Longer
maturities must be covered by the curve. Data-source provenance is the
caller's responsibility. No interpolation/extrapolation of stock forwards
outside explicit `tenors` is allowed.

```bash
python examples/calibrate_snapshot.py 'IV Data(3).zip' \
  --ticker SPY --date 2026-10-27 \
  --curves-json verified_curve_inputs.json \
  --exercise-style american --allow-american-proxy \
  --output spy_curve_proxy_diagnostics.json
```

`--curves-json` is mutually exclusive with the legacy flat `--rate` /
`--dividend-yield` switches. American proxy opt-in remains mandatory.

## APIs

```python
from equity_derivatives import (
    DiscountCurve, EquityCarryCurves, EquityMarket, HestonParams,
    heston_european, heston_surface_prices, price_mc, calibrate_heston,
)

market = EquityMarket(spot=100.)
D = DiscountCurve.flat(0.025, 2.)
Q = DiscountCurve.flat(0.015, 2.)
curves = EquityCarryCurves(100., D, Q, asof="2026-10-27")
p = HestonParams(2., .04, .5, -.6, .04)
value = heston_european(market, p, 100, 1., carry_curves=curves)
```

Optional `carry_curves` is accepted by `heston_surface_prices`,
`simulate_heston`, `price_mc`, `bump_greeks`, `calibrate_heston`,
`evaluate_heston_fit` and `market_data.filter_option_quotes`.
For bump spot Greeks, the whole proportional-forward curve scales with spot,
**holding proportional yields fixed**; it is NOT a sensitivity with fixed
absolute market forwards, recalibration, or risk-free curve movements.

## Chronological surface history

Use `DatedCalibrationSlice` / `calibrate_history` with **ascending** dated
slices, each with contemporaneous spot, quotes and curves. The function
rejects duplicate/out-of-order dates, nonmatching curve as-of and mixed
tickers. `warm_start=True` carries yesterday's successful calibration as the
initial guess; it never uses future data for earlier fits.

`next_date_cross_section` evaluates **previous date's fitted parameters** on
**next date's observed spot, curves and options**. This is a *parameter
transfer / stability* diagnostic; because next day's spot and term
structure are observed ex-post, it is neither an executable trading signal
nor a next-day option-price forecast. It does **not** automatically process
all files in a ZIP: date-specific snapshots and verified curves must be
supplied to the slices. The module exports `history_summary()` for JSON-safe
per-date summaries. If no spreads are quoted, bid/ask coverage is `None`.

## Validation and known open tasks

- Release tests include log-linear interpolation, flat-curve backward
  equivalence, multi-tenor forward reconstruction, simulation carry applied
  at every Asian fixing, synthetic calibration, Greek spot-bump semantics,
  curve/snapshot date guards, chronological calibration, and next-date
  parameter-transfer diagnostics.
- The supplied Yahoo archive offers chains, spot and dates, **not verified
  historical OIS/repo curves and forward/dividend term structure**. The
  v0.3 SPY calibration with assumed r/q is NOT upgraded by merely enabling
  an adapter.
- American ETF early-exercise premia, discrete dividends, calibration bid/ask
  coverage, numerical Fourier/QE bias, market cross-date quote staleness,
  and market-model misspecification remain separate limitations.

# Reproducible SPY exploration on the supplied archive (not market-validated)

**Snapshot:** `SPY_20261027T230605Z.csv.gz`, timestamp
2026-10-27 23:06:05.566847 UTC, spot 769.659973. The original ZIP was
`IV Data(3).zip`. The October 27 CSV is used because Parquet support was not
available in the verification environment; the Nov 10 Parquet data are
supported by the adapter but **were not quote-validated here**.

**Crucial assumptions:** r=4% per year continuously compounded, continuous
dividend yield q=1%. These are **illustrative input values**, not verified
as-of-2026-10-27 yield curves. SPY options are generally American-style, but
this run explicitly uses the `--allow-american-proxy` European Heston fitting
approximation without an exercise premium adjustment.

Run:

```bash
python examples/calibrate_snapshot.py 'IV Data(3).zip' --ticker SPY \
  --date 2026-10-27 --rate 0.04 --dividend-yield 0.01 \
  --exercise-style american --allow-american-proxy \
  --output calibration_spy_exploratory.json
```

### Observation & fit

| Diagnostic | Result |
|---|---:|
| Total snapshot rows | 7,984 |
| Rows after basic filters | 1,608 |
| Selected liquid OTM quotes / expiries | 96 / 8 |
| Train / within-snapshot holdout quotes | 72 / 24 |
| Multi-start | 4 starts, all converged |
| Train IV RMSE | 0.4003 IV percentage points |
| Held-out IV RMSE | 0.4074 IV percentage points |
| Train / held-out price RMSE | 0.6176 / 0.6446 USD per share |
| Held-out predictions inside bid/ask | **1 / 24** (4.17%) |
| Jacobian condition-number proxy | 3,235 (parameter-unit dependent) |
| Fit kappa, theta, xi, rho, v0 | 3.3418, 0.0550, 1.3483, -0.4889, 0.02323 |

Among the 96 selected quotes, the implemented strike-only checks found
**no executable monotonicity or convexity bid/ask violations**, and one
midpoint convexity deviation. The full raw chain had numerous flagged
adjacent-strike checks, illustrating why unfiltred data are unsuitable.

### Interpretation

The model finds a fairly smooth approximate IV fit across selected strikes,
but it does **not** accurately fit most observed bid/ask ranges. This and the
American-versus-European mismatch are material limitations, despite the
apparently small IV RMSE. The final parameters should **not** be used for
real exotic valuations or trading without proper market forwards/rates,
European-style instrument quotes, proper timing, and out-of-time validation.

The held-out strikes share the calibration date and maturities, so this is
**cross-sectional**, not temporal, out-of-sample evidence.

For more exact numerical error analysis, compare against adaptive scalar
Fourier integration with increased quadrature settings: the local diagnostic
run found a worst-case absolute European price difference of approximately
0.0043 per share between 96-node/180 and 160-node/220 batch settings on the
96 selected quotes, showing that numerical integration convergence is not
zero, but is smaller than the observed ~0.6 price RMSE in this example.

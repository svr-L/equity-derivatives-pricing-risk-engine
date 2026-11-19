# Validation record — v0.2.0 (2026-11-10)

## Execution environment and reproducibility

Local execution: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
pytest 9.0.2. Source archives of FX/FI were not modified.

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m compileall -q src tests examples
python examples/price_exotics.py
python examples/compare_schemes.py --csv docs/benchmark_results.csv
python examples/barrier_validation.py
python examples/heston_barrier_grid.py
```

The suite passed **59/59 tests locally** and includes **all v0.1 tests unchanged** plus v0.2 regression tests
for QE-M and continuous barriers. The test outcomes are recorded below;
GitHub Actions has not been run externally. A local `ruff` lint
pass was not possible because `ruff` was not installed in the runtime. The v0.2 code is a *research
numerical engine*, not independently audited production code.

## Vanilla Heston: Euler vs Andersen QE-M

Case: spot=100, rate=1%, continuous dividend yield=2%, kappa=2,
theta=v0=0.04, xi=0.5, rho=-0.7, T=1, strike=100;
12000 paths, antithetic pairs, seed 2026. **Independent Heston Fourier
European benchmark = 6.51384202.** Measured on the local environment:

| scheme | steps | MC price | SE (pairs) | MC minus Fourier | runtime (s) |
|---|---:|---:|---:|---:|---:|
| FT Euler | 24 | 6.548280 | .063589 | +.034438 | .083 |
| FT Euler | 96 | 6.605119 | .062686 | +.091277 | .081 |
| FT Euler | 252 | 6.585242 | .060444 | +.071400 | .236 |
| QE-M | 24 | 6.520022 | .060446 | +.006180 | .166 |
| QE-M | 96 | 6.624961 | .061718 | +.111119 | .158 |
| QE-M | 252 | 6.595116 | .060019 | +.081274 | .759 |

Values are stochastic for fixed RNG seeds. **The six measured deviations
alone do not establish that QE-M is more accurate than FT Euler**: differences
of this magnitude are comparable with MC standard errors. QE-M is locally
slower in this comparison. It is included for its moment-matching/
nonnegative variance and conditional stock martingale structure, not a
claim of universal speed or empirical pricing superiority. Full data and
run durations are in [`benchmark_results.csv`](benchmark_results.csv).
Tracemalloc peak allocations are NOT guaranteed to equal whole-process RSS.

## Independent Black-Scholes continuous-barrier benchmark

For xi=0, theta=v0=.04 (constant annual volatility 20%), spot=100,
strike=100, up-and-out call with barrier=125, T=1, r=1%, q=2%, zero rebate:

- **Exact continuously monitored reflection benchmark: 1.890595**.
- Andersen QE-M, seed 2026, 40000 paths, antithetic, conditional bridge:

| steps | discrete KO MC | bridge KO MC | bridge SE | bridge minus exact |
|---|---:|---:|---:|---:|
| 1 | 3.477549 | 1.868861 | .012286 | -.021734 |
| 4 | 2.968620 | 1.889815 | .016409 | -.000780 |
| 16 | 2.529560 | 1.924907 | .018387 | +.034311 |
| 64 | 2.158234 | 1.853054 | .018855 | -.037541 |

The continuously monitored bridge estimator agrees with the independent
closed-form value within a few SE, while discrete-grid monitoring shows a
visible *upward* monitoring bias. Nonmonotone bridge prices reflect MC noise
from different random sequences at different grid sizes.

## Heston stochastic-variance bridge sensitivity

The script `examples/heston_barrier_grid.py` compares both schemes across
24/96/252 steps and reports discrete and approximate-bridge KO prices, but
**there is no exact continuously monitored Heston barrier benchmark** here.
The residual stochastic-variance bridge error is unquantified: the test
cannot validate exact Heston continuous monitoring or an optimal time step.

## Structural and boundary tests

- QE-M CIR first two conditional moments (both Gaussian-quadratic and
  mixed point-mass/exponential branches).
- Nonnegative QE-M simulated variance even with Feller condition violated,
  and absorbing zero state with `theta=v0=0`.
- Conditional stock martingality (estimated by Monte Carlo), deterministic
  `xi=0` exact variance grid values and Black-Scholes vanilla matching.
- Seed reproducibility, antithetic-pair effective sample size and common
  random numbers for bump Greeks.
- Continuous Black-Scholes barrier benchmark: all up/down, in/out and
  call/put combinations, time-zero touches, deterministic zero volatility,
  zero rebate and parity.
- Bridge survival weights in [0,1], bridge KO <= discrete KO on identical
  paths, and KI+KO exactly reproduces European MC under same shocks.

## Outstanding and known limitations

Neither the Fourier quadrature truncation across extreme strikes and maturities
nor Heston barrier continuous-monitoring error has a rigorous global error
bound. Only risk-neutral *fixed-parameter* Greeks are computed; v0 sensitivity
is not market IV Vega. Asian and lookback sampling dates depend on the grid.
Calibration to observed market vanilla option surfaces, hedging, portfolio
risk analytics, performance tuning and independent QuantLib cross-checks
are outside v0.2.


## v0.3.0 calibration and empirical tests (2026-11-11)

- `python -m pytest -q`: **74 passed** in local environment, preserving all
  59 v0.2 tests.
- `python examples/calibrate_synthetic.py`: input `(kappa=2.5, theta=.05,
  xi=.6, rho=-.65, v0=.045)` recovered by batch-Fourier calibration to
  independently integrated *same-model* scalar Fourier prices: near
  `(2.50085,.04999,.59928,-.65053,.04499)`; within-synthetic held-out IV
  RMSE 0.0036 IV percentage points. This is a numerical recovery test, not
  real-world model-identification evidence.
- SPY October 27 CSV proxy fit: train / held-out IV RMSE 0.4003 / 0.4074
  volatility percentage points, but holdout bid/ask coverage **4.17%** and
  American/European mismatch. Detailed reproduction in `EMPIRICAL_SPY.md`.
- Parquet snapshots are supported with optional `pyarrow`, but the local
  environment could not install it; no latest-Parquet calibration is claimed.
- No release-level cross-date OOS test, historical curve validation,
  calibrated exotic market quoting or independent production-grade validation.

## v0.4.0 term curves and history validation (2026-11-14)

- 86/86 tests: 74 prior v0.3 tests preserved plus 12 added tests.
- Scalar and batch Heston Fourier and both QE-M/FT MC schemes reproduce the
  previous flat-rate results when supplied equivalent flat curves.
- Deterministic zero-variance path with **non-flat** piecewise curves reproduces
  exact intermediate forwards, terminal forwards, Asian average payoff and
  maturity discounting. This checks path carry, not just terminal carry.
- Curve CSV/JSON adapter date mismatch, quote/curve spot mismatch, out-of-range
  tenor errors, validity of forward reconstruction, and histogram chronology
  guards tested. Quote filter can discard expiries beyond curve coverage without
  extrapolating prices for those rejected rows.
- Synthetic per-date Heston calibration and next-date parameter-transfer
  diagnostics tested. This is neither a market fit nor a tradable forecast.
- SPY collector CSV dated 2026-10-27 re-run in **legacy flat/proxy mode**:
  72 train / 24 holdout; IV RMSE 0.4003 / 0.4074 percentage points;
  train bid/ask coverage 2/72, holdout 1/24; exploratory r=4%, q=1%.
  This is **not** a calibration using real historical OIS/forwards.
- Term-structure implementation does not include scheduled cash dividends;
  American ETF exercise not handled; confidence intervals exclude model and
  discretization bias.

## v0.4.1 numerical fixes and independent validation (2026-11-19)

- `python -m pytest -q`: **190 passed** (86 prior + 104 new) in about 7 s.
  Live QuantLib tests need QuantLib >= 1.32 and are skipped otherwise; the
  embedded reference values need no extra package.
- Heston European prices vs QuantLib `AnalyticHestonEngine` (AngledContour
  contour, exp-sinh integration): 150 random parameter sets over the default
  calibration box x 12 maturities (1 day to 10 years) x 7 strikes
  (|log-moneyness| <= 0.5): maximum difference **3.6e-12 x spot**, no accuracy
  warning. QuantLib's exp-sinh integration itself failed (NaN) in 5 of the
  1800 maturity groups, which were excluded.
- 18 reference prices from a 40-digit mpmath Lewis integral: QuantLib agrees
  to <= 1.1e-13, the engine to about 2e-11 or better with the default
  `rel_tol=1e-10`. Andersen (2008) case I gives 13.08467 (paper: 13.0847).
- QuantLib's *default* Heston engine (Gatheral formula + Gauss–Lobatto) is not
  a safe reference: against the 40-digit values it is off by up to 1.9e-5
  (xi=0.01, kappa=15, v0=0.5, T=1) and 3.2e-6 (kappa*T about 118, T=10) on a
  100 spot. The live tests therefore use the AngledContour configuration.
- Calibration-box corners: the automatic cut-off reaches its 2**16 cap and
  warns only for tiny v0 (0.0004) with xi=3 at short expiries (estimated error
  up to about 1e-6 x spot); everywhere else no warning was raised.
- v0.4.0 failure cases, before -> after: 7-day 105 call with xi=2, v0=0.01:
  -0.00715 -> 2.24083e-5 (reference 2.24083e-5); Fourier with xi=1e-9:
  0.84111 -> 7.36429 (BS 20%: 7.36429); QE-M ATM with xi=1e-7 (52 steps,
  100k paths, seed 7): 5.151 -> 7.406 +/- 0.032, identical to the xi=0 run;
  batch ATM price error vs QuantLib: +2.1e-4 -> about 1e-9.
- Monte Carlo tests use several seeds and pooled standard errors: QE-M
  European within 4 SE + 0.005 of the reference at 12 and 24 steps per year;
  constant-volatility bridge unbiased within 4 SE; CRN bump delta, gamma and v0
  sensitivity within 4 SE of QuantLib finite differences; Heston bridge
  up-and-out call at 126 steps within 2% of QuantLib FdHestonBarrierEngine
  (finest grid 4.3886, grid sensitivity about 0.005).
- Performance: batch surface with 64 quotes over 8 expiries about 2 ms (v0.4.0:
  0.8 ms); 65-quote calibration with 4 starts about 0.9 s (v0.4.0: 0.4 s).

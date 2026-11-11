# Changelog

## 0.2.0 — 2026-11-10

**Equity Derivatives Pricing & Risk Engine** (separate standalone repository;
import namespace `equity_derivatives` preserved).

### Added

- Andersen **QE-M** (quadratic-exponential with conditional stock
  martingale correction) as `MCConfig(..., scheme="qe_m")`; variance
  transition matches CIR conditional first two moments. Handles an atom at
  zero and `xi=0` exact deterministic integrated variance.
- Continuous-barrier **conditional Brownian-bridge survival weighting** for
  `Barrier(..., monitoring="continuous_bridge")`. Exact under constant
  Black-Scholes volatility; only **approximate under Heston**.
- Independent exact continuously monitored Black-Scholes barrier benchmark
  `black_scholes_barrier` (zero rebate, up/down, in/out, call/put).
- Validation tests for QE moments, martingality, Feller violation, CRN,
  bridge convergence to exact Black-Scholes and knock-in/out parity.
- Cross-scheme comparison and barrier-benchmark runnable examples; expanded
  methodology, design decisions and recorded QA.

### Preserved

- Full Truncation Euler default and discrete-barrier default.
- v0.1 simulation, payoff, pricing, vanilla, risk and Fourier call signatures.
- No runtime dependency on or modification of the FX/FI repositories.

### Not implemented

- Calibration to market IV/vanilla quotes, market Vega/recalibrated Greeks,
  exact continuous-time Heston barrier crossing, QMC, control variates,
  non-flat rate/dividend curves, American features or production guarantees.

## 0.1.0 — 2026-11-10

Initial **standalone** Equity Derivatives Engine milestone.

- Vanilla Black-76/Black-Scholes European pricing and IV inversion.
- Heston characteristic function and Lewis Fourier European vanilla pricer.
- Correlated Heston full-truncation Euler Monte Carlo with reproducible shocks.
- European, arithmetic Asian, fixed-strike lookback and discretely monitored
  barrier knock-in/knock-out payoff implementations.
- Antithetic simulation and pair-correct sampling error/normal CIs.
- CRN finite-difference Delta/Gamma/initial-variance sensitivity.
- Executable demos, convergence study and Python test suite.
- Upstream FX/FI repositories remain untouched and are not package dependencies.

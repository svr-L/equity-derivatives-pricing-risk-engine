# Changelog

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

Unimplemented capabilities are described in `docs/METHODOLOGY.md`. Future
changes must be explicitly proposed/versioned rather than silently adopted.

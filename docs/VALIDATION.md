# Validation record — v0.1.0 (2026-11-10)

Execution environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2.

```bash
python -m pip install -e . --no-deps --no-build-isolation
PYTHONPATH=src pytest -q
python -m compileall -q src tests examples
python examples/price_exotics.py
python examples/convergence_study.py
```

**Result:** 31 pytest tests passed locally. Both an editable install and a
wheel build were verified. CI config runs Python 3.10–3.13 tests and
compile checks. CI has not yet run on GitHub; no claim is made about
external multi-version test results. Ruff linting could not be executed in
this environment (package unavailable); it is not a CI gate in v0.1.

## Numerical sanity checks

European Heston with parameters `spot=100, r=1%, q=2%, kappa=2,
theta=0.04, xi=0.5, rho=-0.7, v0=0.04, maturity=1`:

- Heston Fourier benchmark at strike 100: **6.513842**.
- Monte Carlo example (`n_steps=252`, `n_paths=20,000`, seed 2026,
  antithetic): **6.41141**, standard error **0.04646**.
- The point difference is about **-2.20 Monte Carlo standard errors**,
  *before* accounting for Euler discretization bias. A particular Monte Carlo
  draw should not be taken as proof of estimator bias or as model validation.
- The convergence script varies steps and paths and compares with the
  deterministic Fourier pricer. Higher path count reduces simulation noise
  in expectation, but a specific seed will not show monotone convergence.
- For `xi=0`, `theta=v0`, vanilla MC matches Black-Scholes within simulation
  uncertainty; Fourier matches analytically to machine precision.
- Zero-rebate, identically monitored knock-in/out parity holds pathwise.

**What is NOT proven:** no quantitative bound on Euler time-step weak bias;
no calibrated out-of-sample fit; no continuous-barrier correctness; no
independent QuantLib cross-validation; no claim of hedging effectiveness.

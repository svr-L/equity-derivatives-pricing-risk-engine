# Equity Derivatives Pricing Engine

**Standalone Python library — v0.1.0.**

Unified risk-neutral Monte Carlo pricing for equity options, with Heston
stochastic variance, European/Asian/lookback/barrier payoffs, antithetic
variance reduction, CRN bump Greeks, Black-Scholes/Black-76 plus Heston Fourier
European benchmarks. No dependency on the separate FX or fixed-income packages.

> Scope is intentionally explicit: this is an *initial numerical library*,
> not yet a full calibrated market-production engine. See
> [methodology and limitations](docs/METHODOLOGY.md).

## Install

From the repository root:

```bash
python -m pip install -e '.[dev]'
pytest -q
python examples/price_exotics.py
```

Core requirements: Python >=3.10, NumPy, SciPy. No external data connection.

## Minimal usage

```python
from equity_derivatives import (
    EquityMarket, HestonParams, MCConfig,
    European, ArithmeticAsian, FixedStrikeLookback, Barrier,
    price_mc, heston_european, implied_vol, bump_greeks,
)

market = EquityMarket(spot=100, rate=0.01, dividend_yield=0.02)
heston = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
mc = MCConfig(maturity=1.0, n_steps=252, n_paths=20_000,
              antithetic=True, seed=2026)

for product in (
    European(100),
    ArithmeticAsian(100),
    FixedStrikeLookback(100),
    Barrier(strike=100, barrier=125, direction="up", knock="out"),
):
    result = price_mc(market, heston, product, mc)
    print(type(product).__name__, result.price, result.stderr)

vanilla = heston_european(market, heston, strike=100, maturity=1.0)
print("Heston IV:", implied_vol(vanilla, market, 100, 1.0))
print("Barrier Greeks:", bump_greeks(market, heston,
                                    Barrier(100, 125), mc))
```

## What is validated

- European Black-76 vs equity Black-Scholes with dividends.
- Inverse Black implied volatility and put-call parity.
- Heston characteristic function martingale identity, Heston Fourier parity.
- Heston Fourier vs Black-Scholes when volatility-of-variance is zero.
- Heston MC European vs Heston Fourier, tolerance allowing sampling/grid error.
- Antithetic pair-level Monte Carlo standard errors, seed reproducibility.
- Knock-in + knock-out = European (discrete monitoring, zero rebate).
- Controlled payoff/monitoring conventions and input validation.

## Architecture

```text
src/equity_derivatives/
    models.py           market, Heston and MC configuration
    vanilla.py          Black-76 / Black-Scholes / implied vol / Vega
    heston_fourier.py   semi-analytical vanilla benchmark
    simulation.py       correlated full-truncation Heston path engine
    payoffs.py          reusable product payoffs
    pricing.py          discounted MC estimator + confidence intervals
    greeks.py           CRN bump Delta, Gamma and dPrice/dv0
examples/price_exotics.py
examples/convergence_study.py
tests/
docs/METHODOLOGY.md
docs/PROVENANCE.md
```

## Milestone roadmap (not claims of completed features)

- **v0.1 (this release):** independently installable research core, vanilla
  numerical benchmarks, discrete-monitoring exotics, Monte Carlo statistics.
- **v0.2:** robust Andersen QE / QE-M or other verified second
  scheme, convergence diagnostics, independent barrier benchmarks, performance
  and memory profiling, conditional bridge approximation documented for Heston.
- **v0.3:** calibration of Heston to vanilla *prices/IV* with
  arbitrage/data checks and parameter-identifiability diagnostics.
- **v0.4:** production-quality risk surface scenarios and explicit
  sticky-parameter vs recalibrated-market-IV Greek conventions.

## Validation record

See [the validation note](docs/VALIDATION.md) for the test command, a
reproducible benchmark run and the distinction between sampling and
time-discretization error.

## Source-code provenance

The pricing conventions in `vanilla.py` are independently adapted from the
user's existing FX and FI pricing repositories. See
[PROVENANCE](docs/PROVENANCE.md) for files, version compatibility and license.
Both source repositories remain **unchanged**.

Copyright (c) 2026 Saverio Lauriola. MIT License.

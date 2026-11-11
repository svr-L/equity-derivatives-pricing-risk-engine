# Equity Derivatives Pricing & Risk Engine

**Standalone Python research library — v0.2.0.**

A unified equity-derivatives numerical core with Heston stochastic variance,
European/Asian/lookback/barrier payoffs, two Monte Carlo schemes, paired
antithetics, common-random-number Greeks and independent vanilla/barrier
Black-Scholes benchmarks. Version 0.2 adds **Andersen QE-M** and **conditional
Brownian-bridge barrier survival weighting**.

**Important:** QE-M is a discretization, not exact Heston simulation. The
continuous-barrier correction is **exact conditional on endpoints for constant
Black-Scholes volatility** but is **only approximate for stochastic Heston
variance**. No market-surface calibration, portfolio hedging P&L or certified
production accuracy is claimed.

## Installation

Python >=3.10, NumPy >=1.26, SciPy >=1.11:

```bash
python -m pip install -e '.[dev]'
pytest -q
python examples/price_exotics.py
python examples/compare_schemes.py
python examples/barrier_validation.py
python examples/heston_barrier_grid.py
```

From a standalone wheel, `pip install dist/*.whl`. The import name remains
**`equity_derivatives`**, unchanged from v0.1. **Neither FX nor FI pricing
repositories are modified or required.** Because the distribution name has
changed while the import name has not, use a fresh virtual environment or
uninstall the old `equity-derivatives-pricing-engine` distribution before
installing v0.2; do not install both into the same environment.

## First pricing example

```python
from equity_derivatives import (
    EquityMarket, HestonParams, MCConfig, European, ArithmeticAsian,
    FixedStrikeLookback, Barrier, price_mc, heston_european,
    black_scholes_barrier, bump_greeks,
)

market = EquityMarket(spot=100, rate=0.01, dividend_yield=0.02)
params = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
mc = MCConfig(maturity=1.0, n_steps=96, n_paths=20_000,
              seed=2026, antithetic=True, scheme="qe_m")

products = (
    European(100),
    ArithmeticAsian(100),
    FixedStrikeLookback(100),
    Barrier(100, 125, direction="up", knock="out",
            monitoring="continuous_bridge"),
)
for product in products:
    result = price_mc(market, params, product, mc)
    print(type(product).__name__, round(result.price, 4), round(result.stderr, 4))

print("Heston Fourier European:", heston_european(market, params, 100, 1.0))
print("Spot/variance Greeks:", bump_greeks(market, params, products[3], mc))

# Independent exact reference exists only for constant-vol Black-Scholes:
print("BS continuously monitored barrier reference:",
      black_scholes_barrier(market, products[3], maturity=1.0, vol=0.20))
```

## Supported functionality

| Layer | v0.2 functionality | Caveat |
|---|---|---|
| Vanilla | Black-Scholes/Black-76, implied Black IV, Heston Fourier | Flat r and q; European exercise |
| Model | Heston correlated price/variance | Q-dynamics, no jumps |
| Simulation | Full Truncation Euler (v0.1 default), Andersen **QE-M** | Both time discretized under stochastic variance |
| Exotics | Arithmetic Asian, fixed-strike lookback, up/down knock-in/out barrier | Grid fixings for Asian/lookback; no rebates |
| Barrier | Grid monitoring or conditional bridge weights | Continuous bridge approximate for Heston |
| Risk | CRN spot Delta/Gamma and sensitivity to initial variance `v0` | Not market IV Vega or recalibrated Greeks |
| Statistics | Antithetics, independent-pair SE, normal CIs | CIs exclude time/grid and model bias |
| Validation | Fourier vanilla; BS continuous-barrier reflection formula; tests, convergence and runtime demos | Not a substitute for independent model validation |

`MCConfig.scheme='full_truncation_euler'` and `Barrier.monitoring='discrete'`
remain the defaults for backwards compatibility with **v0.1**.

## Repository layout

```text
src/equity_derivatives/
    models.py               market, model and MC configuration
    simulation.py           FT Euler and Andersen QE-M
    payoffs.py              payoff definitions and monitoring convention
    pricing.py              estimator, survival-weighted barriers and CIs
    barrier_bridge.py       conditional no-touch weights
    barrier_reference.py    independent exact Black-Scholes barrier benchmark
    vanilla.py              Black-76 / Black-Scholes / implied vol
    heston_fourier.py       semi-analytical Heston vanilla
    greeks.py               CRN bump risks
examples/
    price_exotics.py
    compare_schemes.py
    barrier_validation.py
    heston_barrier_grid.py
    convergence_study.py
tests/
docs/
    METHODOLOGY.md
    VALIDATION.md
    PROVENANCE.md
    DESIGN_DECISIONS.md
```

## Reliability and validation

1. Run `pytest -q` for regressions, variance CIR moment matching, Heston
   martingale tests, Black-Scholes continuous-barrier reference convergence,
   and knock-in/out parity.
2. Run `python examples/compare_schemes.py` to examine MC estimates, standard
   errors and runtimes **against the independent Heston Fourier price**.
3. Run `python examples/barrier_validation.py` to compare **constant-vol**
   bridge results against the analytic continuous-monitoring barrier price.
4. Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for derivations,
   estimator conventions and limitations.
5. Read [`docs/VALIDATION.md`](docs/VALIDATION.md) for actual local outcomes.

## Versioned scope / roadmap

- **v0.1.0:** standalone baseline (Full Truncation Euler, vanilla Fourier,
  discrete exotics, CRN bump risks).
- **v0.2.0 (current):** QE-M scheme, conditional barrier bridge, independent
  Black-Scholes continuous-barrier benchmark, expanded tests and experiments.
- **v0.3, not implemented:** vanilla-market IV/price surface
  calibration, quote-quality/no-arbitrage checks, fit and identifiability
  diagnostics, optional fixture data.
- **v0.4, not implemented:** expanded sensitivity conventions
  (sticky model parameters vs recalibrated surface risk), scenario and
  portfolio-risk outputs.

## License / upstream provenance

MIT. [`docs/PROVENANCE.md`](docs/PROVENANCE.md) describes how standalone
pricing conventions were informed by the independently supplied FX and FI
pricing packages and why no runtime dependency was introduced.

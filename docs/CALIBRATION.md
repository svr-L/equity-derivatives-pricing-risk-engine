# Heston vanilla calibration — v0.3.0 methodology

## Objective and instrument convention

Heston is calibrated to *discounted European vanilla prices*. Each input
`VanillaQuote` provides strike, ACT/365F maturity, call/put right and mid;
optional bid/ask are used to reduce the influence of wide spreads, not to
infer reliable fair values. The market spot, continuously compounded flat
`r` and continuous dividend yield `q` are mandatory market inputs.

### Market IV and objective

For each quote, `implied_vol(mid, market, K, T, right)` solves a European
Black–Scholes/Black-76 inversion. For American option prices that inversion is
not a true comparable European implied volatility, hence the explicit opt-in
and warning.

Three objectives are available, with quote-level fixed spread weights `w_i`:

- `price`: residual `w_i * (HestonPrice_i - Mid_i) / Spot`.
- `vega_scaled_price` (default): residual
  `w_i * (HestonPrice_i - Mid_i) / max(BlackVega_i, 0.001 * Spot)`.
- `iv`: residual `w_i * (BlackIV(HestonPrice_i) - BlackIV(Mid_i))`.

These objectives are **not equivalent**. Vega-scaled price differences act as a
first-order proxy for implied-vol differences, preventing negligible-Vega
quotes from receiving unbounded weight through a finite Vega floor. The
optimizer minimizes one-half of the squared residual norm via bounded
`scipy.optimize.least_squares` (trust-region reflective). An optional robust
loss or stronger market microstructure weighting has **not** been implemented.

## Model parameters and bounds

Calibrated parameters `(kappa, theta, xi, rho, v0)` represent mean-reversion,
long-run instantaneous variance, vol-of-variance, Brownian correlation and
initial instantaneous variance. Default bounds are explicitly configured,
not estimated from the market. No Feller condition is imposed; crossing the
Feller boundary is not itself a reason to reject a well-defined CIR process.

Random-start optimization is deterministic given a fixed seed. The winning
start minimizes objective cost; a returned optimizer `success=True` only
means numerical convergence. Parameters can be weakly identified even if
price/IV RMSE is low, especially if expiry/moneyness coverage is sparse.

`jacobian_condition_number` is the condition number of the local residual
Jacobian **in the current native parameter units/objective**; it is a
scale-dependent diagnostic, not a universal identifiability statistic.
Check individual starting points, parameter bounds, forward/discount inputs,
objective variants and out-of-sample errors before drawing conclusions.

## Numerical pricer

The new batch Heston Fourier pricer uses the same risk-neutral Lewis integral
as `heston_fourier.heston_european`. It shares characteristic-function
integrals across strikes of each maturity and evaluates them using fixed-order
Gauss–Legendre quadrature (96 nodes up to 180 by default). This is very fast
for calibration, but can introduce systematic pricing errors. Test alternative
nodes / truncation cutoffs and cross-check against the original adaptive
quadrature pricer (and ideally a distinct external implementation).

Heston exotics continue to use the existing v0.1/v0.2 Monte Carlo engine and
methods; the new calibration output is a `HestonParams` instance accepted by
the existing pricing/Greeks APIs. No new Greeks have been invented.

## Holdout convention

`evaluate_heston_fit` reports price/IV RMSE, signed biases and bid/ask coverage
without recalibrating. The archived SPY example holds out approximately one
in four selected strikes **within each maturity at the same timestamp**. That
is a cross-sectional holdout, not a historical/temporal backtest, does not
measure future surface stability, and can still have strong cross-sectional
information leakage from nearby strikes. A separate historical date and
ex-ante filter design are required for genuine time-series out-of-sample work.

## Practical limitations

1. American ETF options require an American-aware Heston pricer or a carefully
   justified exercise premium adjustment for reliable calibration. The flag
   `allow_american_proxy=True` **does not** perform such an adjustment.
2. Rates/borrowing/dividend term structures are not downloaded or bootstrapped.
   Input example `r=4%, q=1%` is hypothetical, not a historical data assertion.
3. The Yahoo collector's spot quote and listed option quotes may not be fully
   synchronous. Last-trade timestamps are not a substitute for bid/ask stamps.
4. Recalibration risk (sticky strike/IV vs parameter risk), arbitrage-free
   surface smoothing and credible hedging P&L are future work.

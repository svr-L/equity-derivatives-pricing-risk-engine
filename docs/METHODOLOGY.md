# Numerical methodology and limitations — v0.1.0

## Model and conventions

Risk-neutral Heston with continuous dividend yield q:

    dS_t / S_t = (r-q) dt + sqrt(v_t) dW^S_t
    dv_t       = kappa(theta-v_t) dt + xi sqrt(v_t) dW^v_t
    d<W^S,W^v>_t = rho dt.

`theta` and `v0` are **variances**. `xi` is vol of variance. `rate` and
`dividend_yield` are continuously compounded annualized inputs, not curve
objects. The code does not require the Feller condition and does not claim
strictly positive Heston variance paths.

## Heston simulation

- Full truncation Euler: the *positive part* of pre-step raw variance enters
  both variance drift/diffusion and the log-spot Euler step. Raw variance is
  retained internally; exported paths are clipped to zero.
- Joint Brownian shocks: `z_s = rho*z_v + sqrt(1-rho²)*z_i`.
- Spot is log-stepped to stay strictly positive.
- Euler introduces time-discretization bias and a nontrivial weak error.
- Antithetic signs are reflected for *both* independent Brownian shocks.
- Confidence intervals: the iid sampling units are **pair averages** if
  antithetic=True, not the individual paths. A normal 95% CI describes
  finite-sample simulation noise **only**, not time/grid or model bias.

## Heston vanilla Fourier

- Uses the Heston log-return characteristic function (Little-Trap choice)
  with Lewis' half-imaginary-shift integral for European vanilla options.
- Puts are calculated by call-put parity.
- For `xi=0` variance is deterministic mean-reverting and an exact
  integrated-variance Black-Scholes formula is used.
- Numerical integration truncates at `integration_upper` (default 200).
  Test truncation stability for short maturities and extreme params.
- Market calibration is **not yet included** in v0.1.

## Payoffs and exercise

European exercise at T only. All products are cash-settled, with no rebates.
- `European`: `max(phi*(S_T-K),0)`.
- `ArithmeticAsian`: arithmetic average of all n_steps *positive time*
  observations by default; optional t=0 inclusion. The grid defines fixing
  dates; custom fixing calendars are not yet supported.
- `FixedStrikeLookback`: `max(S_max-K,0)` call and `max(K-S_min,0)` put;
  observation includes t=0 and T.
- `Barrier`: up/down knock-in/out, touch at or beyond barrier counts as hit,
  monitoring includes t=0 and T. Monitoring is **discrete**, not continuous;
  continuous barrier pricing requires a separate conditional-crossing model.
- Barrier in + barrier out equals European pathwise (zero rebate), a useful test.

## Greeks

`bump_greeks` estimates spot delta and gamma by *central* bump-and-revalue
using common random numbers, with default 1% spot bump. Gamma remains a noisy
second difference for non-smooth barrier payoffs and depends on bump size.
`v0_sensitivity` is the derivative wrt **initial instantaneous variance**;
not smile recalibration Vega, not Black implied Vega. Holding parameters
fixed when bumping spot is an explicit "sticky parameters" convention.

## Not implemented in this release

No model calibration, no American exercise, no pathwise or LRM Greeks,
no Brownian-bridge barrier correction, no curves/dividend dates, no
Andersen-QE variance scheme, no quasi-MC, no control variates, no stochastic
interest rates and no PDE/QuantLib cross-validation. Market data retrieval
is not part of this release. An IV computed from an exotic payoff is not in
general a unique "market IV"; `implied_vol` applies to European vanilla
prices only.

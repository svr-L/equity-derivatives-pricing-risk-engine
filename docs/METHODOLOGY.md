# Methodology and numerical limitations — v0.2.0

## Model and units

Under the risk-neutral measure Q, with continuous dividend yield q:

    dS_t / S_t = (r-q) dt + sqrt(v_t) dW^S_t
    dv_t       = kappa(theta-v_t) dt + xi sqrt(v_t) dW^v_t
    d<W^S,W^v>_t = rho dt.

`v0` and `theta` are **variances**, `xi` is volatility of the variance
process, `kappa` is mean reversion. The Feller condition `2*kappa*theta >=
xi**2` is NOT imposed. Both model schemes accept variance touching zero;
negative simulated variance is never used for spot diffusion. The current
market object has a flat `rate` and continuous dividend yield, not actual
market discount curves, discrete dividend schedules or stock borrow curves.

## Simulation scheme A — Full Truncation Euler (v0.1 default)

The raw numerical variance may become negative. Its positive part enters the
variance drift/diffusion and the log-stock update; the raw state is retained
internally, while exported variance grid values are nonnegative. Correlated
Brownian shocks are constructed as `zS=rho*zV+sqrt(1-rho**2)*zI`. Spot is
log-stepped with the **pre-step variance**, and so is strictly positive.
There is weak time-discretization bias, especially for large vol-of-vol,
strong leverage correlation and near-zero instantaneous variance.

## Simulation scheme B — Andersen Quadratic-Exponential with martingale correction (QE-M)

At every time step of length h, calculate CIR's *exact conditional moments*

    m  = theta + (v-theta)*exp(-kappa*h)
    s2 = v*xi²*exp(-kappa*h)*(1-exp(-kappa*h))/kappa
         + theta*xi²*(1-exp(-kappa*h))²/(2*kappa)
    psi = s2/m².

For `psi <= 1.5`, approximate conditional variance by a scaled shifted
squared Gaussian `a*(sqrt(b²)+Z)²`; choose `a,b²` to match `m,s2`. For
`psi>1.5`, use a point mass at zero plus exponential tail, with
`p=(psi-1)/(psi+1)`, `beta=(1-p)/m`. The required uniform for this branch
is `Phi(z_variance)`, which preserves reproducible paired antithetic samples.
Zero conditional variance is handled as a degenerate transition.

Let `gamma1=gamma2=0.5` and define:

    K1 = gamma1*h*(kappa*rho/xi-1/2) - rho/xi
    K2 = gamma2*h*(kappa*rho/xi-1/2) + rho/xi
    K3 = gamma1*h*(1-rho²)
    K4 = gamma2*h*(1-rho²)
    A  = K2 + K4/2

The log-stock update uses `K1*v+K2*v_next + sqrt(K3*v+K4*v_next)*Z_I`
and subtracts the conditional log-moment generating function of `v_next`
so that `E[S_next | S,v] = S*exp((r-q)*h)` whenever its relevant exponential
moment exists. This is the **QE-M** martingale correction; it is distinct
from naive variance QE with log-price Euler. Parameter choices for which
this exponential moment is not finite raise a descriptive `ValueError`.

The special case `xi=0` uses exact deterministic mean-reverting variance,
exact *integrated* variance over each grid interval, and an exact Gaussian
log-stock increment (no time-step bias in terminal vanilla distribution).
When `xi>0`, QE-M remains approximate and must be tested for time-step
bias; moments alone do not make it an exact discretization.

## Monitoring and payoffs

European cash-settled expiry payoffs, no rebate and no exercise before T:

- `European`: call/put `max(±(S_T-K),0)`.
- `ArithmeticAsian`: arithmetic averaging of each positive-time grid node,
  optionally including time zero. Grid refinements *change the fixing
  schedule*, so do NOT interpret its price changes as pure numerical time
  discretization error for a single fixed contract.
- `FixedStrikeLookback`: maximum/minimum over time-zero and subsequent grid
  nodes. This is **discretely sampled**, not a continuously monitored
  lookback; no continuous-extremum correction is implemented.
- `Barrier`: up/down, knock-in/out. A touch of or beyond the barrier knocks
  out/activates (inclusive). Default monitoring at every grid node INCLUDING
  time zero and expiry (`monitoring="discrete"`).

### Conditional bridge weighting

For `Barrier(..., monitoring="continuous_bridge")`, `price_mc` computes
conditional **no-hit probabilities** between adjacent log-price endpoints
`x_i = log(S_i)` and `x_{i+1}=log(S_{i+1})`. If both points lie on the
safe side of log-barrier `b`, then the local survival factor is:

    p_survival_i = 1 - exp(-2*abs(b-x_i)*abs(b-x_{i+1})/I_i),

where `I_i` is the estimated integrated variance over that interval.
Endpoint touches cause zero survival. Multiply these factors across all
intervals (stably via log probabilities) to obtain `w`. The MC payoff
conditional expectations are `w*vanilla_payoff` for knock-out and
`(1-w)*vanilla_payoff` for knock-in. This is Rao-Blackwellized relative to
extra Bernoulli crossing draws and ensures knock-in/out parity on the same
underlying paths (with zero rebates).

For **constant Black-Scholes variance** (xi=0, v0=theta), and the exact
node-grid log-price process, this formula is exact conditional on endpoints.
For Full Truncation Euler, `I_i=dt*max(v_i,0)` follows the frozen per-step
variance. For QE-M with `xi>0`, `I_i≈(v_i+v_{i+1})*dt/2`, an approximation.
For deterministic *time-varying* variance (`xi=0, v0!=theta`) integrated
variance is exact in QE-M, but a constant-barrier reflection formula with a
changing deterministic log drift is not, in general, exactly valid.

**Warning:** Under **stochastic Heston variance**, bridge weighting is only
an approximation; it is not the exact conditional crossing law when
variance varies inside the interval, especially with `rho≠0`. A residual
barrier-monitoring bias remains. Report the scheme, grid, bridge convention
and a convergence study; never claim exact continuously monitored Heston
barrier prices. An `option.evaluate(spot_paths)` call always evaluates the
grid-monitoring payoff. The bridge is applied by `price_mc()` because it
requires Heston variance paths, not just spot observations.

### Independent constant-vol reference

`black_scholes_barrier` implements the **exact continuously monitored**
no-rebate Black-Scholes barrier price by reflecting the drifted normal
log-terminal density at the barrier and integrating a call or put payoff.
It supports up/down, knock-in/out, call/put, zero time and deterministic
zero volatility. For knock-in it uses vanilla minus knock-out.
This is an independent *constant volatility* benchmark, not an analytic
Heston-barrier formula; tests verify bridge estimates against it within
sampling uncertainty. As an extra sanity check, any continuous knock-out
price should not exceed its discretely monitored counterpart on identical
simulated paths when the conditional bridge estimator is used.

## Vanilla Heston Fourier pricing and implied volatility

European vanilla pricing uses the Heston log-return characteristic function
with the Lewis half-imaginary contour and numerical integration to a finite
upper limit. Put-call parity gives puts; for `xi=0` the vanilla pricer uses
Black-Scholes with exact deterministic integrated variance. Truncation and
quadrature errors can matter for extreme parameters or maturities. Only
**vanilla** price-to-Black-IV inversion is supported; reporting an exotic's
Black IV as a model calibration target is not generally well-defined.

Calibration to observed vanilla prices/IV surfaces is **NOT IMPLEMENTED**.
Passing a fixed illustrative Heston parameter set to the engine is not
market calibration. Flat continuous dividends omit discrete corporate
actions. American options are outside scope.

## Monte Carlo confidence intervals and Greeks

For antithetic=True, pairs `(0,1), (2,3), ...` are the independent
sampling units. The standard error uses the sample standard deviation of
pair-average *discounted* payoffs divided by sqrt(number of pairs).
Without antithetics, individual discounted path payoffs are the units.
Normal-theory confidence intervals describe MC **sampling uncertainty only**:
not discretization error, bridge approximation error, calibration error,
model error or possible hedge slippage. Samples must remain finite.

`bump_greeks` reuses the same Gaussian shock arrays across central spot and
`v0` bumps (common random numbers), and respects the selected MCConfig
scheme and barrier monitoring convention. It provides spot Delta, spot
Gamma and `dPrice/d(v0)` at **fixed Heston parameters**; `dPrice/d(v0)` is
NOT Black market Vega. Bump size affects noise/bias; barrier Greeks are
particularly discontinuous/noisy. No analytic/pathwise/likelihood-ratio
Greeks, volatility-surface bucket Vega, sticky-IV or recalibration risk is
implemented yet.

## Numerical QA checklist

1. Check model inputs/units, Feller ratio and QE-M moment-domain errors.
2. Cross-check European MC against Heston Fourier within MC SE and grid bias.
3. Compare multiple step counts; fixed seed does not imply monotone error.
4. For constant vol, compare barrier-bridge price to independent closed form.
5. Check discrete KI+KO and bridge KI+KO = same European payoff price.
6. Reproduce tests, examples, runtimes and record scheme and seed.
7. For any risk output, document the *bump* and what market quantities are
   kept fixed. No calibration or model risk is quantified in v0.2.

# Architecture and backward compatibility decisions (v0.3.0)

These are the explicit decisions applied in **this** release; future design
changes should be versioned and reviewed rather than silently substituted.

1. **New repository, separate sources.** Distribution name is
   `equity-derivatives-pricing-risk-engine`. The original user-supplied FX
   and fixed-income archives remain unchanged. Equity engine has no runtime
   dependency on them.
2. **Stable Python import.** `equity_derivatives` stays the import namespace.
   The existing public functions, model classes and default behaviors remain
   usable without code changes.
3. **Stable MC default.** `MCConfig(...).scheme` defaults to
   `full_truncation_euler`, exactly as v0.1; passing `scheme="qe_m"`
   chooses the new Andersen QE martingale-corrected transition.
4. **Stable barrier default.** `Barrier(...).monitoring` defaults to
   `"discrete"`, exactly as v0.1; `"continuous_bridge"` requests an
   explicitly labelled Heston crossing approximation in `price_mc()`.
5. **Risk terminology.** dPrice/d(v0) is not renamed to "market Vega".
   Calibration and model-recalibration sensitivities remain unimplemented.
6. **Separate analytically benchmarked limits.** An independent exact
   Black-Scholes continuous-barrier reference is supplied. It is *not*
   misrepresented as an exact continuous Heston barrier pricer.
7. **Tracking quality.** Reproducible new tests/experiments and explicit
   limitations accompany every numerical addition.

**v0.3 additive extension:** batch Heston Fourier, multi-start calibration
on European vanilla mids/IVs, archived option snapshot adapters and limited
strike-arbitrage diagnostics. The existing numerical engine and APIs are
preserved; the new market-data dependency is optional. American quotes demand
explicit European-proxy opt-in. Explicit user-supplied rates and dividends
avoid silently guessed market inputs. Historical Yahoo data are not bundled.

**Not introduced:** full market-data API, arbitrage smoothing,
local volatility, Bates, early exercise, PCA/smile risk, arbitrary fixings,
exotic IV, portfolio VaR/ES, hedge P&L or production deployment.

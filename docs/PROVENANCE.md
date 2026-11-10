# Provenance and interoperability (v0.1)

The new repository is **standalone**. It neither imports from nor changes the
following user-supplied archives:

- `fx-pricing-python-0.2.0(1).zip` — package `fx-pricing`, version 0.2.0.
- `fixed-income-pricing-python-public-v0_13.zip` — package `fi-pricing`, version 0.13.0.

The standalone `equity_derivatives.vanilla` module was developed using the
**forward Black-76 convention and inverse-IV design** established in
`fi_pricing/core.py`, `fi_pricing/volatility.py` and `fx_pricing/black.py`.
This module is new standalone code, not a direct file copy. The modules are
published under MIT in the provided archives. The copyright statement and
MIT license are retained in this repository.

The product-level architecture and the distinction between continuous and
discrete barrier observation are informed by `fx_pricing/barriers.py`, but
no FX-specific barrier formula was copied. The MC payoff implementations
and Heston Fourier simulation/pricing were developed afresh.

**Compatibility note:** The provided FX v0.2.0 declares `fi-pricing>=1.0.0`
and imports `fi_pricing.options`. The provided FI v0.13.0 instead exposes its
Black-76 functions in `fi_pricing.core`; the pair cannot simply be installed
as-is to provide one shared API. We intentionally avoid this fragile runtime
dependency and keep both original repositories untouched.

Potential future *optional* adapters may accept external `DiscountCurve` or
`VolSurface2D` instances via plain-data conversion. They must not create a
mandatory dependency on FX or FI packages.

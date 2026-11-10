"""Sampling and time-grid diagnostics, against independent Fourier vanilla.

Run: python examples/convergence_study.py
Displayed error = MC - Fourier (contains both stochastic and discretization error).
"""
from equity_derivatives import (EquityMarket, European, HestonParams,
                                MCConfig, heston_european, price_mc)


def main() -> None:
    mkt = EquityMarket(100, .01, .02)
    par = HestonParams(2, .04, .5, -.7, .04)
    target = heston_european(mkt, par, strike=100, maturity=1)
    print(f"Heston Fourier European benchmark: {target:.6f}")
    print("  steps  paths      estimate        stderr       error      error/SE")
    for steps in (24, 96, 252):
        for paths in (4000, 16000):
            out = price_mc(mkt, par, European(100),
                           MCConfig(1, steps, paths, seed=2026))
            err = out.price - target
            print(f"{steps:7} {paths:6}   {out.price:11.6f}   {out.stderr:11.6f}"
                  f"   {err:+9.6f}  {err/out.stderr:+9.3f}")
    print("Important: |error| includes finite-grid bias, not estimated by stderr.")


if __name__ == "__main__":
    main()

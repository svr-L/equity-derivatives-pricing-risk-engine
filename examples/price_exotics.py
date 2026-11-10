"""Run: python examples/price_exotics.py (after pip install -e .)."""
from equity_derivatives import (ArithmeticAsian, Barrier, EquityMarket, European,
                                FixedStrikeLookback, HestonParams, MCConfig,
                                black_scholes, bump_greeks, heston_european,
                                implied_vol, price_mc)


def main() -> None:
    market = EquityMarket(spot=100, rate=.01, dividend_yield=.02)
    params = HestonParams(kappa=2, theta=.04, xi=.5, rho=-.7, v0=.04)
    config = MCConfig(maturity=1, n_steps=252, n_paths=20_000, seed=2026)
    print("Model: Heston | Q-dynamics | full truncation Euler | discrete monitoring")
    print(f"Feller ratio: {params.feller_ratio:.3f} (below 1: variance may reach zero)")
    print(f"BS reference (sigma=20%): {black_scholes(market, 100, 1, .2):.5f}")
    vanilla_heston = heston_european(market, params, 100, 1)
    print(f"Heston Fourier European:  {vanilla_heston:.5f} | BS IV: "
          f"{implied_vol(vanilla_heston, market, 100, 1):.3%}")
    products = {
        "European call": European(100),
        "Arithmetic Asian call": ArithmeticAsian(100),
        "Fixed-strike lookback call": FixedStrikeLookback(100),
        "Up-and-out call (H=125)": Barrier(100, 125, "up", "out", True),
        "Down-and-out put (H=80)": Barrier(100, 80, "down", "out", False),
    }
    for name, product in products.items():
        result = price_mc(market, params, product, config)
        print(f"{name:29} {result.price:9.5f} +/- {result.stderr:7.5f} "
              f"95% CI [{result.ci_low:7.4f}, {result.ci_high:7.4f}]")
    risk = bump_greeks(market, params, Barrier(100, 125), config)
    print(f"Up-and-out spot Delta: {risk.delta:.5f} | Gamma: {risk.gamma:.5f} "
          f"| dPrice/d(v0): {risk.v0_sensitivity:.5f}")


if __name__ == "__main__":
    main()

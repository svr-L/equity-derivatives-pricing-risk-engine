"""Bridge vs discrete barrier against exact Black-Scholes reflection price.

Run: python examples/barrier_validation.py

This experiment is for *constant* Black-Scholes variance xi=0, v0=theta;
the corresponding correction under stochastic Heston variance is NOT exact.
"""
from equity_derivatives import (Barrier, EquityMarket, HestonParams, MCConfig,
                                black_scholes_barrier, price_mc)


def main() -> None:
    market = EquityMarket(100., .01, .02)
    params = HestonParams(2., .04, 0., -.7, .04)
    product = Barrier(100., 125., 'up','out',True,'continuous_bridge')
    exact = black_scholes_barrier(market,product,1.,.2)
    print(f'Exact continuous BS barrier: {exact:.6f}')
    print('steps      discrete     bridge    bridge_se    bridge_error')
    for n in (1,4,16,64):
        c = MCConfig(1.,n,40000,2026,True,'qe_m')
        continuous = price_mc(market,params,product,c)
        discrete = price_mc(market,params,Barrier(100,125),c)
        print(f'{n:5}   {discrete.price:11.6f} {continuous.price:11.6f}'
              f' {continuous.stderr:11.6f} {continuous.price-exact:+13.6f}')
    print('The bridge is exact conditional on GBM endpoints at constant volatility, '
          'but MC sampling error remains. For nonzero Heston vol-of-vol it is only approximate.')


if __name__ == '__main__':
    main()

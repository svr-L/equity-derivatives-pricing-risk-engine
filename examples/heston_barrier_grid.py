"""Heston barrier grid sensitivity: no exact continuous Heston reference.

Run: python examples/heston_barrier_grid.py
Deliberately reports approximate bridge result rather than claiming exactness.
"""
from equity_derivatives import Barrier, EquityMarket, HestonParams, MCConfig, price_mc


def main() -> None:
    m = EquityMarket(100,.01,.02)
    p = HestonParams(2.,.04,.5,-.7,.04)
    print('Heston barrier grid sensitivity (NOT exact continuously monitored pricing)')
    print('scheme                   steps   discrete_KO    approx_bridge_KO   bridge_SE')
    for scheme in ('full_truncation_euler','qe_m'):
        for n in (24,96,252):
            c = MCConfig(1.,n,12000,2026,True,scheme)
            discrete=price_mc(m,p,Barrier(100,125),c)
            bridge=price_mc(m,p,Barrier(100,125,monitoring='continuous_bridge'),c)
            print(f'{scheme:24} {n:4} {discrete.price:13.6f} {bridge.price:18.6f} '
                  f'{bridge.stderr:10.6f}')
    print('The bridge is approximate for Heston stochastic variance (xi > 0). '
          'Grid refinement is a diagnostic, not proof of zero monitoring bias.')


if __name__ == '__main__':
    main()

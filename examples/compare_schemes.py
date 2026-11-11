"""Reproducible Euler vs Andersen QE-M vanilla experiment.

Run: python examples/compare_schemes.py
Options: --csv docs/benchmark_results.csv  --paths 12000
This is a sampling/error study, not proof of scheme domination.
"""
from __future__ import annotations

import argparse
import csv
import time
import tracemalloc
from pathlib import Path

from equity_derivatives import EquityMarket, European, HestonParams, MCConfig, heston_european, price_mc


def main() -> None:
    arg = argparse.ArgumentParser()
    arg.add_argument('--paths', type=int, default=12000)
    arg.add_argument('--csv', type=Path, default=None)
    args = arg.parse_args()
    market = EquityMarket(100, .01, .02)
    params = HestonParams(2., .04, .5, -.7, .04)
    target = heston_european(market,params,100,1.)
    print(f'Heston Fourier reference: {target:.8f}')
    print('scheme                  steps paths    mc_price    mc_se       error    runtime_s peak_traced_mib')
    rows = []
    for scheme in ('full_truncation_euler','qe_m'):
        for steps in (24,96,252):
            c = MCConfig(1.,steps,args.paths,2026,True,scheme)
            tracemalloc.start()
            begin = time.perf_counter()
            result = price_mc(market,params,European(100),c)
            elapsed = time.perf_counter()-begin
            _, peak=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            err = result.price-target
            row = dict(scheme=scheme,n_steps=steps,n_paths=args.paths,
                       mc_price=result.price,stderr=result.stderr,
                       fourier_price=target,error=err,ratio_error_se=err/result.stderr,
                       elapsed_seconds=elapsed,peak_traced_mib=peak/(1024**2))
            rows.append(row)
            print(f'{scheme:24} {steps:4d} {args.paths:5d} {result.price:11.6f} '
                  f'{result.stderr:9.6f} {err:+11.6f} {elapsed:10.3f} {peak/(1024**2):13.1f}')
    print('Caution: stderr excludes time discretization bias; peak_traced_mib '
          'is tracemalloc peak, not guaranteed whole-process RSS.')
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('Wrote:', args.csv)


if __name__ == '__main__':
    main()

"""Calibrate on an *existing* Yahoo collector snapshot (ZIP/CSV.GZ/Parquet).

User must explicitly supply annualized continuous rate and dividend yield.
American-style ETF options require --allow-american-proxy. The output is
exploratory and NOT an exercise-adjusted European-market calibration.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from equity_derivatives import (
    CalibrationConfig, EquityMarket, DiscountCurve, EquityCarryCurves,
    calibrate_heston, evaluate_heston_fit,
)
from equity_derivatives.market_data import (
    QuoteFilterConfig, filter_option_quotes, load_archive_snapshot, load_snapshot,
    static_arbitrage_report,
)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path, help='Collector archive .zip or one snapshot .csv.gz/.parquet')
    ap.add_argument('--ticker',required=True)
    ap.add_argument('--date',help='YYYY-MM-DD snapshot date (ZIP only; otherwise latest dated snapshot)')
    ap.add_argument('--rate',type=float,
                    help='Explicit annualized continuously compounded discount rate (decimal)')
    ap.add_argument('--dividend-yield',type=float,
                    help='Explicit annualized continuous dividend yield (decimal)')
    ap.add_argument('--curves-json',type=Path,
                    help='Dated supplied curves: asof, discount{tenors,dfs}, forwards{tenors,values}')
    ap.add_argument('--exercise-style',choices=['european','american'],default='american')
    ap.add_argument('--allow-american-proxy',action='store_true')
    ap.add_argument('--max-expiries',type=int,default=8)
    ap.add_argument('--per-side',type=int,default=6)
    ap.add_argument('--starts',type=int,default=4)
    ap.add_argument('--max-nfev',type=int,default=100)
    ap.add_argument('--output',type=Path,help='Optional JSON diagnostics path')
    args=ap.parse_args()
    if args.source.suffix.lower() == '.zip':
        frame, source=load_archive_snapshot(args.source,args.ticker,date=args.date)
    else:
        if args.date:
            ap.error('--date is only supported with ZIP archive inputs')
        frame = load_snapshot(args.source)
        source=str(args.source)
    if frame.empty:
        raise ValueError('empty data frame')
    if str(frame.underlying_ticker.iloc[0]).upper() != args.ticker.upper():
        raise ValueError('ticker mismatch between snapshot and CLI')
    spot=float(frame.underlying_spot.iloc[0])
    curves=None
    if args.curves_json is not None:
        if args.rate is not None or args.dividend_yield is not None:
            ap.error('--curves-json is mutually exclusive with --rate/--dividend-yield')
        raw=json.loads(args.curves_json.read_text(encoding='utf-8'))
        dc=DiscountCurve(tuple(raw['discount']['tenors']),tuple(raw['discount']['dfs']))
        curves=EquityCarryCurves.from_forwards(
            spot, dc, raw['forwards']['tenors'],raw['forwards']['values'],
            asof=raw['asof'])
        curves.validate_snapshot(str(frame.snapshot_ts_utc.iloc[0]))
        market=EquityMarket(spot)
    else:
        if args.rate is None or args.dividend_yield is None:
            ap.error('supply both --rate and --dividend-yield, or --curves-json')
        market=EquityMarket(spot,args.rate,args.dividend_yield)
    quotes, report=filter_option_quotes(
        frame,market,
        config=QuoteFilterConfig(max_expiries=args.max_expiries,
                                 max_quotes_per_expiry_per_side=args.per_side),
        exercise_style=args.exercise_style,
        allow_american_proxy=args.allow_american_proxy,
        carry_curves=curves)
    if len(quotes)<8:
        raise ValueError('insufficient liquid quotes for stable 5-parameter calibration')
    # Within-snapshot holdout: NOT a time-series/out-of-sample market backtest.
    train=[]
    held=[]
    for t in sorted(set(q.maturity for q in quotes)):
        same=[q for q in quotes if q.maturity==t]
        for i,q in enumerate(same):
            (held if i%4==3 else train).append(q)
    if len(train)<6 or not held:
        raise ValueError('insufficient train/holdout quotes')
    result=calibrate_heston(market,train,
                            config=CalibrationConfig(n_starts=args.starts,max_nfev=args.max_nfev),
                            exercise_style=args.exercise_style,
                            allow_american_proxy=args.allow_american_proxy,carry_curves=curves)
    test=evaluate_heston_fit(market,result.params,held,carry_curves=curves)
    selected=frame.loc[frame.contractSymbol.astype(str).isin({q.identifier for q in quotes})]
    arb=static_arbitrage_report(selected)
    payload={
        'version':'0.4.0', 'source':source, 'snapshot_utc':report.snapshot_timestamp,
        'ticker':report.ticker,'spot':market.spot,
        'flat_rate_assumption':(market.rate if curves is None else None),
        'flat_dividend_assumption':(market.dividend_yield if curves is None else None),
        'dated_market_curves':(str(args.curves_json) if curves is not None else None),
        'conventions':result.assumptions,
        'note':'Within-snapshot holdout, not temporal out-of-sample; American proxy may be biased',
        'filter_report':asdict(report),'strike_arbitrage_selected':asdict(arb),
        'n_train':len(train),'n_holdout':len(held),
        'fit_parameters':result.params_dict,'fit_success':result.success,
        'training_iv_rmse_vol_points':100*result.market_rmse_iv,
        'holdout_iv_rmse_vol_points':100*test.rmse_iv,
        'training_price_rmse':result.market_rmse_price,
        'training_bidask_coverage':result.bid_ask_coverage,
        'holdout_price_rmse':test.rmse_price,
        'holdout_bidask_coverage':test.bid_ask_coverage,
        'jacobian_condition_number':(result.jacobian_condition_number
                                    if math.isfinite(result.jacobian_condition_number) else None),
        'multi_start':[asdict(s) for s in result.starts],
    }
    print(json.dumps(payload,indent=2,allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()

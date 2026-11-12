"""Collector schema v1 adapter + file selection/quote hygiene."""
import io
import zipfile

import pytest
pd = pytest.importorskip("pandas")

from equity_derivatives.models import EquityMarket
from equity_derivatives.market_data import (
    filter_option_quotes, list_archive_snapshots,
    load_archive_snapshot, static_arbitrage_report,
)


def _synthetic_dataframe():
    s = 100.0
    rows = []
    for dte in (35, 75, 165, 300):
        for kind in ('call','put'):
            for strike in (85., 90., 95., 100., 105., 110., 115.):
                # Use exact BS prices to make an internally coherent test.
                from equity_derivatives.vanilla import black_scholes
                px=black_scholes(EquityMarket(s,.02,.01),strike,dte/365.,.20,kind=='call')
                rows.append(dict(underlying_ticker='TEST',snapshot_ts_utc='2026-10-27T22:00:00Z',
                                 underlying_spot=s,option_type=kind,strike=strike,
                                 expiry=str((pd.Timestamp('2026-10-27')+pd.Timedelta(days=dte)).date()),dte_calendar=dte,
                                 bid=max(.01,px-.015),ask=px+.015,
                                 openInterest=200, contractSymbol=f'{dte}-{kind}-{strike}'))
    return pd.DataFrame(rows)


def test_filter_exercise_guard():
    f=_synthetic_dataframe()
    with pytest.raises(ValueError,match='American'):
        filter_option_quotes(f, EquityMarket(100.,.02,.01))
    quotes,report=filter_option_quotes(f,EquityMarket(100.,.02,.01),
                                       exercise_style='european')
    assert report.original_rows == len(f)
    assert report.selected_expiries >= 2
    assert 8 <= len(quotes) <= 60
    assert all(q.mid > 0 for q in quotes)
    assert 'European' in report.caveat


def test_spot_mismatch_and_duplicate_rejected():
    f=_synthetic_dataframe()
    with pytest.raises(ValueError,match='spot'):
        filter_option_quotes(f, EquityMarket(101.,.02,.01),exercise_style='european')
    f=pd.concat([f, f.loc[(f.dte_calendar == 75) & (f.option_type == 'call') & (f.strike == 105.)]], ignore_index=True)
    with pytest.raises(ValueError,match='duplicate'):
        filter_option_quotes(f,EquityMarket(100.,.02,.01),exercise_style='european')


def test_select_dated_archived_not_latest(tmp_path):
    f=_synthetic_dataframe()
    archive=tmp_path/'options.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for timestamp, n in [('20261027T200000Z',3),('20261027T220000Z',5)]:
            f2=f.iloc[:n]
            buffer=io.BytesIO()
            f2.to_csv(buffer,index=False,compression='gzip')
            z.writestr(f'IV Data/data/options/TEST/2026/10/27/TEST_{timestamp}.csv.gz',buffer.getvalue())
        z.writestr('IV Data/data/options/TEST/latest.csv.gz','stale')
    found=list_archive_snapshots(archive,'TEST')
    assert len(found)==2
    loaded,name=load_archive_snapshot(archive,'TEST',date='2026-10-27')
    assert len(loaded)==5
    assert 'T220000Z' in name
    with pytest.raises(FileNotFoundError):
        load_archive_snapshot(archive,'TEST',date='2026-10-25')


def test_static_arbitrage_detects_clear_violations():
    base = pd.DataFrame([
        dict(expiry='2027-01-15',option_type='call',strike=90.,bid=13.,ask=14.),
        dict(expiry='2027-01-15',option_type='call',strike=100.,bid=16.,ask=17.),
        dict(expiry='2027-01-15',option_type='call',strike=110.,bid=4.,ask=5.),
    ])
    r = static_arbitrage_report(base)
    assert r.n_slices == 1
    assert r.mid_monotonicity_violations == 1
    assert r.executable_monotonicity_violations == 1
    assert r.mid_convexity_violations == 1
    assert r.executable_convexity_violations == 1


def test_static_arbitrage_clean_linear():
    base = pd.DataFrame([
        dict(expiry='2027-01-15',option_type='call',strike=k,bid=p-.1,ask=p+.1)
        for k,p in ((90,15),(100,10),(110,5))])
    r=static_arbitrage_report(base)
    assert r.mid_monotonicity_violations == 0
    assert r.executable_monotonicity_violations == 0
    assert r.mid_convexity_violations == 0

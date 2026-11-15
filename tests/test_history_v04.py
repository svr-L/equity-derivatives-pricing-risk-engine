"""Chronological point-in-time calibration and quote diagnostics."""
from dataclasses import replace
import math
import numpy as np
import pytest

from equity_derivatives import (
    DiscountCurve, EquityCarryCurves, EquityMarket, HestonParams,
    DatedCalibrationSlice, CalibrationConfig, VanillaQuote,
    calibrate_history, history_summary, heston_surface_prices,
)
from equity_derivatives.market_data import filter_option_quotes
from test_market_data import _synthetic_dataframe


def make_day(day, spot=100.):
    rate=0.025
    c=EquityCarryCurves(spot,DiscountCurve.flat(rate,2.),
                        DiscountCurve.flat(.015,2.),asof=day)
    m=EquityMarket(spot)
    p=HestonParams(2.,.04,.5,-.6,.04)
    ks=np.tile([90.,100.,110.],3);ts=np.repeat([.3,.8,1.7],3)
    px=heston_surface_prices(m,p,ks,ts,carry_curves=c)
    quotes=tuple(VanillaQuote(float(k),float(t),float(v),bid=max(0.,float(v)-.2),ask=float(v)+.2)
                 for k,t,v in zip(ks,ts,px))
    return DatedCalibrationSlice(day,'TEST',m,c,quotes)


def test_chronological_recalibration_and_cross_date_eval():
    ds=(make_day('2026-10-27'),make_day('2026-10-28',101.))
    history=calibrate_history(ds,initial=HestonParams(2.,.04,.5,-.6,.04),
                  config=CalibrationConfig(n_starts=1,max_nfev=25))
    assert len(history)==2
    assert history[0].next_asof=='2026-10-28'
    assert history[0].next_date_cross_section is not None
    assert history[0].next_date_cross_section.rmse_iv < .002
    assert history[1].next_date_cross_section is None
    assert history[0].calibration.bid_ask_coverage == 1.
    assert history[0].calibration.bid_ask_n == 9
    summary=history_summary(history)
    assert summary[0]['next_asof']=='2026-10-28'
    assert summary[1]['next_date_iv_rmse'] is None


def test_chronology_and_asof_safeguards():
    a=make_day('2026-10-27');b=make_day('2026-10-28')
    with pytest.raises(ValueError,match='ascending'):
        calibrate_history([b,a])
    with pytest.raises(ValueError,match='ascending'):
        calibrate_history([a,a])
    with pytest.raises(ValueError,match='valuation dates'):
        replace(a,asof='2026-10-26')
    with pytest.raises(ValueError,match='ticker'):
        calibrate_history([a,replace(b,ticker='OTHER')])
    with pytest.raises(ValueError,match='American'):
        replace(a,exercise_style='american')
    with pytest.raises(ValueError,match='nonempty'):
        calibrate_history([])


def test_collector_curve_date_and_flat_curve_equivalence():
    f=_synthetic_dataframe()
    m=EquityMarket(100.,.02,.01)
    c=EquityCarryCurves(100.,DiscountCurve.flat(.02,2.),
                        DiscountCurve.flat(.01,2.),asof='2026-10-27')
    flat,report1=filter_option_quotes(f,m,exercise_style='european')
    term,report2=filter_option_quotes(f,m,exercise_style='european',carry_curves=c)
    assert [(v.strike,v.maturity,v.mid) for v in term] == [
           (v.strike,v.maturity,v.mid) for v in flat]
    assert report1.selected_quotes==report2.selected_quotes
    bad=replace(c,asof='2026-10-28')
    with pytest.raises(ValueError,match='asof'):
        filter_option_quotes(f,m,exercise_style='european',carry_curves=bad)


def test_curve_only_needs_to_cover_selected_expiries():
    f=_synthetic_dataframe()
    m=EquityMarket(100.,.02,.01)
    c=EquityCarryCurves(100.,DiscountCurve.flat(.02,.5),
                        DiscountCurve.flat(.01,.5),asof='2026-10-27')
    quotes, report=filter_option_quotes(f,m,exercise_style='european',carry_curves=c)
    assert quotes
    assert all(q.maturity <= .5 for q in quotes)

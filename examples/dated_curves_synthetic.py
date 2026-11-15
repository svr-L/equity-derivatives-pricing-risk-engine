"""Purely synthetic dated curves + Heston pricing/calibration example (no data files)."""
from math import exp
from equity_derivatives import (DiscountCurve, EquityCarryCurves, EquityMarket,
    HestonParams, VanillaQuote, CalibrationConfig, calibrate_heston,
    heston_surface_prices)

market=EquityMarket(100.)
D=DiscountCurve((.25, .75, 1.5), tuple(exp(-r*t) for r,t in
       zip((.025,.03,.032),(.25,.75,1.5))))
F=(100.2,100.9,102.1)
curves=EquityCarryCurves.from_forwards(100.,D,(.25,.75,1.5),F,asof='2026-10-27')
true=HestonParams(2.5,.05,.6,-.65,.045)
ks=[90.,100.,110.]*3
ts=[.25]*3+[.75]*3+[1.5]*3
px=heston_surface_prices(market,true,ks,ts,carry_curves=curves)
quotes=[VanillaQuote(k,t,float(v)) for k,t,v in zip(ks,ts,px)]
fit=calibrate_heston(market,quotes,initial=true,
     config=CalibrationConfig(n_starts=1,max_nfev=10),carry_curves=curves)
print('Synthetic forward(1.5):',curves.forward(1.5))
print('Fit params:',fit.params_dict)
print('Price RMSE:',fit.market_rmse_price)
print('Assumptions:',fit.assumptions)

"""Self-contained, European-style synthetic Heston calibration demo."""
from __future__ import annotations

import numpy as np

from equity_derivatives import (
    CalibrationConfig, EquityMarket, HestonParams, VanillaQuote,
    calibrate_heston, evaluate_heston_fit, heston_european,
)


def main():
    market = EquityMarket(spot=100., rate=.02, dividend_yield=.01)
    truth = HestonParams(2.5, .05, .6, -.65, .045)
    k = np.tile(np.array([80, 90, 100, 110, 120], dtype=float), 4)
    t = np.repeat([.25, .5, 1., 2.], len(k)//4)
    calls = k >= market.spot
    prices = np.array([heston_european(market, truth, float(kk), float(tt), bool(cc))
                       for kk,tt,cc in zip(k,t,calls)])  # independent scalar quadrature
    quotes = [VanillaQuote(float(s),float(tt),float(p),bool(c))
              for s, tt, p, c in zip(k, t, prices, calls)]
    training = [q for i,q in enumerate(quotes) if i % 5 != 0]
    holdout = [q for i,q in enumerate(quotes) if i % 5 == 0]
    result = calibrate_heston(market, training, config=CalibrationConfig(n_starts=4))
    test = evaluate_heston_fit(market, result.params, holdout)
    print('True Heston params:      ', truth)
    print('Calibrated Heston params:', result.params)
    print('Training IV RMSE:', f'{100*result.market_rmse_iv:.4f} vol points')
    print('Holdout  IV RMSE:', f'{100*test.rmse_iv:.4f} vol points')
    print('Jacobian condition:', result.jacobian_condition_number)
    print('WARNING: synthetic model-recovery experiment; not live market accuracy')


if __name__ == '__main__':
    main()

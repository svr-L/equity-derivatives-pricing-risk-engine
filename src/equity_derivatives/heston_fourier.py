"""Semi-analytic Heston vanilla European prices by Lewis Fourier integral.

Risk-neutral log-return CF with Little-Trap orientation and quadrature on a
finite interval. Intended for valuation/benchmarking, not yet a calibrated
market-surface production implementation. Requires flat r and q.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from .models import EquityMarket, HestonParams
from .vanilla import black_scholes


def heston_log_return_cf(u: complex, params: HestonParams, maturity: float,
                         rate: float = 0., dividend_yield: float = 0.) -> complex:
    """E_Q[exp(i*u*log(S_T/S_0))], complex u (strip including Im(u)=-1/2)."""
    if maturity < 0 or not math.isfinite(maturity):
        raise ValueError("maturity must be finite and nonnegative")
    if not all(math.isfinite(z) for z in (rate, dividend_yield)):
        raise ValueError("rates must be finite")
    if maturity == 0:
        return 1. + 0j
    k, t, xi, rho, v0 = (params.kappa, params.theta, params.xi,
                           params.rho, params.v0)
    drift = rate - dividend_yield
    if xi == 0:
        # Deterministic mean-reverting variance: exact total integrated variance.
        int_var = t*maturity + (v0-t)*(-math.expm1(-k*maturity))/k
        return complex(np.exp(1j*u*drift*maturity - 0.5*int_var*(u*u+1j*u)))
    a = k - rho*xi*1j*u
    d = np.sqrt(a*a + xi*xi*(u*u + 1j*u))
    if d.real < 0:
        d = -d
    g = (a-d)/(a+d)
    e = np.exp(-d*maturity)
    one_minus_g = 1. - g
    one_minus_g_e = 1. - g*e
    c = (1j*u*drift*maturity + k*t/(xi*xi)
         * ((a-d)*maturity - 2.*np.log(one_minus_g_e/one_minus_g)))
    dd = (a-d)/(xi*xi) * (1.-e)/one_minus_g_e
    return complex(np.exp(c + dd*v0))


def heston_european(market: EquityMarket, params: HestonParams,
                    strike: float, maturity: float, is_call: bool = True,
                    *, integration_upper: float = 200., epsabs: float = 1e-8) -> float:
    """Price European by Lewis integral; put from put-call parity.

    'integration_upper' is the Fourier truncation boundary and must be
    convergence-tested for extreme Heston parameters or tiny maturities.
    The ξ=0 case uses a *closed-form* deterministic-integrated-variance BS
    benchmark instead of performing unnecessary quadrature.
    """
    if not math.isfinite(strike) or strike <= 0 or not math.isfinite(maturity) or maturity < 0:
        raise ValueError("require strike>0 and maturity>=0")
    if not math.isfinite(integration_upper) or integration_upper <= 0:
        raise ValueError("integration_upper must be positive")
    if not math.isfinite(epsabs) or epsabs <= 0:
        raise ValueError("epsabs must be positive")
    if maturity == 0:
        return max((1 if is_call else -1)*(market.spot-strike), 0.)
    if params.xi == 0:
        int_var = params.theta*maturity + (params.v0 - params.theta) * (
            -math.expm1(-params.kappa*maturity)) / params.kappa
        return black_scholes(market, strike, maturity,
                             math.sqrt(max(int_var, 0.)/maturity), is_call)
    log_moneyness = math.log(market.spot/strike)

    def integrand(u: float) -> float:
        cf = heston_log_return_cf(u - 0.5j, params, maturity,
                                  market.rate, market.dividend_yield)
        return float((np.exp(1j*u*log_moneyness)*cf).real/(u*u+0.25))

    integral, _ = quad(integrand, 0., integration_upper,
                       epsabs=epsabs, epsrel=1e-8, limit=500)
    call = (market.spot*math.exp(-market.dividend_yield*maturity)
            - math.exp(-market.rate*maturity)*math.sqrt(market.spot*strike)
            * integral/math.pi)
    if is_call:
        return float(call)
    return float(call - market.spot*math.exp(-market.dividend_yield*maturity)
                 + strike*math.exp(-market.rate*maturity))

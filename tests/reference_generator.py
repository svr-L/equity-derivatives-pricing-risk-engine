"""Regenerate the reference values used by test_reference_values.py and
test_mc_accuracy.py. Not collected by pytest. Requires mpmath and QuantLib.

    python tests/reference_generator.py

Heston European: 40-digit mpmath Lewis integral (classic little-trap CF, an
implementation independent from the package) and QuantLib AnalyticHestonEngine
(AngledContour + exp-sinh); prints both and their difference.
Barriers: QuantLib AnalyticBarrierEngine (BS) and FdHestonBarrierEngine (Heston).
Greeks: QuantLib central differences with bump_greeks' default bumps.
"""

import mpmath as mp
import QuantLib as ql
from test_reference_values import BARRIER_REFERENCE, HESTON_REFERENCE

mp.mp.dps = 40
TODAY = ql.Date(19, 11, 2026)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()


def flat(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DC, ql.Continuous))


def heston_model(spot, r, q, p):
    kappa, theta, xi, rho, v0 = p
    return ql.HestonModel(
        ql.HestonProcess(
            flat(r), flat(q), ql.QuoteHandle(ql.SimpleQuote(spot)), v0, kappa, theta, xi, rho
        )
    )


def ql_heston(spot, r, q, p, days, strike, is_call):
    engine = ql.AnalyticHestonEngine(
        heston_model(spot, r, q, p),
        ql.AnalyticHestonEngine.AngledContour,
        ql.FourierIntegration.expSinh(1e-13),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        ql.EuropeanExercise(TODAY + days),
    )
    option.setPricingEngine(engine)
    return option.NPV()


def mp_heston(spot, r, q, p, days, strike, is_call):
    kappa, theta, xi, rho, v0 = (mp.mpf(x) for x in p)
    S, r, q, K = (mp.mpf(x) for x in (spot, r, q, strike))
    T = mp.mpf(days) / 365
    lm = mp.log(S / K)

    def cf(z):
        a = kappa - rho * xi * 1j * z
        d = mp.sqrt(a * a + xi**2 * (z * z + 1j * z))
        if mp.re(d) < 0:
            d = -d
        g = (a - d) / (a + d)
        e = mp.exp(-d * T)
        c = 1j * z * (r - q) * T + kappa * theta / xi**2 * (
            (a - d) * T - 2 * mp.log((1 - g * e) / (1 - g))
        )
        return mp.exp(c + (a - d) / xi**2 * (1 - e) / (1 - g * e) * v0)

    upper = mp.mpf(16)
    while abs(cf(upper - 0.5j)) / upper > mp.mpf("1e-22") and upper < 2**18:
        upper *= 2
    step = mp.mpf(4) if abs(lm) < 0.5 else mp.mpf(2)
    points = [mp.mpf(x) for x in ("0", "0.25", "0.5", "1", "2", "4")]
    while points[-1] < upper:
        points.append(min(points[-1] + step * max(1, points[-1] / 64), upper))
    integral = mp.quad(
        lambda u: mp.re(mp.exp(1j * u * lm) * cf(u - 0.5j)) / (u * u + 0.25), points, maxdegree=8
    )
    call = S * mp.exp(-q * T) - mp.exp(-r * T) * mp.sqrt(S * K) * integral / mp.pi
    return call if is_call else call - S * mp.exp(-q * T) + K * mp.exp(-r * T)


print("# HESTON_REFERENCE (mpmath value; QuantLib - mpmath in comment)")
for label, spot, r, q, p, days, strike, is_call, _ in HESTON_REFERENCE:
    ref = mp_heston(spot, r, q, p, days, strike, is_call)
    diff = ql_heston(spot, r, q, p, days, strike, is_call) - float(ref)
    print(
        f"    ({label!r}, {spot}, {r}, {q}, {p}, {days}, {strike}, {is_call}, {mp.nstr(ref, 15)}),"
        f"  # QL-mp={diff:+.1e}",
        flush=True,
    )

print("# BARRIER_REFERENCE: S=100 r=.01 q=.02 vol=.2 T=1 (365 days)")
vts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), 0.2, DC))
process = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(ql.SimpleQuote(100.0)), flat(0.02), flat(0.01), vts
)
kinds = {
    ("up", "out"): ql.Barrier.UpOut,
    ("up", "in"): ql.Barrier.UpIn,
    ("down", "out"): ql.Barrier.DownOut,
    ("down", "in"): ql.Barrier.DownIn,
}
for strike, barrier, direction, knock, is_call, _ in BARRIER_REFERENCE:
    option = ql.BarrierOption(
        kinds[(direction, knock)],
        barrier,
        0.0,
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        ql.EuropeanExercise(TODAY + 365),
    )
    option.setPricingEngine(ql.AnalyticBarrierEngine(process))
    print(f"    ({strike}, {barrier}, {direction!r}, {knock!r}, {is_call}, {option.NPV()!r}),")

print("# Greeks (doc parameters, K=100, T=1): spot bump 1%, v0 bump 0.001")
doc = (2.0, 0.04, 0.5, -0.7, 0.04)
price = {
    key: ql_heston(spot, 0.01, 0.02, (*doc[:4], v0), 365, 100.0, True)
    for key, spot, v0 in (
        ("0", 100.0, 0.04),
        ("s+", 101.0, 0.04),
        ("s-", 99.0, 0.04),
        ("v+", 100.0, 0.041),
        ("v-", 100.0, 0.039),
    )
}
print(
    f"    delta={(price['s+'] - price['s-']) / 2!r} "
    f"gamma={price['s+'] - 2 * price['0'] + price['s-']!r} "
    f"v0_sensitivity={(price['v+'] - price['v-']) / 0.002!r}"
)

print("# Continuous Heston up-and-out call K=100 H=125 (doc parameters, T=1): FD grids")
for grid in ((100, 200, 50), (200, 400, 100), (400, 800, 200)):
    option = ql.BarrierOption(
        ql.Barrier.UpOut,
        125.0,
        0.0,
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
        ql.EuropeanExercise(TODAY + 365),
    )
    option.setPricingEngine(ql.FdHestonBarrierEngine(heston_model(100.0, 0.01, 0.02, doc), *grid))
    print(f"    grid {grid}: {option.NPV()!r}", flush=True)

"""Conditional survival weighting for continuously monitored barriers.

For *constant* variance (xi=0 and v0=theta) this formula is
exact conditional on GBM endpoints: reflection principle applied to log S.
For stochastic Heston variance, conditioning only on the simulated endpoint
variances is NOT exact because leverage/correlation and within-step stochastic
variance paths are ignored. This is a documented approximation, not an
unbiased continuous-time Heston price nor an exact bridge of the QE-M scheme.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import HestonParams, MCConfig
from .payoffs import Barrier
from .simulation import HestonPaths


def barrier_survival_probabilities(paths: HestonPaths, params: HestonParams,
                                   config: MCConfig, option: Barrier) -> NDArray[np.float64]:
    """Compute conditional no-touch weights in [0,1] for each whole path.

    Under Euler, step integrated variance is pre-step positive v*dt.
    Under QE-M with xi>0, trapezoidal v*dt is an approximation.
    Under QE-M with xi=0, deterministic integrated variance is exact,
    but the bridge survival formula is exact only if variance is constant.
    """
    s = paths.spot
    v = paths.variance
    if s.shape != v.shape or s.shape != (config.n_steps+1, config.n_paths):
        raise ValueError("path shape and MCConfig mismatch")
    dt = config.maturity/config.n_steps
    if option.direction == "up":
        dist = np.log(option.barrier/s)
    else:
        dist = np.log(s/option.barrier)
    # Any sampled endpoint at/beyond barrier kills the entire survival weight.
    touched = np.any(dist <= 0., axis=0)
    if config.scheme == "full_truncation_euler":
        integrated = dt * v[:-1]
    elif params.xi == 0:
        integrated = (params.theta*dt + (v[:-1]-params.theta)
                      * (-np.expm1(-params.kappa*dt))/params.kappa)
    else:
        integrated = .5*dt*(v[:-1]+v[1:])
    prod = np.maximum(dist[:-1],0.)*np.maximum(dist[1:],0.)
    ratio = np.divide(2.*prod, integrated, out=np.full_like(prod, np.inf),
                      where=integrated > 0.)
    # stable no-hit probability: -expm1(-ratio); sum logs to avoid underflow.
    # For zero variance, no unobserved crossings between distinct safe nodes.
    with np.errstate(divide="ignore"):
        log_survival = np.log(-np.expm1(-ratio)).sum(axis=0)
    weight = np.exp(log_survival)
    weight[touched] = 0.
    return np.clip(weight, 0., 1.)

"""Deterministic surrogate prediction of PrPl stopping times (no fitting).

For the wedge sampler Y ~ Bernoulli(nu), sigma^2 = nu(1 - nu). Replace the
running variance by sigma^2 (EB) or 1/4 (Hoeffding), and the EB penalty
4 (Y_i - mu_{i-1})^2 psi_E(lambda_i) by its leading expectation
4 sigma^2 psi_E(lambda_i). The radius is then deterministic:
  r(t) = (log(2/alpha) + sum_{i<=t} pen_i) / sum_{i<=t} lambda_i,
and the predicted stop is the first monitoring time t >= t0 with
2 r(t) / nu <= eps. Computed in chunks, so memory stays small."""
import math

import numpy as np

ALPHA = 0.10
T0 = 100
STEP = 100
CAP = 0.5
CHUNK = 1_000_000


def predicted_stop(nu, eps, variance, horizon, sigma2=None):
    """sigma2: Var(Y) for the EB variant; defaults to the Bernoulli value
    nu(1 - nu) (wedge sampler). For other samplers pass the exact variance."""
    log_term = math.log(2.0 / ALPHA)
    if variance == "hoeffding":
        sigma2 = 0.25
    elif sigma2 is None:
        sigma2 = nu * (1.0 - nu)
    sum_lam = 0.0
    sum_pen = 0.0
    for start in range(1, horizon + 1, CHUNK):
        t = np.arange(start, min(start + CHUNK, horizon + 1), dtype=float)
        lam = np.minimum(np.sqrt(2.0 * log_term / (sigma2 * t * np.log1p(t))), CAP)
        if variance == "hoeffding":
            pen = lam ** 2 / 8.0
        else:
            pen = 4.0 * sigma2 * (-np.log1p(-lam) - lam) / 4.0
        cum_lam = sum_lam + np.cumsum(lam)
        cum_pen = sum_pen + np.cumsum(pen)
        radius = (log_term + cum_pen) / cum_lam
        is_check = (t >= T0) & (t % STEP == 0)
        ok = np.nonzero(is_check & (2.0 * radius / nu <= eps))[0]
        if len(ok):
            return int(t[ok[0]])
        sum_lam = float(cum_lam[-1])
        sum_pen = float(cum_pen[-1])
    return None

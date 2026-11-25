"""
Four anytime-valid confidence sequences (CSs) for the mean nu of i.i.d.
observations Y_i in [0, 1], designed as a matched ablation.

All four arms share: the level alpha (alpha/2 per side, union bound), the
predictable bet sizes lambda_t capped at 1/2, the monitoring schedule (the
caller decides when to call `interval()`), and the running intersection over
monitoring times. They differ in exactly two switches:

  construction  "betting": the set of m not rejected by either one-sided
                capital process  K_t^{+/-}(m) = prod_i (1 +/- lambda_i (Y_i - m))
                (Waudby-Smith & Ramdas 2024, Sec. 4; Ville's inequality).
                "prpl": the closed-form predictable plug-in CSs of
                Waudby-Smith & Ramdas (2024), Prop. 1 (Hoeffding) and Thm. 2
                (empirical Bernstein).
  variance      "eb": lambda_t uses the running variance estimate.
                "hoeffding": the variance estimate is replaced by its
                worst case 1/4 (and, for prpl, the Hoeffding penalty
                lambda^2/8 replaces the empirical-Bernstein penalty).

Running estimates follow Waudby-Smith & Ramdas (2024):
  mu_t     = (1/2 + sum_{i<=t} Y_i) / (t + 1)
  sigma2_t = (1/4 + sum_{i<=t} (Y_i - mu_i)^2) / (t + 1)
  lambda_t = min(1/2, sqrt(2 log(2/alpha) / (sigma2_{t-1} t log(1 + t))))
with sigma2_{t-1} := 1/4 for the "hoeffding" variant.

Betting endpoints are found by bisection on each side separately (each
one-sided capital is monotone in m). Capital is evaluated in O(K) per probe
from running power sums (see `_PowerSumCapital`) with a rigorous truncation
bound. A candidate counts as rejected only when the bound CERTIFIES
log K >= threshold; an uncertified candidate is treated as not rejected. Every
reported endpoint is therefore either the previous endpoint or a certified
rejected point, so the reported interval always contains the exact one, and
no step ever costs O(t).
"""

import math
import numpy as np

CAP = 0.5
N_TERMS = 48
DECISION_MARGIN = 1e-9
BISECTION_REL_TOL = 1e-6
BISECTION_MAX_ITERS = 200


class _RunningEstimates:
    """Predictable lambda sequence, computed chunk by chunk."""

    def __init__(self, alpha, variance):
        self.log_term = math.log(2.0 / alpha)
        self.variance = variance
        self.t = 0
        self.sum_y = 0.0
        self.sum_sq_dev = 0.0

    def lambdas_and_prev_means(self, y):
        """Return (lambda_i, mu_{i-1}) for the samples in chunk y, and update."""
        n = len(y)
        idx = np.arange(self.t + 1, self.t + n + 1, dtype=float)
        cum_y = self.sum_y + np.cumsum(y)
        mu_i = (0.5 + cum_y) / (idx + 1.0)
        mu_prev = np.empty(n)
        mu_prev[0] = (0.5 + self.sum_y) / (self.t + 1.0)
        mu_prev[1:] = mu_i[:-1]
        cum_sq = self.sum_sq_dev + np.cumsum((y - mu_i) ** 2)
        sigma2_prev = np.empty(n)
        # sigma2_{t-1} = (1/4 + sum_{i<=t-1} (Y_i - mu_i)^2) / t, at global time t = idx
        sigma2_prev[0] = (0.25 + self.sum_sq_dev) / idx[0]
        sigma2_prev[1:] = (0.25 + cum_sq[:-1]) / idx[1:]
        if self.variance == "hoeffding":
            sigma2_prev = np.full(n, 0.25)
        denominator = sigma2_prev * idx * np.log1p(idx)
        if not np.all(denominator > 0):
            bad = int(np.argmin(denominator > 0))
            raise ValueError(f"non-positive lambda denominator at chunk position {bad}: "
                             f"t={idx[bad]}, sigma2={sigma2_prev[bad]!r}, state t={self.t}, "
                             f"sum_y={self.sum_y!r}, sum_sq_dev={self.sum_sq_dev!r}, "
                             f"y={y[bad]!r}, dtype={y.dtype}")
        lam = np.minimum(np.sqrt(2.0 * self.log_term / denominator), CAP)
        self.t += n
        self.sum_y = float(cum_y[-1])
        self.sum_sq_dev = float(cum_sq[-1])
        return lam, mu_prev


class _PowerSumCapital:
    """Log-capital of the two one-sided betting processes at any m in [0,1],
    from running sums, with a certified error bound.

    log(1 + lam(Y - m)) = log(1 + lam Y) + log(1 - m r_plus),  r_plus = lam/(1+lam Y) <= 1/2
    log(1 - lam(Y - m)) = log(1 - lam Y) + log(1 + m r_minus), r_minus = lam/(1-lam Y) <= 1
    expanded in powers of m; tail bounds:
      upper process: 0 <= tail <= m^(K+1) S+_(K+1) / ((K+1)(1 - m/2))
      lower process: |tail| <= m^(K+1) S-_(K+1) / (K+1)   (alternating, terms decreasing)
    """

    def __init__(self):
        self.a_plus = 0.0
        self.a_minus = 0.0
        self.s_plus = np.zeros(N_TERMS + 2)
        self.s_minus = np.zeros(N_TERMS + 2)
        self.ks = np.arange(1, N_TERMS + 2, dtype=float)
        self.inv_k = 1.0 / np.arange(1, N_TERMS + 1, dtype=float)
        self.signs = np.where(np.arange(1, N_TERMS + 1) % 2 == 1, 1.0, -1.0)
        self.n_uncertified = 0

    def add(self, y, lam):
        self.a_plus += float(np.sum(np.log1p(lam * y)))
        self.a_minus += float(np.sum(np.log1p(-lam * y)))
        r_plus = lam / (1.0 + lam * y)
        r_minus = lam / (1.0 - lam * y)
        self.s_plus[1:] += np.power.outer(r_plus, self.ks).sum(axis=0)
        self.s_minus[1:] += np.power.outer(r_minus, self.ks).sum(axis=0)

    def upper_log_capital_bounds(self, m):
        """Return (low, high, scale): low <= log K^+_t(m) <= high; scale bounds
        the magnitude of the summed terms (for floating-point slack)."""
        powers = m ** self.ks[:N_TERMS]
        series = float(np.dot(powers, self.s_plus[1:N_TERMS + 1] * self.inv_k))
        value = self.a_plus - series
        tail = (m ** (N_TERMS + 1)) * self.s_plus[N_TERMS + 1] / ((N_TERMS + 1) * (1.0 - m / 2.0))
        return value - tail, value, abs(self.a_plus) + abs(series)

    def lower_log_capital_bounds(self, m):
        powers = m ** self.ks[:N_TERMS]
        series = float(np.dot(self.signs * powers, self.s_minus[1:N_TERMS + 1] * self.inv_k))
        value = self.a_minus + series
        tail = (m ** (N_TERMS + 1)) * self.s_minus[N_TERMS + 1] / (N_TERMS + 1)
        scale = abs(self.a_minus) + float(np.dot(powers, self.s_minus[1:N_TERMS + 1] * self.inv_k))
        return value - tail, value + tail, scale

    def rejects(self, m, side, threshold):
        """True only if log K^{side}_t(m) >= threshold is certified by the
        truncation bound plus a floating-point margin; otherwise False."""
        if side > 0:
            low, high, scale = self.upper_log_capital_bounds(m)
        else:
            low, high, scale = self.lower_log_capital_bounds(m)
        slack = DECISION_MARGIN * (1.0 + scale)
        if low >= threshold + slack:
            return True
        if high >= threshold - slack:
            self.n_uncertified += 1
        return False


class ConfidenceSequenceArm:
    """One arm of the matched ablation; feed chunks with `add`, query `interval`."""

    def __init__(self, construction, variance, alpha):
        assert construction in ("betting", "prpl")
        assert variance in ("eb", "hoeffding")
        self.construction = construction
        self.variance = variance
        self.alpha = alpha
        self.threshold = math.log(2.0 / alpha)
        self.estimates = _RunningEstimates(alpha, variance)
        self.lower = 0.0
        self.upper = 1.0
        self.t = 0
        if construction == "betting":
            self.capital = _PowerSumCapital()
        else:
            self.sum_lam = 0.0
            self.sum_lam_y = 0.0
            self.sum_penalty = 0.0

    @property
    def name(self):
        return f"{self.construction}-{self.variance}"

    def add(self, y):
        lam, mu_prev = self.estimates.lambdas_and_prev_means(y)
        self.t += len(y)
        if self.construction == "betting":
            self.capital.add(y, lam)
            return
        self.sum_lam += float(lam.sum())
        self.sum_lam_y += float(np.dot(lam, y))
        if self.variance == "hoeffding":
            self.sum_penalty += float(np.sum(lam ** 2) / 8.0)
        else:
            v = 4.0 * (y - mu_prev) ** 2
            psi_e = (-np.log1p(-lam) - lam) / 4.0
            self.sum_penalty += float(np.dot(v, psi_e))

    def interval(self):
        """Running-intersection CS at the current time, in Y units.
        Returns (L, U); L > U means the intersection is empty."""
        if self.construction == "prpl":
            center = self.sum_lam_y / self.sum_lam
            radius = (self.threshold + self.sum_penalty) / self.sum_lam
            self.lower = max(self.lower, center - radius)
            self.upper = min(self.upper, center + radius)
            return self.lower, self.upper
        self.lower = max(self.lower, self._betting_lower_endpoint())
        self.upper = min(self.upper, self._betting_upper_endpoint())
        return self.lower, self.upper

    def advance(self, y, step, eps, t0, cap):
        """Feed a batch y (length a multiple of `step`), evaluating the
        running-intersection CS at every `step` samples exactly as repeated
        add(step samples) + interval() would. Returns None if no stopping
        event occurs in the batch, else (outcome, T, L, U) at the first
        checkpoint where the CS is empty ("empty"), the relative width is
        at most eps with T >= t0 ("stopped"), or T >= cap ("cap")."""
        if self.construction == "betting":
            for start in range(0, len(y), step):
                self.add(y[start:start + step])
                L, U = self.interval()
                outcome = stopping_outcome(L, U, self.t, eps, t0, cap)
                if outcome:
                    return outcome, self.t, L, U
            return None
        lam, mu_prev = self.estimates.lambdas_and_prev_means(y)
        cum_lam = self.sum_lam + np.cumsum(lam)
        cum_lam_y = self.sum_lam_y + np.cumsum(lam * y)
        if self.variance == "hoeffding":
            penalty = lam ** 2 / 8.0
        else:
            penalty = 4.0 * (y - mu_prev) ** 2 * (-np.log1p(-lam) - lam) / 4.0
        cum_pen = self.sum_penalty + np.cumsum(penalty)
        positions = np.arange(step - 1, len(y), step)
        center = cum_lam_y[positions] / cum_lam[positions]
        radius = (self.threshold + cum_pen[positions]) / cum_lam[positions]
        lower = np.maximum.accumulate(np.concatenate([[self.lower], center - radius]))[1:]
        upper = np.minimum.accumulate(np.concatenate([[self.upper], center + radius]))[1:]
        times = self.t + positions + 1
        for j in range(len(positions)):
            outcome = stopping_outcome(lower[j], upper[j], int(times[j]), eps, t0, cap)
            if outcome:
                self.t = int(times[j])
                self.lower, self.upper = float(lower[j]), float(upper[j])
                return outcome, self.t, self.lower, self.upper
        self.t = int(times[-1])
        self.sum_lam = float(cum_lam[-1])
        self.sum_lam_y = float(cum_lam_y[-1])
        self.sum_penalty = float(cum_pen[-1])
        self.lower, self.upper = float(lower[-1]), float(upper[-1])
        return None

    def _betting_lower_endpoint(self):
        # Upper-side capital decreases in m: rejected below the endpoint.
        start = self.lower
        if not self.capital.rejects(start, +1, self.threshold):
            return start
        rejected, accepted = start, 1.0
        for _ in range(BISECTION_MAX_ITERS):
            if accepted - rejected <= BISECTION_REL_TOL * max(accepted, 1e-300):
                break
            mid = 0.5 * (rejected + accepted)
            if self.capital.rejects(mid, +1, self.threshold):
                rejected = mid
            else:
                accepted = mid
        return rejected

    def _betting_upper_endpoint(self):
        # Lower-side capital increases in m: rejected above the endpoint.
        start = self.upper
        if not self.capital.rejects(start, -1, self.threshold):
            return start
        accepted, rejected = 0.0, start
        for _ in range(BISECTION_MAX_ITERS):
            if rejected - accepted <= BISECTION_REL_TOL * max(rejected, 1e-300):
                break
            mid = 0.5 * (accepted + rejected)
            if self.capital.rejects(mid, -1, self.threshold):
                rejected = mid
            else:
                accepted = mid
        return rejected


def stopping_outcome(L, U, t, eps, t0, cap):
    """The stopping rule shared by every monitor."""
    if L > U:
        return "empty"
    if t >= t0:
        mid = 0.5 * (L + U)
        if mid > 0 and (U - L) / mid <= eps:
            return "stopped"
    if t >= cap:
        return "cap"
    return None


def make_matched_arms(alpha):
    return [ConfidenceSequenceArm("betting", "eb", alpha),
            ConfidenceSequenceArm("betting", "hoeffding", alpha),
            ConfidenceSequenceArm("prpl", "eb", alpha),
            ConfidenceSequenceArm("prpl", "hoeffding", alpha)]

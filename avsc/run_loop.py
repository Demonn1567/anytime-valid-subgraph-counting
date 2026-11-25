"""
The run loop shared by the triangle and k-clique experiments.

One i.i.d. sample stream is drawn in batches of BATCH samples and fed to every
monitor; each monitor evaluates its interval at every CHECK_EVERY samples
(the monitoring times) and stops independently by avsc.cs_arms.stopping_outcome.
Drawing in larger batches changes nothing statistically: the monitoring times
and the stopping rule are those of a monitor fed CHECK_EVERY samples at a time.

Sampling time is attributed to a monitor that stops inside a batch in
proportion to the samples it used from that batch.
"""
import math
import time

import numpy as np

from .cs_arms import make_matched_arms, stopping_outcome

Z90 = 1.6448536269514722


class WaldPeeking:
    """NOT anytime-valid: 90% normal-approximation interval re-checked at
    every monitoring time (the naive 'peek until narrow' rule)."""
    name = "wald-peeking"

    def __init__(self):
        self.t = 0
        self.total = 0.0
        self.total_sq = 0.0

    def advance(self, y, step, eps, t0, cap):
        cum = self.total + np.cumsum(y)
        cum_sq = self.total_sq + np.cumsum(y * y)
        positions = np.arange(step - 1, len(y), step)
        times = (self.t + positions + 1).astype(float)
        mean = cum[positions] / times
        var = np.maximum(cum_sq[positions] / times - mean * mean, 0.0) * times / (times - 1.0)
        half = Z90 * np.sqrt(var / times)
        for j in range(len(positions)):
            L, U = float(mean[j] - half[j]), float(mean[j] + half[j])
            outcome = stopping_outcome(L, U, int(times[j]), eps, t0, cap)
            if outcome:
                self.t = int(times[j])
                return outcome, self.t, L, U
        self.t = int(times[-1])
        self.total = float(cum[-1])
        self.total_sq = float(cum_sq[-1])
        return None


def run_monitors(sampler, nu, eps, alpha, t0, check_every, batch, cap):
    """Run the four matched CSs and the Wald peeking rule on one stream.
    Returns {monitor name: dict(outcome, T, L, U, mean, sampling_s, update_s, uncertified)}."""
    assert batch % check_every == 0 and cap % batch == 0
    monitors = make_matched_arms(alpha) + [WaldPeeking()]
    active = list(monitors)
    results = {}
    update_s = {m.name: 0.0 for m in monitors}
    sampling_s = 0.0
    total = 0.0
    t = 0
    while active:
        s0 = time.perf_counter()
        y = sampler.draw(batch)
        batch_s = time.perf_counter() - s0
        cum = total + np.cumsum(y)
        still = []
        for mon in active:
            u0 = time.perf_counter()
            event = mon.advance(y, check_every, eps, t0, cap)
            update_s[mon.name] += time.perf_counter() - u0
            if event is None:
                still.append(mon)
                continue
            outcome, T, L, U = event
            used = T - t
            results[mon.name] = {
                "outcome": outcome, "T": T, "L": L, "U": U,
                "mean": float(cum[used - 1]) / T,
                "sampling_s": sampling_s + batch_s * used / batch,
                "update_s": update_s[mon.name],
                "uncertified": mon.capital.n_uncertified if hasattr(mon, "capital") else 0,
            }
        active = still
        sampling_s += batch_s
        total = float(cum[-1])
        t += batch
    return [(m.name, results[m.name]) for m in monitors]


def result_row(name, info, nu):
    L, U, mean = info["L"], info["U"], info["mean"]
    mid = 0.5 * (L + U)
    return {"monitor": name, "outcome": info["outcome"], "T": info["T"],
            "L_y": repr(L), "U_y": repr(U), "nu": repr(nu), "covered": bool(L <= nu <= U),
            "mid_rel_err": repr((mid - nu) / nu), "mean_at_T": repr(mean),
            "mean_in_interval": bool(L <= mean <= U), "mean_rel_err": repr((mean - nu) / nu),
            "sampling_s_at_T": repr(info["sampling_s"]), "update_s": repr(info["update_s"]),
            "uncertified": info["uncertified"]}

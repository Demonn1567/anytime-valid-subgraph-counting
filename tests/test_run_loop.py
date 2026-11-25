"""Batched evaluation (advance) must equal step-by-step evaluation.

Run:  .venv/bin/python tests/test_run_loop.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from avsc.cs_arms import ConfidenceSequenceArm, stopping_outcome
from avsc.run_loop import Z90, WaldPeeking

ALPHA = 0.10
STEP = 100
BATCH = 10_000
T0 = 100


def stepwise_arm(construction, variance, y, eps, cap):
    arm = ConfidenceSequenceArm(construction, variance, ALPHA)
    for start in range(0, len(y), STEP):
        arm.add(y[start:start + STEP])
        L, U = arm.interval()
        outcome = stopping_outcome(L, U, arm.t, eps, T0, cap)
        if outcome:
            return outcome, arm.t, L, U
    return None


def batched(monitor, y, eps, cap):
    for start in range(0, len(y), BATCH):
        event = monitor.advance(y[start:start + BATCH], STEP, eps, T0, cap)
        if event:
            return event
    return None


def stepwise_wald(y, eps, cap):
    for t in range(STEP, len(y) + 1, STEP):
        prefix = y[:t]
        mean = prefix.mean()
        half = Z90 * math.sqrt(prefix.var(ddof=1) / t)
        outcome = stopping_outcome(mean - half, mean + half, t, eps, T0, cap)
        if outcome:
            return outcome, t, mean - half, mean + half
    return None


def streams(rng, n):
    yield "Bernoulli(0.5)", (rng.random(n) < 0.5).astype(float)
    yield "Bernoulli(0.05)", (rng.random(n) < 0.05).astype(float)
    yield "Bernoulli(0.005)", (rng.random(n) < 0.005).astype(float)
    yield "Beta(1,30)*mask", rng.beta(1, 30, n) * (rng.random(n) < 0.4)


def main():
    rng = np.random.default_rng(20261117)
    n = 200_000
    cap = 200_000
    checked = 0
    for label, y in streams(rng, n):
        for eps in (0.10, 0.30, 0.02):
            for construction in ("betting", "prpl"):
                for variance in ("eb", "hoeffding"):
                    a = stepwise_arm(construction, variance, y, eps, cap)
                    b = batched(ConfidenceSequenceArm(construction, variance, ALPHA), y, eps, cap)
                    assert a[0] == b[0] and a[1] == b[1], (label, eps, construction, variance, a, b)
                    assert abs(a[2] - b[2]) <= 1e-12 and abs(a[3] - b[3]) <= 1e-12, (label, eps, a, b)
                    checked += 1
            a = stepwise_wald(y, eps, cap)
            b = batched(WaldPeeking(), y, eps, cap)
            assert a[0] == b[0] and a[1] == b[1], (label, eps, "wald", a, b)
            assert abs(a[2] - b[2]) <= 1e-12 and abs(a[3] - b[3]) <= 1e-12, (label, eps, "wald", a, b)
            checked += 1
            print(f"    {label:17s} eps={eps:.2f} outcomes/T identical for all five monitors")
    print(f"ok  batched evaluation equals step-by-step evaluation ({checked} monitor runs, incl. cap outcomes at eps=0.02)")
    print("ALL TESTS PASSED")


main()

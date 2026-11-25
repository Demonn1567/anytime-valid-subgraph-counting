"""Correctness tests for avsc/cs_arms.py against naive reference implementations.

Run:  .venv/bin/python tests/test_cs_arms.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from avsc.cs_arms import ConfidenceSequenceArm, _RunningEstimates, make_matched_arms

ALPHA = 0.10
THR = math.log(2.0 / ALPHA)


def reference_lambdas(y, variance):
    """Per-sample loop straight from the definitions in the module docstring."""
    lam = []
    mu_prev_list = []
    sum_y = 0.0
    sum_sq = 0.0
    mus = []
    for t in range(1, len(y) + 1):
        sigma2_prev = (0.25 + sum_sq) / t
        mu_prev = (0.5 + sum_y) / t
        if variance == "hoeffding":
            sigma2_prev = 0.25
        value = math.sqrt(2.0 * THR / (sigma2_prev * t * math.log(1.0 + t)))
        lam.append(min(0.5, value))
        mu_prev_list.append(mu_prev)
        sum_y += y[t - 1]
        mu_t = (0.5 + sum_y) / (t + 1)
        mus.append(mu_t)
        sum_sq += (y[t - 1] - mu_t) ** 2
    return np.array(lam), np.array(mu_prev_list)


def test_lambda_sequence(rng):
    for variance in ("eb", "hoeffding"):
        for trial in range(5):
            y = rng.random(1000) if trial % 2 else (rng.random(1000) < 0.05).astype(float)
            ref_lam, ref_mu = reference_lambdas(y, variance)
            for chunk in (1, 7, 100, 1000):
                est = _RunningEstimates(ALPHA, variance)
                parts_lam = []
                parts_mu = []
                for start in range(0, len(y), chunk):
                    lam, mu_prev = est.lambdas_and_prev_means(y[start:start + chunk])
                    parts_lam.append(lam)
                    parts_mu.append(mu_prev)
                assert np.allclose(np.concatenate(parts_lam), ref_lam, rtol=1e-12, atol=1e-14)
                assert np.allclose(np.concatenate(parts_mu), ref_mu, rtol=1e-12, atol=1e-14)
    print("ok  lambda sequence matches per-sample reference for all chunk sizes")


def test_power_sum_bounds(rng):
    worst = 0.0
    for trial in range(40):
        n = int(rng.integers(50, 5000))
        y = rng.random(n) if trial % 2 else (rng.random(n) < rng.uniform(0.002, 0.9)).astype(float)
        arm = ConfidenceSequenceArm("betting", "eb", ALPHA)
        arm.add(y)
        cap = arm.capital
        lam, _ = reference_lambdas(y, "eb")
        for m in np.concatenate([rng.random(20), [0.0, 1e-6, 0.5, 0.9, 0.999, 1.0]]):
            exact_up = float(np.sum(np.log1p(lam * (y - m))))
            exact_lo = float(np.sum(np.log1p(-lam * (y - m))))
            lo_u, hi_u, _ = cap.upper_log_capital_bounds(m)
            lo_l, hi_l, _ = cap.lower_log_capital_bounds(m)
            tol = 1e-9 * (1.0 + abs(exact_up) + abs(exact_lo))
            assert lo_u - tol <= exact_up <= hi_u + tol, (m, lo_u, exact_up, hi_u)
            assert lo_l - tol <= exact_lo <= hi_l + tol, (m, lo_l, exact_lo, hi_l)
            for side, exact in ((+1, exact_up), (-1, exact_lo)):
                decision = cap.rejects(m, side, THR)
                if decision:
                    assert exact >= THR - 1e-12, (m, side, exact)
                elif abs(exact - THR) > 1e-6:
                    assert exact < THR, (m, side, exact)
            if m < 0.8:
                worst = max(worst, hi_u - lo_u, hi_l - lo_l)
    print(f"ok  power-sum bounds bracket the exact log-capital; every certified rejection is a true rejection, and decisions agree with the exact value outside a 1e-6 band; widest bracket for m<0.8: {worst:.2e}")


def grid_reference_betting(y, lam, checkpoints, grid):
    """Exact running-intersection betting CS on a dense grid of m."""
    alive = np.ones(len(grid), dtype=bool)
    results = []
    for c in checkpoints:
        yc = y[:c]
        lc = lam[:c]
        up = np.array([np.sum(np.log1p(lc * (yc - m))) for m in grid])
        lo = np.array([np.sum(np.log1p(-lc * (yc - m))) for m in grid])
        alive &= (up < THR) & (lo < THR)
        results.append(alive.copy())
    return results


def test_betting_endpoints_vs_grid(rng):
    grid = np.linspace(0.0, 1.0, 20001)
    spacing = grid[1] - grid[0]
    for variance in ("eb", "hoeffding"):
        for trial in range(6):
            n = 3000
            p = [0.02, 0.2, 0.55][trial % 3]
            y = (rng.random(n) < p).astype(float) if trial < 3 else np.clip(rng.beta(2, 8 - trial, n), 0, 1)
            arm = ConfidenceSequenceArm("betting", variance, ALPHA)
            checkpoints = list(range(100, n + 1, 100))
            got = []
            for start in range(0, n, 100):
                arm.add(y[start:start + 100])
                if start + 100 >= 100:
                    got.append(arm.interval())
            lam, _ = reference_lambdas(y, variance)
            ref = grid_reference_betting(y, lam, checkpoints, grid)
            for (L, U), alive in zip(got, ref):
                if not alive.any():
                    assert L >= U - 2 * spacing
                    continue
                gl = grid[alive].min()
                gu = grid[alive].max()
                assert L <= gl + 1e-12 and U >= gu - 1e-12, (L, gl, U, gu)
                assert gl - L <= spacing + 1e-6 * gl and U - gu <= spacing + 1e-6 * gu, (L, gl, U, gu)
    print("ok  betting endpoints contain the grid-exact CS and are within one grid step of it")


def test_prpl_vs_loop(rng):
    for variance in ("eb", "hoeffding"):
        y = rng.random(2000) * (rng.random(2000) < 0.3)
        arm = ConfidenceSequenceArm("prpl", variance, ALPHA)
        lam, mu_prev = reference_lambdas(y, variance)
        lower, upper = 0.0, 1.0
        for start in range(0, 2000, 100):
            arm.add(y[start:start + 100])
            L, U = arm.interval()
            t = start + 100
            lt = lam[:t]
            center = np.sum(lt * y[:t]) / np.sum(lt)
            if variance == "hoeffding":
                penalty = np.sum(lt ** 2) / 8.0
            else:
                penalty = np.sum(4.0 * (y[:t] - mu_prev[:t]) ** 2 * (-np.log(1.0 - lt) - lt) / 4.0)
            radius = (THR + penalty) / np.sum(lt)
            lower = max(lower, center - radius)
            upper = min(upper, center + radius)
            assert abs(L - lower) < 1e-12 and abs(U - upper) < 1e-12
    print("ok  closed-form PrPl intervals match a direct loop")


def test_time_uniform_coverage(rng):
    reps = 2000
    horizon = 4000
    print("    Monte Carlo time-uniform miss rate (truth outside CS at ANY checkpoint), alpha=0.10:")
    for label, nu, sampler in (("Bernoulli(0.5)", 0.5, "bern"), ("Bernoulli(0.05)", 0.05, "bern"),
                               ("Bernoulli(0.005)", 0.005, "bern"), ("Beta(1,9)", 0.1, "beta")):
        misses = {}
        for r in range(reps):
            if sampler == "bern":
                y = (rng.random(horizon) < nu).astype(float)
            else:
                y = rng.beta(1, 9, horizon)
            for arm in make_matched_arms(ALPHA):
                missed = False
                for start in range(0, horizon, 100):
                    arm.add(y[start:start + 100])
                    L, U = arm.interval()
                    if not (L <= nu <= U):
                        missed = True
                        break
                misses[arm.name] = misses.get(arm.name, 0) + int(missed)
        line = ", ".join(f"{k}={v / reps:.4f}" for k, v in misses.items())
        print(f"    {label:17s} {line}")
        for name, count in misses.items():
            rate = count / reps
            se = math.sqrt(ALPHA * (1 - ALPHA) / reps)
            assert rate <= ALPHA + 3 * se, (label, name, rate)
    print("ok  every arm's time-uniform miss rate is within alpha + 3 SE")


def main():
    rng = np.random.default_rng(20261117)
    test_lambda_sequence(rng)
    test_power_sum_bounds(rng)
    test_prpl_vs_loop(rng)
    test_betting_endpoints_vs_grid(rng)
    test_time_uniform_coverage(rng)
    print("ALL TESTS PASSED")


main()

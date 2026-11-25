"""
Summaries of the camera-ready result files, shared by make_paper_artifacts.py
and by interactive checks, so that the paper and the checks read the same
numbers.

Conventions:
  coverage   fraction of ALL runs whose final interval contains the truth
             (stopped, capped and empty runs all count; empty = miss)
  T          mean and sample standard deviation over runs that stopped
  speedup    paired, per run: T(arm A) / T(arm B) on the same sample stream,
             only for runs in which both arms stopped; we report the ratio of
             means (the quantity the old paper reported) and the median of
             the paired per-run ratios
"""
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
MONITORS = ["betting-eb", "betting-hoeffding", "prpl-eb", "prpl-hoeffding", "wald-peeking"]


def load(kind):
    paths = sorted(glob.glob(os.path.join(HERE, "results_cr", kind, "*.csv")))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for col in ("covered", "mean_in_interval"):
        frame[col] = frame[col].astype(str) == "True"
    return frame


def wilson(successes, n):
    ci = binomtest(int(successes), int(n)).proportion_ci(0.95, method="wilson")
    return ci.low, ci.high


def midpoint_exceedances(stopped):
    """Number of stopped runs whose midpoint relative error exceeds
    eps/(2-eps). Corollary 1 allows this only on runs that miss; a covered run
    that exceeds the bound would be a bug, so it raises."""
    if not len(stopped):
        return 0
    bound = stopped["eps"] / (2.0 - stopped["eps"])
    exceeds = stopped["mid_rel_err"].abs() > bound + 1e-12
    violations = stopped[exceeds & stopped["covered"]]
    if len(violations):
        raise AssertionError(f"covered runs violating the midpoint bound:\n{violations}")
    return int(exceeds.sum())


def cell_summary(frame, keys):
    rows = []
    for key, group in frame.groupby(keys + ["monitor"], sort=False):
        stopped = group[group["outcome"] == "stopped"]
        n = len(group)
        covered = int(group["covered"].sum())
        low, high = wilson(covered, n)
        t = stopped["T"].astype(float)
        row = dict(zip(keys + ["monitor"], key))
        row.update({
            "runs": n,
            "stopped": len(stopped),
            "capped": int((group["outcome"] == "cap").sum()),
            "empty": int((group["outcome"] == "empty").sum()),
            "T_mean": t.mean() if len(t) else math.nan,
            "T_sd": t.std(ddof=1) if len(t) > 1 else math.nan,
            "T_median": t.median() if len(t) else math.nan,
            "coverage": covered / n,
            "cov_low": low,
            "cov_high": high,
            "mean_outside": float((~stopped["mean_in_interval"]).mean()) if len(stopped) else math.nan,
            "max_abs_mid_rel_err": stopped["mid_rel_err"].abs().max() if len(stopped) else math.nan,
            "mid_exceeds_bound": midpoint_exceedances(stopped),
            "update_s_mean": stopped["update_s"].mean() if len(stopped) else math.nan,
            "sampling_s_mean": stopped["sampling_s_at_T"].mean() if len(stopped) else math.nan,
            "rho": float(group["rho"].iloc[0]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def paired_speedup(frame, keys, numerator, denominator):
    """T(numerator) / T(denominator) per run, both stopped, same stream."""
    a = frame[frame["monitor"] == numerator].set_index(keys + ["run"])
    b = frame[frame["monitor"] == denominator].set_index(keys + ["run"])
    joined = a[["T", "outcome"]].join(b[["T", "outcome"]], lsuffix="_num", rsuffix="_den")
    both = joined[(joined["outcome_num"] == "stopped") & (joined["outcome_den"] == "stopped")]
    rows = []
    for key, group in both.groupby(level=list(range(len(keys)))):
        ratio = group["T_num"] / group["T_den"]
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row.update({"pairs": len(group),
                    "ratio_of_means": group["T_num"].mean() / group["T_den"].mean(),
                    "median_ratio": float(ratio.median()),
                    "q10_ratio": float(ratio.quantile(0.10)),
                    "q90_ratio": float(ratio.quantile(0.90))})
        rows.append(row)
    return pd.DataFrame(rows)


def triangle_summary():
    frame = load("triangles")
    if frame.empty:
        return frame, frame
    return frame, cell_summary(frame, ["graph", "eps"])


def clique_summary():
    frame = load("cliques")
    if frame.empty:
        return frame, frame
    return frame, cell_summary(frame, ["graph", "k", "eps"])

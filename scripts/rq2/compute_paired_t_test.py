"""Gate 1 paired t-test (Holm-Bonferroni) — paper §4.2 statistical protocol.

Computes:
  - Test (i)  : (a) vs (b) on 6 BN-augmented encoders × 2 datasets,
                Holm-Bonferroni correction within dataset (paper convention).
  - Test (ii) : (b) vs (d) on GBT only (single test, no correction).
  - Test (ii-extended): (b) vs (d) on all 10 encoders × 2 datasets,
                Holm-Bonferroni within dataset (Gate 1 new analysis).

Inputs:
  results/rq1/main_sweep.csv
  results/rq4/amlworld_main_sweep.csv

Output:
  results/rq2/paired_t_test.csv  (long-format with raw + Holm p-values)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats

ENCODER_BN = {
    "gbt":      ("per-layer", True),
    "dgi_bn":   ("per-layer", True),
    "mvgrl_bn": ("per-layer", True),
    "grace_bn": ("per-layer", True),
    "gin":      ("per-layer", True),
    "bgrl":     ("final-only", True),
    "dgi":      ("none", False),
    "mvgrl":    ("none", False),
    "grace":    ("none", False),
    "gca":      ("none", False),
}
BN_ENCODERS = [e for e, (_, b) in ENCODER_BN.items() if b]


def parse_model_name(name):
    parts = name.split("_")
    seed = int(parts[-1].lstrip("s"))
    setting = parts[-2]
    enc = "_".join(parts[1:-2])
    return enc, setting, seed


def get_paired_arrays(df, enc, s1, s2, metric):
    sub = df[df["encoder"] == enc]
    pairs = []
    for seed in sorted(sub["seed"].unique()):
        v1 = sub[(sub["setting"] == s1) & (sub["seed"] == seed)][metric].values
        v2 = sub[(sub["setting"] == s2) & (sub["seed"] == seed)][metric].values
        if len(v1) == 1 and len(v2) == 1:
            pairs.append((float(v1[0]), float(v2[0])))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return a, b


def holm_bonferroni(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running_max = 0.0
    for rank, idx in enumerate(order):
        a = p[idx] * (n - rank)
        running_max = max(running_max, min(a, 1.0))
        adj[idx] = running_max
    return adj


def load(path):
    df = pd.read_csv(path)
    parsed = df["Model"].apply(parse_model_name)
    df["encoder"] = [r[0] for r in parsed]
    df["setting"] = [r[1] for r in parsed]
    df["seed"] = [r[2] for r in parsed]
    return df


def run_tests(df, dataset, metric="f1_1"):
    rows = []
    # Test (i): (a) vs (b), 6 BN encoders, Holm-Bonferroni
    raw_i = []
    cache_i = []
    for enc in BN_ENCODERS:
        a, b = get_paired_arrays(df, enc, "a", "b", metric)
        t, p = stats.ttest_rel(b, a)
        cache_i.append((enc, a, b, t, p))
        raw_i.append(p)
    adj_i = holm_bonferroni(raw_i)
    for (enc, a, b, t, p), pa in zip(cache_i, adj_i):
        rows.append({
            "dataset": dataset, "test": "(i) a_vs_b",
            "encoder": enc, "bn_type": ENCODER_BN[enc][0],
            "n_seeds": len(a),
            "mean_lhs": a.mean(), "mean_rhs": b.mean(),
            "delta": (b - a).mean(), "delta_std": (b - a).std(),
            "t_stat": t, "p_raw": p, "p_holm": pa,
        })
    # Test (ii): (b) vs (d), GBT only, no correction
    a, b = get_paired_arrays(df, "gbt", "b", "d", metric)
    t, p = stats.ttest_rel(b, a)
    rows.append({
        "dataset": dataset, "test": "(ii) b_vs_d_gbt",
        "encoder": "gbt", "bn_type": "per-layer",
        "n_seeds": len(a),
        "mean_lhs": a.mean(), "mean_rhs": b.mean(),
        "delta": (b - a).mean(), "delta_std": (b - a).std(),
        "t_stat": t, "p_raw": p, "p_holm": float("nan"),
    })
    # Test (ii-extended): (b) vs (d), all 10 encoders, Holm-Bonferroni
    raw_ii = []
    cache_ii = []
    for enc in ENCODER_BN.keys():
        a, b = get_paired_arrays(df, enc, "b", "d", metric)
        t, p = stats.ttest_rel(b, a)
        cache_ii.append((enc, a, b, t, p))
        raw_ii.append(p)
    adj_ii = holm_bonferroni(raw_ii)
    for (enc, a, b, t, p), pa in zip(cache_ii, adj_ii):
        rows.append({
            "dataset": dataset, "test": "(ii-ext) b_vs_d_all10",
            "encoder": enc, "bn_type": ENCODER_BN[enc][0],
            "n_seeds": len(a),
            "mean_lhs": a.mean(), "mean_rhs": b.mean(),
            "delta": (b - a).mean(), "delta_std": (b - a).std(),
            "t_stat": t, "p_raw": p, "p_holm": pa,
        })
    return rows


if __name__ == "__main__":
    rows = []
    for ds, path in [("HOFINET", "results/rq1/main_sweep.csv"),
                     ("AMLworld", "results/rq4/amlworld_main_sweep.csv")]:
        df = load(path)
        rows.extend(run_tests(df, ds))
    out = pd.DataFrame(rows)
    out.to_csv("results/rq2/paired_t_test.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: results/rq2/paired_t_test.csv ({len(out)} rows)")

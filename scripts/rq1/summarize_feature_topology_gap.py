"""Summarize the feature-separability vs topology-repair diagnostic.

This script is intentionally lightweight: it does not train a new GNN. It
combines a feature-only linear probe on the 20 behavioral variables with the
existing GBT rows from the RQ1 topology-by-level sweep.

Output:
  results/rq1/feature_topology_gap.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from utils import make_split  # noqa: E402

NODE_FEAT = BASE / "datasets" / "HOFINET_NODE_FEAT.csv"
MAIN_SWEEP = BASE / "results" / "rq1" / "main_sweep.csv"
OUT = BASE / "results" / "rq1" / "feature_topology_gap.csv"
SEEDS = (2024, 2025, 2026, 2027)

BEHAVIOR_COLS = [
    "out_mean",
    "out_max",
    "out_std",
    "in_mean",
    "in_max",
    "in_std",
    "out_3m_mean",
    "out_3m_count",
    "in_3m_mean",
    "in_3m_count",
    "out_6m_mean",
    "out_6m_count",
    "in_6m_mean",
    "in_6m_count",
    "out_12m_mean",
    "out_12m_count",
    "in_12m_mean",
    "in_12m_count",
    "md_type_entropy",
    "fnd_type_entropy",
]


def mean_std(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def metric_summary(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for col in ("f1_1", "pre_1", "rec_1", "auroc", "auprc"):
        mean, std = mean_std(df[col])
        out[f"{col}_mean"] = mean
        out[f"{col}_std"] = std
    return out


def feature_only_probe() -> dict:
    df = pd.read_csv(NODE_FEAT)
    x = df[BEHAVIOR_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    rows = []
    for seed in SEEDS:
        split = make_split(len(y), train_ratio=0.1, val_ratio=0.1, seed=seed)
        train_idx = split["train"].numpy()
        test_idx = split["test"].numpy()

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        )
        clf.fit(x[train_idx], y[train_idx])
        y_score = clf.predict_proba(x[test_idx])[:, 1]
        y_pred = (y_score >= 0.5).astype(np.int64)
        pre, rec, f1, _ = precision_recall_fscore_support(
            y[test_idx], y_pred, labels=[1], zero_division=0
        )
        rows.append(
            {
                "f1_1": f1[0],
                "pre_1": pre[0],
                "rec_1": rec[0],
                "auroc": roc_auc_score(y[test_idx], y_score),
                "auprc": average_precision_score(y[test_idx], y_score),
            }
        )

    return {
        "condition": "Feature-only LR",
        "graph": "none",
        "topology_repaired": "no",
        "diagnostic": "behavioral feature separability without graph repair",
        **metric_summary(rows),
    }


def gbt_row(setting: str, label: str, graph: str, repaired: bool, diagnostic: str) -> dict:
    df = pd.read_csv(MAIN_SWEEP)
    mask = df["Model"].str.contains(fr"hof_gbt_{setting}_s", regex=True)
    rows = df.loc[mask].copy()
    if rows.empty:
        raise RuntimeError(f"No GBT rows found for setting {setting} in {MAIN_SWEEP}")
    summary = {}
    for col in ("f1_1", "pre_1", "rec_1", "auroc", "auprc"):
        mean, std = mean_std(rows[col])
        summary[f"{col}_mean"] = mean
        summary[f"{col}_std"] = std
    return {
        "condition": label,
        "graph": graph,
        "topology_repaired": "yes" if repaired else "no",
        "diagnostic": diagnostic,
        **summary,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        feature_only_probe(),
        gbt_row(
            "a",
            "GBT Tx graph node",
            "transaction",
            False,
            "same self-supervised protocol, original transaction topology",
        ),
        gbt_row(
            "b",
            "GBT recovered node",
            "behavioral kNN",
            True,
            "replace topology with recovered behavioral-neighborhood graph",
        ),
        gbt_row(
            "c",
            "GBT Tx graph pool",
            "transaction",
            False,
            "subgraph pooling on unrepaired transaction topology",
        ),
        gbt_row(
            "d",
            "GBT repaired pool",
            "behavioral kNN",
            True,
            "subgraph pooling on repaired topology",
        ),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()

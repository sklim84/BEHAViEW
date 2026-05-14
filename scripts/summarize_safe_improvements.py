#!/usr/bin/env python3
"""Summarize safe-improvement sweeps by encoder and setting."""
import re
import sys

import pandas as pd


def parse_model(model):
    match = re.match(r"aml_(?P<encoder>.+?)_(?P<setting>b_tuned|d_hap_tuned|d_tuned|d_hap|b|d)_s(?P<seed>\d+)$", model)
    if not match:
        return None
    return match.groupdict()


def main(path):
    df = pd.read_csv(path)
    parsed = df["Model"].map(parse_model)
    keep = parsed.notna()
    if not keep.any():
        raise SystemExit(f"No safe-improvement rows found in {path}")

    meta = pd.DataFrame(parsed[keep].tolist(), index=df.index[keep])
    df = pd.concat([df.loc[keep].copy(), meta], axis=1)
    df["f1_1"] = pd.to_numeric(df["f1_1"], errors="coerce")
    df["threshold"] = pd.to_numeric(df.get("threshold", 0.5), errors="coerce")

    summary = (
        df.groupby(["encoder", "setting"])
        .agg(
            n=("f1_1", "count"),
            f1_mean=("f1_1", "mean"),
            f1_std=("f1_1", "std"),
            threshold_mean=("threshold", "mean"),
        )
        .reset_index()
        .sort_values(["encoder", "f1_mean"], ascending=[True, False])
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = summary.loc[summary.groupby("encoder")["f1_mean"].idxmax()]
    print("\nBest per encoder")
    print(best[["encoder", "setting", "f1_mean", "f1_std", "n"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: scripts/summarize_safe_improvements.py <results.csv>")
    main(sys.argv[1])

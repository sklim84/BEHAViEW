#!/bin/bash
# Merge per-tag temp CSVs into one CSV per dataset.
# HOFINET (RQ1) -> results/rq1/main_sweep.csv
# AMLworld/AMLNet (RQ4) -> results/rq4/${ds}_main_sweep.csv
set -e

python - <<'EOF'
import pandas as pd, glob, os

JOBS = [
    ("hofinet",  "results/rq1",        "main_sweep.csv"),
    ("amlworld", "results/rq4",        "amlworld_main_sweep.csv"),
    ("amlnet",   "results/rq4",        "amlnet_main_sweep.csv"),
]
for ds, out_dir, final_name in JOBS:
    tmps = sorted(glob.glob(f"{out_dir}/.tmp_{ds}_main_sweep_*.csv"))
    if not tmps:
        continue
    dfs = [pd.read_csv(p) for p in tmps]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates(subset='Model', keep='last').reset_index(drop=True)
    final = f"{out_dir}/{final_name}"
    df.to_csv(final, index=False)
    print(f"merged {len(tmps)} tmp files -> {final} ({len(df)} unique rows)")
EOF

#!/bin/bash
# Merge per-tag temp CSVs into one CSV per dataset under results/main_table/.
# Run after all run_main_table.sh dispatches finish.
set -e
OUTPUT_DIR="${OUTPUT_DIR:-results/main_table}"

python - <<EOF
import pandas as pd, glob, os
OUT = "${OUTPUT_DIR}"
for ds in ("hofinet", "amlworld", "amlnet"):
    tmps = sorted(glob.glob(f"{OUT}/.tmp_{ds}_*.csv"))
    if not tmps:
        continue
    dfs = [pd.read_csv(p) for p in tmps]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates(subset='Model', keep='last').reset_index(drop=True)
    final = f"{OUT}/{ds}.csv"
    df.to_csv(final, index=False)
    print(f"merged {len(tmps)} tmp files -> {final} ({len(df)} unique rows)")
EOF

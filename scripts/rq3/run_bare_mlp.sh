#!/bin/bash
# =============================================================
# Bare MLP sweep (standard MLP convention: Linear + ReLU only,
# no BatchNorm, no Dropout). Replaces modern MLP rows.
#
# Env vars: GPU, DATASET (hofinet/amlworld/amlnet)
# Output: appended to results/rq3/supervised_${DATASET}.csv
# =============================================================
set -e

GPU="${GPU:-1}"
DATASET="${DATASET:-hofinet}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
RATIOS="${RATIOS:-0.01 0.05 0.10}"

FINAL="results/rq3/supervised_${DATASET}.csv"
TMP_DIR="results/bare_mlp_tmp/${DATASET}"
mkdir -p "$TMP_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=64

echo "[$(date)] === Bare MLP sweep: $DATASET on GPU $GPU ==="

for RATIO in $RATIOS; do
    OUT="$TMP_DIR/mlp_r${RATIO}.csv"
    echo "[$(date)] === train_ratio=$RATIO -> $OUT ==="
    python3 -u models/supervised_baselines.py \
        --gpu "$GPU" --dataset "$DATASET" \
        --seeds $SEEDS \
        --models mlp \
        --exclude_struct \
        --train_ratio "$RATIO" \
        --result_file "$OUT" 2>&1 | grep -vE '^/home/work/.local|^\s*$' || true
done

# Merge into final CSV (append to existing rows, add train_ratio column)
python3 -c "
import pandas as pd, glob, os

# Load all per-ratio MLP CSVs
new_rows = []
for f in sorted(glob.glob('$TMP_DIR/mlp_r*.csv')):
    df = pd.read_csv(f)
    ratio = float(os.path.basename(f).split('_r')[1].rsplit('.csv', 1)[0])
    df['train_ratio'] = ratio
    new_rows.append(df)
new_df = pd.concat(new_rows, ignore_index=True)

# Append to existing final
existing = pd.read_csv('$FINAL')
combined = pd.concat([existing, new_df], ignore_index=True)
combined.to_csv('$FINAL', index=False)
print(f'Appended {len(new_df)} bare-MLP rows to $FINAL (total: {len(combined)})')
"

echo "[$(date)] === Done: bare MLP $DATASET ==="

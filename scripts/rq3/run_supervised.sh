#!/bin/bash
# =============================================================
# Supervised label-efficiency sweep for one dataset.
# Env vars: GPU, DATASET (atnet/amlworld/amlnet)
#
# 6 baselines × 3 fractions × 4 seeds = 72 runs/dataset
# Boosting models (xgb, lgbm) use --exclude_struct for fair
# comparison with BehaView (behavioral features only).
# =============================================================
set -e

GPU="${GPU:-1}"
DATASET="${DATASET:-atnet}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
MODELS="${MODELS:-mlp xgb lgbm gcn gat caregnn}"
RATIOS="${RATIOS:-0.01 0.05 0.10}"

OUT_DIR="results/rq3/.tmp_supervised/${DATASET}"
FINAL="results/rq3/supervised_${DATASET}.csv"

mkdir -p "$OUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=64

echo "[$(date)] === Supervised label-efficiency sweep ==="
echo "  GPU=$GPU  Dataset=$DATASET  Models=$MODELS  Ratios=$RATIOS"

for RATIO in $RATIOS; do
    OUT="$OUT_DIR/labeff_r${RATIO}.csv"
    echo ""
    echo "[$(date)] === Dataset=$DATASET  train_ratio=$RATIO -> $OUT ==="
    python3 -u models/supervised_baselines.py \
        --gpu "$GPU" --dataset "$DATASET" \
        --seeds $SEEDS \
        --models $MODELS \
        --exclude_struct \
        --train_ratio "$RATIO" \
        --result_file "$OUT" 2>&1 | grep -vE '^/home/work/.local|^\s*$' || true
done

echo ""
echo "[$(date)] === Merging $DATASET results into $FINAL ==="
python3 -c "
import pandas as pd, glob, os
dfs = []
for f in sorted(glob.glob('$OUT_DIR/labeff_r*.csv')):
    df = pd.read_csv(f)
    ratio = os.path.basename(f).split('_r')[1].rsplit('.csv', 1)[0]
    df['train_ratio'] = float(ratio)
    dfs.append(df)
out = pd.concat(dfs, ignore_index=True)
out.to_csv('$FINAL', index=False)
print(f'Saved {len(out)} rows -> $FINAL')
print()
agg = out.groupby(['model','train_ratio']).agg(
    f1_1_mean=('f1_1','mean'), f1_1_std=('f1_1','std')
).round(4)
print(agg.to_string())
"

echo "[$(date)] === Done: $DATASET ==="

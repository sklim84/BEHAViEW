#!/bin/bash
# =============================================================
# Label-efficiency sweep: BehaView vs supervised baselines under
# varying train_ratio (1% / 5% / 10%) on HOFINET (\hofinet).
#
# Supervised: MLP, XGBoost, GCN, CARE-GNN
# BehaView label-fraction results already exist in
#   results/rq3/behaview_hofinet.csv (GBT + setting d, 12 runs)
#
# Output: results/exp_results_label_efficiency.csv
#         (48 rows = 4 models × 3 fractions × 4 seeds)
# =============================================================
set -e

GPU="${GPU:-1}"
DATASET="${DATASET:-hofinet}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
MODELS="${MODELS:-mlp xgb gcn caregnn}"
RATIOS="${RATIOS:-0.01 0.05 0.10}"

OUT_DIR="results/labeff_tmp"
FINAL="results/exp_results_label_efficiency.csv"

mkdir -p "$OUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=64

echo "[$(date)] === Label-efficiency sweep ==="
echo "  GPU=$GPU  Dataset=$DATASET"
echo "  Models=$MODELS  Ratios=$RATIOS  Seeds=$SEEDS"

for RATIO in $RATIOS; do
    OUT="$OUT_DIR/labeff_r${RATIO}.csv"
    echo ""
    echo "[$(date)] === train_ratio=$RATIO -> $OUT ==="
    python3 -u models/supervised_baselines.py \
        --gpu "$GPU" --dataset "$DATASET" \
        --seeds $SEEDS \
        --models $MODELS \
        --train_ratio "$RATIO" \
        --result_file "$OUT" 2>&1 | grep -vE '^/home/work/.local|^\s*$' || true
done

echo ""
echo "[$(date)] === Merging per-ratio CSVs into $FINAL ==="
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
print('Summary (mean over seeds):')
agg = out.groupby(['model','train_ratio']).agg(
    f1_1_mean=('f1_1','mean'), f1_1_std=('f1_1','std'),
    auroc_mean=('auroc','mean'), auprc_mean=('auprc','mean'),
).round(4)
print(agg.to_string())
"

echo ""
echo "[$(date)] === Sweep complete ==="

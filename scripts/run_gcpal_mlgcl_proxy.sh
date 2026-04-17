#!/bin/bash
# GCPAL/MLGCL proxy comparison experiment
# HYBRID k-NN (all 31 feats) ≈ MLGCL/GCPAL approach
# CEN k-NN (7 centrality feats) = centrality-only baseline
set -euo pipefail

GPU=1
BASE="--encoder_type gbt --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --skip_tsne"
RESULT="./results/exp_results_feature_ablation.csv"

echo "[$(date)] === GCPAL/MLGCL Proxy Experiments ==="

for KNN_TYPE in HYBRID CEN; do
  KNN_FILE="HOFINET_KNN_${KNN_TYPE}_k10"
  if [ ! -f "datasets/${KNN_FILE}.csv" ]; then
    echo "  ERROR: $KNN_FILE not found!"
    exit 1
  fi
  for SEED in 2024 2025 2026 2027; do
    NAME="feat_gbt_d_${KNN_TYPE}_s${SEED}"
    if grep -q "$NAME" "$RESULT" 2>/dev/null; then
      echo "  skip $NAME"
      continue
    fi
    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py --model_name "$NAME" --gpu $GPU --seed $SEED \
      $BASE --knn_graph "$KNN_FILE" --subgraph_pool \
      --metric_save_path "$RESULT" 2>&1 | tail -2
  done
done

echo "[$(date)] === Done: $(wc -l < $RESULT) rows ==="

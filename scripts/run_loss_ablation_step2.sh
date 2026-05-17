#!/bin/bash
# =============================================================
# Loss ablation Step 2: 7 baseline encoders, each paired with its
#                       paper-native contrastive loss, on the
#                       BEHAViEW (d) framework.
#
# Purpose: Check whether baseline GCL encoders close the gap with
# BEHAViEW (GBT) when allowed to use their native loss, instead of
# the uniform BootstrapLatent loss used in the main tables.
#
# Encoder -> native loss mapping
#   bgrl      -> BootstrapLatent
#   dgi       -> JSD
#   dgi_bn    -> JSD
#   mvgrl     -> JSD
#   mvgrl_bn  -> JSD
#   grace     -> InfoNCE
#   grace_bn  -> InfoNCE
#
#   Seeds:  2024 2025 2026 2027
#   Setting: (d) bhv + pool
#   Total:  7 * 1 * 4 = 28 runs
#
# Output: results/exp_results_loss_ablation_step2.csv
#
# Usage:
#   nice -n 19 GPU=7 bash scripts/run_loss_ablation_step2.sh
# =============================================================
set -e

GPU="${GPU:-7}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"

HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

NODE="HOFINET_NODE_FEAT"
EDGE="HOFINET_EDGES"
KNN="HOFINET_KNN_BEHAV_k10"
RESULT="./results/exp_results_loss_ablation_step2.csv"

# Encoder -> native loss pairs (space-separated "enc:loss")
PAIRS="${PAIRS:-bgrl:BootstrapLatent dgi:JSD dgi_bn:JSD mvgrl:JSD mvgrl_bn:JSD grace:InfoNCE grace_bn:InfoNCE}"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Loss ablation Step 2 ==="
echo "  GPU=$GPU"
echo "  PAIRS=$PAIRS"
echo "  SEEDS=$SEEDS"
echo "  OUT=$RESULT"

for PAIR in $PAIRS; do
    ENC="${PAIR%:*}"
    LOSS="${PAIR#*:}"
    for SEED in $SEEDS; do
        NAME="loss_step2_${ENC}_${LOSS}_s${SEED}"
        echo "[$(date)] $NAME"
        python -u models/subgraph_cl.py \
            --model_name "$NAME" \
            --gpu "$GPU" --seed "$SEED" \
            --encoder_type "$ENC" \
            --node_data_name "$NODE" --edge_data_name "$EDGE" \
            --knn_graph "$KNN" --subgraph_pool \
            --skip_tsne \
            --metric_save_path "$RESULT" \
            $HP --loss "$LOSS" 2>&1 | grep -E "^\(E\)" || true
    done
done

echo "[$(date)] === Step 2 complete ==="

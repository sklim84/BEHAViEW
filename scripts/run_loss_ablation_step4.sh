#!/bin/bash
# =============================================================
# Loss ablation Step 4: 7 baselines under BYOL BootstrapLatent
#                       on setting (d) (BYOL column re-run for
#                       consistency with Step 2's native column).
#
# Purpose: Provide same-environment BYOL-bootstrap measurements
# for all 7 baseline encoders, so the new Table 9's "BYOL" and
# "Native" columns come from runs under identical code, k-NN
# graph, and seed pool, removing the run-to-run drift visible in
# the BGRL row (0.639 in Table 3 vs 0.627 in Step 2).
#
# Setup:
#   Encoders: bgrl dgi dgi_bn mvgrl mvgrl_bn grace grace_bn
#   Setting:  (d) bhv + pool
#   Loss:     BootstrapLatent
#   Seeds:    2024 2025 2026 2027
#   Total:    7 * 4 = 28 runs (~2h on GPU 1)
#
# Output: results/exp_results_loss_ablation_step4.csv
# =============================================================
set -e

GPU="${GPU:-1}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
ENCODERS="${ENCODERS:-bgrl dgi dgi_bn mvgrl mvgrl_bn grace grace_bn}"

HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

NODE="HOFINET_NODE_FEAT"
EDGE="HOFINET_EDGES"
KNN="HOFINET_KNN_BEHAV_k10"
RESULT="./results/exp_results_loss_ablation_step4.csv"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Loss ablation Step 4 (BYOL consistency) ==="
echo "  GPU=$GPU"
echo "  ENCODERS=$ENCODERS"
echo "  SEEDS=$SEEDS"
echo "  OUT=$RESULT"

for ENC in $ENCODERS; do
    for SEED in $SEEDS; do
        NAME="loss_step4_${ENC}_BootstrapLatent_s${SEED}"
        echo "[$(date)] $NAME"
        python -u models/subgraph_cl.py \
            --model_name "$NAME" \
            --gpu "$GPU" --seed "$SEED" \
            --encoder_type "$ENC" \
            --node_data_name "$NODE" --edge_data_name "$EDGE" \
            --knn_graph "$KNN" --subgraph_pool \
            --skip_tsne \
            --metric_save_path "$RESULT" \
            $HP --loss BootstrapLatent 2>&1 | grep -E "^\((E|L)\)" || true
    done
done

echo "[$(date)] === Step 4 complete ==="

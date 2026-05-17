#!/bin/bash
# Loss ablation Step 6: GIN x Bootstrap x (d) for Table 9
# completeness (Step 1 covers GBT; Step 4 covers 7 baselines;
# GIN is the remaining encoder).
set -e
GPU="${GPU:-1}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
NODE="HOFINET_NODE_FEAT"; EDGE="HOFINET_EDGES"; KNN="HOFINET_KNN_BEHAV_k10"
RESULT="./results/exp_results_loss_ablation_step6.csv"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Step 6: GIN x Bootstrap x (d) ==="
for SEED in $SEEDS; do
    NAME="loss_step6_gin_BootstrapLatent_s${SEED}"
    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py \
        --model_name "$NAME" --gpu "$GPU" --seed "$SEED" \
        --encoder_type gin \
        --node_data_name "$NODE" --edge_data_name "$EDGE" \
        --knn_graph "$KNN" --subgraph_pool \
        --skip_tsne --metric_save_path "$RESULT" \
        $HP --loss BootstrapLatent 2>&1 | grep -E "^\((E|L)\)" || true
done
echo "[$(date)] === Step 6 complete ==="

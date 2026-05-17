#!/bin/bash
# Loss ablation Step 7: cross-dataset native-loss verification.
# 3 BN-equipped baselines x native loss x (d) x AMLworld+AMLNet.
# Confirms that the loss substitution does not penalize baselines
# beyond HOFINET.
set -e
GPU="${GPU:-1}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
PAIRS="${PAIRS:-dgi_bn:JSD mvgrl_bn:JSD grace_bn:InfoNCE}"
DATASETS="${DATASETS:-aml amlnet}"
TAG="${TAG:-}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
if [ -n "$TAG" ]; then
    RESULT="./results/exp_results_loss_ablation_step7_${TAG}.csv"
else
    RESULT="./results/exp_results_loss_ablation_step7.csv"
fi
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

run_ds() {
    local DS_TAG="$1" NODE="$2" EDGE="$3" KNN="$4"
    for PAIR in $PAIRS; do
        ENC="${PAIR%:*}"; LOSS="${PAIR#*:}"
        for SEED in $SEEDS; do
            NAME="loss_step7_${DS_TAG}_${ENC}_${LOSS}_s${SEED}"
            echo "[$(date)] $NAME"
            python -u models/subgraph_cl.py \
                --model_name "$NAME" --gpu "$GPU" --seed "$SEED" \
                --encoder_type "$ENC" \
                --node_data_name "$NODE" --edge_data_name "$EDGE" \
                --knn_graph "$KNN" --subgraph_pool \
                --skip_tsne --metric_save_path "$RESULT" \
                $HP --loss "$LOSS" 2>&1 | grep -E "^\((E|L)\)" || true
        done
    done
}

echo "[$(date)] === Step 7: TAG=$TAG, DATASETS=$DATASETS, PAIRS=$PAIRS, GPU=$GPU, OUT=$RESULT ==="
for DS in $DATASETS; do
    case "$DS" in
        aml)    run_ds "aml"    "amlworld/AMLWORLD_NODE_FEAT" "amlworld/AMLWORLD_EDGES" "amlworld/AMLWORLD_KNN_BEHAV_k10" ;;
        amlnet) run_ds "amlnet" "amlnet/AMLNET_NODE_FEAT"     "amlnet/AMLNET_EDGES"     "amlnet/AMLNET_KNN_BEHAV_k10" ;;
        *) echo "Unknown dataset: $DS"; exit 1 ;;
    esac
done
echo "[$(date)] === Step 7 complete (TAG=$TAG) ==="

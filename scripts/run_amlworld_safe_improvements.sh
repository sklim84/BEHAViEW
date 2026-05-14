#!/bin/bash
# =============================================================
# AMLworld safe-improvement sweep
#
# Narrow tests that keep the BehaView design intact:
#   - behavioral k-NN view remains the auxiliary view
#   - subgraph pooling remains the proposed contrastive level
#   - HAP is tested as a theoretically motivated pool variant
#   - validation-threshold tuning changes only the downstream LR decision rule
#
# Output: results/exp_results_amlworld_safe_improvements.csv
# =============================================================
set -e

GPU="${GPU:-0}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
ENCODERS="${ENCODERS:-gbt dgi_bn grace_bn bgrl}"
SETTINGS="${SETTINGS:-b b_tuned d d_tuned d_hap d_hap_tuned}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-auto}"

NODE="${NODE:-amlworld/AMLWORLD_NODE_FEAT}"
EDGE="${EDGE:-amlworld/AMLWORLD_EDGES}"
KNN="${KNN:-amlworld/AMLWORLD_KNN_BEHAV_k10}"
RESULT="${RESULT:-./results/exp_results_amlworld_safe_improvements.csv}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path"
        echo "Run preprocessing/k-NN construction first, or point NODE/EDGE/KNN to mounted data."
        exit 1
    fi
}

require_file "datasets/${NODE}.csv"
require_file "datasets/${EDGE}.csv"
require_file "datasets/${KNN}.csv"

run_one() {
    local ENC="$1"
    local SETTING="$2"
    local SEED="$3"

    local FLAGS="--knn_graph $KNN"
    case "$SETTING" in
        b) ;;
        b_tuned) FLAGS="$FLAGS --tune_threshold" ;;
        d) FLAGS="$FLAGS --subgraph_pool" ;;
        d_tuned) FLAGS="$FLAGS --subgraph_pool --tune_threshold" ;;
        d_hap) FLAGS="$FLAGS --subgraph_pool --pool_variant heterophily" ;;
        d_hap_tuned) FLAGS="$FLAGS --subgraph_pool --pool_variant heterophily --tune_threshold" ;;
        *) echo "unknown setting: $SETTING"; exit 1 ;;
    esac

    local NAME="aml_${ENC}_${SETTING}_s${SEED}"
    echo "[$(date)] $NAME"
    python3 -u models/subgraph_cl.py \
        --model_name "$NAME" \
        --gpu "$GPU" --seed "$SEED" \
        --device "$DEVICE" --epochs "$EPOCHS" \
        --encoder_type "$ENC" \
        --node_data_name "$NODE" --edge_data_name "$EDGE" \
        --skip_tsne \
        --metric_save_path "$RESULT" \
        $HP --loss BarlowTwins \
        $FLAGS 2>&1 | grep -E "^\(E\)|Tuned threshold" || true
}

echo "[$(date)] === AMLworld safe-improvement sweep ==="
echo "  GPU=$GPU"
echo "  ENCODERS=$ENCODERS"
echo "  SETTINGS=$SETTINGS"
echo "  RESULT=$RESULT"

for ENC in $ENCODERS; do
    for SETTING in $SETTINGS; do
        for SEED in $SEEDS; do
            run_one "$ENC" "$SETTING" "$SEED"
        done
    done
done

echo "[$(date)] === Sweep complete ==="
python3 scripts/summarize_safe_improvements.py "$RESULT"

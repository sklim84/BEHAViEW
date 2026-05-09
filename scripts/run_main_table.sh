#!/bin/bash
# =============================================================
# Gate 1 — Main table sweep on corrected 10/10/80 split
#
# 10 encoders × 4 settings (a/b/c/d) × 4 seeds × 2 datasets = 320 runs
# Replaces all paper Tables 1-3 cells which were produced under the
# PyGCL get_split valid/test inversion; uses utils.make_split for the
# corrected protocol.
#
# Output:
#   - HOFINET  → results/exp_results_hofinet_ab.csv  (ab = "ablation"; full 4-setting)
#   - AMLworld → results/exp_results_amlworld.csv
#
# Parallel launch on two GPUs (default 6, 7):
#   GPU=6 DATASETS=hofinet  bash scripts/run_main_table.sh &
#   GPU=7 DATASETS=amlworld bash scripts/run_main_table.sh &
#   wait
#
# Sequential (single GPU):
#   GPU=6 bash scripts/run_main_table.sh
# =============================================================
set -e

# --- Configuration (env-overridable) ---
GPU="${GPU:-0}"
DATASETS="${DATASETS:-hofinet amlworld}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
ENCODERS="${ENCODERS:-gbt bgrl dgi mvgrl grace gca dgi_bn mvgrl_bn grace_bn gin}"
SETTINGS="${SETTINGS:-a b c d}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === Gate 1 main table sweep ==="
echo "  GPU=$GPU"
echo "  DATASETS=$DATASETS"
echo "  SEEDS=$SEEDS"
echo "  ENCODERS=$ENCODERS"
echo "  SETTINGS=$SETTINGS"

run_one() {
    local DS_TAG="$1"      # hof | aml
    local NODE="$2"        # node_data_name
    local EDGE="$3"        # edge_data_name
    local KNN="$4"         # knn_graph name
    local RESULT="$5"      # output CSV
    local ENC="$6"
    local SETTING="$7"
    local SEED="$8"

    local FLAGS=""
    case "$SETTING" in
        a) FLAGS="" ;;
        b) FLAGS="--knn_graph $KNN" ;;
        c) FLAGS="--subgraph_pool" ;;
        d) FLAGS="--knn_graph $KNN --subgraph_pool" ;;
        *) echo "unknown setting: $SETTING"; exit 1 ;;
    esac

    local NAME="${DS_TAG}_${ENC}_${SETTING}_s${SEED}"

    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py \
        --model_name "$NAME" \
        --gpu "$GPU" --seed "$SEED" \
        --encoder_type "$ENC" \
        --node_data_name "$NODE" --edge_data_name "$EDGE" \
        --skip_tsne \
        --metric_save_path "$RESULT" \
        $HP --loss BarlowTwins \
        $FLAGS 2>&1 | grep -E "^\(E\)" || true
}

for DS in $DATASETS; do
    case "$DS" in
        hofinet)
            DS_TAG="hof"
            NODE="HOFINET_NODE_FEAT"
            EDGE="HOFINET_EDGES"
            KNN="HOFINET_KNN_BEHAV_k10"
            RESULT="./results/exp_results_hofinet_ab.csv"
            ;;
        amlworld)
            DS_TAG="aml"
            NODE="amlworld/AMLWORLD_NODE_FEAT"
            EDGE="amlworld/AMLWORLD_EDGES"
            KNN="amlworld/AMLWORLD_KNN_BEHAV_k10"
            RESULT="./results/exp_results_amlworld.csv"
            ;;
        *)
            echo "unknown dataset: $DS"; exit 1
            ;;
    esac

    echo ""
    echo "[$(date)] === Dataset: $DS → $RESULT ==="

    for ENC in $ENCODERS; do
        echo ""
        echo "----- $DS_TAG / $ENC -----"
        for SETTING in $SETTINGS; do
            for SEED in $SEEDS; do
                run_one "$DS_TAG" "$NODE" "$EDGE" "$KNN" "$RESULT" \
                        "$ENC" "$SETTING" "$SEED"
            done
        done
    done
done

echo ""
echo "[$(date)] === Sweep complete ==="

#!/bin/bash
# =============================================================
# Main table sweep — re-run for Tables 3 (tab:rq1) and 6 (tab:rq4)
# under the current code revision.
#
# 9 encoders x 4 settings (a/b/c/d) x 4 seeds x DATASET
# = 144 runs per dataset on HOFINET
# = 64 runs per dataset on AMLworld / AMLNet (4 encoders only)
#
# Encoder set:
#   HOFINET: gbt bgrl dgi mvgrl grace dgi_bn mvgrl_bn grace_bn gin
#   Cross-dataset (Table 6): bgrl gbt dgi_bn grace_bn
#
# Loss: BootstrapLatent (matches the unified BYOL bootstrap used
#       by Table 9). Earlier scripts used --loss BarlowTwins, which
#       was silently ignored in the previous code but is now a real
#       implementation, so this flag is set explicitly.
#
# Output: results/rq1/${DATASET}.csv
#         Each parallel dispatch writes to a per-tag temp file
#         (results/rq1/.tmp_${DATASET}_${TAG}.csv) to avoid
#         CSV append races; merge with merge_main_table.sh after
#         all dispatches finish.
#
# Process management:
#   - setsid creates a new process group so all children share PGID
#   - trap on EXIT/INT/TERM kills the whole group, avoiding orphan
#     python processes that previously held GPU memory
#
# Env vars (all overridable):
#   GPU         single GPU index (e.g., 0)
#   DATASETS    space-separated subset of {hofinet amlworld amlnet}
#   ENCODERS    space-separated encoder subset
#   SETTINGS    space-separated subset of {a b c d}
#   SEEDS       space-separated seed list
#   TAG         suffix for the temp CSV (default: gpu${GPU})
#   OUTPUT_DIR_HOFINET / OUTPUT_DIR_AMLWORLD / OUTPUT_DIR_AMLNET
#               per-dataset output dirs (defaults: results/rq1,
#               results/rq4, results/rq4)
#
# Example (4-GPU parallel):
#   GPU=0 ENCODERS="gbt bgrl dgi"        DATASETS=hofinet  TAG=g0 bash scripts/run_main_table.sh &
#   GPU=1 ENCODERS="mvgrl grace dgi_bn"  DATASETS=hofinet  TAG=g1 bash scripts/run_main_table.sh &
#   GPU=2 ENCODERS="mvgrl_bn grace_bn gin" DATASETS=hofinet TAG=g2 bash scripts/run_main_table.sh &
#   GPU=3 ENCODERS="bgrl gbt dgi_bn grace_bn" DATASETS="amlworld amlnet" TAG=g3 bash scripts/run_main_table.sh &
#   wait
#   bash scripts/merge_main_table.sh
# =============================================================
set -e

GPU="${GPU:-0}"
DATASETS="${DATASETS:-hofinet amlworld amlnet}"
ENCODERS="${ENCODERS:-gbt bgrl dgi mvgrl grace dgi_bn mvgrl_bn grace_bn gin}"
SETTINGS="${SETTINGS:-a b c d}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
TAG="${TAG:-gpu${GPU}}"
# Per-dataset output directories: HOFINET (RQ1) and AMLworld/AMLNet (RQ4).
# Override OUTPUT_DIR_${DS} to point elsewhere.
OUTPUT_DIR_HOFINET="${OUTPUT_DIR_HOFINET:-results/rq1}"
OUTPUT_DIR_AMLWORLD="${OUTPUT_DIR_AMLWORLD:-results/rq4}"
OUTPUT_DIR_AMLNET="${OUTPUT_DIR_AMLNET:-results/rq4}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

mkdir -p "$OUTPUT_DIR_HOFINET" "$OUTPUT_DIR_AMLWORLD" "$OUTPUT_DIR_AMLNET"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

# Process-group cleanup: kill all children on signal so no orphans linger.
PGID=$$
trap 'echo "[$(date)] cleanup: killing PGID $PGID"; kill -- -$PGID 2>/dev/null; exit 130' INT TERM

echo "[$(date)] === main_table sweep (TAG=$TAG, GPU=$GPU, PGID=$PGID) ==="
echo "  DATASETS=$DATASETS"
echo "  ENCODERS=$ENCODERS"
echo "  SETTINGS=$SETTINGS"
echo "  SEEDS=$SEEDS"
echo "  OUTPUT (hofinet)=$OUTPUT_DIR_HOFINET / (amlworld)=$OUTPUT_DIR_AMLWORLD / (amlnet)=$OUTPUT_DIR_AMLNET"

# is_done $NAME $TMP $FINAL: 0 (done, skip) | 1 (not done, run)
is_done() {
    local NAME="$1" TMP="$2" FINAL="$3"
    [ -f "$TMP"   ] && grep -q ",${NAME}," "$TMP"   2>/dev/null && return 0
    [ -f "$FINAL" ] && grep -q ",${NAME}," "$FINAL" 2>/dev/null && return 0
    return 1
}

run_one() {
    local DS_TAG="$1" NODE="$2" EDGE="$3" KNN="$4" RESULT="$5" FINAL="$6"
    local ENC="$7" SETTING="$8" SEED="$9"

    local FLAGS=""
    case "$SETTING" in
        a) FLAGS="" ;;
        b) FLAGS="--knn_graph $KNN" ;;
        c) FLAGS="--subgraph_pool" ;;
        d) FLAGS="--knn_graph $KNN --subgraph_pool" ;;
        *) echo "unknown setting: $SETTING"; exit 1 ;;
    esac

    local NAME="${DS_TAG}_${ENC}_${SETTING}_s${SEED}"
    if is_done "$NAME" "$RESULT" "$FINAL"; then
        echo "[$(date)] [skip] $NAME (already in CSV)"
        return
    fi
    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py \
        --model_name "$NAME" \
        --gpu "$GPU" --seed "$SEED" \
        --encoder_type "$ENC" \
        --node_data_name "$NODE" --edge_data_name "$EDGE" \
        --skip_tsne \
        --metric_save_path "$RESULT" \
        $HP --loss BootstrapLatent \
        $FLAGS 2>&1 | grep -E "^\((E|L)\)" || true
}

for DS in $DATASETS; do
    case "$DS" in
        hofinet)
            DS_TAG="hof"
            NODE="HOFINET_NODE_FEAT"
            EDGE="HOFINET_EDGES"
            KNN="HOFINET_KNN_BEHAV_k10"
            OUT_DIR="$OUTPUT_DIR_HOFINET"
            ;;
        amlworld)
            DS_TAG="aml"
            NODE="amlworld/AMLWORLD_NODE_FEAT"
            EDGE="amlworld/AMLWORLD_EDGES"
            KNN="amlworld/AMLWORLD_KNN_BEHAV_k10"
            OUT_DIR="$OUTPUT_DIR_AMLWORLD"
            ;;
        amlnet)
            DS_TAG="amlnet"
            NODE="amlnet/AMLNET_NODE_FEAT"
            EDGE="amlnet/AMLNET_EDGES"
            KNN="amlnet/AMLNET_KNN_BEHAV_k10"
            OUT_DIR="$OUTPUT_DIR_AMLNET"
            ;;
        *) echo "unknown dataset: $DS"; exit 1 ;;
    esac

    RESULT="${OUT_DIR}/.tmp_${DS}_main_sweep_${TAG}.csv"
    case "$DS" in
        hofinet)  FINAL="${OUT_DIR}/main_sweep.csv" ;;
        amlworld) FINAL="${OUT_DIR}/amlworld_main_sweep.csv" ;;
        amlnet)   FINAL="${OUT_DIR}/amlnet_main_sweep.csv" ;;
    esac
    echo ""
    echo "[$(date)] === Dataset: $DS -> tmp=$RESULT, final=$FINAL ==="

    for ENC in $ENCODERS; do
        echo ""
        echo "----- $DS_TAG / $ENC -----"
        for SETTING in $SETTINGS; do
            for SEED in $SEEDS; do
                run_one "$DS_TAG" "$NODE" "$EDGE" "$KNN" "$RESULT" "$FINAL" \
                        "$ENC" "$SETTING" "$SEED"
            done
        done
    done
done

echo ""
echo "[$(date)] === sweep complete (TAG=$TAG) ==="

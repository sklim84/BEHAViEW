#!/bin/bash
# =============================================================
# HOFINET: 10 encoders × 2 settings (a, b) × 4 seeds = 80 experiments
# For tab:rq1 mean±std
# =============================================================
set -e
GPU=3
RESULT="./results/exp_results_hofinet_ab.csv"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --metric_save_path $RESULT --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins"
KNN="--knn_graph HOFINET_KNN_BEHAV_k10"

echo "[$(date)] === HOFINET (a)/(b) Multi-seed ==="

for ENC in gbt bgrl dgi_bn mvgrl_bn grace_bn gin dgi mvgrl grace gca; do
  echo ""
  echo "===== $ENC ====="
  for SEED in 2024 2025 2026 2027; do
    echo "[$(date)] $ENC s$SEED (a)"
    python -u models/subgraph_cl.py --model_name hof_${ENC}_a_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON 2>&1 | grep "^(E):"

    echo "[$(date)] $ENC s$SEED (b)"
    python -u models/subgraph_cl.py --model_name hof_${ENC}_b_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON $KNN 2>&1 | grep "^(E):"
  done
done

echo ""
echo "[$(date)] === Complete ==="
wc -l $RESULT

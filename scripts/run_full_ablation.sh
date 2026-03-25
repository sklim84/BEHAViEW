#!/bin/bash
# =============================================================
# Full ablation: 5 encoders × 4 settings × 4 seeds = 80 experiments
# =============================================================
GPU=3
RESULT="./results/exp_results_rq.csv"
COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --metric_save_path $RESULT --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins --struct_feats dc pagerank hits_hub hits_auth kcore triangle betweenness"
KNN="--knn_graph HOFINET_KNN_BEHAV_k10"
SUB="--subgraph_pool"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === Full Ablation (5 encoders × 4 settings) ==="

for ENC in bgrl gbt grace dgi mvgrl; do
  echo ""
  echo "===== Encoder: $ENC ====="
  for SEED in 2024 2025 2026 2027; do
    echo "[$(date)] $ENC seed=$SEED"

    # (a) aug view + node-level
    echo "  (a)"
    python -u models/subgraph_cl.py --model_name fa_${ENC}_a_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON 2>&1 | grep "^(E):"

    # (b) behavioral view + node-level
    echo "  (b)"
    python -u models/subgraph_cl.py --model_name fa_${ENC}_b_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON $KNN 2>&1 | grep "^(E):"

    # (c) aug view + subgraph
    echo "  (c)"
    python -u models/subgraph_cl.py --model_name fa_${ENC}_c_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON $SUB 2>&1 | grep "^(E):"

    # (d) behavioral view + subgraph
    echo "  (d)"
    python -u models/subgraph_cl.py --model_name fa_${ENC}_d_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON $KNN $SUB 2>&1 | grep "^(E):"
  done
done

echo ""
echo "[$(date)] === Full Ablation Complete ==="

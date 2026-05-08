#!/bin/bash
# =============================================================
# 4-setting ablation: (a) org, (b) behavioral view, (c) subgraph pool, (d) both
# 3 backbones × 4 settings × 4 seeds = 48 experiments
# =============================================================
GPU=3
RESULT="./results/exp_results_rq.csv"
COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --metric_save_path $RESULT"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
STRUCT="--struct_feats dc pagerank hits_hub hits_auth kcore triangle betweenness"
KNN="--knn_graph HOFINET_KNN_BEHAV_k10"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === 4-Setting Ablation (a/b/c/d) ==="

# ============= BGRL =============
echo ""
echo "===== BGRL ====="
for SEED in 2024 2025 2026 2027; do
  echo "[$(date)] BGRL seed=$SEED"

  # (a) org: augmentation view + node-level
  python -u models/bgrl_w_org.py --model_name abl_bgrl_a_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BarlowTwins 2>&1 | grep "^(E):"

  # (b) behavioral view + node-level
  python -u models/bgrl_w_knn.py --model_name abl_bgrl_b_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BarlowTwins $STRUCT $KNN 2>&1 | grep "^(E):"

  # (c) augmentation view + subgraph pool
  python -u models/bgrl_w_org.py --model_name abl_bgrl_c_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BarlowTwins --subgraph_pool 2>&1 | grep "^(E):"

  # (d) behavioral view + subgraph pool
  python -u models/bgrl_w_knn.py --model_name abl_bgrl_d_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BarlowTwins $STRUCT $KNN --subgraph_pool 2>&1 | grep "^(E):"
done

# ============= DGI-trs =============
echo ""
echo "===== DGI-trs ====="
for SEED in 2024 2025 2026 2027; do
  echo "[$(date)] DGI-trs seed=$SEED"

  # (a) org
  python -u models/dgi_transductive_w_org.py --model_name abl_dgitrs_a_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss JSD 2>&1 | grep "^(E):"

  # (b) behavioral view
  python -u models/dgi_transductive_w_knn.py --model_name abl_dgitrs_b_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss JSD $STRUCT $KNN 2>&1 | grep "^(E):"

  # (c) subgraph pool
  python -u models/dgi_transductive_w_org.py --model_name abl_dgitrs_c_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss JSD --subgraph_pool 2>&1 | grep "^(E):"

  # (d) behavioral view + subgraph pool
  python -u models/dgi_transductive_w_knn.py --model_name abl_dgitrs_d_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss JSD $STRUCT $KNN --subgraph_pool 2>&1 | grep "^(E):"
done

# ============= MVGRL =============
echo ""
echo "===== MVGRL ====="
for SEED in 2024 2025 2026 2027; do
  echo "[$(date)] MVGRL seed=$SEED"

  # (a) org
  python -u models/mvgrl_w_org.py --model_name abl_mvgrl_a_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BootstrapLatent 2>&1 | grep "^(E):"

  # (b) behavioral view
  python -u models/mvgrl_w_knn.py --model_name abl_mvgrl_b_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BootstrapLatent $STRUCT $KNN 2>&1 | grep "^(E):"

  # (c) subgraph pool
  python -u models/mvgrl_w_org.py --model_name abl_mvgrl_c_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BootstrapLatent --subgraph_pool 2>&1 | grep "^(E):"

  # (d) behavioral view + subgraph pool
  python -u models/mvgrl_w_knn.py --model_name abl_mvgrl_d_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BootstrapLatent $STRUCT $KNN --subgraph_pool 2>&1 | grep "^(E):"
done

echo ""
echo "[$(date)] === Ablation Complete ==="
wc -l $RESULT

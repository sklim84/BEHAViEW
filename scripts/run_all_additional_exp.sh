#!/bin/bash
# All additional experiments for reviewer response
# Run with: nohup bash scripts/run_all_additional_exp.sh > logs_additional_exp.log 2>&1 &
GPU=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins --encoder_type gbt"

echo "[$(date)] === START ALL ADDITIONAL EXPERIMENTS ==="

# =============================================================
# Experiment A: Label Fraction (1%, 5%, 10%)
# =============================================================
echo ""
echo "===== [A] Label Fraction ====="
RESULT_A="./results/exp_results_label_fraction.csv"
for RATIO in 0.01 0.05 0.10; do
  for SEED in 2024 2025 2026 2027; do
    NAME="labfrac_gbt_d_r${RATIO}_s${SEED}"
    grep -q "$NAME" $RESULT_A 2>/dev/null && echo "  skip $NAME" && continue
    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py --model_name $NAME --gpu $GPU --seed $SEED \
      $BASE --knn_graph HOFINET_KNN_BEHAV_k10 --subgraph_pool \
      --train_ratio $RATIO --metric_save_path $RESULT_A 2>&1 | tail -2
  done
done
echo "[$(date)] Label fraction done: $(wc -l < $RESULT_A) rows"

# =============================================================
# Experiment B+E: Feature Ablation / GCPAL-MLGCL proxy
# =============================================================
echo ""
echo "===== [B+E] Feature Ablation (k-NN type) ====="

# Build missing k-NN graphs if needed
for F in HOFINET_KNN_FEAT_k10 HOFINET_KNN_STRUCT_k10; do
  if [ ! -f "datasets/${F}.csv" ]; then
    echo "[$(date)] Building $F..."
    python datasets/build_knn_graph.py --k 10
    break
  fi
done

RESULT_B="./results/exp_results_feature_ablation.csv"
for KNN_TYPE in FEAT BEHAV STRUCT; do
  KNN_FILE="HOFINET_KNN_${KNN_TYPE}_k10"
  if [ ! -f "datasets/${KNN_FILE}.csv" ]; then
    echo "  WARN: $KNN_FILE not found, skipping"
    continue
  fi
  for SEED in 2024 2025 2026 2027; do
    NAME="feat_gbt_d_${KNN_TYPE}_s${SEED}"
    grep -q "$NAME" $RESULT_B 2>/dev/null && echo "  skip $NAME" && continue
    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py --model_name $NAME --gpu $GPU --seed $SEED \
      $BASE --knn_graph $KNN_FILE --subgraph_pool \
      --metric_save_path $RESULT_B 2>&1 | tail -2
  done
done
echo "[$(date)] Feature ablation done: $(wc -l < $RESULT_B) rows"

# =============================================================
# Experiment D: Computational Cost
# =============================================================
echo ""
echo "===== [D] Computational Cost ====="
python3 -c "
import time, torch, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. k-NN construction time
df = pd.read_csv('datasets/HOFINET_NODE_FEAT.csv')
struct_cols = {'in_dc','out_dc','in_count','out_count'}
agg_cols = [c for c in df.columns if c.startswith(('out_','in_','md_','fnd_','entropy'))]
behav_cols = [c for c in agg_cols if c not in struct_cols]
X = df[behav_cols].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0]=1; X = X/norms

t0 = time.time()
nn = NearestNeighbors(n_neighbors=11, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
nn.fit(X)
nn.kneighbors(X)
t_knn = time.time() - t0
print(f'k-NN construction (k=10, N={len(X)}): {t_knn:.1f}s')

# 2. Training time (200 epochs)
import sys; sys.path.insert(0,'.')
from models.subgraph_cl import *
from data_loader import load_graph_data, load_knn_graph
from config import parser
args = parser.parse_args(['--gpu','$GPU','--node_data_name','HOFINET_NODE_FEAT',
    '--edge_data_name','HOFINET_EDGES','--lr','0.0005','--hidden_dim','256',
    '--gconv_nlayers','2','--loss','BarlowTwins','--encoder_type','gbt',
    '--knn_graph','HOFINET_KNN_BEHAV_k10','--subgraph_pool','--skip_tsne',
    '--model_name','timing_test','--seed','2024'])
device = torch.device('cuda:$GPU')
data, _ = load_graph_data(args, device=device)
edge_index_knn = load_knn_graph(args.knn_graph, device=device)
encoder = ENCODERS['gbt'](input_dim=data.x.size(1), hidden_dim=256, num_layers=2, dropout=0.2)
model = SubgraphCL(encoder=encoder, hidden_dim=256, use_subgraph=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

torch.cuda.synchronize()
t0 = time.time()
for epoch in range(200):
    train(model, data.x, data.edge_index, edge_index_knn, optimizer)
torch.cuda.synchronize()
t_train = time.time() - t0
print(f'Training (200 epochs): {t_train:.1f}s ({t_train/200:.2f}s/epoch)')

# 3. Inference time
torch.cuda.synchronize()
t0 = time.time()
z = get_embeddings(model, data.x, data.edge_index, edge_index_knn)
torch.cuda.synchronize()
t_infer = time.time() - t0
print(f'Inference: {t_infer:.2f}s')

print(f'Total: {t_knn + t_train + t_infer:.1f}s')
" 2>&1
echo "[$(date)] Computational cost done"

echo ""
echo "[$(date)] === ALL ADDITIONAL EXPERIMENTS COMPLETE ==="

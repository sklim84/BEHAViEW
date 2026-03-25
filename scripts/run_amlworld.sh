#!/bin/bash
# =============================================================
# AMLworld HI-Small: 전처리 → k-NN 구축 → (a)(b)(c)(d) ablation
# =============================================================
set -e
GPU=2
RESULT="./results/exp_results_amlworld.csv"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === AMLworld Pipeline ==="

# Step 1: Preprocessing
echo "[$(date)] Step 1: Preprocessing"
python -u datasets/pp_amlworld.py 2>&1

# Step 2: Build behavioral k-NN
echo ""
echo "[$(date)] Step 2: Build behavioral k-NN (k=10)"
python3 -c "
import pandas as pd, numpy as np, time
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('datasets/amlworld/AMLWORLD_NODE_FEAT.csv')
structural = {'dc','in_dc','out_dc','pagerank','hits_hub','hits_auth','kcore','triangle','betweenness'}
behav_cols = [c for c in df.columns if c not in ('account','label') and c not in structural]
# Exclude count columns for behavioral k-NN (same as HOFINET)
behav_knn_cols = [c for c in behav_cols if 'count' not in c]
X = df[behav_knn_cols].values
print(f'Behavioral k-NN features ({len(behav_knn_cols)}): {behav_knn_cols}')

X = StandardScaler().fit_transform(X)
norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0]=1; X = X/norms
t0 = time.time()
nn = NearestNeighbors(n_neighbors=11, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
nn.fit(X); _, indices = nn.kneighbors(X)
src, tgt = [], []
for i in range(len(X)):
    for j in range(1,11):
        src.append(i); tgt.append(indices[i,j])
pd.DataFrame({'source':src,'target':tgt}).to_csv('datasets/amlworld/AMLWORLD_KNN_BEHAV_k10.csv', index=False)
print(f'Saved: {len(src)} edges, {time.time()-t0:.0f}s')
" 2>&1

# Step 3: Ablation experiments
echo ""
echo "[$(date)] Step 3: Ablation (5 encoders × 4 settings × 4 seeds)"
COMMON="--node_data_name amlworld/AMLWORLD_NODE_FEAT --edge_data_name amlworld/AMLWORLD_EDGES --skip_tsne --metric_save_path $RESULT --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins"
KNN="--knn_graph amlworld/AMLWORLD_KNN_BEHAV_k10"
SUB="--subgraph_pool"

for ENC in gbt bgrl dgi_bn grace_bn dgi; do
  echo ""
  echo "===== $ENC ====="
  for SEED in 2024 2025 2026 2027; do
    echo "[$(date)] $ENC s$SEED (a)"
    python -u models/subgraph_cl.py --model_name aml_${ENC}_a_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON 2>&1 | grep "^(E):"
    echo "[$(date)] $ENC s$SEED (b)"
    python -u models/subgraph_cl.py --model_name aml_${ENC}_b_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON $KNN 2>&1 | grep "^(E):"
    echo "[$(date)] $ENC s$SEED (c)"
    python -u models/subgraph_cl.py --model_name aml_${ENC}_c_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON $SUB 2>&1 | grep "^(E):"
    echo "[$(date)] $ENC s$SEED (d)"
    python -u models/subgraph_cl.py --model_name aml_${ENC}_d_s${SEED} --gpu $GPU --seed $SEED --encoder_type $ENC $COMMON $KNN $SUB 2>&1 | grep "^(E):"
  done
done

# Step 4: Homophily analysis
echo ""
echo "[$(date)] Step 4: Homophily"
python3 -c "
import pandas as pd, numpy as np
nf = pd.read_csv('datasets/amlworld/AMLWORLD_NODE_FEAT.csv')
labels = nf['label'].values
n_fraud = (labels==1).sum()
print(f'Nodes: {len(nf):,} (laundering: {n_fraud:,}, {n_fraud*100/len(nf):.2f}%)')
for name, path in [('Transaction', 'datasets/amlworld/AMLWORLD_EDGES.csv'),
                    ('Behavioral k-NN', 'datasets/amlworld/AMLWORLD_KNN_BEHAV_k10.csv')]:
    df = pd.read_csv(path)
    src, tgt = df['source'].values, df['target'].values
    if name == 'Transaction':
        ni = {acc:i for i,acc in enumerate(nf['account'])}
        src = np.array([ni.get(s,-1) for s in src])
        tgt = np.array([ni.get(t,-1) for t in tgt])
        valid = (src>=0)&(tgt>=0); src=src[valid]; tgt=tgt[valid]
    ff = np.sum((labels[src]==1)&(labels[tgt]==1))
    fb = np.sum((labels[src]==1)^(labels[tgt]==1))
    homo = np.sum(labels[src]==labels[tgt])/len(src)
    print(f'{name:20s}: homo={homo:.4f}, F-F={ff:,}, F-B={fb:,}, ratio=1:{fb/max(ff,1):.1f}')
" 2>&1

echo ""
echo "[$(date)] === AMLworld Complete ==="

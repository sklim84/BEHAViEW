"""Build behavioral k-NN graphs for AMLNet and PaySim.

Behavioral feature selection follows BECON convention:
  - Use amount stats + temporal + entropy
  - Exclude count features (consistent with HOFINET/AMLworld pure behavioral)

Output:
  datasets/amlnet/AMLNET_KNN_BEHAV_k10.csv
  datasets/paysim/PAYSIM_KNN_BEHAV_k10.csv
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_behav_knn(node_csv, out_csv, k=10):
    df = pd.read_csv(node_csv)
    structural = {'in_dc', 'out_dc'}  # not present in AMLNet/PaySim, kept for symmetry
    excluded = {'account', 'label'}
    # Behavioral k-NN: drop count + structural columns (BECON pure-behavioral convention)
    behav_knn_cols = [c for c in df.columns
                      if c not in excluded
                      and c not in structural
                      and 'count' not in c]
    print(f'[INFO] {os.path.basename(node_csv)}: '
          f'{len(behav_knn_cols)} behavioral k-NN features')
    print(f'[INFO]   features = {behav_knn_cols}')

    X = df[behav_knn_cols].values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms

    print(f'[INFO] Building k-NN (k={k}, ball_tree)...')
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree',
                          metric='euclidean', n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    elapsed = time.time() - t0

    src, tgt = [], []
    for i in range(len(X)):
        for j in range(1, k + 1):
            src.append(i)
            tgt.append(indices[i, j])
    pd.DataFrame({'source': src, 'target': tgt}).to_csv(out_csv, index=False)
    print(f'[INFO] Saved {out_csv}: {len(src):,} edges, {elapsed:.0f}s')


if __name__ == '__main__':
    build_behav_knn(
        'datasets/amlnet/AMLNET_NODE_FEAT.csv',
        'datasets/amlnet/AMLNET_KNN_BEHAV_k10.csv',
        k=10,
    )
    build_behav_knn(
        'datasets/paysim/PAYSIM_NODE_FEAT.csv',
        'datasets/paysim/PAYSIM_KNN_BEHAV_k10.csv',
        k=10,
    )

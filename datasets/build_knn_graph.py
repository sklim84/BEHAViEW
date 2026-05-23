"""
k-NN 그래프 사전 구축 — feature 유사도 및 centrality 유사도 기반.

사용법:
    python datasets/build_knn_graph.py --k 10
    python datasets/build_knn_graph.py --k 10 20 50
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_knn_edges(features, k):
    """feature matrix로부터 k-NN edge list 생성.
    L2-normalized + euclidean ≈ cosine similarity, but uses ball_tree for speed.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    # L2 normalize → euclidean distance ∝ cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree',
                          metric='euclidean', n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    src, tgt = [], []
    for i in range(len(X)):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            j = indices[i, j_idx]
            src.append(i)
            tgt.append(j)

    return np.array(src), np.array(tgt)


def main():
    parser = argparse.ArgumentParser(description='Build k-NN graphs')
    parser.add_argument('--node_data', default='datasets/hofinet/HOFINET_NODE_FEAT.csv')
    parser.add_argument('--k', nargs='+', type=int, default=[10])
    parser.add_argument('--output_dir', default='datasets/hofinet')
    args = parser.parse_args()

    print(f'Loading {args.node_data}...')
    df = pd.read_csv(args.node_data)

    # Feature columns
    agg_cols = [c for c in df.columns if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy'))]
    cen_cols = ['dc', 'cc', 'pagerank', 'hits_hub', 'hits_auth', 'kcore', 'triangle']
    cen_cols = [c for c in cen_cols if c in df.columns]

    x_behav = df[agg_cols].values
    x_struct = df[cen_cols].values

    print(f'  Nodes: {len(df)}')
    print(f'  x_behav features: {len(agg_cols)}')
    print(f'  x_struct features: {len(cen_cols)} — {cen_cols}')

    for k in args.k:
        # (D) Feature k-NN graph
        if not os.path.exists(os.path.join(args.output_dir, f'HOFINET_KNN_FEAT_k{k}.csv')):
            print(f'\n[k={k}] Building feature k-NN graph...')
            t0 = time.time()
            src, tgt = build_knn_edges(x_behav, k)
            elapsed = time.time() - t0
            out_path = os.path.join(args.output_dir, f'HOFINET_KNN_FEAT_k{k}.csv')
            pd.DataFrame({'source': src, 'target': tgt}).to_csv(out_path, index=False)
            print(f'  Saved: {out_path} ({len(src)} edges, {elapsed:.1f}s)')
        else:
            print(f'\n[k={k}] Feature k-NN graph already exists, skipping.')

        # (F) Centrality k-NN graph
        if not os.path.exists(os.path.join(args.output_dir, f'HOFINET_KNN_CEN_k{k}.csv')):
            print(f'[k={k}] Building centrality k-NN graph...')
            t0 = time.time()
            src, tgt = build_knn_edges(x_struct, k)
            elapsed = time.time() - t0
            out_path = os.path.join(args.output_dir, f'HOFINET_KNN_CEN_k{k}.csv')
            pd.DataFrame({'source': src, 'target': tgt}).to_csv(out_path, index=False)
            print(f'  Saved: {out_path} ({len(src)} edges, {elapsed:.1f}s)')
        else:
            print(f'[k={k}] Centrality k-NN graph already exists, skipping.')

        # Pure behavioral k-NN (exclude in_dc, out_dc)
        out_pure = os.path.join(args.output_dir, f'HOFINET_KNN_PURE_k{k}.csv')
        if not os.path.exists(out_pure):
            pure_cols = [c for c in agg_cols if c not in ('in_dc', 'out_dc')]
            x_pure = df[pure_cols].values
            print(f'[k={k}] Building pure behavioral k-NN graph ({len(pure_cols)} feats, no dc)...')
            t0 = time.time()
            src, tgt = build_knn_edges(x_pure, k)
            elapsed = time.time() - t0
            pd.DataFrame({'source': src, 'target': tgt}).to_csv(out_pure, index=False)
            print(f'  Saved: {out_pure} ({len(src)} edges, {elapsed:.1f}s)')
        else:
            print(f'[k={k}] Pure behavioral k-NN already exists, skipping.')

        # Strict behavioral k-NN (amount/entropy only, no count/dc)
        out_strict = os.path.join(args.output_dir, f'HOFINET_KNN_BEHAV_k{k}.csv')
        if not os.path.exists(out_strict):
            struct_cols = {'in_dc', 'out_dc', 'in_count', 'out_count'}
            behav_cols = [c for c in agg_cols if c not in struct_cols]
            x_behav = df[behav_cols].values
            print(f'[k={k}] Building strict behavioral k-NN graph ({len(behav_cols)} feats: {behav_cols})...')
            t0 = time.time()
            src, tgt = build_knn_edges(x_behav, k)
            elapsed = time.time() - t0
            pd.DataFrame({'source': src, 'target': tgt}).to_csv(out_strict, index=False)
            print(f'  Saved: {out_strict} ({len(src)} edges, {elapsed:.1f}s)')
        else:
            print(f'[k={k}] Strict behavioral k-NN already exists, skipping.')

        # Structural k-NN (count + dc from x_behav + all centrality)
        out_struct = os.path.join(args.output_dir, f'HOFINET_KNN_STRUCT_k{k}.csv')
        if not os.path.exists(out_struct):
            struct_from_agg = [c for c in agg_cols if c in ('out_count', 'in_count', 'in_dc', 'out_dc')]
            struct_cols = struct_from_agg + [c for c in cen_cols if c not in struct_from_agg]
            x_struct = df[struct_cols].values
            print(f'[k={k}] Building structural k-NN graph ({len(struct_cols)} feats: {struct_cols})...')
            t0 = time.time()
            src, tgt = build_knn_edges(x_struct, k)
            elapsed = time.time() - t0
            pd.DataFrame({'source': src, 'target': tgt}).to_csv(out_struct, index=False)
            print(f'  Saved: {out_struct} ({len(src)} edges, {elapsed:.1f}s)')
        else:
            print(f'[k={k}] Structural k-NN already exists, skipping.')

        # Hybrid k-NN graph (feature + centrality concatenated)
        out_hybrid = os.path.join(args.output_dir, f'HOFINET_KNN_HYBRID_k{k}.csv')
        if not os.path.exists(out_hybrid):
            print(f'[k={k}] Building hybrid k-NN graph (feat+cen)...')
            t0 = time.time()
            x_hybrid = np.hstack([x_behav, x_struct])
            src, tgt = build_knn_edges(x_hybrid, k)
            elapsed = time.time() - t0
            pd.DataFrame({'source': src, 'target': tgt}).to_csv(out_hybrid, index=False)
            print(f'  Saved: {out_hybrid} ({len(src)} edges, {elapsed:.1f}s)')
        else:
            print(f'[k={k}] Hybrid k-NN graph already exists, skipping.')


if __name__ == '__main__':
    main()

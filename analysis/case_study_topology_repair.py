"""
Case study for topology repair: per-suspicious-account S-ratio in the
transaction graph vs the behavioral k-NN graph.

Outputs (under results/case_study/):
  - distributions.csv        per-account S-ratios + counts (one row per suspicious account)
  - representative.json      selected extreme-mismatch account + its 1-hop neighbors in each graph
  - aggregate_stats.json     summary statistics

Usage:
    python analysis/case_study_topology_repair.py
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Behavioral features used to build G_bhv (matches build_knn_graph.py BEHAV variant).
BEHAV_EXCLUDE = {'in_dc', 'out_dc', 'in_count', 'out_count'}
K = 10


def load_features(path):
    df = pd.read_csv(path)
    agg_cols = [c for c in df.columns if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy'))]
    behav_cols = [c for c in agg_cols if c not in BEHAV_EXCLUDE]
    return df, behav_cols


def build_behavioral_knn(features, k):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree',
                          metric='euclidean', n_jobs=-1)
    nn.fit(X)
    return nn, X


def build_tx_adjacency(edges_path, account_to_idx):
    df = pd.read_csv(edges_path)
    src = df['source'].map(account_to_idx).to_numpy()
    tgt = df['target'].map(account_to_idx).to_numpy()
    valid = ~(np.isnan(src) | np.isnan(tgt))
    src = src[valid].astype(np.int64)
    tgt = tgt[valid].astype(np.int64)
    adj = defaultdict(set)
    for s, t in zip(src, tgt):
        adj[int(s)].add(int(t))
        adj[int(t)].add(int(s))
    return adj


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, 'results', 'case_study')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print('Loading node features...')
    df, behav_cols = load_features(os.path.join(base, 'datasets', 'hofinet', 'HOFINET_NODE_FEAT.csv'))
    print(f'  Nodes: {len(df):,}  behavioral features: {len(behav_cols)}')
    labels = df['label'].to_numpy()
    accounts = df['account'].to_numpy()
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}
    susp_indices = np.where(labels == 1)[0]
    print(f'  Suspicious accounts: {len(susp_indices):,} ({len(susp_indices) / len(df) * 100:.2f}%)')

    print(f'Building tx-graph adjacency from edges...')
    adj_tx = build_tx_adjacency(os.path.join(base, 'datasets', 'hofinet', 'HOFINET_EDGES.csv'),
                                account_to_idx)
    print(f'  Adjacency for {len(adj_tx):,} nodes, took {time.time() - t0:.1f}s total')

    print(f'Building behavioral k-NN ball_tree (k={K})...')
    t1 = time.time()
    nn, X = build_behavioral_knn(df[behav_cols].to_numpy(), K)
    print(f'  Built in {time.time() - t1:.1f}s')

    print('Querying behavioral k-NN for every suspicious account...')
    t2 = time.time()
    _, bhv_idx = nn.kneighbors(X[susp_indices])  # shape (n_susp, K+1)
    bhv_idx = bhv_idx[:, 1:]  # drop self
    print(f'  Queried in {time.time() - t2:.1f}s')

    print('Computing S-ratios...')
    rows = []
    for j, i in enumerate(susp_indices):
        tx_neighbors = adj_tx.get(int(i), set())
        tx_deg = len(tx_neighbors)
        if tx_deg == 0:
            tx_S = 0
            tx_B = 0
            tx_ratio = float('nan')
        else:
            tx_S = sum(1 for n in tx_neighbors if labels[n] == 1)
            tx_B = tx_deg - tx_S
            tx_ratio = tx_S / tx_deg

        bhv_neighbors = bhv_idx[j].tolist()
        bhv_S = int(sum(labels[n] == 1 for n in bhv_neighbors))
        bhv_B = K - bhv_S
        bhv_ratio = bhv_S / K

        rows.append({
            'idx': int(i),
            'account': accounts[i],
            'tx_deg': tx_deg,
            'tx_S': tx_S,
            'tx_B': tx_B,
            'tx_S_ratio': tx_ratio,
            'bhv_S': bhv_S,
            'bhv_B': bhv_B,
            'bhv_S_ratio': bhv_ratio,
        })

    dist = pd.DataFrame(rows)
    dist['shift'] = dist['bhv_S_ratio'] - dist['tx_S_ratio']
    dist.to_csv(os.path.join(out_dir, 'distributions.csv'), index=False)
    print(f"  Saved distributions.csv ({len(dist):,} rows)")

    # ---------- aggregate summary ----------
    with_tx = dist.dropna(subset=['tx_S_ratio'])
    summary = {
        'k': K,
        'n_suspicious_total': int(len(dist)),
        'n_with_tx_neighbors': int(len(with_tx)),
        'n_isolated_in_tx': int(dist['tx_deg'].eq(0).sum()),
        'tx_S_ratio_mean': float(with_tx['tx_S_ratio'].mean()),
        'tx_S_ratio_median': float(with_tx['tx_S_ratio'].median()),
        'bhv_S_ratio_mean': float(dist['bhv_S_ratio'].mean()),
        'bhv_S_ratio_median': float(dist['bhv_S_ratio'].median()),
        'shift_mean': float(with_tx['shift'].mean()),
        'shift_median': float(with_tx['shift'].median()),
        'p_tx_majority_susp': float((with_tx['tx_S_ratio'] >= 0.5).mean()),
        'p_bhv_majority_susp': float((dist['bhv_S_ratio'] >= 0.5).mean()),
    }
    print('Aggregate summary:')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    with open(os.path.join(out_dir, 'aggregate_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---------- pick representative: most extreme mismatch ----------
    # Filter candidates: tx_deg in a reasonable readable range, behavioral mostly suspicious
    candidates = with_tx[
        (with_tx['tx_deg'] >= 15) &
        (with_tx['tx_deg'] <= 60) &
        (with_tx['tx_S_ratio'] <= 0.10) &
        (with_tx['bhv_S_ratio'] >= 0.50)
    ].sort_values(['shift', 'tx_deg'], ascending=[False, True])
    if len(candidates) == 0:
        # relax to top by shift
        candidates = with_tx.sort_values('shift', ascending=False)
    rep_row = candidates.iloc[0]
    rep_idx = int(rep_row['idx'])
    print(f"\nRepresentative: idx={rep_idx} account={rep_row['account']}")
    print(f"  tx_deg={int(rep_row['tx_deg'])} tx_S={int(rep_row['tx_S'])} "
          f"tx_B={int(rep_row['tx_B'])} tx_S_ratio={rep_row['tx_S_ratio']:.3f}")
    print(f"  bhv_S={int(rep_row['bhv_S'])} bhv_B={int(rep_row['bhv_B'])} "
          f"bhv_S_ratio={rep_row['bhv_S_ratio']:.3f}")

    tx_neighbors = sorted(adj_tx[rep_idx])
    bhv_neighbors_idx = bhv_idx[np.where(susp_indices == rep_idx)[0][0]].tolist()

    # Induced subgraph edges: ego + 1-hop neighbors
    tx_node_set = set(tx_neighbors + [rep_idx])
    tx_induced = []
    for u in tx_node_set:
        for v in adj_tx.get(u, set()):
            if v in tx_node_set and u < v:  # undirected, dedup
                tx_induced.append((int(u), int(v)))

    # For behavioral: each node's k-NN may include other nodes in the same set
    bhv_node_set = set(bhv_neighbors_idx + [rep_idx])
    bhv_query_indices = np.array(sorted(bhv_node_set))
    _, bhv_query_knn = nn.kneighbors(X[bhv_query_indices])  # (n_set, K+1)
    bhv_induced = []
    seen = set()
    for src, knn_row in zip(bhv_query_indices, bhv_query_knn):
        for tgt in knn_row[1:]:  # skip self
            if int(tgt) in bhv_node_set:
                a, b = (int(src), int(tgt)) if src < tgt else (int(tgt), int(src))
                if (a, b) not in seen:
                    seen.add((a, b))
                    bhv_induced.append((a, b))

    print(f"  Induced edges: tx={len(tx_induced)}, bhv={len(bhv_induced)}")

    rep_payload = {
        'idx': rep_idx,
        'account': rep_row['account'],
        'label': int(labels[rep_idx]),
        'tx_neighbors': [
            {'idx': int(n), 'account': accounts[n], 'label': int(labels[n])}
            for n in tx_neighbors
        ],
        'bhv_neighbors': [
            {'idx': int(n), 'account': accounts[n], 'label': int(labels[n])}
            for n in bhv_neighbors_idx
        ],
        'tx_induced_edges': tx_induced,
        'bhv_induced_edges': bhv_induced,
        'tx_S_ratio': float(rep_row['tx_S_ratio']),
        'bhv_S_ratio': float(rep_row['bhv_S_ratio']),
        'tx_deg': int(rep_row['tx_deg']),
        'tx_S': int(rep_row['tx_S']),
        'tx_B': int(rep_row['tx_B']),
        'bhv_S': int(rep_row['bhv_S']),
        'bhv_B': int(rep_row['bhv_B']),
    }
    with open(os.path.join(out_dir, 'representative.json'), 'w') as f:
        json.dump(rep_payload, f, indent=2)
    print(f"  Saved representative.json")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()

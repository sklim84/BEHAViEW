"""
RQ4: k-NN graph별 homophily ratio 측정.
각 k-NN graph에서 같은 label의 노드끼리 연결된 edge 비율을 계산.

사용법:
    python analysis/homophily_knn.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np



def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load labels
    node_df = pd.read_csv(os.path.join(base, 'datasets/HOFINET_NODE_FEAT.csv'))
    labels = node_df['label'].values
    n_susp = (labels == 1).sum()
    n_total = len(labels)
    print(f'Nodes: {n_total:,} (suspicious: {n_susp:,}, {n_susp/n_total*100:.2f}%)')
    print()

    # Graphs to measure
    graphs = [
        ('Transaction (original)', 'datasets/HOFINET_EDGES.csv', True),
        ('Behavioral k-NN (A, 8 feats)', 'datasets/HOFINET_KNN_BEHAV_k10.csv', False),
        ('Feature k-NN (A+B1, 12 feats)', 'datasets/HOFINET_KNN_FEAT_k10.csv', False),
        ('Structural k-NN (B, 11 feats)', 'datasets/HOFINET_KNN_STRUCT_k10.csv', False),
        ('Centrality k-NN (B2, 7 feats)', 'datasets/HOFINET_KNN_CEN_k10.csv', False),
        ('Hybrid k-NN (A+B, 19 feats)', 'datasets/HOFINET_KNN_HYBRID_k10.csv', False),
    ]

    print(f'{"Graph":<35s} {"Edges":>12s} {"Homophily":>10s} {"S-S":>12s} {"S-B":>13s}')
    print('-' * 85)

    for name, path, is_directed in graphs:
        full_path = os.path.join(base, path)
        if not os.path.exists(full_path):
            print(f'{name:<35s} {"(not found)":>12s}')
            continue

        df = pd.read_csv(full_path)
        src = df['source'].values
        tgt = df['target'].values

        # For transaction graph, need to map account names to indices
        if is_directed:
            node_index = {acc: i for i, acc in enumerate(node_df['account'])}
            src_idx = np.array([node_index.get(s, -1) for s in src])
            tgt_idx = np.array([node_index.get(t, -1) for t in tgt])
            valid = (src_idx >= 0) & (tgt_idx >= 0)
            src_idx = src_idx[valid]
            tgt_idx = tgt_idx[valid]
        else:
            src_idx = src
            tgt_idx = tgt

        total = len(src_idx)
        same_label = np.sum(labels[src_idx] == labels[tgt_idx])
        homophily = same_label / total

        # Suspicious-suspicious and suspicious-benign edges
        src_susp = labels[src_idx] == 1
        tgt_susp = labels[tgt_idx] == 1
        ss = np.sum(src_susp & tgt_susp)
        sb = np.sum(src_susp ^ tgt_susp)  # XOR = one suspicious, one benign

        print(f'{name:<35s} {total:>12,} {homophily:>10.4f} {ss:>12,} {sb:>13,}')

    print()
    print('Homophily = proportion of edges connecting same-label nodes')
    print('Higher homophily → GNN neighborhood aggregation more effective for classification')


if __name__ == '__main__':
    main()

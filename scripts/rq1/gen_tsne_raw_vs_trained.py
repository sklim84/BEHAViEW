"""Two-panel t-SNE: raw 20-d behavioral features vs BehaView (d) trained
joint embeddings, on the same ATNET subsample.

Narrative: behavioral features already separate the suspicious class
somewhat (left); BehaView's repaired-topology contrastive training
further compresses suspicious accounts into a tighter cluster (right).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

DATA_DIR = 'results/embeddings/tsne_v0'
FIG_DIR = 'results/rq1/figures'
os.makedirs(FIG_DIR, exist_ok=True)

NODE_DATA = 'datasets/atnet/ATNET_NODE_FEAT.csv'

# Same as build_knn_graph_variants._behavioral_columns(): exclude label,
# account, network-derived counts/degree, and structural features.
EXCLUDE = {
    'label', 'account',
    'in_dc', 'out_dc', 'in_count', 'out_count',
    'dc', 'pagerank', 'hits_hub', 'hits_auth', 'kcore', 'triangle', 'betweenness',
}


def _preprocess(X):
    X = StandardScaler().fit_transform(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def load_raw_behavioral():
    df = pd.read_csv(NODE_DATA)
    y = df['label'].to_numpy(dtype=np.int64)
    cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[cols].values
    X = _preprocess(X)
    print(f'raw behavioral: n={X.shape[0]:,}, d={X.shape[1]}')
    return X, y, cols


def load_trained(setting, seed):
    p = os.path.join(DATA_DIR, f'ATNET_z_{setting}_s{seed}.npz')
    d = np.load(p, allow_pickle=True)
    return d['z'], d['y']


def subsample(y, n_total=10000, n_susp=2000, seed=2025):
    rng = np.random.default_rng(seed)
    idx_susp = np.where(y == 1)[0]
    idx_ben = np.where(y == 0)[0]
    take_susp = min(n_susp, len(idx_susp))
    take_ben = min(n_total - take_susp, len(idx_ben))
    susp_pick = rng.choice(idx_susp, take_susp, replace=False)
    ben_pick = rng.choice(idx_ben, take_ben, replace=False)
    return np.concatenate([ben_pick, susp_pick])


def tsne(z, perplexity=30, seed=2025):
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                init='pca', learning_rate='auto', n_jobs=-1).fit_transform(z)


def silhouette(xy, y):
    from sklearn.metrics import silhouette_score
    try:
        return silhouette_score(xy, y)
    except Exception:
        return float('nan')


def plot(ax, xy, y):
    ben = y == 0
    susp = y == 1
    ax.scatter(xy[ben, 0], xy[ben, 1], s=3, c='#bbbbbb', alpha=0.45, linewidths=0)
    ax.scatter(xy[susp, 0], xy[susp, 1], s=10, c='#d62728', alpha=0.85, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)


def save_single_panel(xy, y, base_path):
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    plot(ax, xy, y)
    plt.tight_layout()
    plt.savefig(base_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(base_path + '.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Saved: {base_path}.pdf')
    print(f'Saved: {base_path}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--n_total', type=int, default=10000)
    parser.add_argument('--n_susp', type=int, default=2000)
    parser.add_argument('--perplexity', type=int, default=30)
    args = parser.parse_args()

    X_raw, y_raw, cols = load_raw_behavioral()
    z_trained, y_trained = load_trained('d', args.seed)
    # node order is identical across raw and trained — same data, same row order
    assert (y_raw == y_trained).all(), 'label order mismatch'

    sub_idx = subsample(y_raw, n_total=args.n_total, n_susp=args.n_susp, seed=args.seed)
    y_sub = y_raw[sub_idx]
    print(f'subsampled n={len(sub_idx):,} (susp={int((y_sub == 1).sum()):,})')

    print('t-SNE on raw behavioral features...')
    raw_xy = tsne(X_raw[sub_idx], perplexity=args.perplexity, seed=args.seed)
    sil_raw = silhouette(raw_xy, y_sub)
    print(f'  silhouette = {sil_raw:.4f}')

    print('t-SNE on BehaView (d) trained embeddings...')
    trained_xy = tsne(z_trained[sub_idx], perplexity=args.perplexity, seed=args.seed)
    sil_trained = silhouette(trained_xy, y_sub)
    print(f'  silhouette = {sil_trained:.4f}')

    save_single_panel(raw_xy, y_sub, os.path.join(FIG_DIR, 'fig_tsne_raw'))
    save_single_panel(trained_xy, y_sub, os.path.join(FIG_DIR, 'fig_tsne_trained'))
    print(f'\nsilhouette: raw={sil_raw:+.4f}  trained={sil_trained:+.4f}')


if __name__ == '__main__':
    main()

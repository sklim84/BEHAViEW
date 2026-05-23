"""Render the t-SNE figure from BehaView embeddings dumped by
`extract_embeddings.sh`. Two-panel comparison:
    (a) baseline   -- transaction graph, node-level
    (d) proposed  -- repaired topology + subgraph pooling

Color = suspicious label. Suspicious nodes are plotted on top so the
minority class stays visible.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

DATA_DIR = 'results/embeddings/tsne_v0'
FIG_DIR = 'results/rq1/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def load_npz(setting, seed):
    p = os.path.join(DATA_DIR, f'HOFINET_z_{setting}_s{seed}.npz')
    if not os.path.exists(p):
        raise FileNotFoundError(p)
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


def plot(ax, xy, y, title):
    ben = y == 0
    susp = y == 1
    ax.scatter(xy[ben, 0], xy[ben, 1], s=3, c='#bbbbbb', alpha=0.45, label='benign', linewidths=0)
    ax.scatter(xy[susp, 0], xy[susp, 1], s=8, c='#d62728', alpha=0.85, label='suspicious', linewidths=0)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--n_total', type=int, default=10000)
    parser.add_argument('--n_susp', type=int, default=2000)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--settings', nargs='+', default=['a', 'b', 'c', 'd'])
    args = parser.parse_args()

    embeds = {}
    for s in args.settings:
        embeds[s] = load_npz(s, args.seed)
        print(f'  loaded ({s}): z shape {embeds[s][0].shape}')

    # Use the labels from setting (a); all settings share the same node order.
    _, y_all = embeds[args.settings[0]]
    sub_idx = subsample(y_all, n_total=args.n_total, n_susp=args.n_susp, seed=args.seed)
    y_sub = y_all[sub_idx]
    print(f'subsampled n={len(sub_idx):,}  (susp={int((y_sub == 1).sum()):,})')

    coords = {}
    for s in args.settings:
        z, _ = embeds[s]
        print(f't-SNE on setting ({s})...')
        coords[s] = tsne(z[sub_idx], perplexity=args.perplexity, seed=args.seed)

    titles = {
        'a': r'(a) Transaction + node',
        'b': r'(b) Recovered + node',
        'c': r'(c) Transaction + pool',
        'd': r'(d) Repaired + pool  (proposed)',
    }
    n_panels = len(args.settings)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
    if n_panels == 1:
        axes = [axes]
    for ax, s in zip(axes, args.settings):
        plot(ax, coords[s], y_sub, titles.get(s, s))
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=11)
    plt.tight_layout()
    out_pdf = os.path.join(FIG_DIR, 'fig_tsne_hofinet.pdf')
    out_png = os.path.join(FIG_DIR, 'fig_tsne_hofinet.png')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()

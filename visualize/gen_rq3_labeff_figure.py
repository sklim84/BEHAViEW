"""Generate RQ3 label-efficiency line plot for BehaView paper.

Single figure with 3 panels (HOFINET, AMLworld, AMLNet).
X-axis: train_ratio (log scale 1%, 5%, 10%).
Y-axis: F1_susp with shaded ±std band.
Lines: BehaView, MLP, XGBoost, LightGBM, GAT, GCN, CARE-GNN.

Usage:
    python visualize/gen_rq3_labeff_figure.py
Output:
    _paper/figures/fig_rq3_labeff.pdf
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIG_OUT = os.path.join(ROOT, '_paper', 'figures', 'fig_rq3_labeff.pdf')


def load_behaview(ds):
    if ds == 'hofinet':
        df = pd.read_csv(os.path.join(RESULTS, 'rq3/behaview_hofinet.csv'))
        df['train_ratio'] = df['Model'].apply(lambda m: float(m.split('_r')[1].split('_s')[0]))
    else:
        df = pd.read_csv(os.path.join(RESULTS, f'rq3/behaview_{ds}.csv'))
        df['train_ratio'] = df['Model'].apply(lambda m: float(m.split('_r')[1].split('_s')[0]))
    return df[['train_ratio', 'f1_1']].assign(model='BehaView')


def load_supervised(ds):
    df = pd.read_csv(os.path.join(RESULTS, f'rq3/supervised_{ds}.csv'))
    rename = {'mlp': 'MLP', 'xgb': 'XGBoost', 'lgbm': 'LightGBM',
              'gat': 'GAT', 'gcn': 'GCN', 'caregnn': 'CARE-GNN'}
    df['model'] = df['model'].map(rename)
    return df[['train_ratio', 'f1_1', 'model']]


def aggregate(df):
    return df.groupby(['model', 'train_ratio'])['f1_1'].agg(['mean', 'std']).reset_index()


# Plot config — distinct colors and markers
STYLE = {
    'BehaView':  dict(color='#d62728', marker='o', lw=1.6, ms=5, zorder=10),
    'MLP':       dict(color='#7f7f7f', marker='s', lw=1.0, ms=4, ls='--'),
    'XGBoost':   dict(color='#1f77b4', marker='^', lw=1.0, ms=4),
    'LightGBM':  dict(color='#2ca02c', marker='v', lw=1.0, ms=4),
    'GAT':       dict(color='#ff7f0e', marker='D', lw=1.0, ms=3.5),
    'GCN':       dict(color='#9467bd', marker='P', lw=1.0, ms=4),
    'CARE-GNN':  dict(color='#8c564b', marker='X', lw=1.0, ms=4),
}
MODEL_ORDER = ['BehaView', 'XGBoost', 'LightGBM', 'MLP', 'GAT', 'GCN', 'CARE-GNN']

DATASETS = [
    ('hofinet',  'HOFINET ($\\rho{=}2.13\\%$)'),
    ('amlworld', 'AMLworld ($\\rho{=}1.23\\%$)'),
    ('amlnet',   'AMLNet ($\\rho{=}13.52\\%$)'),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    for ax, (ds, title) in zip(axes, DATASETS):
        bv = aggregate(load_behaview(ds))
        sup = aggregate(load_supervised(ds))
        all_data = pd.concat([bv, sup], ignore_index=True)

        for model in MODEL_ORDER:
            sub = all_data[all_data['model'] == model].sort_values('train_ratio')
            if len(sub) == 0:
                continue
            x = sub['train_ratio'].values * 100  # to percent
            m = sub['mean'].values
            s = sub['std'].values
            style = STYLE[model]
            ax.plot(x, m, label=model, **style)
            ax.fill_between(x, m - s, m + s, color=style['color'], alpha=0.12, lw=0)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Train label fraction')
        ax.set_xscale('log')
        ax.set_xticks([1, 5, 10])
        ax.set_xticklabels(['1%', '5%', '10%'])
        ax.invert_xaxis()  # decreasing labels left -> right (10% -> 1%)
        ax.grid(True, ls=':', alpha=0.5)

    axes[0].set_ylabel('$F1_{\\mathrm{susp}}$')

    # Y-axis scales: HOFINET/AMLNet medium-high, AMLworld very small
    axes[0].set_ylim(0, 0.75)
    axes[1].set_ylim(0, 0.12)
    axes[2].set_ylim(0, 0.75)

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # Ensure BehaView appears first
    order = sorted(range(len(labels)), key=lambda i: MODEL_ORDER.index(labels[i]) if labels[i] in MODEL_ORDER else 99)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), frameon=False, columnspacing=1.2)

    plt.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.20, wspace=0.22)
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight', dpi=300)
    # Also write a 300-DPI PNG sibling for slides / quick preview
    png_out = FIG_OUT.replace('.pdf', '.png')
    fig.savefig(png_out, bbox_inches='tight', dpi=300)
    print(f'Saved: {FIG_OUT}')
    print(f'Saved: {png_out}')


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
    main()

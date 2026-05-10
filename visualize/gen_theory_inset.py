"""Theory-Empirical scatter (Option B for Figure 3, redesigned).

Categorical x-axis: 3 datasets evenly spaced for clear cluster separation.
Y-axis: ΔF1 = F1(c) - F1(a), per encoder.
Bottom annotation: theoretical context (all η_h > 0.55 threshold).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

OUT = '/home/work/kftc_model/KA-003-FraudCenGCL/_paper/figures'
os.makedirs(OUT, exist_ok=True)

# ============ Data ============
DATASET_ORDER = ['HOFINET', 'AMLNet', 'AMLworld']
DATASET_ETA = {'HOFINET': 0.851, 'AMLNet': 0.878, 'AMLworld': 0.941}
DATASET_COLOR = {
    'HOFINET':  '#2E86AB',
    'AMLNet':   '#2A9D8F',
    'AMLworld': '#E07A5F',
}

# Encoder family classification
BN_FAMILIES = {
    'gbt': 'per-layer BN', 'dgi_bn': 'per-layer BN',
    'mvgrl_bn': 'per-layer BN', 'grace_bn': 'per-layer BN',
    'gin': 'per-layer BN',
    'bgrl': 'final BN',
    'dgi': 'BN-free', 'mvgrl': 'BN-free',
    'grace': 'BN-free', 'gca': 'BN-free',
}
FAMILY_MARKER = {'per-layer BN': 'o', 'final BN': 's', 'BN-free': '^'}

def parse_csv(path, ds_prefix):
    df = pd.read_csv(path)
    pat = rf'{ds_prefix}_(\w+?)_([abcd])_'
    parsed = df['Model'].str.extract(pat)
    df['enc'] = parsed[0]
    df['setting'] = parsed[1]
    df = df.dropna(subset=['enc', 'setting'])
    g = df.groupby(['enc','setting'])['f1_1'].agg(['mean','std']).unstack(level=-1)
    means, stds = g['mean'], g['std']
    rows = []
    for enc in means.index:
        if 'a' in means.columns and 'c' in means.columns:
            delta = means.at[enc,'c'] - means.at[enc,'a']
            err = (stds.at[enc,'a']**2 + stds.at[enc,'c']**2)**0.5
            rows.append({'enc': enc, 'delta': delta, 'err': err})
    return pd.DataFrame(rows)

ds_data = {
    'HOFINET':  parse_csv('results/exp_results_hofinet_ab.csv', 'hof'),
    'AMLworld': parse_csv('results/exp_results_amlworld.csv', 'aml'),
    'AMLNet':   parse_csv('results/exp_results_amlnet.csv', 'amlnet'),
}

# ============ Plot ============
fig, ax = plt.subplots(figsize=(5.4, 3.6))

C_ZERO  = '#6B7280'
C_FLIP  = '#C73E1D'
C_BG    = '#F9FAFB'

# Zero line
ax.axhline(0, color=C_ZERO, linewidth=0.9, zorder=1)
ax.text(2.55, 0.005, 'no change ($\\Delta F1=0$)', fontsize=9, color=C_ZERO,
        ha='right', va='bottom', style='italic')

# Plot data
np.random.seed(0)
ds_x_centers = {ds: i for i, ds in enumerate(DATASET_ORDER)}
JITTER_W = 0.30

plotted_families = set()
for ds_name in DATASET_ORDER:
    x_center = ds_x_centers[ds_name]
    ds_df = ds_data[ds_name]
    color = DATASET_COLOR[ds_name]
    n = len(ds_df)
    # Even spread within cluster
    xs = x_center + np.linspace(-JITTER_W, JITTER_W, n)
    # Sort by encoder family to group same-marker points
    family_idx = {'per-layer BN': 0, 'final BN': 1, 'BN-free': 2}
    ds_df = ds_df.copy()
    ds_df['fam_order'] = ds_df['enc'].map(lambda e: family_idx[BN_FAMILIES[e]])
    ds_df = ds_df.sort_values(['fam_order', 'enc']).reset_index(drop=True)
    xs = x_center + np.linspace(-JITTER_W, JITTER_W, n)

    for i, row in ds_df.iterrows():
        enc = row['enc']
        family = BN_FAMILIES.get(enc, 'BN-free')
        marker = FAMILY_MARKER[family]
        ax.errorbar(xs[i], row['delta'], yerr=row['err'],
                    fmt=marker, markersize=8, color=color,
                    markeredgecolor='white', markeredgewidth=1.0,
                    capsize=2.5, ecolor=color, alpha=0.85, zorder=4)

# X-axis: dataset names with η_h sub-labels
ax.set_xticks(list(ds_x_centers.values()))
labels = [f'{ds}\n($\\eta_h{{=}}{DATASET_ETA[ds]:.2f}$)' for ds in DATASET_ORDER]
ax.set_xticklabels(labels, fontsize=11.5)

# Color the dataset names on x-axis
for tick, ds in zip(ax.get_xticklabels(), DATASET_ORDER):
    tick.set_color(DATASET_COLOR[ds])

# Theoretical context (top right)
ax.text(0.98, 0.97,
        'All $\\eta_h > \\eta_h^*\\!=\\!0.55$ (Thm~5 flip threshold)',
        transform=ax.transAxes, fontsize=10, color=C_FLIP,
        ha='right', va='top', fontweight='bold', style='italic',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEE2E2',
                  edgecolor=C_FLIP, linewidth=0.9, alpha=0.9))

# Marker family legend
legend_handles = []
for fam, m in FAMILY_MARKER.items():
    legend_handles.append(plt.Line2D([], [], marker=m, linestyle='None',
                                      color='#37474F', markersize=8,
                                      markeredgecolor='white', label=fam))
ax.legend(handles=legend_handles, loc='lower right',
          fontsize=10, framealpha=0.95, edgecolor='#CCCCCC',
          handletextpad=0.4, title='Encoder family', title_fontsize=10)

ax.set_ylabel('$\\Delta F1_{\\mathrm{susp}}$ = $F1(c) - F1(a)$', fontsize=13)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.20, 0.10)
ax.grid(axis='y', alpha=0.18, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(OUT, 'fig_theory_inset')
plt.savefig(out + '.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out + '.png', bbox_inches='tight', dpi=300)
plt.close()
print('Saved:', out + '.pdf')

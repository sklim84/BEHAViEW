"""Theory-Empirical scatter (Option B for Figure 3).

Connects Theorem 5 (mean-pool signal flip threshold) to empirical RQ2
finding (setting (c) F1_susp drop):
- X axis: measured ego-neighborhood heterophily η_h per dataset
- Y axis: ΔF1 = F1(c) - F1(a) per encoder
- Markers: each (dataset, encoder) pair
- Vertical dashed line: theoretical flip threshold η_h* = 0.55 (k=10)
- Horizontal line at y=0: no change
- Background: red shading for flip region (η_h > 0.55)
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
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

OUT = '/home/work/kftc_model/KA-003-FraudCenGCL/_paper/figures'
os.makedirs(OUT, exist_ok=True)

# Measured η_h per dataset (from Table 2 / app:theory_verify)
DATASET_ETA = {
    'HOFINET':  0.851,
    'AMLNet':   0.878,
    'AMLworld': 0.941,
}
DATASET_COLOR = {
    'HOFINET':  '#2E86AB',
    'AMLNet':   '#2A9D8F',
    'AMLworld': '#E07A5F',
}

# Encoder family classification
BN_FAMILIES = {
    'gbt':       'per-layer BN',
    'dgi_bn':    'per-layer BN',
    'mvgrl_bn':  'per-layer BN',
    'grace_bn':  'per-layer BN',
    'gin':       'per-layer BN',
    'bgrl':      'final BN',
    'dgi':       'BN-free',
    'mvgrl':     'BN-free',
    'grace':     'BN-free',
    'gca':       'BN-free',
}
FAMILY_MARKER = {
    'per-layer BN': 'o',
    'final BN':     's',
    'BN-free':      '^',
}

# Load data
def parse_csv(path, ds_prefix):
    df = pd.read_csv(path)
    pat = rf'{ds_prefix}_(\w+?)_([abcd])_'
    parsed = df['Model'].str.extract(pat)
    df['enc'] = parsed[0]
    df['setting'] = parsed[1]
    df = df.dropna(subset=['enc', 'setting'])
    g = df.groupby(['enc','setting'])['f1_1'].agg(['mean','std']).unstack(level=-1)
    means = g['mean']
    stds = g['std']
    rows = []
    for enc in means.index:
        if 'a' in means.columns and 'c' in means.columns:
            f1_a = means.at[enc, 'a']
            f1_c = means.at[enc, 'c']
            std_a = stds.at[enc, 'a'] if 'a' in stds.columns else 0
            std_c = stds.at[enc, 'c'] if 'c' in stds.columns else 0
            delta = f1_c - f1_a
            err = (std_a**2 + std_c**2)**0.5
            rows.append({'enc': enc, 'delta': delta, 'err': err})
    return pd.DataFrame(rows)

datasets_data = {
    'HOFINET':  parse_csv('results/exp_results_hofinet_ab.csv', 'hof'),
    'AMLworld': parse_csv('results/exp_results_amlworld.csv', 'aml'),
    'AMLNet':   parse_csv('results/exp_results_amlnet.csv', 'amlnet'),
}

# ============ Plot ============
C_FLIP     = '#C73E1D'
C_FLIP_BG  = '#FEE2E2'
C_THRESH   = '#9CA3AF'
C_TEXT     = '#1F2937'

fig, ax = plt.subplots(figsize=(5.0, 3.4))

# Flip region background (η_h > 0.55) — shading is self-explanatory
ax.axvspan(0.55, 1.0, color=C_FLIP_BG, alpha=0.4, zorder=0)

# Zero line (no change)
ax.axhline(0, color='#6B7280', linewidth=0.8, zorder=1)

# Flip threshold vertical line + compact annotation
ax.axvline(0.55, color=C_FLIP, linewidth=1.4, linestyle='--', zorder=2)
ax.text(0.555, 0.09, '$\\eta_h^* = 0.55$',
        fontsize=10, color=C_FLIP, fontweight='bold', va='top', ha='left')

# Plot each (dataset, encoder) point
np.random.seed(0)
plotted_families = set()
for ds_name, ds_df in datasets_data.items():
    eta = DATASET_ETA[ds_name]
    color = DATASET_COLOR[ds_name]
    for _, row in ds_df.iterrows():
        enc = row['enc']
        family = BN_FAMILIES.get(enc, 'BN-free')
        marker = FAMILY_MARKER[family]
        # Small jitter on x to avoid overlap (within each dataset cluster)
        x_jitter = eta + np.random.uniform(-0.006, 0.006)
        ax.errorbar(x_jitter, row['delta'], yerr=row['err'],
                    fmt=marker, markersize=8, color=color,
                    markeredgecolor='white', markeredgewidth=1.0,
                    capsize=2.5, ecolor=color, alpha=0.85, zorder=4)

# Dataset name annotations: short labels only (η_h is implicit from x-axis).
# Stagger HOFINET / AMLNet vertically since their x positions are close.
ds_label_positions = {
    'HOFINET':  (0.851, 0.090),
    'AMLNet':   (0.878, 0.045),
    'AMLworld': (0.941, 0.090),
}
for ds_name, (lx, ly) in ds_label_positions.items():
    color = DATASET_COLOR[ds_name]
    ax.text(lx, ly, ds_name, fontsize=10, color=color,
            ha='center', va='top', fontweight='bold')

# Build legend (encoder family by marker)
legend_handles = []
for fam, m in FAMILY_MARKER.items():
    legend_handles.append(plt.Line2D([], [], marker=m, linestyle='None',
                                      color='#37474F', markersize=8,
                                      markeredgecolor='white', label=fam))
ax.legend(handles=legend_handles, loc='lower left',
          fontsize=10, framealpha=0.95, edgecolor='#CCCCCC', handletextpad=0.4)

ax.set_xlabel('Ego-neighborhood heterophily $\\eta_h$', fontsize=13)
ax.set_ylabel('$\\Delta F1_{\\mathrm{susp}}$ = $F1(c) - F1(a)$', fontsize=13)
ax.set_xlim(0.50, 1.00)
ax.set_ylim(-0.20, 0.10)
ax.grid(alpha=0.18, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(OUT, 'fig_theory_inset')
plt.savefig(out + '.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out + '.png', bbox_inches='tight', dpi=300)
plt.close()
print('Saved:', out + '.pdf')

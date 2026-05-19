"""
논문용 Figure 생성 스크립트.
사용법: python visualize/gen_paper_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIRS = ['_paper/figures', 'results/rq2/figures']
for _d in OUT_DIRS:
    os.makedirs(_d, exist_ok=True)
OUT_DIR = OUT_DIRS[0]   # primary path (paper); loop OUT_DIRS to mirror on save

def _save_all(path, **kw):
    """Save the current figure to OUT_DIRS mirroring `path`'s basename."""
    import shutil
    plt.savefig(path, **kw)
    base = os.path.basename(path)
    for d in OUT_DIRS[1:]:
        os.makedirs(d, exist_ok=True)
        mirror = os.path.join(d, base)
        shutil.copy(path, mirror)
        png = path.replace('.pdf', '.png')
        if os.path.exists(png) and path != png:
            shutil.copy(png, mirror.replace('.pdf', '.png'))

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})


def fig2_ablation_matrix():
    """Fig.2 (paper Fig.3): (a)(b)(c)(d) 2x2 ablation matrix heatmap.

    Cells = mean F1_susp over 4 per-layer BN encoders (GBT, DGI+BN, MVGRL+BN,
    GRACE+BN), 4 seeds, on the corrected 80% holdout (Gate 1 sweep).
    """
    import pandas as pd
    df = pd.read_csv('results/rq1/main_sweep.csv')
    parsed = []
    for n in df['Model']:
        parts = n.split('_')
        parsed.append((('_'.join(parts[1:-2])), parts[-2]))
    df['encoder'] = [p[0] for p in parsed]
    df['setting'] = [p[1] for p in parsed]
    bn_per_layer = ['gbt', 'dgi_bn', 'mvgrl_bn', 'grace_bn']
    means = {s: df[(df['encoder'].isin(bn_per_layer)) & (df['setting'] == s)]['f1_1'].mean()
             for s in 'abcd'}
    data = np.array([
        [means['a'], means['b']],   # node-level row
        [means['c'], means['d']],   # subgraph-pool row
    ])
    base = means['a']
    pct = np.array([
        ['baseline', f'+{(means["b"]/base - 1)*100:.0f}%'],
        [f'{(means["c"]/base - 1)*100:.0f}%', f'+{(means["d"]/base - 1)*100:.0f}%'],
    ])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(data, cmap='GnBu', vmin=0.1, vmax=0.75, aspect='equal')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Augmentation\nView', 'Behavioral\nk-NN View'], fontsize=11, ha='center')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Node-Level', 'Subgraph\nPooling'], fontsize=11)
    ax.set_xlabel('View Construction', fontsize=13, labelpad=8)
    ax.set_ylabel('Contrastive Level', fontsize=13, labelpad=8)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, pad=6)

    for i in range(2):
        for j in range(2):
            setting = [['(a)', '(b)'], ['(c)', '(d)']][i][j]
            color = 'white' if data[i, j] > 0.45 else '#333333'
            star = ' ★' if (i == 1 and j == 1) else ''
            ax.text(j, i - 0.12, f'{setting}{star}',
                    ha='center', va='center', color=color, fontsize=12, fontweight='bold')
            ax.text(j, i + 0.08, f'{data[i,j]:.3f}',
                    ha='center', va='center', color=color, fontsize=14, fontweight='bold')
            ax.text(j, i + 0.3, pct[i, j],
                    ha='center', va='center', color=color, fontsize=9,
                    fontstyle='italic', alpha=0.85)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label('$F1_{susp}$', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_rq2_ablation.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    import shutil; shutil.copy(path, os.path.join('results/rq2/figures', os.path.basename(path))); shutil.copy(path.replace('.pdf', '.png'), os.path.join('results/rq2/figures', os.path.basename(path).replace('.pdf', '.png')))
    plt.close()
    print(f'Saved: {path}')


def fig3_susp_connectivity():
    """Fig.3: Suspicious connectivity comparison (S-S/S-B ratio)."""
    graphs = ['Transaction\nGraph', 'Structural\nk-NN', 'Behavioral\nk-NN']
    ss = [257384, 16416, 61987]
    sb = [1468962, 158973, 87555]
    ratio = [sb[i]/ss[i] for i in range(3)]
    homophily = [0.690, 0.965, 0.981]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: S-S/S-B ratio (lower is better for suspicious detection)
    ax = axes[0]
    colors = ['#d9534f', '#f0ad4e', '#5cb85c']
    bars = ax.bar(graphs, ratio, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Susp-Benign / Susp-Susp Ratio', fontsize=11)
    ax.set_title('(a) Suspicious Neighbor Dilution', fontsize=12, fontweight='bold')
    for bar, r in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'1:{r:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 12)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Balanced (1:1)')
    ax.legend()

    # Right: Homophily vs F1_susp
    ax = axes[1]
    f1 = [0.351, 0.333, 0.567]  # BGRL results for each graph type
    scatter_colors = colors
    for i, (h, f, g) in enumerate(zip(homophily, f1, graphs)):
        ax.scatter(h, f, c=scatter_colors[i], s=200, edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(g.replace('\n', ' '), (h, f), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)
    ax.set_xlabel('Edge Homophily', fontsize=11)
    ax.set_ylabel('F1_susp', fontsize=11)
    ax.set_title('(b) Homophily vs Detection Performance', fontsize=12, fontweight='bold')
    ax.set_xlim(0.65, 1.0)
    ax.set_ylim(0.2, 0.7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_fraud_connectivity.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved: {path}')


def fig4_bn_effect():
    """Fig.4: F1_susp trajectory across (a/b/c/d) settings, faceted by BN family.

    Replaces the prior 2-panel bar chart with a faceted line plot that exposes
    the encoder-conditional pattern: per-layer BN encoders saturate the (b)->(d)
    transition (subgraph pool marginal), final-only BN (BGRL) shows substantial
    amplification, and BN-free encoders gain in absolute floor terms.

    Reads from corrected sweep results (Gate 1):
        results/rq1/main_sweep.csv
    """
    import pandas as pd

    csv_path = 'results/rq1/main_sweep.csv'
    df = pd.read_csv(csv_path)
    parsed = []
    for name in df['Model']:
        parts = name.split('_')
        seed = int(parts[-1].lstrip('s'))
        setting = parts[-2]
        enc = '_'.join(parts[1:-2])
        parsed.append((enc, setting, seed))
    df['encoder'] = [p[0] for p in parsed]
    df['setting'] = [p[1] for p in parsed]

    enc_labels = {
        'gbt': 'GBT', 'dgi_bn': 'DGI+BN', 'mvgrl_bn': 'MVGRL+BN',
        'grace_bn': 'GRACE+BN', 'gin': 'GIN', 'bgrl': 'BGRL',
        'dgi': 'DGI', 'mvgrl': 'MVGRL', 'grace': 'GRACE', 'gca': 'GCA',
    }
    bn_aug_styles = {
        'gbt':       ('#1f4e79', 'o'),
        'dgi_bn':    ('#2e6da4', 's'),
        'mvgrl_bn':  ('#3978b3', 'D'),
        'grace_bn':  ('#5b9bd5', '^'),
        'gin':       ('#9bbedc', 'v'),
        'bgrl':      ('#2ca02c', 'P'),  # green: final-only BN (distinct from per-layer family)
    }
    bn_aug_order = ['gbt', 'dgi_bn', 'mvgrl_bn', 'grace_bn', 'gin', 'bgrl']
    bn_free_order = ['dgi', 'mvgrl', 'grace']
    bn_free_colors = ['#d62728', '#ff7f0e', '#bcbd22']

    settings = ['a', 'b', 'c', 'd']
    setting_labels = ['(a)\nbaseline', '(b)\n+view', '(c)\n+level', '(d)\n+both']
    x = np.arange(len(settings))

    def _plot_panel(ax, encoders, style_map, ylim, legend_loc, legend_ncol):
        for enc in encoders:
            ys = [df[(df['encoder'] == enc) & (df['setting'] == s)]['f1_1'].mean()
                  for s in settings]
            errs = [df[(df['encoder'] == enc) & (df['setting'] == s)]['f1_1'].std()
                    for s in settings]
            color, marker = style_map[enc] if isinstance(style_map, dict) else style_map[encoders.index(enc)]
            ax.errorbar(x, ys, yerr=errs, marker=marker if isinstance(style_map, dict) else 'o',
                        markersize=7, linewidth=1.8, capsize=2.5, color=color,
                        label=enc_labels[enc])
        ax.set_xticks(x)
        ax.set_xticklabels(setting_labels, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10, loc=legend_loc, framealpha=0.9, edgecolor='#CCCCCC',
                  ncol=legend_ncol)
        ax.set_ylabel('$F1_{susp}$', fontsize=13)

    PANEL_FIGSIZE = (4.2, 3.2)

    # Panel A: BN-augmented encoders
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    _plot_panel(ax, bn_aug_order, bn_aug_styles, (0.0, 0.75), 'lower right', 2)
    plt.tight_layout()
    path_a = os.path.join(OUT_DIR, 'fig_rq3_bn_a.pdf')
    plt.savefig(path_a, bbox_inches='tight', dpi=300)
    plt.savefig(path_a.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    import shutil; shutil.copy(path_a, os.path.join('results/rq2/figures', os.path.basename(path_a))); shutil.copy(path_a.replace('.pdf', '.png'), os.path.join('results/rq2/figures', os.path.basename(path_a).replace('.pdf', '.png')))
    plt.close()
    print(f'Saved: {path_a}')

    # Panel B: BN-free encoders
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    bn_free_styles = {enc: (col, 'o') for enc, col in zip(bn_free_order, bn_free_colors)}
    _plot_panel(ax, bn_free_order, bn_free_styles, (0.0, 0.10), 'upper left', 1)
    plt.tight_layout()
    path_b = os.path.join(OUT_DIR, 'fig_rq3_bn_b.pdf')
    plt.savefig(path_b, bbox_inches='tight', dpi=300)
    plt.savefig(path_b.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    import shutil; shutil.copy(path_b, os.path.join('results/rq2/figures', os.path.basename(path_b))); shutil.copy(path_b.replace('.pdf', '.png'), os.path.join('results/rq2/figures', os.path.basename(path_b).replace('.pdf', '.png')))
    plt.close()
    print(f'Saved: {path_b}')


def fig4_bn_heatmap():
    """Alternative Fig.4: 4 settings x 9 encoders heatmap of F1_susp.

    Single-panel sequel to fig4_bn_effect. Same data, denser layout:
    each row is an ablation setting (a/b/c/d), each column is an encoder
    (ordered per-layer BN -> final-only BGRL -> BN-free), each cell is
    the 4-seed mean F1_susp. A vertical divider separates BN-equipped
    from BN-free encoders to make the BN dependence visible at a glance.

    Reads from results/rq1/main_sweep.csv.
    """
    import pandas as pd

    csv_path = 'results/rq1/main_sweep.csv'
    df = pd.read_csv(csv_path)
    parsed = []
    for name in df['Model']:
        parts = name.split('_')
        setting = parts[-2]
        enc = '_'.join(parts[1:-2])
        parsed.append((enc, setting))
    df['encoder'] = [p[0] for p in parsed]
    df['setting'] = [p[1] for p in parsed]

    enc_order = ['gbt', 'dgi_bn', 'mvgrl_bn', 'grace_bn', 'gin', 'bgrl',
                 'dgi', 'mvgrl', 'grace']
    enc_labels = {
        'gbt': 'GBT', 'dgi_bn': 'DGI+BN', 'mvgrl_bn': 'MVGRL+BN',
        'grace_bn': 'GRACE+BN', 'gin': 'GIN', 'bgrl': 'BGRL',
        'dgi': 'DGI', 'mvgrl': 'MVGRL', 'grace': 'GRACE',
    }
    settings = ['a', 'b', 'c', 'd']
    setting_labels = ['(a)\nbaseline', '(b)\n+view', '(c)\n+level', '(d)\n+both']

    # rows = settings, cols = encoders
    matrix = np.zeros((len(settings), len(enc_order)))
    for i, s in enumerate(settings):
        for j, enc in enumerate(enc_order):
            matrix[i, j] = df[(df['encoder'] == enc) & (df['setting'] == s)]['f1_1'].mean()

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(matrix, cmap='Blues', vmin=0.0, vmax=0.75, aspect='auto')

    ax.set_xticks(np.arange(len(enc_order)))
    ax.set_xticklabels([enc_labels[e] for e in enc_order], fontsize=13, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(settings)))
    ax.set_yticklabels(setting_labels, fontsize=13)

    for i in range(len(settings)):
        for j in range(len(enc_order)):
            val = matrix[i, j]
            # Blues: light (low) -> dark (high); white text on dark cells
            color = 'white' if val > 0.45 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=12)

    ax.axvline(x=5.5, color='red', linestyle='--', linewidth=1.5, alpha=0.85)
    ax.text(2.5, -0.75, 'BN-equipped', fontsize=12,
            ha='center', va='bottom', color='dimgray')
    ax.text(7.0, -0.75, 'BN-free', fontsize=12,
            ha='center', va='bottom', color='dimgray')

    ax.set_ylabel('Ablation setting', fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('$F1_{susp}$', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_rq3_bn_heatmap.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    import shutil
    shutil.copy(path, os.path.join('results/rq2/figures', os.path.basename(path)))
    shutil.copy(path.replace('.pdf', '.png'),
                os.path.join('results/rq2/figures', os.path.basename(path).replace('.pdf', '.png')))
    plt.close()
    print(f'Saved: {path}')


def fig5_setting_comparison():
    """Fig.5: (a)(b)(c)(d) comparison across BN/non-BN encoders."""
    settings = ['(a)\norg', '(b)\nbehav', '(c)\nsub', '(d)\nboth']

    # Average across GCN+BN encoders (GBT, DGI+BN, MVGRL+BN, GRACE+BN)
    bn_avg = [0.274, 0.677, 0.208, 0.682]
    # BGRL (final-only BN)
    bgrl = [0.315, 0.566, 0.254, 0.647]
    # GIN+BN
    gin = [0.149, 0.545, 0.121, 0.570]
    # Average across non-BN encoders (DGI, MVGRL, GRACE)
    nobn_avg = [0.043, 0.054, 0.047, 0.070]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(range(4), bn_avg, 'o-', color='#e74c3c', linewidth=2.5, markersize=10,
            label='GCN + per-layer BN (avg)', zorder=5)
    ax.plot(range(4), bgrl, 's--', color='#e67e22', linewidth=2, markersize=8,
            label='BGRL (final-only BN)')
    ax.plot(range(4), gin, '^--', color='#3498db', linewidth=2, markersize=8,
            label='GIN + BN')
    ax.plot(range(4), nobn_avg, 'D-', color='#95a5a6', linewidth=2, markersize=8,
            label='GCN without BN (avg)')

    ax.set_xticks(range(4))
    ax.set_xticklabels(settings, fontsize=11)
    ax.set_ylabel('F1_susp', fontsize=12)
    ax.set_xlabel('Setting', fontsize=12)
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Annotate key points
    ax.annotate('★ 0.682', xy=(3, 0.682), xytext=(3.2, 0.72),
                fontsize=10, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    ax.set_title('F1_susp across Settings and Encoder Types', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_setting_comparison.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved: {path}')


def fig_intro_homophily():
    """Intro Figure: ego-network homophily contrast.

    The visual language intentionally follows Figure 3's TikZ schematic:
    red suspicious nodes, blue benign nodes, thin black edges, and compact
    explanatory notes rather than chart-like styling.
    """
    import matplotlib.patches as mpatches

    C_SUSP = '#F4CCCC'      # close to TikZ red!25
    C_BENIGN = '#DDE8F7'    # close to TikZ blue!18
    C_AUX = '#FCE0BC'       # close to TikZ orange!25
    C_EDGE = '#5F6368'
    C_TEXT = '#202124'
    C_MUTED = '#4B5563'
    C_BOX = '#F4F4F5'
    C_BOX_EDGE = '#9CA3AF'

    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.65))
    fig.patch.set_facecolor('white')

    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 12
    radius = 0.98

    panels = [
        {
            'ax': axes[0],
            'title': 'Transaction view',
            'ratio': 'S-S:S-B = 1:5.7',
            'note': 'Suspicious signal is\ndiluted by benign neighbors',
            'labels': ['B', 'B', 'B', 'S', 'B', 'B', 'B', 'B', 'B', 'S', 'B', 'B'],
        },
        {
            'ax': axes[1],
            'title': 'Behavioral k-NN view',
            'ratio': 'S-S:S-B = 1:1.4',
            'note': 'Behaviorally similar accounts\nbecome neighbors',
            'labels': ['S', 'B', 'S', 'S', 'B', 'B', 'S', 'B', 'S', 'B', 'B', 'B'],
        },
    ]

    def draw_node(ax, x, y, label, size=185, is_ego=False):
        face = C_SUSP if label == 'S' else C_BENIGN
        if label == 'k':
            face = C_AUX
        ax.scatter([x], [y], s=size, marker='o', facecolors=face,
                   edgecolors='#555555', linewidths=0.55, zorder=4)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.4 if not is_ego else 9.0,
                color=C_TEXT, fontweight='bold', zorder=5)

    for panel in panels:
        ax = panel['ax']
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-2.2, 1.7)
        ax.set_aspect('equal')
        ax.axis('off')

        coords = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        for (x, y), label in zip(coords, panel['labels']):
            lw = 0.7 if label == 'S' else 0.45
            alpha = 0.78 if label == 'S' else 0.48
            ax.plot([0, x], [0, y], color=C_EDGE, linewidth=lw, alpha=alpha,
                    zorder=1, solid_capstyle='round')

        for (x, y), label in zip(coords, panel['labels']):
            draw_node(ax, x, y, label)

        draw_node(ax, 0, 0, 'S', size=290, is_ego=True)

        ax.text(0, 1.56, panel['title'], ha='center', va='center',
                fontsize=8.8, color=C_TEXT, fontweight='bold')
        ax.text(0, -1.42, panel['ratio'], ha='center', va='center',
                fontsize=7.8, color=C_TEXT, fontweight='bold')

        note_box = mpatches.FancyBboxPatch(
            (-1.25, -2.08), 2.5, 0.32,
            boxstyle='round,pad=0.06,rounding_size=0.04',
            facecolor=C_BOX, edgecolor=C_BOX_EDGE, linewidth=0.45, zorder=0)
        ax.add_patch(note_box)
        ax.text(0, -1.92, panel['note'], ha='center', va='center',
                fontsize=6.7, color=C_MUTED, linespacing=1.05)

    fig.patches.append(mpatches.FancyArrowPatch(
        (0.475, 0.53), (0.525, 0.53), transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=10, color=C_EDGE, linewidth=0.7,
        zorder=10))
    fig.text(0.5, 0.455, 'behavioral\nk-NN', ha='center', va='center',
             fontsize=6.4, color=C_MUTED, linespacing=0.95)

    legend_handles = [
        mpatches.Patch(facecolor=C_SUSP, edgecolor='#555555', label='S = suspicious'),
        mpatches.Patch(facecolor=C_BENIGN, edgecolor='#555555', label='B = benign'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=2, frameon=False, fontsize=7.2, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.2)

    plt.tight_layout(rect=[0, 0.07, 1, 1], w_pad=1.1)
    path = os.path.join(OUT_DIR, 'fig_intro_homophily.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def fig_dual_view_motivation():
    """Figure 3: dual-view motivation in the same style as Figure 1."""
    import matplotlib.patches as mpatches

    C_SUSP = '#F4CCCC'
    C_BENIGN = '#DDE8F7'
    C_AUX = '#FCE0BC'
    C_EDGE = '#5F6368'
    C_TEXT = '#202124'
    C_MUTED = '#4B5563'
    C_BOX = '#F4F4F5'
    C_BOX_EDGE = '#9CA3AF'

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    def draw_node(x, y, label, size=230):
        face = C_AUX if label == 'k' else (C_SUSP if label == 'S' else C_BENIGN)
        ax.scatter([x], [y], s=size, marker='o', facecolors=face,
                   edgecolors='#555555', linewidths=0.6, zorder=4)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9.8, color=C_TEXT, fontweight='bold', zorder=5)

    def draw_edge(x1, y1, x2, y2, strong=False, dashed=False):
        ax.plot([x1, x2], [y1, y2], color=C_EDGE,
                linewidth=0.8 if strong else 0.5,
                alpha=0.78 if strong else 0.5,
                linestyle='--' if dashed else '-',
                zorder=1, solid_capstyle='round')

    def note_box(cx, cy, text, width=3.8):
        box = mpatches.FancyBboxPatch(
            (cx - width / 2, cy - 0.32), width, 0.64,
            boxstyle='round,pad=0.06,rounding_size=0.04',
            facecolor=C_BOX, edgecolor=C_BOX_EDGE, linewidth=0.45, zorder=0)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=8.2, color=C_MUTED, linespacing=1.05)

    left_c = (2.3, 4.0)
    right_c = (7.7, 4.0)
    r = 1.0
    left_nodes = [
        ('B', -0.9, 0.55), ('B', 0.0, 0.95), ('B', 0.9, 0.55),
        ('B', -0.9, -0.45), ('B', 0.9, -0.45), ('S', 0.0, -1.0),
    ]
    right_nodes = [
        ('S', -0.9, 0.55), ('S', 0.0, 0.95), ('S', 0.9, 0.55),
        ('S', -0.9, -0.45), ('B', 0.9, -0.45), ('k', 0.0, -1.0),
    ]

    for cx, cy, title, nodes, note in [
        (*left_c, 'Transaction view', left_nodes,
         'Suspicious signal is\ndiluted by benign neighbors'),
        (*right_c, 'Behavioral k-NN view', right_nodes,
         'Behaviorally similar accounts\nbecome neighbors'),
    ]:
        ax.text(cx, 5.78, title, ha='center', va='center',
                fontsize=10.8, color=C_TEXT, fontweight='bold')

        draw_node(cx, cy, 'S', size=330)
        for label, dx, dy in nodes:
            x, y = cx + r * dx, cy + r * dy
            draw_edge(cx, cy, x, y, strong=(label == 'S'), dashed=(label == 'B' and dx > 0.8))
            draw_node(x, y, label, size=210)

        note_box(cx, 2.25, note, width=3.75)

    bridge = mpatches.FancyBboxPatch(
        (1.9, 0.55), 6.2, 0.62,
        boxstyle='round,pad=0.06,rounding_size=0.04',
        facecolor=C_BOX, edgecolor=C_BOX_EDGE, linewidth=0.45, zorder=0)
    ax.add_patch(bridge)
    ax.text(5.0, 0.83, 'Align the same account across two views',
            ha='center', va='center', fontsize=8.0, color=C_TEXT, fontweight='bold')
    ax.text(5.0, 0.25, 'Transaction structure + behavioral similarity',
            ha='center', va='center', fontsize=7.8, color=C_MUTED)

    for x0, x1 in [(2.6, 3.35), (7.4, 6.65)]:
        ax.annotate('', xy=(x1, 1.2), xytext=(x0, 1.88),
                    arrowprops=dict(arrowstyle='-|>', color=C_EDGE, lw=0.7,
                                    mutation_scale=10, shrinkA=8, shrinkB=2))

    legend_handles = [
        mpatches.Patch(facecolor=C_SUSP, edgecolor='#555555', label='S = suspicious'),
        mpatches.Patch(facecolor=C_BENIGN, edgecolor='#555555', label='B = benign'),
        mpatches.Patch(facecolor=C_AUX, edgecolor='#555555', label='k = k-NN auxiliary'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.015),
               ncol=3, frameon=False, fontsize=7.8, handlelength=1.0,
               handletextpad=0.35, columnspacing=0.85)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(OUT_DIR, 'fig_dual_view_motivation.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def fig1_framework():
    """Fig.1: Simplified framework overview."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.7, 'Behavioral Subgraph Contrast Framework', ha='center',
            fontsize=14, fontweight='bold')

    # Input
    box = mpatches.FancyBboxPatch((3.5, 8.8), 3, 0.6, boxstyle="round,pad=0.1",
                                   facecolor='#ecf0f1', edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, 9.1, 'Transaction Data (x_behav)', ha='center', fontsize=10, fontweight='bold')

    # Arrow down split
    ax.annotate('', xy=(3, 8.3), xytext=(5, 8.8), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(7, 8.3), xytext=(5, 8.8), arrowprops=dict(arrowstyle='->', lw=1.5))

    # View 1: Transaction Graph
    box1 = mpatches.FancyBboxPatch((1, 7.5), 3.5, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#a8d8ea', edgecolor='black', linewidth=1.5)
    ax.add_patch(box1)
    ax.text(2.75, 7.95, 'View 1: Transaction Graph', ha='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 7.7, 'S-S/S-B =1:5.7', ha='center', fontsize=8, color='#c0392b')

    # View 2: Behavioral k-NN
    box2 = mpatches.FancyBboxPatch((5.5, 7.5), 3.5, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#ffd6a5', edgecolor='black', linewidth=1.5)
    ax.add_patch(box2)
    ax.text(7.25, 7.95, 'View 2: Behavioral k-NN', ha='center', fontsize=9, fontweight='bold')
    ax.text(7.25, 7.7, 'S-S/S-B =1:1.4', ha='center', fontsize=8, color='#27ae60')

    # k-NN build annotation
    ax.text(8.5, 8.5, 'cosine sim\n→ top-k', ha='center', fontsize=8, style='italic', color='#7f8c8d')

    # Arrows to encoder
    ax.annotate('', xy=(2.75, 6.5), xytext=(2.75, 7.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(7.25, 6.5), xytext=(7.25, 7.5), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Shared GNN Encoder
    box3 = mpatches.FancyBboxPatch((1.5, 5.8), 7, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#c8e6c9', edgecolor='black', linewidth=1.5)
    ax.add_patch(box3)
    ax.text(5, 6.25, 'Shared GNN Encoder (GCNConv + BatchNorm + PReLU + Dropout)',
            ha='center', fontsize=9, fontweight='bold')

    # Arrows to subgraph pooling
    ax.annotate('', xy=(2.75, 5.0), xytext=(2.75, 5.8), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(7.25, 5.0), xytext=(7.25, 5.8), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Subgraph Pooling
    box4a = mpatches.FancyBboxPatch((1, 4.3), 3.5, 0.7, boxstyle="round,pad=0.1",
                                     facecolor='#e1bee7', edgecolor='black', linewidth=1.5)
    ax.add_patch(box4a)
    ax.text(2.75, 4.7, 'Subgraph Pooling', ha='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 4.5, 'mean(self + neighbors)', ha='center', fontsize=8)

    box4b = mpatches.FancyBboxPatch((5.5, 4.3), 3.5, 0.7, boxstyle="round,pad=0.1",
                                     facecolor='#e1bee7', edgecolor='black', linewidth=1.5)
    ax.add_patch(box4b)
    ax.text(7.25, 4.7, 'Subgraph Pooling', ha='center', fontsize=9, fontweight='bold')
    ax.text(7.25, 4.5, 'mean(self + neighbors)', ha='center', fontsize=8)

    # Arrows to loss
    ax.annotate('', xy=(5, 3.5), xytext=(2.75, 4.3), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(5, 3.5), xytext=(7.25, 4.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Bootstrap Loss
    box5 = mpatches.FancyBboxPatch((2.5, 2.8), 5, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#ffcdd2', edgecolor='black', linewidth=1.5)
    ax.add_patch(box5)
    ax.text(5, 3.25, 'Bootstrap L2L Loss (BYOL-style)', ha='center', fontsize=9, fontweight='bold')
    ax.text(5, 3.0, 'online pred(s₁) ↔ target proj(s₂)', ha='center', fontsize=8)

    # Arrow to evaluation
    ax.annotate('', xy=(5, 2.0), xytext=(5, 2.8), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Evaluation
    box6 = mpatches.FancyBboxPatch((2.5, 1.3), 5, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#fff9c4', edgecolor='black', linewidth=1.5)
    ax.add_patch(box6)
    ax.text(5, 1.75, 'Frozen [s₁ ∥ s₂] → LogReg Evaluation', ha='center', fontsize=9, fontweight='bold')
    ax.text(5, 1.5, 'F1_susp = 0.682 (GBT encoder, HOFINET)', ha='center', fontsize=8, color='#e74c3c')

    # Axis labels for two dimensions
    ax.annotate('', xy=(0.3, 4.65), xytext=(0.3, 6.15),
                arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=2))
    ax.text(0.15, 5.4, 'Level', ha='center', fontsize=9, fontweight='bold',
            rotation=90, color='#2c3e50')

    ax.annotate('', xy=(1.5, 8.55), xytext=(8.5, 8.55),
                arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=2))
    ax.text(5, 8.65, 'View Construction', ha='center', fontsize=9, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_framework.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved: {path}')


def fig_rq3_baseline():
    """RQ3: Self-supervised BehaView vs 11 supervised baselines across 3 datasets.

    Values mirror Table tab:rq3 exactly (4-seed mean, std for errorbars).
    BehaView uses setting (d) with the strongest encoder per dataset
    (GBT on HOFINET/AMLNet, BGRL on AMLworld).
    """
    datasets = [
        ('HOFINET',  r'HOFINET ($\rho{=}2.13\%$)',  0.85),
        ('AMLworld', r'AMLworld ($\rho{=}1.23\%$)', 0.10),
        ('AMLNet',   r'AMLNet ($\rho{=}13.52\%$)',  0.85),
    ]

    # (model, category, HOFINET[mean,std], AMLworld[mean,std], AMLNet[mean,std])
    rows = [
        ('XGBoost',   'Tabular',  (0.675, 0.001), (0.073, 0.002), (0.699, 0.007)),
        ('LightGBM',  'Tabular',  (0.670, 0.001), (0.070, 0.002), (0.687, 0.011)),
        ('MLP',       'Tabular',  (0.679, 0.002), (0.048, 0.002), (0.714, 0.005)),
        ('GCN',       'Generic',  (0.250, 0.007), (0.045, 0.001), (0.467, 0.006)),
        ('GAT',       'Generic',  (0.534, 0.024), (0.047, 0.001), (0.500, 0.047)),
        ('GraphSAGE', 'Generic',  (0.677, 0.002), (0.050, 0.001), (0.708, 0.007)),
        ('CARE-GNN',  'Fraud',    (0.108, 0.007), (0.041, 0.000), (0.443, 0.013)),
        ('PC-GNN',    'Fraud',    (0.245, 0.005), (0.044, 0.001), (0.458, 0.006)),
        ('BWGNN',     'Fraud',    (0.348, 0.022), (0.049, 0.001), (0.509, 0.021)),
        ('GAGA',      'Fraud',    (0.516, 0.006), (0.046, 0.004), (0.475, 0.031)),
        ('ConsisGAD', 'Fraud',    (0.234, 0.006), (0.045, 0.001), (0.456, 0.008)),
        ('BEHAViEW',  'Ours',     (0.673, 0.003), (0.066, 0.009), (0.679, 0.002)),
    ]

    cat_color = {
        'Tabular':  '#a8c5e6',
        'Generic':  '#bdbdbd',
        'Fraud':    '#fcae91',
        'Ours':     '#1f4e79',
    }
    cat_edge = {
        'Tabular':  '#5b9bd5',
        'Generic':  '#8c8c8c',
        'Fraud':    '#d94801',
        'Ours':     'black',
    }

    plot_rows = [rows[-1]] + rows[:-1]
    y = np.arange(len(plot_rows))

    fig, axes = plt.subplots(3, 1, figsize=(3.55, 7.85))
    for ax, (key, title, ymax) in zip(axes, datasets):
        col = {'HOFINET': 2, 'AMLworld': 3, 'AMLNet': 4}[key]
        means = [r[col][0] for r in plot_rows]
        stds  = [r[col][1] for r in plot_rows]
        colors = [cat_color[r[1]] for r in plot_rows]
        edges  = [cat_edge[r[1]]  for r in plot_rows]
        lws    = [1.8 if r[1] == 'Ours' else 0.7 for r in plot_rows]

        ax.barh(y, means, xerr=stds, color=colors, edgecolor=edges,
                linewidth=lws, capsize=2.0, height=0.62,
                error_kw={'elinewidth': 0.8})

        # Highlight BehaView value on top
        ours_idx = 0
        ax.text(means[ours_idx] + stds[ours_idx] + ymax * 0.015, ours_idx,
                f'{means[ours_idx]:.3f}', ha='left', va='center', fontsize=8.8,
                fontweight='bold', color='#1f4e79')

        # Reference line at BehaView value
        ax.axvline(means[ours_idx], color='#1f4e79', linestyle=':',
                   linewidth=0.9, alpha=0.65, zorder=0)

        for sep in [0.5, 3.5, 6.5]:
            ax.axhline(sep, color='#d9d9d9', linewidth=0.6, zorder=0)

        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in plot_rows], fontsize=8.6)
        ax.invert_yaxis()
        ax.set_xlim(0, ymax)
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=3)
        ax.tick_params(axis='x', labelsize=8.6, pad=1)
        ax.tick_params(axis='y', length=0, pad=2)
        ax.grid(axis='x', alpha=0.25, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Shared legend at top
    legend_handles = [
        mpatches.Patch(facecolor=cat_color['Tabular'], edgecolor=cat_edge['Tabular'],
                       label='Tabular'),
        mpatches.Patch(facecolor=cat_color['Generic'], edgecolor=cat_edge['Generic'],
                       label='Generic GNN'),
        mpatches.Patch(facecolor=cat_color['Fraud'],   edgecolor=cat_edge['Fraud'],
                       label='Fraud-GNN'),
        mpatches.Patch(facecolor=cat_color['Ours'],    edgecolor=cat_edge['Ours'],
                       linewidth=1.5, label='BEHAViEW'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.992), ncol=4, fontsize=7.2,
               framealpha=0.95, edgecolor='#cccccc',
               handlelength=0.9, handletextpad=0.3, columnspacing=0.55,
               borderpad=0.3)

    axes[-1].set_xlabel(r'$F1_{\mathrm{susp}}$', fontsize=9.6, labelpad=2)
    fig.subplots_adjust(left=0.31, right=0.99, top=0.895, bottom=0.055, hspace=0.34)
    path = os.path.join(OUT_DIR, 'fig_rq3_baseline.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    import shutil; shutil.copy(path, os.path.join('results/rq2/figures', os.path.basename(path))); shutil.copy(path.replace('.pdf', '.png'), os.path.join('results/rq2/figures', os.path.basename(path).replace('.pdf', '.png')))
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating paper figures...')
    fig_intro_homophily()
    fig_dual_view_motivation()
    fig1_framework()
    fig2_ablation_matrix()
    fig3_susp_connectivity()
    fig4_bn_effect()
    fig4_bn_heatmap()
    fig5_setting_comparison()
    fig_rq3_baseline()
    print(f'\nAll figures saved to {OUT_DIR}/')

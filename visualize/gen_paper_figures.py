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

OUT_DIR = 'visualize/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})


def fig2_ablation_matrix():
    """Fig.2 (paper Fig.3): (a)(b)(c)(d) 2x2 ablation matrix heatmap.

    Cells = mean F1_susp over 4 per-layer BN encoders (GBT, DGI+BN, MVGRL+BN,
    GRACE+BN), 4 seeds, on the corrected 80% holdout (Gate 1 sweep).
    """
    import pandas as pd
    df = pd.read_csv('results/exp_results_hofinet_ab.csv')
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
    ax.set_xticklabels(['Augmentation\nView', 'Behavioral\nk-NN View'], fontsize=9.5, ha='center')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Node-Level', 'Subgraph\nPooling'], fontsize=9.5)
    ax.set_xlabel('View Construction', fontsize=11, labelpad=8)
    ax.set_ylabel('Contrastive Level', fontsize=11, labelpad=8)

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
        results/exp_results_hofinet_ab.csv
    """
    import pandas as pd

    csv_path = 'results/exp_results_hofinet_ab.csv'
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
    bn_free_order = ['dgi', 'mvgrl', 'grace', 'gca']
    bn_free_colors = ['#d62728', '#ff7f0e', '#bcbd22', '#9467bd']

    settings = ['a', 'b', 'c', 'd']
    setting_labels = ['(a)\nbaseline', '(b)\n+view', '(c)\n+level', '(d)\n+both']
    x = np.arange(len(settings))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2),
                              gridspec_kw={'width_ratios': [6, 4], 'wspace': 0.20})

    def _plot_panel(ax, encoders, style_map, ylim, title, legend_loc, legend_ncol):
        for enc in encoders:
            ys = [df[(df['encoder'] == enc) & (df['setting'] == s)]['f1_1'].mean()
                  for s in settings]
            errs = [df[(df['encoder'] == enc) & (df['setting'] == s)]['f1_1'].std()
                    for s in settings]
            color, marker = style_map[enc] if isinstance(style_map, dict) else style_map[encoders.index(enc)]
            ax.errorbar(x, ys, yerr=errs, marker=marker if isinstance(style_map, dict) else 'o',
                        markersize=6, linewidth=1.6, capsize=2.2, color=color,
                        label=enc_labels[enc])
        ax.set_xticks(x)
        ax.set_xticklabels(setting_labels, fontsize=9.5)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, loc=legend_loc, framealpha=0.9, edgecolor='#CCCCCC',
                  ncol=legend_ncol)

    _plot_panel(axes[0], bn_aug_order, bn_aug_styles,
                (0.0, 0.75), '(a) BN-augmented encoders',
                'lower right', 2)
    axes[0].set_ylabel('$F1_{susp}$', fontsize=11)

    bn_free_styles = {enc: (col, 'o') for enc, col in zip(bn_free_order, bn_free_colors)}
    _plot_panel(axes[1], bn_free_order, bn_free_styles,
                (0.0, 0.10), '(b) BN-free encoders',
                'upper left', 1)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_rq3_bn.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
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
    # Average across non-BN encoders (DGI, MVGRL, GRACE, GCA)
    nobn_avg = [0.046, 0.057, 0.048, 0.075]

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
    """Intro Figure: S-S/S-B ratio comparison — Transaction vs Behavioral k-NN."""
    categories = ['S-S', 'S-B']
    trans_vals = [257384, 1468962]  # Transaction graph
    knn_vals = [61987, 87555]       # Behavioral k-NN

    # Normalize to percentage for clearer comparison
    trans_pct = [v / sum(trans_vals) * 100 for v in trans_vals]
    knn_pct = [v / sum(knn_vals) * 100 for v in knn_vals]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4), sharey=True)

    colors = ['#e74c3c', '#3498db']  # red for S-S, blue for S-B
    bar_width = 0.55

    for ax, pcts, vals, title, ratio in [
        (axes[0], trans_pct, trans_vals, 'Transaction Graph', '1 : 5.7'),
        (axes[1], knn_pct, knn_vals, 'Behavioral k-NN', '1 : 1.4'),
    ]:
        bars = ax.bar(categories, pcts, color=colors, width=bar_width,
                      edgecolor='black', linewidth=0.6)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
        ax.set_ylim(0, 105)
        ax.tick_params(axis='both', labelsize=7)

        # Value labels
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

        # Ratio annotation
        ax.text(0.5, -0.22, f'S-S : S-B = {ratio}', transform=ax.transAxes,
                ha='center', fontsize=7, style='italic', color='#555555')

    axes[0].set_ylabel('Edge Proportion (%)', fontsize=8)

    plt.tight_layout(w_pad=1.0)
    path = os.path.join(OUT_DIR, 'fig_intro_homophily.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
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


if __name__ == '__main__':
    print('Generating paper figures...')
    fig_intro_homophily()
    fig1_framework()
    fig2_ablation_matrix()
    fig3_susp_connectivity()
    fig4_bn_effect()
    fig5_setting_comparison()
    print(f'\nAll figures saved to {OUT_DIR}/')

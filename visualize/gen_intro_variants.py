"""
Intro figure variants for style comparison.
Usage: python visualize/gen_intro_variants.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = 'visualize/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Common data
SS_TRANS, SB_TRANS = 257384, 1468962
SS_KNN, SB_KNN = 61987, 87555

# Color palette (academic, colorblind-friendly)
C_SUSP = '#D64045'   # muted red
C_BENIGN = '#467599'  # steel blue
C_SUSP_L = '#F0A0A2' # light red
C_BENIGN_L = '#A3C4D9' # light blue
C_BG = '#FAFAFA'
C_GRAY = '#888888'
C_DARK = '#2C3E50'
C_ARROW = '#E67E22'


def variant_a_donut():
    """Style A: Side-by-side donut charts."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0))
    fig.patch.set_facecolor('white')

    for ax, ss, sb, title, ratio_str in [
        (axes[0], SS_TRANS, SB_TRANS, 'Transaction Graph', '1 : 5.7'),
        (axes[1], SS_KNN, SB_KNN, 'Behavioral k-NN', '1 : 1.4'),
    ]:
        total = ss + sb
        ss_pct = ss / total * 100
        sb_pct = sb / total * 100
        sizes = [ss_pct, sb_pct]
        colors = [C_SUSP, C_BENIGN]
        explode = (0.03, 0.0)

        wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.38, edgecolor='white', linewidth=1.5),
                           explode=explode)

        # Center text
        ax.text(0, 0.05, f'S-S : S-B', ha='center', va='center',
                fontsize=6, color=C_GRAY)
        ax.text(0, -0.15, ratio_str, ha='center', va='center',
                fontsize=8, fontweight='bold', color=C_DARK)

        ax.set_title(title, fontsize=8, fontweight='bold', color=C_DARK, pad=6)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_SUSP, edgecolor='white', label='Suspicious–Suspicious'),
        mpatches.Patch(facecolor=C_BENIGN, edgecolor='white', label='Suspicious–Benign'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, 'intro_A_donut.pdf')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def variant_b_stacked():
    """Style B: Horizontal stacked bars with arrow annotation."""
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    labels = ['Transaction\nGraph', 'Behavioral\nk-NN']
    ss_pcts = [SS_TRANS/(SS_TRANS+SB_TRANS)*100, SS_KNN/(SS_KNN+SB_KNN)*100]
    sb_pcts = [SB_TRANS/(SS_TRANS+SB_TRANS)*100, SB_KNN/(SS_KNN+SB_KNN)*100]

    y = np.arange(len(labels))
    h = 0.45

    # Stacked bars
    bars_ss = ax.barh(y, ss_pcts, h, color=C_SUSP, edgecolor='white', linewidth=0.8, label='Suspicious–Suspicious', zorder=3)
    bars_sb = ax.barh(y, sb_pcts, h, left=ss_pcts, color=C_BENIGN, edgecolor='white', linewidth=0.8, label='Suspicious–Benign', zorder=3)

    # Percentage labels
    for i, (ss, sb) in enumerate(zip(ss_pcts, sb_pcts)):
        if ss > 8:
            ax.text(ss/2, i, f'{ss:.0f}%', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=4)
        ax.text(ss + sb/2, i, f'{sb:.0f}%', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=4)

    # Ratio annotations on the right
    ax.text(103, 0, '1:5.7', ha='left', va='center', fontsize=7.5, color=C_GRAY, style='italic')
    ax.text(103, 1, '1:1.4', ha='left', va='center', fontsize=7.5, color=C_DARK, fontweight='bold', style='italic')

    # Arrow between bars
    ax.annotate('', xy=(ss_pcts[1], 0.72), xytext=(ss_pcts[0], 0.28),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.8,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(24, 0.5, '×2.7', ha='center', va='center', fontsize=7,
            fontweight='bold', color=C_ARROW)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, fontweight='bold', color=C_DARK)
    ax.set_xlim(0, 115)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.invert_yaxis()

    ax.legend(fontsize=6, loc='lower right', frameon=False, ncol=1,
              bbox_to_anchor=(0.98, -0.02))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'intro_B_stacked.pdf')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def variant_c_ego():
    """Style C: Ego-network conceptual diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4))
    fig.patch.set_facecolor('white')

    np.random.seed(42)

    for ax, ss_ratio, title, ss_pct, sb_pct in [
        (axes[0], 0.15, 'Transaction Graph', 15, 85),
        (axes[1], 0.41, 'Behavioral k-NN', 41, 59),
    ]:
        ax.set_facecolor('white')
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect('equal')
        ax.axis('off')

        # Center suspicious node
        ax.scatter(0, 0, s=180, c=C_SUSP, edgecolors='white', linewidths=1.5, zorder=5)
        ax.text(0, 0, 'S', ha='center', va='center', fontsize=6, fontweight='bold',
                color='white', zorder=6)

        # Neighbor nodes
        n_neighbors = 12
        n_susp = int(n_neighbors * ss_ratio)
        n_benign = n_neighbors - n_susp

        angles = np.linspace(0, 2*np.pi, n_neighbors, endpoint=False)
        radius = 1.15
        is_susp = [True]*n_susp + [False]*n_benign
        np.random.shuffle(is_susp)

        for angle, susp in zip(angles, is_susp):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            color = C_SUSP if susp else C_BENIGN
            label = 'S' if susp else 'B'

            # Edge
            edge_color = C_SUSP_L if susp else C_BENIGN_L
            ax.plot([0, x], [0, y], color=edge_color, linewidth=1.2, zorder=1, alpha=0.7)

            # Node with letter
            ax.scatter(x, y, s=100, c=color, edgecolors='white', linewidths=1.0, zorder=4)
            ax.text(x, y, label, ha='center', va='center', fontsize=5, fontweight='bold',
                    color='white', zorder=5)

        ax.set_title(title, fontsize=8, fontweight='bold', color=C_DARK, pad=6)
        ax.text(0, -1.55, f'S-S {ss_pct}% : S-B {sb_pct}%', ha='center', va='top',
                fontsize=6.5, color=C_DARK, fontweight='bold')

    # Arrow between subplots
    fig.patches.append(mpatches.FancyArrowPatch(
        (0.48, 0.5), (0.52, 0.5), transform=fig.transFigure,
        arrowstyle='->', mutation_scale=12, color=C_ARROW, linewidth=2,
        zorder=10))

    plt.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.2)
    path = os.path.join(OUT_DIR, 'intro_C_ego.pdf')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def variant_d_combined():
    """Style D: Ego-network (top) + ratio bar (bottom) combined."""
    fig = plt.figure(figsize=(3.4, 3.2))
    fig.patch.set_facecolor('white')

    # Top row: ego networks
    ax1 = fig.add_axes([0.02, 0.40, 0.44, 0.55])
    ax2 = fig.add_axes([0.54, 0.40, 0.44, 0.55])
    # Bottom: ratio bar
    ax3 = fig.add_axes([0.08, 0.08, 0.84, 0.22])

    np.random.seed(42)

    for ax, ss_ratio, title in [
        (ax1, 0.15, 'Transaction Graph'),
        (ax2, 0.41, 'Behavioral k-NN'),
    ]:
        ax.set_facecolor('white')
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Center suspicious node
        ax.scatter(0, 0, s=200, c=C_SUSP, edgecolors='#333333', linewidths=1.2, zorder=5)
        ax.text(0, 0, '?', ha='center', va='center', fontsize=7, fontweight='bold',
                color='white', zorder=6)

        # Neighbor nodes
        n_neighbors = 10
        n_susp = round(n_neighbors * ss_ratio)
        n_benign = n_neighbors - n_susp
        angles = np.linspace(0, 2*np.pi, n_neighbors, endpoint=False) + np.pi/10
        radius = 1.1
        is_susp = [True]*n_susp + [False]*n_benign
        np.random.shuffle(is_susp)

        for angle, susp in zip(angles, is_susp):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            color = C_SUSP if susp else C_BENIGN
            edge_alpha = 0.8 if susp else 0.4
            ax.plot([0, x], [0, y], color=color, linewidth=1.0, zorder=1, alpha=edge_alpha)
            ax.scatter(x, y, s=90, c=color, edgecolors='#333333', linewidths=0.6, zorder=4)

        ax.set_title(title, fontsize=8, fontweight='bold', color=C_DARK, pad=4)

    # Arrow between ego networks
    fig.patches.append(mpatches.FancyArrowPatch(
        (0.46, 0.67), (0.54, 0.67), transform=fig.transFigure,
        arrowstyle='->', mutation_scale=14, color=C_ARROW, linewidth=2.5, zorder=10))

    # Bottom bar: S-S ratio comparison
    ax3.set_facecolor('white')
    categories = ['Transaction\nGraph', 'Behavioral\nk-NN']
    ss_pcts = [SS_TRANS/(SS_TRANS+SB_TRANS)*100, SS_KNN/(SS_KNN+SB_KNN)*100]
    sb_pcts = [100 - p for p in ss_pcts]

    y = np.arange(2)
    h = 0.5
    ax3.barh(y, ss_pcts, h, color=C_SUSP, edgecolor='white', linewidth=0.8, zorder=3)
    ax3.barh(y, sb_pcts, h, left=ss_pcts, color=C_BENIGN, edgecolor='white', linewidth=0.8, zorder=3)

    for i, (ss, sb) in enumerate(zip(ss_pcts, sb_pcts)):
        if ss > 10:
            ax3.text(ss/2, i, f'S-S {ss:.0f}%', ha='center', va='center',
                     fontsize=6, fontweight='bold', color='white', zorder=4)
        ax3.text(ss + sb/2, i, f'S-B {sb:.0f}%', ha='center', va='center',
                 fontsize=6, fontweight='bold', color='white', zorder=4)

    ax3.set_yticks(y)
    ax3.set_yticklabels(categories, fontsize=6.5, color=C_DARK)
    ax3.set_xticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.invert_yaxis()

    path = os.path.join(OUT_DIR, 'intro_D_combined.pdf')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating intro figure variants...')
    variant_a_donut()
    variant_b_stacked()
    variant_c_ego()
    variant_d_combined()
    print('Done!')

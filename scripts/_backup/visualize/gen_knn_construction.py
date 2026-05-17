"""Mini-figure: k-NN view construction pipeline."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig, ax = plt.subplots(figsize=(7.0, 1.6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

# Helper to draw a labeled box
def box(x, y, w, h, label, color='#ecf0f1', edge='black', fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                    facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize)

def arrow(x1, y1, x2, y2, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.4, color='#555555'))
    if label:
        ax.text((x1+x2)/2, max(y1, y2)+0.3, label, ha='center',
                fontsize=8, style='italic', color='#555555')

# Step 1: Behavioral features matrix
box(0.2, 1.0, 2.4, 2.0, '$\\mathbf{X}^{\\mathrm{behav}}$\n(20 vars)', '#a8d8ea')
ax.text(1.4, 0.5, '(1) Pure\nbehavioral', ha='center', fontsize=8, color='#555')

# Arrow 1
arrow(2.7, 2.0, 3.7, 2.0)

# Step 2: StandardScaler + L2-norm
box(3.7, 1.0, 2.6, 2.0, 'StandardScaler\n + L2-norm', '#ffd6a5')
ax.text(5.0, 0.5, '(2) Eq.~(1)', ha='center', fontsize=8, color='#555')

# Arrow 2
arrow(6.4, 2.0, 7.4, 2.0)

# Step 3: ball-tree k-NN search
box(7.4, 1.0, 2.6, 2.0, 'Ball-tree\nk-NN (k=10)', '#c8e6c9')
ax.text(8.7, 0.5, '(3) $O(N \\log N)$', ha='center', fontsize=8, color='#555')

# Arrow 3
arrow(10.1, 2.0, 11.1, 2.0)

# Step 4: G_knn
box(11.1, 0.7, 2.6, 2.6, '$\\mathcal{G}_{\\mathrm{knn}}$\nS-S/S-B\n1:5.7$\\to$1:1.4', '#f8bbd0')
ax.text(12.4, 0.2, '(4) Eq.~(2)', ha='center', fontsize=8, color='#555')

plt.tight_layout()
out = '/home/work/kftc_model/KA-003-FraudCenGCL/_paper/figures/fig_knn_construction'
plt.savefig(out + '.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out + '.png', bbox_inches='tight', dpi=300)
plt.close()
print('Saved:', out + '.pdf')

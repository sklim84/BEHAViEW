"""Theory inset: mean-pool signal-flip threshold visualization.

Single-panel design (Variant A simplification):
- Main curve: k=10 (paper's default) in bold teal
- Other k values (5, 20, 50) as faint context background
- Flip threshold (eta_h* = 0.55 for k=10) marked with red dashed line
- Three dataset markers showing all measured eta_h exceed the threshold
- Shaded "flip region" (eta_h > 0.55) where mean-pool signal sign reverses

Message: All AML datasets we use have eta_h > flip threshold,
explaining why setting (c) (transaction graph + mean pool) fails.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

OUT = '/home/work/kftc_model/KA-003-FraudCenGCL/_paper/figures'
os.makedirs(OUT, exist_ok=True)


def fig_theory_inset():
    """Single-panel: signal flip threshold + dataset markers."""
    C_MAIN     = '#1B5E7D'   # dark teal — k=10 (main)
    C_CTX      = '#9CA3AF'   # gray — other k context
    C_FLIP     = '#C73E1D'   # warm red — flip threshold
    C_FLIP_BG  = '#FEE2E2'   # light red — flip region shading
    C_DOT_HOF  = '#2E86AB'
    C_DOT_AML  = '#E07A5F'
    C_DOT_NET  = '#2A9D8F'
    C_TEXT     = '#1F2937'

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    eta = np.linspace(0, 1, 200)

    # Background: flip region (eta > 0.55 for k=10)
    ax.axvspan(0.55, 1.0, color=C_FLIP_BG, alpha=0.45, zorder=0)
    # Region label placed at bottom of red area (out of legend's way)
    ax.text(0.78, -0.97, 'signal-flip region (mean pool fails)',
            fontsize=8.5, color=C_FLIP, ha='center', va='bottom', style='italic',
            fontweight='bold')

    # Context curves (k=5, 20, 50) — faint
    for k in [5, 20, 50]:
        rho = (1 + k * (1 - 2*eta)) / (k + 1)
        ax.plot(eta, rho, color=C_CTX, linewidth=1.0, alpha=0.45,
                linestyle=':', zorder=1)
        ax.text(1.005, rho[-1], f'$k{{=}}{k}$', fontsize=7,
                color=C_CTX, va='center', ha='left')

    # Main curve: k=10
    k = 10
    rho_main = (1 + k * (1 - 2*eta)) / (k + 1)
    ax.plot(eta, rho_main, color=C_MAIN, linewidth=2.4, zorder=3,
            label='$k=10$ (paper default)')
    ax.text(1.005, rho_main[-1], '$k{=}10$', fontsize=8.5,
            color=C_MAIN, fontweight='bold', va='center', ha='left')

    # Zero line
    ax.axhline(0, color='#6B7280', linewidth=0.7, zorder=1)

    # Flip threshold line + label (positioned bottom-left to avoid region label)
    ax.axvline(0.55, color=C_FLIP, linewidth=1.4, linestyle='--', zorder=2)
    ax.annotate('flip threshold\n$\\eta_h^* = 1/2 + 1/(2k) = 0.55$',
                xy=(0.55, -0.30), xytext=(0.05, -0.55),
                fontsize=8.5, color=C_FLIP, fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='->', color=C_FLIP, lw=0.9, alpha=0.85))

    # Dataset markers (all on the k=10 curve)
    datasets = [
        ('HOFINET',  0.851, C_DOT_HOF, 'o'),
        ('AMLworld', 0.941, C_DOT_AML, 's'),
        ('AMLNet',   0.878, C_DOT_NET, '^'),
    ]
    for name, x, color, marker in datasets:
        y = (1 + 10*(1 - 2*x)) / 11
        ax.scatter([x], [y], s=120, marker=marker, color=color,
                   edgecolors='white', linewidths=1.4, zorder=6,
                   label=f'{name} ($\\eta_h{{=}}{x:.2f}$)')

    ax.set_xlabel('Ego-neighborhood heterophily $\\eta_h$', fontsize=12)
    ax.set_ylabel('Mean-pool signal $\\mathbb{E}[\\rho^{(\\mathrm{mean})}]$', fontsize=12)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-1, 1.1)
    ax.grid(alpha=0.18, zorder=0)
    # Legend at upper-right
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.96),
              fontsize=8.5, framealpha=0.95,
              edgecolor='#CCCCCC', handlelength=1.4, borderaxespad=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT, 'fig_theory_inset')
    plt.savefig(out + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(out + '.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved:', out + '.pdf')


if __name__ == '__main__':
    fig_theory_inset()

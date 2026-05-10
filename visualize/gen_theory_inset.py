"""Theory inset: mean-pool signal vs heterophily, with flip threshold.

Generates two separate PDFs (one per subfigure) so they can be embedded
via LaTeX subfigure with their own captions.
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


def panel_a_mean_pool_flip():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    eta = np.linspace(0, 1, 200)
    for k, color, ls in [(5, '#999999', ':'), (10, '#1f77b4', '-'),
                         (20, '#666666', '--'), (50, '#333333', '-.')]:
        rho = (1 + k * (1 - 2*eta)) / (k + 1)
        ax.plot(eta, rho, color=color, linestyle=ls, linewidth=1.6, label=f'$k={k}$')

    ax.axhline(0, color='red', linewidth=0.8, alpha=0.5)
    ax.axvline(0.55, color='red', linewidth=1.0, linestyle=':', alpha=0.7)
    ax.text(0.56, 0.85, '$\\eta_h^* = 0.55$\n(flip,\n$k{=}10$)', fontsize=8, color='red', va='top')

    for x, m, label, dy in [(0.851, 'o', 'HOFINET', -0.10),
                             (0.941, 's', 'AMLworld', 0.0),
                             (0.878, '^', 'AMLNet', +0.10)]:
        y = (1 + 10*(1 - 2*x)) / 11
        ax.scatter([x], [y], s=50, marker=m, color='#1f77b4',
                   zorder=5, edgecolors='black', linewidth=0.8)
        if label == 'AMLworld':
            ax.text(x + 0.04, y, label, fontsize=7, ha='left', va='center', color='#1f77b4')
        else:
            ax.text(x, y + dy, label, fontsize=7, ha='center', color='#1f77b4')

    ax.set_xlabel('Heterophily $\\eta_h$', fontsize=12)
    ax.set_ylabel('$\\mathbb{E}[\\rho^{(\\mathrm{mean})}]$', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT, 'fig_theory_inset_a')
    plt.savefig(out + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(out + '.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved:', out + '.pdf')


def panel_b_hap_vs_mean():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    eta = np.linspace(0, 1, 200)
    k, eps = 10, 0.2
    rho_mean = (1 + k * (1 - 2*eta)) / (k + 1)
    rho_hap = 1 - 4 * eps**2 * eta - eps**2

    gap = rho_hap - rho_mean
    ax.fill_between(eta, 0, gap, where=(gap > 0), alpha=0.25, color='#2ca02c', label='HAP advantage')
    ax.plot(eta, rho_mean, color='#d62728', linewidth=1.6, label='Mean pool', linestyle='--')
    ax.plot(eta, rho_hap, color='#2ca02c', linewidth=1.8, label='HAP (Thm 6 LB)')
    ax.axhline(0, color='gray', linewidth=0.6)
    ax.axvline(0.55, color='red', linewidth=0.8, linestyle=':', alpha=0.6)

    ax.set_xlabel('Heterophily $\\eta_h$', fontsize=12)
    ax.set_ylabel('Post-pool signal $\\mathbb{E}[\\rho]$', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1.1)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT, 'fig_theory_inset_b')
    plt.savefig(out + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(out + '.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved:', out + '.pdf')


if __name__ == '__main__':
    panel_a_mean_pool_flip()
    panel_b_hap_vs_mean()

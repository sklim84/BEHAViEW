"""Theory inset: mean-pool signal vs heterophily, with flip threshold."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

# Panel A: E[rho^mean] vs eta_h, multiple k values
ax = axes[0]
eta = np.linspace(0, 1, 200)
for k, color, ls in [(5, '#999999', ':'), (10, '#1f77b4', '-'), (20, '#666666', '--'), (50, '#333333', '-.')]:
    rho_mean = (1 + k * (1 - 2*eta)) / (k + 1)
    ax.plot(eta, rho_mean, color=color, linestyle=ls, linewidth=1.6, label=f'$k={k}$')

# Threshold line for k=10
ax.axhline(0, color='red', linewidth=0.8, alpha=0.5)
ax.axvline(0.55, color='red', linewidth=1.0, linestyle=':', alpha=0.7)
ax.text(0.56, 0.85, '$\\eta_h^* = 0.55$\n(flip,\n$k{=}10$)', fontsize=8, color='red', va='top')

# Markers for measured eta_h on three datasets
ax.scatter([0.851], [(1 + 10*(1 - 2*0.851)) / 11], s=50, marker='o', color='#1f77b4', zorder=5, edgecolors='black', linewidth=0.8)
ax.scatter([0.941], [(1 + 10*(1 - 2*0.941)) / 11], s=50, marker='s', color='#1f77b4', zorder=5, edgecolors='black', linewidth=0.8)
ax.scatter([0.878], [(1 + 10*(1 - 2*0.878)) / 11], s=50, marker='^', color='#1f77b4', zorder=5, edgecolors='black', linewidth=0.8)
ax.text(0.851, (1 + 10*(1 - 2*0.851)) / 11 - 0.10, 'HOFINET', fontsize=7, ha='center', color='#1f77b4')
ax.text(0.941+0.04, (1 + 10*(1 - 2*0.941)) / 11, 'AMLworld', fontsize=7, ha='left', va='center', color='#1f77b4')
ax.text(0.878-0.02, (1 + 10*(1 - 2*0.878)) / 11 + 0.10, 'AMLNet', fontsize=7, ha='center', color='#1f77b4')

ax.set_xlabel('Heterophily $\\eta_h$', fontsize=10)
ax.set_ylabel('$\\mathbb{E}[\\rho^{(\\mathrm{mean})}]$', fontsize=10)
ax.set_title('(a) Mean-pool signal flip (Thm~5)', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1.05)
ax.grid(alpha=0.25)
ax.legend(loc='lower left', fontsize=8, framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: HAP vs mean preservation (E[rho^hap] - E[rho^mean]) vs eta_h
ax = axes[1]
k = 10
eps = 0.2
rho_mean = (1 + k * (1 - 2*eta)) / (k + 1)
# HAP lower bound: 1 - O(eps^2) - 4*eps^2 * eta (approx)
rho_hap = 1 - 4 * eps**2 * eta - eps**2  # approximation
gap = rho_hap - rho_mean
ax.fill_between(eta, 0, gap, where=(gap > 0), alpha=0.25, color='#2ca02c', label='HAP advantage')
ax.plot(eta, rho_mean, color='#d62728', linewidth=1.6, label='Mean pool', linestyle='--')
ax.plot(eta, rho_hap, color='#2ca02c', linewidth=1.8, label='HAP (Thm 6 LB)')
ax.axhline(0, color='gray', linewidth=0.6)
ax.axvline(0.55, color='red', linewidth=0.8, linestyle=':', alpha=0.6)

ax.set_xlabel('Heterophily $\\eta_h$', fontsize=10)
ax.set_ylabel('Post-pool signal $\\mathbb{E}[\\rho]$', fontsize=10)
ax.set_title('(b) Mean vs HAP ($k{=}10$, $\\epsilon{=}0.2$)', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1.1)
ax.grid(alpha=0.25)
ax.legend(loc='lower left', fontsize=8, framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = '/home/work/kftc_model/KA-003-FraudCenGCL/_paper/figures/fig_theory_inset'
plt.savefig(out + '.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out + '.png', bbox_inches='tight', dpi=300)
plt.close()
print('Saved:', out + '.pdf')

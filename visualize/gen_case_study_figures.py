"""
Generate the case-study Figure 1 (sampled subgraph for topology repair) and
the aggregate S-ratio shift histogram for the paper.

Inputs (from analysis/case_study_topology_repair.py):
  results/case_study/representative.json
  results/case_study/distributions.csv
  results/case_study/aggregate_stats.json

Outputs:
  _paper/figures/fig_intro_topology_repair.pdf      (Figure 1, replaces fig_intro_homophily.pdf)
  _paper/figures/fig_rq1_shift_hist.pdf             (aggregate distribution shift)
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS_DIR = os.path.join(BASE, 'results', 'case_study')
OUT_DIR = os.path.join(BASE, '_paper', 'figures')

# Colorblind-safe palette (matches existing intro variants).
C_SUSP = '#D64045'   # muted red
C_BENIGN = '#467599'  # steel blue
C_EGO = '#F0A000'    # gold for ego node
C_TX_EDGE = '#7A8A99'
C_BHV_EDGE = '#9E6B45'


def _circle_layout(n, radius, center=(0.0, 0.0), start_angle=math.pi / 2):
    """Evenly spaced points on a circle."""
    if n == 0:
        return np.zeros((0, 2))
    angles = start_angle - np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    return pts


def fig1_topology_repair(rep, dpi=300):
    """Side-by-side: tx-overcrowded (mostly benign) vs behavioral-clean."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.patch.set_facecolor('white')

    panels = [
        ('left', axes[0], 'Transaction graph (1-hop ego)',
         rep['tx_neighbors'], rep['tx_S'], rep['tx_B'], C_TX_EDGE),
        ('right', axes[1], r'Recovered neighborhood ($k$-NN, $k{=}10$)',
         rep['bhv_neighbors'], rep['bhv_S'], rep['bhv_B'], C_BHV_EDGE),
    ]

    for side, ax, title, neighbors, n_s, n_b, edge_color in panels:
        n = len(neighbors)
        # Place neighbors evenly on a circle around the ego.
        positions = _circle_layout(n, radius=1.0)
        # Edges first
        for i, (x, y) in enumerate(positions):
            ax.plot([0, x], [0, y], color=edge_color, alpha=0.45, lw=0.85, zorder=1)
        # Neighbor nodes
        for i, (x, y) in enumerate(positions):
            c = C_SUSP if neighbors[i]['label'] == 1 else C_BENIGN
            ax.scatter(x, y, s=58, color=c, edgecolor='white', linewidth=0.8, zorder=2)
        # Ego node
        ax.scatter(0, 0, s=180, color=C_EGO, edgecolor='black', linewidth=1.2,
                   zorder=3, marker='o')
        ax.text(0, -0.28, 'ego', ha='center', va='top', fontsize=7.4,
                color='black', zorder=4)

        # Ratio annotation: report count + fraction, plus S-S:S-B
        total = n_s + n_b
        frac = n_s / total if total > 0 else 0.0
        if n_s == 0:
            ratio_str = f'S-S : S-B = 0 : {n_b}'
        elif n_b == 0:
            ratio_str = f'S-S : S-B = {n_s} : 0'
        elif n_s >= n_b:
            ratio_str = f'S-S : S-B = {n_s/n_b:.1f} : 1'
        else:
            ratio_str = f'S-S : S-B = 1 : {n_b/n_s:.1f}'
        ax.text(0.0, -1.30, f'{n_s} of {total} neighbors suspicious ({frac*100:.1f}%)',
                ha='center', va='top', fontsize=8.4, color='#222222', fontweight='bold')
        ax.text(0.0, -1.50, ratio_str, ha='center', va='top', fontsize=7.6, color='#444444')

        ax.set_title(title, fontsize=9.6, pad=4)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.75, 1.4)
        ax.set_aspect('equal')
        ax.axis('off')

    # Shared legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_EGO,
                   markeredgecolor='black', markersize=10, label='Suspicious ego'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SUSP,
                   markersize=8, label='Suspicious neighbor'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BENIGN,
                   markersize=8, label='Benign neighbor'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.13, wspace=0.05)

    pdf = os.path.join(OUT_DIR, 'fig_intro_topology_repair.pdf')
    fig.savefig(pdf, bbox_inches='tight', dpi=dpi)
    fig.savefig(pdf.replace('.pdf', '.png'), bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'  Saved {pdf}')


def fig_rq1_shift_hist(dist, stats, dpi=300):
    """Overlay histogram of suspicious-share in tx vs behavioral neighborhoods."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    bins = np.linspace(0, 1.0, 21)
    tx = dist['tx_S_ratio'].dropna().to_numpy()
    bhv = dist['bhv_S_ratio'].to_numpy()
    ax.hist(tx, bins=bins, color=C_TX_EDGE, alpha=0.6, label='Transaction graph',
            edgecolor='white', linewidth=0.4)
    ax.hist(bhv, bins=bins, color=C_SUSP, alpha=0.55, label='Recovered neighborhood',
            edgecolor='white', linewidth=0.4)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.18)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.text(0.505, ymax * 1.04, 'majority-S threshold', fontsize=6.4,
            color='black', ha='left', va='bottom')

    ax.set_xlabel('Fraction of suspicious 1-hop neighbors', fontsize=8.5)
    ax.set_ylabel('# suspicious accounts', fontsize=8.5)
    ax.tick_params(labelsize=7.8)
    ax.legend(fontsize=7.4, loc='upper left', frameon=False,
              bbox_to_anchor=(0.0, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    p_tx = stats['p_tx_majority_susp'] * 100
    p_bhv = stats['p_bhv_majority_susp'] * 100
    ax.text(0.99, 0.98,
            f'Majority-S share\n{p_tx:.0f}% (tx) $\\to$ {p_bhv:.0f}% (recov.)',
            transform=ax.transAxes, fontsize=7.0, va='top', ha='right',
            color='#222222',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#bbbbbb', linewidth=0.5))
    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, 'fig_rq1_shift_hist.pdf')
    fig.savefig(pdf, bbox_inches='tight', dpi=dpi)
    fig.savefig(pdf.replace('.pdf', '.png'), bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'  Saved {pdf}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(CS_DIR, 'representative.json')) as f:
        rep = json.load(f)
    dist = pd.read_csv(os.path.join(CS_DIR, 'distributions.csv'))
    with open(os.path.join(CS_DIR, 'aggregate_stats.json')) as f:
        stats = json.load(f)

    print('Building Figure 1 (topology repair case study)...')
    fig1_topology_repair(rep)
    print('Building RQ1 shift histogram...')
    fig_rq1_shift_hist(dist, stats)


if __name__ == '__main__':
    main()

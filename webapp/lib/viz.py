"""Plotting helpers for the BEHAViEW walkthrough app."""
from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


C_SUSP = '#D64045'
C_BENIGN = '#467599'
C_EGO = '#F0A000'
C_TX_EDGE = '#7A8A99'
C_BHV_EDGE = '#9E6B45'


def render_ego_subgraph(ax, ego_idx: int, neighbors: list[dict],
                        induced_edges: list[tuple[int, int]], title: str,
                        edge_color: str, seed: int = 2):
    """Draw an induced 1-hop subgraph in matplotlib `ax`."""
    G = nx.Graph()
    G.add_node(ego_idx, label=1)
    for nb in neighbors:
        G.add_node(nb['idx'], label=nb['label'])
    for u, v in induced_edges:
        if u in G and v in G:
            G.add_edge(u, v)
    pos = nx.spring_layout(G, seed=seed, k=0.55, iterations=300)
    ego_pos = np.array(pos[ego_idx])
    pos = {n: (np.array(p) - ego_pos) for n, p in pos.items()}

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        is_ego_edge = (u == ego_idx) or (v == ego_idx)
        ax.plot([x1, x2], [y1, y2],
                color=edge_color,
                alpha=0.55 if is_ego_edge else 0.35,
                lw=1.0 if is_ego_edge else 0.65, zorder=1)
    for n, (x, y) in pos.items():
        if n == ego_idx:
            continue
        c = C_SUSP if G.nodes[n]['label'] == 1 else C_BENIGN
        ax.scatter(x, y, s=72, color=c, edgecolor='white', linewidth=0.8, zorder=2)
    ex, ey = pos[ego_idx]
    ax.scatter(ex, ey, s=210, color=C_EGO, edgecolor='black', linewidth=1.2,
               zorder=3, marker='o')
    ax.text(ex, ey - 0.10, 'ego', ha='center', va='top', fontsize=7.4,
            color='black', zorder=4)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.2)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')


def histogram_shift(ax, tx_ratios: np.ndarray, bhv_ratios: np.ndarray,
                    stats: dict | None = None) -> None:
    bins = np.linspace(0, 1.0, 21)
    ax.hist(tx_ratios, bins=bins, color=C_TX_EDGE, alpha=0.6,
            label='Transaction graph', edgecolor='white', linewidth=0.4)
    ax.hist(bhv_ratios, bins=bins, color=C_SUSP, alpha=0.55,
            label='Recovered neighborhood', edgecolor='white', linewidth=0.4)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('Fraction of suspicious 1-hop neighbors')
    ax.set_ylabel('# suspicious accounts')
    ax.legend(loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if stats is not None:
        p_tx = stats['p_tx_majority_susp'] * 100
        p_bhv = stats['p_bhv_majority_susp'] * 100
        ax.text(0.99, 0.95,
                f'Majority-S share\n{p_tx:.0f}% (tx) → {p_bhv:.0f}% (recov.)',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#bbbbbb', linewidth=0.5))

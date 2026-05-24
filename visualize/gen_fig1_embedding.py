"""
Generate Figure 1: 3D embedding visualization of topology repair via behavioral k-NN,
rendered using a single case study account (idx=62263) with transaction and behavioral neighborhoods.

Maximizes space utilization: large panels, minimal margins, efficient legend placement.
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_case_study():
    """Load representative case study account."""
    path = 'results/case_study/representative.json'
    with open(path) as f:
        return json.load(f)


def load_embeddings():
    """Load DGI+BN (b) embeddings."""
    path = 'results/embeddings/dgi_bn_b_seed2024.pt'
    z = torch.load(path, map_location='cpu')
    return z.numpy()


def load_node_features():
    """Load node labels and features."""
    df = pd.read_csv('datasets/atnet/ATNET_NODE_FEAT.csv')
    return df['label'].to_numpy(), df


def local_layout_3d(z, ego_idx, neighbor_indices, labels, k=3):
    """
    Project displacements in 3D via PCA, align PC1 with suspicious-to-benign direction.
    Returns 3D coordinates centered at ego = (0, 0, 0).
    """
    # Displacements from ego
    z_ego = z[ego_idx]
    displacements = z[neighbor_indices] - z_ego

    # PCA
    pca = PCA(n_components=k)
    coords_pca = pca.fit_transform(displacements)  # shape (n_neighbors, 3)

    # Sign-align PC1 with suspicious direction
    suspicious = labels[neighbor_indices] == 1
    if len(set(suspicious)) > 1:  # both S and B present
        mu_S = coords_pca[suspicious].mean(axis=0)
        mu_B = coords_pca[~suspicious].mean(axis=0)
        pc1_dir = mu_S[0] - mu_B[0]  # compare along PC1 only
        if pc1_dir < 0:
            coords_pca[:, 0] *= -1

    # Ego at origin
    coords_pca = np.vstack([[0, 0, 0], coords_pca])

    return coords_pca, pca


def render_panel(ax, coords, edge_list_tx, edge_list_bhv, labels, ego_at_0=True):
    """Render a single 3D panel."""
    if ego_at_0:
        ego_idx = 0
        neighbor_indices = np.arange(1, len(coords))

    # Wireframe halo spheres (distance markers)
    p25 = np.percentile(np.linalg.norm(coords[1:], axis=1), 25)
    p75 = np.percentile(np.linalg.norm(coords[1:], axis=1), 75)

    u = np.linspace(0, 2 * np.pi, 16)
    v = np.linspace(0, np.pi, 8)
    for r, alpha_val in [(p25, 0.15), (p75, 0.15)]:
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=alpha_val, color='gray', linewidth=0)

    # Transaction edges (gray lines from ego)
    for src, tgt in edge_list_tx:
        if src == ego_idx or tgt == ego_idx:
            other = tgt if src == ego_idx else src
            ax.plot(coords[[ego_idx, other], 0],
                   coords[[ego_idx, other], 1],
                   coords[[ego_idx, other], 2],
                   color='gray', linewidth=1.5, alpha=0.6, zorder=1)

    # Behavioral edges (orange dashed lines from ego)
    for src, tgt in edge_list_bhv:
        if src == ego_idx or tgt == ego_idx:
            other = tgt if src == ego_idx else src
            ax.plot(coords[[ego_idx, other], 0],
                   coords[[ego_idx, other], 1],
                   coords[[ego_idx, other], 2],
                   color='darkorange', linewidth=2.0, linestyle='--', alpha=0.7, zorder=2)

    # Node markers
    for i in range(1, len(coords)):
        if labels[neighbor_indices[i-1]] == 1:  # suspicious
            ax.scatter(*coords[i], s=520, marker='s', color='red', alpha=0.7,
                      edgecolor='darkred', linewidth=1, zorder=3)
        else:  # benign
            ax.scatter(*coords[i], s=420, marker='o', color='lightblue', alpha=0.6,
                      edgecolor='darkblue', linewidth=1, zorder=3)

    # Ego marker
    ax.scatter(*coords[ego_idx], s=2400, marker='*', color='gold',
              edgecolor='goldenrod', linewidth=1.5, zorder=5)

    # Centroid of suspicious nodes
    if labels[neighbor_indices].sum() > 0:
        susp_coords = coords[1:][labels[neighbor_indices] == 1]
        centroid = susp_coords.mean(axis=0)

        # Dashed line from ego to centroid
        ax.plot(coords[[ego_idx, ego_idx+1], 0],
               coords[[ego_idx, ego_idx+1], 1],
               coords[[ego_idx, ego_idx+1], 2],
               color='none')  # dummy for z-order
        ax.plot([coords[ego_idx, 0], centroid[0]],
               [coords[ego_idx, 1], centroid[1]],
               [coords[ego_idx, 2], centroid[2]],
               color='red', linewidth=2, linestyle=':', alpha=0.5, zorder=2)

        # Centroid marker (open red ring)
        ax.scatter(*centroid, s=500, marker='o', color='none',
                  edgecolor='red', linewidth=2.5, zorder=4)

        # d_S distance badge (positioned in 2D axes coords)
        d_S = np.linalg.norm(centroid - coords[ego_idx])
        ax.text2D(0.80, 0.92, f'd_S = {d_S:.2f}',
                 transform=ax.transAxes,
                 fontsize=64, fontweight='bold', color='red',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=2),
                 zorder=100)

    # Axis labels and ticks
    ax.set_xlabel('PC1', fontsize=60, fontweight='bold', labelpad=60)
    ax.set_ylabel('PC2', fontsize=60, fontweight='bold', labelpad=90)
    ax.set_zlabel('PC3', fontsize=60, fontweight='bold', labelpad=60)

    ax.tick_params(labelsize=50, pad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    # Set viewing angle
    ax.view_init(elev=25, azim=45)


def main():
    # Load data
    print("Loading embeddings and case study...")
    case_study = load_case_study()
    z = load_embeddings()
    labels, df_nodes = load_node_features()

    ego_idx = case_study['idx']
    tx_neighbors_data = case_study['tx_neighbors']
    bhv_neighbors_data = case_study['bhv_neighbors']
    tx_edges = case_study['tx_induced_edges']
    bhv_edges = case_study['bhv_induced_edges']

    # Extract neighbor indices
    tx_neighbors = [n['idx'] for n in tx_neighbors_data]
    bhv_neighbors = [n['idx'] for n in bhv_neighbors_data]

    all_nodes = [ego_idx] + tx_neighbors + bhv_neighbors
    all_nodes = sorted(set(all_nodes))
    node_to_local = {n: i for i, n in enumerate(all_nodes)}

    # PCA layout
    print("Computing 3D layout...")
    coords, pca = local_layout_3d(z, ego_idx, tx_neighbors + bhv_neighbors, labels)

    # Prepare edges for rendering
    edge_list_tx = [(node_to_local[s], node_to_local[t]) for s, t in tx_edges if s in node_to_local and t in node_to_local]
    edge_list_bhv = [(node_to_local[s], node_to_local[t]) for s, t in bhv_edges if s in node_to_local and t in node_to_local]

    # Create figure: MAXIMIZE SPACE
    print("Rendering figure...")
    # Large figsize for big panels
    fig_w, fig_h = 40, 24  # wide, tall
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)

    # Create two subplots side-by-side
    ax_tx = fig.add_subplot(121, projection='3d')
    ax_bhv = fig.add_subplot(122, projection='3d')

    # Render transaction graph
    print("  Rendering transaction graph...")
    render_panel(ax_tx, coords, edge_list_tx, [], labels)
    ax_tx.set_title('Transaction Graph\n(1-hop from ego)', fontsize=72, fontweight='bold', pad=20)

    # Render behavioral graph
    print("  Rendering behavioral k-NN graph...")
    render_panel(ax_bhv, coords, [], edge_list_bhv, labels)
    ax_bhv.set_title('Behavioral k-NN Graph\n(k=10 from ego)', fontsize=72, fontweight='bold', pad=20)

    # Create legend handles
    h_ego = Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                   markersize=30, label='Ego account', markeredgecolor='goldenrod', markeredgewidth=1)
    h_circ = Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                    markersize=15, label='Benign neighbor', markeredgecolor='darkblue', markeredgewidth=1)
    h_susp = Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                    markersize=15, label='Suspicious neighbor', markeredgecolor='darkred', markeredgewidth=1)
    h_tx = Line2D([0], [0], color='gray', linewidth=3, label='Transaction edge')
    h_bhv = Line2D([0], [0], color='darkorange', linewidth=3, linestyle='--', label='Behavioral k-NN edge')

    # Legend: 2-row layout, centered below
    handles = [h_ego, h_circ, h_susp, h_tx, h_bhv]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=56,
              bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True,
              handletextpad=1.0, columnspacing=2.0, labelspacing=0.8)

    # Tight layout: minimize margins
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12, wspace=0.25)

    # Enlarge panels to fill available space
    for ax in [ax_tx, ax_bhv]:
        pos = ax.get_position()
        new_w = pos.width * 1.15
        new_h = pos.height * 1.15
        ax.set_position([pos.x0 - 0.5*(new_w - pos.width),
                        pos.y0 - 0.5*(new_h - pos.height),
                        new_w, new_h])

    # Save
    pdf_path = '_manuscript/figures/fig_intro_topology_repair.pdf'
    png_path = '_manuscript/figures/fig_intro_topology_repair.png'
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    print(f"Saving {pdf_path}...")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f"Done. Saved PDF and PNG.")
    plt.close()


if __name__ == '__main__':
    main()

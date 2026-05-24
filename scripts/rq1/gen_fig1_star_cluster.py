"""
Generate an additional Figure 1 candidate:

  left  : transaction star graph around a suspicious ego account
  right : recovered behavioral-homophily suspicious cluster

The figure uses the representative case-study fixture produced by
analysis/case_study_topology_repair.py and mirrors outputs into both the paper
figure directory and the RQ1 result figure directory.

Usage:
    python3 scripts/rq1/gen_fig1_star_cluster.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[2]
CASE_STUDY_DIRS = [
    BASE / "results" / "rq1" / "case_study",
    BASE / "results" / "case_study",
]
OUT_DIRS = [
    BASE / "_paper" / "figures",
    BASE / "results" / "rq1" / "figures",
]

C_SUSP = "#D64045"
C_BENIGN = "#467599"
C_EGO = "#F0A000"
C_TX_EDGE = "#7A8A99"
C_BHV_EDGE = "#9E6B45"
C_TEXT = "#000000"
C_MUTED = "#000000"
C_PANEL = "#F8FAFC"
C_PANEL_EDGE = "#D4D8DF"
C_FLAG = "#D62728"
C_REPAIR_BG = "#FDE8E8"
C_REPAIR_EDGE = "#ECA0A5"
BEHAV_EXCLUDE = {"in_dc", "out_dc", "in_count", "out_count"}
K_BHV = 10
STAR_SCALE = 1.18
CLUSTER_SCALE = 1.28


class HandlerDashedCircle(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx = xdescent + width / 2.0
        cy = ydescent + height * 0.36
        radius = min(width, height) / 1.75
        artists = []
        for start in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            theta = np.linspace(start, start + np.pi / 5.0, 12)
            artists.append(
                mlines.Line2D(
                    cx + radius * np.cos(theta),
                    cy + radius * np.sin(theta),
                    color=orig_handle.color,
                    linewidth=orig_handle.linewidth,
                    transform=trans,
                )
            )
        return artists


class DashedCircleLegend:
    color = C_FLAG
    linewidth = 1.75


def load_representative() -> dict:
    for case_dir in CASE_STUDY_DIRS:
        path = case_dir / "representative.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    searched = ", ".join(str(d / "representative.json") for d in CASE_STUDY_DIRS)
    raise FileNotFoundError(f"representative.json not found. Searched: {searched}")


def save_all(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / f"{name}.pdf"
        png = out_dir / f"{name}.png"
        fig.savefig(pdf, bbox_inches="tight", dpi=dpi, facecolor="white")
        fig.savefig(png, bbox_inches="tight", dpi=dpi, facecolor="white")
        print(f"Saved: {pdf}")
        print(f"Saved: {png}")


def node_color(label: int, is_ego: bool = False) -> str:
    if is_ego:
        return C_EGO
    return C_SUSP if int(label) == 1 else C_BENIGN


def select_flagged_benign_neighbors(rep: dict, max_flags: int = 1) -> set[int]:
    """Find benign transaction neighbors that look suspicious in behavioral k-NN."""
    benign_tx = [int(nb["idx"]) for nb in rep["tx_neighbors"] if int(nb["label"]) == 0]
    if not benign_tx:
        return set()

    path = BASE / "datasets" / "atnet" / "ATNET_NODE_FEAT.csv"
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    labels = df["label"].to_numpy()
    agg_cols = [c for c in df.columns if c.startswith(("out_", "in_", "md_", "fnd_", "entropy"))]
    behav_cols = [c for c in agg_cols if c not in BEHAV_EXCLUDE]

    scaler = StandardScaler()
    x = scaler.fit_transform(df[behav_cols].to_numpy())
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    x = x / norms

    nn = NearestNeighbors(n_neighbors=K_BHV + 1, algorithm="ball_tree", metric="euclidean", n_jobs=-1)
    nn.fit(x)
    _, idxs = nn.kneighbors(x[benign_tx])

    ranked: list[tuple[int, int, int]] = []
    for node_idx, knn_row in zip(benign_tx, idxs):
        neighbors = [int(n) for n in knn_row[1:]]
        suspicious_count = int(sum(labels[n] == 1 for n in neighbors))
        ranked.append((suspicious_count, -node_idx, node_idx))

    ranked.sort(reverse=True)
    return {node_idx for suspicious_count, _, node_idx in ranked[:max_flags] if suspicious_count >= K_BHV // 2}


def draw_node(
    ax,
    xy,
    label: int,
    *,
    is_ego: bool = False,
    size: float = 170.0,
    text: str | None = None,
    alpha: float = 1.0,
) -> None:
    x, y = xy
    label_text = text if text is not None else ("S" if int(label) == 1 else "B")
    ax.scatter(
        [x],
        [y],
        s=size if not is_ego else size * 1.55,
        color=node_color(label, is_ego),
        edgecolor="white" if not is_ego else "#111111",
        linewidth=0.75 if not is_ego else 1.1,
        alpha=alpha,
        zorder=4,
    )
    ax.text(
        x,
        y,
        label_text,
        ha="center",
        va="center",
        fontsize=9.3 if len(label_text) > 1 and not is_ego else (10.2 if not is_ego else 11.8),
        color="white" if not is_ego else "#111111",
        fontweight="bold",
        zorder=5,
    )


def draw_potential_suspicious_ring(ax, xy, radius: float = 0.165) -> None:
    x, y = xy
    ring = mpatches.Circle(
        (x, y),
        radius,
        facecolor="none",
        edgecolor=C_FLAG,
        linewidth=1.5,
        linestyle=(0, (3.0, 2.0)),
        zorder=6,
    )
    ax.add_patch(ring)


def draw_node_id_callout(
    ax,
    xy,
    node_idx: int,
    *,
    text: str | None = None,
    x_offset: float = 0.0,
    y_offset: float = -0.235,
    ha: str = "center",
    va: str = "top",
) -> None:
    x, y = xy
    ax.text(
        x + x_offset,
        y + y_offset,
        text if text is not None else f"idx {node_idx}",
        ha=ha,
        va=va,
        fontsize=7.4,
        color=C_MUTED,
        linespacing=0.9,
        zorder=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
    )


def draw_external_node_id_callout(
    ax,
    node_xy: tuple[float, float],
    label_xy: tuple[float, float],
    text: str,
) -> None:
    ax.plot(
        [node_xy[0], label_xy[0] - 0.04],
        [node_xy[1], label_xy[1]],
        color=C_MUTED,
        linewidth=0.55,
        alpha=0.55,
        linestyle=(0, (1.2, 2.0)),
        zorder=3,
    )
    ax.text(
        label_xy[0],
        label_xy[1],
        text,
        ha="left",
        va="center",
        fontsize=7.4,
        color=C_MUTED,
        linespacing=0.9,
        zorder=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.7},
    )


def draw_panel_box(
    ax,
    title: str,
    subtitle: str,
    facecolor: str = "white",
    edgecolor: str = "white",
) -> None:
    box = mpatches.FancyBboxPatch(
        (-1.57, -2.02),
        3.14,
        4.06,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.7,
        zorder=-10,
    )
    ax.add_patch(box)
    ax.text(
        0,
        1.82,
        title,
        ha="center",
        va="center",
        fontsize=12.4,
        color=C_TEXT,
        fontweight="bold",
        linespacing=0.92,
    )
    ax.text(0, 1.47, subtitle, ha="center", va="center", fontsize=9.2, color=C_MUTED)


def circular_star_layout(neighbors: list[dict]) -> dict[int, tuple[float, float]]:
    """Place benign-heavy transaction neighbors around the suspicious ego."""
    n = len(neighbors)
    positions: dict[int, tuple[float, float]] = {}
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Put suspicious neighbors toward the top-right so the dilution is visible.
    ordered = sorted(neighbors, key=lambda nb: (0 if int(nb["label"]) == 1 else 1, nb["idx"]))
    susp_count = sum(1 for nb in ordered if int(nb["label"]) == 1)
    susp_angles = np.linspace(np.pi / 7, np.pi / 3, max(susp_count, 1), endpoint=True)
    benign_angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, max(n - susp_count, 1), endpoint=False)

    s_i = 0
    b_i = 0
    for nb in ordered:
        if int(nb["label"]) == 1:
            angle = float(susp_angles[s_i])
            radius = 0.82
            s_i += 1
        else:
            angle = float(benign_angles[b_i])
            radius = 1.03 + 0.08 * ((b_i % 2) - 0.5)
            b_i += 1
        positions[int(nb["idx"])] = (radius * math.cos(angle), radius * math.sin(angle))

    return positions


def cluster_layout(ego_idx: int, neighbors: list[dict]) -> dict[int, tuple[float, float]]:
    """Place suspicious behavioral neighbors as a compact recovered cluster."""
    positions: dict[int, tuple[float, float]] = {ego_idx: (0.0, 0.0)}
    susp = [nb for nb in neighbors if int(nb["label"]) == 1]
    benign = [nb for nb in neighbors if int(nb["label"]) == 0]

    # Wider rings make the recovered cluster visually comparable to the star graph.
    susp_angles = np.linspace(np.pi / 10, 2 * np.pi + np.pi / 10, max(len(susp), 1), endpoint=False)
    for i, nb in enumerate(susp):
        radius = 0.62 if i < 5 else 0.88
        angle = float(susp_angles[i])
        positions[int(nb["idx"])] = (radius * math.cos(angle), radius * math.sin(angle))

    # Benign k-NN outliers are intentionally offset from the recovered cluster.
    benign_angles = np.linspace(-np.pi / 4, np.pi / 4, max(len(benign), 1), endpoint=True)
    for i, nb in enumerate(benign):
        angle = float(benign_angles[i])
        positions[int(nb["idx"])] = (1.18 * math.cos(angle), 1.18 * math.sin(angle) - 0.12)

    return positions


def draw_edges(ax, edges: list[list[int]], positions: dict[int, tuple[float, float]], color: str, *, dashed: bool) -> None:
    seen = set()
    for u, v in edges:
        u = int(u)
        v = int(v)
        if u not in positions or v not in positions:
            continue
        key = tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=0.76 if not dashed else 1.10,
            alpha=0.38 if not dashed else 0.78,
            linestyle="--" if dashed else "-",
            solid_capstyle="round",
            zorder=1,
        )


def draw_repair_hull(ax, positions: dict[int, tuple[float, float]], neighbors: list[dict]) -> None:
    suspicious_points = np.array([positions[int(nb["idx"])] for nb in neighbors if int(nb["label"]) == 1])
    if len(suspicious_points) < 3:
        return
    center = suspicious_points.mean(axis=0)
    width = float(suspicious_points[:, 0].max() - suspicious_points[:, 0].min() + 0.58)
    height = float(suspicious_points[:, 1].max() - suspicious_points[:, 1].min() + 0.58)
    hull = mpatches.Ellipse(
        center,
        width,
        height,
        facecolor=C_REPAIR_BG,
        edgecolor=C_REPAIR_EDGE,
        linewidth=0.95,
        linestyle=(0, (4.0, 2.4)),
        alpha=0.30,
        zorder=0,
    )
    ax.add_patch(hull)


def draw_transaction_star(ax, rep: dict) -> None:
    ego_idx = int(rep["idx"])
    neighbors = rep["tx_neighbors"]
    positions = circular_star_layout(neighbors)
    positions[ego_idx] = (0.0, 0.0)
    positions = {idx: (x * STAR_SCALE, y * STAR_SCALE) for idx, (x, y) in positions.items()}
    flagged_neighbors = select_flagged_benign_neighbors(rep)

    draw_panel_box(
        ax,
        "Observed\nTransaction Topology",
        "suspicious signal dilution",
        facecolor="white",
        edgecolor="white",
    )
    draw_edges(ax, rep["tx_induced_edges"], positions, C_TX_EDGE, dashed=False)
    for nb in neighbors:
        draw_node(ax, positions[int(nb["idx"])], int(nb["label"]), size=172)
    draw_node(ax, positions[ego_idx], 1, is_ego=True, size=214)
    for nb in neighbors:
        if int(nb["idx"]) in flagged_neighbors:
            node_idx = int(nb["idx"])
            draw_potential_suspicious_ring(ax, positions[node_idx])
            draw_external_node_id_callout(
                ax,
                positions[node_idx],
                (1.30, -1.08),
                f"bridge\nidx {node_idx}",
            )

    total = int(rep["tx_S"]) + int(rep["tx_B"])
    ax.text(
        0,
        -1.55,
        f"{int(rep['tx_S'])}/{total} suspicious neighbors",
        ha="center",
        va="center",
        fontsize=10.8,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(0, -1.80, "S-S : S-B = 1 : 25", ha="center", va="center", fontsize=9.9, color=C_MUTED)


def draw_behavioral_cluster(ax, rep: dict) -> None:
    ego_idx = int(rep["idx"])
    neighbors = rep["bhv_neighbors"]
    positions = cluster_layout(ego_idx, neighbors)
    positions = {idx: (x * CLUSTER_SCALE, y * CLUSTER_SCALE) for idx, (x, y) in positions.items()}

    draw_panel_box(
        ax,
        "Repaired\nBehavioral Topology",
        r"behavior-only $k$-NN recovery",
        facecolor="white",
        edgecolor="white",
    )
    draw_repair_hull(ax, positions, neighbors)
    draw_edges(ax, rep["bhv_induced_edges"], positions, C_BHV_EDGE, dashed=True)
    for nb in neighbors:
        node_idx = int(nb["idx"])
        node_label = int(nb["label"])
        node_text = "B" if node_label == 0 else None
        draw_node(ax, positions[node_idx], node_label, size=218, text=node_text)
        if node_label == 0:
            draw_node_id_callout(
                ax,
                positions[node_idx],
                node_idx,
                text=f"recovered B\nidx {node_idx}",
                x_offset=0.21,
                y_offset=-0.03,
                ha="left",
                va="center",
            )
    draw_node(ax, positions[ego_idx], 1, is_ego=True, size=238)

    total = int(rep["bhv_S"]) + int(rep["bhv_B"])
    ax.text(
        0,
        -1.55,
        f"{int(rep['bhv_S'])}/{total} suspicious peers recovered",
        ha="center",
        va="center",
        fontsize=10.8,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(0, -1.80, "S-S : S-B = 9 : 1", ha="center", va="center", fontsize=9.9, color=C_MUTED)


def build_figure(rep: dict) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.28))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_xlim(-1.52, 1.52)
        ax.set_ylim(-2.08, 2.08)
        ax.set_aspect("equal")
        ax.axis("off")

    draw_transaction_star(axes[0], rep)
    draw_behavioral_cluster(axes[1], rep)

    arrow = mpatches.FancyArrowPatch(
        (0.474, 0.545),
        (0.526, 0.545),
        transform=fig.transFigure,
        arrowstyle="Simple,tail_width=0.28,head_width=4.8,head_length=6.4",
        mutation_scale=1,
        color=C_MUTED,
        linewidth=0.0,
        zorder=20,
    )
    fig.patches.append(arrow)
    fig.text(
        0.5,
        0.497,
        "label-free\nbehavior-only\nk-NN repair",
        ha="center",
        va="center",
        fontsize=8.0,
        color=C_MUTED,
        linespacing=0.92,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#E5E7EB", "linewidth": 0.55},
    )

    node_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_EGO, markeredgecolor="#111111", markersize=10.0, label="suspicious ego"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_SUSP, markeredgecolor="white", markersize=9.6, label="suspicious"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_BENIGN, markeredgecolor="white", markersize=9.6, label="benign"),
    ]
    flag_handle = DashedCircleLegend()
    edge_handles = [
        mlines.Line2D([], [], color=C_TX_EDGE, linewidth=1.2, label="transaction edge"),
        mlines.Line2D([], [], color=C_BHV_EDGE, linewidth=1.2, linestyle="--", label=r"behavioral-homophily $k$-NN edge"),
        flag_handle,
    ]
    fig.legend(
        handles=node_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.112),
        ncol=3,
        frameon=False,
        fontsize=9.0,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    fig.legend(
        handles=edge_handles,
        labels=["transaction\nedge", "behavioral-homophily\n$k$-NN edge", "repaired-topology\nsuspect"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=3,
        frameon=False,
        fontsize=8.7,
        handlelength=1.8,
        handleheight=1.85,
        handletextpad=0.45,
        columnspacing=1.05,
        handler_map={DashedCircleLegend: HandlerDashedCircle()},
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.205, wspace=-0.04)
    return fig


def main() -> None:
    rep = load_representative()
    fig = build_figure(rep)
    save_all(fig, "fig1_star_vs_recovered_cluster")
    plt.close(fig)


if __name__ == "__main__":
    main()

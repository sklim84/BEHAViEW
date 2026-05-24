"""
Generate a layered distillation-style Figure 1 for topology repair.

Bottom layer: observed multi-hop transaction topology.
Top layer   : behavior-only recovered suspicious peer cluster.

The visual metaphor is that behaviorally suspicious-like accounts are lifted
from the transaction floor into a recovered homophily layer. The node labels are
used only for visualization/evaluation, not to construct the recovered graph.

Usage:
    python3 scripts/rq1/gen_fig1_layered_distillation.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
CASE_STUDY = BASE / "results" / "rq1" / "case_study" / "representative.json"
NODE_FEAT = BASE / "datasets" / "atnet" / "ATNET_NODE_FEAT.csv"
TX_EDGES = BASE / "datasets" / "atnet" / "ATNET_EDGES.csv"
OUT_DIRS = [BASE / "_paper" / "figures", BASE / "results" / "rq1" / "figures"]

C_SUSP = "#D64045"
C_BENIGN = "#467599"
C_EGO = "#F0A000"
C_TX_EDGE = "#9CA8B4"
C_BHV_EDGE = "#9E6B45"
C_LIFT = "#111827"
C_TEXT = "#111827"
C_MUTED = "#4B5563"
C_FLOOR = "#EEF4F9"
C_SKY = "#FFF1F2"
C_FINGERPRINT = "#F5B642"
Y_SPREAD = 1.10
VISUAL_Y_SCALE = 1.55
TOP_CENTER_Y = 2.13
TOP_Y_MIN = TOP_CENTER_Y - 0.78
TOP_Y_MAX = TOP_CENTER_Y + 0.78
TOP_ELLIPSE_CENTER = (-0.18, TOP_CENTER_Y)
TOP_ELLIPSE_WIDTH = 6.12
TOP_ELLIPSE_HEIGHT = 1.72
SPECIAL_BOTTOM_POSITIONS = {
    1: (-0.34, -1.50),
    4: (-0.56, 0.68),
}
BOTTOM_LABEL_Y_OFFSETS = {
    3: -0.16,
}
BEHAVIOR_SCRIPT_LINE1 = "Behaviorally similar transaction features:"
BEHAVIOR_SCRIPT_LINE2 = "amount scale | in/out frequency | activity windows | type entropy"


def load_representative() -> dict:
    with open(CASE_STUDY) as f:
        return json.load(f)


def save_all(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / f"{name}.pdf"
        png = out_dir / f"{name}.png"
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0, dpi=dpi, facecolor="white")
        fig.savefig(png, bbox_inches="tight", pad_inches=0, dpi=dpi, facecolor="white")
        print(f"Saved: {pdf}")
        print(f"Saved: {png}")


def load_labels_and_graph() -> tuple[np.ndarray, dict[int, list[int]]]:
    node_df = pd.read_csv(NODE_FEAT, usecols=["account", "label"])
    account_to_idx = {acct: idx for idx, acct in enumerate(node_df["account"].astype(str))}
    labels = node_df["label"].astype(int).to_numpy()

    adj: dict[int, list[int]] = defaultdict(list)
    for chunk in pd.read_csv(TX_EDGES, chunksize=700_000):
        src = chunk["source"].astype(str).map(account_to_idx).astype(int).to_numpy()
        dst = chunk["target"].astype(str).map(account_to_idx).astype(int).to_numpy()
        for u, v in zip(src, dst):
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
    return labels, adj


def shortest_paths(ego: int, targets: list[int], adj: dict[int, list[int]]) -> dict[int, list[int]]:
    remaining = set(targets)
    parent: dict[int, int | None] = {ego: None}
    q: deque[int] = deque([ego])

    while q and remaining:
        u = q.popleft()
        for v in adj.get(u, []):
            if v in parent:
                continue
            parent[v] = u
            if v in remaining:
                remaining.remove(v)
            q.append(v)

    paths: dict[int, list[int]] = {}
    for target in targets:
        if target not in parent:
            continue
        path: list[int] = []
        cur: int | None = target
        while cur is not None:
            path.append(int(cur))
            cur = parent[cur]
        paths[target] = path[::-1]
    return paths


def node_color(label: int, *, ego: bool = False) -> str:
    if ego:
        return C_EGO
    return C_SUSP if int(label) == 1 else C_BENIGN


def visual_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * VISUAL_Y_SCALE)


def separate_node_position(
    x: float,
    y: float,
    radius: float,
    occupied: list[tuple[float, float, float]],
    *,
    origin: tuple[float, float],
    max_iter: int = 28,
) -> tuple[float, float]:
    for _ in range(max_iter):
        nearest: tuple[float, float, float, float, float] | None = None
        for ox, oy, oradius in occupied:
            min_gap = radius + oradius + 0.060
            dist = visual_distance((x, y), (ox, oy))
            if dist >= min_gap:
                continue
            deficit = min_gap - dist
            if nearest is None or deficit > nearest[0]:
                nearest = (deficit, ox, oy, oradius, dist)
        if nearest is None:
            break

        deficit, ox, oy, _oradius, _dist = nearest
        dx = x - ox
        dy = y - oy
        if abs(dx) + abs(dy) < 1e-6:
            dx = x - origin[0]
            dy = y - origin[1]
        norm = math.hypot(dx, dy) or 1.0
        push = min(0.16, deficit + 0.025)
        x += push * dx / norm
        y += push * dy / norm
        y = max(y, -1.48)
    return x, y


def separate_top_y_position(
    x: float,
    y: float,
    radius: float,
    occupied: list[tuple[float, float, float]],
) -> float:
    for step in range(18):
        changed = False
        for ox, oy, oradius in occupied:
            min_gap = radius + oradius + 0.070
            dx = abs(x - ox)
            dy = abs(y - oy) * VISUAL_Y_SCALE
            if dx >= min_gap or dy >= min_gap:
                continue
            direction = 1.0 if (step % 2 == 0) else -1.0
            y += direction * ((min_gap - dy) / VISUAL_Y_SCALE + 0.035)
            y = float(np.clip(y, TOP_Y_MIN, TOP_Y_MAX))
            changed = True
        if not changed:
            break
    return y


def draw_node(
    ax,
    xy: tuple[float, float],
    label: int,
    *,
    text: str | None = None,
    ego: bool = False,
    selected: bool = False,
    radius: float = 0.115,
    alpha: float = 1.0,
) -> None:
    x, y = xy
    label_text = text if text is not None else ("S" if int(label) == 1 else "B")
    suspicious_boost = 1.08 if int(label) == 1 and not ego else 1.0
    selected_boost = 1.14 if selected else 1.0
    digit_boost = 1.34 if label_text and len(str(label_text)) >= 2 else 1.0
    actual_radius = radius * (1.24 if ego else suspicious_boost) * selected_boost * digit_boost
    font_size = max(6.2, min(12.4, actual_radius * 92.0))
    node_z = 8 if selected or ego else 6
    shadow_alpha = 0.12 if selected or ego else 0.07
    if ego:
        edge_width = 1.15
    elif selected:
        edge_width = 0.78
    else:
        edge_width = 0.55 if int(label) == 1 else 0.75
    ax.add_patch(
        mpatches.Ellipse(
            (x + actual_radius * 0.16, y - actual_radius * 0.20),
            actual_radius * 2.16,
            actual_radius * 0.82,
            facecolor="#0F172A",
            edgecolor="none",
            alpha=shadow_alpha * alpha,
            zorder=node_z - 1,
        )
    )
    circ = mpatches.Circle(
        (x, y),
        actual_radius,
        facecolor=node_color(label, ego=ego),
        edgecolor="#111827" if ego or selected else "white",
        linewidth=edge_width,
        alpha=alpha,
        zorder=node_z,
    )
    ax.add_patch(circ)
    if label_text:
        ax.text(
            x,
            y,
            label_text,
            ha="center",
            va="center",
            fontsize=font_size,
            color="#111827" if ego else "white",
            fontweight="bold",
            alpha=alpha,
            zorder=9,
        )


def draw_layer_plate(ax, center: tuple[float, float], width: float, height: float, color: str, *, edge: str, alpha: float) -> None:
    plate = mpatches.Ellipse(
        center,
        width,
        height,
        facecolor=color,
        edgecolor=edge,
        linewidth=1.0,
        alpha=alpha,
        zorder=0,
    )
    ax.add_patch(plate)
    back = mpatches.Arc(
        (center[0], center[1] - 0.03),
        width,
        height * 0.90,
        theta1=200,
        theta2=340,
        color=edge,
        linewidth=0.75,
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(back)


def draw_transaction_layer(
    ax,
    rep: dict,
    labels: np.ndarray,
    paths: dict[int, list[int]],
) -> tuple[dict[int, tuple[float, float]], list[int]]:
    ego = int(rep["idx"])
    recovered_targets = [int(nb["idx"]) for nb in rep["bhv_neighbors"]]
    reachable = [target for target in recovered_targets if target in paths]
    chosen = reachable[:8]

    ego_xy = (0.0, -0.44)
    positions: dict[int, tuple[float, float]] = {ego: ego_xy}
    occupied: list[tuple[float, float, float]] = [(ego_xy[0], ego_xy[1], 0.16)]

    # Ambient one-hop counterparties around the ego to show dilution.
    tx_neighbors = rep["tx_neighbors"]
    star_neighbors = [nb for nb in tx_neighbors if int(nb["label"]) == 0][:24]
    star_neighbors += [nb for nb in tx_neighbors if int(nb["label"]) == 1][:1]
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    star_positions: list[tuple[float, float]] = []
    for i, nb in enumerate(star_neighbors):
        angle = (i * golden_angle + 0.26 + 0.19 * math.sin(i * 0.61)) % (2 * math.pi)
        radial = 0.76 + 0.74 * (((i * 7) % max(len(star_neighbors), 1)) / max(len(star_neighbors) - 1, 1))
        radial += 0.13 * math.sin(i * 1.37)
        y_perspective = 0.48 + 0.17 * math.cos(angle - 0.55)
        x = ego_xy[0] + radial * math.cos(angle)
        y = ego_xy[1] + radial * y_perspective * math.sin(angle)

        for _ in range(10):
            if all(math.hypot(x - px, (y - py) * 1.55) > 0.16 for px, py in star_positions):
                break
            radial += 0.085
            angle += 0.045
            y_perspective = 0.48 + 0.17 * math.cos(angle - 0.55)
            x = ego_xy[0] + radial * math.cos(angle)
            y = ego_xy[1] + radial * y_perspective * math.sin(angle)

        node_radius = 0.054
        x, y = separate_node_position(x, y, node_radius, occupied, origin=ego_xy)
        star_positions.append((x, y))
        occupied.append((x, y, node_radius))
        ax.plot([ego_xy[0], x], [ego_xy[1], y], color=C_TX_EDGE, linewidth=0.52, alpha=0.24, zorder=1)
        draw_node(ax, (x, y), int(nb["label"]), text="", radius=node_radius, alpha=0.96)

    # Arrange multi-hop paths as an ego-centered radial transaction structure.
    # Distances vary by branch and hop so that equal-hop nodes are not forced
    # onto a flat concentric ring.
    branch_angles = np.linspace(np.deg2rad(156), np.deg2rad(-156), max(len(chosen), 1))
    all_paths = [paths[target] for target in chosen]
    for branch_idx, (target, path, angle) in enumerate(zip(chosen, all_paths, branch_angles)):
        depth = max(len(path) - 1, 1)
        angle += np.deg2rad(5.5 * math.sin((branch_idx + 1) * 1.67))
        step_weights = np.array(
            [
                1.0
                + 0.24 * math.sin((branch_idx + 1) * 0.93 + step * 1.21)
                + 0.14 * math.cos((branch_idx + 2) * 1.47 - step * 0.68)
                for step in range(1, depth + 1)
            ]
        )
        step_fracs = np.cumsum(step_weights) / step_weights.sum()
        branch_x_span = 1.46 + 0.28 * math.sin((branch_idx + 1) * 1.09)
        branch_y_span = 0.44 + 0.18 * math.cos((branch_idx + 1) * 1.33)
        branch_depth = 1.0 + 0.10 * math.sin((branch_idx + 1) * 2.11)
        branch_curve = np.deg2rad(9.0 * math.sin((branch_idx + 1) * 0.79))
        prev_xy = ego_xy
        for d, node in enumerate(path[1:], start=1):
            frac = float(step_fracs[d - 1])
            local_angle = angle + branch_curve * math.sin(frac * math.pi)
            hop_jitter = 0.10 * math.sin((branch_idx + 2) * 0.83 + d * 1.17)
            hop_jitter += 0.06 * math.cos((branch_idx + 1) * 1.71 - d * 0.62)
            rx = (1.04 + frac * branch_x_span + hop_jitter) * branch_depth
            ry = 0.62 + frac * branch_y_span + 0.07 * math.cos((branch_idx + 1) * 0.91 + d * 1.23)
            perp = 0.08 * math.sin((d + branch_idx) * 1.35)
            x = ego_xy[0] + rx * math.cos(local_angle) - perp * math.sin(local_angle)
            y = ego_xy[1] + ry * math.sin(local_angle) + perp * math.cos(local_angle)

            min_next_hop_gap = 0.27
            for _ in range(12):
                if math.hypot(x - prev_xy[0], (y - prev_xy[1]) * 1.55) >= min_next_hop_gap:
                    break
                rx += 0.10
                ry += 0.040
                perp += 0.015
                x = ego_xy[0] + rx * math.cos(local_angle) - perp * math.sin(local_angle)
                y = ego_xy[1] + ry * math.sin(local_angle) + perp * math.cos(local_angle)

            if node in recovered_targets:
                node_radius = 0.124 + 0.006 * (1.0 - frac)
            else:
                node_radius = 0.054
            x, y = separate_node_position(x, y, node_radius, occupied, origin=ego_xy)
            if node in recovered_targets:
                label_id = recovered_targets.index(node) + 1
                y += BOTTOM_LABEL_Y_OFFSETS.get(label_id, 0.0)
            xy = (x, y)
            occupied.append((x, y, node_radius))
            edge_width = 0.78 + 0.18 * (1.0 - frac)
            edge_alpha = 0.45 + 0.16 * (1.0 - frac)
            ax.plot([prev_xy[0], x], [prev_xy[1], y], color=C_TX_EDGE, linewidth=edge_width, alpha=edge_alpha, zorder=2)
            if node in recovered_targets:
                positions[node] = xy
                draw_node(ax, xy, int(labels[node]), text=str(recovered_targets.index(node) + 1), selected=True, radius=node_radius)
            else:
                draw_node(ax, xy, int(labels[node]), text="", radius=node_radius, alpha=0.96)
            prev_xy = xy

    draw_node(ax, ego_xy, int(labels[ego]), text="E", ego=True, radius=0.145)

    # Non-reachable recovered neighbors are shown as reference markers only:
    # no transaction edge is drawn from the ego.
    for label_id, xy in SPECIAL_BOTTOM_POSITIONS.items():
        target = recovered_targets[label_id - 1]
        radius = 0.124
        positions[target] = xy
        occupied.append((xy[0], xy[1], radius))
        draw_node(ax, xy, int(labels[target]), text=str(label_id), selected=True, radius=radius)

    ax.text(0.0, -2.02, "Observed multi-hop transaction topology", fontsize=10.6, color=C_TEXT, fontweight="bold", ha="center", va="center")
    ax.text(0.0, -2.24, "(1/26 suspicious one-hop neighbors)", fontsize=8.4, color=C_MUTED, ha="center", va="center")

    return positions, chosen


def draw_behavior_layer(
    ax,
    rep: dict,
    labels: np.ndarray,
    bottom_positions: dict[int, tuple[float, float]],
    chosen: list[int],
) -> dict[int, tuple[float, float]]:
    recovered_targets = [int(nb["idx"]) for nb in rep["bhv_neighbors"]]
    top_center = (0.0, TOP_CENTER_Y)

    top_positions: dict[int, tuple[float, float]] = {}
    top_occupied: list[tuple[float, float, float]] = [(top_center[0], top_center[1], 0.20)]
    angles = np.linspace(np.pi * 0.03, 2 * np.pi + np.pi * 0.03, len(recovered_targets), endpoint=False)
    aligned_bottom = [
        bottom_positions[node]
        for node in recovered_targets
        if node in bottom_positions and recovered_targets.index(node) + 1 not in {1, 4}
    ]
    if aligned_bottom:
        bottom_ys = [pos[1] for pos in aligned_bottom]
        bottom_y_mid = (min(bottom_ys) + max(bottom_ys)) / 2.0
        bottom_y_span = max(max(bottom_ys) - min(bottom_ys), 1e-6)
    else:
        bottom_y_mid = 0.0
        bottom_y_span = 1.0
    for i, (node, angle) in enumerate(zip(recovered_targets, angles)):
        label_id = i + 1
        if label_id in {1, 4}:
            bx, _by = bottom_positions.get(node, top_center)
            x = float(bx)
            y = top_center[1] + (-0.56 if label_id == 1 else 0.56)
        elif node in bottom_positions:
            bx, by = bottom_positions[node]
            x = float(bx)
            y = top_center[1] + 0.82 * ((by - bottom_y_mid) / bottom_y_span)
            y = separate_top_y_position(x, y, 0.132, top_occupied)
        else:
            rx = 1.88 + 0.12 * (i % 2)
            ry = 0.66 + 0.05 * ((i + 1) % 2)
            x = top_center[0] + rx * math.cos(angle)
            y = top_center[1] + ry * math.sin(angle)
            x, y = separate_node_position(x, y, 0.132, top_occupied, origin=top_center, max_iter=18)
            y = float(np.clip(y, TOP_Y_MIN, TOP_Y_MAX))
        top_positions[node] = (x, y)
        top_occupied.append((x, y, 0.132))

    ax.add_patch(
        mpatches.Ellipse(
            TOP_ELLIPSE_CENTER,
            width=TOP_ELLIPSE_WIDTH,
            height=TOP_ELLIPSE_HEIGHT,
            facecolor="#9CA3AF",
            edgecolor="none",
            alpha=0.20,
            zorder=2,
        )
    )

    ego_top = top_center
    for u, v in rep["bhv_induced_edges"]:
        u = int(u)
        v = int(v)
        ego_id = int(rep["idx"])
        if u == ego_id and v in recovered_targets and recovered_targets.index(v) + 1 in {1, 4}:
            continue
        if v == ego_id and u in recovered_targets and recovered_targets.index(u) + 1 in {1, 4}:
            continue
        if u == ego_id:
            x1, y1 = ego_top
        elif u in top_positions:
            x1, y1 = top_positions[u]
        else:
            continue
        if v == ego_id:
            x2, y2 = ego_top
        elif v in top_positions:
            x2, y2 = top_positions[v]
        else:
            continue
        ax.plot([x1, x2], [y1, y2], color=C_BHV_EDGE, linewidth=0.50, linestyle="--", alpha=0.46, zorder=4)

    for node in recovered_targets:
        if node in bottom_positions:
            bx, by = bottom_positions[node]
            tx, ty = top_positions[node]
            label_id = recovered_targets.index(node) + 1
            if node not in chosen and label_id not in {1, 4}:
                continue
            ax.plot([bx, tx], [by, ty], color=C_LIFT, linewidth=0.80, alpha=0.55, linestyle=(0, (1.2, 2.2)), zorder=3)
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (bx, by + 0.15),
                    (tx, ty - 0.16),
                    arrowstyle="-|>",
                    mutation_scale=6.0,
                    color=C_LIFT,
                    linewidth=0.0,
                    alpha=0.75,
                    zorder=3,
                )
            )

    draw_node(ax, ego_top, 1, text="E", ego=True, radius=0.145)
    for node, xy in top_positions.items():
        text = str(recovered_targets.index(node) + 1)
        draw_node(ax, xy, int(labels[node]), text=text, selected=True, radius=0.132)

    top_label_y = max(y for _x, y in top_positions.values())
    ax.text(0.0, top_label_y + 0.46, "Recovered behavioral-homophily cluster", fontsize=10.9, color=C_TEXT, fontweight="bold", ha="center", va="center")
    ax.text(0.0, top_label_y + 0.25, "(9/10 suspicious behavioral neighbors)", fontsize=8.4, color=C_MUTED, ha="center", va="center")

    return top_positions


def draw_behavior_script(ax) -> None:
    cx, cy = TOP_ELLIPSE_CENTER
    ry = TOP_ELLIPSE_HEIGHT / 2.0
    font_prop = FontProperties(family="DejaVu Sans", weight="normal", style="normal", size=1.0)

    def add_cylinder_wall_text(text: str, y_base: float, scale: float, arc_lift: float) -> None:
        text_path = TextPath((0, 0), text, prop=font_prop)
        bbox = text_path.get_extents()
        x_mid = (bbox.xmin + bbox.xmax) / 2.0
        y_mid = (bbox.ymin + bbox.ymax) / 2.0
        half_width = max((bbox.xmax - bbox.xmin) * scale / 2.0, 1e-6)
        mapped_vertices: list[tuple[float, float]] = []
        background_vertices: list[tuple[float, float]] = []

        bg_pad = 0.135
        for t in np.linspace(-1.0, 1.0, 48):
            dx = t * half_width
            baseline_y = y_base + arc_lift * (t * t)
            background_vertices.append((cx + dx, baseline_y + bg_pad))
        for t in np.linspace(1.0, -1.0, 48):
            dx = t * half_width
            baseline_y = y_base + arc_lift * (t * t)
            background_vertices.append((cx + dx, baseline_y - bg_pad))
        background_vertices.append(background_vertices[0])
        background_path = MplPath(
            background_vertices,
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(background_vertices) - 2) + [MplPath.CLOSEPOLY],
        )
        ax.add_patch(
            mpatches.PathPatch(
                background_path,
                facecolor="white",
                edgecolor="none",
                alpha=1.0,
                zorder=10,
            )
        )

        for vx, vy in text_path.vertices:
            dx = (vx - x_mid) * scale
            frac = float(np.clip(dx / half_width, -1.0, 1.0))
            baseline_y = y_base + arc_lift * (frac * frac)
            mapped_vertices.append((cx + dx, baseline_y + (vy - y_mid) * scale))

        ax.add_patch(
            mpatches.PathPatch(
                MplPath(mapped_vertices, text_path.codes),
                facecolor="#000000",
                edgecolor="none",
                linewidth=0,
                alpha=1.0,
                zorder=11,
            )
        )

    wall_y = cy - ry - 0.070
    add_cylinder_wall_text(BEHAVIOR_SCRIPT_LINE1, wall_y - 0.004, 0.186, 0.150)
    add_cylinder_wall_text(BEHAVIOR_SCRIPT_LINE2, wall_y - 0.166, 0.173, 0.140)


def build_figure(rep: dict, labels: np.ndarray, paths: dict[int, list[int]]) -> plt.Figure:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.dpi": 300, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(5.10, 4.55))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-3.18, 2.96)
    ax.set_ylim(-2.95, 3.55)
    ax.axis("off")

    bottom_positions, chosen = draw_transaction_layer(ax, rep, labels, paths)
    draw_behavior_layer(ax, rep, labels, bottom_positions, chosen)
    draw_behavior_script(ax)

    handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_EGO, markeredgecolor="#111827", markersize=7.5, label="ego"),
        mlines.Line2D([], [], color=C_TX_EDGE, linewidth=1.0, label="tx path"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_SUSP, markeredgecolor="#111827", markersize=7.5, label="suspicious"),
        mlines.Line2D([], [], color=C_BHV_EDGE, linewidth=1.0, linestyle="--", label="behavioral similarity"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_BENIGN, markeredgecolor="white", markersize=7.5, label="benign"),
        mlines.Line2D([], [], color=C_LIFT, linewidth=1.0, linestyle=(0, (1.2, 2.2)), label="node correspondence"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.040),
        ncol=3,
        frameon=False,
        fontsize=8.6,
        handlelength=1.30,
        handletextpad=0.42,
        columnspacing=0.92,
        labelspacing=0.34,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main() -> None:
    rep = load_representative()
    labels, adj = load_labels_and_graph()
    targets = [int(nb["idx"]) for nb in rep["bhv_neighbors"]]
    paths = shortest_paths(int(rep["idx"]), targets, adj)
    fig = build_figure(rep, labels, paths)
    save_all(fig, "fig1_layered_distillation")
    plt.close(fig)


if __name__ == "__main__":
    main()

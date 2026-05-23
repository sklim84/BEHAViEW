"""
Generate a multi-hop correspondence figure for topology repair.

Left: shortest transaction-graph paths from the suspicious ego to the nodes
      recovered by behavioral homophily.
Right: recovered behavioral-homophily neighborhood.

The same numbered nodes appear on both sides, and dotted connectors show the
one-to-one correspondence between the recovered graph and the original topology.

Usage:
    python3 scripts/rq1/gen_fig_multihop_correspondence.py
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
CASE_STUDY = BASE / "results" / "rq1" / "case_study" / "representative.json"
NODE_FEAT = BASE / "datasets" / "hofinet" / "HOFINET_NODE_FEAT.csv"
TX_EDGES = BASE / "datasets" / "hofinet" / "HOFINET_EDGES.csv"
OUT_DIRS = [BASE / "_paper" / "figures", BASE / "results" / "rq1" / "figures"]

C_SUSP = "#D64045"
C_BENIGN = "#467599"
C_EGO = "#F0A000"
C_TX_EDGE = "#7A8A99"
C_BHV_EDGE = "#9E6B45"
C_MAP = "#6B7280"
C_TRACE = "#334155"
C_TEXT = "#000000"
C_MUTED = "#000000"
C_UNREACH = "#F3F4F6"
C_REPAIR_BG = "#FDE8E8"
C_REPAIR_EDGE = "#ECA0A5"
X_STEP = 1.02
Y_SPREAD = 1.30


def load_representative() -> dict:
    with open(CASE_STUDY) as f:
        return json.load(f)


def save_all(fig: plt.Figure, name: str) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / f"{name}.pdf"
        png = out_dir / f"{name}.png"
        fig.savefig(pdf, bbox_inches="tight", dpi=300, facecolor="white")
        fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Saved: {pdf}")
        print(f"Saved: {png}")


def node_color(label: int, is_ego: bool = False) -> str:
    if is_ego:
        return C_EGO
    return C_SUSP if int(label) == 1 else C_BENIGN


def recovered_node_text(node: int, labels: np.ndarray, target_ids: dict[int, int]) -> str:
    target_id = target_ids[node]
    return "B" if int(labels[node]) == 0 else str(target_id)


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


def shortest_paths_from_ego(
    ego: int,
    targets: list[int],
    adj: dict[int, list[int]],
) -> tuple[dict[int, list[int]], set[int]]:
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

    return paths, remaining


def shortest_path_between(src: int, dst: int, adj: dict[int, list[int]]) -> list[int] | None:
    parent: dict[int, int | None] = {src: None}
    q: deque[int] = deque([src])

    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj.get(u, []):
            if v in parent:
                continue
            parent[v] = u
            q.append(v)

    if dst not in parent:
        return None

    path: list[int] = []
    cur: int | None = dst
    while cur is not None:
        path.append(int(cur))
        cur = parent[cur]
    return path[::-1]


def separate_component_paths(
    disconnected_targets: set[int],
    target_ids: dict[int, int],
    adj: dict[int, list[int]],
) -> list[list[int]]:
    ordered = sorted(disconnected_targets, key=lambda t: target_ids[t])
    paths: list[list[int]] = []
    for src, dst in zip(ordered[:-1], ordered[1:]):
        path = shortest_path_between(src, dst, adj)
        if path:
            paths.append(path)
    return paths


def path_layout(
    ego: int,
    target_ids: dict[int, int],
    paths: dict[int, list[int]],
    unreachable: set[int],
    separate_paths: list[list[int]],
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    ordered_targets = sorted(target_ids, key=lambda t: target_ids[t])
    target_y = {
        target: y
        for target, y in zip(
            ordered_targets,
            np.linspace(Y_SPREAD, -Y_SPREAD, max(len(ordered_targets), 1)),
        )
    }

    positions: dict[int, tuple[float, float]] = {ego: (0.0, 0.0)}
    child_y: dict[int, list[float]] = defaultdict(list)

    for target, path in paths.items():
        positions[target] = (X_STEP * (len(path) - 1), float(target_y[target]))
        for parent, child in zip(path[:-1], path[1:]):
            child_y[parent].append(positions[child][1] if child in positions else target_y[target])

    max_depth = max((len(path) - 1 for path in paths.values()), default=1)
    for depth in range(max_depth - 1, 0, -1):
        nodes_at_depth = sorted({path[depth] for path in paths.values() if len(path) > depth})
        for node in nodes_at_depth:
            ys = []
            for target, path in paths.items():
                if node in path:
                    ys.append(target_y[target])
            positions[node] = (X_STEP * depth, float(np.mean(ys)) if ys else 0.0)

    # Light relaxation so shared intermediate branches do not overlap exactly.
    by_depth: dict[int, list[int]] = defaultdict(list)
    for node, (x, _) in positions.items():
        by_depth[int(round(x / X_STEP))].append(node)
    for nodes in by_depth.values():
        grouped = sorted(nodes, key=lambda n: positions[n][1], reverse=True)
        for i, node in enumerate(grouped):
            if node == ego or node in target_ids:
                continue
            y = positions[node][1]
            positions[node] = (positions[node][0], y + (i - (len(grouped) - 1) / 2) * 0.03)

    separate_positions: dict[int, tuple[float, float]] = {}
    separate_x = X_STEP * max_depth + 0.82
    for path_idx, path in enumerate(separate_paths):
        endpoint_ys = [target_y[node] for node in path if node in target_y]
        if len(endpoint_ys) >= 2:
            y_top = max(endpoint_ys)
            y_bottom = min(endpoint_ys) - 0.16
        elif endpoint_ys:
            y_top = endpoint_ys[0] + 0.25
            y_bottom = endpoint_ys[0] - 0.25
        else:
            y_top = 1.15 - 0.7 * path_idx
            y_bottom = y_top - 0.75

        ys = np.linspace(y_top, y_bottom, len(path))
        for i, (node, y) in enumerate(zip(path, ys)):
            x = separate_x + (0.10 if i % 2 else -0.10)
            if node in target_ids:
                x = separate_x
            positions[node] = (float(x), float(y))
            if node in unreachable:
                separate_positions[node] = positions[node]

    for target in sorted(unreachable, key=lambda t: target_ids[t]):
        if target not in separate_positions:
            separate_positions[target] = (separate_x, float(target_y[target]))
            positions[target] = separate_positions[target]

    return positions, separate_positions


def recovered_layout(
    ego: int,
    targets: list[dict],
    target_ids: dict[int, int],
) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {ego: (0.0, 0.0)}
    ordered_targets = sorted((int(nb["idx"]) for nb in targets), key=lambda n: target_ids[n])

    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, max(len(ordered_targets), 1), endpoint=False)
    for rank, (node, angle) in enumerate(zip(ordered_targets, angles)):
        radius = 1.02 + 0.08 * (rank % 2)
        positions[node] = (float(radius * np.cos(angle)), float(radius * np.sin(angle)))

    return positions


def draw_node(
    ax,
    xy: tuple[float, float],
    label: int,
    text: str,
    *,
    is_ego: bool = False,
    is_target: bool = False,
    faded: bool = False,
) -> None:
    size = 280 if is_ego else (252 if is_target else 148)
    face = C_UNREACH if faded else node_color(label, is_ego)
    edge = "#111111" if is_ego or is_target else "white"
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=size,
        color=face,
        edgecolor=edge,
        linewidth=1.2 if is_target or is_ego else 0.7,
        zorder=4,
    )
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=9.8 if not is_ego else 10.8,
        color="#111111" if is_ego or faded else "white",
        fontweight="bold",
        zorder=5,
    )


def draw_node_id_callout(ax, xy: tuple[float, float], node_idx: int, *, side: str = "right") -> None:
    x, y = xy
    if side == "right":
        dx, dy, ha, va = 0.24, 0.06, "left", "bottom"
    elif side == "left":
        dx, dy, ha, va = -0.24, 0.06, "right", "bottom"
    elif side == "above":
        dx, dy, ha, va = 0.0, 0.25, "center", "bottom"
    elif side == "above_left":
        dx, dy, ha, va = -0.22, 0.25, "right", "bottom"
    elif side == "below":
        dx, dy, ha, va = 0.0, -0.27, "center", "top"
    elif side == "below_left":
        dx, dy, ha, va = -0.22, -0.25, "right", "top"
    else:
        dx, dy, ha, va = 0.0, -0.20, "center", "top"
    ax.annotate(
        f"idx {node_idx}",
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=7.7,
        color=C_MUTED,
        zorder=7,
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.35, "alpha": 0.9, "pad": 0.65},
        arrowprops={
            "arrowstyle": "-",
            "color": C_TRACE,
            "linewidth": 0.55,
            "alpha": 0.72,
            "shrinkA": 1,
            "shrinkB": 7,
        },
    )


def recovered_b_path(paths: dict[int, list[int]], labels: np.ndarray) -> list[int]:
    benign_targets = [target for target in paths if int(labels[target]) == 0]
    if not benign_targets:
        return []
    return paths[min(benign_targets, key=lambda node: len(paths[node]))]


def draw_path_idx_callouts(
    ax,
    positions: dict[int, tuple[float, float]],
    path: list[int],
) -> None:
    sides = ["below", "below_left", "above_left", "below_left", "right"]
    for depth, node in enumerate(path[1:], start=1):
        if node not in positions:
            continue
        side = sides[depth] if depth < len(sides) else "right"
        draw_node_id_callout(ax, positions[node], node, side=side)


def draw_trace_path(
    ax,
    positions: dict[int, tuple[float, float]],
    path: list[int],
) -> None:
    for u, v in zip(path[:-1], path[1:]):
        if u not in positions or v not in positions:
            continue
        ax.plot(
            [positions[u][0], positions[v][0]],
            [positions[u][1], positions[v][1]],
            color=C_TRACE,
            linewidth=1.45,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )


def draw_repair_hull(
    ax,
    positions: dict[int, tuple[float, float]],
    targets: list[dict],
) -> None:
    suspicious_points = np.array([positions[int(nb["idx"])] for nb in targets if int(nb["label"]) == 1])
    if len(suspicious_points) < 3:
        return
    center = suspicious_points.mean(axis=0)
    hull = mpatches.Ellipse(
        center,
        width=float(suspicious_points[:, 0].max() - suspicious_points[:, 0].min() + 0.66),
        height=float(suspicious_points[:, 1].max() - suspicious_points[:, 1].min() + 0.66),
        facecolor=C_REPAIR_BG,
        edgecolor=C_REPAIR_EDGE,
        linewidth=0.95,
        linestyle=(0, (4.0, 2.4)),
        alpha=0.30,
        zorder=0,
    )
    ax.add_patch(hull)


def draw_transaction_paths(
    ax,
    rep: dict,
    labels: np.ndarray,
    paths: dict[int, list[int]],
    unreachable: set[int],
    separate_paths: list[list[int]],
    target_ids: dict[int, int],
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    ego = int(rep["idx"])
    positions, separate_positions = path_layout(ego, target_ids, paths, unreachable, separate_paths)

    max_depth = max((len(path) - 1 for path in paths.values()), default=1)
    separate_x = X_STEP * max_depth + 0.82
    title_x = (X_STEP * max_depth + (separate_x if separate_positions else X_STEP * max_depth)) / 2

    ax.text(title_x, 1.93, "Original\nTransaction Topology", ha="center", fontsize=12.3, fontweight="bold", color=C_TEXT, linespacing=0.92)
    ax.text(title_x, 1.62, "recovered peers appear as\n2-4 hop transaction paths", ha="center", fontsize=9.2, color=C_MUTED, linespacing=1.0)

    edges = set()
    for path in [*paths.values(), *separate_paths]:
        for u, v in zip(path[:-1], path[1:]):
            edges.add(tuple(sorted((u, v))))
    for u, v in edges:
        ax.plot(
            [positions[u][0], positions[v][0]],
            [positions[u][1], positions[v][1]],
            color=C_TX_EDGE,
            linewidth=0.95,
            alpha=0.62,
            zorder=1,
        )

    for depth in range(max_depth + 1):
        ax.text(X_STEP * depth, 1.48, f"{depth}-hop" if depth else "ego", ha="center", fontsize=9.5, color=C_MUTED)
    if separate_positions:
        ax.text(separate_x, 1.48, "other component", ha="center", fontsize=9.5, color=C_MUTED)

    trace_path = recovered_b_path(paths, labels)
    draw_trace_path(ax, positions, trace_path)

    target_set = set(target_ids)
    for node, xy in sorted(positions.items(), key=lambda item: (item[1][0], -item[1][1])):
        if node == ego:
            draw_node(ax, xy, int(labels[node]), "E", is_ego=True)
        elif node in target_set:
            draw_node(ax, xy, int(labels[node]), recovered_node_text(node, labels, target_ids), is_target=True)
        else:
            draw_node(ax, xy, int(labels[node]), "S" if labels[node] else "B")

    draw_path_idx_callouts(ax, positions, trace_path)

    if separate_positions:
        separate_nodes = {
            node
            for path in separate_paths
            for node in path
        } or set(separate_positions)
        xs = [positions[node][0] for node in separate_nodes if node in positions]
        ys = [positions[node][1] for node in separate_nodes if node in positions]
        box_x = min(xs) - 0.32
        box_y = min(ys) - 0.22
        box_w = max(xs) - min(xs) + 0.64
        box_h = max(ys) - min(ys) + 0.44
        box = mpatches.FancyBboxPatch(
            (box_x, box_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="white",
            edgecolor="#CBD5E1",
            linestyle="--",
            linewidth=0.9,
            zorder=0,
        )
        ax.add_patch(box)

    return positions, separate_positions


def draw_recovered_graph(
    ax,
    rep: dict,
    labels: np.ndarray,
    target_ids: dict[int, int],
) -> dict[int, tuple[float, float]]:
    ego = int(rep["idx"])
    targets = rep["bhv_neighbors"]
    positions = recovered_layout(ego, targets, target_ids)

    ax.text(0, 1.93, "Repaired\nBehavioral Topology", ha="center", fontsize=12.3, fontweight="bold", color=C_TEXT, linespacing=0.92)
    ax.text(0, 1.62, "same accounts become\n1-hop behavioral neighbors", ha="center", fontsize=9.2, color=C_MUTED, linespacing=1.0)

    draw_repair_hull(ax, positions, targets)

    for u, v in rep["bhv_induced_edges"]:
        u = int(u)
        v = int(v)
        if u not in positions or v not in positions:
            continue
        ax.plot(
            [positions[u][0], positions[v][0]],
            [positions[u][1], positions[v][1]],
            color=C_BHV_EDGE,
            linewidth=0.95,
            linestyle="--",
            alpha=0.52,
            zorder=1,
        )

    for nb in targets:
        node = int(nb["idx"])
        draw_node(ax, positions[node], int(labels[node]), recovered_node_text(node, labels, target_ids), is_target=True)
        if int(labels[node]) == 0:
            draw_node_id_callout(ax, positions[node], node, side="left")
    draw_node(ax, positions[ego], int(labels[ego]), "E", is_ego=True)

    return positions


def build_figure(
    rep: dict,
    labels: np.ndarray,
    paths: dict[int, list[int]],
    unreachable: set[int],
    separate_paths: list[list[int]],
) -> plt.Figure:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.dpi": 300})
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(10.6, 4.85),
        gridspec_kw={"width_ratios": [1.9, 1.1]},
    )
    fig.patch.set_facecolor("white")

    targets = [int(nb["idx"]) for nb in rep["bhv_neighbors"]]
    target_ids = {node: i + 1 for i, node in enumerate(targets)}

    left_positions, separate_positions = draw_transaction_paths(
        ax_left,
        rep,
        labels,
        paths,
        unreachable,
        separate_paths,
        target_ids,
    )
    right_positions = draw_recovered_graph(ax_right, rep, labels, target_ids)

    for target in targets:
        if target in left_positions:
            xy_left = left_positions[target]
        else:
            xy_left = separate_positions[target]
        xy_right = right_positions[target]
        rank = target_ids[target] - 1
        rad = np.linspace(-0.08, 0.08, max(len(targets), 1))[rank]
        con = ConnectionPatch(
            xyA=xy_right,
            xyB=xy_left,
            coordsA="data",
            coordsB="data",
            axesA=ax_right,
            axesB=ax_left,
            arrowstyle="-",
            linestyle=(0, (1.0, 2.4)),
            connectionstyle=f"arc3,rad={rad:.3f}",
            linewidth=0.65,
            color=C_MAP,
            alpha=0.36,
            zorder=0,
        )
        fig.add_artist(con)

    for ax in (ax_left, ax_right):
        ax.axis("off")
        ax.set_facecolor("white")

    ax_left.set_xlim(-0.42, 5.65)
    ax_left.set_ylim(-1.82, 2.08)
    ax_right.set_xlim(-1.28, 1.28)
    ax_right.set_ylim(-1.82, 2.08)

    reachable_depths = [len(path) - 1 for path in paths.values()]
    ax_left.text(
        2.35,
        -1.73,
        f"Same real accounts: {len(paths)}/10 recovered peers are reachable from the ego transaction component "
        f"({reachable_depths.count(2)} at 2-hop, {reachable_depths.count(4)} at 4-hop)\n"
        f"Topology repair rewires distant transaction nodes into a 1-hop behavioral neighborhood",
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=C_MUTED,
    )

    handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_EGO, markeredgecolor="#111111", markersize=9.5, label="ego"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_SUSP, markeredgecolor="#111111", markersize=9.0, label="suspicious recovered"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_BENIGN, markeredgecolor="#111111", markersize=9.0, label="benign recovered"),
        mlines.Line2D([], [], color=C_TX_EDGE, linewidth=1.2, label="transaction path"),
        mlines.Line2D([], [], color=C_BHV_EDGE, linewidth=1.2, linestyle="--", label="behavioral k-NN"),
        mlines.Line2D([], [], color=C_MAP, linewidth=1.0, linestyle=(0, (1.0, 2.4)), label="1:1 correspondence"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        frameon=False,
        fontsize=8.8,
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=0.95,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.955, bottom=0.095, wspace=0.05)
    return fig


def main() -> None:
    rep = load_representative()
    labels, adj = load_labels_and_graph()
    targets = [int(nb["idx"]) for nb in rep["bhv_neighbors"]]
    paths, unreachable = shortest_paths_from_ego(int(rep["idx"]), targets, adj)
    target_ids = {node: i + 1 for i, node in enumerate(targets)}
    separate_paths = separate_component_paths(unreachable, target_ids, adj)
    print(f"Reachable recovered nodes: {len(paths)}/{len(targets)}")
    for target, path in paths.items():
        print(f"  target={target} depth={len(path)-1} path={path}")
    if unreachable:
        print(f"Unreachable in ego transaction component: {sorted(unreachable)}")
    for path in separate_paths:
        print(f"  separate_component_path depth={len(path)-1} path={path}")
    fig = build_figure(rep, labels, paths, unreachable, separate_paths)
    save_all(fig, "fig_multihop_correspondence")
    plt.close(fig)


if __name__ == "__main__":
    main()

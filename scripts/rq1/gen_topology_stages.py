"""Generate a PCA style topology evolution figure for BehaView.

The figure uses the representative HOFINET account selected by
analysis/case_study_topology_repair.py. It places the ego, transaction
neighbors, and recovered behavioral kNN neighbors in a 2D behavioral-feature
projection, then overlays the topology used at each stage.

Usage:
    python3 scripts/rq1/gen_topology_stages.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize


BASE = Path(__file__).resolve().parents[2]
CASE_STUDY_DIRS = [
    BASE / "results" / "rq1" / "case_study",
    BASE / "results" / "case_study",
]
NODE_FEAT = BASE / "datasets" / "HOFINET_NODE_FEAT.csv"
EDGE_FILE = BASE / "datasets" / "HOFINET_EDGES.csv"
OUT_DIRS = [BASE / "_paper" / "figures", BASE / "results" / "rq1" / "figures"]

BEHAVIOR_COLS = [
    "out_mean",
    "out_max",
    "out_std",
    "in_mean",
    "in_max",
    "in_std",
    "out_3m_mean",
    "out_3m_count",
    "in_3m_mean",
    "in_3m_count",
    "out_6m_mean",
    "out_6m_count",
    "in_6m_mean",
    "in_6m_count",
    "out_12m_mean",
    "out_12m_count",
    "in_12m_mean",
    "in_12m_count",
    "md_type_entropy",
    "fnd_type_entropy",
]

C_SUSP = "#D84C5B"
C_BENIGN = "#3F7396"
C_EGO = "#F2A51A"
C_TX_EDGE = "#B8C2CC"
C_BHV_EDGE = "#9A6A43"
C_TEXT = "#111827"
C_MUTED = C_TEXT
C_SOFT_RED = "#FCE8EA"
C_SOFT_BLUE = "#EEF4F9"
C_SOFT_GREEN = "#E9F7EF"
C_POOL_TEXT = "#8A2F2F"
C_FINAL_TEXT = "#2E6F4E"
C_BYOL = "#6D5EF7"
C_BYOL_DARK = "#4F46B8"
C_ARROW = "#6F7785"
PANEL_VALUE_Y = 0.67
PANEL_NOTE_Y = 0.34
PANEL_VALUE_SIZE = 14.2
PANEL_VALUE_SMALL_SIZE = 11.7
PANEL_NOTE_SIZE = 9.8


def load_representative() -> dict:
    for case_dir in CASE_STUDY_DIRS:
        path = case_dir / "representative.json"
        if path.exists():
            return json.loads(path.read_text())
    searched = ", ".join(str(d / "representative.json") for d in CASE_STUDY_DIRS)
    raise FileNotFoundError(f"representative.json not found. Searched: {searched}")


def save_all(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / f"{name}.pdf"
        png = out_dir / f"{name}.png"
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.0, dpi=dpi, facecolor="white")
        fig.savefig(png, bbox_inches="tight", pad_inches=0.0, dpi=dpi, facecolor="white")
        print(f"Saved: {pdf}")
        print(f"Saved: {png}")


def build_tx_context(rep: dict, accounts: np.ndarray, labels: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    """Add a small actual 2-hop transaction context so the initial graph is not a pure ego star."""
    account_to_idx = {str(acc): i for i, acc in enumerate(accounts)}
    ego_account = str(rep["account"])
    tx_account_to_idx = {str(n["account"]): int(n["idx"]) for n in rep["tx_neighbors"]}
    excluded_accounts = set(tx_account_to_idx) | {ego_account}
    candidates: dict[int, list[int]] = {idx: [] for idx in tx_account_to_idx.values()}

    for chunk in pd.read_csv(EDGE_FILE, usecols=["source", "target"], chunksize=500_000):
        mask = chunk["source"].isin(tx_account_to_idx) | chunk["target"].isin(tx_account_to_idx)
        if not mask.any():
            continue
        for source, target in chunk.loc[mask].itertuples(index=False):
            source = str(source)
            target = str(target)
            if source in tx_account_to_idx:
                bridge = tx_account_to_idx[source]
                other = target
            else:
                bridge = tx_account_to_idx[target]
                other = source
            if other in excluded_accounts:
                continue
            other_idx = account_to_idx.get(other)
            if other_idx is None or int(labels[other_idx]) != 0:
                continue
            bucket = candidates[bridge]
            if other_idx not in bucket:
                bucket.append(int(other_idx))
        if sum(min(2, len(v)) for v in candidates.values()) >= 14:
            break

    bridge_items = sorted(
        ((bridge, ids) for bridge, ids in candidates.items() if ids),
        key=lambda item: (-len(item[1]), item[0]),
    )[:7]
    context_ids: list[int] = []
    context_edges: list[tuple[int, int]] = []
    seen_context: set[int] = set()
    for bridge, ids in bridge_items:
        picked = 0
        for node_id in ids:
            if node_id in seen_context:
                continue
            seen_context.add(node_id)
            context_ids.append(node_id)
            context_edges.append((bridge, node_id))
            picked += 1
            if picked == 2:
                break
    return context_ids, context_edges


def build_case_layout(rep: dict) -> dict:
    ego = int(rep["idx"])
    tx_ids = [int(n["idx"]) for n in rep["tx_neighbors"]]
    bhv_ids = [int(n["idx"]) for n in rep["bhv_neighbors"]]

    labels = {ego: int(rep["label"])}
    labels.update({int(n["idx"]): int(n["label"]) for n in rep["tx_neighbors"]})
    labels.update({int(n["idx"]): int(n["label"]) for n in rep["bhv_neighbors"]})

    feat = pd.read_csv(NODE_FEAT, usecols=["account", "label", *BEHAVIOR_COLS])
    tx_context_ids, tx_context_edges = build_tx_context(rep, feat["account"].to_numpy(), feat["label"].to_numpy())
    labels.update({node_id: int(feat.iloc[node_id]["label"]) for node_id in tx_context_ids})

    all_ids = list(dict.fromkeys([ego] + tx_ids + bhv_ids + tx_context_ids))
    X = feat.iloc[all_ids][BEHAVIOR_COLS].to_numpy(dtype=float)
    X = np.nan_to_num(X, copy=False)
    X = normalize(StandardScaler().fit_transform(X))

    perplexity = max(3, min(8, (len(all_ids) - 1) // 3))
    try:
        emb = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=17,
            max_iter=1500,
        ).fit_transform(X)
    except TypeError:
        emb = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=17,
            n_iter=1500,
        ).fit_transform(X)

    pca3d = PCA(n_components=3).fit_transform(X)
    label_arr = np.array([labels[i] for i in all_ids], dtype=int)
    if np.any(label_arr == 1) and np.any(label_arr == 0):
        gaps = np.abs(pca3d[label_arr == 1].mean(axis=0) - pca3d[label_arr == 0].mean(axis=0))
        susp_axis = int(np.argmax(gaps))
        order = [susp_axis] + [j for j in range(3) if j != susp_axis]
        pca3d = pca3d[:, order]
        if pca3d[label_arr == 1, 0].mean() < pca3d[label_arr == 0, 0].mean():
            pca3d[:, 0] *= -1

    raw = {node_id: emb[i] for i, node_id in enumerate(all_ids)}
    raw3d = {node_id: pca3d[i] for i, node_id in enumerate(all_ids)}
    return {
        "ego": ego,
        "tx_ids": tx_ids,
        "tx_context_ids": tx_context_ids,
        "tx_context_edges": tx_context_edges,
        "bhv_ids": bhv_ids,
        "all_ids": all_ids,
        "labels": labels,
        "raw": raw,
        "raw3d": raw3d,
        "rep": rep,
    }


def text(ax, x: float, y: float, s: str, *, size: float = 7.0, color: str = C_TEXT, weight: str = "normal") -> None:
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color, fontweight=weight, linespacing=0.92, zorder=50)


def stage_title(ax, cx: float, title: str, subtitle: str, *, title_size: float = 9.8, subtitle_y: float = 2.71) -> None:
    text(ax, cx, 3.03, title, size=title_size, weight="bold")
    text(ax, cx, subtitle_y, subtitle, size=8.3, color=C_MUTED)


def node_color(label: int, *, ego: bool = False) -> str:
    if ego:
        return C_EGO
    return C_SUSP if int(label) == 1 else C_BENIGN


def node(
    ax,
    x: float,
    y: float,
    color: str,
    *,
    r: float = 0.047,
    label: str | None = None,
    edge: str = "white",
    lw: float = 0.42,
    alpha: float = 1.0,
    z: int = 10,
) -> None:
    ax.add_patch(
        mpatches.Circle(
            (x, y),
            r,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            zorder=z,
        )
    )
    if label:
        text(ax, x, y, label, size=max(5.2, r * 88), color="#111827" if color == C_EGO else "white", weight="bold")


def edge(
    ax,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    color: str,
    lw: float = 0.5,
    alpha: float = 0.55,
    dashed: bool = False,
    rad: float = 0.0,
    z: int = 2,
) -> None:
    patch = mpatches.FancyArrowPatch(
        a,
        b,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "-",
        color=color,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = C_ARROW, lw: float = 0.9) -> None:
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=lw,
            color=color,
            alpha=0.9,
            zorder=35,
        )
    )


def panel_positions(layout: dict, node_ids: list[int], cx: float, cy: float, *, scale: float = 0.72) -> dict[int, tuple[float, float]]:
    arr = np.array([layout["raw"][i] for i in node_ids], dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    span = np.percentile(np.abs(arr), 92)
    if span <= 1e-9:
        span = np.max(np.abs(arr)) or 1.0
    arr = np.clip(arr / span, -1.25, 1.25)
    arr[:, 1] *= 0.72
    return {node_id: (cx + scale * arr[j, 0], cy + scale * arr[j, 1]) for j, node_id in enumerate(node_ids)}


def focus_cluster_positions(
    layout: dict,
    cx: float,
    cy: float,
    *,
    scale: float = 0.44,
    offset: tuple[float, float] = (0.03, -0.17),
) -> dict[int, tuple[float, float]]:
    """Locally magnify the behavioral kNN cluster while preserving t-SNE shape."""
    node_ids = [layout["ego"]] + layout["bhv_ids"]
    arr = np.array([layout["raw"][i] for i in node_ids], dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    span = np.percentile(np.abs(arr), 90)
    if span <= 1e-9:
        span = np.max(np.abs(arr)) or 1.0
    arr = np.clip(arr / span, -1.20, 1.20)
    arr[:, 1] *= 0.76
    ox, oy = offset
    return {node_id: (cx + ox + scale * arr[j, 0], cy + oy + scale * arr[j, 1]) for j, node_id in enumerate(node_ids)}


def class_separated_behavior_positions(layout: dict, cx: float, cy: float) -> dict[int, tuple[float, float]]:
    """Use a clean 2D behavioral layout where benign and suspicious accounts separate."""
    ego = layout["ego"]
    benign_ids = [i for i in layout["all_ids"] if i != ego and layout["labels"][i] == 0]
    suspicious_ids = [i for i in layout["all_ids"] if i != ego and layout["labels"][i] == 1]
    pos: dict[int, tuple[float, float]] = {}

    def place_blob(node_ids: list[int], center: tuple[float, float], rx: float, ry: float, phase: float) -> None:
        n = max(1, len(node_ids))
        for j, node_id in enumerate(node_ids):
            angle = phase + 2.399963 * j
            radius = 0.25 + 0.72 * ((j * 5) % max(7, n + 5)) / max(7, n + 5)
            wobble = 0.030 * math.sin(j * 1.73)
            pos[node_id] = (
                center[0] + (rx * radius + wobble) * math.cos(angle),
                center[1] + (ry * radius + wobble) * math.sin(angle),
            )

    place_blob(benign_ids, (cx - 0.62, cy - 0.03), 0.28, 0.46, 0.45)
    place_blob(suspicious_ids, (cx + 0.68, cy + 0.05), 0.30, 0.42, 1.10)
    pos[ego] = (cx + 0.63, cy + 0.03)

    bhv_offsets = [
        (-0.18, 0.23),
        (0.03, 0.28),
        (0.21, 0.18),
        (0.26, -0.02),
        (0.11, -0.23),
        (-0.10, -0.25),
        (-0.25, -0.08),
        (-0.25, 0.11),
        (0.01, 0.03),
        (0.24, -0.24),
    ]
    j_susp = 0
    for node_id in layout["bhv_ids"]:
        if layout["labels"][node_id] == 0:
            pos[node_id] = (cx - 0.66, cy - 0.48)
            continue
        dx, dy = bhv_offsets[j_susp % len(bhv_offsets)]
        pos[node_id] = (cx + 0.70 + dx, cy + 0.05 + dy)
        j_susp += 1
    return pos


def class_separated_tsne_positions(layout: dict, cx: float, cy: float) -> dict[int, tuple[float, float]]:
    """Display the actual behavioral t-SNE geometry as two readable class islands."""
    ego = layout["ego"]
    benign_ids = [i for i in layout["all_ids"] if i != ego and layout["labels"][i] == 0]
    suspicious_ids = [i for i in layout["all_ids"] if layout["labels"][i] == 1]
    pos: dict[int, tuple[float, float]] = {}

    def place_group(node_ids: list[int], center: tuple[float, float], sx: float, sy: float) -> None:
        arr = np.array([layout["raw"][i] for i in node_ids], dtype=float)
        arr = arr - arr.mean(axis=0, keepdims=True)
        span = np.percentile(np.abs(arr), 88)
        if span <= 1e-9:
            span = np.max(np.abs(arr)) or 1.0
        arr = np.clip(arr / span, -1.10, 1.10)
        for j, node_id in enumerate(node_ids):
            pos[node_id] = (center[0] + sx * arr[j, 0], center[1] + sy * arr[j, 1])

    place_group(benign_ids, (cx - 0.56, cy - 0.02), 0.28, 0.43)
    place_group(suspicious_ids, (cx + 0.56, cy + 0.03), 0.30, 0.43)
    return pos


def project_3d(cx: float, cy: float, x: float, y: float, z: float) -> tuple[float, float]:
    return (
        cx + 0.58 * x + 0.24 * y + 0.20 * z,
        cy - 0.28 * x + 0.50 * y + 0.36 * z,
    )


def draw_3d_frame(ax, cx: float, cy: float) -> None:
    x0, x1 = -0.90, 0.88
    y0, y1 = -0.58, 0.60
    z0, z1 = -0.44, 0.78
    origin = project_3d(cx, cy, x0, y0, z0)
    x_end = project_3d(cx, cy, x1, y0, z0)
    y_end = project_3d(cx, cy, x0, y1, z0)
    z_end = project_3d(cx, cy, x0, y0, z1)
    xy_far = project_3d(cx, cy, x1, y1, z0)

    ax.add_patch(
        mpatches.Polygon(
            [origin, x_end, xy_far, y_end],
            closed=True,
            facecolor="white",
            edgecolor="#CBD5E1",
            linewidth=0.36,
            alpha=1.0,
            zorder=0,
        )
    )

    for t in np.linspace(x0, x1, 5)[1:-1]:
        edge(ax, project_3d(cx, cy, t, y0, z0), project_3d(cx, cy, t, y1, z0), color="#CBD5E1", lw=0.18, alpha=1.0, z=1)
    for t in np.linspace(y0, y1, 5)[1:-1]:
        edge(ax, project_3d(cx, cy, x0, t, z0), project_3d(cx, cy, x1, t, z0), color="#CBD5E1", lw=0.18, alpha=1.0, z=1)

    for end, label, dx, dy in [
        (x_end, "susp. PC", 0.03, 0.04),
        (y_end, "PC2", 0.03, 0.04),
        (z_end, "PC3", 0.04, 0.03),
    ]:
        ax.add_patch(
            mpatches.FancyArrowPatch(
                origin,
                end,
                arrowstyle="-|>",
                mutation_scale=5.6,
                linewidth=0.62,
                color="#64748B",
                alpha=1.0,
                zorder=5,
            )
        )
        text(ax, end[0] + dx, end[1] + dy, label, size=5.8, color=C_TEXT)


def class_separated_pca3d_positions(
    layout: dict,
    cx: float,
    cy: float,
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Display 3D behavioral PCA as class-separated perspective islands."""
    ego = layout["ego"]
    benign_ids = [i for i in layout["all_ids"] if i != ego and layout["labels"][i] == 0]
    suspicious_ids = [i for i in layout["all_ids"] if layout["labels"][i] == 1]
    coords: dict[int, tuple[float, float, float]] = {}

    def place_group(node_ids: list[int], center: tuple[float, float, float], sx: float, sy: float, sz: float) -> None:
        arr = np.array([layout["raw3d"][i] for i in node_ids], dtype=float)
        arr = arr - arr.mean(axis=0, keepdims=True)
        span = np.percentile(np.abs(arr), 88)
        if span <= 1e-9:
            span = np.max(np.abs(arr)) or 1.0
        arr = np.clip(arr / span, -1.10, 1.10)
        for j, node_id in enumerate(node_ids):
            coords[node_id] = (
                center[0] + sx * arr[j, 0],
                center[1] + sy * arr[j, 1],
                center[2] + sz * arr[j, 2],
            )

    place_group(benign_ids, (-0.76, 0.18, -0.28), 0.36, 0.44, 0.18)
    place_group(suspicious_ids, (0.80, -0.16, 0.40), 0.38, 0.42, 0.20)
    pos = {node_id: project_3d(cx, cy, *xyz) for node_id, xyz in coords.items()}
    depth = {node_id: xyz[0] - 0.35 * xyz[1] + 0.55 * xyz[2] for node_id, xyz in coords.items()}
    return pos, depth


def pca3d_display_coords(layout: dict) -> dict[int, np.ndarray]:
    node_ids = layout["all_ids"]
    arr = np.array([layout["raw3d"][i] for i in node_ids], dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    span = np.percentile(np.abs(arr), 92)
    if span <= 1e-9:
        span = np.max(np.abs(arr)) or 1.0
    arr = np.clip(arr / span, -1.25, 1.25)
    arr[:, 0] *= 1.12
    arr[:, 1] *= 0.86
    arr[:, 2] *= 1.02
    return {node_id: arr[j] for j, node_id in enumerate(node_ids)}


def add_3d_group_ellipse(ax3, pts: np.ndarray) -> None:
    if len(pts) < 3:
        return
    center = pts.mean(axis=0)
    centered = pts - center
    cov = centered.T @ centered / max(len(pts) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    u = vecs[:, order[0]]
    v = vecs[:, order[1]]
    proj_u = centered @ u
    proj_v = centered @ v
    radius_u = max(float(np.max(np.abs(proj_u))) + 0.30, 0.42)
    radius_v = max(float(np.max(np.abs(proj_v))) + 0.24, 0.32)
    theta = np.linspace(0, 2 * np.pi, 96)
    ring = center + np.outer(np.cos(theta) * radius_u, u) + np.outer(np.sin(theta) * radius_v, v)
    poly = Poly3DCollection([ring], facecolors=[(0.98, 0.68, 0.70, 0.26)], edgecolors="none", zorder=1)
    ax3.add_collection3d(poly)
    ax3.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=C_SUSP, linewidth=0.42, alpha=0.42, zorder=2)


def add_pca3d_panel(fig: plt.Figure, base_ax, layout: dict, cx: float, *, with_edges: bool) -> None:
    bbox = base_ax.get_position()
    x0, x1 = base_ax.get_xlim()
    y0, y1 = base_ax.get_ylim()
    left_data = cx - 1.53
    bottom_data = 0.74
    width_data = 3.06
    height_data = 2.19
    rect = [
        bbox.x0 + bbox.width * ((left_data - x0) / (x1 - x0)),
        bbox.y0 + bbox.height * ((bottom_data - y0) / (y1 - y0)),
        bbox.width * (width_data / (x1 - x0)),
        bbox.height * (height_data / (y1 - y0)),
    ]
    ax3 = fig.add_axes(rect, projection="3d")
    ax3.set_facecolor((1, 1, 1, 0))

    coords = pca3d_display_coords(layout)
    ego = layout["ego"]
    bhv_set = set(layout["bhv_ids"])
    labels = layout["labels"]

    if with_edges:
        group_nodes = [node_id for node_id in layout["bhv_ids"] if int(labels[node_id]) == 1 and node_id in coords]
        add_3d_group_ellipse(ax3, np.array([coords[node_id] for node_id in group_nodes]))
        for node_id in layout["bhv_ids"]:
            a = coords[ego]
            b = coords[node_id]
            ax3.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color=C_BHV_EDGE,
                linewidth=0.45,
                linestyle="--",
                alpha=0.75,
                zorder=2,
            )
        for u, v in layout["rep"]["bhv_induced_edges"]:
            u = int(u)
            v = int(v)
            if u in coords and v in coords and u != ego and v != ego:
                a = coords[u]
                b = coords[v]
                ax3.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color=C_BHV_EDGE,
                    linewidth=0.30,
                    linestyle="--",
                    alpha=0.45,
                    zorder=1,
                )

    for label, color, marker, size, nodes in [
        (0, C_BENIGN, "o", 14, [i for i in layout["all_ids"] if i != ego and labels[i] == 0]),
        (1, C_SUSP, "o", 20, [i for i in layout["all_ids"] if i != ego and labels[i] == 1]),
    ]:
        if not nodes:
            continue
        pts = np.array([coords[i] for i in nodes])
        ax3.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=color,
            s=size,
            marker=marker,
            edgecolors="white",
            linewidths=0.35,
            depthshade=False,
            zorder=4,
        )

    e = coords[ego]
    ax3.scatter(
        [e[0]],
        [e[1]],
        [e[2]],
        c=C_EGO,
        s=36,
        marker="o",
        edgecolors="#111827",
        linewidths=0.75,
        depthshade=False,
        zorder=8,
    )

    lim = 1.28
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_zlim(-1.12, 1.12)
    ax3.view_init(elev=18, azim=-60)
    ax3.set_box_aspect((1.25, 1.0, 0.95))
    ax3.set_xlabel("Suspicious ->", fontsize=6.2, labelpad=-17, color=C_TEXT)
    ax3.set_ylabel("")
    ax3.set_zlabel("")
    ax3.set_xticks([-1.0, 0.0, 1.0])
    ax3.set_yticks([-1.0, 0.0, 1.0])
    ax3.set_zticks([-0.8, 0.0, 0.8])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    ax3.tick_params(axis="both", which="major", labelsize=4.4, pad=-3, length=0, width=0, colors=(0, 0, 0, 0))

    for axis in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor("#B8C0CC")
        axis.pane.set_linewidth(0.30)
        axis.line.set_color((0.28, 0.32, 0.36, 0.0))
        axis.line.set_linewidth(0.0)
        axis._axinfo["grid"]["color"] = (0.58, 0.63, 0.69, 0.72)
        axis._axinfo["grid"]["linewidth"] = 0.24
        axis._axinfo["axisline"]["color"] = (0.28, 0.32, 0.36, 0.0)
        axis._axinfo["axisline"]["linewidth"] = 0.0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
        axis._axinfo["tick"]["color"] = (0, 0, 0, 0)
    ax3.grid(True)


def add_halo(
    ax,
    pts: np.ndarray,
    *,
    facecolor: str | None = C_SOFT_RED,
    edgecolor: str | None = None,
    dashed: bool = False,
    alpha: float = 0.36,
    pad: tuple[float, float] = (0.22, 0.16),
    z: int = 0,
) -> None:
    center = pts.mean(axis=0)
    width = max(0.60, float(pts[:, 0].max() - pts[:, 0].min()) + pad[0])
    height = max(0.42, float(pts[:, 1].max() - pts[:, 1].min()) + pad[1])
    ax.add_patch(
        mpatches.Ellipse(
            center,
            width,
            height,
            facecolor=facecolor if facecolor else "none",
            edgecolor=edgecolor if edgecolor else "none",
            linestyle=(0, (3, 2)) if dashed else "-",
            linewidth=0.82 if edgecolor else 0.0,
            alpha=alpha,
            zorder=z,
        )
    )


def draw_nodes(ax, layout: dict, pos: dict[int, tuple[float, float]], *, faded: set[int] | None = None, small: bool = False) -> None:
    faded = faded or set()
    for node_id, (x, y) in pos.items():
        is_ego = node_id == layout["ego"]
        label = layout["labels"][node_id]
        alpha = 0.24 if node_id in faded else 1.0
        r = 0.035 if small else 0.045
        if is_ego:
            r = 0.075 if not small else 0.052
        node(
            ax,
            x,
            y,
            node_color(label, ego=is_ego),
            r=r,
            label="E" if is_ego else None,
            edge="#111827" if is_ego else "white",
            lw=0.65 if is_ego else 0.34,
            alpha=alpha,
            z=16 if is_ego else 10,
        )


def tx_context_positions(layout: dict, cx: float, cy: float, *, scale: float = 1.0) -> dict[int, tuple[float, float]]:
    """Place the transaction view as a compact radial ego graph."""
    pos: dict[int, tuple[float, float]] = {layout["ego"]: (cx, cy)}
    suspicious_tx = [node_id for node_id in layout["tx_ids"] if layout["labels"][node_id] == 1]
    regular_tx = [node_id for node_id in layout["tx_ids"] if node_id not in suspicious_tx]

    for j, node_id in enumerate(regular_tx):
        angle = 2 * math.pi * j / max(len(regular_tx), 1) + 0.18
        radius = (0.55 + 0.12 * (j % 2)) * scale
        pos[node_id] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    for j, node_id in enumerate(suspicious_tx):
        angle = -0.42 + 0.38 * j
        radius = 0.78 * scale
        pos[node_id] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    by_bridge: dict[int, list[int]] = {}
    for bridge, context in layout["tx_context_edges"]:
        by_bridge.setdefault(bridge, []).append(context)
    for bridge, nodes in by_bridge.items():
        bx, by = pos.get(bridge, (cx + 0.45 * scale, cy))
        base_angle = math.atan2(by - cy, bx - cx)
        for j, node_id in enumerate(nodes):
            offset = (-0.18 if j == 0 else 0.18) + 0.08 * j
            radius = 0.23 * scale
            pos[node_id] = (
                bx + radius * math.cos(base_angle + offset),
                by + radius * math.sin(base_angle + offset),
            )
    return pos


def draw_tx_graph(ax, layout: dict, cx: float, cy: float) -> None:
    stage_title(ax, cx, "3.1 Problem\nSetup", "observed Tx topology")
    egos = [
        (cx - 0.72, cy - 0.03, 0.074, "E"),
        (cx - 0.05, cy + 0.44, 0.052, ""),
        (cx + 0.34, cy - 0.28, 0.052, ""),
        (cx + 0.59, cy + 0.19, 0.046, ""),
    ]
    benign = [
        (cx - 0.48, cy + 0.42), (cx - 0.34, cy + 0.54), (cx - 0.18, cy + 0.30),
        (cx + 0.12, cy + 0.62), (cx + 0.31, cy + 0.48), (cx + 0.50, cy + 0.55),
        (cx - 0.54, cy + 0.12), (cx - 0.37, cy + 0.04), (cx - 0.17, cy + 0.12),
        (cx + 0.05, cy + 0.20), (cx + 0.27, cy + 0.12), (cx + 0.49, cy + 0.02),
        (cx - 0.46, cy - 0.18), (cx - 0.25, cy - 0.26), (cx - 0.04, cy - 0.16),
        (cx + 0.16, cy - 0.08), (cx + 0.42, cy - 0.05), (cx + 0.64, cy - 0.12),
        (cx - 0.58, cy - 0.43), (cx - 0.34, cy - 0.52), (cx - 0.10, cy - 0.47),
        (cx + 0.14, cy - 0.56), (cx + 0.41, cy - 0.50), (cx + 0.63, cy - 0.42),
        (cx + 0.76, cy + 0.43), (cx + 0.84, cy - 0.31),
    ]
    suspicious = [(cx + 0.22, cy + 0.31), (cx + 0.73, cy + 0.05), (cx - 0.06, cy - 0.40)]

    edge_pairs = [
        (egos[0][:2], benign[6]), (egos[0][:2], benign[7]), (egos[0][:2], benign[12]),
        (egos[0][:2], benign[13]), (egos[0][:2], suspicious[0]),
        (egos[1][:2], benign[0]), (egos[1][:2], benign[2]), (egos[1][:2], benign[4]),
        (egos[1][:2], suspicious[0]), (egos[2][:2], benign[15]), (egos[2][:2], benign[18]),
        (egos[2][:2], benign[21]), (egos[2][:2], suspicious[2]), (egos[3][:2], benign[10]),
        (egos[3][:2], benign[16]), (egos[3][:2], benign[24]), (egos[3][:2], suspicious[1]),
        (benign[0], benign[1]), (benign[1], benign[3]), (benign[3], benign[5]),
        (benign[2], benign[8]), (benign[8], benign[9]), (benign[9], benign[10]),
        (benign[10], benign[11]), (benign[12], benign[13]), (benign[13], benign[14]),
        (benign[14], benign[15]), (benign[15], benign[16]), (benign[16], benign[17]),
        (benign[18], benign[19]), (benign[19], benign[20]), (benign[20], benign[21]),
        (benign[21], benign[22]), (benign[22], benign[23]), (benign[5], benign[24]),
        (benign[17], benign[25]), (benign[4], suspicious[0]), (benign[11], suspicious[1]),
    ]
    for a, b in edge_pairs:
        edge(ax, a, b, color=C_TX_EDGE, lw=0.31, alpha=0.56, rad=0.05 * math.sin((a[0] + b[1]) * 8.0))

    for x, y in benign:
        node(ax, x, y, C_BENIGN, r=0.026, lw=0.18, alpha=0.90, z=9)
    for x, y in suspicious:
        node(ax, x, y, C_SUSP, r=0.040, lw=0.24, alpha=0.98, z=12)
    for x, y, radius, label in egos[1:]:
        node(ax, x, y, C_EGO, r=radius, label=label, edge="#111827", lw=0.48, z=15)
    node(ax, egos[0][0], egos[0][1], C_EGO, r=egos[0][2], label="E", edge="#111827", lw=0.72, z=16)
    text(ax, cx, PANEL_VALUE_Y, "1/26", size=PANEL_VALUE_SIZE, weight="bold")
    text(ax, cx, PANEL_NOTE_Y, "focal 1-hop suspicious", size=PANEL_NOTE_SIZE, color=C_MUTED)


def draw_behavior_space(ax, layout: dict, cx: float, cy: float) -> None:
    stage_title(ax, cx, "3.2 Behavior\nView", r"$X^{bhv}$ feature space")
    text(ax, cx, PANEL_VALUE_Y, "no edges yet", size=PANEL_VALUE_SMALL_SIZE, weight="bold")
    text(ax, cx, PANEL_NOTE_Y, "suspicious-side PC", size=PANEL_NOTE_SIZE, color=C_MUTED)


def draw_knn_candidates(ax, layout: dict, cx: float, cy: float) -> None:
    stage_title(ax, cx, "3.3 Graph\nRecovery", "behaviorally similar neighborhood")
    text(ax, cx, PANEL_VALUE_Y, "top-10", size=PANEL_VALUE_SMALL_SIZE, weight="bold")
    text(ax, cx, PANEL_NOTE_Y, "recovered neighbors", size=PANEL_NOTE_SIZE, color=C_MUTED)


def draw_repaired_graph(ax, layout: dict, cx: float, cy: float) -> None:
    stage_title(ax, cx, "3.4 GNN\nEncoding", r"encode $\mathcal{G}_{bhv}$ branch")
    node_ids = [layout["ego"]] + layout["bhv_ids"]
    pos = panel_positions(layout, node_ids, cx, cy, scale=0.74)
    ego = layout["ego"]
    for j, node_id in enumerate(layout["bhv_ids"]):
        edge(ax, pos[ego], pos[node_id], color=C_BHV_EDGE, lw=0.55, alpha=0.62, dashed=True, rad=0.05 * ((j % 3) - 1))
    for u, v in layout["rep"]["bhv_induced_edges"]:
        u = int(u)
        v = int(v)
        if u in pos and v in pos and u != ego and v != ego:
            edge(ax, pos[u], pos[v], color=C_BHV_EDGE, lw=0.30, alpha=0.25, dashed=True, rad=0.14)
    draw_nodes(ax, layout, pos)
    text(ax, cx, PANEL_VALUE_Y, "9/10", size=PANEL_VALUE_SIZE, weight="bold")
    text(ax, cx, PANEL_NOTE_Y, "suspicious neighbors", size=PANEL_NOTE_SIZE, color=C_MUTED)


def draw_pooling(ax, layout: dict, cx: float, cy: float) -> None:
    stage_title(ax, cx, "3.5 Subgraph\nPooling", r"repaired $\mathcal{N}(v)$", title_size=9.6)
    node_ids = [layout["ego"]] + layout["bhv_ids"]
    pos = panel_positions(layout, node_ids, cx - 0.50, cy + 0.02, scale=0.44)
    ego = layout["ego"]
    pts = np.array([pos[i] for i in node_ids])
    add_halo(ax, pts, facecolor=C_SOFT_RED, edgecolor="#E7A4AA", alpha=0.72, pad=(0.28, 0.18), z=0)
    for node_id in layout["bhv_ids"]:
        edge(ax, pos[ego], pos[node_id], color=C_BHV_EDGE, lw=0.28, alpha=0.28, dashed=True)
    for u, v in layout["rep"]["bhv_induced_edges"]:
        u = int(u)
        v = int(v)
        if u in pos and v in pos and u != ego and v != ego:
            edge(ax, pos[u], pos[v], color=C_BHV_EDGE, lw=0.22, alpha=0.22, dashed=True, rad=0.12)
    draw_nodes(ax, layout, pos, small=True)

    pool_x = cx + 0.28
    pool = mpatches.FancyBboxPatch(
        (pool_x - 0.18, cy - 0.17),
        0.36,
        0.34,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor="#FFFFFF",
        edgecolor=C_BHV_EDGE,
        linewidth=0.62,
        zorder=28,
    )
    ax.add_patch(pool)
    text(ax, pool_x, cy + 0.03, r"$\Sigma$", size=9.5, weight="bold")
    text(ax, pool_x, cy - 0.10, "pool", size=6.3, color=C_MUTED)
    for start in [(cx - 0.14, cy + 0.22), (cx - 0.08, cy + 0.02), (cx - 0.14, cy - 0.20)]:
        arrow(ax, start, (pool_x - 0.20, cy + 0.42 * (start[1] - cy)), color=C_BHV_EDGE, lw=0.42)
    arrow(ax, (pool_x + 0.20, cy), (cx + 0.74, cy), color=C_BHV_EDGE, lw=0.82)
    out_x = cx + 1.18
    text(ax, out_x, cy + 0.045, r"$s_i^{bhv}$", size=8.2, color=C_POOL_TEXT, weight="bold")
    text(ax, out_x, cy - 0.085, "pooled vector", size=6.4, color=C_POOL_TEXT)
    text(ax, cx, PANEL_VALUE_Y, "90%", size=PANEL_VALUE_SMALL_SIZE, weight="bold")
    text(ax, cx, PANEL_NOTE_Y, "pooled suspicious\nsubgraph", size=PANEL_NOTE_SIZE, color=C_MUTED)


def draw_alignment(ax, layout: dict, cx: float, cy: float) -> None:
    stage_title(ax, cx, "3.6 Training\n& Inference", "BYOL loss -> concat", title_size=9.6)
    ego = layout["ego"]

    def widen(pos: dict[int, tuple[float, float]], center_x: float, factor: float) -> dict[int, tuple[float, float]]:
        return {node_id: (center_x + factor * (x - center_x), y) for node_id, (x, y) in pos.items()}

    tx_pos = tx_context_positions(layout, cx - 0.48, cy + 0.26, scale=0.46)
    tx_pos = widen(tx_pos, cx - 0.48, 1.28)
    for node_id in layout["tx_ids"]:
        edge(ax, tx_pos[ego], tx_pos[node_id], color=C_TX_EDGE, lw=0.18, alpha=0.35)
    for bridge, context in layout["tx_context_edges"]:
        edge(ax, tx_pos[bridge], tx_pos[context], color=C_TX_EDGE, lw=0.20, alpha=0.46)
    for node_id in layout["tx_context_ids"]:
        node(ax, *tx_pos[node_id], C_BENIGN, r=0.014, lw=0.10, alpha=0.55, z=7)
    for node_id in layout["tx_ids"]:
        node(ax, *tx_pos[node_id], node_color(layout["labels"][node_id]), r=0.019, lw=0.13, alpha=0.90, z=9)
    node(ax, *tx_pos[ego], C_EGO, r=0.033, label="E", edge="#111827", lw=0.34, z=12)

    bhv_nodes = [ego] + layout["bhv_ids"]
    bhv_pos = panel_positions(layout, bhv_nodes, cx - 0.47, cy - 0.36, scale=0.38)
    bhv_pos = widen(bhv_pos, cx - 0.47, 1.22)
    for node_id in bhv_nodes[1:]:
        edge(ax, bhv_pos[ego], bhv_pos[node_id], color=C_BHV_EDGE, lw=0.25, alpha=0.42, dashed=True)
    draw_nodes(ax, layout, bhv_pos, small=True)

    branch_pts = np.array(list(tx_pos.values()) + list(bhv_pos.values()))
    bx0, by0 = branch_pts.min(axis=0)
    bx1, by1 = branch_pts.max(axis=0)
    pad_x, pad_y = 0.23, 0.08
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (bx0 - pad_x, by0 - pad_y),
            (bx1 - bx0) + 2 * pad_x,
            (by1 - by0) + 2 * pad_y,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor="none",
            edgecolor=C_BYOL,
            linestyle=(0, (3, 2)),
            linewidth=0.88,
            zorder=28,
        )
    )
    text(ax, (bx0 + bx1) / 2, by0 - pad_y - 0.10, "BYOL loss", size=6.7, color=C_BYOL_DARK, weight="bold")

    concat_x = cx + 0.60
    text(ax, concat_x, cy + 0.20, "CONCAT", size=7.5, color=C_TEXT, weight="bold")
    text(ax, concat_x, cy - 0.005, r"$\Vert$", size=11.2, color=C_TEXT, weight="bold")
    arrow(ax, (cx + 0.08, cy + 0.14), (concat_x - 0.12, cy + 0.055), color=C_ARROW, lw=0.68)
    arrow(ax, (cx + 0.08, cy - 0.30), (concat_x - 0.12, cy - 0.055), color=C_ARROW, lw=0.68)
    arrow(ax, (concat_x + 0.10, cy), (cx + 1.05, cy), color=C_ARROW, lw=0.80)

    final_x = cx + 1.36
    text(ax, final_x, cy + 0.045, "$z_i$", size=9.4, color=C_FINAL_TEXT, weight="bold")
    text(ax, final_x, cy - 0.085, "final vector", size=6.4, color=C_FINAL_TEXT)
    text(ax, cx, PANEL_VALUE_Y, "$z_i=[s_i^{tx}\\Vert s_i^{bhv}]$", size=10.3, weight="bold")
    text(ax, cx, PANEL_NOTE_Y, "final ranking\nrepresentation", size=PANEL_NOTE_SIZE, color=C_MUTED)


def build_figure(layout: dict) -> plt.Figure:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.8, "figure.dpi": 300, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(10.25, 3.18))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 19.8)
    ax.set_ylim(0, 3.06)
    ax.axis("off")

    centers = [1.42, 4.70, 7.98, 11.26, 14.54, 17.82]
    for draw, cx in zip(
        [draw_tx_graph, draw_behavior_space, draw_knn_candidates, draw_repaired_graph, draw_pooling, draw_alignment],
        centers,
    ):
        draw(ax, layout, cx, 1.62)

    handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_EGO, markeredgecolor="#111827", markersize=7.2, label="ego"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_SUSP, markeredgecolor="white", markersize=7.2, label="suspicious"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=C_BENIGN, markeredgecolor="white", markersize=7.2, label="benign"),
        mlines.Line2D([], [], color=C_TX_EDGE, linewidth=1.0, label="transaction edge"),
        mlines.Line2D([], [], color=C_BHV_EDGE, linewidth=1.0, linestyle="--", label="behavioral kNN edge"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=5,
        frameon=False,
        fontsize=8.8,
        handlelength=1.35,
        handletextpad=0.34,
        columnspacing=0.82,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=1.0, bottom=0.085)
    add_pca3d_panel(fig, ax, layout, centers[1], with_edges=False)
    add_pca3d_panel(fig, ax, layout, centers[2], with_edges=True)
    return fig


def main() -> None:
    rep = load_representative()
    layout = build_case_layout(rep)
    fig = build_figure(layout)
    save_all(fig, "fig_topology_stages")
    plt.close(fig)


if __name__ == "__main__":
    main()

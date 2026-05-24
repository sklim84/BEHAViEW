from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    px = None
    go = None


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
CASE_STUDY_PATH = BASE_DIR / "results" / "rq1" / "case_study" / "representative.json"

DATASETS = {
    "ATNET": {
        "node_path": BASE_DIR / "datasets" / "atnet" / "ATNET_NODE_FEAT.csv",
        "edge_path": BASE_DIR / "datasets" / "atnet" / "ATNET_EDGES.csv",
        "ablation_path": BASE_DIR / "results" / "rq1" / "main_sweep.csv",
        "supervised_path": BASE_DIR / "results" / "rq3" / "supervised_atnet.csv",
        "extra_paths": [
            BASE_DIR / "results" / "rq3" / "consisgad_atnet.csv",
            BASE_DIR / "results" / "rq3" / "caregnn_pcgnn_atnet.csv",
            BASE_DIR / "results" / "rq3" / "bwgnn_gaga_atnet.csv",
        ],
        "node_arg": "atnet/ATNET_NODE_FEAT",
        "edge_arg": "atnet/ATNET_EDGES",
        "knn_prefix": "atnet/ATNET_KNN_BEHAV",
        "model_prefix": "atn",
    },
    "AMLworld": {
        "node_path": BASE_DIR / "datasets" / "amlworld" / "AMLWORLD_NODE_FEAT.csv",
        "edge_path": BASE_DIR / "datasets" / "amlworld" / "AMLWORLD_EDGES.csv",
        "ablation_path": BASE_DIR / "results" / "rq4" / "amlworld_main_sweep.csv",
        "supervised_path": BASE_DIR / "results" / "rq3" / "supervised_amlworld.csv",
        "extra_paths": [
            BASE_DIR / "results" / "rq3" / "consisgad_amlworld.csv",
            BASE_DIR / "results" / "rq3" / "caregnn_pcgnn_amlworld.csv",
            BASE_DIR / "results" / "rq3" / "bwgnn_gaga_amlworld.csv",
        ],
        "node_arg": "amlworld/AMLWORLD_NODE_FEAT",
        "edge_arg": "amlworld/AMLWORLD_EDGES",
        "knn_prefix": "amlworld/AMLWORLD_KNN_BEHAV",
        "model_prefix": "aml",
    },
}

STRUCTURAL_COLS = [
    "dc",
    "in_dc",
    "out_dc",
    "pagerank",
    "hits_hub",
    "hits_auth",
    "kcore",
    "triangle",
    "betweenness",
]
BEHAVIOR_EXCLUDE = {
    "dc",
    "in_dc",
    "out_dc",
    "in_count",
    "out_count",
    "pagerank",
    "hits_hub",
    "hits_auth",
    "kcore",
    "triangle",
    "betweenness",
}

ENCODERS = [
    "gbt",
    "dgi_bn",
    "mvgrl_bn",
    "grace_bn",
    "bgrl",
    "gin",
    "gca",
    "dgi",
    "mvgrl",
    "grace",
]
ENCODER_LABELS = {
    "gbt": "GBT",
    "dgi_bn": "DGI+BN",
    "mvgrl_bn": "MVGRL+BN",
    "grace_bn": "GRACE+BN",
    "bgrl": "BGRL",
    "gin": "GIN",
    "gca": "GCA",
    "dgi": "DGI",
    "mvgrl": "MVGRL",
    "grace": "GRACE",
}
DEFAULT_LOSS = {
    "gbt": "BarlowTwins",
    "dgi_bn": "JSD",
    "mvgrl_bn": "JSD",
    "grace_bn": "InfoNCE",
    "bgrl": "BootstrapLatent",
    "gin": "InfoNCE",
    "gca": "InfoNCE",
    "dgi": "JSD",
    "mvgrl": "JSD",
    "grace": "InfoNCE",
}
LOSS_CHOICES = ["BootstrapLatent", "InfoNCE", "BarlowTwins", "JSD", "None"]

SETTINGS = {
    "a": {
        "label": "(a) Transaction topology + node representation",
        "short": "Transaction/Node",
        "use_knn": False,
        "use_pool": False,
        "ours": False,
    },
    "b": {
        "label": "(b) Repaired k-NN topology + node representation",
        "short": "Repaired/Node",
        "use_knn": True,
        "use_pool": False,
        "ours": False,
    },
    "c": {
        "label": "(c) Transaction topology + subgraph pooling",
        "short": "Transaction/Pooling",
        "use_knn": False,
        "use_pool": True,
        "ours": False,
    },
    "d": {
        "label": "(d) BehaView: repaired topology + subgraph pooling",
        "short": "Repaired/Pooling",
        "use_knn": True,
        "use_pool": True,
        "ours": True,
    },
}

C_BENIGN = "#4E79A7"
C_SUSPICIOUS = "#D64045"
C_EGO = "#F0A000"
C_TX_EDGE = "#5F6F7D"
C_BHV_EDGE = "#B66A3C"
C_MAP_EDGE = "rgba(107, 114, 128, 0.28)"
C_BOX = "rgba(148, 163, 184, 0.45)"


st.set_page_config(
    page_title="BehaView 2D method comparator",
    page_icon="B",
    layout="wide",
)


def _format_int(value: int | float) -> str:
    return f"{int(value):,}"


def _ratio_string(ss: int, sb: int) -> str:
    if ss == 0:
        return f"0 : {sb:,}"
    if sb == 0:
        return f"{ss:,} : 0"
    return f"1 : {sb / ss:.1f}"


@st.cache_data(show_spinner=False)
def load_nodes(dataset_name: str) -> pd.DataFrame:
    path = DATASETS[dataset_name]["node_path"]
    df = pd.read_csv(path)
    df["account"] = df["account"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_ablation_results(dataset_name: str) -> pd.DataFrame:
    path = DATASETS[dataset_name]["ablation_path"]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Model" not in df.columns:
        return pd.DataFrame()

    def parse_model(model_name: str) -> tuple[str | None, str | None, int | None]:
        match = re.match(r"^[^_]+_(?P<encoder>.+)_(?P<setting>[abcd])_s(?P<seed>\d+)$", model_name)
        if not match:
            return None, None, None
        return match.group("encoder"), match.group("setting"), int(match.group("seed"))

    parsed = df["Model"].astype(str).map(parse_model)
    df[["encoder", "setting", "seed"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df = df.dropna(subset=["encoder", "setting"])
    return df


@st.cache_data(show_spinner=False)
def load_comparison_results(dataset_name: str) -> pd.DataFrame:
    rows = []
    ablation = load_ablation_results(dataset_name)
    if len(ablation) > 0:
        for (encoder, setting), grp in ablation.groupby(["encoder", "setting"]):
            rows.append(
                {
                    "family": "Self-supervised GCL",
                    "method": f"{ENCODER_LABELS.get(encoder, encoder.upper())} {SETTINGS[setting]['short']}",
                    "encoder": encoder,
                    "setting": setting,
                    "f1_mean": grp["f1_1"].mean(),
                    "f1_std": grp["f1_1"].std(ddof=0),
                    "auroc": grp["auroc"].mean() if "auroc" in grp else np.nan,
                    "auprc": grp["auprc"].mean() if "auprc" in grp else np.nan,
                    "n": len(grp),
                }
            )

    supervised = DATASETS[dataset_name]["supervised_path"]
    if supervised.exists():
        df = pd.read_csv(supervised)
        if {"model", "f1_1"}.issubset(df.columns):
            for model, grp in df.groupby("model"):
                rows.append(
                    {
                        "family": "Supervised/tabular baseline",
                        "method": model.upper(),
                        "encoder": model,
                        "setting": "",
                        "f1_mean": grp["f1_1"].mean(),
                        "f1_std": grp["f1_1"].std(ddof=0),
                        "auroc": grp["auroc"].mean() if "auroc" in grp else np.nan,
                        "auprc": grp["auprc"].mean() if "auprc" in grp else np.nan,
                        "n": len(grp),
                    }
                )

    for path in DATASETS[dataset_name]["extra_paths"]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"model", "f1_1"}.issubset(df.columns):
            for model, grp in df.groupby("model"):
                rows.append(
                    {
                        "family": "Existing graph baseline",
                        "method": str(model).upper(),
                        "encoder": model,
                        "setting": "",
                        "f1_mean": grp["f1_1"].mean(),
                        "f1_std": grp["f1_1"].std(ddof=0),
                        "auroc": grp["auroc"].mean() if "auroc" in grp else np.nan,
                        "auprc": grp["auprc"].mean() if "auprc" in grp else np.nan,
                        "n": len(grp),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False)


def behavior_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in BEHAVIOR_EXCLUDE:
            continue
        if col.startswith(("out_", "in_", "md_", "fnd_")) or col.endswith("_entropy") or col.startswith("entropy"):
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
    return cols


def selected_feature_columns(df: pd.DataFrame, feature_family: str) -> list[str]:
    behav = behavior_columns(df)
    struct = [c for c in STRUCTURAL_COLS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if feature_family == "Behavioral":
        return behav
    if feature_family == "Structural":
        return struct if struct else behav
    return list(dict.fromkeys(behav + struct))


@st.cache_data(show_spinner=False)
def make_sample(dataset_name: str, n_points: int, suspicious_share: float, seed: int) -> pd.DataFrame:
    df = load_nodes(dataset_name)
    n_points = min(int(n_points), len(df))
    labels = df["label"].to_numpy()
    rng = np.random.default_rng(seed)
    susp_idx = np.flatnonzero(labels == 1)
    benign_idx = np.flatnonzero(labels == 0)

    target_susp = int(round(n_points * suspicious_share))
    target_susp = min(target_susp, len(susp_idx))
    target_benign = n_points - target_susp
    if target_benign > len(benign_idx):
        target_benign = len(benign_idx)
        target_susp = min(n_points - target_benign, len(susp_idx))

    sampled_susp = rng.choice(susp_idx, size=target_susp, replace=False) if target_susp > 0 else np.array([], dtype=int)
    sampled_benign = rng.choice(benign_idx, size=target_benign, replace=False) if target_benign > 0 else np.array([], dtype=int)
    sampled = np.concatenate([sampled_susp, sampled_benign])
    rng.shuffle(sampled)

    out = df.iloc[sampled].copy()
    out.insert(0, "global_idx", sampled.astype(np.int64))
    out.insert(1, "sample_id", np.arange(len(out), dtype=np.int64))
    return out.reset_index(drop=True)


@st.cache_data(show_spinner="Scanning transaction edges for the sampled nodes...")
def transaction_edges_for_sample(dataset_name: str, account_tuple: tuple[str, ...]) -> np.ndarray:
    edge_path = DATASETS[dataset_name]["edge_path"]
    if not edge_path.exists() or len(account_tuple) == 0:
        return np.empty((0, 2), dtype=np.int64)

    account_set = set(account_tuple)
    account_to_local = {account: i for i, account in enumerate(account_tuple)}
    edge_chunks: list[np.ndarray] = []

    for chunk in pd.read_csv(edge_path, usecols=["source", "target"], dtype=str, chunksize=500_000):
        mask = chunk["source"].isin(account_set) & chunk["target"].isin(account_set)
        if not bool(mask.any()):
            continue
        sub = chunk.loc[mask]
        src = sub["source"].map(account_to_local).to_numpy(dtype=np.int64)
        tgt = sub["target"].map(account_to_local).to_numpy(dtype=np.int64)
        valid = src != tgt
        if valid.any():
            edge_chunks.append(np.column_stack([src[valid], tgt[valid]]))

    if not edge_chunks:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack(edge_chunks)
    return unique_edges(edges)


def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 1), dtype=np.float64)
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    X = StandardScaler().fit_transform(X)
    return np.nan_to_num(X, copy=False)


def build_knn_edges(X: np.ndarray, k: int) -> np.ndarray:
    n = len(X)
    if n <= 1:
        return np.empty((0, 2), dtype=np.int64)
    k = int(max(1, min(k, n - 1)))
    Xn = normalize(X)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(Xn)
    _, idx = nn.kneighbors(Xn)
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    tgt = idx[:, 1 : k + 1].reshape(-1).astype(np.int64)
    return unique_edges(np.column_stack([src, tgt]))


def unique_edges(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = edges.astype(np.int64, copy=False)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(edges, axis=0)


def bidirectional_edges(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return unique_edges(np.vstack([edges, edges[:, ::-1]]))


def graph_metrics(edges: np.ndarray, labels: np.ndarray) -> dict[str, float | int | str]:
    if edges.size == 0:
        return {"edges": 0, "homophily": np.nan, "ss": 0, "sb": 0, "bb": 0, "ratio": "n/a"}
    src, tgt = edges[:, 0], edges[:, 1]
    same = labels[src] == labels[tgt]
    src_s = labels[src] == 1
    tgt_s = labels[tgt] == 1
    ss = int(np.sum(src_s & tgt_s))
    sb = int(np.sum(src_s ^ tgt_s))
    bb = int(np.sum((~src_s) & (~tgt_s)))
    return {
        "edges": int(len(edges)),
        "homophily": float(np.mean(same)),
        "ss": ss,
        "sb": sb,
        "bb": bb,
        "ratio": _ratio_string(ss, sb),
    }


def neighbor_pool(
    Z: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    variant: str,
    cycle_signal: np.ndarray | None = None,
    cycle_alpha: float = 2.0,
) -> np.ndarray:
    if edges.size == 0 or alpha <= 0:
        return Z

    n, d = Z.shape
    edges = bidirectional_edges(edges)
    src, tgt = edges[:, 0], edges[:, 1]
    weights = np.ones((len(edges), 1), dtype=np.float64)

    if variant in {"Heterophily-aware", "Cycle-aware"}:
        Zn = normalize(Z)
        weights = np.sum(Zn[src] * Zn[tgt], axis=1, keepdims=True)
        weights = np.clip(weights, 0.0, None)

    if variant == "Cycle-aware" and cycle_signal is not None:
        boost = 1.0 + float(cycle_alpha) * cycle_signal[src].reshape(-1, 1)
        weights = weights * boost

    pooled_sum = np.zeros((n, d), dtype=np.float64)
    weight_sum = np.zeros((n, 1), dtype=np.float64)
    np.add.at(pooled_sum, tgt, weights * Z[src])
    np.add.at(weight_sum, tgt, weights)

    if variant == "Cycle-aware" and cycle_signal is not None:
        self_weight = 1.0 + float(cycle_alpha) * cycle_signal.reshape(-1, 1)
    else:
        self_weight = 1.0
    pooled = (pooled_sum + self_weight * Z) / np.maximum(weight_sum + self_weight, 1e-9)
    return (1.0 - alpha) * Z + alpha * pooled


def apply_encoder_profile(Z: np.ndarray, encoder: str) -> np.ndarray:
    if encoder in {"gbt", "dgi_bn", "mvgrl_bn", "grace_bn", "gin"}:
        return StandardScaler().fit_transform(Z)
    if encoder in {"gca", "dgi", "mvgrl", "grace"}:
        return np.tanh(Z)
    if encoder == "bgrl":
        return normalize(Z)
    return Z


def apply_loss_profile(Z: np.ndarray, y: np.ndarray, loss_name: str, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if loss_name == "None" or strength <= 0:
        return Z

    if loss_name == "BootstrapLatent":
        shaped = normalize(Z)
    elif loss_name == "InfoNCE":
        axis = suspicious_axis(Z, y)
        signed = (y * 2 - 1).reshape(-1, 1)
        shaped = Z + signed * axis.reshape(1, -1) * np.std(Z)
    elif loss_name == "BarlowTwins":
        centered = Z - Z.mean(axis=0, keepdims=True)
        cov = np.atleast_2d(np.cov(centered, rowvar=False))
        vals, vecs = np.linalg.eigh(cov + np.eye(cov.shape[0]) * 1e-5)
        shaped = centered @ vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-5)))
    elif loss_name == "JSD":
        shaped = np.tanh(Z)
    else:
        shaped = Z
    shaped = np.nan_to_num(shaped)
    if shaped.ndim == 1:
        shaped = shaped.reshape(-1, 1)
    if shaped.shape != Z.shape:
        shaped = shaped[:, : Z.shape[1]]
    return (1.0 - strength) * Z + strength * shaped


def suspicious_axis(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(np.unique(y)) < 2:
        axis = np.zeros(Z.shape[1], dtype=np.float64)
        axis[0] = 1.0
    else:
        axis = Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        axis = np.zeros(Z.shape[1], dtype=np.float64)
        axis[0] = 1.0
        norm = np.linalg.norm(axis)
    return axis / max(norm, 1e-12)


def method_representation(
    X: np.ndarray,
    y: np.ndarray,
    tx_edges: np.ndarray,
    knn_edges: np.ndarray,
    setting: str,
    encoder: str,
    loss_name: str,
    graph_alpha: float,
    view_weight: float,
    pool_strength: float,
    pool_variant: str,
    cycle_signal: np.ndarray | None,
    cycle_alpha: float,
    loss_strength: float,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray, str]]]:
    Z = apply_encoder_profile(X, encoder)
    Z = apply_loss_profile(Z, y, loss_name, loss_strength)

    tx_node = neighbor_pool(Z, tx_edges, graph_alpha, "Mean")
    bhv_node = neighbor_pool(Z, knn_edges, graph_alpha, "Mean")
    spec = SETTINGS[setting]
    edge_groups: list[tuple[str, np.ndarray, str]] = []

    if spec["use_pool"]:
        tx_pooled = neighbor_pool(tx_node, tx_edges, pool_strength, pool_variant, cycle_signal, cycle_alpha)
        bhv_pooled = neighbor_pool(bhv_node, knn_edges, pool_strength, pool_variant, cycle_signal, cycle_alpha)
    else:
        tx_pooled = tx_node
        bhv_pooled = bhv_node

    if spec["use_knn"]:
        Z_out = (1.0 - view_weight) * tx_pooled + view_weight * bhv_pooled
        edge_groups.append(("Transaction edges", tx_edges, C_TX_EDGE))
        edge_groups.append(("Repaired k-NN edges", knn_edges, C_BHV_EDGE))
    else:
        Z_out = tx_pooled
        edge_groups.append(("Transaction edges", tx_edges, C_TX_EDGE))

    return np.nan_to_num(Z_out), edge_groups


def graph_edges_for_layout(edge_groups: list[tuple[str, np.ndarray, str]], max_edges: int, seed: int) -> np.ndarray:
    chunks = [edges for _, edges, _ in edge_groups if edges.size]
    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack(chunks).astype(np.int64, copy=False)
    if len(edges) > max_edges * 2:
        edges = sample_edges(edges, max_edges * 2, seed)
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    edges = unique_edges(np.column_stack([lo, hi]))
    if len(edges) > max_edges:
        edges = sample_edges(edges, max_edges, seed + 17)
    return edges


def initialize_graph_coords(Z: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(Z)
    coords = rng.normal(0.0, 1.0, size=(n, 2))
    Z = np.nan_to_num(Z)
    if Z.ndim == 2 and Z.shape[1] > 0:
        dims = min(2, Z.shape[1])
        coords[:, :dims] += StandardScaler().fit_transform(Z[:, :dims])
    coords = np.nan_to_num(coords)
    coords -= coords.mean(axis=0, keepdims=True)
    scale = np.std(coords, axis=0, keepdims=True)
    coords = coords / np.maximum(scale, 1e-6)
    return coords


def enforce_minimum_node_spacing(coords: np.ndarray, min_distance: float, seed: int, rounds: int = 4) -> np.ndarray:
    if len(coords) <= 2 or min_distance <= 0:
        return coords

    rng = np.random.default_rng(seed)
    coords = coords + rng.normal(0.0, min_distance * 0.03, size=coords.shape)
    neighbor_k = min(8, len(coords) - 1)

    for _ in range(rounds):
        nn = NearestNeighbors(n_neighbors=neighbor_k + 1, algorithm="auto")
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        delta = np.zeros_like(coords)

        for rank in range(1, neighbor_k + 1):
            other = indices[:, rank]
            diff = coords - coords[other]
            dist = distances[:, rank].reshape(-1, 1) + 1e-6
            mask = distances[:, rank] < min_distance
            if not mask.any():
                continue
            push = 0.5 * (min_distance - dist) * diff / dist
            push = np.clip(push, -0.06, 0.06)
            delta[mask] += push[mask]
            np.add.at(delta, other[mask], -push[mask])

        coords += delta / max(1, neighbor_k)
        coords -= coords.mean(axis=0, keepdims=True)

    return coords


def graph_layout_2d(
    Z: np.ndarray,
    y: np.ndarray,
    edge_groups: list[tuple[str, np.ndarray, str]],
    seed: int,
    iterations: int,
    max_layout_edges: int,
    node_spacing: float,
) -> tuple[np.ndarray, float]:
    """Force-directed 2D graph layout driven by the displayed graph edges."""
    n = len(y)
    coords = initialize_graph_coords(Z, seed)
    edges = graph_edges_for_layout(edge_groups, max_layout_edges, seed)
    if n == 0:
        return coords, 0.0
    if edges.size == 0:
        gap = 0.0 if len(np.unique(y)) < 2 else float(np.linalg.norm(coords[y == 1].mean(0) - coords[y == 0].mean(0)))
        return coords, gap

    rng = np.random.default_rng(seed + 101)
    src = edges[:, 0]
    tgt = edges[:, 1]
    degree = np.bincount(np.concatenate([src, tgt]), minlength=n).reshape(-1, 1)
    degree_scale = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    node_spacing = float(np.clip(node_spacing, 0.6, 2.8))
    rest_length = 0.38 * node_spacing
    step = 0.045
    repulse_pairs = min(max(n * 7, 3000), 55000)
    repulse_strength = 0.016 * node_spacing

    for _ in range(int(iterations)):
        delta = np.zeros_like(coords)

        diff = coords[tgt] - coords[src]
        dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        spring = step * (dist - rest_length) * diff / dist
        np.add.at(delta, src, spring)
        np.add.at(delta, tgt, -spring)

        a = rng.integers(0, n, size=repulse_pairs)
        b = rng.integers(0, n, size=repulse_pairs)
        mask = a != b
        a = a[mask]
        b = b[mask]
        diff = coords[a] - coords[b]
        dist2 = np.sum(diff * diff, axis=1, keepdims=True) + 0.05
        repel = repulse_strength * diff / dist2
        np.add.at(delta, a, repel)
        np.add.at(delta, b, -repel)

        delta -= 0.018 * coords * degree_scale
        coords += np.clip(delta, -0.16 * node_spacing, 0.16 * node_spacing)
        coords -= coords.mean(axis=0, keepdims=True)

    coords = StandardScaler().fit_transform(np.nan_to_num(coords))
    coords = enforce_minimum_node_spacing(coords, min_distance=0.11 * node_spacing, seed=seed + 211)
    if len(np.unique(y)) < 2:
        gap = 0.0
    else:
        gap = float(np.linalg.norm(coords[y == 1].mean(axis=0) - coords[y == 0].mean(axis=0)))
    return coords, gap


def sample_edges(edges: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    if edges.size == 0 or len(edges) <= max_edges:
        return edges
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(edges)), size=max_edges, replace=False)
    return edges[idx]


def edge_trace(
    coords: np.ndarray,
    edges: np.ndarray,
    color: str,
    name: str,
    max_edges: int,
    seed: int,
    edge_width: int,
    edge_opacity: float,
):
    if not PLOTLY_AVAILABLE or edges.size == 0 or max_edges <= 0:
        return None
    edges = sample_edges(edges, max_edges, seed)
    xs: list[float | None] = []
    ys: list[float | None] = []
    for src, tgt in edges:
        xs.extend([coords[src, 0], coords[tgt, 0], None])
        ys.extend([coords[src, 1], coords[tgt, 1], None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": color, "width": edge_width},
        opacity=edge_opacity,
        hoverinfo="skip",
        name=f"{name} ({len(edges):,})",
        showlegend=True,
    )


def recolor_edge_groups(edge_groups: list[tuple[str, np.ndarray, str]], color: str) -> list[tuple[str, np.ndarray, str]]:
    return [(name, edges, color) for name, edges, _ in edge_groups]


def plot_2d(
    coords: np.ndarray,
    sample: pd.DataFrame,
    edge_groups: list[tuple[str, np.ndarray, str]],
    title: str,
    max_edges: int,
    edge_width: int,
    edge_opacity: float,
    node_size_scale: float,
    coordinate_tick: float,
    seed: int,
):
    y = sample["label"].to_numpy()
    hover = [
        f"Account={acc}<br>Label={'Suspicious' if label == 1 else 'Benign'}<br>Source index={idx}"
        for acc, label, idx in zip(sample["account"], y, sample["global_idx"])
    ]

    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed, so the 2D interactive graph cannot be drawn.")
        return

    fig = go.Figure()
    per_group_edge_budget = max(1, max_edges)
    for i, (name, edges, color) in enumerate(edge_groups):
        trace = edge_trace(coords, edges, color, name, per_group_edge_budget, seed + i, edge_width, edge_opacity)
        if trace is not None:
            fig.add_trace(trace)

    for label_value, label_name, color, size in [
        (0, "Benign", C_BENIGN, 2.8 * node_size_scale),
        (1, "Suspicious", C_SUSPICIOUS, 4.4 * node_size_scale),
    ]:
        mask = y == label_value
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker={"size": size, "color": color, "opacity": 0.88, "line": {"width": 0}},
                text=np.array(hover, dtype=object)[mask],
                hovertemplate="%{text}<extra></extra>",
                name=label_name,
            )
        )

    axis_style = {
        "showgrid": True,
        "gridcolor": "rgba(148, 163, 184, 0.28)",
        "zeroline": False,
        "dtick": coordinate_tick,
        "title": {"text": ""},
        "showticklabels": False,
        "ticks": "",
        "showspikes": False,
    }

    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        margin={"l": 0, "r": 0, "t": 46, "b": 0},
        height=620,
        xaxis=axis_style,
        yaxis={**axis_style, "scaleanchor": "x", "scaleratio": 1},
        dragmode="pan",
        legend={"orientation": "h", "x": 0.0, "y": 1.02},
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    visible_edges = [f"{name}: {len(edges):,}" for name, edges, _ in edge_groups]
    st.caption("Computed edge counts - " + " / ".join(visible_edges))


def selected_metric(dataset_name: str, encoder: str, setting: str) -> dict[str, float] | None:
    df = load_ablation_results(dataset_name)
    if len(df) == 0:
        return None
    sub = df[(df["encoder"] == encoder) & (df["setting"] == setting)]
    if len(sub) == 0:
        return None
    return {
        "f1": float(sub["f1_1"].mean()),
        "f1_std": float(sub["f1_1"].std(ddof=0)),
        "auroc": float(sub["auroc"].mean()) if "auroc" in sub else np.nan,
        "auprc": float(sub["auprc"].mean()) if "auprc" in sub else np.nan,
        "n": int(len(sub)),
    }


def metric_text(metric: dict[str, float] | None, key: str) -> str:
    if metric is None or key not in metric or math.isnan(metric[key]):
        return "n/a"
    if key == "f1":
        return f"{metric[key]:.3f} +/- {metric.get('f1_std', 0.0):.3f}"
    return f"{metric[key]:.3f}"


def training_command(dataset_name: str, encoder: str, setting: str, loss_name: str, k: int, pool_variant: str, cycle_alpha: float) -> str:
    cfg = DATASETS[dataset_name]
    parts = [
        "python models/subgraph_cl.py",
        f"  --encoder_type {encoder}",
        f"  --node_data_name {cfg['node_arg']}",
        f"  --edge_data_name {cfg['edge_arg']}",
        f"  --loss {loss_name}",
        "  --hidden_dim 256",
        "  --gconv_nlayers 2",
        "  --lr 0.0005",
    ]
    if SETTINGS[setting]["use_knn"]:
        parts.append(f"  --knn_graph {cfg['knn_prefix']}_k{k}")
    if SETTINGS[setting]["use_pool"]:
        parts.append("  --subgraph_pool")
        parts.append(f"  --pool_variant {pool_variant.lower().replace('-aware', '').replace('mean', 'mean')}")
        if pool_variant == "Cycle-aware":
            parts.append(f"  --cycle_alpha {cycle_alpha:.2f}")
    return " \\\n".join(parts)


def render_metric_cards(
    sample: pd.DataFrame,
    tx_metrics: dict[str, float | int | str],
    bhv_metrics: dict[str, float | int | str],
    left_metric: dict[str, float] | None,
    right_metric: dict[str, float] | None,
):
    y = sample["label"].to_numpy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample nodes", _format_int(len(sample)), f"Suspicious {(y == 1).mean() * 100:.1f}%")
    c2.metric(
        "Transaction S-S:S-B",
        str(tx_metrics["ratio"]),
        f"Homophily {tx_metrics['homophily']:.3f}" if not math.isnan(float(tx_metrics["homophily"])) else "No edges",
    )
    c3.metric(
        "Repaired S-S:S-B",
        str(bhv_metrics["ratio"]),
        f"Homophily {bhv_metrics['homophily']:.3f}" if not math.isnan(float(bhv_metrics["homophily"])) else "No edges",
    )
    if left_metric and right_metric:
        delta = right_metric["f1"] - left_metric["f1"]
        c4.metric("Suspicious F1 change", f"{delta:+.3f}", "Right - Left")
    else:
        c4.metric("Suspicious F1 change", "n/a", "No result CSV")


def method_label(encoder: str, setting: str, loss_name: str) -> str:
    suffix = " (ours)" if SETTINGS[setting]["ours"] else ""
    return f"{ENCODER_LABELS.get(encoder, encoder.upper())} {SETTINGS[setting]['short']} / {loss_name}{suffix}"


FEATURE_FAMILY_LABELS = {
    "Behavioral features": "Behavioral",
    "Hybrid features": "Hybrid",
    "Structural features": "Structural",
}
POOL_LABELS = {
    "Mean": "Mean pooling",
    "Heterophily-aware": "Heterophily-aware pooling",
    "Cycle-aware": "Cycle-emphasis pooling",
}


def label_name_array(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == 1, "Suspicious", "Benign")


def plot_selected_metric_summary(left_metric: dict[str, float] | None, right_metric: dict[str, float] | None) -> None:
    if not PLOTLY_AVAILABLE:
        return
    rows = []
    for side, metric in [("Left", left_metric), ("Right", right_metric)]:
        if not metric:
            continue
        for key, label in [("f1", "Suspicious F1"), ("auroc", "AUROC"), ("auprc", "AUPRC")]:
            value = metric.get(key, np.nan)
            if not math.isnan(float(value)):
                rows.append({"Selection": side, "Metric": label, "Value": float(value)})
    if not rows:
        return
    fig = px.bar(
        pd.DataFrame(rows),
        x="Metric",
        y="Value",
        color="Selection",
        barmode="group",
        text_auto=".3f",
        height=320,
        color_discrete_map={"Left": "#7A8A99", "Right": C_SUSPICIOUS},
    )
    fig.update_layout(title="Performance comparison of the two selected methods", margin={"l": 0, "r": 10, "t": 48, "b": 0}, yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")


def plot_graph_diagnostics(tx_metrics: dict[str, float | int | str], bhv_metrics: dict[str, float | int | str]) -> None:
    if not PLOTLY_AVAILABLE:
        return
    hom = pd.DataFrame(
        [
            {"Topology": "Transaction graph", "Homophily": float(tx_metrics["homophily"]) if not math.isnan(float(tx_metrics["homophily"])) else 0.0},
            {"Topology": "Repaired k-NN graph", "Homophily": float(bhv_metrics["homophily"]) if not math.isnan(float(bhv_metrics["homophily"])) else 0.0},
        ]
    )
    comp = pd.DataFrame(
        [
            {"Topology": "Transaction graph", "Edge type": "S-S", "Count": int(tx_metrics["ss"])},
            {"Topology": "Transaction graph", "Edge type": "S-B", "Count": int(tx_metrics["sb"])},
            {"Topology": "Transaction graph", "Edge type": "B-B", "Count": int(tx_metrics["bb"])},
            {"Topology": "Repaired k-NN graph", "Edge type": "S-S", "Count": int(bhv_metrics["ss"])},
            {"Topology": "Repaired k-NN graph", "Edge type": "S-B", "Count": int(bhv_metrics["sb"])},
            {"Topology": "Repaired k-NN graph", "Edge type": "B-B", "Count": int(bhv_metrics["bb"])},
        ]
    )
    left, right = st.columns([0.8, 1.2])
    with left:
        fig_h = px.bar(hom, x="Topology", y="Homophily", color="Topology", text_auto=".3f", height=330)
        fig_h.update_layout(title="Sample graph homophily", showlegend=False, margin={"l": 0, "r": 10, "t": 48, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig_h, width="stretch")
    with right:
        fig_c = px.bar(
            comp,
            x="Topology",
            y="Count",
            color="Edge type",
            barmode="stack",
            text_auto=True,
            height=330,
            color_discrete_map={"S-S": C_SUSPICIOUS, "S-B": "#9E6B45", "B-B": C_BENIGN},
        )
        fig_c.update_layout(title="Edge composition comparison", margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig_c, width="stretch")


with st.sidebar:
    st.header("Data")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
    n_points = st.slider("Sample size", min_value=800, max_value=12_000, value=3_000, step=200)
    suspicious_share = st.slider("Suspicious ratio in sample", min_value=0.01, max_value=0.70, value=0.01, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=99_999, value=2025, step=1)
    feature_family_label = st.selectbox("Feature group", list(FEATURE_FAMILY_LABELS.keys()), index=0)
    feature_family = FEATURE_FAMILY_LABELS[feature_family_label]

    st.header("Left method")
    left_encoder = st.selectbox("Base encoder", ENCODERS, format_func=lambda x: ENCODER_LABELS[x], index=1, key="left_encoder")
    left_setting = st.selectbox("Topology setting", list(SETTINGS.keys()), format_func=lambda x: SETTINGS[x]["label"], index=0, key="left_setting")
    left_default_loss = st.checkbox("Use encoder default loss", value=True, key="left_default_loss")
    if left_default_loss:
        left_loss = DEFAULT_LOSS[left_encoder]
        st.caption(f"Default: {left_loss}")
    else:
        left_loss = st.selectbox(
            "Loss function",
            LOSS_CHOICES,
            index=LOSS_CHOICES.index(DEFAULT_LOSS[left_encoder]),
            key="left_loss",
        )

    st.header("Right method")
    right_encoder = st.selectbox("Base encoder", ENCODERS, format_func=lambda x: ENCODER_LABELS[x], index=0, key="right_encoder")
    right_setting = st.selectbox("Topology setting", list(SETTINGS.keys()), format_func=lambda x: SETTINGS[x]["label"], index=3, key="right_setting")
    right_default_loss = st.checkbox("Use encoder default loss", value=True, key="right_default_loss")
    if right_default_loss:
        right_loss = DEFAULT_LOSS[right_encoder]
        st.caption(f"Default: {right_loss}")
    else:
        right_loss = st.selectbox(
            "Loss function",
            LOSS_CHOICES,
            index=LOSS_CHOICES.index(DEFAULT_LOSS[right_encoder]),
            key="right_loss",
        )

    st.header("Graph parameters")
    k = st.slider("k for repaired k-NN", min_value=3, max_value=50, value=10, step=1)
    graph_alpha = st.slider("Message-passing strength", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    view_weight = st.slider("Repaired topology weight", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
    pool_strength = st.slider("Subgraph pooling strength", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
    pool_variant = st.selectbox("Pooling method", ["Mean", "Heterophily-aware", "Cycle-aware"], format_func=lambda x: POOL_LABELS[x], index=0)
    cycle_alpha = st.slider("Cycle-emphasis coefficient", min_value=0.0, max_value=5.0, value=2.0, step=0.25)
    loss_strength = st.slider("Visualization loss profile strength", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

    st.header("Rendering")
    max_edges = st.slider("Max edges shown per type", min_value=0, max_value=20_000, value=6_000, step=500)
    edge_width = st.slider("Edge line width", min_value=1, max_value=8, value=1, step=1)
    edge_opacity = st.slider("Edge line opacity", min_value=0.10, max_value=1.00, value=0.25, step=0.05)
    node_spacing = st.slider("Coordinate spacing between nodes", min_value=0.6, max_value=2.8, value=1.7, step=0.1)
    node_size_scale = st.slider("Node display size", min_value=0.45, max_value=1.30, value=0.75, step=0.05)
    coordinate_tick = st.slider("Coordinate tick unit", min_value=0.05, max_value=0.50, value=0.10, step=0.05)
    layout_iterations = st.slider("2D graph layout iterations", min_value=10, max_value=120, value=45, step=5)


st.title("BehaView 2D method comparator")
st.caption(
    "On ATNET and AMLworld, this compares the transaction-graph-based existing method against our repaired-topology method as 2D topology graphs."
)

st.info(
    "The 2D output is a force-directed graph laid out from the actual edges used by the selected method. "
    "Node color encodes the label, lines represent transaction edges and repaired k-NN edges, and line width and node coordinate spacing can be adjusted in the Rendering menu on the left."
)

df_all = load_nodes(dataset_name)
feature_cols = selected_feature_columns(df_all, feature_family)
sample = make_sample(dataset_name, n_points, suspicious_share, int(seed))
labels = sample["label"].to_numpy(dtype=np.int64)
X = numeric_matrix(sample, feature_cols)
tx_edges = transaction_edges_for_sample(dataset_name, tuple(sample["account"].astype(str).tolist()))
knn_edges = build_knn_edges(X, k)

if pool_variant == "Cycle-aware" and "triangle" in sample.columns:
    cycle_signal = (sample["triangle"].to_numpy(dtype=float) > 0).astype(float)
elif pool_variant == "Cycle-aware":
    degree_proxy = np.zeros(len(sample), dtype=float)
    if tx_edges.size:
        np.add.at(degree_proxy, tx_edges[:, 0], 1.0)
        np.add.at(degree_proxy, tx_edges[:, 1], 1.0)
    threshold = np.quantile(degree_proxy, 0.75) if degree_proxy.size else 0.0
    cycle_signal = (degree_proxy > threshold).astype(float)
else:
    cycle_signal = None

tx_metrics = graph_metrics(tx_edges, labels)
bhv_metrics = graph_metrics(knn_edges, labels)
left_metric = selected_metric(dataset_name, left_encoder, left_setting)
right_metric = selected_metric(dataset_name, right_encoder, right_setting)
render_metric_cards(sample, tx_metrics, bhv_metrics, left_metric, right_metric)

st.markdown("---")

with st.spinner("Computing per-method representations and the actual 2D graph layout..."):
    left_Z, left_edges = method_representation(
        X,
        labels,
        tx_edges,
        knn_edges,
        left_setting,
        left_encoder,
        left_loss,
        graph_alpha,
        view_weight,
        pool_strength,
        pool_variant,
        cycle_signal,
        cycle_alpha,
        loss_strength,
    )
    right_Z, right_edges = method_representation(
        X,
        labels,
        tx_edges,
        knn_edges,
        right_setting,
        right_encoder,
        right_loss,
        graph_alpha,
        view_weight,
        pool_strength,
        pool_variant,
        cycle_signal,
        cycle_alpha,
        loss_strength,
    )
    layout_edge_budget = max(1, max(max_edges * 3, int(n_points * 6)))
    left_coords, left_gap = graph_layout_2d(left_Z, labels, left_edges, int(seed), layout_iterations, layout_edge_budget, node_spacing)
    right_coords, right_gap = graph_layout_2d(right_Z, labels, right_edges, int(seed) + 7, layout_iterations, layout_edge_budget, node_spacing)

left_title = f"Left: {method_label(left_encoder, left_setting, left_loss)} | Suspicious F1 {metric_text(left_metric, 'f1')} | 2D cluster gap {left_gap:.2f}"
right_title = f"Right: {method_label(right_encoder, right_setting, right_loss)} | Suspicious F1 {metric_text(right_metric, 'f1')} | 2D cluster gap {right_gap:.2f}"

col_left, col_right = st.columns(2)
with col_left:
    plot_2d(left_coords, sample, left_edges, left_title, max_edges, edge_width, edge_opacity, node_size_scale, coordinate_tick, int(seed))
with col_right:
    right_display_edges = recolor_edge_groups(right_edges, C_TX_EDGE)
    plot_2d(right_coords, sample, right_display_edges, right_title, max_edges, edge_width, edge_opacity, node_size_scale, coordinate_tick, int(seed) + 11)

st.markdown("---")

tab_rq1, tab_rq2, tab_rq3, tab_rq4 = st.tabs(
    [
        "RQ1. Behavioral homophily recovery",
        "RQ2. Signal preservation under aggregation",
        "RQ3. Self-supervised money laundering detection under label scarcity",
        "RQ4. Robustness across AML regimes",
    ]
)

with tab_rq1:
    st.subheader("Check how much the repaired topology recovers suspicious neighbors")
    plot_graph_diagnostics(tx_metrics, bhv_metrics)
    if PLOTLY_AVAILABLE:
        class_df = pd.DataFrame({"Label": label_name_array(labels)})
        fig = px.pie(
            class_df,
            names="Label",
            color="Label",
            color_discrete_map={"Suspicious": C_SUSPICIOUS, "Benign": C_BENIGN},
            title="Label composition of the current sample",
            height=330,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig, width="stretch")
    if feature_cols and PLOTLY_AVAILABLE:
        feature_for_plot = st.selectbox("Behavioral feature to view as a distribution", feature_cols[: min(60, len(feature_cols))], key="rq1_feature_plot")
        plot_df = sample[[feature_for_plot, "label"]].copy()
        plot_df["Label"] = label_name_array(plot_df["label"].to_numpy())
        fig = px.histogram(
            plot_df,
            x=feature_for_plot,
            color="Label",
            barmode="overlay",
            opacity=0.62,
            nbins=45,
            title=f"{feature_for_plot} distribution",
            color_discrete_map={"Suspicious": C_SUSPICIOUS, "Benign": C_BENIGN},
            height=360,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig, width="stretch")

with tab_rq2:
    st.subheader("Compare whether the aggregation method and topology preserve the suspicious signal")
    plot_selected_metric_summary(left_metric, right_metric)
    ablation = load_ablation_results(dataset_name)
    if len(ablation) == 0 or not PLOTLY_AVAILABLE:
        st.warning("Could not find the RQ2 ablation result CSV.")
    else:
        rows = []
        for (encoder, setting), grp in ablation.groupby(["encoder", "setting"]):
            if setting not in SETTINGS:
                continue
            rows.append(
                {
                    "Encoder": ENCODER_LABELS.get(encoder, encoder.upper()),
                    "Setting": SETTINGS[setting]["short"],
                    "setting_order": "abcd".index(setting),
                    "Suspicious F1": float(grp["f1_1"].mean()),
                }
            )
        chart = pd.DataFrame(rows).sort_values(["Encoder", "setting_order"])
        keep = [ENCODER_LABELS.get(e, e.upper()) for e in dict.fromkeys([left_encoder, right_encoder, "gbt", "bgrl"])]
        chart = chart[chart["Encoder"].isin(keep)] if len(chart) else chart
        fig = px.line(
            chart,
            x="Setting",
            y="Suspicious F1",
            color="Encoder",
            markers=True,
            category_orders={"Setting": [SETTINGS[s]["short"] for s in "abcd"]},
            title="Four-setting ablation: topology branch x contrastive level",
            height=430,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 52, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    loss_df = pd.DataFrame({"Encoder": [ENCODER_LABELS[e] for e in ENCODERS], "Default loss": [DEFAULT_LOSS[e] for e in ENCODERS]})
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            loss_df,
            x="Encoder",
            y="Default loss",
            color="Default loss",
            title="Default loss mapping per method",
            height=310,
        )
        fig.update_traces(marker={"size": 14})
        fig.update_layout(margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig, width="stretch")
    st.caption("In actual training, `compute_loss(...)` in `models/subgraph_cl.py` dispatches on `--loss`. This app applies a visualization profile without retraining.")

with tab_rq3:
    st.subheader("Check whether our representation is competitive with supervised baselines under label scarcity")
    rq3_rows = []
    for ds in ["atnet", "amlworld", "amlnet"]:
        behav_path = BASE_DIR / "results" / "rq3" / f"behaview_{ds}.csv"
        if behav_path.exists():
            df = pd.read_csv(behav_path)
            ratios = df["Model"].astype(str).str.extract(r"_r(?P<ratio>0\.\d+)_")["ratio"].astype(float)
            for ratio, grp in df.assign(train_ratio=ratios).dropna(subset=["train_ratio"]).groupby("train_ratio"):
                rq3_rows.append({"Dataset": ds.upper(), "Model": "BehaView", "Label ratio": float(ratio), "Suspicious F1": float(grp["f1_1"].mean())})
        sup_path = BASE_DIR / "results" / "rq3" / f"supervised_{ds}.csv"
        if sup_path.exists():
            df = pd.read_csv(sup_path)
            for (model, ratio), grp in df.groupby(["model", "train_ratio"]):
                rq3_rows.append({"Dataset": ds.upper(), "Model": str(model).upper(), "Label ratio": float(ratio), "Suspicious F1": float(grp["f1_1"].mean())})
    boost_path = BASE_DIR / "results" / "rq3" / "boosting_behav.csv"
    if boost_path.exists():
        df = pd.read_csv(boost_path)
        for (ds, model, ratio), grp in df.groupby(["dataset", "model", "train_ratio"]):
            rq3_rows.append({"Dataset": str(ds).upper(), "Model": str(model).upper(), "Label ratio": float(ratio), "Suspicious F1": float(grp["f1_1"].mean())})
    if rq3_rows and PLOTLY_AVAILABLE:
        chart = pd.DataFrame(rq3_rows)
        fig = px.line(
            chart,
            x="Label ratio",
            y="Suspicious F1",
            color="Model",
            facet_col="Dataset",
            markers=True,
            title="Suspicious-class F1 by label ratio",
            height=500,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 56, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Could not find the RQ3 label-efficiency result CSV.")

with tab_rq4:
    st.subheader("Check whether the same pattern holds across different data scales and suspicious ratios")
    rq4_sources = [
        ("ATNET", BASE_DIR / "results" / "rq1" / "main_sweep.csv"),
        ("AMLWORLD", BASE_DIR / "results" / "rq4" / "amlworld_main_sweep.csv"),
        ("AMLNET", BASE_DIR / "results" / "rq4" / "amlnet_main_sweep.csv"),
    ]
    rows = []
    for ds, path in rq4_sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        parsed = df["Model"].astype(str).str.extract(r"^[^_]+_(?P<encoder>.+)_(?P<setting>[abcd])_s(?P<seed>\d+)$")
        df = df.join(parsed).dropna(subset=["encoder", "setting"])
        for (encoder, setting), grp in df.groupby(["encoder", "setting"]):
            rows.append(
                {
                    "Dataset": ds,
                    "Encoder": ENCODER_LABELS.get(str(encoder), str(encoder).upper()),
                    "Setting": SETTINGS[str(setting)]["short"],
                    "setting_order": "abcd".index(str(setting)),
                    "Suspicious F1": float(grp["f1_1"].mean()),
                }
            )
    if rows and PLOTLY_AVAILABLE:
        chart = pd.DataFrame(rows).sort_values(["Dataset", "setting_order"])
        fig = px.bar(
            chart,
            x="Setting",
            y="Suspicious F1",
            color="Dataset",
            barmode="group",
            facet_col="Encoder",
            category_orders={"Setting": [SETTINGS[s]["short"] for s in "abcd"]},
            title="Topology repair pattern by dataset",
            height=520,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 56, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Could not find the RQ4 cross-dataset result CSV.")

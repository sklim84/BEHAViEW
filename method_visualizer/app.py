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
    "HOFINET": {
        "node_path": BASE_DIR / "datasets" / "HOFINET_NODE_FEAT.csv",
        "edge_path": BASE_DIR / "datasets" / "HOFINET_EDGES.csv",
        "ablation_path": BASE_DIR / "results" / "exp_results_hofinet_ab.csv",
        "supervised_path": BASE_DIR / "results" / "exp_results_supervised_hofinet.csv",
        "extra_paths": [
            BASE_DIR / "results" / "exp_results_consisgad_hofinet.csv",
            BASE_DIR / "results" / "exp_results_caregnn_pcgnn_hofinet.csv",
            BASE_DIR / "results" / "exp_results_bwgnn_gaga_hofinet.csv",
        ],
        "node_arg": "HOFINET_NODE_FEAT",
        "edge_arg": "HOFINET_EDGES",
        "knn_prefix": "HOFINET_KNN_BEHAV",
        "model_prefix": "hof",
    },
    "AMLworld": {
        "node_path": BASE_DIR / "datasets" / "amlworld" / "AMLWORLD_NODE_FEAT.csv",
        "edge_path": BASE_DIR / "datasets" / "amlworld" / "AMLWORLD_EDGES.csv",
        "ablation_path": BASE_DIR / "results" / "exp_results_amlworld.csv",
        "supervised_path": BASE_DIR / "results" / "exp_results_supervised_amlworld.csv",
        "extra_paths": [
            BASE_DIR / "results" / "exp_results_consisgad_amlworld.csv",
            BASE_DIR / "results" / "exp_results_caregnn_pcgnn_amlworld.csv",
            BASE_DIR / "results" / "exp_results_bwgnn_gaga_amlworld.csv",
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
        "label": "(a) 거래 토폴로지 + 노드 표현",
        "short": "거래/노드",
        "use_knn": False,
        "use_pool": False,
        "ours": False,
    },
    "b": {
        "label": "(b) 복구 k-NN 토폴로지 + 노드 표현",
        "short": "복구/노드",
        "use_knn": True,
        "use_pool": False,
        "ours": False,
    },
    "c": {
        "label": "(c) 거래 토폴로지 + 서브그래프 풀링",
        "short": "거래/풀링",
        "use_knn": False,
        "use_pool": True,
        "ours": False,
    },
    "d": {
        "label": "(d) BehaView: 복구 토폴로지 + 서브그래프 풀링",
        "short": "복구/풀링",
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
    page_title="BehaView 3D 기법 비교기",
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
                    "family": "자기지도 GCL",
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
                        "family": "지도학습/표형 기준선",
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
                        "family": "기존 그래프 기준선",
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
        edge_groups.append(("거래 엣지", tx_edges, C_TX_EDGE))
        edge_groups.append(("복구 k-NN 엣지", knn_edges, C_BHV_EDGE))
    else:
        Z_out = tx_pooled
        edge_groups.append(("거래 엣지", tx_edges, C_TX_EDGE))

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
    coords = rng.normal(0.0, 1.0, size=(n, 3))
    Z = np.nan_to_num(Z)
    if Z.ndim == 2 and Z.shape[1] > 0:
        dims = min(3, Z.shape[1])
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


def graph_layout_3d(
    Z: np.ndarray,
    y: np.ndarray,
    edge_groups: list[tuple[str, np.ndarray, str]],
    seed: int,
    iterations: int,
    max_layout_edges: int,
    node_spacing: float,
) -> tuple[np.ndarray, float]:
    """Force-directed 3D graph layout driven by the displayed graph edges."""
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
    repulse_pairs = min(max(n * 5, 2500), 45000)
    repulse_strength = 0.012 * node_spacing

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
    coords = enforce_minimum_node_spacing(coords, min_distance=0.09 * node_spacing, seed=seed + 211)
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
    zs: list[float | None] = []
    for src, tgt in edges:
        xs.extend([coords[src, 0], coords[tgt, 0], None])
        ys.extend([coords[src, 1], coords[tgt, 1], None])
        zs.extend([coords[src, 2], coords[tgt, 2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": color, "width": edge_width},
        opacity=edge_opacity,
        hoverinfo="skip",
        name=f"{name} ({len(edges):,})",
        showlegend=True,
    )


def plot_3d(
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
        f"계좌={acc}<br>라벨={'의심' if label == 1 else '정상'}<br>원본 인덱스={idx}"
        for acc, label, idx in zip(sample["account"], y, sample["global_idx"])
    ]

    if not PLOTLY_AVAILABLE:
        st.warning("Plotly가 설치되어 있지 않아 3D 인터랙티브 그래프를 그릴 수 없습니다.")
        return

    fig = go.Figure()
    per_group_edge_budget = max(1, max_edges)
    for i, (name, edges, color) in enumerate(edge_groups):
        trace = edge_trace(coords, edges, color, name, per_group_edge_budget, seed + i, edge_width, edge_opacity)
        if trace is not None:
            fig.add_trace(trace)

    for label_value, label_name, color, size in [
        (0, "정상", C_BENIGN, 2.8 * node_size_scale),
        (1, "의심", C_SUSPICIOUS, 4.4 * node_size_scale),
    ]:
        mask = y == label_value
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
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
        scene={
            "xaxis": axis_style,
            "yaxis": axis_style,
            "zaxis": axis_style,
            "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.92}},
        },
        legend={"orientation": "h", "x": 0.0, "y": 1.02},
    )
    st.plotly_chart(fig, width="stretch")
    visible_edges = [f"{name}: {len(edges):,}" for name, edges, _ in edge_groups]
    st.caption("계산된 엣지 수 - " + " / ".join(visible_edges))


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
    c1.metric("샘플 노드", _format_int(len(sample)), f"의심 {(y == 1).mean() * 100:.1f}%")
    c2.metric(
        "거래 S-S:S-B",
        str(tx_metrics["ratio"]),
        f"동질성 {tx_metrics['homophily']:.3f}" if not math.isnan(float(tx_metrics["homophily"])) else "엣지 없음",
    )
    c3.metric(
        "복구 S-S:S-B",
        str(bhv_metrics["ratio"]),
        f"동질성 {bhv_metrics['homophily']:.3f}" if not math.isnan(float(bhv_metrics["homophily"])) else "엣지 없음",
    )
    if left_metric and right_metric:
        delta = right_metric["f1"] - left_metric["f1"]
        c4.metric("의심 F1 변화", f"{delta:+.3f}", "오른쪽 - 왼쪽")
    else:
        c4.metric("의심 F1 변화", "n/a", "결과 CSV 없음")


def method_label(encoder: str, setting: str, loss_name: str) -> str:
    suffix = " (우리 기법)" if SETTINGS[setting]["ours"] else ""
    return f"{ENCODER_LABELS.get(encoder, encoder.upper())} {SETTINGS[setting]['short']} / {loss_name}{suffix}"


FEATURE_FAMILY_LABELS = {
    "행동 피처": "Behavioral",
    "혼합 피처": "Hybrid",
    "구조 피처": "Structural",
}
POOL_LABELS = {
    "Mean": "평균 풀링",
    "Heterophily-aware": "이질성 완화 풀링",
    "Cycle-aware": "사이클 강조 풀링",
}


def label_name_array(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == 1, "의심", "정상")


def plot_selected_metric_summary(left_metric: dict[str, float] | None, right_metric: dict[str, float] | None) -> None:
    if not PLOTLY_AVAILABLE:
        return
    rows = []
    for side, metric in [("왼쪽", left_metric), ("오른쪽", right_metric)]:
        if not metric:
            continue
        for key, label in [("f1", "의심 F1"), ("auroc", "AUROC"), ("auprc", "AUPRC")]:
            value = metric.get(key, np.nan)
            if not math.isnan(float(value)):
                rows.append({"선택": side, "지표": label, "값": float(value)})
    if not rows:
        return
    fig = px.bar(
        pd.DataFrame(rows),
        x="지표",
        y="값",
        color="선택",
        barmode="group",
        text_auto=".3f",
        height=320,
        color_discrete_map={"왼쪽": "#7A8A99", "오른쪽": C_SUSPICIOUS},
    )
    fig.update_layout(title="선택한 두 기법의 성능 비교", margin={"l": 0, "r": 10, "t": 48, "b": 0}, yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")


def plot_graph_diagnostics(tx_metrics: dict[str, float | int | str], bhv_metrics: dict[str, float | int | str]) -> None:
    if not PLOTLY_AVAILABLE:
        return
    hom = pd.DataFrame(
        [
            {"토폴로지": "거래 그래프", "동질성": float(tx_metrics["homophily"]) if not math.isnan(float(tx_metrics["homophily"])) else 0.0},
            {"토폴로지": "복구 k-NN 그래프", "동질성": float(bhv_metrics["homophily"]) if not math.isnan(float(bhv_metrics["homophily"])) else 0.0},
        ]
    )
    comp = pd.DataFrame(
        [
            {"토폴로지": "거래 그래프", "엣지 유형": "S-S", "개수": int(tx_metrics["ss"])},
            {"토폴로지": "거래 그래프", "엣지 유형": "S-B", "개수": int(tx_metrics["sb"])},
            {"토폴로지": "거래 그래프", "엣지 유형": "B-B", "개수": int(tx_metrics["bb"])},
            {"토폴로지": "복구 k-NN 그래프", "엣지 유형": "S-S", "개수": int(bhv_metrics["ss"])},
            {"토폴로지": "복구 k-NN 그래프", "엣지 유형": "S-B", "개수": int(bhv_metrics["sb"])},
            {"토폴로지": "복구 k-NN 그래프", "엣지 유형": "B-B", "개수": int(bhv_metrics["bb"])},
        ]
    )
    left, right = st.columns([0.8, 1.2])
    with left:
        fig_h = px.bar(hom, x="토폴로지", y="동질성", color="토폴로지", text_auto=".3f", height=330)
        fig_h.update_layout(title="샘플 그래프 동질성", showlegend=False, margin={"l": 0, "r": 10, "t": 48, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig_h, width="stretch")
    with right:
        fig_c = px.bar(
            comp,
            x="토폴로지",
            y="개수",
            color="엣지 유형",
            barmode="stack",
            text_auto=True,
            height=330,
            color_discrete_map={"S-S": C_SUSPICIOUS, "S-B": "#9E6B45", "B-B": C_BENIGN},
        )
        fig_c.update_layout(title="엣지 구성 비교", margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig_c, width="stretch")


with st.sidebar:
    st.header("데이터")
    dataset_name = st.selectbox("데이터셋", list(DATASETS.keys()), index=0)
    n_points = st.slider("샘플 수", min_value=800, max_value=12_000, value=3_000, step=200)
    suspicious_share = st.slider("샘플 내 의심 비율", min_value=0.05, max_value=0.70, value=0.35, step=0.05)
    seed = st.number_input("랜덤 시드", min_value=0, max_value=99_999, value=2025, step=1)
    feature_family_label = st.selectbox("피처 묶음", list(FEATURE_FAMILY_LABELS.keys()), index=0)
    feature_family = FEATURE_FAMILY_LABELS[feature_family_label]

    st.header("왼쪽 기법")
    left_encoder = st.selectbox("기본 인코더", ENCODERS, format_func=lambda x: ENCODER_LABELS[x], index=1, key="left_encoder")
    left_setting = st.selectbox("토폴로지 설정", list(SETTINGS.keys()), format_func=lambda x: SETTINGS[x]["label"], index=0, key="left_setting")
    left_default_loss = st.checkbox("인코더 기본 loss 사용", value=True, key="left_default_loss")
    if left_default_loss:
        left_loss = DEFAULT_LOSS[left_encoder]
        st.caption(f"기본값: {left_loss}")
    else:
        left_loss = st.selectbox(
            "Loss 함수",
            LOSS_CHOICES,
            index=LOSS_CHOICES.index(DEFAULT_LOSS[left_encoder]),
            key="left_loss",
        )

    st.header("오른쪽 기법")
    right_encoder = st.selectbox("기본 인코더", ENCODERS, format_func=lambda x: ENCODER_LABELS[x], index=0, key="right_encoder")
    right_setting = st.selectbox("토폴로지 설정", list(SETTINGS.keys()), format_func=lambda x: SETTINGS[x]["label"], index=3, key="right_setting")
    right_default_loss = st.checkbox("인코더 기본 loss 사용", value=True, key="right_default_loss")
    if right_default_loss:
        right_loss = DEFAULT_LOSS[right_encoder]
        st.caption(f"기본값: {right_loss}")
    else:
        right_loss = st.selectbox(
            "Loss 함수",
            LOSS_CHOICES,
            index=LOSS_CHOICES.index(DEFAULT_LOSS[right_encoder]),
            key="right_loss",
        )

    st.header("그래프 파라미터")
    k = st.slider("복구 k-NN의 k", min_value=3, max_value=50, value=10, step=1)
    graph_alpha = st.slider("메시지 패싱 강도", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    view_weight = st.slider("복구 토폴로지 가중치", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
    pool_strength = st.slider("서브그래프 풀링 강도", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
    pool_variant = st.selectbox("풀링 방식", ["Mean", "Heterophily-aware", "Cycle-aware"], format_func=lambda x: POOL_LABELS[x], index=0)
    cycle_alpha = st.slider("사이클 강조 계수", min_value=0.0, max_value=5.0, value=2.0, step=0.25)
    loss_strength = st.slider("시각화 loss 프로파일 강도", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

    st.header("렌더링")
    max_edges = st.slider("엣지 유형별 최대 표시 수", min_value=0, max_value=20_000, value=6_000, step=500)
    edge_width = st.slider("엣지 선 두께", min_value=1, max_value=8, value=4, step=1)
    edge_opacity = st.slider("엣지 선 불투명도", min_value=0.10, max_value=1.00, value=0.82, step=0.05)
    node_spacing = st.slider("노드 간 좌표 간격", min_value=0.6, max_value=2.8, value=1.7, step=0.1)
    node_size_scale = st.slider("노드 표시 크기", min_value=0.45, max_value=1.30, value=0.75, step=0.05)
    coordinate_tick = st.slider("좌표 눈금 단위", min_value=0.05, max_value=0.50, value=0.10, step=0.05)
    layout_iterations = st.slider("3D 그래프 배치 반복 수", min_value=10, max_value=120, value=45, step=5)


st.title("BehaView 3D 기법 비교기")
st.caption(
    "HOFINET과 AMLworld에서 거래 그래프 기반 기존 기법과 복구 토폴로지 기반 우리 기법을 3D로 비교합니다."
)

st.info(
    "3D 출력은 선택한 기법이 사용하는 실제 엣지로 배치한 force-directed 그래프입니다. "
    "노드 색은 라벨, 선은 거래 엣지와 복구 k-NN 엣지를 나타내며, 선 두께와 노드 간 좌표 간격은 왼쪽 렌더링 메뉴에서 조정할 수 있습니다."
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

with st.spinner("기법별 표현과 실제 3D 그래프 배치를 계산하는 중입니다..."):
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
    left_coords, left_gap = graph_layout_3d(left_Z, labels, left_edges, int(seed), layout_iterations, layout_edge_budget, node_spacing)
    right_coords, right_gap = graph_layout_3d(right_Z, labels, right_edges, int(seed) + 7, layout_iterations, layout_edge_budget, node_spacing)

left_title = f"왼쪽: {method_label(left_encoder, left_setting, left_loss)} | 의심 F1 {metric_text(left_metric, 'f1')} | 3D 군집거리 {left_gap:.2f}"
right_title = f"오른쪽: {method_label(right_encoder, right_setting, right_loss)} | 의심 F1 {metric_text(right_metric, 'f1')} | 3D 군집거리 {right_gap:.2f}"

col_left, col_right = st.columns(2)
with col_left:
    plot_3d(left_coords, sample, left_edges, left_title, max_edges, edge_width, edge_opacity, node_size_scale, coordinate_tick, int(seed))
with col_right:
    plot_3d(right_coords, sample, right_edges, right_title, max_edges, edge_width, edge_opacity, node_size_scale, coordinate_tick, int(seed) + 11)

st.markdown("---")

tab_rq1, tab_rq2, tab_rq3, tab_rq4 = st.tabs(
    [
        "RQ1. Behavioral homophily recovery",
        "RQ2. Signal preservation under aggregation",
        "RQ3. Self-supervised AML detection under label scarcity",
        "RQ4. Robustness across AML regimes",
    ]
)

with tab_rq1:
    st.subheader("복구 토폴로지가 의심 이웃을 얼마나 회복하는지 확인")
    plot_graph_diagnostics(tx_metrics, bhv_metrics)
    if PLOTLY_AVAILABLE:
        class_df = pd.DataFrame({"라벨": label_name_array(labels)})
        fig = px.pie(
            class_df,
            names="라벨",
            color="라벨",
            color_discrete_map={"의심": C_SUSPICIOUS, "정상": C_BENIGN},
            title="현재 샘플의 라벨 구성",
            height=330,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig, width="stretch")
    if feature_cols and PLOTLY_AVAILABLE:
        feature_for_plot = st.selectbox("분포로 볼 행동 피처", feature_cols[: min(60, len(feature_cols))], key="rq1_feature_plot")
        plot_df = sample[[feature_for_plot, "label"]].copy()
        plot_df["라벨"] = label_name_array(plot_df["label"].to_numpy())
        fig = px.histogram(
            plot_df,
            x=feature_for_plot,
            color="라벨",
            barmode="overlay",
            opacity=0.62,
            nbins=45,
            title=f"{feature_for_plot} 분포",
            color_discrete_map={"의심": C_SUSPICIOUS, "정상": C_BENIGN},
            height=360,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig, width="stretch")

with tab_rq2:
    st.subheader("집계 방식과 토폴로지가 의심 신호를 보존하는지 비교")
    plot_selected_metric_summary(left_metric, right_metric)
    ablation = load_ablation_results(dataset_name)
    if len(ablation) == 0 or not PLOTLY_AVAILABLE:
        st.warning("RQ2 ablation 결과 CSV를 찾지 못했습니다.")
    else:
        rows = []
        for (encoder, setting), grp in ablation.groupby(["encoder", "setting"]):
            if setting not in SETTINGS:
                continue
            rows.append(
                {
                    "인코더": ENCODER_LABELS.get(encoder, encoder.upper()),
                    "설정": SETTINGS[setting]["short"],
                    "setting_order": "abcd".index(setting),
                    "의심 F1": float(grp["f1_1"].mean()),
                }
            )
        chart = pd.DataFrame(rows).sort_values(["인코더", "setting_order"])
        keep = [ENCODER_LABELS.get(e, e.upper()) for e in dict.fromkeys([left_encoder, right_encoder, "gbt", "bgrl"])]
        chart = chart[chart["인코더"].isin(keep)] if len(chart) else chart
        fig = px.line(
            chart,
            x="설정",
            y="의심 F1",
            color="인코더",
            markers=True,
            category_orders={"설정": [SETTINGS[s]["short"] for s in "abcd"]},
            title="4개 설정 ablation: 토폴로지 branch × contrastive level",
            height=430,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 52, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    loss_df = pd.DataFrame({"인코더": [ENCODER_LABELS[e] for e in ENCODERS], "기본 loss": [DEFAULT_LOSS[e] for e in ENCODERS]})
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            loss_df,
            x="인코더",
            y="기본 loss",
            color="기본 loss",
            title="기법별 기본 loss 매핑",
            height=310,
        )
        fig.update_traces(marker={"size": 14})
        fig.update_layout(margin={"l": 0, "r": 10, "t": 48, "b": 0})
        st.plotly_chart(fig, width="stretch")
    st.caption("실제 학습은 `models/subgraph_cl.py`의 `compute_loss(...)`가 `--loss`를 dispatch합니다. 이 앱은 재학습 없이 시각화용 프로파일을 적용합니다.")

with tab_rq3:
    st.subheader("라벨이 적을 때 우리 표현이 지도학습 기준선과 경쟁적인지 확인")
    rq3_rows = []
    for ds in ["hofinet", "amlworld", "amlnet"]:
        behav_path = BASE_DIR / "results" / "rq3" / f"behaview_{ds}.csv"
        if behav_path.exists():
            df = pd.read_csv(behav_path)
            ratios = df["Model"].astype(str).str.extract(r"_r(?P<ratio>0\.\d+)_")["ratio"].astype(float)
            for ratio, grp in df.assign(train_ratio=ratios).dropna(subset=["train_ratio"]).groupby("train_ratio"):
                rq3_rows.append({"데이터셋": ds.upper(), "모델": "BehaView", "라벨 비율": float(ratio), "의심 F1": float(grp["f1_1"].mean())})
        sup_path = BASE_DIR / "results" / "rq3" / f"supervised_{ds}.csv"
        if sup_path.exists():
            df = pd.read_csv(sup_path)
            for (model, ratio), grp in df.groupby(["model", "train_ratio"]):
                rq3_rows.append({"데이터셋": ds.upper(), "모델": str(model).upper(), "라벨 비율": float(ratio), "의심 F1": float(grp["f1_1"].mean())})
    boost_path = BASE_DIR / "results" / "rq3" / "boosting_behav.csv"
    if boost_path.exists():
        df = pd.read_csv(boost_path)
        for (ds, model, ratio), grp in df.groupby(["dataset", "model", "train_ratio"]):
            rq3_rows.append({"데이터셋": str(ds).upper(), "모델": str(model).upper(), "라벨 비율": float(ratio), "의심 F1": float(grp["f1_1"].mean())})
    if rq3_rows and PLOTLY_AVAILABLE:
        chart = pd.DataFrame(rq3_rows)
        fig = px.line(
            chart,
            x="라벨 비율",
            y="의심 F1",
            color="모델",
            facet_col="데이터셋",
            markers=True,
            title="라벨 비율별 의심 클래스 F1",
            height=500,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 56, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("RQ3 label-efficiency 결과 CSV를 찾지 못했습니다.")

with tab_rq4:
    st.subheader("데이터 규모와 의심 비율이 달라도 같은 패턴이 유지되는지 확인")
    rq4_sources = [
        ("HOFINET", BASE_DIR / "results" / "rq1" / "main_sweep.csv"),
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
                    "데이터셋": ds,
                    "인코더": ENCODER_LABELS.get(str(encoder), str(encoder).upper()),
                    "설정": SETTINGS[str(setting)]["short"],
                    "setting_order": "abcd".index(str(setting)),
                    "의심 F1": float(grp["f1_1"].mean()),
                }
            )
    if rows and PLOTLY_AVAILABLE:
        chart = pd.DataFrame(rows).sort_values(["데이터셋", "setting_order"])
        fig = px.bar(
            chart,
            x="설정",
            y="의심 F1",
            color="데이터셋",
            barmode="group",
            facet_col="인코더",
            category_orders={"설정": [SETTINGS[s]["short"] for s in "abcd"]},
            title="데이터셋별 topology repair 패턴",
            height=520,
        )
        fig.update_layout(margin={"l": 0, "r": 10, "t": 56, "b": 0}, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("RQ4 cross-dataset 결과 CSV를 찾지 못했습니다.")

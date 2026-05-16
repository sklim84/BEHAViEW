"""Cached data loaders for the BEHAViEW walkthrough app.

All loaders return the same arrays/DataFrames as the offline analysis scripts so
the app stays consistent with the paper's numbers.
"""
from __future__ import annotations

import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These ranges are identical to datasets/build_knn_graph.py.
BEHAV_EXCLUDE = {'in_dc', 'out_dc', 'in_count', 'out_count'}
STRUCT_COLS = ['dc', 'pagerank', 'betweenness', 'hits_hub', 'hits_auth', 'kcore', 'triangle']


@st.cache_data(show_spinner='Loading ATNet node features...')
def load_node_features() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, 'datasets', 'HOFINET_NODE_FEAT.csv'))


@st.cache_data(show_spinner='Building tx-graph adjacency (4.7M edges)...')
def load_tx_adjacency(account_to_idx: dict[str, int]) -> Dict[int, set]:
    df = pd.read_csv(os.path.join(BASE, 'datasets', 'HOFINET_EDGES.csv'))
    src = df['source'].map(account_to_idx).to_numpy()
    tgt = df['target'].map(account_to_idx).to_numpy()
    valid = ~(np.isnan(src) | np.isnan(tgt))
    src = src[valid].astype(np.int64)
    tgt = tgt[valid].astype(np.int64)
    adj: Dict[int, set] = defaultdict(set)
    for s, t in zip(src, tgt):
        adj[int(s)].add(int(t))
        adj[int(t)].add(int(s))
    return adj


@st.cache_resource(show_spinner='Building behavioral k-NN ball tree...')
def build_behavioral_knn(behav_cols: tuple[str, ...], k_max: int = 50):
    """Return a fitted NearestNeighbors + the L2-normalized feature matrix."""
    df = load_node_features()
    feats = df[list(behav_cols)].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    nn = NearestNeighbors(n_neighbors=k_max + 1, algorithm='ball_tree',
                          metric='euclidean', n_jobs=-1)
    nn.fit(X)
    return nn, X


@st.cache_data
def behavioral_columns() -> list[str]:
    df = load_node_features()
    agg_cols = [c for c in df.columns if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy'))]
    return [c for c in agg_cols if c not in BEHAV_EXCLUDE]


@st.cache_data
def network_derived_columns() -> list[str]:
    return ['in_count', 'out_count', 'in_dc', 'out_dc']


@st.cache_data
def structural_columns() -> list[str]:
    df = load_node_features()
    return [c for c in STRUCT_COLS if c in df.columns]


@st.cache_data(show_spinner='Loading case-study distribution table...')
def load_case_study_distribution() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, 'results', 'case_study', 'distributions.csv'))


@st.cache_data
def load_case_study_aggregate() -> dict:
    import json
    with open(os.path.join(BASE, 'results', 'case_study', 'aggregate_stats.json')) as f:
        return json.load(f)


def dataset_stats(df: pd.DataFrame) -> dict:
    labels = df['label'].to_numpy()
    return {
        'n_nodes': int(len(df)),
        'n_suspicious': int((labels == 1).sum()),
        'n_benign': int((labels == 0).sum()),
        'rho_pct': float((labels == 1).mean() * 100),
    }


def neighbors_in_graph(node_idx: int, k: int, nn: NearestNeighbors,
                       X: np.ndarray) -> np.ndarray:
    """Return the k nearest behavioral neighbors of one node (excluding self)."""
    _, idx = nn.kneighbors(X[node_idx:node_idx + 1], n_neighbors=k + 1)
    return idx[0][1:]


def induced_edges_tx(node_set: set[int], adj: Dict[int, set]) -> List[Tuple[int, int]]:
    edges = []
    for u in node_set:
        for v in adj.get(u, ()):
            if v in node_set and u < v:
                edges.append((int(u), int(v)))
    return edges


def induced_edges_bhv(node_set: set[int], nn: NearestNeighbors, X: np.ndarray,
                      k: int) -> List[Tuple[int, int]]:
    arr = np.array(sorted(node_set))
    _, knn = nn.kneighbors(X[arr], n_neighbors=k + 1)
    seen = set()
    edges = []
    for src, row in zip(arr, knn):
        for tgt in row[1:]:
            t = int(tgt)
            if t in node_set:
                a, b = (int(src), t) if src < t else (t, int(src))
                if (a, b) not in seen:
                    seen.add((a, b))
                    edges.append((a, b))
    return edges

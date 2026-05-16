"""Graph metric helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np


def edge_homophily(src: np.ndarray, tgt: np.ndarray, labels: np.ndarray) -> float:
    same = np.sum(labels[src] == labels[tgt])
    return float(same / len(src))


def ss_sb_counts(src: np.ndarray, tgt: np.ndarray, labels: np.ndarray) -> tuple[int, int, int]:
    """Return (SS, SB, BB) edge counts."""
    src_susp = labels[src] == 1
    tgt_susp = labels[tgt] == 1
    ss = int((src_susp & tgt_susp).sum())
    bb = int(((~src_susp) & (~tgt_susp)).sum())
    sb = int((src_susp ^ tgt_susp).sum())
    return ss, sb, bb


def ratio_string(ss: int, sb: int) -> str:
    if ss == 0:
        return f'0 : {sb}'
    if sb == 0:
        return f'{ss} : 0'
    return f'1 : {sb / ss:.1f}'


def knn_edges_for_nodes(node_indices: np.ndarray, X: np.ndarray, nn, k: int):
    """Run k-NN queries for a subset of nodes; return arrays of src, tgt indices."""
    _, knn = nn.kneighbors(X[node_indices], n_neighbors=k + 1)
    knn = knn[:, 1:]  # drop self
    src = np.repeat(node_indices, k)
    tgt = knn.reshape(-1)
    return src.astype(np.int64), tgt.astype(np.int64)


def chernoff_homophily_bound(N: int, N_min: int, k: int, eps: float) -> float:
    """Theorem 1 bound from the paper."""
    p = (1 - eps) * (N_min - 1)
    if k >= p:
        return 0.0
    exponent = -((p - k) ** 2) / (2 * p)
    multiplier = (N - 1) / k
    delta = multiplier * np.exp(exponent)
    return max(0.0, 1.0 - delta)

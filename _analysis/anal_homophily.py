import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from typing import Dict, Tuple
from torch_geometric.utils import homophily


# ---------------------------------------------------------------------
# 1) 전처리: source, target 보장
# ---------------------------------------------------------------------
def _ensure_source_target(df: pd.DataFrame) -> pd.DataFrame:
    if "source" not in df.columns or "target" not in df.columns:
        if all(c in df.columns for c in ["wd_fc_sn", "wd_ac_sn", "dps_fc_sn", "dps_ac_sn"]):
            df = df.copy()
            df["source"] = df["wd_fc_sn"].astype(str) + "_" + df["wd_ac_sn"].astype(str)
            df["target"] = df["dps_fc_sn"].astype(str) + "_" + df["dps_ac_sn"].astype(str)
        else:
            raise ValueError("필수 컬럼이 없어 source/target을 생성할 수 없습니다.")
    return df


# ---------------------------------------------------------------------
# 2) 노드 라벨 만들기
# ---------------------------------------------------------------------
def _make_node_labels(
    df: pd.DataFrame,
    label_col: str = "ff_sp_ai",
    positive_values: set = {"SP", "01", "02"},
    agg: str = "mode",
) -> Tuple[Dict[str, int], torch.Tensor]:
    df = df.copy()
    df[label_col] = df[label_col].fillna("0")
    df[label_col] = df[label_col].apply(lambda x: 1 if str(x) in positive_values else 0).astype(int)

    label_dict = defaultdict(list)
    for _, row in df[["source", "target", label_col]].iterrows():
        label_dict[row["source"]].append(row[label_col])
        label_dict[row["target"]].append(row[label_col])

    node_label_map: Dict[str, int] = {}
    for node, labels in label_dict.items():
        if agg == "max":
            node_label_map[node] = int(max(labels))
        else:
            node_label_map[node] = Counter(labels).most_common(1)[0][0]

    node_list = pd.Index(node_label_map.keys())
    node_id_map = {node: i for i, node in enumerate(node_list)}

    y = torch.zeros(len(node_id_map), dtype=torch.long)
    for node, lab in node_label_map.items():
        y[node_id_map[node]] = int(lab)

    return node_id_map, y


# ---------------------------------------------------------------------
# 3) edge_index 만들기
# ---------------------------------------------------------------------
def _build_edge_index(df: pd.DataFrame, node_id_map: Dict[str, int]) -> torch.Tensor:
    src, dst = [], []
    for s, t in zip(df["source"], df["target"]):
        if s in node_id_map and t in node_id_map:
            src.append(node_id_map[s])
            dst.append(node_id_map[t])
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------
# 4) homophily 계산 유틸
# ---------------------------------------------------------------------
def _overall_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    same = (y[edge_index[0]] == y[edge_index[1]]).sum().item()
    total = edge_index.size(1)
    return same / total if total > 0 else float("nan")


def _class_conditional_homophily(edge_index: torch.Tensor, y: torch.Tensor, cls: int) -> float:
    mask_idx = torch.where((y[edge_index[0]] == cls) | (y[edge_index[1]] == cls))[0]
    if mask_idx.numel() == 0:
        return float("nan")
    same = (y[edge_index[0][mask_idx]] == y[edge_index[1][mask_idx]]).sum().item()
    return same / mask_idx.numel()


def _macro_weighted_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> dict:
    vals, counts = torch.unique(y, return_counts=True)
    cls_rates: Dict[int, float] = {}
    micro_num, micro_den = 0.0, 0.0
    for c, cnt in zip(vals.tolist(), counts.tolist()):
        h = _class_conditional_homophily(edge_index, y, int(c))
        cls_rates[int(c)] = h
        if not np.isnan(h):
            micro_num += h * cnt
            micro_den += cnt
    macro = np.nanmean(list(cls_rates.values())) if cls_rates else float("nan")
    micro = (micro_num / micro_den) if micro_den > 0 else float("nan")
    return {"by_class": cls_rates, "macro": macro, "micro": micro}


def _node_level_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    n = y.size(0)
    same_cnt, deg = np.zeros(n, int), np.zeros(n, int)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for s, t in zip(src, dst):
        deg[s] += 1
        if y[s] == y[t]:
            same_cnt[s] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(deg > 0, same_cnt / deg, np.nan)
    return ratio


# ---------------------------------------------------------------------
# 4-1) PyG homophily 안전 계산
# ---------------------------------------------------------------------
def _safe_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> dict:
    results = {}
    results["edge"] = float(homophily(edge_index, y, method="edge"))
    results["node"] = float(homophily(edge_index, y, method="node"))
    try:
        results["node_insensitive"] = float(homophily(edge_index, y, method="node_insensitive"))
    except Exception:
        results["node_insensitive"] = None
    return results


# ---------------------------------------------------------------------
# 5) 통합 메트릭 계산
# ---------------------------------------------------------------------
def compute_homophily_metrics(
    df: pd.DataFrame,
    label_col: str = "ff_sp_ai",
    positive_values: set = {"SP", "01", "02"},
    agg: str = "mode",
) -> dict:
    df = _ensure_source_target(df)
    node_id_map, y = _make_node_labels(df, label_col, positive_values, agg)
    edge_index = _build_edge_index(df, node_id_map)

    overall = _overall_homophily(edge_index, y)
    cc = _macro_weighted_homophily(edge_index, y)
    node_h = _node_level_homophily(edge_index, y)

    pyg_h = _safe_homophily(edge_index, y)

    def _summ(a: np.ndarray) -> dict:
        a = a[~np.isnan(a)]
        if a.size == 0:
            return {"count": 0, "mean": float("nan")}
        return {
            "count": int(a.size),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
        }

    return {
        "overall_homophily": overall,
        "class_conditional": cc["by_class"],
        "macro_avg": cc["macro"],
        "micro_avg": cc["micro"],
        "node_level_summary": {
            "all": _summ(node_h),
            "class_0": _summ(node_h[y.numpy() == 0]),
            "class_1": _summ(node_h[y.numpy() == 1]),
        },
        "edge_count": int(edge_index.size(1)),
        "num_nodes": int(y.size(0)),
        "pyg_homophily": pyg_h,
    }


# ---------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------
if __name__ == "__main__":
    data_name = "HF_FND_2024_03_SP_H1"
    input_path = os.path.join("..", "_datasets", f"{data_name}.csv")

    df = pd.read_csv(input_path)
    metrics = compute_homophily_metrics(df)

    print("===== Homophily Metrics =====")
    print("[Overall]")
    print("  ours:", metrics["overall_homophily"])
    print("  PyG (edge):", metrics["pyg_homophily"]["edge"])
    print()
    print("[Class-Conditional]")
    print("  by class:", metrics["class_conditional"])
    print("  macro avg:", metrics["macro_avg"])
    print("  micro avg:", metrics["micro_avg"])
    print()
    print("[Node-Level Homophily]")
    print("  all:", metrics["node_level_summary"]["all"])
    print("  class_0:", metrics["node_level_summary"]["class_0"])
    print("  class_1:", metrics["node_level_summary"]["class_1"])
    print("  PyG (node):", metrics["pyg_homophily"]["node"])
    print("  PyG (node_insensitive):", metrics["pyg_homophily"].get("node_insensitive"))
    print()
    print("[Graph Info]")
    print("  num_nodes:", metrics["num_nodes"])
    print("  edge_count:", metrics["edge_count"])
    print("==============================")

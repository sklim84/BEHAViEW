import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from torch_geometric.utils import homophily

def _ensure_source_target(df: pd.DataFrame) -> pd.DataFrame:
    """source, target 컬럼이 없으면 결합키로 생성"""
    if 'source' not in df.columns or 'target' not in df.columns:
        df = df.copy()
        df['source'] = df['wd_fc_sn'].astype(str) + '_' + df['wd_ac_sn'].astype(str)
        df['target'] = df['dps_fc_sn'].astype(str) + '_' + df['dps_ac_sn'].astype(str)
    return df

def _make_node_labels(df: pd.DataFrame,
                      label_col: str = 'ff_sp_ai',
                      positive_value: str = 'SP',
                      agg: str = 'mode'):
    """
    계좌별 대표 라벨 생성
    agg = 'mode' 또는 'max'
    """
    df = df.copy()
    # 라벨을 0,1로 정규화
    df[label_col] = df[label_col].fillna(0)
    if df[label_col].dtype != int:
        df[label_col] = df[label_col].apply(lambda x: 1 if x == positive_value else 0).astype(int)

    label_dict = defaultdict(list)
    for _, row in df[['source', 'target', label_col]].iterrows():
        label_dict[row['source']].append(row[label_col])
        label_dict[row['target']].append(row[label_col])

    node_label_map = {}
    for node, labels in label_dict.items():
        if agg == 'max':
            node_label_map[node] = int(max(labels))
        else:
            node_label_map[node] = Counter(labels).most_common(1)[0][0]

    node_list = pd.Index(node_label_map.keys())
    node_id_map = {node: i for i, node in enumerate(node_list)}

    num_nodes = len(node_id_map)
    y = torch.zeros(num_nodes, dtype=torch.long)
    for node, lab in node_label_map.items():
        y[node_id_map[node]] = int(lab)

    return node_id_map, y

def _build_edge_index(df: pd.DataFrame, node_id_map: dict) -> torch.Tensor:
    src = [node_id_map[s] for s in df['source'] if s in node_id_map]
    dst = [node_id_map[t] for t in df['target'] if t in node_id_map]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def _overall_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    same = (y[edge_index[0]] == y[edge_index[1]]).sum().item()
    total = edge_index.size(1)
    return same / total if total > 0 else float('nan')

def _class_conditional_homophily(edge_index: torch.Tensor, y: torch.Tensor, cls: int) -> float:
    mask = (y[edge_index[0]] == cls) | (y[edge_index[1]] == cls)
    if mask.sum().item() == 0:
        return float('nan')
    same = ((y[edge_index[0]] == y[edge_index[1]]) & mask).sum().item()
    total = mask.sum().item()
    return same / total

def _macro_weighted_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> dict:
    """클래스별 homophily의 macro 평균과 support 가중 micro 평균"""
    vals, counts = torch.unique(y, return_counts=True)
    cls_rates = {}
    micro_num, micro_den = 0.0, 0.0
    for c, cnt in zip(vals.tolist(), counts.tolist()):
        h = _class_conditional_homophily(edge_index, y, int(c))
        cls_rates[int(c)] = h
        if not np.isnan(h):
            micro_num += h * cnt
            micro_den += cnt
    macro = np.nanmean(list(cls_rates.values())) if cls_rates else float('nan')
    micro = (micro_num / micro_den) if micro_den > 0 else float('nan')
    return {'by_class': cls_rates, 'macro': macro, 'micro': micro}

def _node_level_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """
    각 노드별로 이웃 중 같은 라벨 비율.
    무차별 방향 그래프처럼 계산하려면 양방향 edge를 추가해도 됨.
    """
    n = y.size(0)
    same_cnt = np.zeros(n, dtype=np.int64)
    deg = np.zeros(n, dtype=np.int64)

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for s, t in zip(src, dst):
        deg[s] += 1
        if y[s] == y[t]:
            same_cnt[s] += 1
    # 분모 0 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(deg > 0, same_cnt / deg, np.nan)
    return ratio  # 길이 n, 각 노드의 homophily 비율

def compute_homophily_metrics(df: pd.DataFrame, label_col: str = 'ff_sp_ai', positive_value: str = 'SP',
                              agg: str = 'mode'):
    # 전체 homophily, 클래스별 homophily, macro/micro 평균, 노드별 homophily 분포 요약을 모두 반환
    df = _ensure_source_target(df)
    node_id_map, y = _make_node_labels(df, label_col, positive_value, agg)
    edge_index = _build_edge_index(df, node_id_map)

    overall = _overall_homophily(edge_index, y)
    cc = _macro_weighted_homophily(edge_index, y)
    node_h = _node_level_homophily(edge_index, y)

    # 노드별 분포 요약 통계
    def _summ(a):
        a = a[~np.isnan(a)]
        if a.size == 0:
            return {}
        return {
            'count': int(a.size), 'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'p25': float(np.percentile(a, 25)), 'p50': float(np.percentile(a, 50)), 'p75': float(np.percentile(a, 75)),
        }

    # 클래스별 노드 분포
    node_stats_all = _summ(node_h)
    node_stats_0 = _summ(node_h[y.numpy() == 0])
    node_stats_1 = _summ(node_h[y.numpy() == 1])

    return {
        'overall_homophily': overall,
        'class_conditional': cc['by_class'],   # 예: {0: 0.9995, 1: 0.12}
        'macro_avg': cc['macro'],
        'micro_avg': cc['micro'],
        'node_level_summary': {'all': node_stats_all, 'class_0': node_stats_0, 'class_1': node_stats_1,},
        'edge_count': int(edge_index.size(1)),
        'num_nodes': int(y.size(0)),
    }


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    # HF_FND_2024_03 : 0.9998
    # HF_FND_2024_03_SP_H1 : 0.9990
    # HF_FND_2024_03_SAMPLE_12K : 0.9885
    # HF_FND_2024_03_SP_H1 :
    data_name = 'HF_FND_2024_03_SP_H1'
    input_path = f'../_datasets/{data_name}.csv'
    df = pd.read_csv(input_path)

    # Homophily 계산
    metrics = compute_homophily_metrics(df)
    print('overall:', metrics['overall_homophily'])
    print('by class:', metrics['class_conditional'])  # {0: ..., 1: ...}
    print('macro:', metrics['macro_avg'], 'micro:', metrics['micro_avg'])
    print('node stats all:', metrics['node_level_summary']['all'])
    print('node stats class_0:', metrics['node_level_summary']['class_0'])
    print('node stats class_1:', metrics['node_level_summary']['class_1'])
import pandas as pd
import torch

import pandas as pd
import torch
from collections import defaultdict, Counter

def compute_node_level_homophily(df: pd.DataFrame):
    """
    거래 단위 레이블(ff_sp_ai)을 기반으로 GAGA 논문식 homophily 계산.
    각 계좌에 대해 단일 노드 레이블을 정한 뒤,
    source-target 노드 간 label이 같은 edge 비율을 계산.
    """

    # source/target 컬럼이 없다면 생성
    if 'source' not in df.columns or 'target' not in df.columns:
        df['source'] = df['wd_fc_sn'].astype(str) + '_' + df['wd_ac_sn'].astype(str)
        df['target'] = df['dps_fc_sn'].astype(str) + '_' + df['dps_ac_sn'].astype(str)

    # 거래 레이블 처리
    df['ff_sp_ai'] = df['ff_sp_ai'].fillna(0)
    df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if x == 'SP' else 0).astype(int)

    # 고유 계좌 목록 및 참여한 거래의 라벨 모음
    label_dict = defaultdict(list)
    for i, row in df.iterrows():
        label_dict[row['source']].append(row['ff_sp_ai'])
        label_dict[row['target']].append(row['ff_sp_ai'])

    # 각 노드별 대표 레이블 설정 (예: max 또는 mode)
    node_label_map = {}
    for node, labels in label_dict.items():
        # 방법 1: max (1이 하나라도 있으면 1)
        # node_label_map[node] = max(labels)
        # 방법 2: 다수결
        node_label_map[node] = Counter(labels).most_common(1)[0][0]

    # 고유 노드 ID 할당
    node_list = pd.Index(node_label_map.keys())
    node_id_map = {node: i for i, node in enumerate(node_list)}

    # label vector 생성
    num_nodes = len(node_id_map)
    label_tensor = torch.zeros(num_nodes, dtype=torch.long)
    for node, label in node_label_map.items():
        label_tensor[node_id_map[node]] = label

    # edge 구성
    edge_index = torch.tensor([
        [node_id_map[s] for s in df['source']],
        [node_id_map[t] for t in df['target']]
    ], dtype=torch.long)

    # GAGA식 homophily 계산
    same_label_count = (label_tensor[edge_index[0]] == label_tensor[edge_index[1]]).sum().item()
    total_edges = edge_index.size(1)
    homophily_ratio = same_label_count / total_edges

    return homophily_ratio, total_edges


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
    homophily, edge_count = compute_node_level_homophily(df)
    print(f"Homophily Ratio: {homophily:.4f} (based on {edge_count} edges)")

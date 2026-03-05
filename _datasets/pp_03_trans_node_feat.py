import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import entropy

# 1. 거래 데이터 불러오기
seed = 2025
# HF_TRANS_TRAN_2024_03_SP_H1 HF_FND_2024_03_SP_H0 HF_FND_2024_03_SP_H1
# HF_FND_2024_03_SAMPLE_12K HF_FND_2024_03_SAMPLE_100K
data_name = 'HF_FND_2024_03_SP_H1'
df = pd.read_csv(f'./{data_name}.csv')

# 2. 계좌 ID 생성 (출금/입금)
df['source'] = df['wd_fc_sn'].astype(str) + ':' + df['wd_ac_sn'].astype(str)
df['target'] = df['dps_fc_sn'].astype(str) + ':' + df['dps_ac_sn'].astype(str)

# 3. 사기 여부 이진화
df['ff_sp_ai'] = df['ff_sp_ai'].replace({'01': 0, '02': 0, 'SP': 1}).fillna(0).astype(int)

# 4. 계좌별 거래 집계 기반 node feature 생성
features_out = df.groupby('source')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('out_')
features_in = df.groupby('target')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('in_')
node_features = pd.concat([features_out, features_in], axis=1).fillna(0)

# 5. 계좌별 사기 label 생성 (입금/출금 중 하나라도 여부 시)
src_label = df.groupby('source')['ff_sp_ai'].max()
tgt_label = df.groupby('target')['ff_sp_ai'].max()
labels = src_label.combine(tgt_label, func=lambda s, t: max(s, t)).astype(int)
node_features['label'] = labels

# 6. 중심성 계산을 위한 그래프 생성
G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph())

# 7. 중심성 계산 값 추가
node_features['dc'] = pd.Series(nx.degree_centrality(G))
node_features['cc'] = pd.Series(nx.closeness_centrality(G))
node_features['bc'] = pd.Series(nx.betweenness_centrality(G, seed=seed))  # 근사치 계산

# 8. md_type 및 fnd_type entropy 계산
def compute_entropy(group):
    counts = group.value_counts(normalize=True)
    return entropy(counts)

md_entropy = df.groupby('source')['md_type'].apply(compute_entropy).rename('md_type_entropy')
fnd_entropy = df.groupby('source')['fnd_type'].apply(compute_entropy).rename('fnd_type_entropy')

node_features = node_features.join(md_entropy, how='left').join(fnd_entropy, how='left')
node_features['account'] = node_features.index

# 9. 최종 저장
node_features.reset_index().to_csv(f'{data_name}_NODE_FEAT.csv', index=False)

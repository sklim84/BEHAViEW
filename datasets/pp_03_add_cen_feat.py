import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse.csgraph import *
from scipy.stats import chi2

def extract_value_from_dict(row, centralities, column_name):
    key = row[column_name]
    value = centralities.get(key, 0)
    return value

seed = 2025
data_name = 'HF_FND_2024_03_SAMPLE_12K'
df = pd.read_csv(f'./{data_name}.csv')

# 그래프 생성
G = nx.from_pandas_edgelist(df, source='source', target='target',
                            edge_attr=['tran_dt', 'tran_tmrg', 'tran_amt', 'md_type', 'fnd_type', 'ff_sp_ai'],
                            create_using=nx.MultiDiGraph())

# Degree Centrality
print('##### Degree Centrality')
degree_centralities = nx.degree_centrality(G)
df['source_dc'] = df.apply(extract_value_from_dict, axis=1, args=(degree_centralities, 'source'))
df['target_dc'] = df.apply(extract_value_from_dict, axis=1, args=(degree_centralities, 'target'))
df_DC = df
df_DC = df_DC.drop(columns=['source_dc', 'target_dc'])

# Closeness Centrality
print('##### Closeness Centrality')
closeness_centralities = nx.closeness_centrality(G)
df['source_cc'] = df.apply(extract_value_from_dict, axis=1, args=(closeness_centralities, 'source'))
df['target_cc'] = df.apply(extract_value_from_dict, axis=1, args=(closeness_centralities, 'target'))
df_CC = df
df_CC = df_CC.drop(columns=['source_cc', 'target_cc'])

# Betweenness Centrality
print('##### Betweenness Centrality')
betweenness_centralities = nx.betweenness_centrality(G, k=1000, seed=seed)
betweenness_centralities = nx.betweenness_centrality(G, seed=seed)
df['source_bc'] = df.apply(extract_value_from_dict, axis=1, args=(betweenness_centralities, 'source'))
df['target_bc'] = df.apply(extract_value_from_dict, axis=1, args=(betweenness_centralities, 'target'))
df_BC = df
df_BC = df_BC.drop(columns=['source_bc', 'target_bc'])

# 체크 저장
print(df.head())
pd.set_option('display.max_columns', None)
df.to_csv(f'./{data_name}_CEN_FEAT_ALL.csv', index=False)

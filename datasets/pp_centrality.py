import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csgraph
from scipy.stats import chi2

def extract_value_from_dict(row, centralities, column_name):
    key = row[column_name]
    value = centralities.get(key, 0)
    return value


def compute_closeness_centrality(G):
    A = nx.adj_matrix(G).tolil()
    D = scipy.sparse.csgraph.floyd_warshall(A, directed=True, unweighted=False)
    n = D.shape[0]
    closeness_centrality = {}

    for r in range(0, n):
        print(f'### compute closeness centrality... {r}/{n}')
        cc = 0.0
        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter(lambda x: not x[1] == np.inf, possible_paths))

        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0

        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s

        closeness_centrality[r] = cc

    return closeness_centrality


data_name = 'hf_ctgan_base_10000_3'
df = pd.read_csv(f'./{data_name}.csv')
df['Source'] = df['wd_fc_sn'].astype(str) + '_' + df['wd_ac_sn'].astype(str)
df['Target'] = df['dps_fc_sn'].astype(str) + '_' + df['dps_ac_sn'].astype(str)
df.drop(columns=['WD_NODE', 'DPS_NODE'], inplace=True)

# 그래프 생성
G = nx.from_pandas_edgelist(df, source='Source', target='Target',
                            edge_attr=['tran_dt', 'tran_tmrg', 'tran_amt', 'md_type', 'fnd_type', 'ff_sp_ai'],
                            create_using=nx.MultiDiGraph())

print(G.number_of_nodes())
print(G.number_of_edges())

# MultiDiGraph에서는 Eigenvector Centrality를 지원하지 않음
# Eigenvector Centrality
# print('##### Eigenvector Centrality')
# eigenvector_centralities = nx.eigenvector_centrality(G, tol=1e-03)
# df['Source_EC'] = df.apply(extract_value_from_dict, axis=1,
#                            args=(eigenvector_centralities, 'Source'))
# df['Target_EC'] = df.apply(extract_value_from_dict, axis=1,
#                            args=(eigenvector_centralities, 'Target'))
# df_EC = df
# df_EC.to_csv(f'../_datasets/{data_name}_CEN_EC.csv', index=False)
# print(df_EC.head())

# Degree Centrality
print('##### Degree Centrality')
degree_centralities = nx.degree_centrality(G)
df['Source_DC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(degree_centralities, 'Source'))
df['Target_DC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(degree_centralities, 'Target'))
df_DC = df
# df_DC.to_csv(f'../_datasets/{data_name}_CEN_DC.csv', index=False)
print(df_DC.head())

# Closeness Centrality
print('##### Closeness Centrality')
closeness_centralities = nx.closeness_centrality(G)
df['Source_CC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(closeness_centralities, 'Source'))
df['Target_CC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(closeness_centralities, 'Target'))
df_CC = df.drop(columns=['Source_DC', 'Target_DC'])
# df_CC.to_csv(f'../_datasets/{data_name}_CEN_CC.csv', index=False)
print(df_CC.head())

# Betweenness Centrality
print('##### Betweenness Centrality')
betweenness_centralities = nx.betweenness_centrality(G, k=1000)
df['Source_BC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(betweenness_centralities, 'Source'))
df['Target_BC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(betweenness_centralities, 'Target'))
df_BC = df.drop(columns=['Source_DC', 'Target_DC', 'Source_CC', 'Target_CC'])
# df_BC.to_csv(f'../_datasets/{data_name}_CEN_BC.csv', index=False)
print(df_BC.head())


# 지표 저장
print(df.head())
pd.set_option('display.max_columns', None)
df.to_csv(f'./{data_name}_CEN_FEAT_ALL.csv', index=False)

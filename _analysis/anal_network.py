import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 설정
seed = 2025
# HF_TRNS_TRAN_2024_03_SP_H1
# HF_FND_SP_H0
# HF_FND_2024_03_SP_H1
# HF_FND_2024_03_SAMPLE_12K
data_name = 'HF_FND_2024_03_SAMPLE_12K'
input_path = f'../_datasets/{data_name}.csv'

# 데이터 로딩
df = pd.read_csv(input_path)

# NetworkX 그래프 생성 (유향 or 무향 선택 가능)
G = nx.DiGraph()  # 또는 nx.Graph() for undirected

# 엣지 추가 (source → target)
edges = df[['source', 'target']].dropna().values.tolist()
G.add_edges_from(edges)

# 기본 통계
print("📊 네트워크 기본 통계")
print(f"- 전체 거래 수: {len(df)}")
print(f"- 전체 SP거래 수: {len(df[df['ff_sp_ai'] == 'SP'])}")
print(f"- 전체 노드 수: {G.number_of_nodes():,}")
print(f"- 전체 엣지 수: {G.number_of_edges():,}")
print(f"- 평균 차수: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"- 네트워크 밀도: {nx.density(G):.6f}")

# 연결 컴포넌트 분석 (무향 그래프 기준)
undirected = G.to_undirected()
components = list(nx.connected_components(undirected))
largest_cc = max(components, key=len)

print(f"- 연결 컴포넌트 수: {len(components)}")
print(f"- 최대 컴포넌트 크기: {len(largest_cc)} 노드")

# 중심성 분석 (상위 5개 출력)
print("\n🔎 상위 중심성 노드 (degree centrality)")
dc = nx.degree_centrality(G)
for node, score in sorted(dc.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {node}: {score:.8e}")

print("\n🔎 상위 근접 중심성 노드 (closeness centrality)")
cc = nx.closeness_centrality(G)
for node, score in sorted(cc.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {node}: {score:.8e}")

print("\n🔎 상위 중개 중심성 노드 (betweenness centrality)")
bc = nx.betweenness_centrality(G, k=1000, seed=seed)
for node, score in sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {node}: {score:.8e}")

# # (선택) 시각화: 최대 연결 컴포넌트만
# subG = G.subgraph(largest_cc)
# pos = nx.spring_layout(subG, k=0.15, seed=42)
# plt.figure(figsize=(12, 8))
# nx.draw(subG, pos, node_size=10, edge_color='gray', alpha=0.6)
# plt.title("H1-Hop 거래 네트워크 (최대 컴포넌트)")
# plt.show()

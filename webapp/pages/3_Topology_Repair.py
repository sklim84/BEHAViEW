"""Page 3: Topology Repair centerpiece. Interactive account selector with side-by-side ego subgraph viz."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from lib.data import (
    behavioral_columns, build_behavioral_knn, induced_edges_bhv,
    induced_edges_tx, load_case_study_aggregate, load_case_study_distribution,
    load_node_features, load_tx_adjacency, neighbors_in_graph,
)
from lib.viz import C_BHV_EDGE, C_TX_EDGE, histogram_shift, render_ego_subgraph

st.set_page_config(page_title='Topology Repair', page_icon='🔄', layout='wide')

st.title('3. Topology Repair (RQ1)')
st.caption('Pick a suspicious account and watch its 1-hop neighborhood get repaired.')

df = load_node_features()
labels = df['label'].to_numpy()
accounts = df['account'].to_numpy()
account_to_idx = {acc: i for i, acc in enumerate(accounts)}
behav_cols = behavioral_columns()
nn, X = build_behavioral_knn(tuple(behav_cols), k_max=50)
adj_tx = load_tx_adjacency(account_to_idx)
distribution = load_case_study_distribution()
aggregate = load_case_study_aggregate()

# ---------- Side: account selection ----------
st.subheader('Account selection')

mode = st.radio(
    'How do you want to pick an account?',
    ['Paper default (idx 62263)', 'Random suspicious sample',
     'Filter by tx-S share', 'Type an index'],
    horizontal=True,
)

if mode == 'Paper default (idx 62263)':
    ego_idx = 62263
elif mode == 'Random suspicious sample':
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 0
    if st.button('Resample'):
        st.session_state.random_seed += 1
    rng = np.random.default_rng(st.session_state.random_seed)
    susp = distribution['idx'].to_numpy()
    ego_idx = int(rng.choice(susp))
elif mode == 'Filter by tx-S share':
    lo, hi = st.slider('tx-S share range', 0.0, 1.0, (0.0, 0.10), step=0.05,
                       help='Restrict to suspicious accounts whose tx ego is this fraction suspicious.')
    pool = distribution[(distribution['tx_S_ratio'] >= lo) &
                        (distribution['tx_S_ratio'] <= hi) &
                        (distribution['tx_deg'].between(8, 60))]
    st.caption(f'{len(pool):,} candidates with tx_deg in [8,60].')
    if len(pool) == 0:
        st.warning('No candidate matches the filter — relax it.')
        st.stop()
    if 'filter_idx_pick' not in st.session_state:
        st.session_state.filter_idx_pick = 0
    if st.button('Resample inside filter'):
        st.session_state.filter_idx_pick += 1
    ego_idx = int(pool.iloc[st.session_state.filter_idx_pick % len(pool)]['idx'])
else:  # 'Type an index'
    ego_idx = st.number_input('Suspicious account index', min_value=0,
                              max_value=int(len(df) - 1), value=62263, step=1)
    if labels[ego_idx] != 1:
        st.warning('That node is not labeled suspicious. The repair story applies to suspicious egos.')

st.write(f'**Selected:** idx = {ego_idx}, account = `{accounts[ego_idx]}`, label = {int(labels[ego_idx])}')

# ---------- Compute neighborhoods ----------
k_view = st.slider('Behavioral k-NN $k$ for this account', min_value=5, max_value=20, value=10)

tx_neighbors = sorted(adj_tx.get(ego_idx, set()))
bhv_neighbors = neighbors_in_graph(ego_idx, k_view, nn, X).tolist()

if len(tx_neighbors) == 0:
    st.warning('Selected account has no 1-hop transaction neighbors. Try another one.')
    st.stop()

# Neighborhood compositions
def n_susp(idx_list):
    if len(idx_list) == 0:
        return 0
    return int(np.sum(labels[idx_list] == 1))

tx_S = n_susp(tx_neighbors)
tx_B = len(tx_neighbors) - tx_S
bhv_S = n_susp(bhv_neighbors)
bhv_B = len(bhv_neighbors) - bhv_S

# Induced edges
tx_set = set(tx_neighbors + [ego_idx])
bhv_set = set(bhv_neighbors + [ego_idx])
tx_induced = induced_edges_tx(tx_set, adj_tx)
bhv_induced = induced_edges_bhv(bhv_set, nn, X, k_view)

# ---------- Side-by-side viz ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
neighbors_meta_tx = [{'idx': i, 'label': int(labels[i])} for i in tx_neighbors]
neighbors_meta_bhv = [{'idx': i, 'label': int(labels[i])} for i in bhv_neighbors]
render_ego_subgraph(axes[0], ego_idx, neighbors_meta_tx, tx_induced,
                    f'Transaction graph (1-hop, |N|={len(tx_neighbors)})',
                    C_TX_EDGE)
render_ego_subgraph(axes[1], ego_idx, neighbors_meta_bhv, bhv_induced,
                    f'Recovered neighborhood (k-NN, k={k_view})',
                    C_BHV_EDGE)
st.pyplot(fig, use_container_width=True)

# ---------- 4-setting trace table ----------
st.subheader('Trace through the four BEHAViEW settings')

# Aggregation neighborhoods
def s_share(n_s, total):
    return float(n_s) / total if total > 0 else 0.0

tx_share = s_share(tx_S, len(tx_neighbors))
bhv_share = s_share(bhv_S, len(bhv_neighbors))
tx_pool_share = s_share(tx_S + 1, len(tx_neighbors) + 1)
bhv_pool_share = s_share(bhv_S + 1, len(bhv_neighbors) + 1)

paper_F1 = {'(a)': 0.266, '(b)': 0.671, '(c)': 0.197, '(d)': 0.673}
trace = pd.DataFrame([
    {'Setting': '(a) tx, node',  'Aggregation source': f'{len(tx_neighbors)} tx neighbors',
     'S-share': f'{tx_share*100:.1f}%',
     'Naive label (majority)': 'suspicious ✅' if tx_share >= 0.5 else 'benign ❌',
     'Table 3 (GBT) F1_susp': paper_F1['(a)']},
    {'Setting': '(b) bhv, node', 'Aggregation source': f'{len(bhv_neighbors)} bhv peers',
     'S-share': f'{bhv_share*100:.1f}%',
     'Naive label (majority)': 'suspicious ✅' if bhv_share >= 0.5 else 'benign ❌',
     'Table 3 (GBT) F1_susp': paper_F1['(b)']},
    {'Setting': '(c) tx, pool',  'Aggregation source': f'ego + {len(tx_neighbors)} tx',
     'S-share': f'{tx_pool_share*100:.1f}%',
     'Naive label (majority)': 'suspicious ✅' if tx_pool_share >= 0.5 else 'benign ❌',
     'Table 3 (GBT) F1_susp': paper_F1['(c)']},
    {'Setting': '(d) bhv, pool', 'Aggregation source': f'ego + {len(bhv_neighbors)} bhv',
     'S-share': f'{bhv_pool_share*100:.1f}%',
     'Naive label (majority)': 'suspicious ✅' if bhv_pool_share >= 0.5 else 'benign ❌',
     'Table 3 (GBT) F1_susp': paper_F1['(d)']},
])
st.dataframe(trace, use_container_width=True, hide_index=True)

with st.expander('What is a *naive label*?'):
    st.markdown("""
The Table column shows what a *label-majority* classifier would predict given each
setting's aggregation neighborhood: predict suspicious if the share of suspicious
neighbors is $\\ge 50\\%$.

The trained GNN classifier is much richer than a label vote, but for a suspicious
account whose neighborhood is overwhelmingly benign (S-share $\\ll 50\\%$), the
encoder's aggregated representation lands inside the benign cluster — the
classifier mislabels it. This single-account flip multiplied across thousands of
suspicious accounts yields the F1 jumps in Table~3.
""")

# ---------- Aggregate population shift ----------
st.markdown('---')
st.subheader('Population-level shift')

tx_ratios = distribution['tx_S_ratio'].dropna().to_numpy()
bhv_ratios = distribution['bhv_S_ratio'].to_numpy()
fig2, ax = plt.subplots(figsize=(7.5, 4))
histogram_shift(ax, tx_ratios, bhv_ratios, aggregate)
st.pyplot(fig2)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Suspicious accounts', f"{aggregate['n_suspicious_total']:,}")
c2.metric('Tx majority-S share',  f"{aggregate['p_tx_majority_susp']*100:.1f}%")
c3.metric('Bhv majority-S share', f"{aggregate['p_bhv_majority_susp']*100:.1f}%")
c4.metric('Median shift', f"{aggregate['shift_median']:+.3f}")

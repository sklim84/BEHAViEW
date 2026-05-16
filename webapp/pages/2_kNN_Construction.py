"""Page 2: Behavioral k-NN construction with adjustable k."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from lib.data import behavioral_columns, build_behavioral_knn, load_node_features
from lib.metrics import edge_homophily, knn_edges_for_nodes, ratio_string, ss_sb_counts

st.set_page_config(page_title='k-NN Construction', page_icon='🔧', layout='wide')

st.title('2. Behavioral k-NN Construction')
st.caption('Build the recovered neighborhood graph $\\mathcal{G}_{bhv}$ from the 20 behavioral features alone, then read off its class-level structure.')

st.markdown("""
**Construction recipe** (Section 3.3 of the paper):
1. Take the 20 behavioral features $\\mathbf{x}_i^{\\text{bhv}} \\in \\mathbb{R}^{20}$.
2. Apply `StandardScaler` to remove unit/scale differences across features.
3. L2-normalize so that Euclidean distance is monotone in cosine similarity.
4. Build a ball-tree and connect each node to its $k$ nearest behavioral neighbors.

Connectivity-derived features (degree, counts, centrality) are deliberately
excluded so the repair does not leak the adversarial topology back in.
""")

df = load_node_features()
labels = df['label'].to_numpy()
behav_cols = behavioral_columns()
st.write(f'Using {len(behav_cols)} behavioral features. Total nodes: **{len(df):,}**.')

K_MAX = 50
nn, X = build_behavioral_knn(tuple(behav_cols), k_max=K_MAX)

st.markdown('---')
st.subheader('Pick $k$ and measure homophily')

k = st.slider('$k$ (neighbors per node)', min_value=5, max_value=K_MAX, value=10, step=1)

sample_size = st.select_slider(
    'k-NN sample size for the homophily measurement',
    options=[5_000, 10_000, 50_000, 100_000, 452_816],
    value=50_000,
    help='452,816 = full graph (≈45 sec). Smaller samples return in a few seconds.',
)

rng = np.random.default_rng(0)
if sample_size >= len(df):
    sample_indices = np.arange(len(df))
else:
    sample_indices = rng.choice(len(df), size=sample_size, replace=False)

with st.spinner(f'Querying k-NN for {len(sample_indices):,} nodes at $k$={k}...'):
    src, tgt = knn_edges_for_nodes(sample_indices, X, nn, k)

h = edge_homophily(src, tgt, labels)
ss, sb, bb = ss_sb_counts(src, tgt, labels)
total = ss + sb + bb

c1, c2, c3, c4 = st.columns(4)
c1.metric('Edges queried', f'{total:,}')
c2.metric('Edge homophily', f'{h:.3f}')
c3.metric('S-S : S-B', ratio_string(ss, sb))
c4.metric('B-B share', f'{bb / total * 100:.1f}%')

st.markdown('---')
st.subheader('How homophily moves with $k$')

if st.button('Sweep $k \\in \\{5, 10, 20, 50\\}$ on this sample'):
    rows = []
    for k_try in (5, 10, 20, 50):
        s, t = knn_edges_for_nodes(sample_indices, X, nn, k_try)
        h_try = edge_homophily(s, t, labels)
        ss_t, sb_t, _ = ss_sb_counts(s, t, labels)
        rows.append({'k': k_try, 'edges': len(s),
                     'homophily': round(h_try, 4),
                     'S-S:S-B': ratio_string(ss_t, sb_t)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption('The paper reports the spread across $k$ as 0.021 $F1_{susp}$ — the recovered graph is robust to $k$.')

with st.expander('Reference: paper Table 2 (full-graph numbers)'):
    st.dataframe(pd.DataFrame([
        {'Graph': 'Transaction', 'Homophily': 0.690, 'S-S:S-B': '1:5.7'},
        {'Graph': 'Structural view (analysis)', 'Homophily': 0.965, 'S-S:S-B': '1:9.7'},
        {'Graph': 'Behavioral view (proposed)', 'Homophily': 0.981, 'S-S:S-B': '1:1.4'},
    ]), use_container_width=True, hide_index=True)

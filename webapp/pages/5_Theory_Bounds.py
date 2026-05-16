"""Page 5: Theory 1 bound calculator + measured-vs-predicted comparison."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from lib.metrics import chernoff_homophily_bound

st.set_page_config(page_title='Theory Bounds', page_icon='📐', layout='wide')

st.title('5. Theory Bounds')
st.caption('Theorem 1 (probabilistic homophily recovery) — plug $N, N_{\\min}, k, \\varepsilon$ and read off the bound.')

st.markdown(r"""
**Theorem 1** (Appendix B): under the separability assumption with margin
$\varepsilon$ and $k \leq (1-\varepsilon)(N_{\min}-1) - 1$,
$$
h(G_{\text{knn}, k}) \;\geq\; 1 - \frac{N - 1}{k}\,
\exp\!\left(-\frac{((N_{\min}-1)(1-\varepsilon) - k)^{2}}{2 (N_{\min}-1)(1-\varepsilon)}\right).
$$
""")

st.subheader('Pick parameters')

dataset_presets = {
    'ATNet (HOFINET)': {'N': 452_816, 'N_min': 9_644, 'k': 10, 'eps': 0.05, 'measured_gap': 0.291, 'h_tx': 0.690},
    'AMLworld (HI-Small)': {'N': 515_088, 'N_min': 6_335, 'k': 10, 'eps': 0.05, 'measured_gap': 0.094, 'h_tx': 0.886},
    'AMLNet': {'N': 11_000, 'N_min': 1_487, 'k': 10, 'eps': 0.05, 'measured_gap': 0.109, 'h_tx': 0.791},
    'Custom': None,
}

preset = st.selectbox('Dataset preset', list(dataset_presets.keys()), index=0)

c1, c2, c3, c4 = st.columns(4)
if preset == 'Custom':
    N = c1.number_input('$N$ (total nodes)', min_value=100, value=100_000, step=1000)
    N_min = c2.number_input('$N_{\\min}$ (minority class size)', min_value=10, value=2000, step=10)
    k = c3.number_input('$k$', min_value=1, value=10, step=1)
    eps = c4.slider('$\\varepsilon$', min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    measured_gap = None
    h_tx = None
else:
    p = dataset_presets[preset]
    N = c1.number_input('$N$', min_value=100, value=p['N'], step=1)
    N_min = c2.number_input('$N_{\\min}$', min_value=10, value=p['N_min'], step=1)
    k = c3.number_input('$k$', min_value=1, value=p['k'], step=1)
    eps = c4.slider('$\\varepsilon$', min_value=0.0, max_value=0.5, value=p['eps'], step=0.01)
    measured_gap = p['measured_gap']
    h_tx = p['h_tx']

bound = chernoff_homophily_bound(int(N), int(N_min), int(k), float(eps))

st.subheader('Result')
b1, b2 = st.columns(2)
b1.metric('Lower bound on $h(G_{knn,k})$', f'{bound:.4f}' if bound > 0 else '< 0 (vacuous)')
if h_tx is not None:
    gap_pred = max(0.0, bound - h_tx)
    b2.metric('Lower bound on the homophily gap',
              f'{gap_pred:.4f}',
              delta=f'measured = {measured_gap:.4f}, slack = {gap_pred - measured_gap:+.4f}',
              delta_color='off')
else:
    b2.write('Custom mode — no measured gap to compare.')

st.markdown('---')
st.subheader('Cross-dataset summary')
rows = []
for name, p in dataset_presets.items():
    if p is None:
        continue
    bnd = chernoff_homophily_bound(p['N'], p['N_min'], p['k'], p['eps'])
    rows.append({
        'Dataset': name,
        'N': p['N'],
        'N_min': p['N_min'],
        'k': p['k'],
        'ε': p['eps'],
        'h(tx)': p['h_tx'],
        'h(knn) ≥': round(bnd, 4),
        'predicted gap ≥': round(max(0.0, bnd - p['h_tx']), 4),
        'measured gap': p['measured_gap'],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption('Slack columns reproduce Appendix B: ATNet within 0.019, AMLworld within ~0.02, AMLNet within ~0.10.')

with st.expander('What does each parameter mean?'):
    st.markdown(r"""
- $N$ — total number of accounts.
- $N_{\min}$ — number of accounts in the minority (suspicious) class.
- $k$ — neighbors per node in the behavioral k-NN graph.
- $\varepsilon$ — separability margin: probability that a different-class node is
  closer than a same-class node, in the L2-normalized behavioral feature space.
  Small $\varepsilon$ ⇒ classes are well-separated in feature space ⇒ tighter bound.
""")

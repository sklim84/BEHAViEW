"""Page 1: Dataset and feature taxonomy."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from lib.data import (
    behavioral_columns, dataset_stats, load_node_features,
    network_derived_columns, structural_columns,
)

st.set_page_config(page_title='Dataset & Features', page_icon='📊', layout='wide')

st.title('1. Dataset & Features')
st.caption('ATNet node features and the three feature categories used by BEHAViEW.')

df = load_node_features()
stats = dataset_stats(df)

st.subheader('ATNet (HOFINET) statistics')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Nodes |V|', f"{stats['n_nodes']:,}")
c2.metric('Suspicious', f"{stats['n_suspicious']:,}")
c3.metric('Benign', f"{stats['n_benign']:,}")
c4.metric('Suspicious rate ρ', f"{stats['rho_pct']:.2f}%")

st.markdown('---')
st.subheader('Feature taxonomy')

behav = behavioral_columns()
net = network_derived_columns()
struct = structural_columns()

taxonomy = pd.DataFrame([
    {'Category': 'Behavioral (in $G_{bhv}$ + encoder)', '# features': len(behav),
     'Features': ', '.join(behav)},
    {'Category': 'Network-derived (encoder only)', '# features': len(net),
     'Features': ', '.join(net)},
    {'Category': 'Structural (analysis only)', '# features': len(struct),
     'Features': ', '.join(struct)},
])
st.dataframe(taxonomy, use_container_width=True, hide_index=True)

with st.expander('Why exclude network-derived and structural features from the view?'):
    st.markdown("""
The recovered neighborhood graph $\\mathcal{G}_{bhv}$ is meant to repair the topology,
so it must be **independent of the transaction edges**. Any feature derived from
edge structure (degree counts, centrality, PageRank, k-core, triangles, …) inherits
the adversarial mule pattern and would leak the broken structure back into the
auxiliary view. The 20 behavioral features (amount statistics, time-window
frequencies, transaction-type entropies) are pure account behavior with no
connectivity input.
""")

st.markdown('---')
st.subheader('Sample rows')

label_choice = st.radio('Label filter', ['suspicious', 'benign', 'both'],
                        horizontal=True)
n_rows = st.slider('Rows to show', 5, 50, 10)
if label_choice == 'suspicious':
    sample = df[df['label'] == 1].head(n_rows)
elif label_choice == 'benign':
    sample = df[df['label'] == 0].head(n_rows)
else:
    sample = df.head(n_rows)

view_cols = st.multiselect(
    'Columns to display',
    options=behav + net + structural_columns() + ['label', 'account'],
    default=['account', 'label', 'out_mean', 'in_mean', 'out_count', 'in_count', 'md_type_entropy'],
)
st.dataframe(sample[view_cols], use_container_width=True, hide_index=True)

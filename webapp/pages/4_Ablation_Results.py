"""Page 4: Ablation results — load pre-computed experiment CSVs and pivot."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title='Ablation Results', page_icon='📉', layout='wide')

st.title('4. Ablation Results (RQ1–RQ4)')
st.caption('F1_susp / AUROC / AUPRC across the four settings × ten encoders × three datasets, from cached experiment CSVs.')

st.markdown("""
The training runs themselves take hours on an H100; this page reads their saved
per-seed CSVs and aggregates them in the same way as the paper tables.
""")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = os.path.join(BASE, 'results', name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def parse_hofinet_ab() -> pd.DataFrame:
    """Build a (encoder, setting) -> mean ± std F1_susp table from the HOFINET 4-setting sweep CSV."""
    df = load_csv('exp_results_hofinet_ab.csv')
    if df is None or 'Model' not in df.columns:
        return pd.DataFrame()
    # Model strings look like hof_gbt_a_s2024 -> parse encoder + setting
    def parse(row):
        m = row['Model']
        parts = m.split('_')
        if len(parts) < 4:
            return None, None
        enc = parts[1].upper()
        setting = parts[2]
        return enc, setting
    df[['encoder', 'setting']] = df.apply(parse, axis=1, result_type='expand')
    df = df.dropna(subset=['encoder', 'setting'])
    grp = df.groupby(['encoder', 'setting'])['f1_1'].agg(['mean', 'std', 'count']).reset_index()
    grp['cell'] = grp.apply(lambda r: f"{r['mean']:.3f}±{r['std']:.3f}", axis=1)
    pivot = grp.pivot(index='encoder', columns='setting', values='cell')
    # Order columns
    desired = [c for c in ['a', 'b', 'c', 'd'] if c in pivot.columns]
    return pivot[desired]


st.subheader('HOFINET 4-setting × 10 encoders (Table 3 reproduction)')
table_ab = parse_hofinet_ab()
if len(table_ab) > 0:
    st.dataframe(table_ab, use_container_width=True)
    st.caption('(a) tx, node · (b) bhv, node · (c) tx, pool · (d) bhv, pool')
else:
    st.info('CSV `results/exp_results_hofinet_ab.csv` not found or in an unexpected format.')

st.markdown('---')
st.subheader('Browse other result CSVs')

result_csvs = sorted([f for f in os.listdir(os.path.join(BASE, 'results'))
                      if f.endswith('.csv')])
choice = st.selectbox('Pick a CSV under results/', result_csvs)
if choice:
    df = load_csv(choice)
    st.write(f'**{choice}** — {len(df):,} rows, {df.shape[1]} columns')
    st.dataframe(df.head(50), use_container_width=True)
    with st.expander('Columns'):
        st.write(list(df.columns))

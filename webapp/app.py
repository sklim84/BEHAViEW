"""BEHAViEW interactive walkthrough — home page."""
import streamlit as st

st.set_page_config(
    page_title='BEHAViEW Walkthrough',
    page_icon='🔧',
    layout='wide',
)

st.title('BEHAViEW — Topology Repair for AML')
st.caption('Interactive walkthrough of *Self-Supervised Money Laundering Detection via Topology Repair*')

st.markdown("""
This app lets you step through every analytical stage of the paper on the ATNet
(HOFINET) dataset, with the same loaders and the same numerical procedures used
in the offline scripts. Use the sidebar to navigate between pages.
""")

st.subheader('Headline numbers (ATNet)')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Edge homophily', '0.690 → 0.981', '+0.291')
c2.metric('S-S : S-B ratio', '1:5.7 → 1:1.4', 'topology repair')
c3.metric('F1_susp, GBT (a→b)', '0.266 → 0.671', '+0.405')
c4.metric('Majority-S 1-hop egos', '47.7% → 71.6%', '+23.9 pp')

st.markdown('---')
st.subheader('Pipeline pages')

st.markdown("""
| # | Page | What you can do |
|---|------|------------------|
| 1 | **Dataset & Features** | Inspect the 20+4+7 feature taxonomy and the suspicious / benign class balance. |
| 2 | **Behavioral k-NN Construction** | Build the recovered neighborhood graph $\\mathcal{G}_{bhv}$ for an adjustable $k$ and read off the resulting homophily. |
| 3 | **Topology Repair (RQ1)** | Centerpiece — pick a suspicious account and view its 1-hop ego in the transaction graph vs the recovered graph side by side; trace it through the four ablation settings. |
| 4 | **Ablation Results (RQ1–RQ4)** | Browse the F1_susp / AUROC / AUPRC tables for the 4 settings × 10 encoders × 3 datasets. |
| 5 | **Theory Bounds** | Plug $N, N_{\\min}, k, \\varepsilon$ into Theorem 1 and compare the lower bound to the measured homophily gap. |
""")

st.markdown('---')
with st.expander('How to launch this app'):
    st.code(
        '# from the repository root\n'
        'pip install streamlit\n'
        'git lfs pull   # only the first time; pulls the HOFINET CSVs\n'
        'streamlit run webapp/app.py',
        language='bash',
    )

with st.expander('Reproducibility note'):
    st.markdown("""
- **ATNet** (anonymized HOFINET) is read from `datasets/HOFINET_NODE_FEAT.csv` and `datasets/HOFINET_EDGES.csv`.
  These are tracked in Git LFS — run `git lfs pull` after cloning.
- The behavioral k-NN graph uses the same 20 features as `datasets/build_knn_graph.py` (BEHAV variant).
- Pre-computed case-study tables under `results/case_study/` come from
  `analysis/case_study_topology_repair.py`.
""")

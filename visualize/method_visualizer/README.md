# BehaView 3D Method Comparator

Independent Streamlit app for comparing the repository's BehaView setting
against existing transaction-view GCL settings on HOFINET and AMLworld.

The app reads the current local CSV assets:

- `datasets/hofinet/HOFINET_NODE_FEAT.csv`
- `datasets/hofinet/HOFINET_EDGES.csv`
- `datasets/amlworld/AMLWORLD_NODE_FEAT.csv`
- `datasets/amlworld/AMLWORLD_EDGES.csv`
- cached experiment CSVs under `results/`

It builds a sampled transaction subgraph and a dynamic behavioral k-NN graph,
then renders two selected methods side by side in 3D. The x-axis is always a
label-aligned suspicious component, while PCA or t-SNE supplies the remaining
two axes.

## Run

From the repository root:

```bash
python3 -m streamlit run method_visualizer/app.py
```

If a fresh environment is missing UI packages:

```bash
python3 -m pip install -r method_visualizer/requirements.txt
python3 -m streamlit run method_visualizer/app.py
```

## Notes

- The default right-hand method is BehaView: behavioral k-NN view plus subgraph
  pooling, matching setting `(d)` in `models/subgraph_cl.py`.
- The app exposes loss choices per base encoder for interactive comparison.
  Full experiments dispatch `--loss` through `models/subgraph_cl.py::compute_loss`.
  The app itself does not re-train the GNN; it applies a lightweight
  visualization profile and emits command templates with the selected loss.
- For `k` values other than 10, the page computes behavioral k-NN dynamically
  for visualization. Full training needs the corresponding `*_k{k}.csv` graph
  file to exist under `datasets/`.

# Reviewer 3 — Presentation Quality, Completeness, and Practical Impact

**Paper:** BECON: Behavioral Subgraph Contrast for Anti-Money Laundering in Low-Homophily Transaction Networks

---

## 1. Summary

This paper proposes BECON, a graph contrastive learning framework for anti-money laundering (AML) that constructs a behavioral k-NN graph as an auxiliary contrastive view and applies subgraph-level pooling. The authors systematically analyze two orthogonal axes—view construction (augmented vs. behavioral k-NN) and contrastive level (node vs. subgraph)—across 10 GCL encoders, yielding five research questions. Experiments on a real Korean inter-bank transfer dataset (HOFINET) and a synthetic benchmark (AMLworld HI-Small) show that the behavioral k-NN view dramatically improves suspicious-to-suspicious connectivity, enabling self-supervised representations that match or exceed supervised baselines. The paper also identifies BatchNorm as the dominant factor for encoder performance under extreme class imbalance.

---

## 2. Strengths

1. **Clear and well-motivated problem framing (Section 1, lines 119--134).** The two challenges of low homophily and extreme class imbalance are concretely quantified (S-S/S-B = 1:5.7, fraud ratio 2.13%), making the motivation immediately tangible. The introductory Figure 1 is effective.

2. **Systematic 2x2 ablation design (Section 3, Figure 2, Section 4.2).** The decomposition into view axis and contrastive-level axis is clean and allows each contribution to be isolated. The consistent ordering (d) > (b) > (a) > (c) across all encoders is a strong empirical result.

3. **Mechanistic explanation via S-S/S-B ratios (Tables 2 and 4).** The paper does not merely report performance gains but provides a quantitative explanation for *why* the behavioral k-NN view works—by restoring homophily from 0.690 to 0.981. This is the paper's strongest analytical contribution.

4. **Practical significance (RQ5, Table 5).** Demonstrating that a label-free self-supervised method matches MLP/XGBoost/GraphSAGE (F1_susp 0.682 vs. 0.674--0.678) on a real production dataset is highly relevant for industry deployment where labels are scarce.

5. **Broad encoder coverage (10 encoders x 4 settings = 40 configurations).** The finding that per-layer BatchNorm is the decisive factor (Section 4.3) rather than the contrastive paradigm itself is a valuable insight for practitioners.

6. **Two-dataset validation (RQ4).** Testing on both a proprietary real dataset and a public benchmark strengthens claims of generalizability.

7. **Mathematical formulation is complete and clear (Eqs. 1--7).** All symbols are defined before use. The notation is consistent throughout.

---

## 3. Weaknesses

1. **Related Work section is completely empty (Section 2, line 153).** This is a critical gap. A top-venue submission *must* include Related Work covering at minimum: (a) GNN-based fraud detection (CARE-GNN, PC-GNN, GTAN, etc.), (b) graph contrastive learning methods and augmentation strategies, (c) k-NN graph construction in GCL (MLGCL, GCPAL), (d) handling low homophily in GNNs, and (e) class imbalance techniques for graphs. Without this section, the paper cannot position itself relative to prior art. **Suggestion:** Write 1.5--2 pages covering the five areas above with explicit differentiation from GCPAL and MLGCL.

2. **Conclusion section is empty (Section 5, line 462).** There is no conclusion, no summary of findings, no limitations discussion, and no future work beyond the section title. **Suggestion:** Write a conclusion that (a) summarizes the three main findings (behavioral k-NN view, subgraph pooling conditionality, BatchNorm dominance), (b) acknowledges limitations, and (c) outlines future directions (e.g., dynamic graphs, multi-hop k-NN, inductive settings).

3. **No direct comparison with GCPAL [lu2024graph].** GCPAL is the most directly related work—it also uses k-NN views for AML—yet it appears only in a citation in the introduction (line 130) with no experimental comparison or qualitative differentiation. **Suggestion:** Add GCPAL as a baseline in RQ1/RQ5, or at minimum provide a detailed qualitative comparison table in Related Work explaining how BECON differs (behavioral vs. structural features for k-NN, subgraph pooling, BatchNorm analysis).

4. **Paper is entirely in Korean except for section headings and table headers.** While the instructions state Korean text is acceptable, a KDD/WWW submission must be in English. The current state is a draft; the full English translation is required before submission.

5. **No limitations section.** Top venues increasingly require explicit discussion of limitations. Key limitations to address include:
   - The k-NN graph requires computing pairwise behavioral similarity for all N nodes (O(N log N) with ball tree, but still 452K nodes).
   - The method assumes behavioral features are available and discriminative, which may not hold for all AML datasets.
   - The 10%/80% train/test split for LogReg evaluation is generous; real-world label availability may be far lower.
   - The method is transductive; inductive capability for new accounts is not demonstrated.

6. **Missing standard deviation for BECON (d) in Table 5 (HOFINET section, line 436).** The self-supervised result reports F1_susp = 0.682 without a standard deviation, while all supervised baselines include one. Also, AUPRC is listed as "---" for BECON. **Suggestion:** Report all metrics consistently for all models.

7. **Page count is approximately 7--8 pages of content (excluding references).** With Related Work and Conclusion added, this should reach the 8--10 page target, but currently falls short of a complete submission.

8. **Figure 3 (RQ3) uses subfigure with different y-axis scales.** While the caption mentions this ("y축 스케일이 다름에 유의"), the difference is extreme (0.0--0.7 vs. 0.0--0.09). This visual design could mislead a quick reader into thinking BN-free models perform comparably. **Suggestion:** Consider a single figure with a broken y-axis or log scale, or add explicit annotations showing the 10x gap.

9. **No hyperparameter sensitivity analysis.** The paper uses fixed hyperparameters (k=10 for k-NN, 3 GNN layers, lr=0.001, etc.) without justification or sensitivity study. **Suggestion:** Add an appendix or subsection showing sensitivity to k (k-NN neighbors), number of GNN layers, and learning rate.

10. **No computational cost analysis.** The paper does not report training time, memory usage, or the cost of k-NN construction. For a 452K-node graph, the k-NN construction step could be a bottleneck. **Suggestion:** Add a table comparing wall-clock time for k-NN construction, GCL training, and inference.

11. **Duplicate LaTeX packages.** Lines 5/24 (multirow), 11/19 (bm), 15/21 (subfigure), 16/23 (booktabs), 20/7 (adjustbox). While this does not affect compilation, it reflects incomplete cleanup. The deprecated `subfigure` package should be replaced with `subcaption` or `subfig`.

12. **Table formatting inconsistency.** Table 3 (RQ4) uses checkmark/blank for BN column, while Table 1 (RQ1) uses descriptive text ("Per-layer", "Final", "None"). **Suggestion:** Unify the BN indicator format across all tables.

---

## 4. Questions for Authors

1. **Why is GCPAL not included as a baseline?** Given that GCPAL [lu2024graph] is the most directly comparable method (k-NN view for AML with GCL), its absence as an experimental baseline is a significant gap. Can the authors provide a comparison, or explain why it was excluded?

2. **How sensitive is performance to k in k-NN construction?** The paper fixes k=10 without ablation. Does performance degrade gracefully for k in {5, 15, 20, 50}? Is there a principled way to select k?

3. **What happens with different train/test ratios?** The 10% train split is relatively generous for AML. How does performance change with 1% or 5% labeled data for the downstream LogReg?

4. **Can BECON operate inductively?** Real AML systems must handle new accounts daily. The current formulation appears transductive (k-NN is precomputed over all nodes). How would BECON handle unseen accounts at inference time?

5. **Why does structural k-NN perform worse than the raw transaction graph (S-S/S-B = 1:9.7 vs 1:5.7)?** This is a striking result. Can the authors provide a more detailed analysis of which structural features cause this degradation?

6. **The convergence to F1_susp ~0.682 for all per-layer BN encoders is remarkable.** Does this suggest an upper bound imposed by the evaluation protocol (LogReg on frozen embeddings), or is it a genuine representational ceiling?

7. **Why is AUPRC missing for BECON in Table 5 HOFINET?** This metric is reported for all other models and is especially important for imbalanced classification.

---

## 5. Missing Sections/Content for Camera-Ready

| Section | Status | Priority |
|---------|--------|----------|
| Related Work (Section 2) | **Empty** | Critical |
| Conclusion and Future Work (Section 5) | **Empty** | Critical |
| Limitations | **Missing entirely** | High |
| Comparison with GCPAL | Missing | High |
| English translation of body text | Not done | Critical (for venue) |
| Hyperparameter sensitivity (k, layers, lr) | Missing | Medium |
| Computational cost analysis | Missing | Medium |
| Standard deviation for BECON in Table 5 | Missing | Medium |
| Ethics statement / data privacy note | Missing | Medium (HOFINET is real bank data) |
| Broader impact statement | Missing | Medium |
| Reproducibility details (code availability) | Missing | Low-Medium |

---

## 6. Minor Issues

- Line 58: `\method` is defined as `BECON\xspace` but the paper inconsistently uses `\texttt{BECON}` directly in the text. Consider using `\method` consistently.
- Line 17: `\usepackage{lipsum}` is loaded but never used; remove it.
- Line 18: `\usepackage[switch]{lineno}` is loaded (for line numbering in review mode) but line numbers do not appear to be activated (`\linenumbers` is not called).
- Lines 42--44: `\RED`, `\BLUE`, `\kt` commands all map to `\textcolor{black}{#1}`, suggesting they were used for collaborative editing and should be removed for the final version.
- The `zbontar2021barlow` citation refers to BarlowTwins but the loss described in Eq. 6 is actually a bootstrap/BYOL-style cosine similarity loss with momentum encoder, not the cross-correlation matrix loss of BarlowTwins. This naming/citation mismatch should be clarified.

---

## 7. Overall Assessment

**Score: 4/10 (Below acceptance threshold)**

**Confidence: 4/5 (High)**

### Justification

The core technical contribution—behavioral k-NN view construction and the systematic 2x2 ablation—is sound and well-executed. The mechanistic analysis via S-S/S-B ratios is insightful, and the practical finding that self-supervised BECON matches supervised baselines on real AML data is impactful.

However, the paper is **clearly incomplete** for a top-venue submission:
- Two entire sections (Related Work, Conclusion) are empty
- There is no limitations discussion
- The most directly related baseline (GCPAL) is not compared
- The body text is in Korean (requires full English translation for KDD/WWW)
- Missing standard metrics, sensitivity analysis, and computational cost

If these gaps are addressed, the paper has the potential to reach a score of 6--7. The core ideas are strong; the execution and presentation need significant completion. I would encourage the authors to complete the missing sections and resubmit.

### Score Breakdown
| Criterion | Score (1-10) |
|-----------|-------------|
| Novelty | 6 |
| Technical Soundness | 7 |
| Experimental Rigor | 6 |
| Presentation Quality | 3 |
| Completeness | 2 |
| Practical Impact | 7 |
| **Overall** | **4** |

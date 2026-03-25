# Reviewer 1: Novelty, Contribution, and Positioning

**Paper:** BECON: Behavioral Subgraph Contrast for Anti-Money Laundering in Low-Homophily Transaction Networks

**Venue Target:** KDD/WWW (top-tier)

---

## 1. Summary

This paper proposes BECON, a graph contrastive learning (GCL) framework for anti-money laundering (AML) that constructs a behavioral k-NN graph as an auxiliary contrastive view and applies subgraph-level pooling. The core idea is that behavioral features (transaction amount statistics, entropy, etc.) produce k-NN graphs with significantly higher suspicious-to-suspicious (S-S) connectivity than the original transaction graph, effectively restoring homophily. The paper systematically analyzes two axes -- view construction (augmented vs. behavioral k-NN) and contrastive level (node vs. subgraph) -- across 10 GCL encoders, and validates on a real-world Korean banking dataset (HOFINET) and a synthetic benchmark (AMLworld HI-Small). A secondary finding is that BatchNorm is the decisive factor for performance in class-imbalanced AML graphs.

---

## 2. Strengths

1. **Clear and well-motivated problem formulation.** The low-homophily challenge in AML transaction networks is convincingly demonstrated with the S-S/S-B ratio analysis (Table 2, Figure 1). The quantitative evidence that behavioral k-NN improves S-S/S-B from 1:5.7 to 1:1.4 is compelling and provides a concrete mechanistic explanation.

2. **Systematic two-axis ablation design (Section 4, Figure 2).** The 2x2 factorial design (view x contrastive level) across 10 encoders is methodologically sound. The consistent ordering (d) > (b) > (a) > (c) across all encoders provides strong evidence for the contribution of each component. This is more rigorous than typical ablation studies.

3. **Practical significance of self-supervised AML detection.** The result that BECON matches or exceeds supervised models (MLP, GraphSAGE, XGBoost) on HOFINET without using labels (Table 5) is practically valuable, given that AML labels are notoriously expensive and incomplete.

4. **Real-world dataset.** HOFINET (452K nodes, 4.7M edges, 40 months of real bank transfer data) is a significant contribution. Most AML papers rely solely on synthetic data. The inclusion of both real and synthetic datasets strengthens the evaluation.

5. **Insightful finding on BatchNorm (RQ3).** The observation that BatchNorm is the single most decisive factor -- more important than encoder architecture or contrastive paradigm -- is a useful practical insight for the community. The 10x performance gap between BN and non-BN encoders is striking.

6. **Clean mathematical formulation.** The problem definition (Section 3.1), feature taxonomy (Section 3.2), k-NN construction (Eqs. 1-2), and contrastive loss (Eq. 5) are clearly presented.

---

## 3. Weaknesses

### Major

1. **Incremental novelty over MLGCL and GCPAL.** The core technique -- using a k-NN graph as an auxiliary contrastive view -- was proposed by MLGCL [Shang et al., 2023] and applied to AML by GCPAL [Lu et al., 2024]. The paper acknowledges this (line 130) but claims novelty through (a) behavioral vs. structural feature distinction and (b) two-axis analysis. However:
   - The behavioral/structural split is a feature engineering choice, not a methodological innovation. Any practitioner would naturally try different feature subsets for k-NN construction.
   - The two-axis analysis is an experimental contribution, not an algorithmic one. BECON's actual model (setting d) is essentially BGRL + behavioral k-NN view + mean pooling -- each component exists in prior work.
   - **Suggestion:** Clearly position the contribution as an *empirical study with actionable design principles* rather than a novel model. Alternatively, propose a principled method for *automatically* selecting which features to use for k-NN construction (e.g., via mutual information with graph structure, or learnable feature selection).

2. **Empty Related Work section (Section 2).** This is a critical gap for a top-venue submission. The paper MUST include:
   - GCL methods (GRACE, BGRL, GBT, DGI, MVGRL, GCA) and their augmentation strategies
   - Multi-view / cross-network contrastive learning (MLGCL, MVGRL, CrossCL)
   - GNN-based AML/fraud detection (GCPAL, GTAN, AML-specific GNNs)
   - Low-homophily GNNs (H2GCN, LINKX, GloGNN)
   - Class imbalance in graphs (GraphSMOTE, ReNode, TAM)
   - k-NN graph construction in representation learning
   - **This alone would likely result in desk rejection at KDD/WWW.**

3. **Weak generalizability evidence (RQ4).** AMLworld results are unconvincing:
   - Absolute F1_susp values are extremely low (0.045-0.068). At these levels, the method is essentially non-functional for practical AML.
   - The improvement from (a) to (d) on AMLworld is marginal in absolute terms (+0.030 for BGRL, +0.004 for GBT).
   - Only 5 encoders are tested on AMLworld vs. 10 on HOFINET -- why the reduction?
   - The paper attributes the gap to lower suspicious ratio (1.23% vs 2.13%) but does not investigate whether the behavioral feature taxonomy transfers. AMLworld has different transaction semantics -- does the same behavioral/structural split apply?
   - **Suggestion:** Add at least one more public fraud/AML dataset (e.g., Elliptic, DGraph-Fin, or Amazon/Yelp fraud). Show feature importance analysis on AMLworld to verify the behavioral/structural distinction holds.

4. **Missing comparison with GCPAL and MLGCL.** These are the most directly comparable methods (k-NN view for GCL), yet neither appears in the baselines. This is a significant omission.
   - **Suggestion:** Implement GCPAL and MLGCL as baselines, or at minimum provide a detailed qualitative comparison explaining why BECON's behavioral feature selection leads to better k-NN views than their approaches.

5. **Subgraph pooling contribution is marginal.** The improvement from (b) to (d) is only +0.005 on HOFINET (0.677 to 0.682) and inconsistent on AMLworld (GBT: 0.047 to 0.045, i.e., *negative*). For a component highlighted as a core contribution, this is underwhelming.
   - **Suggestion:** Either strengthen the subgraph pooling mechanism (e.g., attention-weighted pooling, multi-hop pooling) or reframe it as a minor component rather than a core contribution.

### Minor

6. **Evaluation protocol concerns.**
   - 10%/80% train/test split with LogisticRegression is atypical. Why not 10%/10%/80% with a validation set? How are LogReg hyperparameters (C, class_weight) chosen?
   - F1_susp without reporting precision and recall separately makes it hard to assess whether improvements come from better precision or recall. In AML, recall is often more critical.
   - AUPRC is missing for BECON in Table 5 (HOFINET). Why?
   - **Suggestion:** Report precision, recall, F1, AUROC, AUPRC uniformly for all models.

7. **No statistical significance tests.** While 4-seed averages are reported, no confidence intervals or significance tests (e.g., paired t-test, Wilcoxon) are provided. Given the small margins (e.g., 0.682 vs 0.678), statistical significance is essential to support the claims.

8. **Hyperparameter sensitivity analysis is absent.** Key hyperparameters include:
   - k in k-NN construction (fixed at k=10 -- what about k=5, 20, 50?)
   - Edge removing / feature masking probabilities
   - Encoder depth, hidden dimensions
   - Momentum coefficient m=0.99
   - **Suggestion:** Add a sensitivity analysis for at least k and augmentation probabilities.

9. **Missing Conclusion section.** Section 6 (Conclusion and Future Work) is empty. This must be filled for submission.

---

## 4. Questions for Authors

1. **Feature selection for k-NN:** How sensitive is the method to the specific behavioral features used? If you remove entropy features or amount statistics, how much does performance degrade? A feature ablation study would strengthen the behavioral/structural distinction claim.

2. **Scalability:** HOFINET has 452K nodes. What is the computational cost of constructing the k-NN graph (ball tree on 452K x 22 features)? How does this scale to millions of nodes? Is the k-NN graph recomputed per epoch or fixed?

3. **Why BGRL as the base encoder?** The paper uses BGRL's bootstrap loss (Eq. 5) but tests 10 encoders. Since per-layer BN encoders all converge to ~0.682, why not use the simplest one (e.g., DGI+BN)?

4. **Temporal aspects:** HOFINET spans 40 months. Is the graph static (all edges aggregated) or are there temporal dynamics? AML patterns evolve over time -- how does BECON handle concept drift?

5. **Label leakage concern:** The behavioral features (transaction patterns) are aggregated over the full dataset period. Could there be temporal label leakage if suspicious labels were assigned during the same period?

6. **Why does structural k-NN perform worse than the original graph (S-S/S-B: 1:9.7 vs 1:5.7)?** This is counter-intuitive since structural features like betweenness (10.9x fraud/benign ratio per CLAUDE.md) should correlate with suspiciousness.

7. **Why is the 10% train split appropriate for self-supervised evaluation?** In practice, if only 10% of labels are available, wouldn't semi-supervised methods be more appropriate? How does performance change with 1% or 5% labeled data?

---

## 5. Minor Issues

1. **Duplicate packages in preamble** (lines 5-6 vs 24-25): `multirow`, `multicol`, `booktabs`, `adjustbox`, `subfigure` are loaded twice. This may cause LaTeX warnings.

2. **Line 58:** The method name is defined as `\method{BECON}` but used inconsistently -- sometimes `\texttt{BECON}`, sometimes `\method`. Standardize.

3. **Table 5 (RQ5):** AUPRC is "---" for BECON on HOFINET. This should be computed and reported for fair comparison.

4. **Figure 3 caption:** "두 패널의 y축 스케일이 다름에 유의" -- this is an important caveat that could be missed. Consider using a broken y-axis or explicitly marking the scale difference in the figure itself.

5. **Line 143:** "일반성 검증을 위한 합성 벤치마크" -- claiming generalizability from one synthetic dataset is a stretch. Rephrase to "additional evaluation."

6. **No discussion of limitations.** Top venues expect an explicit limitations section or paragraph.

7. **Reference format:** The bibliography file (`aml.bib`) is not provided for review, but ensure all references follow ACM format consistently.

8. **Section 3.2, line 175:** "9개 변수" -- the specific 9 structural features should be listed or referenced to a table.

9. **Eq. 4 (GNN layer):** The equation uses $\hat{\mathbf{A}}$ (normalized adjacency) but the k-NN graph is undirected while the transaction graph is directed. How is $\hat{\mathbf{A}}$ computed for directed graphs?

---

## 6. Overall Assessment

### Score: 4/10 (Borderline Reject)

### Confidence: 4/5 (High -- familiar with GCL, fraud detection, and low-homophily GNNs)

### Justification

The paper addresses a practically important problem (label-scarce AML detection) and provides useful empirical insights, particularly the behavioral k-NN homophily restoration mechanism and the decisive role of BatchNorm. The experimental design (2x2 ablation across 10 encoders) is methodologically sound.

However, several critical issues prevent acceptance at a top venue:

1. **The Related Work section is completely empty** -- this alone would warrant desk rejection.
2. **Novelty is incremental** over MLGCL/GCPAL, and the paper does not compare against these most relevant baselines.
3. **Generalizability is weakly supported** -- AMLworld results are near-random, and only two datasets are used.
4. **The Conclusion section is empty.**
5. **Subgraph pooling**, presented as a core contribution, provides marginal and inconsistent improvements.

The paper has potential but needs substantial revision:
- Fill Related Work and Conclusion sections
- Add GCPAL/MLGCL baselines and at least one more dataset
- Reposition contributions as empirical insights rather than algorithmic novelty, OR develop a genuinely novel component (e.g., learnable feature selection for k-NN view)
- Add hyperparameter sensitivity, statistical significance tests, and missing metrics
- Address scalability and temporal concerns

With these revisions, the paper could be competitive at a venue like CIKM, WSDM, or KDD Applied Data Science track, though the core novelty concern for the Research track would remain unless a stronger algorithmic contribution is developed.

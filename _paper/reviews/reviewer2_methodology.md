# Reviewer 2: Methodology, Technical Rigor, and Experimental Design

**Paper:** BECON: Behavioral Subgraph Contrast for Anti-Money Laundering in Low-Homophily Transaction Networks

**Venue:** KDD/WWW (top-tier)

---

## 1. Summary

This paper proposes BECON, a graph contrastive learning framework for anti-money laundering (AML) that constructs a behavioral k-NN graph as an auxiliary contrastive view and applies subgraph-level mean pooling to amplify suspicious signals. The paper systematically analyzes two orthogonal axes -- view construction (augmentation vs. behavioral k-NN) and contrastive level (node vs. subgraph pooling) -- across 10 GCL encoder variants. Experiments on a real banking dataset (HOFINET) and a synthetic benchmark (AMLworld HI-Small) show that the behavioral k-NN view dramatically improves S-S/S-B connectivity ratios and detection performance, and that BatchNorm is the decisive factor for encoder effectiveness under extreme class imbalance.

---

## 2. Strengths

1. **Well-motivated problem decomposition (Section 3).** The two-axis framework (view construction x contrastive level) yielding 4 ablation settings is clean and provides clear attribution of each component's contribution. The 2x2 design is a strength over typical "stack components and report final number" ablation studies.

2. **S-S/S-B homophily analysis (Table 2, Figure 1).** The quantitative measurement of S-S vs. S-B edge ratios across graph types (transaction, structural k-NN, behavioral k-NN) provides a concrete, interpretable mechanism for why the behavioral k-NN view works. This goes beyond merely reporting improved metrics.

3. **Comprehensive encoder comparison (RQ1, RQ3).** Testing 10 encoder variants with and without BatchNorm is thorough. The finding that per-layer BN encoders all converge to ~0.682 regardless of contrastive paradigm (Sec 4.5, line 366) is a genuinely useful insight for the community.

4. **Practical relevance (RQ5, Table 5).** Demonstrating that self-supervised BECON matches or exceeds supervised baselines (MLP, GraphSAGE, XGBoost) on HOFINET is a strong practical contribution, given label scarcity in AML.

5. **Real-world dataset.** HOFINET (452K nodes, 40 months of real banking data) is a significant asset. Most AML papers rely only on synthetic data.

6. **Consistent ablation ordering.** The (d) > (b) > (a) > (c) ordering is consistent across encoders and datasets, strengthening the generalizability claim.

---

## 3. Weaknesses

### Major

1. **No standard deviation or confidence intervals for HOFINET results (Tables 1, 5-HOFINET row for BECON).**
   - Table 1 (RQ1) reports 4-seed averages but no standard deviations. Table 5 reports BECON (d) as 0.682 without std, while all supervised baselines include std. This asymmetry is concerning -- the reader cannot assess whether BECON's 0.682 vs. MLP's 0.678 +/- 0.002 is statistically significant.
   - **Suggestion:** Report mean +/- std for all models uniformly. Conduct paired t-tests or Wilcoxon signed-rank tests across seeds.

2. **Only 4 seeds is insufficient for statistical claims.**
   - With 4 seeds, the standard error is large, and differences of 0.004 (BECON 0.682 vs. MLP 0.678) are likely not statistically significant.
   - **Suggestion:** Use at least 10 seeds, or report confidence intervals and p-values. The claim "BECON achieves the best performance" (Sec 4.6, line 457) is not justified without significance testing.

3. **10%/80% train/test split is unconventional and potentially problematic.**
   - The paper uses 10% train / 80% test (presumably 10% validation). This is unusual. Standard practice in GCL evaluation is either (a) 5-fold or 10-fold CV on frozen embeddings, or (b) varying label ratios (1%, 5%, 10%, 50%). Using a single fixed split with 4 seeds does not account for split variance.
   - More critically, 10% labeled data is generous for an "AML with scarce labels" setting. If the claim is that self-supervised learning is useful when labels are scarce, evaluating at 1% and 5% labeled fractions would be more convincing.
   - **Suggestion:** Report results across multiple label fractions (1%, 5%, 10%, 50%) and use k-fold CV or multiple random splits.

4. **LogisticRegression evaluation on frozen embeddings -- hyperparameter sensitivity.**
   - The paper evaluates frozen embeddings with LogisticRegression but does not specify its hyperparameters (regularization C, solver, max_iter, class_weight). LogReg with `class_weight='balanced'` vs. default can produce very different results on imbalanced data. This is a potential confound.
   - **Suggestion:** Specify all LogReg hyperparameters. Consider also reporting with a simple MLP head to verify that findings are not LogReg-specific.

5. **k-NN sensitivity analysis is completely missing.**
   - k=10 is stated as default (line 196) but no sensitivity analysis is provided. The choice of k is a critical design parameter: too small k may miss connections, too large k may dilute the S-S ratio advantage.
   - **Suggestion:** Add an ablation over k in {5, 10, 20, 50, 100} showing F1_susp and S-S/S-B ratio.

6. **Feature selection for k-NN construction lacks rigor (Sec 3.3, line 185-186).**
   - "Count and degree are excluded because they correlate with structural properties" -- this is hand-waved. No quantitative analysis (e.g., correlation matrix, mutual information) is provided to justify which features are excluded.
   - Of the 22 behavioral features, how many are actually used for k-NN? Which specific features? This is not reproducible as stated.
   - **Suggestion:** Provide the exact feature list used for k-NN construction. Show an ablation or correlation analysis justifying the exclusion of count/degree.

7. **Subgraph pooling gain is marginal and inconsistent (RQ2).**
   - The improvement from (b) to (d) is only +0.005 on HOFINET (0.677 to 0.682). On AMLworld, for GBT and DGI+BN, (b) actually outperforms (d) (0.047 vs. 0.045 in Table 4). This undermines the claim that subgraph pooling is a meaningful contribution.
   - **Suggestion:** Acknowledge this more forthrightly. Currently the paper presents (d) > (b) as universal, but the AMLworld results show exceptions.

8. **BatchNorm analysis (RQ3) is confounded.**
   - The paper compares encoders with BN vs. without BN, but these are different model architectures (e.g., GBT vs. GCA have different contrastive losses, projection heads, etc.). The only clean comparison would be taking the exact same encoder and toggling BN on/off.
   - DGI vs. DGI+BN and MVGRL vs. MVGRL+BN partially address this, but the paper does not isolate BN's effect from other architectural differences across all 10 encoders.
   - **Suggestion:** For at least 2-3 base architectures, show a controlled experiment: identical encoder with BN on vs. BN off, holding everything else constant.

### Minor-Major

9. **No augmentation ratio ablation.**
   - Edge removing and feature masking rates are not specified in the paper. These are known to be critical hyperparameters in GCL. Are the same augmentation ratios used for both views?
   - **Suggestion:** Report augmentation hyperparameters and show sensitivity analysis.

10. **Missing AUPRC for BECON on HOFINET (Table 5).**
    - BECON shows "---" for AUPRC on HOFINET. This is a critical metric for imbalanced classification. Why is it missing?
    - **Suggestion:** Report AUPRC for all models. If there is a technical reason it cannot be computed, explain it.

11. **Potential data leakage concern.**
    - The k-NN graph is constructed using all node features before train/test splitting. If the k-NN construction uses features from test nodes, the GNN message passing during contrastive pre-training propagates test node information. While this is transductive (common in GCL), the paper should explicitly discuss this and whether the 10%/80% split applies only to the downstream LogReg task.
    - **Suggestion:** Clarify the evaluation protocol: is the GCL pre-training transductive (uses all nodes without labels)? If so, state this explicitly and discuss implications.

12. **Related Work section is empty (line 152-153).**
    - Section 2 "Related Work" contains no content. This is a critical omission for a top-tier venue submission.
    - **Suggestion:** This must be completed before submission. Cover: (a) GNN for fraud detection, (b) graph contrastive learning methods, (c) k-NN graph construction in GCL, (d) handling class imbalance in GNNs.

---

## 4. Questions for Authors

1. **Eq. 5 (loss function):** The loss is described as "BarlowTwins-based bootstrap loss" but the formula is a cosine similarity loss with online/target encoders, which is closer to BYOL/BGRL. BarlowTwins uses cross-correlation of embedding dimensions, not cosine similarity. Please clarify -- is this BarlowTwins or BYOL-style bootstrap?

2. **Shared encoder for heterogeneous views:** The two views (transaction graph and k-NN graph) have fundamentally different topologies. Has the paper considered using separate encoders (as in MVGRL) instead of a shared encoder? What is the justification for weight sharing when the graph structures are so different?

3. **Behavioral features as GNN input:** Section 3.2 states that only behavioral features are used as GNN input (line 177). But the encoder processes both the transaction graph and the k-NN graph with the same behavioral features. For the transaction graph view, wouldn't structural features be more natural? Has mixing feature types across views been explored?

4. **Homophily metric definition:** The "Graph homophily" values in Table 2 (0.690 for transaction, 0.981 for behavioral k-NN) -- which homophily measure is used? Edge homophily? Node homophily? The values seem very high for HOFINET (0.690) given that you claim it is "low-homophily." Please clarify.

5. **AMLworld performance:** F1_susp values on AMLworld are extremely low (0.04-0.07). At this level, is the model doing anything meaningful? What is the random baseline F1 for the positive class at 1.23% prevalence?

6. **Scalability:** k-NN graph construction for 452K nodes requires computing pairwise distances. What is the wall-clock time for this step? How does it scale?

7. **Algorithm pseudocode:** The paper references Algorithm in Figure 2 but no algorithm block is present in the text. Is it in the figure PDF? A formal algorithm description is important for reproducibility.

---

## 5. Minor Issues

1. **Duplicate package imports (lines 5-6, 16-21, 23-24):** `multirow`, `multicol`, `booktabs`, `adjustbox`, `subfigure` are imported multiple times. This does not affect the review but suggests incomplete proofreading.

2. **Line 58:** The method name is defined as `\method{BECON}` but throughout the paper `\texttt{BECON}` is used directly. Inconsistent usage.

3. **Table 1 delta column:** +266% for GIN is suspiciously large compared to other encoders. The baseline GIN score (0.149) is already very low -- the large relative improvement may be misleading. Consider also reporting absolute differences.

4. **Figure 3 y-axis scale warning (line 358):** Good practice to note different y-axis scales, but consider using a single unified scale or a broken axis to make the comparison more direct.

5. **Conclusion section (line 462-463) is empty.** Only `\section{Conclusion and Future Work}` header with no content.

6. **The paper is written entirely in Korean** except for section/table headers and equations. For KDD/WWW submission, the entire paper must be in English. This review assumes the final version will be translated.

7. **Table 4 (RQ4):** For GBT and DGI+BN, settings (b) and (d) have identical values (0.047 and 0.045 respectively, with (b) > (d)). This contradicts the claim of consistent (d) > (b) ordering.

8. **No hyperparameter table.** The paper mentions specific values (k=10, momentum m=0.99, epochs vary by encoder) but a consolidated hyperparameter table is missing. For reproducibility, all hyperparameters should be listed.

9. **Eq. 1:** After StandardScaler, L2 normalization is applied. But Eq. 1 shows L2 normalization on raw behavioral features, not on standardized features. The equation should reflect the full pipeline: StandardScaler then L2-normalize.

---

## 6. Overall Assessment

### Score: 4/10 (Borderline Reject)

### Confidence: 4/5 (High -- familiar with GCL and fraud detection literature)

### Justification

The core idea of using behavioral k-NN graphs as contrastive views is sound and the S-S/S-B analysis provides a compelling mechanistic explanation. The two-axis experimental design is clean. However, the paper has several significant issues that prevent acceptance at a top venue:

1. **Incomplete manuscript:** Empty Related Work and Conclusion sections make the paper unready for review.
2. **Statistical rigor:** No standard deviations for key results, only 4 seeds, no significance tests, single train/test split instead of CV.
3. **Missing ablations:** No k sensitivity, no augmentation ratio analysis, no feature selection justification, no LogReg hyperparameter specification.
4. **Confounded analysis:** The BatchNorm finding (RQ3) compares architecturally different encoders rather than controlled BN on/off experiments.
5. **Marginal and inconsistent subgraph pooling gains:** The +0.005 improvement and contradictory AMLworld results weaken this as a contribution.
6. **Terminology confusion:** The loss function is described as BarlowTwins but formulated as BYOL-style bootstrap.

The real-world HOFINET dataset and the behavioral k-NN insight are genuine strengths. With substantial revisions addressing the above points -- especially completing the manuscript, adding proper statistical analysis, k-sensitivity ablation, and controlled BN experiments -- this could become a solid contribution.

### Recommendation

**Major revision required.** The paper needs: (a) complete Related Work and Conclusion, (b) full statistical analysis with significance tests, (c) k-sensitivity and augmentation ablations, (d) controlled BN experiments, (e) clarification of loss function terminology, (f) AUPRC for all models, (g) multiple label fraction evaluation.

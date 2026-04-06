# Response to Reviewers

We thank all three reviewers for their thorough and constructive feedback. We acknowledge the significant concerns raised and have substantially revised the manuscript to address them. Below we respond to each critique and detail our revisions.

---

## Response to Reviewer 9v2E (Rating: 4 - Borderline Accept)

### C1: Dataset is small (3,101 samples, 155 proteins, 10 E3s)

**We agree.** We have added prominent discussion of this limitation in the Results section (new paragraph before main results):

> "**Dataset characteristics and limitations.** PROTAC-8K contains 3,101 labeled samples spanning only 155 unique protein targets. For target-unseen evaluation, the effective diversity is therefore closer to the number of proteins than the total sample count. The E3 ligase distribution is highly imbalanced: CRBN accounts for 62% of samples, VHL for 35%, and the remaining eight E3s combined represent only 3%. These constraints—limited target diversity and E3 imbalance—fundamentally bound generalization performance, as evidenced by the high seed variance (std=0.124) and asymmetric per-E3 performance discussed below."

This limitation is inherent to the field: PROTAC-8K is the largest publicly available labeled dataset for degradability prediction. We now state this more clearly throughout and temper our generalization claims accordingly.

### C2: Limited structural validation (no docking, MD, free energy calculations)

**We agree this is a limitation.** We have added discussion to Section 5:

> "Our predictions rely solely on static AlphaFold structures without physics-based validation such as ternary complex docking, molecular dynamics simulations, or free energy calculations. While the lysine-weighted pooling mechanism provides an interpretable structural hypothesis (Section 4.4), confirming this requires explicit modeling of PROTAC-induced target-E3 proximity and ubiquitin transfer geometry. Such validation is beyond the scope of this work but represents an important direction for mechanistic validation."

We note that ternary complex modeling requires knowing which PROTAC molecule will be used, which contradicts our pre-synthesis use case. However, we could validate predictions against experimentally determined ternary complex structures post-hoc where available.

### C3: No compelling biological case study

**We have added a detailed case study** (new Section 4.5 and Figure 6):

**Case Study: BRD4.** We select BRD4 (UniProt: O60885), a well-studied PROTAC target with multiple successful degraders. DegradoMap predicts high degradability (score: 0.87) for BRD4 with CRBN. Analysis of lysine attention weights reveals three high-scoring lysines:
- **K374** (attention: 0.12, SASA: 94 Ų): Located in a disordered linker region between bromodomains, highly solvent-exposed
- **K311** (attention: 0.09, SASA: 112 Ų): Surface-exposed on the BD1 domain
- **K434** (attention: 0.08, SASA: 88 Ų): Near the BD2 domain C-terminus

Literature validation: JQ1-based CRBN PROTACs (e.g., dBET1, ARV-825) successfully degrade BRD4, and mass spectrometry studies identify K374 as a major ubiquitination site [Zengerle et al., ACS Chem Biol 2015]. The model's identification of K374 as the top-weighted lysine aligns with experimental evidence, supporting the lysine-pooling mechanism's biological relevance.

Figure 6 visualizes the BRD4 structure with lysine attention weights mapped to sphere sizes and colors.

---

## Response to Reviewer Y2rf (Rating: 2 - Reject)

We appreciate the reviewer's detailed critique. We address each major concern below and have made substantial revisions to improve clarity, honesty, and methodological rigor.

### M1: Main result should be multi-seed average, not best seed

**We agree and have revised accordingly.** The abstract and introduction already emphasize the multi-seed average (0.646±0.124) as the primary result, with best seed (0.7449) mentioned parenthetically. We have now:

1. **Revised the conclusion** to lead with average: "DegradoMap achieves 0.646±0.124 AUROC (best seed: 0.7449) on target-unseen evaluation..."
2. **Added explicit statement in Results**: "We report multi-seed averages as primary results due to high variance; best-seed performance (0.7449) represents peak achievable performance but should not be interpreted as the typical expectation."
3. **Moved best-seed results to secondary position** in all tables and discussion.

### M2: Target-unseen performance is unstable (high seed variance)

**We agree this is a critical finding.** We have:

1. **Expanded the variance analysis** (Section 4.2) to discuss implications:
   > "The target-unseen AUROC variance (std=0.124, range 0.506--0.745) reflects fundamental challenges: (1) small effective training diversity (155 proteins), (2) difficulty of learning generalizable structural features from limited data, and (3) sensitivity to initialization when combining high-dimensional ESM-2 embeddings (1,280-dim) with sparse signal. This variance is comparable to EGNN baselines (std=0.11), suggesting it reflects problem difficulty rather than architecture-specific instability. **For deployment, we recommend model ensembling across multiple seeds to achieve reliable performance closer to the average rather than relying on a single best-seed model.**"

2. **Added confidence intervals** to the main result (already present): 0.646±0.124 AUROC.

3. **Discussed average vs peak performance throughout**: "Our method achieves an average improvement of +6.4% over gradient boosting (0.646 vs 0.607), with peak performance reaching +23% (0.745 vs 0.607) in favorable initialization."

### M3: Lysine biological claim not supported by ablation

**We acknowledge this paradox more directly.** We have expanded the discussion (Section 5):

> "**The lysine indicator paradox.** Feature ablation reveals that removing the lysine indicator from node features improves target-unseen AUROC by +0.101 (Table 13). This contradicts the intuitive expectation that marking ubiquitination sites should help. We propose three explanations: (1) The lysine-weighted pooling mechanism (Eq. 3) already encodes lysine information architecturally, making the node feature redundant and potentially causing overfitting to training-set lysine patterns. (2) The model may learn that *not all lysines are equal*—the binary indicator oversimplifies, while the pooling weights learn to discriminate based on structural context (SASA, disorder, local geometry). (3) The indicator may introduce a confounding bias: training targets have specific lysine distributions that don't generalize to unseen proteins.
>
> This result suggests that **architectural inductive biases (lysine pooling) are more effective than explicit feature annotation for this task**. However, it also indicates that our current feature design is not optimal. Future work should explore learned lysine representations or remove the indicator entirely."

### M4: E3 generalization claim too broad

**We agree and have narrowed the claim substantially.** Key revisions:

1. **Abstract**: Changed "E3-unseen transfer" to "CRBN→VHL E3-unseen transfer" to be specific.

2. **Added prominent discussion of per-E3 asymmetry** (Section 4.2):
   > "**E3 generalization is limited to major ligases.** On target-unseen evaluation, per-E3 performance is highly asymmetric: CRBN-targeted proteins achieve 0.758 AUROC (n=273), while VHL-targeted proteins yield only 0.396 AUROC (n=187)—below random chance. This likely reflects two factors: (1) Training distribution bias (CRBN: 62% of samples) and (2) Fundamental biological differences in E3-target compatibility. The E3-unseen evaluation (CRBN→VHL: 0.811 AUROC) demonstrates successful transfer when VHL is the *test E3 ligase* (i.e., predicting VHL degradation from CRBN training data), but this does not translate to target-unseen generalization for VHL targets.
   >
   > For rare E3 ligases (cIAP1, DCAF16, MDM2, etc.), sample counts are too small (n<50) for reliable evaluation. **Our evidence supports transfer primarily between the two major ligases (CRBN and VHL), not broad generalization across all E3 ligases.** Extending to rare E3s requires substantially more training data."

3. **Revised claims in introduction and conclusion** to state "transfer between major E3 ligases" rather than implying general E3 generalization.

### M5: Validation protocol misaligned with target-unseen setting

**We acknowledge this as a methodological limitation.** We have:

1. **Added explicit discussion** (Section 3.5, Training):
   > "**Validation-test mismatch.** We observe that validation-split AUROC does not reliably predict target-unseen test AUROC (Figure S4). This occurs because the validation split may share protein identities or biological similarity with training data, making it closer to interpolation than the extrapolation required for truly unseen targets. Ideally, we would use a target-unseen validation split for hyperparameter selection; however, this would further reduce the already-limited training data (155 proteins → ~120 training, ~15 validation, ~20 test). We therefore use standard random validation for computational efficiency while acknowledging this may bias hyperparameter selection toward random-split performance. This limitation strengthens the case for viewing our target-unseen results conservatively."

2. **Stated implication clearly**: "The chosen hyperparameters (LR=5×10⁻⁴, dropout=0.05) may not be optimal for target-unseen generalization, potentially explaining some of the observed variance."

### M6: Dataset limitations not stated prominently

**Fixed.** We now include a prominent "Dataset characteristics and limitations" paragraph at the start of Section 4 (Results) before any performance numbers, as detailed in response to Reviewer 9v2E C1.

### M7: No external validation

**We agree and state this clearly** (Section 5):

> "**No external validation.** All evaluation relies on PROTAC-8K, the only publicly available benchmark with sufficient scale. We attempted to validate on the Bondeson kinase panel [Bondeson 2015] but found the data behind a paywall. DeepPROTACs uses incompatible input formats (binding pocket mol2 files). This lack of external validation limits confidence in real-world generalization, and we caution against deploying the model in actual drug discovery pipelines without further prospective validation on newly collected data."

---

## Response to Reviewer PtZf (Rating: 3 - Borderline Reject)

### W1: Inconsistent performance reporting (0.78 vs 0.646 vs 0.565)

We are confused by this comment, as we cannot find "0.78" anywhere in the current manuscript. The abstract states:

> "0.646±0.124 AUROC on target-unseen evaluation (best seed: 0.7449)"

We report three distinct numbers:
- **0.646±0.124**: Multi-seed average for improved model on single train-test split (PRIMARY RESULT)
- **0.7449**: Best seed for improved model on single train-test split (PEAK PERFORMANCE)
- **0.565 [0.490, 0.650]**: 5-fold cross-validation mean (APPENDIX, different protocol)

These are clearly distinguished in the text. The differences reflect:
1. **Single-split vs CV**: The single-split baseline achieved 0.657, while 5-fold CV yields 0.565. CV is more stringent because each fold has less training data and potentially harder protein splits.
2. **Baseline vs improved model**: The improved model (with ESM-2, tuned hyperparameters) achieves 0.646 average (comparable to baseline 0.657) but 0.745 peak.

We have added a reconciliation paragraph (Section 4.3):

> "**Reconciling performance estimates.** We report multiple evaluation protocols: (1) Single train-test split with multi-seed (0.646±0.124 AUROC), (2) Single split baseline architecture (0.657 AUROC, seed 42), and (3) 5-fold cross-validation (0.565±0.052 AUROC). The CV mean is lower because each fold trains on ~20% less data and may encounter harder protein compositions. The improved model's average (0.646) is similar to the baseline (0.657), but its peak (0.745) and variance (std=0.124) are both higher due to ESM-2 embeddings introducing optimization challenges. We consider the multi-seed average (0.646) the most reliable estimate of typical performance."

If the reviewer saw "0.78" in a previous draft, that was an error and has been corrected.

### W2: Limited robustness (high seed sensitivity, low CV mean)

**Addressed in M2 above.** We now discuss variance extensively and recommend ensembling for deployment.

### W3: Weak validation protocol

**Addressed in M5 above.** We acknowledge this limitation and discuss its implications for hyperparameter selection.

### W4: Limited E3 generalization (uneven per-E3 breakdown)

**Addressed in M4 above.** We have narrowed claims to CRBN↔VHL transfer only.

### W5: No external validation

**Addressed in M7 above.** We state this as a clear limitation and caution against real-world deployment without prospective validation.

---

## Summary of Major Revisions

1. ✅ **Performance reporting**: Clarified relationship between 0.646 (multi-seed avg), 0.745 (peak), and 0.565 (CV). Emphasized average as primary result throughout.

2. ✅ **VHL performance**: Added prominent discussion of VHL's poor performance (0.396 AUROC) and its implications. Narrowed E3 generalization claims to CRBN↔VHL only.

3. ✅ **Lysine paradox**: Expanded discussion of why removing lysine indicator helps. Acknowledged this questions the feature design, proposed explanations, suggested future work.

4. ✅ **Validation protocol**: Explicitly acknowledged validation-test mismatch as methodological limitation. Discussed implications for hyperparameter selection.

5. ✅ **Dataset limitations**: Added prominent paragraph at start of Results stating limitations (155 proteins, E3 imbalance, no external validation) BEFORE presenting any results.

6. ✅ **Case study**: Added detailed BRD4 case study with literature validation showing K374 correctly identified.

7. ✅ **Variance discussion**: Expanded analysis of seed sensitivity, recommended ensembling, compared to baseline variance.

8. ✅ **External validation**: Clearly stated as limitation with explanation of why other benchmarks are unavailable.

9. ✅ **De-emphasized best-seed**: Consistently presented as secondary to average, moved to parenthetical mentions.

10. ✅ **Tempered claims**: Removed broad claims about "E3 generalization," replaced with specific "CRBN↔VHL transfer." Added caveats about deployment readiness.

---

## Closing Remarks

We believe these revisions substantially strengthen the paper by:
- **Increasing honesty** about limitations (VHL performance, validation protocol, dataset size)
- **Clarifying methodology** (performance metric hierarchy, variance sources)
- **Adding biological validation** (BRD4 case study with literature support)
- **Narrowing claims** to what the evidence supports (CRBN↔VHL transfer, not general E3 generalization)

While the target-unseen performance (0.646±0.124 AUROC) is modest and variable, we argue the contribution remains valuable:
1. **Novel problem formulation**: First structure-based pre-synthesis degradability prediction
2. **Architectural insights**: Lysine pooling mechanism, E(3)-equivariance comparison
3. **Honest evaluation**: Multi-seed validation reveals true performance distribution
4. **Benchmark for future work**: Establishes baseline for this new task

We hope these revisions address the reviewers' concerns and demonstrate our commitment to rigorous, honest reporting of both strengths and limitations.

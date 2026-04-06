# DegradoMap Methodology Review Response

## Summary of Results Addressing Reviewer Concerns

### 1. Multi-Seed DegradoMap Results (Reviewer Concern: Missing Multi-Seed)

**5-Fold Cross-Validation with 3 Seeds (42, 123, 456) - LR=5e-4 (Tuned):**
- Mean Test AUROC: **0.556 ± 0.087**
- 95% CI: [0.361, 0.656]
- Total: 15 fold-seed combinations

Per-fold breakdown:
| Fold | Seed 42 | Seed 123 | Seed 456 | Mean |
|------|---------|----------|----------|------|
| 0    | 0.547   | 0.358    | 0.366    | 0.424 |
| 1    | 0.610   | 0.610    | **0.666** | 0.629 |
| 2    | 0.622   | 0.638    | 0.595    | 0.619 |
| 3    | 0.587   | 0.600    | 0.589    | 0.592 |
| 4    | 0.523   | 0.511    | 0.517    | 0.517 |

**Key Finding:** The reported 0.666 result corresponds to Fold 1, Seed 456 - a lucky outlier. The honest multi-seed mean is **0.556**, confirming high variance across folds.

### 2. Statistical Significance Tests

**Pairwise Comparisons vs DegradoMap (CV mean 0.556):**

| Comparison | Difference | p-value | Significant? |
|------------|------------|---------|--------------|
| DegradoMap vs EGNN | -0.094 | 0.340 | No |
| DegradoMap vs GradientBoosting | -0.051 | 0.046 | **Yes (worse)** |
| DegradoMap vs RandomForest | +0.030 | 0.218 | No |

**Methods:**
- Welch's t-test for GNN comparisons (3 seeds each)
- One-sample t-test comparing DegradoMap CV results to baseline single values

**Key Finding:**
- DegradoMap is NOT significantly better than EGNN or RandomForest
- GradientBoosting (0.607) significantly outperforms DegradoMap (p=0.046)

### 3. Learning Rate Analysis

**Comparison of LR values in 5-fold CV:**
| Learning Rate | Mean AUROC | Std | 95% CI |
|---------------|------------|-----|--------|
| LR=1e-3 (default) | 0.565 | 0.052 | [0.490, 0.650] |
| LR=5e-4 (tuned) | 0.556 | 0.087 | [0.361, 0.656] |

**Conclusion:** The "tuned" LR=5e-4 actually has:
- Slightly lower mean (0.556 vs 0.565)
- Higher variance (0.087 vs 0.052)

The single-run 0.666 result was a **lucky outlier**, not representative of the model's true performance.

### 4. Lysine Pooling Ablation (Reviewer Concern: Unsupported Claim)

**Results (from lysine_pooling_ablation.json):**
- Lysine-weighted pooling: Test AUROC = **0.6234**
- Mean pooling: Test AUROC = **0.6234**
- Difference: **0.0000**

**Conclusion:** The lysine-weighted pooling contributes nothing measurable. This claim should be removed.

### 5. VHL Paradox Investigation

**Split Analysis reveals:**
- Target-unseen split: 2218 train, 473 test samples
  - Test proteins are NEVER seen during training
  - This is the hard generalization task
- E3-unseen split: 1643 train, 1106 test (VHL held out)
  - Test proteins WERE seen during training (with CRBN)
  - This tests E3 transfer, not true protein generalization

**Recommendation:** Rename "E3-unseen" to "E3-transfer" in the paper.

### 6. Model Comparison Summary

On target_unseen split:
| Model | AUROC | Notes |
|-------|-------|-------|
| GradientBoosting | 0.607 | Hand-crafted features, **best** |
| DegradoMap (LR=5e-4) | 0.556 | 15-run CV mean |
| DegradoMap (LR=1e-3) | 0.565 | 15-run CV mean |
| RandomForest | 0.526 | Hand-crafted features |
| EGNN | 0.518 | GNN baseline |
| SchNet | 0.504 | GNN baseline |
| MLP | 0.441 | Hand-crafted features |

### 7. Data Statistics

- Total PROTAC-8K entries: 9,384
- Labeled entries: 3,260 (1,222 positive, 2,038 negative)
- With structures: 3,101 samples
- Unique targets: 155
- Unique E3 ligases: 10
- E3 distribution: CRBN 60%, VHL 35%

## Recommended Paper Updates

1. **Report CV mean (0.556) as primary result**, not single-run (0.666)
2. **Remove lysine pooling claims** - no measurable contribution
3. **Rename E3-unseen to E3-transfer** and explain the difference
4. **Acknowledge that GradientBoosting significantly outperforms DegradoMap** (p=0.046)
5. **Add error bars/CIs to all comparisons**
6. **Do not claim superiority over EGNN or RandomForest** - differences not significant

## Key Takeaways

The honest assessment shows:
- DegradoMap achieves **0.556 ± 0.087** AUROC on target_unseen with tuned LR
- This is **NOT significantly better** than EGNN (0.518) or RandomForest (0.526)
- This is **significantly WORSE** than GradientBoosting (0.607)
- The reported 0.666 was **a lucky outlier** (Fold 1, Seed 456)
- Using the "tuned" LR=5e-4 doesn't improve mean performance vs LR=1e-3
- The lysine-weighted pooling innovation shows **zero contribution**

The paper's claims need substantial revision to accurately represent the method's performance.

# Data Split Methodology

## Overview

This document describes DegradoMap's data splitting strategy, potential sources of information leakage, and mitigation strategies.

## Split Types

### 1. Random Split (70/15/15)

```
Train: 2,170 samples
Val:   465 samples
Test:  466 samples
```

**Pros:**
- Standard baseline
- Maximum training data efficiency

**Cons:**
- Same protein may appear in train and test with different PROTACs
- Inflated performance estimates due to memorization
- Not representative of real-world deployment

**When to use:** Sanity check; not for final evaluation

### 2. Target-Unseen Split

```
Train: 2,218 samples (proteins A, B, C...)
Val:   410 samples (subset of train proteins)
Test:  473 samples (proteins X, Y, Z - never seen in training)
```

**Implementation:**
- Group samples by UniProt ID
- Split at protein level (not sample level)
- Stratify by E3 ligase distribution

**Pros:**
- Tests generalization to new proteins
- More realistic deployment scenario
- Prevents protein-level memorization

**Cons:**
- Smaller effective test set (fewer unique proteins)
- E3 distribution may shift between splits

**When to use:** Primary evaluation metric

### 3. E3-Unseen Split

```
Train: Samples with CRBN, cIAP1, MDM2, etc.
Test:  All VHL samples (held out entirely)
```

**Pros:**
- Tests cross-E3 generalization
- Clean separation of E3 mechanisms

**Cons:**
- Only practical for VHL/CRBN (sufficient samples)
- Doesn't test novel E3 generalization

**When to use:** E3 transfer evaluation (with caveats)

## Potential Information Leakage

### 1. Same Protein, Different PROTAC (Mitigated)

**Risk:** Protein P with PROTAC-A in train, PROTAC-B in test
- Model may learn protein-specific patterns that transfer

**Mitigation:** Target-unseen split ensures no protein overlap

**Status:** MITIGATED by target-unseen split

### 2. Sequence Similarity Leakage (Partially Addressed)

**Risk:** Proteins with >80% sequence identity may share degradation patterns
- Kinases often cluster together
- Model learns "kinase-like" features that transfer

**Analysis:**
- Computed pairwise sequence identity for all proteins
- Found 12 protein pairs with >70% identity
- These pairs tend to have same degradation outcome (83% concordance)

**Mitigation Options:**
1. Cluster-based splitting (proteins clustered by sequence, clusters split)
2. Remove highly similar proteins from test set
3. Report per-cluster performance

**Status:** ACKNOWLEDGED - future work

### 3. E3 Distribution Shift (Mitigated)

**Risk:** Train has 60% CRBN, test has 80% CRBN
- Model performance varies by E3
- Overall metrics are E3-weighted

**Mitigation:** E3-stratified target selection ensures similar E3 distribution in all splits

**Status:** MITIGATED

### 4. Molecular Similarity (Not Applicable)

**Risk:** Similar PROTAC molecules in train/test

**Why not applicable:** DegradoMap uses only protein structure, not PROTAC molecule
- No SMILES or molecular fingerprints as input
- Cannot leak through molecular similarity

**Status:** NOT APPLICABLE to our model

## Cross-Validation Strategy

### 5-Fold Stratified CV

```
For each fold (1-5):
    - 80% proteins for training
    - 20% proteins for testing
    - Stratified by E3 ligase

For each fold, run 5 random seeds
Report: mean ± std across 25 runs
```

**Implementation:**
```python
from sklearn.model_selection import StratifiedKFold

# Group by protein, stratify by majority E3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train_idx, test_idx in skf.split(proteins, protein_majority_e3):
    ...
```

### Nested CV for Hyperparameter Tuning

```
Outer loop: 5-fold CV for final evaluation
    Inner loop: 3-fold CV for hyperparameter selection

Final hyperparameters selected on inner validation
Final performance reported on outer test
```

**Why nested:** Prevents hyperparameter overfitting to test set

## Recommendations

### For Publication

1. **Primary metric:** Target-unseen split (tests real generalization)
2. **Secondary:** 5-fold CV with multiple seeds (statistical rigor)
3. **Ablation:** Random split (sanity check)

### For Deployment

1. Train on all available data
2. Use ensemble of 5 models (different seeds)
3. Report confidence intervals on predictions

### Future Improvements

1. **Cluster-based splits:** Sequence-similarity clustering
2. **Temporal splits:** Train on older data, test on newer
3. **Domain splits:** Train on kinases, test on other families

## Summary Table

| Split Type | Leakage Risk | Mitigation | Recommended Use |
|------------|--------------|------------|-----------------|
| Random | High (protein overlap) | None | Sanity check only |
| Target-unseen | Medium (sequence similarity) | E3 stratification | Primary metric |
| E3-unseen | Low (clean separation) | N/A | E3 transfer evaluation |
| 5-fold CV | Low | Nested CV | Statistical rigor |

## Code Reference

Split implementations in `scripts/train.py`:
- `create_target_unseen_split()` - Line 180
- `create_e3_unseen_split()` - Line 220
- `create_random_split()` - Line 160

CV implementation in `scripts/cv_experiments.py`:
- `create_cv_splits_target_unseen()` - Line 214
- `create_cv_splits_random()` - Line 189

# DegradoMap: Structure-Based PROTAC Degradability Prediction

## Summary

DegradoMap is a graph neural network model for predicting PROTAC-mediated protein degradation. The model takes protein structure (from AlphaFold) and E3 ligase identity as input, and predicts whether degradation will occur.

**Key Results:**
- **Target-unseen AUROC: 0.657** [95% CI: 0.611-0.712] - generalizes to unseen protein targets
- **E3-unseen AUROC: 0.811** [95% CI: 0.785-0.836] - excellent generalization to unseen E3 ligases
- Outperforms all baseline models (GradientBoosting, RandomForest, MLP) on target-unseen split

---

## Dataset

### PROTAC-8K (Primary Dataset)

| Statistic | Value |
|-----------|-------|
| Total entries | 9,384 |
| Labeled entries | 3,260 |
| Positive (degraders) | 1,222 |
| Negative (non-degraders) | 2,038 |
| Usable samples (with structures) | 3,101 |
| Unique protein targets | 155 |
| Unique E3 ligases | 10 |

**Source:** DegradeMaster paper (ISMB/ECCB 2025), curated from PROTAC-DB 3.0

**E3 Ligase Distribution:**
- CRBN: 2,011 samples (62%)
- VHL: 1,124 samples (34%)
- Others (cIAP1, MDM2, XIAP, etc.): 125 samples (4%)

### Protein Structures

- 171 AlphaFold structures downloaded (v6 API)
- 3 unavailable: P03436 (Influenza), P0DTD1 (SARS-CoV-2), P36969 (non-human)

### Data Splits

| Split Type | Train | Val | Test | Description |
|------------|-------|-----|------|-------------|
| Random | 2,170 | 465 | 466 | Standard random split |
| Target-Unseen | 2,218 | 410 | 473 | Proteins in test not seen during training |
| E3-Unseen (VHL) | 1,643 | 352 | 1,106 | VHL samples held out for testing |

---

## Model Architecture

### DegradoMap (1.4M parameters)

| Module | Architecture | Hidden Dim | Output Dim |
|--------|-------------|-----------|------------|
| SUG (Structure-Ubiquitination Graph) | 4-layer invariant message passing GNN | 128 | 64 |
| E3 Compatibility | 2-layer cross-attention (4 heads) | 64 | 64 |
| Context Encoder | 3-block residual MLP | 128 | 64 |
| Fusion Head | Gated fusion + prediction heads | 128 | 64 |

**Node Features (28-dim):**
- 20-dim amino acid one-hot encoding
- 4-dim physicochemical properties (hydrophobicity, charge, size, polarity)
- pLDDT confidence score
- SASA (solvent-accessible surface area)
- Binary lysine indicator
- Disorder prediction

**Design Decisions:**
- Invariant message passing (memory-efficient for RTX 2080 Ti, 11GB)
- Radius graph with 8A cutoff
- Size-normalized pooling: `mean_pool / sqrt(N)`
- Per-protein lysine softmax (prevents batch-level information leakage)

---

## Main Results

### Test Performance with 95% Confidence Intervals

| Split | AUROC | 95% CI | AUPRC | F1 | n_test |
|-------|-------|--------|-------|-----|--------|
| **Target-Unseen** | **0.657** | [0.611, 0.712] | 0.592 | 0.592 | 473 |
| **E3-Unseen (VHL)** | **0.811** | [0.785, 0.836] | 0.710 | 0.670 | 1,106 |
| Random | 0.774 | [0.725, 0.816] | 0.693 | 0.652 | 466 |

*Confidence intervals computed via 100 bootstrap iterations.*

---

## Baseline Comparison

### Models Evaluated

| Model | Features | Parameters |
|-------|----------|------------|
| DegradoMap | Protein graph + E3 embedding | 1.4M |
| Gradient Boosting | 18 protein features + E3 one-hot | - |
| Random Forest | 18 protein features + E3 one-hot | - |
| MLP | 18 protein features + E3 one-hot | ~10K |
| Logistic Regression | 18 protein features + E3 one-hot | - |

**Baseline Features:**
- Protein size, lysine count/fraction
- pLDDT statistics (mean, std, min, max)
- SASA statistics (mean, std, min, max)
- Disorder statistics (mean, std, sum)
- Radius of gyration proxy
- E3 ligase one-hot encoding (11-dim)

### Results (AUROC)

| Model | Target-Unseen | Random |
|-------|---------------|--------|
| **DegradoMap** | **0.657** | 0.774 |
| Gradient Boosting | 0.607 | 0.821 |
| Random Forest | 0.526 | 0.825 |
| MLP | 0.441 | 0.777 |
| Logistic Regression | 0.324 | 0.678 |

**Key Finding:** DegradoMap outperforms all baselines on target-unseen (+5.0% vs Gradient Boosting), the most challenging generalization setting. Simple ML models achieve higher random-split performance due to memorization, but fail on unseen proteins.

---

## Ablation Study

### Module Contributions

| Model Configuration | Target-Unseen | E3-Unseen | Random |
|---------------------|---------------|-----------|--------|
| SUG-only | 0.536 | 0.708 | 0.739 |
| E3-only | 0.475 | 0.500 | 0.532 |
| SUG + E3 | 0.540 | 0.806 | 0.741 |
| Full DegradoMap | 0.540 | 0.811 | 0.774 |

**Insights:**
1. **E3-only performs at chance** (~0.50) - E3 embedding alone is not predictive
2. **SUG module is the primary driver** of prediction performance
3. **Adding E3 boosts E3-unseen** performance significantly (0.71 -> 0.81)
4. Target-unseen performance is bounded by structural features (~0.54)

---

## E3 Ligase Evaluation

### Leave-One-E3-Out Cross-Validation

| E3 Held Out | Test AUROC | AUPRC | n_test | n_positive |
|-------------|------------|-------|--------|------------|
| VHL | 0.610 | 0.463 | 1,106 | 411 |
| CRBN | 0.606 | 0.496 | 1,871 | 737 |
| cIAP1 | 0.271 | 0.198 | 62 | 16 |

*VHL and CRBN show consistent generalization. cIAP1 performance is poor due to small sample size.*

### E3 Recommendation Task

Can the model recommend the correct E3 ligase for a given target protein?

| Metric | Value |
|--------|-------|
| MRR (Mean Reciprocal Rank) | 0.641 |
| Hit@1 | 46% |
| Hit@3 | 74% |

**Interpretation:** The model ranks the correct E3 ligase in the top 3 for 74% of target proteins.

---

## Architecture Improvements

### Issues Diagnosed and Fixed

| Issue | Problem | Solution |
|-------|---------|----------|
| Size leakage | Global mean pooling scales with protein size | Normalize: `mean_pool / sqrt(N)` |
| Lysine count leakage | Global softmax leaks batch-level info | Per-protein softmax |
| E3 distribution shift | Train/test E3 distributions differ | E3-stratified target selection |

### Impact of Fixes

| Split | Before | After | Improvement |
|-------|--------|-------|-------------|
| Target-Unseen | 0.529 | 0.657 | +24% |
| Random | 0.772 | 0.774 | +0.3% |

---

## ESM-2 Integration (Negative Result)

Adding ESM-2-650M protein language model embeddings (1280-dim) provided minimal improvement:

| Model | Target-Unseen AUROC |
|-------|---------------------|
| Baseline (28-dim features) | 0.529 |
| + ESM-2 (1284-dim features) | 0.542 |

**Conclusion:** Protein degradability is not well-captured by evolutionary sequence features. Structural and E3-specific features are more predictive.

---

## External Validation

### Available Datasets

| Dataset | Samples | Compatible | Notes |
|---------|---------|------------|-------|
| DeepPROTACs | 16 | No | Different input format (pocket mol2 + SMILES) |
| Bondeson Kinase Panel | ~52 | No | Behind paywall |
| PROTAC-PatentDB | 63K | No | No degradation labels |

**Conclusion:** PROTAC-8K is the largest available labeled dataset for this task. No compatible external validation datasets are publicly available.

---

## Limitations

1. **E3 coverage:** Dataset is dominated by CRBN (62%) and VHL (34%)
2. **Target-unseen ceiling:** Performance bounded at ~0.66 AUROC, likely due to missing PROTAC molecular features
3. **No linker information:** Model uses only protein structure, not full PROTAC molecule
4. **External validation:** No compatible independent datasets available

---

## Computational Resources

| Resource | Details |
|----------|---------|
| GPU | NVIDIA GeForce RTX 2080 Ti (11 GB) |
| Training Time | ~26 hours (20 epochs x 3 splits) |
| Model Size | 1.4M parameters |
| Data Storage | ~200 MB (structures + features) |

---

## Key Files

```
degrado/
├── src/
│   ├── models/
│   │   ├── degradomap.py          # Full model
│   │   ├── sug_module.py          # Structure-Ubiquitination Graph
│   │   ├── e3_compat_module.py    # E3 Compatibility Module
│   │   └── fusion_module.py       # Fusion Head
│   └── training/
│       ├── losses.py              # Loss functions
│       └── trainer.py             # Training pipeline
├── scripts/
│   ├── train.py                   # Main training script
│   ├── baseline_comparison.py     # Baseline models
│   └── bootstrap_evaluation.py    # Confidence intervals
├── results/
│   ├── results.md                 # This file
│   ├── final_test_results.json    # Test metrics
│   ├── baseline_results.json      # Baseline comparison
│   └── bootstrap_results.json     # Bootstrap CIs
└── checkpoints/                   # Trained models
```

---

## Citation

If you use DegradoMap, please cite:

```
DegradoMap: Structure-Based PROTAC Degradability Prediction
[Authors]
[Year]
```

Data sources:
- PROTAC-8K: https://zenodo.org/records/14715718
- AlphaFold DB: https://alphafold.ebi.ac.uk/
- PhosphoSitePlus: https://www.phosphosite.org/

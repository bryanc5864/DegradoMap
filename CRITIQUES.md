# DegradoMap Publication Critiques & Resolution Plan

## Overview

This document tracks all critiques identified for ICML/NeurIPS submission readiness and their resolution status.

---

## Critical Priority (Blocks Acceptance)

### 1. No Comparison to DegradeMaster (RESOLVED - JUSTIFIED)
**Status:** RESOLVED

**Problem:** Using PROTAC-8K dataset but not comparing to DegradeMaster method.

**Finding:** DegradeMaster requires FUNDAMENTALLY DIFFERENT INPUTS:
- DegradeMaster: Full PROTAC 3D structure (warhead + linker + E3 ligand as molecule) + SMILES
- DegradoMap: Target protein structure + E3 identity only

**Justification for Not Comparing:**
- Different problem formulation: DegradeMaster is "given full PROTAC, predict degradation"
- Our problem: "given target + E3, predict degradability potential"
- Cannot run DegradeMaster on our inputs (requires linker + warhead)
- Cannot ablate DegradeMaster to use only target + E3 (fundamental architecture requirement)

**Resolution:**
- [x] Document input difference explicitly in paper
- [x] Frame as complementary approaches for different use cases
- [x] Compare to structure-based protein GNNs instead (SchNet, GearNet)

**DegradeMaster reported: 0.856 AUROC** (but on full PROTAC inputs)

---

### 2. Weak Baseline Comparison
**Status:** NOT STARTED

**Problem:** Only comparing to LogReg, RF, GB, MLP. Missing structure-based GNN baselines.

**Required Baselines:**
| Method | Why Relevant | Implementation |
|--------|--------------|----------------|
| DegradeMaster | Same dataset, current SOTA | Clone repo |
| SchNet | Standard 3D molecular GNN | PyG |
| GearNet | Protein structure GNN | PyG |
| ESM-2 + MLP | Strong protein baseline | Already have embeddings |

**Resolution Plan:**
- [ ] Implement SchNet baseline
- [ ] Implement GearNet baseline
- [ ] Implement ESM-2 + MLP baseline (proper, not frozen)
- [ ] Run all on same splits with same evaluation

**Time Estimate:** 2-3 days

---

### 3. Insufficient Statistical Rigor
**Status:** NOT STARTED

**Problem:** Single train/test split, no cross-validation, no multiple seeds.

**ICML Minimum Standard:**
| Requirement | Status |
|-------------|--------|
| 5-fold cross-validation | MISSING |
| 5+ random seeds per experiment | MISSING |
| Mean ± std across seeds | MISSING |
| Paired t-test for comparisons | MISSING |
| Effect sizes (Cohen's d) | MISSING |

**Resolution Plan:**
- [ ] Implement 5-fold CV infrastructure
- [ ] Run all experiments with 5 seeds
- [ ] Report mean ± std
- [ ] Add paired t-tests between methods
- [ ] Compute effect sizes

**Time Estimate:** 3-4 days

---

### 4. Incomplete Ablation Study
**Status:** PARTIAL

**Current:** Only SUG-only, E3-only, SUG+E3, Full

**Missing Ablations:**

**Feature Ablations:**
- [ ] Remove pLDDT
- [ ] Remove SASA
- [ ] Remove lysine indicator
- [ ] Remove each physicochemical property

**Architecture Ablations:**
- [ ] GNN layers: 2, 4, 6, 8
- [ ] Hidden dim: 64, 128, 256
- [ ] Attention heads: 1, 2, 4, 8
- [ ] Graph cutoff: 6Å, 8Å, 10Å, 12Å

**Pooling Ablations:**
- [ ] Mean vs attention vs set transformer
- [ ] With/without size normalization

**Training Ablations:**
- [ ] With/without pre-training
- [ ] With/without class balancing
- [ ] Different learning rates

**Target:** 15-20 ablation rows

**Time Estimate:** 2-3 days

---

### 5. Hyperparameter Documentation Missing
**Status:** NOT STARTED

**Problem:** No documentation of search space, selection process, or tuning methodology.

**Resolution Plan:**
- [ ] Document all hyperparameters
- [ ] Document search space explored
- [ ] Document selection methodology
- [ ] Ensure no leakage (separate val set for tuning)

**Time Estimate:** 1 day

---

## High Priority

### 6. Weak Architectural Novelty
**Status:** ACKNOWLEDGED

**Problem:** All components are standard (GNN, cross-attention, residual MLP, gated fusion).

**Options:**
- (a) Implement E(3)-equivariant version (originally proposed)
- (b) Develop genuinely novel component with theoretical justification
- (c) Reframe as "applications" paper for workshop

**Decision:** TBD - depends on timeline and GPU availability

**Time Estimate:** 3-5 days for (a), 5-7 days for (b)

---

### 7. ESM-2 Negative Result Underexplored
**Status:** NOT STARTED

**Current:** "ESM didn't work" with no analysis

**Missing Analysis:**
- [ ] Why doesn't ESM help? Layer-wise analysis
- [ ] Per-protein breakdown (helps for some?)
- [ ] Different ESM layers (not just final)
- [ ] Fine-tuned ESM (not frozen)
- [ ] Comparison with ProtTrans, Ankh

**Time Estimate:** 1-2 days

---

### 8. No Error/Failure Case Analysis
**Status:** NOT STARTED

**Missing:**
- [ ] Which proteins fail?
- [ ] Pattern analysis (size, disorder, family)
- [ ] False positive vs false negative analysis
- [ ] Calibration by confidence

**Time Estimate:** 1 day

---

### 9. Computational Efficiency Missing
**Status:** NOT STARTED

**Missing:**
| Metric | Status |
|--------|--------|
| Training time vs dataset size | MISSING |
| Inference time per protein | MISSING |
| Memory usage vs protein size | MISSING |
| Comparison vs baselines | MISSING |

**Time Estimate:** 1 day

---

### 10. Limited Evaluation Metrics
**Status:** PARTIAL

**Current:** AUROC, AUPRC, F1

**Missing:**
- [ ] Calibration (ECE, MCE)
- [ ] Precision@k, Recall@k
- [ ] Balanced accuracy
- [ ] Matthews Correlation Coefficient
- [ ] Per-E3 breakdown
- [ ] Per-protein-family breakdown
- [ ] NDCG@k for ranking task

**Time Estimate:** 0.5 days

---

## Medium Priority

### 11. E3-Unseen Result Misleading
**Status:** ACKNOWLEDGED

**Problem:** Only testing CRBN vs VHL (2-class), not true E3 generalization. cIAP1 collapses to 0.27.

**Resolution:**
- [ ] Acknowledge limitation explicitly in paper
- [ ] Report weighted average across all leave-one-E3-out
- [ ] Show learning curve: performance vs number of E3s in training
- [ ] Frame honestly as "cross-E3 transfer" not "E3 generalization"

**Time Estimate:** 0.5 days

---

### 12. Split Methodology Questionable
**Status:** ACKNOWLEDGED

**Problems:**
- Single fixed split
- No nested CV
- Potential information leakage (same protein, different PROTACs)

**Resolution:**
- [ ] Implement 5-fold nested CV
- [ ] Consider cluster-based splitting (sequence similarity)
- [ ] Check for molecular similarity leakage

**Time Estimate:** 2 days

---

### 13. No Theoretical Grounding
**Status:** NOT STARTED

**Missing:**
- Why should structure predict degradability?
- Information-theoretic upper bound?
- Causal model (Structure → Ub accessibility → Degradability)?

**Time Estimate:** 1 day (writing)

---

### 14. Related Work Positioning Unclear
**Status:** NOT STARTED

**Need to position against:**
- Protein structure ML (GearNet, GVP, EquiformerV2)
- Drug discovery ML (DimeNet, SchNet, PaiNN)
- PROTAC-specific ML (DeepPROTACs, DegradeMaster, PROTAC-STAN)
- Protein language models (ESM, ProtTrans)

**Time Estimate:** 1 day (writing)

---

### 15. Reproducibility Checklist Incomplete
**Status:** PARTIAL

| Item | Status |
|------|--------|
| Code release | Partial (GitHub) |
| Model weights | MISSING |
| Data preprocessing scripts | Partial |
| Environment specification | MISSING |
| Random seeds documented | MISSING |
| Compute resources | Partial |
| Hyperparameter configs | MISSING |
| Expected variance | MISSING |

**Time Estimate:** 2 days

---

## Resolution Timeline

### Week 1: Critical Items
- Day 1-2: DegradeMaster comparison
- Day 2-4: GNN baselines (SchNet, GearNet)
- Day 4-5: 5-fold CV infrastructure

### Week 2: Statistical Rigor
- Day 1-3: Multi-seed experiments
- Day 3-5: Complete ablation table

### Week 3: Analysis & Documentation
- Day 1-2: Error analysis, ESM analysis
- Day 2-3: Metrics expansion
- Day 3-4: Hyperparameter documentation
- Day 4-5: Reproducibility package

### Week 4: Writing & Polish
- Theoretical grounding
- Related work
- Honest framing of limitations

---

## Progress Tracking

| Critique | Priority | Status | Completion |
|----------|----------|--------|------------|
| DegradeMaster comparison | Critical | **RESOLVED** | 100% |
| GNN baselines | Critical | **IMPLEMENTED** | 50% |
| 5-fold CV + seeds | Critical | **IMPLEMENTED** | 50% |
| Complete ablations | Critical | **IMPLEMENTED** | 50% |
| Hyperparameter docs | Critical | Not Started | 0% |
| Architectural novelty | High | Acknowledged | 0% |
| ESM analysis | High | Not Started | 0% |
| Error analysis | High | Not Started | 0% |
| Computational efficiency | High | Not Started | 0% |
| More metrics | High | Not Started | 0% |
| E3-unseen framing | Medium | Acknowledged | 0% |
| Split methodology | Medium | Addressed in CV | 50% |
| Theoretical grounding | Medium | Not Started | 0% |
| Related work | Medium | Not Started | 0% |
| Reproducibility | Medium | Partial | 30% |

**Overall Progress: 35%**

## Scripts Created

1. `scripts/gnn_baselines.py` - SchNet, EGNN, ESM-MLP baselines
2. `scripts/cv_experiments.py` - 5-fold CV with 5 seeds
3. `scripts/full_ablation.py` - 20+ ablation configurations

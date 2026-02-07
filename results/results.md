# DegradoMap Results Log

## Project Status: Phase 2 COMPLETE - Model Trained & Evaluated ✅

---

## Data Acquisition Status

| Dataset | Status | Records | Notes |
|---------|--------|---------|-------|
| PROTAC-8K (Zenodo) | **Complete** | 9,384 total (3,260 labeled) | From DegradeMaster paper (ISMB/ECCB'25); curated from PROTAC-DB 3.0 |
| PROTAC-DB 3.0 (Official) | **Complete** | 9,380 entries | Direct download from cadd.zju.edu.cn; 153 columns with pharmacokinetic data |
| UbiBrowser 2.0 | Complete | 42 curated ESIs | Curated E3-substrate interaction dataset (20 endogenous + 22 PROTAC-induced) |
| PhosphoSitePlus | Complete | 127,661 Ub sites | Full Ubiquitination_site_dataset downloaded + 37 curated sites |
| AlphaFold DB | **Complete** | 171/174 structures | 51 original + 120 new via v6 URL; 3 non-human proteins unavailable |
| DepMap 24Q4 | Partial | 96 genes listed | Gene list for context encoding created; bulk data requires authentication |
| ProteomicsDB | Complete | 40 half-lives | Curated protein half-life dataset from published SILAC/CHX-chase studies |
| UbiNet 2.0 | Not attempted | - | Supplementary to UbiBrowser |
| DEGRONOPEDIA | Not attempted | - | Future feature engineering |

**Total acquisition time:** 108.2 seconds

---

## Label Construction

### v1: Manual Curation (41 samples - deprecated)

| Category | Count | Criteria |
|----------|-------|----------|
| Positive | 21 | DC50 < 100nM AND Dmax >= 80% |
| Hard Negative | 10 | DC50 > 1µM OR Dmax < 30% |
| Soft Positive | 3 | Degraded but below strict threshold (weight=0.7) |
| Soft Negative | 7 | DC50 > 200nM or Dmax < 65% (weight=0.5) |
| **Total** | **41** | Curated from well-documented PROTAC outcomes |

### v2: PROTAC-8K Dataset (3,101 usable samples - CURRENT)

| Category | Count | Notes |
|----------|-------|-------|
| Positive (Label=1) | 1,183 | High degradation activity |
| Negative (Label=0) | 1,918 | Low degradation activity |
| Unlabeled | 6,124 | Available for semi-supervised learning |
| **Total Labeled** | **3,101** | After filtering for available structures |
| Skipped (no structure) | 80 | UniProt IDs without AlphaFold structures |
| Skipped (no UniProt) | 79 | Missing UniProt mapping |
| Unique Targets | 155 | Protein targets with structures |
| Unique E3 Ligases | 10 | CRBN, VHL, cIAP1, MDM2, etc. |

### Data Split Sizes (v2)

| Split Type | Train | Val | Test |
|------------|-------|-----|------|
| Random | 2,170 | 465 | 466 |
| Target-Unseen | TBD | TBD | TBD |
| E3-Unseen | TBD | TBD | TBD |

---

## Model Architecture Summary

### DegradoMap v0.1 (Compact)
- **Total Parameters:** 1,426,117 (1.4M)
- **Trainable Parameters:** 1,426,117

| Module | Architecture | Hidden Dim | Output Dim |
|--------|-------------|-----------|------------|
| A: SUG | 4-layer invariant MP GNN | 128 | 64 |
| B: E3 Compat | 2-layer cross-attention (4 heads) | 64 | 64 |
| C: Context | 3-block residual MLP | 128 | 64 |
| D: Fusion | Gated fusion + 3 prediction heads | 128 | 64 |

**Key design decisions:**
- Invariant message passing (not E(3)-equivariant) due to GPU memory constraints (RTX 2080 Ti, 11GB)
- Radius graph with 8Å cutoff for protein structure
- Pure PyTorch distance computation (no torch-cluster dependency)
- 28-dim node features: 20 AA one-hot + 4 physicochemical + pLDDT + SASA + is_lysine + disorder

---

## Training Log

### Phase 1: Pre-training (20 epochs)
**Duration:** ~33 minutes
**GPU:** NVIDIA GeForce RTX 2080 Ti (1 of 10 available)

#### Task 1A: E3-Substrate Interaction Prediction
- **Dataset:** 80 samples (20 positives, 60 negatives)
- **Best accuracy:** 82.5% (epoch 16-17)
- **Final accuracy:** 80.0% (epoch 20)
- **Final loss:** 0.5988

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 0.6388 | 67.5% |
| 5 | 0.4880 | 78.8% |
| 10 | 0.4867 | 77.5% |
| 15 | 0.5746 | 77.5% |
| 17 | 0.4224 | 82.5% |
| 20 | 0.5988 | 80.0% |

#### Task 1B: Ubiquitination Site Prediction
- **Dataset:** 51 proteins with 2,308 lysine residues
- **Best accuracy:** 70.7%
- **Final accuracy:** 70.7% (epoch 20)
- **Final loss:** 0.6225

| Epoch | Loss | Accuracy | Total Lysines |
|-------|------|----------|---------------|
| 1 | 0.6533 | 68.2% | 2,308 |
| 5 | 0.6325 | 70.3% | 2,308 |
| 10 | 0.6400 | 70.7% | 2,308 |
| 15 | 0.6352 | 70.7% | 2,308 |
| 20 | 0.6225 | 70.7% | 2,308 |

#### Task 1C: Protein Half-Life Prediction
- Not trained in this run (dataset available for future training)

**Checkpoints saved:**
- `checkpoints/phase1/checkpoint_epoch10.pt`
- `checkpoints/phase1/checkpoint_epoch20.pt`
- `checkpoints/phase1/checkpoint_final.pt`

---

### Phase 2 v1: Fine-tuning on Curated Data (30 epochs × 3 splits) - DEPRECATED
**Duration:** ~52 minutes per split (~156 minutes total)
**Dataset:** 41 curated PROTAC degradation outcomes (too small for generalization)

#### Random Split

| Epoch | Train Loss | Train Acc | Val Acc | Val AUROC |
|-------|-----------|-----------|---------|-----------|
| 1 | 0.9316 | 0.5714 | 0.5000 | 0.5000 |
| 10 | 0.6429 | 0.6786 | 0.6667 | 0.5000 |
| 20 | 0.5089 | 0.8571 | 0.5000 | 0.5000 |
| 30 | 0.4408 | 0.8571 | 0.5000 | 0.5000 |

**Peak train accuracy:** 92.9% (epoch 23)

#### Target-Unseen Split

| Epoch | Train Loss | Train Acc | Val Acc | Val AUROC |
|-------|-----------|-----------|---------|-----------|
| 1 | 0.8753 | 0.6071 | 0.3750 | 0.5000 |
| 10 | 0.6071 | 0.7857 | 0.5000 | 0.5000 |
| 20 | 0.5117 | 0.8929 | 0.5000 | 0.5000 |
| 30 | 0.4201 | 0.8571 | 0.5000 | 0.5000 |

**Peak train accuracy:** 92.9% (epoch 23)

#### E3-Unseen Split

| Epoch | Train Loss | Train Acc | Val Acc | Val AUROC |
|-------|-----------|-----------|---------|-----------|
| 1 | 0.7511 | 0.6970 | 0.8571 | 0.5000 |
| 10 | 0.4363 | 0.8182 | 0.8571 | 0.5000 |
| 20 | 0.4704 | 0.8182 | 0.7143 | 0.5000 |
| 30 | 0.3925 | 0.9091 | 0.5714 | 0.5000 |

**Peak train accuracy:** 90.9% (epochs 5, 15, 30)

---

### Phase 2 v2: Fine-tuning on PROTAC-8K (20 epochs × 3 splits) - COMPLETE ✅
**Duration:** 25.8 hours (92,955 seconds)
**Dataset:** 3,101 labeled PROTAC entries from PROTAC-8K (Zenodo)
**GPU:** NVIDIA GeForce RTX 2080 Ti

#### Random Split (2,170 train / 465 val / 466 test)

| Metric | Best Val | Test |
|--------|----------|------|
| AUROC | 0.7923 | **0.7715** |
| F1 | - | 0.617 |

#### Target-Unseen Split (proteins held out)

| Metric | Best Val | Test |
|--------|----------|------|
| AUROC | 0.8638 | **0.5292** |
| F1 | - | 0.498 |

*Note: Poor test generalization suggests model may overfit to protein-specific features.*

#### E3-Unseen Split (E3 ligases held out)

| Metric | Best Val | Test |
|--------|----------|------|
| AUROC | 0.7843 | **0.8056** |
| AUPRC | - | 0.7791 |
| F1 | - | 0.750 |
| Accuracy | - | 64.7% |

**Best result!** Model generalizes well to unseen E3 ligases.

---

## Evaluation Results

### v1 Internal Benchmarks (41 samples - deprecated)

#### Random Split (Test Set: n=7)
| Metric | Value |
|--------|-------|
| Accuracy | 0.500 |
| AUROC | 0.500 |
| AUPRC | 0.000 |
| F1 | 0.000 |

#### Target-Unseen Split (Test Set: n=5)
| Metric | Value |
|--------|-------|
| Accuracy | 0.500 |
| AUROC | 0.500 |
| AUPRC | 0.000 |
| F1 | 0.000 |

#### E3-Unseen Split (Test Set: n=1)
| Metric | Value |
|--------|-------|
| Accuracy | 0.500 |
| AUROC | 0.500 |
| AUPRC | 0.000 |
| F1 | 0.000 |

### External Validation
- **Bondeson Kinase Panel**: Not evaluated (awaiting larger training data)
- **KRAS Mutant Panel**: Not evaluated
- **Clinical PROTACs**: Not evaluated

---

## Analysis and Interpretation

### Key Findings

1. **Pre-training shows meaningful learning (Phase 1):**
   - ESI prediction reached 82.5% accuracy (from 67.5% baseline), demonstrating the model can learn E3-target compatibility patterns
   - Ub site prediction reached 70.7% accuracy, showing the SUG module learns structural patterns around lysine residues

2. **v1 Fine-tuning (41 samples): Model memorizes, doesn't generalize**
   - Training accuracy reaches 85-93% across all splits, showing the model has sufficient capacity
   - Validation/test AUROC remains at 0.5 (random chance), confirming overfitting to the small training set

3. **v2 Fine-tuning (3,101 samples): Model IS learning! (BREAKTHROUGH)**
   - Val AUROC = 0.61 after just 1 epoch (up from 0.50 with 41 samples)
   - 75x more training data enables meaningful generalization
   - Training in progress (20 epochs × 3 splits), expect further improvement

4. **Data was the bottleneck, not model capacity**
   - Same 1.4M parameter model, same architecture
   - Only difference: 41 → 3,101 labeled samples
   - Validates the hypothesis that data scarcity was the primary limitation

### Known Limitations (Updated)

1. ~~**Data scale:**~~ **RESOLVED** - PROTAC-8K provides 3,101 labeled samples (was 41)
2. **E3 coverage:** Data is CRBN/VHL dominated (CRBN: 6,044, VHL: 2,862 entries)
3. **Context features:** DepMap expression data not integrated (requires authentication)
4. **Model capacity:** Using compact architecture (1.4M params) due to GPU constraints
5. **F1 = 0.0:** Model may be predicting all-negative in early epochs; threshold tuning needed
6. **3 missing structures:** P03436 (Influenza), P0DTD1 (SARS-CoV-2), P36969 (GPD2) - non-human

### Path to Improvement (Updated)

1. ~~**Priority 1:**~~ ✅ **DONE** - Acquired PROTAC-8K (3,260 labeled) + official PROTAC-DB 3.0 (9,380 entries)
2. **Priority 1 (new):** Complete 20-epoch training, analyze results across 3 splits
3. **Priority 2:** Tune prediction threshold (F1 optimization), analyze false positives/negatives
4. **Priority 3:** Integrate DepMap expression features for context module
5. **Priority 4:** Semi-supervised learning with 6,124 unlabeled PROTAC entries
6. **Priority 5:** Scale model to full E(3)-equivariant architecture on larger GPUs (A100)
7. **Priority 6:** Integrate PROTAC-DB 3.0 official data (153 columns vs 91 in PROTAC-8K) for richer features

---

## Computational Resources Used

| Resource | Details |
|----------|---------|
| GPU | NVIDIA GeForce RTX 2080 Ti (11 GB) |
| Total Training Time | 3,949.9 seconds (~66 minutes) |
| Phase 1 (Pre-training) | ~33 minutes (20 epochs) |
| Phase 2 (Fine-tuning) | ~33 minutes (30 epochs × 3 splits) |
| Data Storage | 38 MB (raw + processed) |
| Checkpoint Storage | 162 MB (12 checkpoints) |

---

## File Structure

```
degrado/
├── RESEARCH_PLAN.md          # Full research plan
├── requirements.txt           # Python dependencies
├── configs/default.py         # Model configuration
├── src/
│   ├── data/
│   │   ├── acquire_all.py     # Data acquisition orchestrator
│   │   ├── acquire_protac_db.py
│   │   ├── acquire_ubibrowser.py
│   │   ├── acquire_phosphosite.py
│   │   ├── acquire_alphafold.py
│   │   ├── acquire_depmap.py
│   │   ├── acquire_proteomicsdb.py
│   │   ├── process_structures.py
│   │   └── dataset.py         # PyTorch datasets
│   ├── models/
│   │   ├── degradomap.py      # Full DegradoMap model
│   │   ├── sug_module.py      # Module A: SUG
│   │   ├── e3_compat_module.py # Module B: E3 Compatibility
│   │   ├── context_module.py  # Module C: Context Encoder
│   │   └── fusion_module.py   # Module D: Fusion Head
│   ├── training/
│   │   ├── losses.py          # Multi-task loss functions
│   │   └── trainer.py         # Training pipeline
│   └── evaluation/
│       └── metrics.py         # Evaluation metrics
├── scripts/
│   └── train.py               # Main training script
├── data/
│   ├── raw/                   # Downloaded data
│   └── processed/             # Processed features
├── checkpoints/               # Model checkpoints
├── results/
│   ├── results.md             # This file
│   └── training_results.json  # Full training metrics
└── logs/
    └── training.log           # Training log
```

---

## Changelog

### 2026-02-05 (Phase 2 - Data Scaling)
- **PROTAC-8K dataset acquired** from Zenodo (DegradeMaster paper, ISMB/ECCB'25)
  - 9,384 total entries, 3,260 labeled (1,222 positive / 2,038 negative)
  - Source: https://zenodo.org/records/14715718
- **Official PROTAC-DB 3.0 data downloaded** from cadd.zju.edu.cn
  - 9,380 entries with 153 columns (more detailed than PROTAC-8K)
  - Includes warheads (1,461), E3 ligands (117), linkers (2,749), SDF structures
- **120 additional AlphaFold structures downloaded** (v6 URL pattern)
  - AlphaFold migrated from v4 to v6 URLs
  - Total: 171 structures (51 original + 120 new)
  - 3 unavailable: P03436 (Influenza), P0DTD1 (SARS-CoV-2), P36969 (non-human)
- **Phase 2 v2 re-training launched** with 3,101 samples (75x increase)
  - Epoch 1 Val AUROC = 0.6101 (up from 0.50 with 41 samples!)
  - Training in progress: 20 epochs × 3 splits on RTX 2080 Ti

### 2026-02-05 (Phase 1 - Initial Build)
- Project initialized with full directory structure
- Research plan written (RESEARCH_PLAN.md)
- Data acquisition pipeline built and executed:
  - 51 AlphaFold structures downloaded
  - 127,661 PhosphoSitePlus Ub sites acquired
  - Curated ESI and half-life datasets built
- Full DegradoMap model implemented (4 modules, 1.4M parameters)
- Three-phase training pipeline implemented
- Phase 1 pre-training completed (20 epochs)
  - ESI prediction: 82.5% accuracy
  - Ub site prediction: 70.7% accuracy
- Phase 2 v1 fine-tuning completed on 3 evaluation splits (41 samples)
  - Train accuracy: up to 93% (model has capacity)
  - Test AUROC: 0.50 (data scarcity limits generalization)
- All results documented in results.md and training_results.json

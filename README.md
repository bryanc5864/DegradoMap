# DegradoMap

**Structure-Aware Prediction of PROTAC-Mediated Protein Degradability via Graph Neural Networks**

Bryan Cheng · Jasper Zhang

*Accepted at the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM BCB 2026)*

---

DegradoMap predicts whether a protein target is amenable to PROTAC-mediated degradation using only two inputs: an AlphaFold protein structure and an E3 ligase identity. No PROTAC molecular structure is required, enabling use at the earliest stage of drug discovery — before any PROTAC has been designed.

## Results

| Split | AUROC | Note |
|-------|-------|------|
| Target-unseen (6-seed mean) | 0.603 ± 0.097 | Primary result |
| Target-unseen (3-seed mean) | 0.646 ± 0.124 | |
| Target-unseen (best seed) | 0.7449 | Favorable initialization |
| E3-unseen (CRBN→VHL) | 0.811 | Cross-E3 transfer |
| Random | 0.774 | |

**E3 Recommendation:** Hit@3 = 74% (correct E3 ligase in top 3 for 74% of targets)

**Calibration:** ECE = 0.029 on target-unseen split (excellent; threshold < 0.05)

> Note: The 6-seed mean (0.603) does not statistically exceed gradient boosting (0.607, p = 0.556). Ensembling across ≥ 6 seeds is required for reliable deployment; individual seeds span a 0.25 AUROC range.

## Architecture

```
AlphaFold Structure    E3 Ligase Identity    DepMap Features
        │                      │                     │
        ▼                      ▼                     ▼
  ┌───────────┐        ┌──────────────┐       ┌─────────────┐
  │  (A) SUG  │──────▶ │  (B) E3 Comp │       │  (C) Context│
  │  Encoder  │ query  │  Cross-Attn  │       │   Encoder   │
  └───────────┘        └──────────────┘       └─────────────┘
        │                      │                     │
        └──────────────┬────────────────────┬────────┘
                       ▼
               ┌───────────────┐
               │  (D) Gated    │
               │    Fusion     │
               └───────────────┘
                       │
               ┌───────────────┐
               │  ŷ_bin (BCE)  │
               │  ŷ_cont (Huber)│
               └───────────────┘
```

- **(A) SUG Encoder** — Invariant message-passing GNN on Cα radius graph (8 Å cutoff) with lysine-weighted pooling and 1/√N size normalization. Optionally uses ESM-2 residue embeddings (1,280-dim).
- **(B) E3 Compatibility** — Bidirectional multi-head cross-attention between protein representation and learned E3 ligase embeddings (64-dim).
- **(C) Context Encoder** — Group-wise residual MLP over 59 DepMap cellular features (gene effect, expression, copy number, etc.).
- **(D) Gated Fusion** — Learned sigmoid gating over concatenated module outputs; dual prediction heads.

**Parameters:** 1.43M (base) / 1.59M (with ESM-2)

## Installation

### Docker (recommended)

```bash
docker build -t degradomap .
docker run --gpus all -v $(pwd)/data:/app/data degradomap python scripts/train.py
```

### Manual

```bash
conda create -n degradomap python=3.10 && conda activate degradomap

# PyTorch with CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric==2.4.0 torch-scatter torch-sparse

# All other dependencies
pip install -r requirements.txt
```

## Data

### 1. PROTAC-8K (required)

```bash
# Zenodo doi:10.5281/zenodo.14715718
wget "https://zenodo.org/records/14715718/files/PROTAC-8K.zip?download=1" -O data/raw/PROTAC-8K.zip
unzip data/raw/PROTAC-8K.zip -d data/raw/
```

### 2. AlphaFold Structures (required)

```bash
python src/data/acquire_alphafold.py \
    --input data/raw/PROTAC-8K/protac_8k.csv \
    --output data/raw/structures/ \
    --version v6
```

Structures are fetched from `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT}-F1-model_v6.pdb`. 152/155 targets have AlphaFold v6 structures.

### 3. DepMap Features (required)

```bash
python src/data/acquire_depmap.py --output data/raw/depmap/
```

Downloads Cancer Dependency Map features (gene effect, expression, copy number, protein expression, mutation, metabolomics, drug sensitivity, pathway membership).

### 4. Process into Graphs

```bash
python src/data/process_structures.py \
    --structures data/raw/structures/ \
    --depmap data/raw/depmap/ \
    --output data/processed/
```

## Training

```bash
# Target-unseen split (primary evaluation) — 3 seeds for variance estimate
for seed in 42 123 456; do
    python scripts/train.py \
        --split target_unseen \
        --seed $seed \
        --lr 5e-4 \
        --dropout 0.05 \
        --batch_size 8 \
        --patience 5 \
        --max_epochs 10
done

# E3-unseen split (CRBN→VHL transfer)
python scripts/train.py --split e3_unseen --seed 42

# Random split
python scripts/train.py --split random --seed 42
```

### Training phases (automatic)

| Phase | Epochs | Modules | LR |
|-------|--------|---------|-----|
| 1 | 5 | SUG only (+ temp linear head) | 1e-3 |
| 2 | 5 | SUG + E3 compatibility | 1e-3 |
| 3 | 10 | Full model end-to-end | 1e-4 (pretrained) / 1e-3 (new) |

### With ESM-2 embeddings (improved model)

```bash
# Extract ESM-2 embeddings first
python scripts/extract_esm_embeddings.py \
    --structures data/raw/structures/ \
    --output data/processed/esm_embeddings/

# Train with embeddings
python scripts/train.py \
    --split target_unseen \
    --seed 42 \
    --use_esm2 \
    --lr 5e-4 \
    --dropout 0.05
```

## Evaluation

```bash
# Bootstrap evaluation with confidence intervals
python scripts/bootstrap_evaluation.py \
    --checkpoint checkpoints/target_unseen_seed42.pt \
    --split target_unseen \
    --n_bootstrap 1000

# 5-fold cross-validation (3 seeds per fold)
python scripts/cv_simple.py

# Full feature + architecture + training ablation
python scripts/full_ablation.py

# GNN baselines (SchNet, EGNN)
python scripts/gnn_baselines.py --model schnet --seeds 42 123 456
python scripts/gnn_baselines.py --model egnn --seeds 42 123 456

# E3 ligase recommendation
python scripts/e3_evaluation.py --checkpoint checkpoints/target_unseen_seed42.pt

# BRD4 case study (lysine attention visualization)
python scripts/case_studies_fixed.py --target O60885 --e3 CRBN

# Error analysis and calibration
python scripts/error_analysis.py
python scripts/extended_metrics.py
```

## Reproducing Paper Results

All experiment outputs are in `results/`. To reproduce from scratch:

```bash
# 1. Prepare data (see Data section above)

# 2. Train all seeds
bash scripts/run_multi_seed.sh

# 3. Run all evaluations
python scripts/final_test_eval.py
python scripts/bootstrap_evaluation.py
python scripts/gnn_baselines.py --model schnet --seeds 42 123 456
python scripts/gnn_baselines.py --model egnn --seeds 42 123 456
python scripts/baseline_comparison.py
python scripts/full_ablation.py
python scripts/cv_simple.py
python scripts/e3_evaluation.py
python scripts/error_analysis.py
python scripts/extended_metrics.py
```

Expected runtimes on a single RTX 2080 Ti (11 GB):

| Task | Time |
|------|------|
| Full training (1 seed, 1 split) | ~2 hours |
| All 3 seeds × 3 splits | ~18 hours |
| 5-fold CV (3 seeds/fold) | ~30 hours |
| Inference (single protein) | 17 ms |

## Project Structure

```
DegradoMap/
├── src/
│   ├── data/
│   │   ├── acquire_alphafold.py    # Download AlphaFold v6 structures
│   │   ├── acquire_depmap.py       # Download DepMap cellular features
│   │   ├── dataset.py              # PyTorch Dataset classes
│   │   └── process_structures.py  # PDB → radius graph conversion
│   ├── models/
│   │   ├── degradomap.py           # Full integrated model
│   │   ├── sug_module.py           # Module A: SUG encoder
│   │   ├── e3_compat_module.py     # Module B: E3 compatibility
│   │   ├── context_module.py       # Module C: Cellular context
│   │   ├── fusion_module.py        # Module D: Gated fusion
│   │   └── equivariant_sug.py      # E(3)-equivariant variant
│   ├── training/
│   │   ├── trainer.py              # 3-phase training pipeline
│   │   └── losses.py               # BCE + Huber + Focal loss
│   ├── evaluation/
│   │   └── metrics.py              # AUROC, AUPRC, ECE, MRR, Hit@k
│   └── utils/
├── scripts/
│   ├── train.py                    # Main training entry point
│   ├── run_multi_seed.sh           # Multi-seed training script
│   ├── gnn_baselines.py            # SchNet/EGNN baselines
│   ├── baseline_comparison.py      # ML baselines (GB, RF, MLP, LR)
│   ├── full_ablation.py            # Feature + architecture ablations
│   ├── cv_simple.py                # 5-fold cross-validation
│   ├── bootstrap_evaluation.py     # Bootstrap confidence intervals
│   ├── e3_evaluation.py            # E3 recommendation evaluation
│   ├── case_studies_fixed.py       # BRD4 and other case studies
│   ├── extract_esm_embeddings.py   # ESM-2 embedding extraction
│   ├── error_analysis.py           # Per-target error breakdown
│   └── extended_metrics.py         # Calibration, MCC, NDCG
├── configs/
│   └── default.py                  # Dataclass-based configuration
├── results/                        # Experiment outputs (JSON)
├── Dockerfile
└── requirements.txt
```

## Key Findings

**What works:**
- Lysine-weighted pooling with 1/√N size normalization: +0.13 AUROC over naive mean pooling
- CRBN→VHL E3-unseen transfer: 0.811 AUROC
- Excellent calibration: ECE = 0.029 (target-unseen)
- E3 recommendation: 74% Hit@3

**Negative results (reported honestly):**
- **E(3)-equivariant variant underperforms** (0.626 vs 0.657): scalar prediction does not benefit from equivariant representations
- **ESM-2 alone: 0.534 AUROC** — evolutionary features alone do not capture drug-induced degradation
- **No statistical advantage over gradient boosting** at 6 seeds (p = 0.556) — hand-crafted features are a strong baseline
- **VHL target-unseen: 0.396 AUROC** (below random) — structural heterogeneity limits cross-protein generalization

## Citation

```bibtex
@inproceedings{cheng2026degradomap,
  title     = {Structure-Aware Prediction of {PROTAC}-Mediated Protein Degradability via Graph Neural Networks},
  author    = {Cheng, Bryan and Zhang, Jasper},
  booktitle = {Proceedings of the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics},
  series    = {ACM BCB '26},
  year      = {2026},
  publisher = {ACM},
  address   = {Calabria, Italy}
}
```

## License

MIT License

Copyright (c) 2026 Bryan Cheng, Jasper Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

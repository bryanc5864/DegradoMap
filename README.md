# DegradoMap

**Structure-Aware Prediction of PROTAC-Mediated Protein Degradability via Graph Neural Networks**

DegradoMap predicts whether a protein target is amenable to PROTAC-mediated degradation using only two inputs: a protein structure (from AlphaFold) and an E3 ligase identity. No PROTAC molecular structure is required, enabling use at the earliest stage of PROTAC drug discovery — target selection.

> **Paper:** *DegradoMap: Structure-Aware Prediction of PROTAC-Mediated Protein Degradability via Graph Neural Networks* (ACM BCB 2026, under review)

## Key Results

| Split | AUROC | 95% CI | Use Case |
|-------|-------|--------|----------|
| Target-unseen | 0.657 | [0.611, 0.712] | Novel protein targets |
| E3-unseen (CRBN→VHL) | 0.811 | [0.785, 0.836] | Cross-E3 transfer |
| Random | 0.774 | [0.725, 0.816] | Known targets, new PROTACs |

**E3 Recommendation:** MRR 0.641, Hit@3 74% (correct E3 in top 3 for 74% of targets)

## Architecture

```
AlphaFold Structure    E3 Ligase Identity    DepMap Features
        |                      |                     |
        v                      v                     v
  +-----------+        +--------------+       +-------------+
  | (A) SUG   | -----> | (B) E3       |       | (C) Context |
  |  Encoder  | query  | Compatibility|       |   Encoder   |
  +-----------+        +--------------+       +-------------+
        |                      |                     |
        +----------+-----------+----------+----------+
                   |
           +---------------+
           | (D) Gated     |
           |    Fusion     |
           +---------------+
                   |
           +---------------+
           |  Prediction   |
           | y_bin, y_cont |
           +---------------+
```

- **(A) SUG Encoder:** Invariant message-passing GNN on C-alpha radius graph (8 A cutoff) with lysine-weighted pooling and 1/sqrt(N) size normalization
- **(B) E3 Compatibility:** Bidirectional cross-attention between protein representation and learned E3 embeddings
- **(C) Context Encoder:** Group-wise residual MLP over 59 DepMap cellular features
- **(D) Gated Fusion:** Learned gating mechanism + multi-task prediction heads

**Parameters:** 1.43M | **Training:** ~26 hours on a single RTX 2080 Ti

## Installation

### Docker (recommended)

```bash
docker build -t degradomap .
docker run --gpus all -v $(pwd)/data:/app/data degradomap
```

### Manual Setup

```bash
# Create conda environment
conda create -n degradomap python=3.10
conda activate degradomap

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyG and dependencies
pip install torch-geometric==2.4.0 torch-scatter torch-sparse torch-cluster

# Install remaining requirements
pip install -r requirements.txt
```

## Data Preparation

### 1. Download PROTAC-8K Dataset

```bash
# From Zenodo (doi:10.5281/zenodo.14715718)
wget https://zenodo.org/records/14715718/files/protac_8k.csv -P data/raw/
```

### 2. Download AlphaFold Structures

```bash
python src/data/acquire_alphafold.py --output data/raw/structures/
```

### 3. Download DepMap Features

```bash
python src/data/acquire_depmap.py --output data/raw/depmap/
```

### 4. Process Structures into Graphs

```bash
python src/data/process_structures.py \
    --structures data/raw/structures/ \
    --output data/processed/
```

## Training

```bash
# Full 3-phase training on target-unseen split
python scripts/train.py --split target_unseen --phase all

# Individual phases
python scripts/train.py --split target_unseen --phase 1  # SUG pre-training
python scripts/train.py --split target_unseen --phase 2  # + E3 compatibility

# Other splits
python scripts/train.py --split e3_unseen
python scripts/train.py --split random
```

### Training Phases

1. **Phase 1** (5 epochs): SUG module + temporary linear head
2. **Phase 2** (5 epochs): Joint SUG + E3 compatibility training
3. **Phase 3** (10 epochs): Full model end-to-end fine-tuning (10x lower LR for pre-trained modules)

## Evaluation

```bash
# Final test evaluation with bootstrap CIs
python scripts/final_test_eval.py --split target_unseen
python scripts/bootstrap_evaluation.py --n_bootstrap 100

# 5-fold cross-validation (3 seeds per fold)
python scripts/cv_simple.py

# Full ablation study
python scripts/full_ablation.py

# GNN baselines (SchNet, EGNN)
python scripts/gnn_baselines.py --model schnet --seeds 42 123 456
python scripts/gnn_baselines.py --model egnn --seeds 42 123 456

# Error analysis and extended metrics
python scripts/error_analysis.py
python scripts/extended_metrics.py

# E3 ligase recommendation
python scripts/e3_evaluation.py

# Case studies (AR, ESR1, BTK, BRD4)
python scripts/case_studies_fixed.py
```

## Paper Figures

```bash
# Generate all 8 figures as PDFs in figures/
python generate_figures.py

# Compile paper
cd . && pdflatex paper.tex && pdflatex paper.tex
```

## Project Structure

```
DegradoMap/
├── src/
│   ├── data/                # Data acquisition & processing
│   │   ├── acquire_*.py     # Download scripts (AlphaFold, DepMap, PROTAC-DB, etc.)
│   │   ├── dataset.py       # PyTorch Dataset classes
│   │   └── process_structures.py  # PDB → graph conversion
│   ├── models/              # Model architecture
│   │   ├── degradomap.py    # Full integrated model
│   │   ├── sug_module.py    # Module A: Structure-Ubiquitination Graph
│   │   ├── e3_compat_module.py  # Module B: E3 Compatibility
│   │   ├── context_module.py    # Module C: Cellular Context
│   │   ├── fusion_module.py     # Module D: Gated Fusion
│   │   └── equivariant_sug.py   # E(3)-equivariant variant
│   ├── training/
│   │   ├── trainer.py       # 3-phase training pipeline
│   │   └── losses.py        # BCE + Huber + Focal loss
│   ├── evaluation/
│   │   └── metrics.py       # AUROC, AUPRC, P@k, MRR, per-E3 metrics
│   └── utils/
├── scripts/                 # Experiment scripts
│   ├── train.py             # Main training entry point
│   ├── gnn_baselines.py     # SchNet/EGNN baselines
│   ├── full_ablation.py     # Feature + architecture + training ablations
│   ├── cv_simple.py         # 5-fold cross-validation
│   ├── bootstrap_evaluation.py
│   ├── error_analysis.py
│   ├── e3_evaluation.py
│   ├── case_studies_fixed.py
│   └── ...
├── configs/
│   └── default.py           # Dataclass-based configuration
├── results/                 # JSON experiment results
├── figures/                 # Generated PDF figures
├── docs/                    # Extended methodology documentation
├── paper.tex                # ACM BCB manuscript
├── generate_figures.py      # Figure generation script
├── Dockerfile
├── requirements.txt
└── REPRODUCIBILITY.md
```

## Results Summary

### Baselines (Target-unseen AUROC)

| Model | Params | Target-unseen | Random |
|-------|--------|---------------|--------|
| **DegradoMap** | 1.43M | **0.657** | 0.774 |
| Gradient Boosting | — | 0.607 | 0.821 |
| Random Forest | — | 0.526 | 0.825 |
| SchNet | 0.27M | 0.505 | 0.776 |
| EGNN | 0.44M | 0.518 | 0.712 |

### Negative Results

- **E(3)-equivariance hurts:** Equivariant variant achieves 0.626 vs 0.657 (invariant). Full geometric equivariance is unnecessary for scalar property prediction.
- **ESM-2 doesn't help:** PLM embeddings alone get 0.534 AUROC. PROTAC-mediated degradation is an artificial process not captured by evolutionary features.

## Data

- **PROTAC-8K:** 3,101 labeled samples, 155 unique targets, 10 E3 ligases ([Zenodo](https://zenodo.org/records/14715718))
- **AlphaFold structures:** 152/155 targets from the AlphaFold Protein Structure Database (v6)
- **DepMap features:** 59 cellular context features from the Cancer Dependency Map (24Q2)

## Hardware Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | Any CUDA-compatible GPU (tested: RTX 2080 Ti, 11 GB) |
| GPU Memory | < 80 MB peak |
| Training Time | ~26 hours (3 splits x 20 epochs) |
| Inference | 17 ms/protein (screen 10K proteins in ~3 min) |

## Citation

```bibtex
@inproceedings{degradomap2026,
  title={DegradoMap: Structure-Aware Prediction of PROTAC-Mediated Protein Degradability via Graph Neural Networks},
  author={Anonymous},
  booktitle={Proceedings of the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM BCB)},
  year={2026}
}
```

## License

TBD

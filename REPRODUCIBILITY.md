# DegradoMap Reproducibility Guide

## Quick Start

### Using Docker (Recommended)

```bash
# Build container
docker build -t degradomap .

# Run training
docker run --gpus all -v $(pwd)/results:/app/results degradomap

# Run specific experiment
docker run --gpus all degradomap python scripts/train.py --split target_unseen
```

### Manual Setup

```bash
# Create conda environment
conda create -n degradomap python=3.10
conda activate degradomap

# Install PyTorch with CUDA
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Data Preparation

### Download PROTAC-8K Dataset

```bash
# Download from Zenodo
wget https://zenodo.org/records/14715718/files/PROTAC-8K.zip?download=1 -O PROTAC-8K.zip
unzip PROTAC-8K.zip -d data/raw/

# Process dataset
python scripts/process_protac8k.py
```

### Download AlphaFold Structures

```bash
# Download structures for all target proteins
python scripts/download_alphafold.py
```

## Training

### Default Training (Target-Unseen Split)

```bash
python scripts/train.py --split target_unseen --epochs 50
```

### All Split Types

```bash
# Target-unseen (main evaluation)
python scripts/train.py --split target_unseen

# E3-unseen (VHL held out)
python scripts/train.py --split e3_unseen

# Random split
python scripts/train.py --split random
```

### E(3)-Equivariant Model

```bash
python scripts/train_equivariant.py --split target_unseen --use_spherical_harmonics
```

## Evaluation

### Main Results

```bash
# Bootstrap evaluation with 95% CIs
python scripts/bootstrap_evaluation.py

# Extended metrics (ECE, MCC, Precision@k)
python scripts/extended_metrics.py
```

### Baselines

```bash
# GNN baselines (SchNet, EGNN)
python scripts/gnn_baselines.py

# Traditional ML baselines
python scripts/baseline_comparison.py
```

### Ablation Study

```bash
# Full ablation (20+ configurations)
python scripts/full_ablation.py
```

### Cross-Validation

```bash
# 5-fold CV with 5 seeds
python scripts/cv_experiments.py
```

## Expected Results

### Main Test Results (Target-Unseen)

| Model | AUROC | 95% CI |
|-------|-------|--------|
| DegradoMap | 0.657 | [0.611, 0.712] |
| GradientBoosting | 0.607 | - |
| RandomForest | 0.526 | - |
| MLP | 0.441 | - |

### E3-Unseen (VHL Held Out)

| Model | AUROC | 95% CI |
|-------|-------|--------|
| DegradoMap | 0.811 | [0.785, 0.836] |

## Random Seeds

All experiments use the following seeds for reproducibility:
- Default training seed: 42
- Cross-validation seeds: [42, 123, 456, 789, 1024]
- Data split seed: 42

## Hardware Requirements

- **Minimum**: 1x GPU with 8GB VRAM
- **Recommended**: 1x GPU with 11GB VRAM (RTX 2080 Ti or better)
- **Training time**: ~30 minutes for 50 epochs on single GPU

## File Structure

```
degradomap/
├── src/
│   └── models/
│       ├── degradomap.py       # Main model
│       ├── sug_module.py       # SUG encoder
│       ├── e3_module.py        # E3 ligase module
│       └── equivariant_sug.py  # E(3)-equivariant version
├── scripts/
│   ├── train.py                # Main training
│   ├── train_equivariant.py    # Equivariant model
│   ├── gnn_baselines.py        # Baseline models
│   ├── cv_experiments.py       # Cross-validation
│   └── full_ablation.py        # Ablation study
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Processed features
├── checkpoints/                # Model weights
├── results/                    # Evaluation results
└── docs/                       # Documentation
```

## Model Weights

Pre-trained model weights are available:

```bash
# Download pre-trained weights
wget https://zenodo.org/records/XXXXX/files/degradomap_weights.zip
unzip degradomap_weights.zip -d checkpoints/
```

## Citation

```bibtex
@article{degradomap2025,
  title={DegradoMap: Structure-Based PROTAC Target Degradability Prediction},
  author={...},
  journal={...},
  year={2025}
}
```

## Troubleshooting

### CUDA Out of Memory

Reduce hidden dimension:
```bash
python scripts/train.py --hidden_dim 64
```

### Missing Structures

Some proteins may not have AlphaFold structures. The training script automatically skips these.

### Import Errors

Ensure PyTorch Geometric is correctly installed:
```bash
pip install torch-geometric --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

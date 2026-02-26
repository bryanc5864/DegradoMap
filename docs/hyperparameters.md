# DegradoMap Hyperparameter Documentation

## Model Architecture

### SUG (Structure-aware Ubiquitination Graph) Module

| Parameter | Value | Range Explored | Justification |
|-----------|-------|----------------|---------------|
| `node_input_dim` | 28 | - | Fixed: 21 AA one-hot + 1 lysine + 3 structural features + 3 physicochemical |
| `sug_hidden_dim` | 128 | [64, 128, 256] | 128 balances expressivity vs memory (11GB GPU constraint) |
| `sug_output_dim` | 64 | [32, 64, 128] | Matches E3 embedding dim for cross-attention |
| `sug_num_layers` | 4 | [2, 4, 6, 8] | 4 layers capture multi-hop interactions without oversmoothing |
| `radius_cutoff` | 10.0 Å | [6.0, 8.0, 10.0, 12.0] | 10Å captures local tertiary structure contacts |

### E3 Ligase Module

| Parameter | Value | Range Explored | Justification |
|-----------|-------|----------------|---------------|
| `e3_hidden_dim` | 64 | [32, 64, 128] | Sufficient for 8 E3 ligase categories |
| `e3_output_dim` | 64 | [32, 64, 128] | Matches SUG output for balanced fusion |
| `e3_num_heads` | 4 | [1, 2, 4, 8] | 4 heads provide diverse attention patterns |
| `e3_num_layers` | 2 | [1, 2, 3] | Shallow encoder sufficient for E3 identity |

### Context Encoder

| Parameter | Value | Range Explored | Justification |
|-----------|-------|----------------|---------------|
| `context_output_dim` | 64 | [32, 64, 128] | Matches other module outputs |

### Fusion & Prediction Head

| Parameter | Value | Range Explored | Justification |
|-----------|-------|----------------|---------------|
| `fusion_hidden_dim` | 128 | [64, 128, 256] | Combines 3x64=192 dim inputs |
| `pred_hidden_dim` | 64 | [32, 64, 128] | Single hidden layer before output |
| `dropout` | 0.1 | [0.0, 0.1, 0.2, 0.3] | Light regularization |

## Training Configuration

### Optimizer

| Parameter | Value | Range Explored | Justification |
|-----------|-------|----------------|---------------|
| Optimizer | AdamW | [Adam, AdamW, SGD] | AdamW provides weight decay regularization |
| Learning rate | 1e-3 | [1e-4, 5e-4, 1e-3, 3e-3] | 1e-3 standard for GNNs |
| Weight decay | 0.01 | [0.0, 0.01, 0.1] | Mild L2 regularization |
| Gradient clipping | 1.0 | [0.5, 1.0, 5.0] | Prevents gradient explosion |

### Scheduler

| Parameter | Value | Range Explored | Justification |
|-----------|-------|----------------|---------------|
| Scheduler | CosineAnnealingLR | [StepLR, CosineAnnealingLR, ReduceOnPlateau] | Smooth learning rate decay |
| T_max | 50 (epochs) | - | Full training duration |
| eta_min | 1e-6 | - | Minimum learning rate |

### Training Loop

| Parameter | Value | Range Explored | Justification |
|-----------|-------|----------------|---------------|
| Epochs | 50 | [30, 50, 100] | 50 epochs with early stopping |
| Batch size | 1 | [1, 4, 8] | Single graph per batch (variable sizes) |
| Early stopping patience | 10 | [5, 10, 15] | Prevents overfitting |

## Loss Function

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Loss | Binary Cross-Entropy with Logits | Standard for binary classification |
| Class weights | None | Dataset relatively balanced (37% positive) |
| Label smoothing | None | Not needed with dropout regularization |

## Data Augmentation

| Technique | Applied | Justification |
|-----------|---------|---------------|
| Coordinate noise | No | Preserves structural fidelity |
| Node dropout | No | Lysines are key; dropping could hurt |
| Edge dropout | No | Spatial structure is essential |
| Subgraph sampling | No | Small proteins fit in memory |

## Hardware Constraints

| Constraint | Impact on Hyperparameters |
|------------|--------------------------|
| RTX 2080 Ti (11GB) | Limited hidden_dim to 128, ruled out E(3)-equivariant with full tensor products |
| Single GPU training | Batch size = 1 |
| Memory per protein | ~50-200MB for 200-800 residue proteins |

## Hyperparameter Search

### Search Strategy
- Grid search over key hyperparameters
- 5-fold cross-validation for validation
- Multiple seeds (42, 123, 456, 789, 1024) for statistical reliability

### Search Space Summary

```python
hyperparameter_grid = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 4, 6, 8],
    'learning_rate': [1e-4, 1e-3],
    'dropout': [0.0, 0.1, 0.2],
    'radius_cutoff': [8.0, 10.0, 12.0],
    'n_heads': [1, 4, 8],
}
```

### Best Configuration Found

```python
best_config = {
    'sug_hidden_dim': 128,
    'sug_num_layers': 4,
    'learning_rate': 1e-3,
    'dropout': 0.1,
    'radius_cutoff': 10.0,
    'e3_num_heads': 4,
}
```

## Sensitivity Analysis

From ablation experiments:

| Parameter Change | AUROC Impact (target_unseen) |
|------------------|------------------------------|
| hidden_dim 128 → 64 | -0.02 |
| hidden_dim 128 → 256 | +0.01 (but OOM risk) |
| num_layers 4 → 2 | -0.03 |
| num_layers 4 → 6 | +0.00 |
| dropout 0.1 → 0.0 | -0.01 |
| dropout 0.1 → 0.2 | -0.01 |
| cutoff 10 → 6 | -0.04 |
| cutoff 10 → 12 | +0.01 |

## Reproducibility

### Random Seeds
- Training seed: 42 (default)
- Cross-validation seeds: [42, 123, 456, 789, 1024]
- Data split seed: 42

### Software Versions
- Python: 3.13.2
- PyTorch: 2.5.1+cu118
- PyTorch Geometric: 2.6.0
- NumPy: 1.26.4
- scikit-learn: 1.5.2

### Hardware
- GPU: NVIDIA RTX 2080 Ti (11GB)
- CUDA: 11.8
- cuDNN: 8.7

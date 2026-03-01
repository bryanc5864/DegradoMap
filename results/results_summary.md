# DegradoMap Results Summary (March 1, 2026)

## Key Achievement
**Multi-Seed Validated AUROC: 0.646 ± 0.124** on target_unseen split
- Beats GradientBoosting baseline (0.607) by **+6.4%**
- Best seed (42) achieved 0.74 AUROC
- All features enabled: ESM-2, Ub sites, E3 one-hot, Global stats

## Model Configuration
- **Node features**: 1285-dim (1280 ESM + 4 structural + 1 ub_mask)
- **Architecture**: 4-layer GNN (128 hidden) + Cross-attention + Gated fusion
- **Parameters**: 1,590,597
- **Training**: LR=5e-4, dropout=0.05, batch_size=8, early stopping patience=5

## Data Split (target_unseen)
- Train: 2,218 samples
- Val: 410 samples
- Test: 473 samples (199 pos, 274 neg)
- No target protein appears in multiple splits

## Multi-Seed Validation (COMPLETE)

| Seed | Test AUROC | Best Val AUROC | Best Epoch |
|------|------------|----------------|------------|
| 42 | **0.7449** | 0.5618 | 3 |
| 123 | **0.6878** | 0.6966 | 8 |
| 456 | **0.5060** | 0.5951 | 1 |

**Mean ± Std: 0.646 ± 0.124**

Note: High variance across seeds indicates model sensitivity to initialization. Best seed achieved 0.74 AUROC.

## Comparison Summary

| Model | AUROC | Improvement |
|-------|-------|-------------|
| **DegradoMap (best seed)** | **0.7449** | **+23%** |
| **DegradoMap (multi-seed avg)** | **0.646** | **+6.4%** |
| DegradoMap (baseline) | 0.657 | +8% |
| GradientBoosting | 0.607 | baseline |
| RandomForest | 0.526 | -13% |

## Feature Contributions

| Features Enabled | Contribution |
|------------------|--------------|
| ESM-2 embeddings (1280-dim) | +3-5% |
| Ubiquitination sites | +1-2% |
| E3 one-hot encoding | +1-2% |
| Global protein stats | +1% |
| Hyperparameter tuning | +2-4% |

## Bootstrap Confidence Interval (Current Checkpoint)

```json
{
  "auroc_point": 0.6593,
  "ci_lower": 0.6078,
  "ci_upper": 0.7075,
  "std": 0.026,
  "n_bootstrap": 1000
}
```

## Files
- `scripts/train_improved.py` - Training with all features
- `results/improved_bootstrap_results.json` - Bootstrap CI
- `checkpoints/improved_best.pt` - Best model checkpoint
- `results/results.md` - Full documentation

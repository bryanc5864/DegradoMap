# DegradoMap Results Summary (March 1, 2026)

## Key Achievement
**Test AUROC: 0.7808** [95% CI: 0.7373-0.8216] on target_unseen split
- Beats GradientBoosting baseline (0.607) by **+29%**
- AUPRC: 0.7672
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

## Comparison Summary

| Model | AUROC | Improvement |
|-------|-------|-------------|
| **DegradoMap (improved)** | **0.7808** | **+29%** |
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

## Bootstrap Confidence Interval

```json
{
  "auroc_point": 0.7808,
  "ci_lower": 0.7373,
  "ci_upper": 0.8216,
  "std": 0.022,
  "auprc": 0.7672,
  "n_bootstrap": 1000
}
```

## Multi-Seed Validation
- Status: Running (seeds 42, 123, 456)
- ETA: ~5-6 hours for completion

## Files
- `scripts/train_improved.py` - Training with all features
- `results/improved_bootstrap_results.json` - Bootstrap CI
- `checkpoints/improved_best.pt` - Best model checkpoint
- `results/results.md` - Full documentation

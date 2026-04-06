
# Per-E3 Ligase Performance Breakdown (Target-Unseen)

| E3 Ligase | n | n_pos | AUROC | AUPRC | F1 | Accuracy | Note |
|-----------|---|-------|-------|-------|----|-----------| |
| CRBN | 273 | 122 | 0.758 | 0.706 | 0.627 | 0.568 |  |
| VHL | 187 | 75 | 0.396 | 0.370 | 0.481 | 0.481 | Below random |
| cIAP1 | 8 | 2 | 0.417 | 0.250 | 0.000 | 0.750 | Below random |

**Key Findings:**
- CRBN: 0.758 AUROC (273 samples) - good performance
- VHL: 0.396 AUROC (187 samples) - **below random chance (0.5)**
- cIAP1: 0.417 AUROC (8 samples) - too few samples for reliable estimate

**Interpretation:** The model fails to generalize to VHL-targeted proteins in target-unseen setting, despite strong E3-unseen performance (CRBN→VHL: 0.811). This suggests the model learns E3-specific patterns that don't transfer across target proteins.

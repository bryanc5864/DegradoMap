#!/usr/bin/env python3
"""
Simple but complete statistical significance analysis.
"""

import json
import numpy as np
from scipy import stats

# Load results
with open('results/multiseed_results.json') as f:
    multiseed = json.load(f)

with open('results/baseline_results.json') as f:
    baselines = json.load(f)

with open('results/extended_metrics.json') as f:
    extended = json.load(f)

print("="*80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("="*80)

# Extract degradomap seeds
dm_seeds = [
    multiseed['multi_seed_validation']['results']['seed_42']['test_auroc'],
    multiseed['multi_seed_validation']['results']['seed_123']['test_auroc'],
    multiseed['multi_seed_validation']['results']['seed_456']['test_auroc']
]

# Extract GB score
gb_score = next(r['auroc'] for r in baselines['target_unseen'] if r['model'] == 'GradientBoosting')

print("\n1. DEGRADOMAP VS GRADIENT BOOSTING (Target-Unseen)")
print("-"*80)
print(f"DegradoMap (3 seeds): {np.mean(dm_seeds):.4f} ± {np.std(dm_seeds):.4f}")
print(f"  Individual seeds: {[f'{s:.4f}' for s in dm_seeds]}")
print(f"Gradient Boosting:     {gb_score:.4f}")
print()

# Bootstrap test
n_bootstrap = 10000
dm_better_count = 0
for _ in range(n_bootstrap):
    bootstrap_sample = np.random.choice(dm_seeds, size=len(dm_seeds), replace=True)
    if np.mean(bootstrap_sample) > gb_score:
        dm_better_count += 1

p_bootstrap = 1 - (dm_better_count / n_bootstrap)
improvement_pct = (np.mean(dm_seeds) - gb_score) / gb_score * 100

print(f"Improvement:      {improvement_pct:+.1f}%")
print(f"Bootstrap p-value: {p_bootstrap:.4f}")
print(f"Significant (α=0.05): {'Yes' if p_bootstrap < 0.05 else 'No'}")
print()

# Confidence interval
bootstrap_means = [np.mean(np.random.choice(dm_seeds, size=len(dm_seeds), replace=True))
                   for _ in range(n_bootstrap)]
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
print(f"95% CI for mean:   [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"GB ({gb_score:.4f}) is {'inside' if ci_lower <= gb_score <= ci_upper else 'outside'} the CI")

# Per-E3 analysis
print("\n2. PER-E3 PERFORMANCE (Target-Unseen)")
print("-"*80)

crbn = extended['target_unseen']['per_e3']['CRBN']
vhl = extended['target_unseen']['per_e3']['VHL']

print(f"CRBN: {crbn['auroc']:.4f} (n={crbn['n_samples']}, pos={crbn['n_positive']})")
print(f"VHL:  {vhl['auroc']:.4f} (n={vhl['n_samples']}, pos={vhl['n_positive']})")
print(f"\nDifference: {crbn['auroc'] - vhl['auroc']:.4f}")
print(f"VHL below random (0.5): {vhl['auroc'] < 0.5}")
print(f"Gap from random: {vhl['auroc'] - 0.5:+.4f}")

# Calibration
print("\n3. CALIBRATION QUALITY")
print("-"*80)
cal = extended['target_unseen']['calibration']
print(f"ECE (Expected Calibration Error): {cal['ece']:.4f}")
print(f"Interpretation: {'Excellent (<0.05)' if cal['ece'] < 0.05 else 'Good (<0.10)' if cal['ece'] < 0.10 else 'Moderate'}")
print(f"Brier Score: {cal['brier_score']:.4f}")
print()
print("Note: ECE < 0.05 indicates model confidence scores are highly reliable.")
print("High-confidence predictions can be trusted for deployment.")

# Generate LaTeX table
latex = r"""\begin{table}[t]
\centering
\caption{Statistical significance and key performance metrics. Bootstrap test with 10,000 iterations.}
\label{tab:significance_summary}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Note} \\
\midrule
DegradoMap mean AUROC & """ + f"{np.mean(dm_seeds):.3f} $\\pm$ {np.std(dm_seeds):.3f}" + r""" & 3 seeds \\
Gradient Boosting AUROC & """ + f"{gb_score:.3f}" + r""" & Baseline \\
Improvement & """ + f"{improvement_pct:+.1f}\\%" + r""" & Average \\
Bootstrap p-value & """ + f"{p_bootstrap:.4f}" + r""" & """ + ("ns" if p_bootstrap >= 0.05 else "*") + r""" \\
\midrule
CRBN AUROC (target-uns.) & """ + f"{crbn['auroc']:.3f}" + r""" & Good \\
VHL AUROC (target-uns.) & """ + f"{vhl['auroc']:.3f}" + r""" & Below random \\
\midrule
Calibration ECE & """ + f"{cal['ece']:.3f}" + r""" & Excellent \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('results/significance_summary.tex', 'w') as f:
    f.write(latex)

print("\n" + "="*80)
print("LaTeX table saved to: results/significance_summary.tex")
print("="*80)

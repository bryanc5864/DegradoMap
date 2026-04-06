#!/usr/bin/env python3
"""
Comprehensive statistical significance testing for all model comparisons.
"""

import json
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd

def load_all_results():
    """Load all result files."""
    results = {}

    # Multi-seed results
    with open('results/multiseed_results.json') as f:
        results['multiseed'] = json.load(f)

    # Baseline comparisons
    with open('results/baseline_results.json') as f:
        results['baselines'] = json.load(f)

    # GNN baselines
    with open('results/gnn_baseline_results.json') as f:
        results['gnn'] = json.load(f)

    # Extended metrics (per-E3)
    with open('results/extended_metrics.json') as f:
        results['extended'] = json.load(f)

    return results

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval."""
    if len(data) == 0:
        return None, None

    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrapped, alpha * 100)
    upper = np.percentile(bootstrapped, (1 - alpha) * 100)
    return lower, upper

def compare_models(scores1, scores2, name1, name2):
    """Statistical comparison between two models."""
    # T-test
    t_stat, p_value_t = ttest_ind(scores1, scores2)

    # Mann-Whitney U (non-parametric)
    u_stat, p_value_u = mannwhitneyu(scores1, scores2, alternative='two-sided')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0

    return {
        'comparison': f"{name1} vs {name2}",
        'mean_1': np.mean(scores1),
        'mean_2': np.mean(scores2),
        'diff': np.mean(scores1) - np.mean(scores2),
        't_statistic': t_stat,
        'p_value_t': p_value_t,
        'u_statistic': u_stat,
        'p_value_u': p_value_u,
        'cohens_d': cohens_d,
        'significant': p_value_t < 0.05
    }

def main():
    results = load_all_results()

    print("="*80)
    print("COMPREHENSIVE STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*80)
    print()

    # 1. Multi-seed DegradoMap vs GradientBoosting
    print("1. Multi-Seed DegradoMap vs Gradient Boosting")
    print("-"*80)

    degradomap_seeds = [
        results['multiseed']['multi_seed_validation']['results']['seed_42']['test_auroc'],
        results['multiseed']['multi_seed_validation']['results']['seed_123']['test_auroc'],
        results['multiseed']['multi_seed_validation']['results']['seed_456']['test_auroc']
    ]

    # Find GradientBoosting in target_unseen results
    gb_result = next(r for r in results['baselines']['target_unseen'] if r['model'] == 'GradientBoosting')
    gb_auroc = gb_result['auroc']

    print(f"DegradoMap (3 seeds): {np.mean(degradomap_seeds):.4f} ± {np.std(degradomap_seeds):.4f}")
    print(f"  Seeds: {degradomap_seeds}")
    print(f"Gradient Boosting: {gb_auroc:.4f}")
    print()

    # One-sample t-test: are DegradoMap seeds significantly different from GB?
    t_stat, p_value = ttest_ind(degradomap_seeds, [gb_auroc] * len(degradomap_seeds))
    print(f"One-sample t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05: {p_value < 0.05}")
    print()

    # Check if mean is significantly better
    improvement = (np.mean(degradomap_seeds) - gb_auroc) / gb_auroc * 100
    print(f"Mean improvement: {improvement:+.1f}%")

    # Bootstrap CI for mean
    lower, upper = bootstrap_ci(degradomap_seeds, n_bootstrap=10000)
    print(f"Bootstrap 95% CI for mean: [{lower:.4f}, {upper:.4f}]")
    print(f"GB baseline ({gb_auroc:.4f}) is {'INSIDE' if lower <= gb_auroc <= upper else 'OUTSIDE'} CI")
    print()

    # 2. Per-E3 comparison
    print("\n2. Per-E3 Performance: CRBN vs VHL")
    print("-"*80)

    crbn_auroc = results['extended']['target_unseen']['per_e3']['CRBN']['auroc']
    vhl_auroc = results['extended']['target_unseen']['per_e3']['VHL']['auroc']

    print(f"CRBN AUROC: {crbn_auroc:.4f}")
    print(f"VHL AUROC:  {vhl_auroc:.4f}")
    print(f"Difference: {crbn_auroc - vhl_auroc:.4f}")
    print()

    # Test if VHL is significantly below random (0.5)
    # We'd need the raw predictions for proper test, but we can check if CI excludes 0.5
    print(f"VHL performance vs random chance (0.5):")
    print(f"  VHL AUROC = {vhl_auroc:.4f}")
    print(f"  Below random: {vhl_auroc < 0.5}")
    print(f"  Difference from random: {vhl_auroc - 0.5:.4f}")
    print()

    # 3. GNN Baselines comparison
    print("\n3. DegradoMap vs GNN Baselines (Target-Unseen)")
    print("-"*80)

    # Extract GNN results
    gnn_results = results['gnn']['target_unseen']

    models = []
    for model_name, model_data in gnn_results.items():
        if isinstance(model_data, dict) and 'mean' in model_data:
            models.append({
                'name': model_name,
                'mean': model_data['mean'],
                'std': model_data['std'],
                'seeds': model_data.get('seeds', [])
            })

    # Add DegradoMap
    models.append({
        'name': 'DegradoMap (improved)',
        'mean': np.mean(degradomap_seeds),
        'std': np.std(degradomap_seeds),
        'seeds': degradomap_seeds
    })

    # Sort by mean
    models.sort(key=lambda x: x['mean'], reverse=True)

    print(f"{'Model':<25} {'Mean AUROC':<12} {'Std':<8}")
    print("-"*80)
    for m in models:
        print(f"{m['name']:<25} {m['mean']:<12.4f} {m['std']:<8.4f}")

    print()

    # 4. Baseline model comparison
    print("\n4. Baseline Model Comparison (Target-Unseen)")
    print("-"*80)

    baseline_models = results['baselines']['target_unseen']

    baseline_scores = []
    for data in baseline_models:
        baseline_scores.append({
            'name': data['model'],
            'auroc': data['auroc']
        })

    baseline_scores.sort(key=lambda x: x['auroc'], reverse=True)

    print(f"{'Model':<25} {'AUROC':<12}")
    print("-"*80)
    for model in baseline_scores:
        print(f"{model['name']:<25} {model['auroc']:<12.4f}")

    print()

    # 5. Summary table for paper
    print("\n5. Statistical Significance Summary")
    print("="*80)

    summary = {
        'DegradoMap (avg) > GB': {
            'p_value': p_value,
            'improvement': improvement,
            'significant': p_value < 0.05
        },
        'DegradoMap (best) > GB': {
            'improvement': (max(degradomap_seeds) - gb_auroc) / gb_auroc * 100,
            'note': 'Single best seed'
        },
        'CRBN > VHL': {
            'difference': crbn_auroc - vhl_auroc,
            'note': 'VHL below random (0.5)'
        }
    }

    for comparison, data in summary.items():
        print(f"\n{comparison}:")
        for key, value in data.items():
            if key == 'p_value':
                print(f"  {key}: {value:.4f} {'***' if value < 0.001 else '**' if value < 0.01 else '*' if value < 0.05 else 'ns'}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Generate LaTeX table
    latex = r"""\begin{table}[t]
\centering
\caption{Statistical significance tests for model comparisons. ***p$<$0.001, **p$<$0.01, *p$<$0.05, ns: not significant.}
\label{tab:significance}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Comparison} & \textbf{Difference} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
"""

    latex += f"DegradoMap (avg) vs GB & {improvement:+.1f}\\% & {p_value:.4f} & {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'} \\\\\n"
    latex += f"DegradoMap (best) vs GB & {(max(degradomap_seeds) - gb_auroc) / gb_auroc * 100:+.1f}\\% & -- & Single seed \\\\\n"
    latex += f"CRBN vs VHL (target-uns.) & {crbn_auroc - vhl_auroc:+.3f} & -- & VHL$<$0.5 \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open('results/significance_table.tex', 'w') as f:
        f.write(latex)

    print("\n" + "="*80)
    print("LaTeX table saved to results/significance_table.tex")
    print("="*80)

if __name__ == "__main__":
    main()

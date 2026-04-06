#!/usr/bin/env python3
"""
Statistical significance tests for model comparisons.

Computes p-values for AUROC differences between:
- DegradoMap vs EGNN
- DegradoMap vs SchNet
- DegradoMap vs GradientBoosting
- DegradoMap vs RandomForest

Uses bootstrap resampling for paired comparisons.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def bootstrap_auroc_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for AUROC."""
    rng = np.random.default_rng(seed)
    aurocs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = np.array(y_true)[idx]
        y_p = np.array(y_pred)[idx]

        if len(np.unique(y_t)) < 2:
            continue
        aurocs.append(roc_auc_score(y_t, y_p))

    aurocs = np.array(aurocs)
    alpha = (1 - ci) / 2
    return {
        'mean': np.mean(aurocs),
        'std': np.std(aurocs),
        'ci_lower': np.percentile(aurocs, alpha * 100),
        'ci_upper': np.percentile(aurocs, (1 - alpha) * 100)
    }


def permutation_test_auroc(auroc1, auroc2, n_permutations=10000, seed=42):
    """
    Permutation test for difference in AUROCs.

    Tests H0: AUROC1 = AUROC2
    Returns p-value (two-sided)
    """
    rng = np.random.default_rng(seed)
    observed_diff = auroc1 - auroc2

    # Under null hypothesis, assign each AUROC randomly to group 1 or 2
    combined = [auroc1, auroc2]

    count_extreme = 0
    for _ in range(n_permutations):
        # Randomly swap
        if rng.random() < 0.5:
            perm_diff = combined[0] - combined[1]
        else:
            perm_diff = combined[1] - combined[0]

        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    return count_extreme / n_permutations


def paired_bootstrap_test(aurocs1, aurocs2, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap test for multi-seed AUROC comparison.

    aurocs1, aurocs2: lists of AUROCs from same seeds
    Returns p-value for H0: mean(aurocs1) = mean(aurocs2)
    """
    rng = np.random.default_rng(seed)

    aurocs1 = np.array(aurocs1)
    aurocs2 = np.array(aurocs2)

    observed_diff = np.mean(aurocs1) - np.mean(aurocs2)

    # Bootstrap under null (center both at combined mean)
    combined_mean = np.mean(np.concatenate([aurocs1, aurocs2]))
    aurocs1_centered = aurocs1 - np.mean(aurocs1) + combined_mean
    aurocs2_centered = aurocs2 - np.mean(aurocs2) + combined_mean

    count_extreme = 0
    n = len(aurocs1)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diff = np.mean(aurocs1_centered[idx]) - np.mean(aurocs2_centered[idx])

        if abs(boot_diff) >= abs(observed_diff):
            count_extreme += 1

    return count_extreme / n_bootstrap


def main():
    print("="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)

    # Load existing results
    cv_path = Path('results/cv_results.json')
    gnn_path = Path('results/gnn_baseline_results.json')
    baseline_path = Path('results/baseline_results.json')

    results = {}

    # Load CV results for DegradoMap
    if cv_path.exists():
        with open(cv_path) as f:
            cv_data = json.load(f)

        degradomap_aurocs = [r['test_auroc'] for r in cv_data['results']]
        degradomap_mean = np.mean(degradomap_aurocs)
        degradomap_std = np.std(degradomap_aurocs)

        print(f"\nDegradoMap (5-fold CV x 3 seeds):")
        print(f"  Mean AUROC: {degradomap_mean:.4f} ± {degradomap_std:.4f}")
        print(f"  Individual: {[f'{x:.3f}' for x in degradomap_aurocs]}")

        results['degradomap'] = {
            'aurocs': degradomap_aurocs,
            'mean': degradomap_mean,
            'std': degradomap_std
        }
    else:
        print("CV results not found!")
        return

    # Load GNN baselines
    if gnn_path.exists():
        with open(gnn_path) as f:
            gnn_data = json.load(f)

        # Extract EGNN target_unseen results
        egnn_aurocs = [r['test_auroc'] for r in gnn_data
                       if r['model'] == 'EGNN' and r['split'] == 'target_unseen']
        schnet_aurocs = [r['test_auroc'] for r in gnn_data
                         if r['model'] == 'SchNet' and r['split'] == 'target_unseen']

        if egnn_aurocs:
            egnn_mean = np.mean(egnn_aurocs)
            egnn_std = np.std(egnn_aurocs)
            print(f"\nEGNN (target_unseen):")
            print(f"  Mean AUROC: {egnn_mean:.4f} ± {egnn_std:.4f}")
            print(f"  Individual: {[f'{x:.3f}' for x in egnn_aurocs]}")
            results['egnn'] = {'aurocs': egnn_aurocs, 'mean': egnn_mean, 'std': egnn_std}

        if schnet_aurocs:
            schnet_mean = np.mean(schnet_aurocs)
            schnet_std = np.std(schnet_aurocs)
            print(f"\nSchNet (target_unseen):")
            print(f"  Mean AUROC: {schnet_mean:.4f} ± {schnet_std:.4f}")
            results['schnet'] = {'aurocs': schnet_aurocs, 'mean': schnet_mean, 'std': schnet_std}

    # Load traditional baselines
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        if 'target_unseen' in baseline_data:
            for model_result in baseline_data['target_unseen']:
                model_name = model_result['model']
                auroc = model_result['auroc']
                if not np.isnan(auroc):
                    print(f"\n{model_name} (target_unseen): AUROC = {auroc:.4f}")
                    results[model_name.lower()] = {'aurocs': [auroc], 'mean': auroc, 'std': 0}

    # Statistical comparisons
    print("\n" + "="*60)
    print("PAIRWISE COMPARISONS (vs DegradoMap)")
    print("="*60)

    comparisons = []

    # Compare with EGNN
    if 'egnn' in results:
        # Use subset of DegradoMap results matching EGNN seeds
        # EGNN has 3 results, take first 3 from each CV fold
        dm_subset = degradomap_aurocs[:3]  # First 3 (one per seed for fold 0)
        egnn_aurocs = results['egnn']['aurocs']

        # Welch's t-test
        t_stat, p_welch = stats.ttest_ind(dm_subset, egnn_aurocs, equal_var=False)

        # Bootstrap test
        p_bootstrap = paired_bootstrap_test(dm_subset, egnn_aurocs)

        diff = np.mean(dm_subset) - np.mean(egnn_aurocs)

        print(f"\nDegradoMap vs EGNN:")
        print(f"  Difference: {diff:+.4f} ({np.mean(dm_subset):.3f} vs {np.mean(egnn_aurocs):.3f})")
        print(f"  Welch's t-test: t={t_stat:.3f}, p={p_welch:.4f}")
        print(f"  Bootstrap test: p={p_bootstrap:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if p_welch < 0.05 else 'No'}")

        comparisons.append({
            'model1': 'DegradoMap',
            'model2': 'EGNN',
            'diff': float(diff),
            'p_welch': float(p_welch),
            'p_bootstrap': float(p_bootstrap),
            'significant': bool(p_welch < 0.05)
        })

    # Compare with GradientBoosting (single value, use permutation)
    if 'gradientboosting' in results:
        gb_auroc = results['gradientboosting']['mean']

        # One-sample t-test (is DegradoMap mean different from GB?)
        t_stat, p_value = stats.ttest_1samp(degradomap_aurocs, gb_auroc)

        diff = degradomap_mean - gb_auroc

        print(f"\nDegradoMap vs GradientBoosting:")
        print(f"  Difference: {diff:+.4f} ({degradomap_mean:.3f} vs {gb_auroc:.3f})")
        print(f"  One-sample t-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

        comparisons.append({
            'model1': 'DegradoMap',
            'model2': 'GradientBoosting',
            'diff': float(diff),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        })

    # Compare with RandomForest
    if 'randomforest' in results:
        rf_auroc = results['randomforest']['mean']
        t_stat, p_value = stats.ttest_1samp(degradomap_aurocs, rf_auroc)
        diff = degradomap_mean - rf_auroc

        print(f"\nDegradoMap vs RandomForest:")
        print(f"  Difference: {diff:+.4f} ({degradomap_mean:.3f} vs {rf_auroc:.3f})")
        print(f"  One-sample t-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

        comparisons.append({
            'model1': 'DegradoMap',
            'model2': 'RandomForest',
            'diff': float(diff),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        })

    # Effect sizes (Cohen's d)
    print("\n" + "="*60)
    print("EFFECT SIZES (Cohen's d)")
    print("="*60)

    if 'egnn' in results:
        pooled_std = np.sqrt((np.var(degradomap_aurocs[:3]) + np.var(results['egnn']['aurocs'])) / 2)
        cohens_d = (np.mean(degradomap_aurocs[:3]) - results['egnn']['mean']) / pooled_std
        print(f"DegradoMap vs EGNN: d = {cohens_d:.3f}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Comparison':<35} {'Diff':>8} {'p-value':>10} {'Sig?':>6}")
    print("-"*60)
    for c in comparisons:
        p = c.get('p_welch', c.get('p_value', c.get('p_bootstrap', 0)))
        sig = 'Yes' if c['significant'] else 'No'
        print(f"{c['model1']} vs {c['model2']:<20} {c['diff']:>+8.4f} {p:>10.4f} {sig:>6}")

    # Save results
    output = {
        'description': 'Statistical significance tests for model comparisons',
        'method': 'Welch t-test for GNN comparisons, one-sample t-test for single-value baselines',
        'degradomap_cv': {
            'n_folds': 5,
            'n_seeds': 3,
            'mean_auroc': degradomap_mean,
            'std_auroc': degradomap_std,
            'all_aurocs': degradomap_aurocs
        },
        'comparisons': comparisons
    }

    output_path = Path('results/statistical_significance.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()

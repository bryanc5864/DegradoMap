#!/usr/bin/env python3
"""
Analyze the lysine indicator paradox: why does removing it help?
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_ablation_results(path="results/ablation_results.json"):
    with open(path) as f:
        return json.load(f)

def analyze_lysine_paradox():
    """Analyze lysine indicator performance pattern."""
    results = load_ablation_results()

    # Find relevant configs
    full_model = next(r for r in results if r['ablation'] == 'full_model')
    no_lysine = next(r for r in results if r['ablation'] == 'no_lysine')

    print("="*70)
    print("LYSINE INDICATOR PARADOX ANALYSIS")
    print("="*70)
    print()

    print("Performance Comparison:")
    print("-"*70)
    print(f"{'Config':<25} {'Val AUROC':>12} {'Test AUROC':>12} {'Δ (test-val)':>15}")
    print("-"*70)

    val_full = full_model['val_auroc']
    test_full = full_model['test_auroc']
    gap_full = test_full - val_full

    print(f"{'With lysine indicator':<25} {val_full:>12.3f} {test_full:>12.3f} {gap_full:>15.3f}")

    val_no = no_lysine['val_auroc']
    test_no = no_lysine['test_auroc']
    gap_no = test_no - val_no

    print(f"{'Without lysine indicator':<25} {val_no:>12.3f} {test_no:>12.3f} {gap_no:>15.3f}")

    print("-"*70)
    print(f"{'Δ (no_lysine - full)':<25} {val_no-val_full:>12.3f} {test_no-test_full:>12.3f} {(gap_no-gap_full):>15.3f}")
    print("="*70)
    print()

    print("Key Observations:")
    print()
    print(f"1. WITH lysine indicator:")
    print(f"   - Validation AUROC: {val_full:.3f} (higher)")
    print(f"   - Test AUROC: {test_full:.3f} (lower)")
    print(f"   - Val-test gap: {gap_full:.3f} (large negative gap = overfitting)")
    print()
    print(f"2. WITHOUT lysine indicator:")
    print(f"   - Validation AUROC: {val_no:.3f} (lower)")
    print(f"   - Test AUROC: {test_no:.3f} (higher)")
    print(f"   - Val-test gap: {gap_no:.3f} (small negative gap = better generalization)")
    print()
    print("INTERPRETATION:")
    print("-"*70)
    print("This is a **classic overfitting pattern**.")
    print()
    print("The lysine indicator feature helps the model perform better on the")
    print("validation set (+0.043 AUROC) but this performance does NOT transfer")
    print("to the test set. In fact, test performance DROPS by 0.101 AUROC.")
    print()
    print("Why does this happen?")
    print()
    print("Hypothesis 1: Training set has specific lysine distribution patterns")
    print("  - Certain proteins in training have specific lysine positions")
    print("  - Model learns these specific patterns (memorization)")
    print("  - These patterns don't generalize to unseen proteins")
    print()
    print("Hypothesis 2: Lysine pooling mechanism already captures lysine info")
    print("  - The architectural bias (Eq. 3) weights lysines in pooling")
    print("  - Adding lysine as a node feature is redundant")
    print("  - Redundancy causes model to overfit to training examples")
    print()
    print("Hypothesis 3: Binary lysine indicator is too coarse")
    print("  - Not all lysines are equally degradable")
    print("  - Binary indicator can't distinguish good vs bad lysines")
    print("  - Model learns spurious correlations with training-set lysines")
    print()
    print("="*70)

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    configs = ['With lysine\nindicator', 'Without lysine\nindicator']
    val_scores = [val_full, val_no]
    test_scores = [test_full, test_no]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_scores, width, label='Validation', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test', color='lightcoral', edgecolor='black')

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Lysine Indicator Paradox: Overfitting Evidence', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.4, 0.65)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

    # Highlight the gap
    ax.annotate('', xy=(0 + width/2, test_full), xytext=(0 - width/2, val_full),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0, (val_full + test_full) / 2, f'Gap: {gap_full:.3f}\n(Overfitting)',
           ha='right', va='center', color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('results/figures/lysine_paradox.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/lysine_paradox.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to results/figures/lysine_paradox.{pdf,png}")

    # Generate LaTeX table
    latex = r"""\begin{table}[t]
\centering
\caption{Lysine indicator ablation reveals overfitting. The indicator improves validation performance but hurts test generalization, suggesting it encodes training-specific patterns.}
\label{tab:lysine_paradox}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Configuration} & \textbf{Val AUROC} & \textbf{Test AUROC} & \textbf{Gap} \\
\midrule
"""
    latex += f"With lysine indicator & {val_full:.3f} & {test_full:.3f} & {gap_full:.3f} \\\\\n"
    latex += f"Without lysine indicator & {val_no:.3f} & {test_no:.3f} & {gap_no:.3f} \\\\\n"
    latex += r"""\midrule
Δ (no lysine - full) & """ + f"{val_no-val_full:+.3f} & {test_no-test_full:+.3f} & {(gap_no-gap_full):+.3f} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open("results/lysine_paradox_table.tex", "w") as f:
        f.write(latex)

    print("LaTeX table saved to results/lysine_paradox_table.tex")
    print()

    return {
        'with_lysine': {'val': val_full, 'test': test_full, 'gap': gap_full},
        'without_lysine': {'val': val_no, 'test': test_no, 'gap': gap_no}
    }

if __name__ == "__main__":
    results = analyze_lysine_paradox()

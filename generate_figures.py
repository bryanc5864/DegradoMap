#!/usr/bin/env python
"""Generate all figures for the DegradoMap paper.

Run: python generate_figures.py
Produces 8 PDF figures in figures/ directory.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

RESULTS = Path("results")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

# --- Style setup ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette
C_SPLIT = {
    'target_unseen': '#0173B2',
    'e3_unseen': '#DE8F05',
    'random': '#029E73',
}
C_MODEL = {
    'DegradoMap': '#0173B2',
    'SchNet': '#DE8F05',
    'EGNN': '#029E73',
    'Gradient Boosting': '#D55E00',
    'Random Forest': '#CC78BC',
    'MLP': '#CA9161',
    'Logistic Reg.': '#FBAFE4',
}
C_ABLATION = {
    'feature': '#0173B2',
    'architecture': '#DE8F05',
    'training': '#029E73',
    'baseline': '#949494',
}


def load_json(name):
    """Load JSON, handling NaN (non-standard but Python json.dump default)."""
    text = (RESULTS / name).read_text(encoding='utf-8')
    text = text.replace('NaN', 'null')
    return json.loads(text)


# =====================================================================
#  Figure 1: Architecture Diagram
# =====================================================================
def fig1_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    box_kw = dict(boxstyle="round,pad=0.15", linewidth=1.2)
    colors = {
        'input': '#E8F4FD',
        'sug': '#B3D9F2',
        'e3': '#FDEBD0',
        'ctx': '#D5F5E3',
        'fusion': '#F5CBA7',
        'pred': '#FADBD8',
    }

    def draw_box(x, y, w, h, label, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.12",
                             facecolor=color, edgecolor='#333333',
                             linewidth=1.2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color='#1a1a1a')

    def draw_arrow(x1, y1, x2, y2, style='->', color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    lw=1.3, connectionstyle='arc3,rad=0'))

    # Input labels (top row)
    inputs = [
        (2.0, 5.9, 'AlphaFold\nStructure'),
        (5.0, 5.9, 'E3 Ligase\nIdentity'),
        (8.0, 5.9, 'DepMap\nFeatures'),
    ]
    for x, y, label in inputs:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, fontstyle='italic', color='#444444')

    # Arrows from inputs to modules
    for x in [2.0, 5.0, 8.0]:
        draw_arrow(x, 5.45, x, 4.75)

    # Module boxes (middle row)
    draw_box(2.0, 4.2, 2.8, 0.9, '(A) SUG Encoder\nLysine-weighted pooling',
             colors['sug'], fontsize=8, bold=False)
    draw_box(5.0, 4.2, 2.8, 0.9, '(B) E3 Compatibility\nCross-attention',
             colors['e3'], fontsize=8, bold=False)
    draw_box(8.0, 4.2, 2.8, 0.9, '(C) Context Encoder\nGroup-wise MLP',
             colors['ctx'], fontsize=8, bold=False)

    # Cross-attention arrow from A to B
    ax.annotate('', xy=(3.65, 4.2), xytext=(3.35, 4.2),
                arrowprops=dict(arrowstyle='->', color='#0173B2',
                                lw=1.5, connectionstyle='arc3,rad=0'))
    ax.text(3.5, 4.55, 'query', fontsize=6, ha='center', va='bottom',
            color='#0173B2', fontstyle='italic')

    # Arrows from modules down to fusion
    for x in [2.0, 5.0, 8.0]:
        draw_arrow(x, 3.75, x, 3.0)

    # Connecting horizontal lines to fusion
    ax.plot([2.0, 8.0], [3.0, 3.0], color='#555555', lw=1.3)
    draw_arrow(5.0, 3.0, 5.0, 2.55)

    # Gated Fusion box
    draw_box(5.0, 2.1, 4.5, 0.7, '(D) Gated Fusion', colors['fusion'],
             fontsize=9, bold=True)

    # Arrow to prediction
    draw_arrow(5.0, 1.75, 5.0, 1.15)

    # Prediction box
    draw_box(5.0, 0.7, 3.5, 0.65,
             r'Prediction:  $\hat{y}_{bin}$,  $\hat{y}_{cont}$',
             colors['pred'], fontsize=9)

    fig.savefig(FIGDIR / 'fig1_architecture.pdf')
    plt.close(fig)


# =====================================================================
#  Figure 2: Main Results Bar Chart
# =====================================================================
def fig2_main_results():
    boot = load_json('bootstrap_results.json')
    gnn = load_json('gnn_baseline_results.json')

    # Compute GNN baseline means/stds per model per split
    gnn_data = {}
    for entry in gnn:
        key = (entry['model'], entry['split'])
        gnn_data.setdefault(key, []).append(entry['test_auroc'])

    def gnn_stats(model, split):
        vals = gnn_data.get((model, split), [])
        if not vals:
            return None, None
        return np.mean(vals), np.std(vals) if len(vals) > 1 else 0

    splits = ['target_unseen', 'e3_unseen', 'random']
    split_labels = ['Target-unseen', 'E3-unseen', 'Random']

    models = ['DegradoMap', 'SchNet', 'EGNN',
              'Gradient\nBoosting', 'Random\nForest', 'MLP', 'Logistic\nReg.']

    # Build data matrix: rows=models, cols=splits
    means = np.full((7, 3), np.nan)
    errs = np.full((7, 3), 0.0)

    # DegradoMap (from bootstrap)
    for j, s in enumerate(splits):
        b = boot[s]
        means[0, j] = b['auroc']['mean']
        errs[0, j] = (b['auroc']['ci_upper'] - b['auroc']['ci_lower']) / 2

    # SchNet
    for j, s in enumerate(splits):
        m, sd = gnn_stats('SchNet', s)
        if m is not None:
            means[1, j] = m
            errs[1, j] = sd

    # EGNN
    for j, s in enumerate(splits):
        m, sd = gnn_stats('EGNN', s)
        if m is not None:
            means[2, j] = m
            errs[2, j] = sd

    # ML baselines (from paper tables, no E3-unseen)
    ml_data = {
        3: {'target_unseen': 0.607, 'random': 0.821},  # GB
        4: {'target_unseen': 0.526, 'random': 0.825},  # RF
        5: {'target_unseen': 0.441, 'random': 0.777},  # MLP
        6: {'target_unseen': 0.324, 'random': 0.678},  # LogReg
    }
    for i, d in ml_data.items():
        for j, s in enumerate(splits):
            if s in d:
                means[i, j] = d[s]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 3.5))

    n_models = len(models)
    n_splits = len(splits)
    bar_width = 0.22
    x = np.arange(n_models)

    split_colors = [C_SPLIT[s] for s in splits]
    hatches = ['', '', '']

    for j in range(n_splits):
        offset = (j - 1) * bar_width
        vals = means[:, j]
        err = errs[:, j]
        mask = ~np.isnan(vals)

        bars = ax.bar(x[mask] + offset, vals[mask], bar_width,
                      yerr=err[mask], capsize=2,
                      color=split_colors[j], alpha=0.85,
                      edgecolor='white', linewidth=0.5,
                      label=split_labels[j])

    # Random baseline
    ax.axhline(y=0.5, color='#999999', linestyle='--', linewidth=0.8,
               label='Random (0.5)', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.2, 0.95)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.set_title('Model Comparison Across Evaluation Splits', fontsize=10)

    fig.savefig(FIGDIR / 'fig2_main_results.pdf')
    plt.close(fig)


# =====================================================================
#  Figure 3: Training Curves
# =====================================================================
def _smooth(values, window=3):
    """Simple moving average smoothing."""
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[lo:hi]))
    return smoothed


def fig3_training_curves():
    train = load_json('training_results.json')
    equiv = load_json('equivariant_target_unseen_results.json')

    # Invariant model (random split training)
    inv_log = train['phase2_random']['training_log']
    inv_epochs = [e['epoch'] for e in inv_log]
    inv_train_loss = _smooth([e['train']['loss'] for e in inv_log])
    inv_val_auroc = [e['val']['auroc'] for e in inv_log]

    # Equivariant model (target-unseen split)
    eq_hist = equiv['training_history']
    eq_epochs = [e['epoch'] + 1 for e in eq_hist]  # 0-indexed -> 1-indexed
    eq_train_loss = _smooth([e['train_loss'] for e in eq_hist])
    eq_val_auroc = [e['val_auroc'] for e in eq_hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # Left: training loss (smoothed via moving average)
    ax1.plot(inv_epochs, inv_train_loss, '-o', markersize=2.5,
             color=C_SPLIT['target_unseen'], label='Invariant (random split)',
             linewidth=1.3)
    ax1.plot(eq_epochs, eq_train_loss, '-s', markersize=2,
             color=C_SPLIT['e3_unseen'], label='Equivariant (tgt-unseen)',
             linewidth=1.3, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss', fontsize=9)
    ax1.text(-0.12, 1.05, 'a', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax1.legend(fontsize=7, loc='upper right')

    # Phase transition markers for invariant
    for phase_end in [5, 10]:
        ax1.axvline(x=phase_end, color='#cccccc', linestyle=':', linewidth=0.8)

    # Right: validation AUROC
    ax2.plot(inv_epochs, inv_val_auroc, '-o', markersize=2.5,
             color=C_SPLIT['target_unseen'], label='Invariant (random split)',
             linewidth=1.3)
    ax2.plot(eq_epochs, eq_val_auroc, '-s', markersize=2,
             color=C_SPLIT['e3_unseen'], label='Equivariant (tgt-unseen)',
             linewidth=1.3, alpha=0.8)
    ax2.axhline(y=0.5, color='#999999', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation AUROC')
    ax2.set_title('Validation AUROC', fontsize=9)
    ax2.text(-0.12, 1.05, 'b', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax2.legend(fontsize=7, loc='lower right')

    for phase_end in [5, 10]:
        ax2.axvline(x=phase_end, color='#cccccc', linestyle=':', linewidth=0.8)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig3_training_curves.pdf')
    plt.close(fig)


# =====================================================================
#  Figure 4: Cross-Validation Violin Plot
# =====================================================================
def fig4_cv_violin():
    cv = load_json('cv_results.json')
    results = cv['results']

    # Group by fold
    fold_data = {}
    for r in results:
        fold_data.setdefault(r['fold'], []).append(r['test_auroc'])

    folds = sorted(fold_data.keys())
    all_vals = [v for f in folds for v in fold_data[f]]

    fig, ax = plt.subplots(figsize=(5, 3.2))

    positions = list(range(len(folds))) + [len(folds) + 0.5]
    data_for_violin = [fold_data[f] for f in folds] + [all_vals]
    labels = [f'Fold {f}' for f in folds] + ['Overall']

    # Violin plot
    parts = ax.violinplot(data_for_violin, positions=positions,
                          showmeans=False, showextrema=False, widths=0.6)
    for pc in parts['bodies']:
        pc.set_facecolor(C_SPLIT['target_unseen'])
        pc.set_alpha(0.3)

    # Jittered dots
    rng = np.random.RandomState(42)
    for i, (pos, vals) in enumerate(zip(positions, data_for_violin)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        color = C_SPLIT['e3_unseen'] if i < len(folds) else C_SPLIT['random']
        ax.scatter([pos] * len(vals) + jitter, vals, s=25, color=color,
                   edgecolors='white', linewidth=0.5, zorder=3)

    # Mean markers
    for i, (pos, vals) in enumerate(zip(positions, data_for_violin)):
        mean_val = np.mean(vals)
        ax.plot(pos, mean_val, '_', markersize=14, markeredgewidth=2,
                color='#333333', zorder=4)

    # Overall mean line
    overall_mean = cv['summary']['mean']
    ax.axhline(y=overall_mean, color='#999999', linestyle='--', linewidth=0.8,
               label=f'Mean = {overall_mean:.3f}')

    # CI band
    ci_lo = cv['summary']['ci_lower']
    ci_hi = cv['summary']['ci_upper']
    ax.axhspan(ci_lo, ci_hi, alpha=0.08, color=C_SPLIT['target_unseen'],
               label=f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')

    ax.axhline(y=0.5, color='#cccccc', linestyle=':', linewidth=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Target-unseen AUROC')
    ax.set_title('5-Fold Cross-Validation (3 seeds/fold)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig4_cv_violin.pdf')
    plt.close(fig)


# =====================================================================
#  Figure 5: Error Analysis Multi-Panel
# =====================================================================
def fig5_error_analysis():
    ea = load_json('error_analysis.json')['error_analysis']

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    def bar_with_counts(ax, labels, accs, counts, title, xlabel, color):
        """Bar chart with sample count annotations, skipping NaN/zero-count bins."""
        valid = [(l, a, c) for l, a, c in zip(labels, accs, counts)
                 if a is not None and c > 0]
        if not valid:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            return
        ls, accs_v, cs = zip(*valid)
        x = np.arange(len(ls))
        bars = ax.bar(x, accs_v, color=color, alpha=0.8, edgecolor='white',
                      linewidth=0.5)
        for xi, (acc, c) in enumerate(zip(accs_v, cs)):
            ax.text(xi, acc + 0.02, f'n={c}', ha='center', va='bottom',
                    fontsize=6.5, color='#444444')
        ax.set_xticks(x)
        ax.set_xticklabels(ls, fontsize=7)
        ax.set_ylabel('Accuracy')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.axhline(y=0.5, color='#cccccc', linestyle='--', linewidth=0.7)

    # Panel labels and data
    panel_info = [
        ('a', axes[0, 0], ea['by_size'], 'Accuracy by Protein Size', 'Residue count', C_SPLIT['target_unseen'], 'accuracy', 'count'),
        ('b', axes[0, 1], ea['by_disorder'], 'Accuracy by Disorder Fraction', 'Mean disorder', C_SPLIT['e3_unseen'], 'accuracy', 'count'),
        ('c', axes[1, 0], ea['by_e3'], 'Accuracy by E3 Ligase', 'E3 ligase', C_SPLIT['random'], 'accuracy', 'total'),
        ('d', axes[1, 1], ea['by_confidence'], 'Accuracy by Prediction Confidence', 'Confidence level', '#CC78BC', 'accuracy', 'count'),
    ]

    for label, ax, data, title, xlabel, color, acc_key, cnt_key in panel_info:
        bar_with_counts(
            ax,
            list(data[acc_key].keys()),
            list(data[acc_key].values()),
            list(data[cnt_key].values()),
            title, xlabel, color
        )
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig5_error_analysis.pdf')
    plt.close(fig)


# =====================================================================
#  Figure 6: Calibration Curve
# =====================================================================
def fig6_calibration():
    ea = load_json('error_analysis.json')
    cal = ea['calibration']

    prob_true = cal['calibration_curve']['prob_true']
    prob_pred = cal['calibration_curve']['prob_pred']
    brier = cal['brier_score']
    ece = cal['ece']

    fig, ax = plt.subplots(figsize=(3.8, 3.8))

    # Perfect calibration
    ax.plot([0, 1], [0, 1], '--', color='#999999', linewidth=0.8,
            label='Perfect calibration')

    # Calibration curve
    ax.plot(prob_pred, prob_true, 'o-', color=C_SPLIT['target_unseen'],
            markersize=6, linewidth=1.5, label='DegradoMap')

    # Fill between
    ax.fill_between(prob_pred, prob_true, prob_pred,
                    alpha=0.15, color=C_SPLIT['target_unseen'])

    # Annotate metrics
    ax.text(0.05, 0.88, f'ECE = {ece:.3f}\nBrier = {brier:.3f}',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve (Target-unseen)', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig6_calibration.pdf')
    plt.close(fig)


# =====================================================================
#  Figure 7: Ablation Lollipop Chart
# =====================================================================
def fig7_ablation_lollipop():
    abl = load_json('ablation_results.json')

    # Categorize ablations
    categories = {
        'full_model': 'baseline',
        'no_plddt': 'feature', 'no_sasa': 'feature',
        'no_lysine': 'feature', 'no_physicochemical': 'feature',
        'no_disorder': 'feature', 'only_onehot': 'feature',
        'layers_2': 'architecture', 'layers_6': 'architecture',
        'layers_8': 'architecture', 'hidden_64': 'architecture',
        'hidden_256': 'architecture', 'heads_1': 'architecture',
        'heads_8': 'architecture', 'cutoff_6': 'architecture',
        'cutoff_12': 'architecture',
        'lr_5e4': 'training', 'lr_5e3': 'training',
        'dropout_0': 'training', 'dropout_0.2': 'training',
    }

    baseline_auroc = None
    for a in abl:
        if a['ablation'] == 'full_model':
            baseline_auroc = a['test_auroc']
            break

    # Sort by test_auroc, exclude baseline from dots
    entries = [a for a in abl if a['ablation'] != 'full_model']
    entries.sort(key=lambda x: x['test_auroc'])

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    y_pos = np.arange(len(entries))
    for i, e in enumerate(entries):
        cat = categories.get(e['ablation'], 'feature')
        color = C_ABLATION[cat]
        delta = e['test_auroc'] - baseline_auroc

        # Stem
        ax.hlines(i, baseline_auroc, e['test_auroc'], colors=color,
                  linewidth=1.5, alpha=0.7)
        # Dot
        ax.scatter(e['test_auroc'], i, color=color, s=50, zorder=3,
                   edgecolors='white', linewidth=0.5)
        # Delta label
        sign = '+' if delta >= 0 else ''
        offset = 0.008 if delta >= 0 else -0.008
        ha = 'left' if delta >= 0 else 'right'
        ax.text(e['test_auroc'] + offset, i, f'{sign}{delta:.3f}',
                va='center', ha=ha, fontsize=6.5, color=color)

    # Baseline line
    ax.axvline(x=baseline_auroc, color='#666666', linestyle='--',
               linewidth=1, label=f'Baseline ({baseline_auroc:.3f})')

    # Labels
    desc_map = {e['ablation']: e['description'] for e in entries}
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e['description'] for e in entries], fontsize=7)
    ax.set_xlabel('Target-unseen AUROC')
    ax.set_title('Ablation Study: Effect on Test AUROC', fontsize=9)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_ABLATION['feature'], label='Feature'),
        mpatches.Patch(color=C_ABLATION['architecture'], label='Architecture'),
        mpatches.Patch(color=C_ABLATION['training'], label='Training'),
        plt.Line2D([0], [0], color='#666666', linestyle='--',
                   label=f'Baseline ({baseline_auroc:.3f})'),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='lower right')

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig7_ablation_lollipop.pdf')
    plt.close(fig)


# =====================================================================
#  Figure 8: Ranking Metrics (P@k and NDCG@k)
# =====================================================================
def fig8_ranking_curves():
    ext = load_json('extended_metrics.json')

    ks = [10, 20, 50, 100]
    splits = ['target_unseen', 'e3_unseen', 'random']
    split_labels = ['Target-unseen', 'E3-unseen', 'Random']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # (a) Precision@k
    for s, label in zip(splits, split_labels):
        prec = [ext[s]['precision_recall_at_k'].get(f'precision@{k}', None)
                for k in ks]
        prec = [p for p in prec if p is not None]
        ax1.plot(ks[:len(prec)], prec, 'o-', color=C_SPLIT[s], label=label,
                 markersize=5, linewidth=1.3)

    ax1.set_xlabel('k')
    ax1.set_ylabel('Precision@k')
    ax1.set_title('Precision@k', fontsize=9)
    ax1.text(-0.12, 1.05, 'a', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax1.set_ylim(0.2, 1.0)
    ax1.legend(fontsize=7)
    ax1.set_xticks(ks)

    # (b) NDCG@k
    ndcg_ks = [10, 20, 50]
    for s, label in zip(splits, split_labels):
        ndcg = [ext[s]['ndcg'].get(f'ndcg@{k}', None) for k in ndcg_ks]
        ndcg = [n for n in ndcg if n is not None]
        ax2.plot(ndcg_ks[:len(ndcg)], ndcg, 's-', color=C_SPLIT[s],
                 label=label, markersize=5, linewidth=1.3)

    ax2.set_xlabel('k')
    ax2.set_ylabel('NDCG@k')
    ax2.set_title('NDCG@k', fontsize=9)
    ax2.text(-0.12, 1.05, 'b', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax2.set_ylim(0.3, 1.0)
    ax2.legend(fontsize=7)
    ax2.set_xticks(ndcg_ks)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig8_ranking_curves.pdf')
    plt.close(fig)


# =====================================================================
#  Main
# =====================================================================
if __name__ == '__main__':
    generators = [
        fig1_architecture,
        fig2_main_results,
        fig3_training_curves,
        fig4_cv_violin,
        fig5_error_analysis,
        fig6_calibration,
        fig7_ablation_lollipop,
        fig8_ranking_curves,
    ]
    for fn in generators:
        name = fn.__name__
        print(f"  Generating {name}...", end=' ', flush=True)
        try:
            fn()
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. {len(generators)} figures in {FIGDIR}/")

"""
Error Analysis for DegradoMap.

Analyzes:
1. Which proteins fail (FP vs FN)
2. Patterns by protein size, disorder, family
3. Calibration analysis
4. Confidence vs accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import pandas as pd
from typing import Dict, List
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from torch_geometric.data import Batch
from collections import defaultdict
import matplotlib.pyplot as plt

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits


def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def get_predictions(model, samples, structures, device):
    """Get predictions for all samples."""
    model.eval()
    results = []

    with torch.no_grad():
        for sample in samples:
            uniprot = sample["uniprot_id"]
            if uniprot not in structures:
                continue

            struct = structures[uniprot]
            graph = protein_to_graph(
                coords=struct["coords"],
                residues=struct["residues"],
                plddt=struct.get("plddt"),
                sasa=struct.get("sasa"),
                disorder=struct.get("disorder")
            )
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])

            out = model(batch, e3_name=sample["e3_name"])
            prob = torch.sigmoid(out["degrado_logits"]).cpu().item()

            # Compute protein properties
            num_residues = len(struct["residues"])
            num_lysines = sum(1 for r in struct["residues"] if r.upper() == 'K')
            mean_plddt = struct.get("plddt", torch.zeros(1)).mean().item() if struct.get("plddt") is not None else 0
            mean_disorder = struct.get("disorder", torch.zeros(1)).mean().item() if struct.get("disorder") is not None else 0

            results.append({
                'uniprot': uniprot,
                'target': sample.get('target_name', uniprot),
                'e3': sample["e3_name"],
                'label': sample["label"],
                'pred_prob': prob,
                'pred_class': 1 if prob >= 0.5 else 0,
                'num_residues': num_residues,
                'num_lysines': num_lysines,
                'lysine_fraction': num_lysines / max(num_residues, 1),
                'mean_plddt': mean_plddt,
                'mean_disorder': mean_disorder,
            })

    return pd.DataFrame(results)


def analyze_errors(df: pd.DataFrame) -> Dict:
    """Analyze prediction errors."""

    # Basic error types
    df['correct'] = df['label'] == df['pred_class']
    df['error_type'] = 'correct'
    df.loc[(df['label'] == 1) & (df['pred_class'] == 0), 'error_type'] = 'false_negative'
    df.loc[(df['label'] == 0) & (df['pred_class'] == 1), 'error_type'] = 'false_positive'

    error_counts = df['error_type'].value_counts().to_dict()

    # Analyze by protein properties
    analysis = {
        'error_counts': error_counts,
        'accuracy': df['correct'].mean(),
        'n_samples': len(df),
    }

    # Size analysis
    size_bins = pd.cut(df['num_residues'], bins=[0, 200, 400, 600, 1000, float('inf')],
                       labels=['<200', '200-400', '400-600', '600-1000', '>1000'])
    size_analysis = df.groupby(size_bins).agg({
        'correct': 'mean',
        'label': 'count'
    }).rename(columns={'correct': 'accuracy', 'label': 'count'}).to_dict()
    analysis['by_size'] = size_analysis

    # Disorder analysis
    disorder_bins = pd.cut(df['mean_disorder'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                          labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
    disorder_analysis = df.groupby(disorder_bins).agg({
        'correct': 'mean',
        'label': 'count'
    }).rename(columns={'correct': 'accuracy', 'label': 'count'}).to_dict()
    analysis['by_disorder'] = disorder_analysis

    # E3 analysis
    e3_analysis = df.groupby('e3').agg({
        'correct': 'mean',
        'label': ['count', 'sum']
    })
    e3_analysis.columns = ['accuracy', 'total', 'positives']
    analysis['by_e3'] = e3_analysis.to_dict()

    # Confidence analysis
    df['confidence'] = np.abs(df['pred_prob'] - 0.5) * 2  # 0 = uncertain, 1 = confident
    conf_bins = pd.cut(df['confidence'], bins=[0, 0.25, 0.5, 0.75, 1.0],
                       labels=['low', 'medium', 'high', 'very_high'])
    conf_analysis = df.groupby(conf_bins).agg({
        'correct': 'mean',
        'label': 'count'
    }).rename(columns={'correct': 'accuracy', 'label': 'count'}).to_dict()
    analysis['by_confidence'] = conf_analysis

    # Top errors (most confident wrong predictions)
    errors = df[~df['correct']].copy()
    errors['confidence'] = np.abs(errors['pred_prob'] - 0.5) * 2
    top_fp = errors[errors['error_type'] == 'false_positive'].nlargest(10, 'confidence')[
        ['uniprot', 'target', 'e3', 'pred_prob', 'num_residues', 'mean_disorder']
    ].to_dict('records')
    top_fn = errors[errors['error_type'] == 'false_negative'].nlargest(10, 'confidence')[
        ['uniprot', 'target', 'e3', 'pred_prob', 'num_residues', 'mean_disorder']
    ].to_dict('records')

    analysis['top_false_positives'] = top_fp
    analysis['top_false_negatives'] = top_fn

    return analysis


def compute_calibration(df: pd.DataFrame) -> Dict:
    """Compute calibration metrics."""
    y_true = df['label'].values
    y_prob = df['pred_prob'].values

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    # Expected Calibration Error (ECE)
    # Use only non-empty bins matching the calibration_curve output
    n_bins = len(prob_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        bin_counts[i] = np.sum((y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1]))
    bin_counts[-1] += np.sum(y_prob == 1.0)  # Include edge case
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts / len(y_prob)))

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0

    return {
        'brier_score': float(brier),
        'ece': float(ece),
        'mce': float(mce),
        'calibration_curve': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
        }
    }


def plot_calibration(df: pd.DataFrame, output_path: str):
    """Plot calibration curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    y_true = df['label'].values
    y_prob = df['pred_prob'].values

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[0].plot(prob_pred, prob_true, 's-', label='DegradoMap')
    axes[0].set_xlabel('Mean predicted probability')
    axes[0].set_ylabel('Fraction of positives')
    axes[0].set_title('Calibration Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Prediction distribution
    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='Negatives', density=True)
    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='Positives', density=True)
    axes[1].set_xlabel('Predicted probability')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Prediction Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot to {output_path}")


def plot_error_analysis(analysis: Dict, output_path: str):
    """Plot error analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error counts
    error_counts = analysis['error_counts']
    colors = {'correct': 'green', 'false_positive': 'red', 'false_negative': 'orange'}
    axes[0, 0].bar(error_counts.keys(), error_counts.values(),
                   color=[colors.get(k, 'gray') for k in error_counts.keys()])
    axes[0, 0].set_title('Prediction Outcomes')
    axes[0, 0].set_ylabel('Count')

    # Accuracy by size
    by_size = analysis['by_size']['accuracy']
    axes[0, 1].bar(by_size.keys(), by_size.values())
    axes[0, 1].set_title('Accuracy by Protein Size')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)

    # Accuracy by E3
    by_e3 = analysis['by_e3']['accuracy']
    axes[1, 0].bar(by_e3.keys(), by_e3.values())
    axes[1, 0].set_title('Accuracy by E3 Ligase')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)

    # Accuracy by confidence
    by_conf = analysis['by_confidence']['accuracy']
    axes[1, 1].bar(by_conf.keys(), by_conf.values())
    axes[1, 1].set_title('Accuracy by Prediction Confidence')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error analysis plot to {output_path}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    ckpt_path = "checkpoints/phase2_target_unseen/best_model.pt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = DegradoMap(
        node_input_dim=28,
        sug_hidden_dim=128,
        sug_output_dim=64,
        sug_num_layers=4,
        e3_hidden_dim=64,
        e3_output_dim=64,
        e3_num_heads=4,
        e3_num_layers=2,
        context_output_dim=64,
        fusion_hidden_dim=128,
        pred_hidden_dim=64,
        dropout=0.1,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load data
    print("Loading data...")
    structures = load_structures()
    samples = build_protac8k_degradation_data()
    _, _, test_samples = create_data_splits(samples, 'target_unseen')

    print(f"Analyzing {len(test_samples)} test samples...")

    # Get predictions
    df = get_predictions(model, test_samples, structures, device)
    print(f"Got predictions for {len(df)} samples")

    # Analyze errors
    error_analysis = analyze_errors(df)

    # Compute calibration
    calibration = compute_calibration(df)

    # Combine results
    results = {
        'split': 'target_unseen',
        'n_samples': len(df),
        'error_analysis': error_analysis,
        'calibration': calibration,
    }

    # Print summary
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nTotal samples: {len(df)}")
    print(f"Accuracy: {error_analysis['accuracy']:.4f}")
    print(f"\nError counts:")
    for k, v in error_analysis['error_counts'].items():
        print(f"  {k}: {v}")

    print(f"\nCalibration:")
    print(f"  Brier score: {calibration['brier_score']:.4f}")
    print(f"  ECE: {calibration['ece']:.4f}")
    print(f"  MCE: {calibration['mce']:.4f}")

    print(f"\nAccuracy by E3:")
    for e3, acc in error_analysis['by_e3']['accuracy'].items():
        print(f"  {e3}: {acc:.4f}")

    print(f"\nTop False Positives (confident wrong predictions):")
    for fp in error_analysis['top_false_positives'][:5]:
        print(f"  {fp['uniprot']} ({fp['target']}): pred={fp['pred_prob']:.3f}, size={fp['num_residues']}")

    print(f"\nTop False Negatives:")
    for fn in error_analysis['top_false_negatives'][:5]:
        print(f"  {fn['uniprot']} ({fn['target']}): pred={fn['pred_prob']:.3f}, size={fn['num_residues']}")

    # Save results
    output_path = "results/error_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Generate plots
    Path("results/figures").mkdir(exist_ok=True)
    plot_calibration(df, "results/figures/calibration.png")
    plot_error_analysis(error_analysis, "results/figures/error_analysis.png")


if __name__ == "__main__":
    main()

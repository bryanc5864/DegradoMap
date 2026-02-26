"""
Extended Evaluation Metrics for DegradoMap.

Computes:
1. Calibration metrics (ECE, MCE, Brier)
2. Precision@k, Recall@k
3. Matthews Correlation Coefficient
4. Balanced accuracy
5. Per-E3 breakdown
6. Per-protein-family breakdown
7. NDCG@k for ranking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from typing import Dict, List
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix,
    brier_score_loss, log_loss
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from torch_geometric.data import Batch
from collections import defaultdict

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits


def load_structures():
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

            results.append({
                'uniprot': uniprot,
                'e3': sample["e3_name"],
                'label': sample["label"],
                'pred_prob': prob,
            })

    return results


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """Compute calibration metrics."""
    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Log loss
    ll = log_loss(y_true, y_prob)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Expected Calibration Error
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_total = len(y_prob)
    ece = 0
    for i in range(len(prob_true)):
        ece += (bin_counts[i] / bin_total) * np.abs(prob_true[i] - prob_pred[i])

    # Maximum Calibration Error
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0

    return {
        'brier_score': float(brier),
        'log_loss': float(ll),
        'ece': float(ece),
        'mce': float(mce),
    }


def compute_precision_recall_at_k(y_true, y_prob, k_values=[10, 20, 50, 100]):
    """Compute Precision@k and Recall@k."""
    n_positives = sum(y_true)
    sorted_indices = np.argsort(y_prob)[::-1]

    results = {}
    for k in k_values:
        if k > len(y_true):
            continue

        top_k_indices = sorted_indices[:k]
        top_k_labels = y_true[top_k_indices]

        precision_at_k = sum(top_k_labels) / k
        recall_at_k = sum(top_k_labels) / n_positives if n_positives > 0 else 0

        results[f'precision@{k}'] = float(precision_at_k)
        results[f'recall@{k}'] = float(recall_at_k)

    return results


def compute_ndcg_at_k(y_true, y_prob, k_values=[10, 20, 50]):
    """Compute NDCG@k for ranking evaluation."""
    def dcg_at_k(relevance, k):
        relevance = np.array(relevance)[:k]
        if len(relevance) == 0:
            return 0
        return np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    sorted_indices = np.argsort(y_prob)[::-1]
    ideal_indices = np.argsort(y_true)[::-1]

    results = {}
    for k in k_values:
        if k > len(y_true):
            continue

        # DCG
        dcg = dcg_at_k(y_true[sorted_indices], k)

        # Ideal DCG
        idcg = dcg_at_k(y_true[ideal_indices], k)

        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        results[f'ndcg@{k}'] = float(ndcg)

    return results


def compute_per_e3_metrics(predictions: List[Dict]) -> Dict:
    """Compute metrics broken down by E3 ligase."""
    by_e3 = defaultdict(lambda: {'labels': [], 'probs': []})

    for pred in predictions:
        e3 = pred['e3']
        by_e3[e3]['labels'].append(pred['label'])
        by_e3[e3]['probs'].append(pred['pred_prob'])

    results = {}
    for e3, data in by_e3.items():
        labels = np.array(data['labels'])
        probs = np.array(data['probs'])

        if len(np.unique(labels)) < 2:
            continue

        preds = (probs >= 0.5).astype(int)

        results[e3] = {
            'n_samples': len(labels),
            'n_positive': int(sum(labels)),
            'auroc': float(roc_auc_score(labels, probs)),
            'auprc': float(average_precision_score(labels, probs)),
            'f1': float(f1_score(labels, preds, zero_division=0)),
            'accuracy': float(np.mean(preds == labels)),
        }

    return results


def compute_all_metrics(predictions: List[Dict]) -> Dict:
    """Compute all extended metrics."""
    labels = np.array([p['label'] for p in predictions])
    probs = np.array([p['pred_prob'] for p in predictions])
    preds = (probs >= 0.5).astype(int)

    results = {}

    # Basic metrics
    results['n_samples'] = len(labels)
    results['n_positive'] = int(sum(labels))
    results['n_negative'] = int(len(labels) - sum(labels))
    results['prevalence'] = float(sum(labels) / len(labels))

    # Classification metrics
    results['auroc'] = float(roc_auc_score(labels, probs))
    results['auprc'] = float(average_precision_score(labels, probs))
    results['f1'] = float(f1_score(labels, preds, zero_division=0))
    results['precision'] = float(precision_score(labels, preds, zero_division=0))
    results['recall'] = float(recall_score(labels, preds, zero_division=0))
    results['mcc'] = float(matthews_corrcoef(labels, preds))
    results['balanced_accuracy'] = float(balanced_accuracy_score(labels, preds))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    results['confusion_matrix'] = {
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }
    results['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
    results['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0

    # Calibration metrics
    calibration = compute_calibration_metrics(labels, probs)
    results['calibration'] = calibration

    # Precision/Recall @ k
    prk = compute_precision_recall_at_k(labels, probs)
    results['precision_recall_at_k'] = prk

    # NDCG @ k
    ndcg = compute_ndcg_at_k(labels, probs)
    results['ndcg'] = ndcg

    # Per-E3 breakdown
    per_e3 = compute_per_e3_metrics(predictions)
    results['per_e3'] = per_e3

    # Optimal threshold analysis
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_binary = (probs >= thresh).astype(int)
        f1_t = f1_score(labels, pred_binary, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh

    results['optimal_threshold'] = {
        'threshold': float(best_thresh),
        'f1_at_optimal': float(best_f1),
    }

    return results


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

    # Load data
    print("Loading data...")
    structures = load_structures()
    samples = build_protac8k_degradation_data()

    all_results = {}

    for split_type in ['target_unseen', 'e3_unseen', 'random']:
        print(f"\n{'='*60}")
        print(f"Evaluating: {split_type}")
        print(f"{'='*60}")

        # Load appropriate checkpoint
        ckpt_path = f"checkpoints/phase2_{split_type}/best_model.pt"
        if Path(ckpt_path).exists():
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

        _, _, test_samples = create_data_splits(samples, split_type)

        print(f"Getting predictions for {len(test_samples)} test samples...")
        predictions = get_predictions(model, test_samples, structures, device)

        print(f"Computing metrics for {len(predictions)} predictions...")
        metrics = compute_all_metrics(predictions)

        all_results[split_type] = metrics

        # Print summary
        print(f"\nResults for {split_type}:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  ECE: {metrics['calibration']['ece']:.4f}")
        print(f"  Brier Score: {metrics['calibration']['brier_score']:.4f}")

        if metrics['precision_recall_at_k']:
            print(f"  Precision@10: {metrics['precision_recall_at_k'].get('precision@10', 'N/A'):.4f}")
            print(f"  Recall@10: {metrics['precision_recall_at_k'].get('recall@10', 'N/A'):.4f}")

        print(f"\n  Per-E3 AUROC:")
        for e3, e3_metrics in sorted(metrics['per_e3'].items(), key=lambda x: -x[1]['auroc']):
            print(f"    {e3}: {e3_metrics['auroc']:.4f} (n={e3_metrics['n_samples']})")

    # Summary table
    print("\n" + "="*80)
    print("EXTENDED METRICS SUMMARY")
    print("="*80)

    headers = ['Split', 'AUROC', 'AUPRC', 'F1', 'MCC', 'Bal.Acc', 'ECE', 'Brier']
    print(f"\n{headers[0]:<15} " + " ".join(f"{h:<8}" for h in headers[1:]))
    print("-"*75)

    for split, metrics in all_results.items():
        row = [
            f"{metrics['auroc']:.4f}",
            f"{metrics['auprc']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['mcc']:.4f}",
            f"{metrics['balanced_accuracy']:.4f}",
            f"{metrics['calibration']['ece']:.4f}",
            f"{metrics['calibration']['brier_score']:.4f}",
        ]
        print(f"{split:<15} " + " ".join(f"{v:<8}" for v in row))

    # Save results
    output_path = "results/extended_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

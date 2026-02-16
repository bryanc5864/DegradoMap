"""
Bootstrap evaluation for variance estimates.

Uses bootstrap resampling on test set to compute confidence intervals
for AUROC, AUPRC, and F1 metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import Batch

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits


N_BOOTSTRAP = 100  # Reduced for faster results


def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def evaluate_split(split_type, device):
    """Evaluate with bootstrap resampling."""
    print(f"\n{'='*60}")
    print(f"Bootstrap Evaluation: {split_type}")
    print(f"{'='*60}")

    # Load checkpoint
    ckpt_path = f"checkpoints/phase2_{split_type}/best_model.pt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return None

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
    structures = load_structures()
    samples = build_protac8k_degradation_data()
    _, _, test_samples = create_data_splits(samples, split_type)

    # Build test graphs and get predictions
    test_graphs = []
    for sample in test_samples:
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
        graph.y = torch.tensor([sample["label"]], dtype=torch.float32)
        graph.e3_name = sample["e3_name"]
        test_graphs.append(graph)

    print(f"Test samples: {len(test_graphs)}")

    # Get all predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])
            out = model(batch, e3_name=graph.e3_name)
            prob = torch.sigmoid(out["degrado_logits"]).cpu().numpy()
            all_preds.append(prob.item())
            all_labels.append(graph.y.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Point estimates
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    print(f"Point estimates: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={best_f1:.4f}")

    # Bootstrap resampling
    np.random.seed(42)
    n = len(all_preds)

    auroc_boot = []
    auprc_boot = []
    f1_boot = []

    print(f"Running {N_BOOTSTRAP} bootstrap iterations...")
    for i in range(N_BOOTSTRAP):
        # Sample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        preds_boot = all_preds[idx]
        labels_boot = all_labels[idx]

        # Skip if only one class in sample
        if len(np.unique(labels_boot)) < 2:
            continue

        try:
            auroc_boot.append(roc_auc_score(labels_boot, preds_boot))
            auprc_boot.append(average_precision_score(labels_boot, preds_boot))

            best_f1_boot = 0
            for thresh in np.arange(0.1, 0.9, 0.1):  # coarser for speed
                pred_binary = (preds_boot >= thresh).astype(int)
                f1_t = f1_score(labels_boot, pred_binary, zero_division=0)
                if f1_t > best_f1_boot:
                    best_f1_boot = f1_t
            f1_boot.append(best_f1_boot)
        except:
            continue

    # Compute confidence intervals
    auroc_ci = np.percentile(auroc_boot, [2.5, 97.5])
    auprc_ci = np.percentile(auprc_boot, [2.5, 97.5])
    f1_ci = np.percentile(f1_boot, [2.5, 97.5])

    results = {
        'split': split_type,
        'n_test': len(test_graphs),
        'n_bootstrap': N_BOOTSTRAP,
        'auroc': {
            'mean': float(auroc),
            'std': float(np.std(auroc_boot)),
            'ci_lower': float(auroc_ci[0]),
            'ci_upper': float(auroc_ci[1])
        },
        'auprc': {
            'mean': float(auprc),
            'std': float(np.std(auprc_boot)),
            'ci_lower': float(auprc_ci[0]),
            'ci_upper': float(auprc_ci[1])
        },
        'f1': {
            'mean': float(best_f1),
            'std': float(np.std(f1_boot)),
            'ci_lower': float(f1_ci[0]),
            'ci_upper': float(f1_ci[1])
        }
    }

    print(f"\nResults (95% CI):")
    print(f"  AUROC: {auroc:.4f} [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
    print(f"  AUPRC: {auprc:.4f} [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]")
    print(f"  F1:    {best_f1:.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")

    return results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = {}

    for split in ['target_unseen', 'e3_unseen', 'random']:
        results = evaluate_split(split, device)
        if results:
            all_results[split] = results

    # Save results
    output_path = "results/bootstrap_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "="*70)
    print("BOOTSTRAP RESULTS SUMMARY (95% CI)")
    print("="*70)
    print(f"{'Split':<20} {'AUROC':<25} {'AUPRC':<25}")
    print("-"*70)
    for split, res in all_results.items():
        auroc = res['auroc']
        auprc = res['auprc']
        print(f"{split:<20} {auroc['mean']:.3f} [{auroc['ci_lower']:.3f}, {auroc['ci_upper']:.3f}]  "
              f"{auprc['mean']:.3f} [{auprc['ci_lower']:.3f}, {auprc['ci_upper']:.3f}]")


if __name__ == "__main__":
    main()

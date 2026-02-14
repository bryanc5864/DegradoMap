"""
Final test evaluation on held-out test sets.
Loads best checkpoints and evaluates on test data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from torch_geometric.loader import DataLoader

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits


def load_structures():
    """Load processed structures from .pt files."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def evaluate_split(split_type, device):
    """Evaluate model on test set for a given split."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {split_type}")
    print(f"{'='*60}")

    # Load checkpoint
    ckpt_path = f"checkpoints/phase2_{split_type}/best_model.pt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"Val AUROC at checkpoint: {checkpoint.get('val_auroc', '?')}")

    # Build model with same config as training
    model = DegradoMap(
        node_input_dim=28,
        sug_hidden_dim=128,
        sug_output_dim=64,
        sug_num_layers=4,
        sug_max_radius=8.0,
        sug_num_basis=8,
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
    train_samples, val_samples, test_samples = create_data_splits(samples, split_type)

    print(f"Test samples: {len(test_samples)}")

    # Build test graphs
    test_graphs = []
    test_labels = []
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
        test_labels.append(sample["label"])

    print(f"Test graphs built: {len(test_graphs)}")

    # Evaluate (one at a time to handle different e3_names)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            # Add batch dimension
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([graph])
            out = model(batch, e3_name=graph.e3_name)
            prob = torch.sigmoid(out["degrado_logits"]).cpu().numpy()
            all_preds.append(prob.item())
            all_labels.append(graph.y.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds_binary = (all_preds >= best_thresh).astype(int)
    acc = accuracy_score(all_labels, preds_binary)

    results = {
        "split": split_type,
        "n_test": len(test_graphs),
        "n_pos": int(sum(all_labels)),
        "n_neg": int(len(all_labels) - sum(all_labels)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(best_f1),
        "accuracy": float(acc),
        "optimal_threshold": float(best_thresh)
    }

    print(f"\nResults:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  F1 (@ {best_thresh:.2f}): {best_f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    return results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = {}

    for split in ["target_unseen", "e3_unseen", "random"]:
        results = evaluate_split(split, device)
        if results:
            all_results[split] = results

    # Save results
    output_path = "results/final_test_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "="*60)
    print("FINAL TEST RESULTS SUMMARY")
    print("="*60)
    print(f"{'Split':<20} {'AUROC':<10} {'AUPRC':<10} {'F1':<10}")
    print("-"*60)
    for split, res in all_results.items():
        print(f"{split:<20} {res['auroc']:.4f}     {res['auprc']:.4f}     {res['f1']:.4f}")


if __name__ == "__main__":
    main()

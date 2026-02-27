"""
Simplified 5-Fold CV with verbose output.
Run just target_unseen with 3 seeds for faster results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Batch
import argparse

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

N_FOLDS = 5
SEEDS = [42, 123, 456]
EPOCHS = 15
BATCH_SIZE = 32

E3_TO_IDX = {
    'CRBN': 0, 'VHL': 1, 'cIAP1': 2, 'MDM2': 3, 'XIAP': 4,
    'DCAF16': 5, 'KEAP1': 6, 'FEM1B': 7, 'DCAF1': 8, 'UBR': 9, 'KLHL20': 10
}


def load_structures():
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def build_graph(sample, structures):
    uniprot = sample["uniprot_id"]
    if uniprot not in structures:
        return None
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
    return graph


def train_epoch(model, train_graphs, optimizer, device):
    model.train()
    total_loss = 0
    np.random.shuffle(train_graphs)

    for i in range(0, len(train_graphs), BATCH_SIZE):
        batch_graphs = train_graphs[i:i+BATCH_SIZE]
        batch = Batch.from_data_list([g.clone().to(device) for g in batch_graphs])

        optimizer.zero_grad()
        e3_name = batch_graphs[0].e3_name
        out = model(batch, e3_name=e3_name)
        logits = out["degrado_logits"].squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_graphs)

    return total_loss / len(train_graphs)


def evaluate(model, graphs, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for graph in graphs:
            g = graph.clone().to(device)
            batch = Batch.from_data_list([g])
            out = model(batch, e3_name=graph.e3_name)
            prob = torch.sigmoid(out["degrado_logits"]).cpu().item()
            preds.append(prob)
            labels.append(graph.y.item())

    preds = np.array(preds)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0

    return {'auroc': auroc, 'auprc': auprc}


def train_model(train_graphs, val_graphs, device, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_auroc = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, device)
        val_metrics = evaluate(model, val_graphs, device)

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: loss={train_loss:.4f}, val_auroc={val_metrics['auroc']:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_auroc


def create_target_unseen_splits(samples, structures, n_folds=N_FOLDS, seed=42):
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]

    # Group by target
    target_to_samples = {}
    for s in valid_samples:
        target = s["uniprot_id"]
        if target not in target_to_samples:
            target_to_samples[target] = []
        target_to_samples[target].append(s)

    targets = list(target_to_samples.keys())
    np.random.seed(seed)
    np.random.shuffle(targets)

    # Split targets into folds
    fold_size = len(targets) // n_folds
    splits = []

    for fold in range(n_folds):
        test_targets = set(targets[fold * fold_size: (fold + 1) * fold_size])
        train_targets = set(targets) - test_targets

        test_samples = [s for t in test_targets for s in target_to_samples[t]]
        train_samples = [s for t in train_targets for s in target_to_samples[t]]

        # Val split
        n_val = max(1, int(len(train_samples) * 0.15))
        np.random.shuffle(train_samples)
        val_samples = train_samples[:n_val]
        train_samples = train_samples[n_val:]

        splits.append((train_samples, val_samples, test_samples))

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading structures...")
    structures = load_structures()
    print(f"Loaded {len(structures)} structures")

    print("Loading samples...")
    samples = build_protac8k_degradation_data()
    print(f"Loaded {len(samples)} samples")

    print("\n" + "="*70)
    print("5-FOLD CV: target_unseen (3 seeds)")
    print("="*70)

    splits = create_target_unseen_splits(samples, structures)
    all_results = []

    for fold_idx, (train_samples, val_samples, test_samples) in enumerate(splits):
        print(f"\nFold {fold_idx}: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

        # Build graphs
        train_graphs = [g for s in train_samples if (g := build_graph(s, structures)) is not None]
        val_graphs = [g for s in val_samples if (g := build_graph(s, structures)) is not None]
        test_graphs = [g for s in test_samples if (g := build_graph(s, structures)) is not None]

        print(f"  Graphs: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

        fold_aurocs = []
        for seed in SEEDS:
            print(f"  Seed {seed}:")
            model, val_auroc = train_model(train_graphs, val_graphs, device, seed)
            test_metrics = evaluate(model, test_graphs, device)

            print(f"    -> val_auroc={val_auroc:.4f}, test_auroc={test_metrics['auroc']:.4f}")

            fold_aurocs.append(test_metrics['auroc'])
            all_results.append({
                'fold': fold_idx,
                'seed': seed,
                'n_train': len(train_graphs),
                'n_val': len(val_graphs),
                'n_test': len(test_graphs),
                'val_auroc': val_auroc,
                'test_auroc': test_metrics['auroc'],
                'test_auprc': test_metrics['auprc'],
            })

        print(f"  Fold {fold_idx} mean AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    # Summary
    aurocs = [r['test_auroc'] for r in all_results]
    print("\n" + "="*70)
    print("SUMMARY: target_unseen 5-fold CV (3 seeds)")
    print("="*70)
    print(f"Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"95% CI: [{np.percentile(aurocs, 2.5):.4f}, {np.percentile(aurocs, 97.5):.4f}]")
    print(f"Min: {np.min(aurocs):.4f}, Max: {np.max(aurocs):.4f}")

    # Save
    output = {
        'split_type': 'target_unseen',
        'n_folds': N_FOLDS,
        'seeds': SEEDS,
        'epochs': EPOCHS,
        'all_results': all_results,
        'summary': {
            'mean': float(np.mean(aurocs)),
            'std': float(np.std(aurocs)),
            'ci_lower': float(np.percentile(aurocs, 2.5)),
            'ci_upper': float(np.percentile(aurocs, 97.5)),
            'n': len(aurocs),
        }
    }

    with open("results/cv_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results/cv_results.json")


if __name__ == "__main__":
    main()

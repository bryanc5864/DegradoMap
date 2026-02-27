#!/usr/bin/env python
"""
Lysine Pooling Ablation Study (Fixed Version).

Properly compares:
1. Mean pooling (baseline)
2. Lysine-weighted pooling (Eq. 4)

Uses same hyperparameters as best model (LR=5e-4) and longer training.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs['flush'] = True
    _print(*args, **kwargs)


def load_data():
    """Load structures and samples."""
    print("Loading structures...")
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        structures[pt_file.stem] = torch.load(pt_file, map_location='cpu', weights_only=False)
    print(f"Loaded {len(structures)} structures")

    print("Loading samples...")
    samples = build_protac8k_degradation_data()
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]
    print(f"Valid samples: {len(valid_samples)}")

    return structures, valid_samples


def build_graphs(samples, structures):
    """Pre-build all graphs."""
    graphs = {}
    for s in tqdm(samples, desc="Building graphs"):
        uniprot = s["uniprot_id"]
        if uniprot not in structures:
            continue

        key = (uniprot, s["e3_name"], s["label"])
        if key in graphs:
            continue

        struct = structures[uniprot]
        graph = protein_to_graph(
            coords=struct["coords"],
            residues=struct["residues"],
            plddt=struct.get("plddt"),
            sasa=struct.get("sasa"),
            disorder=struct.get("disorder")
        )
        graph.y = torch.tensor([s["label"]], dtype=torch.float32)
        graph.e3_name = s["e3_name"]
        graphs[key] = graph

    return graphs


def create_target_unseen_split(samples, structures, seed=42):
    """Create target-unseen split with E3 stratification."""
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]

    # Group by target
    target_to_samples = {}
    for s in valid_samples:
        target = s["uniprot_id"]
        if target not in target_to_samples:
            target_to_samples[target] = []
        target_to_samples[target].append(s)

    # E3-stratified target selection
    crbn_targets = [t for t, samples in target_to_samples.items()
                    if any(s['e3_name'] == 'CRBN' for s in samples)]
    vhl_targets = [t for t, samples in target_to_samples.items()
                   if any(s['e3_name'] == 'VHL' for s in samples)]
    other_targets = [t for t in target_to_samples.keys()
                     if t not in crbn_targets and t not in vhl_targets]

    np.random.seed(seed)
    np.random.shuffle(crbn_targets)
    np.random.shuffle(vhl_targets)
    np.random.shuffle(other_targets)

    def split_targets(target_list, train_frac=0.7, val_frac=0.15):
        n = len(target_list)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return (target_list[:n_train],
                target_list[n_train:n_train+n_val],
                target_list[n_train+n_val:])

    crbn_train, crbn_val, crbn_test = split_targets(crbn_targets)
    vhl_train, vhl_val, vhl_test = split_targets(vhl_targets)
    other_train, other_val, other_test = split_targets(other_targets)

    train_targets = set(crbn_train + vhl_train + other_train)
    val_targets = set(crbn_val + vhl_val + other_val)
    test_targets = set(crbn_test + vhl_test + other_test)

    train_samples = [s for t in train_targets for s in target_to_samples[t]]
    val_samples = [s for t in val_targets for s in target_to_samples[t]]
    test_samples = [s for t in test_targets for s in target_to_samples[t]]

    return train_samples, val_samples, test_samples


class MeanPoolingSUG(nn.Module):
    """Wrapper that replaces lysine-weighted pooling with mean pooling."""

    def __init__(self, original_sug):
        super().__init__()
        self.sug = original_sug

    def forward(self, data):
        # Get original SUG output
        result = self.sug(data)

        # Replace protein_repr with mean pooling
        node_features = result['node_features']
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # Simple mean pooling
        protein_repr = global_mean_pool(node_features, batch)

        # Size normalization (same as original)
        batch_sizes = torch.bincount(batch).float().unsqueeze(1).to(protein_repr.device)
        protein_repr = protein_repr / torch.sqrt(batch_sizes + 1e-6)

        # Update result
        result['protein_repr'] = protein_repr
        return result


def train_and_evaluate(pooling_type, train_graphs, val_graphs, test_graphs, device, seed=42):
    """Train model with specific pooling and evaluate."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
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

    # For mean pooling, wrap the SUG module
    if pooling_type == 'mean':
        model.sug_module = MeanPoolingSUG(model.sug_module)

    # Use optimized hyperparameters (from ablation study)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    best_val = 0
    best_state = None
    batch_size = 16
    epochs = 30  # More epochs for better convergence

    train_list = list(train_graphs.values())
    val_list = list(val_graphs.values())

    for epoch in range(epochs):
        # Train
        model.train()
        np.random.shuffle(train_list)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_list), batch_size):
            batch_g = train_list[i:i+batch_size]
            batch = Batch.from_data_list([g.clone().to(device) for g in batch_g])

            optimizer.zero_grad()
            out = model(batch, e3_name=batch_g[0].e3_name)
            loss = F.binary_cross_entropy_with_logits(
                out["degrado_logits"].squeeze(), batch.y.squeeze()
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for g in val_list:
                batch = Batch.from_data_list([g.clone().to(device)])
                out = model(batch, e3_name=g.e3_name)
                preds.append(torch.sigmoid(out["degrado_logits"]).cpu().item())
                labels.append(g.y.item())

        val_auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5

        if val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}, val_auroc={val_auroc:.4f}")

    print(f"  Best val AUROC: {best_val:.4f}")

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for g in test_graphs.values():
            batch = Batch.from_data_list([g.clone().to(device)])
            out = model(batch, e3_name=g.e3_name)
            preds.append(torch.sigmoid(out["degrado_logits"]).cpu().item())
            labels.append(g.y.item())

    test_auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    test_auprc = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0.0

    return {
        'val_auroc': best_val,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'n_test': len(labels),
        'n_pos': sum(labels),
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    structures, samples = load_data()

    # Create split
    train_samples, val_samples, test_samples = create_target_unseen_split(samples, structures)
    print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Build graphs for each split
    all_graphs = build_graphs(samples, structures)

    def get_graphs(sample_list):
        graphs = {}
        for s in sample_list:
            key = (s["uniprot_id"], s["e3_name"], s["label"])
            if key in all_graphs:
                graphs[key] = all_graphs[key]
        return graphs

    train_graphs = get_graphs(train_samples)
    val_graphs = get_graphs(val_samples)
    test_graphs = get_graphs(test_samples)

    print(f"Graphs: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    # Run ablation for each pooling type
    pooling_types = ['lysine_weighted', 'mean']
    results = []

    for pooling in pooling_types:
        print(f"\n{'='*60}")
        print(f"Pooling: {pooling}")
        print(f"{'='*60}")

        metrics = train_and_evaluate(
            pooling, train_graphs, val_graphs, test_graphs, device
        )

        print(f"Results: val={metrics['val_auroc']:.4f}, test={metrics['test_auroc']:.4f}")

        results.append({
            'pooling_type': pooling,
            **metrics
        })

        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("LYSINE POOLING ABLATION RESULTS")
    print("="*60)
    print(f"{'Pooling Method':<20} {'Val AUROC':<12} {'Test AUROC':<12} {'Test AUPRC':<12}")
    print("-"*56)
    for r in results:
        print(f"{r['pooling_type']:<20} {r['val_auroc']:.4f}       {r['test_auroc']:.4f}       {r['test_auprc']:.4f}")

    # Key finding
    lysine_auroc = results[0]['test_auroc']  # lysine_weighted first
    mean_auroc = results[1]['test_auroc']
    improvement = lysine_auroc - mean_auroc
    print(f"\nLysine-weighted pooling vs mean: {improvement:+.4f} ({improvement/mean_auroc*100:+.1f}%)")

    # Save
    with open("results/lysine_pooling_ablation.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/lysine_pooling_ablation.json")


if __name__ == "__main__":
    main()

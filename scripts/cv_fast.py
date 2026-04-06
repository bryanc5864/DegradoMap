#!/usr/bin/env python
"""
Fast 5-Fold CV - Pre-cache graphs, run on GPU.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Batch
from tqdm import tqdm

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs['flush'] = True
    _print(*args, **kwargs)

N_FOLDS = 5
SEEDS = [42, 123, 456]
EPOCHS = 15
BATCH_SIZE = 16  # Reduced for memory


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load structures
    print("Loading structures...")
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        structures[pt_file.stem] = torch.load(pt_file, map_location='cpu', weights_only=False)
    print(f"Loaded {len(structures)} structures")

    # Load samples
    print("Loading samples...")
    samples = build_protac8k_degradation_data()
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]
    print(f"Valid samples: {len(valid_samples)}")

    # Pre-build ALL graphs once
    print("Pre-building all graphs (this takes ~30 min)...")
    all_graphs = {}
    for i, s in enumerate(tqdm(valid_samples, desc="Building graphs")):
        uniprot = s["uniprot_id"]
        struct = structures[uniprot]

        key = (uniprot, s["e3_name"], s["label"])
        if key in all_graphs:
            continue

        graph = protein_to_graph(
            coords=struct["coords"],
            residues=struct["residues"],
            plddt=struct.get("plddt"),
            sasa=struct.get("sasa"),
            disorder=struct.get("disorder")
        )
        graph.y = torch.tensor([s["label"]], dtype=torch.float32)
        graph.e3_name = s["e3_name"]
        graph.uniprot = uniprot
        all_graphs[key] = graph

    print(f"Built {len(all_graphs)} unique graphs")

    # Create sample -> graph mapping
    sample_to_graph = {}
    for i, s in enumerate(valid_samples):
        key = (s["uniprot_id"], s["e3_name"], s["label"])
        sample_to_graph[i] = all_graphs[key]

    # Group by target for CV
    target_to_indices = {}
    for i, s in enumerate(valid_samples):
        target = s["uniprot_id"]
        if target not in target_to_indices:
            target_to_indices[target] = []
        target_to_indices[target].append(i)

    targets = list(target_to_indices.keys())
    np.random.seed(42)
    np.random.shuffle(targets)

    fold_size = len(targets) // N_FOLDS
    all_results = []

    print(f"\n{'='*70}")
    print(f"5-FOLD CV: target_unseen ({len(SEEDS)} seeds)")
    print(f"{'='*70}")

    for fold in range(N_FOLDS):
        test_targets = set(targets[fold * fold_size: (fold + 1) * fold_size])
        train_targets = set(targets) - test_targets

        test_idx = [i for t in test_targets for i in target_to_indices[t]]
        train_idx = [i for t in train_targets for i in target_to_indices[t]]

        # Val split
        np.random.shuffle(train_idx)
        n_val = max(1, int(len(train_idx) * 0.15))
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]

        train_graphs = [sample_to_graph[i] for i in train_idx]
        val_graphs = [sample_to_graph[i] for i in val_idx]
        test_graphs = [sample_to_graph[i] for i in test_idx]

        print(f"\nFold {fold}: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

        fold_aurocs = []
        for seed in SEEDS:
            print(f"  Seed {seed}:", end=" ")

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = DegradoMap(
                node_input_dim=28, sug_hidden_dim=128, sug_output_dim=64,
                sug_num_layers=4, e3_hidden_dim=64, e3_output_dim=64,
                e3_num_heads=4, e3_num_layers=2, context_output_dim=64,
                fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

            best_val = 0
            best_state = None

            for epoch in range(EPOCHS):
                # Train
                model.train()
                np.random.shuffle(train_graphs)
                for i in range(0, len(train_graphs), BATCH_SIZE):
                    batch_g = train_graphs[i:i+BATCH_SIZE]
                    batch = Batch.from_data_list([g.clone().to(device) for g in batch_g])
                    optimizer.zero_grad()
                    out = model(batch, e3_name=batch_g[0].e3_name)
                    loss = F.binary_cross_entropy_with_logits(
                        out["degrado_logits"].squeeze(), batch.y.squeeze()
                    )
                    loss.backward()
                    optimizer.step()

                # Val
                model.eval()
                preds, labels = [], []
                with torch.no_grad():
                    for g in val_graphs:
                        batch = Batch.from_data_list([g.clone().to(device)])
                        out = model(batch, e3_name=g.e3_name)
                        preds.append(torch.sigmoid(out["degrado_logits"]).cpu().item())
                        labels.append(g.y.item())

                val_auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
                if val_auroc > best_val:
                    best_val = val_auroc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Test
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for g in test_graphs:
                    batch = Batch.from_data_list([g.clone().to(device)])
                    out = model(batch, e3_name=g.e3_name)
                    preds.append(torch.sigmoid(out["degrado_logits"]).cpu().item())
                    labels.append(g.y.item())

            test_auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
            test_auprc = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0.0

            print(f"val={best_val:.4f}, test={test_auroc:.4f}")

            fold_aurocs.append(test_auroc)
            all_results.append({
                'fold': fold, 'seed': seed,
                'val_auroc': best_val, 'test_auroc': test_auroc, 'test_auprc': test_auprc,
                'n_train': len(train_graphs), 'n_test': len(test_graphs)
            })

        print(f"  Fold {fold} mean: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

        # Clear CUDA cache between folds
        torch.cuda.empty_cache()

    # Summary
    aurocs = [r['test_auroc'] for r in all_results]
    print(f"\n{'='*70}")
    print(f"FINAL: target_unseen 5-fold CV")
    print(f"{'='*70}")
    print(f"Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"95% CI: [{np.percentile(aurocs, 2.5):.4f}, {np.percentile(aurocs, 97.5):.4f}]")

    # Save
    output = {
        'split': 'target_unseen',
        'n_folds': N_FOLDS, 'seeds': SEEDS, 'epochs': EPOCHS,
        'results': all_results,
        'summary': {
            'mean': float(np.mean(aurocs)),
            'std': float(np.std(aurocs)),
            'ci_lower': float(np.percentile(aurocs, 2.5)),
            'ci_upper': float(np.percentile(aurocs, 97.5)),
        }
    }
    with open("results/cv_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/cv_results.json")


if __name__ == "__main__":
    main()

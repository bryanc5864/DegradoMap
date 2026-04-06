#!/usr/bin/env python3
"""
Multi-seed DegradoMap training for honest performance estimation.

Addresses reviewer concern: "You report multi-seed for SchNet and EGNN
but not for your own model."

Uses DEFAULT hyperparameters (LR=1e-3) to avoid test-set peeking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.utils.data import DataLoader

from src.models.degradomap import DegradoMap
from src.data.dataset import DegradationDataset
from src.training.trainer import collate_graph_batch
from scripts.train import build_protac8k_degradation_data, create_data_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_seed(seed: int, device: torch.device):
    """Train DegradoMap with a specific seed."""
    print(f"\n{'='*60}")
    print(f"SEED: {seed}")
    print(f"{'='*60}")

    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build samples from PROTAC-8K
    samples = build_protac8k_degradation_data(
        csv_path="data/raw/protac_8k/PROTAC-8K/protac.csv",
        structure_dir="data/processed/structures",
        require_structure=True,
    )
    print(f"Total samples: {len(samples)}")

    # Create target-unseen split with this seed
    train_data, val_data, test_data = create_data_splits(
        samples, split_type="target_unseen",
        train_ratio=0.7, val_ratio=0.15, seed=seed
    )
    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Create datasets
    train_dataset = DegradationDataset(
        train_data,
        structure_dir="data/processed/structures",
        esm_dir="data/processed/esm_embeddings",
        use_esm=False,
    )
    val_dataset = DegradationDataset(
        val_data,
        structure_dir="data/processed/structures",
        esm_dir="data/processed/esm_embeddings",
        use_esm=False,
    )
    test_dataset = DegradationDataset(
        test_data,
        structure_dir="data/processed/structures",
        esm_dir="data/processed/esm_embeddings",
        use_esm=False,
    )

    print(f"Datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=0, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=collate_graph_batch)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=0, collate_fn=collate_graph_batch)

    # Create model with DEFAULT hyperparameters
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
    ).to(device)

    # DEFAULT learning rate (not the tuned 5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    best_val_auroc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(50):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            if batch is None:
                continue

            # Batch is a dictionary with 'graph', 'label', 'e3_name' keys
            graph = batch['graph'].to(device)
            labels = batch['label'].to(device)
            e3_names = batch['e3_name']  # List of e3 names

            optimizer.zero_grad()

            # Forward pass with batched graph and first e3_name (they're often all the same in a batch)
            out = model(graph, e3_name=e3_names[0])
            logits = out['degrado_logits'].squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                graph = batch['graph'].to(device)
                labels = batch['label']
                e3_names = batch['e3_name']

                out = model(graph, e3_name=e3_names[0])
                logits = out['degrado_logits'].squeeze()
                probs = torch.sigmoid(logits)
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        if len(val_preds) > 0 and len(set(val_labels)) > 1:
            val_auroc = roc_auc_score(val_labels, val_preds)
        else:
            val_auroc = 0.5

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_auroc={val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            graph = batch['graph'].to(device)
            labels = batch['label']
            e3_names = batch['e3_name']

            out = model(graph, e3_name=e3_names[0])
            logits = out['degrado_logits'].squeeze()
            probs = torch.sigmoid(logits)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    if len(test_preds) > 0 and len(set(test_labels)) > 1:
        test_auroc = roc_auc_score(test_labels, test_preds)
        test_auprc = average_precision_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, [1 if p > 0.5 else 0 for p in test_preds])
    else:
        test_auroc = 0.5
        test_auprc = 0.0
        test_f1 = 0.0

    print(f"\nSeed {seed} Results:")
    print(f"  Best Val AUROC: {best_val_auroc:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"  Test AUPRC: {test_auprc:.4f}")
    print(f"  Test F1: {test_f1:.4f}")

    return {
        'seed': seed,
        'best_val_auroc': best_val_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'test_f1': test_f1,
        'n_train': len(train_data),
        'n_val': len(val_data),
        'n_test': len(test_data)
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Run with multiple seeds
    seeds = [42, 123, 456]
    results = []

    for seed in seeds:
        result = train_with_seed(seed, device)
        results.append(result)

    # Compute statistics
    test_aurocs = [r['test_auroc'] for r in results]
    test_auprcs = [r['test_auprc'] for r in results]

    mean_auroc = np.mean(test_aurocs)
    std_auroc = np.std(test_aurocs)
    mean_auprc = np.mean(test_auprcs)
    std_auprc = np.std(test_auprcs)

    print("\n" + "="*60)
    print("MULTI-SEED DEGRADOMAP RESULTS (DEFAULT HP: LR=1e-3)")
    print("="*60)
    print(f"\nTest AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Test AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"\nPer-seed results:")
    for r in results:
        print(f"  Seed {r['seed']}: AUROC={r['test_auroc']:.4f}, AUPRC={r['test_auprc']:.4f}")

    # Save results
    output = {
        'description': 'Multi-seed DegradoMap with DEFAULT hyperparameters (LR=1e-3)',
        'note': 'Addresses reviewer concern about missing multi-seed results',
        'hyperparameters': {
            'lr': 1e-3,
            'hidden_dim': 128,
            'num_gnn_layers': 4,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'batch_size': 32
        },
        'seeds': seeds,
        'per_seed_results': results,
        'summary': {
            'test_auroc_mean': mean_auroc,
            'test_auroc_std': std_auroc,
            'test_auprc_mean': mean_auprc,
            'test_auprc_std': std_auprc
        }
    }

    output_path = Path('results/multi_seed_degradomap.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Separate ablations for 1/sqrt(N) normalization and per-protein softmax.

Addresses reviewer concern: "The 1/sqrt(N) normalization is credited with
much of the 0.529→0.657 improvement, but confounded with the per-protein
softmax fix — they're reported together. Need separate ablations."

Tests 4 configurations:
1. Baseline (no fixes)
2. +1/sqrt(N) only
3. +per-protein softmax only
4. Both fixes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool

from src.models.degradomap import DegradoMap
from src.data.dataset import DegradationDataset
from src.training.trainer import collate_graph_batch
from scripts.train import build_protac8k_degradation_data, create_data_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblatedSUGModule(nn.Module):
    """SUG module with configurable normalization options for ablation."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, radius=8.0,
                 use_sqrt_norm=True, use_per_protein_softmax=True, dropout=0.1):
        super().__init__()
        self.use_sqrt_norm = use_sqrt_norm
        self.use_per_protein_softmax = use_per_protein_softmax

        # GNN layers (simple message passing)
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Lysine attention
        self.lysine_attention = nn.Linear(hidden_dim, 1)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        # Pooling with optional sqrt normalization
        protein_repr = global_mean_pool(x, batch)

        if self.use_sqrt_norm:
            # Count nodes per graph and normalize
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
            node_counts = torch.zeros(batch_size, device=x.device)
            for i in range(batch_size):
                node_counts[i] = (batch == i).sum()
            protein_repr = protein_repr / torch.sqrt(node_counts.unsqueeze(1) + 1e-6)

        # Lysine attention (with optional per-protein softmax)
        lysine_mask = data.lysine_mask if hasattr(data, 'lysine_mask') and data.lysine_mask is not None else None

        if lysine_mask is not None and lysine_mask.any():
            lysine_scores = self.lysine_attention(x).squeeze(-1)
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1

            if self.use_per_protein_softmax:
                # Per-protein softmax (correct version)
                lysine_attention = torch.zeros_like(lysine_scores)
                for i in range(batch_size):
                    mask_i = (batch == i) & lysine_mask
                    if mask_i.any():
                        scores_i = lysine_scores[mask_i]
                        lysine_attention[mask_i] = torch.softmax(scores_i, dim=0)
            else:
                # Global softmax (the problematic version)
                masked_scores = lysine_scores.clone()
                masked_scores[~lysine_mask.bool()] = float('-inf')
                lysine_attention = torch.softmax(masked_scores, dim=0)

            # Weighted lysine representation
            lysine_repr = torch.zeros(batch_size, x.size(-1), device=x.device)
            for i in range(batch_size):
                mask_i = batch == i
                lysine_repr[i] = (x[mask_i] * lysine_attention[mask_i].unsqueeze(-1)).sum(0)

            protein_repr = protein_repr + lysine_repr

        return self.output_proj(protein_repr)


class AblatedDegradoMap(nn.Module):
    """Simplified DegradoMap for ablation study."""

    def __init__(self, input_dim=28, hidden_dim=128, output_dim=64,
                 use_sqrt_norm=True, use_per_protein_softmax=True, dropout=0.1):
        super().__init__()

        self.sug = AblatedSUGModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_sqrt_norm=use_sqrt_norm,
            use_per_protein_softmax=use_per_protein_softmax,
            dropout=dropout,
        )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        protein_repr = self.sug(data)
        logit = self.pred_head(protein_repr)
        return {'degrad_logit': logit.squeeze(-1)}


def train_and_evaluate(config_name, use_sqrt_norm, use_per_protein_softmax, device, seed=42):
    """Train and evaluate a single configuration."""
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"  sqrt_norm: {use_sqrt_norm}")
    print(f"  per_protein_softmax: {use_per_protein_softmax}")
    print(f"{'='*60}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build samples
    samples = build_protac8k_degradation_data(
        csv_path="data/raw/protac_8k/PROTAC-8K/protac.csv",
        structure_dir="data/processed/structures",
        require_structure=True,
    )

    # Create target-unseen split
    train_data, val_data, test_data = create_data_splits(
        samples, split_type="target_unseen", seed=seed
    )

    # Create datasets
    train_dataset = DegradationDataset(train_data, structure_dir="data/processed/structures", use_esm=False)
    val_dataset = DegradationDataset(val_data, structure_dir="data/processed/structures", use_esm=False)
    test_dataset = DegradationDataset(test_data, structure_dir="data/processed/structures", use_esm=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=0, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=collate_graph_batch)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=0, collate_fn=collate_graph_batch)

    # Create ablated model
    model = AblatedDegradoMap(
        input_dim=28,
        hidden_dim=128,
        output_dim=64,
        use_sqrt_norm=use_sqrt_norm,
        use_per_protein_softmax=use_per_protein_softmax,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    best_val_auroc = 0
    best_model_state = None

    for epoch in range(30):
        model.train()
        print(f"    Epoch {epoch} starting...", flush=True)
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            graph = batch['graph'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            out = model(graph)
            logits = out['degrad_logit'].squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                graph = batch['graph'].to(device)
                labels = batch['label']

                out = model(graph)
                logits = out['degrad_logit'].squeeze()
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
            print(f"  Epoch {epoch}: val_auroc={val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            graph = batch['graph'].to(device)
            labels = batch['label']

            out = model(graph)
            logits = out['degrad_logit'].squeeze()
            probs = torch.sigmoid(logits)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    if len(test_preds) > 0 and len(set(test_labels)) > 1:
        test_auroc = roc_auc_score(test_labels, test_preds)
        test_auprc = average_precision_score(test_labels, test_preds)
    else:
        test_auroc = 0.5
        test_auprc = 0.0

    print(f"\nResults:")
    print(f"  Val AUROC: {best_val_auroc:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}")

    return {
        'config': config_name,
        'sqrt_norm': use_sqrt_norm,
        'per_protein_softmax': use_per_protein_softmax,
        'val_auroc': best_val_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Run all 4 configurations
    configs = [
        ('baseline', False, False),
        ('sqrt_norm_only', True, False),
        ('per_protein_softmax_only', False, True),
        ('both_fixes', True, True)
    ]

    results = []
    for name, sqrt_norm, per_protein_softmax in configs:
        result = train_and_evaluate(name, sqrt_norm, per_protein_softmax, device)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("NORMALIZATION ABLATION RESULTS")
    print("="*60)
    print(f"\n{'Config':<30} {'Val AUROC':<12} {'Test AUROC':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['config']:<30} {r['val_auroc']:.4f}       {r['test_auroc']:.4f}")

    # Calculate individual contributions
    baseline = results[0]['test_auroc']
    sqrt_only = results[1]['test_auroc']
    softmax_only = results[2]['test_auroc']
    both = results[3]['test_auroc']

    print(f"\nIndividual contributions:")
    print(f"  sqrt_norm alone: {sqrt_only - baseline:+.4f}")
    print(f"  per_protein_softmax alone: {softmax_only - baseline:+.4f}")
    print(f"  Both combined: {both - baseline:+.4f}")
    print(f"  Interaction effect: {(both - baseline) - (sqrt_only - baseline) - (softmax_only - baseline):+.4f}")

    # Save results
    output = {
        'description': 'Separate ablations for 1/sqrt(N) normalization and per-protein softmax',
        'results': results,
        'summary': {
            'baseline': baseline,
            'sqrt_norm_contribution': sqrt_only - baseline,
            'per_protein_softmax_contribution': softmax_only - baseline,
            'combined_contribution': both - baseline,
            'interaction_effect': (both - baseline) - (sqrt_only - baseline) - (softmax_only - baseline)
        }
    }

    output_path = Path('results/normalization_ablation.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()

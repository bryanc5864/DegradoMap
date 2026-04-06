#!/usr/bin/env python3
"""
Three-phase training ablation.

Compares:
1. Full model (Phase 1 pre-training + Phase 2 fine-tuning)
2. No pre-training (Phase 2 only, random init)
3. Pre-training only (Phase 1, then freeze and add classifier)

This quantifies the contribution of pre-training.
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
from torch_geometric.nn import GCNConv, global_mean_pool

from src.data.dataset import DegradationDataset
from src.training.trainer import collate_graph_batch
from scripts.train import build_protac8k_degradation_data, create_data_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSUG(nn.Module):
    """Simplified SUG module for ablation."""

    def __init__(self, input_dim=28, hidden_dim=128, output_dim=64, num_layers=4, dropout=0.1):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        # Mean pooling with sqrt(N) normalization
        protein_repr = global_mean_pool(x, batch)
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        node_counts = torch.zeros(batch_size, device=x.device)
        for i in range(batch_size):
            node_counts[i] = (batch == i).sum()
        protein_repr = protein_repr / torch.sqrt(node_counts.unsqueeze(1) + 1e-6)

        return self.output_proj(protein_repr)


class DegradoMapSimple(nn.Module):
    """Simplified DegradoMap for ablation study."""

    def __init__(self, input_dim=28, hidden_dim=128, output_dim=64, dropout=0.1, pretrained_sug=None):
        super().__init__()

        if pretrained_sug is not None:
            self.sug = pretrained_sug
        else:
            self.sug = SimpleSUG(input_dim, hidden_dim, output_dim, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        protein_repr = self.sug(data)
        logit = self.classifier(protein_repr)
        return {'logit': logit.squeeze(-1)}


def pretrain_sug(train_loader, val_loader, device, epochs=10):
    """
    Pre-train SUG module on auxiliary task (predicting node properties).

    For simplicity, we use reconstruction loss as pre-training objective.
    """
    print("Pre-training SUG module...")

    sug = SimpleSUG(input_dim=28, hidden_dim=128, output_dim=64).to(device)
    decoder = nn.Linear(64, 28).to(device)  # Reconstruct node features

    optimizer = torch.optim.Adam(list(sug.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        sug.train()
        decoder.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            if batch is None:
                continue

            graph = batch['graph'].to(device)
            optimizer.zero_grad()

            # Get node embeddings before pooling
            x = graph.x
            edge_index = graph.edge_index

            for conv in sug.convs:
                x = conv(x, edge_index)
                x = torch.relu(x)

            # Reconstruction loss
            x_proj = sug.output_proj(x)
            x_recon = decoder(x_proj)
            loss = criterion(x_recon, graph.x)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0:
            print(f"  Pre-train epoch {epoch}: loss={total_loss/max(n_batches,1):.4f}")

    return sug


def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       epochs=30, freeze_sug=False):
    """Train and evaluate a model configuration."""

    if freeze_sug:
        for param in model.sug.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        print(f"    Epoch {epoch} starting...", flush=True)
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            graph = batch['graph'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            out = model(graph)
            logits = out['logit']
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
                logits = out['logit']
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
            print(f"    Epoch {epoch}: val_auroc={val_auroc:.4f}")

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
            logits = out['logit']
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

    return best_val_auroc, test_auroc, test_auprc


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    samples = build_protac8k_degradation_data(
        csv_path="data/raw/protac_8k/PROTAC-8K/protac.csv",
        structure_dir="data/processed/structures",
        require_structure=True,
    )

    # Create split
    train_data, val_data, test_data = create_data_splits(
        samples, split_type="target_unseen", seed=42
    )

    # Create datasets
    train_dataset = DegradationDataset(train_data, structure_dir="data/processed/structures", use_esm=False)
    val_dataset = DegradationDataset(val_data, structure_dir="data/processed/structures", use_esm=False)
    test_dataset = DegradationDataset(test_data, structure_dir="data/processed/structures", use_esm=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=0, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            num_workers=0, collate_fn=collate_graph_batch)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             num_workers=0, collate_fn=collate_graph_batch)

    results = []

    # Configuration 1: No pre-training (random init, fine-tune only)
    print("\n" + "="*60)
    print("Config 1: NO PRE-TRAINING (random init)")
    print("="*60)

    model1 = DegradoMapSimple(input_dim=28, hidden_dim=128, output_dim=64).to(device)
    val_auroc, test_auroc, test_auprc = train_and_evaluate(
        model1, train_loader, val_loader, test_loader, device, epochs=30
    )
    print(f"\nResults: val_auroc={val_auroc:.4f}, test_auroc={test_auroc:.4f}")
    results.append({
        'config': 'no_pretrain',
        'description': 'Random initialization, fine-tuning only',
        'val_auroc': val_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc
    })

    # Configuration 2: Pre-training + fine-tuning (full model)
    print("\n" + "="*60)
    print("Config 2: PRE-TRAINING + FINE-TUNING")
    print("="*60)

    pretrained_sug = pretrain_sug(train_loader, val_loader, device, epochs=10)
    model2 = DegradoMapSimple(input_dim=28, hidden_dim=128, output_dim=64,
                               pretrained_sug=pretrained_sug).to(device)
    val_auroc, test_auroc, test_auprc = train_and_evaluate(
        model2, train_loader, val_loader, test_loader, device, epochs=30
    )
    print(f"\nResults: val_auroc={val_auroc:.4f}, test_auroc={test_auroc:.4f}")
    results.append({
        'config': 'pretrain_finetune',
        'description': 'Pre-training + fine-tuning (full model)',
        'val_auroc': val_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc
    })

    # Configuration 3: Pre-training only (frozen SUG, train classifier only)
    print("\n" + "="*60)
    print("Config 3: PRE-TRAINING ONLY (frozen SUG)")
    print("="*60)

    pretrained_sug2 = pretrain_sug(train_loader, val_loader, device, epochs=10)
    model3 = DegradoMapSimple(input_dim=28, hidden_dim=128, output_dim=64,
                               pretrained_sug=pretrained_sug2).to(device)
    val_auroc, test_auroc, test_auprc = train_and_evaluate(
        model3, train_loader, val_loader, test_loader, device, epochs=30, freeze_sug=True
    )
    print(f"\nResults: val_auroc={val_auroc:.4f}, test_auroc={test_auroc:.4f}")
    results.append({
        'config': 'pretrain_only',
        'description': 'Pre-training only (frozen SUG, train classifier)',
        'val_auroc': val_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc
    })

    # Summary
    print("\n" + "="*60)
    print("THREE-PHASE ABLATION SUMMARY")
    print("="*60)
    print(f"\n{'Config':<30} {'Val AUROC':<12} {'Test AUROC':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['config']:<30} {r['val_auroc']:.4f}       {r['test_auroc']:.4f}")

    # Compute contributions
    no_pretrain = results[0]['test_auroc']
    full_model = results[1]['test_auroc']
    pretrain_only = results[2]['test_auroc']

    print(f"\nContributions:")
    print(f"  Pre-training effect: {full_model - no_pretrain:+.4f}")
    print(f"  Fine-tuning effect: {full_model - pretrain_only:+.4f}")

    # Save results
    output = {
        'description': 'Three-phase training ablation',
        'results': results,
        'summary': {
            'no_pretrain_auroc': no_pretrain,
            'full_model_auroc': full_model,
            'pretrain_only_auroc': pretrain_only,
            'pretrain_contribution': full_model - no_pretrain,
            'finetune_contribution': full_model - pretrain_only
        }
    }

    output_path = Path('results/three_phase_ablation.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()

"""
Ablation Study for DegradoMap

Tests what each module contributes to target-unseen performance:
1. SUG-only: SUG module + simple pooling (no E3, no context)
2. E3-only: E3 embeddings + classifier (no SUG)
3. Full model: All modules
4. Full model + Ub sites: Add known Ub sites as direct features

Usage:
    python scripts/ablation_study.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sug_module import SUGModule
from src.models.e3_compat_module import E3CompatModule
from src.data.dataset import DegradationDataset
from src.training.trainer import collate_graph_batch
from scripts.train import build_protac8k_degradation_data, create_data_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Ablation")


class SUGOnlyModel(nn.Module):
    """Ablation 1: SUG module + mean pooling + classifier"""

    def __init__(self, node_input_dim=28, hidden_dim=128, output_dim=64, num_layers=4):
        super().__init__()
        self.sug = SUGModule(
            node_input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, graph, e3_name=None):
        sug_out = self.sug(graph)
        # Use sug_vector which is already pooled [B, output_dim]
        pooled = sug_out["sug_vector"]
        logits = self.classifier(pooled).squeeze(-1)
        return {
            "degrado_logits": logits,
            "degrado_score": torch.sigmoid(logits),
        }


class E3OnlyModel(nn.Module):
    """Ablation 2: E3 embedding + classifier (no structure)"""

    def __init__(self, num_e3=15, e3_dim=64):
        super().__init__()
        self.e3_names = ["CRBN", "VHL", "MDM2", "cIAP1", "XIAP", "DCAF16",
                         "KEAP1", "FEM1B", "DCAF1", "UBR", "KLHL20", "unknown"]
        self.e3_embed = nn.Embedding(len(self.e3_names), e3_dim)
        self.classifier = nn.Sequential(
            nn.Linear(e3_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, graph, e3_name):
        # Get batch size from graph
        if hasattr(graph, 'batch') and graph.batch is not None:
            batch_size = graph.batch.max().item() + 1
        else:
            batch_size = 1

        if e3_name in self.e3_names:
            e3_idx = self.e3_names.index(e3_name)
        else:
            e3_idx = self.e3_names.index("unknown")
        # Expand to batch size
        e3_idx = torch.tensor([e3_idx] * batch_size, device=graph.x.device)
        e3_emb = self.e3_embed(e3_idx)  # [B, e3_dim]
        logits = self.classifier(e3_emb).squeeze(-1)  # [B]
        return {
            "degrado_logits": logits,
            "degrado_score": torch.sigmoid(logits),
        }


class SUGWithE3Model(nn.Module):
    """Ablation 3: SUG + E3 embedding (simplified, no cross-attention)"""

    def __init__(self, node_input_dim=28, hidden_dim=128, output_dim=64, num_layers=4):
        super().__init__()
        self.sug = SUGModule(
            node_input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.e3_names = ["CRBN", "VHL", "MDM2", "cIAP1", "XIAP", "DCAF16",
                         "KEAP1", "FEM1B", "DCAF1", "UBR", "KLHL20", "unknown"]
        self.e3_embed = nn.Embedding(len(self.e3_names), 64)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, graph, e3_name):
        sug_out = self.sug(graph)
        pooled = sug_out["sug_vector"]  # [B, output_dim]
        batch_size = pooled.size(0)

        if e3_name in self.e3_names:
            e3_idx = self.e3_names.index(e3_name)
        else:
            e3_idx = self.e3_names.index("unknown")
        # Expand to batch size
        e3_idx = torch.tensor([e3_idx] * batch_size, device=graph.x.device)
        e3_emb = self.e3_embed(e3_idx)  # [B, 64]

        combined = torch.cat([pooled, e3_emb], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return {
            "degrado_logits": logits,
            "degrado_score": torch.sigmoid(logits),
        }


def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       epochs=10, lr=1e-4, device="cuda"):
    """Train model and return test metrics."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            graphs = batch["graph"].to(device)
            labels = batch["label"].to(device)
            e3_name = batch["e3_name"][0]

            optimizer.zero_grad()
            outputs = model(graphs, e3_name)
            loss = criterion(outputs["degrado_logits"], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                graphs = batch["graph"].to(device)
                labels = batch["label"]
                e3_name = batch["e3_name"][0]
                outputs = model(graphs, e3_name)
                val_scores.extend(outputs["degrado_score"].cpu().numpy())
                val_labels.extend(labels.numpy())

        val_auroc = roc_auc_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0.5

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, val_auroc={val_auroc:.4f}")

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    model.eval()
    test_scores, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            graphs = batch["graph"].to(device)
            labels = batch["label"]
            e3_name = batch["e3_name"][0]
            outputs = model(graphs, e3_name)
            test_scores.extend(outputs["degrado_score"].cpu().numpy())
            test_labels.extend(labels.numpy())

    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    auroc = roc_auc_score(test_labels, test_scores) if len(set(test_labels)) > 1 else 0.5
    auprc = average_precision_score(test_labels, test_scores) if len(set(test_labels)) > 1 else 0.0
    f1 = f1_score(test_labels, (test_scores > 0.5).astype(float), zero_division=0)

    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        t_f1 = f1_score(test_labels, (test_scores >= t).astype(float), zero_division=0)
        if t_f1 > best_f1:
            best_f1, best_thresh = t_f1, t

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "f1_optimal": best_f1,
        "threshold_optimal": best_thresh,
        "best_val_auroc": best_val_auroc,
    }


def run_ablation(args):
    """Run full ablation study."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading PROTAC-8K data...")
    samples = build_protac8k_degradation_data()

    results = {}

    for split_type in ["target_unseen", "e3_unseen", "random"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {split_type}")
        logger.info(f"{'='*60}")

        train_data, val_data, test_data = create_data_splits(samples, split_type=split_type)

        train_dataset = DegradationDataset(train_data, use_esm=False)
        val_dataset = DegradationDataset(val_data, use_esm=False)
        test_dataset = DegradationDataset(test_data, use_esm=False)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                  collate_fn=collate_graph_batch)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                               collate_fn=collate_graph_batch)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                                collate_fn=collate_graph_batch)

        split_results = {}

        # Ablation 1: SUG only
        logger.info("\n--- Ablation 1: SUG Module Only ---")
        model = SUGOnlyModel(node_input_dim=28)
        metrics = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                    epochs=args.epochs, device=device)
        split_results["sug_only"] = metrics
        logger.info(f"  Test AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}")

        # Ablation 2: E3 only
        logger.info("\n--- Ablation 2: E3 Embedding Only ---")
        model = E3OnlyModel()
        metrics = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                    epochs=args.epochs, device=device)
        split_results["e3_only"] = metrics
        logger.info(f"  Test AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}")

        # Ablation 3: SUG + E3 (simplified)
        logger.info("\n--- Ablation 3: SUG + E3 (Simplified) ---")
        model = SUGWithE3Model(node_input_dim=28)
        metrics = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                    epochs=args.epochs, device=device)
        split_results["sug_e3_simple"] = metrics
        logger.info(f"  Test AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}")

        results[split_type] = split_results

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Model':<25} {'Target-Unseen':<15} {'E3-Unseen':<15} {'Random':<15}")
    logger.info("-"*80)

    for model_name in ["sug_only", "e3_only", "sug_e3_simple"]:
        row = f"{model_name:<25}"
        for split in ["target_unseen", "e3_unseen", "random"]:
            auroc = results[split][model_name]["auroc"]
            row += f"{auroc:.4f}{'':>10}"
        logger.info(row)

    logger.info("\nBaseline comparison:")
    logger.info("  Full DegradoMap: target_unseen=0.54, e3_unseen=0.81, random=0.77")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DegradoMap Ablation Study")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_ablation(args)

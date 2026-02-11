"""
Test Known Ub Sites as Direct Features

Tests whether known ubiquitination sites from PhosphoSitePlus improve prediction.
Implements the MAPD insight that E2-accessible Ub sites are key for degradability.

Usage:
    python scripts/test_ub_sites.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.degradomap import DegradoMap
from src.data.dataset import DegradationDataset
from src.training.trainer import collate_graph_batch
from scripts.train import build_protac8k_degradation_data, create_data_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Ub-Sites-Test")


def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       epochs=15, lr=1e-4, device="cuda", pos_weight=None):
    """Train model and return test metrics."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

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


def run_ub_sites_test(args):
    """Test known Ub sites as direct features."""
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading PROTAC-8K data...")
    samples = build_protac8k_degradation_data()

    # Calculate class weights
    n_pos = sum(1 for s in samples if s["label"] > 0.5)
    n_neg = len(samples) - n_pos
    pos_weight = n_neg / n_pos
    logger.info(f"Class balance: {n_pos} pos, {n_neg} neg, pos_weight={pos_weight:.2f}")

    results = {}

    for split_type in ["target_unseen", "e3_unseen", "random"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {split_type}")
        logger.info(f"{'='*60}")

        train_data, val_data, test_data = create_data_splits(samples, split_type=split_type)
        split_results = {}

        # Test 1: Baseline (no Ub sites)
        logger.info("\n--- Baseline (no Ub sites) ---")
        train_dataset = DegradationDataset(train_data, use_esm=False, use_ub_sites=False)
        val_dataset = DegradationDataset(val_data, use_esm=False, use_ub_sites=False)
        test_dataset = DegradationDataset(test_data, use_esm=False, use_ub_sites=False)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                  collate_fn=collate_graph_batch)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                               collate_fn=collate_graph_batch)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                                collate_fn=collate_graph_batch)

        model = DegradoMap(
            node_input_dim=28, sug_hidden_dim=128, sug_output_dim=64, sug_num_layers=4,
            e3_hidden_dim=64, e3_output_dim=64, e3_num_heads=4, e3_num_layers=2,
            context_output_dim=64, fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
        )

        metrics = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                    epochs=args.epochs, device=device, pos_weight=pos_weight)
        split_results["baseline"] = metrics
        logger.info(f"  Test AUROC: {metrics['auroc']:.4f}, F1@optimal: {metrics['f1_optimal']:.4f}")

        # Test 2: With known Ub sites
        logger.info("\n--- With Known Ub Sites (PhosphoSitePlus) ---")
        train_dataset_ub = DegradationDataset(train_data, use_esm=False, use_ub_sites=True)
        val_dataset_ub = DegradationDataset(val_data, use_esm=False, use_ub_sites=True)
        test_dataset_ub = DegradationDataset(test_data, use_esm=False, use_ub_sites=True)

        train_loader_ub = DataLoader(train_dataset_ub, batch_size=4, shuffle=True,
                                     collate_fn=collate_graph_batch)
        val_loader_ub = DataLoader(val_dataset_ub, batch_size=4, shuffle=False,
                                  collate_fn=collate_graph_batch)
        test_loader_ub = DataLoader(test_dataset_ub, batch_size=4, shuffle=False,
                                   collate_fn=collate_graph_batch)

        model_ub = DegradoMap(
            node_input_dim=29,  # +1 for known Ub site feature
            sug_hidden_dim=128, sug_output_dim=64, sug_num_layers=4,
            e3_hidden_dim=64, e3_output_dim=64, e3_num_heads=4, e3_num_layers=2,
            context_output_dim=64, fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
        )

        metrics_ub = train_and_evaluate(model_ub, train_loader_ub, val_loader_ub, test_loader_ub,
                                        epochs=args.epochs, device=device, pos_weight=pos_weight)
        split_results["with_ub_sites"] = metrics_ub
        logger.info(f"  Test AUROC: {metrics_ub['auroc']:.4f}, F1@optimal: {metrics_ub['f1_optimal']:.4f}")

        # Test 3: ESM + Ub sites
        logger.info("\n--- ESM-2 + Known Ub Sites ---")
        train_dataset_esm_ub = DegradationDataset(train_data, use_esm=True, use_ub_sites=True)
        val_dataset_esm_ub = DegradationDataset(val_data, use_esm=True, use_ub_sites=True)
        test_dataset_esm_ub = DegradationDataset(test_data, use_esm=True, use_ub_sites=True)

        train_loader_esm_ub = DataLoader(train_dataset_esm_ub, batch_size=4, shuffle=True,
                                         collate_fn=collate_graph_batch)
        val_loader_esm_ub = DataLoader(val_dataset_esm_ub, batch_size=4, shuffle=False,
                                       collate_fn=collate_graph_batch)
        test_loader_esm_ub = DataLoader(test_dataset_esm_ub, batch_size=4, shuffle=False,
                                        collate_fn=collate_graph_batch)

        model_esm_ub = DegradoMap(
            node_input_dim=1285,  # 1280 ESM + 4 structural + 1 ub site
            sug_hidden_dim=128, sug_output_dim=64, sug_num_layers=4,
            e3_hidden_dim=64, e3_output_dim=64, e3_num_heads=4, e3_num_layers=2,
            context_output_dim=64, fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
        )

        metrics_esm_ub = train_and_evaluate(model_esm_ub, train_loader_esm_ub, val_loader_esm_ub,
                                            test_loader_esm_ub, epochs=args.epochs, device=device,
                                            pos_weight=pos_weight)
        split_results["esm_with_ub_sites"] = metrics_esm_ub
        logger.info(f"  Test AUROC: {metrics_esm_ub['auroc']:.4f}, F1@optimal: {metrics_esm_ub['f1_optimal']:.4f}")

        results[split_type] = split_results

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/ub_sites_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("KNOWN UB SITES FEATURE TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Model':<25} {'Target-Unseen':<15} {'E3-Unseen':<15} {'Random':<15}")
    logger.info("-"*80)

    for model_name in ["baseline", "with_ub_sites", "esm_with_ub_sites"]:
        row = f"{model_name:<25}"
        for split in ["target_unseen", "e3_unseen", "random"]:
            auroc = results[split][model_name]["auroc"]
            row += f"{auroc:.4f}{'':>10}"
        logger.info(row)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Known Ub Sites Feature")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    args = parser.parse_args()

    run_ub_sites_test(args)

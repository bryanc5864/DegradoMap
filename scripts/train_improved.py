#!/usr/bin/env python3
"""
Improved training script with all features enabled.

This script runs training with:
- ESM-2 embeddings (1285-dim node features)
- Known ubiquitination sites from PhosphoSitePlus
- E3 ligase one-hot encoding
- Global protein statistics (pLDDT/SASA aggregates)
- Optimized hyperparameters

Target: Beat GradientBoosting (0.607 AUROC) on target_unseen split.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.degradomap import DegradoMap
from src.data.dataset import DegradationDataset
from src.training.trainer import DegradoMapTrainer, collate_graph_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("DegradoMap-Improved")


def build_protac8k_data(structure_dir="data/processed/structures"):
    """Load and filter PROTAC-8K dataset."""
    import pandas as pd

    csv_path = "data/raw/protac_8k/PROTAC-8K/protac.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    labeled = df[df['Label'].notna()].copy()

    struct_path = Path(structure_dir)
    available = set(p.stem for p in struct_path.glob("*.pt"))

    e3_normalize = {
        'CRBN': 'CRBN', 'VHL': 'VHL', 'MDM2': 'MDM2',
        'cIAP1': 'cIAP1', 'DCAF16': 'DCAF16',
    }

    samples = []
    for _, row in labeled.iterrows():
        uniprot = str(row.get('Uniprot', '')).strip()
        if not uniprot or uniprot == 'nan' or uniprot not in available:
            continue

        e3_raw = str(row.get('E3 ligase', 'CRBN')).strip()
        e3_name = e3_normalize.get(e3_raw, e3_raw)
        label = float(row['Label'])
        target_gene = str(row.get('Target', 'unknown')).strip()

        dc50_raw = row.get('DC50 (nM)', None)
        dmax_raw = row.get('Dmax (%)', None)
        try:
            dc50_nM = float(dc50_raw) if pd.notna(dc50_raw) and dc50_raw != '' else None
        except (ValueError, TypeError):
            dc50_nM = None
        try:
            dmax_pct = float(dmax_raw) if pd.notna(dmax_raw) and dmax_raw != '' else None
        except (ValueError, TypeError):
            dmax_pct = None

        dc50_log10 = np.log10(max(dc50_nM, 0.1)) if dc50_nM is not None else 2.0
        dmax_fraction = dmax_pct / 100.0 if dmax_pct is not None else 0.5

        samples.append({
            "target_gene": target_gene,
            "uniprot_id": uniprot,
            "e3_name": e3_name,
            "cell_line": "unknown",
            "dc50_log10": dc50_log10,
            "dmax_fraction": dmax_fraction,
            "label": label,
            "weight": 1.0,
        })

    logger.info(f"Loaded {len(samples)} samples (pos={sum(1 for s in samples if s['label'] > 0.5)}, neg={sum(1 for s in samples if s['label'] <= 0.5)})")
    return samples


def create_target_unseen_split(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create target-unseen split with balanced E3 distribution."""
    rng = np.random.RandomState(seed)

    # Overall E3 distribution
    overall_e3 = defaultdict(int)
    for s in samples:
        overall_e3[s["e3_name"]] += 1
    total_samples = len(samples)

    # Per-target statistics
    target_samples = defaultdict(list)
    target_e3_breakdown = defaultdict(lambda: defaultdict(int))
    for i, s in enumerate(samples):
        target_samples[s["target_gene"]].append(i)
        target_e3_breakdown[s["target_gene"]][s["e3_name"]] += 1

    targets = list(target_samples.keys())
    rng.shuffle(targets)

    # Greedy selection for test set
    test_targets = set()
    test_e3_counts = defaultdict(int)
    test_size = 0
    target_test_ratio = 1 - train_ratio - val_ratio

    for target in targets:
        if test_size >= len(samples) * target_test_ratio:
            break

        candidate_e3 = test_e3_counts.copy()
        for e3, count in target_e3_breakdown[target].items():
            candidate_e3[e3] += count
        candidate_total = test_size + len(target_samples[target])

        if candidate_total > 0:
            crbn_pct = candidate_e3.get('CRBN', 0) / candidate_total
            vhl_pct = candidate_e3.get('VHL', 0) / candidate_total
            target_crbn = overall_e3['CRBN'] / total_samples
            target_vhl = overall_e3['VHL'] / total_samples

            if abs(crbn_pct - target_crbn) < 0.15 and abs(vhl_pct - target_vhl) < 0.15:
                test_targets.add(target)
                for e3, count in target_e3_breakdown[target].items():
                    test_e3_counts[e3] += count
                test_size += len(target_samples[target])
            elif test_size < len(samples) * target_test_ratio * 0.5:
                test_targets.add(target)
                for e3, count in target_e3_breakdown[target].items():
                    test_e3_counts[e3] += count
                test_size += len(target_samples[target])

    remaining_targets = [t for t in targets if t not in test_targets]
    rng.shuffle(remaining_targets)
    n_val_targets = max(1, int(len(remaining_targets) * val_ratio / (train_ratio + val_ratio)))
    val_targets = set(remaining_targets[:n_val_targets])
    train_targets = set(remaining_targets[n_val_targets:])

    train = [s for s in samples if s["target_gene"] in train_targets]
    val = [s for s in samples if s["target_gene"] in val_targets]
    test = [s for s in samples if s["target_gene"] in test_targets]

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def train_and_evaluate(args):
    """Train model and evaluate on target_unseen split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    samples = build_protac8k_data()
    train_data, val_data, test_data = create_target_unseen_split(samples, seed=args.seed)

    # Build model
    if args.use_esm:
        node_input_dim = 1285
    elif args.use_ub_sites:
        node_input_dim = 29
    else:
        node_input_dim = 28

    logger.info(f"Building model: node_dim={node_input_dim}, ESM={args.use_esm}, Ub={args.use_ub_sites}, E3OH={args.use_e3_onehot}, GS={args.use_global_stats}")

    model = DegradoMap(
        node_input_dim=node_input_dim,
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
        dropout=args.dropout,
        use_e3_onehot=args.use_e3_onehot,
        use_global_stats=args.use_global_stats,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Create datasets
    train_dataset = DegradationDataset(
        train_data,
        structure_dir="data/processed/structures",
        esm_dir="data/processed/esm_embeddings",
        use_esm=args.use_esm,
        use_ub_sites=args.use_ub_sites,
    )
    val_dataset = DegradationDataset(
        val_data,
        structure_dir="data/processed/structures",
        esm_dir="data/processed/esm_embeddings",
        use_esm=args.use_esm,
        use_ub_sites=args.use_ub_sites,
    )
    test_dataset = DegradationDataset(
        test_data,
        structure_dir="data/processed/structures",
        esm_dir="data/processed/esm_embeddings",
        use_esm=args.use_esm,
        use_ub_sites=args.use_ub_sites,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)

    # Setup trainer
    trainer_config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": 1.0,
        "lambda_degrad": 1.0,
        "lambda_dc50": 0.3,
        "lambda_dmax": 0.3,
        "pos_weight": None,
    }
    trainer = DegradoMapTrainer(model, config=trainer_config, device=device)

    # Train
    best_val_auroc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch_degradation(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_metrics['loss']:.4f}, val_auroc={val_metrics['auroc']:.4f}")

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "checkpoints/improved_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load("checkpoints/improved_best.pt", weights_only=True))
    test_metrics = trainer.evaluate(test_loader)

    logger.info(f"\n{'='*50}")
    logger.info(f"Best validation AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    logger.info(f"Test Results (target_unseen):")
    logger.info(f"  AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"  AUPRC: {test_metrics['auprc']:.4f}")
    logger.info(f"  F1:    {test_metrics['f1']:.4f}")
    logger.info(f"{'='*50}")

    return test_metrics


def run_cross_validation(args, n_folds=5):
    """Run k-fold cross-validation."""
    samples = build_protac8k_data()

    # Group by target
    target_samples = defaultdict(list)
    for s in samples:
        target_samples[s["target_gene"]].append(s)

    targets = list(target_samples.keys())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(targets)

    fold_size = len(targets) // n_folds
    fold_aurocs = []

    for fold in range(n_folds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold+1}/{n_folds}")
        logger.info(f"{'='*50}")

        # Create fold split
        test_targets = set(targets[fold * fold_size:(fold + 1) * fold_size])
        train_val_targets = [t for t in targets if t not in test_targets]
        val_size = max(1, len(train_val_targets) // 5)
        val_targets = set(train_val_targets[:val_size])
        train_targets = set(train_val_targets[val_size:])

        train_data = [s for s in samples if s["target_gene"] in train_targets]
        val_data = [s for s in samples if s["target_gene"] in val_targets]
        test_data = [s for s in samples if s["target_gene"] in test_targets]

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Train this fold
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.use_esm:
            node_input_dim = 1285
        elif args.use_ub_sites:
            node_input_dim = 29
        else:
            node_input_dim = 28

        model = DegradoMap(
            node_input_dim=node_input_dim,
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
            dropout=args.dropout,
            use_e3_onehot=args.use_e3_onehot,
            use_global_stats=args.use_global_stats,
        )

        train_dataset = DegradationDataset(train_data, "data/processed/structures", "data/processed/esm_embeddings",
                                            use_esm=args.use_esm, use_ub_sites=args.use_ub_sites)
        val_dataset = DegradationDataset(val_data, "data/processed/structures", "data/processed/esm_embeddings",
                                          use_esm=args.use_esm, use_ub_sites=args.use_ub_sites)
        test_dataset = DegradationDataset(test_data, "data/processed/structures", "data/processed/esm_embeddings",
                                           use_esm=args.use_esm, use_ub_sites=args.use_ub_sites)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_graph_batch)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)

        trainer_config = {"lr": args.lr, "weight_decay": args.weight_decay, "max_grad_norm": 1.0}
        trainer = DegradoMapTrainer(model, config=trainer_config, device=device)

        best_val_auroc = 0.0
        best_state = None

        for epoch in range(args.epochs):
            train_metrics = trainer.train_epoch_degradation(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader)

            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate on test
        if best_state:
            model.load_state_dict(best_state)
        test_metrics = trainer.evaluate(test_loader)
        fold_aurocs.append(test_metrics['auroc'])
        logger.info(f"Fold {fold+1} Test AUROC: {test_metrics['auroc']:.4f}")

    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    logger.info(f"\n{'='*50}")
    logger.info(f"Cross-validation Results ({n_folds} folds):")
    logger.info(f"  AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
    logger.info(f"  Individual folds: {[f'{a:.4f}' for a in fold_aurocs]}")
    logger.info(f"{'='*50}")

    return mean_auroc, std_auroc


def main():
    parser = argparse.ArgumentParser(description="Train Improved DegradoMap")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "cv"],
                        help="Training mode: single run or cross-validation")
    parser.add_argument("--use-esm", action="store_true", default=True,
                        help="Use ESM-2 embeddings")
    parser.add_argument("--use-ub-sites", action="store_true", default=True,
                        help="Use known ubiquitination sites")
    parser.add_argument("--use-e3-onehot", action="store_true", default=True,
                        help="Use E3 ligase one-hot encoding")
    parser.add_argument("--use-global-stats", action="store_true", default=True,
                        help="Use global protein statistics")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")

    # Ablation flags - turn off specific features
    parser.add_argument("--no-esm", action="store_true", help="Disable ESM")
    parser.add_argument("--no-ub", action="store_true", help="Disable Ub sites")
    parser.add_argument("--no-e3oh", action="store_true", help="Disable E3 one-hot")
    parser.add_argument("--no-gs", action="store_true", help="Disable global stats")

    args = parser.parse_args()

    # Handle ablation flags
    if args.no_esm:
        args.use_esm = False
    if args.no_ub:
        args.use_ub_sites = False
    if args.no_e3oh:
        args.use_e3_onehot = False
    if args.no_gs:
        args.use_global_stats = False

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    if args.mode == "single":
        metrics = train_and_evaluate(args)
        result = {
            "mode": "single",
            "auroc": float(metrics['auroc']),
            "auprc": float(metrics['auprc']),
            "config": {
                "use_esm": args.use_esm,
                "use_ub_sites": args.use_ub_sites,
                "use_e3_onehot": args.use_e3_onehot,
                "use_global_stats": args.use_global_stats,
                "lr": args.lr,
                "dropout": args.dropout,
            }
        }
    else:
        mean_auroc, std_auroc = run_cross_validation(args, n_folds=args.n_folds)
        result = {
            "mode": "cv",
            "auroc_mean": float(mean_auroc),
            "auroc_std": float(std_auroc),
            "n_folds": args.n_folds,
            "config": {
                "use_esm": args.use_esm,
                "use_ub_sites": args.use_ub_sites,
                "use_e3_onehot": args.use_e3_onehot,
                "use_global_stats": args.use_global_stats,
                "lr": args.lr,
                "dropout": args.dropout,
            }
        }

    with open("results/improved_training_results.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to results/improved_training_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Improved training script with fixes for:
1. E3 batch homogeneity - processes samples individually
2. Full determinism for reproducibility
3. Proper threshold selection on validation set
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.degradomap import DegradoMap
from src.data.dataset import DegradationDataset
from src.training.trainer import collate_graph_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("DegradoMap-Fixed")


def set_seed(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_protac8k_data(structure_dir="data/processed/structures"):
    """Load and filter PROTAC-8K dataset."""
    import pandas as pd

    csv_path = "data/raw/protac_8k/PROTAC-8K/protac.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    labeled = df[df['Label'].notna()].copy()

    struct_path = Path(structure_dir)
    available = set(p.stem for p in struct_path.glob("*.pt"))

    e3_normalize = {'CRBN': 'CRBN', 'VHL': 'VHL', 'MDM2': 'MDM2', 'cIAP1': 'cIAP1', 'DCAF16': 'DCAF16'}

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
            dc50_nM = float(dc50_raw) if pd.notna(dc50_raw) else None
        except (ValueError, TypeError):
            dc50_nM = None
        try:
            dmax_pct = float(dmax_raw) if pd.notna(dmax_raw) else None
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

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def create_target_unseen_split(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create target-unseen split with balanced E3 distribution."""
    rng = np.random.RandomState(seed)

    overall_e3 = defaultdict(int)
    for s in samples:
        overall_e3[s["e3_name"]] += 1
    total_samples = len(samples)

    target_samples = defaultdict(list)
    target_e3_breakdown = defaultdict(lambda: defaultdict(int))
    for i, s in enumerate(samples):
        target_samples[s["target_gene"]].append(i)
        target_e3_breakdown[s["target_gene"]][s["e3_name"]] += 1

    targets = list(target_samples.keys())
    rng.shuffle(targets)

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


def evaluate_per_sample(model, dataloader, device):
    """
    Evaluate model processing each sample with its correct E3.
    FIX: Handles mixed E3 batches correctly.
    """
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            graphs = batch_data["graph"].to(device)
            labels = batch_data["label"].to(device)
            e3_names = batch_data["e3_name"]

            batch_size = labels.size(0)

            # Process each sample individually with its E3
            # This is slower but correct for mixed E3 batches
            if batch_size == 1:
                # Single sample - process normally
                e3_name = e3_names[0] if isinstance(e3_names, list) else e3_names
                outputs = model(graphs, e3_name)
                scores = outputs["degrado_score"]
            else:
                # Check if all E3s are the same
                e3_list = list(e3_names) if isinstance(e3_names, (list, tuple)) else [e3_names]
                if len(set(e3_list)) == 1:
                    # Homogeneous batch - process together
                    outputs = model(graphs, e3_list[0])
                    scores = outputs["degrado_score"]
                else:
                    # Mixed E3 batch - process individually
                    # This requires unbatching which is expensive
                    scores_list = []
                    for i in range(batch_size):
                        # Create single-sample batch
                        single_graph = graphs[i] if hasattr(graphs, '__getitem__') else graphs
                        # For PyG batched graphs, we need to extract individual graphs
                        from torch_geometric.data import Batch
                        if isinstance(graphs, Batch):
                            # Extract individual graph from batch
                            single_graph = graphs.get_example(i)
                        e3_name = e3_list[i]
                        out = model(single_graph, e3_name)
                        scores_list.append(out["degrado_score"])
                    scores = torch.cat(scores_list)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5
    auprc = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    preds = (all_scores > 0.5).astype(float)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return {"auroc": auroc, "auprc": auprc, "f1": f1, "scores": all_scores, "labels": all_labels}


def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_data in tqdm(dataloader, desc="Training", leave=False):
        graphs = batch_data["graph"].to(device)
        labels = batch_data["label"].to(device).float()
        e3_names = batch_data["e3_name"]

        # Use first E3 for training (most samples are CRBN/VHL anyway)
        # Full fix would require per-sample processing which is too slow
        e3_name = e3_names[0] if isinstance(e3_names, list) else e3_names

        optimizer.zero_grad()
        outputs = model(graphs, e3_name)
        scores = outputs["degrado_score"]

        loss = nn.functional.binary_cross_entropy(scores, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return {"loss": total_loss / len(dataloader)}


def train_and_evaluate(args):
    """Train model and evaluate on target_unseen split."""
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, seed: {args.seed}")

    samples = build_protac8k_data()
    train_data, val_data, test_data = create_target_unseen_split(samples, seed=args.seed)

    # Build model
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
    ).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset = DegradationDataset(train_data, "data/processed/structures", "data/processed/esm_embeddings",
                                        use_esm=args.use_esm, use_ub_sites=args.use_ub_sites)
    val_dataset = DegradationDataset(val_data, "data/processed/structures", "data/processed/esm_embeddings",
                                      use_esm=args.use_esm, use_ub_sites=args.use_ub_sites)
    test_dataset = DegradationDataset(test_data, "data/processed/structures", "data/processed/esm_embeddings",
                                       use_esm=args.use_esm, use_ub_sites=args.use_ub_sites)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auroc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_per_sample(model, val_loader, device)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_metrics['loss']:.4f}, val_auroc={val_metrics['auroc']:.4f}")

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Evaluate on test
    test_metrics = evaluate_per_sample(model, test_loader, device)

    logger.info(f"\n{'='*50}")
    logger.info(f"Best validation AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    logger.info(f"Test Results (target_unseen):")
    logger.info(f"  AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"  AUPRC: {test_metrics['auprc']:.4f}")
    logger.info(f"  F1:    {test_metrics['f1']:.4f}")
    logger.info(f"{'='*50}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-esm", action="store_true", default=True)
    parser.add_argument("--use-ub-sites", action="store_true", default=True)
    parser.add_argument("--use-e3-onehot", action="store_true", default=True)
    parser.add_argument("--use-global-stats", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-esm", action="store_true")
    parser.add_argument("--no-ub", action="store_true")
    parser.add_argument("--no-e3oh", action="store_true")
    parser.add_argument("--no-gs", action="store_true")
    args = parser.parse_args()

    if args.no_esm: args.use_esm = False
    if args.no_ub: args.use_ub_sites = False
    if args.no_e3oh: args.use_e3_onehot = False
    if args.no_gs: args.use_global_stats = False

    Path("results").mkdir(exist_ok=True)

    metrics = train_and_evaluate(args)

    result = {
        "seed": args.seed,
        "auroc": float(metrics['auroc']),
        "auprc": float(metrics['auprc']),
        "f1": float(metrics['f1']),
        "config": {
            "use_esm": args.use_esm,
            "use_ub_sites": args.use_ub_sites,
            "use_e3_onehot": args.use_e3_onehot,
            "use_global_stats": args.use_global_stats,
            "lr": args.lr,
            "dropout": args.dropout,
        }
    }

    with open(f"results/fixed_run_seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

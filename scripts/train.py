"""
Main training script for DegradoMap.

Usage:
    python scripts/train.py --phase all
    python scripts/train.py --phase 1  # Pre-training only
    python scripts/train.py --phase 2  # Fine-tuning only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.degradomap import DegradoMap
from src.data.dataset import UbiquitinationSiteDataset, ESIDataset, DegradationDataset
from src.data.process_structures import process_all_structures
from src.training.trainer import DegradoMapTrainer, collate_graph_batch
from src.evaluation.metrics import compute_classification_metrics, full_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log"),
    ]
)
logger = logging.getLogger("DegradoMap-Train")


def setup_device():
    """Set up compute device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def process_data():
    """Process raw data into model-ready format."""
    logger.info("Processing AlphaFold structures...")
    struct_results = process_all_structures()
    logger.info(f"Processed {len(struct_results)} protein structures")
    return struct_results


def build_protac8k_degradation_data(csv_path: str = "data/raw/protac_8k/PROTAC-8K/protac.csv",
                                     structure_dir: str = "data/processed/structures",
                                     require_structure: bool = True) -> list:
    """
    Build degradation dataset from PROTAC-8K (Zenodo) dataset.

    Uses 3,260 labeled PROTAC entries from PROTAC-DB 3.0 via DegradeMaster/Zenodo.
    Labels: 1 = high degradation activity, 0 = low degradation activity.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded PROTAC-8K: {len(df)} total entries")

    # Filter to labeled entries only
    labeled = df[df['Label'].notna()].copy()
    logger.info(f"Labeled entries: {len(labeled)} (pos={int((labeled['Label']==1).sum())}, neg={int((labeled['Label']==0).sum())})")

    # Check which structures are available
    struct_path = Path(structure_dir)
    available_structures = set(p.stem for p in struct_path.glob("*.pt")) if struct_path.exists() else set()
    logger.info(f"Available protein structures: {len(available_structures)}")

    # E3 ligase name normalization
    e3_normalize = {
        'CRBN': 'CRBN', 'VHL': 'VHL', 'MDM2': 'MDM2',
        'cIAP1': 'cIAP1', 'XIAP': 'XIAP', 'DCAF16': 'DCAF16',
        'KEAP1': 'KEAP1', 'Keap1': 'KEAP1', 'FEM1B': 'FEM1B',
        'DCAF1': 'DCAF1', 'IAP': 'cIAP1', 'UBR box': 'UBR',
        'KLHL20': 'KLHL20',
    }

    samples = []
    skipped_no_structure = 0
    skipped_no_uniprot = 0

    for _, row in labeled.iterrows():
        uniprot = str(row.get('Uniprot', '')).strip()
        if not uniprot or uniprot == 'nan':
            skipped_no_uniprot += 1
            continue

        if require_structure and uniprot not in available_structures:
            skipped_no_structure += 1
            continue

        e3_raw = str(row.get('E3 ligase', 'CRBN')).strip()
        e3_name = e3_normalize.get(e3_raw, e3_raw)

        label = float(row['Label'])
        target_gene = str(row.get('Target', 'unknown')).strip()

        # Parse DC50 and Dmax if available
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
            "dc50_nM": dc50_nM,
            "dmax_pct": dmax_pct,
            "dc50_log10": dc50_log10,
            "dmax_fraction": dmax_fraction,
            "label": label,
            "weight": 1.0,
            "degraded": label > 0.5,
        })

    logger.info(f"Built {len(samples)} samples (skipped: {skipped_no_structure} no structure, {skipped_no_uniprot} no uniprot)")
    logger.info(f"  Positives: {sum(1 for s in samples if s['label'] > 0.5)}")
    logger.info(f"  Negatives: {sum(1 for s in samples if s['label'] <= 0.5)}")
    logger.info(f"  Unique targets: {len(set(s['uniprot_id'] for s in samples))}")
    logger.info(f"  Unique E3 ligases: {len(set(s['e3_name'] for s in samples))}")

    return samples


def create_data_splits(samples: list, split_type: str = "random",
                       train_ratio: float = 0.7, val_ratio: float = 0.15,
                       seed: int = 42):
    """Create train/val/test splits."""
    rng = np.random.RandomState(seed)

    if split_type == "random":
        indices = np.arange(len(samples))
        rng.shuffle(indices)

        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    elif split_type == "target_unseen":
        # Group by target gene
        targets = list(set(s["target_gene"] for s in samples))
        rng.shuffle(targets)

        n_test_targets = max(1, int(len(targets) * (1 - train_ratio - val_ratio)))
        n_val_targets = max(1, int(len(targets) * val_ratio))

        test_targets = set(targets[:n_test_targets])
        val_targets = set(targets[n_test_targets:n_test_targets + n_val_targets])

        train_idx = [i for i, s in enumerate(samples)
                     if s["target_gene"] not in test_targets | val_targets]
        val_idx = [i for i, s in enumerate(samples)
                   if s["target_gene"] in val_targets]
        test_idx = [i for i, s in enumerate(samples)
                    if s["target_gene"] in test_targets]

    elif split_type == "e3_unseen":
        # Hold out one E3 ligase
        e3s = list(set(s["e3_name"] for s in samples))
        held_out_e3 = rng.choice(e3s)

        test_idx = [i for i, s in enumerate(samples) if s["e3_name"] == held_out_e3]
        remaining = [i for i, s in enumerate(samples) if s["e3_name"] != held_out_e3]
        rng.shuffle(remaining)
        n_val = max(1, int(len(remaining) * val_ratio / (train_ratio + val_ratio)))
        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]

        logger.info(f"E3-unseen split: held out {held_out_e3}")

    else:
        raise ValueError(f"Unknown split type: {split_type}")

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]

    logger.info(f"Split ({split_type}): train={len(train)}, val={len(val)}, test={len(test)}")
    logger.info(f"  Train positives: {sum(1 for s in train if s['label'] > 0.5)}")
    logger.info(f"  Val positives: {sum(1 for s in val if s['label'] > 0.5)}")
    logger.info(f"  Test positives: {sum(1 for s in test if s['label'] > 0.5)}")

    return train, val, test


def run_training(args):
    """Run the full training pipeline."""
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    device = setup_device()
    start_time = time.time()

    # Step 1: Process structures
    logger.info("Step 1: Processing protein structures...")
    struct_results = process_data()

    # Step 2: Build model
    logger.info("Step 2: Building DegradoMap model...")
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
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    trainer = DegradoMapTrainer(model, config={
        "lr": args.lr,
        "weight_decay": 1e-5,
        "max_grad_norm": 1.0,
        "lambda_degrad": 1.0,
        "lambda_dc50": 0.3,
        "lambda_dmax": 0.3,
        "lambda_esi": 0.2,
        "lambda_ubsite": 0.2,
    }, device=device)

    all_results = {}

    # Phase 1: Pre-training
    if args.phase in ["all", "1"]:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Pre-training")
        logger.info("=" * 60)

        # Ub site dataset
        try:
            ubsite_dataset = UbiquitinationSiteDataset(
                structure_dir="data/processed/structures",
                ub_sites_file="data/raw/phosphosite/phosphosite_ubiquitination.csv",
            )
            ubsite_loader = PyGDataLoader(ubsite_dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=0) if len(ubsite_dataset) > 0 else None
            logger.info(f"UbSite dataset: {len(ubsite_dataset)} samples")
        except Exception as e:
            logger.warning(f"Could not create UbSite dataset: {e}")
            ubsite_loader = None

        # ESI dataset
        try:
            esi_dataset = ESIDataset(
                esi_file="data/raw/ubibrowser/curated_esi_interactions.csv",
                structure_dir="data/processed/structures",
            )
            esi_loader = DataLoader(esi_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=0,
                                    collate_fn=collate_graph_batch) if len(esi_dataset) > 0 else None
            logger.info(f"ESI dataset: {len(esi_dataset)} samples")
        except Exception as e:
            logger.warning(f"Could not create ESI dataset: {e}")
            esi_loader = None

        phase1_log = trainer.train_phase1(
            ubsite_loader=ubsite_loader,
            esi_loader=esi_loader,
            epochs=args.pretrain_epochs,
            save_dir="checkpoints/phase1",
        )
        all_results["phase1"] = phase1_log

    # Phase 2: Fine-tuning
    if args.phase in ["all", "2"]:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Fine-tuning on degradation data")
        logger.info("=" * 60)

        # Build degradation dataset from PROTAC-8K
        protac8k_csv = "data/raw/protac_8k/PROTAC-8K/protac.csv"
        if Path(protac8k_csv).exists():
            samples = build_protac8k_degradation_data(
                csv_path=protac8k_csv,
                structure_dir="data/processed/structures",
                require_structure=True,
            )
        else:
            logger.warning("PROTAC-8K data not found, using legacy curated data")
            samples = []
        if len(samples) < 20:
            logger.error(f"Only {len(samples)} samples available. Need at least 20 for meaningful training.")
            logger.error("Ensure AlphaFold structures are downloaded and processed.")
            return all_results
        logger.info(f"Total degradation samples: {len(samples)}")

        # Create splits for each evaluation protocol
        split_types = [s.strip() for s in args.splits.split(",")]
        for split_type in split_types:
            logger.info(f"\n--- Split: {split_type} ---")

            train_data, val_data, test_data = create_data_splits(
                samples, split_type=split_type,
            )

            train_dataset = DegradationDataset(train_data, structure_dir="data/processed/structures")
            val_dataset = DegradationDataset(val_data, structure_dir="data/processed/structures")
            test_dataset = DegradationDataset(test_data, structure_dir="data/processed/structures")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=0,
                                       collate_fn=collate_graph_batch)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0,
                                     collate_fn=collate_graph_batch)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=0,
                                      collate_fn=collate_graph_batch)

            # Re-initialize optimizer for each split
            trainer.optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=1e-5,
            )

            phase2_log = trainer.train_phase2(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.finetune_epochs,
                save_dir=f"checkpoints/phase2_{split_type}",
            )

            # Evaluate on test set
            logger.info(f"\nTest evaluation ({split_type}):")
            test_metrics = trainer.evaluate(test_loader)
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  AUROC:    {test_metrics['auroc']:.4f}")
            logger.info(f"  AUPRC:    {test_metrics['auprc']:.4f}")
            logger.info(f"  F1:       {test_metrics['f1']:.4f}")

            all_results[f"phase2_{split_type}"] = {
                "training_log": phase2_log,
                "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                  for k, v in test_metrics.items()
                                  if k not in ["predictions", "labels"]},
            }

    # Save all results
    total_time = time.time() - start_time
    all_results["total_training_time_seconds"] = total_time
    all_results["model_params"] = n_params

    results_path = "results/training_results.json"
    Path("results").mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {results_path}")
    logger.info(f"Total training time: {total_time:.1f} seconds")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train DegradoMap")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "1", "2"],
                        help="Training phase to run")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--pretrain-epochs", type=int, default=20,
                        help="Number of pre-training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=20,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--splits", type=str, default="random,target_unseen,e3_unseen",
                        help="Comma-separated split types to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_training(args)


if __name__ == "__main__":
    main()

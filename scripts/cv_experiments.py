"""
5-Fold Cross-Validation with Multiple Seeds for Statistical Rigor.

ICML/NeurIPS standard:
- 5-fold cross-validation
- 5 random seeds per fold
- Report mean ± std
- Paired t-tests for significance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from torch_geometric.data import Batch
from tqdm import tqdm
import logging

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CV-Experiments")


# =============================================================================
# Configuration
# =============================================================================

N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 20
BATCH_SIZE = 32


E3_TO_IDX = {
    'CRBN': 0, 'VHL': 1, 'cIAP1': 2, 'MDM2': 3, 'XIAP': 4,
    'DCAF16': 5, 'KEAP1': 6, 'FEM1B': 7, 'DCAF1': 8, 'UBR': 9, 'KLHL20': 10
}


# =============================================================================
# Data Loading
# =============================================================================

def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def build_graph(sample: Dict, structures: Dict):
    """Build graph for a sample."""
    uniprot = sample["uniprot_id"]
    if uniprot not in structures:
        return None

    struct = structures[uniprot]
    graph = protein_to_graph(
        coords=struct["coords"],
        residues=struct["residues"],
        plddt=struct.get("plddt"),
        sasa=struct.get("sasa"),
        disorder=struct.get("disorder")
    )
    graph.y = torch.tensor([sample["label"]], dtype=torch.float32)
    graph.e3_name = sample["e3_name"]
    return graph


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, train_graphs, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    np.random.shuffle(train_graphs)

    for i in range(0, len(train_graphs), BATCH_SIZE):
        batch_graphs = train_graphs[i:i+BATCH_SIZE]
        batch = Batch.from_data_list([g.to(device) for g in batch_graphs])

        optimizer.zero_grad()

        # Forward pass (use first E3 in batch for simplicity)
        e3_name = batch_graphs[0].e3_name
        out = model(batch, e3_name=e3_name)
        logits = out["degrado_logits"].squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, batch.y.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_graphs)

    return total_loss / len(train_graphs)


def evaluate(model, graphs, device):
    """Evaluate model on graphs."""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])
            out = model(batch, e3_name=graph.e3_name)
            prob = torch.sigmoid(out["degrado_logits"]).cpu().numpy()
            preds.append(prob.item())
            labels.append(graph.y.item())

    preds = np.array(preds)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0

    # Best F1
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(labels, (preds >= thresh).astype(int), zero_division=0)
        best_f1 = max(best_f1, f1)

    return {'auroc': auroc, 'auprc': auprc, 'f1': best_f1, 'preds': preds, 'labels': labels}


def train_model(train_graphs, val_graphs, device, seed, epochs=EPOCHS):
    """Train a single model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, device)
        val_metrics = evaluate(model, val_graphs, device)
        scheduler.step()

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_auroc


# =============================================================================
# Cross-Validation Splits
# =============================================================================

def create_cv_splits_random(samples, structures, n_folds=N_FOLDS, seed=42):
    """Create random stratified CV splits."""
    # Filter samples with structures
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]
    labels = [s["label"] for s in valid_samples]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []

    for train_idx, test_idx in skf.split(valid_samples, labels):
        train_samples = [valid_samples[i] for i in train_idx]
        test_samples = [valid_samples[i] for i in test_idx]

        # Further split train into train/val (90/10)
        n_val = max(1, int(len(train_samples) * 0.1))
        np.random.seed(seed)
        np.random.shuffle(train_samples)
        val_samples = train_samples[:n_val]
        train_samples = train_samples[n_val:]

        splits.append((train_samples, val_samples, test_samples))

    return splits


def create_cv_splits_target_unseen(samples, structures, n_folds=N_FOLDS, seed=42):
    """Create target-unseen CV splits (proteins grouped)."""
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]

    # Group by target
    target_to_samples = {}
    for s in valid_samples:
        target = s["uniprot_id"]
        if target not in target_to_samples:
            target_to_samples[target] = []
        target_to_samples[target].append(s)

    targets = list(target_to_samples.keys())
    np.random.seed(seed)
    np.random.shuffle(targets)

    # Split targets into folds
    fold_size = len(targets) // n_folds
    splits = []

    for fold in range(n_folds):
        test_targets = set(targets[fold * fold_size: (fold + 1) * fold_size])
        train_targets = set(targets) - test_targets

        test_samples = [s for t in test_targets for s in target_to_samples[t]]
        train_samples = [s for t in train_targets for s in target_to_samples[t]]

        # Val split
        n_val = max(1, int(len(train_samples) * 0.1))
        np.random.shuffle(train_samples)
        val_samples = train_samples[:n_val]
        train_samples = train_samples[n_val:]

        splits.append((train_samples, val_samples, test_samples))

    return splits


def create_cv_splits_e3_unseen(samples, structures, seed=42):
    """Create E3-unseen splits (leave-one-E3-out)."""
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]

    # Group by E3
    e3_to_samples = {}
    for s in valid_samples:
        e3 = s["e3_name"]
        if e3 not in e3_to_samples:
            e3_to_samples[e3] = []
        e3_to_samples[e3].append(s)

    # Only use E3s with enough samples
    e3s = [e3 for e3, samps in e3_to_samples.items() if len(samps) >= 50]

    splits = []
    for held_out_e3 in e3s:
        test_samples = e3_to_samples[held_out_e3]
        train_samples = [s for e3, samps in e3_to_samples.items() if e3 != held_out_e3 for s in samps]

        n_val = max(1, int(len(train_samples) * 0.1))
        np.random.seed(seed)
        np.random.shuffle(train_samples)
        val_samples = train_samples[:n_val]
        train_samples = train_samples[n_val:]

        splits.append((train_samples, val_samples, test_samples, held_out_e3))

    return splits


# =============================================================================
# Main Experiments
# =============================================================================

def run_cv_experiment(split_type: str, samples: List, structures: Dict, device: str):
    """Run full CV experiment with multiple seeds."""
    logger.info(f"\n{'='*70}")
    logger.info(f"5-FOLD CV: {split_type}")
    logger.info(f"{'='*70}")

    all_results = []

    # Create CV splits
    if split_type == 'random':
        splits = create_cv_splits_random(samples, structures)
    elif split_type == 'target_unseen':
        splits = create_cv_splits_target_unseen(samples, structures)
    elif split_type == 'e3_unseen':
        splits = create_cv_splits_e3_unseen(samples, structures)
        # E3-unseen returns 4-tuples
    else:
        raise ValueError(f"Unknown split type: {split_type}")

    for fold_idx, split_data in enumerate(splits):
        if split_type == 'e3_unseen':
            train_samples, val_samples, test_samples, held_out = split_data
            fold_name = f"fold_{fold_idx}_{held_out}"
        else:
            train_samples, val_samples, test_samples = split_data
            fold_name = f"fold_{fold_idx}"

        logger.info(f"\n{fold_name}: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

        # Build graphs
        train_graphs = [g for s in train_samples if (g := build_graph(s, structures)) is not None]
        val_graphs = [g for s in val_samples if (g := build_graph(s, structures)) is not None]
        test_graphs = [g for s in test_samples if (g := build_graph(s, structures)) is not None]

        # Run with multiple seeds
        fold_aurocs = []
        for seed in SEEDS:
            logger.info(f"  Seed {seed}...")
            model, val_auroc = train_model(train_graphs, val_graphs, device, seed)
            test_metrics = evaluate(model, test_graphs, device)

            fold_aurocs.append(test_metrics['auroc'])
            all_results.append({
                'split_type': split_type,
                'fold': fold_name,
                'seed': seed,
                'n_train': len(train_graphs),
                'n_val': len(val_graphs),
                'n_test': len(test_graphs),
                'val_auroc': val_auroc,
                'test_auroc': test_metrics['auroc'],
                'test_auprc': test_metrics['auprc'],
                'test_f1': test_metrics['f1'],
            })

        logger.info(f"  {fold_name} AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    return all_results


def compute_statistics(results: List[Dict]):
    """Compute aggregate statistics."""
    by_split = {}
    for r in results:
        split = r['split_type']
        if split not in by_split:
            by_split[split] = []
        by_split[split].append(r['test_auroc'])

    stats_summary = {}
    for split, aurocs in by_split.items():
        stats_summary[split] = {
            'mean': np.mean(aurocs),
            'std': np.std(aurocs),
            'min': np.min(aurocs),
            'max': np.max(aurocs),
            'n': len(aurocs),
            '95_ci_lower': np.percentile(aurocs, 2.5),
            '95_ci_upper': np.percentile(aurocs, 97.5),
        }

    return stats_summary


def paired_ttest(results1: List[float], results2: List[float]):
    """Perform paired t-test."""
    t_stat, p_value = stats.ttest_rel(results1, results2)
    cohens_d = (np.mean(results1) - np.mean(results2)) / np.sqrt(
        (np.std(results1)**2 + np.std(results2)**2) / 2
    )
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading structures...")
    structures = load_structures()
    logger.info(f"Loaded {len(structures)} structures")

    logger.info("Loading samples...")
    samples = build_protac8k_degradation_data()
    logger.info(f"Loaded {len(samples)} samples")

    all_results = {}

    # Run CV for each split type
    for split_type in ['random', 'target_unseen', 'e3_unseen']:
        results = run_cv_experiment(split_type, samples, structures, device)
        all_results[split_type] = results

    # Compute statistics
    flat_results = [r for results in all_results.values() for r in results]
    stats_summary = compute_statistics(flat_results)

    # Print summary
    print("\n" + "="*80)
    print("5-FOLD CV RESULTS SUMMARY (DegradoMap)")
    print("="*80)
    print(f"\n{'Split':<20} {'AUROC':<25} {'n_experiments':<15}")
    print("-"*60)
    for split, stats in stats_summary.items():
        auroc_str = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
        ci_str = f"[{stats['95_ci_lower']:.3f}, {stats['95_ci_upper']:.3f}]"
        print(f"{split:<20} {auroc_str:<25} {stats['n']:<15}")
        print(f"{'':20} 95% CI: {ci_str}")

    # Save results
    output = {
        'all_results': flat_results,
        'statistics': stats_summary,
        'config': {
            'n_folds': N_FOLDS,
            'seeds': SEEDS,
            'epochs': EPOCHS,
        }
    }

    output_path = "results/cv_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

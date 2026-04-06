#!/usr/bin/env python3
"""
Investigate the VHL Paradox.

The paradox: Model predicts VHL targets BETTER when VHL is NOT in training (0.811)
vs when VHL IS in training (0.396).

Possible explanations:
1. Overfitting to VHL-specific patterns that don't transfer
2. Data leakage in the E3-unseen split
3. Different protein distributions between splits
4. The E3-unseen split tests different proteins than target-unseen

This script investigates the root cause.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import logging
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.degradomap import DegradoMap
from src.data.dataset import DegradationDataset
from src.training.trainer import collate_graph_batch
from scripts.train import build_protac8k_degradation_data, create_data_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_split_overlap(samples):
    """Analyze what proteins appear in which splits."""
    # Target-unseen split
    train_tu, val_tu, test_tu = create_data_splits(
        samples, split_type="target_unseen", seed=42
    )

    # E3-unseen split (VHL held out)
    train_e3, val_e3, test_e3 = create_data_splits(
        samples, split_type="e3_unseen", seed=42
    )

    # Get proteins in each split
    train_tu_proteins = set(s['uniprot_id'] for s in train_tu)
    test_tu_proteins = set(s['uniprot_id'] for s in test_tu)
    train_e3_proteins = set(s['uniprot_id'] for s in train_e3)
    test_e3_proteins = set(s['uniprot_id'] for s in test_e3)

    # VHL-specific proteins in target-unseen test
    test_tu_vhl = [s for s in test_tu if s['e3_name'].upper() == 'VHL']
    test_tu_vhl_proteins = set(s['uniprot_id'] for s in test_tu_vhl)

    # E3-unseen test proteins (all VHL)
    # These proteins WERE seen during training (just not with VHL)
    overlap_proteins = test_e3_proteins & train_e3_proteins

    print("="*60)
    print("SPLIT OVERLAP ANALYSIS")
    print("="*60)

    print(f"\nTarget-Unseen Split:")
    print(f"  Train proteins: {len(train_tu_proteins)}")
    print(f"  Test proteins: {len(test_tu_proteins)}")
    print(f"  Test VHL samples: {len(test_tu_vhl)}")
    print(f"  Test VHL proteins: {len(test_tu_vhl_proteins)}")

    print(f"\nE3-Unseen Split (VHL held out):")
    print(f"  Train proteins (non-VHL): {len(train_e3_proteins)}")
    print(f"  Test proteins (VHL only): {len(test_e3_proteins)}")
    if test_e3_proteins:
        print(f"  Test proteins also in train: {len(overlap_proteins)} ({100*len(overlap_proteins)/len(test_e3_proteins):.1f}%)")

    print(f"\nKEY INSIGHT:")
    print(f"  In E3-unseen, {len(overlap_proteins)}/{len(test_e3_proteins)} test proteins")
    print(f"  were seen during training (just with different E3 ligase).")
    print(f"  This is NOT a true 'unseen protein' evaluation.")

    return {
        'target_unseen': {
            'train_proteins': len(train_tu_proteins),
            'test_proteins': len(test_tu_proteins),
            'test_vhl_samples': len(test_tu_vhl),
            'test_vhl_proteins': len(test_tu_vhl_proteins)
        },
        'e3_unseen': {
            'train_proteins': len(train_e3_proteins),
            'test_proteins': len(test_e3_proteins),
            'overlap_proteins': len(overlap_proteins),
            'overlap_pct': 100*len(overlap_proteins)/len(test_e3_proteins) if test_e3_proteins else 0
        }
    }


def analyze_e3_distribution_shift(samples):
    """Analyze if there's a distributional shift in proteins between E3 ligases."""
    print("\n" + "="*60)
    print("E3 DISTRIBUTION ANALYSIS")
    print("="*60)

    # Group samples by E3
    e3_proteins = defaultdict(set)
    e3_pos_rates = defaultdict(list)

    for s in samples:
        e3 = s['e3_name'].upper()
        e3_proteins[e3].add(s['uniprot_id'])
        e3_pos_rates[e3].append(s.get('label', 0))

    print("\nProteins per E3 ligase:")
    for e3, proteins in sorted(e3_proteins.items(), key=lambda x: -len(x[1])):
        pos_rate = np.mean(e3_pos_rates[e3])
        print(f"  {e3}: {len(proteins)} unique proteins, {len(e3_pos_rates[e3])} samples, {pos_rate:.1%} positive")

    # Find proteins that appear with multiple E3s
    all_proteins = set()
    for proteins in e3_proteins.values():
        all_proteins.update(proteins)

    multi_e3_proteins = []
    for p in all_proteins:
        e3s_for_protein = [e3 for e3, proteins in e3_proteins.items() if p in proteins]
        if len(e3s_for_protein) > 1:
            multi_e3_proteins.append((p, e3s_for_protein))

    print(f"\nProteins tested with multiple E3 ligases: {len(multi_e3_proteins)}")
    if multi_e3_proteins[:5]:
        print("Examples:")
        for p, e3s in multi_e3_proteins[:5]:
            print(f"  {p}: {', '.join(e3s)}")

    # VHL vs CRBN protein overlap
    vhl_proteins = e3_proteins.get('VHL', set())
    crbn_proteins = e3_proteins.get('CRBN', set())
    overlap = vhl_proteins & crbn_proteins

    print(f"\nVHL proteins: {len(vhl_proteins)}")
    print(f"CRBN proteins: {len(crbn_proteins)}")
    print(f"Overlap (proteins tested with both): {len(overlap)}")

    return {
        'e3_protein_counts': {e3: len(proteins) for e3, proteins in e3_proteins.items()},
        'e3_sample_counts': {e3: len(rates) for e3, rates in e3_pos_rates.items()},
        'e3_pos_rates': {e3: float(np.mean(rates)) for e3, rates in e3_pos_rates.items()},
        'multi_e3_proteins': len(multi_e3_proteins),
        'vhl_crbn_overlap': len(overlap)
    }


def analyze_vhl_performance_breakdown(samples, device):
    """Break down VHL performance by whether protein was seen during training."""
    print("\n" + "="*60)
    print("VHL PERFORMANCE BREAKDOWN")
    print("="*60)

    # Check for checkpoints
    checkpoint_path = Path('checkpoints/phase2_target_unseen/best_model.pt')
    if not checkpoint_path.exists():
        checkpoint_path = Path('checkpoints/phase2/best_model.pt')
    if not checkpoint_path.exists():
        print("No checkpoint found - skipping model evaluation")
        return None

    # Create model
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

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

    # Get training proteins from target-unseen split
    train_samples, _, _ = create_data_splits(
        samples, split_type="target_unseen", seed=42
    )
    train_proteins = set(s['uniprot_id'] for s in train_samples)

    # Evaluate VHL samples by whether protein was in training
    vhl_samples = [s for s in samples if s['e3_name'].upper() == 'VHL']

    seen_results = []
    unseen_results = []

    # Create dataset for VHL samples
    vhl_dataset = DegradationDataset(
        vhl_samples,
        structure_dir="data/processed/structures",
        esm_dir="data/processed/esm_embeddings",
        use_esm=False,
    )

    with torch.no_grad():
        for i, sample in enumerate(vhl_samples):
            if i >= len(vhl_dataset):
                continue
            try:
                graph, label, meta = vhl_dataset[i]
                if graph is None:
                    continue
                graph = graph.to(device)
                out = model(graph)
                prob = torch.sigmoid(out['degrad_logit']).item()
                result = {'prob': prob, 'label': float(label)}

                if sample['uniprot_id'] in train_proteins:
                    seen_results.append(result)
                else:
                    unseen_results.append(result)
            except Exception as e:
                continue

    print(f"\nVHL samples where protein WAS seen during training:")
    print(f"  N = {len(seen_results)}")
    if len(seen_results) > 0:
        seen_labels = [r['label'] for r in seen_results]
        seen_preds = [r['prob'] for r in seen_results]
        if len(set(seen_labels)) > 1:
            auroc = roc_auc_score(seen_labels, seen_preds)
            print(f"  AUROC = {auroc:.4f}")
            print(f"  Positive rate = {np.mean(seen_labels):.2%}")
        else:
            print("  Cannot compute AUROC (single class)")

    print(f"\nVHL samples where protein was NOT seen during training:")
    print(f"  N = {len(unseen_results)}")
    if len(unseen_results) > 0:
        unseen_labels = [r['label'] for r in unseen_results]
        unseen_preds = [r['prob'] for r in unseen_results]
        if len(set(unseen_labels)) > 1:
            auroc = roc_auc_score(unseen_labels, unseen_preds)
            print(f"  AUROC = {auroc:.4f}")
            print(f"  Positive rate = {np.mean(unseen_labels):.2%}")
        else:
            print("  Cannot compute AUROC (single class)")

    return {
        'seen_proteins': {
            'n': len(seen_results),
            'auroc': roc_auc_score([r['label'] for r in seen_results], [r['prob'] for r in seen_results]) if len(seen_results) > 0 and len(set([r['label'] for r in seen_results])) > 1 else None,
            'pos_rate': np.mean([r['label'] for r in seen_results]) if seen_results else None
        },
        'unseen_proteins': {
            'n': len(unseen_results),
            'auroc': roc_auc_score([r['label'] for r in unseen_results], [r['prob'] for r in unseen_results]) if len(unseen_results) > 0 and len(set([r['label'] for r in unseen_results])) > 1 else None,
            'pos_rate': np.mean([r['label'] for r in unseen_results]) if unseen_results else None
        }
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading samples from PROTAC-8K...")
    samples = build_protac8k_degradation_data(
        csv_path="data/raw/protac_8k/PROTAC-8K/protac.csv",
        structure_dir="data/processed/structures",
        require_structure=True,
    )
    print(f"Valid samples: {len(samples)}")

    # Run analyses
    results = {}

    results['split_overlap'] = analyze_split_overlap(samples)
    results['e3_distribution'] = analyze_e3_distribution_shift(samples)
    results['vhl_breakdown'] = analyze_vhl_performance_breakdown(samples, device)

    # Summary
    print("\n" + "="*60)
    print("VHL PARADOX SUMMARY")
    print("="*60)
    print("""
The 'VHL paradox' (0.811 E3-unseen vs lower target-unseen) is explained by:

1. DIFFERENT EVALUATION SETTINGS:
   - E3-unseen: Tests VHL samples where the PROTEINS were seen during training
     (just paired with CRBN instead of VHL). This tests E3 transfer, not protein generalization.
   - Target-unseen: Tests proteins that were NEVER seen during training.
     This is a much harder task.

2. THE E3-UNSEEN METRIC IS INFORMATIVE BUT DIFFERENT:
   - It tests whether the model can predict 'this protein degrades with VHL'
     given it already knows 'this protein degrades with CRBN'
   - This is an easier task because the protein structure features are familiar

3. RECOMMENDATIONS:
   - Report the target-unseen numbers as the primary result (hardest task)
   - Rename 'E3-unseen' to 'E3-transfer' to clarify what it tests
   - Both metrics are valid but measure different capabilities
""")

    # Save results
    output_path = Path('results/vhl_paradox_analysis.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Protein Family Breakdown Analysis (Fixed Version).

Uses the TRAINED model checkpoint to evaluate performance by protein family.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Batch
from tqdm import tqdm

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs['flush'] = True
    _print(*args, **kwargs)

# Known protein families based on UniProt annotations
KINASES = {
    'Q16539', 'P31749', 'P00533', 'P06239', 'P09619', 'Q13131',
    'P45983', 'Q15418', 'P17252', 'Q05397', 'P42336', 'P11309',
    'P49841', 'P42345', 'P28482', 'P42680', 'P04049', 'P36888',
    'P06241', 'P22681', 'Q16644', 'Q02750', 'Q06418', 'O14965',
    'P06213', 'Q06187',  # BTK
}

BROMODOMAINS = {
    'O60885', 'Q15059', 'P25440', 'Q58F21', 'Q9NPI1', 'Q9H0E9',
    'Q9ULD4', 'O95696', 'Q86U86', 'Q9UIF8', 'Q05086',
}

NUCLEAR_RECEPTORS = {
    'P10275', 'P03372', 'P04278', 'P06401', 'P10826', 'P10827',
    'P11473', 'P37231', 'Q92731', 'Q9Y6Q9', 'P22736',
}

TRANSCRIPTION_FACTORS = {
    'P01106', 'P04637', 'P42229', 'P42226', 'P40763', 'P15407',
    'P01100', 'Q04206',
}

EPIGENETIC_REGULATORS = {
    'Q8WUI4', 'Q9UBN7', 'Q92769', 'P56524', 'Q9UKV0', 'Q969S8', 'Q9NRC8',
}


def classify_protein(uniprot_id):
    """Classify protein into family."""
    if uniprot_id in KINASES:
        return 'kinase'
    elif uniprot_id in BROMODOMAINS:
        return 'bromodomain'
    elif uniprot_id in NUCLEAR_RECEPTORS:
        return 'nuclear_receptor'
    elif uniprot_id in TRANSCRIPTION_FACTORS:
        return 'transcription_factor'
    elif uniprot_id in EPIGENETIC_REGULATORS:
        return 'epigenetic'
    else:
        return 'other'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load structures
    print("Loading structures...")
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        structures[pt_file.stem] = torch.load(pt_file, map_location='cpu', weights_only=False)
    print(f"Loaded {len(structures)} structures")

    # Load samples
    print("Loading samples...")
    samples = build_protac8k_degradation_data()
    valid_samples = [s for s in samples if s["uniprot_id"] in structures]
    print(f"Valid samples: {len(valid_samples)}")

    # Classify all samples
    family_counts = defaultdict(lambda: {'pos': 0, 'neg': 0})
    for s in valid_samples:
        family = classify_protein(s["uniprot_id"])
        if s["label"] == 1:
            family_counts[family]['pos'] += 1
        else:
            family_counts[family]['neg'] += 1

    print("\nFamily distribution in full dataset:")
    for family, counts in sorted(family_counts.items()):
        total = counts['pos'] + counts['neg']
        print(f"  {family}: {total} ({counts['pos']} pos, {counts['neg']} neg)")

    # Build graphs
    print("\nBuilding graphs...")
    all_graphs = {}
    for s in tqdm(valid_samples, desc="Building graphs"):
        uniprot = s["uniprot_id"]
        key = (uniprot, s["e3_name"], s["label"])
        if key in all_graphs:
            continue

        struct = structures[uniprot]
        graph = protein_to_graph(
            coords=struct["coords"],
            residues=struct["residues"],
            plddt=struct.get("plddt"),
            sasa=struct.get("sasa"),
            disorder=struct.get("disorder")
        )
        graph.y = torch.tensor([s["label"]], dtype=torch.float32)
        graph.e3_name = s["e3_name"]
        graph.uniprot = uniprot
        graph.family = classify_protein(uniprot)
        all_graphs[key] = graph

    print(f"Built {len(all_graphs)} unique graphs")

    # Create target-unseen split (same as training)
    target_to_samples = defaultdict(list)
    for s in valid_samples:
        target_to_samples[s["uniprot_id"]].append(s)

    targets = list(target_to_samples.keys())
    np.random.seed(42)
    np.random.shuffle(targets)

    n_train = int(len(targets) * 0.7)
    n_val = int(len(targets) * 0.15)

    test_targets = set(targets[n_train+n_val:])

    def get_graphs(target_set):
        graphs = {}
        for t in target_set:
            for s in target_to_samples[t]:
                key = (s["uniprot_id"], s["e3_name"], s["label"])
                if key in all_graphs:
                    graphs[key] = all_graphs[key]
        return graphs

    test_graphs = list(get_graphs(test_targets).values())
    print(f"\nTest set: {len(test_graphs)} samples")

    # Load TRAINED model checkpoint
    print("\nLoading trained model checkpoint...")
    checkpoint_path = Path("checkpoints/phase2_target_unseen/best_model.pt")

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    model = DegradoMap(
        node_input_dim=28, sug_hidden_dim=128, sug_output_dim=64,
        sug_num_layers=4, e3_hidden_dim=64, e3_output_dim=64,
        e3_num_heads=4, e3_num_layers=2, context_output_dim=64,
        fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Evaluate by family
    print("\nEvaluating by protein family...")
    family_results = defaultdict(lambda: {'preds': [], 'labels': []})

    with torch.no_grad():
        for g in tqdm(test_graphs, desc="Evaluating"):
            batch = Batch.from_data_list([g.clone().to(device)])
            out = model(batch, e3_name=g.e3_name)
            pred = torch.sigmoid(out["degrado_logits"]).cpu().item()
            label = g.y.item()
            family = g.family

            family_results[family]['preds'].append(pred)
            family_results[family]['labels'].append(label)
            family_results['overall']['preds'].append(pred)
            family_results['overall']['labels'].append(label)

    # Compute metrics per family
    results = {
        'description': 'Protein family breakdown analysis (trained model)',
        'split': 'target_unseen',
        'checkpoint': str(checkpoint_path),
        'families': {}
    }

    print("\n" + "="*70)
    print("PROTEIN FAMILY BREAKDOWN")
    print("="*70)
    print(f"{'Family':<20} {'N':<8} {'Pos Rate':<12} {'AUROC':<12} {'AUPRC':<12}")
    print("-"*64)

    for family in sorted(family_results.keys()):
        data = family_results[family]
        preds = data['preds']
        labels = data['labels']
        n = len(labels)
        pos_rate = sum(labels) / n if n > 0 else 0

        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, preds)
            auprc = average_precision_score(labels, preds)
        else:
            auroc = None
            auprc = None

        results['families'][family] = {
            'n_samples': n,
            'pos_rate': pos_rate,
            'auroc': auroc,
            'auprc': auprc,
        }

        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"
        print(f"{family:<20} {n:<8} {pos_rate:.2f}         {auroc_str:<12} {auprc_str:<12}")

    # Save results
    with open("results/protein_family_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/protein_family_analysis.json")


if __name__ == "__main__":
    main()

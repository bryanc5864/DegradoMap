#!/usr/bin/env python
"""
Case Studies: AR, ER, BTK

Detailed analysis of clinically relevant targets:
- AR (Androgen Receptor) - prostate cancer
- ER/ESR1 (Estrogen Receptor) - breast cancer
- BTK (Bruton's Tyrosine Kinase) - lymphoma/leukemia

Provides lysine-level predictions and comparison to known clinical PROTACs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score
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

# Target UniProt IDs for case studies
CASE_STUDY_TARGETS = {
    'P10275': {
        'name': 'AR',
        'full_name': 'Androgen Receptor',
        'disease': 'Prostate cancer',
        'known_protacs': ['ARV-110 (bavdegalutamide)', 'ARV-766'],
        'clinical_stage': 'Phase II',
        'e3_used': 'CRBN',
    },
    'P03372': {
        'name': 'ESR1/ER',
        'full_name': 'Estrogen Receptor Alpha',
        'disease': 'Breast cancer',
        'known_protacs': ['ARV-471'],
        'clinical_stage': 'Phase III',
        'e3_used': 'CRBN',
    },
    'Q06187': {
        'name': 'BTK',
        'full_name': "Bruton's Tyrosine Kinase",
        'disease': 'B-cell malignancies (CLL, MCL)',
        'known_protacs': ['NX-2127', 'NX-5948'],
        'clinical_stage': 'Phase I/II',
        'e3_used': 'CRBN',
    },
    'O60885': {
        'name': 'BRD4',
        'full_name': 'Bromodomain-containing protein 4',
        'disease': 'Various cancers',
        'known_protacs': ['ARV-825', 'dBET1', 'MZ1'],
        'clinical_stage': 'Preclinical',
        'e3_used': 'VHL/CRBN',
    },
}


def get_lysine_positions(residues):
    """Get positions of lysine residues."""
    lysines = []
    for i, res in enumerate(residues):
        if res == 'K':
            lysines.append(i)
    return lysines


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

    # Check which case study targets we have
    available_targets = {}
    for uniprot, info in CASE_STUDY_TARGETS.items():
        if uniprot in structures:
            available_targets[uniprot] = info
            print(f"  Found: {info['name']} ({uniprot})")
        else:
            print(f"  Missing: {info['name']} ({uniprot})")

    if not available_targets:
        print("No case study targets found in structures!")
        return

    # Build graphs for all samples
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
        all_graphs[key] = graph

    # Create target-unseen split
    target_to_samples = defaultdict(list)
    for s in valid_samples:
        target_to_samples[s["uniprot_id"]].append(s)

    targets = list(target_to_samples.keys())
    np.random.seed(42)
    np.random.shuffle(targets)

    n_train = int(len(targets) * 0.7)
    n_val = int(len(targets) * 0.15)
    train_targets = set(targets[:n_train])
    val_targets = set(targets[n_train:n_train+n_val])
    test_targets = set(targets[n_train+n_val:])

    def get_graphs(target_set):
        graphs = {}
        for t in target_set:
            for s in target_to_samples[t]:
                key = (s["uniprot_id"], s["e3_name"], s["label"])
                if key in all_graphs:
                    graphs[key] = all_graphs[key]
        return graphs

    train_graphs = list(get_graphs(train_targets).values())
    val_graphs = list(get_graphs(val_targets).values())

    print(f"\nTraining with {len(train_graphs)} samples, validating with {len(val_graphs)}")

    # Train model
    print("\nTraining model...")
    torch.manual_seed(42)
    np.random.seed(42)

    model = DegradoMap(
        node_input_dim=28, sug_hidden_dim=128, sug_output_dim=64,
        sug_num_layers=4, e3_hidden_dim=64, e3_output_dim=64,
        e3_num_heads=4, e3_num_layers=2, context_output_dim=64,
        fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    best_val = 0
    best_state = None
    batch_size = 16
    epochs = 20

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_graphs)

        for i in range(0, len(train_graphs), batch_size):
            batch_g = train_graphs[i:i+batch_size]
            batch = Batch.from_data_list([g.clone().to(device) for g in batch_g])

            optimizer.zero_grad()
            out = model(batch, e3_name=batch_g[0].e3_name)
            loss = F.binary_cross_entropy_with_logits(
                out["degrado_logits"].squeeze(), batch.y.squeeze()
            )
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for g in val_graphs:
                batch = Batch.from_data_list([g.clone().to(device)])
                out = model(batch, e3_name=g.e3_name)
                preds.append(torch.sigmoid(out["degrado_logits"]).cpu().item())
                labels.append(g.y.item())

        val_auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
        if val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: val_auroc={val_auroc:.4f}")

    print(f"\nBest val AUROC: {best_val:.4f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Case study analysis
    results = {}
    print("\n" + "="*80)
    print("CASE STUDY ANALYSIS")
    print("="*80)

    for uniprot, info in available_targets.items():
        print(f"\n{'='*80}")
        print(f"{info['name']} ({info['full_name']})")
        print(f"UniProt: {uniprot}")
        print(f"Disease: {info['disease']}")
        print(f"Known PROTACs: {', '.join(info['known_protacs'])}")
        print(f"Clinical Stage: {info['clinical_stage']}")
        print(f"E3 Used: {info['e3_used']}")
        print("="*80)

        struct = structures[uniprot]
        residues = struct["residues"]
        lysines = get_lysine_positions(residues)
        print(f"\nProtein length: {len(residues)} residues")
        print(f"Lysine count: {len(lysines)}")

        # Get predictions for this target
        target_results = {
            'info': info,
            'protein_length': len(residues),
            'lysine_count': len(lysines),
            'predictions': {},
        }

        # Check if we have samples for this target
        target_samples = [s for s in valid_samples if s["uniprot_id"] == uniprot]
        if target_samples:
            print(f"\nPROTAC-8K samples for this target: {len(target_samples)}")
            for s in target_samples[:5]:  # Show first 5
                label_str = "Degraded" if s["label"] == 1 else "Not degraded"
                print(f"  E3: {s['e3_name']}, Label: {label_str}")

        # Make predictions with different E3s
        for e3_name in ['CRBN', 'VHL']:
            graph = protein_to_graph(
                coords=struct["coords"],
                residues=struct["residues"],
                plddt=struct.get("plddt"),
                sasa=struct.get("sasa"),
                disorder=struct.get("disorder")
            )
            graph.y = torch.tensor([0], dtype=torch.float32)
            graph.e3_name = e3_name

            with torch.no_grad():
                batch = Batch.from_data_list([graph.clone().to(device)])
                out = model(batch, e3_name=e3_name)
                pred = torch.sigmoid(out["degrado_logits"]).cpu().item()

            target_results['predictions'][e3_name] = pred
            pred_label = "Likely degradable" if pred > 0.5 else "Unlikely degradable"
            print(f"\nPrediction with {e3_name}: {pred:.4f} ({pred_label})")

        # Get actual labels from dataset
        actual_crbn = [s["label"] for s in target_samples if s["e3_name"] == "CRBN"]
        actual_vhl = [s["label"] for s in target_samples if s["e3_name"] == "VHL"]

        if actual_crbn:
            target_results['actual_crbn'] = {
                'degraded': sum(actual_crbn),
                'total': len(actual_crbn),
            }
            print(f"\nActual CRBN data: {sum(actual_crbn)}/{len(actual_crbn)} degraded")

        if actual_vhl:
            target_results['actual_vhl'] = {
                'degraded': sum(actual_vhl),
                'total': len(actual_vhl),
            }
            print(f"Actual VHL data: {sum(actual_vhl)}/{len(actual_vhl)} degraded")

        results[uniprot] = target_results

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Target':<10} {'Length':<8} {'#Lys':<6} {'CRBN Pred':<12} {'VHL Pred':<12}")
    print("-"*48)

    for uniprot, data in results.items():
        name = data['info']['name']
        length = data['protein_length']
        n_lys = data['lysine_count']
        crbn_pred = data['predictions'].get('CRBN', float('nan'))
        vhl_pred = data['predictions'].get('VHL', float('nan'))
        print(f"{name:<10} {length:<8} {n_lys:<6} {crbn_pred:.4f}       {vhl_pred:.4f}")

    # Save results
    output = {
        'description': 'Case study analysis for clinically relevant targets',
        'targets': results,
    }
    with open("results/case_studies.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to results/case_studies.json")


if __name__ == "__main__":
    main()

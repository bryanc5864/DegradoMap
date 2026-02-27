#!/usr/bin/env python
"""
Protein Family Breakdown Analysis.

Analyzes model performance by protein family:
- Kinases
- Bromodomains
- Nuclear receptors
- Other categories

This addresses ACM BCB reviewer request for family-specific analysis.
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
    'Q16539',  # MAPK14/p38
    'P31749',  # AKT1
    'P00533',  # EGFR
    'P06239',  # LCK
    'P09619',  # PDGFRB
    'Q13131',  # AMPK
    'P45983',  # MAPK8/JNK1
    'Q15418',  # RPS6KA1
    'P17252',  # PRKCA
    'Q05397',  # FAK
    'P42336',  # PIK3CA
    'P11309',  # PIM1
    'P49841',  # GSK3B
    'P42345',  # MTOR
    'P28482',  # MAPK1/ERK2
    'P42680',  # TEC
    'P04049',  # RAF1
    'P36888',  # FLT3
    'P06241',  # FYN
    'P22681',  # CBL
    'Q16644',  # MAPKAPK3
    'Q02750',  # MAP2K1/MEK1
    'Q06418',  # TYRO3
    'O14965',  # AURKA
    'P06213',  # INSR
}

BROMODOMAINS = {
    'O60885',  # BRD4
    'Q15059',  # BRD3
    'P25440',  # BRD2
    'Q58F21',  # BRDT
    'Q9NPI1',  # BRD7
    'Q9H0E9',  # BRD8
    'Q9ULD4',  # BRPF3
    'O95696',  # BRPF1
    'Q86U86',  # BRPF2/BRD1
    'Q9UIF8',  # BAZ2B
    'Q05086',  # UBE3A
}

NUCLEAR_RECEPTORS = {
    'P10275',  # AR (androgen receptor)
    'P03372',  # ESR1/ER (estrogen receptor)
    'P04278',  # SHBG
    'P06401',  # PR (progesterone receptor)
    'P10826',  # RARB
    'P10827',  # RARA
    'P11473',  # VDR (vitamin D receptor)
    'P37231',  # PPARG
    'Q92731',  # ESR2
    'Q9Y6Q9',  # NCOA3
    'P22736',  # NR4A1
}

TRANSCRIPTION_FACTORS = {
    'P01106',  # MYC
    'P04637',  # TP53
    'P42229',  # STAT5A
    'P42226',  # STAT6
    'P40763',  # STAT3
    'P15407',  # FOSL1
    'P01100',  # FOS
    'Q04206',  # RELA/NF-kB p65
}

EPIGENETIC_REGULATORS = {
    'Q8WUI4',  # HDAC7
    'Q9UBN7',  # HDAC6
    'Q92769',  # HDAC2
    'P56524',  # HDAC4
    'Q9UKV0',  # HDAC9
    'Q969S8',  # SIRT6
    'Q9NRC8',  # SIRT7
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

    print("\nFamily distribution:")
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
    test_graphs = list(get_graphs(test_targets).values())

    print(f"\nSplit: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

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

    # Evaluate by family
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Collect predictions grouped by family
    family_preds = defaultdict(list)
    family_labels = defaultdict(list)

    with torch.no_grad():
        for g in test_graphs:
            batch = Batch.from_data_list([g.clone().to(device)])
            out = model(batch, e3_name=g.e3_name)
            pred = torch.sigmoid(out["degrado_logits"]).cpu().item()
            label = g.y.item()
            family = g.family

            family_preds[family].append(pred)
            family_labels[family].append(label)

    # Compute metrics per family
    results = {}
    print("\n" + "="*70)
    print("PROTEIN FAMILY BREAKDOWN")
    print("="*70)
    print(f"{'Family':<20} {'N':<8} {'Pos%':<8} {'AUROC':<10} {'AUPRC':<10}")
    print("-"*56)

    for family in sorted(family_preds.keys()):
        preds = family_preds[family]
        labels = family_labels[family]
        n = len(labels)
        pos_rate = sum(labels) / n if n > 0 else 0

        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, preds)
            auprc = average_precision_score(labels, preds)
        else:
            auroc = float('nan')
            auprc = float('nan')

        results[family] = {
            'n_samples': n,
            'pos_rate': pos_rate,
            'auroc': auroc if not np.isnan(auroc) else None,
            'auprc': auprc if not np.isnan(auprc) else None,
        }

        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
        auprc_str = f"{auprc:.4f}" if not np.isnan(auprc) else "N/A"
        print(f"{family:<20} {n:<8} {pos_rate:.2%}    {auroc_str:<10} {auprc_str:<10}")

    # Overall test metrics
    all_preds = [p for preds in family_preds.values() for p in preds]
    all_labels = [l for labels in family_labels.values() for l in labels]
    overall_auroc = roc_auc_score(all_labels, all_preds)
    overall_auprc = average_precision_score(all_labels, all_preds)

    print("-"*56)
    print(f"{'OVERALL':<20} {len(all_labels):<8} {sum(all_labels)/len(all_labels):.2%}    {overall_auroc:.4f}     {overall_auprc:.4f}")

    results['overall'] = {
        'n_samples': len(all_labels),
        'pos_rate': sum(all_labels) / len(all_labels),
        'auroc': overall_auroc,
        'auprc': overall_auprc,
    }

    # Save results
    output = {
        'description': 'Protein family breakdown analysis',
        'split': 'target_unseen',
        'families': results,
    }
    with open("results/protein_family_analysis.json", 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/protein_family_analysis.json")


if __name__ == "__main__":
    main()

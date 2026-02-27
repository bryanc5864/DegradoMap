#!/usr/bin/env python
"""
MAPD (MAssive Protein Degradation) Comparison.

Compares DegradoMap predictions to MAPD database predictions.
MAPD: https://mapd.cistrome.org - comprehensive PROTAC degradation database

Note: MAPD provides degradation predictions based on sequence/structure analysis.
We compare our predictions on test proteins to see concordance.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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
import requests
import time

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs['flush'] = True
    _print(*args, **kwargs)


def query_mapd_api(uniprot_id, timeout=30):
    """
    Query MAPD database for a protein.

    MAPD provides degradation scores for proteins.
    API endpoint: https://mapd.cistrome.org/api/
    """
    try:
        # Try the gene info endpoint
        url = f"https://mapd.cistrome.org/api/gene/{uniprot_id}"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"  API error for {uniprot_id}: {e}")
    return None


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

    # Get unique proteins
    unique_proteins = list(set(s["uniprot_id"] for s in valid_samples))
    print(f"Unique proteins: {len(unique_proteins)}")

    # Query MAPD for a subset of proteins
    print("\nQuerying MAPD database...")
    mapd_data = {}

    # Test first to see if API is accessible
    test_response = query_mapd_api(unique_proteins[0])
    if test_response is None:
        print("MAPD API not accessible. Using fallback analysis...")
        # Fallback: analyze without MAPD comparison
        mapd_available = False
    else:
        mapd_available = True
        print("MAPD API accessible, querying proteins...")

        for uniprot in tqdm(unique_proteins[:50], desc="Querying MAPD"):  # Limit to 50
            data = query_mapd_api(uniprot)
            if data:
                mapd_data[uniprot] = data
            time.sleep(0.5)  # Rate limiting

        print(f"Retrieved MAPD data for {len(mapd_data)} proteins")

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

    # Evaluate on test set
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Get predictions per protein
    protein_predictions = defaultdict(list)
    protein_labels = defaultdict(list)

    with torch.no_grad():
        for g in test_graphs:
            batch = Batch.from_data_list([g.clone().to(device)])
            out = model(batch, e3_name=g.e3_name)
            pred = torch.sigmoid(out["degrado_logits"]).cpu().item()
            label = g.y.item()

            protein_predictions[g.uniprot].append(pred)
            protein_labels[g.uniprot].append(label)

    # Aggregate per-protein
    protein_scores = {}
    for uniprot in protein_predictions:
        preds = protein_predictions[uniprot]
        labels = protein_labels[uniprot]
        protein_scores[uniprot] = {
            'mean_pred': float(np.mean(preds)),
            'max_pred': float(np.max(preds)),
            'n_samples': len(preds),
            'pos_rate': float(np.mean(labels)),
        }

    # Results summary
    print("\n" + "="*70)
    print("MAPD COMPARISON ANALYSIS")
    print("="*70)

    if mapd_available and mapd_data:
        # Compare with MAPD scores
        print("\nComparing DegradoMap vs MAPD predictions:")
        print(f"{'Protein':<12} {'DegradoMap':<12} {'MAPD Score':<12} {'Actual':<10}")
        print("-"*46)

        comparison_results = []
        for uniprot in sorted(protein_scores.keys()):
            if uniprot in mapd_data:
                dm_score = protein_scores[uniprot]['mean_pred']
                mapd_score = mapd_data[uniprot].get('degradation_score', 'N/A')
                actual = protein_scores[uniprot]['pos_rate']
                print(f"{uniprot:<12} {dm_score:.4f}       {mapd_score}         {actual:.2f}")
                comparison_results.append({
                    'uniprot': uniprot,
                    'degradomap': dm_score,
                    'mapd': mapd_score,
                    'actual': actual,
                })
    else:
        print("\nMAPD API not available - showing DegradoMap predictions only:")
        print(f"{'Protein':<12} {'DegradoMap':<12} {'N Samples':<10} {'Actual Rate':<12}")
        print("-"*46)

        for uniprot in sorted(protein_scores.keys())[:20]:  # Show first 20
            data = protein_scores[uniprot]
            print(f"{uniprot:<12} {data['mean_pred']:.4f}       {data['n_samples']:<10} {data['pos_rate']:.2f}")

    # Overall test metrics
    all_preds = [p for g in test_graphs for p in [torch.sigmoid(model(Batch.from_data_list([g.clone().to(device)]), e3_name=g.e3_name)["degrado_logits"]).cpu().item()]]
    all_labels = [g.y.item() for g in test_graphs]

    # Re-compute from stored predictions
    all_preds = []
    all_labels = []
    for uniprot in protein_predictions:
        all_preds.extend(protein_predictions[uniprot])
        all_labels.extend(protein_labels[uniprot])

    test_auroc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    print(f"\nOverall Test AUROC: {test_auroc:.4f}")
    print(f"Test proteins: {len(protein_scores)}")
    print(f"Test samples: {len(all_labels)}")

    # Save results
    output = {
        'description': 'MAPD comparison analysis',
        'mapd_available': mapd_available,
        'n_mapd_proteins': len(mapd_data) if mapd_available else 0,
        'test_auroc': test_auroc,
        'n_test_proteins': len(protein_scores),
        'n_test_samples': len(all_labels),
        'protein_scores': protein_scores,
    }

    if mapd_available and mapd_data:
        output['mapd_comparison'] = comparison_results

    with open("results/mapd_comparison.json", 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/mapd_comparison.json")


if __name__ == "__main__":
    main()

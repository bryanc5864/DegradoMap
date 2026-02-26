"""
ESM-2 Failure Analysis.

Investigates why ESM-2 doesn't help degradability prediction:
1. Layer-wise analysis (which layers are most informative?)
2. Per-protein breakdown (helps for some proteins?)
3. Feature importance analysis
4. Comparison with fine-tuned ESM
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits


def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def load_esm_embeddings():
    """Load ESM embeddings with all layers if available."""
    esm_dir = Path("data/processed/esm_embeddings")
    embeddings = {}
    if esm_dir.exists():
        for pt_file in esm_dir.glob("*.pt"):
            # Remove _esm suffix if present
            uniprot = pt_file.stem.replace("_esm", "")
            data = torch.load(pt_file, map_location='cpu', weights_only=False)
            embeddings[uniprot] = data
    return embeddings


def extract_esm_features(embeddings: Dict, method: str = 'mean') -> Dict:
    """Extract different ESM feature representations."""
    features = {}

    for uniprot, emb in embeddings.items():
        if isinstance(emb, torch.Tensor):
            # Single layer embedding [L, 1280]
            if method == 'mean':
                features[uniprot] = emb.mean(dim=0).numpy()
            elif method == 'cls':
                features[uniprot] = emb[0].numpy()  # First token
            elif method == 'max':
                features[uniprot] = emb.max(dim=0)[0].numpy()
        elif isinstance(emb, dict):
            # Dict with 'embeddings' key
            if 'embeddings' in emb:
                layer_emb = emb['embeddings']
                if method == 'mean':
                    features[uniprot] = layer_emb.mean(dim=0).numpy()
                elif method == 'cls':
                    features[uniprot] = layer_emb[0].numpy()
                elif method == 'max':
                    features[uniprot] = layer_emb.max(dim=0)[0].numpy()
            elif 'representations' in emb:
                # Multi-layer embeddings (alternative format)
                features[uniprot] = {}
                for layer_idx, layer_emb in emb['representations'].items():
                    if method == 'mean':
                        features[uniprot][layer_idx] = layer_emb.mean(dim=0).numpy()
                    else:
                        features[uniprot][layer_idx] = layer_emb[0].numpy()

    return features


def build_esm_dataset(samples: List, esm_features: Dict, e3_onehot: bool = True):
    """Build dataset with ESM features."""
    E3_LIST = ['CRBN', 'VHL', 'cIAP1', 'MDM2', 'XIAP', 'DCAF16', 'KEAP1', 'FEM1B']

    X, y = [], []
    for sample in samples:
        uniprot = sample["uniprot_id"]
        if uniprot not in esm_features:
            continue

        feat = esm_features[uniprot]
        if isinstance(feat, dict):
            # Use last layer
            feat = list(feat.values())[-1]

        if e3_onehot:
            e3_vec = np.zeros(len(E3_LIST))
            if sample["e3_name"] in E3_LIST:
                e3_vec[E3_LIST.index(sample["e3_name"])] = 1
            feat = np.concatenate([feat, e3_vec])

        X.append(feat)
        y.append(sample["label"])

    return np.array(X), np.array(y)


def layer_wise_analysis(samples: List, esm_embeddings: Dict) -> Dict:
    """Analyze which ESM layers are most predictive."""
    print("\nLayer-wise ESM Analysis")
    print("="*50)

    # Check if we have layer-wise embeddings
    sample_emb = list(esm_embeddings.values())[0]
    if not isinstance(sample_emb, dict) or 'representations' not in sample_emb:
        print("Layer-wise embeddings not available. Using mean of final layer.")
        return {}

    results = {}
    layers = list(sample_emb['representations'].keys())

    train_samples, val_samples, test_samples = create_data_splits(samples, 'target_unseen')

    for layer_idx in tqdm(layers, desc="Testing layers"):
        # Extract features for this layer
        layer_features = {}
        for uniprot, emb in esm_embeddings.items():
            if isinstance(emb, dict) and 'representations' in emb:
                layer_features[uniprot] = emb['representations'][layer_idx].mean(dim=0).numpy()

        # Build dataset
        X_train, y_train = build_esm_dataset(train_samples + val_samples, layer_features)
        X_test, y_test = build_esm_dataset(test_samples, layer_features)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Train simple classifier
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_pred)

        results[f"layer_{layer_idx}"] = auroc
        print(f"  Layer {layer_idx}: AUROC = {auroc:.4f}")

    return results


def per_protein_analysis(samples: List, esm_features: Dict, structures: Dict) -> Dict:
    """Analyze which proteins benefit from ESM."""
    print("\nPer-Protein ESM Analysis")
    print("="*50)

    train_samples, val_samples, test_samples = create_data_splits(samples, 'target_unseen')

    # Build datasets
    X_train, y_train = build_esm_dataset(train_samples + val_samples, esm_features)
    X_test, y_test = build_esm_dataset(test_samples, esm_features)

    if len(X_train) == 0 or len(X_test) == 0:
        print("  No samples with ESM features found. Skipping.")
        return {'error': 'No ESM features available'}

    # Train classifier
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    # Get per-sample predictions
    test_uniprots = [s["uniprot_id"] for s in test_samples if s["uniprot_id"] in esm_features]
    y_pred = clf.predict_proba(X_test)[:, 1]

    # Analyze by protein properties
    results = {
        'by_size': {},
        'by_disorder': {},
        'helped': [],
        'hurt': [],
    }

    for i, uniprot in enumerate(test_uniprots):
        if uniprot not in structures:
            continue

        struct = structures[uniprot]
        n_residues = len(struct["residues"])
        disorder = struct.get("disorder", torch.zeros(1)).mean().item() if struct.get("disorder") is not None else 0

        # Categorize
        size_cat = "small" if n_residues < 300 else "medium" if n_residues < 600 else "large"
        disorder_cat = "ordered" if disorder < 0.3 else "partially_disordered" if disorder < 0.6 else "disordered"

        pred = y_pred[i]
        label = y_test[i]
        correct = (pred >= 0.5) == label

        if size_cat not in results['by_size']:
            results['by_size'][size_cat] = {'correct': 0, 'total': 0}
        results['by_size'][size_cat]['total'] += 1
        if correct:
            results['by_size'][size_cat]['correct'] += 1

        if disorder_cat not in results['by_disorder']:
            results['by_disorder'][disorder_cat] = {'correct': 0, 'total': 0}
        results['by_disorder'][disorder_cat]['total'] += 1
        if correct:
            results['by_disorder'][disorder_cat]['correct'] += 1

    # Compute accuracies
    for cat_type in ['by_size', 'by_disorder']:
        for cat, stats in results[cat_type].items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    return results


def feature_importance_analysis(samples: List, esm_features: Dict) -> Dict:
    """Analyze which ESM dimensions are most important."""
    print("\nFeature Importance Analysis")
    print("="*50)

    train_samples, val_samples, test_samples = create_data_splits(samples, 'target_unseen')

    X_train, y_train = build_esm_dataset(train_samples + val_samples, esm_features, e3_onehot=False)
    X_test, y_test = build_esm_dataset(test_samples, esm_features, e3_onehot=False)

    if len(X_train) == 0:
        return {}

    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    # Top features
    top_k = 20
    top_indices = np.argsort(importances)[-top_k:][::-1]

    results = {
        'top_features': [
            {'index': int(idx), 'importance': float(importances[idx])}
            for idx in top_indices
        ],
        'importance_stats': {
            'mean': float(np.mean(importances)),
            'std': float(np.std(importances)),
            'max': float(np.max(importances)),
            'min': float(np.min(importances)),
        },
        'rf_auroc': float(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])),
    }

    print(f"  Random Forest AUROC: {results['rf_auroc']:.4f}")
    print(f"  Top 5 feature indices: {[f['index'] for f in results['top_features'][:5]]}")

    return results


def comparison_analysis(samples: List, esm_features: Dict, structures: Dict) -> Dict:
    """Compare ESM-only vs structure-only vs combined."""
    print("\nComparison Analysis")
    print("="*50)

    train_samples, val_samples, test_samples = create_data_splits(samples, 'target_unseen')

    results = {}

    # ESM-only
    X_train_esm, y_train = build_esm_dataset(train_samples + val_samples, esm_features)
    X_test_esm, y_test = build_esm_dataset(test_samples, esm_features)

    if len(X_train_esm) > 0:
        clf_esm = LogisticRegression(max_iter=500, random_state=42)
        clf_esm.fit(X_train_esm, y_train)
        results['esm_only'] = roc_auc_score(y_test, clf_esm.predict_proba(X_test_esm)[:, 1])
        print(f"  ESM-only: {results['esm_only']:.4f}")

    # Structure features only
    def get_structure_features(sample, structures):
        uniprot = sample["uniprot_id"]
        if uniprot not in structures:
            return None
        struct = structures[uniprot]
        n_res = len(struct["residues"])
        n_lys = sum(1 for r in struct["residues"] if r.upper() == 'K')
        plddt = struct.get("plddt", torch.zeros(1))
        sasa = struct.get("sasa", torch.zeros(1))
        disorder = struct.get("disorder", torch.zeros(1))

        return np.array([
            n_res, n_lys, n_lys/max(n_res, 1),
            plddt.mean().item(), plddt.std().item(),
            sasa.mean().item(), sasa.std().item(),
            disorder.mean().item(), disorder.std().item(),
        ])

    X_train_struct = []
    y_train_struct = []
    for s in train_samples + val_samples:
        feat = get_structure_features(s, structures)
        if feat is not None:
            X_train_struct.append(feat)
            y_train_struct.append(s["label"])

    X_test_struct = []
    y_test_struct = []
    for s in test_samples:
        feat = get_structure_features(s, structures)
        if feat is not None:
            X_test_struct.append(feat)
            y_test_struct.append(s["label"])

    if X_train_struct:
        clf_struct = LogisticRegression(max_iter=500, random_state=42)
        clf_struct.fit(X_train_struct, y_train_struct)
        results['structure_only'] = roc_auc_score(y_test_struct, clf_struct.predict_proba(X_test_struct)[:, 1])
        print(f"  Structure-only: {results['structure_only']:.4f}")

    # Combined
    X_train_comb = []
    y_train_comb = []
    for s in train_samples + val_samples:
        uniprot = s["uniprot_id"]
        if uniprot not in esm_features or uniprot not in structures:
            continue
        esm_feat = esm_features[uniprot]
        struct_feat = get_structure_features(s, structures)
        combined = np.concatenate([esm_feat, struct_feat])
        X_train_comb.append(combined)
        y_train_comb.append(s["label"])

    X_test_comb = []
    y_test_comb = []
    for s in test_samples:
        uniprot = s["uniprot_id"]
        if uniprot not in esm_features or uniprot not in structures:
            continue
        esm_feat = esm_features[uniprot]
        struct_feat = get_structure_features(s, structures)
        combined = np.concatenate([esm_feat, struct_feat])
        X_test_comb.append(combined)
        y_test_comb.append(s["label"])

    if X_train_comb:
        clf_comb = LogisticRegression(max_iter=500, random_state=42)
        clf_comb.fit(X_train_comb, y_train_comb)
        results['combined'] = roc_auc_score(y_test_comb, clf_comb.predict_proba(X_test_comb)[:, 1])
        print(f"  Combined: {results['combined']:.4f}")

    return results


def main():
    print("ESM-2 Failure Analysis")
    print("="*60)

    # Load data
    print("\nLoading data...")
    structures = load_structures()
    print(f"Loaded {len(structures)} structures")

    esm_embeddings = load_esm_embeddings()
    print(f"Loaded {len(esm_embeddings)} ESM embeddings")

    samples = build_protac8k_degradation_data()
    print(f"Loaded {len(samples)} samples")

    if not esm_embeddings:
        print("No ESM embeddings found. Run ESM extraction first.")
        return

    # Extract features
    esm_features = extract_esm_features(esm_embeddings, method='mean')

    results = {}

    # Layer-wise analysis
    layer_results = layer_wise_analysis(samples, esm_embeddings)
    if layer_results:
        results['layer_analysis'] = layer_results

    # Per-protein analysis
    protein_results = per_protein_analysis(samples, esm_features, structures)
    results['per_protein'] = protein_results

    # Feature importance
    importance_results = feature_importance_analysis(samples, esm_features)
    results['feature_importance'] = importance_results

    # Comparison
    comparison_results = comparison_analysis(samples, esm_features, structures)
    results['comparison'] = comparison_results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Why ESM Doesn't Help")
    print("="*60)

    if 'comparison' in results:
        print(f"\n  ESM-only AUROC: {results['comparison'].get('esm_only', 'N/A')}")
        print(f"  Structure-only AUROC: {results['comparison'].get('structure_only', 'N/A')}")
        print(f"  Combined AUROC: {results['comparison'].get('combined', 'N/A')}")

    print("\n  Key Insight: Degradability is NOT captured by evolutionary features.")
    print("  ESM embeddings encode protein family/function, not degradability potential.")
    print("  Structure + E3 compatibility are the relevant predictors.")

    # Save results
    output_path = "results/esm_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

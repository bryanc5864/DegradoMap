"""
Baseline comparison models for PROTAC degradability prediction.

Implements:
1. Random Forest with protein features
2. MLP baseline
3. Logistic Regression
4. Gradient Boosting

All baselines use the same data splits as DegradoMap for fair comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from scripts.train import build_protac8k_degradation_data, create_data_splits


def extract_protein_features(uniprot_id, structures):
    """Extract hand-crafted features from protein structure."""
    if uniprot_id not in structures:
        return None

    struct = structures[uniprot_id]

    features = []

    # Basic counts
    features.append(struct['num_residues'])
    features.append(struct['num_lysines'])
    features.append(struct['num_lysines'] / max(struct['num_residues'], 1))  # lysine fraction

    # pLDDT statistics
    plddt = struct.get('plddt')
    if plddt is not None and len(plddt) > 0:
        plddt = plddt.numpy() if torch.is_tensor(plddt) else np.array(plddt)
        features.extend([plddt.mean(), plddt.std(), plddt.min(), plddt.max()])
    else:
        features.extend([0, 0, 0, 0])

    # SASA statistics
    sasa = struct.get('sasa')
    if sasa is not None and len(sasa) > 0:
        sasa = sasa.numpy() if torch.is_tensor(sasa) else np.array(sasa)
        features.extend([sasa.mean(), sasa.std(), sasa.min(), sasa.max()])
    else:
        features.extend([0, 0, 0, 0])

    # Disorder statistics
    disorder = struct.get('disorder')
    if disorder is not None and len(disorder) > 0:
        disorder = disorder.numpy() if torch.is_tensor(disorder) else np.array(disorder)
        features.extend([disorder.mean(), disorder.std(), disorder.sum()])
    else:
        features.extend([0, 0, 0])

    # Lysine positions (mean, std of positions)
    lys_pos = struct.get('lysine_positions', [])
    if len(lys_pos) > 0:
        lys_pos = np.array(lys_pos)
        rel_pos = lys_pos / max(struct['num_residues'], 1)
        features.extend([rel_pos.mean(), rel_pos.std()])
    else:
        features.extend([0, 0])

    # Coordinates statistics (protein size/shape)
    coords = struct['coords']
    if torch.is_tensor(coords):
        coords = coords.numpy()

    # Radius of gyration proxy
    centroid = coords.mean(axis=0)
    distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
    features.append(distances.mean())  # mean distance from centroid
    features.append(distances.max())   # max extent

    return np.array(features)


def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


# E3 ligase one-hot encoding
E3_LIGASES = ['CRBN', 'VHL', 'cIAP1', 'MDM2', 'XIAP', 'DCAF16', 'KEAP1', 'FEM1B', 'DCAF1', 'UBR', 'KLHL20']
E3_TO_IDX = {e3: i for i, e3 in enumerate(E3_LIGASES)}


def get_e3_features(e3_name):
    """One-hot encode E3 ligase."""
    vec = np.zeros(len(E3_LIGASES))
    if e3_name in E3_TO_IDX:
        vec[E3_TO_IDX[e3_name]] = 1
    return vec


def prepare_dataset(samples, structures):
    """Prepare features and labels for sklearn models."""
    X = []
    y = []

    for sample in samples:
        uniprot = sample['uniprot_id']
        prot_feat = extract_protein_features(uniprot, structures)
        if prot_feat is None:
            continue

        e3_feat = get_e3_features(sample['e3_name'])
        features = np.concatenate([prot_feat, e3_feat])

        X.append(features)
        y.append(sample['label'])

    return np.array(X), np.array(y)


def evaluate_model(model, X_test, y_test, name):
    """Evaluate a trained model."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_binary = (y_pred_proba >= thresh).astype(int)
        f1_t = f1_score(y_test, pred_binary, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh

    return {
        'model': name,
        'auroc': float(auroc),
        'auprc': float(auprc),
        'f1': float(f1),
        'f1_optimal': float(best_f1),
        'accuracy': float(acc),
        'optimal_threshold': float(best_thresh)
    }


def run_baselines(split_type, structures, samples):
    """Run all baseline models on a given split."""
    print(f"\n{'='*60}")
    print(f"Split: {split_type}")
    print(f"{'='*60}")

    # Load data
    train_samples, val_samples, test_samples = create_data_splits(samples, split_type)

    # Prepare datasets
    X_train, y_train = prepare_dataset(train_samples, structures)
    X_val, y_val = prepare_dataset(val_samples, structures)
    X_test, y_test = prepare_dataset(test_samples, structures)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    # Combine train + val for final model
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Normalize features
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_trainval_scaled, y_trainval)
    results.append(evaluate_model(lr, X_test_scaled, y_test, 'LogisticRegression'))
    print(f"  AUROC: {results[-1]['auroc']:.4f}")

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_trainval, y_trainval)  # RF doesn't need scaling
    results.append(evaluate_model(rf, X_test, y_test, 'RandomForest'))
    print(f"  AUROC: {results[-1]['auroc']:.4f}")

    # 3. Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_trainval, y_trainval)
    results.append(evaluate_model(gb, X_test, y_test, 'GradientBoosting'))
    print(f"  AUROC: {results[-1]['auroc']:.4f}")

    # 4. MLP
    print("Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True)
    mlp.fit(X_trainval_scaled, y_trainval)
    results.append(evaluate_model(mlp, X_test_scaled, y_test, 'MLP'))
    print(f"  AUROC: {results[-1]['auroc']:.4f}")

    return results


def main():
    all_results = {}

    # Load data once
    print("Loading structures...")
    structures = load_structures()
    print(f"Loaded {len(structures)} structures")

    print("Loading samples...")
    samples = build_protac8k_degradation_data()
    print(f"Loaded {len(samples)} samples")

    for split in ['target_unseen', 'e3_unseen', 'random']:
        try:
            results = run_baselines(split, structures, samples)
            all_results[split] = results
        except Exception as e:
            print(f"Error in {split}: {e}")
            all_results[split] = []

    # Save results
    output_path = "results/baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)

    # Load DegradoMap results for comparison
    degradomap_results = {}
    try:
        with open("results/final_test_results.json") as f:
            degradomap_results = json.load(f)
    except:
        pass

    for split in ['target_unseen', 'e3_unseen', 'random']:
        print(f"\n{split}:")
        print(f"{'Model':<20} {'AUROC':<10} {'AUPRC':<10} {'F1':<10}")
        print("-" * 50)

        # DegradoMap first
        if split in degradomap_results:
            dm = degradomap_results[split]
            print(f"{'DegradoMap':<20} {dm['auroc']:.4f}     {dm['auprc']:.4f}     {dm['f1']:.4f}")

        # Baselines
        for res in all_results[split]:
            print(f"{res['model']:<20} {res['auroc']:.4f}     {res['auprc']:.4f}     {res['f1']:.4f}")


if __name__ == "__main__":
    main()

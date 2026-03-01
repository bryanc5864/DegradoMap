"""Bootstrap CI for improved model results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bootstrap")

def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence intervals."""
    n = len(y_true)
    scores = []

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            score = roc_auc_score(y_true[idx], y_pred[idx])
            scores.append(score)
        except ValueError:
            continue

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'n_bootstrap': n_bootstrap
    }

def main():
    from src.models.degradomap import DegradoMap
    from src.data.dataset import DegradationDataset
    from torch_geometric.loader import DataLoader

    # Import data building function from train_improved
    from scripts.train_improved import build_protac8k_data, create_target_unseen_split

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load checkpoint
    ckpt_path = "checkpoints/improved_best.pt"
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint with {len(state_dict)} parameters")

    # Build model with exact config from train_improved.py
    model = DegradoMap(
        node_input_dim=1285,
        sug_hidden_dim=128,
        sug_output_dim=64,
        sug_num_layers=4,
        sug_max_radius=8.0,
        sug_num_basis=8,
        e3_hidden_dim=64,
        e3_output_dim=64,
        e3_num_heads=4,
        e3_num_layers=2,  # Important: was 2 not 4
        context_output_dim=64,
        fusion_hidden_dim=128,
        pred_hidden_dim=64,  # Important: was 64 not 128
        dropout=0.05,
        use_e3_onehot=True,
        use_global_stats=True
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Load test data with same split as training
    samples = build_protac8k_data()
    train_data, val_data, test_data = create_target_unseen_split(samples, seed=42)

    test_dataset = DegradationDataset(
        test_data,
        "data/processed/structures",
        "data/processed/esm_embeddings",
        use_esm=True,
        use_ub_sites=True
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    logger.info(f"Test set size: {len(test_dataset)}")

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            graphs = batch_data["graph"].to(device)
            labels = batch_data["label"].to(device)
            e3_names = batch_data["e3_name"]
            e3_name = e3_names[0] if isinstance(e3_names, list) else e3_names
            outputs = model(graphs, e3_name)
            probs = torch.sigmoid(outputs['degrado_logits']).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Compute bootstrap CI
    logger.info("Computing bootstrap confidence intervals (1000 samples)...")
    results = bootstrap_ci(y_true, y_pred, n_bootstrap=1000)

    # Also compute AUPRC
    auprc = average_precision_score(y_true, y_pred)

    results['auroc_point'] = float(roc_auc_score(y_true, y_pred))
    results['auprc'] = float(auprc)
    results['n_test'] = len(y_true)
    results['n_positive'] = int(y_true.sum())
    results['n_negative'] = int(len(y_true) - y_true.sum())

    print(f"\n{'='*50}")
    print("Bootstrap Results (Improved Model)")
    print(f"{'='*50}")
    print(f"AUROC: {results['auroc_point']:.4f}")
    print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    print(f"Std: {results['std']:.4f}")
    print(f"AUPRC: {results['auprc']:.4f}")
    print(f"Test set: {results['n_test']} ({results['n_positive']} pos, {results['n_negative']} neg)")

    # Save results
    with open('results/improved_bootstrap_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/improved_bootstrap_results.json")

if __name__ == '__main__':
    main()

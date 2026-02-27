"""
Multi-seed training for variance estimates.

Runs DegradoMap training with 5 different random seeds and computes
mean ± std for all metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import subprocess
import numpy as np
from collections import defaultdict

SEEDS = [42, 43, 44, 45, 46]
SPLITS = ['target_unseen', 'e3_unseen', 'random']


def run_single_seed(seed, split, gpu_id):
    """Run training for a single seed and split."""
    output_dir = f"checkpoints/seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "scripts/train.py",
        "--phase", "2",
        "--splits", split,
        "--finetune-epochs", "20",
        "--class-weights",
        "--seed", str(seed),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Running seed={seed}, split={split}, GPU={gpu_id}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error for seed={seed}, split={split}:")
        print(result.stderr[-1000:])
        return None

    return True


def evaluate_checkpoint(seed, split):
    """Evaluate a trained checkpoint."""
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    from src.models.degradomap import DegradoMap
    from src.models.sug_module import protein_to_graph
    from scripts.train import build_protac8k_degradation_data, create_data_splits
    from torch_geometric.data import Batch

    # Set seed for reproducibility in data splitting
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load checkpoint
    ckpt_path = f"checkpoints/phase2_{split}/best_model.pt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load structures
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data

    # Load data with this seed
    samples = build_protac8k_degradation_data()
    _, _, test_samples = create_data_splits(samples, split)

    # Build test graphs
    test_graphs = []
    for sample in test_samples:
        uniprot = sample["uniprot_id"]
        if uniprot not in structures:
            continue

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
        test_graphs.append(graph)

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])
            out = model(batch, e3_name=graph.e3_name)
            prob = torch.sigmoid(out["degrado_logits"]).cpu().numpy()
            all_preds.append(prob.item())
            all_labels.append(graph.y.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    # Optimal F1
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    return {
        'auroc': auroc,
        'auprc': auprc,
        'f1': best_f1
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate existing checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()

    results = defaultdict(lambda: defaultdict(list))

    if not args.eval_only:
        # Run training for each seed and split
        for seed in SEEDS:
            for split in SPLITS:
                success = run_single_seed(seed, split, args.gpu)
                if success:
                    print(f"Completed seed={seed}, split={split}")

    # Evaluate all checkpoints
    print("\n" + "="*60)
    print("EVALUATING CHECKPOINTS")
    print("="*60)

    for seed in SEEDS:
        for split in SPLITS:
            metrics = evaluate_checkpoint(seed, split)
            if metrics:
                for k, v in metrics.items():
                    results[split][k].append(v)
                print(f"Seed {seed}, {split}: AUROC={metrics['auroc']:.4f}")

    # Compute statistics
    summary = {}
    print("\n" + "="*60)
    print("MULTI-SEED RESULTS (mean ± std)")
    print("="*60)

    for split in SPLITS:
        summary[split] = {}
        print(f"\n{split}:")

        for metric in ['auroc', 'auprc', 'f1']:
            values = results[split][metric]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                summary[split][metric] = {
                    'mean': float(mean),
                    'std': float(std),
                    'values': [float(v) for v in values]
                }
                print(f"  {metric}: {mean:.4f} ± {std:.4f}")

    # Save results
    output_path = "results/multi_seed_results.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

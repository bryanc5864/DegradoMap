"""
Expanded E3-Unseen Evaluation for DegradoMap

Tests E3 generalization more thoroughly:
1. Leave-one-E3-out for each major E3 ligase
2. E3 recommendation ranking task
3. Per-E3 AUROC analysis
4. Case study: BRD4 E3 selection

Usage:
    python scripts/e3_evaluation.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.degradomap import DegradoMap
from src.data.dataset import DegradationDataset
from src.training.trainer import DegradoMapTrainer, collate_graph_batch
from scripts.train import build_protac8k_degradation_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("E3-Eval")


def leave_one_e3_out_split(samples, held_out_e3, val_ratio=0.15):
    """Create splits holding out a specific E3 ligase."""
    test = [s for s in samples if s["e3_name"] == held_out_e3]
    remaining = [s for s in samples if s["e3_name"] != held_out_e3]

    np.random.shuffle(remaining)
    n_val = int(len(remaining) * val_ratio)
    val = remaining[:n_val]
    train = remaining[n_val:]

    return train, val, test


def train_and_evaluate_e3(model, train_loader, val_loader, test_loader,
                          epochs=10, lr=1e-4, device="cuda"):
    """Train and evaluate model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auroc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            graphs = batch["graph"].to(device)
            labels = batch["label"].to(device)
            e3_name = batch["e3_name"][0]

            optimizer.zero_grad()
            outputs = model(graphs, e3_name)
            loss = criterion(outputs["degrado_logits"], labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["graph"].to(device), batch["e3_name"][0])
                val_scores.extend(outputs["degrado_score"].cpu().numpy())
                val_labels.extend(batch["label"].numpy())

        val_auroc = roc_auc_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0.5
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_scores, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch["graph"].to(device), batch["e3_name"][0])
            test_scores.extend(outputs["degrado_score"].cpu().numpy())
            test_labels.extend(batch["label"].numpy())

    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    if len(set(test_labels)) > 1:
        auroc = roc_auc_score(test_labels, test_scores)
        auprc = average_precision_score(test_labels, test_scores)
    else:
        auroc, auprc = 0.5, 0.0

    f1 = f1_score(test_labels, (test_scores > 0.5).astype(float), zero_division=0)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "n_test": len(test_labels),
        "n_pos": int(test_labels.sum()),
        "best_val_auroc": best_val_auroc,
    }


def e3_recommendation_task(model, samples, device="cuda"):
    """
    E3 Recommendation Task: Given a target protein, rank E3 ligases.

    For each target with multiple E3 annotations, predict scores for all E3s
    and compute ranking metrics.
    """
    model.eval()

    # Group samples by target
    target_to_e3s = defaultdict(list)
    for s in samples:
        target_to_e3s[s["uniprot_id"]].append({
            "e3": s["e3_name"],
            "label": s["label"],
        })

    # Filter to targets with multiple E3s
    multi_e3_targets = {k: v for k, v in target_to_e3s.items() if len(set(x["e3"] for x in v)) >= 2}

    logger.info(f"Found {len(multi_e3_targets)} targets with multiple E3 annotations")

    if len(multi_e3_targets) == 0:
        return {"mrr": 0, "hit_at_1": 0, "hit_at_3": 0}

    # Create dataset for scoring
    all_e3s = list(set(s["e3_name"] for s in samples))

    mrr_scores = []
    hit_at_1 = 0
    hit_at_3 = 0

    for target, e3_info in list(multi_e3_targets.items())[:50]:  # Limit to 50 for speed
        # Get best E3 for this target (highest label)
        best_e3 = max(e3_info, key=lambda x: x["label"])["e3"]

        # Get scores for all E3s
        sample_for_target = next(s for s in samples if s["uniprot_id"] == target)
        test_dataset = DegradationDataset([sample_for_target], use_esm=False)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_graph_batch)
        batch = next(iter(test_loader))

        e3_scores = {}
        with torch.no_grad():
            for e3 in all_e3s:
                outputs = model(batch["graph"].to(device), e3)
                e3_scores[e3] = outputs["degrado_score"].item()

        # Rank E3s
        ranked_e3s = sorted(e3_scores.keys(), key=lambda x: e3_scores[x], reverse=True)

        # Compute metrics
        rank = ranked_e3s.index(best_e3) + 1
        mrr_scores.append(1.0 / rank)
        if rank == 1:
            hit_at_1 += 1
        if rank <= 3:
            hit_at_3 += 1

    n = len(mrr_scores)
    return {
        "mrr": np.mean(mrr_scores),
        "hit_at_1": hit_at_1 / n,
        "hit_at_3": hit_at_3 / n,
        "n_evaluated": n,
    }


def case_study_brd4(model, samples, device="cuda"):
    """
    Case Study: Which E3 is optimal for BRD4?

    BRD4 is a well-studied PROTAC target with data for multiple E3 ligases.
    """
    model.eval()

    # Find BRD4 samples
    brd4_samples = [s for s in samples if "BRD4" in s.get("target_gene", "").upper()]

    if len(brd4_samples) == 0:
        logger.info("No BRD4 samples found")
        return {}

    logger.info(f"Found {len(brd4_samples)} BRD4 samples")

    # Group by E3
    e3_results = defaultdict(lambda: {"pos": 0, "neg": 0, "scores": []})

    for s in brd4_samples:
        e3 = s["e3_name"]
        e3_results[e3]["pos" if s["label"] > 0.5 else "neg"] += 1

        # Get model prediction
        test_dataset = DegradationDataset([s], use_esm=False)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_graph_batch)
        batch = next(iter(test_loader))

        with torch.no_grad():
            outputs = model(batch["graph"].to(device), e3)
            score = outputs["degrado_score"].item()
        e3_results[e3]["scores"].append(score)

    # Summarize
    summary = {}
    for e3, data in e3_results.items():
        summary[e3] = {
            "positive_samples": data["pos"],
            "negative_samples": data["neg"],
            "mean_score": np.mean(data["scores"]),
            "empirical_rate": data["pos"] / (data["pos"] + data["neg"]),
        }

    return summary


def run_e3_evaluation(args):
    """Run comprehensive E3 evaluation."""
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.gpu)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading PROTAC-8K data...")
    samples = build_protac8k_degradation_data()

    # Get E3 distribution
    e3_counts = defaultdict(int)
    for s in samples:
        e3_counts[s["e3_name"]] += 1

    logger.info("\nE3 ligase distribution:")
    for e3, count in sorted(e3_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {e3}: {count} samples")

    results = {"per_e3": {}, "ranking": {}, "case_studies": {}}

    # Part 1: Leave-one-E3-out for each major E3
    logger.info("\n" + "="*60)
    logger.info("Part 1: Leave-One-E3-Out Evaluation")
    logger.info("="*60)

    major_e3s = [e3 for e3, count in e3_counts.items() if count >= 50]
    logger.info(f"Testing {len(major_e3s)} major E3 ligases with >=50 samples each")

    for e3 in major_e3s:
        logger.info(f"\n--- Holding out: {e3} ({e3_counts[e3]} samples) ---")

        train_data, val_data, test_data = leave_one_e3_out_split(samples, e3)

        if len(test_data) < 10:
            logger.info(f"  Skipping - only {len(test_data)} test samples")
            continue

        train_dataset = DegradationDataset(train_data, use_esm=False)
        val_dataset = DegradationDataset(val_data, use_esm=False)
        test_dataset = DegradationDataset(test_data, use_esm=False)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                  collate_fn=collate_graph_batch)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                               collate_fn=collate_graph_batch)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                                collate_fn=collate_graph_batch)

        # Train model
        model = DegradoMap(
            node_input_dim=28, sug_hidden_dim=128, sug_output_dim=64, sug_num_layers=4,
            e3_hidden_dim=64, e3_output_dim=64, e3_num_heads=4, e3_num_layers=2,
            context_output_dim=64, fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
        )

        metrics = train_and_evaluate_e3(
            model, train_loader, val_loader, test_loader,
            epochs=args.epochs, device=device
        )

        results["per_e3"][e3] = metrics
        logger.info(f"  AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f} "
                   f"(n={metrics['n_test']}, pos={metrics['n_pos']})")

    # Part 2: E3 Recommendation Ranking
    logger.info("\n" + "="*60)
    logger.info("Part 2: E3 Recommendation Task")
    logger.info("="*60)

    # Load best model from random split for ranking
    logger.info("Training model on full data for ranking task...")
    np.random.shuffle(samples)
    n_train = int(len(samples) * 0.85)
    train_data = samples[:n_train]
    val_data = samples[n_train:]

    train_dataset = DegradationDataset(train_data, use_esm=False)
    val_dataset = DegradationDataset(val_data, use_esm=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                           collate_fn=collate_graph_batch)

    model = DegradoMap(
        node_input_dim=28, sug_hidden_dim=128, sug_output_dim=64, sug_num_layers=4,
        e3_hidden_dim=64, e3_output_dim=64, e3_num_heads=4, e3_num_layers=2,
        context_output_dim=64, fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch["graph"].to(device), batch["e3_name"][0])
            loss = criterion(outputs["degrado_logits"], batch["label"].to(device))
            loss.backward()
            optimizer.step()

    ranking_metrics = e3_recommendation_task(model, samples, device)
    results["ranking"] = ranking_metrics
    logger.info(f"\nE3 Recommendation Metrics:")
    logger.info(f"  MRR: {ranking_metrics['mrr']:.4f}")
    logger.info(f"  Hit@1: {ranking_metrics['hit_at_1']:.4f}")
    logger.info(f"  Hit@3: {ranking_metrics['hit_at_3']:.4f}")

    # Part 3: Case Study - BRD4
    logger.info("\n" + "="*60)
    logger.info("Part 3: Case Study - BRD4")
    logger.info("="*60)

    brd4_results = case_study_brd4(model, samples, device)
    results["case_studies"]["BRD4"] = brd4_results

    if brd4_results:
        logger.info("\nBRD4 E3 Ligase Analysis:")
        logger.info(f"{'E3':<12} {'Pos':<6} {'Neg':<6} {'Rate':<8} {'Model Score':<12}")
        logger.info("-" * 50)
        for e3, data in sorted(brd4_results.items(), key=lambda x: -x[1]["mean_score"]):
            logger.info(f"{e3:<12} {data['positive_samples']:<6} {data['negative_samples']:<6} "
                       f"{data['empirical_rate']:.3f}{'':>3} {data['mean_score']:.4f}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/e3_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("E3-UNSEEN EVALUATION SUMMARY")
    logger.info("="*80)

    if results["per_e3"]:
        aurocs = [m["auroc"] for m in results["per_e3"].values()]
        logger.info(f"\nLeave-One-E3-Out Results:")
        logger.info(f"  Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        logger.info(f"  Min: {min(aurocs):.4f}, Max: {max(aurocs):.4f}")

        for e3, m in sorted(results["per_e3"].items(), key=lambda x: -x[1]["auroc"]):
            logger.info(f"    {e3}: {m['auroc']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3-Unseen Evaluation")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    args = parser.parse_args()

    run_e3_evaluation(args)

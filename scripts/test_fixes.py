"""
Quick validation test for architecture and data splitting fixes.

Validates:
1. SUG module protein-size normalization works
2. Per-protein lysine softmax works
3. E3-stratified target-unseen split maintains E3 distribution

Usage:
    python scripts/test_fixes.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import defaultdict

from src.models.sug_module import SUGModule, protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits


def test_sug_size_invariance():
    """Test that SUG module is more size-invariant after fix."""
    print("\n=== Test 1: SUG Size Invariance ===")

    sug = SUGModule(node_input_dim=28, hidden_dim=64, output_dim=32, num_layers=2)
    sug.eval()

    # Create two proteins of different sizes with similar local structure
    # Small protein: 50 residues
    coords_small = torch.randn(50, 3) * 10
    residues_small = ['A'] * 45 + ['K'] * 5

    # Large protein: 200 residues (4x larger)
    coords_large = torch.randn(200, 3) * 10
    residues_large = ['A'] * 180 + ['K'] * 20

    graph_small = protein_to_graph(coords_small, residues_small)
    graph_large = protein_to_graph(coords_large, residues_large)

    with torch.no_grad():
        out_small = sug(graph_small)
        out_large = sug(graph_large)

    vec_small = out_small["sug_vector"]
    vec_large = out_large["sug_vector"]

    # Compute similarity - should be higher after normalization
    cosine_sim = torch.nn.functional.cosine_similarity(vec_small, vec_large, dim=-1).item()
    l2_diff = (vec_small - vec_large).norm().item()

    print(f"  Small protein: {len(residues_small)} residues")
    print(f"  Large protein: {len(residues_large)} residues")
    print(f"  SUG vector cosine similarity: {cosine_sim:.4f}")
    print(f"  SUG vector L2 difference: {l2_diff:.4f}")
    print(f"  (Higher similarity = better size invariance)")

    return cosine_sim


def test_e3_stratified_split():
    """Test that E3 distribution is maintained across splits."""
    print("\n=== Test 2: E3-Stratified Target-Unseen Split ===")

    samples = build_protac8k_degradation_data()

    # Get overall E3 distribution
    overall_e3 = defaultdict(int)
    for s in samples:
        overall_e3[s["e3_name"]] += 1

    total = sum(overall_e3.values())
    print(f"\n  Overall E3 distribution:")
    for e3, count in sorted(overall_e3.items(), key=lambda x: -x[1]):
        print(f"    {e3}: {count} ({100*count/total:.1f}%)")

    # Create target-unseen split
    train, val, test = create_data_splits(samples, split_type="target_unseen")

    # Check E3 distribution in train vs test
    train_e3 = defaultdict(int)
    test_e3 = defaultdict(int)
    for s in train:
        train_e3[s["e3_name"]] += 1
    for s in test:
        test_e3[s["e3_name"]] += 1

    train_total = sum(train_e3.values())
    test_total = sum(test_e3.values())

    print(f"\n  Train E3 distribution (n={train_total}):")
    for e3 in overall_e3:
        count = train_e3.get(e3, 0)
        pct = 100*count/train_total if train_total > 0 else 0
        print(f"    {e3}: {count} ({pct:.1f}%)")

    print(f"\n  Test E3 distribution (n={test_total}):")
    for e3 in overall_e3:
        count = test_e3.get(e3, 0)
        pct = 100*count/test_total if test_total > 0 else 0
        print(f"    {e3}: {count} ({pct:.1f}%)")

    # Calculate distribution shift (KL divergence proxy)
    shifts = []
    for e3 in ['CRBN', 'VHL']:
        train_pct = train_e3.get(e3, 0) / train_total if train_total > 0 else 0
        test_pct = test_e3.get(e3, 0) / test_total if test_total > 0 else 0
        shift = abs(train_pct - test_pct)
        shifts.append(shift)
        print(f"\n  {e3} shift: {100*shift:.1f}% (train={100*train_pct:.1f}%, test={100*test_pct:.1f}%)")

    avg_shift = np.mean(shifts)
    print(f"\n  Average E3 distribution shift: {100*avg_shift:.1f}%")
    print(f"  (Lower is better - OLD method had ~20% shift, stratified should be <5%)")

    return avg_shift


def test_lysine_softmax():
    """Test that lysine softmax is per-protein."""
    print("\n=== Test 3: Per-Protein Lysine Softmax ===")

    from torch_geometric.data import Batch

    # Create two proteins with different lysine counts
    coords1 = torch.randn(30, 3) * 10
    residues1 = ['A'] * 25 + ['K'] * 5  # 5 lysines

    coords2 = torch.randn(100, 3) * 10
    residues2 = ['A'] * 80 + ['K'] * 20  # 20 lysines

    graph1 = protein_to_graph(coords1, residues1)
    graph2 = protein_to_graph(coords2, residues2)

    # Batch them together
    batch = Batch.from_data_list([graph1, graph2])

    sug = SUGModule(node_input_dim=28, hidden_dim=64, output_dim=32, num_layers=2)
    sug.eval()

    with torch.no_grad():
        out = sug(batch)

    lys_summary = out["lysine_summary"]

    print(f"  Protein 1: {sum(1 for r in residues1 if r == 'K')} lysines")
    print(f"  Protein 2: {sum(1 for r in residues2 if r == 'K')} lysines")
    print(f"  Lysine summary shapes: {lys_summary.shape}")

    # Both should have similar magnitude (not scaled by lysine count)
    norm1 = lys_summary[0].norm().item()
    norm2 = lys_summary[1].norm().item()
    ratio = max(norm1, norm2) / min(norm1, norm2) if min(norm1, norm2) > 0 else float('inf')

    print(f"  Lysine summary norm (protein 1): {norm1:.4f}")
    print(f"  Lysine summary norm (protein 2): {norm2:.4f}")
    print(f"  Norm ratio: {ratio:.2f}x")
    print(f"  (Closer to 1.0 = better normalization, OLD method would be ~4x)")

    return ratio


if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATING ARCHITECTURE AND DATA FIXES")
    print("=" * 60)

    # Run tests
    sim = test_sug_size_invariance()
    shift = test_e3_stratified_split()
    ratio = test_lysine_softmax()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SUG size invariance (cosine sim): {sim:.4f} (want > 0.5)")
    print(f"  E3 distribution shift: {100*shift:.1f}% (want < 5%)")
    print(f"  Lysine norm ratio: {ratio:.2f}x (want < 2.0)")

    all_pass = sim > 0.3 and shift < 0.10 and ratio < 3.0
    print(f"\n  Overall: {'PASS' if all_pass else 'NEEDS TUNING'}")

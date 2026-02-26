"""
Computational Efficiency Benchmarks for DegradoMap.

Measures:
1. Training time vs dataset size
2. Inference time per protein
3. Memory usage vs protein size
4. Comparison with baselines
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List
from torch_geometric.data import Batch
import psutil
import gc

from src.models.degradomap import DegradoMap
from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data
from scripts.gnn_baselines import SchNetModel, EGNNModel


def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_inference(model, graphs, device, model_name, n_runs=3):
    """Benchmark inference time and memory."""
    model.eval()

    # Warmup
    for graph in graphs[:5]:
        graph = graph.to(device)
        batch = Batch.from_data_list([graph])
        with torch.no_grad():
            if hasattr(model, 'forward'):
                if 'DegradoMap' in model_name:
                    _ = model(batch, e3_name=graph.e3_name)
                else:
                    _ = model(batch, e3_idx=getattr(graph, 'e3_idx', 0))

    # Benchmark
    times = []
    memories = []

    for run in range(n_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        run_times = []
        run_memories = []

        for graph in graphs:
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])

            mem_before = get_gpu_memory()
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start = time.perf_counter()
            with torch.no_grad():
                if 'DegradoMap' in model_name:
                    _ = model(batch, e3_name=graph.e3_name)
                else:
                    _ = model(batch, e3_idx=getattr(graph, 'e3_idx', 0))

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            mem_after = get_gpu_memory()

            run_times.append((end - start) * 1000)  # ms
            run_memories.append(mem_after - mem_before)

        times.append(np.mean(run_times))
        memories.append(np.max(run_memories))

    return {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'max_memory_mb': float(np.max(memories)),
        'n_samples': len(graphs),
    }


def benchmark_training(model_class, train_graphs, device, model_name, epochs=5):
    """Benchmark training time."""
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mem_before = get_gpu_memory()
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        for graph in train_graphs[:100]:  # Use subset for speed
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])
            optimizer.zero_grad()

            if 'DegradoMap' in model_name:
                out = model(batch, e3_name=graph.e3_name)
                logits = out["degrado_logits"]
            else:
                out = model(batch, e3_idx=getattr(graph, 'e3_idx', 0))
                logits = out

            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), graph.y.squeeze())
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    mem_after = get_gpu_memory()

    return {
        'total_time_s': float(end - start),
        'time_per_epoch_s': float((end - start) / epochs),
        'peak_memory_mb': float(mem_after),
        'n_samples': min(100, len(train_graphs)),
        'epochs': epochs,
    }


def benchmark_by_protein_size(model, structures, device):
    """Benchmark inference time by protein size."""
    model.eval()

    size_bins = {
        'small (<200)': [],
        'medium (200-500)': [],
        'large (500-1000)': [],
        'xlarge (>1000)': [],
    }

    for uniprot, struct in structures.items():
        n_res = len(struct["residues"])
        if n_res < 200:
            size_bins['small (<200)'].append((uniprot, struct))
        elif n_res < 500:
            size_bins['medium (200-500)'].append((uniprot, struct))
        elif n_res < 1000:
            size_bins['large (500-1000)'].append((uniprot, struct))
        else:
            size_bins['xlarge (>1000)'].append((uniprot, struct))

    results = {}
    for bin_name, proteins in size_bins.items():
        if not proteins:
            continue

        times = []
        sizes = []
        for uniprot, struct in proteins[:20]:  # Max 20 per bin
            graph = protein_to_graph(
                coords=struct["coords"],
                residues=struct["residues"],
                plddt=struct.get("plddt"),
                sasa=struct.get("sasa"),
                disorder=struct.get("disorder")
            )
            graph.e3_name = "VHL"
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            with torch.no_grad():
                _ = model(batch, e3_name="VHL")

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            times.append((end - start) * 1000)
            sizes.append(len(struct["residues"]))

        results[bin_name] = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'mean_size': float(np.mean(sizes)),
            'n_proteins': len(proteins),
        }

    return results


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    structures = load_structures()
    samples = build_protac8k_degradation_data()

    # Build test graphs
    print("Building graphs...")
    test_graphs = []
    E3_TO_IDX = {'CRBN': 0, 'VHL': 1, 'cIAP1': 2, 'MDM2': 3, 'XIAP': 4}

    for sample in samples[:200]:  # Use subset
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
        graph.e3_idx = E3_TO_IDX.get(sample["e3_name"], 0)
        graph.pos = struct["coords"]
        test_graphs.append(graph)

    print(f"Built {len(test_graphs)} test graphs")

    results = {}

    # Models to benchmark
    models = [
        (DegradoMap, "DegradoMap", {
            'node_input_dim': 28, 'sug_hidden_dim': 128, 'sug_output_dim': 64,
            'sug_num_layers': 4, 'e3_hidden_dim': 64, 'e3_output_dim': 64,
            'e3_num_heads': 4, 'e3_num_layers': 2, 'context_output_dim': 64,
            'fusion_hidden_dim': 128, 'pred_hidden_dim': 64, 'dropout': 0.1
        }),
        (SchNetModel, "SchNet", {}),
        (EGNNModel, "EGNN", {}),
    ]

    for model_class, model_name, kwargs in models:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")

        model = model_class(**kwargs).to(device)

        # Parameter count
        params = count_parameters(model)
        print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

        # Inference benchmark
        print("Benchmarking inference...")
        inference_results = benchmark_inference(model, test_graphs[:50], device, model_name)
        print(f"  Mean time: {inference_results['mean_time_ms']:.2f} ± {inference_results['std_time_ms']:.2f} ms")

        # Training benchmark
        print("Benchmarking training...")
        if model_name == "DegradoMap":
            train_results = benchmark_training(
                lambda: DegradoMap(**kwargs),
                test_graphs, device, model_name
            )
        else:
            train_results = benchmark_training(model_class, test_graphs, device, model_name)
        print(f"  Time per epoch: {train_results['time_per_epoch_s']:.2f} s")

        results[model_name] = {
            'parameters': params,
            'inference': inference_results,
            'training': train_results,
        }

    # Size scaling benchmark (DegradoMap only)
    print("\n" + "="*60)
    print("Benchmarking by protein size (DegradoMap)")
    print("="*60)

    model = DegradoMap(
        node_input_dim=28, sug_hidden_dim=128, sug_output_dim=64,
        sug_num_layers=4, e3_hidden_dim=64, e3_output_dim=64,
        e3_num_heads=4, e3_num_layers=2, context_output_dim=64,
        fusion_hidden_dim=128, pred_hidden_dim=64, dropout=0.1
    ).to(device)

    size_results = benchmark_by_protein_size(model, structures, device)
    results['by_protein_size'] = size_results

    for bin_name, stats in size_results.items():
        print(f"  {bin_name}: {stats['mean_time_ms']:.2f} ms (n={stats['n_proteins']})")

    # Summary table
    print("\n" + "="*80)
    print("EFFICIENCY BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Model':<15} {'Params':<12} {'Inference (ms)':<18} {'Train/epoch (s)':<15}")
    print("-"*60)

    for model_name in ['DegradoMap', 'SchNet', 'EGNN']:
        if model_name in results:
            r = results[model_name]
            params = f"{r['parameters']['total']/1e6:.2f}M"
            inf = f"{r['inference']['mean_time_ms']:.1f} ± {r['inference']['std_time_ms']:.1f}"
            train = f"{r['training']['time_per_epoch_s']:.2f}"
            print(f"{model_name:<15} {params:<12} {inf:<18} {train:<15}")

    # Save results
    output_path = "results/efficiency_benchmark.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

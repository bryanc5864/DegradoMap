#!/usr/bin/env python3
"""
Ablation study for DegradoMap features.

Runs experiments with different feature combinations to quantify
the contribution of each component.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_ablation(gpu_id, name, flags, seed=42, epochs=15):
    """Run a single ablation experiment."""
    cmd = [
        "python", "scripts/train_improved.py",
        "--mode", "single",
        "--epochs", str(epochs),
        "--patience", "5",
        "--seed", str(seed),
    ] + flags

    log_file = f"results/ablation_{name}_seed{seed}.log"

    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"Starting {name} on GPU {gpu_id}...")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**subprocess.os.environ, **env}
        )

    return process, log_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="3,4,5,6,8,9",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # Define ablation configurations
    ablations = [
        # name, flags
        ("baseline", ["--no-esm", "--no-ub", "--no-e3oh", "--no-gs"]),
        ("esm_only", ["--no-ub", "--no-e3oh", "--no-gs"]),
        ("ub_only", ["--no-esm", "--no-e3oh", "--no-gs"]),
        ("e3oh_only", ["--no-esm", "--no-ub", "--no-gs"]),
        ("gs_only", ["--no-esm", "--no-ub", "--no-e3oh"]),
        ("all_features", []),  # All features enabled
    ]

    processes = []

    for i, (name, flags) in enumerate(ablations):
        if i >= len(gpus):
            print(f"Waiting for GPU... (only {len(gpus)} available)")
            # Wait for a process to finish
            for p, _ in processes:
                p.wait()
            processes = []

        gpu = gpus[i % len(gpus)]
        proc, log = run_ablation(gpu, name, flags, args.seed, args.epochs)
        processes.append((proc, log))
        time.sleep(5)  # Stagger starts

    print(f"\nStarted {len(ablations)} ablation experiments")
    print("Logs in results/ablation_*.log")
    print("\nMonitor with: tail -f results/ablation_*.log")

    # Wait for all to complete
    print("\nWaiting for completion...")
    for proc, log in processes:
        proc.wait()
        print(f"Completed: {log}")

    # Collect results
    results = {}
    for name, _ in ablations:
        log_file = f"results/ablation_{name}_seed{args.seed}.log"
        try:
            with open(log_file) as f:
                content = f.read()
                # Extract test AUROC
                for line in content.split("\n"):
                    if "AUROC:" in line and "val_auroc" not in line:
                        auroc = float(line.split("AUROC:")[-1].strip())
                        results[name] = auroc
                        break
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    print("\n" + "="*50)
    print("ABLATION RESULTS")
    print("="*50)
    for name, auroc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:20s}: {auroc:.4f}")

    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/ablation_results.json")


if __name__ == "__main__":
    main()

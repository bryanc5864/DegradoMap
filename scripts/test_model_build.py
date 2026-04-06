#!/usr/bin/env python3
"""
Quick test to verify the model builds correctly with new features.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.data import Data, Batch

from src.models.degradomap import DegradoMap


def test_model_variants():
    """Test model builds with different feature combinations."""

    configs = [
        {"name": "baseline", "node_input_dim": 28, "use_e3_onehot": False, "use_global_stats": False},
        {"name": "with_ub_sites", "node_input_dim": 29, "use_e3_onehot": False, "use_global_stats": False},
        {"name": "with_e3_onehot", "node_input_dim": 28, "use_e3_onehot": True, "use_global_stats": False},
        {"name": "with_global_stats", "node_input_dim": 28, "use_e3_onehot": False, "use_global_stats": True},
        {"name": "with_esm", "node_input_dim": 1285, "use_e3_onehot": False, "use_global_stats": False},
        {"name": "all_features", "node_input_dim": 1285, "use_e3_onehot": True, "use_global_stats": True},
    ]

    for cfg in configs:
        print(f"\nTesting {cfg['name']}...")

        model = DegradoMap(
            node_input_dim=cfg["node_input_dim"],
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
            use_e3_onehot=cfg["use_e3_onehot"],
            use_global_stats=cfg["use_global_stats"],
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Create dummy graph data
        n_nodes = 100
        feat_dim = cfg["node_input_dim"]

        # Generate random node features
        x = torch.randn(n_nodes, feat_dim)
        pos = torch.randn(n_nodes, 3) * 10

        # Create edges (simple radius graph)
        dist = torch.cdist(pos, pos)
        mask = (dist < 8.0) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_len = edge_vec.norm(dim=-1, keepdim=True)

        # Create lysine mask (roughly 10% lysines)
        lysine_mask = torch.zeros(n_nodes)
        lysine_mask[torch.randperm(n_nodes)[:n_nodes//10]] = 1.0

        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_len=edge_len,
            lysine_mask=lysine_mask,
            num_nodes=n_nodes,
        )

        # Batch two graphs
        data_batch = Batch.from_data_list([data, data])

        # Forward pass
        model.eval()
        with torch.no_grad():
            out = model(data_batch, e3_name="CRBN")

        print(f"  degrado_score shape: {out['degrado_score'].shape}")
        print(f"  degrado_score values: {out['degrado_score']}")
        print(f"  sug_vector shape: {out['sug_vector'].shape}")

        if cfg["use_global_stats"]:
            print(f"  (includes global_stats in sug_vector)")

        print(f"  SUCCESS")

    print("\n" + "="*50)
    print("All model variants built and ran successfully!")
    print("="*50)


if __name__ == "__main__":
    test_model_variants()

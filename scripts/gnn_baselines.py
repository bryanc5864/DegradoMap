"""
GNN Baselines for PROTAC Degradability Prediction.

Implements proper structure-based GNN baselines:
1. SchNet - continuous-filter convolutional neural network
2. EGNN - E(n) equivariant graph neural network
3. ESM-2 + MLP - protein language model baseline

All baselines use same data splits and evaluation for fair comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from tqdm import tqdm

from src.models.sug_module import protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits


# =============================================================================
# SchNet Implementation
# =============================================================================

class GaussianSmearing(nn.Module):
    """Gaussian smearing of interatomic distances."""

    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist ** 2)


class CFConv(MessagePassing):
    """Continuous-filter convolution layer from SchNet."""

    def __init__(self, in_channels, out_channels, num_filters, num_gaussians):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )

    def forward(self, x, edge_index, edge_weight, edge_attr):
        W = self.filter_net(edge_attr)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class SchNetModel(nn.Module):
    """SchNet model for protein degradability prediction."""

    def __init__(
        self,
        node_dim: int = 28,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_interactions: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cutoff = cutoff

        # Embedding
        self.embedding = nn.Linear(node_dim, hidden_dim)

        # Distance encoding
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            CFConv(hidden_dim, hidden_dim, hidden_dim, num_gaussians)
            for _ in range(num_interactions)
        ])

        # E3 embedding
        self.e3_embedding = nn.Embedding(12, hidden_dim)  # 11 E3s + unknown

        # Prediction head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data: Data, e3_idx: int = 0) -> torch.Tensor:
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute edge attributes
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_weight = edge_vec.norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        # Initial embedding
        x = self.embedding(x)

        # Interaction blocks with residual connections
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)

        # Pooling with size normalization
        num_nodes = torch.bincount(batch).float().sqrt().unsqueeze(1).to(x.device)
        x = global_mean_pool(x, batch) / num_nodes

        # E3 embedding
        e3_emb = self.e3_embedding(torch.tensor([e3_idx], device=x.device))

        # Combine and predict
        combined = torch.cat([x, e3_emb.expand(x.size(0), -1)], dim=-1)
        return self.output(combined)


# =============================================================================
# EGNN Implementation
# =============================================================================

class EGNNConv(MessagePassing):
    """E(n) Equivariant Graph Neural Network convolution."""

    def __init__(self, in_channels, out_channels, edge_dim=1):
        super().__init__(aggr='add')

        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(msg_input)

    def update(self, aggr_out, x):
        upd_input = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_upd(upd_input)


class EGNNModel(nn.Module):
    """EGNN model for protein degradability prediction."""

    def __init__(
        self,
        node_dim: int = 28,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNConv(hidden_dim, hidden_dim, edge_dim=1)
            for _ in range(num_layers)
        ])

        self.e3_embedding = nn.Embedding(12, hidden_dim)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data: Data, e3_idx: int = 0) -> torch.Tensor:
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute edge distances
        row, col = edge_index
        edge_attr = (pos[col] - pos[row]).norm(dim=-1, keepdim=True)

        # Embedding
        x = self.embedding(x)

        # Message passing with residual
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)

        # Pooling
        num_nodes = torch.bincount(batch).float().sqrt().unsqueeze(1).to(x.device)
        x = global_mean_pool(x, batch) / num_nodes

        # E3 + predict
        e3_emb = self.e3_embedding(torch.tensor([e3_idx], device=x.device))
        combined = torch.cat([x, e3_emb.expand(x.size(0), -1)], dim=-1)
        return self.output(combined)


# =============================================================================
# ESM-2 + MLP Baseline
# =============================================================================

class ESM2MLPModel(nn.Module):
    """ESM-2 embeddings + MLP baseline."""

    def __init__(
        self,
        esm_dim: int = 1280,
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.e3_embedding = nn.Embedding(12, 64)

        self.mlp = nn.Sequential(
            nn.Linear(esm_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, esm_embedding: torch.Tensor, e3_idx: int = 0) -> torch.Tensor:
        e3_emb = self.e3_embedding(torch.tensor([e3_idx], device=esm_embedding.device))
        combined = torch.cat([esm_embedding, e3_emb], dim=-1)
        return self.mlp(combined)


# =============================================================================
# Training and Evaluation
# =============================================================================

E3_TO_IDX = {
    'CRBN': 0, 'VHL': 1, 'cIAP1': 2, 'MDM2': 3, 'XIAP': 4,
    'DCAF16': 5, 'KEAP1': 6, 'FEM1B': 7, 'DCAF1': 8, 'UBR': 9, 'KLHL20': 10
}


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
    """Load pre-computed ESM-2 embeddings."""
    esm_dir = Path("data/processed/esm_embeddings")
    embeddings = {}
    if esm_dir.exists():
        for pt_file in esm_dir.glob("*.pt"):
            uniprot = pt_file.stem
            embeddings[uniprot] = torch.load(pt_file, map_location='cpu', weights_only=False)
    return embeddings


def build_graph(sample: Dict, structures: Dict) -> Optional[Data]:
    """Build graph for a sample."""
    uniprot = sample["uniprot_id"]
    if uniprot not in structures:
        return None

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
    graph.e3_idx = E3_TO_IDX.get(sample["e3_name"], 11)
    graph.pos = struct["coords"]
    return graph


def train_epoch(model, train_graphs, optimizer, device, model_type='gnn'):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for graph in train_graphs:
        graph = graph.to(device)
        optimizer.zero_grad()

        if model_type == 'esm':
            out = model(graph.esm_emb.unsqueeze(0), graph.e3_idx)
        else:
            batch = Batch.from_data_list([graph])
            out = model(batch, graph.e3_idx)

        loss = F.binary_cross_entropy_with_logits(out.squeeze(), graph.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_graphs)


def evaluate(model, graphs, device, model_type='gnn'):
    """Evaluate model."""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)

            if model_type == 'esm':
                out = model(graph.esm_emb.unsqueeze(0), graph.e3_idx)
            else:
                batch = Batch.from_data_list([graph])
                out = model(batch, graph.e3_idx)

            prob = torch.sigmoid(out).cpu().numpy()
            preds.append(prob.item())
            labels.append(graph.y.item())

    preds = np.array(preds)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)

    # Best F1
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(labels, (preds >= thresh).astype(int), zero_division=0)
        best_f1 = max(best_f1, f1)

    return {'auroc': auroc, 'auprc': auprc, 'f1': best_f1}


def run_baseline(model_class, model_name, split_type, structures, samples,
                 esm_embeddings=None, device='cuda', epochs=20, seed=42):
    """Run a single baseline experiment."""
    print(f"\n{'='*60}")
    print(f"{model_name} on {split_type} (seed={seed})")
    print(f"{'='*60}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get splits
    train_samples, val_samples, test_samples = create_data_splits(samples, split_type, seed=seed)

    # Build graphs
    model_type = 'esm' if 'ESM' in model_name else 'gnn'

    def build_graphs(sample_list):
        graphs = []
        for s in sample_list:
            g = build_graph(s, structures)
            if g is not None:
                if model_type == 'esm':
                    if s["uniprot_id"] in esm_embeddings:
                        g.esm_emb = esm_embeddings[s["uniprot_id"]].mean(dim=0)  # Mean pool
                        graphs.append(g)
                else:
                    graphs.append(g)
        return graphs

    train_graphs = build_graphs(train_samples)
    val_graphs = build_graphs(val_samples)
    test_graphs = build_graphs(test_samples)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Initialize model
    if 'ESM' in model_name:
        model = model_class().to(device)
    else:
        model = model_class().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, device, model_type)
        val_metrics = evaluate(model, val_graphs, device, model_type)
        scheduler.step()

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_auroc={val_metrics['auroc']:.4f}")

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_graphs, device, model_type)

    print(f"\nTest Results: AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}, F1={test_metrics['f1']:.4f}")

    return {
        'model': model_name,
        'split': split_type,
        'seed': seed,
        'n_train': len(train_graphs),
        'n_val': len(val_graphs),
        'n_test': len(test_graphs),
        'best_val_auroc': best_val_auroc,
        **{f'test_{k}': v for k, v in test_metrics.items()}
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading structures...")
    structures = load_structures()
    print(f"Loaded {len(structures)} structures")

    print("Loading ESM embeddings...")
    esm_embeddings = load_esm_embeddings()
    print(f"Loaded {len(esm_embeddings)} ESM embeddings")

    print("Loading samples...")
    samples = build_protac8k_degradation_data()
    print(f"Loaded {len(samples)} samples")

    all_results = []

    # Models to test
    models = [
        (SchNetModel, "SchNet"),
        (EGNNModel, "EGNN"),
    ]

    if esm_embeddings:
        models.append((ESM2MLPModel, "ESM2-MLP"))

    # Run experiments
    for split in ['target_unseen', 'e3_unseen', 'random']:
        for model_class, model_name in models:
            # Multiple seeds
            for seed in [42, 123, 456]:
                try:
                    result = run_baseline(
                        model_class, model_name, split, structures, samples,
                        esm_embeddings, device, epochs=20, seed=seed
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Error: {model_name} on {split} seed={seed}: {e}")

    # Aggregate results
    print("\n" + "="*80)
    print("GNN BASELINES SUMMARY (mean ± std across seeds)")
    print("="*80)

    aggregated = {}
    for r in all_results:
        key = (r['model'], r['split'])
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(r['test_auroc'])

    print(f"\n{'Model':<15} {'Split':<15} {'AUROC':<20}")
    print("-"*50)
    for (model, split), aurocs in sorted(aggregated.items()):
        mean = np.mean(aurocs)
        std = np.std(aurocs)
        print(f"{model:<15} {split:<15} {mean:.4f} ± {std:.4f}")

    # Save results
    output_path = "results/gnn_baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

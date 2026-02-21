"""
Comprehensive Ablation Study for DegradoMap.

Covers:
1. Feature ablations (pLDDT, SASA, lysine, physicochemical)
2. Architecture ablations (layers, hidden dim, heads, cutoff)
3. Pooling ablations (mean, attention, with/without size norm)
4. Training ablations (pre-training, class balancing, learning rates)

Target: 15-20 ablation rows for ICML standard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_add_pool
import logging

from src.models.degradomap import DegradoMap
from src.models.sug_module import SUGModule, protein_to_graph
from scripts.train import build_protac8k_degradation_data, create_data_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ablation")


# =============================================================================
# Feature Ablation Utilities
# =============================================================================

def protein_to_graph_ablated(
    coords: torch.Tensor,
    residues: List[str],
    plddt: Optional[torch.Tensor] = None,
    sasa: Optional[torch.Tensor] = None,
    disorder: Optional[torch.Tensor] = None,
    radius: float = 10.0,
    # Ablation flags
    use_plddt: bool = True,
    use_sasa: bool = True,
    use_lysine: bool = True,
    use_physicochemical: bool = True,
    use_disorder: bool = True,
) -> Data:
    """Build graph with ablated features."""

    # Amino acid properties
    AA_PROPERTIES = {
        'A': [0, 0, 0, 0], 'R': [0, 1, 1, 1], 'N': [0, 0, 0, 1], 'D': [-1, 0, 0, 1],
        'C': [0, 0, 0, 0], 'Q': [0, 0, 0, 1], 'E': [-1, 0, 1, 1], 'G': [0, 0, 0, 0],
        'H': [0, 1, 0, 1], 'I': [0, 0, 0, 0], 'L': [0, 0, 0, 0], 'K': [1, 1, 1, 1],
        'M': [0, 0, 0, 0], 'F': [0, 0, 1, 0], 'P': [0, 0, 0, 0], 'S': [0, 0, 0, 1],
        'T': [0, 0, 0, 1], 'W': [0, 0, 1, 0], 'Y': [0, 0, 1, 1], 'V': [0, 0, 0, 0],
    }
    AA_LIST = list('ARNDCQEGHILKMFPSTWYV')
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

    n_residues = len(residues)
    node_feats = []

    for i, res in enumerate(residues):
        res = res.upper()
        if res not in AA_TO_IDX:
            res = 'A'

        feat = []

        # One-hot amino acid (always included)
        onehot = [0.0] * 20
        onehot[AA_TO_IDX[res]] = 1.0
        feat.extend(onehot)

        # Physicochemical properties
        if use_physicochemical:
            feat.extend([float(x) for x in AA_PROPERTIES.get(res, [0, 0, 0, 0])])
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

        # pLDDT
        if use_plddt and plddt is not None:
            feat.append(float(plddt[i]) / 100.0)
        else:
            feat.append(0.0)

        # SASA
        if use_sasa and sasa is not None:
            feat.append(float(sasa[i]) / 200.0)
        else:
            feat.append(0.0)

        # Lysine indicator
        if use_lysine:
            feat.append(1.0 if res == 'K' else 0.0)
        else:
            feat.append(0.0)

        # Disorder
        if use_disorder and disorder is not None:
            feat.append(float(disorder[i]))
        else:
            feat.append(0.0)

        node_feats.append(feat)

    x = torch.tensor(node_feats, dtype=torch.float32)

    # Build edges (radius graph)
    dist_matrix = torch.cdist(coords, coords)
    mask = (dist_matrix < radius) & (dist_matrix > 0)
    edge_index = mask.nonzero(as_tuple=False).t()

    # Edge attributes
    edge_vec = coords[edge_index[1]] - coords[edge_index[0]]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)

    # Lysine mask
    lysine_mask = torch.tensor([1 if r.upper() == 'K' else 0 for r in residues], dtype=torch.bool)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_len=edge_len,
        pos=coords,
        lysine_mask=lysine_mask,
        num_nodes=n_residues,
    )


# =============================================================================
# Model Variants
# =============================================================================

class SUGModuleVariant(nn.Module):
    """SUG module with configurable architecture."""

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        pooling: str = 'mean',  # 'mean', 'sum', 'attention'
        size_normalize: bool = True,
    ):
        super().__init__()
        self.pooling = pooling
        self.size_normalize = size_normalize

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        if pooling == 'attention':
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.input_proj(x)

        for layer in self.layers:
            row, col = edge_index
            msg = torch.cat([h[row], h[col]], dim=-1)
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, layer(msg))
            h = h + agg

        h = self.output_proj(h)

        # Pooling
        if self.pooling == 'mean':
            pooled = global_mean_pool(h, batch)
        elif self.pooling == 'sum':
            pooled = global_add_pool(h, batch)
        elif self.pooling == 'attention':
            attn_weights = torch.softmax(self.attn(h), dim=0)
            pooled = global_add_pool(h * attn_weights, batch)

        # Size normalization
        if self.size_normalize:
            num_nodes = torch.bincount(batch).float().sqrt().unsqueeze(1).to(h.device)
            pooled = pooled / num_nodes

        return h, pooled


# =============================================================================
# Training and Evaluation
# =============================================================================

def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def build_graph_with_ablation(sample: Dict, structures: Dict, **ablation_kwargs) -> Optional[Data]:
    """Build graph with ablation settings."""
    uniprot = sample["uniprot_id"]
    if uniprot not in structures:
        return None

    struct = structures[uniprot]
    graph = protein_to_graph_ablated(
        coords=struct["coords"],
        residues=struct["residues"],
        plddt=struct.get("plddt"),
        sasa=struct.get("sasa"),
        disorder=struct.get("disorder"),
        **ablation_kwargs
    )
    graph.y = torch.tensor([sample["label"]], dtype=torch.float32)
    graph.e3_name = sample["e3_name"]
    return graph


def train_and_evaluate(
    train_graphs: List,
    val_graphs: List,
    test_graphs: List,
    device: str,
    model_config: Dict,
    epochs: int = 15,
    seed: int = 42,
) -> Dict:
    """Train and evaluate with given configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    model = DegradoMap(
        node_input_dim=model_config.get('node_input_dim', 28),
        sug_hidden_dim=model_config.get('sug_hidden_dim', 128),
        sug_output_dim=model_config.get('sug_output_dim', 64),
        sug_num_layers=model_config.get('sug_num_layers', 4),
        e3_hidden_dim=64,
        e3_output_dim=64,
        e3_num_heads=model_config.get('e3_num_heads', 4),
        e3_num_layers=2,
        context_output_dim=64,
        fusion_hidden_dim=128,
        pred_hidden_dim=64,
        dropout=model_config.get('dropout', 0.1),
    ).to(device)

    lr = model_config.get('learning_rate', 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auroc = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for graph in train_graphs:
            graph = graph.to(device)
            optimizer.zero_grad()
            batch = Batch.from_data_list([graph])
            out = model(batch, e3_name=graph.e3_name)
            loss = F.binary_cross_entropy_with_logits(
                out["degrado_logits"].squeeze(), graph.y.squeeze()
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for graph in val_graphs:
                graph = graph.to(device)
                batch = Batch.from_data_list([graph])
                out = model(batch, e3_name=graph.e3_name)
                val_preds.append(torch.sigmoid(out["degrado_logits"]).cpu().item())
                val_labels.append(graph.y.item())

        val_auroc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])
            out = model(batch, e3_name=graph.e3_name)
            test_preds.append(torch.sigmoid(out["degrado_logits"]).cpu().item())
            test_labels.append(graph.y.item())

    test_auroc = roc_auc_score(test_labels, test_preds) if len(set(test_labels)) > 1 else 0.5
    test_auprc = average_precision_score(test_labels, test_preds) if len(set(test_labels)) > 1 else 0.0

    return {
        'val_auroc': best_val_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
    }


# =============================================================================
# Ablation Configurations
# =============================================================================

ABLATIONS = {
    # Feature ablations
    'full_model': {'desc': 'Full model (baseline)'},
    'no_plddt': {'desc': 'Remove pLDDT', 'feature_ablation': {'use_plddt': False}},
    'no_sasa': {'desc': 'Remove SASA', 'feature_ablation': {'use_sasa': False}},
    'no_lysine': {'desc': 'Remove lysine indicator', 'feature_ablation': {'use_lysine': False}},
    'no_physicochemical': {'desc': 'Remove physicochemical', 'feature_ablation': {'use_physicochemical': False}},
    'no_disorder': {'desc': 'Remove disorder', 'feature_ablation': {'use_disorder': False}},
    'only_onehot': {'desc': 'Only AA one-hot', 'feature_ablation': {
        'use_plddt': False, 'use_sasa': False, 'use_lysine': False,
        'use_physicochemical': False, 'use_disorder': False
    }},

    # Architecture ablations
    'layers_2': {'desc': 'GNN layers=2', 'model_config': {'sug_num_layers': 2}},
    'layers_6': {'desc': 'GNN layers=6', 'model_config': {'sug_num_layers': 6}},
    'layers_8': {'desc': 'GNN layers=8', 'model_config': {'sug_num_layers': 8}},
    'hidden_64': {'desc': 'Hidden dim=64', 'model_config': {'sug_hidden_dim': 64}},
    'hidden_256': {'desc': 'Hidden dim=256', 'model_config': {'sug_hidden_dim': 256}},
    'heads_1': {'desc': 'Attention heads=1', 'model_config': {'e3_num_heads': 1}},
    'heads_8': {'desc': 'Attention heads=8', 'model_config': {'e3_num_heads': 8}},

    # Cutoff ablations
    'cutoff_6': {'desc': 'Cutoff=6A', 'graph_config': {'radius': 6.0}},
    'cutoff_12': {'desc': 'Cutoff=12A', 'graph_config': {'radius': 12.0}},

    # Training ablations
    'lr_5e4': {'desc': 'LR=5e-4', 'model_config': {'learning_rate': 5e-4}},
    'lr_5e3': {'desc': 'LR=5e-3', 'model_config': {'learning_rate': 5e-3}},
    'dropout_0': {'desc': 'Dropout=0', 'model_config': {'dropout': 0.0}},
    'dropout_0.2': {'desc': 'Dropout=0.2', 'model_config': {'dropout': 0.2}},
}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading structures...")
    structures = load_structures()
    logger.info(f"Loaded {len(structures)} structures")

    logger.info("Loading samples...")
    samples = build_protac8k_degradation_data()
    logger.info(f"Loaded {len(samples)} samples")

    results = []

    # Run ablations on target_unseen split
    split_type = 'target_unseen'
    train_samples, val_samples, test_samples = create_data_splits(samples, split_type)

    for ablation_name, config in ABLATIONS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: {ablation_name} - {config['desc']}")
        logger.info(f"{'='*60}")

        # Get ablation settings
        feature_ablation = config.get('feature_ablation', {})
        model_config = config.get('model_config', {})
        graph_config = config.get('graph_config', {})

        # Build graphs with ablation
        train_graphs = []
        for s in train_samples:
            g = build_graph_with_ablation(s, structures, **feature_ablation, **graph_config)
            if g is not None:
                train_graphs.append(g)

        val_graphs = []
        for s in val_samples:
            g = build_graph_with_ablation(s, structures, **feature_ablation, **graph_config)
            if g is not None:
                val_graphs.append(g)

        test_graphs = []
        for s in test_samples:
            g = build_graph_with_ablation(s, structures, **feature_ablation, **graph_config)
            if g is not None:
                test_graphs.append(g)

        logger.info(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

        # Train and evaluate
        metrics = train_and_evaluate(
            train_graphs, val_graphs, test_graphs,
            device, model_config, epochs=15
        )

        result = {
            'ablation': ablation_name,
            'description': config['desc'],
            **metrics
        }
        results.append(result)

        logger.info(f"Test AUROC: {metrics['test_auroc']:.4f}")

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS (target_unseen split)")
    print("="*80)
    print(f"\n{'Ablation':<25} {'Description':<30} {'AUROC':<10}")
    print("-"*65)

    # Sort by AUROC descending
    results_sorted = sorted(results, key=lambda x: x['test_auroc'], reverse=True)
    baseline_auroc = next(r['test_auroc'] for r in results if r['ablation'] == 'full_model')

    for r in results_sorted:
        delta = r['test_auroc'] - baseline_auroc
        delta_str = f"({delta:+.3f})" if r['ablation'] != 'full_model' else "(baseline)"
        print(f"{r['ablation']:<25} {r['description']:<30} {r['test_auroc']:.4f} {delta_str}")

    # Save results
    output_path = "results/ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

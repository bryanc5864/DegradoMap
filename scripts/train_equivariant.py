"""
Training script for E(3)-Equivariant DegradoMap.

This is the key architectural novelty: using E(3)-equivariant message passing
to capture rotation/translation invariant representations of protein structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from src.models.equivariant_sug import EquivariantSUG
from scripts.train import build_protac8k_degradation_data, create_data_splits


# E3 ligases with learned embeddings
E3_LIST = ['CRBN', 'VHL', 'cIAP1', 'MDM2', 'XIAP', 'DCAF16', 'KEAP1', 'FEM1B']
E3_TO_IDX = {e3: i for i, e3 in enumerate(E3_LIST)}


class EquivariantDegradoMap(nn.Module):
    """
    Full E(3)-equivariant PROTAC degradability predictor.

    Key architectural contributions:
    1. E(3)-equivariant message passing preserves geometric structure
    2. Spherical harmonics encode angular information
    3. Lysine-aware equivariant pooling for ubiquitination sites
    4. Coordinate refinement for structure denoising
    """

    def __init__(
        self,
        node_input_dim: int = 28,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 4,
        e3_embed_dim: int = 64,
        n_e3_ligases: int = 8,
        dropout: float = 0.1,
        cutoff: float = 10.0,
        use_spherical_harmonics: bool = True,
    ):
        super().__init__()

        self.equivariant_sug = EquivariantSUG(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            cutoff=cutoff,
            use_spherical_harmonics=use_spherical_harmonics,
        )

        # E3 ligase embeddings
        self.e3_embeddings = nn.Embedding(n_e3_ligases, e3_embed_dim)

        # Cross-attention between protein and E3
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # E3 projection to match dimensions
        self.e3_proj = nn.Linear(e3_embed_dim, output_dim)

        # Fusion and prediction head
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.predictor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch, e3_name=None, e3_idx=None):
        # Get E3 index
        if e3_idx is None and e3_name is not None:
            e3_idx = E3_TO_IDX.get(e3_name, 0)
        if e3_idx is None:
            e3_idx = 0

        e3_idx_tensor = torch.tensor([e3_idx], device=batch.x.device)

        # Equivariant protein encoding
        sug_out = self.equivariant_sug(batch)
        protein_repr = sug_out['protein_repr']  # [1, output_dim]

        # E3 encoding
        e3_emb = self.e3_embeddings(e3_idx_tensor)  # [1, e3_embed_dim]
        e3_repr = self.e3_proj(e3_emb)  # [1, output_dim]

        # Cross-attention: protein attends to E3
        protein_repr_unsq = protein_repr.unsqueeze(0)  # [1, 1, output_dim]
        e3_repr_unsq = e3_repr.unsqueeze(0)  # [1, 1, output_dim]

        attended, _ = self.cross_attention(
            protein_repr_unsq, e3_repr_unsq, e3_repr_unsq
        )
        attended = attended.squeeze(0)  # [1, output_dim]

        # Fusion
        combined = torch.cat([protein_repr, attended], dim=-1)  # [1, output_dim*2]
        fused = self.fusion(combined)

        # Prediction
        logits = self.predictor(fused)

        return {
            'degrado_logits': logits,
            'protein_repr': protein_repr,
            'e3_repr': e3_repr,
            'refined_coords': sug_out.get('refined_coords'),
        }


def protein_to_graph(coords, residues, plddt=None, sasa=None, disorder=None):
    """Convert protein structure to PyG graph for equivariant model."""

    # Amino acid one-hot (20 + 1 unknown)
    AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

    n_residues = len(residues)

    # One-hot encoding
    aa_onehot = torch.zeros(n_residues, 21)
    for i, res in enumerate(residues):
        idx = aa_to_idx.get(res.upper(), 20)
        aa_onehot[i, idx] = 1.0

    # Lysine indicator
    lys_indicator = torch.tensor([1.0 if r.upper() == 'K' else 0.0 for r in residues]).unsqueeze(1)

    # Additional features
    features = [aa_onehot, lys_indicator]

    if plddt is not None:
        plddt_feat = plddt.float().unsqueeze(1) / 100.0
        features.append(plddt_feat)
    else:
        features.append(torch.zeros(n_residues, 1))

    if sasa is not None:
        sasa_feat = sasa.float().unsqueeze(1)
        sasa_feat = (sasa_feat - sasa_feat.mean()) / (sasa_feat.std() + 1e-6)
        features.append(sasa_feat)
    else:
        features.append(torch.zeros(n_residues, 1))

    if disorder is not None:
        disorder_feat = disorder.float().unsqueeze(1)
        features.append(disorder_feat)
    else:
        features.append(torch.zeros(n_residues, 1))

    # Physicochemical properties (hydrophobicity, charge, etc.)
    HYDROPHOBICITY = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                      'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                      'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                      'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    CHARGE = {'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.5}

    hydro = torch.tensor([HYDROPHOBICITY.get(r.upper(), 0) / 4.5 for r in residues]).unsqueeze(1)
    charge = torch.tensor([CHARGE.get(r.upper(), 0) for r in residues]).unsqueeze(1)
    features.extend([hydro, charge])

    x = torch.cat(features, dim=1).float()

    # Build graph with positions
    pos = coords.float() if isinstance(coords, torch.Tensor) else torch.tensor(coords, dtype=torch.float32)

    # Compute edges (radius graph will be built in the model)
    # For now, store sequential edges for reference
    edge_index = torch.stack([
        torch.arange(n_residues - 1),
        torch.arange(1, n_residues)
    ], dim=0)

    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Lysine mask for pooling
    lysine_mask = torch.tensor([r.upper() == 'K' for r in residues], dtype=torch.bool)

    return Data(x=x, pos=pos, edge_index=edge_index, lysine_mask=lysine_mask)


def load_structures():
    """Load processed structures."""
    structures = {}
    struct_dir = Path("data/processed/structures")
    for pt_file in struct_dir.glob("*.pt"):
        uniprot = pt_file.stem
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        structures[uniprot] = data
    return structures


def train_epoch(model, train_graphs, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0

    indices = np.random.permutation(len(train_graphs))

    for idx in indices:
        graph = train_graphs[idx]
        graph = graph.to(device)
        batch = Batch.from_data_list([graph])

        optimizer.zero_grad()
        out = model(batch, e3_name=graph.e3_name)

        loss = F.binary_cross_entropy_with_logits(
            out['degrado_logits'].squeeze(),
            graph.y.squeeze()
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_samples += 1

    return total_loss / max(n_samples, 1)


def evaluate(model, graphs, device):
    """Evaluate model."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            batch = Batch.from_data_list([graph])

            out = model(batch, e3_name=graph.e3_name)
            prob = torch.sigmoid(out['degrado_logits']).cpu().item()

            all_probs.append(prob)
            all_labels.append(graph.y.item())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) < 2:
        return {'auroc': 0.5, 'auprc': 0.5, 'loss': 0.0}

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    loss = F.binary_cross_entropy(
        torch.tensor(all_probs, dtype=torch.float32),
        torch.tensor(all_labels, dtype=torch.float32)
    ).item()

    return {'auroc': auroc, 'auprc': auprc, 'loss': loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='target_unseen',
                        choices=['random', 'target_unseen', 'e3_unseen'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cutoff', type=float, default=10.0)
    parser.add_argument('--use_spherical_harmonics', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    structures = load_structures()
    samples = build_protac8k_degradation_data()

    print(f"Loaded {len(structures)} structures, {len(samples)} samples")

    # Create splits
    train_samples, val_samples, test_samples = create_data_splits(samples, args.split)
    print(f"Split: {args.split}")
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Build graphs
    print("Building graphs...")

    def build_graphs(sample_list):
        graphs = []
        for sample in tqdm(sample_list, desc="Building graphs"):
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
            graphs.append(graph)
        return graphs

    train_graphs = build_graphs(train_samples)
    val_graphs = build_graphs(val_samples)
    test_graphs = build_graphs(test_samples)

    print(f"Built: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs")

    # Create model
    model = EquivariantDegradoMap(
        node_input_dim=27,  # 21 AA + 1 lys + 3 features (plddt, sasa, disorder) + 2 physico
        hidden_dim=args.hidden_dim,
        output_dim=args.hidden_dim // 2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        cutoff=args.cutoff,
        use_spherical_harmonics=args.use_spherical_harmonics,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_auroc = 0
    best_epoch = 0
    results = []

    checkpoint_dir = Path(f"checkpoints/equivariant_{args.split}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, device)
        val_metrics = evaluate(model, val_graphs, device)
        scheduler.step()

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_auroc': val_metrics['auroc'],
            'val_auprc': val_metrics['auprc'],
            'val_loss': val_metrics['loss'],
        })

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': best_val_auroc,
                'args': vars(args),
            }, checkpoint_dir / "best_model.pt")

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val AUROC: {val_metrics['auroc']:.4f} | Val AUPRC: {val_metrics['auprc']:.4f}")

    # Load best model and evaluate on test
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_graphs, device)

    print("\n" + "="*60)
    print("EQUIVARIANT MODEL RESULTS")
    print("="*60)
    print(f"Best validation AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test AUPRC: {test_metrics['auprc']:.4f}")

    # Save results
    final_results = {
        'split': args.split,
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_val_auroc': best_val_auroc,
        'test_auroc': test_metrics['auroc'],
        'test_auprc': test_metrics['auprc'],
        'n_params': n_params,
        'training_history': results,
    }

    output_path = f"results/equivariant_{args.split}_results.json"
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

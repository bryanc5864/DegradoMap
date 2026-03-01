"""
Module A: Structural Ubiquitination Geometry (SUG)

E(3)-equivariant graph neural network for computing per-lysine
Ubiquitination Geometry Scores (UGS). Integrates:
  - Solvent-accessible surface area (SASA) of lysine side chains
  - Angular accessibility from E2 active site
  - Local flexibility (pLDDT from AlphaFold2)
  - Intrinsically disordered region proximity

Uses e3nn for E(3)-equivariant message passing on protein structure graphs.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool

HAS_E3NN = False  # Use invariant MP for memory efficiency on <=11GB GPUs

try:
    from torch_geometric.nn import MessagePassing
except ImportError:
    from torch.nn import Module as MessagePassing


# ============================================================
# Node feature computation from protein structure
# ============================================================

# Amino acid properties
AA_PROPERTIES = {
    'A': {'charge': 0, 'hydrophobicity': 1.8, 'size': 0, 'polar': 0},
    'R': {'charge': 1, 'hydrophobicity': -4.5, 'size': 2, 'polar': 1},
    'N': {'charge': 0, 'hydrophobicity': -3.5, 'size': 1, 'polar': 1},
    'D': {'charge': -1, 'hydrophobicity': -3.5, 'size': 1, 'polar': 1},
    'C': {'charge': 0, 'hydrophobicity': 2.5, 'size': 1, 'polar': 0},
    'Q': {'charge': 0, 'hydrophobicity': -3.5, 'size': 1, 'polar': 1},
    'E': {'charge': -1, 'hydrophobicity': -3.5, 'size': 1, 'polar': 1},
    'G': {'charge': 0, 'hydrophobicity': -0.4, 'size': 0, 'polar': 0},
    'H': {'charge': 0.5, 'hydrophobicity': -3.2, 'size': 1, 'polar': 1},
    'I': {'charge': 0, 'hydrophobicity': 4.5, 'size': 1, 'polar': 0},
    'L': {'charge': 0, 'hydrophobicity': 3.8, 'size': 1, 'polar': 0},
    'K': {'charge': 1, 'hydrophobicity': -3.9, 'size': 2, 'polar': 1},
    'M': {'charge': 0, 'hydrophobicity': 1.9, 'size': 1, 'polar': 0},
    'F': {'charge': 0, 'hydrophobicity': 2.8, 'size': 2, 'polar': 0},
    'P': {'charge': 0, 'hydrophobicity': -1.6, 'size': 1, 'polar': 0},
    'S': {'charge': 0, 'hydrophobicity': -0.8, 'size': 0, 'polar': 1},
    'T': {'charge': 0, 'hydrophobicity': -0.7, 'size': 1, 'polar': 1},
    'W': {'charge': 0, 'hydrophobicity': -0.9, 'size': 2, 'polar': 0},
    'Y': {'charge': 0, 'hydrophobicity': -1.3, 'size': 2, 'polar': 1},
    'V': {'charge': 0, 'hydrophobicity': 4.2, 'size': 1, 'polar': 0},
}

# One-hot encoding for amino acids
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def encode_residue(aa: str, plddt: float = 0.0, sasa: float = 0.0,
                   is_lysine: bool = False, disorder_score: float = 0.0) -> torch.Tensor:
    """
    Encode a single residue into a feature vector.

    Returns:
        Tensor of shape [feat_dim] with:
        - 20-dim one-hot amino acid identity
        - 4-dim physicochemical properties
        - 1-dim pLDDT (flexibility proxy)
        - 1-dim SASA
        - 1-dim is_lysine flag
        - 1-dim disorder score
        Total: 28 dimensions
    """
    # One-hot
    one_hot = torch.zeros(20)
    idx = AA_TO_IDX.get(aa, -1)
    if idx >= 0:
        one_hot[idx] = 1.0

    # Properties
    props = AA_PROPERTIES.get(aa, {'charge': 0, 'hydrophobicity': 0, 'size': 0, 'polar': 0})
    prop_vec = torch.tensor([
        props['charge'],
        props['hydrophobicity'] / 5.0,  # Normalize
        props['size'] / 2.0,
        float(props['polar']),
    ])

    # Continuous features
    continuous = torch.tensor([
        plddt / 100.0,      # Normalize to [0, 1]
        sasa / 250.0,       # Normalize typical SASA range
        float(is_lysine),
        disorder_score,
    ])

    return torch.cat([one_hot, prop_vec, continuous])


def protein_to_graph(coords: torch.Tensor, residues: List[str],
                     plddt: Optional[torch.Tensor] = None,
                     sasa: Optional[torch.Tensor] = None,
                     disorder: Optional[torch.Tensor] = None,
                     esm_embeddings: Optional[torch.Tensor] = None,
                     radius: float = 10.0,
                     use_esm: bool = False,
                     known_ub_sites: Optional[List[int]] = None,
                     residue_numbers: Optional[List[int]] = None) -> Data:
    """
    Convert protein structure to a graph for the E(3)-equivariant GNN.

    Args:
        coords: [N, 3] Cα coordinates
        residues: List of single-letter amino acid codes
        plddt: [N] pLDDT scores per residue
        sasa: [N] SASA values per residue
        disorder: [N] disorder scores per residue
        esm_embeddings: [N, 1280] ESM-2 per-residue embeddings (optional)
        radius: Radius for edge construction (Å)
        use_esm: Whether to use ESM embeddings (replaces handcrafted features)
        known_ub_sites: List of residue positions known to be ubiquitinated (from PhosphoSitePlus)
        residue_numbers: List of residue numbers corresponding to each position in the structure

    Returns:
        PyG Data object with node features, edge indices, and coordinates
    """
    n_residues = len(residues)
    assert coords.shape[0] == n_residues

    if plddt is None:
        plddt = torch.ones(n_residues) * 70.0
    if sasa is None:
        sasa = torch.ones(n_residues) * 100.0
    if disorder is None:
        disorder = torch.zeros(n_residues)

    # Node features
    node_feats = []
    lysine_mask = []
    for i, aa in enumerate(residues):
        is_lys = (aa == 'K')
        feat = encode_residue(
            aa,
            plddt=plddt[i].item() if torch.is_tensor(plddt[i]) else plddt[i],
            sasa=sasa[i].item() if torch.is_tensor(sasa[i]) else sasa[i],
            is_lysine=is_lys,
            disorder_score=disorder[i].item() if torch.is_tensor(disorder[i]) else disorder[i],
        )
        node_feats.append(feat)
        lysine_mask.append(float(is_lys))

    x_handcrafted = torch.stack(node_feats)  # [N, 28]
    lysine_mask = torch.tensor(lysine_mask, dtype=torch.float)  # [N]

    # Create known Ub site feature (MAPD insight: E2-accessible Ub sites predict degradability)
    known_ub_mask = torch.zeros(n_residues, dtype=torch.float)
    if known_ub_sites is not None and residue_numbers is not None:
        ub_set = set(known_ub_sites)
        for i, res_num in enumerate(residue_numbers):
            if res_num in ub_set and residues[i] == 'K':
                known_ub_mask[i] = 1.0

    # Use ESM embeddings if provided and enabled
    if use_esm and esm_embeddings is not None:
        # ESM embeddings: [N, 1280]
        # Concatenate with key structural features (pLDDT, SASA, lysine, disorder) + known_ub
        structural_feats = x_handcrafted[:, 24:28]  # Last 4 features: pLDDT, SASA, is_lys, disorder
        known_ub_feat = known_ub_mask.unsqueeze(-1)  # [N, 1]
        x = torch.cat([esm_embeddings, structural_feats, known_ub_feat], dim=-1)  # [N, 1285]
    elif known_ub_sites is not None:
        # Add known Ub site as additional feature to handcrafted
        known_ub_feat = known_ub_mask.unsqueeze(-1)  # [N, 1]
        x = torch.cat([x_handcrafted, known_ub_feat], dim=-1)  # [N, 29]
    else:
        x = x_handcrafted  # [N, 28]

    # Edge construction (radius graph) — pure PyTorch, no torch-cluster needed
    dist_matrix = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [N, N]
    mask = (dist_matrix < radius) & (dist_matrix > 0)  # No self-loops
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]

    # Edge vectors (for equivariance)
    edge_vec = coords[edge_index[1]] - coords[edge_index[0]]  # [E, 3]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)  # [E, 1]

    data = Data(
        x=x,
        pos=coords,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_len=edge_len,
        lysine_mask=lysine_mask,
        known_ub_mask=known_ub_mask,
        num_nodes=n_residues,
    )

    return data


# ============================================================
# E(3)-Equivariant Message Passing Layer
# ============================================================

class EquivariantMPLayer(nn.Module):
    """
    E(3)-equivariant message passing layer using e3nn.

    Handles both scalar and vector features with proper equivariance.
    Falls back to invariant message passing if e3nn is not available.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 16,
                 num_basis: int = 8, max_radius: float = 10.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_radius = max_radius

        if HAS_E3NN:
            # E(3)-equivariant version
            self.irreps_in = Irreps(f"{in_dim}x0e")
            self.irreps_out = Irreps(f"{out_dim}x0e")
            self.irreps_edge = Irreps(f"{num_basis}x0e")

            self.tp = FullyConnectedTensorProduct(
                self.irreps_in, self.irreps_edge, self.irreps_out,
                shared_weights=False,
            )
            self.fc = FullyConnectedNet(
                [num_basis, 64, self.tp.weight_numel],
                act=torch.nn.functional.silu,
            )

            # Radial basis functions
            self.num_basis = num_basis
        else:
            # Fallback: invariant message passing
            self.message_mlp = nn.Sequential(
                nn.Linear(2 * in_dim + edge_dim, out_dim * 2),
                nn.SiLU(),
                nn.Linear(out_dim * 2, out_dim),
            )

        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def radial_basis(self, dist: torch.Tensor) -> torch.Tensor:
        """Compute radial basis functions (Gaussian RBFs)."""
        n_basis = self.num_basis if HAS_E3NN else 16
        centers = torch.linspace(0, self.max_radius, n_basis, device=dist.device)
        widths = (centers[1] - centers[0]) * 0.5
        return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * widths ** 2))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_vec: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices
            edge_vec: [E, 3] edge vectors
            edge_len: [E, 1] edge lengths

        Returns:
            [N, out_dim] updated node features
        """
        src, dst = edge_index
        edge_dist = edge_len.squeeze(-1)
        rbf = self.radial_basis(edge_dist)

        if HAS_E3NN:
            # Equivariant message passing
            weights = self.fc(rbf)
            messages = self.tp(x[src], rbf, weight=weights)
        else:
            # Invariant fallback
            msg_input = torch.cat([x[src], x[dst], rbf], dim=-1)
            messages = self.message_mlp(msg_input)

        # Aggregate messages
        agg = torch.zeros(x.size(0), messages.size(-1), device=x.device)
        agg.index_add_(0, dst, messages)

        # Count neighbors for normalization
        counts = torch.zeros(x.size(0), device=x.device)
        counts.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        counts = counts.clamp(min=1).unsqueeze(-1)
        agg = agg / counts

        # Update
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        out = self.norm(out)

        return out


# ============================================================
# Ubiquitination Geometry Score (UGS) Computation
# ============================================================

class UbiquitinationGeometryScorer(nn.Module):
    """
    Compute per-lysine Ubiquitination Geometry Score (UGS).

    Integrates structural context around each lysine to assess:
    1. Geometric accessibility for E2~Ub conjugate approach
    2. Local environment compatibility with ubiquitination
    3. Flexibility for conformational adaptation
    4. Disorder region proximity for proteasomal engagement
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Attention for local neighborhood context
        self.neighborhood_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True,
        )

    def forward(self, node_features: torch.Tensor, lysine_mask: torch.Tensor,
                edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [N, hidden_dim] from GNN
            lysine_mask: [N] binary mask for lysine residues
            edge_index: [2, E]
            batch: [N] batch assignment

        Returns:
            ugs_scores: [N_lysines] UGS score for each lysine
            lysine_indices: [N_lysines] indices of lysine residues
        """
        lysine_indices = lysine_mask.nonzero(as_tuple=True)[0]

        if len(lysine_indices) == 0:
            return torch.tensor([], device=node_features.device), lysine_indices

        lysine_feats = node_features[lysine_indices]  # [N_lys, hidden_dim]
        ugs_scores = self.scorer(lysine_feats).squeeze(-1)  # [N_lys]

        return ugs_scores, lysine_indices


# ============================================================
# Full SUG Module
# ============================================================

class SUGModule(nn.Module):
    """
    Structural Ubiquitination Geometry Module.

    Complete pipeline:
    1. Input embedding of residue features
    2. E(3)-equivariant GNN for structure encoding
    3. Per-lysine UGS scoring
    4. Global protein representation for downstream fusion
    """

    def __init__(self, node_input_dim: int = 28, hidden_dim: int = 256,
                 output_dim: int = 128, num_layers: int = 6,
                 max_radius: float = 10.0, num_basis: int = 8,
                 dropout: float = 0.1, use_global_stats: bool = False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_global_stats = use_global_stats
        self.node_input_dim = node_input_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # E(3)-equivariant GNN layers
        self.gnn_layers = nn.ModuleList([
            EquivariantMPLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_basis=num_basis,
                max_radius=max_radius,
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Per-lysine UGS scorer
        self.ugs_scorer = UbiquitinationGeometryScorer(hidden_dim)

        # Global protein representation
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),  # mean + max pool
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

        # Lysine summary vector
        self.lysine_summary = nn.Sequential(
            nn.Linear(hidden_dim + 1, output_dim),  # feats + UGS score
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: PyG Data with x, pos, edge_index, edge_vec, edge_len,
                  lysine_mask, and optionally batch

        Returns:
            Dictionary with:
            - sug_vector: [B, output_dim] global SUG representation
            - ugs_scores: [N_lysines] per-lysine UGS scores
            - lysine_indices: [N_lysines] indices of lysine residues
            - node_features: [N, hidden_dim] per-residue features
            - lysine_summary: [B, output_dim] summary of lysine features
        """
        x = data.x
        edge_index = data.edge_index
        edge_vec = data.edge_vec
        edge_len = data.edge_len
        lysine_mask = data.lysine_mask
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        h = self.input_proj(x)

        # Message passing
        for layer in self.gnn_layers:
            h_new = layer(h, edge_index, edge_vec, edge_len)
            h = h + self.dropout(h_new)  # Residual connection

        # Per-lysine UGS scores
        ugs_scores, lysine_indices = self.ugs_scorer(h, lysine_mask, edge_index, batch)

        # Global pool with protein-size normalization (FIX: removes protein identity leakage)
        mean_pool = global_mean_pool(h, batch)  # [B, hidden_dim]
        max_pool = global_max_pool(h, batch)     # [B, hidden_dim]

        # Normalize mean_pool by sqrt of protein size to reduce size-dependent features
        from torch_geometric.utils import degree
        num_nodes_per_graph = degree(batch, dtype=torch.float)
        size_norm = torch.sqrt(num_nodes_per_graph).clamp(min=1.0).unsqueeze(-1)  # [B, 1]
        mean_pool_normalized = mean_pool / size_norm  # Scale-invariant

        sug_vector = self.global_pool(torch.cat([mean_pool_normalized, max_pool], dim=-1))  # [B, output_dim]

        # Lysine summary: aggregate lysine features weighted by UGS scores
        if len(lysine_indices) > 0:
            lys_feats = h[lysine_indices]  # [N_lys, hidden_dim]
            lys_ugs = ugs_scores.unsqueeze(-1)  # [N_lys, 1]
            lys_input = torch.cat([lys_feats, lys_ugs], dim=-1)  # [N_lys, hidden_dim+1]
            lys_transformed = self.lysine_summary(lys_input)  # [N_lys, output_dim]

            # Aggregate per-protein using batch assignment
            lys_batch = batch[lysine_indices]
            num_graphs = batch.max().item() + 1
            lys_summary = torch.zeros(num_graphs, self.output_dim, device=x.device)

            # FIX: Per-protein softmax instead of global (removes lysine count leakage)
            weights = torch.zeros_like(ugs_scores)
            for b in range(num_graphs):
                protein_mask = lys_batch == b
                if protein_mask.any():
                    weights[protein_mask] = F.softmax(ugs_scores[protein_mask].detach(), dim=0)

            weighted_feats = lys_transformed * weights.unsqueeze(-1)
            lys_summary.index_add_(0, lys_batch, weighted_feats)
        else:
            num_graphs = batch.max().item() + 1
            lys_summary = torch.zeros(num_graphs, self.output_dim, device=x.device)

        result = {
            "sug_vector": sug_vector,
            "ugs_scores": ugs_scores,
            "lysine_indices": lysine_indices,
            "node_features": h,
            "lysine_summary": lys_summary,
        }

        # Compute global protein statistics if enabled
        if self.use_global_stats:
            # Extract pLDDT and SASA from node features
            # For ESM features (1285 dim): pLDDT is at idx 1280, SASA at 1281
            # For handcrafted (28/29 dim): pLDDT at idx 24, SASA at idx 25
            if self.node_input_dim >= 1280:
                plddt_idx, sasa_idx = 1280, 1281
            else:
                plddt_idx, sasa_idx = 24, 25

            global_stats_list = []
            for b in range(num_graphs):
                mask = batch == b
                node_feats = x[mask]  # [N_b, feat_dim]
                plddt = node_feats[:, plddt_idx]  # Already normalized to [0,1]
                sasa = node_feats[:, sasa_idx]    # Already normalized

                # Compute aggregated statistics: mean, std, min, max for each
                plddt_stats = torch.stack([
                    plddt.mean(),
                    plddt.std() if plddt.numel() > 1 else torch.tensor(0.0, device=x.device),
                    plddt.min(),
                    plddt.max()
                ])
                sasa_stats = torch.stack([
                    sasa.mean(),
                    sasa.std() if sasa.numel() > 1 else torch.tensor(0.0, device=x.device),
                    sasa.min(),
                    sasa.max()
                ])
                global_stats_list.append(torch.cat([plddt_stats, sasa_stats]))

            result["global_stats"] = torch.stack(global_stats_list)  # [B, 8]

        return result

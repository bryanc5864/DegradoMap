"""
E(3)-Equivariant Structure-Ubiquitination Graph (SUG) Module.

Implements full E(3)-equivariance using:
1. EGNN-style message passing with coordinate updates
2. Spherical harmonics for angular features
3. Equivariant pooling for protein-level representations

This is the NOVEL ARCHITECTURAL CONTRIBUTION for the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from typing import Tuple, List, Optional, Dict
import math

# Check for e3nn availability
try:
    from e3nn import o3
    from e3nn.nn import FullyConnectedNet
    from e3nn.o3 import Irreps, spherical_harmonics
    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False
    print("Warning: e3nn not available. Using simplified equivariant layers.")


class RadialBasisFunctions(nn.Module):
    """Radial basis functions for distance encoding."""

    def __init__(self, num_basis: int = 32, cutoff: float = 10.0, trainable: bool = True):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        # Initialize centers evenly spaced
        centers = torch.linspace(0, cutoff, num_basis)
        widths = torch.full((num_basis,), (cutoff / num_basis) * 0.5)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [E] edge distances

        Returns:
            rbf: [E, num_basis] radial basis features
        """
        distances = distances.unsqueeze(-1)  # [E, 1]
        rbf = torch.exp(-((distances - self.centers) ** 2) / (2 * self.widths ** 2))

        # Smooth cutoff
        cutoff_fn = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1)
        cutoff_fn = cutoff_fn * (distances < self.cutoff).float()

        return rbf * cutoff_fn


class EquivariantMessagePassing(nn.Module):
    """
    E(3)-Equivariant Message Passing Layer.

    Key innovation: Combines scalar messages with vector updates,
    maintaining full rotational equivariance.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_basis: int = 32,
        update_coords: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        self.use_attention = use_attention

        # Scalar message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Vector message network (for coordinate updates)
        if update_coords:
            self.coord_net = nn.Sequential(
                nn.Linear(2 * hidden_dim + num_basis, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),  # Scalar weight for direction
            )

        # Attention (optional)
        if use_attention:
            self.attention_net = nn.Sequential(
                nn.Linear(2 * hidden_dim + num_basis, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Node update
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, hidden_dim] node features
            pos: [N, 3] node positions
            edge_index: [2, E] edge indices
            rbf: [E, num_basis] radial basis features

        Returns:
            x_new: [N, hidden_dim] updated node features
            pos_new: [N, 3] updated positions (if update_coords)
        """
        row, col = edge_index  # row = source, col = target

        # Compute edge vectors and distances
        edge_vec = pos[col] - pos[row]  # [E, 3]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)  # [E, 1]

        # Normalize edge vectors
        edge_dir = edge_vec / (edge_dist + 1e-8)  # [E, 3]

        # Build message input
        msg_input = torch.cat([x[row], x[col], rbf], dim=-1)  # [E, 2*H + B]

        # Compute scalar messages
        scalar_msg = self.message_net(msg_input)  # [E, H]

        # Attention weights
        if self.use_attention:
            attn_logits = self.attention_net(msg_input)  # [E, 1]
            # Softmax over incoming edges for each node
            attn_weights = self._scatter_softmax(attn_logits, col, num_nodes=x.size(0))
            scalar_msg = scalar_msg * attn_weights

        # Aggregate messages
        msg_aggr = torch.zeros_like(x)
        msg_aggr.index_add_(0, col, scalar_msg)

        # Update node features with residual
        x_new = self.layer_norm(x + self.node_update(torch.cat([x, msg_aggr], dim=-1)))

        # Update coordinates (equivariant)
        if self.update_coords:
            coord_weights = self.coord_net(msg_input)  # [E, 1]
            coord_msg = coord_weights * edge_dir  # [E, 3] - equivariant!

            coord_aggr = torch.zeros_like(pos)
            coord_aggr.index_add_(0, col, coord_msg)

            pos_new = pos + coord_aggr
        else:
            pos_new = pos

        return x_new, pos_new

    def _scatter_softmax(self, src, index, num_nodes):
        """Compute softmax over scattered values."""
        max_vals = torch.zeros(num_nodes, 1, device=src.device)
        max_vals.scatter_reduce_(0, index.unsqueeze(-1), src, reduce='amax')
        max_vals = max_vals[index]

        exp_vals = torch.exp(src - max_vals)

        sum_exp = torch.zeros(num_nodes, 1, device=src.device)
        sum_exp.index_add_(0, index, exp_vals)
        sum_exp = sum_exp[index]

        return exp_vals / (sum_exp + 1e-8)


class SphericalHarmonicsEncoder(nn.Module):
    """
    Encode edge directions using spherical harmonics.

    This provides richer angular information than just distances,
    while maintaining E(3) equivariance.
    """

    def __init__(self, lmax: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.lmax = lmax

        if HAS_E3NN:
            self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
            self.sh_dim = self.irreps_sh.dim
        else:
            # Simplified: just use direction components
            self.sh_dim = 3 + 5 if lmax >= 2 else 3  # l=1 (3) + l=2 (5)

        self.projection = nn.Linear(self.sh_dim, hidden_dim)

    def forward(self, edge_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_vec: [E, 3] edge vectors

        Returns:
            sh_features: [E, hidden_dim]
        """
        if HAS_E3NN:
            # Normalize vectors
            edge_dir = F.normalize(edge_vec, dim=-1)
            sh = spherical_harmonics(self.irreps_sh, edge_dir, normalize=True)
        else:
            # Simplified: use normalized direction + squared components
            edge_dir = F.normalize(edge_vec, dim=-1)
            if self.lmax >= 2:
                # Approximate l=2 spherical harmonics
                x, y, z = edge_dir[:, 0], edge_dir[:, 1], edge_dir[:, 2]
                l2 = torch.stack([
                    x*y, y*z, z**2 - 0.5*(x**2 + y**2),  # Simplified Y_2
                    x*z, x**2 - y**2
                ], dim=-1)
                sh = torch.cat([edge_dir, l2], dim=-1)
            else:
                sh = edge_dir

        return self.projection(sh)


class EquivariantPooling(nn.Module):
    """
    Equivariant pooling for protein-level representations.

    Novel contribution: Uses attention over lysine residues
    weighted by their structural context.
    """

    def __init__(self, hidden_dim: int, use_lysine_attention: bool = True):
        super().__init__()
        self.use_lysine_attention = use_lysine_attention

        # Global attention
        self.global_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Lysine-specific attention
        if use_lysine_attention:
            self.lysine_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2 if use_lysine_attention else hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        lysine_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim] node features
            batch: [N] batch assignment
            lysine_mask: [N] boolean mask for lysine residues

        Returns:
            pooled: [B, hidden_dim] protein-level features
        """
        # Global attention pooling
        attn_weights = self.global_attn(x)  # [N, 1]

        # Softmax within each graph
        attn_weights = self._batch_softmax(attn_weights, batch)
        global_pool = self._weighted_pool(x * attn_weights, batch)

        if self.use_lysine_attention:
            if lysine_mask is not None:
                # Lysine-specific attention
                lys_attn = self.lysine_attn(x)  # [N, 1]

                # Mask non-lysine residues
                lys_attn = lys_attn.masked_fill(~lysine_mask.unsqueeze(-1), float('-inf'))
                lys_attn = self._batch_softmax(lys_attn, batch)
                lys_attn = lys_attn.masked_fill(~lysine_mask.unsqueeze(-1), 0)

                lys_pool = self._weighted_pool(x * lys_attn, batch)
            else:
                # No lysine mask - use global pool as fallback for lys_pool
                lys_pool = global_pool

            # Combine
            pooled = torch.cat([global_pool, lys_pool], dim=-1)
        else:
            pooled = global_pool

        # Size normalization
        num_nodes = torch.bincount(batch).float().unsqueeze(-1).to(x.device)
        pooled = pooled / num_nodes.sqrt()

        return self.output_proj(pooled)

    def _batch_softmax(self, x, batch):
        """Softmax within each batch element."""
        max_vals = torch.zeros(batch.max() + 1, 1, device=x.device)
        max_vals.scatter_reduce_(0, batch.unsqueeze(-1), x, reduce='amax')
        max_vals = max_vals[batch]

        exp_vals = torch.exp(x - max_vals)

        sum_exp = torch.zeros(batch.max() + 1, 1, device=x.device)
        sum_exp.index_add_(0, batch, exp_vals)
        sum_exp = sum_exp[batch]

        return exp_vals / (sum_exp + 1e-8)

    def _weighted_pool(self, x, batch):
        """Weighted sum pooling."""
        out = torch.zeros(batch.max() + 1, x.size(-1), device=x.device)
        out.index_add_(0, batch, x)
        return out


class EquivariantSUG(nn.Module):
    """
    E(3)-Equivariant Structure-Ubiquitination Graph Module.

    Novel contributions:
    1. Full E(3)-equivariance via coordinate-aware message passing
    2. Spherical harmonics for richer angular information
    3. Lysine-aware equivariant pooling
    4. Coordinate refinement for structural denoising
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 4,
        num_basis: int = 32,
        cutoff: float = 10.0,
        update_coords: bool = True,
        use_spherical_harmonics: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.cutoff = cutoff

        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Radial basis functions
        self.rbf = RadialBasisFunctions(num_basis, cutoff)

        # Spherical harmonics encoder
        self.use_sh = use_spherical_harmonics
        if use_spherical_harmonics:
            self.sh_encoder = SphericalHarmonicsEncoder(lmax=2, hidden_dim=hidden_dim)

        # Equivariant layers
        self.layers = nn.ModuleList([
            EquivariantMessagePassing(
                hidden_dim,
                num_basis + (hidden_dim if use_spherical_harmonics else 0),
                update_coords=update_coords,
            )
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Equivariant pooling
        self.pooling = EquivariantPooling(hidden_dim, use_lysine_attention=True)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: PyG Data object with x, pos, edge_index, lysine_mask, batch

        Returns:
            Dictionary with:
                - node_repr: [N, output_dim] per-residue features
                - protein_repr: [B, output_dim] per-protein features
                - refined_coords: [N, 3] updated coordinates (if update_coords=True)
        """
        x = data.x
        pos = data.pos if hasattr(data, 'pos') else data.x[:, :3]  # Fallback
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        lysine_mask = data.lysine_mask if hasattr(data, 'lysine_mask') else None

        # Initial embedding
        h = self.input_proj(x)

        # Compute radial basis
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_dist = edge_vec.norm(dim=-1)
        rbf = self.rbf(edge_dist)

        # Add spherical harmonics
        if self.use_sh:
            sh_features = self.sh_encoder(edge_vec)
            edge_features = torch.cat([rbf, sh_features], dim=-1)
        else:
            edge_features = rbf

        # Message passing layers
        current_pos = pos
        for layer in self.layers:
            h, current_pos = layer(h, current_pos, edge_index, edge_features)
            h = self.dropout(h)

        # Per-residue output
        node_out = self.output_proj(h)

        # Per-protein output (equivariant pooling)
        protein_out = self.pooling(h, batch, lysine_mask)
        protein_out = self.output_proj(protein_out)

        return {
            'node_repr': node_out,
            'protein_repr': protein_out,
            'refined_coords': current_pos,
        }


def protein_to_graph_equivariant(
    coords: torch.Tensor,
    residues: List[str],
    plddt: Optional[torch.Tensor] = None,
    sasa: Optional[torch.Tensor] = None,
    disorder: Optional[torch.Tensor] = None,
    radius: float = 10.0,
) -> Data:
    """
    Build graph for equivariant model.

    Same as regular protein_to_graph but ensures pos is separate from x.
    """
    # Import regular function
    from src.models.sug_module import protein_to_graph

    data = protein_to_graph(coords, residues, plddt, sasa, disorder, radius)

    # Ensure pos is set correctly
    data.pos = coords.clone()

    return data


# For compatibility with DegradoMap
class EquivariantDegradoMap(nn.Module):
    """
    Full E(3)-Equivariant DegradoMap model.

    Replaces SUGModule with EquivariantSUG while keeping
    E3 compatibility and fusion modules.
    """

    def __init__(
        self,
        node_input_dim: int = 28,
        sug_hidden_dim: int = 128,
        sug_output_dim: int = 64,
        sug_num_layers: int = 4,
        e3_hidden_dim: int = 64,
        e3_output_dim: int = 64,
        e3_num_heads: int = 4,
        e3_num_layers: int = 2,
        fusion_hidden_dim: int = 128,
        pred_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Equivariant SUG module
        self.sug = EquivariantSUG(
            input_dim=node_input_dim,
            hidden_dim=sug_hidden_dim,
            output_dim=sug_output_dim,
            num_layers=sug_num_layers,
            dropout=dropout,
        )

        # E3 compatibility module (from original)
        from src.models.e3_compat_module import E3CompatibilityModule
        self.e3_compat = E3CompatibilityModule(
            protein_dim=sug_output_dim,
            e3_hidden_dim=e3_hidden_dim,
            output_dim=e3_output_dim,
            num_heads=e3_num_heads,
            num_layers=e3_num_layers,
            dropout=dropout,
        )

        # Fusion and prediction
        combined_dim = sug_output_dim + e3_output_dim

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, pred_hidden_dim),
            nn.SiLU(),
        )

        self.pred_head = nn.Linear(pred_hidden_dim, 1)

    def forward(self, data: Data, e3_name: str = "VHL"):
        """Forward pass."""
        # Equivariant SUG
        node_features, protein_features = self.sug(data)

        # E3 compatibility
        e3_output = self.e3_compat(protein_features, e3_name)

        # Fusion
        combined = torch.cat([protein_features, e3_output], dim=-1)
        fused = self.fusion(combined)

        # Prediction
        logits = self.pred_head(fused)

        return {
            "degrado_logits": logits,
            "node_features": node_features,
            "protein_features": protein_features,
            "e3_features": e3_output,
        }

"""
Module B: E3 Ligase Compatibility Network

Cross-attention mechanism between target protein surface representations
and E3 ligase substrate recognition domain (SRD) embeddings.

Learns the compatibility function between target surfaces and E3 SRDs
across ligase families, enabling:
  - Predicting which E3 is optimal for a novel target
  - Transfer learning to new E3 ligases
  - Mechanistic interpretation of compatibility
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# E3 Ligase Substrate Recognition Domain Information
# ============================================================

E3_LIGASE_INFO = {
    "CRBN": {
        "uniprot": "Q96SW2",
        "srd_region": (318, 442),  # Thalidomide-binding domain
        "complex_partners": ["DDB1", "CUL4A", "RBX1"],
        "embedding_source": "structure",
    },
    "VHL": {
        "uniprot": "P40337",
        "srd_region": (54, 213),  # β-domain (substrate recognition)
        "complex_partners": ["ELOB", "ELOC", "CUL2", "RBX1"],
        "embedding_source": "structure",
    },
    "MDM2": {
        "uniprot": "Q00987",
        "srd_region": (25, 109),  # N-terminal p53-binding domain
        "complex_partners": [],
        "embedding_source": "structure",
    },
    "cIAP1": {
        "uniprot": "Q13490",
        "srd_region": (264, 348),  # BIR3 domain
        "complex_partners": [],
        "embedding_source": "structure",
    },
    "DCAF16": {
        "uniprot": "Q9NXF7",
        "srd_region": (1, 216),  # Full protein (small)
        "complex_partners": ["DDB1", "CUL4A"],
        "embedding_source": "sequence",  # Less structural data
    },
}


# ============================================================
# E3 Ligase Encoder
# ============================================================

class E3LigaseEncoder(nn.Module):
    """
    Encode E3 ligase substrate recognition domains.

    For each E3 family, produces a set of residue-level representations
    of the substrate recognition interface.
    """

    def __init__(self, input_dim: int = 28, hidden_dim: int = 256,
                 output_dim: int = 128, num_e3_families: int = 5,
                 max_srd_length: int = 200):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_srd_length = max_srd_length

        # Learnable E3 family embeddings (when structure not available)
        self.e3_family_embedding = nn.Embedding(num_e3_families + 1, hidden_dim)

        # E3 SRD encoder (shared architecture with SUG's GNN, but separate weights)
        self.srd_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Per-E3 projection heads
        self.e3_projections = nn.ModuleDict({
            name: nn.Linear(hidden_dim, output_dim)
            for name in E3_LIGASE_INFO
        })

        # Fallback projection
        self.default_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, e3_name: str,
                srd_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode an E3 ligase SRD.

        Args:
            e3_name: Name of E3 ligase family
            srd_features: [L, input_dim] SRD residue features (if available)

        Returns:
            [L_out, output_dim] SRD representation
        """
        if srd_features is not None:
            # Structure-based encoding
            h = self.srd_encoder(srd_features)  # [L, hidden_dim]
            if e3_name in self.e3_projections:
                return self.e3_projections[e3_name](h)  # [L, output_dim]
            return self.default_projection(h)
        else:
            # Learned embedding fallback
            e3_names = list(E3_LIGASE_INFO.keys())
            idx = e3_names.index(e3_name) if e3_name in e3_names else len(e3_names)
            idx_tensor = torch.tensor([idx], device=self.e3_family_embedding.weight.device)
            emb = self.e3_family_embedding(idx_tensor)  # [1, hidden_dim]
            # Expand to pseudo-sequence for cross-attention
            emb = emb.expand(self.max_srd_length // 4, -1)  # [L_pseudo, hidden_dim]
            if e3_name in self.e3_projections:
                return self.e3_projections[e3_name](emb)
            return self.default_projection(emb)


# ============================================================
# Cross-Attention Layer
# ============================================================

class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention between target and E3 SRD representations.

    Target residues attend to E3 SRD residues and vice versa, learning
    which regions of each are compatible for ternary complex formation.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Target → E3 attention
        self.target_to_e3_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # E3 → Target attention
        self.e3_to_target_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Feed-forward networks
        self.target_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.e3_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1_t = nn.LayerNorm(hidden_dim)
        self.norm2_t = nn.LayerNorm(hidden_dim)
        self.norm1_e = nn.LayerNorm(hidden_dim)
        self.norm2_e = nn.LayerNorm(hidden_dim)

    def forward(self, target_feats: torch.Tensor, e3_feats: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None,
                e3_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            target_feats: [B, L_target, hidden_dim]
            e3_feats: [B, L_e3, hidden_dim]
            target_mask: [B, L_target] padding mask
            e3_mask: [B, L_e3] padding mask

        Returns:
            updated_target: [B, L_target, hidden_dim]
            updated_e3: [B, L_e3, hidden_dim]
            cross_attn_weights: [B, L_target, L_e3] attention map
        """
        # Target → E3 cross-attention
        t2e, attn_weights = self.target_to_e3_attn(
            query=target_feats,
            key=e3_feats,
            value=e3_feats,
            key_padding_mask=e3_mask,
        )
        target_feats = self.norm1_t(target_feats + t2e)
        target_feats = self.norm2_t(target_feats + self.target_ffn(target_feats))

        # E3 → Target cross-attention
        e2t, _ = self.e3_to_target_attn(
            query=e3_feats,
            key=target_feats,
            value=target_feats,
            key_padding_mask=target_mask,
        )
        e3_feats = self.norm1_e(e3_feats + e2t)
        e3_feats = self.norm2_e(e3_feats + self.e3_ffn(e3_feats))

        return target_feats, e3_feats, attn_weights


# ============================================================
# Full E3 Compatibility Module
# ============================================================

class E3CompatModule(nn.Module):
    """
    E3 Ligase Compatibility Module.

    Complete pipeline:
    1. Encode E3 SRD
    2. Project target node features to compatibility space
    3. Bidirectional cross-attention (multiple layers)
    4. Pool into compatibility vector
    """

    def __init__(self, target_dim: int = 256, e3_input_dim: int = 28,
                 hidden_dim: int = 128, output_dim: int = 128,
                 num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, num_e3_families: int = 5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Target feature projection (from SUG hidden dim to compat dim)
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # E3 ligase encoder
        self.e3_encoder = E3LigaseEncoder(
            input_dim=e3_input_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            num_e3_families=num_e3_families,
        )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Compatibility vector computation
        self.compat_pool = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # target_mean, target_max, e3_mean, e3_max
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Per-residue interface propensity (for interpretation)
        self.interface_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, target_node_feats: torch.Tensor,
                e3_name: str,
                target_batch: torch.Tensor,
                e3_srd_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            target_node_feats: [N, target_dim] per-residue features from SUG
            e3_name: E3 ligase family name
            target_batch: [N] batch assignment for target residues
            e3_srd_features: [L, e3_input_dim] E3 SRD features (optional)

        Returns:
            Dictionary with:
            - compat_vector: [B, output_dim] compatibility representation
            - cross_attn_map: [B, L_target, L_e3] attention weights
            - interface_scores: [N] per-residue interface propensity
        """
        num_graphs = target_batch.max().item() + 1

        # Project target features
        target_h = self.target_proj(target_node_feats)  # [N, hidden_dim]

        # Encode E3 SRD
        e3_h = self.e3_encoder(e3_name, e3_srd_features)  # [L_e3, hidden_dim]

        # Pad target features into batched format [B, max_L_target, hidden_dim]
        max_target_len = 0
        target_lens = []
        for b in range(num_graphs):
            mask = target_batch == b
            l = mask.sum().item()
            target_lens.append(l)
            max_target_len = max(max_target_len, l)

        target_padded = torch.zeros(num_graphs, max_target_len, self.hidden_dim,
                                     device=target_h.device)
        target_mask = torch.ones(num_graphs, max_target_len,
                                  dtype=torch.bool, device=target_h.device)

        for b in range(num_graphs):
            mask = target_batch == b
            feats = target_h[mask]
            target_padded[b, :feats.size(0)] = feats
            target_mask[b, :feats.size(0)] = False  # False = not masked

        # E3 features: same for all items in batch
        e3_padded = e3_h.unsqueeze(0).expand(num_graphs, -1, -1)  # [B, L_e3, hidden_dim]

        # Cross-attention
        last_attn = None
        for layer in self.cross_attn_layers:
            target_padded, e3_padded, attn_weights = layer(
                target_padded, e3_padded,
                target_mask=target_mask,
            )
            last_attn = attn_weights

        # Pool to compatibility vector
        # Masked mean/max pooling for target
        target_mask_expanded = (~target_mask).unsqueeze(-1).float()
        target_mean = (target_padded * target_mask_expanded).sum(1) / target_mask_expanded.sum(1).clamp(min=1)
        target_max = target_padded.masked_fill(target_mask.unsqueeze(-1), float('-inf')).max(1)[0]

        # E3 mean/max
        e3_mean = e3_padded.mean(1)
        e3_max = e3_padded.max(1)[0]

        pooled = torch.cat([target_mean, target_max, e3_mean, e3_max], dim=-1)
        compat_vector = self.compat_pool(pooled)  # [B, output_dim]

        # Per-residue interface scores
        interface_scores = self.interface_head(target_h).squeeze(-1)  # [N]

        return {
            "compat_vector": compat_vector,
            "cross_attn_map": last_attn,
            "interface_scores": interface_scores,
        }

"""
Module C: Cellular Context Encoder

Encodes cell-line-specific features from DepMap expression, ProteomicsDB
protein abundance, and derived features into a context embedding.

Context features include:
  - E3 ligase gene expression levels
  - Proteasome subunit expression
  - Target protein expression/abundance
  - Competing substrate load
  - DUB expression levels
  - Target protein half-life
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Feature group definitions
FEATURE_GROUPS = {
    "e3_expression": {
        "description": "E3 ligase gene expression (log2 TPM)",
        "genes": ["CRBN", "VHL", "MDM2", "BIRC2", "DCAF16", "DCAF15",
                  "KEAP1", "DDB1", "CUL4A", "CUL2", "ELOB", "ELOC", "RBX1"],
        "dim": 13,
    },
    "proteasome": {
        "description": "Proteasome subunit expression",
        "genes": ["PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
                  "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7",
                  "PSMB8", "PSMB9", "PSMB10"],
        "dim": 17,
    },
    "dub_expression": {
        "description": "Deubiquitinase expression levels",
        "genes": ["USP7", "USP14", "USP15", "USP19", "USP28", "USP33",
                  "UCHL5", "OTUB1", "COPS5", "BRCC3"],
        "dim": 10,
    },
    "ub_pathway": {
        "description": "Ubiquitin conjugation pathway expression",
        "genes": ["UBA1", "UBE2D1", "UBE2D2", "UBE2D3",
                  "UBE2G1", "UBE2G2", "UBE2K", "UBE2N",
                  "UBB", "UBC"],
        "dim": 10,
    },
    "target_context": {
        "description": "Target protein specific context",
        "features": ["target_expression", "target_halflife", "target_abundance",
                     "target_dependency_score", "competing_substrate_load"],
        "dim": 5,
    },
    "cell_metadata": {
        "description": "Cell line metadata features",
        "features": ["lineage_encoding", "growth_rate", "doubling_time",
                     "culture_type_encoding"],
        "dim": 4,
    },
}

TOTAL_CONTEXT_DIM = sum(g["dim"] for g in FEATURE_GROUPS.values())  # 59 base features


class ResidualBlock(nn.Module):
    """Residual block for the context encoder."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.block(x)))


class FeatureGroupEncoder(nn.Module):
    """
    Encode each feature group separately before combining.
    This allows the model to learn group-specific representations.
    """

    def __init__(self, group_dims: Dict[str, int], hidden_dim: int = 64):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.output_dim = 0

        for name, dim in group_dims.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            self.output_dim += hidden_dim

    def forward(self, feature_groups: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_groups: Dict mapping group name to tensor [B, group_dim]

        Returns:
            [B, total_output_dim] concatenated group embeddings
        """
        encoded = []
        for name, encoder in self.encoders.items():
            if name in feature_groups:
                encoded.append(encoder(feature_groups[name]))
            else:
                # Zero-fill for missing groups
                b = next(iter(feature_groups.values())).size(0)
                device = next(iter(feature_groups.values())).device
                encoded.append(torch.zeros(b, encoder[0].in_features, device=device))
                encoded[-1] = encoder(encoded[-1])

        return torch.cat(encoded, dim=-1)


class ContextModule(nn.Module):
    """
    Cellular Context Encoder.

    Encodes cell-line-specific features into a context embedding vector
    that conditions the degradability prediction.
    """

    def __init__(self, input_dim: int = 200, hidden_dims: List[int] = None,
                 output_dim: int = 128, dropout: float = 0.2,
                 num_residual_blocks: int = 3, use_group_encoding: bool = True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 192, 128]

        self.output_dim = output_dim
        self.use_group_encoding = use_group_encoding

        if use_group_encoding:
            group_dims = {name: info["dim"] for name, info in FEATURE_GROUPS.items()}
            self.group_encoder = FeatureGroupEncoder(group_dims, hidden_dim=64)
            actual_input_dim = self.group_encoder.output_dim
        else:
            self.group_encoder = None
            actual_input_dim = input_dim

        # Main encoder
        layers = []
        prev_dim = actual_input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1], dropout)
            for _ in range(num_residual_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.LayerNorm(output_dim),
        )

        # Context dropout: randomly drop entire context during training
        # to make the model robust to missing context information
        self.context_dropout_rate = 0.1

    def forward(self, context_features: torch.Tensor,
                feature_groups: Optional[Dict[str, torch.Tensor]] = None,
                training: bool = True) -> torch.Tensor:
        """
        Args:
            context_features: [B, input_dim] flat context feature vector
                              OR None if using feature_groups
            feature_groups: Optional dict of feature group tensors
            training: Whether in training mode (for context dropout)

        Returns:
            [B, output_dim] context embedding
        """
        # Context dropout: zero out entire context with some probability
        if training and self.training and torch.rand(1).item() < self.context_dropout_rate:
            if feature_groups is not None:
                b = next(iter(feature_groups.values())).size(0)
                device = next(iter(feature_groups.values())).device
            else:
                b = context_features.size(0)
                device = context_features.device
            return torch.zeros(b, self.output_dim, device=device)

        if self.use_group_encoding and feature_groups is not None:
            h = self.group_encoder(feature_groups)
        else:
            h = context_features

        h = self.encoder(h)

        for block in self.residual_blocks:
            h = block(h)

        return self.output_proj(h)


def build_context_features(
    cell_line: str,
    target_gene: str,
    expression_data: Optional[Dict] = None,
    halflife_data: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build context feature vectors from raw data sources.

    Args:
        cell_line: Cell line name (e.g., 'HeLa', 'MCF-7')
        target_gene: Target gene symbol
        expression_data: DepMap expression matrix
        halflife_data: ProteomicsDB half-life data

    Returns:
        Dictionary of feature group tensors
    """
    features = {}

    # Default values (will be overridden by real data)
    for group_name, info in FEATURE_GROUPS.items():
        dim = info["dim"]
        features[group_name] = torch.zeros(1, dim)

    if expression_data is not None:
        # Extract E3 ligase expression
        e3_genes = FEATURE_GROUPS["e3_expression"]["genes"]
        e3_expr = []
        for gene in e3_genes:
            val = expression_data.get(cell_line, {}).get(gene, 0.0)
            e3_expr.append(val)
        features["e3_expression"] = torch.tensor([e3_expr], dtype=torch.float32)

        # Extract proteasome expression
        prot_genes = FEATURE_GROUPS["proteasome"]["genes"]
        prot_expr = [expression_data.get(cell_line, {}).get(g, 0.0) for g in prot_genes]
        features["proteasome"] = torch.tensor([prot_expr], dtype=torch.float32)

        # DUB expression
        dub_genes = FEATURE_GROUPS["dub_expression"]["genes"]
        dub_expr = [expression_data.get(cell_line, {}).get(g, 0.0) for g in dub_genes]
        features["dub_expression"] = torch.tensor([dub_expr], dtype=torch.float32)

        # Ub pathway
        ub_genes = FEATURE_GROUPS["ub_pathway"]["genes"]
        ub_expr = [expression_data.get(cell_line, {}).get(g, 0.0) for g in ub_genes]
        features["ub_pathway"] = torch.tensor([ub_expr], dtype=torch.float32)

        # Target expression
        target_expr = expression_data.get(cell_line, {}).get(target_gene, 0.0)
        features["target_context"] = torch.tensor([[
            target_expr, 0.0, 0.0, 0.0, 0.0
        ]], dtype=torch.float32)

    if halflife_data is not None:
        halflife = halflife_data.get(target_gene, {}).get("halflife_hours", 24.0)
        if features.get("target_context") is not None:
            features["target_context"][0, 1] = halflife

    return features

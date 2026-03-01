"""
Module D: Multi-Scale Fusion and Prediction Head

Fuses representations from:
  - Module A (SUG): Structural ubiquitination geometry
  - Module B (E3 Compat): Target-E3 compatibility
  - Module C (Context): Cellular context

Produces:
  - DegradoScore: probability of successful degradation
  - DC50 prediction (auxiliary)
  - Dmax prediction (auxiliary)
  - Per-lysine importance scores
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining multi-modal representations.

    z = σ(W_g [x1; x2; x3]) ⊙ tanh(W_h [x1; x2; x3])

    The gate learns which information sources are most predictive
    for different target-E3-context combinations.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *inputs: Variable number of input tensors, each [B, dim_i]

        Returns:
            [B, output_dim] fused representation
        """
        concatenated = torch.cat(inputs, dim=-1)
        gate_values = self.gate(concatenated)
        transformed = self.transform(concatenated)
        fused = gate_values * transformed
        return self.norm(fused)


class DegradoScoreHead(nn.Module):
    """Binary degradability classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (apply sigmoid for probability)."""
        return self.classifier(x).squeeze(-1)


class DC50RegressionHead(nn.Module):
    """Auxiliary DC50 prediction (log-scale)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted log10(DC50) in nM."""
        return self.regressor(x).squeeze(-1)


class DmaxRegressionHead(nn.Module):
    """Auxiliary Dmax prediction (percentage)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1], multiply by 100 for percentage
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted Dmax as fraction [0, 1]."""
        return self.regressor(x).squeeze(-1)


class FusionModule(nn.Module):
    """
    Complete fusion and prediction module.

    Combines SUG, E3 compatibility, and context vectors through gated fusion,
    then produces predictions via multiple task-specific heads.
    """

    def __init__(self, sug_dim: int = 128, compat_dim: int = 128,
                 context_dim: int = 128, fusion_hidden_dim: int = 256,
                 pred_hidden_dim: int = 128, dropout: float = 0.1,
                 e3_onehot_dim: int = 0, lysine_summary_dim: int = None):
        super().__init__()

        self.e3_onehot_dim = e3_onehot_dim
        total_input_dim = sug_dim + compat_dim + context_dim + e3_onehot_dim

        # lysine_summary_dim is the original SUG output_dim (without global stats)
        # if not provided, assume same as sug_dim
        if lysine_summary_dim is None:
            lysine_summary_dim = sug_dim

        # Gated fusion
        self.fusion = GatedFusion(
            input_dim=total_input_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=pred_hidden_dim,
        )

        # Also incorporate lysine summary (uses original SUG output_dim)
        self.lysine_gate = nn.Sequential(
            nn.Linear(lysine_summary_dim + pred_hidden_dim, pred_hidden_dim),
            nn.Sigmoid(),
        )

        # Prediction heads
        self.degrade_head = DegradoScoreHead(pred_hidden_dim, pred_hidden_dim, dropout)
        self.dc50_head = DC50RegressionHead(pred_hidden_dim, pred_hidden_dim // 2)
        self.dmax_head = DmaxRegressionHead(pred_hidden_dim, pred_hidden_dim // 2)

    def forward(self, sug_vector: torch.Tensor, compat_vector: torch.Tensor,
                context_vector: torch.Tensor,
                lysine_summary: Optional[torch.Tensor] = None,
                e3_onehot: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            sug_vector: [B, sug_dim] from Module A
            compat_vector: [B, compat_dim] from Module B
            context_vector: [B, context_dim] from Module C
            lysine_summary: [B, sug_dim] from Module A's lysine aggregation
            e3_onehot: [B, 6] E3 ligase one-hot encoding (optional)

        Returns:
            Dictionary with:
            - degrado_logits: [B] logits for degradability
            - degrado_score: [B] probability of degradability
            - dc50_pred: [B] predicted log10(DC50)
            - dmax_pred: [B] predicted Dmax fraction
        """
        # Gated fusion of modalities (including E3 one-hot if provided)
        if e3_onehot is not None and self.e3_onehot_dim > 0:
            fused = self.fusion(sug_vector, compat_vector, context_vector, e3_onehot)
        else:
            fused = self.fusion(sug_vector, compat_vector, context_vector)

        # Optionally incorporate lysine summary
        if lysine_summary is not None:
            gate = self.lysine_gate(torch.cat([lysine_summary, fused], dim=-1))
            fused = fused * gate + fused  # Residual gating

        # Predictions
        degrado_logits = self.degrade_head(fused)
        degrado_score = torch.sigmoid(degrado_logits)
        dc50_pred = self.dc50_head(fused)
        dmax_pred = self.dmax_head(fused)

        return {
            "degrado_logits": degrado_logits,
            "degrado_score": degrado_score,
            "dc50_pred": dc50_pred,
            "dmax_pred": dmax_pred,
            "fused_representation": fused,
        }

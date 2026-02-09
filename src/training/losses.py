"""
Loss functions for DegradoMap training.

Multi-task loss combining:
  - Degradation classification (BCE with label smoothing)
  - DC50 regression (Huber loss)
  - Dmax regression (Huber loss)
  - E3-Substrate Interaction consistency (KL divergence)
  - Ubiquitination site prediction (Focal loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in Ub site prediction.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Reduces the loss contribution from easy examples and focuses
    training on hard negatives.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N] raw logits
            targets: [N] binary labels

        Returns:
            Scalar focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class LabelSmoothingBCE(nn.Module):
    """Binary cross-entropy with label smoothing and optional class weighting."""

    def __init__(self, smoothing: float = 0.05, pos_weight: float = None):
        super().__init__()
        self.smoothing = smoothing
        # pos_weight: weight for positive class (typically neg_count / pos_count)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: [B] raw logits
            targets: [B] binary labels
            weights: [B] sample weights (optional)

        Returns:
            Scalar loss
        """
        # Smooth labels
        smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        # Apply class weighting via pos_weight parameter
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)

        loss = F.binary_cross_entropy_with_logits(
            logits, smoothed, reduction='none', pos_weight=pos_weight_tensor
        )

        if weights is not None:
            loss = loss * weights

        return loss.mean()


class DegradoMapLoss(nn.Module):
    """
    Combined multi-task loss for DegradoMap.

    L = λ₁·L_degrad + λ₂·L_DC50 + λ₃·L_Dmax + λ₄·L_ESI + λ₅·L_UbSite
    """

    def __init__(self, lambda_degrad: float = 1.0, lambda_dc50: float = 0.3,
                 lambda_dmax: float = 0.3, lambda_esi: float = 0.2,
                 lambda_ubsite: float = 0.2, label_smoothing: float = 0.05,
                 pos_weight: float = None):
        super().__init__()

        self.lambda_degrad = lambda_degrad
        self.lambda_dc50 = lambda_dc50
        self.lambda_dmax = lambda_dmax
        self.lambda_esi = lambda_esi
        self.lambda_ubsite = lambda_ubsite

        # pos_weight for class balancing: neg_count / pos_count
        self.degrad_loss = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pos_weight)
        self.dc50_loss = nn.HuberLoss(delta=1.0)
        self.dmax_loss = nn.HuberLoss(delta=0.1)
        self.esi_loss = nn.BCEWithLogitsLoss()
        self.ubsite_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Compute combined loss.

        Args:
            predictions: Dict with model outputs
            targets: Dict with ground truth labels

        Returns:
            Dict with individual and total losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions["degrado_logits"].device)

        # Degradation classification
        if "degrad_label" in targets:
            l_degrad = self.degrad_loss(
                predictions["degrado_logits"],
                targets["degrad_label"],
                weights=targets.get("sample_weight"),
            )
            losses["degrad"] = l_degrad
            total_loss = total_loss + self.lambda_degrad * l_degrad

        # DC50 regression (only for samples with quantitative labels)
        if "dc50_label" in targets:
            mask = targets.get("dc50_mask", torch.ones_like(targets["dc50_label"], dtype=torch.bool))
            if mask.any():
                l_dc50 = self.dc50_loss(
                    predictions["dc50_pred"][mask],
                    targets["dc50_label"][mask],
                )
                losses["dc50"] = l_dc50
                total_loss = total_loss + self.lambda_dc50 * l_dc50

        # Dmax regression
        if "dmax_label" in targets:
            mask = targets.get("dmax_mask", torch.ones_like(targets["dmax_label"], dtype=torch.bool))
            if mask.any():
                l_dmax = self.dmax_loss(
                    predictions["dmax_pred"][mask],
                    targets["dmax_label"][mask],
                )
                losses["dmax"] = l_dmax
                total_loss = total_loss + self.lambda_dmax * l_dmax

        # Ubiquitination site prediction
        if "ub_labels" in targets and "ugs_scores" in predictions:
            if len(predictions["ugs_scores"]) > 0:
                lysine_indices = predictions["lysine_indices"]
                ub_targets = targets["ub_labels"][lysine_indices]
                # Need logits, not scores. Use inverse sigmoid
                ugs_logits = torch.log(
                    predictions["ugs_scores"].clamp(1e-6, 1 - 1e-6) /
                    (1 - predictions["ugs_scores"].clamp(1e-6, 1 - 1e-6))
                )
                l_ubsite = self.ubsite_loss(ugs_logits, ub_targets)
                losses["ubsite"] = l_ubsite
                total_loss = total_loss + self.lambda_ubsite * l_ubsite

        losses["total"] = total_loss
        return losses

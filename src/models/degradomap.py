"""
DegradoMap: Full Model

Integrates all four modules:
  A. SUG Module (Structural Ubiquitination Geometry)
  B. E3 Compatibility Module
  C. Cellular Context Encoder
  D. Fusion and Prediction Head

Into a single end-to-end model for protein degradability prediction.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.sug_module import SUGModule
from src.models.e3_compat_module import E3CompatModule
from src.models.context_module import ContextModule
from src.models.fusion_module import FusionModule


class DegradoMap(nn.Module):
    """
    DegradoMap: Structure-Conditioned, E3-Aware, Context-Dependent
    Protein Degradability Prediction.

    Predicts DegradoScore(T, E, C) ∈ [0, 1] representing the probability
    that target protein T can be degraded by a PROTAC recruiting E3 ligase E
    in cellular context C.
    """

    def __init__(
        self,
        # SUG config
        node_input_dim: int = 28,
        sug_hidden_dim: int = 256,
        sug_output_dim: int = 128,
        sug_num_layers: int = 6,
        sug_max_radius: float = 10.0,
        sug_num_basis: int = 8,
        # E3 compat config
        e3_hidden_dim: int = 128,
        e3_output_dim: int = 128,
        e3_num_heads: int = 8,
        e3_num_layers: int = 4,
        e3_num_families: int = 5,
        # Context config
        context_input_dim: int = 200,
        context_output_dim: int = 128,
        context_dropout: float = 0.2,
        # Fusion config
        fusion_hidden_dim: int = 256,
        pred_hidden_dim: int = 128,
        # General
        dropout: float = 0.1,
        # New features
        use_e3_onehot: bool = False,
        use_global_stats: bool = False,
    ):
        super().__init__()

        self.use_e3_onehot = use_e3_onehot
        self.use_global_stats = use_global_stats

        # Module A: Structural Ubiquitination Geometry
        self.sug_module = SUGModule(
            node_input_dim=node_input_dim,
            hidden_dim=sug_hidden_dim,
            output_dim=sug_output_dim,
            num_layers=sug_num_layers,
            max_radius=sug_max_radius,
            num_basis=sug_num_basis,
            dropout=dropout,
            use_global_stats=use_global_stats,
        )

        # Module B: E3 Compatibility Network
        self.e3_compat_module = E3CompatModule(
            target_dim=sug_hidden_dim,  # Takes hidden features from SUG
            hidden_dim=e3_hidden_dim,
            output_dim=e3_output_dim,
            num_heads=e3_num_heads,
            num_layers=e3_num_layers,
            dropout=dropout,
            num_e3_families=e3_num_families,
        )

        # Module C: Cellular Context Encoder
        self.context_module = ContextModule(
            input_dim=context_input_dim,
            output_dim=context_output_dim,
            dropout=context_dropout,
        )

        # E3 one-hot dimension (6 families: CRBN, VHL, MDM2, cIAP1, DCAF16, Other)
        self.e3_onehot_dim = 6 if use_e3_onehot else 0
        # Global stats dimension (8: mean/std/min/max of pLDDT and SASA)
        self.global_stats_dim = 8 if use_global_stats else 0

        # Adjusted SUG dim includes global stats if enabled
        adjusted_sug_dim = sug_output_dim + self.global_stats_dim

        # Module D: Fusion and Prediction
        self.fusion_module = FusionModule(
            sug_dim=adjusted_sug_dim,
            compat_dim=e3_output_dim,
            context_dim=context_output_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            pred_hidden_dim=pred_hidden_dim,
            dropout=dropout,
            e3_onehot_dim=self.e3_onehot_dim,
            lysine_summary_dim=sug_output_dim,  # Original SUG output dim (without global stats)
        )

        # E3 name to index mapping
        self.e3_to_idx = {
            'CRBN': 0, 'VHL': 1, 'MDM2': 2, 'cIAP1': 3, 'DCAF16': 4,
        }

    def forward(
        self,
        protein_graph: Data,
        e3_name: str,
        context_features: Optional[torch.Tensor] = None,
        context_groups: Optional[Dict[str, torch.Tensor]] = None,
        e3_srd_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            protein_graph: PyG Data object for target protein
            e3_name: E3 ligase family name (e.g., "CRBN", "VHL")
            context_features: [B, context_dim] flat context vector
            context_groups: Dict of feature group tensors (alternative to flat)
            e3_srd_features: [L, feat_dim] E3 SRD structural features

        Returns:
            Dictionary with all predictions and intermediate representations:
            - degrado_score: [B] probability of degradability
            - degrado_logits: [B] raw logits
            - dc50_pred: [B] predicted log10(DC50)
            - dmax_pred: [B] predicted Dmax fraction
            - ugs_scores: [N_lys] per-lysine UGS scores
            - lysine_indices: [N_lys] lysine residue indices
            - interface_scores: [N] per-residue E3 interface propensity
            - cross_attn_map: attention weights for interpretation
            - sug_vector: [B] SUG representation
            - compat_vector: [B] compatibility representation
            - context_vector: [B] context representation
        """
        batch = protein_graph.batch if hasattr(protein_graph, 'batch') and protein_graph.batch is not None else torch.zeros(protein_graph.x.size(0), dtype=torch.long, device=protein_graph.x.device)
        num_graphs = batch.max().item() + 1

        # Module A: SUG
        sug_out = self.sug_module(protein_graph)
        sug_vector = sug_out["sug_vector"]

        # Add global stats to SUG vector if enabled
        if self.use_global_stats and "global_stats" in sug_out:
            sug_vector = torch.cat([sug_vector, sug_out["global_stats"]], dim=-1)

        # Module B: E3 Compatibility
        e3_out = self.e3_compat_module(
            target_node_feats=sug_out["node_features"],
            e3_name=e3_name,
            target_batch=batch,
            e3_srd_features=e3_srd_features,
        )

        # Module C: Context
        if context_features is not None or context_groups is not None:
            context_vector = self.context_module(
                context_features=context_features,
                feature_groups=context_groups,
            )
        else:
            # No context: use zero vector
            context_vector = torch.zeros(
                num_graphs, self.context_module.output_dim,
                device=protein_graph.x.device,
            )

        # E3 one-hot encoding
        e3_onehot = None
        if self.use_e3_onehot:
            e3_idx = self.e3_to_idx.get(e3_name, 5)  # 5 = "Other"
            e3_onehot = torch.zeros(num_graphs, 6, device=protein_graph.x.device)
            e3_onehot[:, e3_idx] = 1.0

        # Module D: Fusion and Prediction
        fusion_out = self.fusion_module(
            sug_vector=sug_vector,
            compat_vector=e3_out["compat_vector"],
            context_vector=context_vector,
            lysine_summary=sug_out["lysine_summary"],
            e3_onehot=e3_onehot,
        )

        # Combine all outputs
        return {
            **fusion_out,
            "ugs_scores": sug_out["ugs_scores"],
            "lysine_indices": sug_out["lysine_indices"],
            "interface_scores": e3_out["interface_scores"],
            "cross_attn_map": e3_out["cross_attn_map"],
            "sug_vector": sug_out["sug_vector"],
            "compat_vector": e3_out["compat_vector"],
            "context_vector": context_vector,
            "node_features": sug_out["node_features"],
        }

    def predict(self, protein_graph: Data, e3_name: str,
                context_features: Optional[torch.Tensor] = None) -> float:
        """
        Simple prediction interface.

        Returns:
            DegradoScore as a float
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(protein_graph, e3_name, context_features)
            return out["degrado_score"].item()

    def get_interpretation(self, protein_graph: Data, e3_name: str,
                           context_features: Optional[torch.Tensor] = None,
                           residues: Optional[list] = None) -> Dict:
        """
        Get interpretable predictions.

        Returns:
            Dictionary with human-readable interpretation including
            top lysines, interface residues, and feature importance.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(protein_graph, e3_name, context_features)

            interpretation = {
                "degrado_score": out["degrado_score"].cpu().numpy(),
                "dc50_predicted": 10 ** out["dc50_pred"].cpu().numpy(),  # Convert from log10
                "dmax_predicted": out["dmax_pred"].cpu().numpy() * 100,  # Convert to percentage
            }

            # Top lysines by UGS score
            if len(out["ugs_scores"]) > 0:
                scores = out["ugs_scores"].cpu().numpy()
                indices = out["lysine_indices"].cpu().numpy()
                sorted_idx = scores.argsort()[::-1]

                top_lysines = []
                for i in sorted_idx[:10]:
                    lys_info = {
                        "residue_index": int(indices[i]),
                        "ugs_score": float(scores[i]),
                    }
                    if residues is not None and indices[i] < len(residues):
                        lys_info["residue"] = f"K{indices[i]+1}"
                    top_lysines.append(lys_info)

                interpretation["top_lysines"] = top_lysines

            # Interface scores
            interface = out["interface_scores"].cpu().numpy()
            top_interface = interface.argsort()[::-1][:20]
            interpretation["top_interface_residues"] = [
                {"residue_index": int(idx), "score": float(interface[idx])}
                for idx in top_interface
            ]

            return interpretation

    @classmethod
    def from_config(cls, config) -> "DegradoMap":
        """Create model from config dataclass."""
        return cls(
            node_input_dim=28,
            sug_hidden_dim=config.sug.hidden_dim,
            sug_output_dim=config.sug.output_dim,
            sug_num_layers=config.sug.num_layers,
            sug_max_radius=config.sug.max_radius,
            sug_num_basis=config.sug.num_basis,
            e3_hidden_dim=config.e3_compat.hidden_dim,
            e3_output_dim=config.e3_compat.output_dim,
            e3_num_heads=config.e3_compat.cross_attn_heads,
            e3_num_layers=config.e3_compat.cross_attn_layers,
            e3_num_families=len(config.e3_compat.e3_families),
            context_input_dim=config.context.input_dim,
            context_output_dim=config.context.output_dim,
            context_dropout=config.context.dropout,
            fusion_hidden_dim=config.fusion.gate_hidden_dim,
            pred_hidden_dim=config.fusion.pred_hidden_dim,
            dropout=config.sug.dropout,
        )

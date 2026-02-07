"""Default configuration for DegradoMap."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"

    # PROTAC-DB thresholds
    dc50_pos_threshold: float = 100.0   # nM, below = positive
    dmax_pos_threshold: float = 80.0    # %, above = positive
    dc50_neg_threshold: float = 1000.0  # nM, above = hard negative
    dmax_neg_threshold: float = 30.0    # %, below = hard negative

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class SUGConfig:
    """Module A: Structural Ubiquitination Geometry."""
    node_feat_dim: int = 128
    edge_feat_dim: int = 64
    num_layers: int = 6
    max_radius: float = 10.0  # Angstroms, E2 catalytic distance
    num_basis: int = 8
    lmax: int = 2  # Spherical harmonics order
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1


@dataclass
class E3CompatConfig:
    """Module B: E3 Compatibility Network."""
    cross_attn_heads: int = 8
    cross_attn_layers: int = 4
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    e3_families: List[str] = field(default_factory=lambda: [
        "CRBN", "VHL", "MDM2", "cIAP1", "DCAF16"
    ])


@dataclass
class ContextConfig:
    """Module C: Cellular Context Encoder."""
    input_dim: int = 200
    hidden_dims: List[int] = field(default_factory=lambda: [256, 192, 128])
    output_dim: int = 128
    dropout: float = 0.2
    num_residual_blocks: int = 3


@dataclass
class FusionConfig:
    """Module D: Fusion and Prediction Head."""
    gate_hidden_dim: int = 256
    fusion_dim: int = 384  # SUG(128) + E3(128) + Context(128)
    pred_hidden_dim: int = 128
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    # Phase 1: Pre-training
    pretrain_lr: float = 1e-4
    pretrain_epochs: int = 100
    pretrain_batch_size: int = 32

    # Phase 2: Fine-tuning
    finetune_lr: float = 5e-5
    finetune_epochs: int = 200
    finetune_batch_size: int = 16

    # Loss weights
    lambda_degrad: float = 1.0
    lambda_dc50: float = 0.3
    lambda_dmax: float = 0.3
    lambda_esi: float = 0.2
    lambda_ubsite: float = 0.2

    # Label smoothing
    label_smoothing: float = 0.05

    # Optimizer
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Semi-supervised
    pseudolabel_threshold: float = 0.9
    pseudolabel_start_epoch: int = 50

    # General
    seed: int = 42
    num_workers: int = 8
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class DegradoMapConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sug: SUGConfig = field(default_factory=SUGConfig)
    e3_compat: E3CompatConfig = field(default_factory=E3CompatConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

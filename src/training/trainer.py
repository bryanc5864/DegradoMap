"""
DegradoMap Training Pipeline.

Three-phase training:
  Phase 1: Pre-training on UbiBrowser ESIs + PhosphoSitePlus Ub sites
  Phase 2: Semi-supervised fine-tuning on PROTAC-DB
  Phase 3: Proteome-wide inference
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from src.models.degradomap import DegradoMap
from src.training.losses import DegradoMapLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_graph_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for batching protein graphs."""
    graphs = [item["graph"] for item in batch]
    batched_graph = Batch.from_data_list(graphs)

    result = {"graph": batched_graph}

    # Collate non-graph fields
    for key in batch[0]:
        if key == "graph":
            continue
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = values

    return result


class DegradoMapTrainer:
    """
    Trainer for the DegradoMap model.

    Handles all three training phases and evaluation.
    """

    def __init__(self, model: DegradoMap, config: dict = None,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        # Default training config
        self.lr = self.config.get("lr", 1e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-5)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.warmup_steps = self.config.get("warmup_steps", 500)

        # Loss function with optional class weighting
        self.loss_fn = DegradoMapLoss(
            lambda_degrad=self.config.get("lambda_degrad", 1.0),
            lambda_dc50=self.config.get("lambda_dc50", 0.3),
            lambda_dmax=self.config.get("lambda_dmax", 0.3),
            lambda_esi=self.config.get("lambda_esi", 0.2),
            lambda_ubsite=self.config.get("lambda_ubsite", 0.2),
            label_smoothing=self.config.get("label_smoothing", 0.05),
            pos_weight=self.config.get("pos_weight", None),
        )

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.training_log = []
        self.best_val_metric = 0.0

    def _get_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=num_training_steps,
            pct_start=min(0.1, self.warmup_steps / max(num_training_steps, 1)),
            anneal_strategy='cos',
        )

    def train_epoch_ubsite(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train one epoch for ubiquitination site prediction (Phase 1B)."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_lysines = 0

        for batch in tqdm(dataloader, desc=f"Phase1B Epoch {epoch}", leave=False):
            graph = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward through SUG module only
            sug_out = self.model.sug_module(graph)

            # Compute Ub site loss
            if hasattr(graph, 'ub_labels') and len(sug_out["ugs_scores"]) > 0:
                lysine_indices = sug_out["lysine_indices"]
                ub_targets = graph.ub_labels[lysine_indices].to(self.device)

                # Binary cross-entropy on UGS scores
                loss = nn.functional.binary_cross_entropy(
                    sug_out["ugs_scores"],
                    ub_targets,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                predictions = (sug_out["ugs_scores"] > 0.5).float()
                total_correct += (predictions == ub_targets).sum().item()
                total_lysines += len(ub_targets)

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = total_correct / max(total_lysines, 1)

        return {"loss": avg_loss, "accuracy": accuracy, "total_lysines": total_lysines}

    def train_epoch_esi(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train one epoch for E3-substrate interaction prediction (Phase 1A)."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_data in tqdm(dataloader, desc=f"Phase1A Epoch {epoch}", leave=False):
            graphs = batch_data["graph"].to(self.device)
            labels = batch_data["label"].to(self.device)
            e3_names = batch_data["e3_name"]

            self.optimizer.zero_grad()

            # Process each E3 name group separately
            batch_losses = []
            batch_preds = []

            # For simplicity, process all with the first E3 name
            # In production, group by E3 name
            e3_name = e3_names[0] if isinstance(e3_names, list) else e3_names

            # Forward pass
            outputs = self.model(graphs, e3_name)

            # ESI-style loss: use degrado_score as interaction score
            loss = nn.functional.binary_cross_entropy(
                outputs["degrado_score"],
                labels,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            predictions = (outputs["degrado_score"] > 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = total_correct / max(total_samples, 1)

        return {"loss": avg_loss, "accuracy": accuracy}

    def train_epoch_degradation(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train one epoch for degradation prediction (Phase 2)."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loss_components = {}

        for batch_data in tqdm(dataloader, desc=f"Phase2 Epoch {epoch}", leave=False):
            graphs = batch_data["graph"].to(self.device)
            labels = batch_data["label"].to(self.device)
            e3_names = batch_data["e3_name"]
            weights = batch_data["weight"].to(self.device)
            dc50_labels = batch_data["dc50"].to(self.device)
            dmax_labels = batch_data["dmax"].to(self.device)

            self.optimizer.zero_grad()

            e3_name = e3_names[0] if isinstance(e3_names, list) else e3_names

            # Forward pass
            outputs = self.model(graphs, e3_name)

            # Compute loss
            targets = {
                "degrad_label": labels,
                "sample_weight": weights,
                "dc50_label": dc50_labels,
                "dmax_label": dmax_labels,
            }

            losses = self.loss_fn(outputs, targets)
            loss = losses["total"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            predictions = (outputs["degrado_score"] > 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

            for k, v in losses.items():
                if k != "total":
                    loss_components[k] = loss_components.get(k, 0) + v.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = total_correct / max(total_samples, 1)
        avg_components = {k: v / max(len(dataloader), 1) for k, v in loss_components.items()}

        return {"loss": avg_loss, "accuracy": accuracy, "components": avg_components}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_scores = []
        total_loss = 0.0

        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            graphs = batch_data["graph"].to(self.device)
            labels = batch_data["label"].to(self.device)
            e3_names = batch_data["e3_name"]

            e3_name = e3_names[0] if isinstance(e3_names, list) else e3_names

            outputs = self.model(graphs, e3_name)

            scores = outputs["degrado_score"]
            preds = (scores > 0.5).float()

            loss = nn.functional.binary_cross_entropy(scores, labels)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        # Metrics
        accuracy = (all_preds == all_labels).mean()
        avg_loss = total_loss / max(len(dataloader), 1)

        # AUROC and threshold optimization
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5
            auprc = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            # Find optimal threshold using Youden's J statistic (TPR - FPR)
            best_threshold = 0.5
            best_f1 = f1
            for thresh in np.arange(0.1, 0.9, 0.05):
                thresh_preds = (all_scores >= thresh).astype(float)
                thresh_f1 = f1_score(all_labels, thresh_preds, zero_division=0)
                if thresh_f1 > best_f1:
                    best_f1 = thresh_f1
                    best_threshold = thresh

            # Compute metrics at optimal threshold
            opt_preds = (all_scores >= best_threshold).astype(float)
            opt_f1 = f1_score(all_labels, opt_preds, zero_division=0)
            opt_accuracy = (opt_preds == all_labels).mean()
        except Exception:
            auroc = 0.5
            auprc = 0.0
            f1 = 0.0
            best_threshold = 0.5
            opt_f1 = 0.0
            opt_accuracy = accuracy

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "optimal_threshold": best_threshold,
            "f1_at_optimal": opt_f1,
            "accuracy_at_optimal": opt_accuracy,
            "predictions": all_scores,
            "labels": all_labels,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded: {path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def train_phase1(self, ubsite_loader: DataLoader, esi_loader: DataLoader,
                     val_loader: DataLoader = None, epochs: int = 50,
                     save_dir: str = "checkpoints/phase1"):
        """
        Phase 1: Pre-training on Ub sites and E3-substrate interactions.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Pre-training")
        logger.info("=" * 60)

        phase1_log = []

        for epoch in range(1, epochs + 1):
            # Task 1B: Ub site prediction
            if ubsite_loader is not None and len(ubsite_loader) > 0:
                ubsite_metrics = self.train_epoch_ubsite(ubsite_loader, epoch)
                logger.info(f"Epoch {epoch} - UbSite: loss={ubsite_metrics['loss']:.4f}, "
                           f"acc={ubsite_metrics['accuracy']:.4f}")
            else:
                ubsite_metrics = {"loss": 0, "accuracy": 0}

            # Task 1A: ESI prediction
            if esi_loader is not None and len(esi_loader) > 0:
                esi_metrics = self.train_epoch_esi(esi_loader, epoch)
                logger.info(f"Epoch {epoch} - ESI: loss={esi_metrics['loss']:.4f}, "
                           f"acc={esi_metrics['accuracy']:.4f}")
            else:
                esi_metrics = {"loss": 0, "accuracy": 0}

            # Validation
            val_metrics = {}
            if val_loader is not None and len(val_loader) > 0:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch} - Val: acc={val_metrics['accuracy']:.4f}, "
                           f"auroc={val_metrics['auroc']:.4f}")

            epoch_log = {
                "epoch": epoch,
                "ubsite": ubsite_metrics,
                "esi": esi_metrics,
                "val": val_metrics,
            }
            phase1_log.append(epoch_log)

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(
                    f"{save_dir}/checkpoint_epoch{epoch}.pt",
                    epoch, epoch_log,
                )

        # Save final checkpoint
        self.save_checkpoint(f"{save_dir}/checkpoint_final.pt", epochs, phase1_log[-1])

        return phase1_log

    def train_phase2(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 100, save_dir: str = "checkpoints/phase2"):
        """
        Phase 2: Semi-supervised fine-tuning on PROTAC-DB.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Fine-tuning on PROTAC-DB")
        logger.info("=" * 60)

        phase2_log = []
        best_auroc = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch_degradation(train_loader, epoch)
            logger.info(f"Epoch {epoch} - Train: loss={train_metrics['loss']:.4f}, "
                       f"acc={train_metrics['accuracy']:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch} - Val: acc={val_metrics['accuracy']:.4f}, "
                           f"auroc={val_metrics['auroc']:.4f}, f1={val_metrics['f1']:.4f}")

                # Save best model
                if val_metrics["auroc"] > best_auroc:
                    best_auroc = val_metrics["auroc"]
                    self.save_checkpoint(
                        f"{save_dir}/best_model.pt",
                        epoch, val_metrics,
                    )
                    logger.info(f"New best AUROC: {best_auroc:.4f}")
            else:
                val_metrics = {}

            epoch_log = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            phase2_log.append(epoch_log)

            if epoch % 20 == 0:
                self.save_checkpoint(
                    f"{save_dir}/checkpoint_epoch{epoch}.pt",
                    epoch, epoch_log,
                )

        self.save_checkpoint(f"{save_dir}/checkpoint_final.pt", epochs, phase2_log[-1])

        return phase2_log

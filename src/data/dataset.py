"""
DegradoMap Dataset classes.

Provides PyTorch datasets for:
  1. Pre-training: Ubiquitination site prediction, E3-substrate interaction
  2. Fine-tuning: PROTAC degradation prediction
  3. Inference: Proteome-wide prediction
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.models.sug_module import protein_to_graph, encode_residue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UbiquitinationSiteDataset(Dataset):
    """
    Dataset for pre-training Task 1B: Ubiquitination site prediction.

    Each sample is a protein structure with per-residue labels indicating
    known ubiquitination sites from PhosphoSitePlus.
    """

    def __init__(self, structure_dir: str = "data/processed/structures",
                 ub_sites_file: str = "data/raw/phosphosite/phosphosite_ubiquitination.csv",
                 radius: float = 10.0):
        super().__init__()

        self.structure_dir = Path(structure_dir)
        self.radius = radius

        # Load ubiquitination sites
        self.ub_sites = self._load_ub_sites(ub_sites_file)

        # Find structures that have Ub site annotations
        self.samples = self._match_structures()
        logger.info(f"UbSite dataset: {len(self.samples)} proteins with annotations")

    def _load_ub_sites(self, filepath: str) -> Dict[str, List[int]]:
        """Load and index ubiquitination sites by UniProt ID."""
        sites_by_protein = {}

        try:
            df = pd.read_csv(filepath)

            # PhosphoSitePlus format: ACC_ID for UniProt, MOD_RSD for site
            if "ACC_ID" in df.columns and "MOD_RSD" in df.columns:
                # Filter for human only
                if "ORGANISM" in df.columns:
                    df = df[df["ORGANISM"].str.lower() == "human"]

                for _, row in df.iterrows():
                    uniprot = str(row.get("ACC_ID", "")).strip()
                    mod_rsd = str(row.get("MOD_RSD", ""))

                    if not uniprot or uniprot == "nan":
                        continue

                    # Parse position from MOD_RSD (format: K123-ub)
                    try:
                        pos_str = mod_rsd.split("-")[0]
                        if pos_str.startswith("K"):
                            pos = int(pos_str[1:])
                            if uniprot not in sites_by_protein:
                                sites_by_protein[uniprot] = []
                            sites_by_protein[uniprot].append(pos)
                    except (ValueError, IndexError):
                        continue

            # Also try curated format
            elif "UniProt_ID" in df.columns and "Position" in df.columns:
                for _, row in df.iterrows():
                    uniprot = str(row["UniProt_ID"])
                    pos = int(row["Position"])
                    if uniprot not in sites_by_protein:
                        sites_by_protein[uniprot] = []
                    sites_by_protein[uniprot].append(pos)

        except Exception as e:
            logger.warning(f"Error loading Ub sites: {e}")

        logger.info(f"Loaded Ub sites for {len(sites_by_protein)} proteins")
        return sites_by_protein

    def _match_structures(self) -> List[Dict]:
        """Match available structures with Ub site annotations."""
        samples = []

        for pt_file in sorted(self.structure_dir.glob("*.pt")):
            uniprot_id = pt_file.stem
            if uniprot_id in self.ub_sites:
                samples.append({
                    "uniprot_id": uniprot_id,
                    "structure_path": str(pt_file),
                    "ub_sites": self.ub_sites[uniprot_id],
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        sample = self.samples[idx]
        processed = torch.load(sample["structure_path"], weights_only=False)

        # Build graph
        graph = protein_to_graph(
            coords=processed["coords"],
            residues=processed["residues"],
            plddt=processed["plddt"],
            sasa=processed["sasa"],
            disorder=processed["disorder"],
            radius=self.radius,
        )

        # Create per-residue Ub site labels
        n_residues = len(processed["residues"])
        ub_labels = torch.zeros(n_residues)
        residue_numbers = processed["residue_numbers"]

        for site_pos in sample["ub_sites"]:
            # Find matching residue index
            for i, rn in enumerate(residue_numbers):
                if rn == site_pos and processed["residues"][i] == "K":
                    ub_labels[i] = 1.0
                    break

        graph.ub_labels = ub_labels
        graph.uniprot_id = sample["uniprot_id"]

        return graph


class ESIDataset(Dataset):
    """
    Dataset for pre-training Task 1A: E3-Substrate Interaction prediction.

    Each sample is a (target_protein, E3_ligase, label) triple.
    """

    def __init__(self, esi_file: str = "data/raw/ubibrowser/curated_esi_interactions.csv",
                 structure_dir: str = "data/processed/structures",
                 radius: float = 10.0, neg_ratio: float = 3.0):
        super().__init__()

        self.structure_dir = Path(structure_dir)
        self.radius = radius

        # Load ESI data
        self.esi_df = pd.read_csv(esi_file)
        self.available_structures = set(
            p.stem for p in self.structure_dir.glob("*.pt")
        )

        # Build positive samples
        self.positives = self._build_positives()

        # Generate negative samples
        self.negatives = self._generate_negatives(neg_ratio)

        self.samples = self.positives + self.negatives
        logger.info(f"ESI dataset: {len(self.positives)} positives, {len(self.negatives)} negatives")

    def _build_positives(self) -> List[Dict]:
        """Build positive ESI samples from curated data."""
        positives = []
        # Map gene names to UniProt IDs (from our AlphaFold mapping)
        mapping_path = Path("data/raw/alphafold/protein_structure_mapping.json")
        gene_to_uniprot = {}
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
                for name, info in mapping.items():
                    gene_to_uniprot[name.upper()] = info["uniprot"]
                    if info.get("uniprot"):
                        gene_to_uniprot[info["uniprot"]] = info["uniprot"]

        for _, row in self.esi_df.iterrows():
            substrate_gene = str(row.get("Substrate_Gene", ""))
            e3_name = str(row.get("E3_Ligase", ""))

            # Find UniProt ID
            uniprot = gene_to_uniprot.get(substrate_gene.upper(),
                                           gene_to_uniprot.get(substrate_gene, None))
            if uniprot and uniprot in self.available_structures:
                positives.append({
                    "uniprot_id": uniprot,
                    "e3_name": e3_name,
                    "label": 1.0,
                    "evidence": str(row.get("Evidence_Type", "Unknown")),
                })

        return positives

    def _generate_negatives(self, ratio: float) -> List[Dict]:
        """Generate negative ESI samples by random pairing."""
        negatives = []
        e3_names = list(self.esi_df["E3_Ligase"].unique())
        all_structures = list(self.available_structures)

        # Positive pairs to avoid
        pos_pairs = set(
            (s["uniprot_id"], s["e3_name"]) for s in self.positives
        )

        n_neg = int(len(self.positives) * ratio)
        rng = np.random.RandomState(42)

        for _ in range(n_neg * 3):  # Over-sample then trim
            if len(negatives) >= n_neg:
                break
            uniprot = rng.choice(all_structures)
            e3 = rng.choice(e3_names)
            if (uniprot, e3) not in pos_pairs:
                negatives.append({
                    "uniprot_id": uniprot,
                    "e3_name": e3,
                    "label": 0.0,
                    "evidence": "Negative_sample",
                })
                pos_pairs.add((uniprot, e3))  # Avoid duplicates

        return negatives[:n_neg]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load structure
        struct_path = self.structure_dir / f"{sample['uniprot_id']}.pt"
        if struct_path.exists():
            processed = torch.load(str(struct_path), weights_only=False)
            graph = protein_to_graph(
                coords=processed["coords"],
                residues=processed["residues"],
                plddt=processed["plddt"],
                sasa=processed["sasa"],
                disorder=processed["disorder"],
                radius=self.radius,
            )
        else:
            # Placeholder for missing structures
            graph = Data(
                x=torch.randn(50, 28),
                pos=torch.randn(50, 3),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_vec=torch.zeros(0, 3),
                edge_len=torch.zeros(0, 1),
                lysine_mask=torch.zeros(50),
                num_nodes=50,
            )

        return {
            "graph": graph,
            "e3_name": sample["e3_name"],
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


class DegradationDataset(Dataset):
    """
    Dataset for fine-tuning: PROTAC degradation prediction.

    Each sample is a (target, E3_ligase, cell_line, degradation_label) tuple.
    Uses structure features from Module A input and context features from Module C.
    """

    # Class-level cache for Ub sites (loaded once)
    _ub_sites_cache = None

    def __init__(self, samples: List[Dict],
                 structure_dir: str = "data/processed/structures",
                 esm_dir: str = "data/processed/esm_embeddings",
                 ub_sites_file: str = "data/raw/phosphosite/phosphosite_ubiquitination.csv",
                 radius: float = 10.0,
                 use_esm: bool = False,
                 use_ub_sites: bool = False):
        super().__init__()

        self.samples = samples
        self.structure_dir = Path(structure_dir)
        self.esm_dir = Path(esm_dir)
        self.radius = radius
        self.use_esm = use_esm
        self.use_ub_sites = use_ub_sites

        if use_esm:
            logger.info(f"Using ESM-2 embeddings from {esm_dir}")

        if use_ub_sites:
            if DegradationDataset._ub_sites_cache is None:
                DegradationDataset._ub_sites_cache = self._load_ub_sites(ub_sites_file)
            self.ub_sites = DegradationDataset._ub_sites_cache
            logger.info(f"Using known Ub sites for {len(self.ub_sites)} proteins")
        else:
            self.ub_sites = {}

    def _load_ub_sites(self, filepath: str) -> Dict[str, List[int]]:
        """Load ubiquitination sites from PhosphoSitePlus."""
        sites_by_protein = {}
        try:
            df = pd.read_csv(filepath)
            if "ACC_ID" in df.columns and "MOD_RSD" in df.columns:
                if "ORGANISM" in df.columns:
                    df = df[df["ORGANISM"].str.lower() == "human"]
                for _, row in df.iterrows():
                    uniprot = str(row.get("ACC_ID", "")).strip()
                    mod_rsd = str(row.get("MOD_RSD", ""))
                    if not uniprot or uniprot == "nan":
                        continue
                    try:
                        pos_str = mod_rsd.split("-")[0]
                        if pos_str.startswith("K"):
                            pos = int(pos_str[1:])
                            if uniprot not in sites_by_protein:
                                sites_by_protein[uniprot] = []
                            sites_by_protein[uniprot].append(pos)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.warning(f"Error loading Ub sites: {e}")
        return sites_by_protein

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load structure
        uniprot_id = sample.get("uniprot_id", "unknown")
        struct_path = self.structure_dir / f"{uniprot_id}.pt"
        esm_path = self.esm_dir / f"{uniprot_id}_esm.pt"

        # Load ESM embeddings if available and enabled
        esm_embeddings = None
        if self.use_esm and esm_path.exists():
            esm_data = torch.load(str(esm_path), weights_only=False)
            esm_embeddings = esm_data.get("embeddings", None)

        if struct_path.exists():
            processed = torch.load(str(struct_path), weights_only=False)

            # Handle ESM embedding length mismatch
            n_residues = len(processed["residues"])
            if esm_embeddings is not None and esm_embeddings.shape[0] != n_residues:
                # Pad or truncate ESM embeddings to match structure
                if esm_embeddings.shape[0] < n_residues:
                    padding = torch.zeros(n_residues - esm_embeddings.shape[0], esm_embeddings.shape[1])
                    esm_embeddings = torch.cat([esm_embeddings, padding], dim=0)
                else:
                    esm_embeddings = esm_embeddings[:n_residues]

            # Get known Ub sites for this protein
            known_ub_positions = self.ub_sites.get(uniprot_id, [])
            residue_numbers = processed.get("residue_numbers", list(range(1, n_residues + 1)))

            graph = protein_to_graph(
                coords=processed["coords"],
                residues=processed["residues"],
                plddt=processed["plddt"],
                sasa=processed["sasa"],
                disorder=processed["disorder"],
                esm_embeddings=esm_embeddings,
                radius=self.radius,
                use_esm=self.use_esm,
                known_ub_sites=known_ub_positions if self.use_ub_sites else None,
                residue_numbers=residue_numbers if self.use_ub_sites else None,
            )

            # Add protein-level Ub count as graph attribute
            if self.use_ub_sites:
                graph.ub_site_count = torch.tensor([len(known_ub_positions)], dtype=torch.float32)
        else:
            # Create placeholder
            n = 100
            feat_dim = 1284 if self.use_esm else 28
            graph = Data(
                x=torch.randn(n, feat_dim),
                pos=torch.randn(n, 3) * 10,
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_vec=torch.zeros(0, 3),
                edge_len=torch.zeros(0, 1),
                lysine_mask=torch.zeros(n),
                num_nodes=n,
            )

        return {
            "graph": graph,
            "e3_name": sample.get("e3_name", "CRBN"),
            "cell_line": sample.get("cell_line", "unknown"),
            "label": torch.tensor(sample.get("label", 0.0), dtype=torch.float32),
            "dc50": torch.tensor(sample.get("dc50_log10", 2.0), dtype=torch.float32),
            "dmax": torch.tensor(sample.get("dmax_fraction", 0.5), dtype=torch.float32),
            "weight": torch.tensor(sample.get("weight", 1.0), dtype=torch.float32),
            "target_gene": sample.get("target_gene", "unknown"),
            "uniprot_id": uniprot_id,
        }

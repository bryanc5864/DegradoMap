"""
Process AlphaFold structures into graph representations for the SUG module.

Extracts:
  - Cα coordinates
  - Residue identities
  - pLDDT scores (B-factor column in AlphaFold PDBs)
  - Solvent-accessible surface area (SASA) via Biopython/DSSP
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Three-letter to one-letter amino acid code mapping
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def parse_pdb_file(pdb_path: str) -> Dict:
    """
    Parse a PDB file and extract residue-level features.

    For AlphaFold PDBs, B-factor = pLDDT score.

    Returns:
        Dictionary with:
        - coords: numpy array of Cα coordinates [N, 3]
        - residues: list of one-letter amino acid codes
        - plddt: numpy array of pLDDT scores [N]
        - residue_numbers: list of residue numbers
        - chain: chain ID
    """
    coords = []
    residues = []
    plddt_scores = []
    residue_numbers = []
    chain_id = None
    seen_residues = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue

                resname = line[17:20].strip()
                chain = line[21]
                resnum = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])

                res_key = (chain, resnum)
                if res_key in seen_residues:
                    continue
                seen_residues.add(res_key)

                aa = AA3TO1.get(resname, 'X')
                if aa == 'X':
                    continue

                coords.append([x, y, z])
                residues.append(aa)
                plddt_scores.append(bfactor)
                residue_numbers.append(resnum)
                if chain_id is None:
                    chain_id = chain

    if not coords:
        return None

    return {
        "coords": np.array(coords, dtype=np.float32),
        "residues": residues,
        "plddt": np.array(plddt_scores, dtype=np.float32),
        "residue_numbers": residue_numbers,
        "chain": chain_id,
        "num_residues": len(residues),
    }


def compute_sasa_simple(coords: np.ndarray, probe_radius: float = 1.4) -> np.ndarray:
    """
    Compute approximate per-residue SASA using Shrake-Rupley algorithm (simplified).

    This is a fast approximation using Cα distances. For production, use DSSP or FreeSASA.

    Args:
        coords: [N, 3] Cα coordinates
        probe_radius: Probe radius in Angstroms

    Returns:
        [N] approximate SASA values per residue
    """
    n = len(coords)
    sasa = np.zeros(n)

    # Typical Cα-Cα distances: bonded ~3.8Å, in contact < 8Å
    # Use number of close contacts as inverse proxy for SASA
    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        n_contacts = np.sum((dists > 0) & (dists < 8.0))
        # More contacts = more buried = lower SASA
        # Scale so exposed residues have SASA ~150-200, buried ~20-50
        max_contacts = min(n - 1, 20)
        burial_fraction = min(n_contacts / max_contacts, 1.0)
        sasa[i] = 200.0 * (1.0 - burial_fraction) + 20.0 * burial_fraction

    return sasa


def compute_disorder_proxy(plddt: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """
    Use pLDDT as a proxy for disorder.

    AlphaFold2 assigns low pLDDT (<50) to intrinsically disordered regions.

    Args:
        plddt: [N] pLDDT scores
        threshold: pLDDT threshold for disorder

    Returns:
        [N] disorder scores (0-1, higher = more disordered)
    """
    disorder = np.clip(1.0 - (plddt / 100.0), 0, 1)
    # Sharpen: regions with pLDDT < threshold are strongly disordered
    disorder[plddt < threshold] = disorder[plddt < threshold] ** 0.5
    return disorder.astype(np.float32)


def process_structure(pdb_path: str) -> Optional[Dict]:
    """
    Process a single PDB structure into features for the SUG module.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary with processed features or None if parsing fails
    """
    parsed = parse_pdb_file(pdb_path)
    if parsed is None:
        return None

    coords = parsed["coords"]
    residues = parsed["residues"]
    plddt = parsed["plddt"]

    # Compute derived features
    sasa = compute_sasa_simple(coords)
    disorder = compute_disorder_proxy(plddt)

    # Identify lysines
    lysine_mask = np.array([1.0 if aa == 'K' else 0.0 for aa in residues], dtype=np.float32)
    lysine_positions = np.where(lysine_mask > 0)[0]

    return {
        "coords": torch.from_numpy(coords),
        "residues": residues,
        "plddt": torch.from_numpy(plddt),
        "sasa": torch.from_numpy(sasa),
        "disorder": torch.from_numpy(disorder),
        "lysine_mask": torch.from_numpy(lysine_mask),
        "lysine_positions": lysine_positions.tolist(),
        "residue_numbers": parsed["residue_numbers"],
        "num_residues": parsed["num_residues"],
        "num_lysines": int(lysine_mask.sum()),
    }


def process_all_structures(structure_dir: str = "data/raw/alphafold/structures",
                           output_dir: str = "data/processed/structures") -> Dict:
    """Process all downloaded AlphaFold structures."""
    struct_dir = Path(structure_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(struct_dir.glob("*.pdb"))
    logger.info(f"Processing {len(pdb_files)} PDB structures...")

    results = {}
    for pdb_path in tqdm(pdb_files, desc="Processing structures"):
        # Extract UniProt ID from filename: AF-{UNIPROT}-F1-model_v4.pdb
        parts = pdb_path.stem.split("-")
        if len(parts) >= 2:
            uniprot_id = parts[1]
        else:
            uniprot_id = pdb_path.stem

        processed = process_structure(str(pdb_path))
        if processed is not None:
            # Save processed features
            out_path = out_dir / f"{uniprot_id}.pt"
            torch.save(processed, str(out_path))
            results[uniprot_id] = {
                "path": str(out_path),
                "num_residues": processed["num_residues"],
                "num_lysines": processed["num_lysines"],
                "mean_plddt": processed["plddt"].mean().item(),
            }
        else:
            logger.warning(f"Failed to process {pdb_path.name}")

    logger.info(f"Processed {len(results)}/{len(pdb_files)} structures")

    # Save summary
    with open(out_dir / "processing_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    process_all_structures()

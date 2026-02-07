"""
Acquire AlphaFold DB predicted structures.

AlphaFold DB (https://alphafold.ebi.ac.uk/) provides predicted structures
for the entire human proteome (~20,000 proteins).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/alphafold")
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"


def get_alphafold_structure(uniprot_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download AlphaFold2 predicted structure for a UniProt ID.

    Args:
        uniprot_id: UniProt accession (e.g., 'P04637')
        output_dir: Directory to save the PDB file

    Returns:
        Path to downloaded PDB file, or None if failed
    """
    if output_dir is None:
        output_dir = RAW_DIR / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
    if out_path.exists():
        return out_path

    # AlphaFold DB API
    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                pdb_url = data[0].get("pdbUrl")
                if pdb_url:
                    pdb_resp = requests.get(pdb_url, timeout=60)
                    if pdb_resp.status_code == 200:
                        with open(out_path, "w") as f:
                            f.write(pdb_resp.text)
                        return out_path

        # Try direct URL pattern
        pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        pdb_resp = requests.get(pdb_url, timeout=60)
        if pdb_resp.status_code == 200:
            with open(out_path, "w") as f:
                f.write(pdb_resp.text)
            return out_path

    except Exception as e:
        logger.debug(f"Failed to download structure for {uniprot_id}: {e}")

    return None


def get_alphafold_plddt(uniprot_id: str) -> Optional[dict]:
    """Get pLDDT scores (confidence) for a protein from AlphaFold."""
    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return {
                    "uniprot_id": uniprot_id,
                    "plddt_url": data[0].get("pdbUrl"),
                    "cif_url": data[0].get("cifUrl"),
                    "pae_url": data[0].get("paeImageUrl"),
                }
    except Exception as e:
        logger.debug(f"Failed to get pLDDT for {uniprot_id}: {e}")
    return None


def batch_download_structures(uniprot_ids: List[str],
                               output_dir: Optional[Path] = None,
                               delay: float = 0.1) -> dict:
    """
    Download AlphaFold structures for a batch of UniProt IDs.

    Args:
        uniprot_ids: List of UniProt accessions
        output_dir: Output directory
        delay: Delay between requests (seconds)

    Returns:
        Dict mapping UniProt ID to download status
    """
    results = {}
    failed = []

    for uid in tqdm(uniprot_ids, desc="Downloading AlphaFold structures"):
        path = get_alphafold_structure(uid, output_dir)
        if path:
            results[uid] = str(path)
        else:
            results[uid] = None
            failed.append(uid)
        time.sleep(delay)

    logger.info(f"Downloaded {len(results) - len(failed)}/{len(uniprot_ids)} structures")
    if failed:
        logger.warning(f"Failed for {len(failed)} proteins: {failed[:10]}...")

    return results


# Priority proteins for DegradoMap (targets from PROTAC-DB + E3 ligases)
PRIORITY_PROTEINS = {
    # Key PROTAC targets
    "BRD4": "O60885",
    "BRD2": "P25440",
    "BRD3": "Q15059",
    "BTK": "Q06187",
    "CDK4": "P11802",
    "CDK6": "Q00534",
    "CDK9": "P50750",
    "AR": "P10275",
    "ER": "P03372",
    "TP53": "P04637",
    "KRAS": "P01116",
    "BCL_XL": "Q07817",
    "BCL2": "P10415",
    "SMARCA2": "P51531",
    "SMARCA4": "P51532",
    "STAT3": "P40763",
    "FAK": "Q05397",
    "ALK": "Q9UM73",
    "EGFR": "P00533",
    "HER2": "P04626",
    "RIPK2": "O43353",
    "IRAK4": "Q9NWZ3",
    "HDAC1": "Q13547",
    "HDAC2": "Q92769",
    "HDAC3": "O15379",
    "HDAC6": "Q9UBN7",
    "PARP1": "P09874",
    "MDM2_target": "Q00987",
    "PCAF": "Q92831",
    "TRIM24": "O15164",
    "ERRα": "P11474",
    "KRASG12C": "P01116",
    "FLT3": "P36888",
    "ABL1": "P00519",
    "AKT1": "P31749",
    "MEK1": "Q02750",
    "BRAF": "P15056",
    "SRC": "P12931",

    # E3 ligases
    "CRBN": "Q96SW2",
    "VHL": "P40337",
    "MDM2": "Q00987",
    "cIAP1": "Q13490",
    "DCAF16": "Q9NXF7",
    "DCAF15": "Q66K64",
    "DDB1": "Q16531",
    "CUL4A": "Q13619",
    "CUL2": "Q13617",
    "KEAP1": "Q14145",

    # Important for validation
    "IKZF1": "Q13422",
    "IKZF3": "Q9UKT9",
    "CK1α": "P48729",
    "GSPT1": "P15170",
    "HIF1A": "Q16665",
}


def acquire_alphafold():
    """Main acquisition function for AlphaFold structures."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    struct_dir = RAW_DIR / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AlphaFold DB Structure Acquisition")
    logger.info("=" * 60)

    # Download priority proteins
    uniprot_ids = list(set(PRIORITY_PROTEINS.values()))
    logger.info(f"Downloading structures for {len(uniprot_ids)} priority proteins...")

    results = batch_download_structures(uniprot_ids, struct_dir, delay=0.2)

    # Save results
    success_count = sum(1 for v in results.values() if v is not None)
    logger.info(f"Successfully downloaded {success_count}/{len(uniprot_ids)} structures")

    # Save mapping and log
    mapping = {name: {"uniprot": uid, "structure_path": results.get(uid)}
                for name, uid in PRIORITY_PROTEINS.items()}

    with open(RAW_DIR / "protein_structure_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    meta = {
        "total_requested": len(uniprot_ids),
        "total_downloaded": success_count,
        "failed": [uid for uid, path in results.items() if path is None],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RAW_DIR / "acquisition_log.json", "w") as f:
        json.dump(meta, f, indent=2)

    return results


if __name__ == "__main__":
    acquire_alphafold()

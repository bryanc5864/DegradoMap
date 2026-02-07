"""
Acquire DepMap 24Q4 expression and dependency data.

DepMap (https://depmap.org/) provides gene expression, dependency scores,
and other molecular characterization across ~1,800 cancer cell lines.
"""

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/depmap")

# DepMap portal file download URLs (24Q4 release)
DEPMAP_BASE = "https://depmap.org/portal/api"
DEPMAP_DOWNLOADS = "https://depmap.org/portal/download/api/downloads"

# Known direct download URLs for DepMap public datasets
DEPMAP_FILES = {
    "expression": {
        "filename": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "description": "Gene expression (log2 TPM+1) across cell lines",
    },
    "gene_effect": {
        "filename": "CRISPRGeneEffect.csv",
        "description": "CRISPR gene dependency scores",
    },
    "sample_info": {
        "filename": "Model.csv",
        "description": "Cell line metadata",
    },
}


def download_depmap_file(file_key: str):
    """Download a specific DepMap file."""
    if file_key not in DEPMAP_FILES:
        logger.warning(f"Unknown file key: {file_key}")
        return None

    info = DEPMAP_FILES[file_key]
    filename = info["filename"]
    out_path = RAW_DIR / filename

    if out_path.exists():
        logger.info(f"File already exists: {out_path}")
        return out_path

    # Try multiple download approaches
    urls_to_try = [
        f"https://ndownloader.figshare.com/files/{filename}",  # DepMap uses figshare
        f"https://depmap.org/portal/download/api/download?file_name={filename}",
        f"https://depmap.org/portal/api/download?filename={filename}",
    ]

    for url in urls_to_try:
        try:
            logger.info(f"Trying to download {filename} from {url}...")
            resp = requests.get(url, timeout=300, stream=True,
                                headers={"User-Agent": "DegradoMap-Research/1.0"})
            if resp.status_code == 200:
                total_size = int(resp.headers.get("content-length", 0))
                if total_size > 1000 or not total_size:
                    with open(out_path, "wb") as f:
                        downloaded = 0
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                    logger.info(f"Downloaded {filename} ({downloaded} bytes)")
                    return out_path
        except Exception as e:
            logger.debug(f"Download failed from {url}: {e}")
            continue

    return None


def try_depmap_api():
    """Try to access DepMap data via their API."""
    try:
        # Try to get available downloads list
        resp = requests.get(
            f"{DEPMAP_BASE}/download/files",
            timeout=30,
            headers={"Accept": "application/json"}
        )
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Found {len(data)} available files on DepMap")
            return data
    except Exception as e:
        logger.debug(f"DepMap API failed: {e}")

    return None


def create_e3_gene_list():
    """
    Create list of E3 ligase genes and related genes needed for context encoding.
    These are the specific genes whose expression matters for PROTAC degradation.
    """
    genes = {
        # E3 ligases used in PROTACs
        "e3_ligases": {
            "CRBN": "Cereblon - most common PROTAC E3",
            "VHL": "Von Hippel-Lindau - second most common",
            "MDM2": "MDM2 - emerging PROTAC E3",
            "BIRC2": "cIAP1 - emerging PROTAC E3",
            "DCAF16": "DDB1-CUL4 associated factor 16",
            "DCAF15": "DDB1-CUL4 associated factor 15",
            "KEAP1": "Kelch-like ECH-associated protein 1",
            "DDB1": "Damage-specific DNA binding protein 1 (CRBN complex)",
            "CUL4A": "Cullin-4A (CRBN complex)",
            "CUL2": "Cullin-2 (VHL complex)",
            "ELOB": "Elongin B (VHL complex)",
            "ELOC": "Elongin C (VHL complex)",
            "RBX1": "RING-box protein 1",
        },
        # Proteasome subunits
        "proteasome": [
            "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
            "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7",
            "PSMB8", "PSMB9", "PSMB10",
            "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
            "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD6", "PSMD7",
            "PSMD8", "PSMD11", "PSMD12", "PSMD13", "PSMD14",
        ],
        # Deubiquitinases (DUBs)
        "dubs": [
            "USP1", "USP2", "USP4", "USP5", "USP7", "USP8", "USP10",
            "USP11", "USP14", "USP15", "USP16", "USP19", "USP20",
            "USP22", "USP24", "USP25", "USP28", "USP29", "USP33",
            "USP36", "USP46", "USP47", "USP48",
            "UCHL1", "UCHL3", "UCHL5",
            "OTUB1", "OTUB2",
            "OTUD1", "OTUD4", "OTUD5",
            "BRCC3", "COPS5",
        ],
        # Ubiquitin pathway
        "ub_pathway": [
            "UBA1", "UBE2D1", "UBE2D2", "UBE2D3", "UBE2D4",
            "UBE2G1", "UBE2G2", "UBE2K", "UBE2L3", "UBE2N",
            "UBE2R1", "UBE2R2",
            "UBB", "UBC", "UBA52", "RPS27A",
        ],
    }

    # Flatten for expression extraction
    all_genes = []
    for category, items in genes.items():
        if isinstance(items, dict):
            all_genes.extend(items.keys())
        elif isinstance(items, list):
            all_genes.extend(items)

    return genes, all_genes


def acquire_depmap():
    """Main acquisition function for DepMap data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DepMap 24Q4 Data Acquisition")
    logger.info("=" * 60)

    # Step 1: Try downloading key files
    downloaded = {}
    for key in DEPMAP_FILES:
        logger.info(f"Attempting to download: {key}...")
        path = download_depmap_file(key)
        downloaded[key] = str(path) if path else None

    # Step 2: Try API
    logger.info("Step 2: Checking DepMap API...")
    api_data = try_depmap_api()

    # Step 3: Generate gene list
    logger.info("Step 3: Generating gene lists for context encoding...")
    gene_categories, all_genes = create_e3_gene_list()

    gene_list_path = RAW_DIR / "context_gene_list.json"
    with open(gene_list_path, "w") as f:
        json.dump(gene_categories, f, indent=2)
    logger.info(f"Saved {len(all_genes)} context-relevant genes to {gene_list_path}")

    # Save log
    meta = {
        "downloaded_files": downloaded,
        "api_available": api_data is not None,
        "context_genes_count": len(all_genes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RAW_DIR / "acquisition_log.json", "w") as f:
        json.dump(meta, f, indent=2)

    return downloaded


if __name__ == "__main__":
    acquire_depmap()

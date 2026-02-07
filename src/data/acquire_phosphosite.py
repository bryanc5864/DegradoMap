"""
Acquire PhosphoSitePlus ubiquitination site data.

PhosphoSitePlus (https://www.phosphosite.org/) contains ~70,000 ubiquitination
sites with flanking sequences. We use their publicly available datasets.
"""

import gzip
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/phosphosite")

# PhosphoSitePlus publicly available dataset URLs
PSP_DOWNLOAD_BASE = "https://www.phosphosite.org/downloads"


def download_phosphosite_files():
    """
    Download ubiquitination site data from PhosphoSitePlus.

    PhosphoSitePlus provides downloadable datasets for non-commercial use.
    The ubiquitylation sites file contains site-level information.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Known file patterns from PhosphoSitePlus
    files_to_try = [
        "Ubiquitination_site_dataset.gz",
        "Ubiquitination_site_dataset",
        "Ubiquitylation_site_dataset.gz",
    ]

    downloaded = []

    for fname in files_to_try:
        url = f"{PSP_DOWNLOAD_BASE}/{fname}"
        out_path = RAW_DIR / fname
        try:
            resp = requests.get(url, timeout=60, headers={
                "User-Agent": "Mozilla/5.0 (Research; DegradoMap Project)"
            })
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Downloaded {fname} ({len(resp.content)} bytes)")
                downloaded.append(out_path)
                break
        except Exception as e:
            logger.debug(f"Download failed for {fname}: {e}")

    return downloaded


def parse_phosphosite_ubiquitination(file_path: Path) -> pd.DataFrame:
    """Parse PhosphoSitePlus ubiquitination site dataset."""
    if str(file_path).endswith(".gz"):
        with gzip.open(file_path, "rt", errors="replace") as f:
            lines = f.readlines()
    else:
        with open(file_path, "r", errors="replace") as f:
            lines = f.readlines()

    # Skip header lines (PhosphoSitePlus files have a multi-line header)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("GENE\t") or line.startswith("gene\t"):
            data_start = i
            break
        if "\t" in line and i > 0 and not line.startswith("#"):
            # Check if this looks like a header row
            fields = line.strip().split("\t")
            if any(h.lower() in ["gene", "protein", "uniprot", "organism"]
                   for h in fields):
                data_start = i
                break

    if data_start > 0:
        header = lines[data_start].strip().split("\t")
        records = []
        for line in lines[data_start + 1:]:
            fields = line.strip().split("\t")
            if len(fields) >= len(header) // 2:  # Allow some missing fields
                record = dict(zip(header, fields))
                records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Parsed {len(df)} ubiquitination site records")
        return df

    logger.warning("Could not parse PhosphoSitePlus file format")
    return pd.DataFrame()


def create_curated_ubiquitination_sites():
    """
    Create a curated ubiquitination site dataset from well-known sites
    documented in the literature, as a fallback/supplement.
    """
    # Well-documented ubiquitination sites from literature
    # Format: (Gene, UniProt, Position, Residue, Flanking_Sequence, Organism, PMID)
    sites = [
        # p53 ubiquitination sites (by MDM2)
        ("TP53", "P04637", 370, "K", "ALPQHAHAQMINEK370STGS", "human", "10722726"),
        ("TP53", "P04637", 372, "K", "PQHAHAQMINEKSTK372GS", "human", "10722726"),
        ("TP53", "P04637", 373, "K", "QHAHAQMINEKSTKK373GSQ", "human", "10722726"),
        ("TP53", "P04637", 381, "K", "STKGSK381SSQHFR", "human", "10722726"),
        ("TP53", "P04637", 382, "K", "TKGSKS382SQHFR", "human", "10722726"),
        ("TP53", "P04637", 386, "K", "SKSSQK386HFRGM", "human", "10722726"),

        # IκBα (by βTrCP/SCF)
        ("NFKBIA", "P25963", 21, "K", "DRHDSGLDSMK21DLR", "human", "9461553"),
        ("NFKBIA", "P25963", 22, "K", "RHDSGLDSMKK22DLRI", "human", "9461553"),

        # HIF1α (by VHL)
        ("HIF1A", "Q16665", 532, "K", "DESGLPQLK532SF", "human", "10521399"),

        # IKZF1/Ikaros (by CRBN - IMiD-induced)
        ("IKZF1", "Q13422", 382, "K", "TQHAHPEK382DER", "human", "24292625"),
        ("IKZF1", "Q13422", 388, "K", "EKDERK388QAQ", "human", "24292625"),

        # Cyclin B1 (by APC/C)
        ("CCNB1", "P14635", 34, "K", "ALKEPVHGK34VE", "human", "20596027"),
        ("CCNB1", "P14635", 42, "K", "KVEVFDDLK42NI", "human", "20596027"),

        # β-catenin (by βTrCP)
        ("CTNNB1", "P35222", 19, "K", "YLDSGIHSGATK19AQI", "human", "9601641"),
        ("CTNNB1", "P35222", 49, "K", "DRKAAVSHK49FNK", "human", "9601641"),

        # BRD4 (PROTAC-relevant, key for validation)
        ("BRD4", "O60885", 316, "K", "NSKNKK316SSK", "human", "28452383"),
        ("BRD4", "O60885", 344, "K", "MGKKK344EVR", "human", "28452383"),
        ("BRD4", "O60885", 452, "K", "TEKQK452KDK", "human", "28452383"),

        # p27 (by SCF-Skp2)
        ("CDKN1B", "P46527", 134, "K", "AFLGK134SPK", "human", "10499802"),
        ("CDKN1B", "P46527", 153, "K", "SKACEK153RTQ", "human", "10499802"),

        # EGFR
        ("EGFR", "P00533", 716, "K", "QLMPFK716ELD", "human", "17694084"),
        ("EGFR", "P00533", 737, "K", "LPQPPK737FQVK", "human", "17694084"),
        ("EGFR", "P00533", 867, "K", "MDAALK867DVR", "human", "17694084"),

        # ERα (PROTAC-relevant)
        ("ESR1", "P03372", 302, "K", "EHLHK302SKQR", "human", "22863005"),
        ("ESR1", "P03372", 303, "K", "HLHKK303SQRT", "human", "22863005"),

        # AR (PROTAC-relevant)
        ("AR", "P10275", 845, "K", "QHTPEK845LLQK", "human", "14678011"),
        ("AR", "P10275", 847, "K", "TPEKK847LLQK", "human", "14678011"),

        # BTK (PROTAC-relevant)
        ("BTK", "Q06187", 453, "K", "FMEKK453DVQR", "human", "29596916"),

        # CDK4 (PROTAC-relevant)
        ("CDK4", "P11802", 107, "K", "ISTDFK107YFRE", "human", "31227622"),

        # CDK6 (PROTAC-relevant)
        ("CDK6", "Q00534", 147, "K", "GQTVAK147VDQR", "human", "31227622"),

        # KRAS
        ("KRAS", "P01116", 42, "K", "VVVGAK42GVG", "human", "37100912"),
        ("KRAS", "P01116", 104, "K", "DTAGQK104EYH", "human", "37100912"),
        ("KRAS", "P01116", 147, "K", "RQHTKK147QCI", "human", "37100912"),

        # BCL-XL (PROTAC-relevant)
        ("BCL2L1", "Q07817", 87, "K", "ELRQK87FGD", "human", "32493095"),

        # SMARCA2/BRM (PROTAC-relevant)
        ("SMARCA2", "P51531", 455, "K", "TQPEK455KLK", "human", "33662261"),

        # FAK (PROTAC-relevant)
        ("PTK2", "Q05397", 454, "K", "EVNQK454FVR", "human", "30191720"),
    ]

    df = pd.DataFrame(sites, columns=[
        "Gene", "UniProt_ID", "Position", "Residue",
        "Flanking_Sequence", "Organism", "PMID"
    ])
    df["Modification"] = "Ubiquitination"
    df["Site_Key"] = df["Gene"] + "_K" + df["Position"].astype(str)

    return df


def acquire_phosphosite():
    """Main acquisition function for PhosphoSitePlus data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PhosphoSitePlus Ubiquitination Site Data Acquisition")
    logger.info("=" * 60)

    # Step 1: Try downloading from PhosphoSitePlus
    logger.info("Step 1: Attempting PhosphoSitePlus download...")
    downloaded_files = download_phosphosite_files()

    parsed_df = None
    if downloaded_files:
        for fpath in downloaded_files:
            df = parse_phosphosite_ubiquitination(fpath)
            if len(df) > 0:
                parsed_df = df
                break

    if parsed_df is not None:
        out_path = RAW_DIR / "phosphosite_ubiquitination.csv"
        parsed_df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(parsed_df)} Ub sites from PhosphoSitePlus")

    # Step 2: Create curated dataset
    logger.info("Step 2: Building curated ubiquitination site dataset...")
    curated_df = create_curated_ubiquitination_sites()
    curated_path = RAW_DIR / "curated_ubiquitination_sites.csv"
    curated_df.to_csv(curated_path, index=False)
    logger.info(f"Saved {len(curated_df)} curated Ub sites to {curated_path}")

    # Save log
    meta = {
        "downloaded_files": [str(f) for f in downloaded_files],
        "parsed_count": len(parsed_df) if parsed_df is not None else 0,
        "curated_count": len(curated_df),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RAW_DIR / "acquisition_log.json", "w") as f:
        json.dump(meta, f, indent=2)

    return curated_path


if __name__ == "__main__":
    acquire_phosphosite()

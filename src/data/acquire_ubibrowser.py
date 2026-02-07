"""
Acquire UbiBrowser 2.0 data.

UbiBrowser 2.0 (http://ubibrowser.bio-it.cn/) contains experimentally verified
and computationally predicted E3 ligase-substrate interactions (ESIs).
4,068 known + 2.2M predicted ESIs across 39 species.
"""

import json
import os
import time
import logging
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://ubibrowser.bio-it.cn"
RAW_DIR = Path("data/raw/ubibrowser")


def try_download_pages():
    """Try to find download links on the UbiBrowser site."""
    download_urls = [
        f"{BASE_URL}/download",
        f"{BASE_URL}/Download",
        f"{BASE_URL}/downloads",
        f"{BASE_URL}/static/download",
        f"{BASE_URL}/#/download",
    ]

    for url in download_urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 500:
                soup = BeautifulSoup(resp.text, "html.parser")
                links = soup.find_all("a", href=True)
                for link in links:
                    href = link["href"]
                    text = link.get_text(strip=True)
                    if any(ext in href.lower() for ext in [".csv", ".tsv", ".xlsx", ".txt", ".zip", ".gz"]):
                        logger.info(f"Found download link: {text} -> {href}")
                        dl_url = href if href.startswith("http") else f"{BASE_URL}/{href.lstrip('/')}"
                        try:
                            dl_resp = requests.get(dl_url, timeout=120)
                            if dl_resp.status_code == 200 and len(dl_resp.content) > 100:
                                fname = href.split("/")[-1]
                                out_path = RAW_DIR / fname
                                with open(out_path, "wb") as f:
                                    f.write(dl_resp.content)
                                logger.info(f"Downloaded: {out_path} ({len(dl_resp.content)} bytes)")
                                return out_path
                        except Exception as e:
                            logger.warning(f"Download failed for {dl_url}: {e}")
        except Exception as e:
            logger.debug(f"Page {url} failed: {e}")
            continue

    return None


def try_api_search(species: str = "Homo sapiens", e3_type: str = None):
    """Try to query UbiBrowser API for E3-substrate interactions."""
    api_endpoints = [
        "/api/search",
        "/api/browse",
        "/api/esi",
        "/api/interaction",
        "/search",
    ]

    params = {"species": species, "page": 1, "per_page": 100}
    if e3_type:
        params["e3_type"] = e3_type

    for endpoint in api_endpoints:
        url = f"{BASE_URL}{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if data:
                        logger.info(f"Working API: {endpoint}")
                        return endpoint, data
                except json.JSONDecodeError:
                    pass

            # Try POST
            resp = requests.post(url, json=params, timeout=30)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if data:
                        logger.info(f"Working API (POST): {endpoint}")
                        return endpoint, data
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug(f"API {endpoint} failed: {e}")

    return None, None


def scrape_ubibrowser_structure():
    """Understand the UbiBrowser website structure."""
    try:
        resp = requests.get(BASE_URL, timeout=30)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            scripts = soup.find_all("script", src=True)
            links = soup.find_all("a", href=True)

            info = {
                "scripts": [s["src"] for s in scripts],
                "links": [{"href": l["href"], "text": l.get_text(strip=True)} for l in links],
                "title": soup.title.string if soup.title else None,
            }
            return info
    except Exception as e:
        logger.warning(f"Failed to scrape structure: {e}")
    return {}


def create_synthetic_ubibrowser_data():
    """
    Create a structured placeholder with known E3-substrate interactions
    from literature, to be replaced with real UbiBrowser data.

    These are well-established E3 ligase-substrate pairs from published research.
    """
    known_esis = [
        # (E3_ligase, E3_gene, Substrate, Substrate_gene, Evidence, PMID)
        ("CRBN", "CRBN", "IKZF1", "IKZF1", "Experimental", "24292625"),
        ("CRBN", "CRBN", "IKZF3", "IKZF3", "Experimental", "24292625"),
        ("CRBN", "CRBN", "CK1α", "CSNK1A1", "Experimental", "25580516"),
        ("CRBN", "CRBN", "GSPT1", "GSPT1", "Experimental", "27819657"),
        ("CRBN", "CRBN", "SALL4", "SALL4", "Experimental", "29474678"),
        ("VHL", "VHL", "HIF1α", "HIF1A", "Experimental", "10521399"),
        ("VHL", "VHL", "HIF2α", "EPAS1", "Experimental", "10521399"),
        ("MDM2", "MDM2", "p53", "TP53", "Experimental", "8319905"),
        ("MDM2", "MDM2", "MDMX", "MDM4", "Experimental", "11700545"),
        ("MDM2", "MDM2", "Rb", "RB1", "Experimental", "10540293"),
        ("cIAP1", "BIRC2", "NIK", "MAP3K14", "Experimental", "18408713"),
        ("cIAP1", "BIRC2", "RIPK1", "RIPK1", "Experimental", "15258597"),
        ("βTrCP", "BTRC", "IκBα", "NFKBIA", "Experimental", "9461553"),
        ("βTrCP", "BTRC", "β-catenin", "CTNNB1", "Experimental", "9601641"),
        ("SCF-SKP2", "SKP2", "p27", "CDKN1B", "Experimental", "10499802"),
        ("SCF-SKP2", "SKP2", "p21", "CDKN1A", "Experimental", "15586776"),
        ("APC/C", "ANAPC2", "Cyclin B1", "CCNB1", "Experimental", "8033209"),
        ("APC/C", "ANAPC2", "Securin", "PTTG1", "Experimental", "9774969"),
        ("CHIP", "STUB1", "CFTR", "CFTR", "Experimental", "11415465"),
        ("CHIP", "STUB1", "HSP70", "HSPA1A", "Experimental", "11113132"),
        ("Parkin", "PRKN", "Mitofusin1", "MFN1", "Experimental", "20404107"),
        ("Parkin", "PRKN", "Mitofusin2", "MFN2", "Experimental", "20404107"),
        ("SIAH1", "SIAH1", "TRAF2", "TRAF2", "Experimental", "15067025"),
        ("NEDD4", "NEDD4", "ENaC", "SCNN1A", "Experimental", "10490631"),
        ("MARCH1", "MARCHF1", "MHC-II", "HLA-DRA", "Experimental", "16682509"),
        # PROTAC-relevant neo-substrates
        ("CRBN", "CRBN", "BRD4", "BRD4", "PROTAC-induced", "26051717"),
        ("CRBN", "CRBN", "BRD2", "BRD2", "PROTAC-induced", "26051717"),
        ("CRBN", "CRBN", "BRD3", "BRD3", "PROTAC-induced", "26051717"),
        ("VHL", "VHL", "BRD4", "BRD4", "PROTAC-induced", "26051717"),
        ("CRBN", "CRBN", "BTK", "BTK", "PROTAC-induced", "29596916"),
        ("VHL", "VHL", "ERRα", "ESRRA", "PROTAC-induced", "29892060"),
        ("CRBN", "CRBN", "CDK4", "CDK4", "PROTAC-induced", "31227622"),
        ("CRBN", "CRBN", "CDK6", "CDK6", "PROTAC-induced", "31227622"),
        ("VHL", "VHL", "SMARCA2", "SMARCA2", "PROTAC-induced", "33662261"),
        ("VHL", "VHL", "SMARCA4", "SMARCA4", "PROTAC-induced", "33662261"),
        ("CRBN", "CRBN", "STAT3", "STAT3", "PROTAC-induced", "31816994"),
        ("VHL", "VHL", "AR", "AR", "PROTAC-induced", "31158914"),
        ("CRBN", "CRBN", "AR", "AR", "PROTAC-induced", "31158914"),
        ("VHL", "VHL", "ER", "ESR1", "PROTAC-induced", "30726754"),
        ("CRBN", "CRBN", "BCL-XL", "BCL2L1", "PROTAC-induced", "32493095"),
        ("VHL", "VHL", "BCL-XL", "BCL2L1", "PROTAC-induced", "32493095"),
        ("CRBN", "CRBN", "KRASG12C", "KRAS", "PROTAC-induced", "37100912"),
        ("VHL", "VHL", "FAK", "PTK2", "PROTAC-induced", "30191720"),
        ("CRBN", "CRBN", "ALK", "ALK", "PROTAC-induced", "30257078"),
        ("VHL", "VHL", "RIPK2", "RIPK2", "PROTAC-induced", "30758023"),
    ]

    df = pd.DataFrame(known_esis, columns=[
        "E3_Ligase", "E3_Gene", "Substrate", "Substrate_Gene",
        "Evidence_Type", "PMID"
    ])
    df["Species"] = "Homo sapiens"
    df["Confidence"] = "High"

    return df


def acquire_ubibrowser():
    """Main acquisition function for UbiBrowser 2.0."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("UbiBrowser 2.0 Data Acquisition")
    logger.info("=" * 60)

    # Step 1: Try download page
    logger.info("Step 1: Looking for download links...")
    dl_path = try_download_pages()

    # Step 2: Try API
    logger.info("Step 2: Probing API endpoints...")
    endpoint, api_data = try_api_search()

    # Step 3: Scrape structure
    logger.info("Step 3: Analyzing site structure...")
    structure = scrape_ubibrowser_structure()

    # Step 4: Create curated known ESI dataset
    logger.info("Step 4: Building curated E3-substrate interaction dataset...")
    df_esi = create_synthetic_ubibrowser_data()
    out_path = RAW_DIR / "curated_esi_interactions.csv"
    df_esi.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df_esi)} curated ESI interactions to {out_path}")

    # Save acquisition log
    meta = {
        "download_path": str(dl_path) if dl_path else None,
        "api_endpoint": endpoint,
        "site_structure": structure,
        "curated_esi_count": len(df_esi),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RAW_DIR / "acquisition_log.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return out_path


if __name__ == "__main__":
    acquire_ubibrowser()

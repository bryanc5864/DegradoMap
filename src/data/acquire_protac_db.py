"""
Acquire PROTAC-DB 3.0 data.

PROTAC-DB (http://cadd.zju.edu.cn/protacdb/) contains ~6,111 PROTACs with
DC50/Dmax measurements, target proteins, E3 ligases, and cell lines.
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

BASE_URL = "http://cadd.zju.edu.cn/protacdb"
API_URL = f"{BASE_URL}/api"

RAW_DIR = Path("data/raw/protac_db")


def fetch_protac_list(page: int = 1, per_page: int = 100) -> dict:
    """Fetch a page of PROTACs from the API."""
    url = f"{BASE_URL}/statics/data"
    params = {"page": page, "per_page": per_page}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"API request failed for page {page}: {e}")
        return {}


def scrape_protac_detail(protac_id: str) -> dict:
    """Scrape detailed information for a single PROTAC entry."""
    url = f"{BASE_URL}/molecule/{protac_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        data = {"protac_id": protac_id}

        # Extract table data
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    data[key] = value

        return data
    except Exception as e:
        logger.warning(f"Failed to scrape PROTAC {protac_id}: {e}")
        return {"protac_id": protac_id, "error": str(e)}


def try_download_bulk():
    """Try to download bulk data if available."""
    bulk_urls = [
        f"{BASE_URL}/download",
        f"{BASE_URL}/static/data/protac_db.csv",
        f"{BASE_URL}/static/data/protac_db.xlsx",
        f"{BASE_URL}/api/download",
        f"{BASE_URL}/statics/download",
    ]

    for url in bulk_urls:
        try:
            resp = requests.get(url, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "csv" in content_type or "excel" in content_type or "octet" in content_type:
                    ext = "csv" if "csv" in content_type else "xlsx"
                    out_path = RAW_DIR / f"protac_db_bulk.{ext}"
                    with open(out_path, "wb") as f:
                        f.write(resp.content)
                    logger.info(f"Downloaded bulk data to {out_path}")
                    return out_path
                elif len(resp.content) > 10000:
                    # Might be HTML page with download links
                    soup = BeautifulSoup(resp.text, "html.parser")
                    links = soup.find_all("a", href=True)
                    for link in links:
                        href = link["href"]
                        if any(ext in href.lower() for ext in [".csv", ".xlsx", ".tsv", ".zip"]):
                            dl_url = href if href.startswith("http") else f"{BASE_URL}/{href.lstrip('/')}"
                            dl_resp = requests.get(dl_url, timeout=60)
                            if dl_resp.status_code == 200:
                                fname = href.split("/")[-1]
                                out_path = RAW_DIR / fname
                                with open(out_path, "wb") as f:
                                    f.write(dl_resp.content)
                                logger.info(f"Downloaded {fname}")
                                return out_path
        except Exception as e:
            logger.debug(f"Bulk download attempt failed for {url}: {e}")
            continue

    return None


def try_api_endpoints():
    """Try various API endpoints to get structured data."""
    endpoints = [
        "/api/protacs",
        "/api/molecules",
        "/statics/data",
        "/api/compound/list",
        "/api/v1/protacs",
    ]

    for endpoint in endpoints:
        url = f"{BASE_URL.rstrip('/')}{endpoint}"
        try:
            resp = requests.get(url, timeout=30, params={"page": 1, "per_page": 10})
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, (list, dict)):
                        logger.info(f"Found working API endpoint: {endpoint}")
                        logger.info(f"Response type: {type(data)}, keys/len: {data.keys() if isinstance(data, dict) else len(data)}")
                        return endpoint, data
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug(f"API endpoint {endpoint} failed: {e}")
            continue

    return None, None


def scrape_main_page():
    """Scrape the main PROTAC-DB page structure to understand navigation."""
    try:
        resp = requests.get(BASE_URL, timeout=30)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find all links
            links = soup.find_all("a", href=True)
            relevant_links = []
            for link in links:
                href = link["href"]
                text = link.get_text(strip=True)
                if any(kw in href.lower() or kw in text.lower()
                       for kw in ["download", "data", "browse", "search", "molecule", "protac"]):
                    relevant_links.append({"href": href, "text": text})

            return relevant_links
    except Exception as e:
        logger.warning(f"Failed to scrape main page: {e}")
    return []


def acquire_protac_db():
    """Main acquisition function for PROTAC-DB 3.0."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PROTAC-DB 3.0 Data Acquisition")
    logger.info("=" * 60)

    # Step 1: Try bulk download
    logger.info("Step 1: Attempting bulk download...")
    bulk_path = try_download_bulk()
    if bulk_path:
        logger.info(f"Bulk download successful: {bulk_path}")
        return bulk_path

    # Step 2: Try API endpoints
    logger.info("Step 2: Probing API endpoints...")
    endpoint, sample_data = try_api_endpoints()
    if endpoint:
        logger.info(f"Working API found: {endpoint}")
        # Try to paginate through all data
        all_data = []
        page = 1
        while True:
            url = f"{BASE_URL.rstrip('/')}{endpoint}"
            try:
                resp = requests.get(url, timeout=30, params={"page": page, "per_page": 100})
                if resp.status_code != 200:
                    break
                data = resp.json()
                records = data if isinstance(data, list) else data.get("data", data.get("results", []))
                if not records:
                    break
                all_data.extend(records)
                logger.info(f"Page {page}: got {len(records)} records (total: {len(all_data)})")
                page += 1
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"API pagination failed at page {page}: {e}")
                break

        if all_data:
            df = pd.DataFrame(all_data)
            out_path = RAW_DIR / "protac_db_api.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Saved {len(all_data)} records to {out_path}")
            return out_path

    # Step 3: Scrape main page for structure
    logger.info("Step 3: Scraping main page for navigation structure...")
    links = scrape_main_page()
    logger.info(f"Found {len(links)} relevant links:")
    for link in links[:20]:
        logger.info(f"  {link['text']}: {link['href']}")

    # Save what we found
    meta = {
        "bulk_download_attempted": True,
        "bulk_path": str(bulk_path) if bulk_path else None,
        "api_endpoint": endpoint,
        "navigation_links": links,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RAW_DIR / "acquisition_log.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("PROTAC-DB acquisition attempt completed.")
    logger.info("Note: If automated acquisition fails, manual download may be needed.")
    logger.info("Visit: http://cadd.zju.edu.cn/protacdb/")

    return None


if __name__ == "__main__":
    acquire_protac_db()

"""
Master data acquisition script for DegradoMap.

Orchestrates downloading/scraping all required datasets:
1. PROTAC-DB 3.0 - Degradation labels
2. UbiBrowser 2.0 - E3-substrate interactions
3. PhosphoSitePlus - Ubiquitination sites
4. AlphaFold DB - Protein structures
5. DepMap 24Q4 - Cell line expression
6. ProteomicsDB - Protein half-lives
"""

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/raw/acquisition.log"),
    ]
)
logger = logging.getLogger("DegradoMap-Acquire")


def main():
    """Run all data acquisition steps."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    results = {}
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("DegradoMap Data Acquisition Pipeline")
    logger.info("=" * 80)

    # 1. PROTAC-DB
    logger.info("\n" + "=" * 40)
    logger.info("[1/6] PROTAC-DB 3.0")
    logger.info("=" * 40)
    try:
        from src.data.acquire_protac_db import acquire_protac_db
        results["protac_db"] = {"status": "success", "path": str(acquire_protac_db())}
    except Exception as e:
        logger.error(f"PROTAC-DB acquisition failed: {e}")
        results["protac_db"] = {"status": "failed", "error": str(e)}

    # 2. UbiBrowser
    logger.info("\n" + "=" * 40)
    logger.info("[2/6] UbiBrowser 2.0")
    logger.info("=" * 40)
    try:
        from src.data.acquire_ubibrowser import acquire_ubibrowser
        results["ubibrowser"] = {"status": "success", "path": str(acquire_ubibrowser())}
    except Exception as e:
        logger.error(f"UbiBrowser acquisition failed: {e}")
        results["ubibrowser"] = {"status": "failed", "error": str(e)}

    # 3. PhosphoSitePlus
    logger.info("\n" + "=" * 40)
    logger.info("[3/6] PhosphoSitePlus")
    logger.info("=" * 40)
    try:
        from src.data.acquire_phosphosite import acquire_phosphosite
        results["phosphosite"] = {"status": "success", "path": str(acquire_phosphosite())}
    except Exception as e:
        logger.error(f"PhosphoSitePlus acquisition failed: {e}")
        results["phosphosite"] = {"status": "failed", "error": str(e)}

    # 4. AlphaFold DB
    logger.info("\n" + "=" * 40)
    logger.info("[4/6] AlphaFold DB")
    logger.info("=" * 40)
    try:
        from src.data.acquire_alphafold import acquire_alphafold
        af_results = acquire_alphafold()
        success_count = sum(1 for v in af_results.values() if v is not None)
        results["alphafold"] = {"status": "success", "structures_downloaded": success_count}
    except Exception as e:
        logger.error(f"AlphaFold acquisition failed: {e}")
        results["alphafold"] = {"status": "failed", "error": str(e)}

    # 5. DepMap
    logger.info("\n" + "=" * 40)
    logger.info("[5/6] DepMap 24Q4")
    logger.info("=" * 40)
    try:
        from src.data.acquire_depmap import acquire_depmap
        results["depmap"] = {"status": "success", "files": acquire_depmap()}
    except Exception as e:
        logger.error(f"DepMap acquisition failed: {e}")
        results["depmap"] = {"status": "failed", "error": str(e)}

    # 6. ProteomicsDB
    logger.info("\n" + "=" * 40)
    logger.info("[6/6] ProteomicsDB")
    logger.info("=" * 40)
    try:
        from src.data.acquire_proteomicsdb import acquire_proteomicsdb
        results["proteomicsdb"] = {"status": "success", "path": str(acquire_proteomicsdb())}
    except Exception as e:
        logger.error(f"ProteomicsDB acquisition failed: {e}")
        results["proteomicsdb"] = {"status": "failed", "error": str(e)}

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("ACQUISITION SUMMARY")
    logger.info("=" * 80)

    for source, result in results.items():
        status = result.get("status", "unknown")
        logger.info(f"  {source:20s}: {status}")

    logger.info(f"\nTotal time: {elapsed:.1f} seconds")

    # Save summary
    results["total_time_seconds"] = elapsed
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open("data/raw/acquisition_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()

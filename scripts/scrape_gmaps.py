"""
scripts/scrape_gmaps.py
─────────────────────────────────────────────────────────────────
Standalone Google Maps Places scraper for Greater Taipei restaurants.

Responsibilities
----------------
1. Grid-scan the Greater Taipei area for restaurants by keyword and
   collect unique Place IDs, persisting them incrementally to a text file.
2. For each new Place ID, fetch up to 5 reviews + star ratings via the
   Places Detail API and append them to the training JSONL.

Usage
-----
    export GMAPS_API_KEY="your_key_here"
    python scripts/scrape_gmaps.py \\
        --ids-out  data/place_ids.txt \\
        --jsonl-out data/Train_data1.jsonl

    # Skip the grid scan if place_ids.txt already exists:
    python scripts/scrape_gmaps.py --skip-scan

Dependencies
------------
    pip install googlemaps pandas
"""

import argparse
import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Optional, Set

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from googlemaps import Client as GMapsClient
from googlemaps.exceptions import ApiError

from src.logger import get_logger

logger = get_logger(__name__, log_file="logs/scrape_gmaps.log")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# API throttle parameters
REQUEST_INTERVAL: float = 0.6   # seconds between calls (≈ 100 RPM)
MAX_RETRIES: int        = 5     # per-request retry limit

# Grid origin: 樹林 SW corner (N, E = 24.971421, 121.400178)
GRID_ORIGIN_LAT: float = 24.971421
GRID_ORIGIN_LNG: float = 121.400178

# Grid dimensions (range(0, 14) rows, range(0, 23) cols)
GRID_ROWS: int  = 14
GRID_COLS: int  = 23

# Step sizes (N += 0.009, E += 0.0098)
LAT_STEP: float = 0.009
LNG_STEP: float = 0.0098

# Search radius in metres (radius=500)
SEARCH_RADIUS: int = 500

# Restaurant keywords (Types list, verbatim)
RESTAURANT_TYPES = [
    "Cold noodle restaurant",           "Mandarin restaurant",
    "Hot pot restaurant",               "Dim sum restaurant",
    "Sichuan restaurant",               "Barbecue restaurant",
    "Seafood restaurant",               "Chinese restaurant",
    "Noodle shop",                      "Breakfast restaurant",
    "Porridge restaurant",              "Deli",
    "Restaurant",                       "American restaurant",
    "Syokudo and Teishoku restaurant",  "Japanese restaurant",
    "Yakitori restaurant",              "Teppanyaki restaurant",
    "Indian restaurant",                "Taiwanese restaurant",
    "Uyghur cuisine restaurant",        "Snack bar",
    "Spanish restaurant",               "Chinese noodle restaurant",
    "Vietnamese restaurant",            "Cantonese restaurant",
    "Asian restaurant",                 "Sushi restaurant",
    "Fine dining restaurant",           "Hamburger restaurant",
    "Pizza restaurant",                 "Dumpling restaurant",
    "Steamed bun shop",                 "Italian restaurant",
    "Sukiyaki restaurant",              "Shabu-shabu restaurant",
    "Jiangsu restaurant",               "Pie shop",
    "Bistro",                           "Hakka restaurant",
    "Chop bar",                         "Malaysian restaurant",
    "Korean restaurant",                "Beijing restaurant",
    "Zhejiang restaurant",              "Juice shop",
    "Shanghainese restaurant",          "Pastry shop",
    "Pho restaurant",                   "Box lunch supplier",
    "Chicken restaurant",               "Hunan restaurant",
    "Unagi restaurant",                 "Korean barbecue restaurant",
    "Mongolian barbecue restaurant",    "Thai restaurant",
    "Takeout restaurant",               "Fish restaurant",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_coord(val: float) -> str:
    """
    Format a float coordinate to 8 decimal places.
    """
    return format(Decimal.from_float(val), ".8")


def places_nearby_with_retry(
    gmaps: GMapsClient,
    location,
    radius: int,
    keyword: str,
    language: str = "zh-Hant",
) -> Optional[dict]:
    """
    Call gmaps.places_nearby with exponential back-off on transient errors.

    Parameters
    ----------
    gmaps : GMapsClient
    location : tuple
        (lat_str, lng_str)
    radius : int
        Search radius in metres.
    keyword : str
        Restaurant category keyword.
    language : str
        Response language (default: 'zh-Hant').

    Returns
    -------
    dict | None
        API response dict, or None if all retries are exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = gmaps.places_nearby(
                location=location,
                radius=radius,
                keyword=keyword,
                language=language,
            )
            time.sleep(REQUEST_INTERVAL)   # rate-limit guard
            return result
        except ApiError as exc:
            logger.warning(
                "API error attempt %d/%d (%s @ %s): %s",
                attempt, MAX_RETRIES, keyword, location, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_INTERVAL * attempt)  # exponential back-off

    logger.error("Max retries exceeded — keyword=%s location=%s", keyword, location)
    return None


def load_existing_ids(
    csv_path: Optional[Path],
    ids_file: Path,
) -> Set[str]:
    """
    Collect all Place IDs already scraped to avoid duplicates.
    """
    existing: Set[str] = set()

    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        existing.update(df["Place_ID"].dropna().astype(str).tolist())
        logger.info("Loaded %d existing IDs from CSV %s", len(existing), csv_path)

    if ids_file.exists():
        with ids_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                pid = line.strip()
                if pid:
                    existing.add(pid)
        logger.info("Loaded %d existing IDs from %s", len(existing), ids_file)

    return existing


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Grid scan → collect Place IDs
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_scan(
    gmaps: GMapsClient,
    ids_out: Path,
    existing_ids: Set[str],
) -> Set[str]:
    """
    Scan a rectangular grid over Greater Taipei and append new Place IDs.

    Parameters
    ----------
    gmaps : GMapsClient
    ids_out : Path
        File to append new Place IDs to.
    existing_ids : Set[str]
        Already-known Place IDs (will not be written again).

    Returns
    -------
    Set[str]
        Updated set of known IDs (existing ∪ newly found).
    """
    ids_out.parent.mkdir(parents=True, exist_ok=True)
    new_ids: Set[str] = set()

    with ids_out.open("a", encoding="utf-8") as fh:
        for keyword in RESTAURANT_TYPES:

            lat = GRID_ORIGIN_LAT   # ← INSIDE keyword loop (was outside before)

            for _row in range(GRID_ROWS):          # range(0, 14) 
                lat += LAT_STEP                    # N += 0.009  

                lng = GRID_ORIGIN_LNG              # E_start reset per row 

                for _col in range(GRID_COLS):      # range(0, 23) 
                    lng += LNG_STEP                # E += 0.0098 

                    location = (
                        _fmt_coord(lat),
                        _fmt_coord(lng),
                    )

                    result = places_nearby_with_retry(
                        gmaps, location, SEARCH_RADIUS, keyword
                    )
                    if result is None:
                        continue

                    for place in result.get("results", []):
                        pid = place["place_id"]
                        if pid not in existing_ids and pid not in new_ids:
                            new_ids.add(pid)
                            fh.write(pid + "\n")
                            fh.flush()   # crash-safe

    logger.info("Grid scan complete — %d new Place IDs found.", len(new_ids))
    return existing_ids | new_ids


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Fetch reviews for each Place ID
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reviews_for_ids(
    gmaps: GMapsClient,
    ids_file: Path,
    jsonl_out: Path,
) -> None:
    """
    For each Place ID in ids_file, fetch up to 5 reviews and append them.

    The JSONL output uses the same two-line format:

        {"Reviews": [...]}
        {"Ratings": [...]}

    Parameters
    ----------
    gmaps : GMapsClient
    ids_file : Path
        Text file with one Place ID per line.
    jsonl_out : Path
        JSONL file to update with new (review, rating) pairs.
    """
    # ── Load existing JSONL data (if any) to extend it ────────────────────────
    reviews = []
    ratings = []
    if jsonl_out.exists():
        with jsonl_out.open("r", encoding="utf-8") as fh:
            lines = [json.loads(ln) for ln in fh if ln.strip()]
        if len(lines) >= 2:
            reviews = lines[0].get("Reviews", [])
            ratings = lines[1].get("Ratings", [])

    # ── Read Place IDs ────────────────────────────────────────────────────────
    with ids_file.open("r", encoding="utf-8") as fh:
        place_ids = [stripped for ln in fh if (stripped := ln.strip())]

    new_count = 0
    for pid in place_ids:
        if not isinstance(pid, str) or not pid:
            continue
        try:
            results = gmaps.place(pid.strip(), language="zh-Hant")
            status  = results.get("status")
            if status == "OK":
                all_reviews = results["result"].get("reviews", [])
                for review in all_reviews:
                    text   = review.get("text")
                    rating = review.get("rating")
                    if text and rating:
                        reviews.append(text)
                        ratings.append(int(rating))
                        new_count += 1
        except Exception as exc:        # noqa: BLE001
            print(f"Error fetching place_id {pid}: {exc}")   
        time.sleep(0.1)                                       

    logger.info("Fetched %d new (review, rating) pairs.", new_count)

    # ── Persist  ───────────────────────────────────────────────
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"Reviews": reviews}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"Ratings": ratings}, ensure_ascii=False) + "\n")

    logger.info("Saved %d total samples to %s", len(reviews), jsonl_out)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Google Maps restaurant reviews for Greater Taipei."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GMAPS_API_KEY", ""),
        help="Google Maps API key.  Prefer exporting GMAPS_API_KEY env var.",
    )
    parser.add_argument(
        "--ids-out",
        default="data/place_ids.txt",
        help="File to accumulate Place IDs  (default: data/place_ids.txt).",
    )
    parser.add_argument(
        "--jsonl-out",
        default="data/Train_data1.jsonl",
        help="JSONL file to store (review, rating) pairs.",
    )
    parser.add_argument(
        "--existing-csv",
        default=None,
        help="Optional legacy CSV with a Place_ID column for de-duplication.",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip Phase 1 grid scan; go straight to review fetching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        logger.error(
            "No API key.  Set --api-key or export GMAPS_API_KEY."
        )
        sys.exit(1)

    gmaps     = GMapsClient(key=args.api_key)
    ids_out   = Path(args.ids_out)
    jsonl_out = Path(args.jsonl_out)
    csv_path  = Path(args.existing_csv) if args.existing_csv else None

    existing_ids = load_existing_ids(csv_path, ids_out)

    if not args.skip_scan:
        logger.info("Starting grid scan …")
        existing_ids = run_grid_scan(gmaps, ids_out, existing_ids)

    logger.info("Starting review fetch …")
    fetch_reviews_for_ids(
        gmaps=gmaps,
        ids_file=ids_out,
        jsonl_out=jsonl_out,
    )

    logger.info("Scraping pipeline complete.")


if __name__ == "__main__":
    main()

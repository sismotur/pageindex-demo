#!/usr/bin/env python3
"""
extract_ubeda.py — Fetch POIs for any tourist destination from the Inventrip API.

Calls GET /v120/pois with the specified tourist_destination and saves the raw
response array to data/{destination}_pois_raw.json.

Usage:
    .venv/bin/python scripts/extract_ubeda.py
    .venv/bin/python scripts/extract_ubeda.py --destination fayon --lang es

Environment variables (loaded from .env):
    INVENTRIP_API_BASE_URL  Base URL of the Inventrip API
    INVENTRIP_API_KEY       API key passed as query param ?api_key=...
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).parent.parent
DEFAULT_DESTINATION = "ubeda"
DEFAULT_LANGUAGE    = "en"
TIMEOUT_SECONDS     = 60


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config() -> tuple[str, str]:
    """Load and validate required environment variables.

    Returns:
        (base_url, api_key) strings.

    Raises:
        SystemExit if any required variable is missing or still at its
        placeholder value.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    base_url = os.getenv("INVENTRIP_API_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("INVENTRIP_API_KEY", "").strip()

    errors = []
    if not base_url:
        errors.append("INVENTRIP_API_BASE_URL is not set")
    if not api_key or api_key == "your_api_key_here":
        errors.append("INVENTRIP_API_KEY is not set (edit .env and fill in the real key)")

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)

    return base_url, api_key


def build_request_params(api_key: str, destination: str, lang: str) -> dict:
    """Return the query parameters for the POI endpoint."""
    return {
        "tourist_destination": destination,
        "language": lang,
        "strip_nulls": "true",
        "api_key": api_key,
    }


def fetch_pois(base_url: str, params: dict, destination: str) -> list:
    """Fetch POIs from the /v120/pois endpoint.

    Returns:
        List of POI dicts from the API response.

    Raises:
        SystemExit on HTTP or connection errors.
    """
    url = f"{base_url}/v120/pois"
    print(f"[INFO] GET {url}")
    print(f"[INFO] tourist_destination={destination}  language={params.get('language', '?')}")

    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        print(f"[ERROR] Could not connect to {base_url}: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out after {TIMEOUT_SECONDS}s", file=sys.stderr)
        sys.exit(1)

    # Pre-condition: HTTP response received
    if response.status_code == 401:
        print("[ERROR] 401 Unauthorized — check INVENTRIP_API_KEY", file=sys.stderr)
        sys.exit(1)
    if response.status_code == 404:
        print(f"[ERROR] 404 Not Found — tourist_destination '{destination}' not found",
              file=sys.stderr)
        sys.exit(1)
    if response.status_code != 200:
        print(f"[ERROR] Unexpected status {response.status_code}: {response.text[:200]}",
              file=sys.stderr)
        sys.exit(1)

    pois = response.json()

    # Post-condition: response is a non-empty list
    if not isinstance(pois, list):
        print(f"[ERROR] Expected a JSON array, got {type(pois).__name__}", file=sys.stderr)
        sys.exit(1)
    if len(pois) == 0:
        print(f"[WARN] API returned an empty array — no POIs found for '{destination}'",
              file=sys.stderr)

    return pois


def save_json(data: list, output_path: Path) -> None:
    """Write data to output_path as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(data)} POIs → {output_path}")


def print_summary(pois: list) -> None:
    """Print a human-readable breakdown of the fetched POIs."""
    type_counts: dict[str, int] = {}
    for poi in pois:
        # The UNE 178503 type may appear in different fields depending on
        # the API version — try the most common ones
        poi_type = (
            poi.get("type")
            or poi.get("type_graph")
            or poi.get("id_type")
            or "unknown"
        )
        if isinstance(poi_type, list):
            poi_type = poi_type[0] if poi_type else "unknown"
        type_counts[str(poi_type)] = type_counts.get(str(poi_type), 0) + 1

    print(f"\n[SUMMARY] Total POIs fetched: {len(pois)}")
    print("[SUMMARY] Breakdown by type:")
    for poi_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {poi_type:40s}  {count:4d}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse args, load config, fetch POIs, and save."""
    parser = argparse.ArgumentParser(
        description="Fetch POIs for a tourist destination from the Inventrip API"
    )
    parser.add_argument(
        "--destination", default=DEFAULT_DESTINATION,
        help=f"Tourist destination slug (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--lang", default=DEFAULT_LANGUAGE,
        help=f"Language code for POI content (default: {DEFAULT_LANGUAGE})",
    )
    args = parser.parse_args()

    output_file = PROJECT_ROOT / "data" / f"{args.destination}_pois_raw_{args.lang}.json"

    base_url, api_key = load_config()
    params = build_request_params(api_key, args.destination, args.lang)
    pois = fetch_pois(base_url, params, args.destination)
    print_summary(pois)
    save_json(pois, output_file)


if __name__ == "__main__":
    main()

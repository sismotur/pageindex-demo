#!/usr/bin/env python3
"""
extract_destination_data.py — Fetch destination-level data for any tourist destination.

Collects five data sources and saves to data/{destination}_destination.json:
  1. /v120/tourist-destinations  — destination overview, trip IDs, route IDs
  2. /v120/trips                 — curated trips with full itineraries
  3. /v120/paths                 — walking/driving routes (one fetch per ID)
  4. /v120/interest-levels       — editorial hierarchy: Indispensable / Interesting / Outstanding
  5. /v120/tourist-types         — type codes with human-readable names

Usage:
    .venv/bin/python scripts/extract_destination_data.py
    .venv/bin/python scripts/extract_destination_data.py --destination fayon --lang es

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

PROJECT_ROOT        = Path(__file__).parent.parent
DEFAULT_DESTINATION = "ubeda"
DEFAULT_LANGUAGE    = "en"
TIMEOUT             = 60

load_dotenv(PROJECT_ROOT / ".env")

# Make `from lang_support import ...` work whether run as a script or module
sys.path.insert(0, str(Path(__file__).parent))
from lang_support import SUPPORTED_LANGS, is_supported  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_session(lang: str = DEFAULT_LANGUAGE) -> tuple[requests.Session, str]:
    """Return a configured requests session and the base URL."""
    base_url = os.getenv("INVENTRIP_API_BASE_URL", "").strip().rstrip("/")
    api_key  = os.getenv("INVENTRIP_API_KEY", "").strip()
    if not base_url or not api_key or api_key == "your_api_key_here":
        print("[ERROR] INVENTRIP_API_BASE_URL or INVENTRIP_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    session = requests.Session()
    session.params = {"api_key": api_key, "language": lang, "strip_nulls": "true"}
    return session, base_url


def fetch(session: requests.Session, url: str, extra: dict | None = None) -> list | dict:
    """GET a URL and return the parsed JSON, or exit on error."""
    params = extra or {}
    resp = session.get(url, params=params, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"[ERROR] {resp.status_code} {url}: {resp.text[:200]}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


def get_localized(entries: list[dict], lang: str, key: str = "value") -> str:
    """Return the value matching `lang` from a multilingual list.

    Falls back to the first available entry if the requested language is
    not present (e.g. when `strip_nulls=true` already filtered out the
    other languages and only one entry survives).
    """
    if not entries:
        return ""
    for e in entries:
        if e.get("language") == lang or e.get("id_language") == lang:
            return e.get(key, "") or e.get("value_text", "")
    first = entries[0]
    return first.get(key, "") or first.get("value_text", "")


# ── Fetchers ───────────────────────────────────────────────────────────────────

def fetch_destination(session, base_url: str, destination: str, lang: str) -> dict:
    """Fetch the destination record for the given tourist destination."""
    data = fetch(session, f"{base_url}/v120/tourist-destinations",
                 {"tourist_destination": destination})
    if not isinstance(data, list) or not data:
        print("[ERROR] tourist-destinations returned empty", file=sys.stderr)
        sys.exit(1)
    d = data[0]
    return {
        "name":             get_localized(d.get("name", []), lang, "value_text"),
        "description":      get_localized(d.get("description", []), lang),
        "official_url":     (d.get("url") or [""])[0],
        "tourist_types":    [t["tourist_type"] for t in d.get("tourist_types", [])],
        "tourist_networks": d.get("tourist_networks", []),
        "latitude":         d.get("latitude"),
        "longitude":        d.get("longitude"),
        "trip_ids":         d.get("trips", []),
        "route_ids":        d.get("routes", []),
    }


def fetch_trips(session, base_url: str, destination: str, lang: str) -> list:
    """Fetch all curated trips with full itineraries."""
    raw = fetch(session, f"{base_url}/v120/trips",
                {"tourist_destination": destination,
                 "add_itinerary": "true", "limit": 100, "offset": 0})
    trips = []
    for t in (raw if isinstance(raw, list) else []):
        name     = get_localized(t.get("name", []), lang)
        desc     = get_localized(t.get("description", []), lang)
        itinerary = []
        for step in t.get("itinerary", []):
            step_name = get_localized(step.get("name", []), lang)
            pois = []
            for item in step.get("itemListElement", []):
                poi_name = get_localized(item.get("name", []), lang)
                if poi_name:
                    pois.append(poi_name)
            if step_name or pois:
                itinerary.append({"step": step_name, "pois": pois})
        trips.append({
            "id":          t.get("identifier", ""),
            "name":        name,
            "description": desc,
            "type":        (t.get("type") or [""])[0],
            "url":         (t.get("url") or [""])[0],
            "itinerary":   itinerary,
        })
    return trips


def fetch_paths(session, base_url: str, route_ids: list, lang: str) -> list:
    """Fetch individual path records by ID."""
    paths = []
    for rid in route_ids:
        try:
            data = fetch(session, f"{base_url}/v120/paths",
                         {"id_path": rid, "add_itinerary": "true"})
            if not isinstance(data, list) or not data:
                continue
            p = data[0]
            name = get_localized(p.get("name", []), lang)
            desc = get_localized(p.get("description", []), lang)
            waypoints = []
            for step in p.get("itinerary", []):
                for item in step.get("itemListElement", []):
                    wp = get_localized(item.get("name", []), lang)
                    if wp:
                        waypoints.append(wp)
            paths.append({
                "id":         p.get("identifier", str(rid)),
                "name":       name,
                "description": desc,
                "waypoints":  waypoints,
            })
            print(f"  [path {rid}] \"{name}\"  ({len(waypoints)} waypoints)")
        except SystemExit:
            print(f"  [SKIP path {rid}] fetch failed", file=sys.stderr)
    return paths


def fetch_interest_levels(session, base_url: str, lang: str) -> dict:
    """Return dict mapping id_interest_level -> localized label."""
    data = fetch(session, f"{base_url}/v120/interest-levels")
    mapping = {}
    for item in (data if isinstance(data, list) else []):
        level_id = item.get("id_interest_level")
        label    = get_localized(item.get("name", []), lang)
        if level_id and label:
            mapping[level_id] = label
    return mapping


def fetch_tourist_types(session, base_url: str, lang: str) -> dict:
    """Return dict mapping touristType code -> localized display name."""
    data = fetch(session, f"{base_url}/v120/tourist-types")
    mapping = {}
    for item in (data if isinstance(data, list) else []):
        code  = item.get("touristType", "")
        label = get_localized(item.get("name", []), lang)
        if code and label:
            mapping[code] = label
    return mapping


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch destination-level data from the Inventrip API"
    )
    parser.add_argument(
        "--destination", default=DEFAULT_DESTINATION,
        help=f"Tourist destination slug (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--lang", default=DEFAULT_LANGUAGE,
        help=(f"Language code for content (default: {DEFAULT_LANGUAGE}). "
              f"One of: {', '.join(SUPPORTED_LANGS)}"),
    )
    args = parser.parse_args()

    if not is_supported(args.lang):
        print(f"[ERROR] Unsupported --lang '{args.lang}'. "
              f"Supported codes: {', '.join(SUPPORTED_LANGS)}",
              file=sys.stderr)
        sys.exit(1)

    output_file = PROJECT_ROOT / "data" / f"{args.destination}_destination_{args.lang}.json"

    load_dotenv(PROJECT_ROOT / ".env")
    session, base_url = get_session(lang=args.lang)
    print(f"[INFO] API base:    {base_url}")
    print(f"[INFO] Destination: {args.destination}")
    print(f"[INFO] Language:    {args.lang}")

    print("\n[1/5] Fetching tourist-destination overview...")
    dest_record = fetch_destination(session, base_url, args.destination, args.lang)
    print(f"  {dest_record['name']}  "
          f"({len(dest_record['trip_ids'])} trips, "
          f"{len(dest_record['route_ids'])} routes)")

    print("\n[2/5] Fetching trips with itineraries...")
    trips = fetch_trips(session, base_url, args.destination, args.lang)
    for t in trips:
        total_pois = sum(len(s["pois"]) for s in t["itinerary"])
        print(f"  {t['id']:12s}  \"{t['name']}\"  "
              f"({len(t['itinerary'])} steps, {total_pois} POIs)")

    print("\n[3/5] Fetching walking/driving routes...")
    paths = fetch_paths(session, base_url, dest_record["route_ids"], args.lang)

    print("\n[4/5] Fetching interest-level taxonomy...")
    interest_levels = fetch_interest_levels(session, base_url, args.lang)
    for k, v in sorted(interest_levels.items()):
        print(f"  {k} = {v}")

    print("\n[5/5] Fetching tourist-type names...")
    tourist_types = fetch_tourist_types(session, base_url, args.lang)
    print(f"  {len(tourist_types)} type codes loaded")

    # Save combined output
    output = {
        "destination":     dest_record,
        "trips":           trips,
        "paths":           paths,
        "interest_levels": interest_levels,
        "tourist_types":   tourist_types,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

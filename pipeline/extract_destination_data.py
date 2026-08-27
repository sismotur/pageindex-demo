#!/usr/bin/env python3
"""
pipeline/extract_destination_data.py — Fetch destination-level data for any tourist destination.

Collects five data sources and saves to data/{destination}_destination_{lang}.json:
  1. /v120/tourist-destinations  — destination overview, trip IDs, route IDs
  2. /v120/trips                 — curated trips with full itineraries
  3. /v120/trips (by id)         — walking/driving routes: a route is simply
                                    a trip whose extras.path is non-null;
                                    that field carries the real /v120/paths id
  4. /v120/interest-levels       — editorial hierarchy: Indispensable / Interesting / Outstanding
  5. /v120/tourist-types         — type codes with human-readable names

Usage:
    .venv/bin/python pipeline/extract_destination_data.py
    .venv/bin/python pipeline/extract_destination_data.py --destination fayon --lang es

Environment variables (loaded from .env):
    INVENTRIP_API_BASE_URL  Base URL of the Inventrip API
    INVENTRIP_API_KEY       API key passed as query param ?api_key=...
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT        = Path(__file__).parent.parent
DEFAULT_DESTINATION = "ubeda"
DEFAULT_LANGUAGE    = "en"
TIMEOUT             = 60

load_dotenv(PROJECT_ROOT / ".env")

# Make `from common.lang_support import ...` work whether run as a script or module
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.lang_support import SUPPORTED_LANGS, is_supported  # noqa: E402


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
    Visitor-facing trip/path data must never fall back to a foreign
    language. If the requested translation is absent, return an empty
    string so the build can omit it from tourist output while retaining
    any stable source identifier for QA/resolution.
    """
    if not entries:
        return ""
    for e in entries:
        if e.get("language") == lang or e.get("id_language") == lang:
            return e.get(key, "") or e.get("value_text", "")
    return ""


def get_item_poi_id(item: dict) -> str:
    """Return a stable POI identifier carried by an itinerary item, if any.

    API versions/place types have used `identifier`, `item`, and
    `id_object`.  Preserve a valid value rather than inferring an id from
    a translated name; the index builder will validate it against the
    destination POI map.  Folder entries returned by the API have no
    concrete identifier (the sentinel `Some(None)`/`null`) — skip them so
    later resolution never mistakes a group label for a POI id.
    """
    for key in ("identifier", "item", "id_object", "poi_id"):
        value = item.get(key)
        if value in (None, ""):
            continue
        return str(value)
    return ""


def extract_itinerary_items(item_list_elements: list[dict],
                            lang: str) -> list[dict]:
    """Recursively extract itinerary items, preserving folder/POI hierarchy.

    The Inventrip v120 /trips|/paths payload nests `itemListElement`:
    a step can carry direct POIs and/or `ItemList` folders that group
    child POIs (e.g. "1.1 Plaza Vázquez de Molina").  The previous
    extractor flattened only the top level and lost every child inside
    a folder.  Each returned entry has:
      - name:   localized display name (may be empty)
      - poi_id: stable identifier (present only if the raw item carries one)
      - items:  nested list (present only when the raw item has children)
    """
    out: list[dict] = []
    for elem in item_list_elements or []:
        if not isinstance(elem, dict):
            continue
        name    = get_localized(elem.get("name", []), lang)
        poi_id  = get_item_poi_id(elem)
        children = extract_itinerary_items(
            elem.get("itemListElement") or [], lang
        )
        if not (name or poi_id or children):
            continue
        entry: dict = {"name": name}
        if poi_id:
            entry["poi_id"] = poi_id
        if children:
            entry["items"] = children
        out.append(entry)
    return out


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
    """Fetch all curated trips with full itineraries.

    Excludes route-trips (extras.path non-null) returned by this same bulk
    listing: fetch_paths() fetches those separately so build_itineraries()
    can mark both copies with is_route=True. Without this filter, a route
    already present here creates an unflagged duplicate that shadows the
    correctly flagged copy during get_trip() lookup.
    """
    raw = fetch(session, f"{base_url}/v120/trips",
                {"tourist_destination": destination,
                 "add_itinerary": "true", "limit": 100, "offset": 0})
    trips = []
    for t in (raw if isinstance(raw, list) else []):
        if (t.get("extras") or {}).get("path"):
            continue
        name     = get_localized(t.get("name", []), lang)
        desc     = get_localized(t.get("description", []), lang)
        itinerary = []
        for step in t.get("itinerary", []):
            step_name = get_localized(step.get("name", []), lang)
            items = extract_itinerary_items(
                step.get("itemListElement") or [], lang
            )
            if step_name or items:
                itinerary.append({"step": step_name, "items": items})
        trips.append({
            "id":          t.get("identifier", ""),
            "name":        name,
            "description": desc,
            "type":        (t.get("type") or [""])[0],
            "url":         (t.get("url") or [""])[0],
            "itinerary":   itinerary,
        })
    return trips


def _parse_id_path(path_ref: str) -> str:
    """Extract the numeric id_path from a trip's extras.path reference.

    e.g. 'paths?id_path=255' -> '255'.  Returns '' when absent/unparseable.
    """
    match = re.search(r"id_path=(\d+)", str(path_ref or ""))
    return match.group(1) if match else ""


def fetch_paths(session, base_url: str, route_ids: list, lang: str) -> list:
    """Fetch physical routes referenced by the destination's `routes` ids.

    Each id in `routes` is NOT a /v120/paths id_path — it is the id of a
    trip record. A route is simply a trip whose extras.path field
    ('paths?id_path=N') is non-null; a trip with no extras.path is not a
    route and is skipped. The tag shown to visitors uses this trip's own
    numeric id — the destination's real, stable route identifier — never
    the internal /v120/paths id_path, which has no meaning outside this
    API and must not be exposed. Name, description, url, and itinerary
    all come from the trip record itself (add_itinerary=true), exactly
    like fetch_trips(); the runtime (index_tools._append_route_stops())
    is responsible for dropping degenerate step labels when rendering,
    not the extractor.
    """
    paths = []
    for rid in route_ids:
        try:
            trip_data = fetch(session, f"{base_url}/v120/trips", {
                "trip": rid, "add_itinerary": "true", "limit": 1, "offset": 0,
            })
        except SystemExit:
            print(f"  [SKIP route {rid}] trip lookup failed", file=sys.stderr)
            continue
        if not isinstance(trip_data, list) or not trip_data:
            print(f"  [SKIP route {rid}] trip not found", file=sys.stderr)
            continue

        trip = trip_data[0]
        id_path = _parse_id_path((trip.get("extras") or {}).get("path"))
        if not id_path:
            print(f"  [SKIP route {rid}] no linked path (not a route)",
                  file=sys.stderr)
            continue

        name = get_localized(trip.get("name", []), lang)
        desc = get_localized(trip.get("description", []), lang)
        itinerary = []
        for step in trip.get("itinerary") or []:
            step_name = get_localized(step.get("name", []), lang)
            items = extract_itinerary_items(
                step.get("itemListElement") or [], lang
            )
            if step_name or items:
                itinerary.append({"step": step_name, "items": items})

        paths.append({
            "id":          f"trip/{rid}",
            "name":        name,
            "description": desc,
            "url":         (trip.get("url") or [""])[0],
            "itinerary":   itinerary,
        })
        print(f"  [route {rid}] \"{name}\"  ({len(itinerary)} steps)")
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


def _count_leaf_items(items: list[dict]) -> int:
    total = 0
    for item in items or []:
        children = item.get("items")
        if children:
            total += _count_leaf_items(children)
        else:
            total += 1
    return total


# ── Main ──────────────────────────────────────────────────────────────

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
        total_pois = sum(_count_leaf_items(s["items"]) for s in t["itinerary"])
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

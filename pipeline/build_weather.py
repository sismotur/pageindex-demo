#!/usr/bin/env python3
"""
pipeline/build_weather.py — Fetch /v100/weather-daily and write the offline
weather artifact the phone consumes.

Output: weather/{destination}_{lang}.json

Contract mirror of the eventual Cloudflare daily cron branch (see
docs/cloudflare-worker-spec.md). Keeps the normalisation logic in a
single reference implementation so the TS port can copy it verbatim.

Usage:
    .venv/bin/python pipeline/build_weather.py
    .venv/bin/python pipeline/build_weather.py --destination ubeda --lang es
    .venv/bin/python pipeline/build_weather.py --destination ubeda --all-languages

Environment variables (loaded from .env):
    INVENTRIP_API_BASE_URL  Base URL of the Inventrip API
    INVENTRIP_API_KEY       API key passed as query param ?api_key=...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT        = Path(__file__).parent.parent
DEFAULT_DESTINATION = "ubeda"
DEFAULT_LANGUAGE    = "en"
DEFAULT_UNITS       = "metric"
FORECAST_DAYS       = 7
TIMEOUT             = 30
STALE_AFTER_HOURS   = 24        # phone should refresh past this
SCHEMA_VERSION      = 1

# The Inventrip icon CDN URLs look like:
#   https://inventrip.com/cdn-cgi/imagedelivery/{ACCOUNT}/{ICON_UUID}/public
# The middle UUID is stable per weather condition regardless of language.
_ICON_UUID_RE = re.compile(
    r"imagedelivery/[^/]+/(?P<uuid>[0-9a-fA-F-]{36})/",
)

load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
from common.lang_support import SUPPORTED_LANGS, is_supported  # noqa: E402


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _get_session() -> tuple[requests.Session, str]:
    base_url = os.getenv("INVENTRIP_API_BASE_URL", "").strip().rstrip("/")
    api_key  = os.getenv("INVENTRIP_API_KEY", "").strip()
    if not base_url or not api_key or api_key == "your_api_key_here":
        print("[ERROR] INVENTRIP_API_BASE_URL or INVENTRIP_API_KEY not set",
              file=sys.stderr)
        sys.exit(1)
    session = requests.Session()
    session.params = {"api_key": api_key}
    return session, base_url


def _load_destination_coords(destination: str) -> tuple[float, float]:
    """Return (latitude, longitude) from any language snapshot of the destination.

    The destination record is language-agnostic for coordinates, but the
    file is written per-language, so try a few candidates.
    """
    candidates = [
        PROJECT_ROOT / "data" / f"{destination}_destination_en.json",
        PROJECT_ROOT / "data" / f"{destination}_destination_es.json",
    ]
    candidates.extend(sorted(
        PROJECT_ROOT.joinpath("data").glob(f"{destination}_destination_*.json")
    ))
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        dest = data.get("destination") or {}
        lat  = dest.get("latitude")
        lon  = dest.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
    print(f"[ERROR] No destination coordinates found for '{destination}'. "
          "Run pipeline/extract_destination_data.py first.", file=sys.stderr)
    sys.exit(1)


# ── Normalisation ───────────────────────────────────────────────────────────

def _extract_condition_code(icon_url: str) -> str:
    """Return the stable condition UUID from an Inventrip icon URL, or ""."""
    if not icon_url:
        return ""
    match = _ICON_UUID_RE.search(icon_url)
    return match.group("uuid") if match else ""


def _iso_weekday(date: datetime) -> int:
    """Return 1..7 for Mon..Sun (Python datetime.isoweekday semantics)."""
    return int(date.isoweekday())


def _normalize_forecast(raw: list[dict]) -> list[dict]:
    """Transform the raw Inventrip payload into the offline contract shape.

    Never invents fields: a missing source value becomes an empty string
    or None so the on-device tool can skip rendering it.
    """
    out: list[dict] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        timestamp = entry.get("forecastTimestamp")
        try:
            dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        except (TypeError, ValueError):
            continue
        icon_url = entry.get("icon") or ""
        out.append({
            "date":            dt.date().isoformat(),
            "iso_weekday":     _iso_weekday(dt),
            "day_label":       str(entry.get("forecastDayOfWeek") or "").strip(),
            "temp_min_c":      round(float(entry.get("tempMin")), 1)
                                if entry.get("tempMin") is not None else None,
            "temp_max_c":      round(float(entry.get("tempMax")), 1)
                                if entry.get("tempMax") is not None else None,
            "condition":       str(entry.get("currentWeather") or "").strip(),
            "condition_code":  _extract_condition_code(icon_url),
            "icon_url":        icon_url,
        })
    return out


def build_weather(destination: str, lang: str,
                  raw: list[dict],
                  latitude: float, longitude: float,
                  fetched_at: datetime | None = None) -> dict:
    """Assemble the complete weather artifact (no I/O)."""
    when = fetched_at or datetime.now(timezone.utc)
    expires = when + timedelta(hours=STALE_AFTER_HOURS)
    return {
        "meta": {
            "destination":    destination,
            "lang":           lang,
            "latitude":       round(latitude,  6),
            "longitude":      round(longitude, 6),
            "units":          DEFAULT_UNITS,
            "fetched_at":     when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "expires_at":     expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "schema_version": SCHEMA_VERSION,
        },
        "forecast": _normalize_forecast(raw),
    }


# ── Fetching ────────────────────────────────────────────────────────────────

def fetch_weather_raw(session: requests.Session, base_url: str,
                      latitude: float, longitude: float,
                      lang: str) -> list[dict]:
    """Return the raw Inventrip payload as a list of daily entries."""
    resp = session.get(
        f"{base_url}/v100/weather-daily",
        params={
            "latitude":  latitude,
            "longitude": longitude,
            "cnt":       FORECAST_DAYS,
            "language":  lang,
            "units":     DEFAULT_UNITS,
        },
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        print(f"[ERROR] {resp.status_code} weather-daily: {resp.text[:200]}",
              file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    if not isinstance(data, list):
        print(f"[ERROR] weather-daily returned {type(data).__name__}, expected list",
              file=sys.stderr)
        sys.exit(1)
    return data


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_one(session, base_url, destination: str, lang: str,
               latitude: float, longitude: float,
               output_dir: Path) -> Path:
    raw = fetch_weather_raw(session, base_url, latitude, longitude, lang)
    artifact = build_weather(destination, lang, raw, latitude, longitude)
    output = output_dir / f"{destination}_{lang}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] {destination}/{lang}: {len(artifact['forecast'])} days -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline weather artifact")
    parser.add_argument(
        "--destination", default=DEFAULT_DESTINATION,
        help=f"Tourist destination slug (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--lang", default=DEFAULT_LANGUAGE,
        help=(f"Language code (default: {DEFAULT_LANGUAGE}). "
              f"One of: {', '.join(SUPPORTED_LANGS)}"),
    )
    parser.add_argument(
        "--all-languages", action="store_true",
        help="Iterate every supported language for the destination.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: weather/).",
    )
    args = parser.parse_args()

    if not args.all_languages and not is_supported(args.lang):
        print(f"[ERROR] Unsupported --lang '{args.lang}'. "
              f"Supported codes: {', '.join(SUPPORTED_LANGS)}",
              file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir \
                 else PROJECT_ROOT / "weather"
    session, base_url = _get_session()
    latitude, longitude = _load_destination_coords(args.destination)
    print(f"[INFO] Destination: {args.destination} @ ({latitude}, {longitude})")

    langs = SUPPORTED_LANGS if args.all_languages else [args.lang]
    for lang in langs:
        _build_one(session, base_url, args.destination, lang,
                   latitude, longitude, output_dir)


if __name__ == "__main__":
    main()

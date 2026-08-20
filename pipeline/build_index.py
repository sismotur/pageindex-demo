#!/usr/bin/env python3
"""
pipeline/build_index.py — Build a POI-aware index from the Inventrip API JSON.

Replaces pageindex/run_pageindex.py + add_section_summaries.py.  Reads:
    data/{destination}_pois_raw_{lang}.json     (raw /v120/pois output)
    data/{destination}_destination_{lang}.json  (optional /v120/* metadata)

Writes:
    indexes/{destination}_{lang}.json

The index is consumed by assistant/run_eval.py and assistant/chat_demo.py
via assistant/index_tools.py, and is the single artifact the mobile apps
download for fully-offline use (see docs/mobile-offline-contract.md).
No LLM calls, no Markdown intermediate; deterministic and re-runnable.

Usage:
    .venv/bin/python pipeline/build_index.py
    .venv/bin/python pipeline/build_index.py --destination caceres --lang es
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Shared normaliser + language list from common/ so the name_index keys
# match exactly what assistant/index_tools.find_poi_by_name() looks up.
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.textnorm import normalize_text, tokenize  # noqa: E402
from common.lang_support import SUPPORTED_LANGS, is_supported  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT        = Path(__file__).parent.parent
DEFAULT_DESTINATION = "ubeda"
DEFAULT_LANGUAGE    = "en"
API_BASE_URL        = "https://api.inventrip.com"

# Section grouping rules.  Identical titles to the previous Markdown
# generator so that eval/questions.json's expected_section values keep
# matching without rubric edits.
SECTIONS: list[tuple[str, set[str]]] = [
    ("UNESCO World Heritage and City Overview",
     {"WorldHeritageSite", "City"}),
    # Accommodation must come before Civil Monuments so dual-typed
    # POIs (Hotel + CivilBuilding, e.g. paradores) are classified as
    # accommodation rather than monuments.
    ("Accommodation",
     {"Hotel", "BoutiqueHotel", "LodgingBusiness", "Apartment",
      "RuralHouse", "Hostel", "GuestHouse", "RVPark"}),
    ("Civil and Historical Monuments",
     {"CivilBuilding", "MilitaryBuilding"}),
    ("Religious Heritage",
     {"PlaceOfWorship"}),
    ("Museums and Culture",
     {"Museum", "CultureCenter"}),
    ("Archaeological Sites",
     {"ArchaeologicalArea"}),
    ("Tourist Attractions and Viewpoints",
     {"TouristAttraction", "ViewPoint"}),
    ("Squares, Parks and Natural Areas",
     {"Square", "Park", "LeisureArea"}),
    ("Gastronomy",
     {"Restaurant", "CafeOrCoffeeShop", "BarOrPub", "IceCreamShop",
      "OilMill", "FoodEvent"}),
    ("Guided Tours and Itineraries",
     {"TouristTrip"}),
    ("Events and Festivals",
     {"BusinessEvent", "Event", "TraditionalFestival",
      "MusicEvent", "ReligionEvent"}),
    ("Shopping",
     {"ShoppingCenter", "Store"}),
    ("Tourist Information and Services",
     {"TouristInformationCenter"}),
    ("Health and Beauty",
     {"HealthAndBeautyBusiness", "Pharmacy",
      "MedicalClinic", "PrimaryCare", "Hospital"}),
    ("Practical Information",
     {"ParkingFacility", "GasStation", "BusStation",
      "PoliceStation", "FireStation", "CivilProtection"}),
    ("Sports and Leisure Activities",
     {"SportsActivityLocation", "WaterActivityCenter"}),
    ("Quality, Rules and Visitor Advice",
     {"Certification", "VisitRule", "VisitAdvice"}),
]
OTHER_SECTION_TITLE = "Other Points of Interest"

# Sub-section grouping (schema v2).  Sections larger than GROUP_MIN_POIS
# are broken into per-type groups so on-device models can navigate
# section -> group -> POIs instead of scanning one long flat list
# (66 Shopping POIs in Ubeda already truncate at the default limit=50).
# Types with fewer than GROUP_MIN_SIZE members fold into an "Other" group.
# Inspired by PageIndex Flash's key_items: every group summary preserves
# the names of its top POIs so nothing vanishes from the model's view.
GROUP_MIN_POIS = 30
GROUP_MIN_SIZE = 2
OTHER_GROUP_TITLE = "Other"

# Map-prominence threshold.  POIs with zoom_level <= this are flagged as
# major landmarks in get_poi() output.
PROMINENCE_ZOOM_MAX = 16

# ISO 3166-1 alpha-2 → human-readable country names.  Same list used by
# the previous Markdown generator; kept here so the index file is
# self-contained when read by downstream tools.
COUNTRY_CODES: dict[str, str] = {
    "AD": "Andorra", "AR": "Argentina", "AU": "Australia",
    "BR": "Brazil", "CA": "Canada", "CL": "Chile", "CN": "China",
    "CO": "Colombia", "DE": "Germany", "EG": "Egypt", "ES": "Spain",
    "FR": "France", "GB": "United Kingdom", "GR": "Greece",
    "IN": "India", "IT": "Italy", "JP": "Japan", "MA": "Morocco",
    "MX": "Mexico", "NL": "Netherlands", "PE": "Peru",
    "PT": "Portugal", "TN": "Tunisia", "TR": "Turkey",
    "US": "United States",
}

# Interest-level fallback labels (used when destination JSON lacks the taxonomy).
DEFAULT_INTEREST_LABELS = {1: "Indispensable", 2: "Interesting", 3: "Outstanding"}


# ── Localised value helpers ────────────────────────────────────────────────

def get_text(field: Any, lang: str = "en") -> str:
    """Extract the plain string from a localised list-of-dicts or raw string."""
    if not field:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and item.get("language") == lang:
                return item.get("value", "") or item.get("value_text", "")
        # No exact-language match — fall back to first entry
        first = field[0]
        if isinstance(first, dict):
            return first.get("value", "") or first.get("value_text", "")
        return str(first)
    return str(field)


def get_list_text(field: Any) -> list[str]:
    """Extract a list of strings from a possibly-list field."""
    if not field:
        return []
    if isinstance(field, str):
        return [field]
    return [str(x) for x in field if x]


# ── Section assignment ─────────────────────────────────────────────────────

def _section_id_for(title: str) -> str:
    """Slug the section title for stable IDs in the index."""
    norm = normalize_text(title)
    return re.sub(r"\s+", "-", norm)


SECTION_RULES = [(title, types, _section_id_for(title))
                 for title, types in SECTIONS]
OTHER_SECTION_ID = _section_id_for(OTHER_SECTION_TITLE)


def assign_section(types: list[str]) -> tuple[str, str]:
    """Return (section_id, section_title) for a POI given its type list."""
    type_set = set(types)
    for title, type_set_for_section, sid in SECTION_RULES:
        if type_set & type_set_for_section:
            return sid, title
    return OTHER_SECTION_ID, OTHER_SECTION_TITLE


# ── URL builders ────────────────────────────────────────────────────────────

def image_url(image_ref: str) -> str | None:
    """Convert 'image/44883' to a full API URL with high quality."""
    parts = (image_ref or "").split("/")
    if len(parts) >= 2 and parts[-1].isdigit():
        return f"{API_BASE_URL}/v100/image/{parts[-1]}?image_quality=high"
    return None


def audio_url(audio_id: int | str, lang: str, destination: str) -> str:
    """Build a per-language audio guide URL."""
    return (f"{API_BASE_URL}/v100/audios?language={lang}&offset=1"
            f"&audio={audio_id}&tourist_destination={destination}")


# ── POI normalisation ───────────────────────────────────────────────────────

def normalize_poi(raw: dict, lang: str, destination: str,
                  tourist_type_display: dict[str, str],
                  type_display: dict[str, str],
                  interest_labels: dict[int, str]) -> dict:
    """Convert one raw POI record into the index-internal shape.

    Output keys are stable, snake_case, and have None/empty values stripped
    where appropriate so downstream formatters can use simple truthiness checks.
    """
    name = get_text(raw.get("name"), lang=lang) or "(unnamed)"
    description = get_text(raw.get("description"), lang=lang) or ""
    types = get_list_text(raw.get("type"))
    extras = raw.get("extras") or {}

    interest_level = extras.get("id_interest_level")
    if isinstance(interest_level, int) and interest_level in interest_labels:
        interest_label = interest_labels[interest_level]
    elif interest_level == 0:
        interest_level = None
        interest_label = None
    else:
        interest_label = None

    # Tourist-type display names: prefer destination-supplied mapping
    raw_tourist_types = get_list_text(raw.get("touristType"))
    display_tourist_types = []
    for code in raw_tourist_types:
        label = tourist_type_display.get(code) or tourist_type_display.get(code.upper())
        display_tourist_types.append(label.title() if label else code.title())

    # UNE type display name: take the first type code, prefer destination map
    primary_type = types[0] if types else ""
    display_type = type_display.get(primary_type, primary_type)

    # Image URLs
    image_urls = [u for u in (image_url(ref) for ref in get_list_text(raw.get("image"))) if u]

    # Audio URLs
    audios = raw.get("audios") or []
    audio_urls = [audio_url(a, lang=lang, destination=destination) for a in audios]

    # subjectOf documents
    subjects = (raw.get("extras") or {}).get("subjectOf") or []
    subject_of_urls = []
    for s in subjects:
        if isinstance(s, dict) and s.get("url"):
            label = s.get("name") or "Document"
            subject_of_urls.append(f"{label}: {s['url']}")

    poi_id = raw.get("identifier") or ""

    record = {
        "poi_id":              poi_id,
        "name":                name,
        "normalized_name":     normalize_text(name),
        "description":         description,
        "types":               types,
        "display_type":        display_type,
        "tourist_types":       raw_tourist_types,
        "display_tourist_types": display_tourist_types,
        "interest_level":      interest_level if isinstance(interest_level, int) and interest_level > 0 else None,
        "interest_level_label": interest_label,
        "zoom_level":          extras.get("zoom_level") if extras.get("zoom_level") not in (None, 0) else None,
        "booking_url":         extras.get("booking_url") or "",
        "url":                 get_list_text(raw.get("url")),
        "telephone":           get_list_text(raw.get("telephone")),
        "email":               get_list_text(raw.get("email")),
        "street_address":      raw.get("streetAddress") or "",
        "address_locality":    raw.get("addressLocality") or "",
        "address_province":    raw.get("addressProvince") or "",
        "address_region":      raw.get("addressRegion") or "",
        "postal_code":         raw.get("postalCode") or "",
        "country_code":        raw.get("addressCountry") or "",
        "country":             COUNTRY_CODES.get(raw.get("addressCountry") or "",
                                                 raw.get("addressCountry") or ""),
        "latitude":            raw.get("latitude"),
        "longitude":           raw.get("longitude"),
        "image_urls":          image_urls,
        "audio_urls":          audio_urls,
        "subject_of_urls":     subject_of_urls,
        "start_date":          raw.get("startDate") or "",
        "end_date":            raw.get("endDate") or "",
        "raw_extras":          extras,
    }
    return record


# ── Section building ───────────────────────────────────────────────────────

def build_section_summary(section_pois: list[dict]) -> str:
    """Deterministic 1-line summary: counts + top tourist types + notable POIs.

    No LLM call — this replaces add_section_summaries.py entirely.
    """
    if not section_pois:
        return "No POIs in this section."

    counts_by_label: dict[str, int] = {}
    for p in section_pois:
        label = p.get("interest_level_label")
        if label:
            counts_by_label[label] = counts_by_label.get(label, 0) + 1

    parts = [f"{len(section_pois)} POI{'s' if len(section_pois) != 1 else ''}"]
    if counts_by_label:
        # Order: Indispensable > Interesting > Outstanding
        order = ["Indispensable", "Interesting", "Outstanding"]
        breakdown = [f"{counts_by_label[k]} {k}" for k in order if counts_by_label.get(k)]
        if breakdown:
            parts[0] += f" ({', '.join(breakdown)})"

    # Top tourist types across the section
    tt_counts: dict[str, int] = {}
    for p in section_pois:
        for label in p.get("display_tourist_types") or []:
            tt_counts[label] = tt_counts.get(label, 0) + 1
    if tt_counts:
        top_tt = sorted(tt_counts.items(), key=lambda kv: -kv[1])[:3]
        parts.append("Top interests: " + ", ".join(name for name, _ in top_tt))

    # Notable POIs: top 3 by (interest_level, zoom_level)
    sorted_pois = sorted(section_pois,
                         key=lambda p: (p.get("interest_level") or 99,
                                        p.get("zoom_level") or 99,
                                        p.get("normalized_name") or ""))
    notable = [p["name"] for p in sorted_pois[:3] if p.get("name")]
    if notable:
        parts.append("Notable: " + ", ".join(notable))

    return ". ".join(parts) + "."


def _group_id_for(section_id: str, group_title: str) -> str:
    """Stable group id: '{section_id}--{normalized-title}'."""
    return f"{section_id}--{re.sub(r'\\s+', '-', normalize_text(group_title))}"


def build_section_groups(section_pois: list[dict],
                         section_id: str) -> list[dict] | None:
    """Split a large section into per-type groups (schema v2).

    Returns None when the section is small enough to stay flat, or when
    the type distribution is too homogeneous for groups to help
    navigation.  Groups are ordered by their best POI using the same
    (interest_level, zoom_level, name) composite key as sections.
    """
    if len(section_pois) <= GROUP_MIN_POIS:
        return None

    by_type: dict[str, list[dict]] = {}
    for p in section_pois:
        key = p.get("display_type") or (p.get("types") or [""])[0] or OTHER_GROUP_TITLE
        by_type.setdefault(key, []).append(p)

    # Fold tiny types into "Other"
    groups: dict[str, list[dict]] = {}
    other: list[dict] = []
    for key, members in by_type.items():
        if len(members) >= GROUP_MIN_SIZE:
            groups[key] = members
        else:
            other.extend(members)
    if other:
        groups[OTHER_GROUP_TITLE] = other

    # Not worth grouping if everything landed in one bucket
    if len(groups) < 2:
        return None

    sort_key = lambda p: (p.get("interest_level") or 99,
                          p.get("zoom_level") or 99,
                          p.get("normalized_name") or "")
    out = []
    for title, members in groups.items():
        members = sorted(members, key=sort_key)
        out.append({
            "group_id": _group_id_for(section_id, title),
            "title":    title,
            "poi_ids":  [p["poi_id"] for p in members],
            "summary":  build_section_summary(members),
            # members[0] is the group's best POI after sorting; keep its
            # key so groups can be ordered best-first below.
            "_best":    sort_key(members[0]),
        })
    # Order groups by their best POI
    out.sort(key=lambda g: g.pop("_best"))
    return out


def assemble_sections(pois: list[dict]) -> list[dict]:
    """Group POIs into ordered sections with deterministic summaries."""
    buckets: dict[str, list[dict]] = {}
    titles: dict[str, str] = {}

    for p in pois:
        sid, title = assign_section(p.get("types") or [])
        buckets.setdefault(sid, []).append(p)
        titles[sid] = title

    # Sort within each bucket by composite key
    for sid in buckets:
        buckets[sid].sort(key=lambda p: (p.get("interest_level") or 99,
                                         p.get("zoom_level") or 99,
                                         p.get("normalized_name") or ""))

    # Order sections: priority list first, then OTHER_SECTION at the tail
    ordered = []
    for _, _, sid in SECTION_RULES:
        if sid in buckets:
            ordered.append(sid)
    if OTHER_SECTION_ID in buckets and OTHER_SECTION_ID not in ordered:
        ordered.append(OTHER_SECTION_ID)

    sections = []
    for sid in ordered:
        sec = {
            "section_id": sid,
            "title":      titles[sid],
            "poi_ids":    [p["poi_id"] for p in buckets[sid]],
            "summary":    build_section_summary(buckets[sid]),
        }
        groups = build_section_groups(buckets[sid], sid)
        if groups:
            sec["groups"] = groups
        sections.append(sec)
    return sections


# ── Facets ─────────────────────────────────────────────────────────────────

def build_facets(pois: list[dict], sections: list[dict]) -> dict:
    """Precompute facet → poi_id lookups plus full-text evidence terms.

    `search_terms` is a deterministic inverted index over visitor-facing
    POI content: name, description, category label, tourism-interest
    labels, and locality.  It lets the offline assistant test whether
    several visitor concepts actually coexist on the same POI before it
    claims a relationship (e.g. restaurant + olive oil).
    """
    by_section: dict[str, list[str]] = {s["section_id"]: list(s["poi_ids"]) for s in sections}
    by_type: dict[str, list[str]] = {}
    by_tourist_type: dict[str, list[str]] = {}
    by_interest_level: dict[str, list[str]] = {}
    by_zoom_bucket: dict[str, list[str]] = {"<=14": [], "15-16": [], "17-19": []}
    indispensable: list[str] = []
    search_terms: dict[str, list[str]] = {}

    for p in pois:
        for t in p.get("types") or []:
            by_type.setdefault(t, []).append(p["poi_id"])
        for tt in p.get("tourist_types") or []:
            by_tourist_type.setdefault(tt, []).append(p["poi_id"])
        if p.get("interest_level"):
            by_interest_level.setdefault(str(p["interest_level"]), []).append(p["poi_id"])
        if p.get("interest_level") == 1:
            indispensable.append(p["poi_id"])
        zoom = p.get("zoom_level")
        if isinstance(zoom, int):
            if zoom <= 14:
                by_zoom_bucket["<=14"].append(p["poi_id"])
            elif zoom <= 16:
                by_zoom_bucket["15-16"].append(p["poi_id"])
            else:
                by_zoom_bucket["17-19"].append(p["poi_id"])

        # Index only user-facing content.  The tool output later extracts
        # a description/name evidence snippet so the model can distinguish
        # an explicit match from a merely related suggestion.
        searchable = " ".join([
            p.get("name") or "",
            p.get("description") or "",
            p.get("display_type") or "",
            " ".join(p.get("display_tourist_types") or []),
            p.get("address_locality") or "",
        ])
        for term in set(tokenize(searchable)):
            # Single/two-character tokens add memory cost but no useful
            # tourism retrieval signal (articles, initials, map labels).
            if len(term) >= 3:
                search_terms.setdefault(term, []).append(p["poi_id"])

    return {
        "by_section":        by_section,
        "by_type":           by_type,
        "by_tourist_type":   by_tourist_type,
        "by_interest_level": by_interest_level,
        "by_zoom_bucket":    by_zoom_bucket,
        "indispensable":     indispensable,
        # Sort postings so JSON output is reproducible and Android/iOS
        # implementations can perform stable intersections.
        "search_terms":      {term: sorted(ids)
                              for term, ids in sorted(search_terms.items())},
    }


# ── Destination overview & trips ────────────────────────────────────────────

def build_destination_overview(dest_data: dict | None,
                                tourist_type_display: dict[str, str]) -> str:
    """Compose a multi-line destination overview from /tourist-destinations."""
    if not dest_data:
        return ""
    d = dest_data.get("destination") or {}
    parts: list[str] = []
    if d.get("description"):
        parts.append(d["description"].strip())

    bullets: list[str] = []
    types = d.get("tourist_types") or []
    if types:
        labels = [tourist_type_display.get(t, t).title() for t in types]
        bullets.append(f"Tourism types: {', '.join(labels)}")
    if d.get("tourist_networks"):
        bullets.append(f"Networks: {', '.join(d['tourist_networks'])}")
    if d.get("official_url"):
        bullets.append(f"Official website: {d['official_url']}")
    if bullets:
        parts.append("\n".join(f"- {b}" for b in bullets))

    return "\n\n".join(parts).strip()


def _resolve_itinerary_steps(raw_steps: list[dict],
                             name_index: dict[str, str]) -> list[dict]:
    """Resolve ordered localized waypoint names to POI ids when exact.

    Never discard an unresolved name: catalogues can contain itinerary
    stops that are not represented by an individual POI record, or whose
    translation does not exactly match the POI title.
    """
    steps = []
    for position, raw_step in enumerate(raw_steps, 1):
        names = list(raw_step.get("pois") or [])
        poi_ids: list[str] = []
        unresolved: list[str] = []
        for name in names:
            poi_id = name_index.get(normalize_text(name))
            if poi_id:
                poi_ids.append(poi_id)
            else:
                unresolved.append(name)
        steps.append({
            "position":             position,
            "title":                raw_step.get("step") or "",
            "poi_ids":              poi_ids,
            "unresolved_poi_names": unresolved,
        })
    return steps


def build_itineraries(dest_data: dict | None,
                      name_index: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Return curated trips and paths in the shared schema-v4 shape."""
    if not dest_data:
        return [], []

    trips = []
    for raw_trip in dest_data.get("trips") or []:
        raw_steps = list(raw_trip.get("itinerary") or [])
        # Some valid editorial trip suggestions have only a title and
        # description (no ordered stops yet). Keep them searchable rather
        # than silently deleting the suggestion from the offline corpus.
        if not raw_steps and not (
            raw_trip.get("id") or raw_trip.get("name") or raw_trip.get("description")
        ):
            continue
        itinerary_id = raw_trip.get("id") or ""
        trips.append({
            "itinerary_id": itinerary_id,
            # Kept as a backwards-compatible alias for v1–v3 consumers.
            "trip_id":      itinerary_id,
            "kind":         "trip",
            "source_type":  raw_trip.get("type") or "TouristTrip",
            "name":         raw_trip.get("name") or "",
            "description":  raw_trip.get("description") or "",
            "url":          raw_trip.get("url") or "",
            "steps":        _resolve_itinerary_steps(raw_steps, name_index),
        })

    paths = []
    for raw_path in dest_data.get("paths") or []:
        # New snapshots retain step boundaries; older snapshots only have
        # flat waypoints, which become one ordered "Waypoints" step.
        raw_steps = list(raw_path.get("itinerary") or [])
        if not raw_steps and raw_path.get("waypoints"):
            raw_steps = [{"step": "Waypoints",
                          "pois": list(raw_path.get("waypoints") or [])}]
        if not raw_steps and not raw_path.get("name"):
            continue
        itinerary_id = raw_path.get("id") or ""
        paths.append({
            "itinerary_id": itinerary_id,
            "path_id":      itinerary_id,
            "kind":         "path",
            "source_type":  "Path",
            "name":         raw_path.get("name") or "",
            "description":  raw_path.get("description") or "",
            "url":          raw_path.get("url") or "",
            "steps":        _resolve_itinerary_steps(raw_steps, name_index),
        })

    return trips, paths


# ── Top-level builder ──────────────────────────────────────────────────────

def build_index(raw_pois: list[dict], dest_data: dict | None,
                lang: str, destination: str) -> dict:
    """Assemble the complete index dict (no I/O)."""
    if not raw_pois:
        raise ValueError("POI list is empty")

    tourist_type_display: dict[str, str] = (dest_data or {}).get("tourist_types") or {}
    interest_labels_raw = (dest_data or {}).get("interest_levels") or {}
    # destination JSON stores keys as strings sometimes — coerce
    interest_labels: dict[int, str] = {}
    for k, v in interest_labels_raw.items():
        try:
            interest_labels[int(k)] = v
        except (TypeError, ValueError):
            continue
    if not interest_labels:
        interest_labels = DEFAULT_INTEREST_LABELS

    # Type display map: not currently provided by the API, but exposed
    # as a hook so downstream destinations can override per-type labels.
    type_display: dict[str, str] = (dest_data or {}).get("type_display") or {}

    # Normalise every POI
    normalised: list[dict] = []
    for raw in raw_pois:
        record = normalize_poi(raw, lang=lang, destination=destination,
                               tourist_type_display=tourist_type_display,
                               type_display=type_display,
                               interest_labels=interest_labels)
        if record["poi_id"]:
            normalised.append(record)

    # Name index — lossy on collisions, but at 367 POIs collisions are <2%
    name_index: dict[str, str] = {}
    for p in normalised:
        norm = p["normalized_name"]
        if norm and norm not in name_index:
            name_index[norm] = p["poi_id"]
    # Group into sections + summarise
    sections = assemble_sections(normalised)
    # Materialise the per-POI dictionary
    pois_by_id = {p["poi_id"]: p for p in normalised}
    # Facets
    facets = build_facets(normalised, sections)
    # Curated itinerary records resolve waypoint names against this
    # language's exact name index.
    trips, paths = build_itineraries(dest_data, name_index)

    destination_display = ""
    if dest_data and dest_data.get("destination", {}).get("name"):
        destination_display = dest_data["destination"]["name"]
    if not destination_display:
        destination_display = destination.title()

    return {
        "meta": {
            "destination":         destination,
            "destination_display": destination_display,
            "lang":                lang,
            "generated_at":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "poi_count":           len(normalised),
            "section_count":       len(sections),
            # v4: schema-v3 evidence search plus resolved curated trips
            # and paths. Older readers can ignore the additions.
            "schema_version":      4,
        },
        "destination_overview": build_destination_overview(dest_data, tourist_type_display),
        "trips":                trips,
        "paths":                paths,
        "sections":             sections,
        "pois":                 pois_by_id,
        "facets":               facets,
        "name_index":           name_index,
        "tourist_type_display": tourist_type_display,
        "interest_levels":      {str(k): v for k, v in interest_labels.items()},
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build POI-aware index from API JSON")
    parser.add_argument("--destination", default=DEFAULT_DESTINATION,
                        help=f"Tourist destination slug (default: {DEFAULT_DESTINATION})")
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE,
                        help=(f"Language code (default: {DEFAULT_LANGUAGE}). "
                              f"One of: {', '.join(SUPPORTED_LANGS)}"))
    parser.add_argument("--output", default=None,
                        help="Override output path (default: indexes/{dest}_{lang}.json)")
    args = parser.parse_args()

    if not is_supported(args.lang):
        print(f"[ERROR] Unsupported --lang '{args.lang}'. "
              f"Supported codes: {', '.join(SUPPORTED_LANGS)}",
              file=sys.stderr)
        sys.exit(1)

    pois_file = PROJECT_ROOT / "data" / f"{args.destination}_pois_raw_{args.lang}.json"
    dest_file = PROJECT_ROOT / "data" / f"{args.destination}_destination_{args.lang}.json"

    if not pois_file.exists():
        print(f"[ERROR] POI file not found: {pois_file}", file=sys.stderr)
        print(f"[ERROR] Run: pipeline/extract_pois.py --destination {args.destination} --lang {args.lang}",
              file=sys.stderr)
        sys.exit(1)

    with open(pois_file, encoding="utf-8") as f:
        raw_pois = json.load(f)
    if not isinstance(raw_pois, list) or not raw_pois:
        print(f"[ERROR] {pois_file} is not a non-empty array", file=sys.stderr)
        sys.exit(1)

    dest_data = None
    if dest_file.exists():
        with open(dest_file, encoding="utf-8") as f:
            dest_data = json.load(f)
    else:
        print(f"[WARN] No destination file at {dest_file} — output will be sparser",
              file=sys.stderr)

    print(f"[INFO] Destination: {args.destination}  Language: {args.lang}")
    print(f"[INFO] Loaded {len(raw_pois)} POIs from {pois_file.name}")

    index = build_index(raw_pois, dest_data, lang=args.lang, destination=args.destination)

    output = Path(args.output) if args.output \
             else PROJECT_ROOT / "indexes" / f"{args.destination}_{args.lang}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Sections ({len(index['sections'])}):")
    for s in index["sections"]:
        print(f"  {s['title']:50s}  {len(s['poi_ids']):>3} POIs")

    size_kb = output.stat().st_size / 1024
    print(f"\n[INFO] Saved → {output}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
json_to_markdown.py — Convert ubeda_pois_raw.json into a structured
Markdown document suitable for PageIndex tree-indexing.

PageIndex uses '#' heading hierarchy to build its retrieval tree.
Each UNE 178503 type group becomes a '##' section; each individual
POI becomes a '###' entry.  POIs are sorted by composite key
(id_interest_level, zoom_level) so the most important and most visible
POIs appear first within each section.

If data/ubeda_destination.json exists (produced by extract_destination_data.py),
two additional top-level sections are prepended:
  ## Destination Overview
  ## Curated Trips and Itineraries

Inputs: data/ubeda_pois_raw.json, data/ubeda_destination.json (optional)
Output: data/ubeda_guide.md
"""

import json
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
INPUT_FILE      = PROJECT_ROOT / "data" / "ubeda_pois_raw.json"
DESTINATION_FILE = PROJECT_ROOT / "data" / "ubeda_destination.json"
OUTPUT_FILE     = PROJECT_ROOT / "data" / "ubeda_guide.md"

# Map-prominence threshold: POIs with zoom_level <= this value are labelled
# as major landmarks in the Markdown.
PROMINENCE_ZOOM_MAX = 16

# Section definitions: each entry is (section_heading, {type_strings}).
# Priority is top-to-bottom: the first matching section wins.
SECTIONS = [
    ("UNESCO World Heritage and City Overview",
     {"WorldHeritageSite", "City"}),
    # Accommodation comes before Civil Monuments so that POIs with both
    # Hotel and CivilBuilding types (e.g. the Condestable Dávalos Parador)
    # are classified as accommodation, not as a monument.
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

SECTION_ORDER = {heading: idx for idx, (heading, _) in enumerate(SECTIONS)}
OTHER_SECTION = "Other Points of Interest"


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_text(field) -> str:
    """Extract plain string from a localised value list or a raw string."""
    if not field:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and item.get("language") == "en":
                return item.get("value", "")
        # Fall back to first entry if no English entry
        first = field[0]
        if isinstance(first, dict):
            return first.get("value", "")
        return str(first)
    return str(field)


def get_list_text(field) -> list[str]:
    """Extract a list of plain strings from a field that may be a list."""
    if not field:
        return []
    if isinstance(field, str):
        return [field]
    return [str(item) for item in field if item]


def assign_section(poi_types: list[str]) -> str:
    """Return the section heading for a POI given its type list."""
    type_set = set(poi_types)
    for heading, section_types in SECTIONS:
        if type_set & section_types:
            return heading
    return OTHER_SECTION


def interest_level(poi: dict) -> tuple[int, int]:
    """Return (interest_level, zoom_level) composite sort key.

    Both dimensions are ascending: lower = more important/visible.
    Sorting by this tuple puts Indispensable, high-visibility POIs first.
    """
    extras = poi.get("extras") or {}
    il   = extras.get("id_interest_level") or 99
    zoom = extras.get("zoom_level")         or 99
    return (il, zoom)


def format_poi(poi: dict, tourist_type_map: dict | None = None) -> str:
    """Render a single POI as a Markdown '###' block."""
    name        = get_text(poi.get("name")) or "(Unnamed)"
    description = get_text(poi.get("description"))
    types       = get_list_text(poi.get("type"))
    address     = poi.get("streetAddress") or ""
    locality    = poi.get("addressLocality") or ""
    province    = poi.get("addressProvince") or ""
    phones      = get_list_text(poi.get("telephone"))
    emails      = get_list_text(poi.get("email"))
    urls        = get_list_text(poi.get("url"))
    t_types_raw = get_list_text(poi.get("touristType"))
    start_date  = poi.get("startDate") or ""
    end_date    = poi.get("endDate") or ""
    identifier  = poi.get("identifier") or ""
    extras      = poi.get("extras") or {}
    il          = extras.get("id_interest_level")
    zoom        = extras.get("zoom_level")

    # Translate tourist type codes to human-readable names
    if tourist_type_map and t_types_raw:
        t_types = [tourist_type_map.get(t, t).title() for t in t_types_raw]
    else:
        t_types = t_types_raw

    lines = [f"### {name}"]

    # Interest-level label (Indispensable only — makes must-see status explicit)
    if il == 1:
        lines.append("- **Interest level**: Indispensable")

    # Map-prominence label for major landmarks (zoom <= threshold)
    if zoom is not None and zoom <= PROMINENCE_ZOOM_MAX:
        lines.append(f"- **Map prominence**: Major landmark (zoom {zoom})")

    # Bullet metadata block
    if types:
        lines.append(f"- **Type**: {', '.join(types)}")
    location_parts = [p for p in [address, locality, province] if p]
    if location_parts:
        lines.append(f"- **Address**: {', '.join(location_parts)}")
    if phones:
        lines.append(f"- **Phone**: {', '.join(phones)}")
    if emails:
        lines.append(f"- **Email**: {', '.join(emails)}")
    if urls:
        lines.append(f"- **Website**: {', '.join(urls)}")
    if t_types:
        lines.append(f"- **Tourism interest**: {', '.join(t_types)}")
    if start_date:
        lines.append(f"- **Start date**: {start_date}")
    if end_date:
        lines.append(f"- **End date**: {end_date}")
    if identifier:
        lines.append(f"- **ID**: {identifier}")

    # Description paragraph
    if description:
        lines.append("")
        lines.append(description)

    lines.append("")
    return "\n".join(lines)


def bucket_pois(pois: list[dict]) -> dict[str, list[dict]]:
    """Group POIs into section buckets, each sorted by composite key."""
    buckets: dict[str, list[dict]] = {}
    for poi in pois:
        types   = get_list_text(poi.get("type"))
        section = assign_section(types)
        buckets.setdefault(section, []).append(poi)

    # Sort by (interest_level, zoom_level): lower = more important / visible
    for section in buckets:
        buckets[section].sort(key=interest_level)

    return buckets


# ── Destination-level sections ────────────────────────────────────────────

def load_destination_data() -> dict | None:
    """Load data/ubeda_destination.json if it exists, else return None."""
    if not DESTINATION_FILE.exists():
        return None
    with open(DESTINATION_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def format_destination_overview(dest: dict, tourist_type_map: dict) -> str:
    """Render the ## Destination Overview section."""
    lines = ["## Destination Overview", ""]
    d = dest["destination"]
    # Lead with UNESCO status including the year — critical for retrieval accuracy
    lines.append(
        "Úbeda is a **UNESCO World Heritage City** (designated in **2003**), "
        "located in Andalusia, Spain. It is renowned for its outstanding "
        "Renaissance architecture and monumental heritage."
    )
    lines.append("")
    if d.get("description"):
        lines.append(d["description"])
        lines.append("")
    # Tourist types (human-readable)
    types = [tourist_type_map.get(t, t).title() for t in d.get("tourist_types", [])]
    if types:
        lines.append(f"- **Tourism types**: {', '.join(types)}")
    if d.get("tourist_networks"):
        lines.append(f"- **Networks**: {', '.join(d['tourist_networks'])}")
    if d.get("official_url"):
        lines.append(f"- **Official website**: {d['official_url']}")
    lines.append("")
    return "\n".join(lines)


def format_trips_section(dest: dict, tourist_type_map: dict) -> str:
    """Render the ## Curated Trips and Itineraries section."""
    # Only include trips with at least one itinerary step that has POIs,
    # or at least descriptive steps.
    useful = [
        t for t in dest.get("trips", [])
        if t.get("itinerary") and
        any(s.get("pois") or s.get("step") for s in t["itinerary"])
    ]
    if not useful:
        return ""

    lines = [
        "## Curated Trips and Itineraries",
        "",
        f"Úbeda offers {len(useful)} officially curated themed tours and itineraries.",
        "",
    ]

    for trip in useful:
        name = trip.get("name", "?")
        lines.append(f"### {name}")
        if trip.get("description"):
            lines.append(f"- **Description**: {trip['description']}")
        if trip.get("url"):
            lines.append(f"- **More info**: {trip['url']}")
        lines.append("")
        for step in trip.get("itinerary", []):
            step_name = step.get("step", "")
            pois = step.get("pois", [])
            if step_name:
                lines.append(f"**{step_name}**")
            if pois:
                for poi in pois:
                    lines.append(f"- {poi}")
            elif step_name:
                lines.append("_(no specific POIs listed)_")
        lines.append("")

    return "\n".join(lines)


def build_markdown(pois: list[dict], dest: dict | None = None) -> str:
    """Assemble the complete Markdown document."""
    # Pre-condition: pois is a non-empty list
    if not pois:
        raise ValueError("POI list is empty")

    tourist_type_map = (dest or {}).get("tourist_types", {})
    buckets = bucket_pois(pois)

    # Order sections by the SECTIONS priority list; Other goes last
    ordered_sections = sorted(
        buckets.keys(),
        key=lambda s: SECTION_ORDER.get(s, len(SECTIONS)),
    )

    lines = [
        "# Úbeda Tourism Guide",
        "",
        "Úbeda is a UNESCO World Heritage City in Andalusia, Spain, renowned"
        " for its outstanding Renaissance architecture. This guide covers"
        f" {len(pois)} points of interest across the destination.",
        "",
    ]

    # Destination-level sections (prepended if data available)
    if dest:
        overview = format_destination_overview(dest, tourist_type_map)
        if overview:
            lines.append(overview)
        trips_section = format_trips_section(dest, tourist_type_map)
        if trips_section:
            lines.append(trips_section)

    for section in ordered_sections:
        section_pois = buckets[section]
        lines.append(f"## {section}")
        lines.append("")
        for poi in section_pois:
            lines.append(format_poi(poi, tourist_type_map))

    # Post-condition: at least one section was written
    assert any(line.startswith("## ") for line in lines), \
        "No sections written — check section mapping"

    return "\n".join(lines)


def save_markdown(content: str, output_path: Path) -> None:
    """Write the Markdown content to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def print_summary(buckets: dict[str, list]) -> None:
    """Print a section-by-section count."""
    total = sum(len(v) for v in buckets.values())
    print(f"\n[SUMMARY] Total POIs: {total}")
    print("[SUMMARY] Sections:")
    ordered = sorted(buckets.keys(), key=lambda s: SECTION_ORDER.get(s, 999))
    for section in ordered:
        print(f"  {section:50s}  {len(buckets[section]):4d} POIs")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Load JSON, convert to Markdown, save."""
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input not found: {INPUT_FILE}", file=sys.stderr)
        print("[ERROR] Run scripts/extract_ubeda.py first.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as fh:
        pois = json.load(fh)

    if not isinstance(pois, list) or len(pois) == 0:
        print("[ERROR] Expected a non-empty JSON array", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(pois)} POIs from {INPUT_FILE}")

    dest = load_destination_data()
    if dest:
        n_trips = len([t for t in dest.get("trips", []) if t.get("itinerary")])
        print(f"[INFO] Loaded destination data ({n_trips} trips, "
              f"{len(dest.get('tourist_types', {}))} type codes)")
    else:
        print("[INFO] No destination data — run extract_destination_data.py for richer output")

    content = build_markdown(pois, dest)
    buckets = bucket_pois(pois)
    print_summary(buckets)

    save_markdown(content, OUTPUT_FILE)

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n[INFO] Saved → {OUTPUT_FILE}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

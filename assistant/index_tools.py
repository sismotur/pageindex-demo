#!/usr/bin/env python3
"""
assistant/index_tools.py — Read-side helpers for the POI-aware index.

Pure functions over the dict produced by pipeline/build_index.py.  No I/O
at import, no LLM calls, no global state.  Imported by run_eval.py and
chat_demo.py to back the five LLM tools:

    list_sections()                — embedded into the system prompt
    get_section(section_id, ...)   — list POIs in a section
    get_poi(poi_id)                — full record of one POI
    find_poi_by_name(query, ...)   — fuzzy lookup by name
    filter_pois(**facets)          — facet query (interest_level, type, ...)

All `format_*` functions return strings suitable for tool-call results;
all `index_*` functions return raw structures used internally by the
formatters and by tests.

Text normalisation lives in common/textnorm.py — it is shared with
pipeline/build_index.py and must stay identical across the Cloudflare
and mobile ports (see docs/mobile-offline-contract.md).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# Allow `from common.textnorm import ...` when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.textnorm import normalize_text, tokenize  # noqa: E402, F401

# ── I/O ─────────────────────────────────────────────────────────────────────

def load_index(path: str | Path) -> dict:
    """Read the index JSON from disk and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Section listing ─────────────────────────────────────────────────────────

def format_sections_overview(index: dict) -> str:
    """Render the sections list with deterministic summaries.

    This output is embedded into the system prompt at startup, so the
    model never needs to call list_sections() at runtime.
    """
    meta = index.get("meta", {})
    dest = meta.get("destination_display") or meta.get("destination", "")
    poi_count = meta.get("poi_count", 0)
    sections = index.get("sections", [])

    lines = [
        f"Destination: {dest}  ({poi_count} POIs across {len(sections)} sections)",
        "",
        "SECTIONS:",
    ]
    for sec in sections:
        sid = sec.get("section_id", "?")
        title = sec.get("title", "?")
        n = len(sec.get("poi_ids") or [])
        summary = sec.get("summary", "").strip()
        # Context reduction: the "Top interests:" sentence is facet data the
        # model can discover via filter_pois; counts + notable names are what
        # drive navigation.  The index file keeps the full summary.
        if summary:
            parts = [p for p in summary.split(". ")
                     if not p.startswith("Top interests:")]
            summary = ". ".join(parts)
        lines.append(f"  [{sid}] {title}  ({n} POIs)")
        if summary:
            lines.append(f"      {summary}")
    return "\n".join(lines)


def section_titles(index: dict) -> list[str]:
    """Return a list of section titles in display order."""
    return [s.get("title", "") for s in index.get("sections", [])]


def section_ids(index: dict) -> list[str]:
    """Return a list of section IDs in display order."""
    return [s.get("section_id", "") for s in index.get("sections", [])]


def find_section(index: dict, key: str) -> dict | None:
    """Find a section by ID or by case-insensitive title match.

    Tolerant lookup: tries exact id, then exact title, then substring title,
    then normalised title.  Returns the section dict or None.
    """
    if not key:
        return None
    sections = index.get("sections", [])
    # Exact section_id
    for s in sections:
        if s.get("section_id") == key:
            return s
    # Exact title (case-insensitive)
    key_lower = key.lower()
    for s in sections:
        if s.get("title", "").lower() == key_lower:
            return s
    # Substring title
    for s in sections:
        if key_lower in s.get("title", "").lower():
            return s
    # Normalised title (drops diacritics and punctuation)
    key_norm = normalize_text(key)
    for s in sections:
        if key_norm == normalize_text(s.get("title", "")):
            return s
    return None


# ── POI access ──────────────────────────────────────────────────────────────

def get_poi(index: dict, poi_id: str) -> dict | None:
    """Return the full POI record by ID, or None if missing.

    Accepts both the raw 'poi/5155' format and the bare numeric '5155'
    suffix so the model can use whichever it remembered.
    """
    if poi_id is None:
        return None
    pois = index.get("pois", {})
    poi_id = str(poi_id).strip()
    if poi_id in pois:
        return pois[poi_id]
    # Try with 'poi/' prefix added
    prefixed = f"poi/{poi_id}"
    if prefixed in pois:
        return pois[prefixed]
    # Try without the 'poi/' prefix (rare)
    if poi_id.startswith("poi/"):
        bare = poi_id[len("poi/"):]
        if bare in pois:
            return pois[bare]
    return None


def get_pois(index: dict, poi_ids: Iterable[str]) -> list[dict | None]:
    """Batch variant of get_poi: one record (or None) per requested id."""
    return [get_poi(index, pid) for pid in poi_ids]


# ── POI tags in answers ─────────────────────────────────────────────────────
#
# The system prompt instructs the model to wrap every POI mention in an
# inline tag carrying the POI id it saw in the tool results:
#
#     <poi id=5155>Church of San Nicolás</poi>
#
# The app's text parser catches the tag, displays the inner text as a
# tappable link, and opens the POI by id — no name matching involved.
# Android: PointOfInterestActivity takes the bare numeric id as the
# "poiId" intent extra (PlacesFragment: putExtra("poiId", id.drop(4))).
#
# The parser is deliberately lenient (accepts optional quotes and the
# 'poi/' prefix the model sees in tool output), but the contract is:
# bare numeric id, no quotes, tag wraps the display text.

# POI tag format (v2): <poi id=5155 type=Restaurant>name</poi>
# The `type` attribute is optional for backward compat; the parser accepts
# any attribute order and optional quotes.  The regex captures:
#   group 1 — bare numeric id
#   group 2 — UNE type code (optional, may be absent in old answers)
#   group 3 — inner display text
POI_TAG_RE = re.compile(
    r"<poi\b(?=[^>]*\bid\s*=\s*\"?(?:poi/)?(\d+)\"?)"
    r"(?=[^>]*)(?:[^>]*\btype\s*=\s*\"?(\w+)\"?)?[^>]*>(.*?)</poi>",
    re.IGNORECASE | re.DOTALL,
)

# Lenient fallback: true self-closing tags (<poi id=5149/>) carry no
# display text. Small models sometimes emit these; the parser resolves the
# display name from the index by id. Requiring '/>' avoids matching the
# opening half of a valid full tag.
POI_TAG_EMPTY_RE = re.compile(
    r"<poi\s+(?:[^>]*?\s)?id\s*=\s*\"?(?:poi/)?(\d+)\"?[^>]*/>",
    re.IGNORECASE,
)
TRIP_TAG_RE = re.compile(
    r"<trip\b[^>]*\bid\s*=\s*\"?(?:trip/)?(\d+)\"?[^>]*>(.*?)</trip>",
    re.IGNORECASE | re.DOTALL,
)
PATH_TAG_RE = re.compile(
    r"<path\b[^>]*\bid\s*=\s*\"?(?:path/)?(\d+)\"?[^>]*>(.*?)</path>",
    re.IGNORECASE | re.DOTALL,
)
TRIP_TAG_OPEN_RE = re.compile(
    r"<trip\b[^>]*\bid\s*=\s*\"?(?:trip/)?(\d+)\"?[^>]*>",
    re.IGNORECASE,
)
PATH_TAG_OPEN_RE = re.compile(
    r"<path\b[^>]*\bid\s*=\s*\"?(?:path/)?(\d+)\"?[^>]*>",
    re.IGNORECASE,
)


def poi_uri(destination_slug: str, poi_id: str) -> str:
    """Canonical app/web URI for a POI: what the tag resolves to.

    'poi/5155' + 'ubeda' -> 'https://inventrip.com/ubeda/object/5155'
    Matches the app's existing app-link intent filter (https://inventrip.com/*).
    """
    bare = str(poi_id).split("/", 1)[-1]
    return f"https://inventrip.com/{destination_slug}/object/{bare}"


def extract_poi_tags(answer: str, index: dict) -> list[dict]:
    """Parse <poi id=…>…</poi> tags from an answer, in order of appearance.

    Returns one entry per unique id:
    {poi_id, text, known, name?, uri?} — `known=False` when the id is not
    in the index (the app should still show the inner text, unlinked).

    Empty/self-closing tags (<poi id=5149/>) are accepted: their `text`
    falls back to the POI's name from the index ('' when unknown).
    """
    dest = (index.get("meta") or {}).get("destination", "")
    answer = answer or ""

    # (start, end, bare_id, type_code|None, inner_text|None)
    matches: list[tuple[int, int, str, str | None, str | None]] = [
        (m.start(), m.end(), m.group(1), m.group(2), m.group(3))
        for m in POI_TAG_RE.finditer(answer)
    ]
    covered = [(ms, me) for ms, me, _, _, _ in matches]
    for m in POI_TAG_EMPTY_RE.finditer(answer):
        ms, me = m.start(), m.end()
        if any(ms >= cs and me <= ce for (cs, ce) in covered):
            continue  # part of a full tag already captured
        matches.append((ms, me, m.group(1), None, None))
    matches.sort(key=lambda m: m[0])

    refs: list[dict] = []
    seen: set[str] = set()
    for _, _, bare, type_code, inner in matches:
        pid = f"poi/{bare}"
        if pid in seen:
            continue
        seen.add(pid)
        p = get_poi(index, pid)
        text = (inner or "").strip() or (p.get("name", "") if p else "")
        ref: dict = {"poi_id": pid, "text": text, "known": p is not None}
        if type_code:
            ref["type_code"] = type_code
        elif p is not None and p.get("display_type"):
            ref["type_code"] = p["display_type"]   # fallback: from index
        if inner is None:
            ref["self_closing"] = True
        if p is not None:
            ref["name"] = p.get("name", "")
            if dest:
                ref["uri"] = poi_uri(dest, pid)
        refs.append(ref)
    return refs


def strip_poi_tags(answer: str) -> str:
    """Replace every poi tag with its inner display text (what the user sees).

    Full tags become their inner text (group 3 = text after id + optional type);
    empty/self-closing tags are removed.  Malformed fragments that match neither
    pattern pass through unchanged.
    """
    stripped = POI_TAG_RE.sub(lambda m: m.group(3), answer or "")
    return POI_TAG_EMPTY_RE.sub("", stripped)

def _poi_tag(poi: dict) -> str:
    """Return the canonical tag-ready POI mention for LLM tool results.

    The tag carries stable app-navigation data; the visible inner text
    remains the tourist-facing POI name. Catalog labels never need to
    appear in answer prose.
    """
    poi_id = str(poi.get("poi_id") or "")
    bare_id = poi_id.split("/", 1)[-1]
    raw_type = str(poi.get("display_type") or "Place")
    type_code = re.sub(r"[^\w]", "", raw_type) or "Place"
    name = poi.get("name") or "(unnamed)"
    return f"<poi id={bare_id} type={type_code}>{name}</poi>"
def _poi_tag_with_text(poi: dict, text: str) -> str:
    """Render canonical id/type while keeping a valid visible label."""
    poi_id = str(poi.get("poi_id") or "")
    bare_id = poi_id.split("/", 1)[-1]
    raw_type = str(poi.get("display_type") or "Place")
    type_code = re.sub(r"[^\w]", "", raw_type) or "Place"
    return f"<poi id={bare_id} type={type_code}>{text}</poi>"


def sanitize_poi_tags(answer: str, index: dict) -> str:
    """Keep only tags whose id exists in the downloaded index.

    This validates an ID the model supplied; it never searches names or
    guesses a replacement. Unknown tags become ordinary inner text.
    """
    def full_tag(match: re.Match) -> str:
        poi = get_poi(index, f"poi/{match.group(1)}")
        if poi is None:
            return match.group(3)
        return _poi_tag_with_text(poi, match.group(3))

    def empty_tag(match: re.Match) -> str:
        poi = get_poi(index, f"poi/{match.group(1)}")
        return _poi_tag(poi) if poi is not None else ""

    sanitized = POI_TAG_RE.sub(full_tag, answer or "")
    return POI_TAG_EMPTY_RE.sub(empty_tag, sanitized)


def _sanitize_collection_tags(answer: str, index: dict, *,
                              collection: str, prefix: str,
                              full_re: re.Pattern,
                              open_re: re.Pattern) -> str:
    """Validate and canonicalize trip/path tags against index records.

    A frequent small-model malformed form is:
      **RUTAS POR ÚBEDA** (<trip id=4420>)
    The bare known tag is repaired to a source-backed wrapped tag:
      <trip id=4420>RUTAS POR ÚBEDA</trip>
    Unknown ids become ordinary text and can never be selectable.
    """
    def item_for(bare: str) -> dict | None:
        return _find_curated(index, f"{prefix}/{bare}", collection)
    protected_tags: list[str] = []

    def protect(tag: str) -> str:
        protected_tags.append(tag)
        return f"__INVENTRIP_{collection.upper()}_TAG_{len(protected_tags) - 1}__"

    def full_tag(match: re.Match) -> str:
        item = item_for(match.group(1))
        if item is None:
            return match.group(2)
        tag = _trip_tag_with_text(item, match.group(2)) if collection == "trips" \
            else _path_tag_with_text(item, match.group(2))
        return protect(tag)

    sanitized = full_re.sub(full_tag, answer or "")

    # Repair the normal Markdown bold presentation before handling any
    # remaining dangling tag. The known source label wins over a model
    # rewording, so later follow-up selection is deterministic.
    bold_pattern = re.compile(
        rf"\*\*(?P<label>[^*\n]+)\*\*\s*\(\s*<{collection[:-1]}\b"
        rf"[^>]*\bid\s*=\s*\"?(?:{prefix}/)?(?P<id>\d+)\"?[^>]*>\s*\)",
        re.IGNORECASE,
    )

    def bold_tag(match: re.Match) -> str:
        item = item_for(match.group("id"))
        if item is None:
            return match.group("label")
        tag = _trip_tag(item) if collection == "trips" else _path_tag(item)
        return protect(tag)

    sanitized = bold_pattern.sub(bold_tag, sanitized)

    def dangling_tag(match: re.Match) -> str:
        item = item_for(match.group(1))
        if item is None:
            return ""
        tag = _trip_tag(item) if collection == "trips" else _path_tag(item)
        return protect(tag)

    sanitized = open_re.sub(dangling_tag, sanitized)
    for position, tag in enumerate(protected_tags):
        sanitized = sanitized.replace(
            f"__INVENTRIP_{collection.upper()}_TAG_{position}__", tag
        )
    return sanitized


def sanitize_itinerary_tags(answer: str, index: dict) -> str:
    """Validate full/dangling `<trip>` and `<path>` tags against the index."""
    sanitized = _sanitize_collection_tags(
        answer, index, collection="trips", prefix="trip",
        full_re=TRIP_TAG_RE, open_re=TRIP_TAG_OPEN_RE,
    )
    return _sanitize_collection_tags(
        sanitized, index, collection="paths", prefix="path",
        full_re=PATH_TAG_RE, open_re=PATH_TAG_OPEN_RE,
    )

def sanitize_tourist_answer(answer: str, index: dict) -> str:
    """Apply deterministic presentation rules to a final visitor answer.

    This is deliberately narrow: validate tag ids, then replace only
    catalog-language nouns that should never reach a tourist. It does not
    infer facts, change names, or alter the meaning of retrieved evidence.
    """
    sanitized = sanitize_poi_tags(answer, index)
    sanitized = sanitize_itinerary_tags(sanitized, index)
    protected_tags: list[str] = []

    def protect_tag(match: re.Match) -> str:
        protected_tags.append(match.group(0))
        return f"__INVENTRIP_POI_TAG_{len(protected_tags) - 1}__"

    # Do not apply prose replacements to the literal `poi` tag name or
    # its machine-readable attributes.
    sanitized = POI_TAG_RE.sub(protect_tag, sanitized)
    replacements = (
        (r"\bPOIs\b", "places"),
        (r"\bPOI\b", "place"),
        (r"\bpoints of interest\b", "places"),
        (r"\bpoint of interest\b", "place"),
    )
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    for position, tag in enumerate(protected_tags):
        sanitized = sanitized.replace(f"__INVENTRIP_POI_TAG_{position}__", tag)
    return sanitized



def _poi_section_title(index: dict, poi_id: str) -> str:
    """Return the section title that owns the given POI ID, or ''. """
    by_section = (index.get("facets") or {}).get("by_section") or {}
    for sec_id, ids in by_section.items():
        if poi_id in ids:
            sec = find_section(index, sec_id)
            if sec:
                return sec.get("title", "")
    return ""


# ── Formatting (tool-result text) ───────────────────────────────────────────

# Adaptive default limit for get_section: sections with a v2 group map only
# need a short top list (the groups + filter_pois carry the drill-down);
# flat sections keep the full listing for browse-style questions.
SECTION_LIMIT_FLAT = 50
SECTION_LIMIT_GROUPED = 20


def _short_preview(poi: dict, max_chars: int = 120) -> str:
    """One-line tourist-facing description preview (no catalog metadata)."""
    desc = (poi.get("description") or "").strip()
    if not desc:
        return ""
    sent_end = re.search(r"[.!?]\s", desc)
    snippet = desc[: sent_end.end()] if sent_end else desc[:90]
    if len(snippet) > 90:
        snippet = snippet[:90].rsplit(" ", 1)[0] + "…"
    return snippet.strip()


def format_section(index: dict, section_key: str,
                   sort: str = "interest", limit: int | None = None) -> str:
    """Render a section: title, summary, optional group map, then one
    line per POI.

    Schema v2: when the section carries `groups` (large sections split
    by type at build time), a group map is rendered between the summary
    and the preview list.  Each group line names its top POIs (the
    key-items pattern), and the drill-down path is the existing
    filter_pois(type=..., section_id=...) call.

    `limit=None` applies the adaptive default: SECTION_LIMIT_GROUPED (20)
    for grouped sections, SECTION_LIMIT_FLAT (50) otherwise.  An explicit
    limit always wins.
    """
    sec = find_section(index, section_key)
    if not sec:
        avail = ", ".join(s.get("title", "") for s in index.get("sections", []))
        return f"[ERROR] Section '{section_key}' not found. Available: {avail}"

    if limit is None:
        limit = SECTION_LIMIT_GROUPED if sec.get("groups") else SECTION_LIMIT_FLAT

    poi_ids = list(sec.get("poi_ids") or [])
    pois = [get_poi(index, pid) for pid in poi_ids]
    pois = [p for p in pois if p]

    # Sort
    if sort == "name":
        pois.sort(key=lambda p: normalize_text(p.get("name") or ""))
    elif sort == "zoom":
        pois.sort(key=lambda p: (p.get("zoom_level") or 99,
                                 normalize_text(p.get("name") or "")))
    else:  # default: by (interest_level, zoom_level) — most important first
        pois.sort(key=lambda p: (p.get("interest_level") or 99,
                                 p.get("zoom_level") or 99,
                                 normalize_text(p.get("name") or "")))

    truncated = False
    if limit and len(pois) > limit:
        pois = pois[:limit]
        truncated = True

    lines = [f"{sec.get('title')} — {len(sec.get('poi_ids') or [])} places"]
    if sec.get("summary"):
        lines.append(f"  {sec['summary']}")

    groups = sec.get("groups") or []
    if groups:
        lines.append("")
        lines.append("  Browse groups:")
        for g in groups:
            key_items = []
            for pid in (g.get("poi_ids") or [])[:3]:
                p = get_poi(index, pid)
                if p and p.get("name"):
                    key_items.append(p["name"])
            notable = f"  Notable: {', '.join(key_items)}" if key_items else ""
            lines.append(f"    {g.get('title')} — "
                         f"{len(g.get('poi_ids') or [])} places.{notable}")
    lines.append("")
    for p in pois:
        preview = _short_preview(p)
        if preview:
            lines.append(f"  {_poi_tag(p)} — {preview}")
        else:
            lines.append(f"  {_poi_tag(p)}")
    if truncated:
        lines.append(f"  …{len(sec.get('poi_ids') or []) - limit} more (raise --limit to see all)")
    return "\n".join(lines)


def _format_kv(label: str, value: Any) -> str | None:
    """Render '- **Label**: value' or None if the value is empty."""
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, list):
        # List of strings — comma-join
        rendered = ", ".join(str(v) for v in value if v)
        if not rendered:
            return None
        return f"- **{label}**: {rendered}"
    return f"- **{label}**: {value}"


def _format_single_poi(index: dict, p: dict) -> str:
    """Render one resolved POI record (helper for format_poi)."""

    section_title = _poi_section_title(index, p["poi_id"])

    lines = [f"# {_poi_tag(p)}"]
    if section_title:
        lines.append(f"*Section: {section_title}*")
    lines.append("")

    # Bullet metadata
    bullets: list[str] = []
    location_parts = [s for s in [p.get("street_address"),
                                  p.get("address_locality"),
                                  p.get("address_province")] if s]
    if location_parts:
        bullets.append(_format_kv("Address", ", ".join(location_parts)))
    bullets.append(_format_kv("Postal code", p.get("postal_code")))
    bullets.append(_format_kv("Country", p.get("country")))
    bullets.append(_format_kv("Region", p.get("address_region")))
    if p.get("latitude") is not None and p.get("longitude") is not None:
        bullets.append(f"- **Coordinates**: {p['latitude']:.6f}, {p['longitude']:.6f}")
    bullets.append(_format_kv("Phone", p.get("telephone")))
    bullets.append(_format_kv("Email", p.get("email")))
    bullets.append(_format_kv("Website", p.get("url")))
    bullets.append(_format_kv("Booking", p.get("booking_url")))
    if p.get("zoom_level") is not None and p["zoom_level"] <= 16:
        bullets.append(f"- **Map prominence**: Major landmark (zoom {p['zoom_level']})")
    bullets.append(_format_kv("Start date", p.get("start_date")))
    bullets.append(_format_kv("End date", p.get("end_date")))
    # NOTE: image/audio/document URLs are deliberately NOT rendered here —
    # the model cannot act on media URLs, and they cost ~13% of get_poi
    # tokens.  They stay in the index record for the app UI.

    for b in bullets:
        if b:
            lines.append(b)

    desc = (p.get("description") or "").strip()
    if desc:
        lines.append("")
        lines.append(desc)
    return "\n".join(lines)


def format_poi(index: dict, poi_id: str) -> str:
    """Render full POI record(s). No truncation, no line slicing.

    Accepts a single id ('poi/5155' or '5155') or several comma-separated
    ids ('poi/5155,poi/65804') — the batch form lets the model fetch
    several POIs in one tool call, saving LLM round-trips on comparison
    and synthesis questions.  Multiple records are joined with a
    '\\n\\n---\\n\\n' separator; unknown ids render an inline [ERROR] block
    without failing the whole batch.
    """
    ids = [part.strip() for part in str(poi_id).split(",") if part.strip()]
    if len(ids) <= 1:
        p = get_poi(index, poi_id)
        if not p:
            return (f"[ERROR] POI '{poi_id}' not found. "
                    f"Use find_poi_by_name() if you only know the name.")
        return _format_single_poi(index, p)

    blocks = []
    for pid in ids:
        p = get_poi(index, pid)
        if p:
            blocks.append(_format_single_poi(index, p))
        else:
            blocks.append(f"[ERROR] POI '{pid}' not found.")
    return "\n\n---\n\n".join(blocks)


# ── Name search ─────────────────────────────────────────────────────────────

def find_poi_by_name(index: dict, query: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` matching POIs as light dicts.

    Matching strategy (in order):
      1. exact normalised name match
      2. all query tokens present in normalised name (substring)
      3. any query token present in normalised name
    Higher-quality matches are returned first; ties broken by interest level.
    """
    q_norm = normalize_text(query)
    if not q_norm:
        return []
    q_tokens = set(q_norm.split())

    by_norm = (index.get("name_index") or {})  # {normalized_name: poi_id}
    pois = index.get("pois", {})

    # Tier 1: exact normalised match
    tier1: list[dict] = []
    if q_norm in by_norm:
        pid = by_norm[q_norm]
        if pid in pois:
            tier1.append(pois[pid])

    # Tier 2 + 3: scan all POIs (367 entries — trivial to iterate)
    tier2: list[tuple[int, dict]] = []  # (negative-score, poi)
    tier3: list[tuple[int, dict]] = []
    for pid, p in pois.items():
        norm = p.get("normalized_name") or normalize_text(p.get("name") or "")
        if not norm or norm == q_norm:
            continue  # already in tier 1
        n_tokens = set(norm.split())
        common = q_tokens & n_tokens
        if not common:
            continue
        if q_tokens.issubset(n_tokens) or q_norm in norm:
            tier2.append((-len(common), p))
        else:
            tier3.append((-len(common), p))

    # Sort each tier by token-overlap desc, then interest level asc
    tier2.sort(key=lambda x: (x[0], x[1].get("interest_level") or 99))
    tier3.sort(key=lambda x: (x[0], x[1].get("interest_level") or 99))

    out = list(tier1)
    for _, p in tier2:
        if p not in out:
            out.append(p)
    for _, p in tier3:
        if p not in out:
            out.append(p)
    return out[: max(1, limit)]


def format_find_poi_by_name(index: dict, query: str, limit: int = 5,
                            detail: str = "brief") -> str:
    """Render name-search results.

    detail="full" appends the best match's complete POI record after the
    candidate list, fusing the classic find_poi_by_name -> get_poi pair
    into one tool call (one fewer LLM round on lookup questions).
    """
    matches = find_poi_by_name(index, query, limit=limit)
    if not matches:
        return (f"[INFO] No POI matches '{query}'. "
                f"Try filter_pois() or browse a section with get_section().")
    lines = [f"Matches for '{query}' ({len(matches)} of up to {limit}):"]
    for p in matches:
        preview = _short_preview(p)
        if preview:
            lines.append(f"  {_poi_tag(p)} — {preview}")
        else:
            lines.append(f"  {_poi_tag(p)}")
    if detail == "full":
        lines.append("")
        lines.append("Best match, full record:")
        lines.append(_format_single_poi(index, matches[0]))
    return "\n".join(lines)


# ── Facet filter ────────────────────────────────────────────────────────────

def _resolve_facet_ids(index: dict, facet: str, value: Any) -> set[str] | None:
    """Resolve a facet value to a set of POI IDs, or None if unknown facet/value."""
    facets = index.get("facets") or {}
    if facet == "section_id":
        section = find_section(index, str(value))
        return set(section.get("poi_ids") or []) if section else set()
    if facet == "interest_level":
        # Accept 1/2/3 or labels
        try:
            iv = int(value)
            return set((facets.get("by_interest_level") or {}).get(str(iv), []))
        except (TypeError, ValueError):
            label = str(value).lower()
            label_map = {"indispensable": 1, "interesting": 2, "outstanding": 3}
            iv = label_map.get(label)
            if iv is None:
                return set()
            return set((facets.get("by_interest_level") or {}).get(str(iv), []))
    if facet == "indispensable":
        if value:
            return set(facets.get("indispensable") or [])
        return None  # falsey filter — ignore
    if facet == "type":
        return set((facets.get("by_type") or {}).get(str(value), []))
    if facet == "tourist_type":
        # Match against UNE codes (raw or display name normalised)
        by_tt = facets.get("by_tourist_type") or {}
        v = str(value).strip()
        if v in by_tt:
            return set(by_tt[v])
        # Try uppercase code form
        v_up = v.upper()
        if v_up in by_tt:
            return set(by_tt[v_up])
        # Try display-name reverse lookup
        v_norm = normalize_text(v)
        for code, ids in by_tt.items():
            if normalize_text(code) == v_norm:
                return set(ids)
        # Try via tourist_type_display map if present
        for code, label in (index.get("tourist_type_display") or {}).items():
            if normalize_text(label) == v_norm:
                return set(by_tt.get(code) or [])
        return set()
    return None


def filter_pois(index: dict, **filters: Any) -> list[dict]:
    """Intersect facet sets and return matching POI records."""
    pois = index.get("pois", {})
    candidate_ids: set[str] | None = None
    for facet, value in filters.items():
        if value is None or value == "":
            continue
        ids = _resolve_facet_ids(index, facet, value)
        if ids is None:
            continue
        candidate_ids = ids if candidate_ids is None else (candidate_ids & ids)
        if not candidate_ids:
            break
    if candidate_ids is None:
        # No filters supplied — refuse to return everything
        return []
    out = [pois[pid] for pid in candidate_ids if pid in pois]
    out.sort(key=lambda p: (p.get("interest_level") or 99,
                            p.get("zoom_level") or 99,
                            normalize_text(p.get("name") or "")))
    return out

# ── Full-text evidence search ───────────────────────────────────────────────

def _searchable_text(poi: dict) -> str:
    """Return normalized visitor-facing text indexed for one POI."""
    return normalize_text(" ".join([
        poi.get("name") or "",
        poi.get("description") or "",
        poi.get("display_type") or "",
        " ".join(poi.get("display_tourist_types") or []),
        poi.get("address_locality") or "",
    ]))


def _search_postings(index: dict, term: str) -> set[str]:
    """Resolve a normalized query term to its inverted-index postings.
    Plural variants handle harmless morphology (restaurant/restaurants)
    without a language-specific stemmer. A v2 index falls back to a
    small local scan during a staged corpus upgrade.
    """
    search_terms = ((index.get("facets") or {}).get("search_terms") or {})
    if not search_terms:
        return {
            pid for pid, poi in (index.get("pois") or {}).items()
            if term in set(tokenize(_searchable_text(poi)))
        }

    matched: set[str] = set()
    for variant in _term_variants(term):
        matched.update(search_terms.get(variant) or [])
    return matched


def _term_variants(term: str) -> set[str]:
    """Return conservative spelling variants for a single search term.

    This intentionally avoids general prefix matching: a query such as
    ``riverwalk`` must not match an unrelated ``river`` token. The three
    transformations cover the common singular/plural forms used in the
    catalogue while leaving other languages as exact lexical matches.
    """
    variants = {term}
    if len(term) >= 4 and term.endswith("ies"):
        variants.add(term[:-3] + "y")
    elif len(term) >= 4 and term.endswith("es"):
        variants.add(term[:-2])
    elif len(term) >= 4 and term.endswith("s"):
        variants.add(term[:-1])
    return variants


def _evidence_snippet(poi: dict, terms: list[str],
                      max_chars: int = 180) -> str:
    """Return a concise visitor-facing sentence supporting a search hit."""
    description = (poi.get("description") or "").strip()
    if not description:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", description)
    best = max(
        sentences,
        key=lambda sentence: sum(
            term in set(tokenize(sentence)) for term in terms
        ),
    )
    if len(best) > max_chars:
        best = best[:max_chars].rsplit(" ", 1)[0] + "…"
    return best


def search_pois(index: dict, query: str, section_id: str | None = None,
                limit: int = 10) -> list[dict]:
    """Find POIs whose same record explicitly matches every query term.

    This is deterministic lexical evidence search, not embeddings and not
    a category alias. An empty result means the catalogue does not support
    the requested combination on one POI; callers can search individual
    concepts separately for complementary options.
    """
    terms = [term for term in tokenize(query) if len(term) >= 3]
    if not terms:
        return []

    postings = [_search_postings(index, term) for term in terms]
    if any(not ids for ids in postings):
        return []
    candidate_ids = set.intersection(*postings)

    if section_id:
        section = find_section(index, section_id)
        if not section:
            return []
        candidate_ids &= set(section.get("poi_ids") or [])

    pois = index.get("pois") or {}
    matches = [pois[pid] for pid in candidate_ids if pid in pois]
    matches.sort(key=lambda poi: (
        poi.get("interest_level") or 99,
        poi.get("zoom_level") or 99,
        normalize_text(poi.get("name") or ""),
    ))
    return [
        {
            "poi": poi,
            "matched_terms": terms,
            "evidence": _evidence_snippet(poi, terms),
        }
        for poi in matches[:max(1, limit)]
    ]


def format_search_pois(index: dict, query: str, section_id: str | None = None,
                       limit: int = 10) -> str:
    """Render evidence-backed search results without catalog internals."""
    matches = search_pois(index, query, section_id=section_id, limit=limit)
    terms = [term for term in tokenize(query) if len(term) >= 3]
    if not matches:
        rendered = ", ".join(f'"{term}"' for term in terms)
        return (
            f"No place record explicitly mentions all of: {rendered}. "
            "Search the concepts separately for related visitor options; "
            "do not claim they are the same place."
        )

    lines = [f'Evidence-backed matches for "{query}" ({len(matches)}):']
    for item in matches:
        poi = item["poi"]
        evidence = item["evidence"]
        if evidence:
            lines.append(f"  {_poi_tag(poi)} — {evidence}")
        else:
            lines.append(f"  {_poi_tag(poi)}")
    return "\n".join(lines)
# ── Curated trips and physical paths ────────────────────────────────────────

def _trip_tag(trip: dict) -> str:
    """Return a curated-suggestion tag (not a physical route)."""
    bare_id = str(trip.get("itinerary_id") or "").split("/", 1)[-1]
    return f"<trip id={bare_id}>{trip.get('name') or '(unnamed trip)'}</trip>"
def _trip_tag_with_text(trip: dict, text: str) -> str:
    """Return a canonical trip tag while preserving a visible label."""
    bare_id = str(trip.get("itinerary_id") or "").split("/", 1)[-1]
    return f"<trip id={bare_id}>{text}</trip>"


def _path_tag(path: dict) -> str:
    """Return a physical-route tag sourced only from /v120/paths."""
    bare_id = str(path.get("itinerary_id") or "").split("/", 1)[-1]
    return f"<path id={bare_id}>{path.get('name') or '(unnamed route)'}</path>"
def _path_tag_with_text(path: dict, text: str) -> str:
    """Return a canonical path tag while preserving a visible label."""
    bare_id = str(path.get("itinerary_id") or "").split("/", 1)[-1]
    return f"<path id={bare_id}>{text}</path>"


def _itinerary_search_text(itinerary: dict, index: dict) -> str:
    """Build searchable visitor text without duplicating it in the JSON."""
    parts = [itinerary.get("name") or "", itinerary.get("description") or ""]
    for step in itinerary.get("steps") or []:
        parts.append(step.get("title") or "")
        parts.extend(step.get("unresolved_poi_names") or [])
        for poi_id in step.get("poi_ids") or []:
            poi = get_poi(index, poi_id)
            if poi:
                parts.append(poi.get("name") or "")
                parts.append(poi.get("description") or "")
                parts.append(poi.get("display_type") or "")
                parts.extend(poi.get("display_tourist_types") or [])
    return normalize_text(" ".join(parts))


def _itinerary_relevance(itinerary: dict, terms: list[str],
                         index: dict) -> int:
    """Score a curated item by visitor-facing query evidence.

    Unlike search_pois (which proves every property on the same POI),
    itinerary suggestions are editorial recommendations. A visitor phrase
    such as "food-focused" should find a trip whose resolved stops are
    labelled Food even if the literal adjective "focused" is absent.
    """
    haystack = set(tokenize(_itinerary_search_text(itinerary, index)))
    return sum(bool(_term_variants(term) & haystack) for term in terms)


def _search_curated(index: dict, query: str, collection: str,
                    limit: int) -> list[dict]:
    """Search one curated collection (trips or paths), never both."""
    terms = [term for term in tokenize(query) if len(term) >= 3]
    if not terms:
        return []
    scored = [
        (_itinerary_relevance(item, terms, index), item)
        for item in (index.get(collection) or [])
    ]
    matches = [item for score, item in scored if score > 0]
    matches.sort(key=lambda item: (
        -_itinerary_relevance(item, terms, index),
        normalize_text(item.get("name") or ""),
        item.get("itinerary_id") or "",
    ))
    return matches[:max(1, limit)]


def search_trips(index: dict, query: str, limit: int = 10) -> list[dict]:
    """Find editorial suggestions for what to do; never returns paths."""
    return _search_curated(index, query, "trips", limit)


def search_paths(index: dict, query: str, limit: int = 10) -> list[dict]:
    """Find physical walking/biking routes from /v120/paths; never trips."""
    return _search_curated(index, query, "paths", limit)


def _find_curated(index: dict, itinerary_id: str,
                  collection: str) -> dict | None:
    """Resolve a full or bare id within one collection only."""
    raw = str(itinerary_id or "").strip()
    if not raw:
        return None
    for item in index.get(collection) or []:
        if item.get("itinerary_id") == raw:
            return item
    bare = raw.split("/", 1)[-1]
    matches = [
        item for item in index.get(collection) or []
        if str(item.get("itinerary_id") or "").split("/", 1)[-1] == bare
    ]
    return matches[0] if len(matches) == 1 else None


def get_trip(index: dict, trip_id: str) -> dict | None:
    """Return one curated suggestion by full or bare trip id."""
    return _find_curated(index, trip_id, "trips")


def get_path(index: dict, path_id: str) -> dict | None:
    """Return one physical route by full or bare path id."""
    return _find_curated(index, path_id, "paths")


def resolve_trip_query(question: str, index: dict) -> dict | None:
    """Resolve a direct user reference to a known curated trip title.

    A title such as “RUTAS POR ÚBEDA” contains a route-like word but is an
    editorial trip, not a physical path. Exact or contained source-title
    matches take precedence over generic route intent. Ambiguities are
    deliberately rejected.
    """
    query = normalize_text(question)
    if len(query) < 4:
        return None
    matches = []
    for trip in index.get("trips") or []:
        name = normalize_text(trip.get("name") or "")
        if len(name) >= 4 and (query == name or name in query):
            matches.append(trip)
    if len(matches) != 1:
        return None
    trip = matches[0]
    return {
        "kind": "trip",
        "id": trip["itinerary_id"],
        "label": trip.get("name") or "",
    }


_BARE_ID_TOKEN_RE = re.compile(r"^\d{3,6}$")


def resolve_history_selection(question: str, messages: list[dict],
                              index: dict) -> dict | None:
    """Resolve a concise follow-up against validated prior assistant tags.

    Example: after the assistant offers
    `<trip id=4453>Ú. en Familia-R. Secundaria 2</trip>`, the user can
    say “Secundaria 2”. This returns a validated source selection:
    `{kind: "trip", id: "trip/4453", label: "…"}`.

    Matching is deliberately conservative: a unique normalized substring,
    an all-token match against a shown label, or a bare numeric id token
    that matches a shown tag id. Ambiguous references return None so the
    grounding gate asks the model to retrieve rather than guessing.
    """
    query = normalize_text(question)
    query_tokens = set(query.split())
    if len(query) < 3 or not query_tokens:
        return None

    candidates: list[dict] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        for ref in extract_poi_tags(content, index):
            if ref.get("known"):
                candidates.append({
                    "kind": "poi",
                    "id": ref["poi_id"],
                    "label": ref.get("text") or ref.get("name") or "",
                })
        for match in TRIP_TAG_RE.finditer(content):
            item = get_trip(index, f"trip/{match.group(1)}")
            if item:
                candidates.append({
                    "kind": "trip",
                    "id": item["itinerary_id"],
                    "label": match.group(2).strip() or item.get("name", ""),
                })
        for match in PATH_TAG_RE.finditer(content):
            item = get_path(index, f"path/{match.group(1)}")
            if item:
                candidates.append({
                    "kind": "path",
                    "id": item["itinerary_id"],
                    "label": match.group(2).strip() or item.get("name", ""),
                })

    # Keep one candidate per source id.
    unique = {(item["kind"], item["id"]): item for item in candidates}
    if not unique:
        return None

    # Bare-id follow-up: a numeric token uniquely matching a shown tag id
    # opens that record deterministically (e.g. user types "4457").
    id_matches: list[dict] = []
    for token in query_tokens:
        if not _BARE_ID_TOKEN_RE.match(token):
            continue
        for item in unique.values():
            item_bare = str(item["id"]).split("/", 1)[-1]
            if item_bare == token and item not in id_matches:
                id_matches.append(item)
    if len(id_matches) == 1:
        return id_matches[0]
    if len(id_matches) > 1:
        return None

    scored: list[tuple[int, dict]] = []
    for item in unique.values():
        label = normalize_text(item["label"])
        label_tokens = set(label.split())
        if query in label:
            scored.append((100 + len(query), item))
        elif query_tokens.issubset(label_tokens):
            scored.append((50 + len(query_tokens), item))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = scored[0]
    if len(scored) > 1 and scored[1][0] == best_score:
        return None
    return best

def _curated_preview(item: dict, max_chars: int = 180) -> str:
    """Trim a curated trip/path description for search output."""
    description = (item.get("description") or "").strip()
    if len(description) > max_chars:
        description = description[:max_chars].rsplit(" ", 1)[0] + "…"
    return description


def _format_curated_search(index: dict, query: str, collection: str,
                           label: str, limit: int) -> str:
    """Format one semantically distinct collection without raw internals."""
    matches = _search_curated(index, query, collection, limit)
    if not matches:
        return f"No curated {label.lower()} matched this request."
    tag = _trip_tag if collection == "trips" else _path_tag
    lines = [f'Curated {label.lower()} for "{query}" ({len(matches)}):']
    for item in matches:
        stops = len(item.get("steps") or [])
        preview = _curated_preview(item)
        suffix = f" — {stops} stops"
        if preview:
            suffix += f". {preview}"
        lines.append(f"  {tag(item)}{suffix}")
    return "\n".join(lines)


def format_search_trips(index: dict, query: str, limit: int = 10) -> str:
    """Render editorial suggestions for what to do; never a physical route."""
    return _format_curated_search(index, query, "trips", "Trip suggestions", limit)


def format_search_paths(index: dict, query: str, limit: int = 10) -> str:
    """Render physical walking/biking routes from /v120/paths only."""
    return _format_curated_search(index, query, "paths", "Routes", limit)


# Localized wrappers for the multi-trip choice offer. The offer is
# rendered deterministically by the runtime when the visitor asks for a
# plan/itinerary and several curated trips could match. English is used
# as the safe fallback for languages not explicitly listed.
_TRIP_CHOICE_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "lead":       "Here are a few curated trips that could match your request:",
        "highlights": "Highlights",
        "outro":      "Tell me the name or number of the trip you would like to see.",
    },
    "es": {
        "lead":       "He encontrado varias sugerencias que podrían encajar con tu petición:",
        "highlights": "Destacan",
        "outro":      "Dime el nombre o el número del viaje que prefieras.",
    },
    "it": {
        "lead":       "Ho trovato alcune proposte curate che potrebbero corrispondere:",
        "highlights": "In evidenza",
        "outro":      "Dimmi il nome o il numero del viaggio che preferisci.",
    },
}


def _headline_trip_pois(item: dict, index: dict, count: int = 3) -> list[str]:
    """Return the first `count` resolved POI names in trip reading order.

    Walks the flat `steps[].poi_ids` projection (which is populated in
    reading order even for nested schema-v6 items) so the caller sees the
    same headline POIs a visitor would encounter first.
    """
    if count <= 0:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for step in item.get("steps") or []:
        for pid in step.get("poi_ids") or []:
            if pid in seen:
                continue
            poi = get_poi(index, pid)
            if poi and poi.get("name"):
                names.append(poi["name"])
                seen.add(pid)
                if len(names) >= count:
                    return names
    return names


def format_trip_choice_offer(index: dict, matches: list[dict]) -> str:
    """Deterministic multi-trip choice offer with headline POIs.

    Emits one `<trip id=...>` tag per candidate, a short description, and
    up to three headline POIs so the visitor can compare options without
    the runtime committing to a single trip.  The follow-up selection
    resolves via `resolve_history_selection` (unique substring or bare
    numeric id).
    """
    if not matches:
        return ""
    lang = (index.get("meta") or {}).get("lang") or "en"
    msgs = _TRIP_CHOICE_MESSAGES.get(lang, _TRIP_CHOICE_MESSAGES["en"])
    lines = [msgs["lead"], ""]
    for item in matches:
        tag = _trip_tag(item)
        parts: list[str] = []
        preview = _curated_preview(item, max_chars=140)
        if preview:
            # `_curated_preview` may already trim with a trailing "…";
            # avoid "…." by only adding a period on complete sentences.
            if preview.endswith((".", "…", "!", "?")):
                parts.append(preview)
            else:
                parts.append(preview + ".")
        headline = _headline_trip_pois(item, index, count=3)
        if headline:
            parts.append(f"{msgs['highlights']}: {', '.join(headline)}.")
        summary = " ".join(parts).strip()
        lines.append(f"  - {tag}{f' — {summary}' if summary else ''}")
    lines.extend(["", msgs["outro"]])
    return "\n".join(lines)


def _render_curated_items(items: list[dict], index: dict,
                          lines: list[str], depth: int) -> None:
    """Recursively render a nested item tree (schema v6).

    Folders are shown with an indented dash label so the visitor sees the
    editorial structure; POIs render as tag-ready mentions so the mobile
    parser can deep-link them.  Unresolved items are QA-only and hidden.
    """
    indent = "   " * depth
    for item in items or []:
        kind = item.get("kind")
        if kind == "folder":
            name = item.get("name") or ""
            if name:
                lines.append(f"{indent}- {name}")
            _render_curated_items(
                item.get("items") or [], index, lines, depth + 1
            )
        elif kind == "poi":
            poi_id = item.get("poi_id") or ""
            poi = get_poi(index, poi_id) if poi_id else None
            if poi:
                lines.append(f"{indent}- {_poi_tag(poi)}")
        # "unresolved" is intentionally skipped: source labels may be
        # stale or foreign-language and must not appear as tourist text.


def _format_curated_detail(index: dict, itinerary_id: str, collection: str,
                           tag_builder, unavailable: str) -> str:
    """Render ordered trip/path stops, preserving unlinked source names."""
    item = _find_curated(index, itinerary_id, collection)
    if not item:
        return unavailable
    lines = [f"# {tag_builder(item)}"]
    description = (item.get("description") or "").strip()
    if description:
        lines.extend(["", description])
    steps = item.get("steps") or []
    if not steps:
        lines.extend(["", "No ordered stops are available for this item."])
        return "\n".join(lines)
    lines.append("")
    for step in steps:
        position = step.get("position") or "?"
        title = step.get("title") or f"Stop {position}"
        lines.append(f"{position}. {title}")
        step_items = step.get("items")
        if step_items:
            # Schema v6 nested items keep folder/POI hierarchy visible.
            _render_curated_items(step_items, index, lines, depth=1)
            continue
        # Legacy flat shape (pre-v6): show subfolder labels then POIs.
        for subfolder in step.get("subfolders") or []:
            lines.append(f"   - {subfolder}")
        for poi_id in step.get("poi_ids") or []:
            poi = get_poi(index, poi_id)
            if poi:
                lines.append(f"   - {_poi_tag(poi)}")
        # `unresolved_poi_names` is retained in the index for QA only.
        # It may be a stale or foreign-language source label, so never
        # present it as a tourist-visible or app-openable place.
    return "\n".join(lines)


def format_trip(index: dict, trip_id: str) -> str:
    """Render one curated suggestion; it is not a physical route."""
    return _format_curated_detail(
        index, trip_id, "trips", _trip_tag,
        "That curated trip suggestion is not available.",
    )


def format_path(index: dict, path_id: str) -> str:
    """Render one physical walking/biking route from /v120/paths."""
    return _format_curated_detail(
        index, path_id, "paths", _path_tag,
        "That physical route is not available.",
    )

def format_filter_pois(index: dict, limit: int = 20, **filters: Any) -> str:
    """Render facet-filter results without exposing filter internals."""
    # Drop None/empty filters to decide whether this is a valid request.
    active = {k: v for k, v in filters.items() if v not in (None, "", [], {})}
    if not active:
        return ("[INFO] filter_pois requires at least one filter "
                "(interest_level, type, tourist_type, section_id, indispensable).")
    matches = filter_pois(index, **active)
    if not matches:
        return "No places matched this request."
    truncated = False
    if limit and len(matches) > limit:
        matches = matches[:limit]
        truncated = True
    lines = [f"Found {len(matches)}{'+' if truncated else ''} places:"]
    for p in matches:
        preview = _short_preview(p)
        if preview:
            lines.append(f"  {_poi_tag(p)} — {preview}")
        else:
            lines.append(f"  {_poi_tag(p)}")
    if truncated:
        lines.append(f"  …more matches available (raise limit)")
    return "\n".join(lines)

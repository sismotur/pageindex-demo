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


def load_weather(path: str | Path) -> dict | None:
    """Read the offline weather artifact if it exists on disk.

    Returns None (rather than raising) when the file is missing so the
    runtime can degrade gracefully without a forecast.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


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


# Small models occasionally emit a synonymous element name
# (<place id=12 ...>) or leave the tag dangling after a bold label
# ("**Name** <poi id=12 type=X>:").  Canonicalize both before validation.
_TAG_ALIAS_OPEN_RE = re.compile(r"<place\b(?=[^>]*\bid\s*=)", re.IGNORECASE)
_TAG_ALIAS_CLOSE_RE = re.compile(r"</place\s*>", re.IGNORECASE)
_POI_BOLD_DANGLING_RE = re.compile(
    r"\*\*(?P<label>[^*\n]+?)\*\*\s*\(?\s*<poi\b[^>]*\bid\s*=\s*\"?"
    r"(?:poi/)?(?P<id>\d+)\"?[^>]*>\s*\)?",
    re.IGNORECASE,
)
# Any remaining <poi ...> opening/empty fragment (dangling model output).
# Captures bare numeric id when present so unknown section-style ids
# (e.g. <poi id=13 type=Events and Festivals>) can be stripped.
_POI_OPEN_FRAGMENT_RE = re.compile(
    r"<poi\b(?=[^>]*\bid\s*=\s*\"?(?:poi/)?(\d+)\"?)[^>]*>",
    re.IGNORECASE,
)
# Model list markup often glues the next bullet/label to </poi> with no
# space or newline: "</poi>*   **Descripción**:". Insert a break so the
# visitor-facing answer stays readable and mobile parsers stay clean.
_POI_TAG_GLUE_RE = re.compile(
    r"(</poi>)([*\-•#]|\*\*)",
    re.IGNORECASE,
)


def sanitize_poi_tags(answer: str, index: dict) -> str:
    """Keep only tags whose id exists in the downloaded index.

    This validates an ID the model supplied; it never searches names or
    guesses a replacement. Unknown tags become ordinary inner text.
    Dangling open fragments with unknown ids (often invented section
    numbers) are removed entirely; known dangling ids expand to a full
    canonical tag.
    """
    def bold_dangling(match: re.Match) -> str:
        poi = get_poi(index, f"poi/{match.group('id')}")
        if poi is None:
            return match.group("label")
        return _poi_tag_with_text(poi, match.group("label"))

    def full_tag(match: re.Match) -> str:
        poi = get_poi(index, f"poi/{match.group(1)}")
        if poi is None:
            return match.group(3)
        return _poi_tag_with_text(poi, match.group(3))

    def empty_tag(match: re.Match) -> str:
        poi = get_poi(index, f"poi/{match.group(1)}")
        return _poi_tag(poi) if poi is not None else ""

    def open_fragment(match: re.Match) -> str:
        poi = get_poi(index, f"poi/{match.group(1)}")
        # Unknown id (section number, hallucinated) → drop the fragment.
        return _poi_tag(poi) if poi is not None else ""

    sanitized = _TAG_ALIAS_OPEN_RE.sub("<poi", answer or "")
    sanitized = _TAG_ALIAS_CLOSE_RE.sub("</poi>", sanitized)
    sanitized = _POI_BOLD_DANGLING_RE.sub(bold_dangling, sanitized)
    sanitized = POI_TAG_RE.sub(full_tag, sanitized)
    sanitized = POI_TAG_EMPTY_RE.sub(empty_tag, sanitized)
    # Protect complete tags so the open-fragment pass cannot re-match
    # their openings and duplicate the inner text.
    protected: list[str] = []

    def protect_full(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"__INVENTRIP_FULL_POI_{len(protected) - 1}__"

    sanitized = POI_TAG_RE.sub(protect_full, sanitized)
    sanitized = _POI_OPEN_FRAGMENT_RE.sub(open_fragment, sanitized)
    # Dropping an unknown fragment can leave a double space ("de  para").
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
    for position, tag in enumerate(protected):
        sanitized = sanitized.replace(
            f"__INVENTRIP_FULL_POI_{position}__", tag
        )
    return sanitized


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
    # its machine-readable attributes. Unknown dangling fragments were
    # already stripped by sanitize_poi_tags; protect only remaining full
    # (known) tags.
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
    # Break glued list/markdown markers that small models stick to </poi>.
    return _POI_TAG_GLUE_RE.sub(r"\1\n\2", sanitized)



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
MAX_SECTION_LIMIT = 50
MAX_FILTER_LIMIT = 20
MAX_NAME_SEARCH_LIMIT = 5
MAX_EVIDENCE_SEARCH_LIMIT = 10
MAX_CURATED_SEARCH_LIMIT = 10
MAX_POI_BATCH = 5


def _bounded_limit(value: int | None, default: int, maximum: int) -> int:
    """Return a positive caller limit clamped to a context-safe maximum."""
    if value is None:
        return default
    try:
        return max(1, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


def _short_preview(poi: dict, max_chars: int = 120) -> str:
    """One-line tourist-facing description preview (no catalog metadata).

    Prefer a complete first sentence when it fits; otherwise cut on a word
    boundary. Avoid mid-word ellipsis so models copying the preview into
    visitor answers leave cleaner prose.
    """
    desc = (poi.get("description") or "").strip()
    if not desc:
        return ""
    budget = max(40, min(int(max_chars or 120), 160))
    sent_end = re.search(r"[.!?](?:\s|$)", desc)
    if sent_end and sent_end.end() <= budget:
        return desc[: sent_end.end()].strip()
    if len(desc) <= budget:
        return desc
    cut = desc[:budget].rsplit(" ", 1)[0].rstrip(",;:-")
    return (cut or desc[:budget]).strip() + "…"


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
    for grouped sections, SECTION_LIMIT_FLAT (50) otherwise. An explicit
    limit is capped at MAX_SECTION_LIMIT (50).
    """
    sec = find_section(index, section_key)
    if not sec:
        avail = ", ".join(s.get("title", "") for s in index.get("sections", []))
        return f"[ERROR] Section '{section_key}' not found. Available: {avail}"

    limit = _bounded_limit(
        limit,
        SECTION_LIMIT_GROUPED if sec.get("groups") else SECTION_LIMIT_FLAT,
        MAX_SECTION_LIMIT,
    )

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
        lines.append(
            f"  …{len(sec.get('poi_ids') or []) - limit} more; "
            "refine with filters or a name search."
        )
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
    """Render full POI records, capped at MAX_POI_BATCH with no line slicing.

    Accepts a single id ('poi/5155' or '5155') or several comma-separated
    ids ('poi/123,poi/456') — at most five records per call. The batch
    form saves LLM round-trips on comparison and synthesis questions.
    Multiple records are joined with a
    '\\n\\n---\\n\\n' separator; unknown ids render an inline [ERROR] block
    without failing the whole batch.
    """
    ids = [part.strip() for part in str(poi_id).split(",") if part.strip()]
    omitted = max(0, len(ids) - MAX_POI_BATCH)
    ids = ids[:MAX_POI_BATCH]
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
    if omitted:
        blocks.append(
            f"[INFO] {omitted} additional POI request(s) omitted; "
            f"get_poi accepts at most {MAX_POI_BATCH} records per call."
        )
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
    return out[:_bounded_limit(
        limit, MAX_NAME_SEARCH_LIMIT, MAX_NAME_SEARCH_LIMIT
    )]


def format_find_poi_by_name(index: dict, query: str, limit: int = 5,
                            detail: str = "brief") -> str:
    """Render name-search results.

    detail="full" appends the best match's complete POI record after the
    candidate list, fusing the classic find_poi_by_name -> get_poi pair
    into one tool call (one fewer LLM round on lookup questions).
    """
    limit = _bounded_limit(limit, MAX_NAME_SEARCH_LIMIT, MAX_NAME_SEARCH_LIMIT)
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
    if facet in {"locality", "address_locality"}:
        # Runtime scan — indexes do not ship a by_locality facet. Match
        # address_locality (and name as a weak fallback for City POIs)
        # so "events in Albalá" does not spill into the whole comarca.
        needle = normalize_text(str(value or ""))
        if not needle:
            return set()
        hits: set[str] = set()
        for pid, poi in (index.get("pois") or {}).items():
            loc = normalize_text(poi.get("address_locality") or "")
            name = normalize_text(poi.get("name") or "")
            if needle == loc or needle in loc.split() or needle in name:
                hits.add(pid)
        return hits
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


def index_localities(index: dict) -> list[str]:
    """Distinct address_locality values, longest first for phrase matching."""
    seen: set[str] = set()
    out: list[str] = []
    for poi in (index.get("pois") or {}).values():
        loc = (poi.get("address_locality") or "").strip()
        if not loc or loc in seen:
            continue
        seen.add(loc)
        out.append(loc)
    out.sort(key=lambda s: (-len(normalize_text(s)), normalize_text(s)))
    return out


def match_locality(text: str, index: dict) -> str | None:
    """Return the longest catalogue locality named in `text`, or None.

    Matching is diacritic-insensitive and requires the locality phrase to
    appear on token boundaries so short names do not hit inside longer
    words. Longest match wins (\"Santa Marta de Magasca\" before
    \"Santa Ana\").
    """
    normalized = f" {normalize_text(text)} "
    if not normalized.strip():
        return None
    for loc in index_localities(index):
        needle = normalize_text(loc)
        if not needle:
            continue
        # Space-padded boundary check keeps multi-word towns intact.
        if f" {needle} " in normalized:
            return loc
    return None


def resolve_active_locality(messages: list[dict], index: dict) -> str | None:
    """Most recent visitor-named locality in prior turns, or None.

    Scans user messages newest-first. Assistant text is ignored so a
    reply that mentions another town does not steal the focus the
    visitor set (e.g. Albalá stay after the model drifted to
    Arroyomolinos).
    """
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        # Skip internal runtime injects — they are English instructions,
        # not visitor locality cues.
        if content.startswith("[") or content.startswith("The visitor"):
            continue
        if content.startswith("A ") and "lookup has already" in content:
            continue
        loc = match_locality(content, index)
        if loc:
            return loc
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
        for poi in matches[:_bounded_limit(
            limit, MAX_EVIDENCE_SEARCH_LIMIT, MAX_EVIDENCE_SEARCH_LIMIT
        )]
    ]


def format_search_pois(index: dict, query: str, section_id: str | None = None,
                       limit: int = 10) -> str:
    """Render evidence-backed search results without catalog internals."""
    limit = _bounded_limit(
        limit, MAX_EVIDENCE_SEARCH_LIMIT, MAX_EVIDENCE_SEARCH_LIMIT
    )
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


def _recent_history_poi_ids(messages: list[dict],
                            index: dict) -> list[str]:
    """Return validated POI ids from the most recent assistant turn.

    Walks history from newest to oldest and stops at the first assistant
    turn that contains at least one known `<poi id=...>` tag, preserving
    the reading order of that turn.  Deduplicates by id.
    """
    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        ids: list[str] = []
        seen: set[str] = set()
        for ref in extract_poi_tags(content, index):
            pid = ref.get("poi_id")
            if ref.get("known") and pid and pid not in seen:
                ids.append(pid)
                seen.add(pid)
        if ids:
            return ids
    return []


def _topical_terms(query: str, index: dict) -> list[str]:
    """Keep only visitor terms that carry topical signal for scoring.

    Rejects tokens shorter than three characters and tokens that appear
    in a large fraction of POI records (e.g. “entonces”, “tiene”,
    “about”).  This is a corpus-driven proxy for stopword filtering that
    works across the sixteen supported languages without a hand-curated
    list.
    """
    raw = [term for term in tokenize(query) if len(term) >= 3]
    if not raw:
        return []
    total_pois = max(1, len(index.get("pois") or {}))
    common_threshold = max(1, total_pois // 3)
    kept: list[str] = []
    fallback: list[tuple[int, str]] = []
    for term in raw:
        postings = _search_postings(index, term)
        hits = len(postings)
        if hits == 0:
            continue
        if hits < common_threshold:
            kept.append(term)
        else:
            fallback.append((hits, term))
    if kept:
        return kept
    # Everything is either unseen or too common: keep the rarest single
    # term so the fallback still has some signal.
    if fallback:
        fallback.sort()
        return [fallback[0][1]]
    return []


def _score_history_pois(question: str,
                        poi_ids: list[str],
                        index: dict) -> list[tuple[float, dict]]:
    """Score history POIs, weighting rare topical terms higher than common ones.

    A POI matching a rare term (e.g. “tapas” present in 2 records) gets a
    much higher score than one matching a common connective verb (e.g.
    “tiene” present in dozens of descriptions), even though both are
    below the corpus-frequency stopword cutoff.
    """
    terms = _topical_terms(question, index)
    if not terms or not poi_ids:
        return []
    weights: dict[str, float] = {}
    for term in terms:
        hits = len(_search_postings(index, term))
        weights[term] = 1.0 / max(1, hits)
    pois = index.get("pois") or {}
    scored: list[tuple[float, dict]] = []
    for pid in poi_ids:
        poi = pois.get(pid)
        if not poi:
            continue
        hay = set(tokenize(_searchable_text(poi)))
        score = 0.0
        for term, weight in weights.items():
            if _term_variants(term) & hay:
                score += weight
        if score > 0:
            scored.append((score, poi))
    scored.sort(key=lambda pair: (
        -pair[0],
        pair[1].get("interest_level") or 99,
        pair[1].get("zoom_level") or 99,
        normalize_text(pair[1].get("name") or ""),
    ))
    return scored


def format_history_followup(index: dict, question: str,
                            messages: list[dict],
                            limit: int = 6) -> str:
    """Best-effort deterministic answer for a follow-up on shown POIs.

    Restricts candidates to POI ids that appeared as validated `<poi>`
    tags in the most recent assistant turn, then keeps those whose
    name/description/type overlaps with the visitor's content words.
    Returns an empty string when no candidate scores or when the recent
    turn does not contain any known POI tag — the runtime should then
    fall back to the localized safe failure.
    """
    if limit <= 0:
        return ""
    ids = _recent_history_poi_ids(messages, index)
    if not ids:
        return ""
    scored = _score_history_pois(question, ids, index)
    if not scored:
        return ""
    matches = [poi for _, poi in scored[:limit]]
    terms = _topical_terms(question, index)
    lines = []
    for poi in matches:
        evidence = _evidence_snippet(poi, terms)
        if evidence:
            lines.append(f"  - {_poi_tag(poi)} — {evidence}")
        else:
            lines.append(f"  - {_poi_tag(poi)}")
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
    """Return the tag for a physical route sourced from /v120/paths.

    A route is itself a trip in the source API — a route is simply a
    trip whose extras.path field is non-null (see fetch_paths() in
    extract_destination_data.py) — so it renders and resolves exactly
    like a curated trip (<trip id=...>), never a separate <path> tag,
    which would not resolve via get_trip() for a follow-up question.
    """
    return _trip_tag(path)
def _path_tag_with_text(path: dict, text: str) -> str:
    """Return the trip tag for a physical route, preserving a visible label."""
    return _trip_tag_with_text(path, text)


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
    return matches[:_bounded_limit(
        limit, MAX_CURATED_SEARCH_LIMIT, MAX_CURATED_SEARCH_LIMIT
    )]


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
    """Return one curated suggestion by full or bare trip id.

    Physical routes live only under `paths` (no JSON dual-listing), but
    format_path still emits <trip id=…> tags because source API routes are
    trip records. Fall back to `paths` so history follow-up and get_trip
    keep resolving those tags without putting routes in `trips`.
    """
    item = _find_curated(index, trip_id, "trips")
    if item is not None:
        return item
    return _find_curated(index, trip_id, "paths")


def get_path(index: dict, path_id: str) -> dict | None:
    """Return one physical route by full or bare path id."""
    return _find_curated(index, path_id, "paths")


def resolve_trip_query(question: str, index: dict) -> dict | None:
    """Resolve a direct user reference to a known curated trip title.

    A title such as “RUTAS POR ÚBEDA” contains a route-like word but is an
    editorial trip, not a physical path. Exact or contained source-title
    matches take precedence over generic route intent. Ambiguities are
    deliberately rejected.

    A single-word title is never enough to count as a match: some
    destinations have trips titled just the destination's own name (e.g.
    “Montánchez”) or a generic term (e.g. “Comarca”), which would otherwise
    match almost every question about that destination even though the
    visitor never named the trip. Requiring at least two words keeps the
    original ÚBEDA case working while rejecting those false positives.
    """
    query = normalize_text(question)
    if len(query) < 4:
        return None
    matches = []
    for trip in index.get("trips") or []:
        name = normalize_text(trip.get("name") or "")
        if (len(name) >= 4 and len(name.split()) >= 2
                and (query == name or name in query)):
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


# Lead-ins stripped before matching a follow-up against prior tags.
# Longest first so "dame el detalle de" wins over bare "detalle de".
# Covers the common "give me details about X" shape without requiring
# the visitor to type the full tagged label alone.
_HISTORY_FOCUS_LEADS = tuple(sorted((
    "dame el detalle de la",
    "dame el detalle de",
    "dame los detalles de la",
    "dame los detalles de",
    "dame mas informacion sobre",
    "dame mas info sobre",
    "dame mas detalles de",
    "cuentame mas sobre",
    "cuentame sobre",
    "dime mas sobre",
    "dime sobre",
    "hablame de",
    "hablame sobre",
    "tell me more about",
    "tell me about",
    "more details about",
    "details about",
    "detail about",
    "detalle de la",
    "detalle del",
    "detalle de",
    "detalles de la",
    "detalles del",
    "detalles de",
    "mas sobre",
    "about the",
    "about",
), key=len, reverse=True))


def _history_focus_query(question: str) -> str:
    """Strip detail/about lead-ins; return the place-name focus span."""
    normalized = normalize_text(question)
    if not normalized:
        return ""
    for lead in _HISTORY_FOCUS_LEADS:
        if normalized.startswith(lead + " "):
            return normalized[len(lead):].strip(" ?!.")
        # Also allow the lead mid-phrase after a short verb ("quiero el
        # detalle de X").
        marker = " " + lead + " "
        if marker in normalized:
            return normalized.split(marker, 1)[1].strip(" ?!.")
    return normalized


def resolve_history_selection(question: str, messages: list[dict],
                              index: dict) -> dict | None:
    """Resolve a concise follow-up against validated prior assistant tags.

    Example: after the assistant offers
    `<trip id=4453>Ú. en Familia-R. Secundaria 2</trip>`, the user can
    say “Secundaria 2”. This returns a validated source selection:
    `{kind: "trip", id: "trip/4453", label: "…"}`.

    Matching is deliberately conservative: a unique normalized substring,
    an all-token match against a shown label, a focus-span match after
    stripping "detail about" lead-ins, content-token overlap on labels,
    or a bare numeric id token that matches a shown tag id. Ambiguous
    references return None so the grounding gate asks the model to
    retrieve rather than guessing.
    """
    query = normalize_text(question)
    query_tokens = set(query.split())
    if len(query) < 3 or not query_tokens:
        return None
    focus = _history_focus_query(question)
    focus_tokens = set(focus.split()) if focus else set()

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
        score = 0
        if query in label:
            score = max(score, 100 + len(query))
        elif query_tokens.issubset(label_tokens):
            score = max(score, 50 + len(query_tokens))
        # Focus span after stripping "dame el detalle de …" etc.
        if focus and focus != query:
            if focus in label:
                score = max(score, 90 + len(focus))
            elif focus_tokens and focus_tokens.issubset(label_tokens):
                score = max(score, 55 + len(focus_tokens))
        # Content-token overlap: require ≥2 tokens of length ≥4 so bare
        # "feria" alone cannot steal a unique match among several fairs.
        content = {t for t in (focus_tokens or query_tokens) if len(t) >= 4}
        overlap = content & label_tokens
        if len(overlap) >= 2:
            score = max(score, 40 + 5 * len(overlap) + sum(len(t) for t in overlap))
        if score > 0:
            scored.append((score, item))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = scored[0]
    if len(scored) > 1 and scored[1][0] == best_score:
        return None
    return best


def resolve_sole_recent_source(messages: list[dict], index: dict) -> dict | None:
    """Return the one trip/path shown in the immediately preceding
    assistant turn, or None when that turn showed zero or more than one.

    A generic plan/detail follow-up ("give me the itinerary") names
    nothing itself, so the wording-based resolve_history_selection()
    cannot match it against a shown label. But when exactly one trip or
    path was just shown, it is the unambiguous referent — callers should
    only use this as a fallback for that specific kind of generic
    follow-up, not for arbitrary unrelated questions.
    """
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        candidates: list[dict] = []
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
        unique = {(c["kind"], c["id"]): c for c in candidates}
        return next(iter(unique.values())) if len(unique) == 1 else None
    return None


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
    "ca": {
        "lead":       "Aquí tens algunes suggerències que podrien encaixar amb la teva petició:",
        "highlights": "Destacats",
        "outro":      "Digues-me el nom o el número de la suggerència que vols veure.",
    },
    "de": {
        "lead":       "Hier sind einige kuratierte Vorschläge, die zu Ihrer Anfrage passen könnten:",
        "highlights": "Highlights",
        "outro":      "Nennen Sie mir den Namen oder die Nummer des Vorschlags, den Sie sehen möchten.",
    },
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
    "eu": {
        "lead":       "Hona hemen zure eskariarekin bat datozkeen proposamen batzuk:",
        "highlights": "Nabarmenak",
        "outro":      "Esadazu ikusi nahi duzun proposamenaren izena edo zenbakia.",
    },
    "fr": {
        "lead":       "Voici quelques suggestions qui pourraient correspondre à votre demande :",
        "highlights": "À ne pas manquer",
        "outro":      "Dites-moi le nom ou le numéro de la suggestion que vous souhaitez voir.",
    },
    "gl": {
        "lead":       "Aquí tes algunhas suxestións que poderían encaixar coa túa petición:",
        "highlights": "Destacados",
        "outro":      "Dime o nome ou o número da suxestión que queres ver.",
    },
    "hi": {
        "lead":       "आपके अनुरोध से मेल खाते कुछ चुनिंदा सुझाव यहाँ हैं:",
        "highlights": "मुख्य आकर्षण",
        "outro":      "बताइए आप कौन-सा सुझाव देखना चाहेंगे, नाम या नंबर से।",
    },
    "hr": {
        "lead":       "Evo nekoliko prijedloga koji bi mogli odgovarati vašem upitu:",
        "highlights": "Istaknuto",
        "outro":      "Recite mi naziv ili broj prijedloga koji želite vidjeti.",
    },
    "it": {
        "lead":       "Ho trovato alcune proposte curate che potrebbero corrispondere:",
        "highlights": "In evidenza",
        "outro":      "Dimmi il nome o il numero del viaggio che preferisci.",
    },
    "ja": {
        "lead":       "ご希望に合いそうなおすすめプランがいくつか見つかりました:",
        "highlights": "見どころ",
        "outro":      "ご覧になりたいプランの名前または番号を教えてください。",
    },
    "nl": {
        "lead":       "Hier zijn een paar samengestelde suggesties die bij je verzoek kunnen passen:",
        "highlights": "Hoogtepunten",
        "outro":      "Vertel me de naam of het nummer van de suggestie die je wilt zien.",
    },
    "pt": {
        "lead":       "Aqui estão algumas sugestões que podem corresponder ao seu pedido:",
        "highlights": "Destaques",
        "outro":      "Diga-me o nome ou o número da sugestão que quer ver.",
    },
    "ru": {
        "lead":       "Вот несколько подобранных вариантов, которые могут подойти к вашему запросу:",
        "highlights": "Главное",
        "outro":      "Назовите название или номер варианта, который хотите посмотреть.",
    },
    "uk": {
        "lead":       "Ось кілька підібраних варіантів, які можуть відповідати вашому запиту:",
        "highlights": "Головне",
        "outro":      "Назвіть назву або номер варіанта, який хочете переглянути.",
    },
    "zh": {
        "lead":       "以下是几个可能符合您需求的精选建议:",
        "highlights": "亮点",
        "outro":      "请告诉我您想查看的建议名称或编号。",
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


def _append_route_stops(lines: list[str], item: dict, index: dict) -> str:
    """Append a route's real waypoints, dropping degenerate step headers.

    A route's source itinerary steps are often just a location label or a
    repeat of the route's own title, with no stops of their own (see
    fetch_paths() in extract_destination_data.py) — those carry no
    information and are dropped. A step that does carry real stops is
    still shown, but without the day-by-day "N. Title" numbering
    format_trip() uses for editorial trips: a route's stops are waypoints
    along one path, not sequential options to choose between.
    """
    route_name = normalize_text(item.get("name") or "")
    for step in item.get("steps") or []:
        step_items = step.get("items")
        poi_ids = step.get("poi_ids") or []
        if not step_items and not poi_ids:
            continue  # empty header/location label — nothing to show
        lines.append("")  # separator before this block (and the description)
        title = (step.get("title") or "").strip()
        if title and normalize_text(title) != route_name:
            lines.append(title)
        if step_items:
            _render_curated_items(step_items, index, lines, depth=0)
        else:
            for poi_id in poi_ids:
                poi = get_poi(index, poi_id)
                if poi:
                    lines.append(f"- {_poi_tag(poi)}")
        # `unresolved_poi_names` is retained in the index for QA only.
        # It may be a stale or foreign-language source label, so never
        # present it as a tourist-visible or app-openable place.
    return "\n".join(lines)


def _format_curated_detail(index: dict, itinerary_id: str, collection: str,
                           tag_builder, unavailable: str) -> str:
    """Render ordered trip/path stops, preserving unlinked source names."""
    if collection == "trips":
        # Prefer editorial trips; fall back to paths for route <trip> tags.
        item = get_trip(index, itinerary_id)
    else:
        item = _find_curated(index, itinerary_id, collection)
    if not item:
        return unavailable
    lines = [f"# {tag_builder(item)}"]
    description = (item.get("description") or "").strip()
    if description:
        lines.extend(["", description])
    if item.get("is_route"):
        # Route may be opened via get_trip() (tag follow-up) or get_path();
        # always render with route stop rules when is_route is true.
        return _append_route_stops(lines, item, index)
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
    """Render one physical walking/biking route from /v120/paths.

    Unlike format_trip() on an ordinary editorial trip, a route's real
    waypoints (if any) are shown without the day-by-day "N. Title"
    numbering — see _append_route_stops().
    """
    item = _find_curated(index, path_id, "paths")
    if not item:
        return "That physical route is not available."
    lines = [f"# {_path_tag(item)}"]
    description = (item.get("description") or "").strip()
    if description:
        lines.extend(["", description])
    return _append_route_stops(lines, item, index)

# ── Weather ------------------------------------------------------------------
#
# The offline weather artifact is a small JSON with a `meta` and a
# 7-entry `forecast` list, downloaded by the phone from a Cloudflare
# Worker daily.  Details in docs/mobile-offline-contract.md §2.4.

FORECAST_TAG_RE = re.compile(
    r"<forecast\b[^>]*\bday\s*=\s*\"?(\d{4}-\d{2}-\d{2})\"?[^>]*>(.*?)</forecast>",
    re.IGNORECASE | re.DOTALL,
)

WEATHER_UNAVAILABLE_MESSAGES: dict[str, str] = {
    "ca": "Les dades descarregades no inclouen una previsió vigent.",
    "de": "Die heruntergeladenen Daten enthalten keine aktuelle Vorhersage.",
    "en": "The downloaded data has no current forecast.",
    "es": "Los datos descargados no incluyen una previsión vigente.",
    "eu": "Deskargatutako datuek ez dute indarreko aurreikuspenik.",
    "fr": "Les données téléchargées ne contiennent pas de prévisions actuelles.",
    "gl": "Os datos descargados non inclúen unha previsión vixente.",
    "hi": "डाउनलोड किए गए डेटा में कोई वर्तमान पूर्वानुमान नहीं है।",
    "hr": "Preuzeti podaci ne sadrže aktualnu prognozu.",
    "it": "I dati scaricati non includono una previsione attuale.",
    "ja": "ダウンロード済みデータに現在の予報は含まれていません。",
    "nl": "De gedownloade gegevens bevatten geen actuele voorspelling.",
    "pt": "Os dados transferidos não incluem uma previsão atual.",
    "ru": "В загруженных данных нет актуального прогноза.",
    "uk": "Завантажені дані не містять актуального прогнозу.",
    "zh": "已下载的数据中没有当前预报。",
}

_WEATHER_STALE_MESSAGES: dict[str, str] = {
    "ca": "Previsió estimada obtinguda fa {n} dies",
    "de": "Geschätzte Vorhersage vor {n} Tagen abgerufen",
    "en": "Estimated forecast fetched {n}d ago",
    "es": "Previsión estimada obtenida hace {n} días",
    "eu": "Duela {n} egun eskuratutako aurreikuspen estimatua",
    "fr": "Prévisions estimées récupérées il y a {n}j",
    "gl": "Previsión estimada obtida hai {n} días",
    "hi": "{n} दिन पहले प्राप्त अनुमानित पूर्वानुमान",
    "hr": "Procijenjena prognoza preuzeta prije {n} d.",
    "it": "Previsione stimata recuperata {n} giorni fa",
    "ja": "{n}日前に取得した推定予報",
    "nl": "Geschatte voorspelling van {n} dagen geleden",
    "pt": "Previsão estimada obtida há {n} dias",
    "ru": "Оценочный прогноз, получен {n} дн. назад",
    "uk": "Оціночний прогноз, отримано {n} дн. тому",
    "zh": "{n}天前获取的估计预报",
}

_WEEKDAY_ALIASES: dict[str, int] = {
    # English
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 7,
    # Spanish
    "lunes": 1, "martes": 2, "miercoles": 3, "jueves": 4,
    "viernes": 5, "sabado": 6, "domingo": 7,
    # Italian
    "lunedi": 1, "martedi": 2, "mercoledi": 3, "giovedi": 4,
    "venerdi": 5, "sabato": 6, "domenica": 7,
}

WEATHER_STALE_HOURS  = 24
WEATHER_EXPIRED_DAYS = 7


def _weather_lang(weather: dict) -> str:
    return (weather.get("meta") or {}).get("lang") or "en"


def weather_unavailable_message(weather: dict | None,
                                fallback_lang: str = "en") -> str:
    """Return the localized 'no forecast' message."""
    lang = _weather_lang(weather or {}) if weather else fallback_lang
    return WEATHER_UNAVAILABLE_MESSAGES.get(
        lang, WEATHER_UNAVAILABLE_MESSAGES["en"]
    )


def _parse_utc_timestamp(value: str) -> "datetime | None":
    """Parse the 'YYYY-MM-DDTHH:MM:SSZ' timestamps produced by build_weather."""
    from datetime import datetime, timezone
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError):
        return None


def _weather_age_days(weather: dict,
                     now: "datetime | None" = None) -> float | None:
    """How many whole days have passed since `meta.fetched_at`."""
    from datetime import datetime, timezone
    fetched = _parse_utc_timestamp((weather.get("meta") or {}).get("fetched_at", ""))
    if fetched is None:
        return None
    ref = now or datetime.now(timezone.utc)
    return (ref - fetched).total_seconds() / 86_400.0


def _resolve_forecast_day(weather: dict, day: str | None,
                          now: "datetime | None" = None) -> dict | None:
    """Return the single forecast entry that matches `day`, or None.

    Accepts:
      - None / "" (caller wants the whole week; handled elsewhere)
      - 'today' / 'tomorrow' in English or the localized aliases below
      - an ISO 'YYYY-MM-DD' date
      - a weekday name (monday..sunday, plus localized aliases)
    """
    forecast = weather.get("forecast") or []
    if not forecast:
        return None
    raw = (day or "").strip()
    if not raw:
        return None
    # ISO date matches must be tested before normalize_text, which strips
    # dashes ("2026-08-27" → "2026 08 27").
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        for entry in forecast:
            if entry.get("date") == raw:
                return entry
        return None
    key = normalize_text(raw)
    if not key:
        return None
    from datetime import datetime, timezone, date as date_cls
    ref = now or datetime.now(timezone.utc)
    ref_date = ref.date()
    # today / tomorrow
    today_aliases    = {"today", "hoy", "oggi", "aujourd hui", "heute", "hoje"}
    tomorrow_aliases = {"tomorrow", "manana", "domani", "demain", "morgen", "amanha"}
    if key in today_aliases:
        target = ref_date
    elif key in tomorrow_aliases:
        target = date_cls.fromordinal(ref_date.toordinal() + 1)
    else:
        target = None
    if target is not None:
        iso = target.isoformat()
        for entry in forecast:
            if entry.get("date") == iso:
                return entry
        return None
    # Weekday name (e.g. 'martes'): return the earliest forecast entry
    # matching that iso_weekday.
    weekday = _WEEKDAY_ALIASES.get(key)
    if weekday is not None:
        for entry in forecast:
            if entry.get("iso_weekday") == weekday:
                return entry
    return None


def _forecast_tag(entry: dict) -> str:
    date = entry.get("date") or ""
    label = entry.get("day_label") or date
    return f'<forecast day="{date}">{label}</forecast>'


def _forecast_line(entry: dict) -> str:
    tag = _forecast_tag(entry)
    condition = (entry.get("condition") or "").strip()
    temp_min = entry.get("temp_min_c")
    temp_max = entry.get("temp_max_c")
    parts: list[str] = []
    if condition:
        parts.append(condition)
    if isinstance(temp_min, (int, float)) and isinstance(temp_max, (int, float)):
        parts.append(f"{temp_min:g}–{temp_max:g} °C")
    elif isinstance(temp_max, (int, float)):
        parts.append(f"{temp_max:g} °C")
    suffix = f" — {', '.join(parts)}" if parts else ""
    return f"{tag}{suffix}"


def weather_hint(weather: dict | None, destination_display: str,
                 now: "datetime | None" = None) -> str:
    """Return a one-line 'today' hint for the system prompt, or ''."""
    if not weather:
        return ""
    entry = _resolve_forecast_day(weather, "today", now=now)
    if entry is None:
        return ""
    line = _forecast_line(entry)
    return f"Today in {destination_display}: {line}. Consult get_weather for other days."


def format_weather(weather: dict | None, day: str | None = None,
                   now: "datetime | None" = None) -> str:
    """Render the tourist-safe weather answer for the LLM tool.

    Returns a localized 'unavailable' message when the file is missing
    or expired, an 'estimated' prefix when it is stale but usable, and
    a single-day or full-week block otherwise.
    """
    if not weather:
        return weather_unavailable_message(weather)
    age = _weather_age_days(weather, now=now)
    if age is None or age > WEATHER_EXPIRED_DAYS:
        return weather_unavailable_message(weather)
    prefix = ""
    if age > 1:
        template = _WEATHER_STALE_MESSAGES.get(
            _weather_lang(weather), _WEATHER_STALE_MESSAGES["en"]
        )
        prefix = template.format(n=int(age)) + ":\n"
    if day:
        entry = _resolve_forecast_day(weather, day, now=now)
        if entry is None:
            return weather_unavailable_message(weather)
        return f"{prefix}{_forecast_line(entry)}"
    forecast = weather.get("forecast") or []
    if not forecast:
        return weather_unavailable_message(weather)
    lines = [f"  - {_forecast_line(entry)}" for entry in forecast]
    body = "\n".join(lines)
    return f"{prefix}{body}"


def get_weather_entry(weather: dict | None, day: str | None = None,
                      now: "datetime | None" = None) -> dict | None:
    """Return the raw forecast entry for `day` (default 'today'), or None.

    None is returned when the file is missing/expired or the day cannot
    be resolved — the same staleness gate `format_weather` applies.
    Unlike `format_weather`, this exposes the entry's raw fields (e.g.
    `temp_max_c`) for callers that reason about the data itself rather
    than its rendered, tourist-facing text.
    """
    if not weather:
        return None
    age = _weather_age_days(weather, now=now)
    if age is None or age > WEATHER_EXPIRED_DAYS:
        return None
    return _resolve_forecast_day(weather, day or "today", now=now)


# Exact threshold already documented in the system prompt's outdoor-plan
# rule (run_eval.py::_SYSTEM_PROMPT_TEMPLATE): "...unfavourable (>35 °C,
# rain, storms)...". Kept here as the single source of truth for the
# numeric half of that rule so the two never drift apart.
OUTDOOR_HIGH_TEMP_C = 35.0


def is_forecast_too_hot(entry: dict | None) -> bool:
    """True when a forecast entry's max temperature exceeds the
    documented outdoor-plan threshold (>35 °C).

    Deliberately narrow: a pure numeric comparison on a field that is
    already extracted, so it needs no per-language data and behaves
    identically regardless of the visitor's language. Rain/storm/clear
    judgments are NOT attempted here — the model already receives the
    localized `condition` string verbatim and can read it directly in
    whichever of the 16 supported languages the index uses; hand-built
    per-language condition keyword lists would not scale the way this
    one threshold check does.
    """
    if not entry:
        return False
    temp_max = entry.get("temp_max_c")
    return isinstance(temp_max, (int, float)) and temp_max > OUTDOOR_HIGH_TEMP_C


def format_filter_pois(index: dict, limit: int = 20, **filters: Any) -> str:
    """Render facet-filter results without exposing filter internals."""
    limit = _bounded_limit(limit, MAX_FILTER_LIMIT, MAX_FILTER_LIMIT)
    # Drop None/empty filters to decide whether this is a valid request.
    active = {k: v for k, v in filters.items() if v not in (None, "", [], {})}
    if not active:
        return ("[INFO] filter_pois requires at least one filter "
                "(interest_level, type, tourist_type, section_id, "
                "locality, indispensable).")
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
        lines.append("  …more matches available; refine the filters.")
    return "\n".join(lines)

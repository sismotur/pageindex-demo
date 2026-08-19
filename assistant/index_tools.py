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

def sanitize_tourist_answer(answer: str, index: dict) -> str:
    """Apply deterministic presentation rules to a final visitor answer.

    This is deliberately narrow: validate tag ids, then replace only
    catalog-language nouns that should never reach a tourist. It does not
    infer facts, change names, or alter the meaning of retrieved evidence.
    """
    sanitized = sanitize_poi_tags(answer, index)
    replacements = (
        (r"\bPOIs\b", "places"),
        (r"\bPOI\b", "place"),
        (r"\bpoints of interest\b", "places"),
        (r"\bpoint of interest\b", "place"),
    )
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
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

    Prefix compatibility handles harmless morphology
    (restaurant/restaurants) without a language-specific stemmer. A v2
    index falls back to a small local scan during a staged corpus upgrade.
    """
    search_terms = ((index.get("facets") or {}).get("search_terms") or {})
    if not search_terms:
        return {
            pid for pid, poi in (index.get("pois") or {}).items()
            if term in set(tokenize(_searchable_text(poi)))
        }

    # Union exact and prefix-compatible forms.  A direct plural word can
    # exist in prose ("restaurants") while the category label is singular
    # ("Restaurant"); using only the direct posting would hide the actual
    # restaurant records.
    matched: set[str] = set(search_terms.get(term) or [])
    for indexed_term, ids in search_terms.items():
        if indexed_term.startswith(term) or term.startswith(indexed_term):
            matched.update(ids)
    return matched


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

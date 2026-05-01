#!/usr/bin/env python3
"""
index_tools.py — Read-side helpers for the POI-aware index.

Pure functions over the dict produced by build_index.py.  No I/O at import,
no LLM calls, no global state.  Imported by run_eval.py and chat_demo.py
to back the five LLM tools:

    list_sections()                — embedded into the system prompt
    get_section(section_id, ...)   — list POIs in a section
    get_poi(poi_id)                — full record of one POI
    find_poi_by_name(query, ...)   — fuzzy lookup by name
    filter_pois(**facets)          — facet query (interest_level, type, ...)

All `format_*` functions return strings suitable for tool-call results;
all `index_*` functions return raw structures used internally by the
formatters and by tests.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

# ── I/O ─────────────────────────────────────────────────────────────────────

def load_index(path: str | Path) -> dict:
    """Read the index JSON from disk and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Tokenisation (used by name search) ─────────────────────────────────────

_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace.

    Diacritic stripping makes 'Vázquez' and 'Vazquez' compare equal,
    which is what users type in search boxes.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = _NON_WORD_RE.sub(" ", stripped)
    return " ".join(stripped.lower().split())


def tokenize(text: str) -> list[str]:
    """Split normalised text into tokens (words)."""
    return normalize_text(text).split()


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

def _short_preview(poi: dict, max_chars: int = 120) -> str:
    """One-line preview: type + interest label + first sentence of description."""
    parts = []
    if poi.get("display_type"):
        parts.append(poi["display_type"])
    label = poi.get("interest_level_label")
    if label and label != "Outstanding":
        parts.append(label)
    desc = (poi.get("description") or "").strip()
    if desc:
        # First sentence or first 90 chars, whichever shorter
        sent_end = re.search(r"[.!?]\s", desc)
        snippet = desc[: sent_end.end()] if sent_end else desc[:90]
        if len(snippet) > 90:
            snippet = snippet[:90].rsplit(" ", 1)[0] + "…"
        parts.append(snippet.strip())
    out = " — ".join(p for p in parts if p)
    if len(out) > max_chars:
        out = out[: max_chars - 1] + "…"
    return out


def format_section(index: dict, section_key: str,
                   sort: str = "interest", limit: int = 50) -> str:
    """Render a section: title, summary, then one line per POI."""
    sec = find_section(index, section_key)
    if not sec:
        avail = ", ".join(s.get("title", "") for s in index.get("sections", []))
        return f"[ERROR] Section '{section_key}' not found. Available: {avail}"

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

    lines = [
        f"Section: {sec.get('title')}  (id={sec.get('section_id')}, "
        f"{len(sec.get('poi_ids') or [])} POIs total)",
    ]
    if sec.get("summary"):
        lines.append(f"  {sec['summary']}")
    lines.append("")
    for p in pois:
        pid = p.get("poi_id", "?")
        name = p.get("name", "?")
        preview = _short_preview(p)
        if preview:
            lines.append(f"  [{pid}] {name} — {preview}")
        else:
            lines.append(f"  [{pid}] {name}")
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


def format_poi(index: dict, poi_id: str) -> str:
    """Render the full POI record. No truncation, no line slicing."""
    p = get_poi(index, poi_id)
    if not p:
        return (f"[ERROR] POI '{poi_id}' not found. "
                f"Use find_poi_by_name() if you only know the name.")

    section_title = _poi_section_title(index, p["poi_id"])

    lines = [f"# {p.get('name', '(unnamed)')}  ({p.get('poi_id')})"]
    if section_title:
        lines.append(f"*Section: {section_title}*")
    lines.append("")

    # Bullet metadata
    bullets: list[str] = []
    if p.get("interest_level_label"):
        bullets.append(_format_kv("Interest level", p["interest_level_label"]))
    if p.get("display_type"):
        bullets.append(_format_kv("Type", p["display_type"]))
    if p.get("display_tourist_types"):
        bullets.append(_format_kv("Tourism interest", p["display_tourist_types"]))
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
    bullets.append(_format_kv("Images", p.get("image_urls")))
    bullets.append(_format_kv("Audio guides", p.get("audio_urls")))
    bullets.append(_format_kv("Documents", p.get("subject_of_urls")))

    for b in bullets:
        if b:
            lines.append(b)

    desc = (p.get("description") or "").strip()
    if desc:
        lines.append("")
        lines.append(desc)
    return "\n".join(lines)


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


def format_find_poi_by_name(index: dict, query: str, limit: int = 5) -> str:
    """Render name-search results."""
    matches = find_poi_by_name(index, query, limit=limit)
    if not matches:
        return (f"[INFO] No POI matches '{query}'. "
                f"Try filter_pois() or browse a section with get_section().")
    lines = [f"Matches for '{query}' ({len(matches)} of up to {limit}):"]
    for p in matches:
        sec_title = _poi_section_title(index, p["poi_id"])
        sec_label = f"  [{sec_title}]" if sec_title else ""
        preview = _short_preview(p)
        head = f"  [{p['poi_id']}] {p.get('name','?')}{sec_label}"
        if preview:
            lines.append(f"{head}  — {preview}")
        else:
            lines.append(head)
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


def format_filter_pois(index: dict, limit: int = 20, **filters: Any) -> str:
    """Render facet-filter results."""
    # Drop None/empty filters from the echo line
    active = {k: v for k, v in filters.items() if v not in (None, "", [], {})}
    if not active:
        return ("[INFO] filter_pois requires at least one filter "
                "(interest_level, type, tourist_type, section_id, indispensable).")
    matches = filter_pois(index, **active)
    if not matches:
        return f"[INFO] No POIs match {active}."
    truncated = False
    if limit and len(matches) > limit:
        matches = matches[:limit]
        truncated = True
    lines = [f"Filter {active}: {len(matches)}{'+' if truncated else ''} matches"]
    for p in matches:
        sec_title = _poi_section_title(index, p["poi_id"])
        sec_label = f"  [{sec_title}]" if sec_title else ""
        preview = _short_preview(p)
        head = f"  [{p['poi_id']}] {p.get('name','?')}{sec_label}"
        if preview:
            lines.append(f"{head}  — {preview}")
        else:
            lines.append(head)
    if truncated:
        lines.append(f"  …more matches available (raise limit)")
    return "\n".join(lines)

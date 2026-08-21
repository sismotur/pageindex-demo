#!/usr/bin/env python3
"""
assistant/run_eval.py — Q&A evaluation runner over the POI-aware index.

Loads indexes/{destination}_{lang}.json (built by pipeline/build_index.py)
and runs each question in eval/questions.json through litellm tool calling.

Six tools are exposed to the model:

    get_section(section_id, sort, limit)
        List the POIs inside one section, sorted by (interest_level,
        zoom_level) by default.  Returns id + name + 1-line preview.

    get_poi(poi_id)
        Full record of one POI by ID.  All fields, all paragraphs,
        no truncation, no line slicing.

    find_poi_by_name(query, limit)
        Fuzzy lookup against POI names.  Diacritic-insensitive.

    filter_pois(interest_level, type, tourist_type, section_id,
                indispensable, limit)
        Facet query.  Combine multiple filters with AND.

    list_sections()
        Section catalogue with deterministic 1-line summaries.
        Embedded into the system prompt at startup, so the model
        rarely needs to call it explicitly.

Usage:
    .venv/bin/python assistant/run_eval.py
    .venv/bin/python assistant/run_eval.py --model openai/gemma-4-E2B-it-MLX-8bit
    .venv/bin/python assistant/run_eval.py --lang es \
        --questions eval/questions_es.json --index indexes/ubeda_es.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import litellm
litellm.drop_params = True
litellm.set_verbose = False

# Make package imports work whether you run as a script or module
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from index_tools import (
    load_index,
    format_sections_overview,
    format_section,
    format_poi,
    format_find_poi_by_name,
    format_filter_pois,
    format_search_pois,
    format_search_trips,
    format_trip,
    format_search_paths,
    format_path,
    find_poi_by_name as ix_find_poi_by_name,
    filter_pois as ix_filter_pois,
    search_pois as ix_search_pois,
    search_trips as ix_search_trips,
    search_paths as ix_search_paths,
    get_trip as ix_get_trip,
    get_path as ix_get_path,
    find_section,
    get_poi as ix_get_poi,
    extract_poi_tags,
    resolve_history_selection,
    sanitize_tourist_answer,
)
from common.lang_support import (
    SUPPORTED_LANGS,
    LANG_RULES as _LANG_RULES,         # re-exported for chat_demo.py
    RECOVERY_MSGS as _RECOVERY_MSGS,   # re-exported for chat_demo.py
    lang_rule,
    recovery_msg,
    is_supported,
)
from common.models import DEFAULT_EVAL_MODEL
from common.textnorm import normalize_text, tokenize

# ── Constants ───────────────────────────────────────────────────────────────────────
QUESTIONS_FILE  = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_INDEX   = PROJECT_ROOT / "indexes" / "ubeda_en.json"
RESULTS_DIR     = PROJECT_ROOT / "results"
DEFAULT_MODEL   = DEFAULT_EVAL_MODEL   # oMLX E2B; the mobile deployment target
MAX_TOOL_ROUNDS = 14
NO_DIRECT_EVIDENCE_PREFIX = "No place record explicitly mentions all of:"
COMPLEMENTARY_SEARCH_INSTRUCTION = (
    "The direct evidence search did not find one place that combines all "
    "requested concepts. Do not ask the visitor a follow-up question. In "
    "this same turn, retrieve each concept separately with search_pois, "
    "filter_pois, or get_section, then give clearly labelled complementary "
    "options. State that the available visitor information does not confirm "
    "the combination; do not imply that separate places satisfy it."
)
ROUTE_INTENT_TERMS = frozenset({
    # English
    "route", "walking", "walk", "cycling", "cycle", "bicycle", "bike",
    "trail", "track", "hiking", "hike", "trek",
    # Spanish / Catalan / Galician / Basque
    "ruta", "caminar", "caminata", "sendero", "senderismo", "bicicleta",
    "ciclismo", "paseo", "camin", "bici", "ibilbide",
    # Italian / French / Portuguese
    "percorso", "piedi", "sentiero", "camminata", "bici", "bicicletta",
    "randonnée", "randonnee", "velo", "ciclovia", "caminho",
    # German / Dutch / Croatian
    "wander", "wanderweg", "fahrrad", "radweg", "spaziergang", "route",
    "wandeling", "fiets", "staza", "pjesa", "setnja",
})
ROUTE_SEARCH_INSTRUCTION = (
    "This is a physical walking, cycling, trail, track, or route request. "
    "You must call search_paths now. Do not ask the visitor to clarify "
    "before checking the available physical routes. Never substitute a "
    "curated trip for a physical route."
)
NO_PATH_ANSWER_INSTRUCTION = (
    "The visitor information has no matching physical path. State that "
    "clearly and concisely now. Do not ask a follow-up question and do not "
    "substitute a curated trip as though it were a route."
)


SOURCE_GROUNDING_TOOLS = frozenset({
    "get_poi", "get_section", "find_poi_by_name", "filter_pois",
    "search_pois", "search_trips", "get_trip", "search_paths", "get_path",
})
GROUNDING_REQUIRED_INSTRUCTION = (
    "This is a tourist information request and requires current source "
    "retrieval from the downloaded index before answering. Call the "
    "appropriate retrieval tool now. Do not answer from previous "
    "assistant prose or general knowledge."
)
SOCIAL_ONLY_MESSAGES = frozenset({
    "hola", "hello", "hi", "hey", "gracias", "thanks", "thank you",
    "adios", "adiós", "bye", "bonjour", "ciao", "hallo", "ola",
})
GROUNDING_FAILURE_MESSAGES = {
    "ca": "No he pogut recuperar informació turística verificada de les dades descarregades.",
    "de": "Ich konnte keine verifizierten touristischen Informationen aus den heruntergeladenen Daten abrufen.",
    "en": "I could not retrieve verified visitor information from the downloaded data.",
    "es": "No he podido recuperar información turística verificada de los datos descargados.",
    "eu": "Ezin izan dut deskargatutako datuetatik turismo-informazio egiaztatua berreskuratu.",
    "fr": "Je n’ai pas pu récupérer d’informations touristiques vérifiées depuis les données téléchargées.",
    "gl": "Non puiden recuperar información turística verificada dos datos descargados.",
    "hi": "मैं डाउनलोड किए गए डेटा से सत्यापित पर्यटन जानकारी प्राप्त नहीं कर सका।",
    "hr": "Nisam uspio dohvatiti provjerene turističke informacije iz preuzetih podataka.",
    "it": "Non sono riuscito a recuperare informazioni turistiche verificate dai dati scaricati.",
    "ja": "ダウンロード済みデータから検証済みの観光情報を取得できませんでした。",
    "nl": "Ik kon geen geverifieerde toeristische informatie uit de gedownloade gegevens ophalen.",
    "pt": "Não consegui recuperar informações turísticas verificadas dos dados transferidos.",
    "ru": "Не удалось получить проверенную туристическую информацию из загруженных данных.",
    "uk": "Не вдалося отримати перевірену туристичну інформацію із завантажених даних.",
    "zh": "无法从已下载的数据中获取经过验证的旅游信息。",
}
def is_physical_route_request(question: str) -> bool:
    """Detect a physical-route request across the supported app languages."""
    terms = tokenize(question)
    return any(
        term in ROUTE_INTENT_TERMS
        or any(
            len(term) >= 4 and term.startswith(route)
            for route in ROUTE_INTENT_TERMS if len(route) >= 4
        )
        for term in terms
    )


def requires_current_turn_grounding(question: str) -> bool:
    """True for all non-social requests in the tourist assistant."""
    normalized = normalize_text(question)
    return bool(normalized) and normalized not in SOCIAL_ONLY_MESSAGES


def grounding_failure_message(index: dict) -> str:
    """Return a localized safe failure instead of ungrounded tourism prose."""
    lang = (index.get("meta") or {}).get("lang") or "en"
    return GROUNDING_FAILURE_MESSAGES.get(lang, GROUNDING_FAILURE_MESSAGES["en"])


def selected_source_context(selection: dict, result: str) -> str:
    """Build internal context after a validated prior-tag selection."""
    return (
        f"The visitor selected this previously shown {selection['kind']}. "
        "Use this freshly retrieved source record to answer, including "
        "available ordered stops. Do not answer from earlier assistant "
        f"paraphrase:\n\n{result}"
    )


def route_lookup_context(result: str) -> str:
    """Return an internal instruction after a forced physical path lookup."""
    if result.startswith("No curated routes matched"):
        return (
            "A physical-route lookup has already completed and found no "
            "matching published route. Answer the visitor clearly now. Do "
            "not ask a follow-up question and do not substitute a trip."
        )
    return (
        "A physical-route lookup has already completed. Use these route "
        "results to answer the visitor; do not offer a curated trip as a "
        "physical route:\n\n" + result
    )
_SYSTEM_PROMPT_TEMPLATE = """\
You are a tourism assistant for {destination}.  You answer visitor \
questions using the {destination} POI index, which is a structured catalogue \
of every point of interest, trip and itinerary in the destination.

The full section catalogue is listed below — you do NOT need to call any \
tool to discover it.  Use this information directly.

You have TEN tools. Pick the one that fits the question:

  • get_section(section_id, sort?, limit?)
        List POIs inside one section.  Returns id + name + a one-line preview.
        Use when the user asks "what X exist?", "list all Y in <category>".

  • get_poi(poi_id)
        Full record of one POI: type, address, phone, coordinates, links, \
AND the full description paragraph.
        Use when you need facts (address, phone, dates, description) about \
a specific named POI.  Pass several comma-separated ids \
('poi/123,poi/456') to fetch multiple POIs in one call when comparing \
or synthesising.

  • find_poi_by_name(query, limit?, detail?)
        Fuzzy lookup by POI name.  Returns up to `limit` candidates with id + \
section + preview.  Use when the user names a place but you don't know \
which section it lives in.  Pass detail="full" to also get the best match's \
complete record in the same call; with the default detail="brief", always \
follow up with get_poi() on the best match before answering specific facts.

  • filter_pois(interest_level?, type?, tourist_type?, section_id?, \
indispensable?, limit?)
        Facet query.  All filters AND together.  Examples:
          - filter_pois(indispensable=true) → must-see POIs
          - filter_pois(tourist_type="FOOD TOURISM", limit=10) → food spots
          - filter_pois(type="OilMill") → all olive-oil mills / producers
          - filter_pois(type="Restaurant", section_id="gastronomy") → restaurants
          - filter_pois(interest_level=1, section_id="religious-heritage")
        Use the UNE type codes you see in tool results (e.g. Restaurant, \
OilMill, Museum); do not guess codes from the user's words.
  • search_pois(query, section_id?, limit?)
        Evidence search across POI names and visitor descriptions. Use it to \
check whether several visitor concepts appear on the SAME place, for example \
"olive oil restaurant", "family museum", or "accessible parking". Results \
include the supporting text. Use a short concept phrase, not the entire \
visitor question.

  • search_trips(query, limit?)
        Search curated suggestions for what to do over a theme, day, or
        multi-day visit. These are editorial plans, NOT walking/biking
        routes. Use for "what should I do for two days?", themed visits,
        or curated itinerary suggestions.

  • get_trip(trip_id)
        Return the full ordered stops and description of one curated trip
        suggestion. Use after search_trips().

  • search_paths(query, limit?)
        Search physical routes fetched from the destination's /paths data.
        Use ONLY for walking, cycling, trail, track, or route requests.
        Never use a trip as a substitute for a physical route.

  • get_path(path_id)
        Return the full ordered waypoint stops of one physical route.
        Use after search_paths().

  • list_sections()
        Returns the catalogue below.  Rarely needed — sections are \
pre-loaded.

--- DESTINATION OVERVIEW ---
{destination_overview}

--- SECTIONS (pre-loaded, do not fetch again) ---
{sections_text}
--- END SECTIONS ---

RULES:
- Answer based ONLY on what your tools return.  Do not use outside knowledge.
- Always include the description paragraph from get_poi() when answering \
about a specific place — it carries the most useful detail.
- Quote exact names, addresses, phones, coordinates, and dates when present.
- For "what should I not miss?" / "best of" questions, use \
filter_pois(indispensable=true) before browsing sections.
- For "tell me about <name>" / "what is <name>" questions, call \
find_poi_by_name() with detail="full" first — it returns the best match's \
full record in one call.
- For "what to do", themed visits, one-day, or multi-day plan requests,
  search curated trips first. For walking, cycling, trail, track, or
  route requests, search physical paths first. A curated trip is never a
  physical route; if no path is available, say so rather than substituting
  a trip. Do not ask the visitor to reformulate the route request.
- After filter_pois: if the question needs a description, dates, \
address, phone, architect, or any per-POI detail beyond the name, \
call get_poi on the most relevant result before answering. \
For pure listing questions (e.g. "what hotels are there?", \
"list all museums"), the filter_pois previews already include name + \
type + interest level, so an extra get_poi call is unnecessary.
- If information is not in the index after trying all relevant tools, \
say so simply — do not repeat the question or summarise what you tried.
- When filter_pois returns no results, do not stop: try \
get_section("gastronomy") or another relevant section, or try a \
different UNE type code (e.g. for olive oil: OilMill, Restaurant), \
before concluding nothing was found.  Look at the type codes shown in \
section previews to pick a valid one.
- For any compound visitor request ("X with Y", "X near Y", "X that \
offers Y"), use search_pois() with the combined concepts first. Only say \
a place has BOTH properties when that same record explicitly supports \
both. If there is no direct match, still be helpful: retrieve each \
concept separately and present clearly-labelled complementary options. \
Say the available visitor information does not confirm the combination; \
never imply that separate options satisfy it. Do this in the SAME turn: \
do not stop, do not ask the visitor which search they prefer, and do not \
make them repeat their request.
- Never expose internal details in your answers: no type codes, no \
tool names, no filter parameter names, no catalog terminology, and no \
raw IDs. Speak like a local tourism host, not a database interface.
- Tag every point of interest you mention.  The tag WRAPS the name \
(the name goes BETWEEN opening and closing tag): \
  CORRECT: <poi id=5155 type=PlaceOfWorship>Church of San Nicolás</poi> \
  WRONG:   Church of San Nicolás (<poi id=5155 type=PlaceOfWorship>) \
  WRONG:   <poi id=5155 type=PlaceOfWorship> Church of San Nicolás \
Bare numeric id, no quotes.  Do NOT write the type or interest level \
in the answer prose — they belong only in the tag attribute.  Never \
show raw 'poi/…' ids outside a tag.
- OUTPUT VALIDITY REQUIREMENT: when a tool gives you a POI tag, copy that \
exact tag around every mention of that POI in the final answer. A plain \
POI name without its <poi ...> tag is invalid output because the mobile \
app cannot open it. Do not replace a tagged name with an untagged \
paraphrase. Only use an id that appeared in a tool result; NEVER invent \
or guess an id. If no tool has provided an id for a name, leave that \
name untagged rather than creating a tag.
- {{lang_rule}}
"""


def make_system_prompt(sections_text: str, destination: str,
                       destination_overview: str, lang: str = "en") -> str:
    """Build the system prompt with sections and overview embedded."""
    overview = destination_overview.strip() or "(no overview available)"
    return _SYSTEM_PROMPT_TEMPLATE.replace("{{lang_rule}}", lang_rule(lang)).format(
        sections_text=sections_text,
        destination=destination,
        destination_overview=overview,
    )


# ── Tool definitions exposed to the LLM ─────────────────────────────────────

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": (
                "List the POIs inside one section.  Returns id + name + "
                "a one-line preview for each POI.  Pass the section_id "
                "from the catalogue above (preferred) OR the exact title."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "Section id or title.",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["interest", "name", "zoom"],
                        "description": "Sort order; default 'interest' (most important first).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": ("Max POIs to return (default 50; "
                                        "20 for large sections that show a group map)."),
                    },
                },
                "required": ["section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_poi",
            "description": (
                "Return the full record of one POI: address, phone, "
                "coordinates, links, AND the full description paragraph.  "
                "Pass the full id ('poi/12345'), the bare number, or "
                "several comma-separated ids ('poi/123,poi/456') to fetch "
                "multiple POIs in one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_id": {
                        "type": "string",
                        "description": ("POI id(s): 'poi/5155', '5155', or "
                                        "comma-separated 'poi/123,poi/456'."),
                    },
                },
                "required": ["poi_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_poi_by_name",
            "description": (
                "Fuzzy POI name lookup.  Diacritic-insensitive.  Returns "
                "id + name + section + preview for up to `limit` matches.  "
                "With detail=\"full\" the best match's complete record is "
                "included in the same response — no follow-up get_poi() "
                "needed.  With the default \"brief\", follow up with "
                "get_poi() on the best match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-form POI name to search for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 5).",
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["brief", "full"],
                        "description": ("'full' also returns the best match's "
                                        "complete POI record (default 'brief')."),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_pois",
            "description": (
                "Facet query.  All filters AND together.  Use for "
                "'indispensable POIs', 'all OilMills', 'food-tourism POIs in "
                "<section>', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "interest_level": {
                        "type": "integer",
                        "description": "1=Indispensable, 2=Interesting, 3=Outstanding.",
                    },
                    "type": {
                        "type": "string",
                        "description": "UNE 178503 type code, e.g. 'OilMill', 'Museum'.",
                    },
                    "tourist_type": {
                        "type": "string",
                        "description": "Tourist-type code, e.g. 'FOOD TOURISM', 'HERITAGE TOURISM'.",
                    },
                    "section_id": {
                        "type": "string",
                        "description": "Restrict to a section.",
                    },
                    "indispensable": {
                        "type": "boolean",
                        "description": "Shortcut for interest_level=1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max POIs to return (default 20).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pois",
            "description": (
                "Search POI names and visitor descriptions for explicit "
                "evidence that every word in a concise query applies to "
                "the same place. Use before claiming that a place combines "
                "multiple visitor needs. Results include supporting text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Short evidence phrase, e.g. 'olive oil "
                            "restaurant' or 'accessible parking'."
                        ),
                    },
                    "section_id": {
                        "type": "string",
                        "description": "Optional section to search within.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trips",
            "description": (
                "Search curated suggestions for what to do over a theme, "
                "day, or multi-day visit. These are editorial suggestions, "
                "not physical walking or biking routes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short trip theme or visit goal.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum suggestions (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trip",
            "description": (
                "Return one curated trip suggestion with its ordered stops. "
                "Use only after search_trips."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trip_id": {
                        "type": "string",
                        "description": "Trip id, e.g. 'trip/4407' or '4407'.",
                    },
                },
                "required": ["trip_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_paths",
            "description": (
                "Search physical walking, cycling, trail, track, or route "
                "records fetched from /v120/paths. Never returns a trip."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short route, activity, or trail query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum routes (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_path",
            "description": (
                "Return one physical route from /v120/paths with ordered "
                "waypoint stops. Use only after search_paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path_id": {
                        "type": "string",
                        "description": "Path id, e.g. 'path/123' or '123'.",
                    },
                },
                "required": ["path_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sections",
            "description": (
                "Return the section catalogue.  Already embedded in your "
                "system prompt — call only as a refresher."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ── Tool dispatch ──────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, index: dict,
                 sections_text: str, cache: dict) -> tuple[str, bool]:
    """Run a tool call against the index.

    Returns (text_result, cache_hit).  `cache` is shared across calls within
    a session and is keyed by (tool, normalised-arg-tuple).
    """
    if name == "list_sections":
        return sections_text, True   # always pre-warmed

    if name == "get_section":
        section_id = (args.get("section_id") or "").strip()
        sort = (args.get("sort") or "interest").lower()
        # limit=None (not supplied) -> format_section applies the adaptive
        # default (20 for grouped sections, 50 for flat ones).  The cache
        # key preserves None so the prewarmed entry matches.
        raw_limit = args.get("limit")
        limit = int(raw_limit) if raw_limit not in (None, "") else None
        key = ("get_section", section_id.lower(), sort, limit)
        if key in cache:
            return cache[key], True
        result = format_section(index, section_id, sort=sort, limit=limit)
        cache[key] = result
        return result, False

    if name == "get_poi":
        poi_id = (args.get("poi_id") or "").strip()
        key = ("get_poi", poi_id)
        if key in cache:
            return cache[key], True
        result = format_poi(index, poi_id)
        cache[key] = result
        return result, False

    if name == "find_poi_by_name":
        query = (args.get("query") or "").strip()
        limit = int(args.get("limit") or 5)
        detail = (args.get("detail") or "brief").lower()
        key = ("find_poi_by_name", query.lower(), limit, detail)
        if key in cache:
            return cache[key], True
        result = format_find_poi_by_name(index, query, limit=limit,
                                         detail=detail)
        cache[key] = result
        return result, False

    if name == "filter_pois":
        active = {k: v for k, v in args.items()
                  if v not in (None, "", [], {})}
        limit = int(active.pop("limit", 20))
        key = ("filter_pois", tuple(sorted(active.items())), limit)
        if key in cache:
            return cache[key], True
        result = format_filter_pois(index, limit=limit, **active)
        cache[key] = result
        return result, False
    if name == "search_pois":
        query = (args.get("query") or "").strip()
        section_id = (args.get("section_id") or "").strip() or None
        limit = int(args.get("limit") or 10)
        key = ("search_pois", query.lower(), section_id, limit)
        if key in cache:
            return cache[key], True
        result = format_search_pois(index, query, section_id=section_id,
                                    limit=limit)
        cache[key] = result
        return result, False
    if name == "search_trips":
        query = (args.get("query") or "").strip()
        limit = int(args.get("limit") or 10)
        key = ("search_trips", query.lower(), limit)
        if key in cache:
            return cache[key], True
        result = format_search_trips(index, query, limit=limit)
        cache[key] = result
        return result, False
    if name == "get_trip":
        trip_id = (args.get("trip_id") or "").strip()
        key = ("get_trip", trip_id)
        if key in cache:
            return cache[key], True
        result = format_trip(index, trip_id)
        cache[key] = result
        return result, False
    if name == "search_paths":
        query = (args.get("query") or "").strip()
        limit = int(args.get("limit") or 10)
        key = ("search_paths", query.lower(), limit)
        if key in cache:
            return cache[key], True
        result = format_search_paths(index, query, limit=limit)
        cache[key] = result
        return result, False
    if name == "get_path":
        path_id = (args.get("path_id") or "").strip()
        key = ("get_path", path_id)
        if key in cache:
            return cache[key], True
        result = format_path(index, path_id)
        cache[key] = result
        return result, False

    return f"[ERROR] Unknown tool: {name}", False


# ── Section-access tracking (used by the rubric) ─────────────────────────────

def _section_titles_for_poi(index: dict, poi_id: str) -> list[str]:
    """Return section titles owning a POI (usually one)."""
    by_section = (index.get("facets") or {}).get("by_section") or {}
    out = []
    for sid, ids in by_section.items():
        if poi_id in ids:
            sec = find_section(index, sid)
            if sec:
                out.append(sec.get("title", ""))
    return out


def sections_accessed_from_calls(tool_calls: list, index: dict) -> list[str]:
    """Map a sequence of tool calls to the section titles touched.

    This drives the eval rubric's retrieval-accuracy score.
    """
    seen: list[str] = []

    def add(title: str) -> None:
        if title and title not in seen:
            seen.append(title)

    def add_itinerary_sections(itinerary: dict | None) -> None:
        if not itinerary:
            return
        for step in itinerary.get("steps") or []:
            for poi_id in step.get("poi_ids") or []:
                for title in _section_titles_for_poi(index, poi_id):
                    add(title)

    for call in tool_calls:
        tool = call.get("tool")
        args = call.get("args") or {}

        if tool == "get_section":
            sec = find_section(index, (args.get("section_id") or ""))
            if sec:
                add(sec.get("title", ""))

        elif tool == "get_poi":
            # poi_id may carry several comma-separated ids (batch fetch)
            poi_ids = [p.strip() for p in (args.get("poi_id") or "").split(",")
                       if p.strip()]
            for poi_id in poi_ids:
                poi = ix_get_poi(index, poi_id)
                if poi:
                    for t in _section_titles_for_poi(index, poi["poi_id"]):
                        add(t)

        elif tool == "find_poi_by_name":
            for poi in ix_find_poi_by_name(index,
                                           args.get("query") or "",
                                           limit=int(args.get("limit") or 5)):
                for t in _section_titles_for_poi(index, poi["poi_id"]):
                    add(t)

        elif tool == "filter_pois":
            facet_args = {k: v for k, v in args.items()
                          if k != "limit" and v not in (None, "", [], {})}
            if "section_id" in facet_args:
                sec = find_section(index, facet_args["section_id"])
                if sec:
                    add(sec.get("title", ""))
                continue
            limit = int(args.get("limit") or 20)
            for poi in ix_filter_pois(index, **facet_args)[:limit]:
                for t in _section_titles_for_poi(index, poi["poi_id"]):
                    add(t)

        elif tool == "search_pois":
            section_id = (args.get("section_id") or "").strip() or None
            limit = int(args.get("limit") or 10)
            for item in ix_search_pois(
                index,
                args.get("query") or "",
                section_id=section_id,
                limit=limit,
            ):
                poi = item["poi"]
                for t in _section_titles_for_poi(index, poi["poi_id"]):
                    add(t)

        elif tool == "search_trips":
            for itinerary in ix_search_trips(
                index, args.get("query") or "",
                limit=int(args.get("limit") or 10),
            ):
                add_itinerary_sections(itinerary)

        elif tool == "get_trip":
            add_itinerary_sections(ix_get_trip(
                index, args.get("trip_id") or "",
            ))

        elif tool == "search_paths":
            for itinerary in ix_search_paths(
                index, args.get("query") or "",
                limit=int(args.get("limit") or 10),
            ):
                add_itinerary_sections(itinerary)

        elif tool == "get_path":
            add_itinerary_sections(ix_get_path(
                index, args.get("path_id") or "",
            ))

        # list_sections doesn't access content
    return seen


# ── Agentic loop ───────────────────────────────────────────────────────────

def run_agentic_loop(question: str, system_prompt: str,
                     index: dict, sections_text: str,
                     model: str, cache: dict,
                     recovery_msg: str = "") -> dict:
    """Run the tool-calling loop for one question."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    tool_calls_made = []
    answer = ""
    error  = None
    cache_hits = 0
    rounds = 0
    prompt_tokens = 0
    completion_tokens = 0
    direct_evidence_missing = False
    complementary_retrieval_started = False
    physical_route_request = is_physical_route_request(question)
    path_search_started = False
    no_matching_path = False
    no_path_answer_enforced = False
    route_lookup_enforced = False
    grounding_required = requires_current_turn_grounding(question)
    grounded = False
    grounding_retry_enforced = False
    grounding_tools: list[str] = []
    automatic_source_calls: list[dict] = []

    for round_num in range(MAX_TOOL_ROUNDS):
        rounds = round_num + 1
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=TOOL_DEFS,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            error = str(exc)
            break

        # Token accounting (feeds the on-device budgets in
        # docs/mobile-offline-contract.md)
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens     += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        choice  = response.choices[0]
        message = choice.message

        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            if physical_route_request and not path_search_started:
                # E2B occasionally responds to a route request with a
                # clarification question without calling search_paths.
                # Perform exactly one deterministic lookup ourselves and
                # feed the result back; never re-inject the same demand in
                # a loop.
                if route_lookup_enforced:
                    answer = sanitize_tourist_answer(
                        (message.content or "").strip(), index
                    )
                    break
                route_lookup_enforced = True
                route_result, route_hit = execute_tool(
                    "search_paths", {"query": question}, index,
                    sections_text, cache,
                )
                path_search_started = True
                no_matching_path = route_result.startswith(
                    "No curated routes matched"
                )
                if route_hit:
                    cache_hits += 1
                tool_calls_made.append({
                    "tool": "search_paths",
                    "args": {"query": question},
                    "result_preview": route_result[:300],
                    "cache_hit": route_hit,
                    "automatic": True,
                })
                grounded = True
                grounding_tools.append("search_paths")
                automatic_source_calls.append({
                    "tool": "search_paths",
                    "args": {"query": question},
                })
                messages.append({
                    "role": "user",
                    "content": route_lookup_context(route_result),
                })
                continue
            if no_matching_path and not no_path_answer_enforced:
                no_path_answer_enforced = True
                messages.append({
                    "role": "user",
                    "content": NO_PATH_ANSWER_INSTRUCTION,
                })
                continue
            # A small model can correctly discover that no single record
            # supports a compound request, then wrongly stop and ask the
            # visitor to choose a next search. Force one generic recovery
            # turn: retrieve complementary options in the same answer.
            if direct_evidence_missing and not complementary_retrieval_started:
                messages.append({
                    "role": "user",
                    "content": COMPLEMENTARY_SEARCH_INSTRUCTION,
                })
                continue
            if grounding_required and not grounded:
                if not grounding_retry_enforced:
                    grounding_retry_enforced = True
                    messages.append({
                        "role": "user",
                        "content": GROUNDING_REQUIRED_INSTRUCTION,
                    })
                    continue
                answer = grounding_failure_message(index)
                break
            answer = sanitize_tourist_answer((message.content or "").strip(), index)
            break

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args: dict = {}
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                pass

            result, hit = execute_tool(fn_name, fn_args, index, sections_text, cache)
            if fn_name in SOURCE_GROUNDING_TOOLS:
                grounded = True
                if fn_name not in grounding_tools:
                    grounding_tools.append(fn_name)
            if fn_name == "search_pois":
                if result.startswith(NO_DIRECT_EVIDENCE_PREFIX):
                    direct_evidence_missing = True
                elif direct_evidence_missing:
                    complementary_retrieval_started = True
            elif direct_evidence_missing and fn_name in {
                "filter_pois", "get_section", "find_poi_by_name",
            }:
                complementary_retrieval_started = True
            if fn_name == "search_paths":
                path_search_started = True
                if result.startswith("No curated routes matched"):
                    no_matching_path = True
            if hit:
                cache_hits += 1
            tool_calls_made.append({
                "tool":           fn_name,
                "args":           fn_args,
                "result_preview": result[:300],
                "cache_hit":      hit,
            })
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    if not answer and grounding_required and not grounded:
        answer = grounding_failure_message(index)
    if not answer:
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                answer = sanitize_tourist_answer(msg["content"].strip(), index)
                break

    if not answer and not error:
        msg = recovery_msg or _RECOVERY_MSGS["en"]
        try:
            recovery = litellm.completion(
                model=model,
                messages=messages + [{"role": "user", "content": msg}],
                temperature=0,
            )
            usage = getattr(recovery, "usage", None)
            if usage:
                prompt_tokens     += getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            answer = sanitize_tourist_answer(
                (recovery.choices[0].message.content or "").strip(), index
            )
        except Exception as exc:
            error = f"recovery failed: {exc}"

    return {
        "answer":     answer,
        "tool_calls": tool_calls_made,
        "rounds":     rounds,
        "cache_hits": cache_hits,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "grounded": grounded or not grounding_required,
        "grounding_tools": grounding_tools,
        "automatic_source_calls": automatic_source_calls,
        "error":      error,
    }


# ── Inputs & helpers ───────────────────────────────────────────────────────

def load_inputs(questions_file: Path | None = None,
                index_file: Path | None = None) -> tuple[list, dict]:
    """Load questions and the POI index.  Fail fast if missing."""
    q_file = questions_file or QUESTIONS_FILE
    i_file = index_file or DEFAULT_INDEX
    for path in (q_file, i_file):
        if not path.exists():
            print(f"[ERROR] Missing: {path}", file=sys.stderr)
            sys.exit(1)
    with open(q_file, encoding="utf-8") as f:
        questions = json.load(f)
    index = load_index(i_file)
    return questions, index


# ── Main ───────────────────────────────────────────────────────────────────

def _resolve_index_arg(args) -> Path:
    """Accept --index OR legacy --structure (with deprecation note)."""
    if args.index:
        path = Path(args.index)
    elif args.structure:
        # Legacy compatibility shim: try to remap old structure paths to
        # the new index file by stripping '_guide' and '_structure'.
        legacy = Path(args.structure)
        guess_name = legacy.name.replace("_guide", "").replace(
            "_structure.json", ".json")
        guessed = legacy.parent.parent / "indexes" / guess_name
        if guessed.exists():
            print(f"[WARN] --structure is deprecated; using {guessed}",
                  file=sys.stderr)
            path = guessed
        else:
            path = legacy
    else:
        path = DEFAULT_INDEX
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POI-index Q&A evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"litellm model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--questions", default=None,
                        help="Path to questions JSON (default: eval/questions.json)")
    parser.add_argument("--index", default=None,
                        help=f"Path to POI index JSON (default: {DEFAULT_INDEX})")
    parser.add_argument("--structure", default=None,
                        help=argparse.SUPPRESS)  # legacy, hidden
    parser.add_argument("--lang", default="en",
                        help=("Response language code (default: en). "
                              "One of: " + ", ".join(SUPPORTED_LANGS)))
    args = parser.parse_args()

    if not is_supported(args.lang):
        print(f"[ERROR] Unsupported --lang '{args.lang}'. "
              f"Supported codes: {', '.join(SUPPORTED_LANGS)}",
              file=sys.stderr)
        sys.exit(1)

    questions_file = Path(args.questions) if args.questions else QUESTIONS_FILE
    index_path     = _resolve_index_arg(args)

    questions, index = load_inputs(questions_file, index_path)

    destination_display = (index.get("meta") or {}).get("destination_display") \
                          or (index.get("meta") or {}).get("destination") \
                          or "this destination"
    sections_text = format_sections_overview(index)
    overview_text = index.get("destination_overview", "")
    system_prompt = make_system_prompt(
        sections_text=sections_text,
        destination=destination_display,
        destination_overview=overview_text,
        lang=args.lang,
    )

    # Pre-warm: cache get_section for every section with the adaptive
    # default limit (None), matching what the model's calls produce.
    cache: dict = {}
    for sec in index.get("sections", []):
        sid = sec.get("section_id", "")
        if sid:
            cache[("get_section", sid.lower(), "interest", None)] = format_section(
                index, sid, sort="interest", limit=None)
    print(f"[INFO] Pre-warmed cache: {len(cache)} sections")

    recovery = recovery_msg(args.lang)

    model_tag   = args.model.split("/")[-1].replace(":", "-")
    lang_suffix = f"_{args.lang}" if args.lang != "en" else ""
    output_file = RESULTS_DIR / f"eval_{model_tag}{lang_suffix}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"[INFO] Model:          {args.model}")
    print(f"[INFO] Language:       {args.lang}")
    print(f"[INFO] Index:          {index_path.name}  "
          f"({(index.get('meta') or {}).get('poi_count', '?')} POIs)")
    print(f"[INFO] Questions:      {len(questions)}  ({questions_file.name})")
    print(f"[INFO] Output:         {output_file}")
    print(f"[INFO] System prompt:  {len(system_prompt):,} chars\n")

    results = []
    total_start = time.time()

    for i, q in enumerate(questions, 1):
        qid        = q["id"]
        question   = q["question"]
        difficulty = q.get("difficulty", "?")
        print(f"[{i:2d}/{len(questions)}] {qid} ({difficulty})  {question[:70]}...")
        t0 = time.time()

        loop = run_agentic_loop(
            question, system_prompt, index, sections_text,
            args.model, cache, recovery_msg=recovery,
        )

        elapsed = round(time.time() - t0, 2)
        sections = sections_accessed_from_calls(loop["tool_calls"], index)
        poi_refs = extract_poi_tags(loop["answer"], index)

        result = {
            "id":               qid,
            "model":            args.model,
            "lang":             args.lang,
            "category":         q.get("category"),
            "difficulty":       difficulty,
            "question":         question,
            "expected_section": q.get("expected_section"),
            "grounding_check":  q.get("grounding_check"),
            "answer":           loop["answer"],
            "tool_calls":       loop["tool_calls"],
            "sections_accessed": sections,
            "poi_refs":         poi_refs,
            "rounds":           loop["rounds"],
            "cache_hits":       loop["cache_hits"],
            "latency_seconds":  elapsed,
            "prompt_tokens":     loop["prompt_tokens"],
            "completion_tokens": loop["completion_tokens"],
            "grounded":         loop["grounded"],
            "grounding_tools":  loop["grounding_tools"],
            "automatic_source_calls": loop["automatic_source_calls"],
            "error":            loop["error"],
        }
        results.append(result)

        status = "ERROR" if loop["error"] else "OK"
        tools = [c["tool"] for c in loop["tool_calls"]]
        print(f"  [{status}] {elapsed}s  rounds={loop['rounds']}  "
              f"tools={tools}  cache={loop['cache_hits']}  "
              f"tokens={loop['prompt_tokens']}+{loop['completion_tokens']}  "
              f"tags={len(poi_refs)}  grounded={loop['grounded']}")

    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n[INFO] All questions complete in {total_elapsed}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

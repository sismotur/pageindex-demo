#!/usr/bin/env python3
"""
run_eval.py — Q&A evaluation runner over the POI-aware index.

Loads indexes/{destination}_{lang}.json (built by build_index.py) and
runs each question in eval/questions.json through litellm tool calling.

Five tools are exposed to the model:

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
    .venv/bin/python scripts/run_eval.py
    .venv/bin/python scripts/run_eval.py --model openai/gemma4:e4b
    .venv/bin/python scripts/run_eval.py --lang es \
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

# Make `from index_tools import ...` work whether you run as a script or module
sys.path.insert(0, str(Path(__file__).parent))
from index_tools import (
    load_index,
    format_sections_overview,
    format_section,
    format_poi,
    format_find_poi_by_name,
    format_filter_pois,
    find_poi_by_name as ix_find_poi_by_name,
    filter_pois as ix_filter_pois,
    find_section,
    get_poi as ix_get_poi,
)
from lang_support import (
    SUPPORTED_LANGS,
    LANG_RULES as _LANG_RULES,         # re-exported for chat_demo.py
    RECOVERY_MSGS as _RECOVERY_MSGS,   # re-exported for chat_demo.py
    lang_rule,
    recovery_msg,
    is_supported,
)

# ── Constants ───────────────────────────────────────────────────────────────────────
QUESTIONS_FILE  = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_INDEX   = PROJECT_ROOT / "indexes" / "ubeda_en.json"
RESULTS_DIR     = PROJECT_ROOT / "results"
DEFAULT_MODEL   = "openai/gemma4:e2b"
MAX_TOOL_ROUNDS = 14

_SYSTEM_PROMPT_TEMPLATE = """\
You are a tourism assistant for {destination}.  You answer visitor \
questions using the {destination} POI index, which is a structured catalogue \
of every point of interest, trip and itinerary in the destination.

The full section catalogue is listed below — you do NOT need to call any \
tool to discover it.  Use this information directly.

You have FIVE tools.  Pick the one that fits the question:

  • get_section(section_id, sort?, limit?)
        List POIs inside one section.  Returns id + name + a one-line preview.
        Use when the user asks "what X exist?", "list all Y in <category>".

  • get_poi(poi_id)
        Full record of one POI: type, address, phone, coordinates, images, \
links, AND the full description paragraph.
        Use when you need facts (address, phone, dates, description) about \
a specific named POI.

  • find_poi_by_name(query, limit?)
        Fuzzy lookup by POI name.  Returns up to `limit` candidates with id + \
section + preview.  Use when the user names a place but you don't know \
which section it lives in.  Always follow up with get_poi() on the best \
match before answering specific facts.

  • filter_pois(interest_level?, type?, tourist_type?, section_id?, \
indispensable?, limit?)
        Facet query.  All filters AND together.  Examples:
          - filter_pois(indispensable=true) → must-see POIs
          - filter_pois(tourist_type="FOOD TOURISM", limit=10) → food spots
          - filter_pois(type="OilMill") → all olive-oil mills
          - filter_pois(interest_level=1, section_id="religious-heritage")

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
find_poi_by_name() first, then get_poi() on the best match.
- If information is not in the index, say so clearly.
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
                        "description": "Max POIs to return (default 50).",
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
                "coordinates, images, AND the full description paragraph.  "
                "Pass either the full id ('poi/12345') or the bare number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_id": {
                        "type": "string",
                        "description": "POI id, e.g. 'poi/5155' or '5155'.",
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
                "Always follow up with get_poi() on the best match."
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
        limit = int(args.get("limit") or 50)
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
        key = ("find_poi_by_name", query.lower(), limit)
        if key in cache:
            return cache[key], True
        result = format_find_poi_by_name(index, query, limit=limit)
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

    for call in tool_calls:
        tool = call.get("tool")
        args = call.get("args") or {}

        if tool == "get_section":
            sec = find_section(index, (args.get("section_id") or ""))
            if sec:
                add(sec.get("title", ""))

        elif tool == "get_poi":
            poi_id = (args.get("poi_id") or "").strip()
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
            answer = (message.content or "").strip()
            break

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args: dict = {}
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                pass

            result, hit = execute_tool(fn_name, fn_args, index, sections_text, cache)
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

    if not answer:
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                answer = msg["content"].strip()
                break

    if not answer and not error:
        msg = recovery_msg or _RECOVERY_MSGS["en"]
        try:
            recovery = litellm.completion(
                model=model,
                messages=messages + [{"role": "user", "content": msg}],
                temperature=0,
            )
            answer = (recovery.choices[0].message.content or "").strip()
        except Exception as exc:
            error = f"recovery failed: {exc}"

    return {
        "answer":     answer,
        "tool_calls": tool_calls_made,
        "rounds":     rounds,
        "cache_hits": cache_hits,
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

    # Pre-warm: cache get_section for every section (pure dict traversal,
    # so this is essentially free).  Subsequent get_section calls hit cache.
    cache: dict = {}
    for sec in index.get("sections", []):
        sid = sec.get("section_id", "")
        if sid:
            cache[("get_section", sid.lower(), "interest", 50)] = format_section(
                index, sid, sort="interest", limit=50)
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
            "rounds":           loop["rounds"],
            "cache_hits":       loop["cache_hits"],
            "latency_seconds":  elapsed,
            "error":            loop["error"],
        }
        results.append(result)

        status = "ERROR" if loop["error"] else "OK"
        tools = [c["tool"] for c in loop["tool_calls"]]
        print(f"  [{status}] {elapsed}s  rounds={loop['rounds']}  "
              f"tools={tools}  cache={loop['cache_hits']}")

    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n[INFO] All questions complete in {total_elapsed}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

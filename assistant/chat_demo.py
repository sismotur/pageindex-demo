#!/usr/bin/env python3
"""
assistant/chat_demo.py — Multi-turn conversation demo over the POI-aware index.

Two modes:
  • Scripted: runs every conversation thread in eval/ubeda/conversations.json,
    carrying the full message history across turns within a thread.
  • Interactive: --interactive launches a chat where you type questions
    and the answer streams back; the conversation context carries
    across turns until you exit.

Reuses the agentic loop and the six tools defined in run_eval.py.

Usage:
    .venv/bin/python assistant/chat_demo.py
    .venv/bin/python assistant/chat_demo.py --model openai/gemma-4-E2B-it-MLX-8bit
    .venv/bin/python assistant/chat_demo.py --interactive
    .venv/bin/python assistant/chat_demo.py --interactive --lang es \
        --index indexes/ubeda/es.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import litellm
litellm.drop_params = True
litellm.set_verbose = False

# Shared building blocks from run_eval.py (same package) and common/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import (   # noqa: E402
    TOOL_DEFS,
    MAX_TOOL_ROUNDS,
    compact_tool_history,
    execute_tool,
    make_system_prompt,
    DEFAULT_INDEX,
    NO_DIRECT_EVIDENCE_PREFIX,
    COMPLEMENTARY_SEARCH_INSTRUCTION,
    is_physical_route_request,
    is_weather_request,
    no_path_answer_instruction,
    route_lookup_context,
    SOURCE_GROUNDING_TOOLS,
    GROUNDING_RECOVERY_INSTRUCTION,
    NAMED_POI_LOOKUP_INSTRUCTION,
    REASK_FALLBACK_INSTRUCTION,
    LOOP_REPEAT_CACHE_STUB,
    LOOP_REPEAT_INSTRUCTION,
    LOOP_REPEAT_STUB,
    backstop_named_pois,
    chant_repeat_prefix,
    execute_reask_fallback,
    is_pure_reask,
    is_repeat_tool_call,
    reask_fallback_applies,
    requires_current_turn_grounding,
    grounding_failure_message,
    history_followup_answer,
    selected_source_context,
    requires_trip_detail,
    tool_call_key,
    validate_tool_call,
    TRIP_DETAIL_REQUIRED_INSTRUCTION,
    WEATHER_LOOKUP_ENFORCED_INSTRUCTION,
)
from index_tools import (   # noqa: E402
    load_index,
    load_weather,
    format_sections_overview,
    format_section,
    extract_poi_tags,
    format_trip_choice_offer,
    resolve_history_selection,
    resolve_sole_recent_source,
    resolve_trip_query,
    search_trips,
    sanitize_tourist_answer,
    weather_hint,
)
from common.lang_support import (   # noqa: E402
    SUPPORTED_LANGS,
    display_name,
    is_supported,
    recovery_msg as _recovery_msg,
)
from common.models import DEFAULT_CHAT_MODEL   # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
CONVERSATIONS_FILE = PROJECT_ROOT / "eval" / "ubeda" / "conversations.json"
RESULTS_DIR        = PROJECT_ROOT / "results"
DEFAULT_MODEL      = DEFAULT_CHAT_MODEL   # oMLX E2B; the mobile deployment target


# ── Spinner (background thread) ─────────────────────────────────────────

class Spinner:
    """Lightweight terminal spinner that runs in a background thread."""
    _FRAMES = ["\u28cb", "\u28d9", "\u28b9", "\u28b8", "\u28bc", "\u28b4",
               "\u28a6", "\u28a7", "\u2887", "\u288f"]

    def __init__(self) -> None:
        self._msg     = "Thinking"
        self._active  = False
        self._thread: threading.Thread | None = None
        self._lock    = threading.Lock()

    def update(self, msg: str) -> None:
        with self._lock:
            self._msg = msg

    def start(self, msg: str = "Thinking") -> None:
        self._msg    = msg
        self._active = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        for frame in itertools.cycle(self._FRAMES):
            if not self._active:
                break
            with self._lock:
                text = self._msg
            sys.stdout.write(f"\033[2K\r  {frame}  {text}\u2026")
            sys.stdout.flush()
            time.sleep(0.08)

    def stop(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join()
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()


def _status_for_call(name: str, args: dict) -> str:
    """Produce a short status line shown by the spinner during tool calls."""
    if name == "get_section":
        return "Looking through visitor information"
    if name == "get_poi":
        return "Checking place details"
    if name == "find_poi_by_name":
        return "Finding the place"
    if name == "filter_pois":
        return "Finding suitable places"
    if name == "search_pois":
        return "Checking visitor information"
    if name == "search_trips":
        return "Finding visit suggestions"
    if name == "get_trip":
        return "Loading visit suggestion"
    if name == "search_paths":
        return "Looking for routes"
    if name == "get_path":
        return "Loading route details"
    if name == "list_sections":
        return "Checking available information"
    if name == "get_weather":
        return "Checking the forecast"
    return f"Calling {name}"


# ── Single-turn execution (appends to shared history) ─────────────────────

def run_turn(question: str, messages: list[dict],
             index: dict, sections_text: str,
             model: str, cache: dict,
             on_status=None,
             on_stream_start=None,
             stream: bool = False,
             recovery_msg: str = "",
             weather: dict | None = None) -> dict:
    """Execute one conversation turn over the POI index.

    `messages` is mutated in-place with the new user/assistant/tool turns
    so the next call sees full context.
    """
    messages.append({"role": "user", "content": question})
    weather_lookup_enforced = False
    weather_intent = bool(weather) and is_weather_request(question)

    tool_calls_made = []
    answer     = ""
    error      = None
    cache_hits = 0
    rounds     = 0
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
    trip_detail_required = requires_trip_detail(question)
    trip_search_started = False
    trip_search_has_results = False
    trip_search_default: dict | None = None
    trip_detail_started = False
    trip_detail_enforced = False
    source_detail_answer = ""
    # Loop-detector state is local to run_turn, so each new visitor
    # message resets it (a cross-turn re-lookup is legitimate).
    call_keys: list[str] = []
    loop_correction_given = False
    loop_correction_pending: str | None = None
    loop_abort = False
    results_by_key: dict[str, str] = {}

    # Resolve concise selections against validated tags from earlier
    # assistant turns before asking the model to answer. This prevents
    # "Secundaria 2" from being answered from paraphrased chat memory.
    # A generic plan/detail follow-up ("give me the itinerary") names
    # nothing itself, so it can only match the wording-based check above
    # by coincidence — but when exactly one trip/path was just shown,
    # that is the unambiguous referent, so only try it for that specific
    # kind of follow-up (trip_detail_required), never for arbitrary
    # unrelated questions.
    selection = (
        resolve_history_selection(question, messages[:-1], index)
        or resolve_trip_query(question, index)
        or (resolve_sole_recent_source(messages[:-1], index)
            if trip_detail_required else None)
    )
    if selection:
        tool_name, arg_name = {
            "poi": ("get_poi", "poi_id"),
            "trip": ("get_trip", "trip_id"),
            "path": ("get_path", "path_id"),
        }[selection["kind"]]
        result, hit = execute_tool(
            tool_name, {arg_name: selection["id"]}, index, sections_text, cache,
            weather=weather,
        )
        if hit:
            cache_hits += 1
        tool_calls_made.append({
            "tool": tool_name,
            "args": {arg_name: selection["id"]},
            "result_preview": result[:250],
            "cache_hit": hit,
            "automatic": True,
            "source_selection": selection,
        })
        grounded = True
        grounding_tools.append(tool_name)
        if tool_name == "get_trip":
            trip_detail_started = True
        if tool_name in {"get_trip", "get_path"}:
            source_detail_answer = result
        automatic_source_calls.append({
            "tool": tool_name,
            "args": {arg_name: selection["id"]},
            "source_selection": selection,
        })
        messages.append({
            "role": "user",
            "content": selected_source_context(selection, result),
        })
    elif trip_detail_required:
        # No deterministic selection but the visitor asked for a plan.
        # Present up to three curated trips deterministically so they
        # can choose, instead of letting the model pick one silently.
        offer_candidates = search_trips(index, question, limit=3)
        if len(offer_candidates) >= 2:
            offer_text = format_trip_choice_offer(index, offer_candidates[:3])
            tool_calls_made.append({
                "tool": "search_trips",
                "args": {"query": question, "limit": 3},
                "result_preview": offer_text[:250],
                "cache_hit": False,
                "automatic": True,
            })
            grounded = True
            grounding_tools.append("search_trips")
            automatic_source_calls.append({
                "tool": "search_trips",
                "args": {"query": question, "limit": 3},
            })
            trip_search_started = True
            trip_search_has_results = True
            trip_search_default = offer_candidates[0]
            source_detail_answer = offer_text
    def use_default_trip_detail() -> bool:
        """Fetch the top source trip when E2B ignores get_trip once."""
        nonlocal grounded, cache_hits, source_detail_answer, trip_detail_started
        if not trip_search_default:
            return False
        selected_id = trip_search_default["itinerary_id"]
        result, hit = execute_tool(
            "get_trip", {"trip_id": selected_id}, index, sections_text, cache,
            weather=weather,
        )
        if hit:
            cache_hits += 1
        selection_meta = {
            "kind": "trip",
            "id": selected_id,
            "label": trip_search_default.get("name", ""),
        }
        tool_calls_made.append({
            "tool": "get_trip",
            "args": {"trip_id": selected_id},
            "result_preview": result[:250],
            "cache_hit": hit,
            "automatic": True,
            "source_selection": selection_meta,
        })
        grounded = True
        if "get_trip" not in grounding_tools:
            grounding_tools.append("get_trip")
        automatic_source_calls.append({
            "tool": "get_trip",
            "args": {"trip_id": selected_id},
            "source_selection": selection_meta,
        })
        trip_detail_started = True
        source_detail_answer = result
        return True

    for round_num in range(MAX_TOOL_ROUNDS):
        rounds = round_num + 1
        if source_detail_answer:
            answer = sanitize_tourist_answer(source_detail_answer, index)
            messages.append({"role": "assistant", "content": answer})
            break
        compact_tool_history(messages)

        if stream:
            # ── Streaming round ──────────────────────────────────────────
            acc_content    = ""
            acc_tool_calls: list[dict] = []
            streaming_live = False
            chant_detected = False
            # Raw model deltas may contain malformed <poi>/<trip>/<path>
            # tags. Buffer until sanitization has validated the full answer.
            hold_stream_content = True

            try:
                response_stream = litellm.completion(
                    model=model,
                    messages=messages,
                    tools=TOOL_DEFS,
                    tool_choice="auto",
                    temperature=0,
                    stream=True,
                )
            except Exception as exc:
                error = str(exc)
                break

            for chunk in response_stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    acc_content += delta.content
                    keep = chant_repeat_prefix(acc_content)
                    if keep < len(acc_content):
                        # Text-level loop (content chanting): stop the
                        # stream and keep only the non-repetitive prefix.
                        acc_content = acc_content[:keep]
                        chant_detected = True
                        break

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(acc_tool_calls) <= idx:
                            acc_tool_calls.append({"id": "", "name": "", "arguments": ""})
                        if tc_delta.id:
                            acc_tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                acc_tool_calls[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                acc_tool_calls[idx]["arguments"] += tc_delta.function.arguments

            if chant_detected:
                # Best effort: release the abandoned HTTP stream.
                close_stream = getattr(response_stream, "close", None)
                if callable(close_stream):
                    try:
                        close_stream()
                    except Exception:
                        pass

            if streaming_live:
                print()

            assistant_msg = {"role": "assistant", "content": acc_content or ""}
            if acc_tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id":       tc["id"],
                        "type":     "function",
                        "function": {"name": tc["name"],
                                     "arguments": tc["arguments"]},
                    }
                    for tc in acc_tool_calls
                ]
            messages.append(assistant_msg)

            if not acc_tool_calls:
                if chant_detected:
                    # At temperature=0 re-asking would chant again
                    # deterministically, so serve the trimmed prefix as
                    # the answer.  An empty prefix (the chant started on
                    # the first token) falls through to the tail
                    # recovery below.
                    answer = sanitize_tourist_answer(
                        acc_content.strip(), index
                    )
                    assistant_msg["content"] = answer
                    break
                if weather_intent and "get_weather" not in grounding_tools:
                    if not weather_lookup_enforced:
                        weather_lookup_enforced = True
                        weather_result, weather_hit = execute_tool(
                            "get_weather", {}, index,
                            sections_text, cache, weather=weather,
                        )
                        if weather_hit:
                            cache_hits += 1
                        tool_calls_made.append({
                            "tool": "get_weather",
                            "args": {},
                            "result_preview": weather_result[:250],
                            "cache_hit": weather_hit,
                            "automatic": True,
                        })
                        grounded = True
                        if "get_weather" not in grounding_tools:
                            grounding_tools.append("get_weather")
                        automatic_source_calls.append({
                            "tool": "get_weather",
                            "args": {},
                        })
                        messages.append({
                            "role": "user",
                            "content": (WEATHER_LOOKUP_ENFORCED_INSTRUCTION
                                        + "\n\n" + weather_result),
                        })
                        continue
                if physical_route_request and not path_search_started:
                    if route_lookup_enforced:
                        answer = sanitize_tourist_answer(acc_content.strip(), index)
                        break
                    route_lookup_enforced = True
                    route_result, route_hit = execute_tool(
                        "search_paths", {"query": question}, index,
                        sections_text, cache, weather=weather,
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
                        "result_preview": route_result[:250],
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
                        "content": no_path_answer_instruction(
                            "get_weather" in grounding_tools
                        ),
                    })
                    continue
                if direct_evidence_missing and not complementary_retrieval_started:
                    messages.append({
                        "role": "user",
                        "content": COMPLEMENTARY_SEARCH_INSTRUCTION,
                    })
                    continue
                if (trip_detail_required and trip_search_started
                        and trip_search_has_results and not trip_detail_started):
                    if not trip_detail_enforced:
                        trip_detail_enforced = True
                        messages.append({
                            "role": "user",
                            "content": TRIP_DETAIL_REQUIRED_INSTRUCTION,
                        })
                        continue
                    if use_default_trip_detail():
                        continue
                    answer = grounding_failure_message(index)
                    assistant_msg["content"] = answer
                    break
                if grounding_required and not grounded:
                    if not grounding_retry_enforced:
                        grounding_retry_enforced = True
                        messages.append({
                            "role": "user",
                            "content": GROUNDING_RECOVERY_INSTRUCTION,
                        })
                        continue
                    fallback = history_followup_answer(
                        question, messages, index
                    )
                    if fallback:
                        tool_calls_made.append({
                            "tool": "history_followup",
                            "args": {"query": question},
                            "result_preview": fallback[:250],
                            "cache_hit": False,
                            "automatic": True,
                        })
                        automatic_source_calls.append({
                            "tool": "history_followup",
                            "args": {"query": question},
                        })
                        answer = sanitize_tourist_answer(fallback, index)
                        assistant_msg["content"] = answer
                        break
                    # Last-resort backstop for small models: if the
                    # question explicitly names a known place (or confirms
                    # an assistant offer that named places), fetch those
                    # records deterministically instead of accepting a
                    # brush-off.
                    named_pois = backstop_named_pois(question, messages, index)
                    if named_pois:
                        name_index = index.get("name_index") or {}
                        named_ids = ",".join(dict.fromkeys(
                            name_index[n] for n in named_pois if n in name_index
                        ))
                        result, hit = execute_tool(
                            "get_poi", {"poi_id": named_ids}, index,
                            sections_text, cache, weather=weather,
                        )
                        if hit:
                            cache_hits += 1
                        tool_calls_made.append({
                            "tool": "get_poi",
                            "args": {"poi_id": named_ids},
                            "result_preview": result[:250],
                            "cache_hit": hit,
                            "automatic": True,
                            "source_selection": {
                                "kind": "poi",
                                "id": named_ids,
                                "label": ", ".join(named_pois),
                            },
                        })
                        grounded = True
                        if "get_poi" not in grounding_tools:
                            grounding_tools.append("get_poi")
                        automatic_source_calls.append({
                            "tool": "get_poi",
                            "args": {"poi_id": named_ids},
                            "source_selection": {
                                "kind": "poi",
                                "id": named_ids,
                                "label": ", ".join(named_pois),
                            },
                        })
                        messages.append({
                            "role": "user",
                            "content": (NAMED_POI_LOOKUP_INSTRUCTION
                                        + "\n\n" + result),
                        })
                        continue
                    # The model answered the recovery prompt with yet
                    # another clarifying question — a brush-off. In an
                    # ongoing conversation, retrieve real content
                    # deterministically and make it answer from that
                    # instead of re-asking.
                    if (reask_fallback_applies(messages, question)
                            and is_pure_reask(acc_content.strip())):
                        result, hit, fb_tool, fb_args = execute_reask_fallback(
                            question, index, sections_text, cache,
                            weather=weather,
                        )
                        if hit:
                            cache_hits += 1
                        tool_calls_made.append({
                            "tool": fb_tool,
                            "args": fb_args,
                            "result_preview": result[:250],
                            "cache_hit": hit,
                            "automatic": True,
                        })
                        grounded = True
                        if fb_tool not in grounding_tools:
                            grounding_tools.append(fb_tool)
                        automatic_source_calls.append({
                            "tool": fb_tool,
                            "args": fb_args,
                        })
                        messages.append({
                            "role": "user",
                            "content": (REASK_FALLBACK_INSTRUCTION
                                        + "\n\n" + result),
                        })
                        continue
                    # The model declined retrieval after the recovery
                    # prompt and no known place is named: trust its
                    # generic-question classification and deliver the
                    # answer it composed from the preloaded
                    # overview/catalogue.
                    answer = sanitize_tourist_answer(acc_content.strip(), index)
                    assistant_msg["content"] = answer
                    break
                answer = sanitize_tourist_answer(acc_content.strip(), index)
                assistant_msg["content"] = answer
                break

            # Convert accumulated deltas to tool-call dispatch format
            raw_tool_calls = [
                type("TC", (), {"id": tc["id"],
                                "function": type("F", (), {
                                    "name":      tc["name"],
                                    "arguments": tc["arguments"],
                                })()})()
                for tc in acc_tool_calls
            ]

        else:
            # ── Non-streaming round ──────────────────────────────────────
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
                if weather_intent and "get_weather" not in grounding_tools:
                    if not weather_lookup_enforced:
                        weather_lookup_enforced = True
                        weather_result, weather_hit = execute_tool(
                            "get_weather", {}, index,
                            sections_text, cache, weather=weather,
                        )
                        if weather_hit:
                            cache_hits += 1
                        tool_calls_made.append({
                            "tool": "get_weather",
                            "args": {},
                            "result_preview": weather_result[:250],
                            "cache_hit": weather_hit,
                            "automatic": True,
                        })
                        grounded = True
                        if "get_weather" not in grounding_tools:
                            grounding_tools.append("get_weather")
                        automatic_source_calls.append({
                            "tool": "get_weather",
                            "args": {},
                        })
                        messages.append({
                            "role": "user",
                            "content": (WEATHER_LOOKUP_ENFORCED_INSTRUCTION
                                        + "\n\n" + weather_result),
                        })
                        continue
                if physical_route_request and not path_search_started:
                    if route_lookup_enforced:
                        answer = sanitize_tourist_answer(
                            (message.content or "").strip(), index
                        )
                        assistant_msg["content"] = answer
                        break
                    route_lookup_enforced = True
                    route_result, route_hit = execute_tool(
                        "search_paths", {"query": question}, index,
                        sections_text, cache, weather=weather,
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
                        "result_preview": route_result[:250],
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
                        "content": no_path_answer_instruction(
                            "get_weather" in grounding_tools
                        ),
                    })
                    continue
                if direct_evidence_missing and not complementary_retrieval_started:
                    messages.append({
                        "role": "user",
                        "content": COMPLEMENTARY_SEARCH_INSTRUCTION,
                    })
                    continue
                if (trip_detail_required and trip_search_started
                        and trip_search_has_results and not trip_detail_started):
                    if not trip_detail_enforced:
                        trip_detail_enforced = True
                        messages.append({
                            "role": "user",
                            "content": TRIP_DETAIL_REQUIRED_INSTRUCTION,
                        })
                        continue
                    if use_default_trip_detail():
                        continue
                    answer = grounding_failure_message(index)
                    break
                if grounding_required and not grounded:
                    if not grounding_retry_enforced:
                        grounding_retry_enforced = True
                        messages.append({
                            "role": "user",
                            "content": GROUNDING_RECOVERY_INSTRUCTION,
                        })
                        continue
                    fallback = history_followup_answer(
                        question, messages, index
                    )
                    if fallback:
                        tool_calls_made.append({
                            "tool": "history_followup",
                            "args": {"query": question},
                            "result_preview": fallback[:250],
                            "cache_hit": False,
                            "automatic": True,
                        })
                        automatic_source_calls.append({
                            "tool": "history_followup",
                            "args": {"query": question},
                        })
                        answer = sanitize_tourist_answer(fallback, index)
                        assistant_msg["content"] = answer
                        break
                    # Last-resort backstop for small models: if the
                    # question explicitly names a known place (or confirms
                    # an assistant offer that named places), fetch those
                    # records deterministically instead of accepting a
                    # brush-off.
                    named_pois = backstop_named_pois(question, messages, index)
                    if named_pois:
                        name_index = index.get("name_index") or {}
                        named_ids = ",".join(dict.fromkeys(
                            name_index[n] for n in named_pois if n in name_index
                        ))
                        result, hit = execute_tool(
                            "get_poi", {"poi_id": named_ids}, index,
                            sections_text, cache, weather=weather,
                        )
                        if hit:
                            cache_hits += 1
                        tool_calls_made.append({
                            "tool": "get_poi",
                            "args": {"poi_id": named_ids},
                            "result_preview": result[:250],
                            "cache_hit": hit,
                            "automatic": True,
                            "source_selection": {
                                "kind": "poi",
                                "id": named_ids,
                                "label": ", ".join(named_pois),
                            },
                        })
                        grounded = True
                        if "get_poi" not in grounding_tools:
                            grounding_tools.append("get_poi")
                        automatic_source_calls.append({
                            "tool": "get_poi",
                            "args": {"poi_id": named_ids},
                            "source_selection": {
                                "kind": "poi",
                                "id": named_ids,
                                "label": ", ".join(named_pois),
                            },
                        })
                        messages.append({
                            "role": "user",
                            "content": (NAMED_POI_LOOKUP_INSTRUCTION
                                        + "\n\n" + result),
                        })
                        continue
                    # The model answered the recovery prompt with yet
                    # another clarifying question — a brush-off. In an
                    # ongoing conversation, retrieve real content
                    # deterministically and make it answer from that
                    # instead of re-asking.
                    if (reask_fallback_applies(messages, question)
                            and is_pure_reask((message.content or "").strip())):
                        result, hit, fb_tool, fb_args = execute_reask_fallback(
                            question, index, sections_text, cache,
                            weather=weather,
                        )
                        if hit:
                            cache_hits += 1
                        tool_calls_made.append({
                            "tool": fb_tool,
                            "args": fb_args,
                            "result_preview": result[:250],
                            "cache_hit": hit,
                            "automatic": True,
                        })
                        grounded = True
                        if fb_tool not in grounding_tools:
                            grounding_tools.append(fb_tool)
                        automatic_source_calls.append({
                            "tool": fb_tool,
                            "args": fb_args,
                        })
                        messages.append({
                            "role": "user",
                            "content": (REASK_FALLBACK_INSTRUCTION
                                        + "\n\n" + result),
                        })
                        continue
                    # The model declined retrieval after the recovery
                    # prompt and no known place is named: trust its
                    # generic-question classification and deliver the
                    # answer it composed from the preloaded
                    # overview/catalogue.
                    answer = sanitize_tourist_answer(
                        (message.content or "").strip(), index
                    )
                    assistant_msg["content"] = answer
                    break
                answer = sanitize_tourist_answer((message.content or "").strip(), index)
                assistant_msg["content"] = answer
                break

            raw_tool_calls = message.tool_calls

        # Execute tool calls and append results
        for tc in raw_tool_calls:
            fn_name = tc.function.name
            fn_args: dict | None = {}
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = None

            # Keep distinct broken-JSON payloads distinct for the loop
            # detector; identical ones trip it like any other repeat.
            key_args = fn_args if fn_args is not None else {
                "__raw__": (tc.function.arguments or "")[:200]}
            call_keys.append(tool_call_key(fn_name, key_args))
            if is_repeat_tool_call(call_keys):
                # Third identical call this turn: block the execution and
                # answer with a stub.  Correct the model once; on any
                # further repeat, abort the tool loop — the tail recovery
                # then forces a final answer from what it already has.
                if loop_correction_given:
                    loop_abort = True
                else:
                    loop_correction_pending = fn_name
                tool_calls_made.append({
                    "tool":           fn_name,
                    "args":           fn_args or {},
                    "result_preview": LOOP_REPEAT_STUB,
                    "cache_hit":      False,
                    "loop_blocked":   True,
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      LOOP_REPEAT_STUB,
                })
                continue

            cached_result = results_by_key.get(call_keys[-1])
            if cached_result is not None:
                # Second occurrence of an identical call this turn.  The
                # tools are deterministic lookups, so re-execution can
                # return nothing new: serve a short stub while the
                # original result is still in context, or re-serve the
                # cached result itself when compact_tool_history has
                # since replaced it (its own text tells the model to
                # retrieve the source again when needed).
                still_visible = any(
                    m.get("role") == "tool" and m.get("content") == cached_result
                    for m in messages
                )
                repeat_content = (LOOP_REPEAT_CACHE_STUB if still_visible
                                  else cached_result)
                tool_calls_made.append({
                    "tool":           fn_name,
                    "args":           fn_args or {},
                    "result_preview": repeat_content[:250],
                    "cache_hit":      False,
                    "repeat_cached":  True,
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      repeat_content,
                })
                continue

            invalid = validate_tool_call(fn_name, fn_args,
                                         tc.function.arguments or "")
            if invalid:
                tool_calls_made.append({
                    "tool":           fn_name,
                    "args":           fn_args or {},
                    "result_preview": invalid,
                    "cache_hit":      False,
                    "invalid_args":   True,
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      invalid,
                })
                continue

            if on_status:
                on_status(_status_for_call(fn_name, fn_args))

            result, hit = execute_tool(
                fn_name, fn_args, index, sections_text, cache, weather=weather,
            )
            results_by_key[call_keys[-1]] = result
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
            if fn_name == "search_trips":
                trip_search_started = True
                trip_matches = search_trips(
                    index, fn_args.get("query") or "",
                    limit=int(fn_args.get("limit") or 10),
                )
                trip_search_has_results = bool(trip_matches)
                trip_search_default = trip_matches[0] if trip_matches else None
            if fn_name == "get_trip":
                trip_detail_started = True
                if trip_detail_required:
                    source_detail_answer = result
            if fn_name == "get_path":
                source_detail_answer = result
            if hit:
                cache_hits += 1
            tool_calls_made.append({
                "tool":           fn_name,
                "args":           fn_args,
                "result_preview": result[:250],
                "cache_hit":      hit,
            })
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

        if loop_abort:
            break
        if loop_correction_pending:
            blocked_tool = loop_correction_pending
            loop_correction_pending = None
            loop_correction_given = True
            messages.append({
                "role":    "user",
                "content": LOOP_REPEAT_INSTRUCTION.format(tool=blocked_tool),
            })
            continue

    # Fallback: never recover a model-only tourism answer after the round cap.
    if not answer and grounding_required and not grounded:
        answer = grounding_failure_message(index)
    if not answer:
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                answer = sanitize_tourist_answer(msg["content"].strip(), index)
                break

    if not answer and not error:
        msg = recovery_msg or _recovery_msg("en")
        try:
            recovery = litellm.completion(
                model=model,
                messages=messages + [{"role": "user", "content": msg}],
                temperature=0,
            )
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
        "grounded":   grounded or not grounding_required,
        "grounding_tools": grounding_tools,
        "automatic_source_calls": automatic_source_calls,
        "error":      error,
    }


# ── Cache pre-warm ─────────────────────────────────────────────────────────

def prewarm_cache(index: dict) -> dict:
    """Populate the per-session cache with one get_section per section.

    limit=None matches execute_tool's no-limit calls (format_section then
    applies the adaptive default: 20 grouped / 50 flat).
    """
    cache: dict = {}
    for sec in index.get("sections", []):
        sid = sec.get("section_id", "")
        if sid:
            cache[("get_section", sid.lower(), "interest", None)] = format_section(
                index, sid, sort="interest", limit=None)
    return cache


# ── Conversation runner ────────────────────────────────────────────────────────

def run_conversation(thread: dict, system_prompt: str,
                     index: dict, sections_text: str,
                     model: str,
                     weather: dict | None = None) -> dict:
    """Run all turns of a conversation thread sharing one cache + history."""
    cache    = prewarm_cache(index)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    turns_log = []
    conv_start = time.time()

    print(f"\n{'─'*70}")
    print(f"  {thread['id']}  {thread['title']}")
    if thread.get("description"):
        print(f"  {thread['description']}")
    print(f"{'─'*70}")

    for i, turn_spec in enumerate(thread["turns"], 1):
        question = turn_spec["question"]
        print(f"\n  Turn {i}/{len(thread['turns'])}: {question}")

        t0 = time.time()
        result = run_turn(question, messages,
                          index, sections_text, model, cache,
                          weather=weather)
        elapsed = round(time.time() - t0, 2)

        status = "ERROR" if result["error"] else "OK"
        tools_used = [c["tool"] for c in result["tool_calls"]]
        hits = result["cache_hits"]
        total_calls = len(result["tool_calls"])
        print(f"  [{status}] {elapsed}s | tools: {tools_used} | "
              f"cache: {hits}/{total_calls}")
        print(f"  → {result['answer'][:200].replace(chr(10), ' ')}")

        turns_log.append({
            "turn":       i,
            "question":   question,
            "answer":     result["answer"],
            "tool_calls": result["tool_calls"],
            "poi_refs":   extract_poi_tags(result["answer"], index),
            "latency":    elapsed,
            "cache_hits": result["cache_hits"],
            "grounded":   result["grounded"],
            "grounding_tools": result["grounding_tools"],
            "automatic_source_calls": result["automatic_source_calls"],
            "error":      result["error"],
        })

    total_time      = round(time.time() - conv_start, 1)
    total_cache_hits = sum(t["cache_hits"] for t in turns_log)
    total_tool_calls = sum(len(t["tool_calls"]) for t in turns_log)
    total_latency    = sum(t["latency"] for t in turns_log)

    print(f"\n  ✓ {thread['id']} done in {total_time}s | "
          f"cache hits: {total_cache_hits}/{total_tool_calls} | "
          f"avg turn: {total_latency/len(turns_log):.1f}s")

    return {
        "id":               thread["id"],
        "title":            thread["title"],
        "model":            model,
        "total_time":       total_time,
        "context_turns":    len(thread["turns"]),
        "context_messages": len(messages),
        "cache_hits":       total_cache_hits,
        "total_tool_calls": total_tool_calls,
        "turns":            turns_log,
    }


# ── Interactive mode ───────────────────────────────────────────────────────────

def run_interactive(system_prompt: str, index: dict, sections_text: str,
                    model: str, lang: str,
                    destination_name: str,
                    recovery_msg: str,
                    weather: dict | None = None) -> None:
    """Interactive chat session in the terminal."""
    cache    = prewarm_cache(index)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    turn = 0

    print()
    print("─" * 60)
    print(f"  {destination_name} Assistant — Interactive Mode")
    print(f"  Model: {model}")
    print(f"  Language: {display_name(lang)}")
    print("  Type your question and press Enter. 'exit' to quit.")
    print("─" * 60)
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", "bye"}:
            print("Goodbye!")
            break

        turn += 1
        t0 = time.time()

        spinner = Spinner()
        spinner.start("Thinking")
        spinner_stopped = False

        def on_stream_start():
            nonlocal spinner_stopped
            spinner.stop()
            spinner_stopped = True
            sys.stdout.write("Assistant: ")
            sys.stdout.flush()

        result = run_turn(
            question, messages,
            index, sections_text, model, cache,
            on_status=spinner.update,
            on_stream_start=on_stream_start,
            stream=True,
            recovery_msg=recovery_msg,
            weather=weather,
        )

        if not spinner_stopped:
            spinner.stop()
            print("Assistant:", result["answer"])

        elapsed = round(time.time() - t0, 2)

        tools_used = [c["tool"].replace("get_", "") for c in result["tool_calls"]]
        hits  = result["cache_hits"]
        total = len(result["tool_calls"])
        meta = f"[{elapsed}s"
        if tools_used:
            meta += f" | tools: {', '.join(tools_used)}"
        if total:
            meta += f" | cache {hits}/{total}"
        meta += f" | turn {turn}]"
        print(f"\033[2m{meta}\033[0m")
        # The <poi>/<trip>/<path> tags already carry the app's link targets;
        # no secondary URL list is printed.  extract_poi_tags remains
        # available for scripted logs and downstream parsers.
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def _resolve_index_arg(args) -> Path:
    """Accept --index OR legacy --structure."""
    if args.index:
        path = Path(args.index)
    elif args.structure:
        legacy = Path(args.structure)
        guess_name = legacy.name.replace("_guide", "").replace(
            "_structure.json", ".json")
        guessed = legacy.parent.parent / "indexes" / guess_name
        if not guessed.exists():
            # One subfolder per destination: {dest}_{lang}.json ->
            # {dest}/{lang}.json
            dest, _, lang_file = guess_name.rpartition("_")
            if dest:
                guessed = legacy.parent.parent / "indexes" / dest / lang_file
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
    parser = argparse.ArgumentParser(description="Multi-turn POI-index chat demo")
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help=f"litellm model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--interactive",  action="store_true",
                        help="Start an interactive chat session")
    parser.add_argument("--lang",         default="en",
                        help=("Response language code (default: en). "
                              "One of: " + ", ".join(SUPPORTED_LANGS)))
    parser.add_argument("--index",        default=None,
                        help=f"POI index JSON (default: {DEFAULT_INDEX})")
    parser.add_argument("--structure",    default=None,
                        help=argparse.SUPPRESS)  # legacy, hidden
    parser.add_argument("--conversation", default=None,
                        help="Run only this conversation ID (e.g. C01)")
    parser.add_argument("--conversations-file", default=None,
                        help=("Conversations JSON path (default: "
                              f"{CONVERSATIONS_FILE.relative_to(PROJECT_ROOT)}). "
                              "Use this to point scripted mode at a "
                              "destination-specific thread file."))
    parser.add_argument("--output",       default=None,
                        help="Output path (default: results/conversations_<model>.json)")
    args = parser.parse_args()

    if not is_supported(args.lang):
        print(f"[ERROR] Unsupported --lang '{args.lang}'. "
              f"Supported codes: {', '.join(SUPPORTED_LANGS)}",
              file=sys.stderr)
        sys.exit(1)

    index_path = _resolve_index_arg(args)
    if not index_path.exists():
        print(f"[ERROR] Index not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    index = load_index(index_path)

    destination_display = (index.get("meta") or {}).get("destination_display") \
                          or (index.get("meta") or {}).get("destination") \
                          or "Tourism"
    sections_text = format_sections_overview(index)
    overview_text = index.get("destination_overview", "")
    # Optional per-destination weather artifact next to the index.
    dest_slug = (index.get("meta") or {}).get("destination") or ""
    weather_path = PROJECT_ROOT / "weather" / dest_slug / f"{args.lang}.json"
    weather = load_weather(weather_path) if dest_slug else None
    hint_text = weather_hint(weather, destination_display) if weather else ""
    system_prompt = make_system_prompt(
        sections_text=sections_text,
        destination=destination_display,
        destination_overview=overview_text,
        lang=args.lang,
        weather_hint_text=hint_text,
    )
    recovery_msg = _recovery_msg(args.lang)

    # Interactive mode
    if args.interactive:
        run_interactive(system_prompt, index, sections_text,
                        args.model, args.lang,
                        destination_name=destination_display,
                        recovery_msg=recovery_msg,
                        weather=weather)
        return

    # Scripted mode
    conversations_path = Path(args.conversations_file) if args.conversations_file \
                          else CONVERSATIONS_FILE
    if not conversations_path.is_absolute():
        conversations_path = PROJECT_ROOT / conversations_path
    if not conversations_path.exists():
        print(f"[ERROR] Not found: {conversations_path}", file=sys.stderr)
        sys.exit(1)
    with open(conversations_path, encoding="utf-8") as f:
        threads = json.load(f)

    if args.conversation:
        threads = [t for t in threads if t["id"] == args.conversation]
        if not threads:
            print(f"[ERROR] Conversation '{args.conversation}' not found",
                  file=sys.stderr)
            sys.exit(1)

    model_tag = args.model.split("/")[-1].replace(":", "-")
    output_file = Path(args.output) if args.output \
                  else RESULTS_DIR / f"conversations_{model_tag}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"[INFO] Model:         {args.model}")
    print(f"[INFO] Index:         {index_path.name}")
    print(f"[INFO] Conversations: {len(threads)} (from {conversations_path.name})")
    print(f"[INFO] Output:        {output_file}")

    results = []
    total_start = time.time()
    for thread in threads:
        result = run_conversation(thread, system_prompt,
                                  index, sections_text, args.model,
                                  weather=weather)
        results.append(result)
    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n[INFO] All conversations complete in {total_elapsed}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

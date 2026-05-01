#!/usr/bin/env python3
"""
chat_demo.py — Multi-turn conversation demo over the POI-aware index.

Two modes:
  • Scripted: runs every conversation thread in eval/conversations.json,
    carrying the full message history across turns within a thread.
  • Interactive: --interactive launches a chat where you type questions
    and the answer streams back; the conversation context carries
    across turns until you exit.

Reuses the agentic loop and the five tools defined in run_eval.py.

Usage:
    .venv/bin/python scripts/chat_demo.py
    .venv/bin/python scripts/chat_demo.py --model openai/gemma4:e4b
    .venv/bin/python scripts/chat_demo.py --interactive
    .venv/bin/python scripts/chat_demo.py --interactive --lang es \
        --index indexes/ubeda_es.json
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

# Shared building blocks from run_eval.py
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_eval import (   # noqa: E402
    TOOL_DEFS,
    MAX_TOOL_ROUNDS,
    execute_tool,
    make_system_prompt,
    _LANG_RULES,
    _RECOVERY_MSGS,
    DEFAULT_INDEX,
)
from index_tools import (   # noqa: E402
    load_index,
    format_sections_overview,
    format_section,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CONVERSATIONS_FILE = PROJECT_ROOT / "eval" / "conversations.json"
RESULTS_DIR        = PROJECT_ROOT / "results"
DEFAULT_MODEL      = "openai/gemma4:26b"


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
        return f"Loading section {args.get('section_id', '')}"
    if name == "get_poi":
        return f"Loading POI {args.get('poi_id', '')}"
    if name == "find_poi_by_name":
        return f"Searching '{args.get('query', '')}'"
    if name == "filter_pois":
        echo = ", ".join(f"{k}={v}" for k, v in args.items() if v not in (None, "", [], {}))
        return f"Filtering POIs ({echo})"
    if name == "list_sections":
        return "Listing sections"
    return f"Calling {name}"


# ── Single-turn execution (appends to shared history) ─────────────────────

def run_turn(question: str, messages: list[dict],
             index: dict, sections_text: str,
             model: str, cache: dict,
             on_status=None,
             on_stream_start=None,
             stream: bool = False,
             recovery_msg: str = "") -> dict:
    """Execute one conversation turn over the POI index.

    `messages` is mutated in-place with the new user/assistant/tool turns
    so the next call sees full context.
    """
    messages.append({"role": "user", "content": question})

    tool_calls_made = []
    answer     = ""
    error      = None
    cache_hits = 0
    rounds     = 0

    for round_num in range(MAX_TOOL_ROUNDS):
        rounds = round_num + 1

        if stream:
            # ── Streaming round ──────────────────────────────────────────
            acc_content    = ""
            acc_tool_calls: list[dict] = []
            streaming_live = False

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
                    if not streaming_live:
                        if on_stream_start:
                            on_stream_start()
                        else:
                            sys.stdout.write("\033[2K\rAssistant: ")
                            sys.stdout.flush()
                        streaming_live = True
                    acc_content += delta.content
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()

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
                answer = acc_content.strip()
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
                answer = (message.content or "").strip()
                break

            raw_tool_calls = message.tool_calls

        # Execute tool calls and append results
        for tc in raw_tool_calls:
            fn_name = tc.function.name
            fn_args: dict = {}
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                pass

            if on_status:
                on_status(_status_for_call(fn_name, fn_args))

            result, hit = execute_tool(fn_name, fn_args, index, sections_text, cache)
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

    # Fallback: recover the last assistant text if loop ended without one
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


# ── Cache pre-warm ─────────────────────────────────────────────────────────

def prewarm_cache(index: dict) -> dict:
    """Populate the per-session cache with one get_section per section."""
    cache: dict = {}
    for sec in index.get("sections", []):
        sid = sec.get("section_id", "")
        if sid:
            cache[("get_section", sid.lower(), "interest", 50)] = format_section(
                index, sid, sort="interest", limit=50)
    return cache


# ── Conversation runner ────────────────────────────────────────────────────────

def run_conversation(thread: dict, system_prompt: str,
                     index: dict, sections_text: str,
                     model: str) -> dict:
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
                          index, sections_text, model, cache)
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
            "latency":    elapsed,
            "cache_hits": result["cache_hits"],
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
                    recovery_msg: str) -> None:
    """Interactive chat session in the terminal."""
    cache    = prewarm_cache(index)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    turn = 0

    print()
    print("─" * 60)
    print(f"  {destination_name} Assistant — Interactive Mode")
    print(f"  Model: {model}")
    lang_label = {"en": "English", "es": "Español",
                  "fr": "Français", "de": "Deutsch"}.get(lang, lang.upper())
    print(f"  Language: {lang_label}")
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
                        help="Response language: en, es, fr, de (default: en)")
    parser.add_argument("--index",        default=None,
                        help=f"POI index JSON (default: {DEFAULT_INDEX})")
    parser.add_argument("--structure",    default=None,
                        help=argparse.SUPPRESS)  # legacy, hidden
    parser.add_argument("--conversation", default=None,
                        help="Run only this conversation ID (e.g. C01)")
    parser.add_argument("--output",       default=None,
                        help="Output path (default: results/conversations_<model>.json)")
    args = parser.parse_args()

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
    system_prompt = make_system_prompt(
        sections_text=sections_text,
        destination=destination_display,
        destination_overview=overview_text,
        lang=args.lang,
    )
    recovery_msg = _RECOVERY_MSGS.get(args.lang, _RECOVERY_MSGS["en"])

    # Interactive mode
    if args.interactive:
        run_interactive(system_prompt, index, sections_text,
                        args.model, args.lang,
                        destination_name=destination_display,
                        recovery_msg=recovery_msg)
        return

    # Scripted mode
    if not CONVERSATIONS_FILE.exists():
        print(f"[ERROR] Not found: {CONVERSATIONS_FILE}", file=sys.stderr)
        sys.exit(1)
    with open(CONVERSATIONS_FILE, encoding="utf-8") as f:
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
    print(f"[INFO] Conversations: {len(threads)}")
    print(f"[INFO] Output:        {output_file}")

    results = []
    total_start = time.time()
    for thread in threads:
        result = run_conversation(thread, system_prompt,
                                  index, sections_text, args.model)
        results.append(result)
    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n[INFO] All conversations complete in {total_elapsed}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

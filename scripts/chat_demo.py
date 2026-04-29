#!/usr/bin/env python3
"""
chat_demo.py — Multi-turn conversation demo for the Úbeda tourism assistant.

Runs scripted conversation threads from eval/conversations.json, carrying
the full message history across turns within each thread.  Each turn in a
conversation sees everything the model said in previous turns, so later
questions can reference earlier answers naturally ("you mentioned X — tell
me more about it").

The POI list cache is shared across all turns in a conversation, and the
section context is pre-loaded in the system prompt (identical setup to
run_eval.py).

Key difference from run_eval.py:
  - messages list is NOT reset between turns
  - tool calls within a turn are appended to the shared history
  - the next user question always follows the last assistant message

Usage:
    # Scripted conversations
    .venv/bin/python scripts/chat_demo.py
    .venv/bin/python scripts/chat_demo.py --model openai/gemma4:e4b
    .venv/bin/python scripts/chat_demo.py --conversation C01

    # Interactive mode (type questions, get answers, conversation context carries)
    .venv/bin/python scripts/chat_demo.py --interactive
    .venv/bin/python scripts/chat_demo.py --interactive --model openai/gemma4:26b
"""

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

# Import building blocks from run_eval without modifying it
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_eval import (
    _get_sections,
    build_sections_text,
    build_poi_list_text,
    make_system_prompt,
    execute_tool,
    TOOL_DEFS,
    MAX_TOOL_ROUNDS,
    load_inputs,
    _LANG_RULES,
    _RECOVERY_MSGS,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CONVERSATIONS_FILE = PROJECT_ROOT / "eval" / "conversations.json"
RESULTS_DIR        = PROJECT_ROOT / "results"
DEFAULT_MODEL      = "openai/gemma4:26b"


# ── Spinner (background thread) ─────────────────────────────────────────

class Spinner:
    """Lightweight terminal spinner that runs in a background thread."""
    _FRAMES = ["\u28cb","\u28d9","\u28b9","\u28b8","\u28bc","\u28b4",
               "\u28a6","\u28a7","\u2887","\u288f"]  # braille dots

    def __init__(self) -> None:
        self._msg    = "Thinking"
        self._active = False
        self._thread: threading.Thread | None = None
        self._lock   = threading.Lock()

    def update(self, msg: str) -> None:
        """Change the status text while the spinner is running."""
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
            # \033[2K = erase entire current line; \r = go to start of line
            sys.stdout.write(f"\033[2K\r  {frame}  {text}\u2026")
            sys.stdout.flush()
            time.sleep(0.08)

    def stop(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join()
        # Erase the spinner line completely and return cursor to start
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()


# ── Single-turn execution (appends to shared history) ─────────────────────

def run_turn(question: str, messages: list[dict],
             sections_text: str, poi_list_fn,
             md_lines: list[str], model: str,
             poi_cache: dict,
             on_status=None,
             on_stream_start=None,
             stream: bool = False,
             recovery_msg: str = "") -> dict:
    """
    Execute one conversation turn.

    Appends the user question to `messages`, runs the tool-calling loop,
    appends all intermediate assistant/tool messages, and returns the
    final answer dict.  `messages` is modified in-place so the next
    turn sees the full history.
    """
    messages.append({"role": "user", "content": question})

    tool_calls_made = []
    answer     = ""
    error      = None
    cache_hits = 0

    for round_num in range(MAX_TOOL_ROUNDS):
        if stream:
            # ── Streaming round ───────────────────────────────────────────
            # Accumulate content + tool-call deltas from the stream.
            # Text tokens are printed immediately; tool-call tokens are
            # collected silently (spinner stays active until first text).
            acc_content    = ""
            acc_tool_calls: list[dict] = []  # [{id, name, arguments}]
            streaming_live = False  # True once first text token printed

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

                # ─ Text content token
                if delta.content:
                    if not streaming_live:
                        # Signal caller to stop the spinner, then start printing
                        if on_stream_start:
                            on_stream_start()  # stops spinner + prints prefix
                        else:
                            sys.stdout.write("\033[2K\rAssistant: ")
                            sys.stdout.flush()
                        streaming_live = True
                    acc_content += delta.content
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()

                # ─ Tool-call delta (accumulate silently)
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
                print()   # newline after streamed answer

            # Build assistant message from accumulated deltas
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

            # Process accumulated tool calls
            raw_tool_calls = [
                type("TC", (), {"id": tc["id"],
                                "function": type("F", (), {
                                    "name":      tc["name"],
                                    "arguments": tc["arguments"],
                                })()})()
                for tc in acc_tool_calls
            ]

        else:
            # ── Non-streaming round (scripted / batch use) ──────────────
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

        # Execute tools and append responses to history
        for tc in raw_tool_calls:
            fn_name = tc.function.name
            fn_args = {}
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                pass

            # Update status so the spinner shows what's happening
            if on_status:
                if fn_name == "get_poi_list":
                    sec = fn_args.get("section_title", "section")
                    on_status(f"Searching {sec}")
                elif fn_name == "get_page_content":
                    lines = fn_args.get("lines", "...")
                    on_status(f"Reading guide lines {lines}")

            result, hit = execute_tool(
                fn_name, fn_args, sections_text, poi_list_fn, md_lines, poi_cache
            )
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

    # Safety net for empty answers
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
        "rounds":     round_num + 1,
        "cache_hits": cache_hits,
        "error":      error,
    }


# ── Conversation runner ────────────────────────────────────────────────────────

def run_conversation(thread: dict, system_prompt: str,
                     sections_text: str, poi_list_fn,
                     md_lines: list[str], model: str,
                     structure_data: dict) -> dict:
    """
    Run all turns of a conversation thread.

    A single `messages` list and `poi_cache` are shared across every turn,
    giving the model full conversational context and instant POI lookups.
    """
    # Pre-warm POI cache for this conversation
    poi_cache: dict[str, str] = {}
    for sec in _get_sections(structure_data):
        title = sec.get("title", "")
        if title:
            poi_cache[title.lower()] = poi_list_fn(title)

    # Shared message history — starts with system prompt only
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    turns_log  = []
    conv_start = time.time()

    print(f"\n{'─'*70}")
    print(f"  {thread['id']}  {thread['title']}")
    print(f"  {thread['description']}")
    print(f"{'─'*70}")

    for i, turn_spec in enumerate(thread["turns"], 1):
        question = turn_spec["question"]
        print(f"\n  Turn {i}/{len(thread['turns'])}: {question}")

        t0     = time.time()
        result = run_turn(
            question, messages,
            sections_text, poi_list_fn, md_lines, model, poi_cache,
        )
        elapsed = round(time.time() - t0, 2)

        status = "ERROR" if result["error"] else "OK"
        tools_used  = [c["tool"] for c in result["tool_calls"]]
        hits        = result["cache_hits"]
        total_calls = len(result["tool_calls"])
        print(f"  [{status}] {elapsed}s | tools: {tools_used} | cache: {hits}/{total_calls}")
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

    total_time = round(time.time() - conv_start, 1)
    total_cache_hits  = sum(t["cache_hits"] for t in turns_log)
    total_tool_calls  = sum(len(t["tool_calls"]) for t in turns_log)
    total_latency     = sum(t["latency"] for t in turns_log)

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

def run_interactive(system_prompt: str, sections_text: str, poi_list_fn,
                    md_lines: list[str], model: str,
                    structure_data: dict, lang: str = "en",
                    destination_name: str = "Tourism",
                    recovery_msg: str = "") -> None:
    """Start an interactive chat session in the terminal.

    Full conversation context carries across turns. Type 'exit', 'quit',
    or press Ctrl+C / Ctrl+D to end the session.
    """
    # Pre-warm POI cache once for the whole session
    poi_cache: dict[str, str] = {}
    for sec in _get_sections(structure_data):
        title = sec.get("title", "")
        if title:
            poi_cache[title.lower()] = poi_list_fn(title)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    turn = 0

    print()
    print("─" * 60)
    print(f"  {destination_name} Assistant — Interactive Mode")
    print(f"  Model: {model}")
    lang_label = {"en": "English", "es": "Español", "fr": "Français", "de": "Deutsch"}.get(lang, lang.upper())
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
            sections_text, poi_list_fn, md_lines, model, poi_cache,
            on_status=spinner.update,
            on_stream_start=on_stream_start,
            stream=True,
            recovery_msg=recovery_msg,
        )

        if not spinner_stopped:
            # Streaming didn't happen (empty answer / error): stop normally
            spinner.stop()
            print("Assistant:", result["answer"])
        # else: answer was already streamed to the terminal

        elapsed = round(time.time() - t0, 2)

        # Compact metadata line
        tools_used = [c["tool"].replace("get_", "") for c in result["tool_calls"]]
        hits = result["cache_hits"]
        total = len(result["tool_calls"])
        meta = f"[{elapsed}s"
        if tools_used:
            meta += f" | tools: {', '.join(tools_used)}"
        if total:
            meta += f" | cache {hits}/{total}"
        meta += f" | turn {turn}]"
        print(f"\033[2m{meta}\033[0m")   # dim text
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load conversations or start interactive mode."""
    parser = argparse.ArgumentParser(description="Multi-turn conversation demo")
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help=f"litellm model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--interactive",  action="store_true",
                        help="Start an interactive chat session")
    parser.add_argument("--lang",          default="en",
                        help="Response language: en, es, fr, de (default: en)")
    parser.add_argument("--conversation", default=None,
                        help="Run only this conversation ID (e.g. C01)")
    parser.add_argument("--output",       default=None,
                        help="Output path (default: results/conversations_<model>.json)")
    args = parser.parse_args()

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

    questions, structure_data, md_lines = load_inputs()

    # Derive destination name from root node title
    root_nodes = structure_data.get("structure", [])
    root_title = root_nodes[0].get("title", "") if root_nodes else ""
    destination_name = root_title.replace(" Tourism Guide", "").strip() or "Tourism"

    recovery_msg  = _RECOVERY_MSGS.get(args.lang, _RECOVERY_MSGS["en"])
    sections_text = build_sections_text(structure_data)
    system_prompt = make_system_prompt(sections_text, lang=args.lang,
                                       destination=destination_name)
    poi_list_fn   = lambda title: build_poi_list_text(title, structure_data)

    # ── Interactive mode ──────────────────────────────────────────────────────────────────
    if args.interactive:
        run_interactive(
            system_prompt, sections_text, poi_list_fn,
            md_lines, args.model, structure_data, lang=args.lang,
            destination_name=destination_name, recovery_msg=recovery_msg,
        )
        return

    # ── Scripted mode ───────────────────────────────────────────────────────
    model_tag   = args.model.split("/")[-1].replace(":", "-")
    output_file = Path(args.output) if args.output \
                  else RESULTS_DIR / f"conversations_{model_tag}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"[INFO] Model:         {args.model}")
    print(f"[INFO] Conversations: {len(threads)}")
    print(f"[INFO] Output:        {output_file}")

    results    = []
    total_start = time.time()

    for thread in threads:
        result = run_conversation(
            thread, system_prompt, sections_text,
            poi_list_fn, md_lines, args.model, structure_data,
        )
        results.append(result)

    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n[INFO] All conversations complete in {total_elapsed}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

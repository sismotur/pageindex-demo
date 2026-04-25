#!/usr/bin/env python3
"""
run_eval.py — PageIndex Úbeda Q&A evaluation runner.

Uses litellm tool calling (Ollama backend) to navigate the PageIndex tree
and answer each question in eval/questions.json.

Usage:
    .venv/bin/python scripts/run_eval.py
    .venv/bin/python scripts/run_eval.py --model openai/gemma4:e4b
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import litellm
litellm.drop_params = True
litellm.set_verbose = False

# ── Constants ──────────────────────────────────────────────────────────────────
QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.json"
STRUCTURE_FILE = PROJECT_ROOT / "results" / "ubeda_guide_structure.json"
MD_FILE        = PROJECT_ROOT / "data" / "ubeda_guide.md"
RESULTS_DIR    = PROJECT_ROOT / "results"
DEFAULT_MODEL  = "openai/gemma4:e2b"
MAX_TOOL_ROUNDS = 14

SYSTEM_PROMPT = """\
You are a tourism assistant for Úbeda, Spain. You answer questions using the \
Úbeda Tourism Guide document. You have three tools:

- get_sections(): returns the 18 section titles with their line ranges and summaries.
- get_poi_list(section_title): returns all POI names and their line numbers in a section.
- get_page_content(lines): returns the raw Markdown text for a line range (e.g. "9-28").

STRATEGY — choose the path that fits the question:

A) LISTING questions ("what X exist?", "list all Y")
   1. get_sections() → identify the relevant section.
   2. get_poi_list(section_title) → use the POI names to compose your answer.
   You do NOT need to call get_page_content for every POI in a list.

B) SPECIFIC FACT questions (address, phone, description, dates, measurements,
   or any question about a named place: "tell me about X", "what is X?",
   "describe X", "what are the opening hours of X")
   1. get_sections() → identify the section.
   2. get_poi_list(section_title) → find the exact POI name and its line number.
   3. get_page_content(lines="start-end") → read the POI text (10-25 lines).
   Never answer specific facts from a POI title alone.

RULES FOR ALL QUESTIONS:
- Answer based ONLY on retrieved text. Do not use outside knowledge.
- Include exact names, addresses, phones, and dates when present in the text.
- If information is not in the guide, say so clearly.
- Always respond in English, regardless of the language of any retrieved content.
"""

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_sections",
            "description": (
                "Returns the 18 top-level sections of the Úbeda Tourism Guide, "
                "each with its title, line range, POI count, and a short summary. "
                "Call this first to identify which section(s) are relevant."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_poi_list",
            "description": (
                "Returns the names and line numbers of every POI inside one section. "
                "Use the section_title exactly as returned by get_sections(). "
                "Use the line numbers to target get_page_content() at specific POIs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_title": {
                        "type": "string",
                        "description": "Exact section title from get_sections().",
                    }
                },
                "required": ["section_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": (
                "Returns the raw Markdown text for a line range. "
                "Use line_num values from get_poi_list() as anchors. "
                "Keep ranges tight — a single POI is typically 10-25 lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "string",
                        "description": "Line range like '100-125' or single line '50'.",
                    }
                },
                "required": ["lines"],
            },
        },
    },
]


# ── Two-level navigation helpers ────────────────────────────────────────────

def _get_sections(structure_data: dict) -> list[dict]:
    """Return the list of top-level section nodes (## level)."""
    root_nodes = structure_data.get("structure", [])
    if not root_nodes:
        return []
    # Direct children of the document root are the ## sections
    return root_nodes[0].get("nodes", [])


def build_sections_text(structure_data: dict) -> str:
    """Return a compact sections-only overview for get_sections() tool."""
    doc_name   = structure_data.get("doc_name", "Document")
    line_count = structure_data.get("line_count", 0)
    sections   = _get_sections(structure_data)

    lines = [f"Document: {doc_name}  ({line_count} lines)",
             f"",
             f"SECTIONS ({len(sections)} total):"]

    for i, sec in enumerate(sections):
        title   = sec.get("title", "?")
        lnum    = sec.get("line_num", "?")
        summary = sec.get("summary", "")
        pois    = sec.get("nodes") or []
        n_pois  = len(pois)

        # Compute end line: start of next section - 1, or end of doc
        if i + 1 < len(sections):
            end_line = (sections[i + 1].get("line_num") or line_count) - 1
        else:
            end_line = line_count

        lines.append(f"  [{sec.get('node_id','?')}] {title}")
        lines.append(f"      lines {lnum}–{end_line}  ({n_pois} POIs)")
        if summary:
            lines.append(f"      Summary: {summary}")

    return "\n".join(lines)


def build_poi_list_text(section_title: str, structure_data: dict) -> str:
    """Return POI names + line numbers for a given section title."""
    sections = _get_sections(structure_data)
    # Case-insensitive match
    match = next(
        (s for s in sections
         if s.get("title", "").lower() == section_title.lower()),
        None,
    )
    if not match:
        # Try partial match
        match = next(
            (s for s in sections
             if section_title.lower() in s.get("title", "").lower()),
            None,
        )
    if not match:
        titles = [s.get("title") for s in sections]
        return f"[ERROR] Section '{section_title}' not found. Available: {titles}"

    pois = match.get("nodes") or []
    if not pois:
        return f"[INFO] Section '{match['title']}' has no POI children."

    lines = [f"POIs in '{match['title']}' ({len(pois)} entries):"]
    for poi in pois:
        lines.append(f"  [{poi.get('node_id','?')}] {poi.get('title','?')}  (line {poi.get('line_num','?')})")
    return "\n".join(lines)


# ── Content retrieval ──────────────────────────────────────────────────────────

def parse_line_spec(spec: str) -> tuple[int, int]:
    """Parse '100-150' or '50' into (start, end) 1-indexed line numbers."""
    spec = spec.strip()
    m = re.match(r"^(\d+)-(\d+)$", spec)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)$", spec)
    if m:
        n = int(m.group(1))
        return n, n
    raise ValueError(f"Cannot parse line spec: {spec!r}")


def get_lines(md_lines: list[str], spec: str) -> str:
    """Return Markdown text for the given line spec (1-indexed, inclusive)."""
    try:
        start, end = parse_line_spec(spec)
    except ValueError as exc:
        return f"[ERROR] {exc}"
    start = max(1, start)
    end   = min(len(md_lines), end)
    if start > end:
        return f"[ERROR] Invalid range {spec}: start > end"
    return "\n".join(md_lines[start - 1 : end])


# ── Section mapping ────────────────────────────────────────────────────────────

def build_section_map(nodes: list, parent_section: str = "") -> dict[int, str]:
    """Map every line_num in the tree to its nearest ## section title."""
    mapping = {}
    for node in nodes:
        lnum = node.get("line_num")
        # Detect ## sections (depth 1 children of root)
        title = node.get("title", "")
        section = title if not parent_section else parent_section
        if lnum:
            mapping[lnum] = section
        if node.get("nodes"):
            mapping.update(build_section_map(node["nodes"], section))
    return mapping


def sections_from_tool_calls(tool_calls_made: list, section_map: dict) -> list[str]:
    """Derive accessed section names from get_poi_list and get_page_content calls."""
    sections = []
    for call in tool_calls_made:
        # Direct hit: get_poi_list names the section explicitly
        if call["tool"] == "get_poi_list":
            sec = call.get("args", {}).get("section_title", "")
            if sec and sec not in sections:
                sections.append(sec)
            continue
        # Indirect: get_page_content — map lines back to section
        if call["tool"] == "get_page_content":
            spec = call.get("args", {}).get("lines", "")
            try:
                start, end = parse_line_spec(spec)
            except ValueError:
                continue
            for lnum, sec in section_map.items():
                if start <= lnum <= end and sec not in sections:
                    sections.append(sec)
    return sections


# ── Agentic loop ───────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict,
                 sections_text: str, poi_list_fn, md_lines: list[str]) -> str:
    """Dispatch a tool call and return its result string."""
    if name == "get_sections":
        return sections_text
    if name == "get_poi_list":
        section_title = args.get("section_title", "")
        return poi_list_fn(section_title)
    if name == "get_page_content":
        spec = args.get("lines", "1-20")
        content = get_lines(md_lines, spec)
        if not content.strip():
            return "[WARNING] No content found for that line range."
        return content
    return f"[ERROR] Unknown tool: {name}"


def run_agentic_loop(question: str, sections_text: str,
                     poi_list_fn, md_lines: list[str], model: str) -> dict:
    """
    Run the tool-calling loop for a single question.
    Returns a dict with answer, tool_calls, rounds, and any error.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    tool_calls_made = []
    answer          = ""
    error           = None

    for round_num in range(MAX_TOOL_ROUNDS):
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

        # Build assistant message dict (some fields may be None)
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

        # No more tool calls → final answer
        if not message.tool_calls:
            answer = (message.content or "").strip()
            break

        # Execute each tool call
        for tc in message.tool_calls:
            fn_name  = tc.function.name
            fn_args  = {}
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                pass

            result = execute_tool(fn_name, fn_args, sections_text, poi_list_fn, md_lines)
            tool_calls_made.append({
                "tool":           fn_name,
                "args":           fn_args,
                "result_preview": result[:300],
            })

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    # If the loop ended without a clean answer, take the last assistant message
    if not answer:
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                answer = msg["content"].strip()
                break

    return {
        "answer":         answer,
        "tool_calls":     tool_calls_made,
        "rounds":         round_num + 1,
        "error":          error,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def load_inputs() -> tuple[list, dict, list[str]]:
    """Load questions, structure, and Markdown lines. Fail fast if missing."""
    for path in (QUESTIONS_FILE, STRUCTURE_FILE, MD_FILE):
        if not path.exists():
            print(f"[ERROR] Missing: {path}", file=sys.stderr)
            sys.exit(1)

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    with open(STRUCTURE_FILE, encoding="utf-8") as f:
        structure_data = json.load(f)
    with open(MD_FILE, encoding="utf-8") as f:
        md_lines = f.read().splitlines()

    return questions, structure_data, md_lines


def main() -> None:
    """Parse args, run all questions, save results."""
    parser = argparse.ArgumentParser(description="Run PageIndex Q&A evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"litellm model string (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    questions, structure_data, md_lines = load_inputs()
    sections_text = build_sections_text(structure_data)
    poi_list_fn   = lambda title: build_poi_list_text(title, structure_data)

    # Build line→section map from the root's direct children (## sections)
    root_nodes  = structure_data.get("structure", [])
    section_map = {}
    if root_nodes and root_nodes[0].get("nodes"):
        section_map = build_section_map(root_nodes[0]["nodes"])

    model_tag = args.model.split("/")[-1].replace(":", "-")
    output_file = RESULTS_DIR / f"eval_{model_tag}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"[INFO] Model:     {args.model}")
    print(f"[INFO] Questions: {len(questions)}")
    print(f"[INFO] Output:    {output_file}")
    print(f"[INFO] Sections text: {len(sections_text):,} chars (was ~20,478)\n")

    results = []
    total_start = time.time()

    for i, q in enumerate(questions, 1):
        qid        = q["id"]
        question   = q["question"]
        difficulty = q.get("difficulty", "?")
        print(f"[{i:2d}/{len(questions)}] {qid} ({difficulty})  {question[:70]}...")
        t0 = time.time()

        loop_result = run_agentic_loop(question, sections_text, poi_list_fn, md_lines, args.model)

        elapsed = round(time.time() - t0, 2)
        sections = sections_from_tool_calls(loop_result["tool_calls"], section_map)

        result = {
            "id":               qid,
            "model":            args.model,
            "category":         q.get("category"),
            "difficulty":       difficulty,
            "question":         question,
            "expected_section": q.get("expected_section"),
            "grounding_check":  q.get("grounding_check"),
            "answer":           loop_result["answer"],
            "tool_calls":       loop_result["tool_calls"],
            "sections_accessed":sections,
            "rounds":           loop_result["rounds"],
            "latency_seconds":  elapsed,
            "error":            loop_result["error"],
        }
        results.append(result)

        answer_preview = loop_result["answer"][:120].replace("\n", " ")
        tools_used = [c["tool"] for c in loop_result["tool_calls"]]
        status = "ERROR" if loop_result["error"] else "OK"
        print(f"         [{status}] {elapsed}s | tools: {tools_used}")
        print(f"         → {answer_preview}...\n")

    elapsed_total = round(time.time() - total_start, 1)
    print(f"[INFO] Finished {len(results)} questions in {elapsed_total}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

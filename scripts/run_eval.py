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
MAX_TOOL_ROUNDS = 7

SYSTEM_PROMPT = """\
You are a tourism assistant for Úbeda, Spain. You answer questions using the \
Úbeda Tourism Guide document. You have two tools:

- get_document_structure(): returns the full tree index (all section and POI titles with \
  their line numbers).
- get_page_content(lines): returns the Markdown text for a line range (e.g. "10-50").

STRATEGY:
1. Always call get_document_structure() first to orient yourself.
2. Identify which section(s) contain the relevant information, using line numbers.
3. Call get_page_content(lines="start-end") with a tight range covering just the relevant POIs.
4. Answer based ONLY on the retrieved text. Do not use outside knowledge.
5. Include specific names, addresses, phone numbers, and dates when they appear in the data.
   If information is not in the guide, say so clearly.
"""

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_document_structure",
            "description": (
                "Returns the full hierarchical tree of the Úbeda Tourism Guide, "
                "showing every section (##) and POI (###) with its title and line_num. "
                "Use the line_num values to request content with get_page_content."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": (
                "Returns the raw Markdown text for a range of lines. "
                "Use line_num values from get_document_structure as anchors. "
                "Keep ranges tight — e.g. '182-230' for one section, '9-28' for one POI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "string",
                        "description": "Line range like '100-150' or single line '50'.",
                    }
                },
                "required": ["lines"],
            },
        },
    },
]


# ── Structure rendering ────────────────────────────────────────────────────────

def render_tree(nodes: list, indent: int = 0) -> list[str]:
    """Recursively render the tree as indented text lines."""
    lines = []
    prefix = "  " * indent
    for node in nodes:
        nid   = node.get("node_id", "?")
        title = node.get("title", "?")
        lnum  = node.get("line_num", "?")
        lines.append(f"{prefix}[{nid}] {title}  (line {lnum})")
        if node.get("nodes"):
            lines.extend(render_tree(node["nodes"], indent + 1))
    return lines


def build_structure_text(structure_data: dict) -> str:
    """Return the full tree as a compact string for tool responses."""
    doc_name   = structure_data.get("doc_name", "Document")
    line_count = structure_data.get("line_count", 0)
    tree       = structure_data.get("structure", [])
    header = f"Document: {doc_name}  ({line_count} lines)\n\nSTRUCTURE:\n"
    body = "\n".join(render_tree(tree))
    return header + body


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
    """Derive accessed section names from the lines the model requested."""
    sections = []
    for call in tool_calls_made:
        if call["tool"] != "get_page_content":
            continue
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
                 structure_text: str, md_lines: list[str]) -> str:
    """Dispatch a tool call and return its result string."""
    if name == "get_document_structure":
        return structure_text
    if name == "get_page_content":
        spec = args.get("lines", "1-20")
        content = get_lines(md_lines, spec)
        if not content.strip():
            return "[WARNING] No content found for that line range."
        return content
    return f"[ERROR] Unknown tool: {name}"


def run_agentic_loop(question: str, structure_text: str,
                     md_lines: list[str], model: str) -> dict:
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

            result = execute_tool(fn_name, fn_args, structure_text, md_lines)
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
    structure_text = build_structure_text(structure_data)

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
    print(f"[INFO] Structure: {len(structure_text):,} chars\n")

    results = []
    total_start = time.time()

    for i, q in enumerate(questions, 1):
        qid        = q["id"]
        question   = q["question"]
        difficulty = q.get("difficulty", "?")
        print(f"[{i:2d}/{len(questions)}] {qid} ({difficulty})  {question[:70]}...")
        t0 = time.time()

        loop_result = run_agentic_loop(question, structure_text, md_lines, args.model)

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

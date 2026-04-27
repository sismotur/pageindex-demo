#!/usr/bin/env python3
"""
add_section_summaries.py — Generate LLM summaries for the 18 section nodes.

Reads results/ubeda_guide_structure.json, generates a 2-3 sentence
summary for each top-level section (##) using the POI titles as input,
then writes the summaries back in-place.

Only section nodes are summarised — the 408 POI leaf nodes are left as-is.
This keeps the one-time indexing cost to ~18 LLM calls (~90s on E4B).

Usage:
    .venv/bin/python scripts/add_section_summaries.py
    .venv/bin/python scripts/add_section_summaries.py --model openai/gemma4:e2b
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT   = Path(__file__).parent.parent
STRUCTURE_FILE = PROJECT_ROOT / "results" / "ubeda_guide_structure.json"
MD_FILE        = PROJECT_ROOT / "data" / "ubeda_guide.md"
DEFAULT_MODEL  = "openai/gemma4:26b"

load_dotenv(PROJECT_ROOT / ".env")
import litellm
litellm.drop_params = True
litellm.set_verbose = False


def get_sections(structure_data: dict) -> list[dict]:
    """Return the list of ## section nodes."""
    root = structure_data.get("structure", [])
    if not root:
        return []
    return root[0].get("nodes", [])


def get_poi_text_sample(poi_node: dict, next_line: int,
                        md_lines: list[str], max_chars: int = 250) -> str:
    """Extract actual Markdown text for one POI (heading + metadata + description)."""
    start = (poi_node.get("line_num") or 1) - 1   # 0-indexed
    end   = min(next_line - 1, start + 30)         # at most 30 lines
    text  = "\n".join(md_lines[start:end]).strip()
    return text[:max_chars] + ("\u2026" if len(text) > max_chars else "")


def get_section_content_sample(sec_node: dict, md_lines: list[str],
                               max_pois: int = 5, max_chars: int = 250) -> str:
    """Build a multi-POI content sample for the summary prompt."""
    pois = sec_node.get("nodes") or []
    parts = []
    for i, poi in enumerate(pois[:max_pois]):
        next_start = pois[i + 1].get("line_num", len(md_lines) + 1) \
                     if i + 1 < len(pois) else len(md_lines) + 1
        parts.append(get_poi_text_sample(poi, next_start, md_lines, max_chars))
    return "\n\n---\n\n".join(parts)


def generate_summary(section_title: str, poi_names: list[str], model: str,
                     sec_node: dict | None = None,
                     md_lines: list[str] | None = None) -> str:
    """Ask the model for a 2-sentence tourism summary of the section.

    When sec_node and md_lines are provided, the prompt includes actual
    Markdown content (descriptions, addresses) from the first N POIs,
    producing significantly richer summaries than titles alone.
    """
    if sec_node is not None and md_lines is not None:
        sample = get_section_content_sample(sec_node, md_lines)
        prompt = (
            f"You are summarising a tourism section for a travel guide about Úbeda, Spain.\n\n"
            f"Section: \"{section_title}\" ({len(poi_names)} points of interest)\n\n"
            f"Sample content from the first {min(5, len(poi_names))} POIs:\n\n"
            f"{sample}\n\n"
            f"Write exactly 2 sentences summarising what a visitor will find in this section. "
            f"Be specific — mention the most notable highlights by name. "
            f"Do not use bullet points. Reply in English only."
        )
    else:
        # Fallback: title-only prompt (used when Markdown is unavailable)
        poi_str = ", ".join(poi_names[:30])
        if len(poi_names) > 30:
            poi_str += f", … ({len(poi_names) - 30} more)"
        prompt = (
            f"You are summarising a tourism section for a travel guide about Úbeda, Spain.\n\n"
            f"Section: \"{section_title}\"\n"
            f"Contains {len(poi_names)} points of interest, including: {poi_str}\n\n"
            f"Write exactly 2 sentences summarising what a visitor will find in this section. "
            f"Be specific — name the most notable highlights. "
            f"Do not use bullet points. Reply in English only."
        )

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    """Load structure and Markdown, generate section summaries, save back."""
    parser = argparse.ArgumentParser(description="Add section summaries to structure JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model for summarisation (default: {DEFAULT_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if summary already present")
    args = parser.parse_args()

    if not STRUCTURE_FILE.exists():
        print(f"[ERROR] Not found: {STRUCTURE_FILE}", file=sys.stderr)
        print("[ERROR] Run PageIndex indexing first.", file=sys.stderr)
        sys.exit(1)

    with open(STRUCTURE_FILE, encoding="utf-8") as f:
        structure_data = json.load(f)

    sections = get_sections(structure_data)
    if not sections:
        print("[ERROR] No section nodes found in structure.", file=sys.stderr)
        sys.exit(1)

    # Load Markdown for richer content-based prompts
    md_lines: list[str] | None = None
    if MD_FILE.exists():
        with open(MD_FILE, encoding="utf-8") as f:
            md_lines = f.read().splitlines()
        print(f"[INFO] Markdown loaded: {len(md_lines)} lines (content-based prompts)")
    else:
        print(f"[WARN] Markdown not found at {MD_FILE} — falling back to title-only prompts")

    print(f"[INFO] Model:    {args.model}")
    print(f"[INFO] Sections: {len(sections)}")
    print()

    total_start = time.time()
    updated = 0

    for i, sec in enumerate(sections, 1):
        title    = sec.get("title", "?")
        existing = sec.get("summary", "")

        if existing and not args.force:
            print(f"[{i:2d}/{len(sections)}] {title[:60]}  — skipped (already has summary)")
            continue

        poi_names = [p.get("title", "") for p in (sec.get("nodes") or [])]
        t0 = time.time()

        try:
            summary = generate_summary(title, poi_names, args.model,
                                       sec_node=sec, md_lines=md_lines)
        except Exception as exc:
            print(f"[{i:2d}/{len(sections)}] {title[:60]}  — ERROR: {exc}", file=sys.stderr)
            continue

        sec["summary"] = summary
        updated += 1
        elapsed = round(time.time() - t0, 1)
        print(f"[{i:2d}/{len(sections)}] {title[:60]}  ({elapsed}s)")
        print(f"           {summary[:120]}...")

    total = round(time.time() - total_start, 1)
    print(f"\n[INFO] Generated {updated} summaries in {total}s")

    if updated > 0:
        with open(STRUCTURE_FILE, "w", encoding="utf-8") as f:
            json.dump(structure_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved → {STRUCTURE_FILE}")
    else:
        print("[INFO] No changes written.")


if __name__ == "__main__":
    main()

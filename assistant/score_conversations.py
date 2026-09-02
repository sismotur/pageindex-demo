#!/usr/bin/env python3
"""
score_conversations.py — Score scripted multi-turn conversation runs.

A conversation *run* (chat_demo.py scripted mode output) only records what
happened: question, answer, tool_calls, per turn. The optional rubric
(`must_mention`, `forbidden`, `expected_section` per turn; `outcome` per
thread) lives in the *source* conversations.json fixture instead, resolved
via the run's manifest.conversations_file — the same "authoring lives in
the input fixture, not the output log" pattern score_results.py uses for
FACT_CHECKS/expected_section.

Threads or turns that define none of these stay manual QA: they are still
included in the output for completeness, but do not count toward the
composite average.

Two checks run on EVERY turn regardless of authoring, because they are
runtime health signals rather than content expectations:
  - is_reask   — the answer is only another bare clarifying question.
  - is_repeat  — the answer duplicates the previous turn's answer verbatim
                 (normalized) — the greeting/confirmation "double-parrot"
                 failure mode the repeat-answer guard exists to prevent.
A thread with authored checks whose composite would otherwise be positive
is hard-floored to 0.0 if either fires on any turn, mirroring how a single
error zeroes a question's composite in score_results.py.

Usage:
    .venv/bin/python assistant/score_conversations.py
    .venv/bin/python assistant/score_conversations.py --file results/ubeda/en/runs/<run_id>/conversations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import run_store          # noqa: E402
import score_results      # noqa: E402
from run_eval import is_pure_reask, sections_accessed_from_calls  # noqa: E402
from index_tools import load_index                                # noqa: E402
from common.textnorm import normalize_text                        # noqa: E402


# ── Per-turn / per-thread scoring ────────────────────────────────────────────

def is_repeat_of_previous_turn(turns: list[dict], i: int) -> bool:
    """True when turns[i]'s answer duplicates turns[i-1]'s answer verbatim.

    Post-hoc equivalent of run_eval.is_repeat_of_previous_answer: a saved
    conversation run has one answer per visitor turn already (no raw
    message history needed), so comparing normalized adjacent answers
    reproduces the same "previous VISITOR turn" referent.
    """
    if i == 0:
        return False
    candidate = normalize_text(turns[i].get("answer") or "")
    previous  = normalize_text(turns[i - 1].get("answer") or "")
    return bool(candidate) and candidate == previous


def score_turn(run_turn: dict, source_turn: dict, run_turns: list[dict],
              index: dict | None, lang: str, turn_index: int) -> dict:
    """Score one turn against its (optional) authored expectations."""
    answer       = run_turn.get("answer", "") or ""
    answer_lower = answer.lower()
    must_mention = source_turn.get("must_mention") or []
    forbidden    = source_turn.get("forbidden") or []
    expected_section = source_turn.get("expected_section")
    has_checks   = bool(must_mention or forbidden or expected_section)

    missing = [t for t in must_mention if not score_results._matches(t, answer_lower)]
    mention_score = round((len(must_mention) - len(missing)) / len(must_mention), 2) \
        if must_mention else None

    forbidden_hits = [t for t in forbidden if score_results._matches(t, answer_lower)]

    retrieval = None
    if expected_section and index is not None:
        accessed = sections_accessed_from_calls(run_turn.get("tool_calls") or [], index)
        retrieval = 1.0 if score_results._sections_match(accessed, expected_section) else 0.0

    return {
        "turn":               turn_index + 1,
        "question":           run_turn.get("question"),
        "has_checks":         has_checks,
        "must_mention_score": mention_score,
        "missing_mentions":   missing,
        "forbidden_hits":     forbidden_hits,
        "expected_section":   expected_section,
        "retrieval":          retrieval,
        "language_ok":        score_results.score_language({"answer": answer, "lang": lang}),
        "is_reask":           is_pure_reask(answer),
        "is_repeat":          is_repeat_of_previous_turn(run_turns, turn_index),
        "error":              bool(run_turn.get("error")),
    }


def score_conversation(run_thread: dict, source_thread: dict | None,
                       index: dict | None, lang: str) -> dict:
    """Score one thread's run output against its (optional) source fixture."""
    run_turns    = run_thread.get("turns") or []
    source_turns = (source_thread or {}).get("turns") or [{}] * len(run_turns)

    turn_scores = [
        score_turn(rt, source_turns[i] if i < len(source_turns) else {},
                  run_turns, index, lang, i)
        for i, rt in enumerate(run_turns)
    ]

    outcome_spec = (source_thread or {}).get("outcome") or {}
    outcome_must_mention = outcome_spec.get("must_mention") or []
    outcome_score  = None
    outcome_missing: list[str] = []
    if outcome_must_mention and run_turns:
        final_answer = (run_turns[-1].get("answer") or "").lower()
        outcome_missing = [t for t in outcome_must_mention
                          if not score_results._matches(t, final_answer)]
        outcome_score = round(
            (len(outcome_must_mention) - len(outcome_missing)) / len(outcome_must_mention), 2)

    reask_count    = sum(1 for s in turn_scores if s["is_reask"])
    repeat_count   = sum(1 for s in turn_scores if s["is_repeat"])
    error_count    = sum(1 for s in turn_scores if s["error"])
    lang_fail_count = sum(1 for s in turn_scores if s["language_ok"] == 0.0)
    hard_fail = reask_count > 0 or repeat_count > 0

    check_scores = [s["must_mention_score"] for s in turn_scores if s["must_mention_score"] is not None]
    check_scores += [s["retrieval"] for s in turn_scores if s["retrieval"] is not None]
    if outcome_score is not None:
        check_scores.append(outcome_score)

    composite = round(sum(check_scores) / len(check_scores), 3) if check_scores else None
    if composite is not None and hard_fail:
        composite = 0.0

    return {
        "id":                 run_thread.get("id"),
        "title":              run_thread.get("title"),
        "turns":              turn_scores,
        "outcome_score":      outcome_score,
        "outcome_missing":    outcome_missing,
        "n_turns":            len(run_turns),
        "n_checked_turns":    sum(1 for s in turn_scores if s["has_checks"]),
        "reask_count":        reask_count,
        "repeat_count":       repeat_count,
        "error_count":        error_count,
        "language_fail_count": lang_fail_count,
        "composite":          composite,
    }


# ── Aggregation, discovery, history ─────────────────────────────────────────

def _aggregate_row(manifest: dict, thread_scores: list[dict]) -> dict:
    scored = [t["composite"] for t in thread_scores if t["composite"] is not None]
    n = len(thread_scores)
    return {
        "run_id":          manifest.get("run_id"),
        "kind":            "conversations",
        "scored_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model":           manifest.get("model"),
        "model_tag":       manifest.get("model_tag"),
        "lang":            manifest.get("lang"),
        "destination":     manifest.get("destination"),
        "git_commit":      (manifest.get("git") or {}).get("commit"),
        "index_sha256_12": (manifest.get("index") or {}).get("sha256_12"),
        "n_threads":       n,
        "n_checked_threads": sum(1 for t in thread_scores if t["n_checked_turns"] or t["outcome_score"] is not None),
        "composite":       round(sum(scored) / len(scored), 3) if scored else None,
        "reask_count":     sum(t["reask_count"] for t in thread_scores),
        "repeat_count":    sum(t["repeat_count"] for t in thread_scores),
        "error_count":     sum(t["error_count"] for t in thread_scores),
    }


def _discover_result_files() -> list[Path]:
    """Latest conversations run per (dest, lang) that has one."""
    files: list[Path] = []
    for dest, lang, _lang_dir in run_store.iter_lang_dirs():
        run_dir = run_store.find_latest_run(dest, lang, require="conversations.json")
        if run_dir:
            files.append(run_dir / "conversations.json")
    return files


def _print_summary(thread_scores: list[dict]) -> None:
    print(f"\n{'ID':<6} {'Composite':>9}  {'Checked':>7}  {'Reask':>5}  {'Repeat':>6}  {'Errors':>6}")
    print("-" * 55)
    for t in thread_scores:
        comp = f"{t['composite']:.3f}" if t["composite"] is not None else "  n/a"
        print(f"{t['id']:<6} {comp:>9}  {t['n_checked_turns']:>7}  "
              f"{t['reask_count']:>5}  {t['repeat_count']:>6}  {t['error_count']:>6}")
    scored = [t["composite"] for t in thread_scores if t["composite"] is not None]
    if scored:
        print(f"\nAggregate composite ({len(scored)}/{len(thread_scores)} scored threads): "
              f"{sum(scored) / len(scored):.3f}")
    total_reask  = sum(t["reask_count"] for t in thread_scores)
    total_repeat = sum(t["repeat_count"] for t in thread_scores)
    if total_reask or total_repeat:
        print(f"Guard signals: reask={total_reask}  repeat={total_repeat}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score scripted conversation runs")
    parser.add_argument("--file", type=Path, default=None,
                        help="Path to a conversations.json run (default: latest per destination/lang)")
    args = parser.parse_args()

    if args.file:
        candidate = args.file if args.file.is_absolute() else PROJECT_ROOT / args.file
        result_files = [candidate]
    else:
        result_files = _discover_result_files()
    if not result_files:
        print("[ERROR] No conversation run files found in results/", file=sys.stderr)
        sys.exit(1)

    for result_file in result_files:
        if not result_file.exists():
            print(f"[ERROR] Not found: {result_file}", file=sys.stderr)
            continue

        run_dir  = result_file.parent
        manifest = run_store.read_manifest(run_dir)
        with open(result_file, encoding="utf-8") as f:
            run_threads = json.load(f)

        source_by_id: dict[str, dict] = {}
        index = None
        lang = manifest.get("lang", "en")
        conv_file = manifest.get("conversations_file")
        if conv_file:
            source_path = PROJECT_ROOT / conv_file
            if source_path.exists():
                with open(source_path, encoding="utf-8") as f:
                    source_by_id = {t["id"]: t for t in json.load(f)}
        index_rel = (manifest.get("index") or {}).get("path")
        if index_rel:
            index_path = PROJECT_ROOT / index_rel
            if index_path.exists():
                index = load_index(index_path)

        thread_scores = [
            score_conversation(rt, source_by_id.get(rt.get("id")), index, lang)
            for rt in run_threads
        ]

        print(f"\n{'='*55}\nRun: {run_dir.relative_to(PROJECT_ROOT)}")
        _print_summary(thread_scores)

        if manifest:
            scored_path = run_dir / "conversations_scored.json"
        else:
            scored_path = result_file.parent / result_file.name.replace(
                "conversations", "conversations_scored")
        with open(scored_path, "w", encoding="utf-8") as f:
            json.dump(thread_scores, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Scored conversations saved → {scored_path}")

        if manifest and manifest.get("destination") and manifest.get("lang"):
            row = _aggregate_row(manifest, thread_scores)
            appended = run_store.append_history(
                manifest["destination"], manifest["lang"], row)
            if appended:
                hpath = run_store.history_path(manifest["destination"], manifest["lang"])
                print(f"[INFO] History appended → {hpath}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compare_runs.py — Regression comparison across historized eval runs.

Compares a candidate run (default: latest) against a reference run (default:
the pinned baseline.json) for one (destination, lang) pair, using the same
rubric as score_results.py. Prints per-question and aggregate deltas and
exits non-zero when a regression threshold is crossed — suitable for a
pre-push check.

Also prints a destination x language matrix of the latest composite scores
across every historized run, to compare how well the system performs across
destinations/languages (--matrix).

Usage:
    .venv/bin/python assistant/compare_runs.py
    .venv/bin/python assistant/compare_runs.py --dest ubeda --lang es
    .venv/bin/python assistant/compare_runs.py --candidate results/ubeda/en/runs/<run_id> \\
        --against results/ubeda/en/runs/<other_run_id>
    .venv/bin/python assistant/compare_runs.py --set-baseline
    .venv/bin/python assistant/compare_runs.py --matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import run_store  # noqa: E402
import score_results  # noqa: E402

# Regression thresholds.
COMPOSITE_DROP_MAX  = 0.02   # aggregate composite average
GROUNDING_DROP_MAX  = 0.025  # aggregate grounding average (2.5 pp)
QUESTION_DROP_MAX   = 0.25   # any single question's composite


def _load_scores(run_dir: Path) -> tuple[list[dict], dict]:
    """Return (per-question scores, manifest) for a run dir.

    Uses the run's scored.json when present; otherwise scores eval.json
    on the fly with the same expected-sections resolution score_results.py
    uses, so an unscored run can still be compared.
    """
    manifest = run_store.read_manifest(run_dir)
    scored_path = run_dir / "scored.json"
    if scored_path.exists():
        with open(scored_path, encoding="utf-8") as f:
            return json.load(f), manifest

    eval_path = run_dir / "eval.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"No scored.json or eval.json in {run_dir}")
    with open(eval_path, encoding="utf-8") as f:
        results = json.load(f)
    expected_sections = score_results._expected_sections_for_manifest(manifest) \
        if manifest else score_results.EXPECTED_SECTIONS
    scores = [score_results.score_result(r, expected_sections) for r in results]
    return scores, manifest


def _resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir():
        print(f"[ERROR] Not a run directory: {path}", file=sys.stderr)
        sys.exit(2)
    return path


def _display_path(path: Path) -> str:
    """Path relative to PROJECT_ROOT for display, or the absolute path
    when unrelated (e.g. a tmp_path run dir used in tests)."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _aggregate(scores: list[dict]) -> dict[str, float]:
    n = len(scores)
    if not n:
        return {k: 0.0 for k in
                ("grounding", "retrieval", "content_fetched", "composite", "latency")}
    avg = lambda key: round(sum(s[key] for s in scores) / n, 3)
    return {
        "grounding":       avg("grounding"),
        "retrieval":       avg("retrieval"),
        "content_fetched": avg("content_fetched"),
        "composite":       avg("composite"),
        "latency":         avg("latency"),
    }


def _print_aggregate_deltas(base_agg: dict, cand_agg: dict) -> None:
    print(f"\n{'Metric':<16}  {'Baseline':>9}  {'Candidate':>9}  {'Delta':>8}")
    print("-" * 48)
    for key in ("composite", "grounding", "retrieval", "content_fetched", "latency"):
        delta = round(cand_agg[key] - base_agg[key], 3)
        sign = "+" if delta >= 0 else ""
        print(f"{key:<16}  {base_agg[key]:>9.3f}  {cand_agg[key]:>9.3f}  {sign}{delta:>7.3f}")


def _print_question_deltas(base_scores: list[dict], cand_scores: list[dict]) -> list[str]:
    """Print per-question deltas; return the list of regression reasons found."""
    base_by_id = {s["id"]: s for s in base_scores}
    cand_by_id = {s["id"]: s for s in cand_scores}
    reasons: list[str] = []

    print(f"\n{'ID':>4}  {'Base':>6}  {'Cand':>6}  {'Delta':>7}  Notes")
    print("-" * 70)
    for qid in sorted(cand_by_id, key=lambda x: (len(x), x)):
        cand = cand_by_id[qid]
        base = base_by_id.get(qid)
        if base is None:
            print(f"{qid:>4}  {'—':>6}  {cand['composite']:>6.3f}  {'new':>7}  (question not in baseline)")
            continue

        delta = round(cand["composite"] - base["composite"], 3)
        notes = []

        new_missing = sorted(set(cand.get("missing_facts") or [])
                             - set(base.get("missing_facts") or []))
        if new_missing:
            notes.append(f"newly missing: {', '.join(new_missing)}")

        if cand["error"] and not base["error"]:
            notes.append("NEW ERROR")
            reasons.append(f"{qid}: new error")

        for key, count in (cand.get("guard_events") or {}).items():
            base_count = (base.get("guard_events") or {}).get(key, 0)
            if count > base_count:
                notes.append(f"guard {key} {base_count}->{count}")

        if delta <= -QUESTION_DROP_MAX:
            reasons.append(f"{qid}: composite dropped {abs(delta):.3f} (>= {QUESTION_DROP_MAX})")

        sign = "+" if delta >= 0 else ""
        print(f"{qid:>4}  {base['composite']:>6.3f}  {cand['composite']:>6.3f}  "
              f"{sign}{delta:>6.3f}  {'; '.join(notes)}")

    return reasons


def _run_comparison(base_dir: Path, cand_dir: Path) -> int:
    base_scores, _base_manifest = _load_scores(base_dir)
    cand_scores, _cand_manifest = _load_scores(cand_dir)

    print(f"Baseline:  {_display_path(base_dir)}")
    print(f"Candidate: {_display_path(cand_dir)}")

    base_agg = _aggregate(base_scores)
    cand_agg = _aggregate(cand_scores)
    _print_aggregate_deltas(base_agg, cand_agg)

    reasons = _print_question_deltas(base_scores, cand_scores)

    composite_drop = round(base_agg["composite"] - cand_agg["composite"], 3)
    grounding_drop = round(base_agg["grounding"] - cand_agg["grounding"], 3)
    if composite_drop > COMPOSITE_DROP_MAX:
        reasons.append(f"aggregate composite dropped {composite_drop:.3f} "
                       f"(> {COMPOSITE_DROP_MAX})")
    if grounding_drop > GROUNDING_DROP_MAX:
        reasons.append(f"aggregate grounding dropped {grounding_drop:.3f} "
                       f"(> {GROUNDING_DROP_MAX})")

    print()
    if reasons:
        print("VERDICT: ❌ REGRESSION DETECTED")
        for r in reasons:
            print(f"  - {r}")
        return 1
    print("VERDICT: ✅ NO REGRESSION")
    return 0


def _print_matrix() -> None:
    pairs = run_store.iter_lang_dirs()
    if not pairs:
        print("[INFO] No historized runs found under results/.")
        return

    print(f"{'Destination':<20} {'Lang':<6} {'Composite':>9}  {'Grounding':>9}  "
          f"{'Retrieval':>9}  {'Fetched':>8}  Run")
    print("-" * 90)
    for dest, lang, _lang_dir in pairs:
        history = run_store.load_history(dest, lang)
        latest_eval_run = run_store.find_latest_run(dest, lang, require="eval.json")
        row = None
        if latest_eval_run is not None:
            row = next((h for h in reversed(history)
                       if h.get("run_id") == latest_eval_run.name
                       and h.get("kind") == "eval"), None)
        if row is None:
            print(f"{dest:<20} {lang:<6} {'—':>9}  {'—':>9}  {'—':>9}  {'—':>8}  (unscored)")
            continue
        print(f"{dest:<20} {lang:<6} {row['composite']:>9.3f}  {row['grounding']:>9.1%}  "
              f"{row['retrieval']:>9.1%}  {row['content_fetched']:>8.1%}  {row['run_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare historized eval runs for regressions")
    parser.add_argument("--dest", default="ubeda", help="Destination slug (default: ubeda)")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--candidate", default=None,
                        help="Run dir to evaluate (default: latest eval run)")
    parser.add_argument("--against", default=None,
                        help="Run dir to compare against (default: pinned baseline)")
    parser.add_argument("--set-baseline", action="store_true",
                        help="Pin the candidate (default: latest) as the new baseline and exit")
    parser.add_argument("--matrix", action="store_true",
                        help="Print the destination x language latest-composite matrix and exit")
    args = parser.parse_args()

    if args.matrix:
        _print_matrix()
        return

    cand_dir = _resolve_run_dir(args.candidate) if args.candidate \
        else run_store.find_latest_run(args.dest, args.lang, require="eval.json")
    if cand_dir is None:
        print(f"[ERROR] No eval run found for {args.dest}/{args.lang}", file=sys.stderr)
        sys.exit(2)

    if args.set_baseline:
        run_store.set_baseline(args.dest, args.lang, cand_dir)
        print(f"[INFO] Baseline for {args.dest}/{args.lang} -> {cand_dir.name}")
        return

    base_dir = _resolve_run_dir(args.against) if args.against \
        else run_store.read_baseline(args.dest, args.lang)
    if base_dir is None:
        print(f"[ERROR] No baseline pinned for {args.dest}/{args.lang}. "
              f"Run with --set-baseline first, or pass --against.", file=sys.stderr)
        sys.exit(2)

    if base_dir == cand_dir:
        print("[INFO] Candidate and baseline are the same run; nothing to compare.")
        return

    sys.exit(_run_comparison(base_dir, cand_dir))


if __name__ == "__main__":
    main()

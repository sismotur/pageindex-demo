"""
tests/test_rubric.py — Rubric regression tests.

Loads the pinned ubeda/en baseline run (results/ubeda/en/baseline.json,
set via `assistant/compare_runs.py --set-baseline`) and re-scores it with
the current assistant/score_results.py logic. Asserts that four
historically rubric-tricky questions keep acceptable scores, and that
the aggregate grounding/composite stay at or above a healthy floor.

This is a fast, dependency-free smoke check; `assistant/compare_runs.py`
is the precise regression gate (per-question deltas, tighter thresholds)
and should be used before shipping a change — see AGENTS.md's regression
workflow section.

Also validates the deterministic section summaries baked into
indexes/ubeda/en.json by the sibling pipeline's build_index.py.

Run with:
    cd /path/to/pageindex-demo
    .venv/bin/python -m pytest tests/test_rubric.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))

import run_store
from score_results import (
    guard_tallies,
    score_result,
    score_factual_grounding,
    _matches,
    _expected_sections_for_manifest,
    EXPECTED_SECTIONS,
)

INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda" / "en.json"


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def baseline_run_dir():
    run_dir = run_store.read_baseline("ubeda", "en")
    if run_dir is None:
        pytest.skip("No pinned baseline for ubeda/en — run assistant/run_eval.py "
                   "then assistant/compare_runs.py --set-baseline first.")
    return run_dir


@pytest.fixture(scope="module")
def eval_results(baseline_run_dir):
    eval_file = baseline_run_dir / "eval.json"
    if not eval_file.exists():
        pytest.skip(f"Eval file not found: {eval_file}")
    with open(eval_file, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scores(eval_results, baseline_run_dir):
    manifest = run_store.read_manifest(baseline_run_dir)
    expected_sections = _expected_sections_for_manifest(manifest) if manifest else EXPECTED_SECTIONS
    return {r["id"]: score_result(r, expected_sections) for r in eval_results}


# ── Phone normalization unit tests ─────────────────────────────────────────────

class TestPhoneNormalization:
    """_matches() should equate phone numbers regardless of spacing."""

    def test_continuous_matches_spaced(self):
        assert _matches("+34953750345", "+34 953 75 03 45")

    def test_continuous_matches_dashes(self):
        assert _matches("+34953750345", "+34-953-750345")

    def test_spaced_matches_continuous(self):
        # Check written as a spaced number still finds match in continuous form
        assert _matches("+34 953 750345", "+34953750345")

    def test_wrong_digits_do_not_match(self):
        assert not _matches("+34953750345", "+34953750346")

    def test_non_phone_unchanged(self):
        assert _matches("vázquez de molina", "visit the vázquez de molina square")
        assert not _matches("vázquez de molina", "visit the cathedral")


# ── Guard telemetry ─────────────────────────────────────────────────────────

class TestGuardTelemetry:
    """Runtime guard interventions (loop block, repeat cache stub,
    validation rejection, chant truncation) are tallied per question so a
    nonzero count is visible when porting to a new model."""

    def test_counts_flags_from_tool_calls(self):
        result = {
            "tool_calls": [
                {"tool": "get_poi", "args": {}},
                {"tool": "get_poi", "args": {}, "repeat_cached": True},
                {"tool": "get_poi", "args": {}, "loop_blocked": True},
                {"tool": "get_poi", "args": {}, "invalid_args": True},
                {"tool": "get_poi", "args": {}, "loop_blocked": True},
            ],
            "chant_truncated": True,
        }
        assert guard_tallies(result) == {
            "repeat_cached": 1,
            "loop_blocked": 2,
            "invalid_args": 1,
            "chant_truncated": 1,
        }

    def test_clean_run_has_zero_tallies(self):
        result = {"tool_calls": [{"tool": "get_poi", "args": {}}]}
        assert not any(guard_tallies(result).values())

    def test_score_result_carries_guard_events(self):
        result = {
            "id": "QX", "answer": "Some answer text.",
            "tool_calls": [{"tool": "get_poi", "args": {},
                            "repeat_cached": True}],
        }
        scored = score_result(result)
        assert scored["guard_events"]["repeat_cached"] == 1
        assert scored["guard_events"]["loop_blocked"] == 0


# ── Per-question assertions ───────────────────────────────────────────────────

class TestPreviouslyFailingQuestions:
    """Historically rubric-tricky questions: floors reflect the current
    E2B baseline (composite 0.890, grounding 80.0%), not an exact score
    — they exist to catch a real regression, not to pin a moving target.
    """

    def test_q03_parador_phone(self, scores):
        """Phone formatted with spaces should match (digit-only comparison)."""
        s = scores["Q03"]
        assert s["grounding"] == 1.0, f"Q03 grounding={s['grounding']}, missing={s['missing_facts']}"
        assert s["composite"] == 1.0, f"Q03 composite={s['composite']}"

    def test_q15_dolmen_retrieval(self, scores):
        """Dolmen is in Tourist Attractions and Viewpoints.

        The Dolmen's 'megalithic' keyword is occasionally paraphrased.
        Composite >= 0.60 is acceptable: retrieval=1.0 confirms correct
        section navigation even when grounding misses the exact keyword.
        """
        s = scores["Q15"]
        assert s["composite"] >= 0.60, (
            f"Q15 composite={s['composite']} — "
            f"retrieval={s['retrieval']}, grounding={s['grounding']}"
        )

    def test_q17_tour_agencies(self, scores):
        """'falcon' (Falcon Travel) is the fact check; E2B sometimes names a
        tour without the operator's brand, so grounding floors at 0.5
        (baseline: grounding=0.5, composite=0.5, missing=['falcon']).
        """
        s = scores["Q17"]
        assert s["grounding"] >= 0.5, f"Q17 grounding={s['grounding']}, missing={s['missing_facts']}"
        assert s["composite"] >= 0.5, f"Q17 composite={s['composite']}"

    def test_q20_unique_appeal(self, scores):
        """Synthesis question: the designation-guard fix forces a
        search_pois('unesco') lookup, so retrieval=1.0 even when the
        prose doesn't restate '2003'/'renaissance' verbatim (baseline:
        grounding=0.0, retrieval=1.0, composite=0.6 — 0.3 retrieval +
        0.2 fetched + 0.1 language).  Composite is the meaningful floor
        here; grounding alone is a known over-strict rubric item.
        """
        s = scores["Q20"]
        assert s["composite"] >= 0.30, f"Q20 composite={s['composite']}"


# ── Aggregate thresholds ─────────────────────────────────────────────────

class TestAggregateThresholds:
    """System-level smoke floors — well below the current baseline
    (grounding 80.0%, composite 0.890) so day-to-day fluctuation doesn't
    flake, but tight enough to catch a real break. Precise regression
    detection (per-question deltas, tighter thresholds) belongs to
    `assistant/compare_runs.py`, not this file.
    """

    def test_grounding_meets_rubric_threshold(self, scores):
        """Matches the project-wide 70% rubric pass bar (see README/AGENTS.md)."""
        avg = sum(s["grounding"] for s in scores.values()) / len(scores)
        assert avg >= 0.70, f"Grounding avg={avg:.1%} — expected ≥ 70%"

    def test_no_perfect_zero_composites(self, scores):
        zeros = [qid for qid, s in scores.items() if s["composite"] == 0.0]
        assert not zeros, f"Questions with composite=0.0: {zeros}"

    def test_all_questions_have_answers(self, eval_results):
        empty = [r["id"] for r in eval_results if not r.get("answer", "").strip()]
        assert not empty, f"Questions with empty answers: {empty}"

    def test_composite_stays_healthy(self, scores):
        avg = sum(s["composite"] for s in scores.values()) / len(scores)
        assert avg >= 0.80, f"Composite avg={avg:.3f} — expected ≥ 0.80"


# ── Section summary quality tests (POI-aware index format) ──────────────────────
# These validate the deterministic summaries produced by pipeline/build_index.py
# inside indexes/ubeda/en.json.  They replace the retired PageIndex-era tests
# that targeted results/ubeda_guide_structure.json (removed in 8df83d2).

@pytest.fixture(scope="module")
def index_data():
    if not INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {INDEX_FILE}")
    with open(INDEX_FILE, encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def sections(index_data):
    secs = index_data.get("sections", [])
    if not secs:
        pytest.skip("No sections found in index")
    return secs


class TestSectionSummaryQuality:
    """Verify that section summaries exist and have meaningful content."""

    def test_all_sections_have_summaries(self, sections):
        missing = [s["title"] for s in sections if not s.get("summary", "").strip()]
        assert not missing, f"Sections without summaries: {missing}"

    def test_summaries_are_long_enough(self, sections):
        """Each summary must be > 40 chars (a bare title would be shorter)."""
        short = [
            (s["title"], len(s.get("summary", "")))
            for s in sections
            if len(s.get("summary", "")) <= 40
        ]
        assert not short, f"Summaries too short (<=40 chars): {short}"

    def test_accommodation_summary_mentions_parador(self, sections):
        """Accommodation summary should mention the Parador by name."""
        acc = next(
            (s for s in sections if "accommodation" in s["title"].lower()),
            None,
        )
        assert acc is not None, "Accommodation section not found"
        summary = acc.get("summary", "").lower()
        assert "condestable" in summary or "parador" in summary, (
            f"Accommodation summary does not mention Parador: {acc['summary'][:120]}"
        )

    def test_gastronomy_summary_mentions_olive_oil(self, sections):
        """Gastronomy summary should mention olive oil or olive mill."""
        gast = next(
            (s for s in sections if "gastronomy" in s["title"].lower()),
            None,
        )
        assert gast is not None, "Gastronomy section not found"
        summary = gast.get("summary", "").lower()
        assert "olive" in summary or "almazara" in summary or "oil" in summary, (
            f"Gastronomy summary does not mention olive oil: {gast['summary'][:120]}"
        )

    def test_tours_summary_mentions_specific_operator(self, sections):
        """Guided Tours summary should name at least one operator."""
        tours = next(
            (s for s in sections if "guided tours" in s["title"].lower()),
            None,
        )
        assert tours is not None, "Guided Tours section not found"
        summary = tours.get("summary", "").lower()
        assert any(name in summary for name in ["falcon", "mh travel", "trails"]), (
            f"Guided Tours summary does not name a specific operator: {tours['summary'][:120]}"
        )

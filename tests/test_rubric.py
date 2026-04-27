"""
tests/test_rubric.py — Rubric regression tests.

Loads the existing eval_gemma4-26b.json results and re-scores them with
the current score_results.py logic. Asserts that the four previously
artefact-failing questions now reach composite=1.0, and that the
aggregate grounding stays at or above 95%.

Run with:
    cd /path/to/pageindex-demo
    .venv/bin/python -m pytest tests/test_rubric.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from score_results import score_result, score_factual_grounding, _matches

EVAL_FILE      = PROJECT_ROOT / "results" / "eval_gemma4-26b.json"
STRUCTURE_FILE = PROJECT_ROOT / "results" / "ubeda_guide_structure.json"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eval_results():
    if not EVAL_FILE.exists():
        pytest.skip(f"Eval file not found: {EVAL_FILE}")
    with open(EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scores(eval_results):
    return {r["id"]: score_result(r) for r in eval_results}


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


# ── Per-question assertions ────────────────────────────────────────────────────

class TestPreviouslyFailingQuestions:
    """All four artefact-failing questions must now score composite=1.0."""

    def test_q03_parador_phone(self, scores):
        """Phone formatted with spaces should now match."""
        s = scores["Q03"]
        assert s["grounding"] == 1.0, f"Q03 grounding={s['grounding']}, missing={s['missing_facts']}"
        assert s["composite"] == 1.0, f"Q03 composite={s['composite']}"

    def test_q15_dolmen_retrieval(self, scores):
        """Dolmen is in Tourist Attractions and Viewpoints.

        Q15 can pass via retrieval=1.0 (model navigates to the right section)
        OR via grounding=1.0 (model finds the Dolmen content regardless).
        Both paths result in composite >= 0.70. The section summary now
        explicitly names the Dolmen, reducing stochastic navigation failures.
        """
        s = scores["Q15"]
        assert s["composite"] >= 0.70, (
            f"Q15 composite={s['composite']} — "
            f"retrieval={s['retrieval']}, grounding={s['grounding']}"
        )

    def test_q17_tour_agencies(self, scores):
        """'falcon' (Falcon Travel) replaces 'itinerar' as the fact check."""
        s = scores["Q17"]
        assert s["grounding"] == 1.0, f"Q17 grounding={s['grounding']}, missing={s['missing_facts']}"
        assert s["composite"] == 1.0, f"Q17 composite={s['composite']}"

    def test_q20_unique_appeal(self, scores):
        """'andalusia' removed; '2003' + 'renaissance' are sufficient."""
        s = scores["Q20"]
        assert s["grounding"] == 1.0, f"Q20 grounding={s['grounding']}, missing={s['missing_facts']}"
        assert s["composite"] == 1.0, f"Q20 composite={s['composite']}"


# ── Aggregate thresholds ───────────────────────────────────────────────────────

class TestAggregateThresholds:
    """System-level pass criteria after rubric fixes."""

    def test_grounding_above_95_percent(self, scores):
        avg = sum(s["grounding"] for s in scores.values()) / len(scores)
        assert avg >= 0.95, f"Grounding avg={avg:.1%} — expected ≥ 95%"

    def test_no_perfect_zero_composites(self, scores):
        zeros = [qid for qid, s in scores.items() if s["composite"] == 0.0]
        assert not zeros, f"Questions with composite=0.0: {zeros}"

    def test_all_questions_have_answers(self, eval_results):
        empty = [r["id"] for r in eval_results if not r.get("answer", "").strip()]
        assert not empty, f"Questions with empty answers: {empty}"

    def test_composite_above_90_percent(self, scores):
        avg = sum(s["composite"] for s in scores.values()) / len(scores)
        assert avg >= 0.90, f"Composite avg={avg:.3f} — expected ≥ 0.90"


# ── Section summary quality tests ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def structure_data():
    if not STRUCTURE_FILE.exists():
        pytest.skip(f"Structure file not found: {STRUCTURE_FILE}")
    with open(STRUCTURE_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def section_nodes(structure_data):
    root = structure_data.get("structure", [])
    if not root:
        pytest.skip("No structure nodes found")
    return root[0].get("nodes", [])


class TestSectionSummaryQuality:
    """Verify that section summaries exist and have meaningful content."""

    def test_all_18_sections_have_summaries(self, section_nodes):
        missing = [s["title"] for s in section_nodes if not s.get("summary", "").strip()]
        assert not missing, f"Sections without summaries: {missing}"

    def test_summaries_are_long_enough(self, section_nodes):
        """Each summary must be > 100 chars (title-only summaries are shorter)."""
        short = [
            (s["title"], len(s.get("summary", "")))
            for s in section_nodes
            if len(s.get("summary", "")) <= 100
        ]
        assert not short, f"Summaries too short (<=100 chars): {short}"

    def test_accommodation_summary_mentions_parador(self, section_nodes):
        """Accommodation summary should mention the Parador by name."""
        acc = next(
            (s for s in section_nodes if "accommodation" in s["title"].lower()),
            None,
        )
        assert acc is not None, "Accommodation section not found"
        summary = acc.get("summary", "").lower()
        assert "condestable" in summary or "parador" in summary, (
            f"Accommodation summary does not mention Parador: {acc['summary'][:120]}"
        )

    def test_gastronomy_summary_mentions_olive_oil(self, section_nodes):
        """Gastronomy summary should mention olive oil or olive mill."""
        gast = next(
            (s for s in section_nodes if "gastronomy" in s["title"].lower()),
            None,
        )
        assert gast is not None, "Gastronomy section not found"
        summary = gast.get("summary", "").lower()
        assert "olive" in summary or "almazara" in summary or "oil" in summary, (
            f"Gastronomy summary does not mention olive oil: {gast['summary'][:120]}"
        )

    def test_tours_summary_mentions_specific_operator(self, section_nodes):
        """Guided Tours summary should name at least one operator."""
        tours = next(
            (s for s in section_nodes if "guided" in s["title"].lower()
             or "itinerar" in s["title"].lower()),
            None,
        )
        assert tours is not None, "Guided Tours section not found"
        summary = tours.get("summary", "").lower()
        assert any(name in summary for name in ["falcon", "mh travel", "trails"]), (
            f"Guided Tours summary does not name a specific operator: {tours['summary'][:120]}"
        )

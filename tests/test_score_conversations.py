"""
tests/test_score_conversations.py — Scored-conversation rubric tests.

Run with:
    cd /path/to/pageindex-demo
    .venv/bin/python -m pytest tests/test_score_conversations.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))

import score_conversations as sc
from index_tools import load_index

INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda" / "en.json"


@pytest.fixture(scope="module")
def index():
    if not INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {INDEX_FILE}")
    return load_index(INDEX_FILE)


# ── is_repeat_of_previous_turn ───────────────────────────────────────────────

class TestIsRepeatOfPreviousTurn:
    def test_first_turn_is_never_a_repeat(self):
        turns = [{"answer": "Hello!"}]
        assert sc.is_repeat_of_previous_turn(turns, 0) is False

    def test_identical_answer_is_a_repeat(self):
        turns = [{"answer": "¡Hola! ¿En qué puedo ayudarte?"},
                {"answer": "¡Hola! ¿En qué puedo ayudarte?"}]
        assert sc.is_repeat_of_previous_turn(turns, 1) is True

    def test_normalized_match_ignores_accents_and_case(self):
        turns = [{"answer": "Bienvenido a Úbeda!"},
                {"answer": "bienvenido a ubeda!"}]
        assert sc.is_repeat_of_previous_turn(turns, 1) is True

    def test_different_answer_is_not_a_repeat(self):
        turns = [{"answer": "Hola"}, {"answer": "El castillo está aquí."}]
        assert sc.is_repeat_of_previous_turn(turns, 1) is False

    def test_empty_answer_is_not_a_repeat(self):
        turns = [{"answer": ""}, {"answer": ""}]
        assert sc.is_repeat_of_previous_turn(turns, 1) is False


# ── score_turn ────────────────────────────────────────────────────────────

class TestScoreTurn:
    def test_must_mention_all_present_scores_full(self):
        run_turn = {"answer": "Visit the Parador hotel today.", "tool_calls": []}
        source_turn = {"must_mention": ["parador", "hotel"]}
        result = sc.score_turn(run_turn, source_turn, [run_turn], None, "en", 0)
        assert result["must_mention_score"] == 1.0
        assert result["missing_mentions"] == []
        assert result["has_checks"] is True

    def test_must_mention_partial_credit_and_missing_list(self):
        run_turn = {"answer": "Visit the Parador.", "tool_calls": []}
        source_turn = {"must_mention": ["parador", "hotel"]}
        result = sc.score_turn(run_turn, source_turn, [run_turn], None, "en", 0)
        assert result["must_mention_score"] == 0.5
        assert result["missing_mentions"] == ["hotel"]

    def test_forbidden_hit_is_reported(self):
        run_turn = {"answer": "I don't have that information.", "tool_calls": []}
        source_turn = {"forbidden": ["don't have"]}
        result = sc.score_turn(run_turn, source_turn, [run_turn], None, "en", 0)
        assert result["forbidden_hits"] == ["don't have"]

    def test_no_authored_checks_has_checks_false(self):
        run_turn = {"answer": "Sure, here is more.", "tool_calls": []}
        result = sc.score_turn(run_turn, {}, [run_turn], None, "en", 0)
        assert result["has_checks"] is False
        assert result["must_mention_score"] is None
        assert result["retrieval"] is None

    def test_is_reask_flagged_on_bare_clarifying_question(self):
        run_turn = {"answer": "What would you like to know?", "tool_calls": []}
        result = sc.score_turn(run_turn, {}, [run_turn], None, "en", 0)
        assert result["is_reask"] is True

    def test_error_flag_passthrough(self):
        run_turn = {"answer": "", "tool_calls": [], "error": "timeout"}
        result = sc.score_turn(run_turn, {}, [run_turn], None, "en", 0)
        assert result["error"] is True

    def test_expected_section_retrieval_hit_with_real_index(self, index):
        run_turn = {
            "answer": "Here is the parador.",
            "tool_calls": [{"tool": "get_section", "args": {"section_id": "accommodation"}}],
        }
        source_turn = {"expected_section": "Accommodation"}
        result = sc.score_turn(run_turn, source_turn, [run_turn], index, "en", 0)
        assert result["retrieval"] == 1.0

    def test_expected_section_retrieval_miss_with_real_index(self, index):
        run_turn = {
            "answer": "Here is the parador.",
            "tool_calls": [{"tool": "get_section", "args": {"section_id": "gastronomy"}}],
        }
        source_turn = {"expected_section": "Accommodation"}
        result = sc.score_turn(run_turn, source_turn, [run_turn], index, "en", 0)
        assert result["retrieval"] == 0.0


# ── score_conversation ────────────────────────────────────────────────────

class TestScoreConversation:
    def test_hard_fail_floors_composite_despite_passing_content_checks(self):
        """A repeated (parroted) answer must zero the composite even when
        the content check alone would have scored 1.0 — the whole point
        of the repeat-answer regression coverage."""
        run_thread = {
            "id": "T1", "title": "t",
            "turns": [
                {"answer": "Hola", "tool_calls": []},
                {"answer": "Hola", "tool_calls": []},  # exact repeat of turn 1
            ],
        }
        source_thread = {"turns": [{}, {"must_mention": ["hola"]}]}
        result = sc.score_conversation(run_thread, source_thread, None, "es")

        assert result["repeat_count"] == 1
        assert result["turns"][1]["must_mention_score"] == 1.0  # content alone would pass
        assert result["composite"] == 0.0  # but the repeat hard-floors it

    def test_unannotated_thread_has_none_composite(self):
        run_thread = {
            "id": "T2", "title": "t",
            "turns": [{"answer": "Sure, here is info.", "tool_calls": []}],
        }
        result = sc.score_conversation(run_thread, None, None, "en")
        assert result["composite"] is None
        assert result["n_checked_turns"] == 0

    def test_outcome_check_uses_final_turn_answer(self):
        run_thread = {
            "id": "T3", "title": "t",
            "turns": [
                {"answer": "Let me look that up.", "tool_calls": []},
                {"answer": "The Castillo de Montánchez is a fortress.", "tool_calls": []},
            ],
        }
        source_thread = {"turns": [{}, {}], "outcome": {"must_mention": ["castillo"]}}
        result = sc.score_conversation(run_thread, source_thread, None, "es")
        assert result["outcome_score"] == 1.0
        assert result["composite"] == 1.0

    def test_unannotated_thread_still_reports_reask_but_stays_unscored(self):
        """Structural health signals (reask/repeat) are always computed and
        reported, but only fold into `composite` for threads that opted
        into scoring by authoring at least one check."""
        run_thread = {
            "id": "T4", "title": "t",
            "turns": [{"answer": "What would you like to know?", "tool_calls": []}],
        }
        result = sc.score_conversation(run_thread, None, None, "en")
        assert result["reask_count"] == 1
        assert result["composite"] is None

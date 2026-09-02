"""
tests/test_compare_runs.py — Regression-gate threshold tests.

Exercises assistant/compare_runs.py's decision logic against synthetic
scored.json fixtures written under pytest tmp_path, so these tests never
depend on live eval data or a pinned baseline.

Run with:
    cd /path/to/pageindex-demo
    .venv/bin/python -m pytest tests/test_compare_runs.py -v
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))

import compare_runs


def _score(qid, composite, grounding=0.8, error=False, missing_facts=None):
    """A minimal score_result()-shaped dict. `composite`/`grounding` are
    read directly by compare_runs (never recomputed), so tests can set
    them independently to isolate one regression rule at a time.
    """
    return {
        "id": qid, "difficulty": "easy", "category": "x",
        "grounding": grounding, "retrieval": 0.8, "content_fetched": 0.8,
        "language_ok": 1.0, "composite": composite, "latency": 10.0,
        "prompt_tokens": 100, "completion_tokens": 10,
        "error": error, "missing_facts": missing_facts or [],
        "guard_events": {"repeat_cached": 0, "loop_blocked": 0,
                         "invalid_args": 0, "chant_truncated": 0},
    }


def _write_run(tmp_path, name, scores):
    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "scored.json", "w", encoding="utf-8") as f:
        json.dump(scores, f)
    return run_dir


class TestNoRegression:
    def test_identical_runs_pass(self, tmp_path):
        scores = [_score("Q01", 1.0), _score("Q02", 0.8)]
        base = _write_run(tmp_path, "base", scores)
        cand = _write_run(tmp_path, "cand", scores)
        assert compare_runs._run_comparison(base, cand) == 0

    def test_improvement_passes(self, tmp_path):
        base = _write_run(tmp_path, "base", [_score("Q01", 0.5)])
        cand = _write_run(tmp_path, "cand", [_score("Q01", 0.9)])
        assert compare_runs._run_comparison(base, cand) == 0


class TestPerQuestionRegression:
    def test_single_question_drop_at_threshold_fails_in_isolation(self, tmp_path):
        """20 questions (matching real eval size): one drops by exactly the
        per-question threshold. Aggregate composite drop is only 0.25/20 =
        0.0125 (below the 0.02 aggregate rule), so only the per-question
        rule can be responsible for the failure.
        """
        base = [_score(f"Q{i:02d}", 0.80) for i in range(20)]
        cand = [_score(f"Q{i:02d}", 0.80) for i in range(20)]
        cand[0]["composite"] = 0.55  # drop of 0.25 == QUESTION_DROP_MAX
        base_dir = _write_run(tmp_path, "base", base)
        cand_dir = _write_run(tmp_path, "cand", cand)
        assert compare_runs._run_comparison(base_dir, cand_dir) == 1

    def test_small_per_question_drop_does_not_fail_alone(self, tmp_path):
        base = [_score(f"Q{i:02d}", 0.80) for i in range(20)]
        cand = [_score(f"Q{i:02d}", 0.80) for i in range(20)]
        cand[0]["composite"] = 0.60  # drop of 0.20, below threshold
        base_dir = _write_run(tmp_path, "base", base)
        cand_dir = _write_run(tmp_path, "cand", cand)
        assert compare_runs._run_comparison(base_dir, cand_dir) == 0

    def test_question_missing_from_baseline_is_not_a_regression_reason(self):
        base_scores = [_score("Q01", 1.0)]
        cand_scores = [_score("Q01", 1.0), _score("Q02", 0.1)]
        reasons = compare_runs._print_question_deltas(base_scores, cand_scores)
        assert reasons == []


class TestAggregateRegression:
    def test_aggregate_composite_drop_above_threshold_fails(self, tmp_path):
        base = [_score(f"Q{i:02d}", 0.90) for i in range(10)]
        cand = [_score(f"Q{i:02d}", 0.85) for i in range(10)]  # drop 0.05 > 0.02
        base_dir = _write_run(tmp_path, "base", base)
        cand_dir = _write_run(tmp_path, "cand", cand)
        assert compare_runs._run_comparison(base_dir, cand_dir) == 1

    def test_aggregate_grounding_drop_fails_even_without_composite_drop(self, tmp_path):
        """Composite is unchanged (even improves slightly) but grounding
        alone drops past its own, tighter threshold — grounding regressions
        must be caught independently of the blended composite metric.
        """
        base = [_score("Q01", composite=0.50, grounding=1.00)]
        cand = [_score("Q01", composite=0.55, grounding=0.90)]  # grounding drop 0.10
        base_dir = _write_run(tmp_path, "base", base)
        cand_dir = _write_run(tmp_path, "cand", cand)
        assert compare_runs._run_comparison(base_dir, cand_dir) == 1


class TestNewErrorRegression:
    def test_new_error_fails_regardless_of_composite(self, tmp_path):
        base = _write_run(tmp_path, "base", [_score("Q01", 1.0, error=False)])
        cand = _write_run(tmp_path, "cand", [_score("Q01", 1.0, error=True)])
        assert compare_runs._run_comparison(base, cand) == 1

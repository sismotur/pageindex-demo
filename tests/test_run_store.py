"""
tests/test_run_store.py — Historized run-storage regression tests.

Every test redirects run_store.RESULTS_DIR to a pytest tmp_path via the
`results_dir` fixture, so these tests never touch the real results/
directory on disk.

Run with:
    cd /path/to/pageindex-demo
    .venv/bin/python -m pytest tests/test_run_store.py -v
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))

import run_store


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Redirect run_store's module-level RESULTS_DIR to an empty tmp dir."""
    monkeypatch.setattr(run_store, "RESULTS_DIR", tmp_path)
    return tmp_path


# ── Naming ───────────────────────────────────────────────────────────────────

class TestNaming:
    def test_model_tag_strips_provider_prefix(self):
        assert run_store.model_tag("openai/gemma-4-E2B-it-MLX-8bit") == "gemma-4-E2B-it-MLX-8bit"

    def test_model_tag_replaces_colon(self):
        assert run_store.model_tag("openai/gemma4:e2b") == "gemma4-e2b"

    def test_make_run_id_format(self):
        ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert run_store.make_run_id("openai/gemma4:e2b", ts) == "2026-01-02T03-04-05Z_gemma4-e2b"


# ── Run directory creation ───────────────────────────────────────────────────

class TestRunDir:
    def test_new_run_dir_creates_expected_path_shape(self, results_dir):
        run_dir = run_store.new_run_dir("ubeda", "en", "openai/gemma4:e2b")
        assert run_dir.is_dir()
        assert run_dir.parent.name == "runs"
        assert run_dir.parent.parent.name == "en"
        assert run_dir.parent.parent.parent.name == "ubeda"

    def test_new_run_dir_avoids_collision_with_suffix(self, results_dir):
        fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
        first = run_store.new_run_dir("ubeda", "en", "openai/gemma4:e2b", started_at=fixed)
        second = run_store.new_run_dir("ubeda", "en", "openai/gemma4:e2b", started_at=fixed)
        assert first != second
        assert second.name.endswith("-2")


# ── Manifest ───────────────────────────────────────────────────────────────

class TestManifest:
    def test_build_manifest_round_trip(self, results_dir, tmp_path):
        run_dir = run_store.new_run_dir("ubeda", "en", "openai/gemma4:e2b")
        index_path = tmp_path / "index.json"
        index_data = {"meta": {"generated_at": "2026-01-01", "poi_count": 5}}
        index_path.write_text(json.dumps(index_data), encoding="utf-8")

        manifest = run_store.build_manifest(
            run_dir, kind="eval", model="openai/gemma4:e2b", lang="en",
            destination="ubeda", index_path=index_path, index=index_data,
        )

        assert manifest["destination"] == "ubeda"
        assert manifest["lang"] == "en"
        assert manifest["kind"] == "eval"
        assert manifest["index"]["poi_count"] == 5
        assert "sha256_12" in manifest["index"]
        assert run_store.read_manifest(run_dir) == manifest

    def test_read_manifest_missing_returns_empty_dict(self, results_dir):
        run_dir = run_store.new_run_dir("ubeda", "en", "openai/gemma4:e2b")
        assert run_store.read_manifest(run_dir) == {}


# ── latest pointer & discovery ───────────────────────────────────────────────

class TestLatestAndDiscovery:
    def test_find_latest_run_picks_newest_with_required_file(self, results_dir):
        older = run_store.new_run_dir("ubeda", "en", "modelA",
                                      started_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        (older / "eval.json").write_text("[]", encoding="utf-8")
        run_store.update_latest(older)

        newer = run_store.new_run_dir("ubeda", "en", "modelB",
                                      started_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
        (newer / "eval.json").write_text("[]", encoding="utf-8")
        run_store.update_latest(newer)

        assert run_store.find_latest_run("ubeda", "en", require="eval.json") == newer

    def test_find_latest_run_falls_back_when_latest_lacks_required_file(self, results_dir):
        """A conversations-only run updates `latest` but has no eval.json;
        an eval-scoped lookup must self-heal to the newest run that does."""
        eval_run = run_store.new_run_dir("ubeda", "en", "modelA",
                                         started_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        (eval_run / "eval.json").write_text("[]", encoding="utf-8")
        run_store.update_latest(eval_run)

        conv_run = run_store.new_run_dir("ubeda", "en", "modelA",
                                         started_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
        (conv_run / "conversations.json").write_text("[]", encoding="utf-8")
        run_store.update_latest(conv_run)

        assert run_store.find_latest_run("ubeda", "en", require="eval.json") == eval_run
        assert run_store.find_latest_run("ubeda", "en", require="conversations.json") == conv_run

    def test_find_latest_run_none_when_no_runs(self, results_dir):
        assert run_store.find_latest_run("nowhere", "xx") is None

    def test_iter_lang_dirs_lists_only_dirs_with_runs(self, results_dir):
        run_store.new_run_dir("ubeda", "en", "modelA")
        run_store.new_run_dir("ubeda", "es", "modelA")
        pairs = {(dest, lang) for dest, lang, _ in run_store.iter_lang_dirs()}
        assert pairs == {("ubeda", "en"), ("ubeda", "es")}


# ── Baseline ───────────────────────────────────────────────────────────────

class TestBaseline:
    def test_set_and_read_baseline_round_trip(self, results_dir):
        run_dir = run_store.new_run_dir("ubeda", "en", "modelA")
        run_store.set_baseline("ubeda", "en", run_dir)
        assert run_store.read_baseline("ubeda", "en") == run_dir

    def test_read_baseline_missing_returns_none(self, results_dir):
        assert run_store.read_baseline("ubeda", "en") is None

    def test_read_baseline_dangling_run_returns_none(self, results_dir):
        run_dir = run_store.new_run_dir("ubeda", "en", "modelA")
        run_store.set_baseline("ubeda", "en", run_dir)
        shutil.rmtree(run_dir)
        assert run_store.read_baseline("ubeda", "en") is None


# ── History ────────────────────────────────────────────────────────────────

class TestHistory:
    def test_append_and_load_history_preserves_order(self, results_dir):
        run_store.append_history("ubeda", "en", {"run_id": "r1", "kind": "eval", "composite": 0.9})
        run_store.append_history("ubeda", "en", {"run_id": "r2", "kind": "eval", "composite": 0.8})
        rows = run_store.load_history("ubeda", "en")
        assert [r["run_id"] for r in rows] == ["r1", "r2"]

    def test_append_history_is_idempotent_by_run_id_and_kind(self, results_dir):
        run_store.append_history("ubeda", "en", {"run_id": "r1", "kind": "eval", "composite": 0.9})
        appended_again = run_store.append_history(
            "ubeda", "en", {"run_id": "r1", "kind": "eval", "composite": 0.5})
        rows = run_store.load_history("ubeda", "en")

        assert appended_again is False
        assert len(rows) == 1
        assert rows[0]["composite"] == 0.9  # original row untouched

    def test_append_history_allows_same_run_id_different_kind(self, results_dir):
        run_store.append_history("ubeda", "en", {"run_id": "r1", "kind": "eval"})
        appended = run_store.append_history("ubeda", "en", {"run_id": "r1", "kind": "conversations"})
        assert appended is True
        assert len(run_store.load_history("ubeda", "en")) == 2

    def test_load_history_empty_when_missing(self, results_dir):
        assert run_store.load_history("nowhere", "xx") == []

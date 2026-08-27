"""
tests/test_multi_destination.py — Cross-destination isolation regression tests.

Proves the constraint that motivated onboarding Sierra de Montánchez y
Tamuja as a second destination: a destination's compiled index and
generated system prompt must never contain a reference to another
destination's content. Covers the two places this architecture could
realistically leak content — the built index JSON and the shared
system-prompt template (see assistant/run_eval.py::_SYSTEM_PROMPT_TEMPLATE,
which previously hardcoded a real Úbeda POI as its tag-format example).

Run with:
    cd /path/to/pageindex-demo
    .venv/bin/python -m pytest tests/test_multi_destination.py -v
"""

import json
import sys
import unicodedata
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))

from index_tools import load_index, format_sections_overview  # noqa: E402
from run_eval import make_system_prompt  # noqa: E402

UBEDA_INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda_es.json"
MONTANCHEZ_INDEX_FILE = PROJECT_ROOT / "indexes" / "montancheztamuja_es.json"

# Distinctive, destination-only terms that must never cross over. Deliberately
# narrow (proper nouns, not generic Spanish words) so the check cannot produce
# false positives from ordinary shared vocabulary.
UBEDA_TERMS = ("ubeda",)
MONTANCHEZ_TERMS = ("montanchez", "tamuja")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _normalized_text(value) -> str:
    """Lowercase, accent-stripped text of an arbitrary JSON-serializable value."""
    raw = json.dumps(value, ensure_ascii=False)
    return _strip_accents(raw.lower())


def _system_prompt_for(index: dict) -> str:
    meta = index.get("meta", {})
    destination = meta.get("destination_display") or meta.get("destination", "")
    sections_text = format_sections_overview(index)
    overview_text = index.get("destination_overview", "")
    return make_system_prompt(
        sections_text=sections_text,
        destination=destination,
        destination_overview=overview_text,
        lang=meta.get("lang", "es"),
    )


@pytest.fixture(scope="module")
def ubeda_index():
    if not UBEDA_INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {UBEDA_INDEX_FILE}")
    return load_index(UBEDA_INDEX_FILE)


@pytest.fixture(scope="module")
def montanchez_index():
    if not MONTANCHEZ_INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {MONTANCHEZ_INDEX_FILE}")
    return load_index(MONTANCHEZ_INDEX_FILE)


# ── Index content isolation ──────────────────────────────────────────────────

class TestIndexContentIsolation:
    """The compiled index JSON for one destination must never mention the other."""

    def test_ubeda_index_has_no_montanchez_terms(self, ubeda_index):
        haystack = _normalized_text(ubeda_index)
        for term in MONTANCHEZ_TERMS:
            assert term not in haystack, f"Ubeda index unexpectedly contains '{term}'"

    def test_montanchez_index_has_no_ubeda_terms(self, montanchez_index):
        haystack = _normalized_text(montanchez_index)
        for term in UBEDA_TERMS:
            assert term not in haystack, f"Montanchez index unexpectedly contains '{term}'"


# ── System prompt isolation ──────────────────────────────────────────────────

class TestSystemPromptIsolation:
    """The generated system prompt must never leak the other destination's content."""

    def test_ubeda_prompt_has_no_montanchez_terms(self, ubeda_index):
        prompt = _strip_accents(_system_prompt_for(ubeda_index).lower())
        for term in MONTANCHEZ_TERMS:
            assert term not in prompt, f"Ubeda system prompt unexpectedly contains '{term}'"

    def test_montanchez_prompt_has_no_ubeda_terms(self, montanchez_index):
        prompt = _strip_accents(_system_prompt_for(montanchez_index).lower())
        for term in UBEDA_TERMS:
            assert term not in prompt, f"Montanchez system prompt unexpectedly contains '{term}'"

    def test_prompt_template_has_no_hardcoded_destination_content(self):
        """Regression: the shared TEMPLATE STRING itself (before any real
        destination content is interpolated in) must never hardcode a real
        destination's example content (e.g. a real POI id/name) again. The
        rendered prompt legitimately contains real POI names pulled from
        whichever destination's own index is loaded, so that string is not
        a valid target for this check."""
        from run_eval import _SYSTEM_PROMPT_TEMPLATE
        assert "5155" not in _SYSTEM_PROMPT_TEMPLATE
        assert "San Nicolás" not in _SYSTEM_PROMPT_TEMPLATE


# ── Destination metadata sanity ──────────────────────────────────────────────

class TestDestinationMetaIsolation:
    """Each index must self-report its own destination, never the other's."""

    def test_ubeda_meta_is_ubeda(self, ubeda_index):
        assert ubeda_index["meta"]["destination"] == "ubeda"

    def test_montanchez_meta_is_montanchez(self, montanchez_index):
        assert montanchez_index["meta"]["destination"] == "montancheztamuja"

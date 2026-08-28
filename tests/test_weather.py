"""
tests/test_weather.py — Regression tests for the on-device weather tool
and intent detection.

Covers:
  * assistant/index_tools.py  :: load_weather / format_weather / weather_hint
  * assistant/run_eval.py     :: is_weather_request

pipeline/build_weather.py's own normalisation logic (_normalize_forecast,
build_weather, _extract_condition_code) now has its tests in the sibling
inventrip-rag-data repo's tests/test_build_weather.py.

Run with:
    .venv/bin/python -m pytest tests/test_weather.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))

from index_tools import (
    format_weather,
    load_weather,
    weather_hint,
    weather_unavailable_message,
)
from run_eval import is_weather_request


FIXED_NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)


# ── On-device tool ───────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_weather() -> dict:
    """An already-built weather artifact, loaded directly rather than via
    build_weather() — this matches production reality, where the runtime
    only ever reads a pre-built file (see tests/fixtures/weather_ubeda_es_
    built.json, precomputed from tests/fixtures/weather_ubeda_es_raw.json
    in the sibling inventrip-rag-data repo with fetched_at=FIXED_NOW).
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "weather_ubeda_es_built.json"
    with open(fixture_path, encoding="utf-8") as fh:
        return json.load(fh)


class TestFormatWeather:
    def test_missing_file_returns_unavailable(self):
        out = format_weather(None)
        assert out == weather_unavailable_message(None)

    def test_load_weather_missing_file_returns_none(self, tmp_path):
        assert load_weather(tmp_path / "does_not_exist.json") is None

    def test_full_week_lists_all_days(self, sample_weather):
        out = format_weather(sample_weather, now=FIXED_NOW)
        # Seven bulleted lines, one per day (leading `  - ` per line).
        assert out.count("  - ") == 7
        # Every forecast tag is present.
        assert '<forecast day="2026-08-25">Mar 25</forecast>' in out
        assert '<forecast day="2026-08-31">Lun 31</forecast>' in out

    def test_single_day_by_iso_date(self, sample_weather):
        out = format_weather(sample_weather, day="2026-08-27", now=FIXED_NOW)
        assert out.startswith('<forecast day="2026-08-27">Jue 27</forecast>')
        assert "Cielo claro" in out
        assert "32.9\u201345.3 \u00b0C" in out

    def test_today_alias(self, sample_weather):
        out = format_weather(sample_weather, day="hoy", now=FIXED_NOW)
        assert out.startswith('<forecast day="2026-08-25">')

    def test_tomorrow_alias(self, sample_weather):
        out = format_weather(sample_weather, day="tomorrow", now=FIXED_NOW)
        assert out.startswith('<forecast day="2026-08-26">')

    def test_weekday_alias(self, sample_weather):
        # Spanish "s\u00e1bado" -> iso_weekday 6 (Saturday) -> 2026-08-29.
        out = format_weather(sample_weather, day="s\u00e1bado", now=FIXED_NOW)
        assert out.startswith('<forecast day="2026-08-29">')

    def test_stale_prefix_after_24h(self, sample_weather):
        later = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
        out = format_weather(sample_weather, day="today", now=later)
        assert out.startswith("Previsi\u00f3n estimada obtenida hace 3 d\u00edas:")

    def test_expired_after_seven_days(self, sample_weather):
        later = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)
        assert format_weather(sample_weather, now=later) == \
            weather_unavailable_message(sample_weather)

    def test_unknown_day_returns_unavailable(self, sample_weather):
        out = format_weather(sample_weather, day="never", now=FIXED_NOW)
        assert out == weather_unavailable_message(sample_weather)


class TestWeatherHint:
    def test_localized_today_line(self, sample_weather):
        line = weather_hint(sample_weather, "\u00dabeda", now=FIXED_NOW)
        assert line.startswith("Today in \u00dabeda:")
        assert "Cielo claro" in line
        assert '<forecast day="2026-08-25">Mar 25</forecast>' in line

    def test_missing_weather_returns_empty_string(self):
        assert weather_hint(None, "\u00dabeda") == ""


# ── Intent detection ─────────────────────────────────────────────────────────

class TestWeatherIntent:
    @pytest.mark.parametrize("question", [
        "\u00bfqu\u00e9 tiempo har\u00e1 ma\u00f1ana?",
        "what's the weather like this weekend?",
        "che tempo far\u00e0 sabato?",
        "quelle est la m\u00e9t\u00e9o demain?",
        "wie ist das wetter am wochenende?",
        "temperatura mi\u00e9rcoles",
    ])
    def test_matches_weather_intents(self, question):
        assert is_weather_request(question)

    @pytest.mark.parametrize("question", [
        "dame un plan para dos d\u00edas",
        "which hotels are indispensable?",
        "\u00bfd\u00f3nde puedo comer bien?",
    ])
    def test_ignores_non_weather_turns(self, question):
        assert not is_weather_request(question)

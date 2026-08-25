"""
tests/test_weather.py — Regression tests for the weather pipeline,
on-device tool, and intent detection.

Covers:
  * pipeline/build_weather.py :: _normalize_forecast (schema shape)
  * pipeline/build_weather.py :: build_weather (meta wrapper)
  * assistant/index_tools.py  :: load_weather / format_weather / weather_hint
  * assistant/run_eval.py     :: is_weather_request

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

from pipeline.build_weather import (
    _normalize_forecast,
    build_weather,
    _extract_condition_code,
)
from index_tools import (
    format_weather,
    load_weather,
    weather_hint,
    weather_unavailable_message,
)
from run_eval import is_weather_request


FIXTURE_RAW = PROJECT_ROOT / "tests" / "fixtures" / "weather_ubeda_es_raw.json"
FIXED_NOW   = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)


# ── Pipeline normalisation ───────────────────────────────────────────────────

class TestNormalizeForecast:
    """The Cloudflare TS port has to reproduce these fields byte-for-byte."""

    @pytest.fixture
    def raw(self) -> list[dict]:
        with open(FIXTURE_RAW, encoding="utf-8") as fh:
            return json.load(fh)

    def test_produces_seven_days(self, raw):
        assert len(_normalize_forecast(raw)) == 7

    def test_first_entry_shape(self, raw):
        entry = _normalize_forecast(raw)[0]
        assert entry == {
            "date":           "2026-08-25",
            "iso_weekday":    2,
            "day_label":      "Mar 25",
            "temp_min_c":     31.6,
            "temp_max_c":     46.0,
            "condition":      "Cielo claro",
            "condition_code": "2119bfd6-a006-4577-e0f0-bba80a256700",
            "icon_url":       raw[0]["icon"],
        }

    def test_missing_temp_leaves_none(self):
        entry = _normalize_forecast([{
            "forecastTimestamp": 1787655600,
            "forecastDayOfWeek": "Mar 25",
            "currentWeather": "",
        }])[0]
        assert entry["temp_min_c"] is None
        assert entry["temp_max_c"] is None
        assert entry["condition"] == ""
        assert entry["condition_code"] == ""

    def test_invalid_timestamps_are_dropped(self):
        assert _normalize_forecast([{"forecastTimestamp": "invalid"}]) == []

    def test_condition_code_extraction(self):
        # A well-formed CDN URL keeps the 36-char UUID in the middle.
        assert _extract_condition_code(
            "https://inventrip.com/cdn-cgi/imagedelivery/AC/"
            "deadbeef-1234-4321-9abc-abcdef012345/public"
        ) == "deadbeef-1234-4321-9abc-abcdef012345"
        assert _extract_condition_code("") == ""
        assert _extract_condition_code("https://example.com/nothing-here") == ""


class TestBuildWeather:
    """build_weather wraps the normalised forecast with a meta block."""

    def test_meta_carries_destination_and_units(self):
        artifact = build_weather(
            destination="ubeda", lang="es",
            raw=[], latitude=38.01, longitude=-3.37,
            fetched_at=FIXED_NOW,
        )
        assert artifact["meta"] == {
            "destination":    "ubeda",
            "lang":           "es",
            "latitude":       38.01,
            "longitude":      -3.37,
            "units":          "metric",
            "fetched_at":     "2026-08-25T12:00:00Z",
            "expires_at":     "2026-08-26T12:00:00Z",
            "schema_version": 1,
        }
        assert artifact["forecast"] == []


# ── On-device tool ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_weather() -> dict:
    with open(FIXTURE_RAW, encoding="utf-8") as fh:
        raw = json.load(fh)
    return build_weather(
        destination="ubeda", lang="es",
        raw=raw, latitude=38.01, longitude=-3.37,
        fetched_at=FIXED_NOW,
    )


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

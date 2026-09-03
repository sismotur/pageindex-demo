"""
tests/test_i18n.py — Translation-completeness guard for visitor-facing strings.

Every language in common.lang_support.SUPPORTED_LANGS must have an entry
in each visitor-facing i18n table.  SUPPORTED_LANGS defaults to the 16
Inventrip app languages and can be overridden with the SUPPORTED_LANGS
environment variable (e.g. in .env) — this test then guards that list.

Model-facing text needs no entries in these tables: it is generated from
English templates (lang_support.lang_rule / recovery_msg), so any
language the LLM understands is covered without translations.

Run with:
    .venv/bin/python -m pytest tests/test_i18n.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))

from common.lang_support import (
    LANG_DISPLAY,
    SUPPORTED_LANGS,
    display_name,
    lang_rule,
    recovery_msg,
)
from run_eval import (
    CATEGORY_INTENT_TERMS,
    DESIGNATION_INTENT_PHRASES,
    DESTINATION_WIDE_TERMS,
    GROUNDING_FAILURE_MESSAGES,
    HISTORY_FOLLOWUP_LEADS,
    RENTAL_INTENT_TERMS,
    ROUTE_INTENT_TERMS,
    TRIP_PLAN_INTENT_TERMS,
    WEATHER_INTENT_TERMS,
)
from index_tools import (
    WEATHER_UNAVAILABLE_MESSAGES,
    _PLACE_TOPIC_CHOICE_MESSAGES,
    _POI_CHOICE_MESSAGES,
    _TRIP_CHOICE_MESSAGES,
    _WEATHER_STALE_MESSAGES,
)

# Every visitor-facing i18n table in the runtime.  When a new localized
# table is added (any module), register it here so the completeness
# guard covers it.
VISITOR_FACING_TABLES = {
    "LANG_DISPLAY": LANG_DISPLAY,
    "GROUNDING_FAILURE_MESSAGES": GROUNDING_FAILURE_MESSAGES,
    "HISTORY_FOLLOWUP_LEADS": HISTORY_FOLLOWUP_LEADS,
    "WEATHER_UNAVAILABLE_MESSAGES": WEATHER_UNAVAILABLE_MESSAGES,
    "_WEATHER_STALE_MESSAGES": _WEATHER_STALE_MESSAGES,
    "_TRIP_CHOICE_MESSAGES": _TRIP_CHOICE_MESSAGES,
    "_POI_CHOICE_MESSAGES": _POI_CHOICE_MESSAGES,
    "_PLACE_TOPIC_CHOICE_MESSAGES": _PLACE_TOPIC_CHOICE_MESSAGES,
}

# Intent-detection lexicons: not visitor-facing text, but the same
# per-language table shape — the guard ensures every supported language
# has a non-empty term set in each lexicon.
INTENT_LEXICON_TABLES = {
    "ROUTE_INTENT_TERMS": ROUTE_INTENT_TERMS,
    "WEATHER_INTENT_TERMS": WEATHER_INTENT_TERMS,
    "RENTAL_INTENT_TERMS": RENTAL_INTENT_TERMS,
    "TRIP_PLAN_INTENT_TERMS": TRIP_PLAN_INTENT_TERMS,
    "DESIGNATION_INTENT_PHRASES": DESIGNATION_INTENT_PHRASES,
    "CATEGORY_INTENT_TERMS": CATEGORY_INTENT_TERMS,
    "DESTINATION_WIDE_TERMS": DESTINATION_WIDE_TERMS,
}


class TestTranslationCompleteness:
    def test_every_supported_language_is_covered(self):
        for name, table in VISITOR_FACING_TABLES.items():
            missing = [code for code in SUPPORTED_LANGS if code not in table]
            assert not missing, f"{name}: missing translations for {missing}"

    def test_no_empty_translations(self):
        for name, table in VISITOR_FACING_TABLES.items():
            for code in SUPPORTED_LANGS:
                value = table[code]
                if isinstance(value, dict):
                    empties = [k for k, v in value.items()
                               if not str(v).strip()]
                    assert not empties, f"{name}[{code}]: empty {empties}"
                elif isinstance(value, tuple):
                    assert all(str(v).strip() for v in value), (
                        f"{name}[{code}]: empty entry"
                    )
                else:
                    assert str(value).strip(), f"{name}[{code}]: empty string"

    def test_every_supported_language_has_intent_terms(self):
        for name, table in INTENT_LEXICON_TABLES.items():
            missing = [code for code in SUPPORTED_LANGS if code not in table]
            assert not missing, f"{name}: missing term set for {missing}"

    def test_no_empty_intent_terms(self):
        for name, table in INTENT_LEXICON_TABLES.items():
            for code in SUPPORTED_LANGS:
                terms = table[code]
                assert terms, f"{name}[{code}]: empty term set"
                assert all(str(t).strip() for t in terms), (
                    f"{name}[{code}]: blank term"
                )

    def test_trip_plan_lexicon_excludes_bare_detail_words(self):
        # "dame el detalle de <POI>" must not enter curated-trip mode.
        banned = {
            "detalle", "detalles", "detall", "detalls", "xehetasun",
            "xehetasunak", "detail", "details", "detalhe", "detalhes",
            "dettaglio", "dettagli", "detaille", "details",
        }
        for code in SUPPORTED_LANGS:
            terms = {str(t).strip().lower() for t in TRIP_PLAN_INTENT_TERMS[code]}
            overlap = terms & banned
            assert not overlap, (
                f"TRIP_PLAN_INTENT_TERMS[{code}] still has detail-words: "
                f"{sorted(overlap)}"
            )

    def test_stale_template_keeps_day_placeholder(self):
        # format_weather() does template.format(n=...); a translation
        # that drops the placeholder would crash at render time.
        for code in SUPPORTED_LANGS:
            assert "{n}" in _WEATHER_STALE_MESSAGES[code], (
                f"_WEATHER_STALE_MESSAGES[{code}] lost the {{n}} placeholder"
            )

    def test_trip_choice_keys_complete(self):
        for code in SUPPORTED_LANGS:
            assert set(_TRIP_CHOICE_MESSAGES[code]) == {
                "lead", "highlights", "outro",
            }, f"_TRIP_CHOICE_MESSAGES[{code}] keys incomplete"

    def test_poi_choice_keys_complete(self):
        for code in SUPPORTED_LANGS:
            assert set(_POI_CHOICE_MESSAGES[code]) == {
                "lead", "outro",
            }, f"_POI_CHOICE_MESSAGES[{code}] keys incomplete"

    def test_place_topic_choice_keys_complete(self):
        for code in SUPPORTED_LANGS:
            assert set(_PLACE_TOPIC_CHOICE_MESSAGES[code]) == {
                "lead", "place", "topic", "outro",
            }, f"_PLACE_TOPIC_CHOICE_MESSAGES[{code}] keys incomplete"

    def test_english_templates_name_the_language(self):
        # Model-facing templates must interpolate the English language
        # name so the model knows which language to answer in.
        for code in SUPPORTED_LANGS:
            english_name = display_name(code, native=False)
            assert english_name in lang_rule(code)
            assert english_name in recovery_msg(code)

    def test_env_override_is_reflected(self, monkeypatch):
        # SUPPORTED_LANGS comes from the env var when set; the module
        # loader is what reads it, so exercise the loader directly.
        from common import lang_support
        monkeypatch.setenv("SUPPORTED_LANGS", "en, es ,XX")
        assert lang_support._load_supported_langs() == ("en", "es", "xx")
        monkeypatch.delenv("SUPPORTED_LANGS")
        assert lang_support._load_supported_langs() == \
            lang_support._DEFAULT_SUPPORTED_LANGS

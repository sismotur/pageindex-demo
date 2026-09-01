#!/usr/bin/env python3
"""
lang_support.py — Supported-language registry for the assistant runtime.

The Inventrip app is limited to the 16 languages the API exposes under
`/v100/configuration-languages?is_active_app=true`.  That whitelist and
the native display names (chat banner) are the only per-language data
left: all model-facing text is generated from English templates
parameterised with the language's English name.  The model renders the
visitor's language itself, so nothing here needs translating and any
language the LLM understands (160+) works — the whitelist only gates
which languages the app offers.

`SUPPORTED_LANGS` defaults to the 16 app languages and can be overridden
with the environment variable of the same name (comma-separated ISO
639-1 codes), e.g. in `.env`:

    SUPPORTED_LANGS=ca,de,en,es,eu,fr,gl,hi,hr,it,ja,nl,pt,ru,uk,zh

Every code in the effective list must have complete visitor-facing
translations — guarded by tests/test_i18n.py.
"""

from __future__ import annotations

import os

# ── Supported language codes ────────────────────────────────────────────────
# Default order mirrors the API response (alphabetical by ISO 639-1 code).
_DEFAULT_SUPPORTED_LANGS: tuple[str, ...] = (
    "ca",  # Catalan
    "de",  # German
    "en",  # English
    "es",  # Spanish
    "eu",  # Basque
    "fr",  # French
    "gl",  # Galician
    "hi",  # Hindi
    "hr",  # Croatian
    "it",  # Italian
    "ja",  # Japanese
    "nl",  # Dutch
    "pt",  # Portuguese
    "ru",  # Russian
    "uk",  # Ukrainian
    "zh",  # Chinese
)


def _load_supported_langs() -> tuple[str, ...]:
    """Read the optional SUPPORTED_LANGS env override (comma-separated)."""
    raw = os.environ.get("SUPPORTED_LANGS", "")
    codes = tuple(c.strip().lower() for c in raw.split(",") if c.strip())
    return codes or _DEFAULT_SUPPORTED_LANGS


SUPPORTED_LANGS: tuple[str, ...] = _load_supported_langs()

# ── Display names (native, English label) ──────────────────────────────────
# Native form is what we show in the interactive banner; English label is
# used in error messages or logs.
LANG_DISPLAY: dict[str, tuple[str, str]] = {
    "ca": ("Català",     "Catalan"),
    "de": ("Deutsch",    "German"),
    "en": ("English",    "English"),
    "es": ("Español",    "Spanish"),
    "eu": ("Euskaraz",   "Basque"),
    "fr": ("Français",   "French"),
    "gl": ("Galego",     "Galician"),
    "hi": ("हिन्दी",        "Hindi"),
    "hr": ("Hrvatski",   "Croatian"),
    "it": ("Italiano",   "Italian"),
    "ja": ("日本語",      "Japanese"),
    "nl": ("Nederlands", "Dutch"),
    "pt": ("Português",  "Portuguese"),
    "ru": ("Pусский",    "Russian"),
    "uk": ("українська", "Ukrainian"),
    "zh": ("中文",        "Chinese"),
}

# ── Model-facing English templates ─────────────────────────────────────────
# The system prompt's closing rule and the empty-answer recovery prompt
# are model-facing, so a single English sentence parameterised with the
# language's English name covers every language the LLM understands.
# Only visitor-facing strings keep per-language tables (guarded by
# tests/test_i18n.py).
_LANG_RULE_TEMPLATE = (
    "Always respond in {language}, regardless of the language of any "
    "retrieved content."
)
_RECOVERY_TEMPLATE = (
    "Based on what you have retrieved above, give your final answer now, "
    "in {language}."
)

# Self-check: every supported language must have a display name (used by
# the chat banner and the templates above).  Raised at import time so a
# missing entry is caught before any request reaches a user.
_missing = [code for code in SUPPORTED_LANGS if code not in LANG_DISPLAY]
if _missing:  # pragma: no cover - sanity check, never fires in practice
    raise RuntimeError(
        f"lang_support.py: missing display names for {_missing}. "
        f"Add entries to LANG_DISPLAY."
    )


def _english_name(code: str) -> str:
    """English language name used to parameterise the templates."""
    pair = LANG_DISPLAY.get(code)
    return pair[1] if pair else "English"


def lang_rule(code: str) -> str:
    """System-prompt rule: answer in the selected language."""
    return _LANG_RULE_TEMPLATE.format(language=_english_name(code))


def recovery_msg(code: str) -> str:
    """Recovery prompt sent if the model returns no final text."""
    return _RECOVERY_TEMPLATE.format(language=_english_name(code))


def display_name(code: str, native: bool = True) -> str:
    """Return the language's display name (native by default)."""
    pair = LANG_DISPLAY.get(code)
    if not pair:
        return code.upper()
    return pair[0] if native else pair[1]


def is_supported(code: str) -> bool:
    """True if `code` is one of the active app languages."""
    return code in SUPPORTED_LANGS

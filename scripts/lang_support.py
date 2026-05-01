#!/usr/bin/env python3
"""
lang_support.py — Multilingual constants for the POI-index pipeline.

Single source of truth for the 16 languages the Inventrip API exposes
under `/v100/configuration-languages?is_active_app=true`.  Imported by
`run_eval.py` and `chat_demo.py` so the per-language rules/labels never
drift between the two entry points.

Anything language-related (system-prompt closing rule, recovery message
sent on empty answers, native + English display name, validation) lives
here.
"""

from __future__ import annotations

# ── Supported language codes ────────────────────────────────────────────────
# Order mirrors the API response (alphabetical by ISO 639-1 code).
SUPPORTED_LANGS: tuple[str, ...] = (
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

# ── System-prompt closing rule per language ────────────────────────────────
# This is appended to the system prompt as the final line, e.g.
#     "- Réponds toujours en français, ..."
# The 26B model honours these rules reliably; smaller models may need
# explicit reinforcement.
LANG_RULES: dict[str, str] = {
    "en": "Always respond in English, regardless of the language of any retrieved content.",
    "es": "Responde siempre en español, independientemente del idioma del contenido recuperado.",
    "fr": "Réponds toujours en français, quelle que soit la langue du contenu récupéré.",
    "de": "Antworte immer auf Deutsch, unabhängig von der Sprache des abgerufenen Inhalts.",
    "it": "Rispondi sempre in italiano, indipendentemente dalla lingua del contenuto recuperato.",
    "pt": "Responde sempre em português, independentemente do idioma do conteúdo recuperado.",
    "nl": "Antwoord altijd in het Nederlands, ongeacht de taal van de opgehaalde inhoud.",
    "ca": "Respon sempre en català, independentment de l'idioma del contingut recuperat.",
    "eu": "Erantzun beti euskaraz, berreskuratutako edukiaren hizkuntza edozein dela ere.",
    "gl": "Responde sempre en galego, independentemente do idioma do contido recuperado.",
    "hr": "Uvijek odgovori na hrvatskom, bez obzira na jezik dohvaćenog sadržaja.",
    "ru": "Всегда отвечай на русском языке, независимо от языка извлечённого содержимого.",
    "uk": "Завжди відповідай українською мовою, незалежно від мови отриманого вмісту.",
    "hi": "हमेशा हिंदी में उत्तर दें, चाहे प्राप्त सामग्री किसी भी भाषा में हो।",
    "ja": "取得したコンテンツの言語に関わらず、常に日本語で回答してください。",
    "zh": "无论检索到的内容是什么语言，请始终用中文回答。",
}

# ── Recovery prompt sent if the model returns no final text ─────────────────
# Wording mirrors the LANG_RULES list (same keys, same imperative voice).
RECOVERY_MSGS: dict[str, str] = {
    "en": "Based on what you have retrieved above, please give your final answer now.",
    "es": "Basándote en lo que has recuperado, da tu respuesta final ahora.",
    "fr": "Sur la base de ce que tu as récupéré, donne ta réponse finale maintenant.",
    "de": "Gib jetzt deine endgültige Antwort, basierend auf dem, was du abgerufen hast.",
    "it": "In base a ciò che hai recuperato, fornisci ora la tua risposta finale.",
    "pt": "Com base no que recuperaste, dá agora a tua resposta final.",
    "nl": "Geef nu je definitieve antwoord op basis van wat je hebt opgehaald.",
    "ca": "Basant-te en el que has recuperat, dona ara la teva resposta final.",
    "eu": "Berreskuratu duzunaren arabera, eman zure azken erantzuna orain.",
    "gl": "En función do que recuperaches, dá agora a túa resposta final.",
    "hr": "Na temelju onoga što si dohvatio, sada daj svoj konačni odgovor.",
    "ru": "Опираясь на полученную выше информацию, дай окончательный ответ сейчас.",
    "uk": "Базуючись на отриманій вище інформації, дай остаточну відповідь зараз.",
    "hi": "ऊपर प्राप्त जानकारी के आधार पर, अब अपना अंतिम उत्तर दें।",
    "ja": "上記で取得した内容に基づいて、最終的な回答をお願いします。",
    "zh": "请根据上述检索内容，给出您的最终答案。",
}

# Self-check: every supported language must have a rule, recovery and label.
# Raised at import time so a missing translation is caught before any
# request reaches a user.
_missing = [
    code for code in SUPPORTED_LANGS
    if code not in LANG_RULES
    or code not in RECOVERY_MSGS
    or code not in LANG_DISPLAY
]
if _missing:  # pragma: no cover - sanity check, never fires in practice
    raise RuntimeError(
        f"lang_support.py: missing translations for {_missing}. "
        f"Add entries to LANG_RULES / RECOVERY_MSGS / LANG_DISPLAY."
    )


def lang_rule(code: str) -> str:
    """Return the system-prompt rule for `code`, falling back to English."""
    return LANG_RULES.get(code, LANG_RULES["en"])


def recovery_msg(code: str) -> str:
    """Return the recovery message for `code`, falling back to English."""
    return RECOVERY_MSGS.get(code, RECOVERY_MSGS["en"])


def display_name(code: str, native: bool = True) -> str:
    """Return the language's display name (native by default)."""
    pair = LANG_DISPLAY.get(code)
    if not pair:
        return code.upper()
    return pair[0] if native else pair[1]


def is_supported(code: str) -> bool:
    """True if `code` is one of the 16 active app languages."""
    return code in SUPPORTED_LANGS

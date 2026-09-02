#!/usr/bin/env python3
"""
assistant/run_eval.py — Q&A evaluation runner over the POI-aware index.

Loads indexes/{destination}/{lang}.json (a committed fixture copy of the
sibling pipeline's build) and runs each question in
eval/ubeda/questions.json through litellm tool calling.

Six tools are exposed to the model:

    get_section(section_id, sort, limit)
        List the POIs inside one section, sorted by (interest_level,
        zoom_level) by default.  Returns id + name + 1-line preview.

    get_poi(poi_id)
        Full record of one POI by ID.  All fields, all paragraphs,
        no truncation, no line slicing.

    find_poi_by_name(query, limit)
        Fuzzy lookup against POI names.  Diacritic-insensitive.

    filter_pois(interest_level, type, tourist_type, section_id,
                indispensable, limit)
        Facet query.  Combine multiple filters with AND.

    list_sections()
        Section catalogue with deterministic 1-line summaries.
        Embedded into the system prompt at startup, so the model
        rarely needs to call it explicitly.

Usage:
    .venv/bin/python assistant/run_eval.py
    .venv/bin/python assistant/run_eval.py --model openai/gemma-4-E2B-it-MLX-8bit
    .venv/bin/python assistant/run_eval.py --lang es \
        --questions eval/ubeda/questions_es.json --index indexes/ubeda/es.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import litellm
litellm.drop_params = True
litellm.set_verbose = False

# Make package imports work whether you run as a script or module
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from index_tools import (
    load_index,
    format_sections_overview,
    format_section,
    format_poi,
    format_find_poi_by_name,
    format_filter_pois,
    format_search_pois,
    format_search_trips,
    format_trip,
    format_search_paths,
    format_path,
    find_poi_by_name as ix_find_poi_by_name,
    filter_pois as ix_filter_pois,
    search_pois as ix_search_pois,
    search_trips as ix_search_trips,
    search_paths as ix_search_paths,
    get_trip as ix_get_trip,
    get_path as ix_get_path,
    find_section,
    get_poi as ix_get_poi,
    extract_poi_tags,
    format_history_followup,
    format_trip_choice_offer,
    format_weather,
    get_weather_entry,
    is_forecast_too_hot,
    load_weather,
    resolve_history_selection,
    resolve_trip_query,
    sanitize_tourist_answer,
    weather_hint,
)
from common.lang_support import (
    SUPPORTED_LANGS,
    lang_rule,
    recovery_msg,
    recovery_msg as _recovery_msg,
    is_supported,
)
from common.models import DEFAULT_EVAL_MODEL
from common.textnorm import normalize_text, tokenize

# ── Constants ───────────────────────────────────────────────────────────────────────
QUESTIONS_FILE  = PROJECT_ROOT / "eval" / "ubeda" / "questions.json"
DEFAULT_INDEX   = PROJECT_ROOT / "indexes" / "ubeda" / "en.json"
RESULTS_DIR     = PROJECT_ROOT / "results"
DEFAULT_MODEL   = DEFAULT_EVAL_MODEL   # oMLX E2B; the mobile deployment target
MAX_TOOL_ROUNDS = 14
# Answers run a few hundred tokens; the cap bounds worst-case latency and
# stops a degenerating non-streaming response from burning the
# server-side default budget (the streaming path has the chant guard).
MAX_ANSWER_TOKENS = 1_024
MAX_TOOL_RESULT_CHARS = 24_000
MAX_TOOL_HISTORY_CHARS = 120_000
COMPACTED_TOOL_RESULT = (
    "[Earlier tool result omitted to preserve E2B context. "
    "Retrieve the source again when needed.]"
)
NO_DIRECT_EVIDENCE_PREFIX = "No place record explicitly mentions all of:"
COMPLEMENTARY_SEARCH_INSTRUCTION = (
    "The direct evidence search did not find one place that combines all "
    "requested concepts. Do not ask the visitor a follow-up question. In "
    "this same turn, retrieve each concept separately with search_pois, "
    "filter_pois, or get_section, then give clearly labelled complementary "
    "options. State that the available visitor information does not confirm "
    "the combination; do not imply that separate places satisfy it."
)
ROUTE_INTENT_TERMS = frozenset({
    # English
    "route", "walking", "walk", "cycling", "cycle", "bicycle", "bike",
    "trail", "track", "hiking", "hike", "trek",
    # Spanish / Catalan / Galician / Basque
    "ruta", "caminar", "caminata", "sendero", "senderismo", "bicicleta",
    "ciclismo", "paseo", "camin", "bici", "ibilbide",
    # Italian / French / Portuguese
    "percorso", "piedi", "sentiero", "camminata", "bici", "bicicletta",
    "randonnée", "randonnee", "velo", "ciclovia", "caminho",
    # German / Dutch / Croatian
    "wander", "wanderweg", "fahrrad", "radweg", "spaziergang", "route",
    "wandeling", "fiets", "staza", "pjesa", "setnja",
})
ROUTE_SEARCH_INSTRUCTION = (
    "This is a physical walking, cycling, trail, track, or route request. "
    "You must call search_paths now. Do not ask the visitor to clarify "
    "before checking the available physical routes. Never substitute a "
    "curated trip for a physical route."
)
NO_PATH_ANSWER_INSTRUCTION = (
    "The visitor information has no matching physical path. State that "
    "clearly and concisely now. Do not ask a follow-up question and do not "
    "substitute a curated trip as though it were a route."
)
# Used instead of NO_PATH_ANSWER_INSTRUCTION when get_weather already ran
# this turn: the lack of a cataloged route must never make the model drop
# the forecast it already has. This is an internal, English-only model
# directive (like every other *_INSTRUCTION constant here) — it is never
# shown to the visitor, so it needs no per-language translation.
NO_PATH_ANSWER_WITH_WEATHER_INSTRUCTION = (
    "The visitor information has no matching physical path, but you "
    "already retrieved the forecast this turn. Answer using BOTH facts: "
    "first state whether the weather is good for walking outdoors based "
    "only on the forecast already retrieved, then separately note that no "
    "specific route is available in the catalog. The lack of a cataloged "
    "route does not change the weather judgment. Do not ask a follow-up "
    "question."
)


def no_path_answer_instruction(weather_grounded: bool) -> str:
    """Pick the no-path instruction that fits whether weather already ran."""
    return (NO_PATH_ANSWER_WITH_WEATHER_INSTRUCTION if weather_grounded
            else NO_PATH_ANSWER_INSTRUCTION)


# ── Tool-call loop detection ─────────────────────────────────────────────
# Small models sometimes re-issue the identical tool call over and over
# (the 26B Spanish eval showed single questions looping until the round
# cap, e.g. 1000 s outliers vs a ~20 s median).  Every tool here is a
# deterministic read-only lookup, so an identical (tool, args) call can
# never return new information — repeat detection has no false-positive
# window beyond legitimate cross-turn re-asks, which is why the state
# resets per question/turn.
LOOP_CALL_LIMIT = 3
LOOP_REPEAT_STUB = (
    "[Repeated call blocked: you already received this exact result "
    "earlier in this turn.]"
)
LOOP_REPEAT_INSTRUCTION = (
    "You already called {tool} with the exact same arguments earlier in "
    "this turn and received that result. Repeating the call returns "
    "nothing new. Do not repeat it. Answer the visitor's question now "
    "from the information you already have, or use a different tool."
)
# Served for the SECOND occurrence of an identical call (before the
# blocking stub above applies to the third): the tools are deterministic
# lookups, so the result already in context is the whole answer.
LOOP_REPEAT_CACHE_STUB = (
    "[Already retrieved this turn: the identical result is in this "
    "conversation above. Use it; repeating the call returns nothing new.]"
)


def tool_call_key(name: str, args: dict) -> str:
    """Canonical identity of one tool call (argument order irrelevant)."""
    return name + "|" + json.dumps(args or {}, sort_keys=True, default=str)


def is_repeat_tool_call(call_keys: list[str],
                        limit: int = LOOP_CALL_LIMIT) -> bool:
    """True when the latest key in `call_keys` has appeared `limit` times.

    Count-based, so consecutive, alternating (A-B-A-B-A) and scattered
    repeats of the same (tool, args) all trip it.  Only model-emitted
    calls are tracked; deterministic runtime lookups are excluded.
    """
    return bool(call_keys) and call_keys.count(call_keys[-1]) >= limit


# ── Streaming content-chant detection ─────────────────────────────────────
# Text-level counterpart of the tool-call loop detector: a small model can
# degenerate into repeating one phrase until the server-side token cap
# (gemini-cli calls this "content chanting").  Streaming only — the eval
# path is non-streaming and bounded by the server.
CHANT_CHUNK_SIZE = 50        # tail chunk compared for repetition
CHANT_WINDOW_CHARS = 2_000   # only the recent window is inspected
CHANT_MAX_REPEATS = 6        # chunk occurrences in the window that trip it


def chant_repeat_prefix(text: str,
                        chunk_size: int = CHANT_CHUNK_SIZE,
                        window_chars: int = CHANT_WINDOW_CHARS,
                        max_repeats: int = CHANT_MAX_REPEATS) -> int:
    """Offset where a repetitive chant run begins, or len(text) if none.

    Detection (gemini-cli style, tightened for short chat answers): the
    last `chunk_size` characters occurring at least `max_repeats` times in
    the trailing `window_chars` window means the stream is chanting.  On
    detection the offset walks back over every earlier occurrence of the
    tail chunk, so text[:offset] is the keepable non-repetitive prefix
    (it may retain up to one partial chant unit — the run start is not
    chunk-aligned in general).
    """
    if len(text) < chunk_size * max_repeats:
        return len(text)
    chunk = text[-chunk_size:]
    if text[-window_chars:].count(chunk) < max_repeats:
        return len(text)
    start = text.find(chunk, max(0, len(text) - window_chars))
    while True:
        prev = text.rfind(chunk, 0, start)
        if prev == -1:
            return start
        start = prev


SOURCE_GROUNDING_TOOLS = frozenset({
    "get_poi", "get_section", "find_poi_by_name", "filter_pois",
    "search_pois", "search_trips", "get_trip", "search_paths", "get_path",
    "get_weather",
})
GROUNDING_RECOVERY_INSTRUCTION = (
    "Look at the visitor's message once more.\n"
    "If it is a short confirmation (\"yes\", \"sí\", \"ok\", \"go ahead\") "
    "answering a question or offer YOU made in your previous message, it is "
    "NOT small talk: act on your own offer now — call find_poi_by_name for "
    "each place you just named (or the tool matching what you offered) and "
    "answer from the retrieved records. Never answer a confirmation with "
    "another \"what would you like to know?\" question.\n"
    "If it is just a greeting, thanks, or small talk, respond warmly and "
    "briefly in the visitor's language and offer your help — no tool call "
    "is needed.\n"
    "If it is a broad overview question that names no specific place, "
    "fact, or date — such as \"What can I see?\" or \"What is there to "
    "do?\" — write the answer right now, in the visitor's language, using "
    "only the destination overview and the section catalogue from the "
    "system prompt: summarize the highlights and mention the most "
    "interesting sections. Never answer a broad overview question with a "
    "failure message.\n"
    "If the question names a specific place, fact, date, or listing, call "
    "one of get_section, filter_pois, search_pois, or find_poi_by_name "
    "now instead of answering from memory — when the question names a "
    "place, the right call is find_poi_by_name with the place name.\n"
    "Never promise to search later, and never ask the visitor to repeat "
    "or choose; reply with the visitor-facing answer or a tool call, "
    "nothing else."
)
# Localized preamble used when the runtime deterministically answers a
# follow-up question from the POIs already shown in the recent history
# (safety net for small models that decline to call tools on broad
# follow-ups).  Visitor-facing, so it stays an i18n table; coverage for
# every supported language is guarded by tests/test_i18n.py.
HISTORY_FOLLOWUP_LEADS: dict[str, str] = {
    "ca": "D'entre els llocs que ja t'he esmentat, aquests encaixen amb la teva pregunta:",
    "de": "Von den bereits genannten Orten passen diese zu Ihrer Frage:",
    "en": "Based on the places I already mentioned, these match your question:",
    "es": "De los lugares que ya te he mencionado, estos encajan con tu pregunta:",
    "eu": "Jada aipatutako lekuen artean, hauek datoz bat zure galderarekin:",
    "fr": "Parmi les lieux déjà mentionnés, voici ceux qui correspondent à votre demande :",
    "gl": "Dos lugares que xa mencionei, estes encaixan coa túa pregunta:",
    "hi": "जिन जगहों का मैंने पहले उल्लेख किया है, उनमें से ये आपके प्रश्न से मेल खाती हैं:",
    "hr": "Od mjesta koja sam već spomenuo, ova odgovaraju vašem pitanju:",
    "it": "Fra i luoghi che ho già menzionato, questi corrispondono alla tua richiesta:",
    "ja": "すでにご紹介した場所の中で、ご質問に合うのはこちらです:",
    "nl": "Van de plaatsen die ik al noemde, passen deze bij je vraag:",
    "pt": "Entre os locais já mencionados, estes correspondem à sua pergunta:",
    "ru": "Из уже упомянутых мест вашему вопросу соответствуют:",
    "uk": "З місць, які я вже згадував, вашому запитанню відповідають:",
    "zh": "在我已经提到的地点中，以下符合您的问题：",
}


def history_followup_lead(index: dict) -> str:
    lang = (index.get("meta") or {}).get("lang") or "en"
    return HISTORY_FOLLOWUP_LEADS.get(lang, HISTORY_FOLLOWUP_LEADS["en"])


GROUNDING_FAILURE_MESSAGES = {
    "ca": "No he pogut recuperar informació turística verificada de les dades descarregades.",
    "de": "Ich konnte keine verifizierten touristischen Informationen aus den heruntergeladenen Daten abrufen.",
    "en": "I could not retrieve verified visitor information from the downloaded data.",
    "es": "No he podido recuperar información turística verificada de los datos descargados.",
    "eu": "Ezin izan dut deskargatutako datuetatik turismo-informazio egiaztatua berreskuratu.",
    "fr": "Je n’ai pas pu récupérer d’informations touristiques vérifiées depuis les données téléchargées.",
    "gl": "Non puiden recuperar información turística verificada dos datos descargados.",
    "hi": "मैं डाउनलोड किए गए डेटा से सत्यापित पर्यटन जानकारी प्राप्त नहीं कर सका।",
    "hr": "Nisam uspio dohvatiti provjerene turističke informacije iz preuzetih podataka.",
    "it": "Non sono riuscito a recuperare informazioni turistiche verificate dai dati scaricati.",
    "ja": "ダウンロード済みデータから検証済みの観光情報を取得できませんでした。",
    "nl": "Ik kon geen geverifieerde toeristische informatie uit de gedownloade gegevens ophalen.",
    "pt": "Não consegui recuperar informações turísticas verificadas dos dados transferidos.",
    "ru": "Не удалось получить проверенную туристическую информацию из загруженных данных.",
    "uk": "Не вдалося отримати перевірену туристичну інформацію із завантажених даних.",
    "zh": "无法从已下载的数据中获取经过验证的旅游信息。",
}
RENTAL_INTENT_TERMS = frozenset({
    # English
    "rent", "rental", "hire",
    # Spanish / Catalan / Galician / Basque
    "alquilar", "alquiler", "lloguer", "alugar",
    # Italian / French / Portuguese
    "noleggiare", "noleggio", "louer", "location", "aluguer",
    # German / Dutch
    "mieten", "verleih", "huren",
})


def is_physical_route_request(question: str) -> bool:
    """Detect a physical-route request across the supported app languages.

    A bare transport-mode word ("bicicleta", "walking") is too weak a
    signal on its own when the visitor is really asking about renting or
    buying equipment rather than requesting a cataloged path — e.g. "can I
    rent a bike here?" is not a route request even though it names a
    transport mode.
    """
    terms = tokenize(question)
    term_set = set(terms)
    if term_set & RENTAL_INTENT_TERMS:
        return False
    return any(
        term in ROUTE_INTENT_TERMS
        or any(
            len(term) >= 4 and term.startswith(route)
            for route in ROUTE_INTENT_TERMS if len(route) >= 4
        )
        for term in terms
    )


WEATHER_INTENT_TERMS = frozenset({
    # English
    "weather", "forecast", "temperature", "temp", "rain", "sunny",
    "cloudy", "windy", "storm", "cold", "hot",
    # Spanish / Catalan / Galician / Basque
    "tiempo", "clima", "temperatura", "lluvia", "soleado", "nublado",
    "tormenta", "calor", "frío", "frio", "eguraldi",
    # Italian / French / Portuguese
    "meteo", "tempo", "previsione", "pioggia", "soleggiato", "nuvoloso",
    "caldo", "freddo", "météo", "meteo", "température", "pluie",
    "ensoleillé", "nuageux", "chaud", "froid", "tempo", "chuva",
    "ensolarado", "nublado", "quente", "frio",
    # German / Dutch / Croatian / Russian / Ukrainian / Japanese / Chinese
    "wetter", "vorhersage", "temperatur", "regen", "sonnig", "bewölkt",
    "heiß", "kalt", "weer", "weersvoorspelling", "regen", "zonnig",
    "vrijeme", "prognoza", "kiša", "sunčano", "vruće", "hladno",
    "погода", "прогноз", "температура", "天気", "天气", "预报",
})


def is_weather_request(question: str) -> bool:
    """Detect a weather-intent visitor turn across the supported languages.

    Uses the same prefix-tolerant match as ROUTE_INTENT_TERMS so morphological
    variants ("weathered", "tempesta") do not trigger false positives.
    """
    terms = tokenize(question)
    return any(
        term in WEATHER_INTENT_TERMS
        or any(
            len(term) >= 4 and term.startswith(intent)
            for intent in WEATHER_INTENT_TERMS if len(intent) >= 4
        )
        for term in terms
    )


WEATHER_LOOKUP_ENFORCED_INSTRUCTION = (
    "A weather lookup has already completed. Use these forecast results to "
    "answer the visitor; do not add outside knowledge about the weather."
)

# English-only, model-facing note appended to a get_weather(day=...) tool
# result when that specific day exceeds the outdoor-plan heat threshold.
# Deliberately not localized: it is read by the model (which already
# renders its final answer in the visitor's language per {lang_rule}),
# never shown to the visitor verbatim.
OUTDOOR_HEAT_NOTE = (
    "Note: exceeds 35 \u00b0C \u2014 treat as unfavourable for a midday walk; "
    "suggest an early morning visit or an indoor plan instead."
)


TRIP_PLAN_INTENT_TERMS = frozenset({
    # English
    "plan", "itinerary", "itiner", "day", "days", "weekend",
    # Spanish / Catalan / Galician / Basque
    # NOTE: "visita"/"visitar" is deliberately excluded — it is too
    # generic a verb (used in ordinary "what can I visit" questions) to
    # reliably signal an itinerary/plan request, and previously forced a
    # curated-trip offer instead of real retrieval on plain sightseeing
    # questions.
    "plan", "itinerario", "recorrido", "dia", "dias", "semana",
    "detalle", "detalles",
    # Italian / French / Portuguese
    "piano", "itinerario", "giorno", "giorni", "fine", "settimana",
    "programme", "jour", "jours", "semaine", "roteiro", "dia", "dias",
    # German / Dutch / Croatian
    "plan", "tag", "tage", "wochenende", "reiseroute", "dagen",
    "weekend", "itinerar", "dan", "dana",
})
TRIP_DETAIL_REQUIRED_INSTRUCTION = (
    "The visitor asked for a plan or itinerary. You have source trip "
    "suggestions, but you must not invent a new option or combine stops "
    "from several trips. Choose the most suitable returned trip and call "
    "get_trip now. Base the final plan only on that retrieved trip detail."
)
def requires_current_turn_grounding(question: str) -> bool:
    """True for every non-empty visitor message.

    There is deliberately no hardcoded greeting/social list to translate:
    greetings and small talk go through the same recovery path as any
    other message, where GROUNDING_RECOVERY_INSTRUCTION's small-talk
    branch lets the model answer them in the visitor's language.  The
    runtime stays language-agnostic — any language the LLM understands
    works, not just the ones a keyword list covers.
    """
    return bool(normalize_text(question))


def requires_trip_detail(question: str) -> bool:
    """True when a visitor requests a concrete curated plan/detail."""
    terms = tokenize(question)
    return any(
        term in TRIP_PLAN_INTENT_TERMS
        or any(
            len(term) >= 4 and term.startswith(intent)
            for intent in TRIP_PLAN_INTENT_TERMS if len(intent) >= 4
        )
        for term in terms
    )


def grounding_failure_message(index: dict) -> str:
    """Return a localized safe failure instead of ungrounded tourism prose."""
    lang = (index.get("meta") or {}).get("lang") or "en"
    return GROUNDING_FAILURE_MESSAGES.get(lang, GROUNDING_FAILURE_MESSAGES["en"])


def question_names_known_poi(question: str, index: dict) -> str | None:
    """Return the `name_index` entry the question explicitly names, if any.

    Language-agnostic 'specific question' probe: no keyword lists to
    translate — the destination's own POI names are the lexicon.  Used as
    the last-resort backstop when a small model twice declines to
    retrieve on a question that names a known place (e.g. it answers
    \"tell me if you want details\" instead of fetching the castle).
    """
    normalized = normalize_text(question)
    if not normalized:
        return None
    best = None
    for name in (index.get("name_index") or {}):
        if " " not in name and len(name) < 8:
            continue  # short single words match too casually
        if name in normalized and (best is None or len(name) > len(best)):
            best = name
    return best


NAMED_POI_LOOKUP_INSTRUCTION = (
    "The visitor is asking about this known place (named by them, or "
    "offered by you and confirmed with a yes), whose record has just been "
    "retrieved. Answer from this record now, in the visitor's language — "
    "do not promise to search later and do not ask the visitor to repeat "
    "or choose."
)


def _current_question_index(messages: list[dict], question: str) -> int:
    """Index of the current visitor question inside `messages` (or -1)."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and (msg.get("content") or "") == question:
            return i
    return -1


def _previous_offer_poi_names(messages: list[dict], question: str,
                              index: dict, limit: int = 3) -> list[str]:
    """Known-place names inside the assistant's pre-question offer.

    When the assistant ended its previous turn by offering places
    (“¿Te gustaría saber más sobre X o Y?”) and the visitor replies with
    a short confirmation, the reply itself names nothing — the named
    places live in the assistant's own previous message.  Probe that
    message against the destination's name_index so the backstop can
    fetch what was offered.  Language-agnostic by design: no affirmation
    word list — the gate is a short visitor reply to an assistant
    message that ends with a question mark.
    """
    if len(tokenize(question)) > 5:
        return []
    prior_end = _current_question_index(messages, question)
    prior = messages[:prior_end] if prior_end >= 0 else messages
    offer = ""
    for msg in reversed(prior):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            offer = (msg["content"] or "").strip()
            break
    if not offer.endswith("?"):
        return []
    normalized = normalize_text(offer)
    matches: list[str] = []
    for name in (index.get("name_index") or {}):
        if " " not in name and len(name) < 8:
            continue  # short single words match too casually
        if name in normalized and name not in matches:
            matches.append(name)
    matches.sort(key=len, reverse=True)
    return matches[:limit]


def backstop_named_pois(question: str, messages: list[dict],
                        index: dict) -> list[str]:
    """Known-place names to fetch when the model twice declined retrieval.

    Primary: the visitor's question explicitly names a known place.
    Secondary: the visitor gave a short confirmation to an assistant offer
    that named known places (the names live in the assistant's message,
    not in the visitor's reply).
    """
    named = question_names_known_poi(question, index)
    if named:
        return [named]
    return _previous_offer_poi_names(messages, question, index)


def is_pure_reask(text: str) -> bool:
    """True when the model's answer is only another clarifying question.

    A helpful tourism turn must carry content; a bare “¿Qué te gustaría
    saber ahora?” after the recovery prompt is a brush-off.  Tagged or
    long answers, and warm greeting-style replies (which use exclamation
    marks in the app languages), count as real answers, not re-asks.
    """
    t = (text or "").strip()
    if not t or len(t) > 300:
        return False
    if "<poi" in t or "<trip" in t:
        return False
    if "!" in t or "¡" in t or "！" in t:
        return False
    return t.endswith("?") or t.endswith("？")


def is_repeat_of_previous_answer(answer: str, messages: list[dict],
                                 question: str = "") -> bool:
    """True when `answer` duplicates the assistant's previous VISITOR turn.

    A small model at temperature=0 can re-emit the reply the visitor last
    saw instead of answering the new message (observed: "Hola" followed
    by a broad question returns the greeting verbatim); serving it is a
    brush-off, exactly like a repeated clarifying question.  Exact match
    after normalisation — near-misses stay the model's answer.

    The referent is the last assistant message BEFORE the current
    `question` (current-turn drafts from earlier rounds of this same
    turn must not count).  When `question` is not found in `messages`,
    falls back to the last assistant message overall.
    """
    candidate = normalize_text(answer or "")
    if not candidate:
        return False
    if question:
        q_idx = _current_question_index(messages, question)
        if q_idx >= 0:
            messages = messages[:q_idx]
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            return normalize_text(msg["content"]) == candidate
    return False


REASK_FALLBACK_INSTRUCTION = (
    "Answering with another clarifying question is not acceptable — the "
    "visitor already engaged and expects real content. Below are visitor "
    "records retrieved now. Present them warmly in the visitor's language, "
    "tagging each place, and invite the visitor to pick one for more "
    "detail. If the result says nothing matched, answer from the "
    "destination overview and the section catalogue instead."
)


def execute_reask_fallback(question: str, index: dict, sections_text: str,
                           cache: dict,
                           weather: dict | None = None) -> tuple[str, bool, str, dict]:
    """Deterministically retrieve content after a repeated re-ask.

    A topical question (any token with 3+ chars) gets a same-record
    evidence search first; a content-free message ("sí", "ok") or a
    search miss falls back to the destination's indispensable highlights.
    Returns (result_text, cache_hit, tool_name, args).
    """
    if any(len(tok) >= 3 for tok in tokenize(question)):
        args = {"query": question, "limit": 5}
        result, hit = execute_tool(
            "search_pois", args, index, sections_text, cache, weather=weather,
        )
        if not result.startswith(NO_DIRECT_EVIDENCE_PREFIX):
            return result, hit, "search_pois", args
    args = {"interest_level": 1, "limit": 10}
    result, hit = execute_tool(
        "filter_pois", args, index, sections_text, cache, weather=weather,
    )
    return result, hit, "filter_pois", args


def reask_fallback_applies(messages: list[dict], question: str) -> bool:
    """Gate the re-ask fallback to ongoing conversations only.

    First-turn messages (greetings like “Hola”, or every single-question
    eval run) keep the trust-the-model path so greeting tone and eval
    scores are unaffected; from the second visitor message on, a repeated
    clarifying question is intercepted with deterministic retrieval.
    """
    q_idx = _current_question_index(messages, question)
    prior = messages[:q_idx] if q_idx >= 0 else messages
    return any(m.get("role") == "user" for m in prior)


def history_followup_answer(question: str, messages: list[dict],
                            index: dict, limit: int = 6) -> str:
    """Build a deterministic follow-up answer, or empty string on miss.

    Uses only POIs already tagged in the most recent assistant turn so
    the safety net never surfaces new places the visitor has not seen.
    """
    body = format_history_followup(index, question, messages, limit=limit)
    if not body:
        return ""
    return f"{history_followup_lead(index)}\n\n{body}"


def selected_source_context(selection: dict, result: str) -> str:
    """Build internal context after a validated prior-tag selection."""
    return (
        f"The visitor selected this previously shown {selection['kind']}. "
        "Use this freshly retrieved source record to answer, including "
        "available ordered stops. Do not answer from earlier assistant "
        f"paraphrase:\n\n{result}"
    )


def route_lookup_context(result: str) -> str:
    """Return an internal instruction after a forced physical path lookup."""
    if result.startswith("No curated routes matched"):
        return (
            "A physical-route lookup has already completed and found no "
            "matching published route. Answer the visitor clearly now. Do "
            "not ask a follow-up question and do not substitute a trip."
        )
    return (
        "A physical-route lookup has already completed. Use these route "
        "results to answer the visitor; do not offer a curated trip as a "
        "physical route:\n\n" + result
    )
_SYSTEM_PROMPT_TEMPLATE = """\
You are a tourism assistant for {destination}.  You answer visitor \
questions using the {destination} POI index, which is a structured catalogue \
of every point of interest, trip and itinerary in the destination.

The full section catalogue is listed below — you do NOT need to call any \
tool to discover it.  Use this information directly.  Generic overview \
questions (\"What can I see?\", \"What is there to do?\") that name no \
specific place, fact, or date can and should be answered from the \
destination overview and this catalogue alone: present the highlights \
and point the visitor to the most interesting sections.

You have ELEVEN tools. Pick the one that fits the question:

  • get_section(section_id, sort?, limit?)
        List POIs inside one section.  Returns id + name + a one-line preview.
        Use when the user asks "what X exist?", "list all Y in <category>".

  • get_poi(poi_id)
        Full record of one POI: type, address, phone, coordinates, links, \
AND the full description paragraph.
        Use when you need facts (address, phone, dates, description) about \
a specific named POI.  Pass several comma-separated ids \
('poi/123,poi/456') to fetch multiple POIs in one call when comparing \
or synthesising.

  • find_poi_by_name(query, limit?, detail?)
        Fuzzy lookup by POI name.  Returns up to `limit` candidates with id + \
section + preview.  Use when the user names a place but you don't know \
which section it lives in.  Pass detail="full" to also get the best match's \
complete record in the same call; with the default detail="brief", always \
follow up with get_poi() on the best match before answering specific facts.

  • filter_pois(interest_level?, type?, tourist_type?, section_id?, \
indispensable?, limit?)
        Facet query.  All filters AND together.  Examples:
          - filter_pois(indispensable=true) → must-see POIs
          - filter_pois(tourist_type="FOOD TOURISM", limit=10) → food spots
          - filter_pois(type="OilMill") → all olive-oil mills / producers
          - filter_pois(type="Restaurant", section_id="gastronomy") → restaurants
          - filter_pois(interest_level=1, section_id="religious-heritage")
        Use the UNE type codes you see in tool results (e.g. Restaurant, \
OilMill, Museum); do not guess codes from the user's words.
  • search_pois(query, section_id?, limit?)
        Evidence search across POI names and visitor descriptions. Use it to \
check whether several visitor concepts appear on the SAME place, for example \
"olive oil restaurant", "family museum", or "accessible parking". Results \
include the supporting text. Use a short concept phrase, not the entire \
visitor question.

  • search_trips(query, limit?)
        Search curated suggestions for what to do over a theme, day, or
        multi-day visit. These are editorial plans, NOT walking/biking
        routes. Use for "what should I do for two days?", themed visits,
        or curated itinerary suggestions.

  • get_trip(trip_id)
        Return the full ordered stops and description of one curated trip
        suggestion. Use after search_trips().

  • search_paths(query, limit?)
        Search physical routes fetched from the destination's /paths data.
        Use ONLY for walking, cycling, trail, track, or route requests.
        Never use a trip as a substitute for a physical route.

  • get_path(path_id)
        Return the full ordered waypoint stops of one physical route.
        Use after search_paths().

  • get_weather(day?)
        Return the downloaded 7-day forecast (or one day: `today`, \
`tomorrow`, an ISO date `YYYY-MM-DD`, or a weekday name). Use it for \
any outdoor/day-plan question. Never invent temperatures or conditions.

  • list_sections()
        Returns the catalogue below.  Rarely needed — sections are \
pre-loaded.

{weather_hint_block}--- DESTINATION OVERVIEW ---
{destination_overview}

--- SECTIONS (pre-loaded, do not fetch again) ---
{sections_text}
--- END SECTIONS ---

RULES:
- Answer based ONLY on what your tools return.  Do not use outside knowledge.
- Always include the description paragraph from get_poi() when answering \
about a specific place — it carries the most useful detail.
- Quote exact names, addresses, phones, coordinates, and dates when present.
- For "what should I not miss?" / "best of" questions, use \
filter_pois(indispensable=true) before browsing sections.
- For "tell me about <name>" / "what is <name>" questions, call \
find_poi_by_name() with detail="full" first — it returns the best match's \
full record in one call.
- For "what to do", themed visits, one-day, or multi-day plan requests,
  search curated trips first. For walking, cycling, trail, track, or
  route requests, search physical paths first. A curated trip is never a
  physical route; if no path is available, say so rather than substituting
  a trip. Do not ask the visitor to reformulate the route request.
- For outdoor, walking, or day-plan questions, call get_weather(day) \
before recommending. When the forecast is unfavourable (>35 °C, rain, \
storms), prefer indoor stops or early-morning windows. If the tool \
reports the forecast is unavailable, do not invent one. Judge whether \
the weather suits an outdoor activity ONLY from the forecast; whether a \
specific physical route exists in the catalog is a separate fact and \
must never change that judgment or replace it in your answer.
- After search_trips, do not invent named plans, option headings, or
  day-by-day stop combinations. Present only returned <trip> tags, or
  call get_trip for one returned source trip before describing its plan.
- After filter_pois: if the question needs a description, dates, \
address, phone, architect, or any per-POI detail beyond the name, \
call get_poi on the most relevant result before answering. \
For pure listing questions (e.g. "what hotels are there?", \
"list all museums"), the filter_pois previews already include name + \
type + interest level, so an extra get_poi call is unnecessary.
- If information is not in the index after trying all relevant tools, \
say so simply — do not repeat the question or summarise what you tried.
- When filter_pois returns no results, do not stop: try \
get_section("gastronomy") or another relevant section, or try a \
different UNE type code (e.g. for olive oil: OilMill, Restaurant), \
before concluding nothing was found.  Look at the type codes shown in \
section previews to pick a valid one.
- For any compound visitor request ("X with Y", "X near Y", "X that \
offers Y"), use search_pois() with the combined concepts first. Only say \
a place has BOTH properties when that same record explicitly supports \
both. If there is no direct match, still be helpful: retrieve each \
concept separately and present clearly-labelled complementary options. \
Say the available visitor information does not confirm the combination; \
never imply that separate options satisfy it. Do this in the SAME turn: \
do not stop, do not ask the visitor which search they prefer, and do not \
make them repeat their request.
- Never expose internal details in your answers: no type codes, no \
tool names, no filter parameter names, no catalog terminology, and no \
raw IDs. Speak like a local tourism host, not a database interface.
- Tag every point of interest you mention.  The tag WRAPS the name \
(the name goes BETWEEN opening and closing tag): \
  CORRECT: <poi id=0 type=TouristAttraction>Example Landmark</poi> \
  WRONG:   Example Landmark (<poi id=0 type=TouristAttraction>) \
  WRONG:   <poi id=0 type=TouristAttraction> Example Landmark \
Bare numeric id, no quotes.  Do NOT write the type or interest level \
in the answer prose — they belong only in the tag attribute.  Never \
show raw 'poi/…' ids outside a tag.
- OUTPUT VALIDITY REQUIREMENT: when a tool gives you a POI tag, copy that \
exact tag around every mention of that POI in the final answer. A plain \
POI name without its <poi ...> tag is invalid output because the mobile \
app cannot open it. Do not replace a tagged name with an untagged \
paraphrase. Only use an id that appeared in a tool result; NEVER invent \
or guess an id. If no tool has provided an id for a name, leave that \
name untagged rather than creating a tag.
- {{lang_rule}}
"""


def make_system_prompt(sections_text: str, destination: str,
                       destination_overview: str, lang: str = "en",
                       weather_hint_text: str = "") -> str:
    """Build the system prompt with sections, overview, and (optional)
    today’s weather hint embedded.
    """
    overview = destination_overview.strip() or "(no overview available)"
    hint = weather_hint_text.strip()
    weather_hint_block = f"--- WEATHER HINT ---\n{hint}\n\n" if hint else ""
    return _SYSTEM_PROMPT_TEMPLATE.replace("{{lang_rule}}", lang_rule(lang)).format(
        sections_text=sections_text,
        destination=destination,
        destination_overview=overview,
        weather_hint_block=weather_hint_block,
    )



# ── Tool definitions exposed to the LLM ─────────────────────────────────────

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": (
                "List the POIs inside one section.  Returns id + name + "
                "a one-line preview for each POI.  Pass the section_id "
                "from the catalogue above (preferred) OR the exact title."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "Section id or title.",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["interest", "name", "zoom"],
                        "description": "Sort order; default 'interest' (most important first).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": ("Max POIs to return (default 50; "
                                        "20 for grouped sections; hard maximum 50)."),
                    },
                },
                "required": ["section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_poi",
            "description": (
                "Return the full record of one POI: address, phone, "
                "coordinates, links, AND the full description paragraph.  "
                "Pass the full id ('poi/12345'), the bare number, or "
                "up to five comma-separated ids ('poi/123,poi/456') to "
                "fetch multiple POIs in one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_id": {
                        "type": "string",
                        "description": ("POI id(s): 'poi/5155', '5155', or "
                                        "comma-separated 'poi/123,poi/456'."),
                    },
                },
                "required": ["poi_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_poi_by_name",
            "description": (
                "Fuzzy POI name lookup.  Diacritic-insensitive.  Returns "
                "id + name + section + preview for up to five matches.  "
                "With detail=\"full\" the best match's complete record is "
                "included in the same response — no follow-up get_poi() "
                "needed.  With the default \"brief\", follow up with "
                "get_poi() on the best match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-form POI name to search for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default and maximum 5).",
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["brief", "full"],
                        "description": ("'full' also returns the best match's "
                                        "complete POI record (default 'brief')."),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_pois",
            "description": (
                "Facet query.  All filters AND together.  Use for "
                "'indispensable POIs', 'all OilMills', 'food-tourism POIs in "
                "<section>', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "interest_level": {
                        "type": "integer",
                        "description": "1=Indispensable, 2=Interesting, 3=Outstanding.",
                    },
                    "type": {
                        "type": "string",
                        "description": "UNE 178503 type code, e.g. 'OilMill', 'Museum'.",
                    },
                    "tourist_type": {
                        "type": "string",
                        "description": "Tourist-type code, e.g. 'FOOD TOURISM', 'HERITAGE TOURISM'.",
                    },
                    "section_id": {
                        "type": "string",
                        "description": "Restrict to a section.",
                    },
                    "indispensable": {
                        "type": "boolean",
                        "description": "Shortcut for interest_level=1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max POIs to return (default and maximum 20).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pois",
            "description": (
                "Search POI names and visitor descriptions for explicit "
                "evidence that every word in a concise query applies to "
                "the same place. Use before claiming that a place combines "
                "multiple visitor needs. Results include supporting text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Short evidence phrase, e.g. 'olive oil "
                            "restaurant' or 'accessible parking'."
                        ),
                    },
                    "section_id": {
                        "type": "string",
                        "description": "Optional section to search within.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trips",
            "description": (
                "Search curated suggestions for what to do over a theme, "
                "day, or multi-day visit. These are editorial suggestions, "
                "not physical walking or biking routes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short trip theme or visit goal.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum suggestions (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trip",
            "description": (
                "Return one curated trip suggestion with its ordered stops. "
                "Use only after search_trips."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trip_id": {
                        "type": "string",
                        "description": "Trip id, e.g. 'trip/4407' or '4407'.",
                    },
                },
                "required": ["trip_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_paths",
            "description": (
                "Search physical walking, cycling, trail, track, or route "
                "records fetched from /v120/paths. Never returns a trip."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short route, activity, or trail query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum routes (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_path",
            "description": (
                "Return one physical route from /v120/paths with ordered "
                "waypoint stops. Use only after search_paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path_id": {
                        "type": "string",
                        "description": "Path id, e.g. 'path/123' or '123'.",
                    },
                },
                "required": ["path_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Return the offline weather forecast downloaded with the "
                "visitor data. Pass no day to see the full 7-day outlook, "
                "or a specific day: 'today', 'tomorrow', an ISO date "
                "'YYYY-MM-DD', or a weekday name (monday..sunday, or the "
                "localized variants). Call this before recommending any "
                "outdoor plan or day-by-day itinerary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": (
                            "Optional day selector: 'today', 'tomorrow', "
                            "'YYYY-MM-DD', or a weekday name."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sections",
            "description": (
                "Return the section catalogue.  Already embedded in your "
                "system prompt — call only as a refresher."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ── Tool-call validation feedback ─────────────────────────────────────────
# Validate each call against its schema BEFORE execution and return the
# problem as the tool result (the gemini-cli scheduler pattern): a small
# model then re-issues a corrected call on the next round instead of
# silently getting a misleading answer from defaulted arguments.
_TOOL_SCHEMAS = {d["function"]["name"]: d["function"]["parameters"]
                 for d in TOOL_DEFS}


def validate_tool_call(name: str, args: dict | None,
                       raw_arguments: str = "") -> str | None:
    """Return an error string for an invalid tool call, else None.

    The error text becomes the tool result, so the model sees exactly
    what was wrong.  Checks: known tool, parseable JSON (args is None
    when the model's arguments string was not valid JSON), required
    arguments present, enum values, and integer/boolean types.
    """
    if name not in _TOOL_SCHEMAS:
        valid = ", ".join(sorted(_TOOL_SCHEMAS))
        return (f"[ERROR] Unknown tool: {name}. "
                f"Available tools: {valid}.")
    if args is None:
        excerpt = (raw_arguments or "")[:200]
        return (f"[ERROR] {name}: invalid JSON in arguments "
                f"(received: {excerpt!r}). Re-issue the call with valid "
                f"JSON matching the tool schema.")
    schema = _TOOL_SCHEMAS[name]
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args or args[req] is None or args[req] == "":
            hint = props.get(req, {}).get("type", "string")
            return (f"[ERROR] {name}: missing required argument '{req}' "
                    f"({hint}). Re-issue the call with it.")
    for arg, value in args.items():
        spec = props.get(arg)
        if not spec:
            continue   # unknown extra arguments are ignored by the tools
        enum = spec.get("enum")
        if enum and str(value).lower() not in [str(v).lower() for v in enum]:
            return (f"[ERROR] {name}: '{arg}' must be one of "
                    f"{', '.join(str(v) for v in enum)}; got {value!r}.")
        expected = spec.get("type")
        if expected == "integer" and (
                not isinstance(value, int) or isinstance(value, bool)):
            return (f"[ERROR] {name}: '{arg}' must be an integer; "
                    f"got {value!r}.")
        if expected == "boolean" and not isinstance(value, bool):
            return (f"[ERROR] {name}: '{arg}' must be a boolean "
                    f"(true/false); got {value!r}.")
    return None


# ── Tool dispatch ──────────────────────────────────────────────────────────

def bound_tool_result(result: str) -> str:
    """Keep one tool result within E2B's per-call context budget."""
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    boundary = result.rfind("\n", 0, MAX_TOOL_RESULT_CHARS)
    if boundary <= 0:
        boundary = MAX_TOOL_RESULT_CHARS
    return (
        result[:boundary]
        + "\n[INFO] Tool result truncated to preserve E2B context. "
          "Refine the lookup or retrieve a specific source.]"
    )


def compact_tool_history(messages: list[dict]) -> int:
    """Replace oldest retained tool payloads after the E2B history budget.

    Tool messages stay in place, preserving the assistant tool-call /
    tool-result pairing required by OpenAI-compatible tool transports.
    The function returns the number of compacted payloads for diagnostics.
    """
    retained_chars = 0
    compacted = 0
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        if content == COMPACTED_TOOL_RESULT:
            continue
        if retained_chars + len(content) <= MAX_TOOL_HISTORY_CHARS:
            retained_chars += len(content)
            continue
        message["content"] = COMPACTED_TOOL_RESULT
        compacted += 1
    return compacted

def execute_tool(name: str, args: dict, index: dict,
                 sections_text: str, cache: dict,
                 weather: dict | None = None) -> tuple[str, bool]:
    """Run a tool call against the index.

    Returns (text_result, cache_hit).  `cache` is shared across calls within
    a session and is keyed by (tool, normalised-arg-tuple). `weather` is
    the optional loaded weather artifact used by `get_weather`.
    """
    def result_limit(key: str, default: int) -> int:
        try:
            return int(args.get(key, default))
        except (TypeError, ValueError):
            return default

    def store(key: tuple, result: str) -> tuple[str, bool]:
        bounded = bound_tool_result(result)
        cache[key] = bounded
        return bounded, False

    if name == "list_sections":
        return bound_tool_result(sections_text), True   # always pre-warmed

    if name == "get_section":
        section_id = (args.get("section_id") or "").strip()
        sort = (args.get("sort") or "interest").lower()
        # limit=None (not supplied) -> format_section applies the adaptive
        # default (20 for grouped sections, 50 for flat ones).  The cache
        # key preserves None so the prewarmed entry matches.
        raw_limit = args.get("limit")
        limit = result_limit("limit", 0) if raw_limit not in (None, "") else None
        key = ("get_section", section_id.lower(), sort, limit)
        if key in cache:
            return cache[key], True
        return store(key, format_section(index, section_id, sort=sort, limit=limit))

    if name == "get_poi":
        poi_id = (args.get("poi_id") or "").strip()
        key = ("get_poi", poi_id)
        if key in cache:
            return cache[key], True
        return store(key, format_poi(index, poi_id))

    if name == "find_poi_by_name":
        query = (args.get("query") or "").strip()
        limit = result_limit("limit", 5)
        detail = (args.get("detail") or "brief").lower()
        key = ("find_poi_by_name", query.lower(), limit, detail)
        if key in cache:
            return cache[key], True
        return store(key, format_find_poi_by_name(
            index, query, limit=limit, detail=detail
        ))

    if name == "filter_pois":
        active = {k: v for k, v in args.items()
                  if v not in (None, "", [], {})}
        limit = result_limit("limit", 20)
        active.pop("limit", None)
        key = ("filter_pois", tuple(sorted(active.items())), limit)
        if key in cache:
            return cache[key], True
        return store(key, format_filter_pois(index, limit=limit, **active))
    if name == "search_pois":
        query = (args.get("query") or "").strip()
        section_id = (args.get("section_id") or "").strip() or None
        limit = result_limit("limit", 10)
        key = ("search_pois", query.lower(), section_id, limit)
        if key in cache:
            return cache[key], True
        return store(key, format_search_pois(
            index, query, section_id=section_id, limit=limit
        ))
    if name == "search_trips":
        query = (args.get("query") or "").strip()
        limit = result_limit("limit", 10)
        key = ("search_trips", query.lower(), limit)
        if key in cache:
            return cache[key], True
        return store(key, format_search_trips(index, query, limit=limit))
    if name == "get_trip":
        trip_id = (args.get("trip_id") or "").strip()
        key = ("get_trip", trip_id)
        if key in cache:
            return cache[key], True
        return store(key, format_trip(index, trip_id))
    if name == "search_paths":
        query = (args.get("query") or "").strip()
        limit = result_limit("limit", 10)
        key = ("search_paths", query.lower(), limit)
        if key in cache:
            return cache[key], True
        return store(key, format_search_paths(index, query, limit=limit))
    if name == "get_path":
        path_id = (args.get("path_id") or "").strip()
        key = ("get_path", path_id)
        if key in cache:
            return cache[key], True
        return store(key, format_path(index, path_id))
    if name == "get_weather":
        day = (args.get("day") or "").strip() or None
        key = ("get_weather", (day or "").lower())
        if key in cache:
            return cache[key], True
        result = format_weather(weather, day)
        # Deterministic, language-agnostic heat check on whichever day
        # was resolved; only fires for a specific day, never the 7-day
        # dump, since "too hot" needs one temperature to compare.
        if day and is_forecast_too_hot(get_weather_entry(weather, day)):
            result = f"{result}\n{OUTDOOR_HEAT_NOTE}"
        return store(key, result)

    return f"[ERROR] Unknown tool: {name}", False


# ── Section-access tracking (used by the rubric) ─────────────────────────────

def _section_titles_for_poi(index: dict, poi_id: str) -> list[str]:
    """Return section titles owning a POI (usually one)."""
    by_section = (index.get("facets") or {}).get("by_section") or {}
    out = []
    for sid, ids in by_section.items():
        if poi_id in ids:
            sec = find_section(index, sid)
            if sec:
                out.append(sec.get("title", ""))
    return out


def sections_accessed_from_calls(tool_calls: list, index: dict) -> list[str]:
    """Map a sequence of tool calls to the section titles touched.

    This drives the eval rubric's retrieval-accuracy score.
    """
    seen: list[str] = []

    def add(title: str) -> None:
        if title and title not in seen:
            seen.append(title)

    def add_itinerary_sections(itinerary: dict | None) -> None:
        if not itinerary:
            return
        for step in itinerary.get("steps") or []:
            for poi_id in step.get("poi_ids") or []:
                for title in _section_titles_for_poi(index, poi_id):
                    add(title)

    for call in tool_calls:
        tool = call.get("tool")
        args = call.get("args") or {}

        if tool == "get_section":
            sec = find_section(index, (args.get("section_id") or ""))
            if sec:
                add(sec.get("title", ""))

        elif tool == "get_poi":
            # poi_id may carry several comma-separated ids (batch fetch)
            poi_ids = [p.strip() for p in (args.get("poi_id") or "").split(",")
                       if p.strip()]
            for poi_id in poi_ids:
                poi = ix_get_poi(index, poi_id)
                if poi:
                    for t in _section_titles_for_poi(index, poi["poi_id"]):
                        add(t)

        elif tool == "find_poi_by_name":
            for poi in ix_find_poi_by_name(index,
                                           args.get("query") or "",
                                           limit=int(args.get("limit") or 5)):
                for t in _section_titles_for_poi(index, poi["poi_id"]):
                    add(t)

        elif tool == "filter_pois":
            facet_args = {k: v for k, v in args.items()
                          if k != "limit" and v not in (None, "", [], {})}
            if "section_id" in facet_args:
                sec = find_section(index, facet_args["section_id"])
                if sec:
                    add(sec.get("title", ""))
                continue
            limit = int(args.get("limit") or 20)
            for poi in ix_filter_pois(index, **facet_args)[:limit]:
                for t in _section_titles_for_poi(index, poi["poi_id"]):
                    add(t)

        elif tool == "search_pois":
            section_id = (args.get("section_id") or "").strip() or None
            limit = int(args.get("limit") or 10)
            for item in ix_search_pois(
                index,
                args.get("query") or "",
                section_id=section_id,
                limit=limit,
            ):
                poi = item["poi"]
                for t in _section_titles_for_poi(index, poi["poi_id"]):
                    add(t)

        elif tool == "search_trips":
            for itinerary in ix_search_trips(
                index, args.get("query") or "",
                limit=int(args.get("limit") or 10),
            ):
                add_itinerary_sections(itinerary)

        elif tool == "get_trip":
            add_itinerary_sections(ix_get_trip(
                index, args.get("trip_id") or "",
            ))

        elif tool == "search_paths":
            for itinerary in ix_search_paths(
                index, args.get("query") or "",
                limit=int(args.get("limit") or 10),
            ):
                add_itinerary_sections(itinerary)

        elif tool == "get_path":
            add_itinerary_sections(ix_get_path(
                index, args.get("path_id") or "",
            ))

        # list_sections doesn't access content
    return seen


# ── Agentic loop ───────────────────────────────────────────────────────────

def run_agentic_loop(question: str, system_prompt: str,
                     index: dict, sections_text: str,
                     model: str, cache: dict,
                     recovery_msg: str = "",
                     weather: dict | None = None) -> dict:
    """Run the tool-calling loop for one question."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    weather_lookup_enforced = False
    weather_intent = bool(weather) and is_weather_request(question)
    tool_calls_made = []
    answer = ""
    error  = None
    cache_hits = 0
    rounds = 0
    prompt_tokens = 0
    completion_tokens = 0
    direct_evidence_missing = False
    complementary_retrieval_started = False
    physical_route_request = is_physical_route_request(question)
    path_search_started = False
    no_matching_path = False
    no_path_answer_enforced = False
    route_lookup_enforced = False
    grounding_required = requires_current_turn_grounding(question)
    grounded = False
    grounding_retry_enforced = False
    grounding_tools: list[str] = []
    automatic_source_calls: list[dict] = []
    trip_detail_required = requires_trip_detail(question)
    trip_search_started = False
    trip_search_has_results = False
    trip_search_default: dict | None = None
    trip_detail_started = False
    trip_detail_enforced = False
    source_detail_answer = ""
    call_keys: list[str] = []
    loop_correction_given = False
    loop_correction_pending: str | None = None
    loop_abort = False
    results_by_key: dict[str, str] = {}
    direct_trip_selection = resolve_trip_query(question, index)
    if direct_trip_selection:
        result, hit = execute_tool(
            "get_trip", {"trip_id": direct_trip_selection["id"]},
            index, sections_text, cache, weather=weather,
        )
        if hit:
            cache_hits += 1
        tool_calls_made.append({
            "tool": "get_trip",
            "args": {"trip_id": direct_trip_selection["id"]},
            "result_preview": result[:300],
            "cache_hit": hit,
            "automatic": True,
            "source_selection": direct_trip_selection,
        })
        grounded = True
        grounding_tools.append("get_trip")
        automatic_source_calls.append({
            "tool": "get_trip",
            "args": {"trip_id": direct_trip_selection["id"]},
            "source_selection": direct_trip_selection,
        })
        trip_detail_started = True
        source_detail_answer = result
    elif trip_detail_required:
        offer_candidates = ix_search_trips(index, question, limit=3)
        if len(offer_candidates) >= 2:
            offer_text = format_trip_choice_offer(index, offer_candidates[:3])
            tool_calls_made.append({
                "tool": "search_trips",
                "args": {"query": question, "limit": 3},
                "result_preview": offer_text[:300],
                "cache_hit": False,
                "automatic": True,
            })
            grounded = True
            grounding_tools.append("search_trips")
            automatic_source_calls.append({
                "tool": "search_trips",
                "args": {"query": question, "limit": 3},
            })
            trip_search_started = True
            trip_search_has_results = True
            trip_search_default = offer_candidates[0]
            source_detail_answer = offer_text

    for round_num in range(MAX_TOOL_ROUNDS):
        rounds = round_num + 1
        if source_detail_answer:
            answer = sanitize_tourist_answer(source_detail_answer, index)
            messages.append({"role": "assistant", "content": answer})
            break
        compact_tool_history(messages)
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=TOOL_DEFS,
                tool_choice="auto",
                temperature=0,
                max_tokens=MAX_ANSWER_TOKENS,
            )
        except Exception as exc:
            error = str(exc)
            break

        # Token accounting (feeds the on-device budgets in
        # docs/mobile-offline-contract.md)
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens     += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        choice  = response.choices[0]
        message = choice.message

        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            if weather_intent and "get_weather" not in grounding_tools:
                if not weather_lookup_enforced:
                    weather_lookup_enforced = True
                    weather_result, weather_hit = execute_tool(
                        "get_weather", {}, index,
                        sections_text, cache, weather=weather,
                    )
                    if weather_hit:
                        cache_hits += 1
                    tool_calls_made.append({
                        "tool": "get_weather",
                        "args": {},
                        "result_preview": weather_result[:300],
                        "cache_hit": weather_hit,
                        "automatic": True,
                    })
                    grounded = True
                    if "get_weather" not in grounding_tools:
                        grounding_tools.append("get_weather")
                    automatic_source_calls.append({
                        "tool": "get_weather",
                        "args": {},
                    })
                    messages.append({
                        "role": "user",
                        "content": (WEATHER_LOOKUP_ENFORCED_INSTRUCTION
                                    + "\n\n" + weather_result),
                    })
                    continue
            if physical_route_request and not path_search_started:
                # E2B occasionally responds to a route request with a
                # clarification question without calling search_paths.
                # Perform exactly one deterministic lookup ourselves and
                # feed the result back; never re-inject the same demand in
                # a loop.
                if route_lookup_enforced:
                    answer = sanitize_tourist_answer(
                        (message.content or "").strip(), index
                    )
                    break
                route_lookup_enforced = True
                route_result, route_hit = execute_tool(
                    "search_paths", {"query": question}, index,
                    sections_text, cache, weather=weather,
                )
                path_search_started = True
                no_matching_path = route_result.startswith(
                    "No curated routes matched"
                )
                if route_hit:
                    cache_hits += 1
                tool_calls_made.append({
                    "tool": "search_paths",
                    "args": {"query": question},
                    "result_preview": route_result[:300],
                    "cache_hit": route_hit,
                    "automatic": True,
                })
                grounded = True
                grounding_tools.append("search_paths")
                automatic_source_calls.append({
                    "tool": "search_paths",
                    "args": {"query": question},
                })
                messages.append({
                    "role": "user",
                    "content": route_lookup_context(route_result),
                })
                continue
            if no_matching_path and not no_path_answer_enforced:
                no_path_answer_enforced = True
                messages.append({
                    "role": "user",
                    "content": no_path_answer_instruction(
                        "get_weather" in grounding_tools
                    ),
                })
                continue
            # A small model can correctly discover that no single record
            # supports a compound request, then wrongly stop and ask the
            # visitor to choose a next search. Force one generic recovery
            # turn: retrieve complementary options in the same answer.
            if direct_evidence_missing and not complementary_retrieval_started:
                messages.append({
                    "role": "user",
                    "content": COMPLEMENTARY_SEARCH_INSTRUCTION,
                })
                continue
            if (trip_detail_required and trip_search_started
                    and trip_search_has_results and not trip_detail_started):
                if not trip_detail_enforced:
                    trip_detail_enforced = True
                    messages.append({
                        "role": "user",
                        "content": TRIP_DETAIL_REQUIRED_INSTRUCTION,
                    })
                    continue
                if trip_search_default:
                    selected_id = trip_search_default["itinerary_id"]
                    result, hit = execute_tool(
                        "get_trip", {"trip_id": selected_id}, index,
                        sections_text, cache, weather=weather,
                    )
                    if hit:
                        cache_hits += 1
                    tool_calls_made.append({
                        "tool": "get_trip",
                        "args": {"trip_id": selected_id},
                        "result_preview": result[:300],
                        "cache_hit": hit,
                        "automatic": True,
                        "source_selection": {
                            "kind": "trip",
                            "id": selected_id,
                            "label": trip_search_default.get("name", ""),
                        },
                    })
                    grounded = True
                    if "get_trip" not in grounding_tools:
                        grounding_tools.append("get_trip")
                    automatic_source_calls.append({
                        "tool": "get_trip",
                        "args": {"trip_id": selected_id},
                        "source_selection": {
                            "kind": "trip",
                            "id": selected_id,
                            "label": trip_search_default.get("name", ""),
                        },
                    })
                    trip_detail_started = True
                    source_detail_answer = result
                    continue
                answer = grounding_failure_message(index)
                break
            if grounding_required and not grounded:
                if not grounding_retry_enforced:
                    grounding_retry_enforced = True
                    messages.append({
                        "role": "user",
                        "content": GROUNDING_RECOVERY_INSTRUCTION,
                    })
                    continue
                fallback = history_followup_answer(question, messages, index)
                if fallback:
                    tool_calls_made.append({
                        "tool": "history_followup",
                        "args": {"query": question},
                        "result_preview": fallback[:300],
                        "cache_hit": False,
                        "automatic": True,
                    })
                    automatic_source_calls.append({
                        "tool": "history_followup",
                        "args": {"query": question},
                    })
                    answer = sanitize_tourist_answer(fallback, index)
                    assistant_msg["content"] = answer
                    break
                # Last-resort backstop for small models: if the question
                # explicitly names a known place (or confirms an assistant
                # offer that named places), fetch those records
                # deterministically instead of accepting a brush-off.
                named_pois = backstop_named_pois(question, messages, index)
                if named_pois:
                    name_index = index.get("name_index") or {}
                    named_ids = ",".join(dict.fromkeys(
                        name_index[n] for n in named_pois if n in name_index
                    ))
                    result, hit = execute_tool(
                        "get_poi", {"poi_id": named_ids}, index,
                        sections_text, cache, weather=weather,
                    )
                    if hit:
                        cache_hits += 1
                    tool_calls_made.append({
                        "tool": "get_poi",
                        "args": {"poi_id": named_ids},
                        "result_preview": result[:300],
                        "cache_hit": hit,
                        "automatic": True,
                        "source_selection": {
                            "kind": "poi",
                            "id": named_ids,
                            "label": ", ".join(named_pois),
                        },
                    })
                    grounded = True
                    if "get_poi" not in grounding_tools:
                        grounding_tools.append("get_poi")
                    automatic_source_calls.append({
                        "tool": "get_poi",
                        "args": {"poi_id": named_ids},
                        "source_selection": {
                            "kind": "poi",
                            "id": named_ids,
                            "label": ", ".join(named_pois),
                        },
                    })
                    messages.append({
                        "role": "user",
                        "content": (NAMED_POI_LOOKUP_INSTRUCTION
                                    + "\n\n" + result),
                    })
                    continue
                # The model answered the recovery prompt with yet another
                # clarifying question — a brush-off. In an ongoing
                # conversation, retrieve real content deterministically
                # and make it answer from that instead of re-asking.
                if (reask_fallback_applies(messages, question)
                        and is_pure_reask((message.content or "").strip())):
                    result, hit, fb_tool, fb_args = execute_reask_fallback(
                        question, index, sections_text, cache, weather=weather,
                    )
                    if hit:
                        cache_hits += 1
                    tool_calls_made.append({
                        "tool": fb_tool,
                        "args": fb_args,
                        "result_preview": result[:300],
                        "cache_hit": hit,
                        "automatic": True,
                    })
                    grounded = True
                    if fb_tool not in grounding_tools:
                        grounding_tools.append(fb_tool)
                    automatic_source_calls.append({
                        "tool": fb_tool,
                        "args": fb_args,
                    })
                    messages.append({
                        "role": "user",
                        "content": (REASK_FALLBACK_INSTRUCTION
                                    + "\n\n" + result),
                    })
                    continue
                # The model declined retrieval after the recovery prompt and
                # no known place is named: trust its generic-question
                # classification and deliver the answer it composed from
                # the preloaded overview/catalogue.
                answer = sanitize_tourist_answer(
                    (message.content or "").strip(), index
                )
                break
            answer = sanitize_tourist_answer((message.content or "").strip(), index)
            break

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args: dict | None = {}
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = None

            # Keep distinct broken-JSON payloads distinct for the loop
            # detector; identical ones trip it like any other repeat.
            key_args = fn_args if fn_args is not None else {
                "__raw__": (tc.function.arguments or "")[:200]}
            call_keys.append(tool_call_key(fn_name, key_args))
            if is_repeat_tool_call(call_keys):
                # Third identical call this turn: block the execution and
                # answer with a stub.  Correct the model once; on any
                # further repeat, abort the tool loop — the tail recovery
                # then forces a final answer from what it already has.
                if loop_correction_given:
                    loop_abort = True
                else:
                    loop_correction_pending = fn_name
                tool_calls_made.append({
                    "tool":           fn_name,
                    "args":           fn_args or {},
                    "result_preview": LOOP_REPEAT_STUB,
                    "cache_hit":      False,
                    "loop_blocked":   True,
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      LOOP_REPEAT_STUB,
                })
                continue

            cached_result = results_by_key.get(call_keys[-1])
            if cached_result is not None:
                # Second occurrence of an identical call this turn.  The
                # tools are deterministic lookups, so re-execution can
                # return nothing new: serve a short stub while the
                # original result is still in context, or re-serve the
                # cached result itself when compact_tool_history has
                # since replaced it (its own text tells the model to
                # retrieve the source again when needed).
                still_visible = any(
                    m.get("role") == "tool" and m.get("content") == cached_result
                    for m in messages
                )
                repeat_content = (LOOP_REPEAT_CACHE_STUB if still_visible
                                  else cached_result)
                tool_calls_made.append({
                    "tool":           fn_name,
                    "args":           fn_args or {},
                    "result_preview": repeat_content[:300],
                    "cache_hit":      False,
                    "repeat_cached":  True,
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      repeat_content,
                })
                continue

            invalid = validate_tool_call(fn_name, fn_args,
                                         tc.function.arguments or "")
            if invalid:
                tool_calls_made.append({
                    "tool":           fn_name,
                    "args":           fn_args or {},
                    "result_preview": invalid,
                    "cache_hit":      False,
                    "invalid_args":   True,
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      invalid,
                })
                continue

            result, hit = execute_tool(
                fn_name, fn_args, index, sections_text, cache, weather=weather,
            )
            results_by_key[call_keys[-1]] = result
            if fn_name in SOURCE_GROUNDING_TOOLS:
                grounded = True
                if fn_name not in grounding_tools:
                    grounding_tools.append(fn_name)
            if fn_name == "search_pois":
                if result.startswith(NO_DIRECT_EVIDENCE_PREFIX):
                    direct_evidence_missing = True
                elif direct_evidence_missing:
                    complementary_retrieval_started = True
            elif direct_evidence_missing and fn_name in {
                "filter_pois", "get_section", "find_poi_by_name",
            }:
                complementary_retrieval_started = True
            if fn_name == "search_paths":
                path_search_started = True
                if result.startswith("No curated routes matched"):
                    no_matching_path = True
            if fn_name == "search_trips":
                trip_search_started = True
                trip_matches = ix_search_trips(
                    index, fn_args.get("query") or "",
                    limit=int(fn_args.get("limit") or 10),
                )
                trip_search_has_results = bool(trip_matches)
                trip_search_default = trip_matches[0] if trip_matches else None
            if fn_name == "get_trip":
                trip_detail_started = True
                if trip_detail_required:
                    source_detail_answer = result
            if fn_name == "get_path":
                source_detail_answer = result
            if hit:
                cache_hits += 1
            tool_calls_made.append({
                "tool":           fn_name,
                "args":           fn_args,
                "result_preview": result[:300],
                "cache_hit":      hit,
            })
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

        if loop_abort:
            break
        if loop_correction_pending:
            blocked_tool = loop_correction_pending
            loop_correction_pending = None
            loop_correction_given = True
            messages.append({
                "role":    "user",
                "content": LOOP_REPEAT_INSTRUCTION.format(tool=blocked_tool),
            })
            continue

    if not answer and grounding_required and not grounded:
        answer = grounding_failure_message(index)
    if not answer:
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                answer = sanitize_tourist_answer(msg["content"].strip(), index)
                break

    if not answer and not error:
        msg = recovery_msg or _recovery_msg("en")
        try:
            compact_tool_history(messages)
            recovery = litellm.completion(
                model=model,
                messages=messages + [{"role": "user", "content": msg}],
                temperature=0,
                max_tokens=MAX_ANSWER_TOKENS,
            )
            usage = getattr(recovery, "usage", None)
            if usage:
                prompt_tokens     += getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            answer = sanitize_tourist_answer(
                (recovery.choices[0].message.content or "").strip(), index
            )
        except Exception as exc:
            error = f"recovery failed: {exc}"

    return {
        "answer":     answer,
        "tool_calls": tool_calls_made,
        "rounds":     rounds,
        "cache_hits": cache_hits,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "grounded": grounded or not grounding_required,
        "grounding_tools": grounding_tools,
        "automatic_source_calls": automatic_source_calls,
        "error":      error,
    }


# ── Inputs & helpers ───────────────────────────────────────────────────────

def load_inputs(questions_file: Path | None = None,
                index_file: Path | None = None) -> tuple[list, dict]:
    """Load questions and the POI index.  Fail fast if missing."""
    q_file = questions_file or QUESTIONS_FILE
    i_file = index_file or DEFAULT_INDEX
    for path in (q_file, i_file):
        if not path.exists():
            print(f"[ERROR] Missing: {path}", file=sys.stderr)
            sys.exit(1)
    with open(q_file, encoding="utf-8") as f:
        questions = json.load(f)
    index = load_index(i_file)
    return questions, index


# ── Main ───────────────────────────────────────────────────────────────────

def _resolve_index_arg(args) -> Path:
    """Accept --index OR legacy --structure (with deprecation note)."""
    if args.index:
        path = Path(args.index)
    elif args.structure:
        # Legacy compatibility shim: try to remap old structure paths to
        # the new index file by stripping '_guide' and '_structure'.
        legacy = Path(args.structure)
        guess_name = legacy.name.replace("_guide", "").replace(
            "_structure.json", ".json")
        guessed = legacy.parent.parent / "indexes" / guess_name
        if not guessed.exists():
            # One subfolder per destination: {dest}_{lang}.json ->
            # {dest}/{lang}.json
            dest, _, lang_file = guess_name.rpartition("_")
            if dest:
                guessed = legacy.parent.parent / "indexes" / dest / lang_file
        if guessed.exists():
            print(f"[WARN] --structure is deprecated; using {guessed}",
                  file=sys.stderr)
            path = guessed
        else:
            path = legacy
    else:
        path = DEFAULT_INDEX
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POI-index Q&A evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"litellm model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--questions", default=None,
                        help="Path to questions JSON (default: eval/ubeda/questions.json)")
    parser.add_argument("--index", default=None,
                        help=f"Path to POI index JSON (default: {DEFAULT_INDEX})")
    parser.add_argument("--structure", default=None,
                        help=argparse.SUPPRESS)  # legacy, hidden
    parser.add_argument("--lang", default="en",
                        help=("Response language code (default: en). "
                              "One of: " + ", ".join(SUPPORTED_LANGS)))
    args = parser.parse_args()

    if not is_supported(args.lang):
        print(f"[ERROR] Unsupported --lang '{args.lang}'. "
              f"Supported codes: {', '.join(SUPPORTED_LANGS)}",
              file=sys.stderr)
        sys.exit(1)

    questions_file = Path(args.questions) if args.questions else QUESTIONS_FILE
    index_path     = _resolve_index_arg(args)

    questions, index = load_inputs(questions_file, index_path)

    destination_display = (index.get("meta") or {}).get("destination_display") \
                          or (index.get("meta") or {}).get("destination") \
                          or "this destination"
    sections_text = format_sections_overview(index)
    overview_text = index.get("destination_overview", "")
    dest_slug = (index.get("meta") or {}).get("destination") or ""
    weather_path = PROJECT_ROOT / "weather" / dest_slug / f"{args.lang}.json"
    weather = load_weather(weather_path) if dest_slug else None
    hint_text = weather_hint(weather, destination_display) if weather else ""
    system_prompt = make_system_prompt(
        sections_text=sections_text,
        destination=destination_display,
        destination_overview=overview_text,
        lang=args.lang,
        weather_hint_text=hint_text,
    )

    # Pre-warm: cache get_section for every section with the adaptive
    # default limit (None), matching what the model's calls produce.
    cache: dict = {}
    for sec in index.get("sections", []):
        sid = sec.get("section_id", "")
        if sid:
            cache[("get_section", sid.lower(), "interest", None)] = format_section(
                index, sid, sort="interest", limit=None)
    print(f"[INFO] Pre-warmed cache: {len(cache)} sections")

    recovery = recovery_msg(args.lang)

    model_tag   = args.model.split("/")[-1].replace(":", "-")
    lang_suffix = f"_{args.lang}" if args.lang != "en" else ""
    output_file = RESULTS_DIR / f"eval_{model_tag}{lang_suffix}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"[INFO] Model:          {args.model}")
    print(f"[INFO] Language:       {args.lang}")
    print(f"[INFO] Index:          {index_path.name}  "
          f"({(index.get('meta') or {}).get('poi_count', '?')} POIs)")
    print(f"[INFO] Questions:      {len(questions)}  ({questions_file.name})")
    print(f"[INFO] Output:         {output_file}")
    print(f"[INFO] System prompt:  {len(system_prompt):,} chars\n")

    results = []
    total_start = time.time()

    for i, q in enumerate(questions, 1):
        qid        = q["id"]
        question   = q["question"]
        difficulty = q.get("difficulty", "?")
        print(f"[{i:2d}/{len(questions)}] {qid} ({difficulty})  {question[:70]}...")
        t0 = time.time()

        loop = run_agentic_loop(
            question, system_prompt, index, sections_text,
            args.model, cache, recovery_msg=recovery,
            weather=weather,
        )

        elapsed = round(time.time() - t0, 2)
        sections = sections_accessed_from_calls(loop["tool_calls"], index)
        poi_refs = extract_poi_tags(loop["answer"], index)

        result = {
            "id":               qid,
            "model":            args.model,
            "lang":             args.lang,
            "category":         q.get("category"),
            "difficulty":       difficulty,
            "question":         question,
            "expected_section": q.get("expected_section"),
            "grounding_check":  q.get("grounding_check"),
            "answer":           loop["answer"],
            "tool_calls":       loop["tool_calls"],
            "sections_accessed": sections,
            "poi_refs":         poi_refs,
            "rounds":           loop["rounds"],
            "cache_hits":       loop["cache_hits"],
            "latency_seconds":  elapsed,
            "prompt_tokens":     loop["prompt_tokens"],
            "completion_tokens": loop["completion_tokens"],
            "grounded":         loop["grounded"],
            "grounding_tools":  loop["grounding_tools"],
            "automatic_source_calls": loop["automatic_source_calls"],
            "error":            loop["error"],
        }
        results.append(result)

        status = "ERROR" if loop["error"] else "OK"
        tools = [c["tool"] for c in loop["tool_calls"]]
        print(f"  [{status}] {elapsed}s  rounds={loop['rounds']}  "
              f"tools={tools}  cache={loop['cache_hits']}  "
              f"tokens={loop['prompt_tokens']}+{loop['completion_tokens']}  "
              f"tags={len(poi_refs)}  grounded={loop['grounded']}")

    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n[INFO] All questions complete in {total_elapsed}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {output_file}")


if __name__ == "__main__":
    main()

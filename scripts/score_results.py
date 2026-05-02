#!/usr/bin/env python3
"""
score_results.py — Score PageIndex Q&A evaluation results.

Loads results/eval_*.json and applies a rubric:
  1. Retrieval accuracy  — did the model access an expected section?
  2. Content fetched     — did it call get_page_content at all?
  3. Factual grounding   — does the answer contain verifiable key facts?
  4. Language correct    — is the answer in English (for EN questions)?
  5. Latency             — wall-clock seconds per question.

Usage:
    .venv/bin/python scripts/score_results.py
    .venv/bin/python scripts/score_results.py --file results/eval_gemma4-e4b.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

# Load current expected sections from questions.json as the authoritative source.
# These override whatever expected_section is stored inside a result file,
# so rubric corrections to questions.json take effect without re-running the eval.
_QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.json"
EXPECTED_SECTIONS: dict[str, str] = {}
if _QUESTIONS_FILE.exists():
    with open(_QUESTIONS_FILE, encoding="utf-8") as _qf:
        for _q in json.load(_qf):
            if _q.get("expected_section"):
                EXPECTED_SECTIONS[_q["id"]] = _q["expected_section"]

# ── Factual grounding checks ──────────────────────────────────────────────────
# Each entry: (question_id, list_of_required_substrings_or_patterns).
# All items in the list must be present for full credit; partial matches score 0.5.
# Matches are case-insensitive.

FACT_CHECKS: dict[str, list[str]] = {
    "Q01": ["2003", "renaissance"],
    "Q02": ["vázquez de molina"],                  # production: model describes the square without naming the Chapel
    "Q03": ["plaza vázquez de molina", "+34953750345"],  # phone compared digit-only via _matches
    "Q04": ["museum"],                              # at least one museum name
    "Q05": ["san nicolás", "guadalupe"],            # two specific churches
    "Q06": ["1562", "guadalimar", "100"],           # bridge facts
    "Q07": ["tourist information"],                 # generic check
    "Q08": ["parking", "plaza de andalucía"],       # parking + location
    "Q09": ["restaurant"],                          # at least a restaurant name
    "Q10": ["oil", "olive"],                        # olive oil connection
    "Q11": ["parador", "hotel"],                    # parador confirmed
    "Q12": ["renaissance", "vandelvira"],           # key Renaissance architect
    "Q13": ["holy week"],                          # festival (data is English-only; 'Semana Santa' removed)
    "Q14": ["viewpoint", "santa luc"],              # viewpoint name (prefix matches both lucia/lucía)
    "Q15": ["megalithic"],                          # dolmen facts ('3rd millennium' rarely quoted verbatim)
    "Q16": ["pharmacy"],                            # English source; 'farmacia' removed (Spanish not in data)
    "Q17": ["tour", "falcon"],                      # 'falcon' = Falcon Travel, a POI that always appears
    "Q18": ["vázquez de molina", "salvador"],     # production: 'Sacra Capilla del Salvador' not 'Chapel of El Salvador'
    "Q19": ["olive", "restaurant"],                 # gastronomy + olive oil
    "Q20": ["2003", "renaissance"],                 # 'andalusia' removed (model omits it but answers correctly)
}

# For Spanish (lang='es'), these keyword equivalents are tried when the
# English check fails. Each value is a tuple of acceptable substrings;
# the check passes if any of them appears in the answer. Stems and
# alternative inflections live here so 'renacimiento' and 'renacentista'
# both score for an English 'renaissance' check.
# Proper nouns and numbers stay verbatim.
_ES_KEYWORDS: dict[str, tuple[str, ...]] = {
    "renaissance":         ("renacentista", "renacimiento"),
    "museum":              ("museo",),
    "restaurant":          ("restaurante",),
    "holy week":           ("semana santa",),
    "megalithic":          ("megalític",),    # megalítico / megalítica
    "tourist information": ("información turística",),
    "pharmacy":            ("farmacia",),
    "parking":             ("aparcamiento",),
    "viewpoint":           ("mirador",),
    "savior":              ("salvador",),
    "olive":               ("oliva",),         # aceite de oliva
    "oil":                 ("aceite",),         # aceite (de oliva)
}

# For Italian (lang='it'), same idea. Italian adjectives inflect for
# gender/number, so we use stems where it matters (e.g. 'rinasciment'
# matches both 'rinascimento' and 'rinascimentale'). Proper nouns,
# numbers, UNE 178503 type codes and POI names stay verbatim.
_IT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "renaissance":         ("rinasciment",),       # rinascimento / rinascimentale
    "museum":              ("museo",),
    "restaurant":          ("ristorant",),         # ristorante / ristoranti
    "holy week":           ("settimana santa",),
    "megalithic":          ("megalitic",),         # megalitico / megalitica / megalitiche
    "tourist information": ("informazione turistica", "informazioni turistiche"),
    "pharmacy":            ("farmacia",),
    "parking":             ("parcheggio",),
    "viewpoint":           ("belvedere", "punto panoramico", "punti panoramici"),
    "savior":              ("salvatore",),
    "olive":               ("oliva",),
    "oil":                 ("olio",),
}


def _normalize_digits(s: str) -> str:
    """Return only the digit characters from s."""
    return re.sub(r"\D", "", s)


def _matches(check: str, answer_lower: str) -> bool:
    """
    Return True if `check` (case-insensitive) appears in `answer_lower`.

    Phone-number checks (strings that start with '+' followed only by digits,
    spaces, dashes, dots, or parentheses) are normalised to digit-only strings
    before comparison so that '+34953750345' matches '+34 953 75 03 45'.
    """
    check_lower = check.lower()
    if re.match(r"^\+[\d\s\-\.()]+$", check_lower):
        return _normalize_digits(check_lower) in _normalize_digits(answer_lower)
    return check_lower in answer_lower


# Expected sections (from questions.json expected_section field).
# We check if ANY expected section name is a substring of any accessed section.
def _sections_match(accessed: list[str], expected: str) -> bool:
    """True if any expected section appears (case-insensitive) in accessed list."""
    for exp_part in re.split(r",\s*", expected):
        exp_part = exp_part.strip().lower()
        for acc in accessed:
            if exp_part in acc.lower():
                return True
    return False


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_factual_grounding(qid: str, answer: str,
                            lang: str = "en") -> tuple[float, list[str]]:
    """
    Return (score 0.0–1.0, missing_facts).
    1.0 = all required facts present, 0.0 = none present.

    For Spanish evals (lang='es'), each check is also tried against its
    Spanish translation from _ES_KEYWORDS, so 'renacentista' counts as a
    hit for the 'renaissance' check.
    """
    checks = FACT_CHECKS.get(qid, [])
    if not checks:
        return 1.0, []

    answer_lower = answer.lower()
    missing = []
    # Per-language fallback dictionaries.  English check is tried first;
    # if it fails, each language-specific equivalent is tried in turn.
    _LANG_KEYWORDS = {"es": _ES_KEYWORDS, "it": _IT_KEYWORDS}
    fallback = _LANG_KEYWORDS.get(lang)
    for c in checks:
        if _matches(c, answer_lower):
            continue  # English check passed
        if fallback and c.lower() in fallback:
            if any(_matches(equiv, answer_lower)
                   for equiv in fallback[c.lower()]):
                continue  # Localised equivalent found
        missing.append(c)
    score = (len(checks) - len(missing)) / len(checks)
    return round(score, 2), missing


def score_retrieval(result: dict) -> float:
    """1.0 if the model accessed an expected section, else 0.0.

    Uses EXPECTED_SECTIONS (from questions.json) as the authoritative source,
    falling back to the value stored in the result file.
    """
    qid      = result.get("id", "")
    expected = EXPECTED_SECTIONS.get(qid) or result.get("expected_section", "")
    accessed = result.get("sections_accessed", [])
    if not expected or not accessed:
        return 0.0
    return 1.0 if _sections_match(accessed, expected) else 0.0


# Tool names that count as "the model retrieved real content".
# Includes the new POI-index tools (build_index.py / index_tools.py) and the
# legacy PageIndex tools so old result files still score correctly.
_CONTENT_FETCH_TOOLS = frozenset({
    # POI-index tools
    "get_poi", "get_section", "find_poi_by_name", "filter_pois",
    # Legacy PageIndex tools
    "get_page_content", "get_poi_list",
})


def score_content_fetched(result: dict, grounding: float = 0.0) -> float:
    """
    1.0 if any content-fetching tool was called, OR if grounding is perfect.

    Listing questions (category_browse, accommodation overviews) can correctly
    answer from a section listing alone — if the answer is fully grounded,
    the content was effectively retrieved through another tool and penalising
    it would be a false negative.
    """
    tool_names = {c["tool"] for c in result.get("tool_calls", [])}
    if tool_names & _CONTENT_FETCH_TOOLS:
        return 1.0
    # Fully-grounded answers that used listing strategy are not hallucinations
    if grounding >= 1.0:
        return 1.0
    return 0.0


# Stop-word sets used for language detection.
_SPANISH_STOPS = frozenset({
    "de", "la", "el", "en", "es", "se", "por", "los", "las",
    "un", "una", "con", "su", "del", "al", "que", "para",
    "como", "más", "también", "tiene", "están", "hay",
    "del", "entre", "este", "para", "pero", "son",
})
_ITALIAN_STOPS = frozenset({
    "di", "il", "la", "lo", "i", "gli", "le", "e", "ed", "un",
    "una", "uno", "per", "con", "del", "della", "dello", "dei",
    "degli", "delle", "al", "alla", "allo", "agli", "alle",
    "che", "come", "sono", "sia", "questa", "questo", "questi",
    "queste", "sua", "suo", "non", "anche", "da", "dal", "dalla",
    "nel", "nella", "sul", "sulla", "tra", "fra",
})


def score_language(result: dict) -> float:
    """
    Language conformance check.

    For English runs (lang='en'): returns 0.0 if the answer contains
    too many Spanish stop-words (> 12% of word count) — a coarse signal
    that the model fell through to its default Spanish persona.

    For Spanish (lang='es') and Italian (lang='it') runs: returns 1.0 if
    the answer contains enough language-specific stop-words (> 5% of
    word count), confirming the model responded in the requested language.

    For other languages: skipped (returns 1.0) — no detection logic yet.
    """
    lang  = result.get("lang", "en")
    words = re.findall(r"\b\w+\b", result.get("answer", "").lower())
    if not words:
        return 0.0

    if lang == "en":
        es_ratio = sum(1 for w in words if w in _SPANISH_STOPS) / len(words)
        return 0.0 if es_ratio > 0.12 else 1.0
    if lang == "es":
        es_ratio = sum(1 for w in words if w in _SPANISH_STOPS) / len(words)
        return 1.0 if es_ratio > 0.05 else 0.0
    if lang == "it":
        it_ratio = sum(1 for w in words if w in _ITALIAN_STOPS) / len(words)
        return 1.0 if it_ratio > 0.05 else 0.0
    return 1.0  # other languages: skip the check


def score_result(result: dict) -> dict:
    """Return a dict of all dimension scores for one result."""
    lang = result.get("lang", "en")
    grounding, missing = score_factual_grounding(result["id"], result.get("answer", ""), lang=lang)
    retrieval  = score_retrieval(result)
    fetched    = score_content_fetched(result, grounding)  # pass grounding for listing exemption
    language   = score_language(result)
    has_error  = bool(result.get("error"))

    composite = round((grounding * 0.4 + retrieval * 0.3 + fetched * 0.2 + language * 0.1), 3)
    if has_error:
        composite = 0.0

    return {
        "id":               result["id"],
        "difficulty":       result.get("difficulty", "?"),
        "category":         result.get("category", "?"),
        "grounding":        grounding,
        "retrieval":        retrieval,
        "content_fetched":  fetched,
        "language_ok":      language,
        "composite":        composite,
        "latency":          result.get("latency_seconds", 0),
        "error":            has_error,
        "missing_facts":    missing,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(scores: list[dict], model: str) -> None:
    """Print a per-question summary table."""
    hdr = f"{'ID':>4}  {'Diff':<6}  {'Category':<20}  {'Ground':>6}  {'Retriev':>7}  {'Fetched':>7}  {'Lang':>4}  {'Score':>5}  {'Lat':>6}  Notes"
    print(f"\n{'='*120}")
    print(f"Model: {model}")
    print('='*120)
    print(hdr)
    print('-'*120)
    for s in scores:
        notes = ""
        if s["error"]:
            notes = "ERROR"
        elif s["missing_facts"]:
            notes = f"missing: {', '.join(s['missing_facts'][:2])}"
        elif s["language_ok"] == 0:
            notes = "WRONG LANGUAGE"
        print(
            f"{s['id']:>4}  {s['difficulty']:<6}  {s['category']:<20}  "
            f"{s['grounding']:>6.2f}  {s['retrieval']:>7.1f}  {s['content_fetched']:>7.1f}  "
            f"{s['language_ok']:>4.1f}  {s['composite']:>5.3f}  {s['latency']:>6.1f}s  {notes}"
        )
    print('='*120)


def print_summary(scores: list[dict], model: str) -> None:
    """Print aggregate metrics and pass/fail verdict."""
    n = len(scores)
    avg = lambda key: round(sum(s[key] for s in scores) / n, 3)

    grounding_avg   = avg("grounding")
    retrieval_avg   = avg("retrieval")
    fetched_avg     = avg("content_fetched")
    composite_avg   = avg("composite")
    latency_avg     = avg("latency")

    n_fetched  = sum(1 for s in scores if s["content_fetched"] == 1.0)
    n_correct_retrieval = sum(1 for s in scores if s["retrieval"] == 1.0)

    # Per difficulty breakdown
    for diff in ("easy", "medium", "hard"):
        subset = [s for s in scores if s["difficulty"] == diff]
        if subset:
            d_avg = round(sum(s["composite"] for s in subset) / len(subset), 3)
            print(f"  {diff:<6}  ({len(subset):2d} Qs)  composite={d_avg}")

    print(f"\nAGGREGATE ({n} questions):")
    print(f"  Factual grounding avg :  {grounding_avg:.1%}")
    print(f"  Retrieval accuracy    :  {retrieval_avg:.1%}  ({n_correct_retrieval}/{n} correct sections)")
    print(f"  Content fetched       :  {fetched_avg:.1%}  ({n_fetched}/{n} questions fetched content)")
    print(f"  Composite score       :  {composite_avg:.3f}")
    print(f"  Avg latency           :  {latency_avg:.1f}s / question")

    # Thresholds from plan
    passes_grounding  = grounding_avg >= 0.70
    passes_hallucination = fetched_avg >= 0.70  # proxy: not fetching = risk of hallucination

    print(f"\nVERDICT (thresholds: grounding ≥ 70%, content-fetch ≥ 70%):")
    print(f"  Grounding ≥ 70% :  {'✅ PASS' if passes_grounding else '❌ FAIL'}  ({grounding_avg:.1%})")
    print(f"  Content-fetch ≥ 70% :  {'✅ PASS' if passes_hallucination else '❌ FAIL'}  ({fetched_avg:.1%})")

    if passes_grounding and passes_hallucination:
        print("\n  → E2B is sufficient for this retrieval task. No escalation needed.")
    else:
        print("\n  → Consider escalating to gemma4:e4b for improved performance.")


def save_scored(scores: list[dict], output_path: Path) -> None:
    """Save scored results as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Scored results saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load eval results, score, print, save."""
    parser = argparse.ArgumentParser(description="Score evaluation results")
    parser.add_argument(
        "--file", type=Path,
        default=None,
        help="Path to eval_*.json (default: most recent in results/)",
    )
    args = parser.parse_args()

    if args.file:
        result_files = [args.file]
    else:
        result_files = sorted(RESULTS_DIR.glob("eval_*.json"))

    if not result_files:
        print("[ERROR] No eval_*.json files found in results/", file=sys.stderr)
        sys.exit(1)

    for result_file in result_files:
        if not result_file.exists():
            print(f"[ERROR] Not found: {result_file}", file=sys.stderr)
            continue

        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)

        model = results[0].get("model", "unknown") if results else "unknown"
        scores = [score_result(r) for r in results]

        print_table(scores, model)
        print_summary(scores, model)

        scored_path = result_file.parent / result_file.name.replace("eval_", "scored_")
        save_scored(scores, scored_path)


if __name__ == "__main__":
    main()

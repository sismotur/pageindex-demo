# Inventrip Offline Tourism Assistant with Gemma 4

A self-contained framework that answers grounded tourism questions for
**any tourist destination** in **any language** by combining a custom
POI-aware index (built directly from the [Inventrip](https://inventrip.com)
API) with the **Gemma 4** family.

The repository is split into the two parts of the production architecture:

1. **`pipeline/` — data preparation.** LLM-free, deterministic scripts that
   fetch the Inventrip API and build `indexes/{dest}_{lang}.json`. Being
   ported to a **Cloudflare cron Worker** that publishes the same files to
   R2 (see `docs/cloudflare-worker-spec.md`).
2. **`assistant/` — offline chatbot.** The reference implementation of the
   on-device runtime: five pure-lookup tools over the index plus the
   agentic loop. Runs against any OpenAI-compatible endpoint (oMLX,
   Ollama) and will be **reimplemented in Android/iOS with Gemma 4 E2B
   fully offline** — the phone only downloads index files
   (see `docs/mobile-offline-contract.md`).

The reference dataset is Úbeda, Spain — 367 POIs returned by `/v120/pois`
under the [UNE 178503](https://www.une.org) Spanish tourism standard.

The pipeline supports **multiple destinations** and **multiple languages**.
All artifacts use the `{destination}_*_{lang}` naming convention so different
`(destination, language)` pairs never overwrite each other.

---

## Quick summary

Four models were evaluated end-to-end on 20 visitor questions per
language. The full report lives in `results/comparison_table.md`.

### Server-side (recommended: gemma4:26b)

| Run                             | Grounding | Retrieval | Composite | Avg latency |
|---------------------------------|-----------|-----------|-----------|-------------|
| 26B — PageIndex (pre-refactor)  | 92.5%     | 80.0%     | 0.910     | 26.5 s      |
| 26B — PageIndex (Spanish)       | 80.0%     | 80.0%     | 0.850     | 47.3 s      |
| **26B — POI-index (English)**   | **90.0%** | **95.0%** | **0.935** | **26.9 s**  |
| **26B — POI-index (Spanish)**   | **90.0%** | **95.0%** | **0.935** | 19.7 s¹     |
| **26B — POI-index (Italian)**   | **87.5%** | **85.0%** | **0.895** | **24.0 s**  |

¹ Median latency excluding three model-side looping outliers (Q09, Q11,
Q12). Mean including those: 132.5 s.

### Offline-mobile candidates (Inventrip Android/iOS app)

| Model         | Disk    | EN comp.  | ES comp.  | IT comp.  | EN lat. | All-lang pass? |
|---------------|---------|-----------|-----------|-----------|---------|----------------|
| **Gemma 4 E2B** — recommended | 7.2 GB | **0.850** | **0.830** | **0.760** | 13.5 s | ✅ yes |
| Qwen 2.5 7B — EN-first alt.   | 4.7 GB | 0.835     | 0.710     | 0.720     | 8.6 s  | ❌ ES/IT 5 pp short |
| Qwen 2.5 3B — unsuitable      | 1.9 GB | 0.745     | 0.590     | 0.525     | 3.0 s  | ❌ |

Latest E2B run on the current **oMLX** serving stack
(`gemma-4-E2B-it-MLX-8bit`, 2026-08-16): composite **0.820**, grounding
**72.5%**, content-fetch **95%**, **3.7 s**/question — pass on both
thresholds at ~4× lower latency than the Ollama baseline.

The POI-aware index lifts retrieval accuracy from 80% to 95% on the
server model and — critically — also unlocks the smallest Gemma 4
variant for **fully-offline mobile use**. The same E2B that scored 54%
grounding on the old PageIndex pipeline scores 85% on this one. See
`results/comparison_table.md` for the full cross-model report and the
offline-mobile integration guidance.

**Why an index instead of PageIndex?**
PageIndex builds a tree from Markdown headings. That works for arbitrary
documents, but the source data here is a fully-typed UNE 178503 dataset.
Routing it through a Markdown intermediate created brittle line-range
slicing, expensive LLM-summary calls, and lossy navigation. The custom
POI-aware index uses the structure that already exists in the source.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Inventrip API (production)                     │
│                  /v120/pois  ·  /v120/tourist-destinations           │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
        PART 1 — DATA PREPARATION (pipeline/, → Cloudflare cron Worker)
                              ▼
              ┌──────────────────────────────────┐
              │  data/{dest}_pois_raw_{lang}.json│  pipeline/extract_pois.py
              │  data/{dest}_destination_{lang}  │  pipeline/extract_destination_data.py
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  pipeline/build_index.py         │       deterministic, < 1 s
              │  (no LLM calls, no Markdown)     │       no preprocessing wallclock
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  indexes/{dest}_{lang}.json      │ ← THE OFFLINE ARTIFACT
              │  meta · destination_overview     │   (phone downloads this
              │  trips · sections (deterministic │   once, then works with
              │     summaries) · pois · facets · │   no internet connection)
              │  name_index                      │
              └──────────────┬───────────────────┘
                             │
        PART 2 — ASSISTANT (assistant/, → Android/iOS, Gemma 4 E2B on-device)
                             ▼
              ┌──────────────────────────────────┐
              │  assistant/run_eval.py           │       litellm tool calls
              │  assistant/chat_demo.py          │       to oMLX / Ollama
              │                                  │       (reference impl of
              │  Five tools (pure dict lookups): │       the on-device loop)
              │   list_sections, get_section,    │
              │   get_poi, find_poi_by_name,     │
              │   filter_pois                    │
              └──────────────────────────────────┘
```

`common/` (`lang_support.py`, `textnorm.py`) is shared by both parts and
pins the constants/normalisation that the Cloudflare and mobile ports
must reproduce byte-for-byte.

The optional `pipeline/json_to_markdown.py` still exists as a
human-readable export of the same data — useful for reading the corpus
in a text editor — but no script consumes it.

### Tools exposed to the LLM

The model has five tools, each a pure dict lookup against the index
(no I/O, no LLM-in-the-loop):

| Tool | Purpose |
|---|---|
| `list_sections()` | Section catalogue (pre-loaded in the system prompt). |
| `get_section(id, sort, limit)` | List POIs in one section (incl. the schema-v2 group map for large sections), sorted by `(interest_level, zoom_level)`. |
| `get_poi(poi_id)` | Full record of one POI — or several at once with comma-separated ids (`'poi/123,poi/456'`). No truncation, no line slicing. |
| `find_poi_by_name(query, limit)` | Diacritic-insensitive fuzzy lookup by POI name. |
| `filter_pois(interest_level, type, tourist_type, section_id, indispensable, limit)` | Facet query, all filters AND together. |

A typical answer flow:

- **"Tell me about X"** → `find_poi_by_name(X)` → `get_poi(id)`
- **"What museums exist?"** → `get_section("museums-and-culture")`
- **"What should I not miss?"** → `filter_pois(indispensable=true)`
- **"Indispensable food spots"** → `filter_pois(indispensable=true, tourist_type="FOOD TOURISM")`

---

## Data source

POI data comes from the **Inventrip API** (`/v120/pois`), which wraps the
PostgreSQL function `it.get_objects_une_v121`.

- **367 POIs** for Úbeda (production data, English) / 369 in Spanish
- **Per-POI fields** (UNE 178503): `identifier`, `name`, `type`,
  `description`, `extras.id_interest_level` (1=Indispensable, 2, 3),
  `extras.zoom_level` (10–19), `extras.booking_url`, `touristType[]`,
  full address (`streetAddress`/`addressLocality`/`addressProvince`/
  `addressRegion`/`addressCountry`/`postalCode`), `latitude`/`longitude`,
  `telephone`, `email`, `url`, image refs (`image/{id}` →
  `/v100/image/{id}?image_quality=high`), audio guide ids
  (`/v100/audios?audio={id}&...`), `extras.subjectOf[]` documents.
- **Destination metadata** from `/v120/tourist-destinations`: description,
  curated trips with itineraries, paths, interest-level taxonomy,
  tourist-type display-name mapping.
- **Languages**: any code accepted by `/v120/pois?language=` —
  available codes at `/v100/configuration-languages?is_active_app=true`.

---

## Project layout

```
pageindex-demo/
├── AGENTS.md                          ← implementation guide for Warp agents
├── README.md                          ← this file
├── .env                               ← credentials (gitignored)
├── docs/
│   ├── README-mobile.md               ← mobile team: API integration + index schema
│   ├── mobile-offline-contract.md     ← Android/iOS port contract (offline E2B)
│   └── cloudflare-worker-spec.md      ← Cloudflare data-prep + distribution spec
│
├── common/                            ← shared by both parts (port byte-for-byte)
│   ├── lang_support.py                ← 16 languages: rules, recovery, display
│   ├── textnorm.py                    ← normalize_text/tokenize (name search)
│   └── models.py                      ← canonical oMLX model IDs + defaults
│
├── pipeline/                          ← PART 1: data preparation (→ Cloudflare)
│   ├── extract_pois.py                ← Step 1a: fetch POIs (--destination, --lang)
│   ├── extract_destination_data.py    ← Step 1b: fetch trips & taxonomies
│   ├── build_index.py                 ← Step 2: build indexes/{dest}_{lang}.json
│   └── json_to_markdown.py            ← optional human-readable export
│
├── assistant/                         ← PART 2: offline chatbot reference (→ mobile)
│   ├── index_tools.py                 ← five tools, pure dict lookups
│   ├── run_eval.py                    ← Step 3: agentic Q&A evaluation
│   ├── chat_demo.py                   ← interactive / scripted chat demo
│   └── score_results.py               ← Step 4: score grounding + retrieval
│
├── data/                              ← raw API output, tracked in git
│   ├── ubeda_pois_raw_{en,es,it}.json ← /v120/pois output (367–369 POIs)
│   └── ubeda_destination_{en,es,it}.json
│
├── indexes/                           ← build_index.py output, tracked
│   └── ubeda_{en,es,it}.json          ← POI-aware index (~720–820 KB)
│
├── eval/
│   ├── questions.json                 ← 20 curated visitor questions (English)
│   ├── questions_es.json              ← Spanish translations
│   ├── questions_it.json              ← Italian translations
│   └── conversations.json             ← multi-turn conversation threads
│
└── results/                           ← gitignored; eval/conversation outputs
```

### Naming convention

```
data/{destination}_pois_raw_{lang}.json
data/{destination}_destination_{lang}.json
indexes/{destination}_{lang}.json
results/eval_{model}_{lang}.json     (results/ is gitignored)
```

---

## Setup

### Prerequisites

- Python 3.11+
- A local OpenAI-compatible serving stack with at least one Gemma 4 model:
  - **oMLX** (current stack, `http://127.0.0.1:8000/v1`) — models
    `gemma-4-26B-A4B-it-MLX-4bit`, `gemma-4-E4B-it-MLX-4bit`,
    `gemma-4-E2B-it-MLX-8bit` (the mobile deployment target)
  - or [Ollama](https://ollama.com) (`http://localhost:11434/v1`) —
    `ollama pull gemma4:26b` / `gemma4:e4b` / `gemma4:e2b`
- An Inventrip API key (only needed when re-extracting data)

### Installation

```bash
git clone https://github.com/sismotur/pageindex-demo.git
cd pageindex-demo

python3 -m venv .venv
.venv/bin/pip install litellm requests python-dotenv pytest
```

### Environment variables

Create `.env` in the project root:

```bash
# LLM serving (OpenAI-compatible endpoint)
# oMLX (current): API key from ~/.omlx/settings.json → auth.api_key
OPENAI_API_KEY=<omlx-api-key>
OPENAI_API_BASE=http://127.0.0.1:8000/v1

# Ollama alternative:
# OPENAI_API_KEY=ollama
# OPENAI_API_BASE=http://localhost:11434/v1

# Inventrip API (only for pipeline/extract_*.py)
INVENTRIP_API_BASE_URL=https://api.inventrip.com
INVENTRIP_API_KEY=your_api_key_here
```

### Technical configuration

| Component       | Value                                                    |
|-----------------|----------------------------------------------------------|
| Hardware        | Apple Silicon Mac, 128 GB unified memory                 |
| Python          | 3.14 in `.venv`                                          |
| LLM serving     | oMLX on `http://127.0.0.1:8000/v1` (OpenAI-compatible)   |
| LLM client      | `litellm` (`openai/<model-id>` model strings)            |
| Server model    | `openai/gemma-4-26B-A4B-it-MLX-4bit` (MoE, 4B active)    |
| Mobile model    | `openai/gemma-4-E2B-it-MLX-8bit` (the on-device target)  |
| Index rebuild   | `< 1 s` per `(destination, language)` pair              |

---

## Running the pipeline

### Úbeda in English (default)

```bash
# 1a. Fetch POIs from the Inventrip API
.venv/bin/python pipeline/extract_pois.py --destination ubeda --lang en

# 1b. Fetch destination metadata (trips, taxonomies)
.venv/bin/python pipeline/extract_destination_data.py --destination ubeda --lang en

# 2. Build the POI-aware index (deterministic, sub-second, no LLM)
.venv/bin/python pipeline/build_index.py --destination ubeda --lang en
# → indexes/ubeda_en.json

# 3. Run the Q&A evaluation (server model ~10 min; E2B ~75 s on oMLX)
.venv/bin/python assistant/run_eval.py \
  --model openai/gemma-4-26B-A4B-it-MLX-4bit --index indexes/ubeda_en.json

# 4. Score and summarise
.venv/bin/python assistant/score_results.py \
  --file results/eval_gemma-4-26B-A4B-it-MLX-4bit.json

# Optional: interactive chat
.venv/bin/python assistant/chat_demo.py --interactive \
  --model openai/gemma-4-26B-A4B-it-MLX-4bit
```

### Spanish

```bash
.venv/bin/python pipeline/extract_pois.py             --destination ubeda --lang es
.venv/bin/python pipeline/extract_destination_data.py --destination ubeda --lang es
.venv/bin/python pipeline/build_index.py              --destination ubeda --lang es

.venv/bin/python assistant/run_eval.py \
  --model openai/gemma-4-26B-A4B-it-MLX-4bit \
  --questions eval/questions_es.json \
  --index indexes/ubeda_es.json \
  --lang es

.venv/bin/python assistant/chat_demo.py --interactive \
  --model openai/gemma-4-26B-A4B-it-MLX-4bit \
  --index indexes/ubeda_es.json \
  --lang es
```

### Adding a new destination

Replace `caceres` with your destination slug. No code changes required.

```bash
.venv/bin/python pipeline/extract_pois.py             --destination caceres --lang en
.venv/bin/python pipeline/extract_destination_data.py --destination caceres --lang en
.venv/bin/python pipeline/build_index.py              --destination caceres --lang en

.venv/bin/python assistant/run_eval.py \
  --model openai/gemma-4-26B-A4B-it-MLX-4bit \
  --index indexes/caceres_en.json
```

The destination display name is taken from the index's `meta.destination_display`
field, which is sourced from `/v120/tourist-destinations`.

---

## Evaluation design

### Question set

20 questions in `eval/questions.json` across three difficulty tiers and
seven categories (overview, monument lookup, category browse, practical
info, gastronomy, accommodation, events, synthesis). Spanish translations
in `eval/questions_es.json`.

### Scoring rubric

Each answer is scored on four dimensions:

| Dimension          | Weight | Method                                     |
|--------------------|--------|--------------------------------------------|
| Factual grounding  | 40%    | Substring match for verifiable facts       |
| Retrieval accuracy | 30%    | Did the model touch the expected section?  |
| Content fetched    | 20%    | Did it call any retrieval tool?            |
| Language correct   | 10%    | Stop-word ratio / language detection       |

Pass thresholds (from the original plan): `grounding ≥ 70%` AND
`content-fetch ≥ 70%`.

### Sections accessed (rubric input)

`run_eval.py` derives `sections_accessed` from each tool call:

- `get_section(id)` — explicit section id.
- `get_poi(id)` — section that owns the POI (via `facets.by_section`).
- `find_poi_by_name(q)` — sections of the matched POIs.
- `filter_pois(...)` — section_id filter if supplied, otherwise the
  sections of the result POIs.

This replaces the previous heuristic that mapped line ranges to section
titles.

---

## Multilingual notes

- **Supported languages (16 total)** — the same set returned by
  `/v100/configuration-languages?is_active_app=true`. Validated at the
  CLI of every entry point via `common/lang_support.py`:
  - `ca` Catalan, `de` German, `en` English, `es` Spanish, `eu` Basque,
    `fr` French, `gl` Galician, `hi` Hindi, `hr` Croatian, `it` Italian,
    `ja` Japanese, `nl` Dutch, `pt` Portuguese, `ru` Russian,
    `uk` Ukrainian, `zh` Chinese.
  - The 16 codes have a per-language **system-prompt rule** and **recovery
    message** in `common/lang_support.py` (`LANG_RULES`, `RECOVERY_MSGS`).
    Smoke-tested in 26B for Italian; Spanish and English are part of the
    full eval baselines.
- Every artifact name carries a `_{lang}` suffix. Pairs never overwrite.
- The system prompt template ends with a per-language rule from
  `LANG_RULES`. The corpus language is independent — a French question
  over the Spanish corpus works because the model handles cross-lingual
  synthesis.
- Tourist-type display names come from
  `data/{dest}_destination_{lang}.json`'s `tourist_types` map, which is
  a per-language code → label dictionary returned by
  `/v120/tourist-types?language={lang}`.
  `extract_destination_data.py` now picks the requested-language entry
  from each multilingual list (previously it always preferred English).
- Interest-level labels (Indispensable / Interesting / Outstanding /
  their localised equivalents) come from the same file's
  `interest_levels` map.
  - To check which languages your API instance currently exposes:
  `curl "$INVENTRIP_API_BASE_URL/v100/configuration-languages?is_active_app=true&api_key=$INVENTRIP_API_KEY"`.
  If the API list ever drifts from the 16 codes hard-coded here,
  update `common/lang_support.py` (the import-time self-check will
  refuse to load with missing translations).

---

## Open improvements

- **Per-question scoring rubric** — Q08, Q15, Q20 currently lose points
  on substring matches that the model arguably answers correctly.
  Loosening the rubric semantically would lift both languages above
  95% grounding. (Q20's test encodes the accepted floor; see
  `tests/test_rubric.py`.)
- ~~**Hierarchical sub-sections**~~ — **done (schema v2)**: sections
  with > 30 POIs now carry per-type `groups` with key-item summaries
  (the PageIndex Flash `key_items` pattern), so on-device models navigate
  section → group → POIs. PageIndex itself is deliberately not
  re-adopted: the typed 5-tool surface beats its tree search on this
  dataset (95% vs 80% retrieval on 26B), and its LLM-free Flash mode
  independently validates the deterministic-build approach taken here.
- **Cloudflare Worker port of `pipeline/`** — spec in
  `docs/cloudflare-worker-spec.md`; the §4.4 contract test keeps the TS
  build byte-equivalent to the Python reference.
- **Android/iOS ports of `assistant/`** — contract in
  `docs/mobile-offline-contract.md`; verification harness is the existing
  20-question eval.
- **Vector RAG baseline** — the original plan included a parallel
  baseline using `nomic-embed-text` over `###`-bounded chunks for
  side-by-side comparison. Still pending.
- **Typed-tool schema** — `filter_pois` currently accepts string values
  for `tourist_type` / `type`. Constraining the JSON schema to the
  enumerated UNE 178503 codes would reduce model errors on small models.

---

## References

- [Inventrip](https://inventrip.com) — UNE 178503 tourism POI platform
- [UNE 178503](https://www.une.org) — Spanish tourism data standard
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- Original PageIndex experiment: see git history before commit
  `refactor: replace pageindex/ with POI-aware index`.

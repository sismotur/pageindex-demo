# Inventrip Offline Tourism Assistant with Gemma 4

A self-contained framework that answers grounded tourism questions for
**any tourist destination** in **any language** using a custom POI-aware
index (built from the [Inventrip](https://inventrip.com) API) with the
**Gemma 4** family.

This repository is the **offline chatbot reference implementation**:
eleven pure-lookup tools over a pre-built index plus the agentic loop.
Runs against any OpenAI-compatible endpoint (oMLX, Ollama) and will be
**reimplemented in Android/iOS with Gemma 4 E2B fully offline** — the
phone only downloads index files (see `docs/mobile-offline-contract.md`).

Data preparation — fetching the Inventrip API and building the
POI-aware index — lives in the sibling
[`inventrip-rag-data`](https://github.com/sismotur/inventrip-rag-data)
repo, a Cloudflare Python Worker that publishes index and weather files
to R2. This repo keeps small **committed fixture copies** of the built
`indexes/`/`weather/` artifacts for local development and testing;
refresh them by copying output from `inventrip-rag-data` when needed.

The reference destination is Úbeda, Spain — 367 POIs returned by
`/v120/pois` under the [UNE 178503](https://www.une.org) Spanish tourism
standard. The index format supports **multiple destinations** and
**multiple languages** (currently also Sierra de Montánchez y Tamuja,
Spanish-only); all artifacts use the `{destination}_*_{lang}` naming
convention so different `(destination, language)` pairs never overwrite
each other.

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
│    inventrip-rag-data (sibling repo, Cloudflare Python Worker)       │
│    fetches Inventrip API → builds POI-aware index → publishes to R2  │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
              ┌──────────────────────────────────┐
              │  indexes/{dest}_{lang}.json      │ ← THE OFFLINE ARTIFACT
              │  meta · overview · trips · paths │   (phone downloads this
              │  sections (deterministic         │   once, then works with
              │     summaries) · pois · facets · │   no internet connection;
              │  name_index · search_terms       │   committed here as a
              └──────────────┬───────────────────┘   local dev/test fixture)
                             │
        ASSISTANT (assistant/, → Android/iOS, Gemma 4 E2B on-device)
                             ▼
              ┌──────────────────────────────────┐
              │  assistant/run_eval.py           │       litellm tool calls
              │  assistant/chat_demo.py          │       to oMLX / Ollama
              │                                  │       (reference impl of
              │  Eleven tools (pure dict lookups):│       the on-device loop)
              │   list_sections, get_section,    │
              │   get_poi, find_poi_by_name,     │
              │   filter_pois, search_pois,      │
              │   trips + physical paths,        │
              │   get_weather                    │
              └─────────────────────────────────────┘
```

`common/` (`lang_support.py`, `textnorm.py`) is duplicated from the
sibling `inventrip-rag-data` repo — both copies must reproduce the same
constants/normalisation byte-for-byte, since mobile clients depend on it.

### Tools exposed to the LLM

The model has eleven tools, each a pure dict lookup against the
index or the daily weather artifact (no I/O, no LLM-in-the-loop):

| Tool | Purpose |
|---|---|
| `list_sections()` | Section catalogue (pre-loaded in the system prompt). |
| `get_section(id, sort, limit)` | List POIs in one section (incl. the schema-v2 group map for large sections), sorted by `(interest_level, zoom_level)`. |
| `get_poi(poi_id)` | Full record of one POI — or several at once with comma-separated ids (`'poi/123,poi/456'`). No truncation, no line slicing. |
| `find_poi_by_name(query, limit)` | Diacritic-insensitive fuzzy lookup by POI name. |
| `filter_pois(interest_level, type, tourist_type, section_id, indispensable, limit)` | Facet query, all filters AND together. |
| `search_pois(query, section_id, limit)` | Same-record full-text evidence search for compound visitor needs; prevents unsupported claims that one place combines separate concepts. |
| `search_trips(query, limit)` / `get_trip(id)` | Curated theme/day/multi-day suggestions from `/v120/trips`; never presented as routes. |
| `search_paths(query, limit)` / `get_path(id)` | Physical walking/cycling/trail routes from `/v120/paths`; never substitutes a trip. |
| `get_weather(day?)` | Local 7-day forecast (or one day: `today`/`tomorrow`/ISO date/weekday) with a stale-file prefix and localized unavailable fallback. |

A typical answer flow:

- **"Tell me about X"** → `find_poi_by_name(X)` → `get_poi(id)`
- **"What museums exist?"** → `get_section("museums-and-culture")`
- **"What should I not miss?"** → `filter_pois(indispensable=true)`
- **"Indispensable food spots"** → `filter_pois(indispensable=true, tourist_type="FOOD TOURISM")`
- **"Restaurants with olive oil"** → `search_pois("olive oil restaurant")`;
  when no direct record exists, retrieve each concept separately and
  present complementary options without inventing the combination.
- **"What should I do for two days?"** → `search_trips(...)`.
- **"Show me a walking route"** → `search_paths(...)`; if the destination
  supplies no `/paths` record, say so without substituting a trip.

For route requests, the runtime forces exactly one physical-path lookup
before accepting an answer and bounds no-route recovery to one turn. This
prevents the small on-device model from repeating clarification prompts.

All non-social tourist requests also require a current-turn content
retrieval. A selected prior POI/trip/path tag is resolved directly to its
source record; if the model fails to retrieve after one bounded retry, the
assistant returns a localized safe failure instead of ungrounded advice.

---

## Data source

Each index is built from the **Inventrip API** (`/v120/pois`, which wraps
the PostgreSQL function `it.get_objects_une_v121`) by the sibling
`inventrip-rag-data` repo — see that repo for extraction/build details.
This repo only reads the already-built index.

- **367 POIs** for Úbeda (English) / 369 (Spanish); Sierra de Montánchez
  y Tamuja is a smaller, Spanish-only second destination.
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
│   └── mobile-offline-contract.md     ← Android/iOS port contract (offline E2B)
│
├── common/                            ← duplicated from inventrip-rag-data
│   │                                     (port byte-for-byte, keep in sync)
│   ├── lang_support.py                ← 16 languages: rules, recovery, display
│   ├── textnorm.py                    ← normalize_text/tokenize (name search)
│   └── models.py                      ← canonical oMLX model IDs + defaults
│
├── assistant/                         ← offline chatbot reference (→ mobile)
│   ├── index_tools.py                 ← eleven tools, evidence + trip/path retrieval
│   ├── run_eval.py                    ← agentic Q&A evaluation
│   ├── chat_demo.py                   ← interactive / scripted chat demo
│   └── score_results.py               ← score grounding + retrieval
│
├── indexes/                           ← committed fixture copies, refreshed from
│   ├── ubeda_{en,es,it}.json             inventrip-rag-data's build output
│   └── montancheztamuja_es.json
│
├── weather/                           ← same fixture-copy convention as indexes/
│   ├── ubeda_{en,es,it}.json
│   └── montancheztamuja_es.json
│
├── eval/
│   ├── questions.json                 ← 20 curated visitor questions (English)
│   ├── questions_es.json              ← Spanish translations
│   ├── questions_it.json              ← Italian translations
│   ├── conversations.json             ← multi-turn conversation threads (Úbeda)
│   └── conversations_montancheztamuja.json
│
└── results/                           ← gitignored; eval/conversation outputs
```

### Naming convention

```
indexes/{destination}_{lang}.json
weather/{destination}_{lang}.json
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

### Installation

```bash
git clone https://github.com/sismotur/pageindex-demo.git
cd pageindex-demo

python3 -m venv .venv
.venv/bin/pip install litellm python-dotenv pytest
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
```

Re-extracting or rebuilding an index requires an Inventrip API key, but
that only happens in the sibling `inventrip-rag-data` repo now.

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

## Running the assistant

### Úbeda in English (default)

```bash
# Run the Q&A evaluation (server model ~10 min; E2B ~75 s on oMLX)
.venv/bin/python assistant/run_eval.py \
  --model openai/gemma-4-26B-A4B-it-MLX-4bit --index indexes/ubeda_en.json

# Score and summarise
.venv/bin/python assistant/score_results.py \
  --file results/eval_gemma-4-26B-A4B-it-MLX-4bit.json

# Optional: interactive chat
.venv/bin/python assistant/chat_demo.py --interactive \
  --model openai/gemma-4-26B-A4B-it-MLX-4bit
```

### Spanish / other languages

```bash
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

### Adding a new destination or refreshing an index

Run the pipeline in the sibling
[`inventrip-rag-data`](https://github.com/sismotur/inventrip-rag-data)
repo (`src/pipeline/extract_pois.py` → `extract_destination_data.py` →
`build_index.py`), then copy the resulting `indexes/{dest}_{lang}.json`
(and `weather/{dest}_{lang}.json`) here. No code changes are required in
this repo — the destination display name comes from the index's own
`meta.destination_display` field.

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
- Tourist-type display names and interest-level labels (Indispensable /
  Interesting / Outstanding / their localised equivalents) are baked
  into each index's `tourist_type_display` and `interest_levels` maps,
  originally sourced per-language from `/v120/tourist-types` and
  `/v120/interest-levels` by the inventrip-rag-data pipeline.
  If the Inventrip API's supported-language list ever drifts from the
  16 codes hard-coded here, update `common/lang_support.py` in both this
  repo and inventrip-rag-data (the import-time self-check will refuse to
  load with missing translations).

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
- ~~**Compound-request evidence**~~ — **done (schema v3)**:
  `facets.search_terms` and `search_pois` prove all requested concepts
  against the same POI before the assistant claims a relationship; when
  direct evidence is absent, the loop retrieves complementary options
  without inventing one.
- **Cloudflare data-prep Worker** — done, see the sibling
  [`inventrip-rag-data`](https://github.com/sismotur/inventrip-rag-data)
  repo (a Python Cloudflare Worker; not deployed to production yet).
- **POI section taxonomy** — the `SECTIONS` grouping in
  inventrip-rag-data's `pipeline/build_index.py` was extended with 22
  UNE type codes discovered while onboarding Montánchez y Tamuja; a
  couple of low-confidence placements (`Cemetery`, `Street`) are
  provisional and flagged for review.
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

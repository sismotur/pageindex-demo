# Inventrip Offline Tourism Assistant: Multi-Destination, Multi-Language

## ✅ Project status

This repository is the **offline chatbot reference implementation**
only. Data preparation — fetching the Inventrip API and building
`indexes/{dest}_{lang}.json` — was split out into the sibling
[`inventrip-rag-data`](https://github.com/sismotur/inventrip-rag-data)
repo, a **Cloudflare Python Worker** (not a TypeScript port — it
imports the same pipeline code directly) that publishes index and
weather files to R2. See that repo's `docs/cloudflare-worker-spec.md`
for the full data-prep design.

**Hard boundary: all edits stay inside this repository.** Never modify
the sibling `inventrip-rag-data` repo or any other project on disk —
if something looks wrong there, report it instead of fixing it. When
the sibling's build output changes, refresh the fixture copies here;
the sibling itself is only ever changed from its own checkout.

**`assistant/`** is the reference implementation of the on-device
runtime (eleven pure-lookup tools + agentic loop), driven by `litellm`
tool calls against a local OpenAI-compatible server (**oMLX**,
`http://127.0.0.1:8000/v1`). Will be reimplemented in Android/iOS
running **Gemma 4 E2B fully offline** — the phone only downloads index
files; see `docs/mobile-offline-contract.md`.

`common/textnorm.py` is **duplicated** from `inventrip-rag-data` (not
shared via a package/submodule — it's small and low-churn). Both copies,
plus the Cloudflare and mobile ports, must reproduce it byte-for-byte —
it defines how POI names are normalised at build time and matched at
runtime. `common/lang_support.py` and `common/models.py` are
runtime-side configuration and may diverge from the sibling's copies
(the pipeline only needs the language-code list).

Repository: <https://github.com/sismotur/pageindex-demo>
Data-prep repository: <https://github.com/sismotur/inventrip-rag-data>

The project previously used [PageIndex](https://github.com/VectifyAI/PageIndex)
to index a generated Markdown document. That stack has been retired
because the source data is already a fully-typed UNE 178503 dataset and
benefits from direct indexing. A 2026-08 review of upstream PageIndex
(Flash mode, file-system meta-index) confirmed the decision — no
re-adoption planned; the one borrowable idea (hierarchical
sub-sections for very large destinations) is listed under the README's
"Open improvements". See the README's "Quick summary" table for the
before/after metrics.

## Latest results

### Server (gemma4:26b)

| Run                      | Grounding | Retrieval | Composite | Avg latency |
|--------------------------|-----------|-----------|-----------|-------------|
| English (POI-index)      | 90.0%     | 95.0%     | 0.935     | 26.9 s      |
| Spanish (POI-index)      | 90.0%     | 95.0%     | 0.935     | 19.7 s¹     |
| Italian (POI-index)      | 87.5%     | 85.0%     | 0.895     | 24.0 s      |
| English (PageIndex base) | 92.5%     | 80.0%     | 0.910     | 26.5 s      |
| Spanish (PageIndex base) | 80.0%     | 80.0%     | 0.850     | 47.3 s      |

¹ Median latency excluding three Spanish questions where the model
loops; mean including outliers is 132.5 s.

### Offline mobile candidates

| Model         | Disk    | EN comp.  | ES comp.  | IT comp.  | EN lat. | All-lang pass? |
|---------------|---------|-----------|-----------|-----------|---------|----------------|
| Gemma 4 E2B (recommended) | 7.2 GB | **0.850** | **0.830** | **0.760** | 13.5 s | ✅ yes |
| Qwen 2.5 7B (EN-first alt.) | 4.7 GB | 0.835    | 0.710     | 0.720     | 8.6 s  | ❌ ES/IT 5 pp short |
| Qwen 2.5 3B (unsuitable)  | 1.9 GB | 0.745    | 0.590     | 0.525     | 3.0 s  | ❌ |

E2B on the current **oMLX** stack (`gemma-4-E2B-it-MLX-8bit`,
2026-08-16): composite **0.820**, grounding **72.5%**, content-fetch
**95%**, **3.7 s**/question EN — pass at ~4× lower latency than the
Ollama baseline. After the 2026-09-01 grounding-recovery change (model
self-classifies generic overview questions instead of forced retrieval):
composite **0.845**, grounding **77.5%**, content-fetch **90%**,
**3.5 s**/question EN.

The **same** Gemma 4 E2B model that scored 54.1% grounding on the old
PageIndex pipeline scores 85.0% / 77.5% / 72.5% on EN/ES/IT with the
new POI-aware index — above the 70% rubric threshold on every measured
language. Architecture matters more than model size for this task.

The "Sections accessed" rubric input is now derived from the actual
tools called (`get_section`, `get_poi`, `find_poi_by_name`,
`filter_pois`) instead of mapping line ranges to section titles.

---

## Purpose

Answer grounded tourism questions for **any tourist destination** in
**any language** using the Inventrip POI catalogue. The reference dataset
is Úbeda, Spain (367 POIs in English, 369 in Spanish); Sierra de
Montánchez y Tamuja is a smaller, Spanish-only second destination.

## Supported languages

The index format targets the **16 languages** the API exposes under
`/v100/configuration-languages?is_active_app=true`:

`ca` Catalan, `de` German, `en` English, `es` Spanish, `eu` Basque,
`fr` French, `gl` Galician, `hi` Hindi, `hr` Croatian, `it` Italian,
`ja` Japanese, `nl` Dutch, `pt` Portuguese, `ru` Russian,
`uk` Ukrainian, `zh` Chinese.

`common/lang_support.py` is the single source of truth: the supported
codes (default 16, overridable via the `SUPPORTED_LANGS` env var as a
comma-separated list), the native display names used by the chat banner,
and the model-facing text — a pair of English templates
(`lang_rule`/`recovery_msg`) parameterised with the language's English
name, so no per-language translations are needed and any language the
LLM understands works. `run_eval.py` and `chat_demo.py` validate
`--lang` against the list and refuse unknown codes. Visitor-facing
strings stay in per-language i18n tables whose completeness is guarded
by `tests/test_i18n.py` (no missing translations for any listed code).

Decision: `gemma-4-E2B` is the **default model for every entry point**
(`run_eval.py`, `chat_demo.py`) — it is the model the Android/iOS apps
will ship, so all evaluation and chatting runs against the deployment
constraint. `gemma-4-26B` remains available via `--model` as the
server-side quality ceiling. All four Gemma 4 variants (`e2b`, `e4b`,
`26b`, `31b`) fit in this machine's 128 GB unified memory.

---

## Technical decisions

### Source: HTTP API, not direct database

The sibling `inventrip-rag-data` repo's pipeline fetches from the
Inventrip API (`/v120/pois`, wrapping the PostgreSQL function
`it.get_objects_une_v121`) rather than a direct DB connection, to keep
credentials out of this demo environment, follow the same path the
mobile app uses, and respect the production database constraint (no DDL
or risky queries on `inventrip-postgres-f24a92b2`). This repo has no
direct API/DB access at all — it only reads the resulting index files.

### Index format: structured JSON, not Markdown

The sibling repo's pipeline produces a single artifact per
`(destination, language)` pair: `indexes/{destination}_{lang}.json`,
stored in this repo as `indexes/{destination}/{lang}.json`.
The shape is:

```
{
  "meta":                 { destination, lang, poi_count, ... },
  "destination_overview": "...",
  "trips":                [ { trip_id, name, description, steps: [...] } ],
  "sections":             [ { section_id, title, summary, poi_ids } ],
  "pois":                 { "poi/5155": { full record + computed fields } },
  "facets": {
    "by_section":        { section_id: [poi_ids] },
    "by_type":           { "OilMill": [poi_ids], ... },
    "by_tourist_type":   { "FOOD TOURISM": [poi_ids], ... },
    "by_interest_level": { "1": [poi_ids], ... },
    "by_zoom_bucket":    { "<=14": [...], "15-16": [...], "17-19": [...] },
    "indispensable":     [poi_ids]
  },
  "name_index":           { "normalized_name": "poi_id" },
  "tourist_type_display": { code -> human label },
  "interest_levels":      { "1": "Indispensable", "2": ..., "3": ... }
}
```

Each POI value contains the raw API fields plus computed fields:
`display_type`, `display_tourist_types`, `interest_level_label`,
`image_urls` (resolved API URLs), `audio_urls`, `subject_of_urls`,
`country` (ISO code → human name), `normalized_name` (used by
`find_poi_by_name`).

Section titles match the `expected_section` strings in
`eval/ubeda/questions.json`, so the rubric does not need to change.

### Section grouping

Sections are derived deterministically from the POI `type` list. The
priority list (`SECTIONS` in the sibling repo's `pipeline/build_index.py`)
places overlapping types in the most appropriate bucket:

```
UNESCO World Heritage and City Overview      ← WorldHeritageSite, City, ...
Accommodation                                ← LodgingBusiness leaves
Civil and Historical Monuments               ← CivilBuilding, MilitaryBuilding
Religious Heritage                           ← PlaceOfWorship
Museums and Culture                          ← Museum, CultureCenter, ...
Archaeological Sites                         ← ArchaeologicalArea
Tourist Attractions and Viewpoints           ← TouristAttraction, ViewPoint, ...
Squares, Parks and Natural Areas             ← Square, Park, Beach, Landform, ...
Gastronomy                                   ← FoodEstablishment leaves, OilMill, ...
Guided Tours and Itineraries                 ← TouristTrip
Events and Festivals                         ← Event leaves
Shopping                                     ← ShoppingCenter, Store, ...
Tourist Information and Services             ← TouristInformationCenter, ...
Health and Beauty                            ← Pharmacy, clinic, spa (not Hospital)
Emergency Services                           ← Police, Fire, Hospital, CivilProtection
Transport and Access                         ← Airport, stations, Port, Taxi
Practical Information                        ← Parking, GasStation, finance, ...
Sports and Leisure Activities                ← trails, sports, tourist bus/train, ...
Quality, Rules and Visitor Advice            ← Certification, VisitRule, VisitAdvice
Other Points of Interest                     ← fallback only (all /v120/types map to named sections)
```

Accommodation appears before Civil and Historical Monuments so that
dual-typed POIs (e.g. paradores typed as both `Hotel` and `CivilBuilding`)
land in Accommodation.

### Section summaries

Deterministic, computed by `build_section_summary()` in the sibling
repo's `pipeline/build_index.py`. The previous LLM-summary step
(`add_section_summaries.py`) is **gone** — it cost ~8 minutes
per `(destination, language)` pair and had to be re-run after every
Markdown rebuild. Each summary now reports POI count, breakdown by
interest level, top tourist types, and the three notable POIs:

> "30 POIs (3 Indispensable, 6 Interesting, 21 Outstanding). Top
> interests: Architecture, Cultural, Heritage. Notable: Hotel Spa Rosaleda
> de Don Pedro, Hostería Los Cerros, Apartamentos Don Sancho."

### Model backend: oMLX OpenAI-compatible endpoint

`litellm` routes `openai/*` strings to the local oMLX server
(`~/.omlx/settings.json`, port 8000, API-key auth):

- Base URL: `http://127.0.0.1:8000/v1`
- API key: read from `~/.omlx/settings.json` → `auth.api_key`
- Model strings: `openai/gemma-4-E2B-it-MLX-8bit` (the mobile
  deployment target — **default**), `openai/gemma-4-E4B-it-MLX-4bit`,
  `openai/gemma-4-26B-A4B-it-MLX-4bit` (server quality ceiling)

Ollama (`http://localhost:11434/v1`, model strings like
`openai/gemma4:26b`) remains a supported alternative — point
`OPENAI_API_BASE` at it. All Gemma 4 variants fit in this machine's
128 GB unified memory:

| Model           | oMLX id                        | Size   | Context |
|-----------------|--------------------------------|--------|---------|
| Gemma 4 E2B     | `gemma-4-E2B-it-MLX-8bit`      | 7.2 GB | 128K    |
| Gemma 4 E4B     | `gemma-4-E4B-it-MLX-4bit`      | 9.6 GB | 128K    |
| Gemma 4 26B MoE | `gemma-4-26B-A4B-it-MLX-4bit`  | 18 GB  | 256K    |

oMLX accepts but **ignores `tool_choice`** (`"required"` and named
function choice were probed 2026-09-02: no error, plain-text answers
on a greeting where a call was forced). The reference loop still passes
it on the two instruction turns that demand a tool call (trip detail →
named `get_trip`; complementary retrieval → `"required"`) as porting
intent — a no-op on oMLX, effective on LiteRT (`ToolChoice`). Every
other turn stays `"auto"` because those instructions have legitimate
no-tool outcomes; see `docs/mobile-offline-contract.md` §6.

oMLX 0.6.4 does **prefix-cache** shared prompt prefixes server-side
(`cache.enabled`, on by default): a repeated 13K-char system prefix
costs ~0.7 s vs ~3.7 s cold (measured 2026-09-02), which is what makes
the multi-round eval latencies possible. The on-device equivalent is
session/KV reuse — a porting requirement documented in
`docs/mobile-offline-contract.md` §7.2.

---

## Data preparation (sibling repo)

This repo contains **no pipeline code**. Index and weather artifacts are
produced by the sibling
[`inventrip-rag-data`](https://github.com/sismotur/inventrip-rag-data)
repo (`src/pipeline/extract_pois.py` → `extract_destination_data.py` →
`build_index.py`, plus `build_weather.py` — see that repo's AGENTS.md).
To add or refresh a destination, run the pipeline there and copy the
resulting `indexes/{dest}_{lang}.json` here as
`indexes/{dest}/{lang}.json` (one subfolder per destination), and
`weather/{dest}_{lang}.json` as `weather/{dest}/{lang}.json`. No code
changes are needed in this repo — the destination display name comes
from the index's own `meta.destination_display` field.

## Running the assistant

### Run the agentic Q&A evaluation

```bash
# English (default). --model defaults to the E2B mobile model; pass
# --model openai/gemma-4-26B-A4B-it-MLX-4bit for the server ceiling.
.venv/bin/python assistant/run_eval.py --index indexes/ubeda/en.json

# Spanish
.venv/bin/python assistant/run_eval.py \
  --questions eval/ubeda/questions_es.json \
  --index indexes/ubeda/es.json \
  --lang es

# Interactive chat
.venv/bin/python assistant/chat_demo.py --interactive
.venv/bin/python assistant/chat_demo.py --interactive --lang es \
  --index indexes/ubeda/es.json
```

`--structure` is accepted as a deprecated alias for `--index` — when
given an old `results/{name}_structure.json` path, it remaps to
`indexes/{dest}/{lang}.json` if that exists.

### Score and report

```bash
.venv/bin/python assistant/score_results.py --file results/eval_gemma4-26b.json
.venv/bin/python assistant/score_results.py --file results/eval_gemma4-26b_es.json
```

Rubric details in `assistant/score_results.py`. The `_CONTENT_FETCH_TOOLS`
set lists every tool that counts as "the model retrieved real content"
(`get_poi`, `get_section`, `find_poi_by_name`, `filter_pois`,
`search_pois`, `search_trips`, `get_trip`, `search_paths`, `get_path`); legacy
tool names from older result files are also accepted so historical
files still score.

---

## LLM tool surface

Eleven tools, all pure dict lookups against the index (plus the small
weather artifact for `get_weather`). No I/O, no LLM-in-the-loop, no
line slicing.

| Tool | Purpose |
|---|---|
| `list_sections()` | Section catalogue (pre-loaded in the system prompt). |
| `get_section(section_id, sort, limit)` | List POIs in one section (schema v2: large sections include a per-type group map), sorted by `(interest_level, zoom_level)`. |
| `get_poi(poi_id)` | Full record of one POI by id; comma-separated ids fetch several records in one call. |
| `find_poi_by_name(query, limit)` | Diacritic-insensitive fuzzy lookup by name. |
| `filter_pois(interest_level, type, tourist_type, section_id, indispensable, limit)` | Facet query, all filters AND together. |
| `search_pois(query, section_id, limit)` | Same-record full-text evidence search for compound visitor requests. |
| `search_trips(query, limit)` / `get_trip(id)` | Editorial day/theme/multi-day suggestions from `/v120/trips`; never physical routes. |
| `search_paths(query, limit)` / `get_path(id)` | Physical walking/cycling/trail routes from `/v120/paths`; never substitutes a trip. |
| `get_weather(day?)` | Local 7-day forecast (or one day: `today`/`tomorrow`/ISO date/weekday) with a stale-file prefix and a localized unavailable fallback. |

Typical flows handled by the model:

- **"Tell me about X"** → `find_poi_by_name(X)` → `get_poi(id)` → answer
  includes the description paragraph, address, phone, etc.
- **"What X exist?"** → `get_section("…")` → answer from the previews.
- **"Indispensable POIs"** → `filter_pois(indispensable=true)`.
- **"Indispensable food spots"** →
  `filter_pois(indispensable=true, tourist_type="FOOD TOURISM")`.
- **Physical route intent** is guarded deterministically: one
  `search_paths` lookup is forced, then the assistant answers once. This
  prevents repeated clarification/instruction loops on small models.
- **Designation/status questions** ("Why is X a UNESCO World Heritage
  City?") name no POI, so small models misclassify them as generic
  overviews and miss the factual anchors (dates, reasons). Guarded like
  route intent: one forced `search_pois` lookup querying the proper noun
  `unesco` (designation records carry it in every language), then the
  model answers from the results.
- **Strict grounding**: specific questions (named places, facts,
  listings) need a current-turn source-bearing retrieval; prior `<poi>`,
  `<trip>`, and `<path>` tag selections resolve directly to `get_*`.
  Generic overview questions ("what can I see?") may be answered from the
  destination overview and section catalogue preloaded in the system
  prompt — that content is itself index-derived, so no tool call is
  required. When a turn ends without retrieval, the model self-classifies
  once (generic → answer from the catalogue; specific → call a tool; a
  short confirmation like "sí" answering the assistant's own offer → act
  on that offer, never re-ask). If the model still declines, two
  deterministic backstops fire: the destination's own `name_index` is
  probed against the question — and, after a short confirmation, against
  the assistant's previous offer — to fetch the named records; and when
  the model answers with yet another bare clarifying question (on ANY
  turn — warm greeting replies carry "!" markers and are unaffected) or
  repeats its previous answer verbatim (ongoing conversations only;
  `is_repeat_of_previous_answer` compares against the last assistant
  message before the current question, so current-turn drafts do not
  count) — the runtime retrieves content itself
  (`search_pois` on the question, else `filter_pois` indispensable
  highlights) and makes the model present it. A per-turn loop detector
  blocks the third identical `(tool, args)` call (every tool is a
  deterministic read-only lookup, so a repeat can return nothing new),
  corrects the model once with the offending call named, and on any
  further repeat aborts the tool loop so the tail recovery forces a
  final answer — worst-case latency stays bounded instead of burning
  all 14 rounds. The second occurrence of an identical call is not
  blocked but answered from a per-turn result cache: a short stub
  while the original result is still in context, or the cached result
  itself when history compaction has since replaced it — a repeated
  full result no longer doubles the context cost. Every tool call is
  validated against its schema before
  execution: malformed JSON, unknown tools, missing required arguments,
  wrong-typed values, and out-of-enum values return an `[ERROR] …`
  tool result naming the problem (never executing), so the model
  re-issues a corrected call instead of running on silently defaulted
  arguments. On the streaming path (interactive chat), a content-chant
  guard (`chant_repeat_prefix`: the trailing 50-char chunk repeated
  six or more times in the last 2 000 characters) stops a degenerating
  stream and serves the non-repetitive prefix as the answer — at
  temperature=0 re-asking would chant again deterministically. All
  generation calls carry a 1 024-token answer cap (`MAX_ANSWER_TOKENS`),
  so a degenerating non-streaming response stays bounded as well. The
  localized safe failure remains only for turns that never produce an
  answer.

Pre-warm: every section's `get_section(id, "interest", 50)` result is
cached at session start, so subsequent calls are instant.

---

## Project layout

```
pageindex-demo/
├── AGENTS.md                          ← this file
├── README.md
├── docs/
│   ├── README-mobile.md               ← mobile team: API integration + index schema
│   └── mobile-offline-contract.md     ← Android/iOS port contract
│
├── common/                            ← duplicated from inventrip-rag-data,
│   │                                     port byte-for-byte, keep in sync
│   ├── lang_support.py
│   ├── textnorm.py
│   └── models.py                      ← canonical oMLX model IDs + defaults
│
├── assistant/                         ← offline chatbot reference
│   ├── index_tools.py                 ← read-side helpers (eleven tools + itineraries)
│   ├── run_eval.py                    ← agentic eval
│   ├── chat_demo.py                   ← interactive / scripted chat demo
│   └── score_results.py               ← score grounding + retrieval
│
├── indexes/                           ← committed fixture copies, refreshed from
│   │                                     inventrip-rag-data's build output;
│   │                                     one subfolder per destination
│   ├── ubeda/
│   │   ├── en.json
│   │   ├── es.json
│   │   └── it.json
│   └── montancheztamuja/
│       └── es.json
│
├── weather/                           ← same fixture-copy convention as indexes/
│   ├── ubeda/
│   │   ├── en.json
│   │   ├── es.json
│   │   └── it.json
│   └── montancheztamuja/
│       └── es.json
│
├── eval/                              ← one subfolder per destination
│   ├── ubeda/
│   │   ├── questions.json             ← 20 visitor questions (English)
│   │   ├── questions_es.json          ← Spanish translations
│   │   ├── questions_it.json          ← Italian translations
│   │   └── conversations.json         ← multi-turn threads (Úbeda)
│   └── montancheztamuja/
│       └── conversations.json         ← multi-turn threads (Montánchez)
│
├── tests/
│   ├── test_index_tools.py            ← index schema + tool-layer regression tests
│   ├── test_weather.py                ← weather tool + intent detection tests
│   ├── test_multi_destination.py      ← cross-destination isolation tests
│   └── test_rubric.py                 ← rubric + index regression tests
│
└── results/                           ← gitignored
```

### Naming convention

```
indexes/{destination}/{lang}.json
weather/{destination}/{lang}.json
eval/{destination}/questions_{lang}.json    (English: questions.json, no suffix)
eval/{destination}/conversations.json
results/eval_{model}_{lang}.json            (results/ is gitignored)
```

---

## Environment variables

```bash
# LLM inference via oMLX (assistant/run_eval.py, chat_demo.py)
# API key: ~/.omlx/settings.json → auth.api_key
OPENAI_API_KEY=<omlx-api-key>
OPENAI_API_BASE=http://127.0.0.1:8000/v1

# Ollama alternative:
# OPENAI_API_KEY=ollama
# OPENAI_API_BASE=http://localhost:11434/v1
```

Inventrip API credentials (`INVENTRIP_API_BASE_URL`, `INVENTRIP_API_KEY`)
are only needed in the sibling `inventrip-rag-data` repo now.

---

## Key external references

- Ollama Gemma 4 tags: <https://ollama.com/library/gemma4/tags>
- Inventrip API source: `/Users/fsanti/Development/inventrip_api`
  - POI route: `src/modules/v120/pois/routes.js`
  - Params builder: `src/modules/v120/pois/params-builder.js`
  - POI v3 schema: `src/schemas/poi_v3.js`
- Cloudflare data-prep spec: `inventrip-rag-data` repo's
  `docs/cloudflare-worker-spec.md`
- Mobile offline contract: `docs/mobile-offline-contract.md`

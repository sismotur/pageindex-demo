# Inventrip POI-Index RAG: Multi-Destination Tourism Assistant

## ✅ Project status

Production pipeline using a **custom POI-aware index** built directly
from the Inventrip API, with retrieval driven by `litellm` tool calls
to a local `gemma4:26b` model (Ollama / MLX).

Repository: <https://github.com/sismotur/pageindex-demo>

The project previously used [PageIndex](https://github.com/VectifyAI/PageIndex)
to index a generated Markdown document. That stack has been retired
because the source data is already a fully-typed UNE 178503 dataset and
benefits from direct indexing. See the README's "Quick summary" table
for the before/after metrics.

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

The **same** Gemma 4 E2B model that scored 54.1% grounding on the old
PageIndex pipeline scores 85.0% / 77.5% / 72.5% on EN/ES/IT with the
new POI-aware index — above the 70% rubric threshold on every measured
language. Architecture matters more than model size for this task.

Full cross-model report: `results/comparison_table.md`.

The "Sections accessed" rubric input is now derived from the actual
tools called (`get_section`, `get_poi`, `find_poi_by_name`,
`filter_pois`) instead of mapping line ranges to section titles.

---

## Purpose

Answer grounded tourism questions for **any tourist destination** in
**any language** using the Inventrip POI catalogue. The reference dataset
is Úbeda, Spain (367 POIs in English, 369 in Spanish).

## Supported languages

The pipeline targets the **16 languages** the API exposes under
`/v100/configuration-languages?is_active_app=true`:

`ca` Catalan, `de` German, `en` English, `es` Spanish, `eu` Basque,
`fr` French, `gl` Galician, `hi` Hindi, `hr` Croatian, `it` Italian,
`ja` Japanese, `nl` Dutch, `pt` Portuguese, `ru` Russian,
`uk` Ukrainian, `zh` Chinese.

`scripts/lang_support.py` is the single source of truth: it carries one
system-prompt rule and one recovery message per code, plus the native
display name used by the chat banner. All five entry points
(`extract_pois.py`, `extract_destination_data.py`, `build_index.py`,
`run_eval.py`, `chat_demo.py`) validate `--lang` against this list and
refuse unknown codes.

Decision: keep `gemma4:26b` as the recommended model. All four Gemma 4
variants (`e2b`, `e4b`, `26b`, `31b`) fit in this machine's 128 GB
unified memory; escalation is unnecessary for this corpus size.

---

## Technical decisions

### Source: HTTP API, not direct database

The Inventrip `it.get_objects_une_v121` PostgreSQL function is wrapped
by `/v120/pois`. Use that endpoint instead of a direct DB connection
because:

- Keeps credentials out of this demo environment.
- Follows the same path the mobile app uses.
- Respects the production database constraint (no DDL or risky queries
  on `inventrip-postgres-f24a92b2`).

Set `INVENTRIP_API_BASE_URL` and `INVENTRIP_API_KEY` as environment
variables in `.env`. Scripts validate these on startup and fail early
when missing.

### Index format: structured JSON, not Markdown

The pipeline produces a single artifact per `(destination, language)`
pair: `indexes/{destination}_{lang}.json`. The shape is:

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
`eval/questions.json`, so the rubric does not need to change.

### Section grouping

Sections are derived deterministically from the POI `type` list. The
priority list (`SECTIONS` in `scripts/build_index.py`) places overlapping
types in the most appropriate bucket:

```
UNESCO World Heritage and City Overview      ← WorldHeritageSite, City
Accommodation                                ← Hotel, BoutiqueHotel, ...
Civil and Historical Monuments               ← CivilBuilding, MilitaryBuilding
Religious Heritage                           ← PlaceOfWorship
Museums and Culture                          ← Museum, CultureCenter
Archaeological Sites                         ← ArchaeologicalArea
Tourist Attractions and Viewpoints           ← TouristAttraction, ViewPoint
Squares, Parks and Natural Areas             ← Square, Park, LeisureArea
Gastronomy                                   ← Restaurant, OilMill, ...
Guided Tours and Itineraries                 ← TouristTrip
Events and Festivals                         ← BusinessEvent, ...
Shopping                                     ← ShoppingCenter, Store
Tourist Information and Services             ← TouristInformationCenter
Health and Beauty                            ← Pharmacy, ...
Practical Information                        ← ParkingFacility, ...
Sports and Leisure Activities                ← SportsActivityLocation, ...
Quality, Rules and Visitor Advice            ← Certification, VisitRule
Other Points of Interest                     ← fallback
```

Accommodation appears before Civil and Historical Monuments so that
dual-typed POIs (e.g. paradores typed as both `Hotel` and `CivilBuilding`)
land in Accommodation.

### Section summaries

Deterministic, computed by `build_section_summary()` in
`scripts/build_index.py`. The previous LLM-summary step
(`scripts/add_section_summaries.py`) is **gone** — it cost ~8 minutes
per `(destination, language)` pair and had to be re-run after every
Markdown rebuild. Each summary now reports POI count, breakdown by
interest level, top tourist types, and the three notable POIs:

> "30 POIs (3 Indispensable, 6 Interesting, 21 Outstanding). Top
> interests: Architecture, Cultural, Heritage. Notable: Hotel Spa Rosaleda
> de Don Pedro, Hostería Los Cerros, Apartamentos Don Sancho."

### Model backend: Ollama OpenAI-compatible endpoint

`litellm` routes `openai/*` strings to the local Ollama endpoint:

- Base URL: `http://localhost:11434/v1`
- API key: `ollama` (any non-empty string)
- Model string: `openai/gemma4:26b` (recommended) or `openai/gemma4:e4b`

All Gemma 4 variants fit in this machine's 128 GB unified memory:

| Model           | Tag           | Size   | Context |
|-----------------|---------------|--------|---------|
| Gemma 4 E2B     | `gemma4:e2b`  | 7.2 GB | 128K    |
| Gemma 4 E4B     | `gemma4:e4b`  | 9.6 GB | 128K    |
| Gemma 4 26B MoE | `gemma4:26b`  | 18 GB  | 256K    |
| Gemma 4 31B     | `gemma4:31b`  | 20 GB  | 256K    |

---

## Implementation steps

### Step 1 — Extract from the Inventrip API

Two scripts, both accept `--destination` and `--lang`:

```bash
.venv/bin/python scripts/extract_pois.py             --destination ubeda --lang en
.venv/bin/python scripts/extract_destination_data.py --destination ubeda --lang en
```

Outputs:

- `data/{destination}_pois_raw_{lang}.json` — raw `/v120/pois` array.
- `data/{destination}_destination_{lang}.json` — destination overview,
  trips, paths, interest-level taxonomy, tourist-type display-name map.

The relevant query parameters (from `params-builder.js`):

- `tourist_destination` → `filter.name_implan`
- `language` → `id_language`
- `strip_nulls=true` → drop null fields from the response

### Step 2 — Build the POI-aware index

```bash
.venv/bin/python scripts/build_index.py --destination ubeda --lang en
# → indexes/ubeda_en.json   (~720 KB; sub-second; deterministic)
```

`build_index.py` consumes only the two JSON artifacts from Step 1. No
LLM calls. No Markdown intermediate. Re-runnable any time without side
effects.

### Step 3 — Run the agentic Q&A evaluation

```bash
# English (default)
.venv/bin/python scripts/run_eval.py \
  --model openai/gemma4:26b \
  --index indexes/ubeda_en.json

# Spanish
.venv/bin/python scripts/run_eval.py \
  --model openai/gemma4:26b \
  --questions eval/questions_es.json \
  --index indexes/ubeda_es.json \
  --lang es

# Interactive chat
.venv/bin/python scripts/chat_demo.py --interactive --model openai/gemma4:26b
.venv/bin/python scripts/chat_demo.py --interactive --lang es \
  --index indexes/ubeda_es.json
```

`--structure` is accepted as a deprecated alias for `--index` — when
given an old `results/{name}_structure.json` path, it remaps to
`indexes/{name}.json` if that exists.

### Step 4 — Score and report

```bash
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-26b.json
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-26b_es.json
```

Rubric details in `scripts/score_results.py`. The `_CONTENT_FETCH_TOOLS`
set lists every tool that counts as "the model retrieved real content"
(`get_poi`, `get_section`, `find_poi_by_name`, `filter_pois`); legacy
tool names from older result files are also accepted so historical
files still score.

---

## LLM tool surface

Five tools, all pure dict lookups against the index. No I/O, no
LLM-in-the-loop, no line slicing.

| Tool | Purpose |
|---|---|
| `list_sections()` | Section catalogue (pre-loaded in the system prompt). |
| `get_section(section_id, sort, limit)` | List POIs in one section, sorted by `(interest_level, zoom_level)`. |
| `get_poi(poi_id)` | Full record of one POI by id. |
| `find_poi_by_name(query, limit)` | Diacritic-insensitive fuzzy lookup by name. |
| `filter_pois(interest_level, type, tourist_type, section_id, indispensable, limit)` | Facet query, all filters AND together. |

Typical flows handled by the model:

- **"Tell me about X"** → `find_poi_by_name(X)` → `get_poi(id)` → answer
  includes the description paragraph, address, phone, etc.
- **"What X exist?"** → `get_section("…")` → answer from the previews.
- **"Indispensable POIs"** → `filter_pois(indispensable=true)`.
- **"Indispensable food spots"** →
  `filter_pois(indispensable=true, tourist_type="FOOD TOURISM")`.

Pre-warm: every section's `get_section(id, "interest", 50)` result is
cached at session start, so subsequent calls are instant.

---

## Project layout

```
pageindex-demo/
├── AGENTS.md                          ← this file
├── README.md
├── docs/
│   └── cloudflare-worker-spec.md
│
├── data/                              ← raw API output, tracked in git
│   ├── ubeda_pois_raw_en.json
│   ├── ubeda_pois_raw_es.json
│   ├── ubeda_destination_en.json
│   └── ubeda_destination_es.json
│
├── indexes/                           ← build_index.py output, tracked
│   ├── ubeda_en.json
│   └── ubeda_es.json
│
├── eval/
│   ├── questions.json                 ← 20 visitor questions (English)
│   ├── questions_es.json              ← Spanish translations
│   └── conversations.json             ← multi-turn threads for chat_demo
│
├── results/                           ← gitignored
│   ├── eval_gemma4-26b.json
│   ├── eval_gemma4-26b_es.json
│   ├── scored_gemma4-26b.json
│   └── scored_gemma4-26b_es.json
│
└── scripts/
    ├── extract_pois.py                ← Step 1a: fetch POIs
    ├── extract_destination_data.py    ← Step 1b: fetch trips & taxonomies
    ├── build_index.py                 ← Step 2: build POI-aware index
    ├── index_tools.py                 ← read-side helpers
    ├── run_eval.py                    ← Step 3: agentic eval
    ├── chat_demo.py                   ← interactive / scripted chat demo
    ├── score_results.py               ← Step 4: score grounding + retrieval
    └── json_to_markdown.py            ← optional human-readable export
```

### Naming convention

```
data/{destination}_pois_raw_{lang}.json
data/{destination}_destination_{lang}.json
indexes/{destination}_{lang}.json
results/eval_{model}_{lang}.json     (results/ is gitignored)
```

---

## Environment variables

```bash
# Inventrip API (extract_pois.py, extract_destination_data.py)
INVENTRIP_API_BASE_URL=https://api.inventrip.com
INVENTRIP_API_KEY=your_api_key_here

# LLM inference via Ollama (run_eval.py, chat_demo.py)
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1

# Apple Silicon MLX engine — all latency figures use this
OLLAMA_NEW_ENGINE=true
OLLAMA_KV_CACHE_TYPE=q8_0
```

---

## Key external references

- Ollama Gemma 4 tags: <https://ollama.com/library/gemma4/tags>
- Inventrip API source: `/Users/fsanti/Development/inventrip_api`
  - POI route: `src/modules/v120/pois/routes.js`
  - Params builder: `src/modules/v120/pois/params-builder.js`
  - POI v3 schema: `src/schemas/poi_v3.js`
- Cloudflare Worker spec: `docs/cloudflare-worker-spec.md`

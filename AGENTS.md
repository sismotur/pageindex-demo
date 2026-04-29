# PageIndex + Gemma 4: Multi-Destination Tourism RAG

## ✅ Project Complete

**Final result: `gemma4:26b` with enriched corpus, multilingual, multi-destination pipeline**
Repository: https://github.com/sismotur/pageindex-demo

## Progress

| Step | Status | Output |
|------|--------|--------|
| 1 — Clone PageIndex + venv | ✅ Done | `pageindex/`, `.venv/` |
| 2 — Smoke test | ⏭ Deferred (setup confirmed working) | — |
| 3 — Extract Úbeda POIs | ✅ Done | `data/ubeda_pois_raw.json` (408 POIs) |
| 4 — JSON → Markdown | ✅ Done | `data/ubeda_guide.md` (160 KB, 18 sections, 408 POIs) |
| 5 — PageIndex indexing | ✅ Done | `results/ubeda_guide_structure.json` (427 nodes: 1 root, 18 sections, 408 POIs) |
| 6 — Question set | ✅ Done | `eval/questions.json` (20 questions, easy/medium/hard) |
| 7 — Run Q&A eval (E2B) | ✅ Done | `results/eval_gemma4-e2b.json` (649s, 32.5s/q avg) |
| 8 — Score + compare | ✅ Done | `results/scored_gemma4-e2b.json` — E2B: 54.1% grounding, 60% retrieval |
| 9 — Run Q&A eval (E4B) | ✅ Done | `results/eval_gemma4-e4b.json` (1593s, 79.7s/q avg) |
| 10 — Cross-model comparison | ✅ Done | `results/comparison_table.md` — E4B: 82.5% grounding ✅ PASS |
| 11 — Two-level navigation | ✅ Done | `scripts/run_eval.py` rewritten; `get_sections` + `get_poi_list` |
| 12 — Section summaries | ✅ Done | `scripts/add_section_summaries.py`; 18 summaries in structure JSON |
| 13 — E4B re-eval (improved) | ✅ Done | `results/eval_gemma4-e4b.json` (1182s, 59.1s/q avg) |
| 14 — Merge + document | ✅ Done | `master`: 85.9% grounding ✅, −26% latency; README updated |
| 15 — Run Q&A eval (26B) | ✅ Done | `results/eval_gemma4-26b.json` (706s, 35.3s/q avg) |
| 16 — Final comparison | ✅ Done | 26B: 90.0% grounding ✅, 95% retrieval, best on all metrics |
| 17 — Rubric + safety net fixes | ✅ Done | Q12 fixed; rubric corrected; 26B run 2: 90% grounding ✅, 37.5 s/q |
| 18 — Content-based summaries (#3) | ✅ Done | `scripts/add_section_summaries.py` rewritten; 18 summaries regenerated with POI content |
| 19 — Rubric fixes (#1) + tests | ✅ Done | Q03/Q15/Q17/Q20 corrected; `tests/test_rubric.py` 18/18 pass |
| 20 — Cache pre-warm (#6) | ✅ Done | All 18 sections loaded at session start; 50% cache hit rate |
| 21 — Final eval (all improvements) | ✅ Done | 100.0% grounding ✅, 100% retrieval, composite 1.000, 19.7 s/q |
| 22 — Destination corpus (#8) | ✅ Done | `extract_destination_data.py`; 22 trips, Indispensable labels, zoom labels, type names |
| 23 — Final eval (enriched corpus) | ✅ Done | 90.0% grounding ✅, 85% retrieval, composite 0.915, 27.2 s/q |
| 24 — Enrich POI Markdown | ✅ Done | Added coordinates, postal code, country (ISO→name), region, image links, audio guide links, booking URL to `json_to_markdown.py` |
| 25 — Remove destination hardcoding | ✅ Done | All 6 scripts fully destination- and language-agnostic; `--destination` + `--lang` CLI args throughout |
| 26 — Language in artifact filenames | ✅ Done | `{dest}_{type}_{lang}` convention; `ubeda_guide.md` → `ubeda_guide_en.md`; no (dest, lang) pair can overwrite another |
| 27 — Rename extract_ubeda.py | ✅ Done | `scripts/extract_ubeda.py` → `scripts/extract_pois.py` |

---

## Purpose

Evaluate whether **PageIndex** (vectorless, reasoning-based RAG) combined
with **Gemma 4** models served via **Ollama** can answer grounded tourism
questions for **any tourist destination in any language** using real
Inventrip POI data. Reference dataset: Úbeda, Spain.

Start with `gemma4:e2b` (smallest, smartphone-viable). Escalate to `e4b`,
`26b`, and `31b` only if answer quality is insufficient at the smaller size.

---

## Technical Decisions

### Data source: HTTP API, not direct database

The Inventrip `it.get_objects_une_v121` PostgreSQL function is already
wrapped by the API route `/v120/pois`. Use that endpoint instead of a
direct DB connection because:
- keeps credentials out of this demo environment,
- follows the same path the mobile app uses,
- respects the production database constraint (no DDL or risky queries on
  `inventrip-postgres-f24a92b2`).

Set `INVENTRIP_API_BASE_URL` and `INVENTRIP_API_KEY` as environment
variables. Scripts must validate these on startup and fail early if missing.

### Document format: Markdown, not raw JSON

PageIndex's tree-builder anchors on `#` heading hierarchy to identify node
boundaries. A flat JSON blob would produce a single undifferentiated node
with no semantic hierarchy, degrading retrieval quality regardless of model
size. The Markdown document must group POIs into named sections:

```
# Úbeda Tourism Guide
## UNESCO World Heritage Monuments
## Museums and Galleries
## Churches and Religious Heritage
## Gastronomy and Local Food
## Practical Information
```

Retain the raw JSON alongside the Markdown so evaluation can be traced back
to the original source facts.

### Model backend: Ollama OpenAI-compatible endpoint

PageIndex uses `litellm` for LLM calls. Configure it to point at the local
Ollama endpoint:

- Base URL: `http://localhost:11434/v1`
- API key: `ollama` (any non-empty string)
- Model string for litellm: `openai/gemma4:e2b` (or whichever variant)

All four Gemma 4 variants fit in this machine's 128 GB unified memory:

| Model           | Ollama tag       | Size   | Context |
|-----------------|------------------|--------|---------|
| Gemma 4 E2B     | `gemma4:e2b`     | 7.2 GB | 128K    |
| Gemma 4 E4B     | `gemma4:e4b`     | 9.6 GB | 128K    |
| Gemma 4 26B MoE | `gemma4:26b`     | 18 GB  | 256K    |
| Gemma 4 31B     | `gemma4:31b`     | 20 GB  | 256K    |

---

## Implementation Steps

### Step 1 — Clone PageIndex and validate config

```bash
git clone https://github.com/VectifyAI/PageIndex.git pageindex
cd pageindex
pip install -r requirements.txt
```

Inspect `config.yaml` and confirm the model and API base URL fields.
The key fields to override for Ollama:

```yaml
model: openai/gemma4:e2b
```

And set env vars:

```bash
export OPENAI_API_KEY=ollama
export OPENAI_API_BASE=http://localhost:11434/v1
```

### Step 2 — Smoke test: PageIndex + Ollama on a tiny Markdown file

Create `smoke_test.md` (a short, 3-section Markdown document) and run:

```bash
python3 pageindex/run_pageindex.py --md_path smoke_test.md --model openai/gemma4:e2b
```

Confirm the tree JSON is produced in `results/`. If this fails, fix the
Ollama config before proceeding.

### Step 3 — Extract POIs from Inventrip API

Script: `scripts/extract_pois.py` (formerly `extract_ubeda.py`)

Calls `GET /v120/pois?tourist_destination={dest}&language={lang}&strip_nulls=true`.
Accepts `--destination` and `--lang` CLI args (defaults: `ubeda`, `en`).

Saves to `data/{destination}_pois_raw_{lang}.json`.

Also run `scripts/extract_destination_data.py --destination {dest} --lang {lang}`
to fetch trips, tourist types, and interest-level taxonomy.
Saves to `data/{destination}_destination_{lang}.json`.

The relevant query params (from `params-builder.js`):
- `tourist_destination` → `filter.name_implan`
- `language` → `id_language`
- `strip_nulls=true` → removes null fields from output

### Step 4 — Transform JSON into Markdown

Script: `scripts/json_to_markdown.py`

Accepts `--destination` and `--lang` CLI args.
Input:  `data/{destination}_pois_raw_{lang}.json`
Output: `data/{destination}_guide_{lang}.md`

Group POIs by UNE 178503 type into `##` sections. For each POI, emit a
`###` heading with: name, description, address, postal code, country,
region, coordinates, phone, website, booking URL, tourist type tags,
image links, and audio guide links.

### Step 5 — Run PageIndex indexing

Two modes:

**Fast (structural only, no LLM calls — recommended):**
Builds the tree from heading hierarchy alone. Instant, deterministic.

```bash
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide_en.md \
  --model openai/gemma4:26b \
  --if-add-node-summary no \
  --if-add-doc-description no
# → produces results/ubeda_guide_en_structure.json
```

Output filename mirrors the input Markdown name: `{dest}_guide_{lang}_structure.json`.
The `results/` directory is created automatically if it does not exist.
The `.env` file in the project root is loaded automatically (OPENAI_API_KEY + OPENAI_API_BASE).

After indexing, run section summaries (one-time per (destination, language)):

```bash
.venv/bin/python scripts/add_section_summaries.py \
  --structure results/ubeda_guide_en_structure.json --lang en
```

### Step 6 — Define the evaluation question set

File: `eval/questions.json`

A curated list of ~20 questions covering:
- UNESCO monuments (simple factual lookup)
- Opening hours and visit logistics
- Museums and cultural venues
- Churches and religious heritage
- Gastronomy and local food
- Short itinerary planning (light reasoning)
- Direct POI lookup by name

Mix of simple retrieval and light synthesis questions to distinguish
retrieval quality (PageIndex) from generation quality (Gemma 4).

### Step 7 — Run Q&A and capture results

Script: `scripts/run_eval.py`

Key options: `--model`, `--questions`, `--structure`, `--lang`.
The destination name and Markdown path are derived automatically from the
structure file. Run in escalation order:

```bash
# English eval
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e2b
.venv/bin/python scripts/run_eval.py --model openai/gemma4:26b   # recommended

# Spanish eval
.venv/bin/python scripts/run_eval.py \
  --model openai/gemma4:26b \
  --questions eval/questions_es.json \
  --structure results/ubeda_guide_es_structure.json \
  --lang es

# Interactive chat (any language)
.venv/bin/python scripts/chat_demo.py --interactive --lang es
```

### Step 8 — Evaluate and compare

Script: `scripts/score_results.py`

Applies the shared rubric to each `results/eval_*.json`:

| Dimension          | Pass threshold                              |
|--------------------|---------------------------------------------|
| Factual grounding  | Claim traceable to a POI in the source JSON |
| Completeness       | Covers what the question asks               |
| Hallucination rate | < 20 % of answers add unsupported facts     |
| Retrieval accuracy | Correct section surfaced by PageIndex       |
| Latency            | Wall-clock seconds per question             |

**Stop escalating when**: ≥ 70 % of answers are factually grounded AND
hallucination rate < 20 %.

Produces `results/comparison_table.md` — a cross-model summary.

---

## Project Layout

```
pageindex-demo/
├── AGENTS.md                        ← this file
├── README.md
├── docs/
│   └── cloudflare-worker-spec.md
├── pageindex/                       ← cloned VectifyAI/PageIndex repo
├── data/                            ← {dest}_{type}_{lang} naming
│   ├── ubeda_pois_raw_en.json   ← 367 POIs from Inventrip API
│   ├── ubeda_destination_en.json← trips, tourist types, interest levels
│   └── ubeda_guide_en.md        ← structured Markdown (240 KB, 20 sections)
├── eval/
│   ├── questions.json           ← 20 English visitor questions
│   ├── questions_es.json        ← 20 Spanish visitor questions
│   └── conversations.json       ← multi-turn conversation threads
├── results/
│   ├── ubeda_guide_en_structure.json  ← PageIndex tree index (gitignored)
│   └── eval_gemma4-26b.json           ← Q&A results
└── scripts/
    ├── extract_pois.py          ← Step 3a: fetch POIs (--destination, --lang)
    ├── extract_destination_data.py  ← Step 3b: fetch trips & metadata
    ├── json_to_markdown.py      ← Step 4: JSON → Markdown (--destination, --lang)
    ├── add_section_summaries.py ← Step 5: LLM section summaries (--structure, --lang)
    ├── run_eval.py              ← Step 6: PageIndex Q&A runner
    ├── score_results.py         ← Step 7: scoring and comparison
    └── chat_demo.py             ← interactive / scripted conversation demo
```

---

## Environment Variables Required

```bash
# Data extraction (extract_pois.py, extract_destination_data.py)
INVENTRIP_API_BASE_URL=https://api.inventrip.com
INVENTRIP_API_KEY=your_api_key_here

# LLM inference via Ollama (all eval/chat/summary scripts)
OPENAI_API_KEY=ollama                  # literal string
OPENAI_API_BASE=http://localhost:11434/v1

# Apple Silicon MLX engine (all latency figures use this)
OLLAMA_NEW_ENGINE=true
OLLAMA_KV_CACHE_TYPE=q8_0
```

---

## Key External References

- PageIndex repo: <https://github.com/VectifyAI/PageIndex>
- Ollama Gemma 4 tags: <https://ollama.com/library/gemma4/tags>
- Inventrip API source: `/Users/fsanti/Development/inventrip_api`
  - POI route: `src/modules/v120/pois/routes.js`
  - Params builder: `src/modules/v120/pois/params-builder.js`

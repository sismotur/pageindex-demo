# PageIndex + Gemma 4 (E2B → 31B): Úbeda Tourism RAG Demo

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

---

## Purpose

Evaluate whether **PageIndex** (vectorless, reasoning-based RAG) combined
with **Gemma 4** models served via **Ollama** can answer grounded tourism
questions about Úbeda using real Inventrip POI data.

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

### Step 3 — Extract Úbeda POIs from Inventrip API

Script: `scripts/extract_ubeda.py`

Calls `GET /v120/pois?tourist_destination=ubeda&language=en&strip_nulls=true`
against `$INVENTRIP_API_BASE_URL` with `Authorization: Bearer $INVENTRIP_API_KEY`.

Saves result to `data/ubeda_pois_raw.json`.

The relevant query params (from `params-builder.js`):
- `tourist_destination` → `filter.name_implan`
- `language` → `id_language`
- `strip_nulls=true` → removes null fields from output

### Step 4 — Transform JSON into Markdown

Script: `scripts/json_to_markdown.py`

Input: `data/ubeda_pois_raw.json`
Output: `data/ubeda_guide.md`

Group POIs by UNE 178503 type into `##` sections. For each POI, emit a
`###` heading with name, then description, address, opening hours, and
tourist tags as bullet points. This gives PageIndex a clean hierarchy to
index.

### Step 5 — Run PageIndex indexing

Two modes:

**Fast (structural only, no LLM calls — recommended first):**
Builds the tree from heading hierarchy alone. Instant, deterministic.

```bash
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide.md \
  --model openai/gemma4:e2b \
  --if-add-node-summary no \
  --if-add-doc-description no
```

**Full (with LLM-generated summaries — ~10-12 min on E2B):**
Calls `gemma4:e2b` once per node above the token threshold.
Summaries improve navigation in the agentic retrieval step.

```bash
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide.md \
  --model openai/gemma4:e2b \
  --if-add-node-summary yes \
  --if-add-doc-description no
```

Output: `results/ubeda_guide_structure.json` — the hierarchical tree index.
The `results/` directory is created automatically if it does not exist.
The `.env` file in the project root is loaded automatically (OPENAI_API_KEY + OPENAI_API_BASE).

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

Uses `PageIndexClient` (same pattern as `examples/agentic_vectorless_rag_demo.py`).
For each question:
1. Ask the model using the indexed document.
2. Record the answer, retrieved sections/line ranges, and wall-clock latency.
3. Append to `results/eval_{model_tag}.json`.

Run with each model in escalation order:

```bash
python3 scripts/run_eval.py --model openai/gemma4:e2b
python3 scripts/run_eval.py --model openai/gemma4:e4b   # only if E2B is poor
python3 scripts/run_eval.py --model openai/gemma4:26b   # only if E4B is poor
python3 scripts/run_eval.py --model openai/gemma4:31b   # only if 26B is poor
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
├── AGENTS.md                  ← this file
├── pageindex/                 ← cloned VectifyAI/PageIndex repo
├── data/
│   ├── ubeda_pois_raw.json    ← raw output from Inventrip API
│   └── ubeda_guide.md         ← structured Markdown for PageIndex
├── eval/
│   └── questions.json         ← curated visitor question set
├── results/
│   ├── ubeda_guide_structure.json   ← PageIndex tree index
│   ├── eval_gemma4-e2b.json         ← Q&A results per model
│   └── comparison_table.md          ← cross-model rubric summary
└── scripts/
    ├── extract_ubeda.py       ← Step 3: fetch from Inventrip API
    ├── json_to_markdown.py    ← Step 4: convert JSON to Markdown
    ├── run_eval.py            ← Step 7: PageIndex Q&A runner
    └── score_results.py       ← Step 8: scoring and comparison
```

---

## Environment Variables Required

```bash
INVENTRIP_API_BASE_URL=https://api.inventrip.com   # or staging URL
INVENTRIP_API_KEY=...                               # your API key
OPENAI_API_KEY=ollama                               # literal string
OPENAI_API_BASE=http://localhost:11434/v1           # local Ollama endpoint
```

---

## Key External References

- PageIndex repo: <https://github.com/VectifyAI/PageIndex>
- Ollama Gemma 4 tags: <https://ollama.com/library/gemma4/tags>
- Inventrip API source: `/Users/fsanti/Development/inventrip_api`
  - POI route: `src/modules/v120/pois/routes.js`
  - Params builder: `src/modules/v120/pois/params-builder.js`

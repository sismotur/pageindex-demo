# PageIndex + Gemma 4: Úbeda Tourism RAG Demo

A self-contained experiment that evaluates whether
[PageIndex](https://github.com/VectifyAI/PageIndex) (vectorless,
reasoning-based RAG) combined with Google's **Gemma 4** family served
via **Ollama** can answer grounded tourism questions about Úbeda, Spain,
using real POI data from the [Inventrip](https://inventrip.com) API.

The experiment starts with the smallest model (`gemma4:e2b`, 7.2 GB —
smartphone-viable) and escalates to `e4b`, `26b`, and `31b` only when
the smaller model fails the quality threshold.

---

## What is PageIndex?

PageIndex builds a hierarchical tree index from a document's heading
structure and uses an LLM to **navigate** that tree (like a human
flipping through a book) rather than relying on vector similarity search
or chunking. No embedding model, no vector database — just a structured
index and a reasoning model.

The retrieval loop:

1. Agent reads `get_document_structure()` → sees all section and POI
   titles with line numbers.
2. Agent identifies the relevant section and calls
   `get_page_content(lines="start-end")` with a tight range.
3. Agent synthesises an answer from the retrieved text.

---

## Data Source

POI data comes from the **Inventrip API** (`/v120/pois`), which wraps
the PostgreSQL function `it.get_objects_une_v121`. The data is
structured according to the **UNE 178503** Spanish tourism standard.

- **408 POIs** for the Úbeda tourist destination
- Fields: name, type (UNE 178503), description, address, phone,
  website, tourist type tags, and coordinates
- Languages: English (`language=en`)

The raw JSON is converted into a structured **Markdown document** (not
fed as raw JSON) because PageIndex's tree-builder anchors on `#` heading
hierarchy. A flat JSON blob would produce a single undifferentiated node
with no semantic hierarchy.

---

## Project Layout

```
pageindex-demo/
├── AGENTS.md                     ← implementation guide (for Warp agents)
├── README.md                     ← this file
├── .env                          ← credentials (gitignored)
├── .gitignore
├── pageindex/                    ← cloned VectifyAI/PageIndex (gitignored)
├── .venv/                        ← Python virtual environment (gitignored)
│
├── data/
│   ├── ubeda_pois_raw.json       ← 408 POIs from Inventrip API (gitignored)
│   └── ubeda_guide.md            ← structured Markdown for PageIndex (gitignored)
│
├── eval/
│   └── questions.json            ← 20 curated visitor questions
│
├── results/
│   ├── ubeda_guide_structure.json     ← PageIndex tree (427 nodes, gitignored)
│   ├── eval_gemma4-e2b.json           ← raw Q&A results (gitignored)
│   └── scored_gemma4-e2b.json         ← scored results (gitignored)
│
└── scripts/
    ├── extract_ubeda.py          ← Step 1: fetch POIs from Inventrip API
    ├── json_to_markdown.py       ← Step 2: convert JSON → structured Markdown
    ├── run_eval.py               ← Step 3: litellm agentic tool-calling eval
    └── score_results.py          ← Step 4: score grounding + retrieval
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) with at least `gemma4:e2b` pulled:

```bash
ollama pull gemma4:e2b
```

- An Inventrip API key (for data extraction only — not needed to re-run
  the eval if you already have `data/ubeda_pois_raw.json`)

### Installation

```bash
git clone https://github.com/felipesanti/pageindex-demo.git
cd pageindex-demo

# Clone PageIndex
git clone https://github.com/VectifyAI/PageIndex.git pageindex

# Create venv and install dependencies
python3 -m venv .venv
grep -v "python-dotenv" pageindex/requirements.txt | \
  .venv/bin/pip install -r /dev/stdin requests
```

### Environment variables

Create `.env` in the project root:

```bash
# Ollama (OpenAI-compatible endpoint)
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1

# Inventrip API (only needed for scripts/extract_ubeda.py)
INVENTRIP_API_BASE_URL=https://stgapi.inventrip.com
INVENTRIP_API_KEY=your_api_key_here
```

---

## Running the Pipeline

```bash
# 1. Fetch Úbeda POIs from the Inventrip API
.venv/bin/python scripts/extract_ubeda.py

# 2. Convert to structured Markdown (18 sections, 408 POI entries)
.venv/bin/python scripts/json_to_markdown.py

# 3. Build the PageIndex tree (deterministic, no LLM calls)
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide.md \
  --model openai/gemma4:e2b \
  --if-add-node-summary no \
  --if-add-doc-description no

# 4. Run the Q&A evaluation (20 questions, ~10 min on E2B)
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e2b

# 5. Score and summarise
.venv/bin/python scripts/score_results.py
```

---

## Evaluation Design

### Question set (`eval/questions.json`)

20 questions across 3 difficulty tiers and 7 categories:

| Category | Count | Example |
|---|---|---|
| Overview / synthesis | 4 | "What makes Úbeda different from other Spanish cities?" |
| Monument / POI lookup | 5 | "Tell me about the Ariza Bridge." |
| Category browse | 5 | "What museums can I visit?" |
| Practical info | 3 | "Where can I park in Úbeda?" |
| Gastronomy | 2 | "Is Úbeda known for olive oil?" |
| Accommodation | 1 | "Is there a parador?" |
| Events | 1 | "What festivals take place in Úbeda?" |

### Scoring rubric

Each answer is scored on four dimensions:

| Dimension | Weight | Method |
|---|---|---|
| Factual grounding | 40% | Substring match for verifiable facts |
| Retrieval accuracy | 30% | Expected section accessed? |
| Content fetched | 20% | `get_page_content` called? |
| Language correct | 10% | Spanish stop-word ratio check |

**Pass thresholds (from plan):** ≥ 70% grounding AND ≥ 70% content-fetch rate.

---

## Results: Gemma 4 E2B

**Overall: ❌ FAIL — escalation to E4B recommended**

| Metric | Score | Threshold |
|---|---|---|
| Factual grounding | 54.1% | ≥ 70% ❌ |
| Retrieval accuracy | 60.0% (12/20) | — |
| Content fetched | 75.0% (15/20) | ≥ 70% ✅ |
| Composite score | 0.637 | — |
| Avg latency | 32.5 s / question | — |
| Total runtime | 649 s (20 questions) | — |

### Per-difficulty breakdown

| Difficulty | Questions | Composite |
|---|---|---|
| Easy | 10 | 0.630 |
| Medium | 7 | 0.671 |
| Hard | 3 | 0.577 |

### Per-question detail

| ID | Category | Ground | Retrieval | Fetched | Score | Notes |
|---|---|---|---|---|---|---|
| Q01 | overview | 1.00 | 1.0 | 1.0 | 1.000 | |
| Q02 | monument_lookup | 0.50 | 1.0 | 1.0 | 0.800 | missed "savior" |
| Q03 | poi_direct_lookup | 0.00 | 0.0 | 1.0 | 0.300 | wrong section (Parador in Civil Monuments) |
| Q04 | category_browse | 0.00 | 0.0 | 1.0 | 0.300 | navigated to Civil Buildings, not Museums |
| Q05 | category_browse | 0.50 | 1.0 | 1.0 | 0.800 | missed Guadalupe |
| Q06 | poi_direct_lookup | 0.00 | 1.0 | 1.0 | 0.600 | retrieved right section; failed to extract facts |
| Q07 | practical_info | 0.00 | 0.0 | 0.0 | 0.100 | skipped get\_page\_content |
| Q08 | practical_info | 1.00 | 1.0 | 1.0 | 1.000 | |
| Q09 | gastronomy | 1.00 | 0.0 | 0.0 | 0.400 | answered in Spanish |
| Q10 | gastronomy | 1.00 | 0.0 | 0.0 | 0.500 | answered from structure only |
| Q11 | accommodation | 1.00 | 1.0 | 1.0 | 1.000 | |
| Q12 | heritage | 1.00 | 1.0 | 1.0 | 1.000 | |
| Q13 | events | 0.50 | 1.0 | 1.0 | 0.800 | missed "Semana Santa" |
| Q14 | category_browse | 1.00 | 1.0 | 1.0 | 1.000 | |
| Q15 | poi_direct_lookup | 0.00 | 0.0 | 1.0 | 0.300 | retrieved Tourist Attractions, not Archaeological |
| Q16 | practical_info | 0.50 | 0.0 | 0.0 | 0.300 | answered from titles only |
| Q17 | category_browse | 0.50 | 1.0 | 1.0 | 0.800 | missed "itinerary" |
| Q18 | synthesis | 0.33 | 0.0 | 0.0 | 0.232 | skipped get\_page\_content |
| Q19 | synthesis | 1.00 | 1.0 | 1.0 | 1.000 | |
| Q20 | synthesis | 0.00 | 1.0 | 1.0 | 0.500 | answered in Spanish |

### Failure analysis

Two distinct root causes were identified:

**1. Retrieval failures** — the model navigated to the wrong section
or skipped `get_page_content` entirely (Q03, Q04, Q07, Q09, Q10, Q15,
Q16, Q18). This is partly a data-preparation issue: the Condestable
Dávalos Parador has both `Hotel` and `CivilBuilding` type tags; because
`CivilBuilding` has priority in the section-assignment map, it was
placed under "Civil and Historical Monuments" instead of
"Accommodation". A more robust priority ordering or a multi-section
assignment strategy would fix Q03 and Q04.

**2. Generation failures** — the model retrieved the right content but
failed to extract specific facts or responded in Spanish (Q06, Q09, Q20).
These are pure model-capability failures: E2B's smaller parameter count
means weaker instruction-following and fact extraction on detailed text.

---

## Escalation Plan

To run the next model:

```bash
ollama pull gemma4:e4b
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e4b
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-e4b.json
```

If E4B also fails, escalate to `gemma4:26b` (MoE, 18 GB), then
`gemma4:31b` (dense, 20 GB). All four models fit in this machine's
128 GB unified memory.

---

## Known Issues / Improvements

- **Section priority bug**: POIs with both accommodation and
  civil-building type tags (e.g. the Parador) land in the wrong section.
  Fix: swap the order of `Hotel` and `CivilBuilding` in the `SECTIONS`
  map in `scripts/json_to_markdown.py`, or allow multi-section assignment.
- **No LLM summaries**: the PageIndex tree was built without
  node summaries (`--if-add-node-summary no`) for speed. Adding summaries
  (~10 min on E2B) would give the agent richer context for navigation
  decisions, potentially improving retrieval accuracy.
- **Language drift**: E2B occasionally answers in Spanish despite an
  English system prompt. Adding an explicit instruction ("Always respond
  in English") in the system prompt should address this.

---

## References

- [PageIndex](https://github.com/VectifyAI/PageIndex) — VectifyAI
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- [Inventrip](https://inventrip.com) — UNE 178503 tourism POI platform
- [UNE 178503](https://www.une.org) — Spanish tourism data standard

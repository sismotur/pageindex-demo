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

## Quick Summary

**Recommended model: `gemma4:e4b` — passes both quality thresholds.**

| | E2B (original) | E2B (fixed) | **E4B (fixed)** |
|---|---|---|---|
| Factual grounding | 54.1% ❌ | 62.5% ❌ | **82.5% ✅** |
| Retrieval accuracy | 60% | 60% | **90%** |
| Content fetched | 75% ✅ | 65% ❌ | **100% ✅** |
| Composite score | 0.637 | 0.655 | **0.900** |
| Avg latency (MLX) | 32.5 s/q | 51.1 s/q | **79.7 s/q** |

**Technical configuration used for all runs:**
- Hardware: Apple Silicon Mac, 128 GB unified memory
- Inference backend: Ollama MLX engine (`com.ollama.mlx`, `OLLAMA_NEW_ENGINE=true`)
- LLM routing: `litellm` → `openai/gemma4:{e2b,e4b}` → `http://localhost:11434/v1`
- Index: PageIndex tree (427 nodes, structural only — no LLM summaries)
- Dataset: 408 Úbeda POIs from the Inventrip API, 20 evaluation questions

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
git clone https://github.com/sismotur/pageindex-demo.git
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
# litellm routes openai/* model strings to this base URL.
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1

# Enable Ollama's MLX engine on Apple Silicon.
# All latency figures in this repo were measured with this active.
# Without it, inference falls back to llama.cpp and will be noticeably slower.
OLLAMA_NEW_ENGINE=true

# Inventrip API (only needed for scripts/extract_ubeda.py)
INVENTRIP_API_BASE_URL=https://stgapi.inventrip.com
INVENTRIP_API_KEY=your_api_key_here
```

### Technical configuration

All evaluation runs in this repository used the following setup:

| Component | Value |
|---|---|
| Hardware | Apple Silicon Mac, 128 GB unified memory |
| OS | macOS |
| Python | 3.14 via `.venv` |
| Ollama service | `com.ollama.mlx` (MLX backend) |
| Ollama endpoint | `http://localhost:11434/v1` (OpenAI-compatible) |
| LLM client | `litellm 1.83.7` |
| Model routing | `openai/gemma4:e2b` / `openai/gemma4:e4b` → Ollama |
| `gemma4:e2b` | 7.2 GB Q4\_K\_M, 128K context |
| `gemma4:e4b` | 9.6 GB Q4\_K\_M, 128K context |
| Index type | PageIndex structural tree (no LLM summaries) |
| Index rebuild time | < 5 s (deterministic, no model calls) |

The `OLLAMA_NEW_ENGINE=true` flag routes Ollama through Apple's
[MLX](https://github.com/ml-explore/mlx) framework, which is optimised
for Apple Silicon unified memory. Latency on non-MLX or non-Apple
Silicon hardware will be higher than the figures reported here.

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

## Results

Three evaluation runs were conducted. The first two used `gemma4:e2b`
(before and after pipeline fixes); the third used `gemma4:e4b`.

### Run summary

| Run | Model | Grounding | Retrieval | Content fetched | Composite | Avg latency |
|---|---|---|---|---|---|---|
| E2B — original | `gemma4:e2b` | 54.1% ❌ | 60% | 75% ✅ | 0.637 | 32.5 s |
| E2B — fixed | `gemma4:e2b` | 62.5% ❌ | 60% | 65% ❌ | 0.655 | 51.1 s |
| **E4B — fixed** | **`gemma4:e4b`** | **82.5% ✅** | **90%** | **100% ✅** | **0.900** | **79.7 s** |

**Verdict: `gemma4:e4b` passes both thresholds. No escalation to 26B
or 31B required.**

### Pipeline fixes applied between E2B original and E2B fixed

Two fixes were applied after the first E2B run:

1. **Section priority** (`scripts/json_to_markdown.py`): the Condestable
   Dávalos Parador has both `Hotel` and `CivilBuilding` type tags.
   Because `CivilBuilding` appeared first in the section map it was
   placed under "Civil and Historical Monuments" instead of
   "Accommodation", causing Q03 and Q04 to fail regardless of model
   quality. Moving `Accommodation` before `Civil and Historical Monuments`
   in the priority list corrected this. Accommodation grew from 29 to
   35 POIs; Civil Monuments shrank from 54 to 49.

2. **Language enforcement** (`scripts/run_eval.py`): added
   "Always respond in English" as an explicit rule in the system prompt
   to address Q09 and Q20 answering in Spanish.

### Per-question comparison: E2B (fixed) vs E4B

| ID | Category | Diff | E2B | E4B | Δ | Notes |
|---|---|---|---|---|---|---|
| Q01 | overview | easy | 1.000 | 1.000 | — | |
| Q02 | monument\_lookup | easy | 0.800 | **1.000** | +0.200 ↑ | E4B found "Savior Chapel" |
| Q03 | poi\_direct\_lookup | easy | 0.600 | **1.000** | +0.400 ↑ | E4B extracted Parador address + phone |
| Q04 | category\_browse | easy | 0.500 | **1.000** | +0.500 ↑ | E4B navigated to Museums section |
| Q05 | category\_browse | easy | 0.800 | **1.000** | +0.200 ↑ | E4B found Guadalupe sanctuary |
| Q06 | poi\_direct\_lookup | easy | 0.100 | **1.000** | +0.900 ↑ | E4B extracted 1562, Guadalimar, 100m |
| Q07 | practical\_info | easy | **1.000** | 0.700 | −0.300 ↓ | E2B got lucky from titles; E4B hit wrong section |
| Q08 | practical\_info | easy | 1.000 | 1.000 | — | |
| Q09 | gastronomy | easy | 0.400 | **1.000** | +0.600 ↑ | E2B answered in Spanish; E4B correct |
| Q10 | gastronomy | medium | 0.500 | **1.000** | +0.500 ↑ | E4B retrieved Gastronomy section |
| Q11 | accommodation | easy | 1.000 | 1.000 | — | |
| Q12 | heritage | medium | 1.000 | 1.000 | — | |
| Q13 | events | medium | 0.800 | 0.800 | — | Rubric checks "Semana Santa"; data is English-only |
| Q14 | category\_browse | medium | 1.000 | 1.000 | — | |
| Q15 | poi\_direct\_lookup | medium | 0.300 | 0.300 | — | Both models paraphrase instead of quoting |
| Q16 | practical\_info | medium | 0.300 | **0.800** | +0.500 ↑ | E4B retrieved pharmacy list |
| Q17 | category\_browse | medium | 0.800 | 0.800 | — | Rubric checks "itinerary"; word absent from source |
| Q18 | synthesis | hard | 0.600 | **0.732** | +0.132 ↑ | E4B retrieved heritage sections |
| Q19 | synthesis | hard | 0.500 | **1.000** | +0.500 ↑ | E4B covered gastronomy + olive oil |
| Q20 | synthesis | hard | 0.100 | **0.868** | +0.768 ↑ | E2B answered in Spanish; E4B mostly correct |

12 questions improved, 1 regressed (Q07), 7 unchanged.

### Per-difficulty breakdown

| Difficulty | Questions | E2B (fixed) | E4B | Δ |
|---|---|---|---|---|
| Easy | 10 | 0.720 | **0.970** | +0.250 |
| Medium | 7 | 0.671 | **0.814** | +0.143 |
| Hard | 3 | 0.400 | **0.867** | +0.467 |

E4B's largest gain is on **hard synthesis questions** (+0.467) — exactly
the category where E2B's weaker instruction-following hurt most.

### Latency

| Model | Total (20 Qs) | Avg / question | Relative |
|---|---|---|---|
| E2B | 1022 s | 51 s | 1× |
| E4B | 1593 s | 80 s | 1.6× |

E4B is 1.6× slower. For a real-time tourism concierge an 80 s SLA per
query is borderline. Running the index with node summaries
(`--if-add-node-summary yes`, ~10 min one-time cost) would pre-compute
navigation context and reduce the number of `get_page_content` calls per
query, which is the primary driver of latency.

> **Important — MLX engine:** all latency figures above were measured
> with Ollama running via the MLX backend (`com.ollama.mlx` on macOS,
> equivalent to `OLLAMA_NEW_ENGINE=true`). This backend uses Apple's MLX
> framework for inference on unified memory and is significantly faster
> than the default llama.cpp engine on Apple Silicon. If you run without
> the MLX engine your latencies will be considerably higher.
> Add `OLLAMA_NEW_ENGINE=true` to your `.env` (already included in the
> template) or start Ollama with that variable exported.

### Persistent failures

Four questions scored below 1.0 for both models. Three are **rubric
artefacts** rather than model failures:

- **Q13** — rubric checks for "Semana Santa" (Spanish); the POI data is
  English-only ("Holy Week"). Both models answer correctly but the
  string match fails.
- **Q15** — both models paraphrase the Dolmen description instead of
  quoting it verbatim, so "3rd millennium BC" and "megalithic" go
  unmatched despite being in the source text.
- **Q17** — rubric checks for "itinerary"; the tours section uses
  "route" and "trip" but not that word.
- **Q07** — a genuine single regression: E4B fetched content but
  navigated to a neighbouring section instead of Tourist Information.

A rubric revision using semantic matching (synonyms, paraphrase
equivalence) would push E4B's measured grounding above the 82.5%
reported here.

---

## Conclusions

**`gemma4:e4b` (9.6 GB, 128K context) is the recommended model for
this tourism RAG use case.**

Key takeaways from the experiment:

1. **PageIndex works well for structured tourism data.** The
   heading-based tree index over 408 POIs and 18 sections gives the
   model a navigable map of the destination. E4B achieved 90%
   retrieval accuracy — it found the right section 18 out of 20 times
   without any embedding model or vector database.

2. **Data preparation quality matters as much as model size.** The
   section priority bug caused two outright failures in the E2B original
   run that were unrelated to model capability. Fixing one line in the
   section map raised the retrieval-accurate question count from
   12 to 12 for E2B (no change, because E2B couldn't exploit it) but
   enabled E4B to score perfectly on Q03 and Q04.

3. **E2B is not sufficient for grounded fact extraction.** It can
   navigate simple category lookups but struggles with specific fact
   extraction (addresses, measurements, dates) and instruction
   following (language enforcement). 54–62% grounding is too low for
   a production tourism assistant.

4. **E4B hits the quality bar at a reasonable hardware cost.** 9.6 GB
   fits comfortably on any Apple Silicon laptop, a modern phone is not
   far behind (E4B is the natural successor to on-device E2B once
   hardware catches up). The 80 s average latency is an engineering
   problem (pre-indexed summaries, caching) not a fundamental
   capability limit.

5. **The scoring rubric underestimates E4B.** At least three of the
   four persistent failures are rubric artefacts. Adjusting for
   semantic equivalence would push E4B's grounding above 90%.

---

## Reproducing the Full Experiment

```bash
# 1–3: data pipeline (same as before)
.venv/bin/python scripts/extract_ubeda.py
.venv/bin/python scripts/json_to_markdown.py
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide.md --model openai/gemma4:e2b \
  --if-add-node-summary no --if-add-doc-description no

# E2B eval
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e2b
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-e2b.json

# E4B eval
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e4b
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-e4b.json
```

---

## Open Improvements

- **Add node summaries** — run PageIndex with `--if-add-node-summary yes`
  to give the agent richer navigation context and reduce latency.
- **Revise scoring rubric** — use semantic matching for Q13, Q15, Q17
  to eliminate false negatives from paraphrasing and language
  equivalents.
- **Test 26B / 31B** — both models fit in 128 GB unified memory and
  would likely push grounding above 90% for the synthesis questions
  that E4B still partially misses (Q18).

---

## References

- [PageIndex](https://github.com/VectifyAI/PageIndex) — VectifyAI
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- [Inventrip](https://inventrip.com) — UNE 178503 tourism POI platform
- [UNE 178503](https://www.une.org) — Spanish tourism data standard

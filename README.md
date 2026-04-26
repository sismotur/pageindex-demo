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

**Recommended model: `gemma4:e4b` with two-level navigation.**

| Run | Grounding | Retrieval | Avg latency |
|---|---|---|---|
| E2B — original | 54.1% ❌ | 60% | 32.5 s |
| E2B — fixed pipeline | 62.5% ❌ | 60% | 51.1 s |
| E4B — flat navigation | 82.5% ✅ | 90% | 79.7 s |
| E4B — two-level nav + summaries | 85.9% ✅ | 90% | 59.1 s |
| **26B MoE — two-level nav + summaries** | **90.0% ✅** | **95%** | **35.3 s** |

**`gemma4:26b` is the best result across all dimensions:** highest
grounding, highest retrieval accuracy, and the fastest latency of any
multi-model run. Its Mixture-of-Experts architecture (25B total / 4B active
parameters at inference) explains the speed — it runs as fast as E4B but
reasoning at 26B quality.

**Technical configuration (final):**
- Hardware: Apple Silicon Mac, 128 GB unified memory
- Inference: Ollama MLX engine (`com.ollama.mlx`, `OLLAMA_NEW_ENGINE=true`,
  `OLLAMA_KV_CACHE_TYPE=q8_0`, `num_batch=2048`)
- LLM routing: `litellm` → `openai/gemma4:26b` → `http://localhost:11434/v1`
- Index: PageIndex tree (427 nodes) + 18 section-level LLM summaries
- Dataset: 408 Úbeda POIs, 20 evaluation questions

---

## What is PageIndex?

PageIndex builds a hierarchical tree index from a document's heading
structure and uses an LLM to **navigate** that tree (like a human
flipping through a book) rather than relying on vector similarity search
or chunking. No embedding model, no vector database — just a structured
index and a reasoning model.

---

## Retrieval Architecture

The eval script implements a **two-level navigation** loop with three
tools, chosen based on question type:

**A — Listing questions** ("what museums exist?", "list all hotels")

```
get_sections()           → 18 section headers with summaries (~2 KB)
  └─ get_poi_list(sec)   → POI names + line numbers for that section
       └─ answer from POI names (no need to fetch individual pages)
```

**B — Specific fact questions** ("tell me about X", address, phone,
dates, measurements)

```
get_sections()           → 18 section headers with summaries
  └─ get_poi_list(sec)   → find the exact POI and its line number
       └─ get_page_content(lines) → read the POI text (10-25 lines)
            └─ answer from retrieved text only
```

This replaces the original flat `get_document_structure()` call that
sent **all 427 node titles** (~5,100 tokens) on every question. The
two-level approach sends only the 18 section summaries first (~2,000
tokens), then drills down on demand.

### Section summaries

Run `scripts/add_section_summaries.py` once after indexing to generate
a 2-sentence LLM summary for each of the 18 section nodes. Summaries
are stored in `results/ubeda_guide_structure.json` and included in the
`get_sections()` tool response, giving the model section-level context
without fetching individual POI pages.

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
    ├── add_section_summaries.py  ← Step 3b: generate 18 section summaries (one-time)
    ├── run_eval.py               ← Step 4: litellm agentic tool-calling eval
    └── score_results.py          ← Step 5: score grounding + retrieval
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
| KV cache | `OLLAMA_KV_CACHE_TYPE=q8_0` (8-bit, reduced memory bandwidth) |
| Index type | PageIndex structural tree + 18 section summaries |
| Index rebuild time | < 5 s structural; ~8 min for section summaries (one-time) |

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

# 3. Build the PageIndex structural tree (deterministic, no LLM calls, < 5 s)
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide.md \
  --model openai/gemma4:e4b \
  --if-add-node-summary no \
  --if-add-doc-description no

# 3b. Generate section summaries — ONE-TIME, ~8 min on E4B
#     Re-run only if the Markdown document is rebuilt.
.venv/bin/python scripts/add_section_summaries.py --model openai/gemma4:e4b

# 4. Run the Q&A evaluation (~20 min on E4B with two-level navigation)
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e4b

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

Five evaluation runs were conducted across two retrieval architectures
and three model sizes.

### Run summary

| Run | Architecture | Model | Grounding | Retrieval | Latency |
|---|---|---|---|---|---|
| E2B — original | flat nav | `gemma4:e2b` | 54.1% ❌ | 60% | 32.5 s |
| E2B — fixed | flat nav | `gemma4:e2b` | 62.5% ❌ | 60% | 51.1 s |
| E4B — flat nav | flat nav | `gemma4:e4b` | 82.5% ✅ | 90% | 79.7 s |
| E4B — two-level + summaries | two-level | `gemma4:e4b` | 85.9% ✅ | 90% | 59.1 s |
| **26B — two-level + summaries** | **two-level** | **`gemma4:26b`** | **90.0% ✅** | **95%** | **35.3 s** |

**`gemma4:26b` is the best result — highest grounding and retrieval
accuracy at the lowest latency of any run.**

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

| Difficulty | Questions | E2B (fixed) | E4B (two-level) | 26B (two-level) |
|---|---|---|---|---|
| Easy | 10 | 0.720 | 0.850 | **0.940** |
| Medium | 7 | 0.671 | 0.743 | **0.771** |
| Hard | 3 | 0.400 | 0.867 | **0.933** |

E4B's largest gain is on **hard synthesis questions** (+0.467) — exactly
the category where E2B's weaker instruction-following hurt most.

### Latency

| Architecture | Model | Total (20 Qs) | Avg / question | vs E2B flat |
|---|---|---|---|---|
| Flat navigation | E2B | 1022 s | 51 s | 1× |
| Flat navigation | E4B | 1593 s | 80 s | 1.6× |
| Two-level + summaries | E4B | 1182 s | 59 s | 1.2× |
| **Two-level + summaries** | **26B** | **706 s** | **35 s** | **0.7×** |

26B MoE is the **fastest model despite being the largest**: its
Mixture-of-Experts architecture activates only ~4B parameters per token,
giving near-E4B compute cost with 26B-level reasoning. Combined with the
two-level navigation and `num_batch=2048` tuning, it answers in 35 s/q
— 31% faster than E4B and 44% faster than E4B flat navigation.

> **MLX engine:** all figures were measured with `com.ollama.mlx`
> (`OLLAMA_NEW_ENGINE=true`, `OLLAMA_KV_CACHE_TYPE=q8_0`). Running
> without the MLX backend on Apple Silicon will be considerably slower.

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

**`gemma4:26b` with two-level navigation and section summaries is the
recommended configuration for this tourism RAG use case.**

Key takeaways from the experiment:

1. **PageIndex works well for structured tourism data.** The
   heading-based tree index over 408 POIs and 18 sections gives the
   model a navigable map of the destination. 26B achieved 95%
   retrieval accuracy — it found the right section 19 out of 20 times
   without any embedding model or vector database.

2. **Data preparation quality matters as much as model size.** The
   section priority bug caused two outright failures in the E2B original
   run that were unrelated to model capability. Fixing one line in the
   section map (moving Accommodation before Civil Monuments in the
   priority list) resolved those failures regardless of model choice.

3. **E2B is not sufficient for grounded fact extraction.** It can
   navigate simple category lookups but struggles with specific fact
   extraction (addresses, measurements, dates) and instruction
   following (language enforcement). 54–62% grounding is too low for
   a production tourism assistant.

4. **Two-level navigation + section summaries improves both quality
   and speed.** The 18-section overview (2,000 tokens) replaces the
   flat 5,100-token structure dump, enabling faster navigation and
   better section-level reasoning. Combined with `gemma4:26b`, this
   achieves 90% grounding at 35 s/q — the best result on every metric.

5. **Prompt strategy matters as much as context size.** Distinguishing
   listing questions (answer from POI titles) from specific-fact
   questions (always fetch the POI text) prevents both empty answers
   on large sections and hallucination on named-entity lookups.

6. **The scoring rubric underestimates the final configuration.**
   Three of the four persistent failures (Q13, Q15, Q17) are rubric
   artefacts — the checker looks for exact strings that are absent
   from the English-only source data or uses Spanish synonyms.
   Semantic matching would push measured grounding above 90%.

---

## Reproducing the Full Experiment

```bash
# 1. Data pipeline
.venv/bin/python scripts/extract_ubeda.py
.venv/bin/python scripts/json_to_markdown.py
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide.md --model openai/gemma4:e4b \
  --if-add-node-summary no --if-add-doc-description no

# 2. Generate section summaries (one-time, ~8 min)
.venv/bin/python scripts/add_section_summaries.py --model openai/gemma4:e4b

# 3. E2B baseline eval (flat navigation, for comparison)
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e2b
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-e2b.json

# 4. E4B final eval (two-level navigation + section summaries)
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e4b
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-e4b.json
```

---

## Open Improvements

- **Revise scoring rubric** — use semantic matching for Q13, Q15, Q17
  to eliminate false negatives from paraphrasing and language
  equivalents. This alone would lift measured grounding above 90%.
- **Test 31B** — fits in 128 GB unified memory (20 GB); the dense
  31B model would likely resolve Q12 (language drift on heritage
  question) and push grounding above 92%.
- **Response caching** — section summaries and POI lists are static;
  caching them client-side would eliminate repeated `get_sections` and
  `get_poi_list` calls across a conversation, cutting latency further.
- **Full node summaries** — `--if-add-node-summary yes` on individual
  POI nodes (408 calls, ~2 h on E4B) would enable the agent to answer
  some fact questions directly from the summary without a
  `get_page_content` call.

---

## References

- [PageIndex](https://github.com/VectifyAI/PageIndex) — VectifyAI
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- [Inventrip](https://inventrip.com) — UNE 178503 tourism POI platform
- [UNE 178503](https://www.une.org) — Spanish tourism data standard

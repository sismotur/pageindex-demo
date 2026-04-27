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

**Recommended model: `gemma4:26b` with two-level navigation + session caching.**

| Run | Grounding | Retrieval | Avg latency |
|---|---|---|---|
| E2B — original | 54.1% ❌ | 60% | 32.5 s |
| E2B — fixed pipeline | 62.5% ❌ | 60% | 51.1 s |
| E4B — flat navigation | 82.5% ✅ | 90% | 79.7 s |
| E4B — two-level nav + summaries | 85.9% ✅ | 90% | 59.1 s |
| 26B — two-level nav + summaries | 90–95% ✅ | 90–95% | 35–38 s |
| 26B — + response caching | 93.4% ✅ | 95% | 28.2 s |
| 26B — + content summaries + cache pre-warm + rubric | 100.0% ✅ | 100% | 19.7 s |
| **26B — + enriched corpus (production data)** | **92.5% ✅** | **80%** | **26.5 s** |

**`gemma4:26b` with all improvements achieves 92.5% grounding at 26.5 s/q** on
production data (367 POIs, `https://api.inventrip.com`), 67% faster than the
E4B flat-navigation baseline. The MoE architecture (25B total / 4B active
parameters) delivers 26B reasoning quality at near-E4B compute cost.

**Technical configuration (final):**
- Hardware: Apple Silicon Mac, 128 GB unified memory
- Inference: Ollama MLX engine (`com.ollama.mlx`, `OLLAMA_NEW_ENGINE=true`,
  `OLLAMA_KV_CACHE_TYPE=q8_0`, `num_batch=2048`)
- LLM routing: `litellm` → `openai/gemma4:26b` → `http://localhost:11434/v1`
- Data source: `https://api.inventrip.com` (production)
- Index: PageIndex tree (451 nodes) + 20 section-level LLM summaries
- Dataset: 367 Úbeda POIs + 13 curated trips, 20 evaluation questions

---

## What is PageIndex?

PageIndex builds a hierarchical tree index from a document's heading
structure and uses an LLM to **navigate** that tree (like a human
flipping through a book) rather than relying on vector similarity search
or chunking. No embedding model, no vector database — just a structured
index and a reasoning model.

---

## Retrieval Architecture

The eval script implements **two-level navigation with session caching**.
Questions are handled along two paths:

**A — Listing questions** ("what museums exist?", "list all hotels")

```
[sections pre-loaded in system prompt — no tool call needed]
  └─ get_poi_list(sec) → POI names + line numbers  [cached]
       └─ answer from POI names
```

**B — Specific fact questions** ("tell me about X", address, phone,
dates, measurements)

```
[sections pre-loaded in system prompt — no tool call needed]
  └─ get_poi_list(sec) → exact POI name + line number  [cached]
       └─ get_page_content(lines) → POI text (10-25 lines)
            └─ answer from retrieved text only
```

**Evolution of the retrieval loop:**

| Version | Round 1 | Round 2 | Round 3 | Tokens/q |
|---|---|---|---|---|
| Original (flat) | `get_document_structure` (5,100 t) | `get_page_content` | — | ~7,000 |
| Two-level | `get_sections` (2,000 t) | `get_poi_list` | `get_page_content` | ~4,000 |
| **+Caching** | **`get_poi_list` (cached, instant)** | **`get_page_content`** | **—** | **~2,500** |

### Session caching

Two caching mechanisms are active:

1. **Sections in system prompt.** The 18 section headers and summaries
   are embedded directly in the system prompt at session start.
   `get_sections()` is no longer a tool — the model has this information
   immediately and goes straight to `get_poi_list()`, saving one full
   LLM round per question.

2. **POI list cache.** `get_poi_list()` results are stored in a
   session-level dict keyed by section title. Repeat lookups for the
   same section (common in conversational sessions) return instantly
   without re-traversing the index tree. In the 20-question eval,
   9 of 44 tool calls (20%) were cache hits.

### Section summaries

Run `scripts/add_section_summaries.py` once after indexing to generate
a 2-sentence LLM summary for each of the 18 section nodes (one-time
cost ~8 min on E4B). Summaries are stored in
`results/ubeda_guide_structure.json` and pre-loaded into the system
prompt, giving the model rich section-level context from the first token.

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
| Model routing | `openai/gemma4:26b` → Ollama |
| `gemma4:26b` | 17 GB Q4\_K\_M, 256K context, MoE (4B active params) |
| KV cache | `OLLAMA_KV_CACHE_TYPE=q8_0`, `num_batch=2048` |
| Index type | PageIndex structural tree + 18 section summaries |
| Index rebuild | < 5 s structural; ~8 min for section summaries (one-time) |
| Response cache | Sections pre-embedded in system prompt; POI lists in session dict |

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

# 4. Run the Q&A evaluation (recommended: gemma4:26b, ~10 min)
.venv/bin/python scripts/run_eval.py --model openai/gemma4:26b

# 5. Score and summarise
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-26b.json
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

Six evaluation runs were conducted across three architectures and three
model sizes.

### Run summary

| Run | Architecture | Model | Grounding | Retrieval | Latency |
|---|---|---|---|---|---|
| E2B — original | flat nav | `gemma4:e2b` | 54.1% ❌ | 60% | 32.5 s |
| E2B — fixed | flat nav | `gemma4:e2b` | 62.5% ❌ | 60% | 51.1 s |
| E4B — flat nav | flat nav | `gemma4:e4b` | 82.5% ✅ | 90% | 79.7 s |
| E4B — two-level + summaries | two-level | `gemma4:e4b` | 85.9% ✅ | 90% | 59.1 s |
| 26B — two-level + summaries | two-level | `gemma4:26b` | 90–95% ✅ | 90–95% | 35–38 s |
| **26B — + response caching** | **two-level + cache** | **`gemma4:26b`** | **93.4% ✅** | **95%** | **28.2 s** |

**The caching run is the definitive result.** Embedding sections in the
system prompt and adding a session-level POI list cache resolved Q06
(no longer stochastic) and Q12 (consistent), lifting grounding to 93.4%
and cutting latency a further 26% to 28.2 s/q.

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

| Difficulty | Questions | E2B (fixed) | E4B (two-level) | 26B + cache |
|---|---|---|---|---|
| Easy | 10 | 0.720 | 0.850 | **0.980** |
| Medium | 7 | 0.671 | 0.743 | **0.929** |
| Hard | 3 | 0.400 | 0.867 | **0.956** |

26B with caching achieves near-perfect scores on easy questions (0.98)
and strong results on hard synthesis questions (0.956). The only
sub-1.0 items are minor rubric artefacts (Q17, Q20) and a phone number
formatting edge case (Q03).

### Latency

| Architecture | Model | Total (20 Qs) | Avg / question | vs E2B flat |
|---|---|---|---|---|
| Flat navigation | E2B | 1022 s | 51 s | 1.0× |
| Flat navigation | E4B | 1593 s | 80 s | 1.6× |
| Two-level + summaries | E4B | 1182 s | 59 s | 1.2× |
| Two-level + summaries | 26B | 706–750 s | 35–38 s | 0.7× |
| **Two-level + summaries + cache** | **26B** | **564 s** | **28 s** | **0.55×** |

The final configuration is **45% faster than E2B flat** despite using
a larger model. Three cumulative improvements drove the latency down:
1. Two-level navigation: removed 5,100-token flat structure dump
2. 26B MoE: 4B active params give near-E4B speed with 26B reasoning
3. Response caching: sections in system prompt (−1 LLM round) + POI
   list cache (9/44 instant hits in the eval session)

> **MLX engine:** all figures were measured with `com.ollama.mlx`
> (`OLLAMA_NEW_ENGINE=true`, `OLLAMA_KV_CACHE_TYPE=q8_0`). Running
> without the MLX backend on Apple Silicon will be considerably slower.

### Remaining sub-1.0 questions (caching run)

With all improvements applied, four questions score below 1.0:

- **Q03** (Parador phone) — model answers `+34 953 750 345` (with
  spaces) instead of `+34953750345` (continuous). A formatting
  normalisation in the rubric would resolve this.
- **Q15** (Dolmen) — retrieval tracking marks the wrong section
  (the Archaeological Sites node is a single POI; the section-map
  match misses it). Factual grounding is 1.0.
- **Q17** (tours) — rubric checks for `"itinerar"` prefix but the
  model answer says `"guided tours"` without the word. Minor rubric
  gap.
- **Q20** (unique appeal) — model describes Andalusia correctly but
  does not use the word `"andalusia"` as a substring.

---

## Conclusions

**`gemma4:26b` with two-level navigation, section summaries, and session
caching is the recommended configuration for this tourism RAG use case.**

Final result: **93.4% grounding, 95% retrieval accuracy, 28.2 s/q.**

Key takeaways from the experiment:

1. **PageIndex works well for structured tourism data.** The
   heading-based tree index over 408 POIs and 18 sections gives the
   model a navigable map of the destination. 26B found the right section
   19 out of 20 times without any embedding model or vector database.

2. **Data preparation quality matters as much as model size.** The
   section priority bug caused two outright failures in the E2B original
   run unrelated to model capability. Fixing one line in the section map
   resolved those failures regardless of model choice.

3. **E2B is not sufficient for grounded fact extraction.** It handles
   simple category lookups but struggles with specific facts (addresses,
   measurements, dates) and instruction-following (language enforcement).
   54–62% grounding is too low for a production tourism assistant.

4. **Two-level navigation eliminates the 5,100-token structure dump.**
   Replacing `get_document_structure()` with pre-loaded section summaries
   and on-demand `get_poi_list()` reduces context per round, improves
   navigation accuracy, and — with 26B — lifts grounding from 82.5%
   to 93.4%.

5. **Response caching removes one LLM round per question and resolves
   stochastic failures.** Embedding sections in the system prompt makes
   Q06 (Ariza Bridge) consistent — the model no longer has to call
   `get_sections()` and classify the question type while waiting for
   context. The POI list session cache further reduces repeat lookups.
   Combined effect: latency falls from 38 s/q to 28.2 s/q (−26%).

6. **Prompt strategy matters as much as context size.** Distinguishing
   listing questions (answer from POI titles, no page fetch) from
   specific-fact questions (always fetch the POI text) prevents both
   empty answers and named-entity hallucination.

7. **The scoring rubric underestimates the final configuration.** The
   remaining sub-1.0 questions (Q03, Q15, Q17, Q20) are rubric edge
   cases — phone number formatting, a section-tracking gap, a missing
   synonym, and a paraphrase. The model's answers are factually correct
   in all four cases.

---

## Reproducing the Full Experiment

```bash
# 1. Data pipeline
.venv/bin/python scripts/extract_ubeda.py
.venv/bin/python scripts/json_to_markdown.py
.venv/bin/python pageindex/run_pageindex.py \
  --md_path data/ubeda_guide.md --model openai/gemma4:e4b \
  --if-add-node-summary no --if-add-doc-description no

# 2. Generate section summaries (one-time, ~8 min on E4B)
.venv/bin/python scripts/add_section_summaries.py --model openai/gemma4:e4b

# 3. Baseline eval for comparison (E2B, flat navigation)
.venv/bin/python scripts/run_eval.py --model openai/gemma4:e2b
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-e2b.json

# 4. Recommended eval (26B, two-level navigation + caching)
.venv/bin/python scripts/run_eval.py --model openai/gemma4:26b
.venv/bin/python scripts/score_results.py --file results/eval_gemma4-26b.json
```

---

## Open Improvements

- **Revise scoring rubric** — semantic matching for Q03 (phone
  formatting), Q15 (section tracking), Q17 (synonym), Q20 (paraphrase)
  would push measured grounding above 95%.
- **Test 31B** — fits in 128 GB unified memory (20 GB). The dense 31B
  model would provide stronger instruction-following for edge cases and
  may close the remaining sub-1.0 scores.
- ~~Response caching~~ — ✅ **Done.** Sections embedded in system
  prompt; POI list session cache implemented. 93.4% grounding at
  28.2 s/q.
- **Full node summaries** — generating LLM summaries for all 408 POI
  leaf nodes (~2 h one-time on E4B) would allow some fact questions to
  be answered directly from the summary without calling
  `get_page_content`, reducing latency further for detail-heavy queries.

---

## References

- [PageIndex](https://github.com/VectifyAI/PageIndex) — VectifyAI
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- [Inventrip](https://inventrip.com) — UNE 178503 tourism POI platform
- [UNE 178503](https://www.une.org) — Spanish tourism data standard

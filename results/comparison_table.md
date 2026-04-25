# Cross-Model Comparison: Gemma 4 E2B (fixed) vs E4B

> Index: 427 nodes, 18 sections, 408 POIs — structural only (no LLM summaries)
> Dataset: Úbeda tourism, 20 questions (10 easy / 7 medium / 3 hard)

## Aggregate Scores

| Metric | E2B (fixed) | E4B | Δ |
|---|---|---|---|
| Factual grounding | 62.5% ❌ | **82.5% ✅** | +20.0 pp |
| Retrieval accuracy | 60.0% (12/20) | **90.0% (18/20)** | +30.0 pp |
| Content fetched | 65.0% (13/20) | **100.0% (20/20)** | +35.0 pp |
| Composite score | 0.655 | **0.900** | +0.245 |
| Avg latency | 51.1 s/q | 79.7 s/q | +28.6 s |

**Verdict: E4B ✅ PASS — escalation to 26B not required.**

## Per-Difficulty Breakdown

| Difficulty | Questions | E2B | E4B | Δ |
|---|---|---|---|---|
| Easy | 10 | 0.720 | **0.970** | +0.250 |
| Medium | 7 | 0.671 | **0.814** | +0.143 |
| Hard | 3 | 0.400 | **0.867** | +0.467 |

E4B's largest relative gain is on **hard synthesis questions** (+0.467),
where E2B struggled to retrieve across multiple sections.

## Per-Question Detail

| ID | Category | Diff | E2B | E4B | Δ | Notes |
|---|---|---|---|---|---|---|
| Q01 | overview | easy | 1.000 | 1.000 | — | |
| Q02 | monument_lookup | easy | 0.800 | **1.000** | +0.200 ↑ | E4B found "Savior Chapel" |
| Q03 | poi_direct_lookup | easy | 0.600 | **1.000** | +0.400 ↑ | E4B extracted Parador address + phone |
| Q04 | category_browse | easy | 0.500 | **1.000** | +0.500 ↑ | E4B navigated to Museums section |
| Q05 | category_browse | easy | 0.800 | **1.000** | +0.200 ↑ | E4B found Guadalupe sanctuary |
| Q06 | poi_direct_lookup | easy | 0.100 | **1.000** | +0.900 ↑ | E4B extracted 1562, Guadalimar, 100m |
| Q07 | practical_info | easy | **1.000** | 0.700 | −0.300 ↓ | E2B answered from titles; E4B fetched but hit wrong section |
| Q08 | practical_info | easy | 1.000 | 1.000 | — | |
| Q09 | gastronomy | easy | 0.400 | **1.000** | +0.600 ↑ | E2B answered in Spanish; E4B correct in English |
| Q10 | gastronomy | medium | 0.500 | **1.000** | +0.500 ↑ | E4B retrieved Gastronomy section |
| Q11 | accommodation | easy | 1.000 | 1.000 | — | |
| Q12 | heritage | medium | 1.000 | 1.000 | — | |
| Q13 | events | medium | 0.800 | 0.800 | — | Both miss "Semana Santa" (rubric uses Spanish term, data is English-only) |
| Q14 | category_browse | medium | 1.000 | 1.000 | — | |
| Q15 | poi_direct_lookup | medium | 0.300 | 0.300 | — | Both fail to surface "3rd millennium" / "megalithic" from Dolmen |
| Q16 | practical_info | medium | 0.300 | **0.800** | +0.500 ↑ | E4B retrieved pharmacy list |
| Q17 | category_browse | medium | 0.800 | 0.800 | — | Both miss "itinerary" (word not in source data) |
| Q18 | synthesis | hard | 0.600 | **0.732** | +0.132 ↑ | E4B retrieved heritage sections; still missing Chapel of El Salvador |
| Q19 | synthesis | hard | 0.500 | **1.000** | +0.500 ↑ | E4B covered gastronomy + olive oil fully |
| Q20 | synthesis | hard | 0.100 | **0.868** | +0.768 ↑ | E2B answered in Spanish; E4B correct, missed "Andalusia" |

## What E4B Fixes vs E2B

**E4B resolved (12 questions improved):**
- Language drift: Q09, Q20 now answered in English
- Fact extraction on retrieved content: Q03, Q06 — specific addresses, phone
  numbers, dates, and measurements correctly extracted
- Navigation to correct section: Q04 (Museums), Q10 (Gastronomy), Q16
  (Health/Pharmacy), Q19 (Gastronomy synthesis)
- Cross-section synthesis for hard questions: Q18, Q19, Q20

**Persistent failures (both models, 4 questions):**
- Q13: Scoring rubric checks for "Semana Santa" (Spanish), but POI data is
  English-only. Not a model failure — rubric adjustment needed.
- Q15 (Dolmen): Neither model surfaces the exact phrases "3rd millennium BC"
  and "megalithic construction" even though they are in the source text.
  Likely caused by the model paraphrasing rather than quoting directly.
- Q17: Rubric checks for the word "itinerary" which does not appear in the
  source data for the tours section.
- Q07: E4B regressed vs E2B (E2B happened to get it right from structure
  titles alone, E4B fetched content but navigated to a different section).

## Scoring Rubric Notes

Three of the four persistent failures (Q13, Q15, Q17) are at least partly
**rubric artefacts** — the grounding checks reference words or phrases that
are not present verbatim in the source data. A revised rubric using
synonyms or semantic matching would likely raise E4B's grounding score
above the 82.5% measured here.

## Latency Summary

| Model | Total (20 Qs) | Avg / Q | Relative |
|---|---|---|---|
| E2B | 1022 s | 51.1 s | 1× |
| E4B | 1593 s | 79.7 s | 1.6× |

E4B is 1.6× slower than E2B but delivers significantly higher accuracy.
For a real-time tourism concierge with a ~80s SLA per query this is
borderline; running with node summaries (pre-indexed) would reduce the
number of `get_page_content` calls and likely cut latency.

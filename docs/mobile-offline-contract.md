# Mobile Offline Contract — Inventrip Assistant (Gemma 4 E2B)

**Version:** 1.0
**Status:** Draft
**Audience:** Android (Kotlin) and iOS (Swift) implementers
**Reference implementation:** `assistant/` (Python) — when in doubt, match
  its behaviour byte-for-byte.
**Data source:** `docs/cloudflare-worker-spec.md` (download endpoints)

This document is the **complete contract** for the on-device assistant.
A phone downloads one index file plus a small daily weather file per
`(destination, language)` and then works **with no internet
connection**: the LLM (Gemma 4 E2B) and all eleven retrieval tools run
locally.

---

## 1. Components on the phone

```
┌─────────────────────────────────────────────────────────────┐
│ Mobile app                                                  │
│                                                             │
│  indexes/{dest}_{lang}.json   ← downloaded once (ETag       │
│                                 refresh when online)        │
│                                                             │
│  weather/{dest}_{lang}.json  ← refreshed daily (ETag)      │
│                                                             │
│  Tool layer  ← port of assistant/index_tools.py             │
│    list_sections / get_section / get_poi /                  │
│    find_poi_by_name / filter_pois / search_pois             │
│    search_trips / get_trip / search_paths / get_path /      │
│    get_weather                                              │
│    (pure lookups over the parsed JSON — no DB, no network)  │
│                                                             │
│  Agentic loop  ← port of assistant/run_eval.py              │
│    system prompt + tools + tool-dispatch loop               │
│                                                             │
│  LLM runtime: Gemma 4 E2B                                   │
│    Android: LiteRT-LM (MediaPipe LLM Inference API)         │
│    iOS:     MLX Swift (or llama.cpp)                        │
└─────────────────────────────────────────────────────────────┘
```

Measured baseline for this exact stack (E2B, oMLX serving, schema v4
index, 20 visitor questions, English, 2026-08-16): composite **0.830**,
grounding **75.0%**, content-fetch **95%**, **3.1 s**/question, 13,950
prompt + 241 completion tokens/question. The same E2B weights are the
deployment target; expect ≥ the 70% rubric thresholds on EN/ES/IT.

---

## 2. The index file (`indexes/{dest}_{lang}.json`)

One JSON object, `meta.schema_version == 6`. Sizes: ~0.7–1.3 MB per pair
(367 POIs, Úbeda). Parse it fully into memory at session start.

Schema v2 adds the optional `sections[].groups` field. Schema v3 adds
`facets.search_terms`, the deterministic full-text evidence index used by
`search_pois`. Schema v4 adds resolved `trips[]` and `paths[]`. Schema v5
adds source-id/cross-language itinerary stop resolution. Schema v6 stores
ordered itinerary stops as a nested tree (`steps[].items[]` with `folder`,
`poi`, and `unresolved` kinds) so subfolder POIs are preserved. Older
readers that only walk the flat `steps[].poi_ids` continue to work —
`sections[].poi_ids` and POI records remain complete.

```jsonc
{
  "meta": {
    "destination": "ubeda",              // slug
    "destination_display": "Úbeda",      // human name for prompts/UI
    "lang": "en",
    "generated_at": "2026-08-16T05:13:09Z",
    "poi_count": 367,
    "section_count": 18,
    "schema_version": 6
  },
  "destination_overview": "…multi-line string, embedded in the system prompt…",
  "trips": [
    { "itinerary_id": "trip/4407", "trip_id": "trip/4407",
      "kind": "trip", "name": "…", "description": "…", "url": "…",
      "steps": [ {
        "position": 1, "title": "1. What not to miss",
        // v6: recursive tree, folder | poi | unresolved
        "items": [
          { "kind": "folder", "name": "1.1 Plaza Vázquez de Molina",
            "items": [
              { "kind": "poi", "poi_id": "poi/30536",
                "source_name": "Plaza Vázquez de Molina",
                "resolution":  "source_id" },
              // …
            ] },
          { "kind": "unresolved", "name": "CR La Casería de Tito" }
        ],
        // Flat reading-order projections kept for pre-v6 readers.
        "poi_ids":              ["poi/30536"],
        "poi_resolutions":      [ /* {poi_id, source_name, resolution} */ ],
        "subfolders":           ["1.1 Plaza Vázquez de Molina"],
        "unresolved_poi_names": ["CR La Casería de Tito"]
      } ] }
  ],
  "sections": [
    { "section_id": "shopping",              // slug, stable
      "title": "Shopping",                   // display title
      "summary": "66 POIs (…). Top interests: … Notable: …",
      "poi_ids": ["poi/123", …],             // ALL section POIs, sorted
      "groups": [                            // v2 only, optional: present
                                             // when the section has > 30 POIs
        { "group_id": "shopping--store",     // "{section_id}--{type-slug}"
          "title": "Store",                  // display_type of the group
          "poi_ids": ["poi/123", …],         // sorted by (interest, zoom, name)
          "summary": "33 POIs. Notable: …" } // key items preserved by name
      ] }
  ],
  "pois": {
    "poi/123": {
      "poi_id": "poi/123",
      "name": "…",
      "normalized_name": "…",              // see §4 — do NOT recompute
      "description": "…",
      "types": ["Museum"],                 // UNE 178503 codes
      "display_type": "Museum",
      "tourist_types": ["CULTURAL TOURISM"],
      "display_tourist_types": ["Cultural Tourism"],
      "interest_level": 1,                 // 1|2|3 or null
      "interest_level_label": "Indispensable",   // localized, or null
      "zoom_level": 15,                    // 10–19 or null
      "booking_url": "…",
      "url": ["…"], "telephone": ["…"], "email": ["…"],
      "street_address": "…", "address_locality": "…",
      "address_province": "…", "address_region": "…",
      "postal_code": "…", "country_code": "ES", "country": "Spain",
      "latitude": 38.0, "longitude": -3.37,   // or null
      "image_urls": ["https://api.inventrip.com/v100/image/…"],
      "audio_urls": ["…"],
      "subject_of_urls": ["Label: https://…"],
      "start_date": "…", "end_date": "…",      // events; "" otherwise
      "raw_extras": { … }                      // UNE extras, passthrough
    }
  },
  "facets": {
    "by_section":        { "museums-and-culture": ["poi/123", …] },
    "by_type":           { "Museum": ["poi/123", …] },
    "by_tourist_type":   { "CULTURAL TOURISM": ["poi/123", …] },
    "by_interest_level": { "1": ["poi/…"], "2": […], "3": […] },
    "by_zoom_bucket":    { "<=14": […], "15-16": […], "17-19": […] },
    "indispensable":     ["poi/…", …],         // interest_level == 1
    "search_terms":      { "olive": ["poi/30124", "poi/36694", …] }
  },
  "name_index": { "sacra capilla del salvador": "poi/5155", … },
  "tourist_type_display": { "CULTURAL TOURISM": "Cultural Tourism", … },
  "interest_levels": { "1": "Indispensable", "2": "Interesting",
                       "3": "Outstanding" }     // localized labels
}
```

Notes for ports:

- `poi_id` keys keep the `poi/` prefix; `get_poi` also accepts the bare
  numeric suffix (§3.2).
- `normalized_name` is precomputed at build time — the phone never runs
  NFKD itself except to normalize the user's **query** (§4).
- `raw_extras` is passthrough UNE 178503 data; the tools never read it,
  but the app UI may (opening hours, prices, …).

---

## 2.4 The weather file (`weather/{dest}_{lang}.json`)

One small JSON per `(destination, language)`, refreshed daily by the
Cloudflare Worker (see `docs/cloudflare-worker-spec.md` §4–5). The phone
downloads it independently of the index via `GET /v1/weather/{dest}/{lang}`
with ETag revalidation, falls back to the last downloaded copy when
offline, and treats a file older than 7 days as expired.

```jsonc
{
  "meta": {
    "destination":    "ubeda",
    "lang":           "es",
    "latitude":       38.0108,
    "longitude":      -3.3717,
    "units":          "metric",
    "fetched_at":     "2026-08-25T04:00:00Z",
    "expires_at":     "2026-08-26T04:00:00Z",   // fetched_at + 24 h
    "schema_version": 1
  },
  "forecast": [
    {
      "date":           "2026-08-25",           // ISO YYYY-MM-DD
      "iso_weekday":    2,                       // 1=Mon…7=Sun
      "day_label":      "Mar 25",                // localized source string
      "temp_min_c":     31.6,
      "temp_max_c":     46.0,
      "condition":      "Cielo claro",           // localized source string
      "condition_code": "2119bfd6-a006-4577-e0f0-bba80a256700",
      "icon_url":       "https://…/public"       // rendered by the app UI
    }
    // …up to 7 entries
  ]
}
```

Notes:
- `condition_code` is the 36-character UUID from the Inventrip CDN URL,
  stable across languages. The mobile app maps codes to bundled icons.
- The runtime treats a file with `now - fetched_at > 24 h` as *stale but
  usable* and prefixes the answer with a localized "estimated forecast"
  note; past 7 days the file is *expired* and the tool returns the
  localized unavailable message.
- Missing file is not an error — the tool simply refuses to answer
  weather questions, and the model must not invent a forecast.

---

## 3. The eleven tools (exact semantics)

All eleven are pure functions over the parsed index (and, for
`get_weather`, the parsed weather file). **Outputs are text blobs
returned to the LLM verbatim** — match the Python formatting exactly,
because the system prompt and the model's learned behaviour depend on
it. Reference: `assistant/index_tools.py`.

### 3.1 `list_sections()`

Returns the sections overview — the same text embedded in the system
prompt (§5), so a correct implementation rarely needs to execute it:

```
Destination: Úbeda  (367 POIs across 18 sections)

SECTIONS:
  [unesco-world-heritage-and-city-overview] UNESCO World Heritage and City Overview  (2 POIs)
      2 POIs (1 Indispensable, 1 Outstanding). Top interests: … Notable: …
  …
```

### 3.2 `get_poi(poi_id)`

Lookup order per id: exact key → `poi/{arg}` prefix added → prefix
stripped. **Batch form:** `poi_id` accepts comma-separated ids
(`"poi/123, poi/456"`, bare numbers also fine); the result is the
records joined with `\n\n---\n\n`. Unknown ids render an inline
`[ERROR] POI '…' not found.` block without failing the batch. A single
unknown id returns:

```
[ERROR] POI '123' not found. Use find_poi_by_name() if you only know the name.
```

Found id returns the full record — header, optional `*Section: …*` line,
`- **Label**: value` bullets (only for non-empty fields, lists
comma-joined), then a blank line and the full description:

```
# Sacra Capilla del Salvador  (poi/5155)
*Section: UNESCO World Heritage and City Overview*

- **Interest level**: Indispensable
- **Type**: CivilBuilding
- **Tourism interest**: Heritage Tourism, Architecture
- **Address**: Plaza Vázquez de Molina, Úbeda, Jaén
- **Coordinates**: 38.007863, -3.367220      (only if both are non-null)
- **Map prominence**: Major landmark (zoom 15)   (only if zoom ≤ 16)
…

Full description paragraph(s) here, never truncated.
```

Media URLs (`image_urls`, `audio_urls`, `subject_of_urls`) are **not**
rendered in the tool output — the model cannot act on them and they cost
~13% of the record's tokens. They remain in the index record for the
app's UI; render them from the parsed POI object, not from the tool text.

### 3.3 `get_section(section_id, sort?, limit?)`

Section resolution is tolerant: exact `section_id` → case-insensitive
exact title → substring title → normalized title. Unknown key returns
`[ERROR] Section '…' not found. Available: <comma-joined titles>`.

`sort`: `"interest"` (default: `(interest_level, zoom_level, name)`
ascending, nulls last as 99) | `"name"` | `"zoom"`. **Adaptive default
limit**: 20 when the section carries a v2 `groups` map, 50 otherwise;
an explicit `limit` always wins. When truncated, append
`  …N more (raise --limit to see all)`.

**Schema v2:** when the section has `groups`, a group map is rendered
between the summary and the preview list. The group map is the navigation
structure — drill down with `filter_pois(type=<group title>,
section_id=<section_id>)`:

```
Section: Shopping  (id=shopping, 66 POIs total)
  66 POIs (…). Top interests: Shopping, … Notable: …

  Groups in this section (drill down with filter_pois(type=..., section_id="shopping")):
    [shopping--shoppingcenter] ShoppingCenter — 33 POIs.  Notable: Mesones and Obispo Cobos Streets, …
    [shopping--store] Store — 33 POIs.  Notable: Juan Tito Pottery, …

  [poi/123] Name — Store — First sentence of description…
  …
```

Flat sections (≤ 30 POIs) render exactly as before, without the group
block:

The one-line preview per POI is
`display_type — interest_level_label (unless "Outstanding") — first
sentence of description (≤ 90 chars, word-boundary truncated with …)`,
joined with `" — "`, hard-capped at 120 chars.

### 3.4 `find_poi_by_name(query, limit?, detail?)`

Three tiers, in order; ties broken by token overlap, then interest level:

1. Exact match of `normalize_text(query)` in `name_index`.
2. All query tokens present in the POI's normalized name, or the whole
   normalized query is a substring of it.
3. Any query token present.

`limit` default 5. No matches →
`[INFO] No POI matches '…'. Try filter_pois() or browse a section with get_section().`
Otherwise one line per match:
`  [poi/123] Name  [Section Title]  — <preview>`.

`detail` is `"brief"` (default) or `"full"`. With `"full"`, the best
match's complete POI record (the §3.2 rendering) is appended after the
candidate list, introduced by the line `Best match, full record:` — this
fuses the classic find→get two-call pattern into one LLM round. With
`"brief"` the model is expected to follow up with `get_poi()`.

### 3.5 `filter_pois(interest_level?, type?, tourist_type?, section_id?, indispensable?, limit?)`

All supplied filters AND together; called with zero filters →

```
[INFO] filter_pois requires at least one filter (interest_level, type, tourist_type, section_id, indispensable).
```

Facet resolution details (must match):

- `interest_level`: integer 1|2|3, or the English labels
  `indispensable|interesting|outstanding` (case-insensitive).
- `indispensable: true` ≡ interest_level 1; `false` is ignored.
- `type`: exact UNE code lookup in `facets.by_type`.
- `tourist_type`: exact code → uppercase code → normalized code →
  normalized display label (via `tourist_type_display`).
- `section_id`: resolved like `get_section`.

Results sorted by `(interest_level, zoom_level, name)`, default
`limit=20`, rendered like `find_poi_by_name` with a header line
`Filter {active_filters}: N matches` and a trailing
`  …more matches available (raise limit)` when truncated.
### 3.6 Schema v3/v4 overrides: evidence, tags, trips, and paths

This section is authoritative for schema v3/v4 and supersedes older v2
examples above that show raw ids, category labels, interest levels, or
filter echoes.

**`search_pois(query, section_id?, limit?)`** is the sixth tool. It
intersects `facets.search_terms`, so every result contains all meaningful
query terms in that same POI's name, description, category label,
tourism-interest label, or locality. It is used before the model claims
that one place combines two visitor concepts.

For example, `search_pois("olive oil restaurant", "gastronomy")` has no
current Úbeda English result: the catalogue includes restaurants and
olive-oil places but does not establish both characteristics for a single
restaurant. The agent then retrieves `"olive oil"` and `"restaurant"`
separately in the same turn, states this evidence gap naturally, and
presents complementary options without claiming either group satisfies
the full combination.

All current LLM-facing POI output uses:

```text
<poi id=36694 type=OilMill>ALMAZARA BALTASAR LARA Y CÍA.</poi>
```

- `id`: bare numeric suffix of `poi_id`; Android passes it to
  `PointOfInterestActivity` as `poiId`.
- `type`: `display_type`, for app routing/presentation only.
- Inner text: the complete tourist-visible name.

The app must keep `type` and IDs out of visible chat prose. Tool results
are intentionally rendered as `Found N places:` plus tags and description
previews; they do not show filter parameters, raw `[poi/…]` ids,
`Type:`, or interest levels. Full POI records use a tag-ready heading and
visitor-facing details only.

Before making a tag tappable, the mobile parser MUST verify that
`poi/{id}` exists in the currently downloaded index. An unknown or
malformed tag renders as ordinary inner text with no navigation. The LLM
is instructed to copy IDs only from tool results, but deterministic
validation is mandatory.

For compound requests (`X with Y`, `X near Y`, `X that offers Y`), the
agent first runs one combined `search_pois` evidence check. If it has no
direct match, the runtime forces separate complementary retrieval in the
same turn; it must not ask the visitor to choose a next search.

### 3.8 Schema v4: curated trips versus physical paths

Schema v4 adds two collections with the same underlying shape but
different visitor meaning:

```json
{
  "trips": [{
    "itinerary_id": "trip/4407",
    "trip_id": "trip/4407",
    "kind": "trip",
    "source_type": "TouristTrip",
    "name": "TASTE ÚBEDA",
    "description": "…",
    "url": "https://inventrip.com/ubeda/trip/4407",
    "steps": [{
      "position": 1,
      "title": "Restaurants",
      "poi_ids": ["poi/35398", "poi/35403"],
      "unresolved_poi_names": []
    }]
  }],
  "paths": [{
    "itinerary_id": "path/9001",
    "path_id": "path/9001",
    "kind": "path",
    "source_type": "Path",
    "name": "Example Walking Route",
    "description": "…",
    "url": "…",
    "steps": [{
      "position": 1,
      "title": "Start",
      "poi_ids": ["poi/…"],
      "unresolved_poi_names": ["Source waypoint with no POI record"]
    }]
  }]
}
```

- **Trips** come from `/v120/trips`: editorial suggestions for what to
  do over a theme, one day, or several days. They are not physical routes.
- **Paths** come only from `/v120/paths`: walking, cycling, trail, or
  route candidates. Never represent a trip as a path.
- `poi_ids` are localized exact name matches from the itinerary source.
  `unresolved_poi_names` preserves stops that cannot be linked offline.
- An empty `paths[]` is valid. Current Úbeda en/es/it snapshots have no
  `/paths` records; the assistant must state that no physical route is
  available rather than substituting a trip.

The four tools remain separate:

| Tool | Visitor intent | Output tag |
|---|---|---|
| `search_trips` / `get_trip` | What to do, themed/day/multi-day suggestions | `<trip id=…>…</trip>` |
| `search_paths` / `get_path` | Walking, cycling, trail, track, route | `<path id=…>…</path>` |

The app may later make trip/path tags tappable, but this reference
implementation does not claim an Android navigation contract for them
yet. POI stops retain the existing `<poi id=… type=…>…</poi>` contract.

### 3.9 Schema v5: stable source ids and localization safety

Itinerary source stop resolution follows this strict order:

1. A stable identifier supplied by the `/trips` or `/paths` item.
2. Exact name in the requested-language POI index.
3. A unique name alias from another downloaded language snapshot, mapped
   through the same stable POI identifier.
4. Unresolved source metadata only.

Only cases 1–3 render a `<poi>` tag, using the localized POI name from
the current index. Case 4 is retained in `unresolved_poi_names` and
`poi_resolutions` for QA but **must not be displayed** in tourist-facing
chat. This prevents a stale or foreign-language name from appearing as
an app-openable location.
Numbered source labels such as `1.1 Plaza Vázquez de Molina` are
itinerary **subfolders**, not POIs. They are preserved in
`steps[].subfolders` and rendered as nested text beneath their parent
step, with no `<poi>` tag. This keeps the editorial trip hierarchy
visible without inventing a navigable location.

The current Spanish `trip/4444` source illustrates the behavior:
English source name `Yit El Postigo Hotel` resolves to
`<poi id=30459 type=Hotel>Hotel Yit El Postigo</poi>`; missing source
stop `CR La Casería de Tito` has no current POI record and is omitted.

### 3.10 Route-intent loop safety

The runtime applies a deterministic multilingual route-intent guard before
accepting an answer to walking/cycling/trail questions:

1. A physical-route request must perform exactly one `search_paths`
   lookup. If the small model initially answers with a clarification
   question instead of a tool call, the runtime performs that lookup
   automatically and supplies the result to the next model turn.
2. If no path matches, the model receives one explicit no-route
   instruction and must answer concisely. It may not ask the visitor to
   reformulate the route request and may not substitute a trip.
3. The automatic lookup is bounded to one attempt per user turn. A model
   that still declines to use data cannot trigger repeated instructions
   or an infinite conversation loop.

The intent detector uses normalized route words/stems across supported
languages and rejects short-token prefix matches. This avoids false
classification of ordinary text such as Spanish `se` as a Croatian route
word (`setnja`).

### 3.11 Strict current-turn grounding

Tourist answers use a fail-closed source rule:

1. Every non-social user turn must have a current-turn source retrieval
   (`get_poi`, `get_section`, name/facet/evidence search, trip, or path)
   before an answer is accepted. `list_sections` alone is not enough.
2. If the model tries to answer without retrieval, the runtime gives it
   one internal grounding retry. A second no-tool answer returns a
   localized safe failure message instead of model-memory prose.
3. A concise selection of a validated prior tag is resolved
   deterministically. For example, after
   `<trip id=4453>Ú. en Familia-R. Secundaria 2</trip>`, the user can say
   `Secundaria 2`; the runtime automatically calls
   `get_trip("trip/4453")` before response generation.
4. The turn log carries `grounded`, `grounding_tools`, and
   `automatic_source_calls` for QA. The app should treat an ungrounded
   failure response as ordinary text, not a recommendation.
5. A plan request that finds a curated trip must retrieve `get_trip`
   before rendering a day-by-day or ordered-stop answer. The runtime
   renders the retrieved source steps directly; it does not let the model
   invent named option headings or merge stops from several trips.
6. When a plan-shaped question has no deterministic selection and
   `search_trips` returns ≥2 candidates, the runtime emits a
   deterministic **choice offer**: up to three `<trip id=…>` tags with a
   short description and 2–3 headline POI names. Visitors then pick a
   trip by name (unique substring against a shown label) or by bare
   numeric id (e.g. typing `4457` opens `<trip id=4457>`). If exactly
   one trip matches, the runtime opens it directly; if none match, the
   loop proceeds normally so the model can answer from other tools.

Grounding means the answer is based on the current downloaded index. It
does not promise verbatim wording: the model may paraphrase retrieved
data, but must not introduce location/trip facts without a current
retrieval result.

### 3.12 `get_weather(day?)`

Returns the tourist-safe forecast produced by
`assistant/index_tools.py::format_weather` against the parsed weather
file (§2.4). `day` accepts one of: omitted (full 7-day outlook),
`today`/`tomorrow` (plus localized aliases), an ISO `YYYY-MM-DD` date,
or a weekday name (`monday`…`sunday`, plus localized aliases).

Every rendered day is wrapped in a `<forecast day="YYYY-MM-DD">day_label</forecast>`
tag so a future mobile parser can deep-link to a per-day forecast
screen without re-parsing the file. The tag never appears without a
validated date.

Staleness handling matches the file semantics in §2.4:
- Age ≤ 24 h: rendered as-is.
- 24 h < age ≤ 7 days: prefixed with a localized “estimated forecast
fetched Nd ago” note; the model must not present it as fresh.
- Age > 7 days or file missing: the tool returns the localized
“forecast unavailable” message and the model must not invent one.

Weather intent safety net (mirrors the route safety net in §3.10): if
the visitor turn contains a weather keyword (e.g. `tiempo`, `weather`,
`meteo`, `temperatura`) and the model answers without calling
`get_weather`, the runtime performs one deterministic full-week
lookup, appends the localized `WEATHER_LOOKUP_ENFORCED_INSTRUCTION`
plus the tool result, and lets the model answer once. Bounded to one
attempt per user turn; no loops.

System-prompt hint: when a weather file is loaded, `make_system_prompt`
embeds a single line above the destination overview:
`Today in {destination_display}: {condition}, {temp_min_c}–{temp_max_c}
°C. Consult get_weather for other days.` (~20 tokens). Omitted when
no weather file is available; the outdoor-plan rule still applies.

---

## 4. Text normalization (critical for name search)

`find_poi_by_name` and the precomputed `name_index` must agree exactly.
Port `common/textnorm.py`:

1. NFKD-decompose the string (Unicode normalization form KD).
2. Drop every combining mark (Unicode category M).
3. Replace every run of non-`\w` non-space characters with one space.
4. Lowercase; split on whitespace; rejoin with single spaces.

Reference outputs: `"Vázquez de Molina"` → `"vazquez de molina"`;
`"Sacra Capilla d'El Salvador"` → `"sacra capilla d el salvador"`.

Kotlin: `java.text.Normalizer.normalize(s, Form.NFKD)` +
`replace(Regex("\\p{M}+"), "")` + `replace(Regex("[^\\w\\s]+"), " ")`.
Swift: `s.decomposedStringWithCompatibilityMapping` +
`UnicodeScalar.Properties.isNonspacingMark` filter + the same regex.

---

## 5. System prompt contract

Built once per session. The authoritative template is
`assistant/run_eval.py::_SYSTEM_PROMPT_TEMPLATE`; the older illustrative
excerpt below remains only as background. Ports must mirror the **current
Python template**, including `search_pois`, `search_trips`, `get_trip`,
`search_paths`, `get_path`, tourist-safe output rules, validated POI
tags, and the rule that trips are never physical routes.

```
You are a tourism assistant for {destination}.  You answer visitor questions using the {destination} POI index, which is a structured catalogue of every point of interest, trip and itinerary in the destination.

The full section catalogue is listed below — you do NOT need to call any tool to discover it.  Use this information directly.

You have ELEVEN tools. Pick the one that fits the question:

  • get_section(section_id, sort?, limit?)
        List POIs inside one section.  Returns id + name + a one-line preview.
        Use when the user asks "what X exist?", "list all Y in <category>".

  • get_poi(poi_id)
        Full record of one POI: type, address, phone, coordinates, links, AND the full description paragraph.
        Use when you need facts (address, phone, dates, description) about a specific named POI.  Pass several comma-separated ids ('poi/123,poi/456') to fetch multiple POIs in one call when comparing or synthesising.

  • find_poi_by_name(query, limit?, detail?)
        Fuzzy lookup by POI name.  Returns up to `limit` candidates with id + section + preview.  Use when the user names a place but you don't know which section it lives in.  Pass detail="full" to also get the best match's complete record in the same call; with the default detail="brief", always follow up with get_poi() on the best match before answering specific facts.

  • filter_pois(interest_level?, type?, tourist_type?, section_id?, indispensable?, limit?)
        Facet query.  All filters AND together.  Examples:
          - filter_pois(indispensable=true) → must-see POIs
          - filter_pois(tourist_type="FOOD TOURISM", limit=10) → food spots
          - filter_pois(type="OilMill") → all olive-oil mills
          - filter_pois(interest_level=1, section_id="religious-heritage")

  • search_trips(query, limit?) / get_trip(trip_id)
        Curated theme/day/multi-day visit suggestions. A trip is not a route.

  • search_paths(query, limit?) / get_path(path_id)
        Physical walking/cycling/trail routes only. Never substitute a trip.

  • get_weather(day?)
        Return the downloaded 7-day forecast (or one day: `today`, `tomorrow`, an ISO date, or a weekday name). Use before any outdoor or day-plan recommendation.

  • list_sections()
        Returns the catalogue below.  Rarely needed — sections are pre-loaded.

--- WEATHER HINT ---     (omitted when no weather file is available)
Today in {destination}: {condition}, {temp_min_c}–{temp_max_c} °C. Consult get_weather for other days.

--- DESTINATION OVERVIEW ---
{destination_overview}

--- SECTIONS (pre-loaded, do not fetch again) ---
{sections_text}
--- END SECTIONS ---

RULES:
- Answer based ONLY on what your tools return.  Do not use outside knowledge.
- Always include the description paragraph from get_poi() when answering about a specific place — it carries the most useful detail.
- Quote exact names, addresses, phones, coordinates, and dates when present.
- For "what should I not miss?" / "best of" questions, use filter_pois(indispensable=true) before browsing sections.
- For "tell me about <name>" / "what is <name>" questions, call find_poi_by_name() with detail="full" first — it returns the best match's full record in one call.
- After filter_pois: if the question needs a description, dates, address, phone, architect, or any per-POI detail beyond the name, call get_poi on the most relevant result before answering. For pure listing questions (e.g. "what hotels are there?", "list all museums"), the filter_pois previews already include name + type + interest level, so an extra get_poi call is unnecessary.
- If information is not in the index, say so clearly.
- {lang_rule}
```

`{lang_rule}` is generated by `common/lang_support.py::lang_rule()`
from a single English template parameterised with the language's
English name — no per-language table ships in the app bundle. The
**recovery message** (`recovery_msg()`, §6) uses the same
template pattern, so any language the on-device LLM understands works.
Úbeda EN reference size: system prompt = 7,953 chars ≈ 2,041 tokens
(measured with cl100k_base as a Gemma approximation).

---

## 6. Agentic loop (pseudocode)

Port of `run_agentic_loop` / `run_turn` (`assistant/run_eval.py`,
`assistant/chat_demo.py`). Conversation history is kept in memory for
multi-turn chat; the eval runs single-turn.

```text
MAX_TOOL_ROUNDS = 14
messages = [system_prompt, ...history, user_question]
cache    = prewarmed: for each section id:
           cache[("get_section", id, "interest", 50)] = get_section(id, "interest", 50)

for round in 1..MAX_TOOL_ROUNDS:
    response = llm.chat(messages, tools=TOOL_DEFS, temperature=0)
    messages += response.assistant_message

    if response has no tool_calls:
        answer = response.content; break

    for call in response.tool_calls:
        args = parse_json(call.arguments) or {}
        key  = (call.name, normalized args)
        result = cache[key] or (cache[key] = execute_tool(call.name, args))
        messages += { role: "tool", tool_call_id: call.id, content: result }

if no answer after the loop:
    answer = last non-empty assistant content, else one recovery call:
    llm.chat(messages + [recovery_msg(lang)])   # no tools
```

- **TOOL_DEFS**: the ten JSON schemas in §3, copied verbatim from
  `assistant/run_eval.py::TOOL_DEFS` (descriptions included — the model
  reads them).
- **Temperature 0**, always.
- **Caching:** `get_section` is fully prewarmed per session (18 entries,
  ~2 ms of dict work) — the model's most common call is then free.
- **Tool-call transport:** if the on-device runtime has no native
  function-calling channel, use the standard Gemma tool-call prompt
  format and parse the model's `<tool_call>` / JSON output app-side.
  The loop semantics above do not change.

---

## 7. Budgets (E2B, measured 2026-08-16/19 via oMLX, schema v3 index)

All token figures are **measured** (`response.usage` logged by
`assistant/run_eval.py` and aggregated by `assistant/score_results.py`;
tokenizer counts below use cl100k_base as a Gemma approximation).

### Per-round fixed base (re-sent every round)

| Component | Tokens | Notes |
|---|---|---|
| System prompt | **2,667** | schema-v3 evidence policy + sections overview + destination overview |
| Tool definitions | 832 | six tool schemas, including evidence search |
| **Base total** | **~3.5K per round** | the dominant cost driver is rounds × this base |

### Per-call tool results (context-reduced)

| Call | Tokens | Reduction applied |
|---|---|---|
| `get_section` (grouped section, default) | 745 | −50% vs limit=50 (adaptive default 20 when a group map exists) |
| `get_section` (flat section) | 300–700 | unchanged, full listing |
| `get_poi` per record | ~164 | −13% (media URLs stripped; they stay in the file for the UI) |
| `find_poi_by_name(detail="full")` | preview list + ~164 | fuses the classic find→get pair into one call |

### Whole-question totals (two stochastic 20-question runs)

| Metric | Run 1 | Run 2 | Notes |
|---|---|---|---|
| Rounds / question | 2.6 avg | 2.7 avg | batch `get_poi` + `detail="full"` adopted by the model unprompted |
| Cumulative prompt tokens | 13,950 avg / 22,646 max | 14,477 avg / 28,992 max | run-to-run spread comes from round-count variance, not per-call size |
| Completion tokens / question | 241 avg | 245 avg | answers are compact |
| Quality gates | 0.830 composite, 75% grounding, 95% fetch | identical | unchanged by the reductions |

**Design rule for ports:** rounds are the multiplier — every avoided
tool round saves ~3.5K tokens of re-sent base plus the previous results.
Per-call trimming (grouped section caps, media stripping, fused lookups)
is real but secondary. Plan context headroom for ~2× the observed peak
(~60K) on long multi-turn chats; expire or compact old tool results
beyond that.

Quality gates from the eval harness (`assistant/score_results.py`):
grounding ≥ 70% AND content-fetch ≥ 70%. Any port must reproduce these
on the 20-question English set before shipping (§8).

---
## 7.1 E2B context guardrails (required)

The phone parses the complete index JSON into app memory. The LLM must
**never** receive the full index as prompt context. A Portuguese Alandroal
fixture (173 POIs, 403 KB JSON) measures about 142K tokens with the
reference approximation, above E2B's 128K context window. The tool layer
must expose only bounded source slices.

Apply these hard maximums in the Kotlin and Swift ports:

| Tool | Default | Maximum |
|---|---:|---:|
| `get_section` | 20 grouped / 50 flat | 50 POIs |
| `filter_pois` | 20 | 20 POIs |
| `find_poi_by_name` | 5 | 5 POIs |
| `search_pois` | 10 | 10 POIs |
| `search_trips` / `search_paths` | 10 | 10 results |
| `get_poi` | 1 | 5 comma-separated POIs |

The Python reference also enforces two text limits:

- A complete tool result is capped at **24,000 characters**. Truncate at a
  newline when possible and append a concise internal instruction to refine
  the lookup or retrieve one source.
- Retain at most **120,000 characters** across prior tool messages.
  Replace older tool-message content with a concise omission marker; do not
  delete the tool message, because its assistant tool-call pairing must
  remain valid for OpenAI-compatible transports.

Apply caps even when the model passes a larger explicit `limit`. Tell the
model to refine filters or perform a name search; do not invite it to raise
the limit. The current Alandroal reference stays comfortably bounded:
initial prompt plus schemas ≈4.3K tokens, largest default section ≈1.3K,
largest POI ≈0.8K, and largest trip detail ≈2.2K.

---

## 8. Verifying a port

1. Load `indexes/ubeda_en.json` (tracked in this repo).
2. Implement §3 tools + §5 prompt + §6 loop.
3. Run the 20 questions of `eval/questions.json` single-turn,
   temperature 0, recording per question: answer, tool calls, sections
   accessed, latency.
4. Score with `assistant/score_results.py` (it accepts any
   `results/eval_*.json`-shaped file).
5. Pass bar: grounding ≥ 70% and content-fetch ≥ 70% — the same gates
   the Python reference meets (E2B: 72.5% / 95%).
6. Repeat with `eval/questions_es.json` + `indexes/ubeda_es.json` and
   `eval/questions_it.json` + `indexes/ubeda_it.json` for multilingual
   coverage (E2B reference: ES 0.830, IT 0.760 composite).

Multi-turn behaviour is exercised by `eval/conversations.json`
(4 threads × 3 turns, shared history) — used for manual QA, not scored.

---

## 9. Download & refresh (summary)

Full detail in `docs/cloudflare-worker-spec.md` §5–§6. Essentials:

- `GET /v1/manifest` — catalogue + per-language ETag/bytes.
- `GET /v1/index/{dest}/{lang}` with `If-None-Match` for 304 refresh.
- Keep the last known-good file until its replacement parses cleanly.
- After download, everything in §2–§6 runs offline.

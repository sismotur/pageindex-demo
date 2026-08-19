# Inventrip Offline Assistant — README for the Mobile Team

**Audience:** Android (Kotlin) and iOS (Swift) developers.
**What you will ship:** an offline tourism assistant. The app downloads one
JSON index file per `(destination, language)` and answers visitor questions
on-device with Gemma 4 E2B. No internet connection is needed after the
download.

This document covers the two things you integrate with:

1. **The data supply chain** — where the index comes from (Inventrip API →
   Cloudflare → your app).
2. **The index file format** — every field you will parse.

For the conversational runtime (the five tools, the system prompt, the
agentic loop, verification harness), see the companion document
`docs/mobile-offline-contract.md`. For the server side of the supply
chain, see `docs/cloudflare-worker-spec.md`.

---

## 1. Architecture in 30 seconds

```
Inventrip API (read-only GETs, api_key auth)
      │  weekly cron build (Cloudflare Worker, deterministic, no LLM)
      ▼
Cloudflare R2:  destinations/{slug}/{slug}_{lang}.json  +  meta/manifest.json
      │  HTTPS download when online (ETag / 304 refresh)
      ▼
Phone: parse index JSON once → fully offline
       Gemma 4 E2B + six local lookup tools over the parsed JSON
```

The index file is the **only** artifact your app consumes. Everything in
it is precomputed server-side; the phone never calls the Inventrip API
for chat content.

The app receives final answers with tag-ready POI mentions:
`<poi id=36694 type=OilMill>ALMAZARA BALTASAR LARA Y CÍA.</poi>`. Render
the inner text; only make it tappable after verifying `poi/36694` exists
in the downloaded `pois` map. Unknown tags remain plain text.

---

## 2. API integration (how the index is produced)

You do not call these endpoints from the app — this section exists so you
know exactly what the data means and can trace any field back to its
source.

**Base URL:** `https://api.inventrip.com`
**Auth:** `?api_key=…` query parameter (server-side only; never ship the
  build-time key in the app).
**Common parameters on every call:** `language={lang}` (ISO 639-1,
  16 supported codes) and `strip_nulls=true` (null fields are omitted
  from responses).

### 2.1 Endpoints fetched per (destination, language) build

| Endpoint | Key parameters | What it feeds in the index |
|---|---|---|
| `GET /v120/pois` | `tourist_destination` | The UNE 178503 POI catalogue: names, types, descriptions, interest level, zoom level, addresses, coordinates, contacts. Becomes `pois`, `sections`, `facets`, `name_index`. |
| `GET /v120/tourist-destinations` | `tourist_destination` | Overview text, display name, official URL, tourist types/networks, coordinates, trip/route ids. Becomes `destination_overview` and `meta.destination_display`. |
| `GET /v120/trips` | `tourist_destination`, `add_itinerary=true`, `limit=100` | Curated trips with step itineraries. Becomes `trips[]`. |
| `GET /v120/paths` | `id_path`, `add_itinerary=true` | Walking/driving routes (one call per route id). Stored in the raw snapshot; not surfaced in the index today. |
| `GET /v120/interest-levels` | — | Localized labels for interest levels 1–3. Becomes `interest_levels`. |
| `GET /v120/tourist-types` | — | Tourist-type code → localized display name. Becomes `tourist_type_display`. |

### 2.2 URLs embedded in the index but not fetched at build time

These are **resolved lazily by your app** (they need connectivity at
display time — show images/audio only when online):

| Field | URL pattern |
|---|---|
| `image_urls[]` | `{base}/v100/image/{id}?image_quality=high` |
| `audio_urls[]` | `{base}/v100/audios?language={lang}&offset=1&audio={id}&tourist_destination={dest}` |
| `subject_of_urls[]` | `"Label: https://…"` passthrough strings (external documents) |

### 2.3 Data freshness

The server rebuilds indexes on a weekly cron. Your app learns about new
builds exclusively through the manifest ETags (§3.2) — there is no push
channel.

---

## 3. Downloading the index

### 3.1 Endpoints (Cloudflare Worker)

```
GET /v1/manifest                      → catalogue of everything available
GET /v1/index/{dest}/{lang}           → one index file
```

Auth: `X-Inventrip-Key: <app key>` header. Errors are JSON:
`{"error": "index_not_found", "dest": …, "lang": …}` (404),
`{"error": "unauthorized"}` (401).

### 3.2 Refresh flow (ETag)

1. First run (online): fetch `/v1/manifest`; for each
   `(destination, language)` the user downloads, store the file **and its
   ETag**.
2. Later runs (online): re-fetch the manifest, or send
   `If-None-Match: <stored etag>` per file. `304` = keep local copy;
   `200` = replace atomically; `404` for a language = fall back to the
   `en` copy if present.
3. Never delete a working local copy until its replacement is fully
   downloaded and parses as valid JSON.
4. Offline: no network calls at all.

### 3.3 Manifest shape

```json
{
  "generated_at": "2026-08-16T03:00:00Z",
  "schema_version": 2,
  "destinations": [
    {
      "slug": "ubeda",
      "name": "Úbeda",
      "country": "ES",
      "region": "Andalusia",
      "latitude": 38.0116,
      "longitude": -3.3733,
      "tourist_types": ["HERITAGE TOURISM", "FOOD TOURISM"],
      "languages": {
        "en": { "etag": "\"a1b2…\"", "bytes": 748544,
                "poi_count": 367, "updated_at": "2026-08-16T03:01:12Z" }
      }
    }
  ]
}
```

Use `bytes` to show download sizes in the UI and `poi_count` as a
coverage hint. `country`/`region`/`latitude`/`longitude` let you sort
destinations by proximity.

---

## 4. The index file (schema v3)

One JSON object per file. Reference size: ~0.75 MB (Úbeda, 367 POIs).
Parse it fully into memory at session start — everything is then
dict/map lookups.

### 4.1 Top level

| Key | Type | Purpose |
|---|---|---|
| `meta` | object | identity and versioning (see §4.2) |
| `destination_overview` | string | multi-line overview; embedded in the LLM system prompt, also usable as "About this destination" UI text |
| `trips` | array | curated trips with itinerary steps (§4.3) |
| `sections` | array | the 17 typed sections + fallback, in display order (§4.4) |
| `pois` | object | every POI keyed by id: `"poi/5155"` → record (§4.5) |
| `facets` | object | precomputed id lists for filtering (§4.6) |
| `name_index` | object | normalized name → `poi_id` for fuzzy name search |
| `tourist_type_display` | object | code → localized label, e.g. `"FOOD TOURISM"` → `"Food Tourism"` |
| `interest_levels` | object | `"1"`/`"2"`/`"3"` → localized label |

### 4.2 `meta`

```json
{ "destination": "ubeda", "destination_display": "Úbeda", "lang": "en",
  "generated_at": "2026-08-16T05:13:09Z", "poi_count": 367,
  "section_count": 18, "schema_version": 3 }
```

**Versioning rule:** additive changes bump `schema_version`; readers
must ignore unknown keys. v2 added `sections[].groups`; v3 adds
`facets.search_terms` for deterministic evidence search. Older readers
still work because `sections[].poi_ids` and POI records remain complete.

### 4.3 `trips[]`

```json
{ "trip_id": "…", "name": "…", "description": "…", "url": "…",
  "steps": [ { "step": "Morning", "pois": ["POI name", …] } ] }
```

`steps[].pois` are POI **names** (not ids) — resolve them through
`name_index` after normalization (§4.8) when you need the full record.

### 4.4 `sections[]` and v2 `groups[]`

Sections group POIs by UNE 178503 type. Fixed display order (the first
match wins for multi-typed POIs — e.g. a hotel-monument lands in
Accommodation):

```
unesco-world-heritage-and-city-overview, accommodation,
civil-and-historical-monuments, religious-heritage, museums-and-culture,
archaeological-sites, tourist-attractions-and-viewpoints,
squares-parks-and-natural-areas, gastronomy, guided-tours-and-itineraries,
events-and-festivals, shopping, tourist-information-and-services,
health-and-beauty, practical-information, sports-and-leisure-activities,
quality-rules-and-visitor-advice, other-points-of-interest
```

```json
{ "section_id": "shopping", "title": "Shopping",
  "summary": "66 POIs. Top interests: Shopping… Notable: …",
  "poi_ids": ["poi/…"],                      // ALL POIs, best first
  "groups": [                                 // v2, optional: sections > 30 POIs
    { "group_id": "shopping--store",          // "{section_id}--{type-slug}"
      "title": "Store",                       // the group's UNE display type
      "poi_ids": ["poi/…"],                   // subset, best first
      "summary": "33 POIs. Notable: …" }      // top-3 names preserved
  ] }
```

- `summary` is deterministic (counts + top tourist types + 3 notable
  names) — safe to display as-is.
- `groups[].poi_ids` partition the section: their union equals
  `poi_ids`, without duplicates.
- Sections with ≤ 30 POIs have **no** `groups` key.
- Group `title` values are raw UNE type codes (`Store`, `OilMill`,
  `PlaceOfWorship`); render them as-is or map to your own labels.

### 4.5 `pois` — the POI record

```json
"poi/30117": {
  "poi_id": "poi/30117",
  "name": "Dean Ortega Palace. Tourism Parador",
  "normalized_name": "dean ortega palace tourism parador",
  "description": "…full paragraph(s), never truncated…",
  "types": ["Hotel", "LodgingBusiness", "CivilBuilding", "Restaurant"],
  "display_type": "Hotel",
  "tourist_types": ["SHORT BREAK", "FOOD TOURISM"],
  "display_tourist_types": ["Short Break", "Food Tourism"],
  "interest_level": 1,
  "interest_level_label": "Indispensable",
  "zoom_level": 17,
  "booking_url": "https://…",
  "url": ["https://…"],
  "telephone": ["+34953750345"],
  "email": ["…"],
  "street_address": "Plaza Vázquez de Molina",
  "address_locality": "Úbeda",
  "address_province": "Jaén",
  "address_region": "Andalusia",
  "postal_code": "23400",
  "country_code": "ES",
  "country": "Spain",
  "latitude": 38.008203654724,
  "longitude": -3.36723561910189,
  "image_urls": ["https://api.inventrip.com/v100/image/40491?image_quality=high"],
  "audio_urls": ["https://api.inventrip.com/v100/audios?…"],
  "subject_of_urls": ["Brochure: https://…"],
  "start_date": "",
  "end_date": "",
  "raw_extras": { }
}
```

Nullability and empty-value rules (the builder strips aggressively):

| Field | Type | Empty means |
|---|---|---|
| `interest_level` / `_label` | int 1–3 or **null** | not editorially ranked (level 0 in the API is dropped) |
| `zoom_level` | int 10–19 or **null** | no map-prominence hint; `≤ 16` means "major landmark" |
| `latitude` / `longitude` | float or **null** | no coordinates — do not place on a map |
| `url` `telephone` `email` `image_urls` `audio_urls` `subject_of_urls` `types` `tourist_types` | array, may be `[]` | absent |
| `booking_url` `start_date` `end_date` | string, may be `""` | absent; dates are only set for events |
| `display_tourist_types` | array of labels | localized; falls back to title-cased code |
| `description` | string, may be `""` | no text available |
| `raw_extras` | object | passthrough UNE `extras` (opening hours, prices, …). The assistant tools never read it; your UI may. |

`start_date`/`end_date` come from the API's `startDate`/`endDate` and
identify event POIs (section `events-and-festivals`).

### 4.6 `facets` — precomputed filters

All values are lists of `poi_id`, unsorted:

| Map | Keyed by | Example |
|---|---|---|
| `by_section` | `section_id` | `"gastronomy"` → 40 ids |
| `by_type` | UNE type code | `"OilMill"` → 7 ids |
| `by_tourist_type` | raw type code | `"FOOD TOURISM"` → ids |
| `by_interest_level` | `"1"`/`"2"`/`"3"` | `"1"` → indispensable ids |
| `by_zoom_bucket` | `"<=14"`, `"15-16"`, `"17-19"` | map-prominence bands |
| `indispensable` | — (list) | all `interest_level == 1` ids |
| `search_terms` | normalized word | sorted POI ids whose visitor-facing record contains the word |

Filtering with AND semantics (e.g. "indispensable food spots") is set
intersection: `by_tourist_type["FOOD TOURISM"] ∩ indispensable`. Sort
results by `(interest_level, zoom_level, normalized_name)`, nulls last.

### 4.7 Same-record evidence search

`facets.search_terms` is a deterministic inverted full-text index over
each POI's name, description, category label, tourism-interest labels,
and locality. The offline `search_pois` tool intersects postings for every
query term to verify that a compound request is supported by the **same**
POI.

Example: `olive oil restaurant` has no direct result in the current
Úbeda English catalogue. The data supports olive-oil places and
restaurants separately, but does not assert that any restaurant serves
olive-oil cuisine. The assistant must explain that evidence gap in normal
tourist language, then offer separately labelled complementary options;
it must never invent the relationship.

### 4.8 `name_index` and name search

`name_index` maps **normalized** POI names to `poi_id`. It is lossy on
collisions (first writer wins; < 2% of POIs in practice) — treat it as
the fast path, not the only path.

The reference fuzzy search (port of `assistant/index_tools.py`) ranks:

1. exact normalized match in `name_index`,
2. names containing **all** query tokens (or the whole normalized query
   as substring),
3. names containing **any** query token,

ties broken by interest level. Return the top 5.

### 4.9 Text normalization — must match exactly

Indexed names and user queries only meet if both sides normalize
identically (port of `common/textnorm.py`):

1. NFKD-decompose (Unicode compatibility decomposition).
2. Drop all combining marks (category M).
3. Replace every run of non-`\w` non-space characters with one space.
4. Lowercase; split on whitespace; rejoin with single spaces.

Reference: `"Vázquez de Molina"` → `"vazquez de molina"`;
`"Sacra Capilla d'El Salvador"` → `"sacra capilla d el salvador"`.

Kotlin: `Normalizer.normalize(s, Normalizer.Form.NFKD)` +
`replace(Regex("\\p{M}+"), "")` + `replace(Regex("[^\\w\\s]+"), " ")`.
Swift: `decomposedStringWithCompatibilityMapping` +
`UnicodeScalar.Properties.isNonspacingMark` filter + same regex.

---

## 5. Minimal data-flow example (no LLM involved)

Everything below is plain JSON work — the same calls the on-device tools
make:

```kotlin
// load once per (destination, language)
val index = Json.parseToJsonElement(File("ubeda_en.json").readText()).jsonObject
val pois  = index.getValue("pois").jsonObject

// "best restaurants" → Gastronomy section, Restaurant group, top entries
val gastro = index.getValue("sections").jsonArray
    .first { it.jsonObject["section_id"]!!.jsonPrimitive.content == "gastronomy" }
val restaurantIds = gastro.jsonObject["groups"]?.jsonArray
    ?.first { it.jsonObject["title"]!!.jsonPrimitive.content == "Restaurant" }
    ?.jsonObject?.get("poi_ids")?.jsonArray
    ?: gastro.jsonObject.getValue("poi_ids").jsonArray   // flat fallback
val top = restaurantIds.take(5).map { pois.getValue(it.jsonPrimitive.content) }

// "tell me about X" → name search
val id = index.getValue("name_index").jsonObject["sacra capilla del salvador"]
val poi = id?.let { pois[it.jsonPrimitive.content] }
```

The LLM layer on top of this (tools, prompt, loop) is specified in
`docs/mobile-offline-contract.md` §§3–6.

---

## 6. Reference numbers (Úbeda, English, measured)

| Metric | Value |
|---|---|
| Index file size | 993 KB / 127 KB gzip (schema v3, English) |
| POIs / sections | 367 / 18 (4 sections carry `groups`) |
| Parse time on desktop | < 10 ms |
| Assistant quality gate (E2B) | grounding 75.0% / content-fetch 95% — both pass |
| Latency per question (E2B, desktop MLX) | 3.1 s avg |
| Tokens per question | 13,950 prompt + 241 completion avg; 22.6K prompt peak |

Rebuild cadence: weekly. Language availability per destination varies;
the manifest is the source of truth.

---

## 7. Where things live

| What | Where |
|---|---|
| This document | `docs/README-mobile.md` |
| Runtime contract (tools, prompt, loop, verification) | `docs/mobile-offline-contract.md` |
| Server-side build & distribution spec | `docs/cloudflare-worker-spec.md` |
| Reference implementation (Python) | `assistant/` (tools) · `pipeline/` (index builder) |
| Sample index files | `indexes/ubeda_{en,es,it}.json` (tracked in git) |
| Sample question sets for QA | `eval/questions_{,es_,it_}json` |

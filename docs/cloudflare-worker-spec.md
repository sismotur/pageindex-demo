# Cloudflare Worker — Inventrip Data Pipeline & Distribution: Technical Specification

**Version:** 3.0
**Status:** Draft
**Scope:** 200 tourist destinations × 16 languages
**Relates to:** `pipeline/extract_pois.py`, `pipeline/extract_destination_data.py`,
  `pipeline/build_index.py`, `common/textnorm.py`, `docs/mobile-offline-contract.md`

> **v3.0 rewrite:** the architecture changed fundamentally. LLM inference moved
> **on-device** (Gemma 4 E2B on Android/iOS, fully offline). This Worker no
> longer runs any RAG/chat loop — its only jobs are:
>
> 1. **Data preparation** (cron): fetch the Inventrip API per
>    `(destination, language)` pair, build the POI-aware index in TypeScript,
>    write it to R2.
> 2. **Distribution** (HTTP): serve the index files to mobile clients with
>    ETag-based refresh, so phones update their offline corpora when — and
>    only when — they have connectivity.
>
> The v2.0 server-side chat design (Durable Objects, SSE streaming, LLM
> endpoint) is retired. The conversational runtime now lives on the phone;
> `assistant/` is its Python reference implementation.

---

## 1. Goals

- **200 tourist destinations** built and served from a single Worker
  deployment, each with one self-contained index per language in R2.
- **16 languages** per destination, built incrementally (top languages first;
  see §8 Rollout).
- **Fully-offline mobile clients**: after downloading an index file once,
  the phone answers questions with zero network access. The Worker is only
  involved when the app refreshes its corpora.
- **Deterministic builds**: the TypeScript index builder is a 1:1 port of
  `pipeline/build_index.py`. Same inputs → byte-equivalent output (modulo
  `meta.generated_at`). No LLM calls anywhere in this Worker.
- The GKE cluster continues to serve the existing Inventrip API unchanged;
  the Worker is a read-only API consumer.

---

## 2. Architecture

```
                    ┌────────────────────────────────────────┐
                    │        Inventrip API (GKE, existing)   │
                    │  /v120/pois · /v120/tourist-destinations│
                    │  /v120/trips · /v120/paths · …         │
                    └───────────────┬────────────────────────┘
                                    │ read-only, api_key auth
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│              Cloudflare Worker "inventrip-rag-data"          │
│                                                              │
│  cron (weekly) ──► build pipeline (TypeScript)               │
│      for each (destination, lang):                           │
│        fetch API ──► buildIndex() ──► R2 put                 │
│      then rebuild meta/manifest.json                         │
│                                                              │
│  HTTP ──► GET /v1/manifest          (JSON, short cache)      │
│       ──► GET /v1/index/{dest}/{lang} (R2 stream, ETag/304)  │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTPS, only when online
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              Mobile app (Android / iOS)                      │
│  downloads indexes/{dest}_{lang}.json once, then offline:    │
│  Gemma 4 E2B + 5 local tools over the index                  │
│  (contract: docs/mobile-offline-contract.md)                 │
└──────────────────────────────────────────────────────────────┘
```

No Durable Objects, no sessions, no SSE, no LLM endpoint. The Worker is a
build-and-serve static-artifact pipeline with an HTTP façade.

---

## 3. R2 Bucket Structure

One bucket (**`inventrip-rag`**) holds everything:

```
inventrip-rag/
├── meta/
│   └── manifest.json                # ← download catalogue (see §5)
└── destinations/
    ├── ubeda/
    │   ├── ubeda_en.json            # POI-aware index (English)
    │   ├── ubeda_es.json            # Spanish
    │   └── ubeda_it.json            # additional languages as built
    ├── baeza/
    │   └── ...
    └── ... (198 more)
```

Object keys mirror the local pipeline output exactly:
`destinations/{slug}/{slug}_{lang}.json`.

**Storage (all languages):** 200 destinations × 16 languages × ~0.75 MB
≈ **2.4 GB**. R2 at $0.015/GB-month ≈ $0.04/month. Negligible; the API
fetch volume is the real cost driver.

Objects are **immutable between pipeline runs** — they change only when the
cron job rebuilds them. Every `put` rotates the ETag, which is the mobile
client's change signal (§6).

---

## 4. Build Pipeline (cron trigger)

### 4.1 Trigger

```toml
# wrangler.toml (excerpt)
[triggers]
crons = ["0 3 * * 0"]    # weekly, Sunday 03:00 UTC
```

Cron invocations allow up to 15 minutes of wall time. One
`(destination, language)` build costs ~5 HTTP fetches and < 50 ms CPU;
200 × 16 pairs ≈ 3,200 builds. The handler processes pairs sequentially
with bounded concurrency (8 in flight), checkpointing progress to R2 so a
timeout can resume on the next run (§4.5).

### 4.2 Per-pair build steps

A TypeScript port of the Python pipeline. For a pair `(dest, lang)`:

1. `fetchPois(dest, lang)` — port of `pipeline/extract_pois.py`.
   `GET /v120/pois?tourist_destination={dest}&language={lang}&strip_nulls=true&api_key=…`
2. `fetchDestinationData(dest, lang)` — port of
   `pipeline/extract_destination_data.py`. Five endpoints:
   `/v120/tourist-destinations`, `/v120/trips?add_itinerary=true`,
   `/v120/paths?id_path=…` (one per route id), `/v120/interest-levels`,
   `/v120/tourist-types`.
3. `buildIndex(rawPois, destData, lang, dest)` — port of
   `pipeline/build_index.py`. Pure transformation, no I/O:
   - `normalizePoi()` per record (localized fields, image/audio/document
     URL builders, interest-level mapping)
   - `assignSection()` — the SECTIONS priority table (17 typed sections +
     "Other Points of Interest"), same order, same titles
   - `buildSectionGroups()` — schema v2: sections with > 30 POIs are split
     into per-`display_type` groups (min group size 2, remainder folds
     into "Other"), each group carrying sorted `poi_ids` and a key-item
     summary; groups ordered best-POI-first
   - `buildSectionSummary()` — deterministic counts + top interests +
     3 notable POIs
   - `buildFacets()` — by_section / by_type / by_tourist_type /
     by_interest_level / by_zoom_bucket / indispensable, plus schema-v3
     `search_terms`: normalized word → sorted POI ids over visitor-facing
     name/description/category/tourism/locality fields
   - `buildItineraries()` — schema-v4 localized waypoint resolution:
     `/trips` records become editorial trip suggestions; `/paths` records
     become physical route candidates. Each ordered step preserves
     resolved `poi_ids` and `unresolved_poi_names`; never substitute a
     trip for a path.
   - `name_index` — **critical:** keyed by `normalizeText(name)` where
     `normalizeText` is the exact port of `common/textnorm.py`
     (NFKD → strip combining marks → non-word runs to single space →
     lowercase → collapse whitespace). If this diverges,
     `find_poi_by_name` breaks on-device.
4. `R2.put(key, json)` with `httpMetadata: { contentType: "application/json" }`.

The Python pipeline remains the **reference implementation**. The contract
test in §4.4 keeps the port honest. The index schema version
(`meta.schema_version`, currently **4**) must match between the two
implementations — bumping it is a coordinated change.

### 4.3 manifest.json

After all pairs finish, rebuild `meta/manifest.json`:

```json
{
  "generated_at": "2026-08-16T03:00:00Z",
  "schema_version": 1,
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
        "en": { "etag": "\"a1b2…\"", "bytes": 737280, "poi_count": 367,
                "updated_at": "2026-08-16T03:01:12Z" },
        "es": { "etag": "\"c3d4…\"", "bytes": 820224, "poi_count": 369,
                "updated_at": "2026-08-16T03:01:19Z" }
      }
    }
  ]
}
```

The manifest doubles as the **destination catalogue** the app shows for
download (name, region, size per language) and the change-detection feed
(per-language ETag). Total size for 200 destinations ≈ 150 KB.

### 4.4 Contract test against the Python reference

CI step (runs locally or in a nightly Worker cron against one canary
destination, e.g. `ubeda`):

1. Run `pipeline/build_index.py --destination ubeda --lang en` locally.
2. Run the TS `buildIndex()` over the same two `data/` snapshots.
3. Assert deep equality of both outputs after stripping
   `meta.generated_at`.

Any regression in the TS port (normalization, section assignment, facet
ordering) fails this test before it can corrupt the mobile corpora.

### 4.5 Checkpointing & partial rebuilds

- A `meta/build_state.json` object records `{ pair: last_built_at }`.
- The cron skips pairs whose source data is unchanged when the Inventrip
  API exposes per-destination `updated_at`; otherwise it rebuilds
  everything weekly (cheap enough).
- On timeout, unprocessed pairs resume on the next cron run.

---

## 5. HTTP API (distribution)

### `GET /v1/manifest`

Returns `meta/manifest.json`. `Cache-Control: public, max-age=300`.
Mobile clients call this on app start **when online** to detect updates.

### `GET /v1/index/{dest}/{lang}`

Streams `destinations/{dest}/{dest}_{lang}.json` from R2.

- **ETag / If-None-Match:** returns `304 Not Modified` when the client's
  stored ETag matches — refresh costs a few hundred bytes.
- `Cache-Control: public, max-age=86400, immutable` is wrong here (content
  changes weekly); use `public, max-age=3600` and rely on ETag
  revalidation instead.
- `Content-Type: application/json; charset=utf-8`.
- 404 JSON body when the pair has not been built:
  `{ "error": "index_not_found", "dest": "…", "lang": "…" }` — the app
  falls back to the English index (`…/{dest}/en`) when available.

```typescript
async function serveIndex(env: Env, dest: string, lang: string,
                          request: Request): Promise<Response> {
  const key = `destinations/${dest}/${dest}_${lang}.json`;
  const obj = await env.RAG_BUCKET.get(key);
  if (!obj) {
    return Response.json(
      { error: "index_not_found", dest, lang }, { status: 404 });
  }
  const headers = new Headers({
    "Content-Type":  "application/json; charset=utf-8",
    "Cache-Control": "public, max-age=3600",
    "ETag":          obj.etag,
  });
  if (request.headers.get("If-None-Match") === obj.etag) {
    return new Response(null, { status: 304, headers });
  }
  return new Response(obj.body, { headers });
}
```

### Authentication

The mobile app already holds an Inventrip API key. Reuse it: requests must
carry `X-Inventrip-Key`; the Worker validates it against a secret list
(`wrangler secret put MOBILE_API_KEYS`). Downloads are otherwise
unmetered. Rate-limit per key at the Cloudflare level (50 req/day is far
above real needs — a client downloads ≤ a few indexes per week).

---

## 6. Mobile Refresh Flow (client-side contract)

1. **First run (online):** app fetches `/v1/manifest`, lets the user pick
   destinations + languages, downloads each
   `/v1/index/{dest}/{lang}`, stores the file and its ETag.
2. **Offline operation:** no network calls at all. The on-device runtime
   loads the local index JSON and answers via the five tools
   (see `docs/mobile-offline-contract.md`).
3. **Refresh (online, e.g. on app start):** conditional GET with the
   stored ETag per downloaded pair. `304` → keep local copy; `200` →
   atomically replace the file and the stored ETag; `404` → language was
   withdrawn, fall back to `en` copy if present.
4. **Failure tolerance:** a failed refresh never deletes a working local
   copy. The last good index is always kept until its replacement is
   fully downloaded and parsed.

---

## 7. Worker Environment (`Env`)

```typescript
interface Env {
  RAG_BUCKET: R2Bucket;             // bound in wrangler.toml

  // Inventrip API (build pipeline only)
  INVENTRIP_API_BASE_URL: string;   // https://api.inventrip.com
  INVENTRIP_API_KEY: string;        // secret: wrangler secret put

  // Download auth
  MOBILE_API_KEYS: string;          // secret, comma-separated

  // Build scope (vars)
  DESTINATIONS: string;             // "all" or comma-separated slugs
  LANGS: string;                    // "en,es,it" — start small, see §8
}
```

`wrangler.toml` binds the bucket and vars; all keys are secrets, never
plaintext in the repo.

---

## 8. Rollout Plan

**Phase 1 — pilot:** destinations = `ubeda` only; `LANGS=en,es,it`
(matching the evaluated baselines). Cron weekly. Internal TestFlight /
Play Console track validating the offline E2B runtime against
`/v1/index/ubeda/*`.

**Phase 2 — top destinations:** the ~20 highest-traffic destinations;
`LANGS=en,es,fr,de`. Measure download sizes and refresh behaviour in the
field.

**Phase 3 — full catalogue:** all 200 destinations; languages extended to
the full 16 based on measured per-language demand (storage is cheap; the
API fetch volume is the cost). Add the §4.4 contract test as a cron
canary over `ubeda` on every run.

---

## 9. Performance & Cost Budget

| Phase | Time | Notes |
|---|---|---|
| Per-pair API fetches | ~2–4 s wall | 5–10 HTTP calls, I/O-bound |
| `buildIndex()` per pair | < 50 ms CPU | 367-POI JSON transform |
| Full 200×16 rebuild | ~2 h wall | spread over cron with checkpointing |
| `GET /v1/index` (cached PoP) | ~10 ms | R2 + Workers Cache API |
| `GET /v1/index` (cold) | ~50 ms | one R2 read, ~0.75 MB |
| Worker CPU per HTTP request | < 5 ms | far under the 30 s limit |

Cost at Phase 3 scale: R2 storage ≈ $0.04/month; Class B operations
(index reads) dominate and stay trivial at realistic refresh rates.

---

## 10. Error Handling

| Condition | Behaviour |
|---|---|
| Inventrip API 5xx during build | retry 3× with backoff; skip pair, log to `build_state.json` |
| Inventrip API 401 | abort run, alert (key rotation needed) |
| Pair produces empty POI list | do NOT overwrite the existing R2 object; keep previous index |
| Build timeout | resume from `build_state.json` on next cron |
| Download: unknown dest/lang | 404 JSON; client falls back to English |
| Download: bad/missing API key | 401; client shows "cannot check for updates" |
| Manifest missing/corrupt | 500; client keeps using local copies (offline-first) |

---

## 11. Open Questions

1. **Per-destination change detection** — does the Inventrip API expose
   `updated_at` per destination so the cron can skip unchanged pairs?
   If not, weekly full rebuilds are acceptable (see §9).
2. **Download auth model** — per-app shared key vs. per-user keys issued
   by the existing Inventrip auth service.
3. **Compression** — indexes are ~70% compressible with gzip
   (~0.75 MB → ~0.2 MB). Enable when mobile data usage matters; R2 +
   Cloudflare serve `Content-Encoding: gzip` transparently.
4. **Delta updates** — not planned; full-file replacement at ~0.75 MB is
   simpler and cheap enough. Revisit only if destinations grow an order
   of magnitude.

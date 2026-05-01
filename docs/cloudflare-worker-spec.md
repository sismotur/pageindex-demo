# Cloudflare Worker — Inventrip RAG: Technical Specification

**Version:** 2.0
**Status:** Draft
**Scope:** 200 tourist destinations × 16 languages
**Relates to:** `scripts/build_index.py`, `scripts/index_tools.py`,
  `scripts/run_eval.py`, `scripts/chat_demo.py`

> **v2.0 update:** the corpus format changed from `{structure.json + guide.md}`
> (PageIndex tree + Markdown) to a single self-contained POI-aware index
> (`indexes/{dest}_{lang}.json`) produced by `scripts/build_index.py`.
> The Worker no longer manipulates Markdown line ranges; it serves five
> dict-lookup tools (`list_sections`, `get_section`, `get_poi`,
> `find_poi_by_name`, `filter_pois`) directly off the index.

---

## 1. Goals

Move the RAG orchestration layer out of GKE into a Cloudflare Worker so
that:

- The service runs at Cloudflare's edge (300+ PoPs) with no additional
  infrastructure to manage.
- **200 tourist destinations** are served from a single Worker deployment,
  each with its own corpus stored in R2.
- **16 languages** are supported; the response language is set once per
  session and never changes mid-conversation.
- Multiple data sources (Inventrip API, weather, events) can be fetched
  in parallel and injected into the LLM context.
- **Cross-destination queries** (e.g. “best restaurants in Spain”) are
  served by a meta-index that identifies relevant destinations before
  routing to their individual corpora.
- The full response is streamed token-by-token to the client from the
  moment the LLM produces the first word.
- **Strict destination isolation:** when a session has a destination set,
  the Worker is physically incapable of returning data from any other
  destination. Cross-destination queries require an explicit separate
  endpoint and are never triggered by a regular `/v1/chat` call.
- The GKE cluster continues to serve the existing Inventrip API unchanged.

---

## 2. Architecture

```
Client (browser / mobile)
    │  POST /v1/chat  (JSON body, includes destination + session_id)
    │  ← SSE stream   (text/event-stream)
    ▼
Cloudflare Worker  (this spec)
    ├── R2: inventrip-rag/             ← one bucket, all destinations
    │     ├── meta/
    │     │   └── all_destinations.json  ← cross-destination meta-index
    │     └── destinations/
    │         ├── ubeda/
    │         │   ├── ubeda_en.json    ← POI-aware index (built by build_index.py)
    │         │   ├── ubeda_es.json    ← Spanish index
    │         │   └── ubeda_fr.json    ← (additional languages as needed)
    │         └── baeza/ ... granada/ ...
    ├── Durable Objects: RagSession    ← session: language + history
    │
    ├── fetch → llm.inventrip.com/v1  ← LLM inference (Ollama / Gemma 4)
    ├── fetch → api.inventrip.com     ← live data enrichment
    └── fetch → external APIs         ← weather, events, etc.
```

The Worker implements the complete agentic loop in TypeScript. No
Python runtime is involved in the request path. The retrieval tools
are the same five exposed by `scripts/index_tools.py` (Python) — the
TypeScript port loads the index JSON once per request and serves
`get_section`, `get_poi`, `find_poi_by_name`, `filter_pois`, and
`list_sections` from in-memory dicts.

**Single-destination query** (most requests): load the
`{destination}_{lang}.json` index from R2, run the agentic loop, stream
the answer.

**Cross-destination query** (e.g. "best restaurants in Spain"): load
the meta-index from R2, identify the top-N relevant destinations, run
one agentic loop per destination in parallel, merge and stream a
synthesised answer.

---

## 3. R2 Bucket Structure

One bucket (**`inventrip-rag`**) holds all 200 destinations. The path
scheme is `destinations/{slug}/{slug}_{lang}.json` — one self-contained
POI-aware index per `(destination, language)` pair. Adding a destination
or language never requires a Worker re-deploy.

```
inventrip-rag/
├── meta/
│   └── all_destinations.json      # ← see §17: cross-destination index
└── destinations/
    ├── ubeda/
    │   ├── ubeda_en.json          # POI-aware index (English)
    │   ├── ubeda_es.json          # POI-aware index (Spanish)
    │   └── ubeda_fr.json          # additional languages as needed
    ├── baeza/
    │   └── ...
    └── ... (198 more)
```

Each index file contains everything the Worker needs: destination
overview, trips, sections with deterministic summaries, all POIs
keyed by id, facet lookups, and the name index for fuzzy search.
No Markdown, no separate metadata file, no LLM-generated summaries.

**Storage estimate (v1, English-only):**
200 destinations × ~720 KB = **144 MB** total. R2 at $0.015/GB/month
≈ $0.002/month. Negligible.

**Storage estimate (v2, all languages):**
200 destinations × 16 languages × ~720 KB = **2.3 GB**. Still cheap
($0.035/month). The data-pipeline cost (Inventrip API calls + index
build) is the real constraint, not R2 storage.

Objects are **immutable during a session** — they change only when the
Inventrip data pipeline runs (weekly Cloud Run job). The Worker caches
all objects in the [Workers Cache API][wcache] after the first fetch;
subsequent requests on the same PoP serve them from memory.

[wcache]: https://developers.cloudflare.com/workers/runtime-apis/cache/

### Uploading index files

Local artifact names follow the `{dest}_{lang}.json` convention.
The upload mirrors them straight into R2:

```bash
# Upload a single (destination, language) pair:
DEST=ubeda
LANG=en

wrangler r2 object put inventrip-rag/destinations/$DEST/${DEST}_${LANG}.json \
  --file indexes/${DEST}_${LANG}.json

# Upload the meta-index (after all destinations are processed):
wrangler r2 object put inventrip-rag/meta/all_destinations.json \
  --file data/all_destinations.json
```

The Cloud Run data pipeline iterates all `(destination, language)` pairs,
runs `scripts/build_index.py` for each, uploads the resulting index, and
then rebuilds `all_destinations.json`.

---

## 4. Worker Environment (`Env`)

```typescript
interface Env {
  // R2 bucket holding the corpus for all destinations
  RAG_BUCKET: R2Bucket;

  // Durable Objects: one instance per session_id
  // Stores { lang, history: ChatMessage[], destination: string }
  SESSIONS: DurableObjectNamespace;

  // LLM inference endpoint (Ollama OpenAI-compatible)
  LLM_ENDPOINT: string;      // e.g. https://llm.inventrip.com/v1
  LLM_API_KEY: string;       // bearer token for llm.inventrip.com

  // Inventrip API (for live data enrichment)
  INVENTRIP_API_BASE_URL: string;   // https://api.inventrip.com
  INVENTRIP_API_KEY: string;

  // Supported languages (comma-separated for validation)
  SUPPORTED_LANGS: string;   // "en,es,fr,de,it,pt,ca,de,nl,pl,ru,zh,ja,ar,tr,uk"
}
```

`wrangler.toml` binds `RAG_BUCKET` to the R2 bucket and reads secrets
from the Workers Secrets store (never in `wrangler.toml` plaintext).

The destination is **not** a static environment variable — it comes
from the request body or the session, allowing one Worker deployment
to serve all 200 destinations.

---

## 5. Request / Response Format

### Session creation (`POST /v1/session`)

Called once at the start of a conversation. The language and destination
are **locked** for the entire session — the model never switches
language mid-conversation.

```
POST /v1/session
Content-Type: application/json

{
  "destination": "ubeda",   // destination slug; omit for cross-dest mode
  "lang":        "es"        // language code: en | es | fr | de | it | pt | ...
}

→ 200 { "session_id": "uuid", "destination": "ubeda", "lang": "es" }
```

### Single-destination chat (`POST /v1/chat`)

```
POST /v1/chat
Content-Type: application/json

{
  "question":   "¿Dónde puedo aparcar en Úbeda?",
  "session_id": "uuid",         // required after /v1/session
  "enrich":     ["weather"]     // optional live data modules
}
```

### Cross-destination chat (`POST /v1/meta-chat`)

```
POST /v1/meta-chat
Content-Type: application/json

{
  "question":   "Cules son los mejores restaurantes en España?",
  "session_id": "uuid",         // language resolved from session
  "scope":      "spain"         // optional: country / region filter
}
```

### Query routing rules and destination isolation

**Rule 1 — session with destination set → strict single-destination mode.**
Every `/v1/chat` call on a session that has a `destination` field loads
*only* the index for that destination from R2. The five retrieval tools
(`get_section`, `get_poi`, `find_poi_by_name`, `filter_pois`,
`list_sections`) operate exclusively on that destination's in-memory
index. There is no code path that can read another destination's data
during the same session. Any answer the model produces is grounded
solely in the loaded index.

```
Session { destination: "baeza" }  →  R2: destinations/baeza/**
                                       NEVER reads destinations/ubeda/**
                                       or any other destination
```

This is structural, not a runtime check: the R2 key prefix is derived
directly from `session.destination`, so accessing a different
destination’s data is physically impossible within the same request.

**Rule 2 — session without destination → cross-destination mode only.**
A session created without a `destination` field cannot call `/v1/chat`
(returns 400). Cross-destination queries must go through `/v1/meta-chat`,
which uses the meta-index to identify relevant destinations and then
runs one isolated `ragLoop` per destination. The final synthesis step
combines results but clearly attributes each fact to its source destination.

```
Session { destination: null }  →  /v1/meta-chat only
                                  /v1/chat → 400 Bad Request
```

**Rule 3 — destination cannot change mid-session.**
Once set, `session.destination` is immutable in the Durable Object.
Subsequent `/v1/chat` calls ignore any `destination` field in the
request body and always use the session value. This prevents
a client from querying a different destination by passing a different
slug in a later request.

| Session type | Endpoint | Corpus loaded | Cross-destination |
|---|---|---|---|
| `destination: "baeza"` | `/v1/chat` | `destinations/baeza/` only | ❌ Not possible |
| `destination: null` | `/v1/meta-chat` | meta-index + N per-dest corpora | ✅ Explicit |
| `destination: null` | `/v1/chat` | — | 400 error |

### Response (SSE stream)

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no

data: {"type":"status","text":"Searching Practical Information…"}

data: {"type":"token","text":"Puedes "}
data: {"type":"token","text":"aparcar "}
data: {"type":"token","text":"en "}
...
data: {"type":"done","latency_ms":4200,"tool_calls":2,"cache_hits":1}
```

Event types:
- `status` — tool call in progress (spinner text for the client)
- `token` — one streamed text fragment
- `done` — stream complete, includes timing metadata
- `error` — unrecoverable failure

---

## 6. Static Asset Loading

Assets are loaded once per Worker invocation and cached. The pattern
uses the Workers Cache API with the R2 ETag as the cache key, ensuring
the Worker automatically picks up corpus updates without a re-deploy.

```typescript
async function loadAsset(
  env: Env, cache: Cache, key: string
): Promise<string> {
  const cacheUrl = `https://rag-cache/${key}`;
  const cached   = await cache.match(cacheUrl);
  if (cached) return cached.text();

  const obj = await env.RAG_BUCKET.get(key);
  if (!obj) throw new Error(`R2 object not found: ${key}`);

  const text     = await obj.text();
  const response = new Response(text, {
    headers: {
      "Cache-Control": "public, max-age=3600",
      "ETag":          obj.etag,
    },
  });
  await cache.put(cacheUrl, response.clone());
  return text;
}
```

---

## 7. Navigation Helpers (TypeScript port)

These are direct ports of the Python helpers in `scripts/index_tools.py`.
All five operate on a single `Index` object loaded once per request.

```typescript
interface Section {
  section_id: string;
  title:      string;
  summary:    string;
  poi_ids:    string[];
}

interface Poi {
  poi_id:                 string;
  name:                   string;
  normalized_name:        string;
  description:            string;
  display_type:           string;
  display_tourist_types:  string[];
  interest_level:         number | null;
  interest_level_label:   string | null;
  zoom_level:             number | null;
  street_address:         string;
  address_locality:       string;
  postal_code:            string;
  country:                string;
  latitude:               number | null;
  longitude:              number | null;
  telephone:              string[];
  email:                  string[];
  url:                    string[];
  booking_url:            string;
  image_urls:             string[];
  audio_urls:             string[];
  subject_of_urls:        string[];
  // ...other fields from build_index.py
}

interface Index {
  meta:                  { destination: string; destination_display: string;
                           lang: string; poi_count: number };
  destination_overview:  string;
  trips:                 { trip_id: string; name: string;
                           description: string; steps: { step: string; pois: string[] }[] }[];
  sections:              Section[];
  pois:                  Record<string, Poi>;
  facets: {
    by_section:        Record<string, string[]>;
    by_type:           Record<string, string[]>;
    by_tourist_type:   Record<string, string[]>;
    by_interest_level: Record<string, string[]>;
    by_zoom_bucket:    Record<string, string[]>;
    indispensable:     string[];
  };
  name_index:            Record<string, string>;
  tourist_type_display:  Record<string, string>;
  interest_levels:       Record<string, string>;
}

function normalize(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function formatSectionsOverview(idx: Index): string {
  const lines: string[] = [
    `Destination: ${idx.meta.destination_display}  (${idx.meta.poi_count} POIs across ${idx.sections.length} sections)`,
    "",
    "SECTIONS:",
  ];
  for (const s of idx.sections) {
    lines.push(`  [${s.section_id}] ${s.title}  (${s.poi_ids.length} POIs)`);
    if (s.summary) lines.push(`      ${s.summary}`);
  }
  return lines.join("\n");
}

function findSection(idx: Index, key: string): Section | null {
  if (!key) return null;
  const exact = idx.sections.find(s => s.section_id === key);
  if (exact) return exact;
  const lower = key.toLowerCase();
  return idx.sections.find(s => s.title.toLowerCase() === lower)
      ?? idx.sections.find(s => s.title.toLowerCase().includes(lower))
      ?? null;
}

function shortPreview(p: Poi): string {
  const parts: string[] = [];
  if (p.display_type) parts.push(p.display_type);
  if (p.interest_level_label && p.interest_level_label !== "Outstanding")
    parts.push(p.interest_level_label);
  if (p.description) {
    const sentenceEnd = p.description.match(/[.!?](\s|$)/);
    const snippet = sentenceEnd
      ? p.description.slice(0, sentenceEnd.index! + 1)
      : p.description.slice(0, 90);
    parts.push(snippet.trim());
  }
  return parts.join(" — ");
}
```

The full TypeScript port (`formatSection`, `formatPoi`, `findPoiByName`,
`filterPois`) mirrors the Python helpers in `scripts/index_tools.py`
1:1 — same input shapes, same output strings, same fallback rules.

---

## 8. Tool Execution

```typescript
type ToolResult = { text: string; cacheHit: boolean };

function executeTool(
  name:         string,
  args:         Record<string, unknown>,
  idx:          Index,
  sectionsText: string,
  cache:        Map<string, string>,
): ToolResult {
  if (name === "list_sections") {
    return { text: sectionsText, cacheHit: true };
  }
  if (name === "get_section") {
    const key = JSON.stringify(["get_section", args]);
    const hit = cache.get(key);
    if (hit) return { text: hit, cacheHit: true };
    const text = formatSection(idx, String(args.section_id ?? ""),
                                String(args.sort ?? "interest"),
                                Number(args.limit ?? 50));
    cache.set(key, text);
    return { text, cacheHit: false };
  }
  if (name === "get_poi") {
    const key = JSON.stringify(["get_poi", args.poi_id]);
    const hit = cache.get(key);
    if (hit) return { text: hit, cacheHit: true };
    const text = formatPoi(idx, String(args.poi_id ?? ""));
    cache.set(key, text);
    return { text, cacheHit: false };
  }
  if (name === "find_poi_by_name") {
    const text = formatFindPoiByName(idx,
                                     String(args.query ?? ""),
                                     Number(args.limit ?? 5));
    return { text, cacheHit: false };
  }
  if (name === "filter_pois") {
    const text = formatFilterPois(idx, args);
    return { text, cacheHit: false };
  }
  return { text: `[ERROR] Unknown tool: ${name}`, cacheHit: false };
}
```

The `cache` (`Map<string, string>`) is created per request. All sections
are pre-warmed at the start of each request (see §10), so every
`get_section` call is a cache hit — zero recomputation after startup.

---

## 9. Streaming Tool-Calling Loop

This is the core of the Worker. It mirrors the Python `run_turn` logic
with `stream=True`.

```typescript
const TOOL_DEFS = [
  {
    type: "function",
    function: {
      name:        "get_section",
      description: "List the POIs inside one section. Returns id + name + 1-line preview.",
      parameters:  {
        type: "object",
        properties: {
          section_id: { type: "string", description: "Section id or title." },
          sort:       { type: "string", enum: ["interest", "name", "zoom"] },
          limit:      { type: "integer" },
        },
        required: ["section_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name:        "get_poi",
      description: "Return the full record of one POI by id (no truncation).",
      parameters:  {
        type: "object",
        properties: { poi_id: { type: "string" } },
        required:   ["poi_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name:        "find_poi_by_name",
      description: "Diacritic-insensitive fuzzy lookup against POI names.",
      parameters:  {
        type: "object",
        properties: { query: { type: "string" }, limit: { type: "integer" } },
        required:   ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name:        "filter_pois",
      description: "Facet query (interest_level, type, tourist_type, section_id, indispensable).",
      parameters:  {
        type: "object",
        properties: {
          interest_level: { type: "integer" },
          type:           { type: "string" },
          tourist_type:   { type: "string" },
          section_id:     { type: "string" },
          indispensable:  { type: "boolean" },
          limit:          { type: "integer" },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name:        "list_sections",
      description: "Return the section catalogue (already in the system prompt).",
      parameters:  { type: "object", properties: {} },
    },
  },
];

async function* ragLoop(
  params: {
    question:     string;
    systemPrompt: string;
    sectionsText: string;
    index:        Index;
    history:      ChatMessage[];
    llmEndpoint:  string;
    llmApiKey:    string;
    cache:        Map<string, string>;
  }
): AsyncGenerator<RagEvent> {
  const { question, systemPrompt, sectionsText, index,
          history, llmEndpoint, llmApiKey, cache } = params;

  const messages: ChatMessage[] = [
    { role: "system",  content: systemPrompt },
    ...history,
    { role: "user",    content: question },
  ];

  const MAX_ROUNDS = 14;

  for (let round = 0; round < MAX_ROUNDS; round++) {
    const response = await fetch(`${llmEndpoint}/chat/completions`, {
      method:  "POST",
      headers: {
        "Content-Type":  "application/json",
        "Authorization": `Bearer ${llmApiKey}`,
      },
      body: JSON.stringify({
        model:       "gemma4:26b",
        messages,
        tools:       TOOL_DEFS,
        tool_choice: "auto",
        temperature: 0,
        stream:      true,
      }),
    });

    if (!response.ok || !response.body) {
      yield { type: "error", text: `LLM error: ${response.status}` };
      return;
    }

    // ── Parse the SSE stream ──────────────────────────────────────────
    let accContent    = "";
    let accToolCalls: AccToolCall[] = [];
    let streamingLive = false;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      for (const line of buffer.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") break;

        let chunk: StreamChunk;
        try { chunk = JSON.parse(data); } catch { continue; }

        const delta = chunk.choices?.[0]?.delta;
        if (!delta) continue;

        // Text content → relay immediately
        if (delta.content) {
          if (!streamingLive) streamingLive = true;
          accContent += delta.content;
          yield { type: "token", text: delta.content };
        }

        // Tool call deltas → accumulate silently
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index;
            while (accToolCalls.length <= idx)
              accToolCalls.push({ id: "", name: "", arguments: "" });
            if (tc.id)              accToolCalls[idx].id         = tc.id;
            if (tc.function?.name)  accToolCalls[idx].name       = tc.function.name;
            if (tc.function?.arguments)
              accToolCalls[idx].arguments += tc.function.arguments;
          }
        }
      }
      buffer = buffer.includes("\n") ? buffer.split("\n").pop()! : buffer;
    }

    // ── Append assistant message ──────────────────────────────────────
    const assistantMsg: ChatMessage = { role: "assistant", content: accContent };
    if (accToolCalls.length) {
      assistantMsg.tool_calls = accToolCalls.map(tc => ({
        id: tc.id, type: "function",
        function: { name: tc.name, arguments: tc.arguments },
      }));
    }
    messages.push(assistantMsg);

    if (!accToolCalls.length) return;  // final answer, generator done

    // ── Execute tool calls ────────────────────────────────────────────
    for (const tc of accToolCalls) {
      let args: Record<string, string> = {};
      try { args = JSON.parse(tc.arguments || "{}"); } catch {}

      // Emit status event for the client spinner
      if (tc.name === "get_section")
        yield { type: "status", text: `Loading section ${args.section_id ?? ""}…` };
      else if (tc.name === "get_poi")
        yield { type: "status", text: `Loading POI ${args.poi_id ?? ""}…` };
      else if (tc.name === "find_poi_by_name")
        yield { type: "status", text: `Searching '${args.query ?? ""}'…` };
      else if (tc.name === "filter_pois")
        yield { type: "status", text: `Filtering POIs…` };

      const { text, cacheHit } = executeTool(
        tc.name, args, index, sectionsText, cache
      );
      messages.push({ role: "tool", tool_call_id: tc.id, content: text });
    }
  }
}
```

---

## 10. Worker Entry Point

```typescript
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method !== "POST") return new Response("Method Not Allowed", { status: 405 });

    let body: { question: string; lang?: string; session_id?: string; enrich?: string[] };
    try { body = await request.json(); }
    catch { return new Response("Invalid JSON", { status: 400 }); }

    const { question, lang = "en", session_id, enrich = [] } = body;
    if (!question?.trim()) return new Response("question is required", { status: 400 });

    const cache = await caches.open("rag-v1");

    // Resolve session (language + destination locked)
    const session  = session_id
      ? await resolveSession(env, session_id)
      : null;
    const destination = session?.destination ?? body.destination ?? "ubeda";
    const lang        = session?.lang        ?? body.lang        ?? "en";
    const history: ChatMessage[] = session?.history ?? [];

    const indexKey = `destinations/${destination}/${destination}_${lang}.json`;

    // Load index from R2 (cached per PoP).  Falls back to English if the
    // requested language has not been built yet.
    let indexJson: string;
    try {
      indexJson = await loadAsset(env, cache, indexKey);
    } catch {
      indexJson = await loadAsset(env, cache,
        `destinations/${destination}/${destination}_en.json`);
    }
    const index = JSON.parse(indexJson) as Index;

    // Pre-warm session cache (one entry per section; pure JS, ~2 ms)
    const sessionCache = new Map<string, string>();
    for (const sec of index.sections) {
      sessionCache.set(
        JSON.stringify(["get_section", { section_id: sec.section_id, sort: "interest", limit: 50 }]),
        formatSection(index, sec.section_id, "interest", 50),
      );
    }

    // Build system prompt (same template as scripts/run_eval.py)
    const sectionsText  = formatSectionsOverview(index);
    const systemPrompt  = makeSystemPrompt(sectionsText, lang,
                                           index.meta.destination_display,
                                           index.destination_overview);

    // Optional: parallel context enrichment
    const enrichedContext = await enrichContext(enrich, env);
    const finalPrompt = enrichedContext
      ? systemPrompt + "\n\n--- LIVE CONTEXT ---\n" + enrichedContext
      : systemPrompt;

    // Persist updated history to Durable Object after response

    // Stream the response
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    const writeEvent = (event: RagEvent) =>
      writer.write(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));

    ctx.waitUntil((async () => {
      const start = Date.now();
      let toolCalls = 0;
      try {
        for await (const event of ragLoop({
          question, systemPrompt: finalPrompt, sectionsText,
          index, history,
          llmEndpoint: env.LLM_ENDPOINT, llmApiKey: env.LLM_API_KEY,
          cache: sessionCache,
        })) {
          await writeEvent(event);
          if (event.type === "status") toolCalls++;
        }
        await writeEvent({
          type: "done",
          latency_ms: Date.now() - start,
          tool_calls: toolCalls,
          cache_hits: sessionCache.size,   // all hits after pre-warm
        });
      } catch (err) {
        await writeEvent({ type: "error", text: String(err) });
      } finally {
        await writer.close();
      }
    })());

    return new Response(readable, {
      headers: {
        "Content-Type":  "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
      },
    });
  },
};
```

---

## 11. Context Enrichment (Optional)

Each enrichment module is a small async function that fetches from an
external API and returns a short text fragment. They run in parallel
before the LLM call.

```typescript
async function enrichContext(
  modules: string[], env: Env
): Promise<string | null> {
  if (!modules.length) return null;

  const tasks: Promise<string>[] = [];

  if (modules.includes("weather")) {
    tasks.push(fetchWeather(env.INVENTRIP_API_BASE_URL, env.INVENTRIP_API_KEY,
      env.DESTINATION).catch(() => ""));
  }
  if (modules.includes("events")) {
    tasks.push(fetchTodayEvents(env.INVENTRIP_API_BASE_URL, env.INVENTRIP_API_KEY,
      env.DESTINATION).catch(() => ""));
  }

  const results = await Promise.all(tasks);
  return results.filter(Boolean).join("\n") || null;
}

async function fetchWeather(base: string, key: string, dest: string): Promise<string> {
  const r = await fetch(`${base}/v100/weather-daily?tourist_destination=${dest}&api_key=${key}`);
  if (!r.ok) return "";
  const data = await r.json();
  const today = data?.[0];
  if (!today) return "";
  return `Current weather in ${dest}: ${today.description}, ${today.temp_max}°C max.`;
}
```

---

## 12. `wrangler.toml`

```toml
name       = "inventrip-rag"
main       = "src/index.ts"
compatibility_date = "2025-04-01"

[[r2_buckets]]
binding     = "RAG_BUCKET"
bucket_name = "inventrip-rag"

[[durable_objects.bindings]]
name  = "SESSIONS"
class = "RagSession"

[vars]
INVENTRIP_API_BASE_URL = "https://api.inventrip.com"
LLM_ENDPOINT           = "https://llm.inventrip.com/v1"
SUPPORTED_LANGS        = "en,es,fr,de,it,pt,ca,nl,pl,ru,zh,ja,ar,tr,uk,hr"
# Secrets: set via `wrangler secret put LLM_API_KEY` etc.
```

---

## 13. Error Handling

| Condition | Behaviour |
|---|---|
| R2 object missing | 500 + `error` event; retry after pipeline re-run |
| LLM endpoint unreachable | 502 + `error` event; client should retry |
| LLM response timeout (> 60 s) | 504 + `error` event |
| Invalid question (empty) | 400, no stream opened |
| Tool call JSON parse failure | skip arguments, call tool with `{}` |
| MAX_ROUNDS exceeded | return whatever partial answer was accumulated |
| `/v1/chat` on session with no destination | 400 `destination_required` — use `/v1/meta-chat` |
| Unknown destination slug in session | 500 `corpus_not_found` — trigger pipeline upload |
| `destination` field in `/v1/chat` body differs from session | silently ignored — session value always wins |

---

## 14. Performance Budget

All figures measured on the existing Apple Silicon / MLX stack.

| Phase | Time | Notes |
|---|---|---|
| R2 load (cold, first PoP hit) | ~10 ms | One index object, ~720 KB |
| R2 load (warm, cached PoP) | < 1 ms | Served from Workers Cache API |
| Index parse (`JSON.parse`) | ~5 ms | Done once per request |
| Session-cache pre-warm (18 sections) | < 2 ms | Pure JS dict traversal |
| Context enrichment (weather) | ~50 ms | One `fetch()`, runs in parallel |
| First LLM token (TTFT) | ~2–3 s | Depends on LLM load |
| Full answer (streaming) | 20–30 s | Wall time; CPU time ≈ 10 ms |
| SSE relay overhead | ~0 ms | TransformStream is zero-copy |

The Worker itself consumes < 5 ms of CPU time per request. Cloudflare
paid plans allow up to 30 s of CPU time, which is never approached.
Streaming responses have no wall-time limit.

---

## 15. Rollout Plan

**v1 — single destination, stateless (pilot):**
- Úbeda only; English corpus
- TypeScript navigation + streaming loop; no session state
- R2 path: `destinations/ubeda/`
- Deploy as `inventrip-rag.inventrip.workers.dev/ubeda`

**v2 — session + multi-language:**
- `POST /v1/session` creates a Durable Object locking `{lang, destination}`
- Language chosen once; response language never changes mid-conversation
- English corpus for all destinations; model responds in requested language
- Weather + events enrichment modules

**v3 — all 200 destinations + per-language indexes:**
- Cloud Run pipeline processes all 200 destinations × 16 languages
- R2 holds `{dest}_en.json`, `{dest}_es.json`, etc. per destination
- Worker falls back to English index if the requested language is missing
- Cross-destination meta-index (`meta/all_destinations.json`) enabled
- `POST /v1/meta-chat` for cross-destination queries

**Index update pipeline (Cloud Run cron, weekly):**
1. For each `(destination, language)` pair:
   `extract_pois.py` → `extract_destination_data.py` → `build_index.py`
   → upload `indexes/{dest}_{lang}.json` to R2.
2. Section summaries are deterministic (no LLM call) and rebuilt as part
   of step 1.
3. Rebuild `meta/all_destinations.json` once all destinations finish.
4. Workers Cache API entries are invalidated automatically when the R2
   ETag changes — no purge needed.

---

## 16. Open Questions

1. **Auth model** — JWT from the mobile app, or reuse the existing
   `api_key` query-param pattern? The session endpoint is the natural
   place to validate auth; subsequent `/v1/chat` calls carry `session_id`.
2. **Rate limiting** — apply at the Worker level (per `session_id` or
   per API key) reusing the existing Cloudflare rate-limit infrastructure.
3. **Model selection** — expose `model` hint in the session payload
   (`gemma4:e4b` for mobile/low latency, `gemma4:26b` for desktop/quality).
4. **Per-language corpora vs. model translation** — v1 uses English
   corpora for all languages; the model translates on-the-fly. v3 adds
   per-language corpora. The quality delta needs measurement before
   committing to the 16× storage and pipeline cost.
5. **Cross-destination answer quality** — the meta-index approach returns
   the top-N destinations and runs per-destination queries in parallel.
   The synthesis step (merging N answers into one) adds one LLM call.
   Maximum parallel queries per cross-destination request: configurable,
   default 5.

---

## 17. Multi-language Corpus Strategy

### v1: Single English corpus, model translates

The simpler approach. All destination corpora are in English.
The system prompt ends with the language rule (e.g. `Respond always
in Spanish`) from `_LANG_RULES`. The 26B model correctly handles
cross-lingual retrieval: a French question retrieves from the English
corpus and the answer is synthesised in French.

**Trade-off:** POI names are in English (`Santa Lucía Viewpoints`
instead of `Miradores de Santa Lucía`). The model's translation is
generally accurate but may lose local terminology or branding.

### v2: Per-language indexes

The local pipeline already supports per-language artifacts. Run the
full 3-step pipeline with `--lang {code}` for each desired language:

```bash
# Spanish index for Ubeda
scripts/extract_pois.py             --destination ubeda --lang es
scripts/extract_destination_data.py --destination ubeda --lang es
scripts/build_index.py              --destination ubeda --lang es
```

This produces `indexes/ubeda_es.json`, which uploads to
`destinations/ubeda/ubeda_es.json` in R2. The Worker tries the
language-specific index first and falls back to the English index if
the requested language is not available.

Storage: 200 destinations × 16 languages × 720 KB ≈ 2.3 GB (negligible
cost; data-pipeline compute is the real constraint).

**Recommendation:** Start with v1. Generate per-language indexes only
for the top 3 languages by traffic (likely es, fr, de), measuring
grounding quality improvement before expanding to all 16.

### Session language lock

Language is set in `POST /v1/session` and stored in the Durable Object.
It is never read from the question text — the client always declares
the language explicitly at session start. This prevents the model from
switching languages if a user types a word in a different language.

```typescript
interface SessionState {
  lang:        string;           // locked at session creation
  destination: string | null;    // null = cross-destination mode
  history:     ChatMessage[];    // appended after each turn
  created_at:  number;           // ms since epoch
}
```

Durable Object TTL: 2 hours of inactivity (configurable). After TTL,
the client must create a new session.

---

## 18. Cross-destination Query Architecture

A question like “what are the best restaurants in Spain?” cannot be
answered from a single destination corpus. The architecture uses a
**two-stage retrieval** approach.

### Stage 1: Meta-index lookup

`meta/all_destinations.json` is a compact JSON array where each entry
describes one destination in ~200 words:

```json
[
  {
    "slug":       "ubeda",
    "name":       "\u00dabeda",
    "country":    "ES",
    "region":     "Andalusia",
    "tourist_types": ["HERITAGE TOURISM", "FOOD TOURISM", "ARCHITECTURE TOURISM"],
    "networks":   ["WORLD HERITAGE CITIES"],
    "highlights": "UNESCO World Heritage city with outstanding Renaissance architecture.
                   Known for olive oil tourism and the Sacred Chapel of El Salvador.",
    "poi_count":  367
  },
  ...
]
```

Total size: 200 destinations × ~300 chars = **~60 KB** — fits comfortably
in the LLM context window.

### Stage 2: Parallel destination queries

The Worker feeds the meta-index to the LLM and asks it to identify the
top-N most relevant destinations for the query. The LLM returns a JSON
list of slugs. The Worker then runs `ragLoop` for each slug **in
parallel** (up to `MAX_PARALLEL = 5` concurrent requests), merges the
results, and asks the LLM for a final synthesised answer.

```typescript
async function* metaRagLoop(params: MetaParams): AsyncGenerator<RagEvent> {
  // 1. Identify relevant destinations from meta-index
  const metaJson   = await loadAsset(env, cache, "meta/all_destinations.json");
  const metaPrompt = buildMetaPrompt(params.question, metaJson, params.lang);
  const slugs      = await identifyDestinations(metaPrompt, params);  // LLM call

  yield { type: "status", text: `Searching ${slugs.length} destinations…` };

  // 2. Query each destination in parallel
  const perDestResults = await Promise.all(
    slugs.slice(0, MAX_PARALLEL).map(slug =>
      queryOneDestination(slug, params).catch(e => ({ slug, error: String(e) }))
    )
  );

  // 3. Synthesise a single answer from the per-destination results
  yield* synthesiseAnswer(params.question, perDestResults, params);
}
```

**Latency:** 2 LLM calls (identification + synthesis) + N parallel
destination queries. If each destination query takes ~25 s and they run
fully in parallel, wall time ≈ 25 s + overhead for the two synthesis
calls ≈ 55 s total. Streaming the identification and per-destination
statuses keeps the UX responsive throughout.

**Scope filtering:** The `scope` field in the request (`"spain"`,
`"andalusia"`) pre-filters the meta-index before sending it to the LLM,
reducing the identification step’s token cost and improving accuracy.

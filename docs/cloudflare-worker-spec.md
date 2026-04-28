# Cloudflare Worker — Inventrip RAG: Technical Specification

**Version:** 1.1  
**Status:** Draft  
**Scope:** 200 tourist destinations × 16 languages  
**Relates to:** `scripts/chat_demo.py`, `scripts/run_eval.py`, `docs/cloudflare-worker-spec.md`

---

## 1. Goals

Move the PageIndex RAG orchestration layer out of GKE into a Cloudflare
Worker so that:

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
    │         │   ├── index/structure.json
    │         │   ├── index/destination.json
    │         │   └── corpus/guide_en.md   ← (+ guide_es.md, etc. in v2)
    │         └── baeza/ ... granada/ ...
    ├── Durable Objects: RagSession    ← session: language + history
    │
    ├── fetch → llm.inventrip.com/v1  ← LLM inference (Ollama / Gemma 4)
    ├── fetch → api.inventrip.com     ← live data enrichment
    └── fetch → external APIs         ← weather, events, etc.
```

The Worker implements the complete two-level navigation loop in
TypeScript. No Python runtime is involved in the request path.

**Single-destination query** (most requests): load the destination
corpus from R2, run the two-level navigation loop, stream the answer.

**Cross-destination query** (e.g. “best restaurants in Spain”): load
the meta-index from R2, identify the top-N relevant destinations, run
one navigation loop per destination in parallel, merge and stream a
synthesised answer.

---

## 3. R2 Bucket Structure

One bucket (**`inventrip-rag`**) holds all 200 destinations. The path
scheme is `destinations/{slug}/` so adding a destination never requires
a Worker re-deploy.

```
inventrip-rag/
├── meta/
│   └── all_destinations.json      # ← see §17: cross-destination index
└── destinations/
    ├── ubeda/
    │   ├── index/
    │   │   ├── structure.json       # PageIndex tree + section summaries
    │   │   └── destination.json     # trips, tourist types, interest levels
    │   └── corpus/
    │       ├── guide_en.md          # English corpus (v1)
    │       ├── guide_es.md          # Spanish corpus (v2, optional)
    │       └── guide_fr.md          # French corpus (v2, optional)
    ├── baeza/
    │   └── ...
    └── ... (198 more)
```

**Storage estimate (v1, English-only):**
200 destinations × ~290 KB = **58 MB** total. R2 at $0.015/GB/month ≈
$0.001/month. Negligible.

**Storage estimate (v2, per-language corpora):**
200 destinations × 16 languages × ~290 KB = **928 MB**. Still cheap
($0.014/month) but the data pipeline becomes 16× heavier.

Objects are **immutable during a session** — they change only when the
Inventrip data pipeline runs (weekly Cloud Run job). The Worker caches
all objects in the [Workers Cache API][wcache] after the first fetch;
subsequent requests on the same PoP serve them from memory.

[wcache]: https://developers.cloudflare.com/workers/runtime-apis/cache/

### Uploading corpus files

```bash
# Upload a single destination (run from pageindex-demo project root):
DEST=ubeda
wrangler r2 object put inventrip-rag/destinations/$DEST/index/structure.json \
  --file results/ubeda_guide_structure.json
wrangler r2 object put inventrip-rag/destinations/$DEST/index/destination.json \
  --file data/ubeda_destination.json
wrangler r2 object put inventrip-rag/destinations/$DEST/corpus/guide_en.md \
  --file data/ubeda_guide.md

# Upload the meta-index (after all destinations are processed):
wrangler r2 object put inventrip-rag/meta/all_destinations.json \
  --file data/all_destinations.json
```

The Cloud Run data pipeline iterates all 200 destinations and calls
this upload sequence for each, then rebuilds `all_destinations.json`.

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

These are direct ports of the Python functions in `run_eval.py`.

```typescript
interface SectionNode {
  node_id: string;
  title:   string;
  line_num: number;
  summary?: string;
  nodes?:  SectionNode[];
}

function getSections(structure: { structure: SectionNode[] }): SectionNode[] {
  return structure.structure[0]?.nodes ?? [];
}

function buildSectionsText(structure: { structure: SectionNode[]; line_count: number }): string {
  const sections  = getSections(structure);
  const lineCount = structure.line_count;
  const lines: string[] = [
    `Document: ubeda_guide  (${lineCount} lines)`,
    "",
    `SECTIONS (${sections.length} total):`,
  ];
  sections.forEach((sec, i) => {
    const endLine = sections[i + 1]?.line_num
      ? sections[i + 1].line_num - 1
      : lineCount;
    lines.push(`  [${sec.node_id}] ${sec.title}`);
    lines.push(`      lines ${sec.line_num}–${endLine}  (${sec.nodes?.length ?? 0} POIs)`);
    if (sec.summary) lines.push(`      Summary: ${sec.summary}`);
  });
  return lines.join("\n");
}

function buildPoiListText(sectionTitle: string, structure: { structure: SectionNode[] }): string {
  const sections = getSections(structure);
  const match    = sections.find(
    s => s.title.toLowerCase() === sectionTitle.toLowerCase()
        || s.title.toLowerCase().includes(sectionTitle.toLowerCase())
  );
  if (!match) {
    const titles = sections.map(s => s.title);
    return `[ERROR] Section '${sectionTitle}' not found. Available: ${JSON.stringify(titles)}`;
  }
  const pois = match.nodes ?? [];
  return [
    `POIs in '${match.title}' (${pois.length} entries):`,
    ...pois.map(p => `  [${p.node_id}] ${p.title}  (line ${p.line_num})`),
  ].join("\n");
}

function getLines(mdLines: string[], spec: string): string {
  const [startStr, endStr] = spec.includes("-")
    ? spec.split("-")
    : [spec, spec];
  const start = Math.max(1, parseInt(startStr, 10));
  const end   = Math.min(mdLines.length, parseInt(endStr, 10));
  return mdLines.slice(start - 1, end).join("\n");
}
```

---

## 8. Tool Execution

```typescript
type ToolResult = { text: string; cacheHit: boolean };

function executeTool(
  name:         string,
  args:         Record<string, string>,
  sectionsText: string,
  structure:    { structure: SectionNode[] },
  mdLines:      string[],
  poiCache:     Map<string, string>,
): ToolResult {
  if (name === "get_sections") {
    return { text: sectionsText, cacheHit: true };
  }
  if (name === "get_poi_list") {
    const key   = (args.section_title ?? "").toLowerCase();
    const cached = poiCache.get(key);
    if (cached) return { text: cached, cacheHit: true };
    const text = buildPoiListText(args.section_title ?? "", structure);
    poiCache.set(key, text);
    return { text, cacheHit: false };
  }
  if (name === "get_page_content") {
    const text = getLines(mdLines, args.lines ?? "1-20");
    return { text: text || "[WARNING] No content found for that range.", cacheHit: false };
  }
  return { text: `[ERROR] Unknown tool: ${name}`, cacheHit: false };
}
```

The `poiCache` (`Map<string, string>`) is created per request. Because
all 20 sections are pre-warmed at the start of each request (see §10),
every `get_poi_list` call is a cache hit — zero R2 reads after startup.

---

## 9. Streaming Tool-Calling Loop

This is the core of the Worker. It mirrors the Python `run_turn` logic
with `stream=True`.

```typescript
const TOOL_DEFS = [
  {
    type: "function",
    function: {
      name:        "get_poi_list",
      description: "Returns all POI names and line numbers inside a section.",
      parameters:  {
        type: "object",
        properties: { section_title: { type: "string" } },
        required:   ["section_title"],
      },
    },
  },
  {
    type: "function",
    function: {
      name:        "get_page_content",
      description: "Returns the raw Markdown text for a line range (e.g. '9-28').",
      parameters:  {
        type: "object",
        properties: { lines: { type: "string" } },
        required:   ["lines"],
      },
    },
  },
];

async function* ragLoop(
  params: {
    question:     string;
    systemPrompt: string;
    sectionsText: string;
    structure:    { structure: SectionNode[] };
    mdLines:      string[];
    history:      ChatMessage[];
    llmEndpoint:  string;
    llmApiKey:    string;
    poiCache:     Map<string, string>;
  }
): AsyncGenerator<RagEvent> {
  const { question, systemPrompt, sectionsText, structure,
          mdLines, history, llmEndpoint, llmApiKey, poiCache } = params;

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
      if (tc.name === "get_poi_list")
        yield { type: "status", text: `Searching ${args.section_title ?? "section"}…` };
      else if (tc.name === "get_page_content")
        yield { type: "status", text: `Reading guide lines ${args.lines ?? "…"}` };

      const { text, cacheHit } = executeTool(
        tc.name, args, sectionsText, structure, mdLines, poiCache
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

    const prefix = `destinations/${destination}`;
    const lang_suffix = lang !== "en" ? `_${lang}` : "";

    // Load corpus (cached after first request per PoP)
    const [structureJson, destinationJson, mdText] = await Promise.all([
      loadAsset(env, cache, `${prefix}/index/structure.json`),
      loadAsset(env, cache, `${prefix}/index/destination.json`),
      // v1: always English corpus; v2: prefer language-specific, fall back to EN
      loadAsset(env, cache, `${prefix}/corpus/guide${lang_suffix}.md`)
        .catch(() => loadAsset(env, cache, `${prefix}/corpus/guide_en.md`)),
    ]);

    const structure   = JSON.parse(structureJson);
    const mdLines     = mdText.split("\n");

    // Pre-warm POI cache (all 20 sections, ~2 ms, pure JS)
    const poiCache = new Map<string, string>();
    for (const sec of getSections(structure)) {
      poiCache.set(sec.title.toLowerCase(), buildPoiListText(sec.title, structure));
    }

    // Build system prompt
    const sectionsText  = buildSectionsText(structure);
    const systemPrompt  = makeSystemPrompt(sectionsText, lang);

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
      let toolCalls = 0, cacheHits = 0;
      try {
        for await (const event of ragLoop({
          question, systemPrompt: finalPrompt, sectionsText,
          structure, mdLines, history,
          llmEndpoint: env.LLM_ENDPOINT, llmApiKey: env.LLM_API_KEY,
          poiCache,
        })) {
          await writeEvent(event);
          if (event.type === "token") { /* count */ }
          if (event.type === "status") toolCalls++;
        }
        await writeEvent({
          type: "done",
          latency_ms: Date.now() - start,
          tool_calls: toolCalls,
          cache_hits: poiCache.size,   // all hits after pre-warm
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

---

## 14. Performance Budget

All figures measured on the existing Apple Silicon / MLX stack.

| Phase | Time | Notes |
|---|---|---|
| R2 load (cold, first PoP hit) | ~10 ms | Three objects, ~290 KB total |
| R2 load (warm, cached PoP) | < 1 ms | Served from Workers Cache API |
| POI cache pre-warm (20 sections) | < 2 ms | Pure JS dict traversal |
| Context enrichment (weather) | ~50 ms | One `fetch()`, runs in parallel |
| First LLM token (TTFT) | ~2–3 s | Depends on LLM load |
| Full answer (streaming) | 20–30 s | Wall time; CPU time ≈ 5 ms |
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

**v3 — all 200 destinations + per-language corpora:**
- Cloud Run pipeline processes all 200 destinations × 16 languages
- R2 holds `guide_en.md`, `guide_es.md`, etc. per destination
- Worker falls back to English corpus if language-specific corpus is missing
- Cross-destination meta-index (`meta/all_destinations.json`) enabled
- `POST /v1/meta-chat` for cross-destination queries

**Corpus update pipeline (Cloud Run cron, weekly):**
1. For each destination × language: fetch POIs → generate Markdown → build index → upload to R2
2. Regenerate section summaries in English only (model handles translation)
3. Rebuild `meta/all_destinations.json`
4. Purge Workers Cache API entries for updated destinations (R2 ETag change handles this automatically)

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
instead of `Miradores de Santa Lucía`). The model’s translation is
generally accurate but may lose local terminology or branding.

### v2: Per-language corpora (optional)

For destinations with high traffic in a specific language (e.g.
Spanish for all Spanish destinations), generate a language-specific
Markdown from the Inventrip API with `language={lang}`. Store as
`corpus/guide_es.md`. The Worker tries the language-specific corpus
first and falls back to English if missing.

Storage: 200 destinations × 16 languages × 290 KB = 928 MB (negligible
cost; data pipeline cost is the real constraint).

**Recommendation:** Start with v1. Add per-language corpora only for
the top 3 languages by traffic (likely es, fr, de), measuring grounding
improvement before expanding to all 16.

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

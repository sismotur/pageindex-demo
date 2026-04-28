# Cloudflare Worker — Úbeda Tourism RAG: Technical Specification

**Version:** 1.0  
**Status:** Draft  
**Relates to:** `scripts/chat_demo.py`, `scripts/run_eval.py`

---

## 1. Goals

Move the PageIndex RAG orchestration layer out of GKE into a Cloudflare
Worker so that:

- The service runs at Cloudflare's edge (300+ PoPs) with no additional
  infrastructure to manage.
- Multiple data sources (Inventrip API, weather, events, live POI status)
  can be fetched in parallel and injected into the LLM context.
- The full response is streamed token-by-token to the client from the
  moment the LLM produces the first word.
- The GKE cluster continues to serve the existing Inventrip API
  unchanged.

---

## 2. Architecture

```
Client (browser / mobile)
    │  POST /v1/chat  (JSON body)
    │  ← SSE stream   (text/event-stream)
    ▼
Cloudflare Worker  (this spec)
    ├── R2: ubeda-rag/                 ← static corpus at edge
    │     ├── index/ubeda_guide_structure.json
    │     ├── index/ubeda_destination.json
    │     └── corpus/ubeda_guide.md
    ├── KV: rag-cache                  ← POI list cache (optional v2)
    ├── Durable Objects: RagSession    ← conversation history (v2)
    │
    ├── fetch → llm.inventrip.com/v1  ← LLM inference (Ollama / Gemma 4)
    ├── fetch → api.inventrip.com     ← live data enrichment
    └── fetch → external APIs         ← weather, events, etc.
```

The Worker implements the complete two-level navigation loop in
TypeScript. No Python runtime is involved in the request path.

---

## 3. R2 Bucket Structure

Bucket name: **`ubeda-rag`** (one bucket per destination; extend with
`seville-rag`, `granada-rag`, etc. as needed).

```
ubeda-rag/
├── index/
│   ├── ubeda_guide_structure.json   # PageIndex tree + section summaries
│   └── ubeda_destination.json       # Trips, tourist types, interest levels
└── corpus/
    └── ubeda_guide.md               # Full Markdown document (line-addressed)
```

Objects are **immutable during a session** — they change only when the
Inventrip data pipeline runs (weekly Cloud Run job). The Worker caches
all three objects in the [Workers Cache API][wcache] after the first
fetch; subsequent requests on the same PoP serve them from memory.

[wcache]: https://developers.cloudflare.com/workers/runtime-apis/cache/

### Uploading corpus files

```bash
# From the pageindex-demo project root after running the data pipeline:
wrangler r2 object put ubeda-rag/index/ubeda_guide_structure.json \
  --file results/ubeda_guide_structure.json
wrangler r2 object put ubeda-rag/index/ubeda_destination.json \
  --file data/ubeda_destination.json
wrangler r2 object put ubeda-rag/corpus/ubeda_guide.md \
  --file data/ubeda_guide.md
```

---

## 4. Worker Environment (`Env`)

```typescript
interface Env {
  // R2 bucket holding the corpus
  RAG_BUCKET: R2Bucket;

  // KV namespace for POI list cache (optional v2 — see §8)
  RAG_KV: KVNamespace;

  // Durable Objects for conversation history (optional v2 — see §9)
  SESSIONS: DurableObjectNamespace;

  // LLM inference endpoint (Ollama OpenAI-compatible)
  LLM_ENDPOINT: string;      // e.g. https://llm.inventrip.com/v1
  LLM_API_KEY: string;       // bearer token for llm.inventrip.com

  // Inventrip API (for live data enrichment)
  INVENTRIP_API_BASE_URL: string;   // https://api.inventrip.com
  INVENTRIP_API_KEY: string;

  // Destination slug (for multi-destination deployments)
  DESTINATION: string;       // e.g. "ubeda"
}
```

`wrangler.toml` binds `RAG_BUCKET` to the R2 bucket and reads secrets
from the Workers Secrets store (never in `wrangler.toml` plaintext).

---

## 5. Request / Response Format

### Request

```
POST /v1/chat
Content-Type: application/json

{
  "question":   "¿Dónde puedo aparcar en Úbeda?",
  "lang":       "es",            // en | es | fr | de  (default: en)
  "session_id": "uuid-optional", // for conversation history (v2)
  "enrich":     ["weather"]      // optional enrichment modules
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

    // Load corpus (cached after first request per PoP)
    const [structureJson, destinationJson, mdText] = await Promise.all([
      loadAsset(env, cache, "index/ubeda_guide_structure.json"),
      loadAsset(env, cache, "index/ubeda_destination.json"),
      loadAsset(env, cache, "corpus/ubeda_guide.md"),
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

    // Load conversation history (v2: Durable Objects; v1: stateless)
    const history: ChatMessage[] = [];

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
name       = "ubeda-rag"
main       = "src/index.ts"
compatibility_date = "2025-04-01"

[[r2_buckets]]
binding    = "RAG_BUCKET"
bucket_name = "ubeda-rag"

[[kv_namespaces]]
binding    = "RAG_KV"
id         = "<kv-namespace-id>"          # create with wrangler kv:namespace create

[vars]
DESTINATION            = "ubeda"
INVENTRIP_API_BASE_URL = "https://api.inventrip.com"
LLM_ENDPOINT           = "https://llm.inventrip.com/v1"
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

**v1 (stateless):**
- TypeScript port of navigation helpers + tool-calling loop
- R2 corpus upload from pipeline output
- Streaming SSE response
- Stateless (no conversation history)
- Deploy as `ubeda-rag.inventrip.workers.dev`

**v2 (stateful + enriched):**
- Durable Objects for conversation history (session continuity)
- Weather + events enrichment modules
- Multi-language routing from `Accept-Language` header
- Cache invalidation webhook from Cloud Run data pipeline

**v3 (multi-destination):**
- Parameterise bucket prefix by destination slug
- Same Worker handles Baeza, Cazorla, etc. with different R2 paths
- Dynamic section summaries regenerated on data change via R2 event trigger

---

## 16. Open Questions

1. **Auth model** — should the Worker validate a JWT from the mobile app,
   or reuse the existing `api_key` query-param pattern from the
   Inventrip API?
2. **Rate limiting** — reuse the existing Cloudflare rate-limit rule on
   `inventrip.com`, or add a dedicated rule for `/v1/chat`?
3. **Model selection** — can the client hint a preferred model
   (`gemma4:e4b` for low latency, `gemma4:26b` for quality)?
4. **Corpus update cadence** — weekly pipeline or event-driven on Inventrip
   data change? The latter requires a Cloud Run → R2 write → Worker cache
   invalidation chain.

# llm-gateway-service

Local HTTP service wrapping `local_first_common`'s `resolve_provider()` /
`FallbackProvider` / `acomplete()` behind one authenticated endpoint, plus a
concurrent multi-model comparison mode that didn't exist anywhere in the
toolkit before this.

Python, not TypeScript -- unlike `http-retriever-service`, all the real
logic (provider dispatch, fallback, tiering) already lives here, tested.
There's no ecosystem asymmetry favoring another language the way Mozilla
Readability favored TS for the retriever; building this in TS would mean
reimplementing and then maintaining a second copy of that dispatch logic.
See BrainSync tool doc 49.

## Quickstart

```bash
export LLM_GATEWAY_API_KEY=$(security find-generic-password -a "$USER" -s "llm-gateway-api-key" -w)
uv run llm-gateway-service   # http://127.0.0.1:8788
```

```bash
curl -X POST http://127.0.0.1:8788/complete \
  -H "Authorization: Bearer $LLM_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider": "anthropic", "system": "...", "user": "..."}'
```

Real callers should go through `resolve_provider()` in `local_first_common`
(set `LLM_GATEWAY_URL`) rather than hitting this HTTP API directly -- see
"Client wiring" below.

## API

All routes except `/health` require `Authorization: Bearer <LLM_GATEWAY_API_KEY>`.

**`POST /complete`** -- `{provider, model?, system, user, images?, trace?}` -> `{provider, model, text, trace_id, duration_ms, input_tokens, output_tokens}`. 502 on provider failure.

- `images`: list of base64-encoded image strings (no data-URI prefix), forwarded to the real provider's vision API. Omit for text-only calls.
- `trace`: when `true`, persists the real prompt/response to a separate trace database (see "Tracing" below). Defaults to `false`.

**`POST /compare`** -- `{system, user, calls: [{provider, model?}, ...], trace?}` -> `{trace_id, results: [...]}`, one result per call, run concurrently via `asyncio.gather`. A failing call becomes `{error: "..."}` in its own result slot rather than failing the whole batch -- comparison is the point, one bad model shouldn't hide the others' answers. Text-only; no `images` field (the one known vision use case is a single call, not a comparison).

**`GET /health`** -- `{"status": "ok"}`, no auth.

## Client wiring

`local_first_common.cli.resolve_provider()` delegates here automatically when `LLM_GATEWAY_URL` is set (via `GatewayProvider` in `local_first_common.providers.gateway`) -- every existing caller of `resolve_provider()` benefits with zero code changes, the same shape as `http-retriever-service`'s `fetch_article_metadata`/`fetch_article_body` wiring. `response_model` (structured/validated JSON output) works transparently: schema-awareness stays entirely client-side (`BaseProvider._get_example_json()`/`_parse_json_response()`), this service never sees a schema.

**Known limitation before 2026-09-19, now resolved:** vision calls used to raise on the client side. They now work end-to-end -- verified with a real Anthropic vision call against an actual image.

**Audited (2026-09-19) which other tools are safe to point at this gateway** -- see BrainSync tool doc 49's rollout-audit section for the full list. Currently only `content-discovery-agent` has `LLM_GATEWAY_URL` set.

For ad-hoc comparisons outside any tool's own code, use `model-compare` (BrainSync tool #51) -- a small CLI built specifically to give `/compare` a real caller: `model-compare "prompt" -c anthropic:claude-haiku-4-5-20251001 -c ollama:llama3.2:3b`.

## Tracing

Metadata-only by default, unchanged for every existing caller: every call gets a `trace_id` (returned to the caller, used as `source_location` in the logged `processing_log` row), so every model in one `/compare` request can be found together later. That log entry never includes prompt/response text -- matching `processing_log`'s existing schema exactly.

On top of that, a caller can opt in per call (`"trace": true`) to persist the *actual* prompt and response -- to a separate database (`~/sync/local-first/llm_gateway_traces.duckdb`, via `traces.py`), never mixed into `processing_log`. This is deliberate and per-call, not automatic: verified that an untraced call leaves the trace database's row count unchanged, and a traced call's real content lands there correctly.

## Auth

Shared static bearer token (`LLM_GATEWAY_API_KEY`), checked with a constant-time comparison. One operator, one secret -- no per-client keys, sessions, or OAuth; see BrainSync tool doc 49's security discussion for why that's proportionate here. Binds to `127.0.0.1` only. The key lives in macOS Keychain (`security find-generic-password -a "$USER" -s "llm-gateway-api-key" -w`), sourced via `.zshenv` -- never written into the LaunchAgent plist in plaintext.

## Testing

```bash
uv run pytest -q --cov=src   # 22 tests, 98% coverage
```

Verified against real providers throughout development, not just mocks: a real Anthropic call through `/complete`; a real concurrent `/compare` against Anthropic + local Ollama (parallelism confirmed by wall-clock time matching the slower call, not the sum); a real vision call against an actual image, correctly described; a traced call's real content landing in the separate trace database while an untraced call left it untouched; and `resolve_provider()`'s delegation verified against content-discovery-agent's actual production binary, including a real bug (the model name defaulting to the provider alias) found and fixed by that real test, not caught by mocks alone.

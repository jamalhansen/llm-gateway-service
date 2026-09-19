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
export LLM_GATEWAY_API_KEY=$(openssl rand -hex 32)
uv run llm-gateway-service   # http://127.0.0.1:8788
```

```bash
curl -X POST http://127.0.0.1:8788/complete \
  -H "Authorization: Bearer $LLM_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider": "anthropic", "system": "...", "user": "..."}'
```

## API

All routes except `/health` require `Authorization: Bearer <LLM_GATEWAY_API_KEY>`.

**`POST /complete`** -- `{provider, model?, system, user}` -> `{provider, model, text, trace_id, duration_ms, input_tokens, output_tokens}`. 502 on provider failure.

**`POST /compare`** -- `{system, user, calls: [{provider, model?}, ...]}` -> `{trace_id, results: [...]}`, one result per call, run concurrently via `asyncio.gather`. A failing call becomes `{error: "..."}` in its own result slot rather than failing the whole batch -- comparison is the point, one bad model shouldn't hide the others' answers.

**`GET /health`** -- `{"status": "ok"}`, no auth.

## Tracing decision (2026-09-19, made without sign-off -- flag for review)

Every call gets a `trace_id` (returned to the caller, and used as `source_location` in the logged `processing_log` row), so every model in one `/compare` request can be found together later. That log entry is metadata only (model, tokens, duration) -- matching `processing_log`'s existing schema exactly, which stores no prompt/response text. **The actual completion text is returned to the caller and not persisted anywhere by this service.** This was the conservative default given an earlier security discussion (today's logging holds no conversation content); revisit if trace-id correlation isn't enough and full-content replay is actually wanted -- that would be a real, deliberate change to what gets centralized, not a small tweak.

## Auth

Shared static bearer token (`LLM_GATEWAY_API_KEY`), checked with a constant-time comparison. One operator, one secret -- no per-client keys, sessions, or OAuth; see BrainSync tool doc 49's security discussion for why that's proportionate here. Binds to `127.0.0.1` only.

## Testing

```bash
uv run pytest -q --cov=src   # 14 tests, 97% coverage
```

Verified against real providers during development, not just mocks: a real
Anthropic call through `/complete`, and a real concurrent `/compare` against
Anthropic + local Ollama (1.9s wall time for both, confirming genuine
parallelism rather than sequential fallback-style calling).

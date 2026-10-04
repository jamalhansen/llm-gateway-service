"""FastAPI app: POST /complete, POST /compare, GET /health.

Auth via a shared bearer token (auth.py) on every route except /health.
Binds to 127.0.0.1 only -- see BrainSync tool doc 49.

Tracing (2026-09-19, revised from the original overnight decision): every
call still gets a trace_id and a metadata-only processing_log entry by
default -- no prompt/response text, matching the existing schema exactly.
On top of that, a caller can now opt in per call (`"trace": true`) to have
the real prompt/response persisted to traces.py's separate database. The
default behavior for every existing caller is unchanged; tracing real
content is deliberate and per-call, never automatic.
"""
import os
import uuid

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from local_first_common.tracking import log_run, register_tool

from .auth import verify_token
from .core import CompletionResult, compare, complete_one
from .schemas import (
    CompareRequest,
    CompareResponse,
    CompareResultItem,
    CompleteRequest,
    CompleteResponse,
)
from .traces import write_trace

TOOL_NAME = "llm-gateway-service"
_TOOL = register_tool(TOOL_NAME)

app = FastAPI(title="llm-gateway-service")


def _log(result: CompletionResult, trace_id: str) -> None:
    # The single database write for this call (2026-09-20) -- callers no
    # longer keep their own duplicate processing_log row, so this is it.
    # Attribute to the real caller when it sent one (via GatewayProvider's
    # tool_name=) instead of always this service -- see gateway.py's
    # docstring. A direct/anonymous caller (no tool_name in the request)
    # still logs as TOOL_NAME, same as before. source_location prefers the
    # caller's own context over the bare trace_id, which identifies nothing
    # on its own; item_count defaults to 1 (one completion) when the caller
    # didn't say it covered more than one logical item.
    #
    # Duration comes from the result, not from timing this function: the
    # completion already happened in complete_one() by the time we get here,
    # so a timed_run() around this block measured ~0 s. Every gateway-routed
    # row from 2026-09-21 to 2026-10-04 has duration_seconds = 0 for that
    # reason, which is why the 16 s/call claude-code slowdown was invisible
    # in processing_log. Failures are logged as failures for the same reason:
    # complete_one() returns the error instead of raising it.
    log_run(
        result.tool_name or TOOL_NAME,
        result.model,
        provider=result.provider,
        source_location=result.source_location or trace_id,
        item_count=result.item_count or 1,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        duration_seconds=result.duration_ms / 1000,
        success=result.error is None,
        error_message=result.error,
        via_gateway=True,
    )


@app.post("/complete", response_model=CompleteResponse, dependencies=[Depends(verify_token)])
async def complete(req: CompleteRequest) -> CompleteResponse:
    trace_id = str(uuid.uuid4())
    result = await complete_one(
        req.provider,
        req.model,
        req.system,
        req.user,
        images=req.images,
        tool_name=req.tool_name,
        source_location=req.source_location,
        item_count=req.item_count,
    )
    _log(result, trace_id)
    if req.trace:
        write_trace(trace_id, "complete", result.provider, result.model, req.system, req.user, result.text, result.error)
    if result.error:
        raise HTTPException(status_code=502, detail=result.error)
    return CompleteResponse(
        provider=result.provider,
        model=result.model,
        text=result.text or "",
        trace_id=trace_id,
        duration_ms=result.duration_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@app.post("/compare", response_model=CompareResponse, dependencies=[Depends(verify_token)])
async def compare_endpoint(req: CompareRequest) -> CompareResponse:
    trace_id = str(uuid.uuid4())
    calls = [(c.provider, c.model) for c in req.calls]
    results = await compare(
        req.system, req.user, calls, tool_name=req.tool_name, source_location=req.source_location, item_count=req.item_count
    )
    for result in results:
        _log(result, trace_id)
        if req.trace:
            write_trace(
                trace_id, "compare", result.provider, result.model, req.system, req.user, result.text, result.error
            )
    return CompareResponse(
        trace_id=trace_id,
        results=[
            CompareResultItem(
                provider=r.provider,
                model=r.model,
                text=r.text,
                error=r.error,
                duration_ms=r.duration_ms,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
            )
            for r in results
        ],
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    port = int(os.environ.get("LLM_GATEWAY_PORT", "8788"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()

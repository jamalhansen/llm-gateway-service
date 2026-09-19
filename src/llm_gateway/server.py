"""FastAPI app: POST /complete, POST /compare, GET /health.

Auth via a shared bearer token (auth.py) on every route except /health.
Binds to 127.0.0.1 only -- see BrainSync tool doc 49.

Tracing decision (made without Jamal's sign-off, since this was built while
he was offline -- flag for review): every call gets a trace_id (returned to
the caller) and is logged via timed_run using that trace_id as
source_location, so related calls -- e.g. every model in one /compare
request -- can be found together later in processing_log. That log entry is
metadata only (model, tokens, duration), matching the existing schema's
"no prompt/response text" property exactly. The actual completion text is
returned to the caller and NOT persisted anywhere by this service. This
was the conservative default given the earlier security discussion (logging
today stores no conversation content); revisit if trace_id-based
correlation alone isn't enough and full-content replay is actually wanted.
"""
import os
import uuid

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from local_first_common.tracking import register_tool, timed_run

from .auth import verify_token
from .core import CompletionResult, compare, complete_one
from .schemas import (
    CompareRequest,
    CompareResponse,
    CompareResultItem,
    CompleteRequest,
    CompleteResponse,
)

TOOL_NAME = "llm-gateway-service"
_TOOL = register_tool(TOOL_NAME)

app = FastAPI(title="llm-gateway-service")


def _log(result: CompletionResult, trace_id: str) -> None:
    with timed_run(TOOL_NAME, result.model, source_location=trace_id) as run:
        run.item_count = 1
        run.input_tokens = result.input_tokens
        run.output_tokens = result.output_tokens


@app.post("/complete", response_model=CompleteResponse, dependencies=[Depends(verify_token)])
async def complete(req: CompleteRequest) -> CompleteResponse:
    trace_id = str(uuid.uuid4())
    result = await complete_one(req.provider, req.model, req.system, req.user)
    _log(result, trace_id)
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
    results = await compare(req.system, req.user, calls)
    for result in results:
        _log(result, trace_id)
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

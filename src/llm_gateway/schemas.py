from pydantic import BaseModel


class CompleteRequest(BaseModel):
    provider: str = "ollama"
    model: str | None = None
    system: str
    user: str
    # Base64-encoded image data, no data-URI prefix -- matches BaseProvider's
    # own `images` convention exactly (see AnthropicProvider._build_messages).
    images: list[str] | None = None
    # Opt-in only: when true, the real prompt/response is persisted to
    # traces.py's separate database. Default logging (processing_log) stays
    # metadata-only either way. See traces.py's module docstring.
    trace: bool = False
    # Sent by GatewayProvider when resolve_provider() was given a tool_name --
    # lets this server's own processing_log row be attributed to the real
    # caller instead of always "llm-gateway-service". Absent for a direct/
    # anonymous caller (e.g. a raw curl test); that still logs as this service.
    tool_name: str | None = None
    # The database write for an LLM call happens exactly once, here in the
    # gateway (2026-09-20) -- callers no longer keep their own duplicate
    # processing_log row, so anything they want persisted about the call has
    # to travel in the request instead. source_location is the caller's own
    # per-call context (a file path, a URL, "example:あ") -- without it this
    # row's source_location falls back to a bare trace_id, which identifies
    # nothing on its own. item_count is how many logical items one completion
    # covered (e.g. notes tagged in a single batched prompt); defaults to 1.
    source_location: str | None = None
    item_count: int | None = None


class CompleteResponse(BaseModel):
    provider: str
    model: str
    text: str
    trace_id: str
    duration_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class CompareCall(BaseModel):
    provider: str
    model: str | None = None


class CompareRequest(BaseModel):
    system: str
    user: str
    calls: list[CompareCall]
    trace: bool = False
    tool_name: str | None = None
    source_location: str | None = None
    item_count: int | None = None


class CompareResultItem(BaseModel):
    provider: str
    model: str
    text: str | None
    error: str | None
    duration_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class CompareResponse(BaseModel):
    trace_id: str
    results: list[CompareResultItem]

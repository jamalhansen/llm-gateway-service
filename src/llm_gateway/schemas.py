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

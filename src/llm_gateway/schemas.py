from pydantic import BaseModel


class CompleteRequest(BaseModel):
    provider: str = "ollama"
    model: str | None = None
    system: str
    user: str


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

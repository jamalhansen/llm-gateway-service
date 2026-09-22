"""Pure(ish) dispatch logic for the LLM gateway: single completions and
concurrent multi-model comparison. HTTP/auth/logging live in server.py so
this stays testable without spinning up FastAPI.

Reuses resolve_provider()/FallbackProvider/acomplete() as-is -- this module
is a thin wrapper, not a reimplementation. The reason this is Python rather
than TypeScript (unlike the http-retriever-service): the actual provider
dispatch, fallback, and tiering logic already lives here, tested, and there
is no equivalent asymmetry favoring another language the way Mozilla
Readability favored TS for the retriever.
"""
from __future__ import annotations

import time
from asyncio import gather
from dataclasses import dataclass

from local_first_common.cli import resolve_provider
from local_first_common.providers import PROVIDERS


@dataclass
class CompletionResult:
    provider: str
    model: str
    text: str | None
    error: str | None
    duration_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_name: str | None = None
    source_location: str | None = None
    item_count: int | None = None


async def complete_one(
    provider_name: str,
    model: str | None,
    system: str,
    user: str,
    images: list[str] | None = None,
    tool_name: str | None = None,
    source_location: str | None = None,
    item_count: int | None = None,
) -> CompletionResult:
    """Run one completion. Provider failures land in the result, not an
    exception -- required for /compare, where one bad model must not sink
    the others in the same batch; /complete reuses this and raises at the
    HTTP layer instead.
    """
    started = time.monotonic()
    try:
        # use_gateway=False: this process IS the gateway -- it inherits
        # LLM_GATEWAY_URL from the same shell env as every other tool, and
        # without this, resolve_provider() would route back through the
        # gateway branch and build a provider that calls this same server,
        # recursively, over HTTP. See local_first_common.cli.resolve_provider's
        # docstring (2026-09-20 incident).
        provider = resolve_provider(PROVIDERS, provider_name, model, use_gateway=False)
        text = await provider.acomplete(system, user, images=images)
        return CompletionResult(
            # provider.provider_name, not the raw provider_name request
            # string: PROVIDERS aliases "local" to OllamaProvider (backward
            # compat) alongside "ollama" -- logging the request string
            # instead of the resolved provider's canonical name split
            # identical Ollama traffic into two provider buckets on the
            # dashboard. Jamal 2026-09-22: "poor stats hygiene."
            provider=provider.provider_name,
            model=provider.model,
            text=text if isinstance(text, str) else str(text),
            error=None,
            duration_ms=int((time.monotonic() - started) * 1000),
            input_tokens=getattr(provider, "input_tokens", None),
            output_tokens=getattr(provider, "output_tokens", None),
            tool_name=tool_name,
            source_location=source_location,
            item_count=item_count,
        )
    except Exception as e:  # noqa: BLE001 - a provider failure is a result, not a crash
        # Still the raw request string here, not provider.provider_name --
        # resolve_provider() itself may have failed (unknown provider name),
        # leaving `provider` unbound, so this is the best info available.
        return CompletionResult(
            provider=provider_name,
            model=model or "",
            text=None,
            error=str(e),
            duration_ms=int((time.monotonic() - started) * 1000),
            tool_name=tool_name,
            source_location=source_location,
            item_count=item_count,
        )


async def compare(
    system: str,
    user: str,
    calls: list[tuple[str, str | None]],
    tool_name: str | None = None,
    source_location: str | None = None,
    item_count: int | None = None,
) -> list[CompletionResult]:
    """Run every requested (provider, model) call concurrently. Comparison is
    the point, so every call runs regardless of whether earlier ones failed.
    """
    return list(
        await gather(
            *(
                complete_one(
                    p, m, system, user, tool_name=tool_name, source_location=source_location, item_count=item_count
                )
                for p, m in calls
            )
        )
    )

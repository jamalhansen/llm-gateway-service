from unittest.mock import patch

import pytest
from local_first_common.testing import MockProvider

from llm_gateway.core import compare, complete_one


class TestCompleteOne:
    @pytest.mark.asyncio
    async def test_returns_text_and_tokens_on_success(self):
        mock = MockProvider(response="a real answer")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            result = await complete_one("anthropic", "claude-haiku", "system", "user")
        assert result.text == "a real answer"
        assert result.error is None
        # provider.provider_name (the resolved provider's canonical name),
        # not the raw request string -- MockProvider's is "mock" regardless
        # of what was requested.
        assert result.provider == "mock"

    @pytest.mark.asyncio
    async def test_provider_alias_normalizes_to_the_resolved_providers_own_name(self):
        """Jamal 2026-09-22: 'local' and 'ollama' both showed up as separate
        providers on the dashboard -- PROVIDERS aliases "local" to
        OllamaProvider (backward compat), but the old code logged the raw
        request string instead of the resolved provider's canonical name, so
        identical Ollama traffic split into two buckets depending on which
        alias a caller happened to use."""

        class FakeOllamaProvider:
            provider_name = "ollama"  # what OllamaProvider actually reports, regardless of which alias resolved to it
            model = "phi4-mini"
            input_tokens = None
            output_tokens = None

            async def acomplete(self, system, user, images=None):
                return "ok"

        with patch("llm_gateway.core.resolve_provider", return_value=FakeOllamaProvider()):
            result = await complete_one("local", "phi4-mini", "system", "user")
        assert result.provider == "ollama"

    @pytest.mark.asyncio
    async def test_tool_name_carried_through_on_success_and_failure(self):
        mock = MockProvider(response="ok")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            result = await complete_one("anthropic", None, "system", "user", tool_name="my-tool")
        assert result.tool_name == "my-tool"

        broken = MockProvider(raise_error="boom")
        with patch("llm_gateway.core.resolve_provider", return_value=broken):
            failed = await complete_one("anthropic", None, "system", "user", tool_name="my-tool")
        assert failed.tool_name == "my-tool"

    @pytest.mark.asyncio
    async def test_source_location_and_item_count_carried_through_on_success_and_failure(self):
        mock = MockProvider(response="ok")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            result = await complete_one("anthropic", None, "system", "user", source_location="example:あ", item_count=5)
        assert result.source_location == "example:あ"
        assert result.item_count == 5

        broken = MockProvider(raise_error="boom")
        with patch("llm_gateway.core.resolve_provider", return_value=broken):
            failed = await complete_one("anthropic", None, "system", "user", source_location="example:あ", item_count=5)
        assert failed.source_location == "example:あ"
        assert failed.item_count == 5

    @pytest.mark.asyncio
    async def test_provider_failure_becomes_a_result_not_an_exception(self):
        mock = MockProvider(raise_error="boom")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            result = await complete_one("anthropic", None, "system", "user")
        assert result.text is None
        assert result.error is not None and "boom" in result.error

    @pytest.mark.asyncio
    async def test_images_are_forwarded_to_the_provider(self):
        captured = {}

        class FakeProvider:
            provider_name = "anthropic"
            model = "vision-model"
            input_tokens = None
            output_tokens = None

            async def acomplete(self, system, user, images=None):
                captured["images"] = images
                return "I see a cat"

        with patch("llm_gateway.core.resolve_provider", return_value=FakeProvider()):
            result = await complete_one("anthropic", None, "system", "user", images=["b64data"])
        assert result.text == "I see a cat"
        assert captured["images"] == ["b64data"]


class TestCompleteOneUsesGatewayFalse:
    @pytest.mark.asyncio
    async def test_resolve_provider_called_with_use_gateway_false(self):
        """Regression for the 2026-09-20 self-recursion incident: this
        process IS the gateway, and inherits LLM_GATEWAY_URL from the same
        shell env as every other tool -- without use_gateway=False its own
        resolve_provider() call would route back through itself."""
        mock = MockProvider(response="ok")
        with patch("llm_gateway.core.resolve_provider", return_value=mock) as fake:
            await complete_one("ollama", "phi4-mini", "system", "user")
        assert fake.call_args.kwargs.get("use_gateway") is False


class TestCompare:
    @pytest.mark.asyncio
    async def test_runs_every_call_even_if_one_fails(self):
        # model set explicitly on each mock: the success path reports
        # provider.model (what the resolved instance actually used), and
        # fake_resolve below ignores the requested model, so without this
        # both mocks would report "mock" and be indistinguishable by model.
        good = MockProvider(response="good answer", model="m1")
        bad = MockProvider(raise_error="bad model", model="m2")

        def fake_resolve(providers, provider_name, model, **kwargs):
            return good if provider_name == "good-provider" else bad

        with patch("llm_gateway.core.resolve_provider", side_effect=fake_resolve):
            results = await compare("system", "user", [("good-provider", "m1"), ("bad-provider", "m2")])

        assert len(results) == 2
        # Keyed by model, not provider: both mocks resolve to the same
        # provider.provider_name ("mock"), so model is what distinguishes
        # the two calls here.
        by_model = {r.model: r for r in results}
        assert by_model["m1"].text == "good answer"
        assert by_model["m2"].error == "bad model"

    @pytest.mark.asyncio
    async def test_empty_calls_returns_empty_list(self):
        results = await compare("system", "user", [])
        assert results == []

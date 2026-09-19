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
        assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_provider_failure_becomes_a_result_not_an_exception(self):
        mock = MockProvider(raise_error="boom")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            result = await complete_one("anthropic", None, "system", "user")
        assert result.text is None
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_images_are_forwarded_to_the_provider(self):
        captured = {}

        class FakeProvider:
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


class TestCompare:
    @pytest.mark.asyncio
    async def test_runs_every_call_even_if_one_fails(self):
        good = MockProvider(response="good answer")
        bad = MockProvider(raise_error="bad model")

        def fake_resolve(providers, provider_name, model):
            return good if provider_name == "good-provider" else bad

        with patch("llm_gateway.core.resolve_provider", side_effect=fake_resolve):
            results = await compare(
                "system", "user", [("good-provider", "m1"), ("bad-provider", "m2")]
            )

        assert len(results) == 2
        by_provider = {r.provider: r for r in results}
        assert by_provider["good-provider"].text == "good answer"
        assert by_provider["bad-provider"].error == "bad model"

    @pytest.mark.asyncio
    async def test_empty_calls_returns_empty_list(self):
        results = await compare("system", "user", [])
        assert results == []

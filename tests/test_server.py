from unittest.mock import patch

from fastapi.testclient import TestClient
from local_first_common.testing import MockProvider

from llm_gateway.server import app

AUTH = {"Authorization": "Bearer test-token"}


def _client(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-token")
    return TestClient(app)


class TestCompleteEndpoint:
    def test_requires_auth(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.post("/complete", json={"system": "s", "user": "u"})
        assert response.status_code == 401

    def test_success_returns_text_and_trace_id(self, monkeypatch):
        client = _client(monkeypatch)
        mock = MockProvider(response="a real answer")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            response = client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u"},
                headers=AUTH,
            )
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "a real answer"
        assert "trace_id" in body

    def test_images_field_is_forwarded_to_complete_one(self, monkeypatch):
        client = _client(monkeypatch)
        captured = {}

        class FakeProvider:
            model = "vision-model"
            input_tokens = None
            output_tokens = None

            async def acomplete(self, system, user, images=None):
                captured["images"] = images
                return "I see a cat"

        with patch("llm_gateway.core.resolve_provider", return_value=FakeProvider()):
            response = client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u", "images": ["b64data"]},
                headers=AUTH,
            )
        assert response.status_code == 200
        assert response.json()["text"] == "I see a cat"
        assert captured["images"] == ["b64data"]

    def test_provider_error_returns_502(self, monkeypatch):
        client = _client(monkeypatch)
        mock = MockProvider(raise_error="provider down")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            response = client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u"},
                headers=AUTH,
            )
        assert response.status_code == 502


class TestCompareEndpoint:
    def test_requires_auth(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.post("/compare", json={"system": "s", "user": "u", "calls": []})
        assert response.status_code == 401

    def test_runs_multiple_models_and_returns_all_results(self, monkeypatch):
        client = _client(monkeypatch)
        good = MockProvider(response="good")
        bad = MockProvider(raise_error="bad")

        def fake_resolve(providers, provider_name, model):
            return good if provider_name == "good-provider" else bad

        with patch("llm_gateway.core.resolve_provider", side_effect=fake_resolve):
            response = client.post(
                "/compare",
                json={
                    "system": "s",
                    "user": "u",
                    "calls": [{"provider": "good-provider"}, {"provider": "bad-provider"}],
                },
                headers=AUTH,
            )
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 2
        by_provider = {r["provider"]: r for r in body["results"]}
        assert by_provider["good-provider"]["text"] == "good"
        assert by_provider["bad-provider"]["error"] == "bad"
        assert "trace_id" in body


class TestHealth:
    def test_health_needs_no_auth(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

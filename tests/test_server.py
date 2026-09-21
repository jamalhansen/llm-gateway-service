from unittest.mock import patch

import duckdb
from fastapi.testclient import TestClient
from local_first_common.testing import MockProvider
from local_first_common.tracking import get_tracking_db_path

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

    def test_processing_log_attributed_to_the_real_caller_when_tool_name_sent(self, monkeypatch):
        """Jamal: "LLM gateway service shouldn't really be the tool that is
        calling LLM in the report, right? It should show the tool it's being
        called by." When the request carries tool_name (sent by
        GatewayProvider whenever resolve_provider() was given one), this
        service's own processing_log row should be attributed to that real
        caller, not always "llm-gateway-service"."""
        client = _client(monkeypatch)
        mock = MockProvider(response="ok")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u", "tool_name": "japanese-tutor"},
                headers=AUTH,
            )
        conn = duckdb.connect(str(get_tracking_db_path()))
        row = conn.execute("SELECT tool_name FROM processing_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        assert row[0] == "japanese-tutor"

    def test_processing_log_attributed_to_this_service_when_no_tool_name_sent(self, monkeypatch):
        """A direct/anonymous caller (no GatewayProvider, e.g. a raw curl
        test) has no tool_name to send -- unchanged behavior."""
        client = _client(monkeypatch)
        mock = MockProvider(response="ok")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u"},
                headers=AUTH,
            )
        conn = duckdb.connect(str(get_tracking_db_path()))
        row = conn.execute("SELECT tool_name FROM processing_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        assert row[0] == "llm-gateway-service"

    def test_processing_log_marked_via_gateway(self, monkeypatch):
        """Jamal: pass the tool name through, but also indicate that the
        call went through the gateway. Every row this service logs for its
        own request handling is via_gateway=True, distinguishing it from
        the calling tool's own timed_run() row for the same logical call --
        both get created for a gateway-routed call, and without this a
        dashboard summing processing_log by tool_name double-counts every
        one of them (confirmed live 2026-09-20: obsidian-vault-auto-tagger's
        real call count was inflated ~2x)."""
        client = _client(monkeypatch)
        mock = MockProvider(response="ok")
        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u", "tool_name": "japanese-tutor"},
                headers=AUTH,
            )
        conn = duckdb.connect(str(get_tracking_db_path()))
        row = conn.execute("SELECT via_gateway FROM processing_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        assert row[0] is True

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

    def test_trace_false_never_calls_write_trace(self, monkeypatch):
        client = _client(monkeypatch)
        mock = MockProvider(response="a real answer")
        with (
            patch("llm_gateway.core.resolve_provider", return_value=mock),
            patch("llm_gateway.server.write_trace") as mock_write,
        ):
            client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u"},
                headers=AUTH,
            )
        mock_write.assert_not_called()

    def test_trace_true_persists_the_real_content(self, monkeypatch):
        client = _client(monkeypatch)
        mock = MockProvider(response="a real answer")
        with (
            patch("llm_gateway.core.resolve_provider", return_value=mock),
            patch("llm_gateway.server.write_trace") as mock_write,
        ):
            response = client.post(
                "/complete",
                json={"provider": "anthropic", "system": "s", "user": "u", "trace": True},
                headers=AUTH,
            )
        trace_id = response.json()["trace_id"]
        mock_write.assert_called_once_with(trace_id, "complete", "anthropic", mock.model, "s", "u", "a real answer", None)


class TestCompareEndpoint:
    def test_requires_auth(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.post("/compare", json={"system": "s", "user": "u", "calls": []})
        assert response.status_code == 401

    def test_runs_multiple_models_and_returns_all_results(self, monkeypatch):
        client = _client(monkeypatch)
        good = MockProvider(response="good")
        bad = MockProvider(raise_error="bad")

        def fake_resolve(providers, provider_name, model, **kwargs):
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

    def test_every_call_in_the_batch_attributed_to_the_real_caller(self, monkeypatch):
        client = _client(monkeypatch)
        mock = MockProvider(response="ok")

        with patch("llm_gateway.core.resolve_provider", return_value=mock):
            client.post(
                "/compare",
                json={
                    "system": "s",
                    "user": "u",
                    "calls": [{"provider": "a"}, {"provider": "b"}],
                    "tool_name": "model-comparison-harness",
                },
                headers=AUTH,
            )
        conn = duckdb.connect(str(get_tracking_db_path()))
        rows = conn.execute(
            "SELECT tool_name FROM processing_log ORDER BY id DESC LIMIT 2"
        ).fetchall()
        conn.close()
        assert all(r[0] == "model-comparison-harness" for r in rows)

    def test_trace_true_persists_every_call_in_the_batch(self, monkeypatch):
        client = _client(monkeypatch)
        good = MockProvider(response="good")

        with (
            patch("llm_gateway.core.resolve_provider", return_value=good),
            patch("llm_gateway.server.write_trace") as mock_write,
        ):
            client.post(
                "/compare",
                json={
                    "system": "s",
                    "user": "u",
                    "calls": [{"provider": "p1"}, {"provider": "p2"}],
                    "trace": True,
                },
                headers=AUTH,
            )
        assert mock_write.call_count == 2


class TestHealth:
    def test_health_needs_no_auth(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

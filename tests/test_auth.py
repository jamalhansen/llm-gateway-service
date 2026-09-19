import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from llm_gateway.auth import verify_token


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "correct-token")
    test_app = FastAPI()

    @test_app.get("/protected", dependencies=[Depends(verify_token)])
    def protected():
        return {"ok": True}

    return test_app


class TestVerifyToken:
    def test_missing_header_rejected(self, app):
        client = TestClient(app)
        response = client.get("/protected")
        assert response.status_code == 401

    def test_wrong_token_rejected(self, app):
        client = TestClient(app)
        response = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_correct_token_allowed(self, app):
        client = TestClient(app)
        response = client.get("/protected", headers={"Authorization": "Bearer correct-token"})
        assert response.status_code == 200

    def test_no_server_key_configured_is_a_server_error(self, monkeypatch):
        monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
        test_app = FastAPI()

        @test_app.get("/protected", dependencies=[Depends(verify_token)])
        def protected():
            return {"ok": True}

        client = TestClient(test_app)
        response = client.get("/protected", headers={"Authorization": "Bearer anything"})
        assert response.status_code == 500

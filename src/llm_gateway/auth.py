"""Shared-secret bearer-token auth.

One operator, one secret -- see BrainSync tool doc 49's security discussion
for why this is the proportionate design for a single-user local service,
not per-client keys, sessions, or OAuth. Unlike http-retriever-service
(no secrets behind it, localhost-only, auth deferred), this service holds
real provider API keys, so auth is required from the start rather than
added later.
"""

import hmac
import os

from fastapi import Header, HTTPException

API_KEY_ENV_VAR = "LLM_GATEWAY_API_KEY"


def verify_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get(API_KEY_ENV_VAR)
    if not expected:
        raise HTTPException(status_code=500, detail=f"{API_KEY_ENV_VAR} is not set on the server")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    provided = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid token")

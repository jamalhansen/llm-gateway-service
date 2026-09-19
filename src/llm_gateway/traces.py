"""Opt-in full-content tracing, deliberately separate from processing_log.

processing_log.duckdb (via local_first_common.tracking) is metadata-only by
design across the whole ecosystem -- no prompt/response text, ever. That's
the right default and this doesn't change it. But "trace and compare LLM
call results" (the original ask) sometimes needs the actual content, not
just a trace_id correlating metadata rows -- so a caller can opt in per
call (`"trace": true`) and get the real request/response persisted here
instead, in its own file, never mixed into the shared metadata table.
"""
from pathlib import Path

import duckdb

TRACE_DB_PATH = Path("~/sync/local-first/llm_gateway_traces.duckdb").expanduser()

_CREATE_SEQUENCE = "CREATE SEQUENCE IF NOT EXISTS traces_id_seq START 1;"
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    id           BIGINT DEFAULT nextval('traces_id_seq') PRIMARY KEY,
    trace_id     VARCHAR NOT NULL,
    kind         VARCHAR NOT NULL,  -- 'complete' or 'compare'
    provider     VARCHAR,
    model        VARCHAR,
    system       VARCHAR,
    user_message VARCHAR,
    response     VARCHAR,
    error        VARCHAR,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
_INSERT = """
INSERT INTO traces (trace_id, kind, provider, model, system, user_message, response, error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""


def write_trace(
    trace_id: str,
    kind: str,
    provider: str,
    model: str,
    system: str,
    user_message: str,
    response: str | None,
    error: str | None = None,
) -> None:
    TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(TRACE_DB_PATH))
    try:
        conn.execute(_CREATE_SEQUENCE)
        conn.execute(_CREATE_TABLE)
        conn.execute(_INSERT, [trace_id, kind, provider, model, system, user_message, response, error])
    finally:
        conn.close()

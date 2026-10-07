import duckdb
import pytest

from llm_gateway import traces


@pytest.fixture(autouse=True)
def isolate_trace_db(tmp_path, monkeypatch):
    monkeypatch.setattr(traces, "TRACE_DB_PATH", tmp_path / "traces.duckdb")


def test_write_trace_creates_a_readable_row():
    traces.write_trace("trace-1", "complete", "anthropic", "claude-haiku", "sys", "hello", "hi there")

    conn = duckdb.connect(str(traces.TRACE_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT trace_id, kind, provider, model, system, user_message, response, error FROM traces"
    ).fetchone()
    conn.close()

    assert row == ("trace-1", "complete", "anthropic", "claude-haiku", "sys", "hello", "hi there", None)


def test_write_trace_records_errors_too():
    traces.write_trace("trace-2", "complete", "ollama", "phi4-mini", "sys", "hello", None, "model not found")

    conn = duckdb.connect(str(traces.TRACE_DB_PATH), read_only=True)
    row = conn.execute("SELECT response, error FROM traces WHERE trace_id = 'trace-2'").fetchone()
    conn.close()

    assert row == (None, "model not found")


def test_multiple_writes_accumulate():
    traces.write_trace("a", "complete", "p", "m", "s", "u", "r1")
    traces.write_trace("b", "compare", "p", "m", "s", "u", "r2")

    conn = duckdb.connect(str(traces.TRACE_DB_PATH), read_only=True)
    count = (conn.execute("SELECT COUNT(*) FROM traces").fetchone() or (0,))[0]
    conn.close()

    assert count == 2

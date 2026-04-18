"""Tests for the FastAPI server — endpoints that don't hit external LLMs.

The /api/query endpoint requires a real retrieval+LLM pipeline, so it's
covered by smoke tests elsewhere. Here we focus on the endpoints that gate
collection state, health, and metrics — the "control plane".
"""

import pytest
from fastapi.testclient import TestClient

from rag.config import get_settings
from rag.server import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate settings cache per test so project_root takes effect.
    get_settings.cache_clear()
    monkeypatch.delenv("RAG_COLLECTION", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    app = create_app(str(tmp_path))
    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "active_collection" in body
    assert "model" in body
    assert "tracing" in body
    assert body["tracing"]["enabled"] is False  # no LANGFUSE_* env vars in tests
    assert body["tracing"]["host"] is None


def test_health_endpoint_reports_tracing_when_enabled(client, monkeypatch):
    from rag import tracing
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    tracing.reset_for_tests()

    r = client.get("/api/health")
    body = r.json()
    assert body["tracing"]["enabled"] is True
    assert body["tracing"]["host"] == "http://localhost:3000"


def test_list_collections_empty(client):
    r = client.get("/api/collections")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == "default"
    assert body["collections"] == []


def test_create_collection(client):
    r = client.post("/api/collections", json={
        "name": "papers",
        "display_name": "Research Papers",
        "description": "PDFs",
    })
    assert r.status_code == 200
    assert r.json()["name"] == "papers"

    r2 = client.get("/api/collections")
    names = [c["name"] for c in r2.json()["collections"]]
    assert "papers" in names


def test_create_rejects_invalid_name(client):
    r = client.post("/api/collections", json={"name": "bad name"})
    assert r.status_code == 400


def test_create_rejects_duplicate(client):
    client.post("/api/collections", json={"name": "papers"})
    r = client.post("/api/collections", json={"name": "papers"})
    assert r.status_code == 409


def test_activate_collection(client):
    client.post("/api/collections", json={"name": "papers"})
    r = client.post("/api/collections/papers/activate")
    assert r.status_code == 200
    assert r.json()["active"] == "papers"


def test_activate_unknown_collection(client):
    r = client.post("/api/collections/ghost/activate")
    assert r.status_code == 404


def test_delete_collection(client):
    client.post("/api/collections", json={"name": "papers"})
    r = client.delete("/api/collections/papers")
    assert r.status_code == 200
    assert r.json()["deleted"] == "papers"


def test_delete_default_is_rejected(client):
    # `default` exists logically even without files (legacy detection falls
    # through to "no legacy => doesn't exist"), so calling delete on it
    # should fail with a helpful error either way.
    r = client.delete("/api/collections/default")
    assert r.status_code == 400


def test_query_without_index_returns_409(client):
    # Create a collection but don't index — query should be rejected cleanly.
    client.post("/api/collections", json={"name": "papers"})
    client.post("/api/collections/papers/activate")

    r = client.post("/api/query", json={"question": "hello"})
    assert r.status_code == 409
    assert "index" in r.json()["detail"].lower()


def test_query_unknown_collection_returns_404(client):
    r = client.post("/api/query", json={"question": "hi", "collection": "ghost"})
    assert r.status_code == 404


def test_stats_endpoint(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["errors"] == 0
    assert body["recent"] == []


def test_pipeline_invalidate(client):
    r = client.post("/api/pipeline/invalidate")
    assert r.status_code == 200
    assert r.json()["invalidated"] == "all"


# ----------------------------------------------------------------------
# Feedback endpoint
# ----------------------------------------------------------------------


def test_feedback_accepts_thumbs_up(client):
    r = client.post("/api/feedback", json={
        "rating": 1,
        "request_id": "abc",
        "question": "what is GRACE?",
        "answer": "GRACE is...",
    })
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_feedback_accepts_thumbs_down_with_comment(client):
    r = client.post("/api/feedback", json={
        "rating": -1,
        "comment": "missed the citation",
        "question": "what is GRACE?",
    })
    assert r.status_code == 200


def test_feedback_rejects_invalid_rating(client):
    r = client.post("/api/feedback", json={"rating": 0, "question": "q"})
    assert r.status_code == 400


def test_feedback_round_trip_appears_in_stats(client):
    client.post("/api/feedback", json={"rating": 1, "question": "ok"})
    client.post("/api/feedback", json={"rating": -1, "question": "bad1"})
    client.post("/api/feedback", json={"rating": -1, "question": "bad2"})

    r = client.get("/api/stats")
    body = r.json()
    assert body["feedback"]["positive"] == 1
    assert body["feedback"]["negative"] == 2

"""Tests for MetricsStore — schema, recording, and summary aggregates."""

import sqlite3

from rag.metrics import MetricsStore


def test_schema_is_created_on_first_use(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.ensure_schema()

    with sqlite3.connect(str(tmp_path / "m.sqlite3")) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "queries" in tables


def test_record_then_summary_totals(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    for i in range(5):
        store.record_query(
            collection="papers",
            question=f"q{i}",
            latency_ms=100 + i * 10,
            answer_len=200,
            token_count=50,
            doc_count=3,
        )

    summary = store.summary()
    assert summary["total"] == 5
    assert summary["errors"] == 0
    assert 100 <= summary["avg_latency_ms"] <= 200
    assert summary["p50_latency_ms"] > 0
    assert summary["p95_latency_ms"] > 0


def test_summary_counts_errors(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.record_query("papers", "ok q", latency_ms=120)
    store.record_query("papers", "fail q", latency_ms=50, error="timeout")

    summary = store.summary()
    assert summary["total"] == 2
    assert summary["errors"] == 1


def test_summary_per_collection_breakdown(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.record_query("papers", "q1", latency_ms=100)
    store.record_query("papers", "q2", latency_ms=200)
    store.record_query("law", "q3", latency_ms=150)

    summary = store.summary()
    per = {row["collection"]: row for row in summary["per_collection"]}
    assert per["papers"]["count"] == 2
    assert per["law"]["count"] == 1


def test_summary_filters_by_collection(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.record_query("papers", "q1", latency_ms=100)
    store.record_query("law", "q2", latency_ms=200)

    summary = store.summary(collection="law")
    assert summary["total"] == 1
    assert all(r["collection"] == "law" for r in summary["recent"])


def test_long_question_is_truncated(tmp_path):
    """Safety: an abusive client sending a huge question shouldn't blow up storage."""
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.record_query("papers", "x" * 10000, latency_ms=10)
    summary = store.summary()
    assert len(summary["recent"][0]["question"]) <= 2000

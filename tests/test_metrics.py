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


# ----------------------------------------------------------------------
# Feedback
# ----------------------------------------------------------------------


def test_feedback_schema_is_created(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.ensure_schema()
    with sqlite3.connect(str(tmp_path / "m.sqlite3")) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "feedback" in tables


def test_record_feedback_persists_row(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.record_feedback(
        rating=1,
        request_id="abc",
        comment="great answer",
        collection="papers",
        question="what is GRACE?",
        answer="GRACE is...",
    )
    with sqlite3.connect(str(tmp_path / "m.sqlite3")) as conn:
        rows = conn.execute("SELECT * FROM feedback").fetchall()
    assert len(rows) == 1


def test_record_feedback_rejects_invalid_rating(tmp_path):
    import pytest
    store = MetricsStore(tmp_path / "m.sqlite3")
    with pytest.raises(ValueError):
        store.record_feedback(rating=0, question="q", collection="c")
    with pytest.raises(ValueError):
        store.record_feedback(rating=2, question="q", collection="c")


def test_summary_includes_feedback_counts(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.record_feedback(rating=1, question="ok", collection="papers")
    store.record_feedback(rating=-1, question="bad1", collection="papers")
    store.record_feedback(rating=-1, question="bad2", collection="papers")

    summary = store.summary()
    assert summary["feedback"]["positive"] == 1
    assert summary["feedback"]["negative"] == 2


def test_summary_feedback_filters_by_collection(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    store.record_feedback(rating=-1, question="papers q", collection="papers")
    store.record_feedback(rating=-1, question="law q", collection="law")

    summary = store.summary(collection="law")
    assert summary["feedback"]["negative"] == 1


def test_list_negative_feedback_returns_distinct_questions(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    # Two downvotes on the same question, one positive, one different downvote
    store.record_feedback(rating=-1, question="q1", collection="papers", comment="a")
    store.record_feedback(rating=-1, question="q1", collection="papers", comment="b")
    store.record_feedback(rating=1, question="q3", collection="papers")
    store.record_feedback(rating=-1, question="q2", collection="papers")

    rows = store.list_negative_feedback(collection="papers")
    questions = [r["question"] for r in rows]
    assert "q1" in questions
    assert "q2" in questions
    assert "q3" not in questions
    assert len(rows) == 2  # q1 deduped


def test_list_negative_feedback_respects_limit(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    for i in range(10):
        store.record_feedback(rating=-1, question=f"q{i}", collection="papers")
    rows = store.list_negative_feedback(collection="papers", limit=3)
    assert len(rows) == 3

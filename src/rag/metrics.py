"""SQLite-backed query metrics.

Tracks one row per answered query: collection, timestamp, latency, token
count, doc count, error. Used by `rag stats`, the /api/stats endpoint, and
the web dashboard tab.

SQLite is deliberate: zero-config, one file, process-safe with the built-in
lock. For the scale of a single-tenant RAG (hundreds to thousands of queries
per day) it's far more than enough.
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from time import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    collection TEXT NOT NULL,
    question TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    answer_len INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    doc_count INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts DESC);
CREATE INDEX IF NOT EXISTS idx_queries_collection ON queries(collection);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT,
    rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
    comment TEXT,
    collection TEXT,
    question TEXT,
    answer TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback(ts DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_collection ON feedback(collection);
"""


class MetricsStore:
    """Thread-safe SQLite metrics log. One file per installation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._schema_ready = True

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(
            str(self.path), timeout=5.0, isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    def record_query(
        self,
        collection: str,
        question: str,
        latency_ms: int,
        answer_len: int = 0,
        token_count: int = 0,
        doc_count: int = 0,
        error: str | None = None,
    ) -> None:
        self.ensure_schema()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO queries "
                "(ts, collection, question, latency_ms, answer_len, token_count, doc_count, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time(),
                    collection,
                    question[:2000],
                    int(latency_ms),
                    int(answer_len),
                    int(token_count),
                    int(doc_count),
                    error,
                ),
            )

    def record_feedback(
        self,
        rating: int,
        request_id: str | None = None,
        comment: str | None = None,
        collection: str | None = None,
        question: str | None = None,
        answer: str | None = None,
    ) -> None:
        """Persist a thumbs up/down + optional comment for a previous answer.

        rating must be +1 or -1. Other fields are denormalized so the row is
        self-contained: we don't need to join against `queries` to know what
        the user was reacting to.
        """
        if rating not in (-1, 1):
            raise ValueError(f"rating must be -1 or +1, got {rating!r}")
        self.ensure_schema()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback "
                "(ts, request_id, rating, comment, collection, question, answer) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time(),
                    request_id,
                    int(rating),
                    (comment[:4000] if comment else None),
                    collection,
                    (question[:2000] if question else None),
                    (answer[:8000] if answer else None),
                ),
            )

    def list_negative_feedback(
        self,
        collection: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent thumbs-down rows for evaluating regressions.

        Deduplicates by question text — multiple downvotes on the same query
        only surface once, with the most recent comment.
        """
        self.ensure_schema()
        params: tuple = (-1,)
        where = "WHERE rating = ?"
        if collection:
            where += " AND collection = ?"
            params = (-1, collection)
        sql = (
            f"SELECT ts, request_id, rating, comment, collection, question, answer "  # noqa: S608
            f"FROM feedback {where} ORDER BY ts DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (*params, limit * 5)).fetchall()
        seen: set[str] = set()
        out: list[dict] = []
        for r in rows:
            q = (r["question"] or "").strip()
            if not q or q in seen:
                continue
            seen.add(q)
            out.append({
                "ts": r["ts"],
                "request_id": r["request_id"],
                "rating": r["rating"],
                "comment": r["comment"],
                "collection": r["collection"],
                "question": q,
                "answer": r["answer"],
            })
            if len(out) >= limit:
                break
        return out

    def summary(self, limit: int = 50, collection: str | None = None) -> dict:
        """Return dashboard-ready aggregates and recent history."""
        self.ensure_schema()
        with self._connect() as conn:
            params: tuple = ()
            where = ""
            if collection:
                where = "WHERE collection = ?"
                params = (collection,)

            # `where` is an internal literal (empty or "WHERE collection = ?");
            # the `collection` value itself is always bound via ? placeholders.
            totals_sql = (
                "SELECT COUNT(*) AS n, "  # noqa: S608
                "  AVG(latency_ms) AS avg_ms, "
                "  SUM(CASE WHEN error IS NULL THEN 0 ELSE 1 END) AS errors "
                f"FROM queries {where}"
            )
            totals = conn.execute(totals_sql, params).fetchone()

            # p50 / p95 via ROW_NUMBER — accurate, O(n log n) on an indexed table.
            percentiles_sql = f"""
                WITH ok AS (
                    SELECT latency_ms FROM queries
                    {where if where else 'WHERE 1=1'}
                    AND error IS NULL
                    ORDER BY latency_ms
                ),
                counted AS (
                    SELECT latency_ms, ROW_NUMBER() OVER () AS rn,
                           COUNT(*) OVER () AS total
                    FROM ok
                )
                SELECT
                  MAX(CASE WHEN rn = CAST(total * 0.50 AS INTEGER) THEN latency_ms END) AS p50,
                  MAX(CASE WHEN rn = CAST(total * 0.95 AS INTEGER) THEN latency_ms END) AS p95
                FROM counted
                """  # noqa: S608
            percentiles = conn.execute(percentiles_sql, params).fetchone()

            per_collection = conn.execute(
                "SELECT collection, COUNT(*) AS n, AVG(latency_ms) AS avg_ms "
                "FROM queries GROUP BY collection ORDER BY n DESC"
            ).fetchall()

            recent_sql = (
                "SELECT ts, collection, question, latency_ms, doc_count, error "  # noqa: S608
                f"FROM queries {where} ORDER BY ts DESC LIMIT ?"
            )
            recent = conn.execute(recent_sql, (*params, limit)).fetchall()

            fb_sql = (
                "SELECT "  # noqa: S608
                "  SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS pos, "
                "  SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS neg "
                f"FROM feedback {where if where else ''}"
            )
            fb = conn.execute(fb_sql, params).fetchone()

        return {
            "total": int(totals["n"] or 0),
            "errors": int(totals["errors"] or 0),
            "avg_latency_ms": int(totals["avg_ms"] or 0),
            "p50_latency_ms": int(percentiles["p50"] or 0),
            "p95_latency_ms": int(percentiles["p95"] or 0),
            "per_collection": [
                {"collection": r["collection"], "count": r["n"], "avg_ms": int(r["avg_ms"] or 0)}
                for r in per_collection
            ],
            "recent": [
                {
                    "ts": r["ts"],
                    "collection": r["collection"],
                    "question": r["question"],
                    "latency_ms": r["latency_ms"],
                    "doc_count": r["doc_count"],
                    "error": r["error"],
                }
                for r in recent
            ],
            "feedback": {
                "positive": int(fb["pos"] or 0),
                "negative": int(fb["neg"] or 0),
            },
        }

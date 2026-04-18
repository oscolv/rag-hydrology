"""Tests for evaluation helpers that don't hit external LLMs.

The full RAGAS run is integration-tested by hand with a real API key; here
we only cover loaders and pure data shaping.
"""

import pytest
import yaml

from rag.evaluation import load_golden_set


def test_load_golden_set_returns_empty_df_when_dir_missing(tmp_path):
    df = load_golden_set(tmp_path / "nope")
    assert df.empty
    assert list(df.columns) == ["user_input", "reference", "expected_sources", "notes"]


def test_load_golden_set_returns_empty_df_when_no_files(tmp_path):
    (tmp_path / "golden").mkdir()
    df = load_golden_set(tmp_path / "golden")
    assert df.empty


def test_load_golden_set_parses_one_file(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "a.yaml").write_text(yaml.safe_dump([
        {
            "question": "What is GRACE?",
            "expected_answer": "A satellite mission.",
            "expected_sources": ["tapley2004.pdf"],
            "notes": "anchor",
        },
    ]))

    df = load_golden_set(golden)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["user_input"] == "What is GRACE?"
    assert row["reference"] == "A satellite mission."
    assert row["expected_sources"] == ["tapley2004.pdf"]
    assert row["notes"] == "anchor"


def test_load_golden_set_concatenates_files_in_sorted_order(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "b.yaml").write_text(yaml.safe_dump([{"question": "second"}]))
    (golden / "a.yaml").write_text(yaml.safe_dump([{"question": "first"}]))

    df = load_golden_set(golden)
    assert list(df["user_input"]) == ["first", "second"]


def test_load_golden_set_defaults_optional_fields(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "x.yaml").write_text(yaml.safe_dump([{"question": "minimal"}]))
    df = load_golden_set(golden)
    row = df.iloc[0]
    assert row["reference"] == ""
    assert row["expected_sources"] == []
    assert row["notes"] == ""


def test_load_golden_set_rejects_top_level_dict(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "bad.yaml").write_text(yaml.safe_dump({"question": "wrong shape"}))
    with pytest.raises(ValueError, match="list of golden entries"):
        load_golden_set(golden)


def test_load_golden_set_rejects_entry_missing_question(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "bad.yaml").write_text(yaml.safe_dump([{"expected_answer": "x"}]))
    with pytest.raises(ValueError, match="must be a dict with a 'question' key"):
        load_golden_set(golden)

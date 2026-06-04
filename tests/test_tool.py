"""
tests/test_tool.py — Unit tests for the RLHF feedback tool.
Run with: pytest
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from comparator import FeedbackCollector, FeedbackEntry, ResponsePair
from dataset import get_sample_pairs, load_pairs_from_file
from analytics import load_feedback, summarize


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pair():
    return ResponsePair(
        prompt="What is 2 + 2?",
        response_a="4",
        response_b="The answer is four.",
    )


@pytest.fixture
def tmp_feedback_path(tmp_path):
    return str(tmp_path / "feedback.jsonl")


@pytest.fixture
def collector(tmp_feedback_path):
    return FeedbackCollector(output_path=tmp_feedback_path)


# ── ResponsePair ──────────────────────────────────────────────────────────────

class TestResponsePair:

    def test_pair_id_auto_generated(self, sample_pair):
        assert sample_pair.pair_id is not None
        assert len(sample_pair.pair_id) == 8

    def test_unique_pair_ids(self):
        p1 = ResponsePair(prompt="x", response_a="a", response_b="b")
        p2 = ResponsePair(prompt="x", response_a="a", response_b="b")
        assert p1.pair_id != p2.pair_id


# ── FeedbackEntry ─────────────────────────────────────────────────────────────

class TestFeedbackEntry:

    def test_to_dict_has_required_keys(self):
        entry = FeedbackEntry(
            pair_id="abc123",
            prompt="Test prompt",
            chosen="Good response",
            rejected="Bad response",
            preference="A",
            reasoning="More helpful and clear",
            quality_score=4,
        )
        d = entry.to_dict()
        for key in ["pair_id", "prompt", "chosen", "rejected", "preference",
                    "reasoning", "quality_score", "timestamp", "annotator_id"]:
            assert key in d

    def test_timestamp_is_set(self):
        entry = FeedbackEntry(
            pair_id="x", prompt="p", chosen="c", rejected="r",
            preference="B", reasoning="Clear and concise", quality_score=3,
        )
        assert entry.timestamp is not None


# ── FeedbackCollector ─────────────────────────────────────────────────────────

class TestFeedbackCollector:

    def _run_collect(self, collector, pair, inputs):
        with patch("builtins.input", side_effect=inputs):
            return collector.collect(pair)

    def test_collect_preference_a(self, collector, sample_pair):
        entry = self._run_collect(
            collector, sample_pair,
            ["A", "Response A was clearer and more direct", "4"]
        )
        assert entry.preference == "A"
        assert entry.chosen == sample_pair.response_a
        assert entry.rejected == sample_pair.response_b

    def test_collect_preference_b(self, collector, sample_pair):
        entry = self._run_collect(
            collector, sample_pair,
            ["B", "Response B was more natural and friendly", "5"]
        )
        assert entry.preference == "B"
        assert entry.chosen == sample_pair.response_b

    def test_collect_tie(self, collector, sample_pair):
        entry = self._run_collect(
            collector, sample_pair,
            ["tie", "Both responses are equally good here", "3"]
        )
        assert entry.preference == "tie"

    def test_feedback_saved_to_file(self, collector, sample_pair, tmp_feedback_path):
        with patch("builtins.input", side_effect=["A", "Much clearer explanation", "4"]):
            collector.collect(sample_pair)
        lines = Path(tmp_feedback_path).read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["preference"] == "A"

    def test_session_count_increments(self, collector, sample_pair):
        assert collector.session_count == 0
        with patch("builtins.input", side_effect=["A", "Response A is more accurate", "4"]):
            collector.collect(sample_pair)
        assert collector.session_count == 1

    def test_invalid_preference_retried(self, collector, sample_pair):
        entry = self._run_collect(
            collector, sample_pair,
            ["X", "Z", "A", "Clear and well structured response", "3"]
        )
        assert entry.preference == "A"

    def test_short_reasoning_retried(self, collector, sample_pair):
        entry = self._run_collect(
            collector, sample_pair,
            ["A", "ok", "Better and more detailed answer here", "5"]
        )
        assert len(entry.reasoning) >= 10


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestDataset:

    def test_sample_pairs_not_empty(self):
        pairs = get_sample_pairs()
        assert len(pairs) >= 4

    def test_sample_pairs_have_required_fields(self):
        for pair in get_sample_pairs():
            assert pair.prompt
            assert pair.response_a
            assert pair.response_b

    def test_load_from_file(self, tmp_path):
        data = [
            {"prompt": "What is AI?", "response_a": "Artificial intelligence.", "response_b": "Smart computers."}
        ]
        f = tmp_path / "pairs.json"
        f.write_text(json.dumps(data))
        pairs = load_pairs_from_file(str(f))
        assert len(pairs) == 1
        assert pairs[0].prompt == "What is AI?"


# ── Analytics ─────────────────────────────────────────────────────────────────

class TestAnalytics:

    def test_load_feedback_empty_when_no_file(self):
        entries = load_feedback("nonexistent/path.jsonl")
        assert entries == []

    def test_load_feedback_returns_correct_count(self, tmp_path):
        f = tmp_path / "feedback.jsonl"
        entries = [
            {"pair_id": "a1", "preference": "A", "quality_score": 4,
             "reasoning": "Good", "prompt": "p", "chosen": "c", "rejected": "r",
             "timestamp": "2026-01-01", "annotator_id": "x", "metadata": {}},
            {"pair_id": "a2", "preference": "B", "quality_score": 3,
             "reasoning": "Fine", "prompt": "p", "chosen": "c", "rejected": "r",
             "timestamp": "2026-01-01", "annotator_id": "x", "metadata": {}},
        ]
        f.write_text("\n".join(json.dumps(e) for e in entries))
        loaded = load_feedback(str(f))
        assert len(loaded) == 2

    def test_summarize_no_crash_on_empty(self, capsys):
        summarize("nonexistent/path.jsonl")
        out = capsys.readouterr().out
        assert "No feedback data" in out

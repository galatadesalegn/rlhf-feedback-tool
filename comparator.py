"""
comparator.py — Presents two AI responses side-by-side and collects
human preference feedback, simulating a real RLHF data collection pipeline.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ResponsePair:
    """A pair of AI responses to the same prompt, ready for human comparison."""
    prompt: str
    response_a: str
    response_b: str
    pair_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)


@dataclass
class FeedbackEntry:
    """
    A single human preference label — the core unit of RLHF training data.

    Fields mirror real preference datasets (e.g. Anthropic HH-RLHF, OpenAI
    summarization dataset) so the output is directly usable for fine-tuning.
    """
    pair_id: str
    prompt: str
    chosen: str          # the preferred response
    rejected: str        # the less preferred response
    preference: str      # "A", "B", or "tie"
    reasoning: str       # annotator's explanation
    quality_score: int   # 1–5 overall quality of the chosen response
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    annotator_id: str = "human_annotator"

    def to_dict(self) -> dict:
        return asdict(self)


class FeedbackCollector:
    """
    Interactive CLI tool for collecting pairwise preference feedback.

    Workflow:
      1. Load a prompt + two candidate responses
      2. Display them side by side
      3. Collect: preference (A/B/tie), reasoning, quality score
      4. Save as structured JSONL — ready for reward model training
    """

    DIVIDER = "─" * 65

    def __init__(self, output_path: str = "feedback/feedback.jsonl"):
        self.output_path = output_path
        self.session_count = 0

    # ── Display ───────────────────────────────────────────────────────────

    def _display_pair(self, pair: ResponsePair) -> None:
        print(f"\n{self.DIVIDER}")
        print(f"  PROMPT  [{pair.pair_id}]")
        print(self.DIVIDER)
        print(f"\n{pair.prompt}\n")

        print(self.DIVIDER)
        print("  RESPONSE A")
        print(self.DIVIDER)
        print(f"\n{pair.response_a}\n")

        print(self.DIVIDER)
        print("  RESPONSE B")
        print(self.DIVIDER)
        print(f"\n{pair.response_b}\n")

        print(self.DIVIDER)

    # ── Input helpers ─────────────────────────────────────────────────────

    def _get_preference(self) -> str:
        while True:
            choice = input("  Which response is better? [A / B / tie]: ").strip().upper()
            if choice in {"A", "B", "TIE"}:
                return "tie" if choice == "TIE" else choice
            print("  ⚠  Please enter A, B, or tie.")

    def _get_reasoning(self) -> str:
        while True:
            reason = input("  Why? (min 10 chars): ").strip()
            if len(reason) >= 10:
                return reason
            print("  ⚠  Please give a bit more detail.")

    def _get_quality_score(self) -> int:
        while True:
            try:
                score = int(input("  Quality score for chosen response [1–5]: ").strip())
                if 1 <= score <= 5:
                    return score
            except ValueError:
                pass
            print("  ⚠  Please enter a number between 1 and 5.")

    # ── Core ──────────────────────────────────────────────────────────────

    def collect(self, pair: ResponsePair) -> FeedbackEntry:
        """Run one annotation session for a single response pair."""
        self._display_pair(pair)

        print("\n📝  Your annotation:\n")
        preference = self._get_preference()
        reasoning = self._get_reasoning()
        quality_score = self._get_quality_score()

        chosen  = pair.response_a if preference == "A" else pair.response_b
        rejected = pair.response_b if preference == "A" else pair.response_a

        if preference == "tie":
            chosen = pair.response_a
            rejected = pair.response_b

        entry = FeedbackEntry(
            pair_id=pair.pair_id,
            prompt=pair.prompt,
            chosen=chosen,
            rejected=rejected,
            preference=preference,
            reasoning=reasoning,
            quality_score=quality_score,
        )

        self._save(entry)
        self.session_count += 1
        print(f"\n✅  Feedback saved. (session total: {self.session_count})\n")
        return entry

    def _save(self, entry: FeedbackEntry) -> None:
        """Append one feedback entry to the JSONL output file."""
        import os
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

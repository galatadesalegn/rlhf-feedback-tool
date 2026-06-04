"""
analytics.py — Summarize and analyze collected RLHF feedback data.
Useful for quality checks and inter-annotator agreement reporting.
"""

import json
from collections import Counter
from pathlib import Path


def load_feedback(path: str = "feedback/feedback.jsonl") -> list[dict]:
    """Load all feedback entries from the JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def summarize(path: str = "feedback/feedback.jsonl") -> None:
    """Print a summary report of collected feedback."""
    entries = load_feedback(path)

    if not entries:
        print("No feedback data found.")
        return

    total = len(entries)
    prefs = Counter(e["preference"] for e in entries)
    scores = [e["quality_score"] for e in entries]
    categories = Counter(
        e.get("metadata", {}).get("category", "unknown") for e in entries
    )

    avg_score = sum(scores) / len(scores) if scores else 0

    divider = "─" * 50

    print(f"\n{divider}")
    print("  RLHF FEEDBACK SUMMARY")
    print(divider)
    print(f"  Total annotations   : {total}")
    print(f"  Preferred A         : {prefs.get('A', 0)}  ({prefs.get('A', 0)/total*100:.1f}%)")
    print(f"  Preferred B         : {prefs.get('B', 0)}  ({prefs.get('B', 0)/total*100:.1f}%)")
    print(f"  Ties                : {prefs.get('tie', 0)}  ({prefs.get('tie', 0)/total*100:.1f}%)")
    print(f"  Avg quality score   : {avg_score:.2f} / 5.00")
    print(f"\n  By category:")
    for cat, count in categories.most_common():
        print(f"    {cat:<20} {count}")
    print(divider)

    print("\n  Recent annotations:\n")
    for entry in entries[-3:]:
        print(f"  [{entry['pair_id']}] Preferred: {entry['preference']}  "
              f"Score: {entry['quality_score']}/5")
        print(f"  Reason: {entry['reasoning'][:80]}...")
        print()


def export_to_json(
    input_path: str = "feedback/feedback.jsonl",
    output_path: str = "feedback/feedback_export.json",
) -> None:
    """Export JSONL feedback to a clean JSON array — ready for reward model training."""
    entries = load_feedback(input_path)
    Path(output_path).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported {len(entries)} entries → {output_path}")

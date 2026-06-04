#!/usr/bin/env python3
"""
main.py — CLI entry point for the RLHF Feedback Collection Tool.

Usage:
  python main.py                        # annotate all sample pairs
  python main.py --pairs data/pairs.json  # annotate from a custom file
  python main.py --summary              # show feedback analytics
  python main.py --export               # export JSONL → JSON
  python main.py --limit 2             # annotate only first N pairs
"""

import argparse
import sys

from comparator import FeedbackCollector
from dataset import get_sample_pairs, load_pairs_from_file
from analytics import summarize, export_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🤖 RLHF Feedback Collection Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs", type=str, default=None,
        help="Path to a custom JSON file with response pairs",
    )
    parser.add_argument(
        "--output", type=str, default="feedback/feedback.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of pairs to annotate",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show analytics summary of collected feedback",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export feedback JSONL to a clean JSON file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.summary:
        summarize(args.output)
        return 0

    if args.export:
        export_to_json(args.output)
        return 0

    # Load pairs
    if args.pairs:
        pairs = load_pairs_from_file(args.pairs)
        print(f"Loaded {len(pairs)} pairs from {args.pairs}")
    else:
        pairs = get_sample_pairs()
        print(f"Using {len(pairs)} built-in sample pairs.")

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"\n🤖  RLHF Feedback Collection Tool")
    print(f"    Pairs to annotate : {len(pairs)}")
    print(f"    Output file       : {args.output}")
    print("\nFor each pair, you'll see a prompt and two responses.")
    print("Your job: pick the better one and explain why.\n")

    collector = FeedbackCollector(output_path=args.output)

    for i, pair in enumerate(pairs, 1):
        print(f"\n{'═'*65}")
        print(f"  Pair {i} of {len(pairs)}")
        print(f"{'═'*65}")
        collector.collect(pair)

        if i < len(pairs):
            cont = input("  Continue to next pair? [Y/n]: ").strip().lower()
            if cont == "n":
                break

    print(f"\n🎉  Session complete! {collector.session_count} annotation(s) saved.")
    print(f"    File: {args.output}")
    print(f"\nRun `python main.py --summary` to see your feedback analytics.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

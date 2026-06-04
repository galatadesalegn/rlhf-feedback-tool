"""
dataset.py — Loads response pairs from JSON files and provides
built-in sample pairs for demo / testing purposes.
"""

import json
from pathlib import Path
from comparator import ResponsePair


SAMPLE_PAIRS = [
    ResponsePair(
        prompt="Explain what a neural network is to a 10-year-old.",
        response_a=(
            "A neural network is a computer system loosely inspired by the human brain. "
            "It consists of layers of interconnected nodes (neurons) that process input data "
            "and learn to recognize patterns through a training process called backpropagation."
        ),
        response_b=(
            "Imagine your brain is made of billions of tiny light bulbs all connected by wires. "
            "When you learn something new — like riding a bike — some of those bulbs light up "
            "together. A neural network is like a pretend brain inside a computer. It has lots "
            "of tiny parts connected together, and when you show it pictures or words, it lights "
            "up the right parts to figure out what something is — just like how you learned to "
            "recognize your dog's face!"
        ),
        metadata={"category": "explanation", "difficulty": "easy"},
    ),
    ResponsePair(
        prompt="What should I do if I feel overwhelmed at work?",
        response_a=(
            "You should prioritize your tasks using a matrix: urgent/important, "
            "urgent/not important, not urgent/important, not urgent/not important. "
            "Delegate what you can, and set firm boundaries on your working hours."
        ),
        response_b=(
            "Feeling overwhelmed is really common, and it's okay to pause and take a breath. "
            "A few things that often help: write down everything on your plate so it's out of "
            "your head, pick just one thing to start with, and don't hesitate to tell your "
            "manager if the workload feels unmanageable. You don't have to handle it alone."
        ),
        metadata={"category": "advice", "difficulty": "medium"},
    ),
    ResponsePair(
        prompt="Write a Python function that checks if a string is a palindrome.",
        response_a=(
            "def is_palindrome(s):\n"
            "    return s == s[::-1]"
        ),
        response_b=(
            "def is_palindrome(s: str) -> bool:\n"
            "    \"\"\"\n"
            "    Check whether a string reads the same forwards and backwards.\n"
            "    Comparison is case-insensitive and ignores spaces.\n\n"
            "    >>> is_palindrome('racecar')\n"
            "    True\n"
            "    >>> is_palindrome('Hello')\n"
            "    False\n"
            "    \"\"\"\n"
            "    cleaned = s.replace(' ', '').lower()\n"
            "    return cleaned == cleaned[::-1]"
        ),
        metadata={"category": "coding", "difficulty": "easy"},
    ),
    ResponsePair(
        prompt="Is it ethical to use AI to replace human customer service agents?",
        response_a=(
            "Yes, it is ethical because AI is more efficient, available 24/7, "
            "and reduces costs for businesses."
        ),
        response_b=(
            "This is a nuanced question. AI can handle repetitive queries efficiently and "
            "improve response times, which benefits customers. However, replacing human agents "
            "entirely raises concerns about job displacement, loss of empathy in sensitive "
            "interactions, and accountability when things go wrong. A thoughtful approach "
            "typically involves AI augmenting human agents rather than fully replacing them, "
            "especially for complex or emotionally charged situations."
        ),
        metadata={"category": "ethics", "difficulty": "hard"},
    ),
]


def load_pairs_from_file(path: str) -> list[ResponsePair]:
    """
    Load response pairs from a JSON file.

    Expected format:
    [
      {
        "prompt": "...",
        "response_a": "...",
        "response_b": "...",
        "metadata": {}   // optional
      },
      ...
    ]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ResponsePair(
            prompt=item["prompt"],
            response_a=item["response_a"],
            response_b=item["response_b"],
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]


def get_sample_pairs() -> list[ResponsePair]:
    return SAMPLE_PAIRS

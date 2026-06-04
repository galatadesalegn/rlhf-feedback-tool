<div align="center">

# 🤖 RLHF Feedback Collection Tool

### Human Preference Annotation Pipeline for AI Training

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![CI](https://github.com/galatadesalegn/rlhf-feedback-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/galatadesalegn/rlhf-feedback-tool/actions)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen?style=flat-square)](#)

<br/>

> A structured CLI tool for collecting pairwise human preference feedback on AI-generated responses — the core data collection step in **Reinforcement Learning from Human Feedback (RLHF)** pipelines.

</div>

---

## 🧠 What is RLHF?

**Reinforcement Learning from Human Feedback** is the technique used to fine-tune large language models (like ChatGPT, Claude, and Gemini) to be helpful, harmless, and honest.

The process works like this:

```
1. Two AI responses are generated for the same prompt
2. A human annotator picks the better one (or marks a tie)
3. This preference data trains a Reward Model
4. The Reward Model guides further LLM fine-tuning via RL
```

This tool automates **step 2** — the human preference collection phase.

---

## ✨ Features

- 📋 **Pairwise comparison UI** — side-by-side display of two AI responses
- 🗳️ **Structured annotation** — preference (A/B/tie), reasoning, quality score (1–5)
- 💾 **JSONL output** — format compatible with reward model training pipelines
- 📊 **Analytics dashboard** — session summaries, preference distribution, avg quality
- 📁 **Custom dataset support** — load your own prompt/response pairs from JSON
- ✅ **Input validation** — retries on invalid preference or too-short reasoning
- 🧪 **17 unit tests** — full coverage of core annotation and analytics logic
- ⚙️ **GitHub Actions CI** — tests run on Python 3.11 and 3.12

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/galatadesalegn/rlhf-feedback-tool.git
cd rlhf-feedback-tool
pip install -r requirements.txt
```

### 2. Run the annotation tool

```bash
# Annotate built-in sample pairs (great for a demo)
python main.py

# Load your own pairs
python main.py --pairs data/sample_pairs.json

# Limit to the first 2 pairs
python main.py --limit 2
```

### 3. View analytics

```bash
python main.py --summary
```

### 4. Export to JSON for training

```bash
python main.py --export
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
rlhf-feedback-tool/
├── comparator.py          # Core annotation logic — ResponsePair, FeedbackEntry, FeedbackCollector
├── dataset.py             # Built-in sample pairs + custom JSON file loader
├── analytics.py           # Feedback summarizer and JSON exporter
├── main.py                # CLI entry point (argparse)
├── data/
│   └── sample_pairs.json  # Example prompt/response pairs
├── feedback/              # Output directory (git-ignored)
│   └── feedback.jsonl     # Collected annotations
├── tests/
│   └── test_tool.py       # 17 unit tests (pytest)
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI
└── requirements.txt
```

---

## 📄 Output Format

Each annotation is saved as one line in `feedback/feedback.jsonl`:

```json
{
  "pair_id": "a3f2b1c0",
  "prompt": "Explain what a neural network is to a 10-year-old.",
  "chosen": "Imagine your brain is made of billions of tiny light bulbs...",
  "rejected": "A neural network is a computer system loosely inspired by the human brain...",
  "preference": "B",
  "reasoning": "Response B uses an analogy that a child can actually visualize and relate to.",
  "quality_score": 5,
  "timestamp": "2026-06-04T10:22:31.004521+00:00",
  "annotator_id": "human_annotator"
}
```

This format mirrors real RLHF datasets like [Anthropic's HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) and [OpenAI's summarization dataset](https://huggingface.co/datasets/openai/summarize_from_feedback), making it directly usable for reward model training with libraries like TRL or trlX.

---

## ⚙️ CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs` | built-in samples | Path to a custom JSON pairs file |
| `--output` | `feedback/feedback.jsonl` | Output file path |
| `--limit` | all pairs | Annotate only the first N pairs |
| `--summary` | — | Show analytics report |
| `--export` | — | Export JSONL → clean JSON |

---

## 🔗 Related Work

This project is inspired by and compatible with:

- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [OpenAI Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [TRL — Transformer Reinforcement Learning](https://github.com/huggingface/trl)

---

## 📄 License

MIT — free to use and adapt. See [`LICENSE`](LICENSE).

---

<div align="center">

Built by **Galata Desalegn** — Addis Ababa, Ethiopia

⭐ Star this repo if it helped you understand RLHF!

</div>

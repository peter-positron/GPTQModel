"""Pre-render Harmony-framed calibration data for GPT-OSS quantization.

Run with paibaker's venv (has Harmony tokenizer):
    /mnt/disk3/repos/paibaker/.venv/bin/python tests/generate_harmony_calibration.py \
        --source wikitext --output /mnt/datasets/calibration/harmony-wikitext-256.json \
        --samples 256

    # Token-budget mode (stop after reaching target tokens):
    /mnt/disk3/repos/paibaker/.venv/bin/python tests/generate_harmony_calibration.py \
        --source code --output /mnt/datasets/calibration/harmony-code-50k.json \
        --target-total-tokens 50000

    # Use train split to avoid calibration/eval leakage:
    /mnt/disk3/repos/paibaker/.venv/bin/python tests/generate_harmony_calibration.py \
        --source wikitext --split train --output harmony-wikitext-train-50k.json \
        --target-total-tokens 50000

Sources:
    wikitext  - Wikitext-2 paragraphs (general/encyclopedic text)
    code      - HumanEval + MBPP code problems with solutions
    stem      - MMLU STEM subjects (math, physics, CS, biology, etc.)
    mixed     - Equal token budget from wikitext + code + stem

Output: JSON file with list of {input_ids: list[int], attention_mask: list[int]}
that GPTQModel.quantize() accepts directly.

The Harmony framing wraps each text as a user turn:
    <|start|>user<|message|>{text}<|end|><|start|>assistant
This matches the inference-time distribution for GPT-OSS / Tron.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

SOURCES = ("wikitext", "code", "stem", "mixed")

STEM_SUBJECTS = {
    "abstract_algebra", "anatomy", "astronomy", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics", "computer_security",
    "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    "medical_genetics", "virology",
}


# ---------------------------------------------------------------------------
# Source loaders — each returns list[str] of text passages
# ---------------------------------------------------------------------------

def load_wikitext(max_texts: int, min_chars: int = 200, split: str = "test") -> list[str]:
    """Load filtered paragraphs from wikitext-2."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if len(t.strip()) >= min_chars]
    print(f"Wikitext-2 ({split}): {len(texts)} paragraphs >= {min_chars} chars")

    if len(texts) < max_texts:
        print(f"Warning: only {len(texts)} paragraphs available, requested {max_texts}")
    return texts[:max_texts]


def load_code(max_texts: int) -> list[str]:
    """Load code problems + solutions from HumanEval and MBPP.

    Each problem becomes one text formatted as a coding task with solution.
    Combined pool: 164 (HumanEval) + 257 (MBPP sanitized test) = 421 problems.
    """
    from datasets import load_dataset

    texts = []

    # HumanEval: prompt (function signature + docstring) + canonical_solution
    print("Loading HumanEval...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    for row in ds:
        text = (
            f"# Python coding problem\n\n"
            f"{row['prompt']}{row['canonical_solution']}\n\n"
            f"# Test cases\n{row['test']}"
        )
        texts.append(text)
    print(f"  HumanEval: {len(ds)} problems")

    # MBPP: prompt (description) + code (solution)
    print("Loading MBPP...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for row in ds:
        tests_str = "\n".join(row.get("test_list", []))
        text = (
            f"# Task: {row['prompt']}\n\n"
            f"{row['code']}\n\n"
            f"# Tests\n{tests_str}"
        )
        texts.append(text)
    print(f"  MBPP: {len(ds)} problems")

    print(f"Code pool: {len(texts)} total problems")
    if len(texts) < max_texts:
        print(f"Warning: only {len(texts)} code problems available, requested {max_texts}")
    return texts[:max_texts]


def load_stem(max_texts: int, questions_per_sample: int = 4) -> list[str]:
    """Load STEM Q&A from MMLU, packing multiple questions per sample.

    Individual MMLU questions are short (~150 chars). We pack several into
    one sample to get reasonable sequence lengths (~170 tokens, matching
    wikitext average). Each sample becomes a "study session" with related
    questions from the same or similar subjects.
    """
    from datasets import load_dataset

    print("Loading MMLU STEM subjects...")
    # Load all MMLU test data (streams lazily)
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Collect STEM questions
    stem_questions = []
    for row in ds:
        if row["subject"] in STEM_SUBJECTS:
            choices = row["choices"]
            ans_idx = row["answer"]
            q_text = (
                f"Subject: {row['subject'].replace('_', ' ').title()}\n"
                f"Q: {row['question']}\n"
            )
            for i, c in enumerate(choices):
                q_text += f"  ({chr(65 + i)}) {c}\n"
            q_text += f"Answer: ({chr(65 + ans_idx)}) {choices[ans_idx]}"
            stem_questions.append(q_text)

    print(f"  STEM questions collected: {len(stem_questions)}")

    # Shuffle for variety within packs
    random.shuffle(stem_questions)

    # Pack into samples of N questions each
    texts = []
    for i in range(0, len(stem_questions), questions_per_sample):
        batch = stem_questions[i:i + questions_per_sample]
        if len(batch) < 2:
            break  # skip tiny remainder
        text = "STEM Practice Questions\n\n" + "\n\n".join(batch)
        texts.append(text)

    print(f"  Packed into {len(texts)} samples ({questions_per_sample} questions each)")
    if len(texts) < max_texts:
        print(f"Warning: only {len(texts)} STEM samples available, requested {max_texts}")
    return texts[:max_texts]


# ---------------------------------------------------------------------------
# Harmony rendering
# ---------------------------------------------------------------------------

def render_harmony(texts: list[str], target_total_tokens: int | None = None) -> list[dict]:
    """Wrap texts in Harmony conversation framing and tokenize.

    If target_total_tokens is set, stops adding samples once the token budget
    is reached. Otherwise renders all provided texts.
    """
    from paibaker.tokenizers import render_conversation_for_completion

    samples = []
    token_counts = []
    running_total = 0

    for text in texts:
        tokens = render_conversation_for_completion(
            [{"role": "user", "content": text}],
            "assistant",
        )
        n_tokens = len(tokens)

        # Token budget check: stop if we've already met the target.
        # Allow the sample that crosses the boundary (slight overshoot OK).
        if target_total_tokens is not None and running_total >= target_total_tokens:
            break

        samples.append({
            "input_ids": tokens,
            "attention_mask": [1] * n_tokens,
        })
        token_counts.append(n_tokens)
        running_total += n_tokens

    total = sum(token_counts)
    avg = total // len(token_counts) if token_counts else 0
    mn = min(token_counts) if token_counts else 0
    mx = max(token_counts) if token_counts else 0
    print(
        f"Rendered {len(samples)} samples: "
        f"{total} total tokens, avg {avg}, min {mn}, max {mx}"
    )
    if target_total_tokens is not None:
        print(f"  Token budget: {target_total_tokens}, actual: {total} ({total/target_total_tokens:.1%})")
    return samples


# ---------------------------------------------------------------------------
# Mixed source
# ---------------------------------------------------------------------------

def load_mixed(max_texts: int, split: str = "test") -> list[str]:
    """Load equal portions from wikitext, code, and stem.

    Interleaves texts so the calibration covers all three distributions.
    """
    per_source = max_texts // 3 + 1  # +1 to account for rounding

    print("--- Loading wikitext portion ---")
    wiki_texts = load_wikitext(per_source, split=split)
    print("--- Loading code portion ---")
    code_texts = load_code(per_source)
    print("--- Loading STEM portion ---")
    stem_texts = load_stem(per_source)

    # Interleave: take from each source in round-robin
    texts = []
    sources = [wiki_texts, code_texts, stem_texts]
    max_len = max(len(s) for s in sources)
    for i in range(max_len):
        for source in sources:
            if i < len(source):
                texts.append(source[i])
            if len(texts) >= max_texts:
                break
        if len(texts) >= max_texts:
            break

    print(f"\nMixed pool: {len(texts)} texts "
          f"(wiki: {min(len(wiki_texts), per_source)}, "
          f"code: {min(len(code_texts), per_source)}, "
          f"stem: {min(len(stem_texts), per_source)})")
    return texts[:max_texts]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-render Harmony-framed calibration data for GPT-OSS",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--source", "-s", choices=SOURCES, default="wikitext",
        help="Calibration data source (default: wikitext)",
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=256,
        help="Max number of calibration samples (default: 256). "
             "May produce fewer if --target-total-tokens is set.",
    )
    parser.add_argument(
        "--target-total-tokens", type=int, default=None,
        help="Stop after reaching this total token count. "
             "Overrides --samples as the primary budget control. "
             "Pass a large --samples value to ensure enough source texts.",
    )
    parser.add_argument(
        "--split", choices=("train", "test"), default="test",
        help="Dataset split for wikitext (default: test). "
             "Use 'train' to avoid calibration/eval leakage.",
    )
    parser.add_argument(
        "--min-chars", type=int, default=200,
        help="Minimum character length for wikitext paragraphs (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for STEM question shuffling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # If token budget is set, request more source texts than needed
    max_texts = args.samples
    if args.target_total_tokens is not None:
        # Estimate: ~170 tokens per sample on average
        estimated_samples = (args.target_total_tokens // 120) + 50
        max_texts = max(max_texts, estimated_samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Warn about flags that only apply to wikitext/mixed sources
    if args.source in ("code", "stem"):
        if args.split != "test":
            print(f"Note: --split is ignored for source '{args.source}' (only applies to wikitext/mixed)")
        if args.min_chars != 200:
            print(f"Note: --min-chars is ignored for source '{args.source}' (only applies to wikitext)")

    print(f"Source: {args.source}")
    if args.target_total_tokens:
        print(f"Token budget: {args.target_total_tokens}")
    else:
        print(f"Sample count: {args.samples}")
    print()

    t0 = time.monotonic()

    if args.source == "wikitext":
        texts = load_wikitext(max_texts, args.min_chars, split=args.split)
        source_desc = f"wikitext-2-raw-v1 ({args.split} split)"
    elif args.source == "code":
        texts = load_code(max_texts)
        source_desc = "HumanEval + MBPP (code problems + solutions)"
    elif args.source == "stem":
        texts = load_stem(max_texts)
        source_desc = "MMLU STEM subjects (packed Q&A)"
    elif args.source == "mixed":
        texts = load_mixed(max_texts, split=args.split)
        source_desc = f"mixed (wikitext-{args.split} + code + stem, interleaved)"
    else:
        print(f"Unknown source: {args.source}")
        return 1

    print(f"\nLoaded {len(texts)} texts in {time.monotonic() - t0:.1f}s")

    print(f"Rendering with Harmony framing...")
    t0 = time.monotonic()
    samples = render_harmony(texts, target_total_tokens=args.target_total_tokens)
    print(f"Rendered in {time.monotonic() - t0:.1f}s")

    if not samples:
        print("ERROR: no samples produced")
        return 1

    # Save with metadata
    total_tokens = sum(len(s["input_ids"]) for s in samples)
    payload = {
        "format": "harmony_conversation",
        "framing": "<|start|>user<|message|>{text}<|end|><|start|>assistant",
        "source": source_desc,
        "n_samples": len(samples),
        "total_tokens": total_tokens,
        "samples": samples,
    }

    output_path.write_text(json.dumps(payload))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
    print(f"  {len(samples)} samples, {total_tokens} tokens")

    return 0


if __name__ == "__main__":
    sys.exit(main())

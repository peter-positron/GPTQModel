"""Pre-render Harmony-framed calibration data for GPT-OSS quantization.

Run with paibaker's venv (has Harmony tokenizer):
    /mnt/disk3/repos/paibaker/.venv/bin/python tests/generate_harmony_calibration.py \
        --output /mnt/datasets/calibration/harmony-wikitext-256.json \
        --samples 256

Output: JSON file with list of {input_ids: list[int], attention_mask: list[int]}
that GPTQModel.quantize() accepts directly.

The Harmony framing wraps each text as a user turn:
    <|start|>user<|message|>{text}<|end|><|start|>assistant
This matches the inference-time distribution for GPT-OSS / Tron.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def load_wikitext(n_samples: int, min_chars: int = 200) -> list[str]:
    """Load filtered paragraphs from wikitext-2."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) >= min_chars]
    print(f"Wikitext-2: {len(texts)} paragraphs >= {min_chars} chars")

    if len(texts) < n_samples:
        print(f"Warning: only {len(texts)} paragraphs available, requested {n_samples}")
    return texts[:n_samples]


def render_harmony(texts: list[str]) -> list[dict]:
    """Wrap texts in Harmony conversation framing and tokenize."""
    from paibaker.tokenizers import (
        HarmonyTokenizerAdapter,
        render_conversation_for_completion,
    )

    tok = HarmonyTokenizerAdapter()
    samples = []
    token_counts = []

    for text in texts:
        tokens = render_conversation_for_completion(
            [{"role": "user", "content": text}],
            "assistant",
        )
        samples.append({
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        })
        token_counts.append(len(tokens))

    total = sum(token_counts)
    avg = total // len(token_counts) if token_counts else 0
    mn = min(token_counts) if token_counts else 0
    mx = max(token_counts) if token_counts else 0
    print(
        f"Rendered {len(samples)} samples: "
        f"{total} total tokens, avg {avg}, min {mn}, max {mx}"
    )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-render Harmony-framed calibration data",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=256,
        help="Number of calibration samples (default: 256)",
    )
    parser.add_argument(
        "--min-chars", type=int, default=200,
        help="Minimum character length for wikitext paragraphs (default: 200)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading wikitext-2...")
    t0 = time.monotonic()
    texts = load_wikitext(args.samples, args.min_chars)
    print(f"Loaded in {time.monotonic() - t0:.1f}s")

    print(f"Rendering with Harmony framing...")
    t0 = time.monotonic()
    samples = render_harmony(texts)
    print(f"Rendered in {time.monotonic() - t0:.1f}s")

    # Save with metadata
    payload = {
        "format": "harmony_conversation",
        "framing": "<|start|>user<|message|>{text}<|end|><|start|>assistant",
        "source": "wikitext-2-raw-v1 (test split)",
        "n_samples": len(samples),
        "total_tokens": sum(len(s["input_ids"]) for s in samples),
        "samples": samples,
    }

    output_path.write_text(json.dumps(payload))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Smoke test: load a quantized GPT-OSS model and generate text.

Usage:
    python tests/smoke_generate.py /raid/peter/quants/gpt-oss-20b-good-gar-on
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROMPTS = [
    "The capital of France is",
    "In quantum computing, a qubit differs from a classical bit because",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
]

MAX_NEW_TOKENS = 64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        return 1

    meta_path = model_dir / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Model: {meta.get('profile', '?')}/gar-{meta.get('gar', '?')}")
        print(f"Config: {meta.get('quantize_config', {})}")
    print(f"Device: {args.device}")
    print()

    # Apply paibaker patches if available
    try:
        from paibaker.patches import apply_all
        apply_all()
    except ImportError:
        pass

    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print(f"Loading quantized model from {model_dir}...")
    t0 = time.monotonic()
    model = GPTQModel.load(str(model_dir), device=args.device)
    load_time = time.monotonic() - t0
    print(f"Loaded in {load_time:.1f}s\n")

    results = []
    all_ok = True

    for i, prompt in enumerate(PROMPTS):
        print(f"--- Prompt {i + 1}/{len(PROMPTS)} ---")
        print(f"Input: {prompt!r}")

        t0 = time.monotonic()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            args.device
        )
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
        )
        gen_time = time.monotonic() - t0
        new_tokens = output_ids.shape[1] - input_ids.shape[1]

        output_text = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        generated = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        print(f"Output ({new_tokens} tokens, {gen_time:.1f}s):")
        print(f"  {generated[:200]}")

        # Basic sanity: generated something, not all garbage
        is_ok = (
            new_tokens > 0
            and len(generated.strip()) > 0
            and not all(c == generated[0] for c in generated[:20])
        )
        status = "OK" if is_ok else "SUSPECT"
        if not is_ok:
            all_ok = False
        print(f"  [{status}]")
        print()

        results.append({
            "prompt": prompt,
            "generated": generated[:500],
            "new_tokens": new_tokens,
            "gen_time_seconds": round(gen_time, 2),
            "status": status,
        })

    # Write results
    results_path = model_dir / "smoke_generate_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {results_path}")

    if all_ok:
        print("\nSMOKE TEST PASSED")
        return 0
    else:
        print("\nSMOKE TEST: some outputs look suspect")
        return 1


if __name__ == "__main__":
    sys.exit(main())

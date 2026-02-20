"""Quantize GPT-OSS with configurable profile and GAR setting.

Usage:
    python tests/quantize_gptoss_matrix.py --profile good --gar on --output-dir /raid/peter/quants/gpt-oss-20b-good-gar-on
    python tests/quantize_gptoss_matrix.py --profile best --gar off --output-dir /raid/peter/quants/gpt-oss-20b-best-gar-off
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from safetensors import safe_open


MODEL_PATH = "/raid/peter/openai--gpt-oss-20b-bf16"

PROFILES = {
    "best": {"bits": 4, "group_size": 64, "sym": True, "desc_act": False},
    "good": {"bits": 4, "group_size": 128, "sym": True, "desc_act": False},
}

STACKED_FRAGMENTS = [
    ".experts.gate_up_proj.",
    ".experts.down_proj.",
]

UNSTACKED_FRAGMENTS = [
    ".experts.gate_up.0.",
    ".experts.gate_up.1.",
    ".experts.down.0.",
    ".experts.down.1.",
]

CALIBRATION_FALLBACK = [
    (
        "The quick brown fox jumps over the lazy dog. "
        "This is a longer sentence to ensure the tokenized "
        "sequence exceeds the minimum length threshold for "
        "calibration data. We need enough tokens to build "
        "a meaningful Hessian for each quantized layer."
    ),
    (
        "In a hole in the ground there lived a hobbit. "
        "Not a nasty, dirty, wet hole, filled with the "
        "ends of worms and an oozy smell, nor yet a dry, "
        "bare, sandy hole with nothing in it to sit down "
        "on or to eat: it was a hobbit-hole."
    ),
]


def load_calibration(
    n_samples: int = 128, use_harmony: bool = True,
) -> list[str | dict]:
    """Load calibration data with Harmony conversation framing.

    Returns a list suitable for GPTQModel.quantize():
    - With Harmony: list of token-ID dicts (pre-tokenized)
    - Without Harmony: list of raw text strings (GPTQModel tokenizes)
    """
    # Load raw texts from wikitext
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 200]
        texts = texts[:n_samples]
        print(f"Calibration: {len(texts)} raw texts from wikitext-2")
    except Exception as e:
        print(f"Calibration: wikitext unavailable ({e}), using fallback")
        texts = CALIBRATION_FALLBACK

    if not use_harmony:
        return texts

    # Wrap in Harmony conversation framing
    try:
        from paibaker.tokenizers import (
            HarmonyTokenizerAdapter,
            render_conversation_for_completion,
        )
        tok = HarmonyTokenizerAdapter()
        samples = []
        for text in texts:
            tokens = render_conversation_for_completion(
                [{"role": "user", "content": text}],
                "assistant",
            )
            samples.append(
                {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
            )
        print(
            f"Calibration: {len(samples)} Harmony-framed samples, "
            f"avg {sum(len(s['input_ids']) for s in samples) // len(samples)} tokens"
        )
        return samples
    except Exception as e:
        print(f"Calibration: Harmony unavailable ({e}), using raw text")
        return texts


def collect_safetensor_keys(model_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in model_dir.glob("*.safetensors"):
        with safe_open(str(path), framework="pt") as f:
            keys.update(f.keys())
    return keys


def verify_keys(keys: set[str]) -> bool:
    ok = True
    for frag in STACKED_FRAGMENTS:
        matches = [k for k in keys if frag in k]
        if matches:
            print(f"  PASS: {len(matches)} keys matching '{frag}'")
        else:
            print(f"  FAIL: no keys matching '{frag}'")
            ok = False

    for frag in UNSTACKED_FRAGMENTS:
        matches = [k for k in keys if frag in k]
        if matches:
            print(f"  FAIL: {len(matches)} unstacked keys matching '{frag}'")
            ok = False
        else:
            print(f"  PASS: no unstacked keys matching '{frag}'")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", required=True, choices=list(PROFILES),
    )
    parser.add_argument(
        "--gar", required=True, choices=["on", "off"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--calibration-samples", type=int, default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--no-harmony", action="store_true",
        help="Skip Harmony framing, use raw text calibration",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    gar = args.gar == "on"
    output_dir = Path(args.output_dir)

    print(f"Profile: {args.profile} -> {profile}")
    print(f"GAR (act_group_aware): {gar}")
    print(f"Output: {output_dir}")
    print(f"Source: {MODEL_PATH}")

    # Apply paibaker runtime patches if available
    try:
        from paibaker.patches import apply_all
        applied = apply_all()
        print(f"Applied paibaker patches: {applied}")
    except ImportError:
        print("paibaker patches not available, continuing without")

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    qcfg = QuantizeConfig(
        bits=profile["bits"],
        group_size=profile["group_size"],
        sym=profile["sym"],
        desc_act=profile["desc_act"],
        act_group_aware=gar,
    )

    print(f"\nLoading model from {MODEL_PATH}...")
    t0 = time.monotonic()
    model = GPTQModel.load(MODEL_PATH, quantize_config=qcfg)
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")

    calibration = load_calibration(
        args.calibration_samples, use_harmony=not args.no_harmony,
    )

    print("Quantizing...")
    t0 = time.monotonic()
    model.quantize(calibration)
    quant_time = time.monotonic() - t0
    print(f"Quantization completed in {quant_time:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_dir}...")
    t0 = time.monotonic()
    model.save_quantized(str(output_dir))
    print(f"Saved in {time.monotonic() - t0:.1f}s")

    # Write run metadata
    meta = {
        "source": MODEL_PATH,
        "profile": args.profile,
        "gar": args.gar,
        "quantize_config": {
            "bits": profile["bits"],
            "group_size": profile["group_size"],
            "sym": profile["sym"],
            "desc_act": profile["desc_act"],
            "act_group_aware": gar,
        },
        "quant_time_seconds": round(quant_time, 1),
        "calibration_samples": len(calibration),
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Verify keys
    keys = collect_safetensor_keys(output_dir)
    print(f"\nTotal keys: {len(keys)}")
    ok = verify_keys(keys)

    if ok:
        print(f"\nSUCCESS: {args.profile}/gar-{args.gar}")
    else:
        print(f"\nFAILURE: {args.profile}/gar-{args.gar}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

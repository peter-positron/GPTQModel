"""Quantize GPT-OSS with configurable profile and GAR setting.

Usage:
    python tests/quantize_gptoss_matrix.py --profile good --gar on --output-dir /path/to/quants/good-gar-on
    python tests/quantize_gptoss_matrix.py --profile best --gar off --output-dir /path/to/quants/best-gar-off \
        --calibration /mnt/datasets/calibration/harmony-wikitext-256.json

Pre-render calibration data first (uses paibaker venv for Harmony tokenizer):
    /mnt/disk3/repos/paibaker/.venv/bin/python tests/generate_harmony_calibration.py \
        --output /mnt/datasets/calibration/harmony-wikitext-256.json --samples 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from safetensors import safe_open


MODEL_PATH = os.environ.get(
    "GPTQMODEL_SOURCE_MODEL",
    "/mnt/datasets/source/openai--gpt-oss-20b-bf16",
)

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
    calibration_path: str | None = None,
    n_samples: int = 128,
) -> list[str | dict]:
    """Load calibration data for GPTQModel.quantize().

    Priority:
    1. Pre-rendered JSON file (from generate_harmony_calibration.py)
    2. Raw wikitext-2 text (GPTQModel tokenizes internally)
    3. Hardcoded fallback strings

    Returns list of dicts (pre-tokenized) or strings (raw text).
    """
    # Path 1: pre-rendered Harmony calibration JSON
    if calibration_path:
        path = Path(calibration_path)
        if not path.exists():
            print(f"ERROR: calibration file not found: {path}")
            sys.exit(1)
        payload = json.loads(path.read_text())
        samples = payload["samples"][:n_samples]
        total_tokens = sum(len(s["input_ids"]) for s in samples)
        avg_tokens = total_tokens // len(samples) if samples else 0
        print(
            f"Calibration: {len(samples)} pre-rendered samples from {path.name}, "
            f"format={payload.get('format', 'unknown')}, "
            f"{total_tokens} total tokens, avg {avg_tokens}"
        )
        return samples

    # Path 2: raw wikitext (no Harmony framing)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 200]
        texts = texts[:n_samples]
        print(f"Calibration: {len(texts)} raw texts from wikitext-2 (no Harmony framing)")
        return texts
    except Exception as e:
        print(f"Calibration: wikitext unavailable ({e}), using fallback")
        return CALIBRATION_FALLBACK


def collect_safetensor_keys(model_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in model_dir.glob("*.safetensors"):
        with safe_open(str(path), framework="pt") as f:
            keys.update(f.keys())
    return keys


def verify_keys(keys: set[str]) -> bool:
    """Verify expert key layout matches the save format mode.

    In stacked mode: stacked keys present, per-expert keys absent.
    In per_expert mode: per-expert keys present, stacked keys absent.
    """
    mode = os.environ.get("GPTQMODEL_MOE_SAVE_FORMAT", "stacked")
    print(f"  Save format mode: {mode}")

    if mode == "per_expert":
        expect_present = UNSTACKED_FRAGMENTS
        expect_absent = STACKED_FRAGMENTS
        present_label, absent_label = "per-expert", "stacked"
    else:
        expect_present = STACKED_FRAGMENTS
        expect_absent = UNSTACKED_FRAGMENTS
        present_label, absent_label = "stacked", "per-expert"

    ok = True
    for frag in expect_present:
        matches = [k for k in keys if frag in k]
        if matches:
            print(f"  PASS: {len(matches)} {present_label} keys matching '{frag}'")
        else:
            print(f"  FAIL: no {present_label} keys matching '{frag}'")
            ok = False

    for frag in expect_absent:
        matches = [k for k in keys if frag in k]
        if matches:
            print(f"  FAIL: {len(matches)} {absent_label} keys matching '{frag}' (should be absent)")
            ok = False
        else:
            print(f"  PASS: no {absent_label} keys matching '{frag}'")

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
        "--calibration", type=str, default=None,
        help="Path to pre-rendered calibration JSON (from generate_harmony_calibration.py)",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=128,
        help="Number of calibration samples to use (default: 128)",
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
        calibration_path=args.calibration,
        n_samples=args.calibration_samples,
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
        "calibration_source": args.calibration or "wikitext-2-raw (no Harmony)",
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

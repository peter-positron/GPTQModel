"""
End-to-end test: quantize GPT-OSS, save, verify stacked keys.

Usage:
    python tests/test_gptoss_restack_e2e.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open

MODEL_PATH = (
    "/opt/positron/weights/huggingface/"
    "positron-ai/openai--gpt-oss-20b-bf16"
)

# Expected stacked key fragments (should appear in saved keys)
STACKED_FRAGMENTS = [
    ".experts.gate_up_proj.",
    ".experts.down_proj.",
]

# Forbidden unstacked key fragments (should NOT appear)
UNSTACKED_FRAGMENTS = [
    ".experts.gate_up.0.",
    ".experts.gate_up.1.",
    ".experts.down.0.",
    ".experts.down.1.",
]


def collect_safetensor_keys(model_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in model_dir.glob("*.safetensors"):
        with safe_open(str(path), framework="pt") as f:
            keys.update(f.keys())
    return keys


def main() -> int:
    # Apply paibaker runtime patches if available (dtype guards etc.)
    try:
        from paibaker.patches import apply_all
        applied = apply_all()
        print(f"Applied paibaker patches: {applied}")
    except ImportError:
        print("paibaker patches not available, continuing without")

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
    )

    print(f"Loading model from {MODEL_PATH}...")
    model = GPTQModel.load(
        MODEL_PATH,
        quantize_config=qcfg,
    )

    print("Preparing calibration data...")
    calibration = [
        (
            "The quick brown fox jumps over the lazy dog. "
            "This is a longer sentence to ensure the tokenized "
            "sequence exceeds the minimum length threshold for "
            "calibration data. We need enough tokens to build "
            "a meaningful Hessian for each quantized layer. "
            "The model has 24 decoder layers with 32 experts "
            "each, attention projections, and layer norms. "
            "Quantization requires representative activations "
            "from the calibration forward pass through every "
            "module in the computation graph."
        ),
        (
            "In a hole in the ground there lived a hobbit. "
            "Not a nasty, dirty, wet hole, filled with the "
            "ends of worms and an oozy smell, nor yet a dry, "
            "bare, sandy hole with nothing in it to sit down "
            "on or to eat: it was a hobbit-hole, and that "
            "means comfort. It had a perfectly round door "
            "like a porthole, painted green, with a shiny "
            "yellow brass knob in the exact middle."
        ),
    ]

    print("Quantizing...")
    model.quantize(calibration)

    with tempfile.TemporaryDirectory(prefix="gptoss_restack_") as tmp:
        save_dir = Path(tmp)
        print(f"Saving to {save_dir}...")
        model.save_quantized(str(save_dir))

        keys = collect_safetensor_keys(save_dir)
        print(f"Total keys in saved model: {len(keys)}")

        # Check for stacked keys
        stacked_found = []
        for frag in STACKED_FRAGMENTS:
            matches = [k for k in keys if frag in k]
            if matches:
                stacked_found.extend(matches[:2])
                print(f"  PASS: found {len(matches)} keys "
                      f"matching '{frag}'")
            else:
                print(f"  FAIL: no keys matching '{frag}'")

        # Check for forbidden unstacked keys
        unstacked_found = []
        for frag in UNSTACKED_FRAGMENTS:
            matches = [k for k in keys if frag in k]
            if matches:
                unstacked_found.extend(matches[:2])
                print(f"  FAIL: found {len(matches)} unstacked "
                      f"keys matching '{frag}'")
            else:
                print(f"  PASS: no unstacked keys "
                      f"matching '{frag}'")

        if stacked_found and not unstacked_found:
            print("\nSUCCESS: all expert keys are in stacked "
                  "format")
            # Print sample keys
            print("\nSample stacked keys:")
            for k in sorted(stacked_found)[:6]:
                print(f"  {k}")
            return 0
        else:
            print("\nFAILURE: key format mismatch")
            if not stacked_found:
                print("  No stacked keys found")
            if unstacked_found:
                print("  Unstacked keys still present:")
                for k in sorted(unstacked_found)[:6]:
                    print(f"    {k}")
            return 1


if __name__ == "__main__":
    sys.exit(main())

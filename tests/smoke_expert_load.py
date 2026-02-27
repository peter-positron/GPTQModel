"""Smoke test: load GPT-OSS GPTQ artifact, verify expert weights
non-zero, router sane, forward pass produces finite logits.

Usage (always from source, never installed wheel):
    PYTHONPATH=/path/to/peter-GPTQModel \
      /path/to/venv/bin/python tests/smoke_expert_load.py \
      /path/to/gptq-artifact

No AutoTokenizer. Uses fixed token IDs for forward pass.
Forces CPU + TORCH backend to avoid kernel selection issues.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: smoke_expert_load.py <artifact_dir>")
        sys.exit(1)

    artifact = sys.argv[1]

    # -- [A] Import + version --
    from gptqmodel.models.auto import GPTQModel
    from gptqmodel.utils.backend import BACKEND
    import transformers
    print(f"[A] transformers={transformers.__version__}")
    print(f"    gptqmodel source from PYTHONPATH")

    # -- [B] Load model (bypass tokenizer entirely) --
    print(f"[B] Loading {artifact}")
    print(f"    device=cpu, backend=TORCH")

    # Patch out tokenizer load — we don't need it
    from transformers import AutoTokenizer
    _orig_from_pt = AutoTokenizer.from_pretrained
    AutoTokenizer.from_pretrained = staticmethod(
        lambda *a, **kw: None
    )
    try:
        model = GPTQModel.load(
            artifact,
            device="cpu",
            backend=BACKEND.TORCH,
            trust_remote_code=True,
        )
    finally:
        AutoTokenizer.from_pretrained = _orig_from_pt

    cfg = model.model.config
    num_experts = cfg.num_local_experts
    num_layers = cfg.num_hidden_layers
    print(f"    layers={num_layers} experts={num_experts}")

    # -- [C] Expert tensor sanity --
    print("[C] Expert tensor check (layer 0)")
    experts = model.model.model.layers[0].mlp.experts
    all_ok = True
    for name, ml in [("gate_up", experts.gate_up),
                     ("down", experts.down)]:
        for i in range(min(num_experts, 4)):
            mod = ml[i]
            qw = mod.qweight.abs().max().item()
            qz = mod.qzeros.abs().max().item()
            sc = mod.scales.abs().max().item()
            ok = qw > 0 and sc > 0
            tag = "OK" if ok else "FAIL"
            print(
                f"    {name}[{i}] qw={qw:.0f}"
                f" qz={qz:.0f} sc={sc:.4f} [{tag}]"
            )
            if not ok:
                all_ok = False
    if not all_ok:
        print("FATAL: expert weights are zero")
        sys.exit(1)

    # -- [D] Router sanity --
    print("[D] Router check (layer 0)")
    router = model.model.model.layers[0].mlp.router
    hidden_size = cfg.hidden_size
    fake = torch.randn(1, 1, hidden_size)
    with torch.no_grad():
        rout = router(fake)
    if isinstance(rout, tuple):
        indices = rout[-1]
    else:
        indices = rout
    uniq = indices.unique()
    print(
        f"    dtype={indices.dtype}"
        f" range=[{indices.min().item()},{indices.max().item()}]"
        f" unique={len(uniq)}"
        f" (expect 0..{num_experts - 1})"
    )
    assert indices.min().item() >= 0
    assert indices.max().item() < num_experts

    # -- [E] Forward pass with fixed token IDs --
    print("[E] Forward pass (4 tokens, CPU)")
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out = model.model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    finite = logits.isfinite().all().item()
    absmax = logits.abs().max().item()
    argmax = logits[0, -1].argmax().item()
    # Deterministic hash of last-position logits
    last = logits[0, -1].float().cpu()
    h = hash(last.numpy().tobytes())
    print(
        f"    shape={tuple(logits.shape)}"
        f" finite={finite} absmax={absmax:.2f}"
        f" argmax={argmax}"
    )
    print(f"    logits_hash={h}")
    assert finite, "logits contain NaN/Inf"
    assert absmax > 0, "logits all zero"

    print("\n=== ALL SMOKE CHECKS PASSED ===")


if __name__ == "__main__":
    main()

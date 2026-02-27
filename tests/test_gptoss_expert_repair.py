"""Test that stacked expert checkpoint tensors are correctly
unstacked into per-expert modules during load.

Pure unit test: requires only torch + safetensors.  Does NOT import
gptqmodel/__init__.py or any heavy dep (triton, accelerate, etc.).
Uses spec_from_file_location to load gpt_oss.py in isolation.
"""

import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

# --------------- dimensions (small, deterministic) ---------------
NUM_EXPERTS = 2
HIDDEN = 32
INTERMEDIATE = 16
NUM_LAYERS = 1
BITS = 4
GROUP_SIZE = 32
PACK_BITS = 32  # int32

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------- isolated module loader ---------------

def _load_module_from_file(name: str, path: Path):
    """Load a single .py file as a module, no package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_gpt_oss():
    """Load gpt_oss.py and its direct deps without the package root.

    Stubs out the two relative imports that gpt_oss.py needs:
      from ..base import BaseQModel
      from ..expert_restack import ExpertRestackSpec
      from ...utils.logger import setup_logger
    """
    # 1) Stub logger
    logger_mod = SimpleNamespace(
        setup_logger=lambda: SimpleNamespace(
            info=lambda *a, **kw: None,
            warning=lambda *a, **kw: None,
        ),
    )
    sys.modules["gptqmodel"] = SimpleNamespace()
    sys.modules["gptqmodel.utils"] = SimpleNamespace()
    sys.modules["gptqmodel.utils.logger"] = logger_mod

    # 2) Load expert_restack (pure Python, only needs torch)
    restack_path = (
        REPO_ROOT / "gptqmodel" / "models" / "expert_restack.py"
    )
    # expert_restack imports from ...utils.model — stub it
    model_utils_stub = SimpleNamespace(
        OffloadTensorRef=object,
        TensorSource=object,
    )
    sys.modules["gptqmodel.utils.model"] = model_utils_stub
    restack_mod = _load_module_from_file(
        "gptqmodel.models.expert_restack", restack_path,
    )

    # 3) Stub BaseQModel (gpt_oss.py inherits from it)
    base_mod = SimpleNamespace(BaseQModel=object)
    sys.modules["gptqmodel.models"] = SimpleNamespace()
    sys.modules["gptqmodel.models.base"] = base_mod
    sys.modules[
        "gptqmodel.models.expert_restack"
    ] = restack_mod

    # 4) Stub nn_modules imports (for _apply_v1_to_v2)
    sys.modules["gptqmodel.nn_modules"] = SimpleNamespace()
    sys.modules["gptqmodel.nn_modules.qlinear"] = SimpleNamespace(
        BaseQuantLinear=torch.nn.Module,
    )
    sys.modules["gptqmodel.utils.model"] = SimpleNamespace(
        convert_gptq_v1_to_v2_format_module=lambda **kw: None,
        OffloadTensorRef=object,
        TensorSource=object,
    )

    # 5) Load gpt_oss.py
    gpt_oss_path = (
        REPO_ROOT
        / "gptqmodel"
        / "models"
        / "definitions"
        / "gpt_oss.py"
    )
    sys.modules["gptqmodel.models.definitions"] = SimpleNamespace()
    return _load_module_from_file(
        "gptqmodel.models.definitions.gpt_oss", gpt_oss_path,
    )


# Load once at module level
_gpt_oss = _load_gpt_oss()
GPTOSSGPTQ = _gpt_oss.GPTOSSGPTQ


# --------------- shape helpers ---------------

def _qweight_shape(in_f: int, out_f: int) -> tuple:
    return (in_f // (PACK_BITS // BITS), out_f)


def _qzeros_shape(in_f: int, out_f: int) -> tuple:
    return (
        math.ceil(in_f / GROUP_SIZE),
        out_f // (PACK_BITS // BITS),
    )


def _scales_shape(in_f: int, out_f: int) -> tuple:
    return (math.ceil(in_f / GROUP_SIZE), out_f)


def _g_idx(in_f: int) -> torch.Tensor:
    return torch.tensor(
        [i // GROUP_SIZE for i in range(in_f)],
        dtype=torch.int32,
    )


# --------------- fixtures ---------------

def _make_stacked_checkpoint(tmpdir: Path):
    """Create safetensors with stacked expert keys + non-zero data."""
    tensors = {}
    prefix = "model.layers.0.mlp.experts"

    for proj_name, in_f, out_f in [
        ("gate_up_proj", HIDDEN, 2 * INTERMEDIATE),
        ("down_proj", INTERMEDIATE, HIDDEN),
    ]:
        for suffix, shape, dtype in [
            ("qweight", _qweight_shape(in_f, out_f), torch.int32),
            ("qzeros", _qzeros_shape(in_f, out_f), torch.int32),
            ("scales", _scales_shape(in_f, out_f), torch.float16),
        ]:
            t = torch.randint(
                1, 127, (NUM_EXPERTS, *shape), dtype=dtype,
            )
            tensors[f"{prefix}.{proj_name}.{suffix}"] = t

        gi = _g_idx(in_f).unsqueeze(0).expand(NUM_EXPERTS, -1)
        tensors[f"{prefix}.{proj_name}.g_idx"] = gi.contiguous()

        bias = torch.randn(NUM_EXPERTS, out_f, dtype=torch.float16)
        tensors[f"{prefix}.{proj_name}_bias"] = bias

    save_file(tensors, str(tmpdir / "model.safetensors"))
    return tensors


class _FakeQuantLinear(torch.nn.Module):
    """Minimal stand-in for TorchQuantLinear with zero-init buffers."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.register_buffer(
            "qweight",
            torch.zeros(_qweight_shape(in_f, out_f), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(_qzeros_shape(in_f, out_f), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros(_scales_shape(in_f, out_f), dtype=torch.float16),
        )
        self.register_buffer(
            "g_idx",
            _g_idx(in_f),
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(out_f, dtype=torch.float16),
        )


def _make_mock_model(tmpdir: Path):
    """Build minimal model with per-expert _FakeQuantLinear modules."""
    gu_in, gu_out = HIDDEN, 2 * INTERMEDIATE
    d_in, d_out = INTERMEDIATE, HIDDEN

    gate_up = torch.nn.ModuleList([
        _FakeQuantLinear(gu_in, gu_out) for _ in range(NUM_EXPERTS)
    ])
    down = torch.nn.ModuleList([
        _FakeQuantLinear(d_in, d_out) for _ in range(NUM_EXPERTS)
    ])

    experts = torch.nn.Module()
    experts.gate_up = gate_up
    experts.down = down

    mlp = torch.nn.Module()
    mlp.experts = experts

    layer = torch.nn.Module()
    layer.mlp = mlp

    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([layer])

    model = torch.nn.Module()
    model.model = inner
    model.config = SimpleNamespace(
        num_local_experts=NUM_EXPERTS,
        num_hidden_layers=NUM_LAYERS,
    )
    model.checkpoint_file_name = str(tmpdir / "model.safetensors")
    return model


def _make_repair_ctx():
    """Minimal GPTOSSGPTQ instance for calling _repair_*."""
    ctx = GPTOSSGPTQ.__new__(GPTOSSGPTQ)
    ctx.quantize_config = SimpleNamespace(
        bits=BITS, pack_dtype=torch.int32, format="gptq",
    )
    ctx.qlinear_kernel = None  # skip V1->V2 in this test
    return ctx


# --------------- tests ---------------

def test_expert_repair_loads_stacked_tensors():
    """Stacked checkpoint tensors must populate per-expert modules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src = _make_stacked_checkpoint(tmpdir)
        model = _make_mock_model(tmpdir)

        # Pre-condition: all expert qweights are zero
        for i in range(NUM_EXPERTS):
            gu = model.model.layers[0].mlp.experts.gate_up[i]
            assert gu.qweight.abs().max().item() == 0
            d = model.model.layers[0].mlp.experts.down[i]
            assert d.qweight.abs().max().item() == 0

        ctx = _make_repair_ctx()
        ctx._repair_stacked_experts_if_needed(model)

        # Post-condition: every expert buffer matches source
        pfx = "model.layers.0.mlp.experts"
        for i in range(NUM_EXPERTS):
            gu = model.model.layers[0].mlp.experts.gate_up[i]
            assert torch.equal(
                gu.qweight, src[f"{pfx}.gate_up_proj.qweight"][i],
            ), f"gate_up[{i}].qweight mismatch"
            assert torch.equal(
                gu.scales, src[f"{pfx}.gate_up_proj.scales"][i],
            ), f"gate_up[{i}].scales mismatch"
            assert torch.equal(
                gu.qzeros, src[f"{pfx}.gate_up_proj.qzeros"][i],
            ), f"gate_up[{i}].qzeros mismatch"
            assert torch.allclose(
                gu.bias.data, src[f"{pfx}.gate_up_proj_bias"][i],
            ), f"gate_up[{i}].bias mismatch"
            assert gu.qweight.abs().max().item() > 0

            d = model.model.layers[0].mlp.experts.down[i]
            assert torch.equal(
                d.qweight, src[f"{pfx}.down_proj.qweight"][i],
            ), f"down[{i}].qweight mismatch"
            assert torch.equal(
                d.scales, src[f"{pfx}.down_proj.scales"][i],
            ), f"down[{i}].scales mismatch"
            assert torch.equal(
                d.qzeros, src[f"{pfx}.down_proj.qzeros"][i],
            ), f"down[{i}].qzeros mismatch"
            assert torch.allclose(
                d.bias.data, src[f"{pfx}.down_proj_bias"][i],
            ), f"down[{i}].bias mismatch"
            assert d.qweight.abs().max().item() > 0
            assert d.scales.abs().max().item() > 0
            assert gu.scales.abs().max().item() > 0


def test_expert_repair_noop_when_already_loaded():
    """Repair must skip when expert weights are already non-zero."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_stacked_checkpoint(tmpdir)
        model = _make_mock_model(tmpdir)

        sentinel = torch.ones_like(
            model.model.layers[0].mlp.experts.gate_up[0].qweight,
        )
        model.model.layers[0].mlp.experts.gate_up[0].qweight.data.copy_(
            sentinel,
        )

        ctx = _make_repair_ctx()
        ctx._repair_stacked_experts_if_needed(model)

        assert torch.equal(
            model.model.layers[0].mlp.experts.gate_up[0].qweight,
            sentinel,
        ), "Repair should be no-op when weights already loaded"

# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for MoE expert restacking logic.

Uses stub imports to avoid pulling in the full gptqmodel package
which has heavy/optional dependencies.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Tuple, Union

import torch

_repo = Path(__file__).resolve().parent.parent

# -- Stub gptqmodel.utils.model with just the types we need --------


@dataclass
class OffloadTensorRef:
    path: str
    torch_dtype: torch.dtype
    shape: Tuple[int, ...]
    format: str
    weight_name: str | None = None
    data_offsets: tuple[int, int] | None = None


@dataclass
class TensorSource:
    name: str
    torch_dtype: torch.dtype
    shape: Tuple[int, ...]
    source: Union[torch.Tensor, OffloadTensorRef]

    @property
    def num_bytes(self) -> int:
        import math
        return 4 * math.prod(self.shape or (1,))


_stub_model = ModuleType("gptqmodel.utils.model")
_stub_model.TensorSource = TensorSource
_stub_model.OffloadTensorRef = OffloadTensorRef

# Register stubs so expert_restack can import from them
_stub_gptqmodel = ModuleType("gptqmodel")
_stub_utils = ModuleType("gptqmodel.utils")
sys.modules.setdefault("gptqmodel", _stub_gptqmodel)
sys.modules.setdefault("gptqmodel.utils", _stub_utils)
sys.modules["gptqmodel.utils.model"] = _stub_model

# Now load expert_restack
_spec = importlib.util.spec_from_file_location(
    "gptqmodel.models.expert_restack",
    _repo / "gptqmodel" / "models" / "expert_restack.py",
)
_restack = importlib.util.module_from_spec(_spec)
sys.modules["gptqmodel.models.expert_restack"] = _restack
_spec.loader.exec_module(_restack)

ExpertRestackSpec = _restack.ExpertRestackSpec
restack_moe_experts = _restack.restack_moe_experts

# -------------------------------------------------------------------

NUM_EXPERTS = 4
A, B = 8, 16


def _make_ts(name: str, tensor: torch.Tensor) -> TensorSource:
    return TensorSource(
        name=name,
        torch_dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        source=tensor,
    )


def _make_config(num_local_experts: int = NUM_EXPERTS):
    return SimpleNamespace(num_local_experts=num_local_experts)


# GPT-OSS specs (transpose on float weight)
GATE_UP_SPEC = ExpertRestackSpec(
    unstacked_template="gate_up.{expert}",
    stacked_name="gate_up_proj",
    stacked_suffix_overrides={"weight": "", "bias": "_bias"},
)
DOWN_SPEC = ExpertRestackSpec(
    unstacked_template="down.{expert}",
    stacked_name="down_proj",
    stacked_suffix_overrides={"weight": "", "bias": "_bias"},
)

# Llama4 specs (no transpose on float weight)
LLAMA4_GATE_UP_SPEC = ExpertRestackSpec(
    unstacked_template="gate_up.{expert}",
    stacked_name="gate_up_proj",
    transpose_suffixes=frozenset(),
    stacked_suffix_overrides={"weight": "", "bias": "_bias"},
)
LLAMA4_DOWN_SPEC = ExpertRestackSpec(
    unstacked_template="down.{expert}",
    stacked_name="down_proj",
    transpose_suffixes=frozenset(),
    stacked_suffix_overrides={"weight": "", "bias": "_bias"},
)


class TestGptqSuffixRestacking:
    """GPTQ suffixes (qweight, qzeros, scales) should be stacked
    without transpose."""

    def test_qweight_stacked(self):
        sd = {}
        prefix = "model.layers.0.mlp.experts"
        for i in range(NUM_EXPERTS):
            t = torch.randn(A, B)
            key = f"{prefix}.gate_up.{i}.qweight"
            sd[key] = _make_ts(key, t)

        result = restack_moe_experts(
            sd, [GATE_UP_SPEC], _make_config()
        )

        stacked_key = f"{prefix}.gate_up_proj.qweight"
        assert stacked_key in result
        assert result[stacked_key].shape == (NUM_EXPERTS, A, B)

        for i in range(NUM_EXPERTS):
            assert f"{prefix}.gate_up.{i}.qweight" not in result


class TestBiasSuffixOverride:
    """Bias suffix should map via stacked_suffix_overrides to
    bare name + _bias."""

    def test_bias_override_key(self):
        sd = {}
        prefix = "model.layers.0.mlp.experts"
        for i in range(NUM_EXPERTS):
            t = torch.randn(A)
            key = f"{prefix}.gate_up.{i}.bias"
            sd[key] = _make_ts(key, t)

        result = restack_moe_experts(
            sd, [GATE_UP_SPEC], _make_config()
        )

        stacked_key = f"{prefix}.gate_up_proj_bias"
        assert stacked_key in result
        assert result[stacked_key].shape == (NUM_EXPERTS, A)


class TestFloatWeightTranspose:
    """Float weight suffix should be stacked then transposed."""

    def test_weight_transposed(self):
        sd = {}
        prefix = "model.layers.0.mlp.experts"
        tensors = []
        for i in range(NUM_EXPERTS):
            t = torch.randn(A, B)
            tensors.append(t)
            key = f"{prefix}.gate_up.{i}.weight"
            sd[key] = _make_ts(key, t)

        result = restack_moe_experts(
            sd, [GATE_UP_SPEC], _make_config()
        )

        stacked_key = f"{prefix}.gate_up_proj"
        assert stacked_key in result
        stacked = result[stacked_key].source
        assert stacked.shape == (NUM_EXPERTS, B, A)

        for i in range(NUM_EXPERTS):
            expected = tensors[i].transpose(-1, -2)
            assert torch.equal(stacked[i], expected)


class TestNonExpertKeysPreserved:
    """Keys that don't match any spec pass through unchanged."""

    def test_passthrough(self):
        sd = {}
        passthrough_keys = [
            "model.layers.0.mlp.router.weight",
            "model.layers.0.input_layernorm.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        for key in passthrough_keys:
            t = torch.randn(4)
            sd[key] = _make_ts(key, t)

        result = restack_moe_experts(
            sd, [GATE_UP_SPEC, DOWN_SPEC], _make_config()
        )

        assert set(result.keys()) == set(passthrough_keys)
        for key in passthrough_keys:
            assert torch.equal(
                result[key].source, sd[key].source
            )


class TestMultiLayerMultiSpec:
    """Multiple layers x both gate_up and down specs."""

    def test_two_layers_two_specs(self):
        sd = {}
        num_layers = 2

        for layer in range(num_layers):
            prefix = f"model.layers.{layer}.mlp.experts"
            for i in range(NUM_EXPERTS):
                gu = torch.randn(A, B)
                key_gu = f"{prefix}.gate_up.{i}.qweight"
                sd[key_gu] = _make_ts(key_gu, gu)

                d = torch.randn(B, A)
                key_d = f"{prefix}.down.{i}.qweight"
                sd[key_d] = _make_ts(key_d, d)

        norm_key = "model.norm.weight"
        sd[norm_key] = _make_ts(norm_key, torch.randn(4))

        result = restack_moe_experts(
            sd, [GATE_UP_SPEC, DOWN_SPEC], _make_config()
        )

        for layer in range(num_layers):
            prefix = f"model.layers.{layer}.mlp.experts"
            assert f"{prefix}.gate_up_proj.qweight" in result
            assert result[
                f"{prefix}.gate_up_proj.qweight"
            ].shape == (NUM_EXPERTS, A, B)
            assert f"{prefix}.down_proj.qweight" in result
            assert result[
                f"{prefix}.down_proj.qweight"
            ].shape == (NUM_EXPERTS, B, A)

            for i in range(NUM_EXPERTS):
                assert (
                    f"{prefix}.gate_up.{i}.qweight"
                    not in result
                )
                assert (
                    f"{prefix}.down.{i}.qweight"
                    not in result
                )

        assert norm_key in result


class TestLlama4NoTranspose:
    """Llama4 stacks float weights without transpose."""

    def test_weight_not_transposed(self):
        sd = {}
        prefix = (
            "language_model.model.layers.0"
            ".feed_forward.experts"
        )
        tensors = []
        for i in range(NUM_EXPERTS):
            t = torch.randn(A, B)
            tensors.append(t)
            key = f"{prefix}.gate_up.{i}.weight"
            sd[key] = _make_ts(key, t)

        result = restack_moe_experts(
            sd,
            [LLAMA4_GATE_UP_SPEC],
            _make_config(),
        )

        stacked_key = f"{prefix}.gate_up_proj"
        assert stacked_key in result
        stacked = result[stacked_key].source
        # No transpose: shape stays [E, A, B]
        assert stacked.shape == (NUM_EXPERTS, A, B)

        for i in range(NUM_EXPERTS):
            assert torch.equal(stacked[i], tensors[i])

    def test_qweight_stacked(self):
        sd = {}
        prefix = (
            "language_model.model.layers.0"
            ".feed_forward.experts"
        )
        for i in range(NUM_EXPERTS):
            t = torch.randn(A, B)
            for suffix in ("qweight", "qzeros", "scales"):
                key = f"{prefix}.gate_up.{i}.{suffix}"
                sd[key] = _make_ts(key, t)

            t_d = torch.randn(B, A)
            for suffix in ("qweight", "qzeros", "scales"):
                key = f"{prefix}.down.{i}.{suffix}"
                sd[key] = _make_ts(key, t_d)

        specs = [LLAMA4_GATE_UP_SPEC, LLAMA4_DOWN_SPEC]
        result = restack_moe_experts(
            sd, specs, _make_config()
        )

        for group, shape in [
            ("gate_up_proj", (NUM_EXPERTS, A, B)),
            ("down_proj", (NUM_EXPERTS, B, A)),
        ]:
            for suffix in ("qweight", "qzeros", "scales"):
                key = f"{prefix}.{group}.{suffix}"
                assert key in result
                assert result[key].shape == shape

        # Per-expert keys removed
        for i in range(NUM_EXPERTS):
            for grp in ("gate_up", "down"):
                for suffix in ("qweight", "qzeros", "scales"):
                    assert (
                        f"{prefix}.{grp}.{i}.{suffix}"
                        not in result
                    )

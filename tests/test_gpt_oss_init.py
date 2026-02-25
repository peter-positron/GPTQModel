"""Tests for GPT-OSS meta device init and corruption sync."""

import torch
from torch import nn

from gptqmodel.models.definitions.gpt_oss import (
    GPT_OSS_CORRUPTION_SYNC_SUPPORTED,
    GPT_OSS_META_DEVICE_INIT_SUPPORTED,
    GptOssExpertsNew,
    GptOssTopKRouterNew,
    _sync_non_quantized_from_turtle,
)


class _MockConfig:
    num_local_experts = 4
    hidden_size = 64
    intermediate_size = 128
    num_experts_per_tok = 2
    dtype = torch.float32


def _make_fake_ori_experts(config, device="cpu"):
    """Build a minimal ori_experts with gate_up_proj / down_proj.

    Original GptOssExperts stores weights as:
      gate_up_proj: [num_experts, hidden_size, 2 * intermediate_size]
      down_proj:    [num_experts, intermediate_size, hidden_size]
    The init code does .t().contiguous() to get Linear weight shape.
    """

    class _FakeOri(nn.Module):
        pass

    ori = _FakeOri()
    ne = config.num_local_experts
    h = config.hidden_size
    d = config.intermediate_size

    # [ne, h, 2*d] -- transposed by init to [2*d, h] for Linear
    ori.gate_up_proj = nn.Parameter(
        torch.randn(ne, h, 2 * d, device=device)
    )
    ori.gate_up_proj_bias = nn.Parameter(
        torch.randn(ne, 2 * d, device=device)
    )
    # [ne, d, h] -- transposed by init to [h, d] for Linear
    ori.down_proj = nn.Parameter(
        torch.randn(ne, d, h, device=device)
    )
    ori.down_proj_bias = nn.Parameter(
        torch.randn(ne, h, device=device)
    )
    return ori


def _make_fake_ori_router(config, device="cpu"):
    """Build a minimal ori_router with weight and bias."""
    router = nn.Module()
    ne = config.num_local_experts
    h = config.hidden_size
    router.weight = nn.Parameter(
        torch.randn(ne, h, device=device)
    )
    router.bias = nn.Parameter(
        torch.randn(ne, device=device)
    )
    return router


def _make_sync_model_pair(param_name, shell_data, turtle_data):
    """Build shell/turtle module trees with dotted param names.

    Splits 'model.embed_tokens.weight' into nested modules so
    named_parameters() produces the correct dotted key.
    """
    parts = param_name.rsplit(".", 1)
    if len(parts) == 2:
        prefix, leaf = parts
    else:
        prefix, leaf = "", parts[0]

    def _build(prefix_str, data):
        # Build nested module hierarchy from dotted prefix
        root = nn.Module()
        if not prefix_str:
            root.register_parameter(leaf, nn.Parameter(data))
            return root
        segments = prefix_str.split(".")
        current = root
        for seg in segments[:-1]:
            child = nn.Module()
            current.add_module(seg, child)
            current = child
        # Last segment holds the parameter
        last = nn.Module()
        last.register_parameter(leaf, nn.Parameter(data))
        current.add_module(segments[-1], last)
        return root

    shell = _build(prefix, shell_data)
    turtle = _build(prefix, turtle_data)
    return shell, turtle


class TestExpertsMetaDevice:
    def test_meta_init_produces_meta_params(self):
        cfg = _MockConfig()
        ori = _make_fake_ori_experts(cfg, device="meta")
        experts = GptOssExpertsNew(cfg, ori_experts=ori)

        for i in range(cfg.num_local_experts):
            assert experts.gate_up[i].weight.device.type == "meta"
            assert experts.gate_up[i].bias.device.type == "meta"
            assert experts.down[i].weight.device.type == "meta"
            assert experts.down[i].bias.device.type == "meta"

    def test_meta_init_params_still_registered(self):
        cfg = _MockConfig()
        ori = _make_fake_ori_experts(cfg, device="meta")
        experts = GptOssExpertsNew(cfg, ori_experts=ori)

        param_names = {n for n, _ in experts.named_parameters()}
        for i in range(cfg.num_local_experts):
            assert f"gate_up.{i}.weight" in param_names
            assert f"gate_up.{i}.bias" in param_names
            assert f"down.{i}.weight" in param_names
            assert f"down.{i}.bias" in param_names

    def test_cpu_init_copies_correctly(self):
        cfg = _MockConfig()
        ori = _make_fake_ori_experts(cfg, device="cpu")
        experts = GptOssExpertsNew(cfg, ori_experts=ori)

        assert experts.gate_up[0].weight.device.type == "cpu"
        assert experts.gate_up[0].weight.std() > 0


class TestRouterMetaDevice:
    def test_meta_init_produces_meta_params(self):
        cfg = _MockConfig()
        ori = _make_fake_ori_router(cfg, device="meta")
        router = GptOssTopKRouterNew(cfg, ori_router=ori)

        assert router.weight.device.type == "meta"
        assert router.bias.device.type == "meta"

    def test_meta_init_params_still_registered(self):
        cfg = _MockConfig()
        ori = _make_fake_ori_router(cfg, device="meta")
        router = GptOssTopKRouterNew(cfg, ori_router=ori)

        param_names = {n for n, _ in router.named_parameters()}
        assert "weight" in param_names
        assert "bias" in param_names

    def test_cpu_init_copies_correctly(self):
        cfg = _MockConfig()
        ori = _make_fake_ori_router(cfg, device="cpu")
        router = GptOssTopKRouterNew(cfg, ori_router=ori)

        assert router.weight.device.type == "cpu"
        assert torch.allclose(router.weight, ori.weight)


class TestCorruptionSync:
    def test_syncs_all_zeros(self):
        shell, turtle = _make_sync_model_pair(
            "model.embed_tokens.weight",
            torch.zeros(16, 64),
            torch.randn(16, 64),
        )

        synced = _sync_non_quantized_from_turtle(shell, turtle)
        assert len(synced) == 1
        assert "model.embed_tokens.weight" in synced
        # Verify data was actually synced
        shell_p = dict(shell.named_parameters())[
            "model.embed_tokens.weight"
        ]
        turtle_p = dict(turtle.named_parameters())[
            "model.embed_tokens.weight"
        ]
        assert torch.allclose(shell_p, turtle_p)

    def test_syncs_near_zero_std(self):
        shell, turtle = _make_sync_model_pair(
            "model.norm.weight",
            torch.full((64,), 1e-5),
            torch.randn(64),
        )

        synced = _sync_non_quantized_from_turtle(shell, turtle)
        assert "model.norm.weight" in synced

    def test_skips_non_matching_patterns(self):
        shell, turtle = _make_sync_model_pair(
            "model.layers.0.mlp.experts.gate_up.0.weight",
            torch.zeros(64, 64),
            torch.randn(64, 64),
        )

        synced = _sync_non_quantized_from_turtle(shell, turtle)
        assert len(synced) == 0


class TestSentinels:
    def test_meta_device_init_sentinel(self):
        assert GPT_OSS_META_DEVICE_INIT_SUPPORTED is True

    def test_corruption_sync_sentinel(self):
        assert GPT_OSS_CORRUPTION_SYNC_SUPPORTED is True

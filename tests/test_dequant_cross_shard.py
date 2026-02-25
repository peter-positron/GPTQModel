# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-shard GPTQ dequantization.

Verifies that _build_gptq_cross_shard_plan() and _execute_gptq_dequant()
correctly handle GPTQ buffers split across multiple safetensors shards.

Test strategy:
- Plan builder tests use synthetic safetensors (key-only, tensor values irrelevant).
- Dequant tests use convert_gptq_file() as oracle for equivalence checks and
  hand-computed golden values for absolute correctness.
- No self-written pack helpers -- synthetic GPTQ tensors use known int32 bit
  patterns whose unpacked values are verified independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.model_dequant import (
    GptqShardPlan,
    _build_gptq_cross_shard_plan,
    _execute_gptq_dequant,
    convert_gptq_file,
    dequantize_model,
    unpack_cols,
    unpack_rows,
)


# -- Constants ---------------------------------------------------------------

_PREFIX = "model.layers.0.self_attn.q_proj"
_EMBED_KEY = "model.embed_tokens.weight"
_BITS = 4
_PACK_FACTOR = 32 // _BITS  # 8


# -- Helpers -----------------------------------------------------------------


def _make_gptq_tensors() -> dict[str, torch.Tensor]:
    """Synthetic GPTQ buffers with known values.

    With V2 format (no zero correction), scales=1, zeros=0:
      dequant weight = unpack_rows(qweight).T  (as float)

    qweight column j has all 4-bit slots set to j, so the dequantized
    weight has row j = [j, j, ..., j].
    """
    # qweight (1, 8): column j packs eight 4-bit slots each = j.
    # j * 0x11111111 puts j into every 4-bit nibble of int32.
    qweight = torch.tensor(
        [[j * 0x11111111 for j in range(_PACK_FACTOR)]],
        dtype=torch.int32,
    )
    # qzeros (1, 1): one group, one packed column, all zero.
    qzeros = torch.zeros((1, 1), dtype=torch.int32)
    # scales (1, 8): one group, all ones.
    scales = torch.ones((1, _PACK_FACTOR), dtype=torch.float16)
    # g_idx (8,): all rows in group 0.
    g_idx = torch.zeros(_PACK_FACTOR, dtype=torch.long)
    # Passthrough embedding.
    embed = torch.randn(4, 8, dtype=torch.float16)

    return {
        f"{_PREFIX}.qweight": qweight,
        f"{_PREFIX}.qzeros": qzeros,
        f"{_PREFIX}.scales": scales,
        f"{_PREFIX}.g_idx": g_idx,
        _EMBED_KEY: embed,
    }


def _write_config(
    model_dir: Path,
    *,
    checkpoint_format: str = "gptq_v2",
    bits: int = _BITS,
) -> None:
    config = {
        "quantization_config": {
            "quant_method": "gptq",
            "bits": bits,
            "checkpoint_format": checkpoint_format,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_index(
    model_dir: Path,
    weight_map: dict[str, str],
) -> None:
    index = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index)
    )


def _expected_weight(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Golden dequantized weight for V2 format, scales=1, zeros=0.

    The transposed output has shape (8, 8) where row j = [j, j, ..., j].
    """
    rows = torch.arange(_PACK_FACTOR, dtype=dtype)
    return rows.unsqueeze(1).expand(_PACK_FACTOR, _PACK_FACTOR).contiguous()


def _split_cross_shard(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split tensors: g_idx + embed -> shard2, rest -> shard1."""
    shard1 = {
        k: v
        for k, v in tensors.items()
        if not k.endswith(".g_idx") and k != _EMBED_KEY
    }
    shard2 = {
        k: v
        for k, v in tensors.items()
        if k.endswith(".g_idx") or k == _EMBED_KEY
    }
    return shard1, shard2


# -- Unpack sanity checks ---------------------------------------------------
#
# These verify the synthetic test tensors have expected properties,
# independent of the dequant pipeline.


class TestUnpackSanity:
    def test_unpack_rows_known_values(self) -> None:
        """unpack_rows on synthetic qweight gives expected integers."""
        qweight = torch.tensor(
            [[j * 0x11111111 for j in range(_PACK_FACTOR)]],
            dtype=torch.int32,
        )
        unpacked = unpack_rows(qweight, _BITS)

        assert unpacked.shape == (_PACK_FACTOR, _PACK_FACTOR)
        for k in range(_PACK_FACTOR):
            for j in range(_PACK_FACTOR):
                assert unpacked[k, j].item() == j

    def test_unpack_cols_zeros(self) -> None:
        """unpack_cols on all-zero qzeros gives all zeros."""
        qzeros = torch.zeros((1, 1), dtype=torch.int32)
        unpacked = unpack_cols(qzeros, _BITS)

        assert unpacked.shape == (1, _PACK_FACTOR)
        assert (unpacked == 0).all()


# -- Plan builder tests ------------------------------------------------------


class TestBuildPlan:
    def test_single_shard(self, tmp_path: Path) -> None:
        tensors = _make_gptq_tensors()
        filename = "model.safetensors"
        save_file(tensors, str(tmp_path / filename))

        plan = _build_gptq_cross_shard_plan(tmp_path, [filename])

        assert _PREFIX in plan.module_buffers
        for buf in ("qweight", "qzeros", "scales", "g_idx"):
            assert buf in plan.module_buffers[_PREFIX]
            assert plan.module_buffers[_PREFIX][buf][0] == filename
        assert _EMBED_KEY in plan.passthrough
        assert plan.passthrough[_EMBED_KEY] == filename

    def test_cross_shard(self, tmp_path: Path) -> None:
        tensors = _make_gptq_tensors()
        s1 = "model-00001-of-00002.safetensors"
        s2 = "model-00002-of-00002.safetensors"
        shard1, shard2 = _split_cross_shard(tensors)
        save_file(shard1, str(tmp_path / s1))
        save_file(shard2, str(tmp_path / s2))

        plan = _build_gptq_cross_shard_plan(tmp_path, [s1, s2])

        bufs = plan.module_buffers[_PREFIX]
        assert bufs["qweight"][0] == s1
        assert bufs["qzeros"][0] == s1
        assert bufs["scales"][0] == s1
        assert bufs["g_idx"][0] == s2
        assert plan.passthrough[_EMBED_KEY] == s2

    def test_missing_buffer_raises(self, tmp_path: Path) -> None:
        tensors = _make_gptq_tensors()
        del tensors[f"{_PREFIX}.g_idx"]
        filename = "model.safetensors"
        save_file(tensors, str(tmp_path / filename))

        with pytest.raises(KeyError, match="Incomplete GPTQ buffers"):
            _build_gptq_cross_shard_plan(tmp_path, [filename])


# -- Dequant equivalence tests ----------------------------------------------


class TestExecuteDequant:
    def test_matches_convert_gptq_file(self, tmp_path: Path) -> None:
        """Cross-shard path on a single shard must match convert_gptq_file."""
        tensors = _make_gptq_tensors()
        filename = "model.safetensors"
        save_file(tensors, str(tmp_path / filename))
        _write_config(tmp_path, checkpoint_format="gptq_v2")

        quant_cfg = json.loads((tmp_path / "config.json").read_text())[
            "quantization_config"
        ]

        # Oracle: existing single-shard function.
        expected = convert_gptq_file(
            tmp_path / filename, torch.bfloat16, quant_cfg, "cpu",
        )

        # Under test: cross-shard pipeline.
        plan = _build_gptq_cross_shard_plan(tmp_path, [filename])
        result = _execute_gptq_dequant(
            plan, tmp_path, quant_cfg, torch.bfloat16, "cpu",
            setup_logger(),
        )

        actual = result[filename]
        for key in expected:
            assert key in actual, f"Missing key {key}"
            assert torch.equal(actual[key], expected[key]), (
                f"Mismatch on {key}"
            )

    def test_cross_shard_matches_single(self, tmp_path: Path) -> None:
        """Splitting g_idx into shard 2 must produce identical weights."""
        tensors = _make_gptq_tensors()
        quant_cfg = {
            "quant_method": "gptq",
            "bits": _BITS,
            "checkpoint_format": "gptq_v2",
        }

        # Oracle: single-shard convert_gptq_file.
        single_dir = tmp_path / "single"
        single_dir.mkdir()
        save_file(tensors, str(single_dir / "model.safetensors"))
        expected = convert_gptq_file(
            single_dir / "model.safetensors",
            torch.bfloat16,
            quant_cfg,
            "cpu",
        )

        # Under test: cross-shard layout.
        cross_dir = tmp_path / "cross"
        cross_dir.mkdir()
        s1 = "model-00001-of-00002.safetensors"
        s2 = "model-00002-of-00002.safetensors"
        shard1, shard2 = _split_cross_shard(tensors)
        save_file(shard1, str(cross_dir / s1))
        save_file(shard2, str(cross_dir / s2))

        plan = _build_gptq_cross_shard_plan(cross_dir, [s1, s2])
        result = _execute_gptq_dequant(
            plan, cross_dir, quant_cfg, torch.bfloat16, "cpu",
            setup_logger(),
        )

        weight_key = f"{_PREFIX}.weight"
        actual = result[s1][weight_key]
        assert torch.equal(actual, expected[weight_key])

    def test_golden_values(self, tmp_path: Path) -> None:
        """Dequantized weight must match hand-computed golden values."""
        tensors = _make_gptq_tensors()
        filename = "model.safetensors"
        save_file(tensors, str(tmp_path / filename))
        quant_cfg = {
            "quant_method": "gptq",
            "bits": _BITS,
            "checkpoint_format": "gptq_v2",
        }

        plan = _build_gptq_cross_shard_plan(tmp_path, [filename])
        result = _execute_gptq_dequant(
            plan, tmp_path, quant_cfg, torch.bfloat16, "cpu",
            setup_logger(),
        )

        actual = result[filename][f"{_PREFIX}.weight"]
        expected = _expected_weight(torch.bfloat16)
        assert torch.equal(actual, expected)

    def test_passthrough_stays_in_original_shard(
        self, tmp_path: Path,
    ) -> None:
        """Passthrough tensors must remain in their original shard."""
        tensors = _make_gptq_tensors()
        s1 = "model-00001-of-00002.safetensors"
        s2 = "model-00002-of-00002.safetensors"
        shard1, shard2 = _split_cross_shard(tensors)
        save_file(shard1, str(tmp_path / s1))
        save_file(shard2, str(tmp_path / s2))
        quant_cfg = {
            "quant_method": "gptq",
            "bits": _BITS,
            "checkpoint_format": "gptq_v2",
        }

        plan = _build_gptq_cross_shard_plan(tmp_path, [s1, s2])
        result = _execute_gptq_dequant(
            plan, tmp_path, quant_cfg, torch.bfloat16, "cpu",
            setup_logger(),
        )

        # Embed was in shard2, must stay there.
        assert _EMBED_KEY not in result.get(s1, {})
        assert _EMBED_KEY in result[s2]

    def test_v1_format_applies_correction(self, tmp_path: Path) -> None:
        """V1 checkpoint format must apply qzeros correction."""
        tensors = _make_gptq_tensors()
        filename = "model.safetensors"
        save_file(tensors, str(tmp_path / filename))

        plan = _build_gptq_cross_shard_plan(tmp_path, [filename])

        # V1 format (applies zero-point correction).
        quant_cfg_v1 = {
            "quant_method": "gptq",
            "bits": _BITS,
            "checkpoint_format": "gptq",
        }
        result_v1 = _execute_gptq_dequant(
            plan, tmp_path, quant_cfg_v1, torch.bfloat16, "cpu",
            setup_logger(),
        )

        # V2 format (no correction).
        quant_cfg_v2 = {
            "quant_method": "gptq",
            "bits": _BITS,
            "checkpoint_format": "gptq_v2",
        }
        result_v2 = _execute_gptq_dequant(
            plan, tmp_path, quant_cfg_v2, torch.bfloat16, "cpu",
            setup_logger(),
        )

        weight_key = f"{_PREFIX}.weight"
        weight_v1 = result_v1[filename][weight_key]
        weight_v2 = result_v2[filename][weight_key]

        # V1 correction changes zeros, so outputs must differ.
        assert not torch.equal(weight_v1, weight_v2), (
            "V1 and V2 produced identical weights -- "
            "V1 correction was not applied"
        )

        # V1 cross-shard must match V1 single-shard oracle.
        expected = convert_gptq_file(
            tmp_path / filename, torch.bfloat16, quant_cfg_v1, "cpu",
        )
        assert torch.equal(weight_v1, expected[weight_key])


# -- End-to-end dequantize_model test ----------------------------------------


class TestDequantizeModelCrossShard:
    def test_end_to_end(self, tmp_path: Path) -> None:
        """Full dequantize_model with cross-shard GPTQ buffers."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        tensors = _make_gptq_tensors()
        s1 = "model-00001-of-00002.safetensors"
        s2 = "model-00002-of-00002.safetensors"
        shard1, shard2 = _split_cross_shard(tensors)
        save_file(shard1, str(model_dir / s1))
        save_file(shard2, str(model_dir / s2))

        weight_map = {}
        for k in shard1:
            weight_map[k] = s1
        for k in shard2:
            weight_map[k] = s2
        _write_index(model_dir, weight_map)
        _write_config(model_dir, checkpoint_format="gptq_v2")

        dequantize_model(model_dir, output_dir)

        # Verify output index.
        out_index = json.loads(
            (output_dir / "model.safetensors.index.json").read_text()
        )
        weight_key = f"{_PREFIX}.weight"
        assert weight_key in out_index["weight_map"]

        # Verify dequantized weight matches golden value.
        weight_file = out_index["weight_map"][weight_key]
        out_tensors = load_file(str(output_dir / weight_file))
        actual = out_tensors[weight_key]
        expected = _expected_weight(torch.bfloat16)
        assert torch.equal(actual, expected)

        # Verify embed passthrough is present.
        assert _EMBED_KEY in out_index["weight_map"]
        embed_file = out_index["weight_map"][_EMBED_KEY]
        embed_out = load_file(str(output_dir / embed_file))
        assert _EMBED_KEY in embed_out

        # Config should have quantization_config removed.
        out_config = json.loads(
            (output_dir / "config.json").read_text()
        )
        assert "quantization_config" not in out_config

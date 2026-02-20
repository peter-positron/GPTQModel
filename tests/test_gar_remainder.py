# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression test for GAR (Group Aware Reordering) with non-divisible
column counts.

When columns % group_size != 0, the GAR permutation must preserve
remainder columns instead of silently dropping them.  This was the
root cause of the GPT-OSS quantization crash (hidden_size=2880,
group_size=128 -> 2880 % 128 = 64 remainder columns dropped).
"""

from __future__ import annotations

import torch

from gptqmodel.quantization.gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
)


def _build_gar_perm(
    columns: int, group_size: int
) -> torch.Tensor:
    """Reproduce the GAR permutation path from gptq.py quantize()."""
    diag_h = torch.randn(columns).abs()
    local_perms, local_values = compute_local_perms(
        diag_h, group_size, return_values=True
    )
    global_perm = compute_global_perm(
        diag_h, group_size, precomputed_values=local_values
    )
    perm = compose_final_perm(local_perms, global_perm, group_size)
    # Append remainder indices (mirrors the fix in gptq.py)
    remainder_start = len(perm)
    if remainder_start < columns:
        remainder = torch.arange(
            remainder_start, columns,
            dtype=perm.dtype, device=perm.device,
        )
        perm = torch.cat([perm, remainder])
    return perm


class TestGarDivisible:
    """Group size divides columns evenly — baseline sanity check."""

    def test_perm_length_matches_columns(self) -> None:
        perm = _build_gar_perm(columns=2560, group_size=128)
        assert len(perm) == 2560

    def test_perm_is_valid_permutation(self) -> None:
        perm = _build_gar_perm(columns=2560, group_size=128)
        assert set(perm.tolist()) == set(range(2560))


class TestGarRemainder:
    """Columns not divisible by group_size — the bug case."""

    def test_perm_length_2880_gs128(self) -> None:
        """GPT-OSS: 2880 cols, group_size=128 -> 64 remainder."""
        perm = _build_gar_perm(columns=2880, group_size=128)
        assert len(perm) == 2880, (
            f"expected 2880, got {len(perm)} "
            f"(dropped {2880 - len(perm)} remainder columns)"
        )

    def test_perm_preserves_all_indices(self) -> None:
        perm = _build_gar_perm(columns=2880, group_size=128)
        assert set(perm.tolist()) == set(range(2880))

    def test_weight_roundtrip(self) -> None:
        """Apply permutation then inverse — weight must be unchanged."""
        columns = 2880
        W = torch.randn(64, columns)
        perm = _build_gar_perm(columns=columns, group_size=128)
        W_perm = W[:, perm]
        assert W_perm.shape == W.shape
        inv_perm = torch.argsort(perm)
        W_restored = W_perm[:, inv_perm]
        assert torch.allclose(W, W_restored)

    def test_small_remainder(self) -> None:
        """17 cols, group_size=5 -> 2 remainder."""
        perm = _build_gar_perm(columns=17, group_size=5)
        assert len(perm) == 17
        assert set(perm.tolist()) == set(range(17))

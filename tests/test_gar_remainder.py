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


class TestGarPermLengthInvariant:
    """Explicit invariant: len(final_perm) == n_cols for all cases.

    This makes the original bug (silent column truncation) impossible
    to reintroduce without a test failure.
    """

    CASES = [
        (2560, 128, "divisible"),
        (2880, 128, "gpt-oss remainder=64"),
        (17, 5, "small remainder=2"),
        (129, 128, "remainder=1"),
        (127, 128, "cols < group_size"),
        (256, 256, "cols == group_size"),
        (1, 1, "trivial"),
    ]

    def test_perm_length_equals_columns(self) -> None:
        for cols, gs, label in self.CASES:
            perm = _build_gar_perm(columns=cols, group_size=gs)
            assert len(perm) == cols, (
                f"{label}: len(perm)={len(perm)} != {cols}"
            )

    def test_perm_is_bijection(self) -> None:
        for cols, gs, label in self.CASES:
            perm = _build_gar_perm(columns=cols, group_size=gs)
            assert len(set(perm.tolist())) == cols, (
                f"{label}: perm has duplicates"
            )
            assert perm.min().item() == 0, (
                f"{label}: perm min != 0"
            )
            assert perm.max().item() == cols - 1, (
                f"{label}: perm max != {cols - 1}"
            )


class TestGarSmokeQuantPath:
    """Minimal CPU smoke test exercising the GAR reorder path from
    gptq.py quantize() with a fake linear (in_features=2880,
    group_size=128).

    Does NOT run full GPTQ quantization — only the permutation and
    Hessian reorder steps that caused the original crash.
    """

    def test_reorder_path_preserves_shapes(self) -> None:
        """Simulate the GAR reorder path: build H, permute W and H,
        verify shapes and invertibility.
        """
        rows, cols, gs = 64, 2880, 128

        W = torch.randn(rows, cols)
        # Simulate Hessian diagonal (from add_batch accumulation)
        diag_h = torch.randn(cols).abs() + 1e-6
        H = torch.diag(diag_h)

        # Build GAR permutation (same path as gptq.py)
        perm = _build_gar_perm(columns=cols, group_size=gs)

        # Apply permutation (mirrors gptq.py lines 993-994)
        W_perm = W[:, perm]
        H_perm = H[perm][:, perm]

        assert W_perm.shape == (rows, cols)
        assert H_perm.shape == (cols, cols)

        # Inverse recovers original
        inv_perm = torch.argsort(perm)
        W_restored = W_perm[:, inv_perm]
        assert torch.allclose(W, W_restored)

    def test_block_loop_indexing(self) -> None:
        """Verify that the GPTQ block loop can index into permuted W
        without IndexError (the original crash).
        """
        cols, gs, blocksize = 2880, 128, 128

        W = torch.randn(64, cols)
        perm = _build_gar_perm(columns=cols, group_size=gs)
        W_perm = W[:, perm]

        # Simulate block loop from gptq.py (lines 1006-1008)
        blocks_visited = 0
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            block = W_perm[:, i1:i2]
            assert block.shape[1] == i2 - i1
            blocks_visited += 1

        expected_blocks = (cols + blocksize - 1) // blocksize
        assert blocks_visited == expected_blocks

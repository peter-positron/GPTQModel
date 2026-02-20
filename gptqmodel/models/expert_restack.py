# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Set, Tuple

import torch

from ..utils.model import OffloadTensorRef, TensorSource

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass(frozen=True)
class ExpertRestackSpec:
    """Defines the key mapping for restacking one expert tensor group.

    unstacked_template: fragment pattern with {expert} placeholder
        e.g. "gate_up.{expert}"
    stacked_name: replacement fragment e.g. "gate_up_proj"
    num_experts_key: config attribute for expert count
    stack_axis: dimension for torch.stack
    transpose_suffixes: suffixes needing .transpose(-1, -2) after
        stacking (only applies to float weights, not GPTQ suffixes)
    stacked_suffix_overrides: maps suffix to custom stacked key
        ending, e.g. {"weight": "", "bias": "_bias"}
    """

    unstacked_template: str
    stacked_name: str
    num_experts_key: str = "num_local_experts"
    stack_axis: int = 0
    transpose_suffixes: FrozenSet[str] = frozenset({"weight"})
    stacked_suffix_overrides: Dict[str, str] = field(
        default_factory=dict,
    )


def restack_moe_experts(
    state_dict: Dict[str, TensorSource],
    specs: List[ExpertRestackSpec],
    config: PretrainedConfig,
) -> Dict[str, TensorSource]:
    """Restack per-expert keys back into HF's stacked format.

    Scans state_dict for keys matching each spec's unstacked_template,
    groups them, stacks the tensors, and replaces the per-expert
    entries with a single stacked TensorSource entry.

    Non-matching keys pass through unchanged.
    """
    keys_to_remove: Set[str] = set()
    entries_to_add: Dict[str, TensorSource] = {}

    for spec in specs:
        num_experts = getattr(config, spec.num_experts_key)

        # Build regex: "gate_up.{expert}" -> "gate_up\.(\d+)"
        escaped = re.escape(spec.unstacked_template)
        pattern_frag = escaped.replace(re.escape("{expert}"), r"(\d+)")

        # Full key pattern: <prefix>.<pattern_frag>.<suffix>
        full_pattern = re.compile(
            r"^(.+)\." + pattern_frag + r"\.([^.]+)$"
        )

        # Group matched keys by (prefix, suffix)
        groups: Dict[
            Tuple[str, str], Dict[int, TensorSource]
        ] = defaultdict(dict)

        for key, ts in state_dict.items():
            m = full_pattern.match(key)
            if m is None:
                continue
            prefix = m.group(1)
            expert_idx = int(m.group(2))
            suffix = m.group(3)
            groups[(prefix, suffix)][expert_idx] = ts

        for (prefix, suffix), expert_map in groups.items():
            if len(expert_map) != num_experts:
                continue

            # Collect tensors in expert order
            tensors: List[torch.Tensor] = []
            for i in range(num_experts):
                ts = expert_map[i]
                if isinstance(ts.source, OffloadTensorRef):
                    raise NotImplementedError(
                        "Expert restacking does not yet support "
                        "disk-offloaded tensors (OffloadTensorRef). "
                        "Quantize with offload_to_disk=False to use "
                        "this feature."
                    )
                tensors.append(ts.source.detach())

            stacked = torch.stack(tensors, dim=spec.stack_axis)

            if suffix in spec.transpose_suffixes:
                stacked = stacked.transpose(-1, -2).contiguous()

            # Compute stacked key name
            if suffix in spec.stacked_suffix_overrides:
                suffix_str = spec.stacked_suffix_overrides[suffix]
                stacked_key = f"{prefix}.{spec.stacked_name}{suffix_str}"
            else:
                stacked_key = (
                    f"{prefix}.{spec.stacked_name}.{suffix}"
                )

            new_ts = TensorSource(
                name=stacked_key,
                torch_dtype=stacked.dtype,
                shape=tuple(stacked.shape),
                source=stacked,
            )
            entries_to_add[stacked_key] = new_ts

            # Mark per-expert keys for removal
            for i in range(num_experts):
                orig_key = (
                    f"{prefix}."
                    f"{spec.unstacked_template.replace('{expert}', str(i))}"
                    f".{suffix}"
                )
                keys_to_remove.add(orig_key)

    # Build result: preserve order, remove old, append new
    result: Dict[str, TensorSource] = {}
    for key, ts in state_dict.items():
        if key not in keys_to_remove:
            result[key] = ts
    result.update(entries_to_add)
    return result

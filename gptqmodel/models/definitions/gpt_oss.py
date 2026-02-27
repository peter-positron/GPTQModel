# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseQModel
from ..expert_restack import ExpertRestackSpec
from ...utils.logger import setup_logger

log = setup_logger()


class GptOssExpertsNew(nn.Module):
    def __init__(self, config, ori_experts=None):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0
        self.quantizing = False

        self.gate_up = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.expert_dim, dtype=config.dtype)
            for _ in range(self.num_experts)
        ])

        self.down = nn.ModuleList([
            nn.Linear(self.expert_dim, self.hidden_size, dtype=config.dtype)
            for _ in range(self.num_experts)
        ])

        if ori_experts is not None:
            self.quantizing = True
            # Detect if source is already a GptOssExpertsNew (has
            # gate_up ModuleList) vs original GptOssExperts (has
            # gate_up_proj stacked Parameter).
            is_converted = (
                hasattr(ori_experts, 'gate_up')
                and isinstance(ori_experts.gate_up, nn.ModuleList)
            )

            for i in range(self.num_experts):
                if is_converted:
                    gu_w_src = ori_experts.gate_up[i].weight.detach()
                    gu_b_src = ori_experts.gate_up[i].bias.detach()
                    d_w_src = ori_experts.down[i].weight.detach()
                    d_b_src = ori_experts.down[i].bias.detach()
                else:
                    gu_w_src = ori_experts.gate_up_proj[i].detach().t().contiguous()
                    gu_b_src = ori_experts.gate_up_proj_bias[i].detach()
                    d_w_src = ori_experts.down_proj[i].detach().t().contiguous()
                    d_b_src = ori_experts.down_proj_bias[i].detach()

                with torch.inference_mode():
                    self.gate_up[i].weight.copy_(gu_w_src)
                    self.gate_up[i].bias.copy_(gu_b_src)
                    self.down[i].weight.copy_(d_w_src)
                    self.down[i].bias.copy_(d_b_src)

        # Prevent transformers _init_weights from clobbering weights
        self._is_hf_initialized = True

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        if self.quantizing:
            # For quantization, we need to trigger computation of all experts
            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
            num_experts = routing_weights.shape[1]

            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = torch.stack([proj(hidden_states[i]) for i, proj in enumerate(self.gate_up)])
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.stack([proj((up[i] + 1) * glu[i]) for i, proj in enumerate(self.down)])
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)

            return next_states

        # For non-quantization forward pass, reduce forward pass time by only computing active experts
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1] if len(hidden_states.shape) > 2 else 1
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)

        active_experts = torch.unique(router_indices.flatten())
        final_output = torch.zeros_like(hidden_states)
        for expert_idx in active_experts:
            expert_mask = (router_indices == expert_idx).any(dim=-1)  # (num_tokens,)
            if not expert_mask.any():
                continue

            expert_tokens = hidden_states[expert_mask]  # (selected_tokens, hidden_size)

            gate_up_output = self.gate_up[expert_idx](expert_tokens)  # (selected_tokens, 2*expert_dim)
            gate, up = gate_up_output[..., ::2], gate_up_output[..., 1::2]

            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)

            expert_output = self.down[expert_idx]((up + 1) * glu)  # (selected_tokens, hidden_size)

            expert_weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)  # (selected_tokens, 1)

            final_output[expert_mask] += expert_output * expert_weights

        if seq_len > 1:
            final_output = final_output.view(batch_size, seq_len, self.hidden_size)
        else:
            final_output = final_output.view(batch_size, self.hidden_size)

        return final_output

class GptOssTopKRouterNew(nn.Module):
    def __init__(self, config, ori_router=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

        if ori_router is not None:
            with torch.inference_mode():
                self.weight.copy_(ori_router.weight.detach())
                self.bias.copy_(ori_router.bias.detach())

        # Prevent transformers _init_weights from clobbering weights
        self._is_hf_initialized = True

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight.to(hidden_states.dtype), self.bias.to(hidden_states.dtype))  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        # Return 3 values to match original GptOssTopKRouter interface
        # expected by GptOssMLP.forward(): (logits, scores, indices)
        return router_logits, router_scores, router_indices

class GPTOSSGPTQ(BaseQModel):
    # Disable shell/turtle disk offload — the default materialization
    # path does not handle GPT-OSS expert weight splitting correctly,
    # causing empty Hessians on attention modules during calibration.
    support_offload_to_disk = False

    dynamic_expert_index = 'num_local_experts'

    expert_restack_specs = [
        ExpertRestackSpec(
            unstacked_template="gate_up.{expert}",
            stacked_name="gate_up_proj",
            stacked_suffix_overrides={
                "weight": "",
                "bias": "_bias",
            },
        ),
        ExpertRestackSpec(
            unstacked_template="down.{expert}",
            stacked_name="down_proj",
            stacked_suffix_overrides={
                "weight": "",
                "bias": "_bias",
            },
        ),
    ]

    pre_lm_head_norm_module = 'model.norm'

    module_tree = [
        'model',
        'layers',
        '#',
        {
            'input_layernorm': ('input_layernorm:!',),
            'self_attn': ('q_proj:0', 'k_proj:0', 'v_proj:0', 'o_proj:1'),
            'post_attention_layernorm': ('post_attention_layernorm:!',),
            'mlp': {
                'experts': {
                    'gate_up': {'#': ('#')},
                    'down': {'#': ('#')},
                }
            },
        },
    ]

    # Stacked-key mapping: (stacked_name, suffix_overrides) per spec.
    # Used by _repair_stacked_experts to reverse the restack transform.
    _EXPERT_GROUPS = [
        ("gate_up_proj", "gate_up", {"weight": "", "bias": "_bias"}),
        ("down_proj", "down", {"weight": "", "bias": "_bias"}),
    ]

    # GPTQ buffer names that get stacked without suffix override
    _GPTQ_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.gpt_oss.modeling_gpt_oss as m

            m.GptOssExperts = GptOssExpertsNew
            m.GptOssTopKRouter = GptOssTopKRouterNew

    def after_model_load(self, model, load_quantized_model=False):
        if load_quantized_model:
            self._repair_stacked_experts_if_needed(model)
        return model

    def _repair_stacked_experts_if_needed(self, model):
        """Load stacked expert tensors into per-expert modules.

        The save path restacks per-expert keys into HF stacked
        format (gate_up.{i}.qweight -> gate_up_proj.qweight[E,...]).
        The load path creates per-expert TorchQuantLinear modules
        that accelerate cannot populate from stacked keys. This
        method detects the mismatch and backfills expert weights.
        """
        # Probe layer 0 expert 0 to decide if repair is needed
        layers = model.model.layers
        probe = layers[0].mlp.experts.gate_up[0]
        if not hasattr(probe, "qweight"):
            return  # not quantized (float model), nothing to do
        if probe.qweight.abs().max().item() > 0:
            return  # already loaded correctly (per-expert checkpoint)

        model_dir = self._resolve_checkpoint_dir(model)
        if model_dir is None:
            log.warning(
                "Expert repair: cannot locate checkpoint directory"
            )
            return

        tensor_map = self._build_shard_map(model_dir)
        if not tensor_map:
            log.warning(
                "Expert repair: no safetensors files found in %s",
                model_dir,
            )
            return

        num_experts = model.config.num_local_experts
        num_layers = model.config.num_hidden_layers
        repaired = 0

        for layer_idx in range(num_layers):
            experts = layers[layer_idx].mlp.experts
            prefix = f"model.layers.{layer_idx}.mlp.experts"

            for stacked_name, ml_attr, suf_map in self._EXPERT_GROUPS:
                module_list = getattr(experts, ml_attr)

                for suffix in self._GPTQ_SUFFIXES:
                    # Stacked key: prefix.gate_up_proj.qweight
                    stacked_key = f"{prefix}.{stacked_name}.{suffix}"
                    stacked = self._load_tensor(
                        stacked_key, tensor_map, model_dir,
                    )
                    if stacked is None:
                        continue
                    if stacked.dim() < 2:
                        continue
                    # Slice: [E, ...] -> per-expert 2D
                    for i in range(num_experts):
                        sliced = stacked[i].contiguous()
                        buf = getattr(module_list[i], suffix, None)
                        if buf is not None:
                            buf.data.copy_(sliced)
                            repaired += 1

                # Bias uses suffix override: gate_up_proj_bias
                bias_suffix = suf_map.get("bias")
                if bias_suffix is not None:
                    bias_key = f"{prefix}.{stacked_name}{bias_suffix}"
                    stacked_bias = self._load_tensor(
                        bias_key, tensor_map, model_dir,
                    )
                    if stacked_bias is not None and stacked_bias.dim() >= 1:
                        for i in range(num_experts):
                            mod = module_list[i]
                            if hasattr(mod, "bias") and mod.bias is not None:
                                mod.bias.data.copy_(stacked_bias[i])
                                repaired += 1

        if repaired > 0:
            self._apply_v1_to_v2_on_experts(model)
            log.info(
                "Expert repair: loaded %d stacked tensors into "
                "per-expert modules via stacked->per-expert bridge",
                repaired,
            )

    def _apply_v1_to_v2_on_experts(self, model):
        """Apply V1->V2 qzeros correction on repaired expert modules.

        The main V1->V2 pass runs before after_model_load, but expert
        qzeros were all zeros at that point. Now that real values are
        loaded, apply the correction.
        """
        from ...nn_modules.qlinear import BaseQuantLinear
        from ...utils.model import convert_gptq_v1_to_v2_format_module

        qcfg = self.quantize_config
        kernel = self.qlinear_kernel
        if kernel is None or not getattr(kernel, "REQUIRES_FORMAT_V2", False):
            return

        for layer in model.model.layers:
            experts = layer.mlp.experts
            for ml_attr in ("gate_up", "down"):
                for mod in getattr(experts, ml_attr):
                    if isinstance(mod, BaseQuantLinear):
                        convert_gptq_v1_to_v2_format_module(
                            module=mod,
                            bits=qcfg.bits,
                            pack_dtype=qcfg.pack_dtype,
                        )

    @staticmethod
    def _resolve_checkpoint_dir(model):
        """Find the directory containing safetensors shards."""
        ckpt = getattr(model, "checkpoint_file_name", None)
        if ckpt is None:
            return None
        p = Path(ckpt)
        if p.is_file():
            return p.parent
        if p.is_dir():
            return p
        return None

    @staticmethod
    def _build_shard_map(model_dir: Path):
        """Return {tensor_key: shard_filename} for all safetensors."""
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            return index.get("weight_map", {})

        # Single-file model
        single = model_dir / "model.safetensors"
        if single.exists():
            from safetensors import safe_open

            with safe_open(str(single), framework="pt") as f:
                return {k: "model.safetensors" for k in f.keys()}
        # Try any .safetensors file
        for sf in sorted(model_dir.glob("*.safetensors")):
            from safetensors import safe_open

            with safe_open(str(sf), framework="pt") as f:
                return {k: sf.name for k in f.keys()}
        return {}

    @staticmethod
    def _load_tensor(key, tensor_map, model_dir):
        """Load a single tensor from the correct shard file."""
        shard_name = tensor_map.get(key)
        if shard_name is None:
            return None
        from safetensors import safe_open

        shard_path = model_dir / shard_name
        with safe_open(str(shard_path), framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
        return None

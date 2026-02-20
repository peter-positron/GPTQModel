# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseQModel
from ..expert_restack import ExpertRestackSpec


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
            for i in range(self.num_experts):
                tgt_gu_w = self.gate_up[i].weight  # [2E, H]
                tgt_gu_b = self.gate_up[i].bias  # [2E]
                tgt_d_w = self.down[i].weight  # [H, E]
                tgt_d_b = self.down[i].bias  # [H]

                gu_w_src = ori_experts.gate_up_proj[i].detach().t().contiguous()
                gu_b_src = ori_experts.gate_up_proj_bias[i].detach()
                d_w_src = ori_experts.down_proj[i].detach().t().contiguous()
                d_b_src = ori_experts.down_proj_bias[i].detach()

                # Handle meta device tensors from shell model - preserve meta state
                # so alias_all_from_turtle_if_meta can sync from turtle later
                if gu_w_src.device.type == 'meta':
                    self.gate_up[i].to('meta')
                    self.down[i].to('meta')
                    continue

                with torch.inference_mode():
                    tgt_gu_w.copy_(gu_w_src)
                    tgt_gu_b.copy_(gu_b_src)
                    tgt_d_w.copy_(d_w_src)
                    tgt_d_b.copy_(d_b_src)

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
            # Handle meta device tensors from shell model - preserve meta state
            # so alias_all_from_turtle_if_meta can sync from turtle later
            if ori_router.weight.device.type == 'meta':
                self.to('meta')
            else:
                with torch.inference_mode():
                    self.weight.copy_(ori_router.weight.detach())
                    self.bias.copy_(ori_router.bias.detach())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight.to(hidden_states.dtype), self.bias.to(hidden_states.dtype))  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices

class GPTOSSGPTQ(BaseQModel):
    # Enable shell/turtle architecture to ensure proper weight syncing during quantization.
    # Without this, weights are loaded directly to CPU and non-quantized tensors
    # (embed_tokens, lm_head, norm) may be corrupted with initialization values.
    support_offload_to_disk = True

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

    def before_model_load(self, load_quantized_model=False):
        # Always apply class replacements for consistent model structure between
        # quantization and loading phases. The original transformers classes have
        # initialization quirks that can cause weight corruption when loading
        # models with trust_remote_code=True.
        import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling

        gpt_oss_modeling.GptOssExperts = GptOssExpertsNew
        gpt_oss_modeling.GptOssTopKRouter = GptOssTopKRouterNew

    def shell_module_materialize(
        self,
        target_submodule: torch.nn.Module,
        device: torch.device,
        non_blocking: bool = False,
    ) -> torch.nn.Module:
        # Custom materialization for GPT-OSS to handle expert weight splitting
        # from stacked tensor format (turtle) to ModuleList format (shell)
        with self._turtle_lock:
            turtle_model = self.turtle_model

            if turtle_model is None:
                # No turtle model, use default behavior
                return super().shell_module_materialize(target_submodule, device, non_blocking)

            # Check if this is an expert ModuleList that needs special handling
            if isinstance(target_submodule, nn.ModuleList):
                # Try to find the parent path
                target_path = None
                for name, mod in self.model.named_modules():
                    if mod is target_submodule:
                        target_path = name
                        break

                # Check if this is a GPT-OSS expert ModuleList (gate_up or down)
                if target_path and '.experts.' in target_path and (target_path.endswith('.gate_up') or target_path.endswith('.down')):
                    # Find the corresponding original experts module in turtle
                    parent_path = target_path.rsplit('.', 1)[0]  # e.g., "model.layers.0.mlp.experts"
                    turtle_experts = None
                    for name, mod in turtle_model.named_modules():
                        if name == parent_path:
                            turtle_experts = mod
                            break

                    if turtle_experts is not None:
                        is_gate_up = target_path.endswith('.gate_up')

                        # Check if turtle has ModuleList format (from paibaker unpacked weights)
                        # or stacked format (original HF checkpoint)
                        turtle_source = getattr(turtle_experts, 'gate_up' if is_gate_up else 'down', None)
                        has_modulelist = isinstance(turtle_source, nn.ModuleList)

                        # Materialize each expert
                        for i, expert_linear in enumerate(target_submodule):
                            if has_modulelist:
                                # ModuleList format: copy directly from turtle's individual experts
                                w_src = turtle_source[i].weight.detach()
                                b_src = turtle_source[i].bias.detach()
                            else:
                                # Stacked format: extract from stacked tensors with transpose
                                if is_gate_up:
                                    # gate_up_proj is [num_experts, hidden, 2*intermediate]
                                    w_src = turtle_experts.gate_up_proj[i].detach().t().contiguous()
                                    b_src = turtle_experts.gate_up_proj_bias[i].detach()
                                else:
                                    # down_proj is [num_experts, intermediate, hidden]
                                    w_src = turtle_experts.down_proj[i].detach().t().contiguous()
                                    b_src = turtle_experts.down_proj_bias[i].detach()

                            # Allocate target tensor on device if needed
                            if expert_linear.weight.device.type == 'meta':
                                expert_linear.weight = nn.Parameter(
                                    torch.empty_like(expert_linear.weight, device=device),
                                    requires_grad=False
                                )
                                expert_linear.bias = nn.Parameter(
                                    torch.empty_like(expert_linear.bias, device=device),
                                    requires_grad=False
                                )
                            elif expert_linear.weight.device != device:
                                expert_linear.weight.data = expert_linear.weight.data.to(device, copy=True)
                                expert_linear.bias.data = expert_linear.bias.data.to(device, copy=True)

                            # Copy the weights
                            with torch.inference_mode():
                                expert_linear.weight.detach().copy_(
                                    w_src.to(device), non_blocking=(non_blocking and w_src.is_pinned())
                                )
                                expert_linear.bias.detach().copy_(
                                    b_src.to(device), non_blocking=(non_blocking and b_src.is_pinned())
                                )

                        self._maybe_auto_reload_after_alias(target_submodule, target_submodule)
                        return target_submodule

        # Fall back to default behavior for non-expert modules
        return super().shell_module_materialize(target_submodule, device, non_blocking)

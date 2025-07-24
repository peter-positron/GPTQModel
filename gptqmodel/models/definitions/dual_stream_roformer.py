# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..base import BaseGPTQModel
import torch


class DualStreamRoFormerGPTQ(BaseGPTQModel):
    """
    DualStreamRoFormer GPTQ model definition.
    
    This model implements a dual-stream transformer architecture with rotary embeddings
    for processing both text and shape tokens. It consists of:
    - Text and shape projection layers
    - Dual-stream decoder layers with cross-attention
    - Single-stream decoder layers for final processing
    - Language modeling head for token generation
    """
    
    # Non-repeating layers at the root level
    base_modules = ["text_proj", "shape_proj", "transformer.wte", "transformer.ln_f"]
    pre_lm_head_norm_module = "transformer.ln_f"
    
    # The model has two types of repeating layers:
    # 1. dual_blocks: dual-stream decoder layers
    # 2. single_blocks: single-stream decoder layers
    layers_node = "transformer.dual_blocks"
    layer_type = "DualStreamDecoderLayerWithRotaryEmbedding"
    
    # Full tree of quantizable modules for dual-stream layers
    # Each dual-stream layer contains attention and MLP components for both streams
    layers_modules_tree = [
        "transformer",
        "dual_blocks", 
        "#",
        {
            "attn.pre_x": ("c_qk", "c_v"),
            "attn.pre_c": ("c_qk", "c_v"),
            "post_1": ("c_proj",),
            "post_1.mlp": ("gate_proj", "up_proj", "down_proj"),
            "post_2": ("c_proj",),
            "post_2.mlp": ("gate_proj", "up_proj", "down_proj")
        }
    ]
    
    # Quantizable modules within each dual-stream layer
    # Based on the DualStreamDecoderLayerWithRotaryEmbedding structure
    layer_modules = [
        # Dual-stream attention projections
        ["attn.pre_x.c_qk", "attn.pre_x.c_v"],
        ["attn.pre_c.c_qk", "attn.pre_c.c_v"],
        # Post-attention projections
        ["post_1.c_proj"],
        ["post_2.c_proj"],
        # MLP layers for both streams
        ["post_1.mlp.gate_proj", "post_1.mlp.up_proj", "post_1.mlp.down_proj"],
        ["post_2.mlp.gate_proj", "post_2.mlp.up_proj", "post_2.mlp.down_proj"],
    ]

    def get_layer_modules(self, layer_name: str) -> list:
        if "dual_blocks" in layer_name:
            return self.layer_modules
        else:
            return [
                ["attn.c_qk", "attn.c_v"],
                ["attn.c_proj"],
                ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            ]

    def should_quantize_layer(self, layer_name: str) -> bool:
        # Do not quantize the last dual-stream layer (22) as its structure is different.
        # Also, do not quantize single-stream layers.
        if "transformer.dual_blocks.22" in layer_name or "transformer.single_blocks" in layer_name:
            return False
        return True
    
    def forward(self, model, input_ids, attention_mask=None, **kwargs):
        """
        Custom forward pass to bridge GPTQModel's input with DualStreamRoformer's needs.
        """
        # Extract the conditioning embeddings from the keyword arguments
        cond = kwargs.get('cond_embeds', None)
        
        # Convert input_ids to shape embeddings ('x')
        x = model.transformer.wte(input_ids)

        # Create the rotary positional embeddings ('freqs_cis')
        seq_len = x.shape[1]
        n_embd = model.cfg.n_embd
        n_head = model.cfg.n_head
        head_dim = n_embd // n_head
        
        freqs_cis = self._precompute_freqs_cis(
            head_dim, torch.arange(seq_len, device=x.device), model.cfg.rope_theta
        )

        # If no conditioning tensor is provided, create a dummy one
        if cond is None:
            cond_seq_len = 77  # Standard CLIP sequence length
            cond = torch.zeros(x.shape[0], cond_seq_len, n_embd, device=x.device, dtype=x.dtype)

        # Call the model's *original* forward method with positional arguments
        return model._original_forward(x, cond, freqs_cis)

    @torch.no_grad()
    def _precompute_freqs_cis(self, dim: int, t: torch.Tensor, theta: float = 10000.0):
        """
        Calculate rotary embedding cos & sin for the model.
        """
        assert dim % 2 == 0, "RoPE only supports embedding dimensions that are multiples of 2"
        
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=t.device) / dim))
        freqs = torch.outer(t.contiguous().view(-1), freqs).reshape(*t.shape, -1)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_cis

    # Model configuration
    require_trust_remote_code = True
    support_batch_quantize = True
    
    def get_model_layers(self, model):
        """
        Override to handle both dual_blocks and single_blocks
        """
        # Get dual blocks first
        dual_blocks = super().get_model_layers(model)
        
        # Now handle single_blocks if they exist
        single_blocks = []
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'single_blocks'):
            single_blocks = list(model.transformer.single_blocks)
            
        return dual_blocks + single_blocks
    
    def get_quantizable_modules(self, model):
        """
        Override to get modules from both dual and single blocks
        """
        from ...utils.model import find_modules
        
        # Get modules from dual blocks using parent method
        modules = super().get_quantizable_modules(model)
        
        # Add modules from single blocks
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'single_blocks'):
            # Single blocks have a simpler structure (DecoderLayerWithRotaryEmbedding)
            single_block_patterns = [
                "transformer.single_blocks.*.attn.c_qk",
                "transformer.single_blocks.*.attn.c_v", 
                "transformer.single_blocks.*.attn.c_proj",
                "transformer.single_blocks.*.mlp.gate_proj",
                "transformer.single_blocks.*.mlp.up_proj",
                "transformer.single_blocks.*.mlp.down_proj"
            ]
            
            for name, module in model.named_modules():
                if "transformer.single_blocks" in name:
                    # Check if this is a quantizable linear layer
                    if any(pattern.replace("*", "") in name for pattern in single_block_patterns):
                        if hasattr(module, 'weight'):  # Ensure it's a linear layer
                            modules[name] = module
                            
        # Also add the lm_head if it exists
        if hasattr(model, 'lm_head'):
            modules['lm_head'] = model.lm_head
            
        # Add bbox_proj if model uses bbox
        if hasattr(model, 'bbox_proj'):
            modules['bbox_proj'] = model.bbox_proj
            
        return modules 
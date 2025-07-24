from typing import List
from .base import BaseGPTQModel

class DualStreamRoformerGPTQ(BaseGPTQModel):
    base_modules = ["text_proj", "shape_proj", "transformer.wte"]
    
    layers_node = "transformer.dual_blocks"
    layer_type = "DualStreamDecoderLayerWithRotaryEmbedding"
    layer_modules = [
        ["attn.pre_x.c_qk", "attn.pre_x.c_v", "attn.pre_x.c_proj"],
        ["attn.pre_c.c_qk", "attn.pre_c.c_v", "attn.pre_c.c_proj"],
        ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    ] 
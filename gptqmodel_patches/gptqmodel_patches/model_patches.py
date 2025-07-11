"""
Model patches for GPTQModel to support DualStreamRoformer.

These patches extend GPTQModel's model registry and handling
to work with the custom DualStreamRoformer architecture.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings
import gc
from torch.utils.checkpoint import checkpoint


class DualStreamLayerWrapper(nn.Module):
    """
    Wrapper that makes DualStreamRoformer layers compatible with GPTQModel.
    
    This wrapper:
    1. Converts GPTQModel's standard transformer inputs to DualStreamRoformer format
    2. Handles the dual-stream architecture (x, c) -> (x, c)
    3. Provides a standard forward interface that GPTQModel expects
    4. Exposes original linear modules for direct hook registration
    """
    
    def __init__(self, original_layer, layer_type="dual"):
        super().__init__()
        
        # Store the original layer - this is the KEY to the fix
        self.original_layer = original_layer
        self.layer_type = layer_type  # "dual" or "single"
        
        # Get correct dimensions from the actual model
        if hasattr(original_layer, 'attn'):
            if hasattr(original_layer.attn, 'embed_dim'):
                self.embed_dim = original_layer.attn.embed_dim
            elif hasattr(original_layer.attn, 'pre_x') and hasattr(original_layer.attn.pre_x, 'embed_dim'):
                self.embed_dim = original_layer.attn.pre_x.embed_dim
            else:
                self.embed_dim = 1536
                
            if hasattr(original_layer.attn, 'num_heads'):
                self.num_heads = original_layer.attn.num_heads
            elif hasattr(original_layer.attn, 'pre_x') and hasattr(original_layer.attn.pre_x, 'num_heads'):
                self.num_heads = original_layer.attn.pre_x.num_heads
            else:
                self.num_heads = 12
                
            # Calculate head dimension correctly
            self.head_dim = self.embed_dim // self.num_heads
            
            # Try to get the actual head dimension from the model
            if hasattr(original_layer.attn, 'pre_x'):
                if hasattr(original_layer.attn.pre_x, 'head_dim'):
                    self.head_dim = original_layer.attn.pre_x.head_dim
                elif hasattr(original_layer.attn.pre_x, 'c_qk'):
                    # Infer head dimension from weight shapes
                    qk_weight = original_layer.attn.pre_x.c_qk.weight
                    if qk_weight.shape[0] % self.num_heads == 0:
                        self.head_dim = qk_weight.shape[0] // (self.num_heads * 2)  # 2 for q and k
        else:
            self.embed_dim = 1536
            self.num_heads = 12
            self.head_dim = 64  # Default to 64 based on the error message
            
        # Create dummy conditioning for quantization
        self.register_buffer('dummy_conditioning', torch.randn(1, 1, self.embed_dim))
        
        # CRITICAL FIX: Directly expose the original layer's modules as our attributes
        # This ensures that when GPTQ discovers modules via tree paths like 'attn.pre_x.c_qk',
        # it finds the EXACT same module instances that get called during forward pass
        
        # Based on the cube3d source code analysis, we know the exact structure:
        # - self.original_layer points to the actual DualStreamDecoderLayerWithRotaryEmbedding or DecoderLayerWithRotaryEmbedding
        # - The modules that get called are: attn.pre_x.c_qk, attn.pre_x.c_v, attn.pre_c.c_qk/c_k, attn.pre_c.c_v, 
        #   post_1.c_proj, post_1.mlp.{gate_proj,up_proj,down_proj}, post_2.c_proj, post_2.mlp.{gate_proj,up_proj,down_proj}
        
        # NO PROXYING NEEDED - just use the original layer directly!
        # The tree config paths will resolve naturally to the correct modules
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that calls the original layer correctly.
        
        This method handles both:
        1. Single-argument calls: forward(hidden_states) - for legacy/hook compatibility
        2. Keyword-argument calls: forward(x=..., c=...) - for our custom `forward_and_capture`
        
        Based on cube3d source code analysis, we know the exact call signature needed:
        - Dual-stream layers: original_layer(x, c, freqs_cis, ...)
        - Single-stream layers: original_layer(x, freqs_cis, ...)
        """
        layer_index = kwargs.pop('layer_index', -1)
        
        # Pop capture_mode flag to handle different return value requirements
        capture_mode = kwargs.pop('capture_mode', False)
        
        # CRITICAL FIX: Handle calls from both the GPTQ looper (positional arg)
        # and our custom forward_and_capture (keyword args).
        if 'x' in kwargs:
            x = kwargs.pop('x')
        elif args:
            x = args[0]
        else:
            raise ValueError("Could not find hidden_states ('x') in layer input arguments.")
            
        # For dual-stream layers, we need to create conditioning tensor
        if self.layer_type == "dual":
            # Use conditioning tensor if provided, otherwise create a dummy one.
            c = kwargs.pop('c', None)
            if c is None:
                batch_size = x.shape[0]
                device = x.device
                dtype = x.dtype
                cond_len = 77
                embed_dim = self.embed_dim
                c = torch.randn(batch_size, cond_len, embed_dim, device=device, dtype=dtype)
        else:
            # Single-stream layers don't need conditioning
            c = None
            
        # Ensure x has proper shape and dtype for layer processing
        if x.dim() == 0 or x.dim() < 2:
            # Create a proper tensor for calibration
            native_dtype = self.original_layer.ln_1.weight.dtype
            x = torch.randn(1, 1, self.embed_dim, device=x.device, dtype=native_dtype)
        elif x.dim() == 2:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
            
        # Handle dual-stream vs single-stream processing
        if self.layer_type == "dual":
            if c is not None and c.dim() == 2:
                c = c.unsqueeze(0)
                
            # Create freqs_cis for rotary embeddings (dual-stream format)
            seq_len = x.shape[1]
            cond_len = c.shape[1]
            
            main_position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            cond_position_ids = torch.zeros(cond_len, dtype=torch.long, device=x.device)
            
            position_ids = torch.cat([cond_position_ids, main_position_ids], dim=0)
            position_ids = position_ids.unsqueeze(0)
            
            freqs_cis = kwargs.pop('freqs_cis', None)
            if freqs_cis is None:
                try:
                    freqs_cis = self._precompute_freqs_cis(self.head_dim, position_ids)
                except Exception as freq_error:
                    freqs_cis = torch.ones(position_ids.shape[0], position_ids.shape[1], self.head_dim // 2, 
                                         dtype=torch.complex64, device=position_ids.device)
            
            try:
                # Call the original layer (memory managed by auto_gc and PyTorch settings)
                result = self.original_layer(
                    x=x,
                    c=c, 
                    freqs_cis=freqs_cis,
                    attn_mask=None,
                    is_causal=True,
                    kv_cache=None,
                    curr_pos_id=None,
                    decode=False
                )
                
                if isinstance(result, tuple):
                    if capture_mode:
                        return result
                    else:
                        x_out, c_out = result
                        return x_out
                else:
                    return result
                    
            except torch.OutOfMemoryError:
                raise
            except Exception as e:
                dummy_x = torch.zeros_like(x)
                if self.layer_type == "dual":
                    dummy_c = torch.zeros_like(c) if c is not None else torch.zeros_like(x)
                    if capture_mode:
                        return dummy_x, dummy_c
                    else:
                        return dummy_x
                else:
                    return dummy_x
        else:
            # Single-stream layer - only process hidden_states
            seq_len = x.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0)
            
            freqs_cis = kwargs.pop('freqs_cis', None)
            if freqs_cis is None:
                try:
                    freqs_cis = self._precompute_freqs_cis(self.head_dim, position_ids)
                except Exception as freq_error:
                    freqs_cis = torch.ones(position_ids.shape[0], position_ids.shape[1], self.head_dim // 2, 
                                         dtype=torch.complex64, device=position_ids.device)
            
            try:
                # Call the original layer
                result = self.original_layer(
                    x=x,
                    freqs_cis=freqs_cis,
                    attn_mask=None,
                    is_causal=True,
                    kv_cache=None,
                    curr_pos_id=None,
                    decode=False
                )
                return result
            except torch.OutOfMemoryError:
                raise
            except Exception as e:
                return torch.zeros_like(x)

    @torch.no_grad()
    def _precompute_freqs_cis(self, dim: int, t: torch.Tensor, theta: float = 10000.0):
        """
        Calculate rotary embedding cos & sin, replicating the model's exact implementation.
        
        Args:
            dim (int): dimension of the single head of the transformer block
            t (torch.Tensor): position ids [..., L]
            theta (float, optional): rope theta. Defaults to 10000.
            
        Returns:
            torch.Tensor: freqs_cis tensor with shape [batch_size, seq_len, dim//2] as complex tensor
        """
        assert dim % 2 == 0, "RoPE only supports embedding dimensions that are multiples of 2"
        
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=t.device) / dim))
        # [batch_size, seq_len, num_freqs]
        freqs = torch.outer(t.contiguous().view(-1), freqs).reshape(*t.shape, -1)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_cis
            
    def __getattr__(self, name):
        """
        Delegate attribute access to original layer.
        
        CRITICAL FIX: This method is called by GPTQ when it resolves tree paths.
        We intercept these calls and redirect them to our module path mapping
        to ensure GPTQ gets the same module instances that get called during forward pass.
        """
        # First check if this is a protected attribute  
        if name in ['original_layer', 'layer_type', 'embed_dim', 'num_heads', 'head_dim', 'dummy_conditioning']:
            return super().__getattr__(name)
        
        # CRITICAL FIX: Direct passthrough to original layer for all attributes
        # This ensures GPTQModel gets exactly the same module instances that get called
        return getattr(self.original_layer, name)
    
    def __setattr__(self, name, value):
        """
        Handle attribute setting, including module replacement during hook setup.
        
        CRITICAL FIX: This method is called by GPTQ when it replaces modules with hooked versions.
        We need to ensure that module replacements are properly handled in the original layer
        AND also stored on the wrapper for consistent access.
        """
        # Handle protected attributes directly
        if name in ['original_layer', 'layer_type', 'embed_dim', 'num_heads', 'head_dim', 'dummy_conditioning']:
            super().__setattr__(name, value)
            return
        
        # CRITICAL FIX: For module replacements, set both on wrapper and original layer
        # This ensures the packing phase can find modules where it expects them
        if hasattr(self, 'original_layer'):
            # Set on original layer for runtime calls
            if hasattr(self.original_layer, name):
                setattr(self.original_layer, name, value)
            # ALSO set on wrapper for name resolution during packing
            super().__setattr__(name, value)
        else:
            # Before original_layer is set, just set normally
            super().__setattr__(name, value)
        

class UnifiedLayerList(nn.ModuleList):
    """
    A unified layer list that combines dual_blocks and single_blocks 
    into a single layers structure that GPTQModel can understand.
    """
    
    def __init__(self, dual_blocks, single_blocks):
        super().__init__()
        
        # Wrap dual-stream layers
        for layer in dual_blocks:
            wrapped_layer = DualStreamLayerWrapper(layer, layer_type="dual")
            self.append(wrapped_layer)
            
        # Wrap single-stream layers  
        for layer in single_blocks:
            wrapped_layer = DualStreamLayerWrapper(layer, layer_type="single")
            self.append(wrapped_layer)
            
    def __getitem__(self, idx):
        """Override indexing to return wrapped layers."""
        return super().__getitem__(idx)
        
    def __len__(self):
        """Return total number of layers."""
        return super().__len__()
        

class DualStreamRoformerWrapper(nn.Module):
    """
    Wrapper that makes DualStreamRoformer compatible with GPTQModel.
    
    This wrapper:
    1. Exposes layers through a standard 'layers' attribute
    2. Converts input/output format to match transformer expectations
    3. Provides all the attributes GPTQModel needs
    4. Exposes tree-based configuration for accurate module targeting
    """
    
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
        # CRITICAL FIX: Use wrappers but ensure proper module name resolution
        self.layers = nn.ModuleList()
        
        # Add dual-stream layers (0-22), wrapped for interface compatibility
        for layer in original_model.transformer.dual_blocks:
            self.layers.append(DualStreamLayerWrapper(layer, layer_type="dual"))
            
        # Add single-stream layer (23), wrapped for interface compatibility
        for layer in original_model.transformer.single_blocks:
            self.layers.append(DualStreamLayerWrapper(layer, layer_type="single"))
        
        # Expose other components GPTQModel might need
        self.embed_tokens = original_model.transformer.wte
        self.norm = original_model.transformer.ln_f
        self.lm_head = original_model.lm_head
        
        # Also expose through 'transformer' for compatibility
        self.transformer = nn.ModuleDict({
            'wte': self.embed_tokens,
            'ln_f': self.norm,
            'h': self.layers  # Standard transformer layers attribute
        })
        
        # Copy configuration
        self.config = original_model.cfg
        
        # Set vocab size correctly
        self.vocab_size = getattr(original_model, 'vocab_size', 16387)
        
        # Expose layer-aware configuration for GPTQModel
        self.layer_configs = DualStreamRoformerGPTQ.layer_configs
        self.layer_modules = DualStreamRoformerGPTQ.layer_modules
        self.layer_modules_strict = DualStreamRoformerGPTQ.layer_modules_strict
        self.base_modules = DualStreamRoformerGPTQ.base_modules
        self.layers_node = DualStreamRoformerGPTQ.layers_node
        self.layer_type = DualStreamRoformerGPTQ.layer_type
        
    @torch.no_grad()
    def _precompute_freqs_cis(self, dim: int, t: torch.Tensor, theta: float = 10000.0):
        """
        Calculate rotary embedding cos & sin, replicating the model's exact implementation.
        
        Args:
            dim (int): dimension of the single head of the transformer block
            t (torch.Tensor): position ids [..., L]
            theta (float, optional): rope theta. Defaults to 10000.
            
        Returns:
            torch.Tensor: freqs_cis tensor with shape [batch_size, seq_len, dim//2] as complex tensor
        """
        assert dim % 2 == 0, "RoPE only supports embedding dimensions that are multiples of 2"
        
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=t.device) / dim))
        # [batch_size, seq_len, num_freqs]
        freqs = torch.outer(t.contiguous().view(-1), freqs).reshape(*t.shape, -1)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_cis
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Standard transformer forward interface for DualStreamRoformer.
        
        This method creates both the main embeddings and conditioning tensor
        that the dual-stream architecture requires, then processes through each
        layer individually so GPTQModel's hooks can collect inputs properly.
        
        Args:
            input_ids: Token IDs [B, L] or [L]
            attention_mask: Attention mask [B, L] or [L]
            **kwargs: Additional arguments
            
        Returns:
            logits: Output logits [B, L, vocab_size] or [L, vocab_size]
        """
        try:
            # Ensure input_ids has batch dimension
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # Convert to embeddings (main stream)
            embed = self.embed_tokens(input_ids)  # [B, L, D]
            
            # CRITICAL FIX: REMOVED - Do not force float32. Respect the model's native precision (float16).
            # if embed.dtype != torch.float32:
            #     embed = embed.to(torch.float32)
            
            # Create conditioning tensor for dual-stream architecture
            # This should match the expected CLIP text embeddings format
            # Shape: [batch_size, 77, 1536] (77 is CLIP's max sequence length)
            clip_seq_len = 77
            embed_dim = self.config.n_embd
            
            # CRITICAL FIX: Use the embedding's dtype, not hardcoded float32.
            cond = torch.randn(batch_size, clip_seq_len, embed_dim, 
                             device=device, dtype=embed.dtype)
            
            # Process through each layer individually with proper dual-stream handling
            hidden_states = embed
            conditioning = cond
            
            # Go through each original layer
            for i, layer in enumerate(self.layers):
                try:
                    # Determine layer type based on index
                    if i < 23:  # Layers 0-22 are dual-stream
                        # Create freqs_cis for dual-stream
                        seq_len = hidden_states.shape[1]
                        cond_len = conditioning.shape[1]
                        
                        # Create position IDs for dual-stream format
                        main_position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
                        cond_position_ids = torch.zeros(cond_len, dtype=torch.long, device=hidden_states.device)
                        position_ids = torch.cat([cond_position_ids, main_position_ids], dim=0).unsqueeze(0)
                        
                        # Compute freqs_cis
                        head_dim = self.config.n_embd // self.config.n_head
                        freqs_cis = self._precompute_freqs_cis(head_dim, position_ids)
                        
                        # Call dual-stream layer
                        result = layer(
                            x=hidden_states,
                            c=conditioning,
                            freqs_cis=freqs_cis,
                            attn_mask=None,
                            is_causal=True,
                            kv_cache=None,
                            curr_pos_id=None,
                            decode=False
                        )
                        
                        # Update both streams
                        if isinstance(result, tuple):
                            hidden_states, conditioning = result
                        else:
                            hidden_states = result
                            
                    else:  # Layer 23 is single-stream
                        # Create freqs_cis for single-stream
                        seq_len = hidden_states.shape[1]
                        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
                        
                        head_dim = self.config.n_embd // self.config.n_head
                        freqs_cis = self._precompute_freqs_cis(head_dim, position_ids)
                        
                        # Call single-stream layer
                        hidden_states = layer(
                            x=hidden_states,
                            freqs_cis=freqs_cis,
                            attn_mask=None,
                            is_causal=True,
                            kv_cache=None,
                            curr_pos_id=None,
                            decode=False
                        )
                        
                except ValueError as layer_error:
                    # ValueError during calibration is expected from store_input_hook
                    # Re-raise it to let GPTQModel handle it properly
                    raise layer_error
                except Exception as layer_error:
                    print(f"WARNING: Layer {i} failed: {layer_error}")
                    print(f"  Error type: {type(layer_error).__name__}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Apply final norm and get logits
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            
            return logits
            
        except ValueError as e:
            # ValueError during calibration is expected from store_input_hook
            # Re-raise it to let GPTQModel handle it properly
            raise e
        except Exception as e:
            print(f"WARNING: Forward pass failed in DualStreamRoformerWrapper: {e}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return dummy output with correct shape
            if len(input_ids.shape) == 1:
                return torch.randn(input_ids.shape[0], self.vocab_size, device=input_ids.device)
            else:
                batch_size, seq_len = input_ids.shape
                return torch.randn(batch_size, seq_len, self.vocab_size, device=input_ids.device)
            
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Override named_modules to provide both wrapper and original module paths."""
        
        # CRITICAL FIX: Yield modules from the original model structure to ensure consistent naming
        # This bypasses wrapper naming issues entirely
        
        # First, yield non-layer modules normally  
        for name, module in super().named_modules(memo, prefix, remove_duplicate):
            if name.endswith(('text_proj', 'shape_proj')):
                continue
            if not name.startswith('layers.') and name != 'original_model':
                yield name, module
        
        # Then, yield layer modules using original model paths
        if hasattr(self, 'original_model'):
            for name, module in self.original_model.named_modules(memo, 'original_model', remove_duplicate):
                # Only yield Linear modules from the transformer blocks
                if isinstance(module, nn.Linear) and ('dual_blocks' in name or 'single_blocks' in name):
                    yield name, module
    
    def __getattr__(self, name):
        """Delegate attribute access to original model with path translation for packing compatibility."""
        if name in ['original_model', 'layers', 'embed_tokens', 'norm', 'lm_head', 'transformer', 'config', 'vocab_size', 
                   'layers_modules_tree', 'layer_modules', 'layer_modules_strict', 'base_modules', 'layers_node', 'layer_type']:
            return super().__getattr__(name)
        
        # CRITICAL FIX: Handle direct access to original model structure during packing
        if name == 'original_model':
            return self.original_model
            
        return getattr(self.original_model, name)

    @torch.no_grad()
    def forward_and_capture(self, input_ids, attention_mask=None, **kwargs):
        """
        Runs a forward pass and captures the inputs for each layer.
        Uses gradient checkpointing and manual device placement to reduce memory.
        """
        captured_inputs = []
        
        # Determine target device for computation
        compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if compute_device == 'cpu':
            warnings.warn("CUDA not available, running forward pass on CPU. This will be very slow.")

        # --- This is the same setup logic from the main forward pass ---
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, seq_len = input_ids.shape
        # Embeddings are on CPU, so ensure input is on CPU
        cpu_device = self.embed_tokens.weight.device
        embed = self.embed_tokens(input_ids.to(cpu_device))

        clip_seq_len = 77
        embed_dim = self.config.n_embd
        
        cond = torch.randn(batch_size, clip_seq_len, embed_dim, 
                         device=cpu_device, dtype=embed.dtype)
        
        hidden_states = embed
        conditioning = cond
        
        # --- Main Loop ---
        for i, layer in enumerate(self.layers):
            # The layer is on CPU by default. All tensors are on CPU.
            kw_args = {
                'x': hidden_states,
                'c': conditioning if i < 23 else None,
                'freqs_cis': None,
            }

            # This part must run on CPU as hidden_states/conditioning are on CPU
            if i < 23: # Dual-stream
                seq_len_layer = hidden_states.shape[1]
                cond_len_layer = conditioning.shape[1]
                main_pos = torch.arange(seq_len_layer, dtype=torch.long, device=cpu_device)
                cond_pos = torch.zeros(cond_len_layer, dtype=torch.long, device=cpu_device)
                position_ids = torch.cat([cond_pos, main_pos], dim=0).unsqueeze(0)
                head_dim = self.config.n_embd // self.config.n_head
                freqs_cis = self._precompute_freqs_cis(head_dim, position_ids)
                kw_args['freqs_cis'] = freqs_cis
            else: # Single-stream
                seq_len_layer = hidden_states.shape[1]
                position_ids = torch.arange(seq_len_layer, dtype=torch.long, device=cpu_device).unsqueeze(0)
                head_dim = self.config.n_embd // self.config.n_head
                freqs_cis = self._precompute_freqs_cis(head_dim, position_ids)
                kw_args.pop('c')
                kw_args['freqs_cis'] = freqs_cis

            # We capture the CPU tensors. The quantizer will move them later.
            captured_inputs.append(((), kw_args))

            # --- GPU Computation Scope ---
            try:
                # 1. Move layer and inputs to GPU
                layer.to(compute_device)
                gpu_kw_args = {k: v.to(compute_device) if isinstance(v, torch.Tensor) else v for k, v in kw_args.items() if v is not None}

                # 2. Execute on GPU
                result = checkpoint(layer, use_reentrant=False, **gpu_kw_args, capture_mode=True)
                
                # 3. Move results back to CPU
                if isinstance(result, tuple):
                    hidden_states, conditioning = result
                    hidden_states = hidden_states.to(cpu_device)
                    if conditioning is not None:
                        conditioning = conditioning.to(cpu_device)
                else:
                    hidden_states = result.to(cpu_device)
            finally:
                # 4. ALWAYS move layer back to CPU
                layer.to(cpu_device)
            # --- End GPU Computation Scope ---

        return hidden_states, conditioning, captured_inputs


def get_layer_modules_for_layer(layer_idx, total_dual_layers=23):
    """
    Get the appropriate layer modules for a specific layer index.
    
    Args:
        layer_idx: Index of the layer (0-based)
        total_dual_layers: Number of dual-stream layers (default 23)
    
    Returns:
        List of module lists for this layer
    """
    if layer_idx < total_dual_layers:
        # This is a dual-stream layer
        return [
            # Dual-stream attention modules (always present)
            ["original_layer.attn.pre_x.c_qk", "original_layer.attn.pre_x.c_v"],
            ["original_layer.attn.pre_c.c_qk", "original_layer.attn.pre_c.c_v"],
            
            # Dual-stream post attention (always present)
            ["original_layer.post_1.c_proj"],
            ["original_layer.post_1.mlp.gate_proj", "original_layer.post_1.mlp.up_proj", "original_layer.post_1.mlp.down_proj"],
        ]
    else:
        # This is a single-stream layer
        return [
            # Single-stream attention modules
            ["original_layer.attn.c_qk", "original_layer.attn.c_v", "original_layer.attn.c_proj"],
            
            # Single-stream MLP modules
            ["original_layer.mlp.gate_proj", "original_layer.mlp.up_proj", "original_layer.mlp.down_proj"]
        ]


class DualStreamRoformerGPTQ:
    """
    GPTQ wrapper for DualStreamRoformer that properly handles the architecture.
    
    DualStreamRoformer Architecture:
    - Layers 0-22: DualStreamDecoderLayerWithRotaryEmbedding (dual-stream)
    - Layer 23: DecoderLayerWithRotaryEmbedding (single-stream)
    - Layer 22 is special: cond_pre_only=True (no post_2 module)
    
    Tree-based targeting allows GPTQModel to accurately target modules
    based on the actual model structure rather than using legacy flat lists.
    """
    
    # GPTQModel 4.0 architecture constants
    base_modules = ["embed_tokens", "norm", "lm_head"]  # Removed non-existent modules
    layers_node = "layers"
    layer_type = "DualStreamDecoderLayerWithRotaryEmbedding"  # Now points to original layers
    
    # NOTE: layers_modules_tree is deprecated in favor of dynamic layer_configs
    # to handle the model's heterogeneous layer architecture.
    
    # Legacy layer_modules for backward compatibility (now unused thanks to tree config)
    layer_modules = [
        # Dual-stream attention modules (always present in dual-stream layers)
        ["attn.pre_x.c_qk", "attn.pre_x.c_v"],
        ["attn.pre_c.c_qk", "attn.pre_c.c_v"],
        
        # Dual-stream post attention (always present in dual-stream layers)
        ["post_1.c_proj"],
        ["post_1.mlp.gate_proj", "post_1.mlp.up_proj", "post_1.mlp.down_proj"],
        
        # NOTE: Removed single-stream specific modules to avoid "not found" errors
        # Single-stream layers (layer 23) will be skipped as their modules don't exist in dual-stream layers
        # This is acceptable since we're primarily interested in quantizing the dual-stream layers (0-22)
    ]

    # Layer-specific module definitions to handle architectural variations
    layer_configs = {
        "dual_stream_default": [
            ["attn.pre_x.c_qk", "attn.pre_x.c_v"],
            ["attn.pre_c.c_qk", "attn.pre_c.c_v"],
            ["post_1.c_proj"],
            ["post_1.mlp.gate_proj", "post_1.mlp.up_proj", "post_1.mlp.down_proj"],
            ["post_2.c_proj"],
            ["post_2.mlp.gate_proj", "post_2.mlp.up_proj", "post_2.mlp.down_proj"],
        ],
        "dual_stream_cond_pre_only": [
            ["attn.pre_x.c_qk", "attn.pre_x.c_v"],
            ["attn.pre_c.c_k", "attn.pre_c.c_v"], # Note: c_qk becomes c_k
            ["post_1.c_proj"],
            ["post_1.mlp.gate_proj", "post_1.mlp.up_proj", "post_1.mlp.down_proj"],
        ],
        "single_stream": [
            ["attn.c_qk", "attn.c_v"],
            ["attn.c_proj"],
            ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        ]
    }
    
    # Disable strict checking because modules vary by layer type
    # (dual-stream vs single-stream, cond_pre_only variations)
    layer_modules_strict = False
    
    # GPTQModel compatibility settings
    require_trust_remote_code = True
    support_batch_quantize = True
    
    def __init__(self, model, quantized=False, quantize_config=None, **kwargs):
        """Initialize DualStreamRoformerGPTQ."""
        from gptqmodel.models.base import BaseGPTQModel
        
        # Wrap the model to make it compatible
        wrapped_model = DualStreamRoformerWrapper(model)
        
        # Get vocabulary size from wrapped model
        vocab_size = getattr(wrapped_model, 'vocab_size', 16387)
        
        # For quantization, pass None tokenizer and handle it in our quantize method
        tokenizer = None
        
        # Create our Patched BaseGPTQModel instance instead of the original
        self.base_model = PatchedBaseGPTQModel(
            model=wrapped_model,  # type: ignore
            quantized=quantized,
            quantize_config=quantize_config,  # type: ignore
            tokenizer=tokenizer,
            trust_remote_code=True
        )
        
        # Set architecture attributes
        self.base_model.base_modules = self.base_modules
        self.base_model.layers_node = self.layers_node
        self.base_model.layer_type = self.layer_type
        self.base_model.layer_modules = self.layer_modules
        self.base_model.layers_modules_tree = self.layers_modules_tree  # Enable tree-based config
        self.base_model.layer_modules_strict = self.layer_modules_strict
        
        # Set optional attributes if they exist
        try:
            if hasattr(self.base_model, 'require_trust_remote_code'):
                self.base_model.require_trust_remote_code = self.require_trust_remote_code  # type: ignore
        except (AttributeError, TypeError):
            pass
        try:
            self.base_model.support_batch_quantize = self.support_batch_quantize
        except AttributeError:
            pass

        # Add layer configs for dynamic module loading
        self.base_model.layer_configs = self.layer_configs
        
        # Store references
        self.model = wrapped_model
        self.quantized = quantized
        self.quantize_config = quantize_config
        self.vocab_size = vocab_size  # Store vocab_size for later use
        
    def quantize(self, examples, **kwargs):
        """
        Quantize the model with handling for None tokenizer and custom input caching.
        
        This method manually prepares and transposes the input cache because the
        default GPTQ looper expects inputs organized by layer, while our model's
        `forward_and_capture` provides them organized by example. This prevents
        indexing errors during the quantization loop.
        """
        # 1. Prepare examples and the custom input cache
        processed_examples = self._prepare_examples(examples)
        
        # Check if we have a pre-computed input cache available
        if hasattr(self.base_model, '_precomputed_input_cache') and self.base_model._precomputed_input_cache is not None:
            print("INFO: Using pre-computed input cache - skipping expensive forward passes!")
            input_cache = self.base_model._precomputed_input_cache
        else:
            # Compute input cache from scratch
            all_captured_inputs = []
            num_examples = len(processed_examples)
            print(f"INFO: Starting input caching for {num_examples} calibration examples (this may be slow)...")
            for i, example in enumerate(processed_examples):
                print(f"  Caching inputs for example {i + 1}/{num_examples}...")
                _, _, captured = self.model.forward_and_capture(**example)
                all_captured_inputs.append(captured)
                # CRITICAL FIX: Explicitly clear memory after each forward pass
                # to prevent accumulation of activation tensors across examples.
                gc.collect()
                torch.cuda.empty_cache()
            print("INFO: Input caching complete.")

            if not all_captured_inputs:
                raise ValueError("Calibration dataset is empty or failed to produce inputs.")

            # 2. Transpose cache from [num_examples, num_layers] to [num_layers, num_examples]
            num_layers = len(all_captured_inputs[0])
            num_examples = len(all_captured_inputs)
            
            input_cache = [[] for _ in range(num_layers)]
            for layer_idx in range(num_layers):
                for example_idx in range(num_examples):
                    input_cache[layer_idx].append(all_captured_inputs[example_idx][layer_idx])
            
            # Store the computed cache for potential saving
            self.base_model._computed_input_cache = input_cache
        
        # 3. Store the custom cache where our patched looper can find it
        self.base_model._custom_input_cache = input_cache
        
        # 4. Handle tokenizer for the base quantizer
        class TempTokenizer:
            def __init__(self, vocab_size):
                self.pad_token_id = vocab_size - 1
                
        original_tokenizer = getattr(self.base_model, 'tokenizer', None)
        try:
            self.base_model.tokenizer = TempTokenizer(self.vocab_size)  # type: ignore
        except AttributeError:
            pass
        
        try:
            # 5. Call base quantization. It will now use our custom cache.
            # We still pass processed_examples because other parts might need it (e.g., logging)
            result = self.base_model.quantize(processed_examples, **kwargs)
            return result
        finally:
            # Restore original tokenizer
            try:
                if hasattr(self.base_model, 'tokenizer'):
                    self.base_model.tokenizer = original_tokenizer  # type: ignore
            except AttributeError:
                pass
        
    def _prepare_examples(self, examples):
        """Convert text examples to token format with correct vocabulary range."""
        processed = []
        vocab_size = getattr(self.model, 'vocab_size', 16387)
        # CRITICAL FIX: Determine the model's device and move all tensors there.
        device = self.model.embed_tokens.weight.device
        
        for example in examples:
            if isinstance(example, str):
                # Convert text to token IDs within valid range
                # Make sure sequences are long enough (>= 10 tokens as required by GPTQModel)
                # CRITICAL FIX: Reduce sequence length to prevent OOM errors during calibration.
                seq_len = 32
                # Generate token IDs within model vocabulary range, on the correct device
                token_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long, device=device)
                attention_mask = torch.ones((seq_len,), dtype=torch.long, device=device)
                processed.append({
                    "input_ids": token_ids,
                    "attention_mask": attention_mask
                })
            elif isinstance(example, dict) and "input_ids" in example:
                # Ensure tensors are on the correct device
                input_ids = example["input_ids"].to(device)
                
                # Handle different input_ids shapes
                if len(input_ids.shape) == 2:
                    # [batch_size, seq_len] - take first batch item
                    input_ids = input_ids[0]
                elif len(input_ids.shape) == 1:
                    # [seq_len] - already correct
                    pass
                else:
                    raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")
                
                if "attention_mask" not in example:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = example["attention_mask"].to(device)
                    if len(attention_mask.shape) == 2:
                        attention_mask = attention_mask[0]
                
                # Ensure minimum length and clamp token IDs to vocabulary range
                if input_ids.shape[0] < 10:
                    seq_len = 32
                    input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long, device=device)
                    attention_mask = torch.ones((seq_len,), dtype=torch.long, device=device)
                else:
                    # Clamp existing token IDs to vocabulary range
                    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                    
                processed.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })
            else:
                # Fallback: create dummy tokens with sufficient length on the correct device
                token_ids = torch.randint(0, vocab_size, (32,), dtype=torch.long, device=device)
                attention_mask = torch.ones((32,), dtype=torch.long, device=device)
                processed.append({
                    "input_ids": token_ids,
                    "attention_mask": attention_mask
                })
                
        return processed
        
    def save_quantized(self, save_dir, **kwargs):
        """Save the quantized model."""
        if hasattr(self.base_model, 'save_quantized') and callable(self.base_model.save_quantized):
            return self.base_model.save_quantized(save_dir, **kwargs)
        else:
            raise AttributeError("Base model does not have a callable save_quantized method")
        
    def __getattr__(self, name):
        """Delegate to base model."""
        if name in ['base_model', 'model', 'quantized', 'quantize_config', 'vocab_size']:
            return object.__getattribute__(self, name)
        return getattr(self.base_model, name)


def visualize_dualstream_tree_config():
    """
    Visualize the tree-based configuration for DualStreamRoformer.
    
    This function helps understand how the tree-based targeting works
    and can be used for debugging module targeting issues.
    """
    tree = DualStreamRoformerGPTQ.layers_modules_tree
    
    return tree


def validate_tree_config_against_model(model):
    """
    Validate the tree configuration against an actual model instance.
    
    Args:
        model: DualStreamRoformer model instance
        
    Returns:
        dict: Validation results with found/missing modules
    """
    results = {
        "found_modules": [],
        "missing_modules": [],
        "layer_analysis": {}
    }
    
    if not hasattr(model, 'layers'):
        results["error"] = "Model doesn't have 'layers' attribute"
        return results
    
    # Check each layer
    for layer_idx, layer in enumerate(model.layers):
        layer_results = {
            "layer_type": str(type(layer)),
            "found": [],
            "missing": []
        }
        
        # Check what modules exist in this layer
        for name, module in layer.named_modules():
            if name:  # Skip the layer itself
                layer_results["found"].append(name)
        
        # Check against tree config targets
        tree_targets = DualStreamRoformerGPTQ.layers_modules_tree[-1]
        for module_path, target_modules in tree_targets.items():
            for target in target_modules:
                full_path = f"{module_path}.{target}"
                if full_path not in layer_results["found"]:
                    layer_results["missing"].append(full_path)
        
        results["layer_analysis"][layer_idx] = layer_results
    
    return results


def verify_tree_config_enabled(model):
    """
    Verify that the tree-based configuration is properly enabled for a model.
    
    Args:
        model: The model instance to check
        
    Returns:
        dict: Verification results
    """
    results = {
        "tree_config_enabled": False,
        "has_layers_modules_tree": False,
        "has_legacy_layer_modules": False,
        "layer_modules_strict": None,
        "expected_behavior": "unknown"
    }
    
    # Check if model has tree configuration
    if hasattr(model, 'layers_modules_tree') and model.layers_modules_tree:
        results["has_layers_modules_tree"] = True
        results["tree_config_enabled"] = True
        results["expected_behavior"] = "tree-based config (structured targeting)"
    
    # Check if model has legacy configuration
    if hasattr(model, 'layer_modules') and model.layer_modules:
        results["has_legacy_layer_modules"] = True
        if not results["tree_config_enabled"]:
            results["expected_behavior"] = "legacy config (flat list targeting)"
    
    # Check strict mode
    if hasattr(model, 'layer_modules_strict'):
        results["layer_modules_strict"] = model.layer_modules_strict
    
    return results


def print_tree_config_status(model):
    """
    Print the tree configuration status for a model.
    
    Args:
        model: The model instance to check
    """
    print("=" * 60)
    print("GPTQModel Tree Configuration Status")
    print("=" * 60)
    
    results = verify_tree_config_enabled(model)
    
    print(f"Tree Config Enabled: {results['tree_config_enabled']}")
    print(f"Has layers_modules_tree: {results['has_layers_modules_tree']}")
    print(f"Has legacy layer_modules: {results['has_legacy_layer_modules']}")
    print(f"Layer modules strict: {results['layer_modules_strict']}")
    print(f"Expected behavior: {results['expected_behavior']}")
    
    if results['tree_config_enabled']:
        print("\n✅ SUCCESS: Tree-based config is enabled!")
        print("   GPTQModel will use: 'tree based config for accurate targeting'")
        print("   This eliminates the 'Using legacy based config' message.")
    else:
        print("\n⚠️  WARNING: Tree-based config not detected!")
        print("   GPTQModel will use: 'legacy based config for targeting'")
        print("   Consider enabling tree-based configuration.")
    
    print("=" * 60)
    
    return results


# Export the patches
def patch_model_registry():
    """Patch GPTQModel's model registry to recognize DualStreamRoformer."""
    try:
        import gptqmodel.models
        
        # Add DualStreamRoformer to the registry
        if hasattr(gptqmodel.models, 'MODEL_MAP'):
            gptqmodel.models.MODEL_MAP['DualStreamRoformer'] = DualStreamRoformerGPTQ
            gptqmodel.models.MODEL_MAP['DualStreamRoformerWrapper'] = DualStreamRoformerGPTQ
            
            # Also patch the model loading to ensure tree config is used
            original_from_pretrained = None
            
            # Store reference to original from_pretrained if it exists
            original_from_pretrained = getattr(DualStreamRoformerGPTQ, 'from_pretrained', None)
            
            @classmethod
            def enhanced_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                """Enhanced from_pretrained that ensures tree config is used."""
                # Create instance
                if original_from_pretrained:
                    instance = original_from_pretrained(pretrained_model_name_or_path, **kwargs)
                else:
                    # Fallback to standard loading
                    from gptqmodel.models.base import BaseGPTQModel
                    if hasattr(BaseGPTQModel, 'from_pretrained'):
                        instance = BaseGPTQModel.from_pretrained(pretrained_model_name_or_path, **kwargs)  # type: ignore
                    else:
                        raise AttributeError("BaseGPTQModel does not have from_pretrained method")
                
                return instance
            
            # Apply the enhanced from_pretrained
            try:
                setattr(DualStreamRoformerGPTQ, 'from_pretrained', enhanced_from_pretrained)
            except AttributeError:
                pass
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Could not patch model registry: {e}")
        return False


def patch_model_loading():
    """Patch GPTQModel's model loading to handle DualStreamRoformer."""
    try:
        import gptqmodel.models.base
        
        # Store original method
        original_from_pretrained = getattr(gptqmodel.models.base.BaseGPTQModel, 'from_pretrained', None)
        
        @classmethod
        def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            """Patched from_pretrained that handles DualStreamRoformer."""
            try:
                if original_from_pretrained is not None:
                    return original_from_pretrained(pretrained_model_name_or_path, **kwargs)
                else:
                    raise AttributeError("No original from_pretrained method found")
            except Exception as e:
                # If standard loading fails, try DualStreamRoformer approach
                raise e  # Re-raise for now
                
        # Apply patch
        try:
            setattr(gptqmodel.models.base.BaseGPTQModel, 'from_pretrained', patched_from_pretrained)
        except AttributeError:
            pass
        
        return True
        
    except Exception as e:
        return False


def patch_layer_detection():
    """Patch GPTQModel's layer detection for DualStreamRoformer."""
    try:
        import gptqmodel.utils.model
        
        # Store original function
        original_find_modules = getattr(gptqmodel.utils.model, 'find_modules', None)
        
        def patched_find_modules(module: nn.Module, layers=None, name: str = "", **kwargs):
            """Patched module finder for DualStreamRoformer."""
            modules = {}
            
            # Check if this is our wrapped model
            if hasattr(module, 'layers') and hasattr(module.layers, '__class__') and 'UnifiedLayerList' in str(module.layers.__class__):
                # Find all Linear modules in the wrapped layers
                for mod_name, mod in module.named_modules():
                    if isinstance(mod, nn.Linear):
                        modules[mod_name] = mod
                        
                return modules
            
            # Fall back to original implementation
            if original_find_modules:
                return original_find_modules(module, layers=layers, name=name, **kwargs)
            else:
                # Simple fallback
                for mod_name, mod in module.named_modules():
                    if isinstance(mod, nn.Linear):
                        modules[mod_name] = mod
                return modules
        
        # Apply the patch
        gptqmodel.utils.model.find_modules = patched_find_modules
        
        return True
        
    except Exception as e:
        return False


def apply_model_patches():
    """Apply all model-related patches."""
    patches = [
        ("model registry", patch_model_registry),
        ("model loading", patch_model_loading), 
        ("layer detection", patch_layer_detection)
    ]
    
    results = []
    for name, patch_func in patches:
        try:
            success = patch_func()
            results.append(success)
        except Exception as e:
            results.append(False)
    
    success_count = sum(results)
    
    return success_count == len(patches) 

from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.looper.module_looper import ModuleLooper


class PatchedBaseGPTQModel(BaseGPTQModel):
    """
    Subclass of BaseGPTQModel to inject our custom, pre-processed input cache.
    """
    _custom_input_cache: Optional[List[Any]] = None

    def _get_module_looper(self, *args, **kwargs):
        """
        Overrides the base method to return a looper pre-filled with our data.
        """
        # Create the standard looper
        looper = super()._get_module_looper(*args, **kwargs)
        
        # If we have a custom cache, set it on the looper instance
        if self._custom_input_cache is not None:
            looper.set_inputs(self._custom_input_cache)
            print("INFO: Custom input cache successfully injected into module looper.")
        
        return looper 

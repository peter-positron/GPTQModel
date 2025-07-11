"""
Utility patches for GPTQModel to support DualStreamRoformer.

These patches provide only the essential compatibility fixes needed
to make GPTQModel 4.0.0 work with DualStreamRoformer architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import warnings


def patch_model_utils():
    """Patch GPTQModel's model utility functions for DualStreamRoformer compatibility."""
    try:
        import gptqmodel.utils.model
        
        # Store original function
        original_find_modules = getattr(gptqmodel.utils.model, 'find_modules', None)
        
        def patched_find_modules(model, **kwargs):
            """Enhanced module finder that works with DualStreamRoformer."""
            # Check if this is a DualStreamRoformer or DualStreamRoformer layer
            is_dualstream = (
                hasattr(model, 'transformer') and 'dual_stream' in str(type(model)).lower()
            ) or (
                'DualStream' in str(type(model)) or 'dual_stream' in str(type(model)).lower()
            )
            
            if is_dualstream:
                # DualStreamRoformer-specific module finding
                modules = {}
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Exclude projection layers that we don't want to quantize
                        if name.endswith(('text_proj', 'shape_proj')):
                            continue
                        modules[name] = module
                return modules
            
            # Fall back to original implementation for other models
            if original_find_modules:
                return original_find_modules(model, **kwargs)
            else:
                # Simple fallback
                modules = {}
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        modules[name] = module
                return modules
        
        # Apply the patch
        gptqmodel.utils.model.find_modules = patched_find_modules
        
        print("Patched model utilities for DualStreamRoformer")
        return True
        
    except Exception as e:
        print(f"Could not patch model utilities: {e}")
        return False


def patch_compatibility():
    """Patch compatibility issues between GPTQModel 4.0.0 and DualStreamRoformer."""
    try:
        # Suppress warnings that may occur during DualStreamRoformer quantization
        warnings.filterwarnings('ignore', category=UserWarning, message='.*GPTQ.*')
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*quantization.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*roformer.*')
        
        print("Patched compatibility issues")
        return True
        
    except Exception as e:
        print(f"Could not patch compatibility: {e}")
        return False


def apply_utils_patches():
    """Apply essential utility patches for DualStreamRoformer support."""
    success_count = 0
    essential_patches = 2  # Only the patches we actually need
    
    if patch_model_utils():
        success_count += 1
    
    if patch_compatibility():
        success_count += 1
    
    print(f"Applied {success_count}/{essential_patches} essential utility patches")
    
    if success_count == essential_patches:
        print("All essential utility patches applied successfully!")
        return True
    else:
        print("WARNING: Some essential utility patches failed")
        return False 
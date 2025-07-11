"""
GPTQModel Patches for DualStreamRoformer Support

This package provides essential patches to make GPTQModel 4.0.0 work with
the DualStreamRoformer architecture from the qube project.

The patches focus on:
1. Model registry and loading (essential)
2. Quantization logic compatibility (essential) 
3. Basic utility functions (essential)

Usage:
    from gptqmodel_patches import apply_patches
    success = apply_patches()
"""

import warnings
from .model_patches import apply_model_patches
from .quantizer_patches import apply_quantizer_patches
from .utils_patches import apply_utils_patches


def check_gptqmodel_version():
    """Check if GPTQModel is the expected version."""
    try:
        import gptqmodel
        version = getattr(gptqmodel, '__version__', 'unknown')
        
        # Check for 4.0.0 or 4.0.0-dev
        if version.startswith('4.0.0'):
            print(f"GPTQModel version {version} detected - compatible")
            return True
        else:
            print(f"WARNING: GPTQModel version {version} detected")
            print("These patches are designed for GPTQModel 4.0.0")
            print("Proceeding anyway, but some features may not work correctly")
            return True  # Still try to apply patches
            
    except ImportError:
        print("ERROR: GPTQModel not found")
        return False


def apply_patches(ignore_failures=False):
    """
    Apply essential patches for DualStreamRoformer support in GPTQModel 4.0.0.
    
    Args:
        ignore_failures (bool): If True, continue even if some patches fail
        
    Returns:
        bool: True if all essential patches were applied successfully
    """
    print("=== Applying GPTQModel Patches for DualStreamRoformer ===")
    
    # Check GPTQModel version
    if not check_gptqmodel_version():
        if not ignore_failures:
            return False
        print("Continuing despite version check failure...")
    
    # Apply patches in order of importance
    patch_results = []
    
    # 1. Utility patches (must be applied first for find_modules to work)
    print("\n1. Applying utility patches...")
    try:
        result = apply_utils_patches()
        patch_results.append(('utils', result))
        if result:
            print("   Utility patches applied successfully")
        else:
            print("   Utility patches failed")
    except Exception as e:
        print(f"   Utility patches failed with error: {e}")
        patch_results.append(('utils', False))
    
    # 2. Model patches (most critical)
    print("\n2. Applying model patches...")
    try:
        result = apply_model_patches()
        patch_results.append(('model', result))
        if result:
            print("   Model patches applied successfully")
        else:
            print("   Model patches failed")
    except Exception as e:
        print(f"   Model patches failed with error: {e}")
        patch_results.append(('model', False))
    
    # 3. Quantizer patches (important)
    print("\n3. Applying quantizer patches...")
    try:
        result = apply_quantizer_patches()
        patch_results.append(('quantizer', result))
        if result:
            print("   Quantizer patches applied successfully")
        else:
            print("   Quantizer patches failed")
    except Exception as e:
        print(f"   Quantizer patches failed with error: {e}")
        patch_results.append(('quantizer', False))
    
    # Evaluate results
    successful_patches = [name for name, success in patch_results if success]
    failed_patches = [name for name, success in patch_results if not success]
    
    print(f"\n=== Patch Results ===")
    print(f"Successful: {successful_patches}")
    if failed_patches:
        print(f"Failed: {failed_patches}")
    
    # Determine overall success
    # Model patches are absolutely essential
    model_success = any(name == 'model' and success for name, success in patch_results)
    
    if model_success:
        print("\nSUCCESS: Essential model patches applied!")
        if failed_patches:
            print(f"WARNING: Some non-essential patches failed: {failed_patches}")
            if not ignore_failures:
                print("Use --ignore_patch_failures to continue anyway")
        return True
    else:
        print("\nERROR: Essential model patches failed!")
        print("DualStreamRoformer quantization will not work without model patches")
        return False


def get_patch_info():
    """Get information about available patches."""
    return {
        'version': '1.0.0',
        'target_gptqmodel': '4.0.0',
        'supported_models': ['DualStreamRoformer'],
        'patches': {
            'model': 'Model registry and loading support',
            'quantizer': 'Quantization logic compatibility',
            'utils': 'Utility function patches'
        }
    } 
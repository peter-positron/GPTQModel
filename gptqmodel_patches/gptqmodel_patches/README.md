# GPTQModel Patches for DualStreamRoformer

This package provides essential patches to make GPTQModel 4.0.0 work with the DualStreamRoformer architecture from the qube project.

## Overview

The patches are streamlined to focus only on what's absolutely necessary for GPTQModel 4.0.0 compatibility:

1. **Model Patches** (Essential): Registers DualStreamRoformer with GPTQModel and provides the quantization implementation
2. **Quantizer Patches** (Essential): Ensures quantization logic works with DualStreamRoformer 
3. **Utility Patches** (Helpful): Basic utility functions for model introspection

## Requirements

- GPTQModel 4.0.0 or 4.0.0-dev
- PyTorch with CUDA support
- The qube project with DualStreamRoformer model

## Installation

No installation required - just ensure the `gptqmodel_patches` directory is in your Python path.

## Usage

### Basic Usage

```python
from gptqmodel_patches import apply_patches

# Apply patches before using GPTQModel
success = apply_patches()
if success:
    print("Patches applied successfully!")
else:
    print("Some patches failed - check output for details")
```

### With Error Handling

```python
from gptqmodel_patches import apply_patches

# Apply patches with tolerance for non-critical failures
success = apply_patches(ignore_failures=True)
```

### Command Line Usage

```bash
# Apply patches and quantize model
python quantize_dualstream_roformer_gptq.py /path/to/model/weights

# Continue even if some patches fail
python quantize_dualstream_roformer_gptq.py /path/to/model/weights --ignore_patch_failures
```

## Patch Details

### Model Patches (`model_patches.py`)

- **DualStreamRoformerGPTQ**: Custom GPTQ implementation that inherits from BaseGPTQModel
- **Model Registry**: Registers DualStreamRoformer with GPTQModel's model registry
- **Calibration Data**: Handles conversion of text prompts to model-compatible format

### Quantizer Patches (`quantizer_patches.py`)

- **Quantization Logic**: Ensures GPTQModel's quantization works with DualStreamRoformer
- **Compatibility**: Handles API differences in GPTQModel 4.0.0

### Utility Patches (`utils_patches.py`)

- **Model Utils**: Enhanced module finding for DualStreamRoformer architecture
- **Compatibility**: Suppresses warnings during quantization

## GPTQModel 4.0.0 Compatibility

These patches are specifically designed for GPTQModel 4.0.0 and later. Key changes handled:

- Model registry changes (`MODEL_MAP` instead of `GPTQ_CAUSAL_LM_MODEL_MAP`)
- Device configuration requirements
- API changes in quantization methods
- New BaseGPTQModel inheritance pattern

## Troubleshooting

### Common Issues

1. **"GPTQModel not found"**: Install with `pip install gptqmodel`
2. **"Essential model patches failed"**: Check GPTQModel version compatibility
3. **"Could not import qube modules"**: Run from qube project root directory
4. **Device errors**: Ensure CUDA is available and properly configured

### Version Compatibility

- ✅ GPTQModel 4.0.0
- ✅ GPTQModel 4.0.0-dev
- ❌ GPTQModel < 4.0.0 (API differences)

### Debug Mode

For detailed debugging, check the console output when applying patches. The system will report:
- Which patches were applied successfully
- Which patches failed and why
- Whether the failure is critical or can be ignored

## Example Output

```
=== Applying GPTQModel Patches for DualStreamRoformer ===
GPTQModel version 4.0.0-dev detected - compatible

1. Applying model patches...
   Model patches applied successfully

2. Applying quantizer patches...
   Quantizer patches applied successfully

3. Applying utility patches...
   Utility patches applied successfully

=== Patch Results ===
Successful: ['model', 'quantizer', 'utils']

SUCCESS: Essential model patches applied!
```

## Architecture Support

Currently supports:
- DualStreamRoformer (qube project)

The patches are designed to be extensible for other custom transformer architectures with minimal modifications. 
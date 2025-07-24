# DualStreamRoFormer Quantization Test

This directory contains testing and debugging tools for the DualStreamRoFormer model quantization support in GPTQModel.

## Files

- `quantize_dual_stream.py`: Main test script for quantization and debugging
- `run_dual_stream_test.sh`: Shell script wrapper for easy execution with uv
- `README_dual_stream_test.md`: This documentation file

## Prerequisites

1. **uv package manager**: Install from [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
2. **DualStreamRoFormer model**: A trained model following the cube3d architecture
3. **GPU with CUDA support**: For quantization (CPU mode may work but is slower)

## Quick Start

### 1. Model Analysis Only (Recommended first step)

```bash
# Make the script executable
chmod +x run_dual_stream_test.sh

# Run analysis to understand model structure
./run_dual_stream_test.sh --model-path /path/to/your/model --analysis-only
```

### 2. Full Quantization Test

```bash
# Run complete quantization test
./run_dual_stream_test.sh --model-path /path/to/your/model --output-path ./quantized_output
```

### 3. Custom Configuration

```bash
# Run with custom settings
./run_dual_stream_test.sh \
    --model-path /path/to/your/model \
    --output-path ./my_quantized_model \
    --bits 8 \
    --group-size 64 \
    --backend torch \
    --debug
```

## Direct Python Usage

If you prefer to run the Python script directly:

```bash
# Basic usage
uv run python quantize_dual_stream.py --model-path /path/to/model --trust-remote-code

# With custom parameters
uv run python quantize_dual_stream.py \
    --model-path /path/to/model \
    --output-path ./quantized_output \
    --bits 4 \
    --group-size 128 \
    --calibration-samples 256 \
    --backend auto \
    --trust-remote-code \
    --debug
```

## Command Line Options

### Main Script Options (`quantize_dual_stream.py`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-path` | str | **required** | Path to the DualStreamRoFormer model |
| `--output-path` | str | `./quantized_dual_stream` | Output directory for quantized model |
| `--calibration-samples` | int | `128` | Number of calibration samples to generate |
| `--seq-len` | int | `512` | Sequence length for calibration data |
| `--bits` | int | `4` | Quantization bits (4, 8, 16) |
| `--group-size` | int | `128` | Group size for quantization |
| `--backend` | str | `auto` | Backend: auto, torch, triton, marlin |
| `--trust-remote-code` | flag | `False` | Trust remote code (required for custom models) |
| `--analysis-only` | flag | `False` | Only run model analysis, skip quantization |
| `--debug` | flag | `False` | Enable debug logging |

### Shell Script Options (`run_dual_stream_test.sh`)

The shell script provides the same options with simplified syntax and additional validation.

## Output Files

When you run the test, the following files will be created:

1. **`quantize_dual_stream.log`**: Detailed logging output
2. **`quantized_dual_stream/`** (or custom output path):
   - `model.safetensors` or `model.safetensors.index.json`: Quantized model weights
   - `config.json`: Model configuration
   - `quantize_config.json`: Quantization configuration
   - `tokenizer.json`, `tokenizer_config.json`: Tokenizer files (if available)

## What the Test Does

### 1. Model Analysis
- Loads the DualStreamRoFormer model
- Analyzes model structure and parameters
- Identifies quantizable modules
- Reports layer types and parameter counts

### 2. Quantization Process
- Creates dummy calibration data with 3D shape prompts
- Applies GPTQ quantization to linear layers
- Tests different quantization backends
- Validates quantized model structure

### 3. Testing and Validation
- Attempts basic inference tests
- Saves quantized model for later use
- Provides detailed logging and error reporting

## Expected Model Structure

The test expects a DualStreamRoFormer model with the following structure:

```
DualStreamRoformer/
├── text_proj              # Text embedding projection
├── shape_proj             # Shape embedding projection
├── transformer/
│   ├── wte                # Word token embeddings
│   ├── dual_blocks/       # Dual-stream transformer layers
│   │   ├── [0-22]/        # 23 dual-stream layers
│   │   │   ├── attn/      # Dual-stream attention
│   │   │   ├── post_1/    # MLP for stream 1
│   │   │   └── post_2/    # MLP for stream 2
│   ├── single_blocks/     # Single-stream transformer layers
│   │   └── [0]/           # 1 single-stream layer
│   └── ln_f               # Final layer normalization
└── lm_head                # Language modeling head
```

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure the model path exists and contains the expected files
2. **CUDA Out of Memory**: Reduce batch size or use CPU mode
3. **Trust Remote Code**: Always use `--trust-remote-code` for custom models
4. **Backend Issues**: Try different backends (torch, triton, marlin) if auto fails

### Debug Mode

Enable debug mode for detailed information:

```bash
./run_dual_stream_test.sh --model-path /path/to/model --debug
```

This will:
- Show detailed parameter analysis
- Log quantization progress
- Report module-level information
- Save verbose logs to `quantize_dual_stream.log`

### Log Files

Check the log file for detailed information:

```bash
tail -f quantize_dual_stream.log
```

## Integration with GPTQModel

Once quantization is successful, you can use the quantized model with GPTQModel:

```python
from gptqmodel import GPTQModel

# Load quantized model
model = GPTQModel.from_quantized(
    "path/to/quantized_dual_stream", 
    trust_remote_code=True
)

# Use for inference
# Note: Actual inference API depends on the specific model implementation
```

## Contributing

If you encounter issues or want to improve the test:

1. Check the logs for detailed error information
2. Verify the model structure matches expectations
3. Test with different quantization parameters
4. Report issues with model path, configuration, and error logs

## Dependencies

The test script automatically manages dependencies through uv, but key requirements include:

- `torch`: PyTorch framework
- `transformers`: HuggingFace Transformers
- `safetensors`: Safe tensor format
- `gptqmodel`: GPTQModel quantization library

These are automatically installed when using `uv run`. 
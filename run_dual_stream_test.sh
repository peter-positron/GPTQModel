#!/bin/bash

# Test script runner for DualStreamRoFormer quantization with uv
# This script demonstrates how to run the quantization test with proper uv dependency management

set -e  # Exit on any error

echo "DualStreamRoFormer Quantization Test Runner"
echo "=========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH"
    echo "Please install uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Default values
MODEL_PATH=""
OUTPUT_PATH="./quantized_dual_stream"
ANALYSIS_ONLY=false
DEBUG=false
BITS=4
GROUP_SIZE=128
BACKEND="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --analysis-only)
            ANALYSIS_ONLY=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --bits)
            BITS="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --model-path <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model-path <path>     Path to DualStreamRoFormer model (required)"
            echo "  --output-path <path>    Output path for quantized model (default: ./quantized_dual_stream)"
            echo "  --analysis-only         Only run model analysis, skip quantization"
            echo "  --debug                 Enable debug logging"
            echo "  --bits <int>            Quantization bits (default: 4)"
            echo "  --group-size <int>      Group size for quantization (default: 128)"
            echo "  --backend <string>      Backend: auto, torch, triton, marlin (default: auto)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model-path /path/to/model --analysis-only"
            echo "  $0 --model-path /path/to/model --bits 8 --group-size 64"
            echo "  $0 --model-path /path/to/model --backend torch --debug"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if model path exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Output Path: $OUTPUT_PATH"
echo "  Bits: $BITS"
echo "  Group Size: $GROUP_SIZE"
echo "  Backend: $BACKEND"
echo "  Analysis Only: $ANALYSIS_ONLY"
echo "  Debug: $DEBUG"
echo ""

# Build the command
CMD="uv run --with gptqmodel[dev] python quantize_dual_stream.py --model-path \"$MODEL_PATH\" --output-path \"$OUTPUT_PATH\" --bits $BITS --group-size $GROUP_SIZE --backend $BACKEND --trust-remote-code"

if [[ "$ANALYSIS_ONLY" == "true" ]]; then
    CMD="$CMD --analysis-only"
fi

if [[ "$DEBUG" == "true" ]]; then
    CMD="$CMD --debug"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD

echo ""
echo "Test completed. Check quantize_dual_stream.log for detailed logs." 
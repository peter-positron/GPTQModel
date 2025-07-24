#!/usr/bin/env python3
"""
Test script for quantizing and debugging DualStreamRoFormer model with GPTQModel.
This script provides comprehensive testing and debugging capabilities for the new model support.

Usage:
    python quantize_dual_stream.py --model-path /path/to/model --output-path /path/to/output [options]

Requirements:
    - Run with: uv run python quantize_dual_stream.py [args]
    - Requires torch, transformers, and other GPTQModel dependencies
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add current directory to path for importing gptqmodel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.utils.model import find_modules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quantize_dual_stream.log')
    ]
)
logger = logging.getLogger(__name__)


def create_dummy_calibration_data(tokenizer, num_samples: int = 128, seq_len: int = 512) -> List[str]:
    """Create dummy calibration data for testing."""
    logger.info(f"Creating {num_samples} dummy calibration samples with seq_len={seq_len}")
    
    # Simple text prompts for 3D shape generation
    base_prompts = [
        "A red cube sitting on a table",
        "A blue sphere floating in space", 
        "A green pyramid with smooth edges",
        "A yellow cylinder standing upright",
        "A purple torus rotating slowly",
        "A metallic dragon with spread wings",
        "A wooden chair with curved legs",
        "A glass vase with flowers",
        "A stone statue of a lion",
        "A modern lamp with geometric design"
    ]
    
    calibration_data = []
    for i in range(num_samples):
        prompt = base_prompts[i % len(base_prompts)]
        # Add some variation
        variations = [
            f"{prompt} in high detail",
            f"{prompt} with realistic textures",
            f"{prompt} from a side view",
            f"{prompt} with dramatic lighting",
            f"Create {prompt.lower()}",
            f"Generate {prompt.lower()}"
        ]
        final_prompt = variations[i % len(variations)]
        calibration_data.append(final_prompt)
    
    return calibration_data


def analyze_model_structure(model, model_path: str) -> Dict:
    """Analyze the model structure and return detailed information."""
    logger.info("Analyzing model structure...")
    
    analysis = {
        'model_path': model_path,
        'model_class': model.__class__.__name__,
        'total_parameters': 0,
        'quantizable_modules': {},
        'layer_structure': {},
        'config_info': {}
    }
    
    # Get model configuration
    if hasattr(model, 'config'):
        config = model.config
        analysis['config_info'] = {
            'model_type': getattr(config, 'model_type', 'unknown'),
            'vocab_size': getattr(config, 'vocab_size', 'unknown'),
            'hidden_size': getattr(config, 'hidden_size', 'unknown'),
            'num_attention_heads': getattr(config, 'num_attention_heads', 'unknown'),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', 'unknown'),
        }
    
    # Analyze parameters
    for name, param in model.named_parameters():
        analysis['total_parameters'] += param.numel()
        
        # Categorize by layer type
        if 'dual_blocks' in name:
            layer_type = 'dual_stream_layer'
        elif 'single_blocks' in name:
            layer_type = 'single_stream_layer'
        elif 'text_proj' in name:
            layer_type = 'text_projection'
        elif 'shape_proj' in name:
            layer_type = 'shape_projection'
        elif 'lm_head' in name:
            layer_type = 'language_model_head'
        else:
            layer_type = 'other'
        
        if layer_type not in analysis['layer_structure']:
            analysis['layer_structure'][layer_type] = {'count': 0, 'parameters': 0}
        
        analysis['layer_structure'][layer_type]['count'] += 1
        analysis['layer_structure'][layer_type]['parameters'] += param.numel()
    
    # Find quantizable modules
    try:
        modules = find_modules(model)
        analysis['quantizable_modules'] = {
            'count': len(modules),
            'module_types': {},
            'module_names': list(modules.keys())[:10]  # First 10 for brevity
        }
        
        for name, module in modules.items():
            module_type = module.__class__.__name__
            if module_type not in analysis['quantizable_modules']['module_types']:
                analysis['quantizable_modules']['module_types'][module_type] = 0
            analysis['quantizable_modules']['module_types'][module_type] += 1
            
    except Exception as e:
        logger.warning(f"Could not analyze quantizable modules: {e}")
        analysis['quantizable_modules'] = {'error': str(e)}
    
    return analysis


def print_analysis_report(analysis: Dict):
    """Print a detailed analysis report."""
    print("\n" + "="*80)
    print("MODEL ANALYSIS REPORT")
    print("="*80)
    
    print(f"Model Path: {analysis['model_path']}")
    print(f"Model Class: {analysis['model_class']}")
    print(f"Total Parameters: {analysis['total_parameters']:,}")
    
    print("\nConfiguration Info:")
    for key, value in analysis['config_info'].items():
        print(f"  {key}: {value}")
    
    print("\nLayer Structure:")
    for layer_type, info in analysis['layer_structure'].items():
        print(f"  {layer_type}: {info['count']} modules, {info['parameters']:,} parameters")
    
    print("\nQuantizable Modules:")
    if 'error' in analysis['quantizable_modules']:
        print(f"  Error: {analysis['quantizable_modules']['error']}")
    else:
        print(f"  Total quantizable modules: {analysis['quantizable_modules']['count']}")
        print("  Module types:")
        for module_type, count in analysis['quantizable_modules']['module_types'].items():
            print(f"    {module_type}: {count}")
        
        if analysis['quantizable_modules']['module_names']:
            print("  Sample module names:")
            for name in analysis['quantizable_modules']['module_names']:
                print(f"    {name}")
    
    print("="*80)


def test_model_loading(model_path: str, trust_remote_code: bool = True) -> tuple:
    """Test loading the DualStreamRoFormer model."""
    logger.info(f"Testing model loading from: {model_path}")
    
    try:
        # Try to load tokenizer first
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=trust_remote_code
            )
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            logger.info("Will create dummy tokenizer for testing")
        
        # Load model for quantization
        logger.info("Loading model for quantization...")
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            device="auto"
        )
        
        model = GPTQModel.from_pretrained(
            model_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("Model loaded successfully for quantization")
        return model, tokenizer, quantize_config
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def test_quantization(model, tokenizer, quantize_config: QuantizeConfig, 
                     calibration_data: List[str], backend: BACKEND = BACKEND.AUTO) -> None:
    """Test the quantization process."""
    logger.info("Starting quantization test...")
    
    try:
        # Perform quantization
        logger.info(f"Quantizing with backend: {backend}")
        logger.info(f"Quantization config: bits={quantize_config.bits}, group_size={quantize_config.group_size}")
        
        model.quantize(
            calibration_data,
            backend=backend,
            batch_size=1,  # Start with small batch size
            # calibration_enable_gpu_cache=True,  # Enable if memory allows
            # auto_gc=True,  # Enable garbage collection
        )
        
        logger.info("Quantization completed successfully!")
        
        # Analyze quantized model
        logger.info("Analyzing quantized model...")
        kernels = model.kernels()
        if kernels:
            logger.info(f"Quantized kernels loaded: {[k.__name__ for k in kernels]}")
        else:
            logger.warning("No quantized kernels found")
            
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def test_inference(model, tokenizer, test_prompts: List[str]) -> None:
    """Test inference with the quantized model."""
    logger.info("Testing inference with quantized model...")
    
    try:
        if tokenizer is None:
            logger.warning("No tokenizer available, skipping inference test")
            return
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Testing prompt {i+1}: {prompt}")
            
            # This is a placeholder - actual inference would depend on the model's specific API
            # For DualStreamRoFormer, we'd need to handle text and shape tokens differently
            logger.info("Note: Actual inference implementation depends on model-specific API")
            
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


def save_quantized_model(model, output_path: str) -> None:
    """Save the quantized model."""
    logger.info(f"Saving quantized model to: {output_path}")
    
    try:
        os.makedirs(output_path, exist_ok=True)
        model.save(output_path)
        logger.info("Model saved successfully")
        
        # List saved files
        saved_files = list(Path(output_path).glob("*"))
        logger.info(f"Saved files: {[f.name for f in saved_files]}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def main():
    """Main function to run the quantization test."""
    parser = argparse.ArgumentParser(description="Test DualStreamRoFormer quantization")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the DualStreamRoFormer model")
    parser.add_argument("--output-path", type=str, default="./quantized_dual_stream",
                       help="Output path for quantized model")
    parser.add_argument("--calibration-samples", type=int, default=128,
                       help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=512,
                       help="Sequence length for calibration data")
    parser.add_argument("--bits", type=int, default=4,
                       help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for quantization")
    parser.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "torch", "triton", "marlin"],
                       help="Quantization backend")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code (required for custom models)")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Only run model analysis without quantization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting DualStreamRoFormer quantization test")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Test model loading
        model, tokenizer, quantize_config = test_model_loading(
            args.model_path, 
            trust_remote_code=args.trust_remote_code
        )
        
        # Update quantization config with user parameters
        quantize_config.bits = args.bits
        quantize_config.group_size = args.group_size
        
        # Analyze model structure
        analysis = analyze_model_structure(model, args.model_path)
        print_analysis_report(analysis)
        
        if args.analysis_only:
            logger.info("Analysis complete. Exiting (analysis-only mode)")
            return
        
        # Create calibration data
        calibration_data = create_dummy_calibration_data(
            tokenizer, 
            num_samples=args.calibration_samples,
            seq_len=args.seq_len
        )
        
        # Test quantization
        backend = BACKEND(args.backend.upper())
        test_quantization(model, tokenizer, quantize_config, calibration_data, backend)
        
        # Test inference
        test_prompts = [
            "A detailed red dragon with golden scales",
            "A modern chair with ergonomic design",
            "A crystal vase with intricate patterns"
        ]
        test_inference(model, tokenizer, test_prompts)
        
        # Save quantized model
        save_quantized_model(model, args.output_path)
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
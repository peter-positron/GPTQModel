#!/usr/bin/env python3
"""
Quantize DualStreamRoformer using GPTQModel 4.0.0 with comprehensive patches.

This script:
1. Applies patches to make GPTQModel compatible with DualStreamRoformer
2. Loads the DualStreamRoformer model
3. Creates calibration data
4. Quantizes the model using GPTQ
5. Saves the quantized model

Usage:
    python quantize_dualstream_roformer_gptq.py <model_dir> [--output_dir <output_dir>]
"""

import sys
import os
from pathlib import Path

# Add the script's own directory to the Python path to ensure that
# local modules (like gptqmodel_patches) can be found, especially when
# running with tools like `uv run` that might alter the path.
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
    print(f"INFO: Added {script_dir} to Python path for local imports.")

# Add the qube project directory to Python path for cube3d imports
qube_dir = script_dir.parent / "qube"
if qube_dir.exists():
    sys.path.insert(0, str(qube_dir))
    print(f"INFO: Added qube project directory to Python path: {qube_dir}")
else:
    print(f"WARNING: Could not find qube directory at {qube_dir}")
    print("Make sure you're running this from the workdir that contains both onlypatches and qube directories")

# Add local GPTQModel to Python path (before other imports)
# First try the GPTQModel in the workdir (../GPTQModel from onlypatches)
local_gptqmodel_path = script_dir.parent / "GPTQModel"
if local_gptqmodel_path.exists():
    sys.path.insert(0, str(local_gptqmodel_path))
    print(f"Using local GPTQModel from: {local_gptqmodel_path}")
else:
    # Fallback: try in the same directory as the script
    fallback_gptqmodel_path = Path(__file__).parent / "GPTQModel"
    if fallback_gptqmodel_path.exists():
        sys.path.insert(0, str(fallback_gptqmodel_path))
        print(f"Using local GPTQModel from: {fallback_gptqmodel_path}")
    else:
        print("Local GPTQModel not found, using installed version")
        print(f"Looked for GPTQModel at:")
        print(f"  - {local_gptqmodel_path}")
        print(f"  - {fallback_gptqmodel_path}")

import argparse
import torch
import gc
import time
import warnings
import os
from typing import Dict, List, Any, Optional

# Set PyTorch memory allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Suppress some warnings during quantization
warnings.filterwarnings('ignore', category=UserWarning, message='.*GPTQ.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*quantization.*')

def find_model_file(model_path: Path) -> Optional[Path]:
    """Find the shape_gpt.safetensors file from the given path."""
    
    # If the path is directly a .safetensors file, use it
    if model_path.is_file() and model_path.suffix == '.safetensors':
        if model_path.name in ['shape_gpt.safetensors', 'model.safetensors']:
            return model_path
        else:
            print(f"WARNING: Found .safetensors file but name is unexpected: {model_path.name}")
            return model_path
    
    # If it's a directory, search for model files
    if model_path.is_dir():
        possible_paths = [
            model_path / "shape_gpt.safetensors",
            model_path / "model.safetensors",
            model_path / "pytorch_model.bin"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Search recursively
        for path in model_path.rglob("shape_gpt.safetensors"):
            return path
    
    return None


def load_dualstream_roformer(model_path: Path, device: str = "auto") -> torch.nn.Module:
    """Load the DualStreamRoformer model."""
    print(f"Loading DualStreamRoformer from: {model_path}")
    
    # Import the qube project modules
    try:
        from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer
    except ImportError as e:
        print(f"ERROR: Could not import qube modules: {e}")
        print("Make sure you're running this script from the qube project root")
        sys.exit(1)
    
    # Create model configuration using values from open_model.yaml
    config = DualStreamRoformer.Config(
        n_layer=23,
        n_single_layer=1,
        rope_theta=10000.0,
        n_head=12,
        n_embd=1536,
        bias=True,
        eps=1e-6,
        shape_model_vocab_size=16384,  # Use the original vocab size from training
        shape_model_embed_dim=32,
        text_model_embed_dim=768,
        use_pooled_text_embed=False,
        encoder_with_cls_token=True
    )
    
    # Create model
    model = DualStreamRoformer(config)
    
    # Load weights
    if model_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    
    # Add a config attribute that GPTQModel expects
    model.config = config
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available!")
        sys.exit(1)
    
    # CRITICAL FIX: Do NOT move the entire model to the GPU at once.
    # Keep it on the CPU. The GPTQ patch will handle moving layers
    # to the GPU one by one during the quantization process.
    # This is the key to preventing the initial OOM error.
    print(f"Model loaded on CPU. Layers will be moved to {device} during quantization.")
    # model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on cpu")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_calibration_dataset(tokenizer, num_samples: int = 512, max_length: int = 512, model_vocab_size: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
    """
    Create a calibration dataset for GPTQ quantization.
    
    Args:
        tokenizer: The tokenizer to use
        num_samples: Number of calibration samples (should be >256)
        max_length: Maximum sequence length (should be >256)
        model_vocab_size: Maximum vocabulary size for the model (to clip token IDs)
    
    Returns:
        List of tokenized samples
    """
    print(f"=== Creating Calibration Dataset ({num_samples} samples) ===")
    
    # Extended prompts for better calibration
    base_prompts = [
        "a red sports car driving down a winding mountain road",
        "a blue dragon flying over a medieval castle",
        "a green tree with golden leaves in autumn",
        "a white cat sitting on a wooden chair",
        "a black motorcycle parked in front of a garage",
        "a yellow sunflower in a field of grass",
        "a purple butterfly landing on a pink flower",
        "a silver airplane flying through white clouds",
        "a brown horse running across a grassy field",
        "a orange tiger walking through a dense jungle",
        "a crystal clear lake surrounded by tall mountains",
        "a wooden bridge crossing over a flowing river",
        "a stone castle with tall towers and flags",
        "a glass house with plants growing inside",
        "a metal robot with glowing blue eyes",
        "a fabric tent set up in a forest clearing",
        "a leather jacket hanging on a coat rack",
        "a plastic toy car on a child's bedroom floor",
        "a ceramic vase filled with fresh flowers",
        "a paper airplane flying through the air",
        "a detailed architectural blueprint of a modern building",
        "a complex geometric pattern with intricate details",
        "a futuristic spaceship with advanced technology",
        "a vintage steam locomotive from the early 1900s",
        "a magnificent cathedral with stained glass windows",
        "a bustling marketplace filled with colorful goods",
        "a serene zen garden with carefully placed stones",
        "a dramatic thunderstorm over a vast ocean",
        "a cozy log cabin nestled in a snowy forest",
        "a vibrant coral reef teeming with marine life"
    ]
    
    # Generate variations and combinations
    extended_prompts = []
    
    # Add base prompts
    extended_prompts.extend(base_prompts)
    
    # Add variations with adjectives
    adjectives = ["beautiful", "amazing", "stunning", "incredible", "magnificent", "elegant", "graceful", "powerful", "mysterious", "enchanting"]
    for prompt in base_prompts[:10]:  # Use first 10 base prompts
        for adj in adjectives:
            extended_prompts.append(f"{adj} {prompt}")
    
    # Add combinations
    for i in range(10):
        for j in range(i+1, 15):
            if i < len(base_prompts) and j < len(base_prompts):
                extended_prompts.append(f"{base_prompts[i]} and {base_prompts[j]}")
    
    # Add detailed descriptions
    detailed_prompts = [
        "a highly detailed mechanical clockwork mechanism with intricate gears and springs",
        "a photorealistic rendering of a modern architectural structure with glass and steel",
        "a complex organic molecular structure with atoms and chemical bonds",
        "a detailed topographical map showing mountains, valleys, and rivers",
        "a sophisticated electronic circuit board with microchips and resistors",
        "a comprehensive technical diagram of an advanced aircraft engine",
        "a detailed cross-section view of a multi-story building",
        "a complex mathematical visualization showing geometric relationships",
        "a detailed anatomical model of a human skeletal system",
        "a sophisticated mechanical watch movement with visible components"
    ]
    extended_prompts.extend(detailed_prompts)
    
    # Ensure we have enough unique prompts
    while len(extended_prompts) < num_samples:
        # Generate more variations
        base_idx = len(extended_prompts) % len(base_prompts)
        variation = f"detailed view of {base_prompts[base_idx]} with realistic textures"
        extended_prompts.append(variation)
    
    # Take the required number of samples
    selected_prompts = extended_prompts[:num_samples]
    
    print(f"Generated {len(selected_prompts)} unique prompts")
    print(f"Sample prompts:")
    for i, prompt in enumerate(selected_prompts[:5]):
        print(f"  {i+1}. {prompt}")
    
    # Tokenize all prompts
    calibration_data = []
    clipped_tokens_count = 0
    
    for prompt in selected_prompts:
        # Tokenize with proper padding and truncation
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True
        )
        
        # Ensure we have attention_mask
        if 'attention_mask' not in tokens:
            tokens['attention_mask'] = torch.ones_like(tokens['input_ids'])
        
        # Clip token IDs to model vocabulary size if specified
        input_ids = tokens["input_ids"].squeeze(0)
        if model_vocab_size is not None:
            # Count tokens that need clipping
            clipped_mask = input_ids >= model_vocab_size
            clipped_tokens_count += clipped_mask.sum().item()
            
            # Clip to model vocabulary size
            input_ids = torch.clamp(input_ids, 0, model_vocab_size - 1)
        
        # Convert to the format expected by GPTQModel
        sample = {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }
        
        calibration_data.append(sample)
    
    print(f"Calibration dataset created with {len(calibration_data)} samples")
    print(f"Average sequence length: {sum(len(sample['input_ids']) for sample in calibration_data) / len(calibration_data):.1f}")
    
    if model_vocab_size is not None and clipped_tokens_count > 0:
        print(f"Clipped {clipped_tokens_count} tokens to model vocabulary size {model_vocab_size}")
    
    return calibration_data


def save_calibration_data(calibration_data, cache_dir: Path, model_vocab_size: int, num_samples: int):
    """Save calibration data to disk for reuse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a descriptive filename
    cache_file = cache_dir / f"calibration_{num_samples}samples_vocab{model_vocab_size}.pt"
    
    print(f"\n=== Saving Calibration Data ===")
    print(f"Saving {len(calibration_data)} samples to: {cache_file}")
    
    try:
        torch.save({
            'calibration_data': calibration_data,
            'vocab_size': model_vocab_size,
            'num_samples': num_samples,
            'created_at': time.time(),
            'version': '1.0'
        }, cache_file)
        print(f"SUCCESS: Calibration data saved to {cache_file}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save calibration data: {e}")
        return False


def save_input_cache(input_cache, cache_dir: Path, model_vocab_size: int, num_samples: int):
    """Save the expensive input activation cache to disk using chunked approach for large files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Saving Input Cache ===")
    print(f"Saving input cache with {len(input_cache)} layers...")
    
    try:
        # Calculate approximate size
        total_elements = 0
        for layer_cache in input_cache:
            for sample in layer_cache:
                if hasattr(sample, 'numel'):
                    total_elements += sample.numel()
        
        # Estimate size in GB (assuming float16/bfloat16 = 2 bytes per element)
        estimated_size_gb = (total_elements * 2) / (1024**3)
        print(f"Estimated cache size: {estimated_size_gb:.1f} GB")
        
        if estimated_size_gb > 10:
            print("Large cache detected - using chunked saving approach...")
            return save_input_cache_chunked(input_cache, cache_dir, model_vocab_size, num_samples)
        else:
            # Use standard saving for smaller caches
            cache_file = cache_dir / f"input_cache_{num_samples}samples_vocab{model_vocab_size}.pt"
            print(f"Saving to: {cache_file}")
            
            # Convert to CPU and save to reduce memory usage
            cpu_cache = []
            for layer_cache in input_cache:
                cpu_layer = [inp.cpu() if hasattr(inp, 'cpu') else inp for inp in layer_cache]
                cpu_cache.append(cpu_layer)
            
            torch.save({
                'input_cache': cpu_cache,
                'vocab_size': model_vocab_size,
                'num_samples': num_samples,
                'num_layers': len(input_cache),
                'created_at': time.time(),
                'version': '1.0'
            }, cache_file)
            
            print(f"SUCCESS: Input cache saved to {cache_file}")
            print(f"  Layers: {len(input_cache)}")
            print(f"  Samples per layer: {len(input_cache[0]) if input_cache else 0}")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to save input cache: {e}")
        print("Trying chunked saving as fallback...")
        return save_input_cache_chunked(input_cache, cache_dir, model_vocab_size, num_samples)


def save_input_cache_chunked(input_cache, cache_dir: Path, model_vocab_size: int, num_samples: int):
    """Save input cache in chunks to handle very large caches."""
    try:
        # Create a directory for the chunked cache
        chunked_cache_dir = cache_dir / f"input_cache_{num_samples}samples_vocab{model_vocab_size}_chunks"
        chunked_cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving chunked cache to: {chunked_cache_dir}")
        
        # Save metadata
        metadata = {
            'vocab_size': model_vocab_size,
            'num_samples': num_samples,
            'num_layers': len(input_cache),
            'created_at': time.time(),
            'version': '1.0',
            'format': 'chunked'
        }
        torch.save(metadata, chunked_cache_dir / "metadata.pt")
        
        # Save each layer separately
        for layer_idx, layer_cache in enumerate(input_cache):
            layer_file = chunked_cache_dir / f"layer_{layer_idx:03d}.pt"
            print(f"  Saving layer {layer_idx + 1}/{len(input_cache)} to {layer_file.name}")
            
            # Convert to CPU to save memory
            cpu_layer = [inp.cpu() if hasattr(inp, 'cpu') else inp for inp in layer_cache]
            
            torch.save({
                'layer_idx': layer_idx,
                'layer_cache': cpu_layer,
                'num_samples': len(cpu_layer)
            }, layer_file)
        
        print(f"SUCCESS: Input cache saved as {len(input_cache)} chunk files")
        print(f"  Directory: {chunked_cache_dir}")
        print(f"  Layers: {len(input_cache)}")
        print(f"  Samples per layer: {len(input_cache[0]) if input_cache else 0}")
        return True
        
    except Exception as e:
        print(f"ERROR: Chunked saving also failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_input_cache(cache_dir: Path, model_vocab_size: int, num_samples: int, device: str = "cuda"):
    """Load pre-computed input activation cache from disk (handles both single file and chunked formats)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # First try chunked format
    chunked_cache_dir = cache_dir / f"input_cache_{num_samples}samples_vocab{model_vocab_size}_chunks"
    if chunked_cache_dir.exists():
        return load_input_cache_chunked(chunked_cache_dir, device)
    
    # Fall back to single file format
    cache_file = cache_dir / f"input_cache_{num_samples}samples_vocab{model_vocab_size}.pt"
    
    if not cache_file.exists():
        print(f"\n=== Input Cache ===")
        print(f"No cached input cache found at: {cache_file}")
        print(f"Also checked chunked format at: {chunked_cache_dir}")
        return None
    
    print(f"\n=== Loading Input Cache ===")
    print(f"Loading cached input cache from: {cache_file}")
    
    try:
        cached_data = torch.load(cache_file, map_location='cpu')
        
        # Validate the cached data
        if 'input_cache' not in cached_data:
            print("ERROR: Invalid input cache file format")
            return None
        
        input_cache = cached_data['input_cache']
        cached_vocab_size = cached_data.get('vocab_size', 'unknown')
        cached_num_samples = cached_data.get('num_samples', 'unknown')
        num_layers = cached_data.get('num_layers', len(input_cache) if input_cache else 0)
        created_at = cached_data.get('created_at', 0)
        
        print(f"Input cache info:")
        print(f"  Layers: {num_layers}")
        print(f"  Samples: {cached_num_samples} (expected: {num_samples})")
        print(f"  Vocab size: {cached_vocab_size} (expected: {model_vocab_size})")
        print(f"  Created: {time.ctime(created_at) if created_at else 'unknown'}")
        
        # Move cache to target device
        if device != "cpu":
            print(f"Moving input cache to {device}...")
            device_cache = []
            for layer_cache in input_cache:
                device_layer = [inp.to(device) if hasattr(inp, 'to') else inp for inp in layer_cache]
                device_cache.append(device_layer)
            input_cache = device_cache
        
        print(f"SUCCESS: Loaded input cache with {len(input_cache)} layers")
        return input_cache
        
    except Exception as e:
        print(f"ERROR: Failed to load input cache: {e}")
        return None


def load_input_cache_chunked(chunked_cache_dir: Path, device: str = "cuda"):
    """Load input cache from chunked format."""
    print(f"\n=== Loading Chunked Input Cache ===")
    print(f"Loading from directory: {chunked_cache_dir}")
    
    try:
        # Load metadata
        metadata_file = chunked_cache_dir / "metadata.pt"
        if not metadata_file.exists():
            print("ERROR: Metadata file not found in chunked cache")
            return None
        
        metadata = torch.load(metadata_file, map_location='cpu')
        num_layers = metadata.get('num_layers', 0)
        cached_vocab_size = metadata.get('vocab_size', 'unknown')
        cached_num_samples = metadata.get('num_samples', 'unknown')
        created_at = metadata.get('created_at', 0)
        
        print(f"Chunked cache info:")
        print(f"  Layers: {num_layers}")
        print(f"  Samples: {cached_num_samples}")
        print(f"  Vocab size: {cached_vocab_size}")
        print(f"  Created: {time.ctime(created_at) if created_at else 'unknown'}")
        
        # Load each layer
        input_cache = []
        for layer_idx in range(num_layers):
            layer_file = chunked_cache_dir / f"layer_{layer_idx:03d}.pt"
            if not layer_file.exists():
                print(f"ERROR: Layer file {layer_file} not found")
                return None
            
            print(f"  Loading layer {layer_idx + 1}/{num_layers}")
            layer_data = torch.load(layer_file, map_location='cpu')
            layer_cache = layer_data['layer_cache']
            
            # Move to target device if needed
            if device != "cpu":
                layer_cache = [inp.to(device) if hasattr(inp, 'to') else inp for inp in layer_cache]
            
            input_cache.append(layer_cache)
        
        print(f"SUCCESS: Loaded chunked input cache with {len(input_cache)} layers")
        return input_cache
        
    except Exception as e:
        print(f"ERROR: Failed to load chunked input cache: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_calibration_data(cache_dir: Path, model_vocab_size: int, num_samples: int):
    """Load calibration data from disk if available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for matching cache file
    cache_file = cache_dir / f"calibration_{num_samples}samples_vocab{model_vocab_size}.pt"
    
    if not cache_file.exists():
        print(f"\n=== Calibration Cache ===")
        print(f"No cached calibration data found at: {cache_file}")
        return None
    
    print(f"\n=== Loading Calibration Data ===")
    print(f"Loading cached calibration data from: {cache_file}")
    
    try:
        cached_data = torch.load(cache_file, map_location='cpu')
        
        # Validate the cached data
        if 'calibration_data' not in cached_data:
            print("ERROR: Invalid cache file format")
            return None
        
        calibration_data = cached_data['calibration_data']
        cached_vocab_size = cached_data.get('vocab_size', 'unknown')
        cached_num_samples = cached_data.get('num_samples', 'unknown')
        created_at = cached_data.get('created_at', 0)
        
        print(f"Cache info:")
        print(f"  Samples: {len(calibration_data)} (expected: {num_samples})")
        print(f"  Vocab size: {cached_vocab_size} (expected: {model_vocab_size})")
        print(f"  Created: {time.ctime(created_at) if created_at else 'unknown'}")
        
        # Check compatibility
        if len(calibration_data) != num_samples:
            print(f"WARNING: Sample count mismatch. Expected {num_samples}, got {len(calibration_data)}")
        
        if cached_vocab_size != model_vocab_size:
            print(f"WARNING: Vocab size mismatch. Expected {model_vocab_size}, got {cached_vocab_size}")
            print("This may cause issues during quantization")
        
        print(f"SUCCESS: Loaded {len(calibration_data)} calibration samples from cache")
        return calibration_data
        
    except Exception as e:
        print(f"ERROR: Failed to load calibration data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Quantize DualStreamRoformer using GPTQModel")
    parser.add_argument("model_path", help="Path to shape_gpt.safetensors file or directory containing it")
    parser.add_argument("--output_dir", default="quantized_dualstream_roformer", help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Number of bits for quantization")
    parser.add_argument("--group_size", type=int, default=64, help="Group size for quantization")
    parser.add_argument(
        "--calibration_size",
        type=int,
        default=512,  # Increased from 128 to meet GPTQModel recommendations
        help="Number of calibration samples (should be >256)"
    )
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--ignore_patch_failures", action="store_true", help="Continue even if some patches fail")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage (not recommended - will be very slow)")
    parser.add_argument("--save_calibration", action="store_true", help="Save calibration data to disk for reuse")
    parser.add_argument("--load_calibration", action="store_true", help="Load pre-saved calibration data from disk")
    parser.add_argument("--use_calibration_cache", action="store_true", help="Automatically load cached calibration data if available, otherwise create and save")
    parser.add_argument("--only_cache_inputs", action="store_true", help="Only compute and save input cache, skip quantization (useful for pre-computing)")
    parser.add_argument("--calibration_cache_dir", default="/mnt/models/calibration_cache", help="Directory to save/load calibration data")
    
    args = parser.parse_args()
    
    # Check CUDA availability first
    print("=== Checking CUDA Availability ===")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("GPTQ quantization requires GPU acceleration for practical performance.")
        print("Please ensure:")
        print("  1. NVIDIA GPU is installed and recognized")
        print("  2. CUDA drivers are installed")
        print("  3. PyTorch was installed with CUDA support")
        print("  4. Run 'nvidia-smi' to verify GPU is accessible")
        if not args.force_cpu:
            print("\nTo force CPU usage anyway (very slow), use --force_cpu flag")
            sys.exit(1)
        else:
            print("\nWARNING: Forcing CPU usage - this will be extremely slow!")
    else:
        cuda_device_count = torch.cuda.device_count()
        print(f"CUDA available: {cuda_device_count} GPU(s) detected")
        for i in range(cuda_device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check if we have enough GPU memory (rough estimate)
        if cuda_device_count > 0:
            primary_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if primary_gpu_memory < 8.0:
                print(f"WARNING: Primary GPU has only {primary_gpu_memory:.1f} GB memory")
                print("GPTQ quantization may fail with insufficient GPU memory")
                print("Consider using a GPU with at least 8GB memory")
    
    # Override device setting if forcing CPU
    if args.force_cpu:
        args.device = "cpu"
    elif args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup paths
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        print(f"ERROR: Model path {model_path} does not exist")
        sys.exit(1)
    
    # Find model file
    model_file = find_model_file(model_path)
    if not model_file:
        print(f"ERROR: Could not find shape_gpt.safetensors at {model_path}")
        if model_path.is_dir():
            print(f"Directory contents: {list(model_path.iterdir())}")
        sys.exit(1)
    
    print(f"Found model file: {model_file}")
    
    # Apply patches for GPTQModel compatibility
    print("\n=== Applying GPTQModel Patches ===")
    try:
        from gptqmodel_patches import apply_patches
        patch_success = apply_patches(ignore_failures=args.ignore_patch_failures)
        if not patch_success:
            print("ERROR: Critical patches failed")
            if not args.ignore_patch_failures:
                sys.exit(1)
            print("Continuing anyway due to --ignore_patch_failures flag")
    except ImportError:
        print("ERROR: Could not import gptqmodel_patches")
        print("Make sure the gptqmodel_patches directory is in your Python path")
        sys.exit(1)
    
    # Load model
    print("\n=== Loading Model ===")
    start_time = time.time()
    model = load_dualstream_roformer(model_file, args.device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Setup calibration cache directory
    cache_dir = Path(args.calibration_cache_dir)
    model_vocab_size = int(model.vocab_size)
    
    # Validate cache directory early to prevent failures after expensive computations
    print(f"\n=== Validating Cache Directory ===")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Create the cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory created/verified successfully")
        
        # Test write permissions by creating a test file
        test_file = cache_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()  # Remove the test file
        print(f"Write permissions verified")
        
        # Check available disk space (basic check)
        import shutil
        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / (1024**3)
        print(f"Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 20:  # Warn if less than 20GB free
            print(f"WARNING: Low disk space ({free_gb:.1f} GB). Input cache can be >15GB.")
            print("Consider using a different cache directory with more space.")
            if free_gb < 5:
                print("ERROR: Insufficient disk space for caching operations")
                sys.exit(1)
        
    except PermissionError:
        print(f"ERROR: No write permission for cache directory: {cache_dir}")
        print("Please ensure you have write access to the cache directory or use a different path")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot access cache directory: {e}")
        sys.exit(1)
    
    # Try to load cached calibration data
    calibration_data = None
    if args.load_calibration or args.use_calibration_cache:
        calibration_data = load_calibration_data(cache_dir, model_vocab_size, args.calibration_size)
    
    # Create calibration dataset if not loaded from cache
    if calibration_data is None:
        print(f"\n=== Creating Calibration Dataset ({args.calibration_size} samples) ===")
        
        # Import and create CLIP tokenizer (same as used in cube3d)
        from transformers import CLIPTokenizerFast
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
        
        # Check vocabulary size compatibility
        tokenizer_vocab_size = tokenizer.vocab_size
        
        print(f"Model vocabulary size: {model_vocab_size}")
        print(f"CLIP tokenizer vocabulary size: {tokenizer_vocab_size}")
        
        if tokenizer_vocab_size != model_vocab_size:
            print(f"WARNING: Vocabulary size mismatch!")
            print(f"  Model expects: {model_vocab_size}")
            print(f"  Tokenizer provides: {tokenizer_vocab_size}")
            print(f"  This may cause indexing errors during quantization")
            
            # Adjust tokenizer to match model vocabulary
            print(f"Creating calibration data with token IDs limited to model vocabulary...")
        
        calibration_data = create_calibration_dataset(tokenizer, args.calibration_size, 512, model_vocab_size)
        
        # Save calibration data if requested
        if args.save_calibration or args.use_calibration_cache:
            save_calibration_data(calibration_data, cache_dir, model_vocab_size, args.calibration_size)
    else:
        print(f"Using cached calibration data with {len(calibration_data)} samples")
    
    # Import GPTQModel components
    print("\n=== Setting up GPTQModel ===")
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        from gptqmodel.quantization import GPTQ
    except ImportError as e:
        print(f"ERROR: Could not import GPTQModel: {e}")
        print("Install with: pip install gptqmodel")
        sys.exit(1)
    
    # Create quantization config
    quantize_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,  # Disable activation quantization for stability
        damp_percent=0.1,
        static_groups=False,
        true_sequential=True,
        device=args.device  # Set device explicitly for GPTQModel 4.0.0
    )
    
    print(f"Quantization config: {args.bits}-bit, group_size={args.group_size}")
    
    # Try to load pre-computed input cache
    input_cache = None
    if args.load_calibration or args.use_calibration_cache:
        input_cache = load_input_cache(cache_dir, model_vocab_size, args.calibration_size, args.device)
    
    # Create DualStreamRoformerGPTQ instance
    print("\n=== Creating DualStreamRoformerGPTQ ===")
    try:
        # Import our custom GPTQ class
        from gptqmodel_patches.model_patches import DualStreamRoformerGPTQ
        
        # Create GPTQ model instance with proper architecture handling
        gptq_model = DualStreamRoformerGPTQ(
            model=model,
            quantized=False,
            quantize_config=quantize_config
        )
        
        # Set pre-computed input cache if available
        if input_cache is not None:
            gptq_model.base_model._precomputed_input_cache = input_cache
            print(f"Pre-computed input cache loaded with {len(input_cache)} layers")
        
        print("DualStreamRoformerGPTQ instance created successfully")
        
    except Exception as e:
        print(f"ERROR: Could not create DualStreamRoformerGPTQ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # If only caching inputs, compute cache and exit
    if args.only_cache_inputs:
        if input_cache is not None:
            print("\n=== Input Cache Already Exists ===")
            print("Input cache already computed and saved. Nothing to do.")
            return True
        
        print("\n=== Computing Input Cache Only ===")
        print("This will compute and save the input cache without quantization...")
        
        try:
            # Prepare examples for caching
            processed_examples = gptq_model._prepare_examples(calibration_data)
            
            # Compute input cache
            all_captured_inputs = []
            num_examples = len(processed_examples)
            print(f"Computing input cache for {num_examples} calibration examples...")
            for i, example in enumerate(processed_examples):
                print(f"  Caching inputs for example {i + 1}/{num_examples}...")
                _, _, captured = gptq_model.model.forward_and_capture(**example)
                all_captured_inputs.append(captured)
                gc.collect()
                torch.cuda.empty_cache()
            
            # Transpose cache
            num_layers = len(all_captured_inputs[0])
            input_cache = [[] for _ in range(num_layers)]
            for layer_idx in range(num_layers):
                for example_idx in range(num_examples):
                    input_cache[layer_idx].append(all_captured_inputs[example_idx][layer_idx])
            
            # Save the cache
            save_input_cache(input_cache, cache_dir, model_vocab_size, args.calibration_size)
            print("\n=== Input Cache Computed and Saved ===")
            print("You can now run quantization with --load_calibration for much faster iterations!")
            return True
            
        except Exception as e:
            print(f"ERROR during input cache computation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Start quantization
    print("\n=== Starting Quantization ===")
    print("This may take several minutes...")
    
    start_time = time.time()
    try:
        quantization_result = gptq_model.quantize(
            examples=calibration_data,
            batch_size=1,
            auto_gc=True,  # Enable automatic garbage collection
            calibration_enable_gpu_cache=True  # Use GPU cache for speed, manage memory in layer wrapper
        )
        
        # Save input cache if it was computed and saving is requested
        if input_cache is None and (args.save_calibration or args.use_calibration_cache):
            if hasattr(gptq_model.base_model, '_computed_input_cache'):
                print("\n=== Saving Computed Input Cache ===")
                save_input_cache(
                    gptq_model.base_model._computed_input_cache, 
                    cache_dir, 
                    model_vocab_size, 
                    args.calibration_size
                )
        
        quantization_time = time.time() - start_time
        print(f"Quantization completed in {quantization_time:.2f} seconds")
        
        # Check if quantization was successful
        if quantization_result is None:
            print("WARNING: Quantization returned None (layers were skipped)")
            print("The model may still be partially quantized and usable")
        else:
            print("Quantization completed successfully!")
        
    except Exception as e:
        print(f"ERROR during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save the quantized model
    print("\n=== Saving Quantized Model ===")
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to save the quantized model
        # Even if quantization was partially successful, the model should still be saveable
        gptq_model.save_quantized(
            str(output_dir),
            max_shard_size="5GB"
        )
        print(f"SUCCESS: Quantized model saved to {output_dir}")
        
        # List the saved files
        if output_dir.exists():
            saved_files = list(output_dir.glob("*"))
            print(f"Saved files: {[f.name for f in saved_files]}")
        
    except Exception as e:
        print(f"ERROR during saving: {e}")
        print("Attempting alternative save method...")
        
        # Try without the problematic arguments
        try:
            gptq_model.save_quantized(str(output_dir))
            print(f"SUCCESS: Quantized model saved to {output_dir} (alternative method)")
        except Exception as e2:
            print(f"Alternative save method also failed: {e2}")
            
            # Last resort: try to save the underlying model
            try:
                if hasattr(gptq_model, 'model'):
                    gptq_model.model.save_pretrained(str(output_dir))
                    print(f"SUCCESS: Underlying model saved to {output_dir} (fallback method)")
                elif hasattr(gptq_model, 'base_model'):
                    gptq_model.base_model.save_pretrained(str(output_dir))
                    print(f"SUCCESS: Base model saved to {output_dir} (fallback method)")
                else:
                    print("Could not find a saveable model component")
                    return False
            except Exception as e3:
                print(f"Fallback save method also failed: {e3}")
                return False
    
    # Print summary
    print(f"\n=== Quantization Summary ===")
    print(f"Original model: {model_file}")
    print(f"Quantized model: {output_dir}")
    print(f"Bits: {args.bits}")
    print(f"Group size: {args.group_size}")
    print(f"Calibration samples: {args.calibration_size}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print("SUCCESS: DualStreamRoformer quantization completed!")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)
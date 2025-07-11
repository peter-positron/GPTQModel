#!/usr/bin/env python3
"""
Example usage of GPTQModel patches for DualStreamRoformer quantization.

This script demonstrates how to use the patches to quantize a DualStreamRoformer
model using GPTQModel with proper handling of the custom architecture.
"""

import sys
import torch
from pathlib import Path

# Add the current directory to path so we can import our patches
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main function demonstrating DualStreamRoformer quantization with GPTQModel."""
    
    print("=== DualStreamRoformer GPTQ Quantization Example ===\n")
    
    # Step 1: Apply patches
    print("1. Applying GPTQModel patches...")
    try:
        from gptqmodel_patches import apply_patches, check_gptqmodel_available
        
        # Check if GPTQModel is available
        if not check_gptqmodel_available():
            print("Please install GPTQModel first: pip install gptqmodel")
            return
        
        # Apply all patches
        apply_patches()
        
    except ImportError as e:
        print(f"Error importing patches: {e}")
        return
    
    # Step 2: Load the model
    print("\n2. Loading DualStreamRoformer model...")
    try:
        # Import after patches are applied
        from gptqmodel import GPTQModel, QuantizeConfig
        
        # Configure quantization
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            damp_percent=0.01,
            static_groups=False
        )
        
        # Load model (replace with your actual model path)
        model_path = "model_weights/shape_gpt.safetensors"
        
        # This would use our patched model loading
        model = GPTQModel.from_pretrained(
            model_path,
            quantize_config=quantize_config,
            config_path="cube3d/configs/open_model.yaml"
        )
        
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This is expected if the model files are not available")
        print("The patches are still applied and ready to use!")
        return
    
    # Step 3: Prepare calibration data
    print("\n3. Preparing calibration data...")
    
    # Generate diverse calibration prompts
    calibration_prompts = [
        "a red dragon",
        "a blue car",
        "a wooden chair",
        "a metal robot",
        "a glass vase",
        "a stone statue",
        "a leather bag",
        "a plastic toy",
        "a ceramic bowl",
        "a fabric pillow",
        "a golden crown",
        "a silver ring",
        "a bronze sculpture",
        "a crystal ball",
        "a wooden table",
        "a metal sword",
        "a glass bottle",
        "a stone bridge",
        "a leather jacket",
        "a plastic container"
          ]
      
      print(f"Generated {len(calibration_prompts)} calibration prompts")
    
    # Step 4: Quantize the model
    print("\n4. Quantizing the model...")
    try:
        # This would use our patched quantization logic
        quantized_model = model.quantize(
            calibration_prompts,
            use_triton=False,  # Set to True if you have Triton installed
            batch_size=1,
            use_cuda_fp16=True,
            autotune_warmup_after_quantized=False
        )
        
        print("Model quantized successfully")
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        return
    
    # Step 5: Save the quantized model
    print("\n5. Saving quantized model...")
    try:
        output_dir = "quantized_model"
        quantized_model.save_quantized(output_dir)
        
        print(f"Quantized model saved to {output_dir}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # Step 6: Test the quantized model
    print("\n6. Testing quantized model...")
    try:
        # Load the quantized model
        loaded_model = GPTQModel.from_quantized(output_dir)
        
        # Test inference (this would need proper input formatting)
        print("Quantized model loaded successfully")
        print("Model is ready for inference!")
        
    except Exception as e:
        print(f"Error testing quantized model: {e}")
        return
    
    print("\n=== Quantization Complete! ===")
    print(f"Your quantized DualStreamRoformer model is saved in: {output_dir}")


def test_patches_only():
    """Test function that only applies patches without loading models."""
    
    print("=== Testing GPTQModel Patches ===\n")
    
    try:
        from gptqmodel_patches import apply_patches, check_gptqmodel_available
        
        # Check GPTQModel availability
        if not check_gptqmodel_available():
            print("GPTQModel not available - install with: pip install gptqmodel")
            return False
        
        # Apply patches
        success = apply_patches()
        
        if success:
            print("\nAll patches applied successfully!")
            print("GPTQModel is now ready to work with DualStreamRoformer")
            return True
        else:
            print("\nSome patches failed to apply")
            return False
            
    except Exception as e:
        print(f"Error during patch testing: {e}")
        return False


def show_usage():
    """Show usage instructions."""
    print("=== GPTQModel Patches for DualStreamRoformer ===\n")
    print("Usage:")
    print("  python example_usage.py                 # Full quantization example")
    print("  python example_usage.py --test-patches  # Test patches only")
    print("  python example_usage.py --help          # Show this help")
    print("\nBefore running:")
    print("  1. Install GPTQModel: pip install gptqmodel")
    print("  2. Ensure you have the DualStreamRoformer model files")
    print("  3. Run from the qube repository root directory")
    print("\nThe patches enable GPTQModel to:")
    print("  • Recognize DualStreamRoformer architecture")
    print("  • Handle custom layer types and structures")
    print("  • Process calibration data appropriately")
    print("  • Apply quantization with proper layer filtering")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-patches":
            test_patches_only()
        elif sys.argv[1] == "--help":
            show_usage()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            show_usage()
    else:
        main() 
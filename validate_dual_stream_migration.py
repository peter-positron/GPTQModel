#!/usr/bin/env python3
"""
Validation script to check if DualStreamRoFormer migration was successful.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print result."""
    if os.path.exists(filepath):
        print(f"[PASS] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} - File not found")
        return False

def check_file_content(filepath, search_strings, description):
    """Check if file contains expected content."""
    if not os.path.exists(filepath):
        print(f"[FAIL] {description}: {filepath} - File not found")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_strings = []
        for search_str in search_strings:
            if search_str not in content:
                missing_strings.append(search_str)
        
        if missing_strings:
            print(f"[FAIL] {description}: Missing content - {missing_strings}")
            return False
        else:
            print(f"[PASS] {description}: Contains expected content")
            return True
            
    except Exception as e:
        print(f"[FAIL] {description}: Error reading file - {e}")
        return False

def main():
    """Main validation function."""
    print("Validating DualStreamRoFormer migration...")
    print("=" * 60)
    
    all_passed = True
    
    # Check if model definition file exists
    model_def_path = "gptqmodel/models/definitions/dual_stream_roformer.py"
    if not check_file_exists(model_def_path, "Model definition file"):
        all_passed = False
    else:
        # Check if model definition contains expected content
        expected_content = [
            "class DualStreamRoFormerGPTQ",
            "BaseGPTQModel",
            "base_modules",
            "layer_modules",
            "transformer.dual_blocks",
            "require_trust_remote_code = True"
        ]
        if not check_file_content(model_def_path, expected_content, "Model definition content"):
            all_passed = False
    
    # Check if __init__.py is updated
    init_path = "gptqmodel/models/definitions/__init__.py"
    if not check_file_exists(init_path, "Definitions __init__.py"):
        all_passed = False
    else:
        expected_imports = [
            "from .dual_stream_roformer import DualStreamRoFormerGPTQ"
        ]
        if not check_file_content(init_path, expected_imports, "Definitions __init__.py import"):
            all_passed = False
    
    # Check if auto.py is updated
    auto_path = "gptqmodel/models/auto.py"
    if not check_file_exists(auto_path, "Auto.py file"):
        all_passed = False
    else:
        expected_auto_content = [
            "from .definitions.dual_stream_roformer import DualStreamRoFormerGPTQ",
            '"dual_stream_roformer": DualStreamRoFormerGPTQ'
        ]
        if not check_file_content(auto_path, expected_auto_content, "Auto.py registration"):
            all_passed = False
    
    # Check test and documentation files
    test_files = [
        ("quantize_dual_stream.py", "Test script"),
        ("run_dual_stream_test.sh", "Test runner script"),
        ("README_dual_stream_test.md", "Test documentation"),
        ("dual_stream_roformer_summary.md", "Implementation summary"),
        ("INSTALL_OPTIONS.md", "Installation guide"),
        ("pyproject.toml", "Project configuration")
    ]
    
    for filename, description in test_files:
        if not check_file_exists(filename, description):
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[PASS] All migration validation checks passed!")
        print("DualStreamRoFormer has been successfully migrated to the target directory.")
        print("\nNext steps:")
        print("1. Test the installation: uv pip install .[dev]")
        print("2. Run the test script: ./run_dual_stream_test.sh --model-path /path/to/model --analysis-only")
        print("3. Commit the changes to git")
    else:
        print("[FAIL] Some migration checks failed.")
        print("Please check the files and fix any issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
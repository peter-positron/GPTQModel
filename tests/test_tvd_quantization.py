
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import GPTQModel
from gptqmodel.quantization import QuantizeConfig
from TVD_test import TVDTester, load_default_test_prompts

logger = logging.getLogger(__name__)

# Small model for quick testing
MODEL_ID = "facebook/opt-125m"
QUANTIZED_MODEL_DIR_PREFIX = "quantized_model_tvd_test_"
ORIGINAL_MODEL_DIR_PREFIX = "original_model_tvd_test_"

def get_calib_dataset(tokenizer, n_samples=32, block_size=128):
    text = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ] * (n_samples // 4) # Repeat to get enough text

    tokenized_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=block_size)
    input_ids = tokenized_text.input_ids
    attn_mask = tokenized_text.attention_mask

    # Ensure all samples have the same length by padding or truncating
    if input_ids.shape[1] < block_size:
        padding = torch.full((input_ids.shape[0], block_size - input_ids.shape[1]), tokenizer.pad_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, padding], dim=1)
        attn_mask_padding = torch.zeros((attn_ids.shape[0], block_size - attn_mask.shape[1]), dtype=torch.long)
        attn_mask = torch.cat([attn_mask, attn_mask_padding], dim=1)
    elif input_ids.shape[1] > block_size:
        input_ids = input_ids[:, :block_size]
        attn_mask = attn_mask[:, :block_size]

    dataset = Dataset.from_dict({
        "input_ids": input_ids[:n_samples].tolist(),
        "attention_mask": attn_mask[:n_samples].tolist(),
    })
    return dataset

@pytest.fixture(scope="module")
def setup_models():
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmp_quantized_model_dir, \
         tempfile.TemporaryDirectory() as tmp_original_model_dir:

        quantized_model_path = Path(tmp_quantized_model_dir)
        original_model_path = Path(tmp_original_model_dir)

        logger.info(f"Loading original model: {MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Ensure pad token is set for tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Save the original model for TVD comparison
        model.save_pretrained(original_model_path)
        tokenizer.save_pretrained(original_model_path)
        logger.info(f"Original model saved to: {original_model_path}")

        # Quantize the model
        logger.info(f"Quantizing model: {MODEL_ID}")
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
        )

        gptq_model = GPTQModel(model, quantize_config)
        calib_dataset = get_calib_dataset(tokenizer)

        gptq_model.quantize(
            calib_dataset,
            batch_size=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        gptq_model.save_pretrained(quantized_model_path)
        logger.info(f"Quantized model saved to: {quantized_model_path}")

        yield str(original_model_path), str(quantized_model_path)

@pytest.mark.gpu
def test_tvd_quantization_comparison(setup_models):
    original_model_path, quantized_model_path = setup_models

    logger.info("Running TVD test comparison...")
    tester = TVDTester(
        model_dir_1=original_model_path,
        model_dir_2=quantized_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use default prompts and a reasonable tolerance
    prompts = load_default_test_prompts()
    # A higher tolerance is expected for 4-bit quantization
    # Adjust this based on empirical testing if needed
    tolerance = 0.1 # Increased tolerance for 4-bit quantization comparison

    results = tester.run_tvd_test(
        prompts=prompts,
        tolerance=tolerance,
        max_new_tokens=20, # Keep token generation short for faster tests
        use_outliers=True,
    )

    logger.info(f"TVD test overall result: {results['test_passed']}")
    logger.info(f"Overall max TVD: {results['overall_max_tvd']:.6f}")
    logger.info(f"Overall mean TVD: {results['overall_mean_tvd']:.6f}")

    assert results["test_passed"] is True, f"TVD test failed: {results['message']}"
    assert results["overall_max_tvd"] <= tolerance, \
        f"Overall max TVD {results['overall_max_tvd']:.6f} exceeded tolerance {tolerance:.6f}"

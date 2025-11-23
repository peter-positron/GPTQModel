import os
import shutil
import tempfile
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.models.definitions.gpt_oss import GPTOSSGPTQ
from gptqmodel.utils.offload import offload_to_disk


class TestOffloadGptOss(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_gpt_oss_offload(self):
        # Create a dummy GPT-OSS config
        config = AutoConfig.for_model("gpt_oss")
        config.hidden_size = 64
        config.intermediate_size = 256
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.head_dim = 16
        config.num_local_experts = 2
        config.num_experts_per_tok = 1
        config.vocab_size = 50300 # match gpt2 tokenizer
        config.max_position_embeddings = 128
        # Fix layer_types to match num_hidden_layers if it exists
        if hasattr(config, "layer_types"):
             # Assuming layer_types is a list of strings, we take the first N
             if isinstance(config.layer_types, list):
                 config.layer_types = config.layer_types[:config.num_hidden_layers]
        
        # Instantiate model
        # We need to ensure we can load this model via AutoModelForCausalLM
        # If gpt_oss is not registered in AutoConfig, we might need to instantiate directly
        # But since the previous import worked, it might be registered or we can import the class.
        
        try:
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
            model = GptOssForCausalLM(config)
        except ImportError:
            self.skipTest("transformers.models.gpt_oss not available")

        # Mock tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2") # just any tokenizer

        # Quantize Config with offload
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            offload_to_disk=True,
            offload_to_disk_path=os.path.join(self.test_dir, "offload"),
        )

        # Initialize GPTQModel
        # We use from_pretrained but with the model instance to simulate loading
        # But GPTQModel.from_pretrained expects path or ID usually.
        # We can use GPTQModel(model, ...) constructor directly but GPTQModel is a factory.
        # We should use `GPTOSSGPTQ` class directly or use `GPTQModel.load` with a fake path if we saved it.
        
        # Let's save the model first to pretend we are loading it
        model_path = os.path.join(self.test_dir, "model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # Now load with GPTQModel
        # This will trigger ModelLoader.from_pretrained
        qmodel = GPTQModel.load(
            model_path,
            quantize_config=quantize_config,
        )

        # Check if support_offload_to_disk is respected (no warning in logs, but hard to check in test)
        # Check if offload_to_disk is still True in config
        self.assertTrue(qmodel.quantize_config.offload_to_disk)
        
        # Run quantization (minimal)
        calibration_data = ["hello world " * 10]
        
        qmodel.quantize(calibration_data, batch_size=1, calibration_data_min_length=1)
        
        # Check if offload happened
        # The offload directory should contain files
        offload_dir = os.path.join(self.test_dir, "offload")
        self.assertTrue(os.path.exists(offload_dir))
        self.assertTrue(len(os.listdir(offload_dir)) > 0)
        
        # Check specific structure for GPT-OSS experts
        # Expected: model.layers.0.mlp.experts.0.gate_up...
        # But offload names depend on module names.
        
        # Verify that we can save the model
        save_path = os.path.join(self.test_dir, "quantized_model")
        qmodel.save(save_path)
        
        self.assertTrue(os.path.exists(os.path.join(save_path, "model.safetensors")))

if __name__ == '__main__':
    unittest.main()
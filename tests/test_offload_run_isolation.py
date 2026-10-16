# SPDX-License-Identifier: Apache-2.0
# GPU=-1

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from test_tiny_moe_quant_smoke import _build_calibration_dataset, _build_local_tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import BACKEND, CheckpointConfig, GPTQModel, QuantizeConfig, RTNConfig
from gptqmodel.looper.gptq_checkpoint import _checkpoint_quantization_identity, checkpoint_session


def _config(root, assign="init", config_cls=QuantizeConfig, **kwargs):
    if assign == "init":
        return config_cls(device="cpu", offload_to_disk=True, offload_to_disk_path=str(root), **kwargs)
    config = config_cls(device="cpu", offload_to_disk=True, **kwargs)
    config.offload_to_disk_path = str(root)
    return config


@pytest.mark.parametrize(("config_cls", "assign"), [(QuantizeConfig, "init"), (RTNConfig, "attribute")])
def test_two_models_sharing_an_offload_root_save_their_own_weights(tmp_path, config_cls, assign):
    sources = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        source = tmp_path / f"native-{seed}"
        LlamaForCausalLM(LlamaConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=4, vocab_size=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )).save_pretrained(source)
        tokenizer = _build_local_tokenizer(source)
        sources.append(source)
    calibration = _build_calibration_dataset(tokenizer)

    def quantize(source, root):
        torch.manual_seed(0)
        config = _config(root, assign, config_cls, bits=4, group_size=32, desc_act=False)
        model = GPTQModel.load(str(source), quantize_config=config, backend=BACKEND.TORCH)
        model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
        return model

    def saved(model, name):
        model.save(tmp_path / name)
        return load_file(tmp_path / name / "model.safetensors")

    expected = saved(quantize(sources[0], tmp_path / "private"), "expected")
    first = quantize(sources[0], tmp_path / "shared")
    quantize(sources[1], tmp_path / "shared")
    actual = saved(first, "actual")
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)


def test_config_and_reload_create_nothing_and_save_the_root(tmp_path):
    root = tmp_path / "offload"
    config = _config(root)
    assert QuantizeConfig.from_quant_config(config.to_dict()).offload_to_disk_path == str(root)
    assert not root.exists()
    identity = _checkpoint_quantization_identity(config)
    config._ensure_offload_run_dir()
    assert Path(config.offload_to_disk_path).parent == root
    assert config.to_dict()["meta"]["offload_to_disk_path"] == str(root)
    # The run directory must not change checkpoint identity, or saved checkpoints stop resuming.
    assert _checkpoint_quantization_identity(config) == identity


def test_run_dir_is_owned_by_the_config(tmp_path):
    config = _config(tmp_path / "offload")
    config._ensure_offload_run_dir()
    run_dir = config.offload_to_disk_path
    config._ensure_offload_run_dir()
    assert config.offload_to_disk_path == run_dir
    del config
    gc.collect()
    assert not Path(run_dir).exists()


def test_disabled_offload_keeps_the_configured_path(tmp_path):
    config = QuantizeConfig(device="cpu", offload_to_disk=False, offload_to_disk_path=str(tmp_path / "offload"))
    config._ensure_offload_run_dir()
    assert config.offload_to_disk_path == str(tmp_path / "offload")
    assert not (tmp_path / "offload").exists()


def test_automatic_offload_dir_is_used_directly():
    config = QuantizeConfig(device="cpu", offload_to_disk=True)
    automatic = config.offload_to_disk_path
    config._ensure_offload_run_dir()
    assert config.offload_to_disk_path == automatic
    assert config._offload_root is None


@pytest.mark.parametrize("plain_run_first", [False, True])
def test_automatic_checkpoint_root_is_the_explicit_root(tmp_path, plain_run_first):
    root = tmp_path / "offload"
    config = _config(root)
    if plain_run_first:
        config._ensure_offload_run_dir()
    checkpoint = CheckpointConfig()
    with checkpoint_session(checkpoint, SimpleNamespace(quantize_config=config)):
        # Attempts live in the checkpoint store, which must outlive any run directory.
        assert Path(config.offload_to_disk_path).parent == root
    assert checkpoint.resolve_path(None) == str(root)

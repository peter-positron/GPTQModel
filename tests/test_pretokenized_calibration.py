"""Tests for pre-tokenized JSON calibration loading (CPU-only)."""

from __future__ import annotations

import json
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample(
    input_ids: list[int],
    attention_mask: list[int] | None = None,
) -> dict:
    d = {"input_ids": input_ids}
    if attention_mask is not None:
        d["attention_mask"] = attention_mask
    return d


@pytest.fixture()
def json_array_file(tmp_path):
    """JSON file containing a bare array of samples."""
    samples = [
        _make_sample([1, 2, 3], [1, 1, 1]),
        _make_sample([4, 5, 6, 7], [1, 1, 1, 1]),
    ]
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(samples))
    return p


@pytest.fixture()
def jsonl_file(tmp_path):
    """JSONL file (one sample per line)."""
    samples = [
        _make_sample([10, 20], [1, 1]),
        _make_sample([30, 40, 50], [1, 1, 0]),
        _make_sample([60, 70, 80, 90]),
    ]
    p = tmp_path / "cal.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in samples))
    return p


@pytest.fixture()
def wrapper_file(tmp_path):
    """JSON wrapper: {"samples": [...]}."""
    samples = [
        _make_sample([100, 200]),
        _make_sample([300, 400, 500]),
    ]
    p = tmp_path / "cal_wrapped.json"
    p.write_text(json.dumps({"samples": samples}))
    return p


# ---------------------------------------------------------------------------
# load_pretokenized_json tests
# ---------------------------------------------------------------------------

class TestLoadPretokenizedJson:

    def test_json_array_loading(self, json_array_file):
        from gptqmodel.utils.calibration import load_pretokenized_json

        result = load_pretokenized_json(str(json_array_file))
        assert len(result) == 2
        assert result[0]["input_ids"] == [1, 2, 3]
        assert result[0]["attention_mask"] == [1, 1, 1]
        assert result[1]["input_ids"] == [4, 5, 6, 7]

    def test_jsonl_loading(self, jsonl_file):
        from gptqmodel.utils.calibration import load_pretokenized_json

        result = load_pretokenized_json(str(jsonl_file))
        assert len(result) == 3
        assert result[0]["input_ids"] == [10, 20]
        assert result[2]["input_ids"] == [60, 70, 80, 90]

    def test_wrapper_format(self, wrapper_file):
        from gptqmodel.utils.calibration import load_pretokenized_json

        result = load_pretokenized_json(str(wrapper_file))
        assert len(result) == 2
        assert result[0]["input_ids"] == [100, 200]

    def test_attention_mask_inference(self, tmp_path):
        """Missing attention_mask should be inferred as all-ones."""
        from gptqmodel.utils.calibration import load_pretokenized_json

        p = tmp_path / "no_mask.json"
        p.write_text(json.dumps([{"input_ids": [1, 2, 3]}]))
        result = load_pretokenized_json(str(p))
        assert result[0]["attention_mask"] == [1, 1, 1]

    def test_nsamples_truncation(self, json_array_file):
        from gptqmodel.utils.calibration import load_pretokenized_json

        result = load_pretokenized_json(str(json_array_file), nsamples=1)
        assert len(result) == 1
        assert result[0]["input_ids"] == [1, 2, 3]

    def test_empty_file_raises(self, tmp_path):
        from gptqmodel.utils.calibration import load_pretokenized_json

        p = tmp_path / "empty.json"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_pretokenized_json(str(p))

    def test_missing_input_ids_raises(self, tmp_path):
        from gptqmodel.utils.calibration import load_pretokenized_json

        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"attention_mask": [1, 1]}]))
        with pytest.raises(ValueError, match="missing 'input_ids'"):
            load_pretokenized_json(str(p))

    def test_non_int_input_ids_raises(self, tmp_path):
        from gptqmodel.utils.calibration import load_pretokenized_json

        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"input_ids": [1.5, 2.0]}]))
        with pytest.raises(ValueError, match="must be list\\[int\\]"):
            load_pretokenized_json(str(p))

    def test_mismatched_mask_length_raises(self, tmp_path):
        from gptqmodel.utils.calibration import load_pretokenized_json

        p = tmp_path / "bad.json"
        p.write_text(json.dumps([
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1]}
        ]))
        with pytest.raises(ValueError, match="length"):
            load_pretokenized_json(str(p))

    def test_wrapper_missing_samples_key_raises(self, tmp_path):
        from gptqmodel.utils.calibration import load_pretokenized_json

        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"data": [{"input_ids": [1]}]}))
        with pytest.raises(ValueError, match="missing 'samples' key"):
            load_pretokenized_json(str(p))


# ---------------------------------------------------------------------------
# prepare_calibration_dataset integration (file path string)
# ---------------------------------------------------------------------------

class TestPrepareCalibrationDatasetFilePath:

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_file_path_string_accepted(self, tmp_path):
        """prepare_calibration_dataset() should accept a file path string."""
        import torch
        from unittest.mock import MagicMock

        from gptqmodel.utils.calibration import prepare_calibration_dataset

        samples = [
            {"input_ids": list(range(20)), "attention_mask": [1] * 20},
        ]
        p = tmp_path / "cal.json"
        p.write_text(json.dumps(samples))

        qmodel = MagicMock()
        qmodel.tokenizer = None
        qmodel.support_batch_quantize = False
        qmodel.model = MagicMock()
        qmodel.model.config = None

        result = prepare_calibration_dataset(
            qmodel,
            str(p),
            calibration_data_min_length=1,
        )
        assert len(result) == 1
        assert isinstance(result[0]["input_ids"], torch.Tensor)

    def test_nonexistent_path_string_raises(self):
        from unittest.mock import MagicMock

        from gptqmodel.utils.calibration import prepare_calibration_dataset

        qmodel = MagicMock()
        with pytest.raises(ValueError, match="does not exist"):
            prepare_calibration_dataset(qmodel, "/no/such/file.json")


# ---------------------------------------------------------------------------
# CAPS reports pretokenized_calibration_json
# ---------------------------------------------------------------------------

class TestCapsReportsPretokenizedJson:

    def test_caps_key_present(self):
        from gptqmodel._paibaker_caps import get_caps

        caps = get_caps()
        assert "pretokenized_calibration_json" in caps
        assert caps["pretokenized_calibration_json"] is True

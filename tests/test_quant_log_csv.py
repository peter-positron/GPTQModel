"""Test that quant_log.csv writer produces correct column alignment."""

from __future__ import annotations

import csv

import pytest

# Re-use the column name constants from the writer module.
from gptqmodel.models.writer import (
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_TIME,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)

HEADER = [PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES, QUANT_LOG_DAMP, PROCESS_LOG_TIME]


def _write_quant_log_csv(path: str, quant_log: list[dict]) -> None:
    """Minimal replica of the CSV-writing logic from writer.py save_quantized."""
    with open(path, mode='w', newline='') as file:
        w = csv.writer(file)
        w.writerow(HEADER)
        w.writerows([[entry.get(PROCESS_LOG_LAYER), entry.get(PROCESS_LOG_MODULE), entry.get(QUANT_LOG_LOSS),
                      entry.get(QUANT_LOG_NSAMPLES), entry.get(QUANT_LOG_DAMP), entry.get(PROCESS_LOG_TIME)] for entry in quant_log])


class TestQuantLogCsv:
    def test_column_count_matches_header(self, tmp_path):
        """Every data row must have the same number of columns as the header."""
        quant_log = [
            {PROCESS_LOG_LAYER: 0, PROCESS_LOG_MODULE: 'mlp.experts.gate_up.0', QUANT_LOG_LOSS: 0.123,
             QUANT_LOG_NSAMPLES: 389582, QUANT_LOG_DAMP: 0.05, PROCESS_LOG_TIME: 1.234},
            {PROCESS_LOG_LAYER: 0, PROCESS_LOG_MODULE: 'mlp.experts.gate_up.1', QUANT_LOG_LOSS: 0.456,
             QUANT_LOG_NSAMPLES: 389582, QUANT_LOG_DAMP: 0.05, PROCESS_LOG_TIME: 2.345},
        ]
        csv_path = str(tmp_path / 'quant_log.csv')
        _write_quant_log_csv(csv_path, quant_log)

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) == 6
            for i, row in enumerate(reader):
                assert len(row) == len(header), f'Row {i} has {len(row)} columns, expected {len(header)}'

    def test_samples_column_contains_nsamples(self, tmp_path):
        """The 'samples' column must contain the actual nsamples value, not damp."""
        quant_log = [
            {PROCESS_LOG_LAYER: 0, PROCESS_LOG_MODULE: 'mlp.experts.gate_up.0', QUANT_LOG_LOSS: 0.1,
             QUANT_LOG_NSAMPLES: 256, QUANT_LOG_DAMP: 0.05, PROCESS_LOG_TIME: 1.0},
        ]
        csv_path = str(tmp_path / 'quant_log.csv')
        _write_quant_log_csv(csv_path, quant_log)

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row['samples'] == '256', f"Expected '256' in samples column, got '{row['samples']}'"
            assert row['damp'] == '0.05', f"Expected '0.05' in damp column, got '{row['damp']}'"

    def test_none_nsamples_still_aligned(self, tmp_path):
        """If nsamples is missing from the entry, the column should be empty but aligned."""
        quant_log = [
            {PROCESS_LOG_LAYER: 0, PROCESS_LOG_MODULE: 'attn.q_proj', QUANT_LOG_LOSS: 0.1,
             QUANT_LOG_DAMP: 0.01, PROCESS_LOG_TIME: 0.5},
        ]
        csv_path = str(tmp_path / 'quant_log.csv')
        _write_quant_log_csv(csv_path, quant_log)

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            assert len(row) == len(header)

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            # damp should still be in the damp column, not shifted
            assert row['damp'] == '0.01'

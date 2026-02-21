# TODO: GPT-OSS per_expert key verification

## Finding

`tests/quantize_gptoss_matrix.py` had a verification check that was
inverted for `GPTQMODEL_MOE_SAVE_FORMAT=per_expert` mode.  It treated
stacked keys as "expected present" and per-expert keys as "unexpected",
regardless of the save format env var.  This caused the script to
report FAIL on a correctly-produced per-expert quantization.

## Fix applied

`verify_keys()` now reads `GPTQMODEL_MOE_SAVE_FORMAT` and flips the
expected/absent key sets accordingly:

- **stacked** (default): stacked keys present, per-expert keys absent
- **per_expert**: per-expert keys present, stacked keys absent

## What PASS means

| Mode | Present keys | Absent keys |
|------|-------------|-------------|
| `stacked` | `.experts.gate_up_proj.`, `.experts.down_proj.` | `.experts.gate_up.0.`, `.experts.down.0.`, etc. |
| `per_expert` | `.experts.gate_up.0.`, `.experts.down.0.`, etc. | `.experts.gate_up_proj.`, `.experts.down_proj.` |

## Acceptance criteria

- [ ] `verify_keys()` returns True for correctly-produced per_expert output
- [ ] `verify_keys()` returns True for correctly-produced stacked output
- [ ] Mixed layouts (both stacked and per-expert keys present) return False
- [ ] Key count sanity: for N layers and E experts, per-expert mode should
      have `N * E * suffixes_per_group * 2` expert keys (gate_up + down)

## Related issues

- Router/MLP return value mismatch at inference time (separate issue):
  transformers `GptOssMLP.forward()` expects 2 return values from router,
  patched router returns 3.  Only affects GPTQModel inference loading on
  machines where `before_model_load()` is conditional.  Does not affect
  quantization or Tron serving path.

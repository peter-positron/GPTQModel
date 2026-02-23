#!/bin/bash
# Calibration SOURCE sweep for GPT-OSS best/64/GAR-on
# Measures PPL + quant loss across different calibration distributions
# All at matched ~50k token budget
#
# Run AFTER calibration_sweep.sh completes (they share GPU).

set -euo pipefail

PROFILE="best"
GAR="on"
BASE_OUTPUT="/mnt/datasets/quants/sweep-source"
CALIB_DIR="/mnt/datasets/calibration"
DEVICE="cuda:0"

# Sweep: (calibration_file, label, description)
# All files are ~50k tokens for apples-to-apples comparison
declare -a SWEEP=(
    "$CALIB_DIR/harmony-wikitext-train-50k.json|wiki-train-50k|Wikitext-2 train split (leakage-free)"
    "$CALIB_DIR/harmony-code-50k.json|code-50k|HumanEval + MBPP code problems"
    "$CALIB_DIR/harmony-stem-50k.json|stem-50k|MMLU STEM Q&A"
    "$CALIB_DIR/harmony-mixed-50k.json|mixed-50k|1:1:1 wikitext+code+stem"
)

mkdir -p "$BASE_OUTPUT"

echo "============================================"
echo "Calibration SOURCE Sweep: best/64/GAR-on"
echo "Token budget: ~50k (matched across sources)"
echo "Points: ${#SWEEP[@]}"
echo "============================================"

for entry in "${SWEEP[@]}"; do
    IFS='|' read -r calib_file label desc <<< "$entry"
    output_dir="$BASE_OUTPUT/$label"

    if [ -f "$output_dir/quality_gate_summary.json" ]; then
        echo ""
        echo "=== SKIP $label (already completed) ==="
        continue
    fi

    echo ""
    echo "=== $label: $desc ==="
    echo "Started: $(date)"

    # Quantize (use all samples in the file — token budget already controlled at generation)
    # Pass large --calibration-samples to avoid the 128 default truncation
    GPTQMODEL_MOE_SAVE_FORMAT=per_expert .venv/bin/python tests/quantize_gptoss_matrix.py \
        --profile "$PROFILE" \
        --gar "$GAR" \
        --calibration "$calib_file" \
        --calibration-samples 9999 \
        --output-dir "$output_dir" || true

    # Run perplexity-only quality gate
    if [ -d "$output_dir" ]; then
        .venv/bin/python tests/quality_gate.py \
            "$output_dir" \
            --perplexity-only \
            --bf16-baseline /mnt/datasets/source/openai--gpt-oss-20b-bf16/quality_gate_summary.json \
            --device "$DEVICE" || true
    fi

    echo "Finished $label: $(date)"
done

echo ""
echo "============================================"
echo "Source sweep complete. Collecting results..."
echo "============================================"

# Print summary table
echo ""
printf "%-18s  %8s  %10s  %10s  %10s  %s\n" "Label" "Tokens" "PPL" "Loss(mean)" "Loss(max)" "Source"
printf "%-18s  %8s  %10s  %10s  %10s  %s\n" "---------" "------" "---" "----------" "---------" "------"

# Include existing cal-256 (wikitext-test) for comparison
if [ -f "/mnt/datasets/quants/gpt-oss-20b-best-gar-on-harmony512/quality_gate_summary.json" ]; then
    .venv/bin/python -c "
import json
d = json.load(open('/mnt/datasets/quants/gpt-oss-20b-best-gar-on-harmony512/quality_gate_summary.json'))
ppl = d.get('perplexity_avg', 0)
ql = d.get('quant_loss', {})
print(f'wiki-test-256       44638     {ppl:>10.2f}  {ql.get(\"mean_loss\", 0):>10.4f}  {ql.get(\"max_loss\", 0):>10.2f}  wikitext-test (existing)')
" 2>/dev/null || true
fi

for entry in "${SWEEP[@]}"; do
    IFS='|' read -r calib_file label desc <<< "$entry"
    output_dir="$BASE_OUTPUT/$label"
    summary="$output_dir/quality_gate_summary.json"
    if [ -f "$summary" ]; then
        .venv/bin/python -c "
import json
d = json.load(open('$summary'))
cal = json.load(open('$calib_file'))
tokens = cal.get('total_tokens', 0)
ppl = d.get('perplexity_avg', 0)
ql = d.get('quant_loss', {})
print(f'$label  {tokens:>8d}  {ppl:>10.2f}  {ql.get(\"mean_loss\", 0):>10.4f}  {ql.get(\"max_loss\", 0):>10.2f}  $desc')
" 2>/dev/null || echo "$label  FAILED"
    fi
done

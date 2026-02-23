#!/bin/bash
# Calibration sample count sweep for GPT-OSS best/64/GAR-on
# Measures PPL + quant loss at different calibration sample counts
# All use Harmony-framed wikitext-2 calibration data

set -euo pipefail

PROFILE="best"
GAR="on"
BASE_OUTPUT="/mnt/datasets/quants/sweep-calibration"
CALIB_DIR="/mnt/datasets/calibration"
DEVICE="cuda:0"

# Sweep points: (calibration_file, n_samples, label)
# Phase 1: from existing 512-sample file
# Phase 2: from larger files
declare -a SWEEP=(
    "$CALIB_DIR/harmony-wikitext-512.json|64|cal-064"
    "$CALIB_DIR/harmony-wikitext-512.json|128|cal-128"
    # 256 already exists as gpt-oss-20b-best-gar-on-harmony512 (PPL 237.9)
    "$CALIB_DIR/harmony-wikitext-512.json|384|cal-384"
    "$CALIB_DIR/harmony-wikitext-512.json|512|cal-512"
    "$CALIB_DIR/harmony-wikitext-768.json|768|cal-768"
    "$CALIB_DIR/harmony-wikitext-1024.json|1024|cal-1024"
    "$CALIB_DIR/harmony-wikitext-1536.json|1536|cal-1536"
)

mkdir -p "$BASE_OUTPUT"

echo "============================================"
echo "Calibration Sweep: best/64/GAR-on"
echo "Points: ${#SWEEP[@]} + 1 existing (cal-256)"
echo "============================================"

for entry in "${SWEEP[@]}"; do
    IFS='|' read -r calib_file n_samples label <<< "$entry"
    output_dir="$BASE_OUTPUT/$label"

    if [ -f "$output_dir/quality_gate_summary.json" ]; then
        echo ""
        echo "=== SKIP $label (already completed) ==="
        continue
    fi

    echo ""
    echo "=== $label: $n_samples samples from $(basename $calib_file) ==="
    echo "Started: $(date)"

    # Quantize
    GPTQMODEL_MOE_SAVE_FORMAT=per_expert .venv/bin/python tests/quantize_gptoss_matrix.py \
        --profile "$PROFILE" \
        --gar "$GAR" \
        --calibration "$calib_file" \
        --calibration-samples "$n_samples" \
        --output-dir "$output_dir" || true

    # Run perplexity-only quality gate (skip generation to save time)
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
echo "Sweep complete. Collecting results..."
echo "============================================"

# Print summary table
echo ""
printf "%-10s  %8s  %10s  %10s  %10s\n" "Label" "Samples" "PPL" "Loss(mean)" "Loss(max)"
printf "%-10s  %8s  %10s  %10s  %10s\n" "-----" "-------" "---" "----------" "---------"

# Include the existing 256-sample result
if [ -f "/mnt/datasets/quants/gpt-oss-20b-best-gar-on-harmony512/quality_gate_summary.json" ]; then
    .venv/bin/python -c "
import json
d = json.load(open('/mnt/datasets/quants/gpt-oss-20b-best-gar-on-harmony512/quality_gate_summary.json'))
ppl = d.get('perplexity_avg', 'n/a')
ql = d.get('quant_loss', {})
print(f'cal-256     256       {ppl:>10.2f}  {ql.get(\"mean_loss\", 0):>10.4f}  {ql.get(\"max_loss\", 0):>10.2f}')
" 2>/dev/null || true
fi

for entry in "${SWEEP[@]}"; do
    IFS='|' read -r calib_file n_samples label <<< "$entry"
    output_dir="$BASE_OUTPUT/$label"
    summary="$output_dir/quality_gate_summary.json"
    if [ -f "$summary" ]; then
        .venv/bin/python -c "
import json
d = json.load(open('$summary'))
ppl = d.get('perplexity_avg', 'n/a')
ql = d.get('quant_loss', {})
print(f'$label  {$n_samples:>8d}  {ppl:>10.2f}  {ql.get(\"mean_loss\", 0):>10.4f}  {ql.get(\"max_loss\", 0):>10.2f}')
" 2>/dev/null || echo "$label  $n_samples  FAILED"
    fi
done

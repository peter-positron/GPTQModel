"""Quality gate eval for GPT-OSS quantized models.

Replaces smoke_generate.py's crash-only check with quantitative loop detection,
derail detection, and capability validation. Reports per-generation metrics to
JSONL and per-cell summary with pass/fail gate thresholds.

Usage:
    python tests/quality_gate.py /path/to/quant-model --device cuda:5
    python tests/quality_gate.py /path/to/model1 /path/to/model2 --device cuda:0
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Decode configurations
# ---------------------------------------------------------------------------

DECODE_CONFIGS = {
    "greedy": {"do_sample": False, "temperature": 1.0},
    "sampling": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}

# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

# Each prompt: (id, category, max_new_tokens, variants, anchors | checker)
# anchors: list of lowercase terms for derail detection (open_ended / instruction)
# checker: callable(generated_text) -> (bool, str) for capability prompts

PROMPTS: list[dict[str, Any]] = [
    # ---- OPEN-ENDED (4 prompts × 3 variants × 256 tokens) ----
    {
        "id": "open_ended_01",
        "category": "open_ended",
        "max_new_tokens": 256,
        "variants": [
            "The capital of France is",
            "France's capital city is",
            "Paris, the capital of France,",
        ],
        "anchors": [
            "france", "paris", "capital", "city", "europe", "french",
            "country", "river", "seine", "eiffel",
        ],
    },
    {
        "id": "open_ended_02",
        "category": "open_ended",
        "max_new_tokens": 256,
        "variants": [
            "The theory of evolution explains",
            "Darwin's theory of evolution",
            "Evolution by natural selection",
        ],
        "anchors": [
            "evolution", "darwin", "natural", "selection", "species",
            "adaptation", "genetic", "survival", "trait", "population",
        ],
    },
    {
        "id": "open_ended_03",
        "category": "open_ended",
        "max_new_tokens": 256,
        "variants": [
            "The Internet was originally developed",
            "ARPANET, the predecessor to the Internet,",
            "The history of the Internet begins",
        ],
        "anchors": [
            "internet", "arpanet", "network", "computer", "communication",
            "protocol", "web", "online", "digital", "tcp",
        ],
    },
    {
        "id": "open_ended_04",
        "category": "open_ended",
        "max_new_tokens": 256,
        "variants": [
            "In classical music, Beethoven",
            "Ludwig van Beethoven composed",
            "Beethoven's symphonies are",
        ],
        "anchors": [
            "beethoven", "symphony", "music", "composer", "classical",
            "piano", "orchestra", "sonata", "deaf", "vienna",
        ],
    },
    # ---- INSTRUCTION-FOLLOWING (4 prompts × 3 variants × 256 tokens) ----
    {
        "id": "instruction_01",
        "category": "instruction",
        "max_new_tokens": 256,
        "variants": [
            "Explain how photosynthesis works in 3 steps.",
            "Describe the process of photosynthesis step by step.",
            "List the main stages of photosynthesis.",
        ],
        "anchors": [
            "photosynthesis", "light", "chlorophyll", "carbon", "dioxide",
            "oxygen", "water", "energy", "glucose", "plant", "sun",
        ],
    },
    {
        "id": "instruction_02",
        "category": "instruction",
        "max_new_tokens": 256,
        "variants": [
            "Compare Python and JavaScript for web development.",
            "What are the key differences between Python and JavaScript?",
            "Contrast Python with JavaScript for building web apps.",
        ],
        "anchors": [
            "python", "javascript", "web", "language", "backend",
            "frontend", "syntax", "typing", "browser", "server",
        ],
    },
    {
        "id": "instruction_03",
        "category": "instruction",
        "max_new_tokens": 256,
        "variants": [
            "Summarize the causes of World War I in a paragraph.",
            "Write a brief summary of what caused World War I.",
            "What were the main causes that led to World War I?",
        ],
        "anchors": [
            "war", "alliance", "assassination", "archduke", "europe",
            "military", "empire", "nation", "conflict", "treaty",
        ],
    },
    {
        "id": "instruction_04",
        "category": "instruction",
        "max_new_tokens": 256,
        "variants": [
            "Explain what a neural network is to a beginner.",
            "Describe neural networks in simple terms.",
            "How would you explain neural networks to someone new to AI?",
        ],
        "anchors": [
            "neural", "network", "layer", "input", "output",
            "learn", "weight", "neuron", "brain", "data", "train",
        ],
    },
    # ---- FACTUAL QA (2 prompts × 3 variants × 64 tokens) ----
    {
        "id": "factual_01",
        "category": "factual",
        "max_new_tokens": 64,
        "variants": [
            "What is the boiling point of water at sea level?",
            "At what temperature does water boil at sea level?",
            "Water boils at sea level at a temperature of",
        ],
        "checker": lambda text: (
            any(v in text for v in ["100", "212"]),
            "expected '100' or '212' in output",
        ),
    },
    {
        "id": "factual_02",
        "category": "factual",
        "max_new_tokens": 64,
        "variants": [
            "How many planets are in our solar system?",
            "The number of planets in our solar system is",
            "Our solar system contains how many planets?",
        ],
        "checker": lambda text: (
            "8" in text or "eight" in text.lower(),
            "expected '8' or 'eight' in output",
        ),
    },
    # ---- CODE COMPLETION (2 prompts × 3 variants × 64 tokens) ----
    {
        "id": "code_01",
        "category": "code",
        "max_new_tokens": 64,
        "variants": [
            'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
            'def fib(n):\n    """Compute Fibonacci number for n."""\n',
            '# Return the nth Fibonacci number\ndef fibonacci(n):\n',
        ],
        "checker": lambda text: _check_code_fibonacci(text),
    },
    {
        "id": "code_02",
        "category": "code",
        "max_new_tokens": 128,
        "variants": [
            'def binary_search(arr, target):\n    """Return index of target in sorted arr, or -1."""\n',
            'def bsearch(nums, target):\n    """Binary search: return index or -1 if not found."""\n',
            '# Binary search on a sorted list\ndef binary_search(arr, target):\n',
        ],
        "checker": lambda text: _check_code_binary_search(text),
    },
]


# ---------------------------------------------------------------------------
# Code capability checkers
# ---------------------------------------------------------------------------

def _extract_function(text: str, func_names: list[str]) -> str | None:
    """Try to extract a complete function from generated text."""
    # Find the function body — take everything up to the next blank line
    # or next def/class or end of text
    lines = text.split("\n")
    result_lines: list[str] = []
    in_function = False
    indent_level = None

    for line in lines:
        stripped = line.rstrip()
        if not in_function:
            for name in func_names:
                if f"def {name}(" in stripped:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    result_lines.append(line[indent_level:])  # dedent to col 0
                    break
        elif in_function:
            if stripped == "":
                result_lines.append("")
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and not line[indent_level:].startswith(" "):
                # New top-level statement — function ended
                if stripped.startswith("def ") or stripped.startswith("class "):
                    break
                # Could be a call like print(fibonacci(10)) — stop
                break
            result_lines.append(line[indent_level:])

    if not result_lines:
        return None

    # Trim trailing empty lines
    while result_lines and result_lines[-1].strip() == "":
        result_lines.pop()

    return "\n".join(result_lines) if result_lines else None


def _check_code_fibonacci(text: str) -> tuple[bool, str]:
    """Check if generated code contains a working fibonacci function."""
    func_src = _extract_function(text, ["fibonacci", "fib"])
    if func_src is None:
        return False, "no function definition found"

    try:
        ast.parse(func_src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    # Determine function name from source
    func_name = "fibonacci" if "def fibonacci(" in func_src else "fib"

    try:
        namespace: dict[str, Any] = {}
        exec(func_src, namespace)  # noqa: S102
        result = namespace[func_name](10)
        if result == 55:
            return True, "fibonacci(10) == 55"
        return False, f"fibonacci(10) == {result}, expected 55"
    except Exception as e:
        return False, f"runtime error: {e}"


def _check_code_binary_search(text: str) -> tuple[bool, str]:
    """Check if generated code contains a working binary search function."""
    func_src = _extract_function(text, ["binary_search", "bsearch"])
    if func_src is None:
        return False, "no function definition found"

    try:
        ast.parse(func_src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    func_name = "binary_search" if "def binary_search(" in func_src else "bsearch"

    try:
        namespace: dict[str, Any] = {}
        exec(func_src, namespace)  # noqa: S102
        fn = namespace[func_name]
        arr = [1, 3, 5, 7, 9, 11, 13]
        # Check: find existing element
        idx = fn(arr, 7)
        if idx != 3:
            return False, f"search([1,3,5,7,9,11,13], 7) == {idx}, expected 3"
        # Check: missing element
        idx_miss = fn(arr, 4)
        if idx_miss != -1:
            return False, f"search([1,3,5,7,9,11,13], 4) == {idx_miss}, expected -1"
        return True, "binary_search passed both cases"
    except Exception as e:
        return False, f"runtime error: {e}"


# ---------------------------------------------------------------------------
# Loop detectors
# ---------------------------------------------------------------------------

def detect_loop_ngram(token_ids: list[int], n_values: tuple[int, ...] = (3, 4, 5),
                      threshold: int = 6) -> dict[str, Any]:
    """Detector 1: n-gram repetition rate.

    Flag if any n-gram (for n in n_values) appears >= threshold times.
    Returns max_4gram_count and repeat_fraction.
    """
    max_4gram_count = 0
    repeat_fraction = 0.0
    flagged = False

    for n in n_values:
        if len(token_ids) < n:
            continue
        ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
        counts = Counter(ngrams)
        if not counts:
            continue

        most_common_count = counts.most_common(1)[0][1]
        if n == 4:
            max_4gram_count = most_common_count

        if most_common_count >= threshold:
            flagged = True

    # Compute repeat_fraction for 4-grams
    if len(token_ids) >= 4:
        ngrams_4 = [tuple(token_ids[i:i + 4]) for i in range(len(token_ids) - 3)]
        counts_4 = Counter(ngrams_4)
        repeated_positions: set[int] = set()
        for ng, cnt in counts_4.items():
            if cnt >= 2:  # appears at least twice
                for i, g in enumerate(ngrams_4):
                    if g == ng:
                        for j in range(4):
                            repeated_positions.add(i + j)
        repeat_fraction = len(repeated_positions) / len(token_ids) if token_ids else 0.0

    return {
        "flagged": flagged,
        "max_4gram_count": max_4gram_count,
        "repeat_fraction": round(repeat_fraction, 4),
    }


def detect_loop_distinct2(token_ids: list[int], window_size: int = 64,
                          threshold: float = 0.2,
                          consecutive_required: int = 2) -> dict[str, Any]:
    """Detector 2: rolling window degeneracy.

    Compute distinct-2 (unique bigrams / total bigrams) over sliding windows.
    Flag if distinct-2 < threshold for >= consecutive_required consecutive windows.
    """
    if len(token_ids) < window_size:
        return {"flagged": False, "min_distinct2": 1.0}

    min_d2 = 1.0
    low_streak = 0
    flagged = False

    for start in range(0, len(token_ids) - window_size + 1, window_size // 2):
        window = token_ids[start:start + window_size]
        bigrams = [(window[i], window[i + 1]) for i in range(len(window) - 1)]
        total = len(bigrams)
        unique = len(set(bigrams))
        d2 = unique / total if total > 0 else 1.0
        min_d2 = min(min_d2, d2)

        if d2 < threshold:
            low_streak += 1
            if low_streak >= consecutive_required:
                flagged = True
        else:
            low_streak = 0

    return {"flagged": flagged, "min_distinct2": round(min_d2, 4)}


def detect_loop_sentence(text: str, threshold: int = 3) -> dict[str, Any]:
    """Detector 3: same-sentence repetition.

    Split on sentence boundaries, normalize, flag if any sentence appears >= threshold times.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Normalize: strip, lowercase, collapse whitespace
    normalized = []
    for s in sentences:
        s = re.sub(r'\s+', ' ', s.strip().lower())
        if len(s) > 5:  # ignore tiny fragments
            normalized.append(s)

    if not normalized:
        return {"flagged": False, "max_sentence_count": 0}

    counts = Counter(normalized)
    max_count = counts.most_common(1)[0][1]
    return {"flagged": max_count >= threshold, "max_sentence_count": max_count}


def check_loops(token_ids: list[int], text: str) -> dict[str, Any]:
    """Run all 3 loop detectors and combine results."""
    ngram = detect_loop_ngram(token_ids)
    distinct2 = detect_loop_distinct2(token_ids)
    sentence = detect_loop_sentence(text)

    detectors_triggered = []
    if ngram["flagged"]:
        detectors_triggered.append("ngram")
    if distinct2["flagged"]:
        detectors_triggered.append("distinct2")
    if sentence["flagged"]:
        detectors_triggered.append("sentence")

    return {
        "loop_flag": len(detectors_triggered) > 0,
        "detectors_triggered": detectors_triggered,
        "max_4gram_count": ngram["max_4gram_count"],
        "repeat_fraction": ngram["repeat_fraction"],
        "min_distinct2": distinct2["min_distinct2"],
        "max_sentence_count": sentence["max_sentence_count"],
    }


# ---------------------------------------------------------------------------
# Derail detector
# ---------------------------------------------------------------------------

def check_derail(text: str, token_ids: list[int], anchors: list[str],
                 coverage_threshold: float = 0.15,
                 decay_threshold: float = 0.5) -> dict[str, Any]:
    """Anchor-term coverage and decay for derail detection.

    anchors: lowercase terms to look for in the generated text.
    Flag "derail" if anchor_coverage < coverage_threshold OR anchor_decay > decay_threshold.
    """
    if not anchors:
        return {"derail_flag": False, "anchor_coverage": 1.0, "anchor_decay": 0.0}

    text_lower = text.lower()

    # Overall coverage
    found = sum(1 for a in anchors if a in text_lower)
    anchor_coverage = found / len(anchors)

    # Decay: compare first 64 tokens vs last 64 tokens
    # Use character-based approximation: split text roughly by token count
    words = text_lower.split()
    n_words = len(words)

    if n_words >= 16:
        quarter = max(n_words // 4, 8)
        first_chunk = " ".join(words[:quarter])
        last_chunk = " ".join(words[-quarter:])

        first_cov = sum(1 for a in anchors if a in first_chunk) / len(anchors)
        last_cov = sum(1 for a in anchors if a in last_chunk) / len(anchors)
        anchor_decay = first_cov - last_cov
    else:
        anchor_decay = 0.0

    derail_flag = anchor_coverage < coverage_threshold or anchor_decay > decay_threshold

    return {
        "derail_flag": derail_flag,
        "anchor_coverage": round(anchor_coverage, 4),
        "anchor_decay": round(anchor_decay, 4),
    }


# ---------------------------------------------------------------------------
# Generation + evaluation
# ---------------------------------------------------------------------------

def get_git_commit() -> str:
    """Get current short git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_cell_name(meta: dict) -> str:
    """Build cell name from run_metadata.json."""
    profile = meta.get("profile", "?")
    qc = meta.get("quantize_config", {})
    gs = qc.get("group_size", "?")
    gar = meta.get("gar", "?")
    return f"{profile}/{gs}/GAR-{gar}"


def _read_quant_log(model_dir: Path) -> dict | None:
    """Read quant_log.csv and compute aggregate loss stats."""
    log_path = model_dir / "quant_log.csv"
    if not log_path.exists():
        return None
    import csv

    losses = []
    worst = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                loss = float(row["loss"])
                losses.append(loss)
                worst.append((int(row["layer"]), row["module"], loss))
            except (ValueError, KeyError):
                pass  # skip failsafe entries
    if not losses:
        return None
    worst.sort(key=lambda x: x[2], reverse=True)
    return {
        "modules": len(losses),
        "mean_loss": round(sum(losses) / len(losses), 6),
        "median_loss": round(sorted(losses)[len(losses) // 2], 6),
        "sum_loss": round(sum(losses), 2),
        "max_loss": round(max(losses), 4),
        "top5_worst": [
            {"layer": l, "module": m, "loss": round(v, 4)}
            for l, m, v in worst[:5]
        ],
    }


def run_quality_gate(
    model_dir: Path,
    device: str,
    bf16: bool = False,
    perplexity_only: bool = False,
    decode_modes: list[str] | None = None,
    bf16_baseline: dict | None = None,
) -> dict[str, Any]:
    """Run full quality gate on one model directory. Returns summary dict.

    If perplexity_only=True, skip generation checks and load existing results
    from quality_gate_results.jsonl. Only runs perplexity + quant_log.

    decode_modes: list of decode config names (default: ["greedy"]).
    bf16_baseline: loaded bf16 summary dict for delta-based metrics.
    """
    if decode_modes is None:
        decode_modes = ["greedy"]
    # Read metadata
    meta_path = model_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cell_name = "bf16/baseline" if bf16 else get_cell_name(meta)
    git_commit = get_git_commit()

    mode_label = "perplexity-only" if perplexity_only else (
        "bf16 (unquantized baseline)" if bf16 else "quantized"
    )
    print(f"\n{'=' * 70}")
    print(f"QUALITY GATE: {cell_name}")
    print(f"  Model: {model_dir}")
    print(f"  Commit: {git_commit}")
    print(f"  Device: {device}")
    print(f"  Mode: {mode_label}")
    print(f"{'=' * 70}\n")

    # Apply paibaker patches if available
    try:
        from paibaker.patches import apply_all
        apply_all()
    except ImportError:
        pass

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"Loading model from {model_dir}...")
    t0 = time.monotonic()

    if bf16:
        import torch
        from transformers import AutoModelForCausalLM

        # bf16 20B model needs ~40GB; single 46GB GPU leaves too little
        # headroom for KV cache. Use device_map="auto" to spread across GPUs.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        from gptqmodel import GPTQModel

        model = GPTQModel.load(str(model_dir), device=device)

    load_time = time.monotonic() - t0
    print(f"Loaded in {load_time:.1f}s\n")

    # Per-generation results
    results: list[dict[str, Any]] = []
    jsonl_path = model_dir / "quality_gate_results.jsonl"

    if perplexity_only:
        # Load existing generation results if available
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
            print(f"Loaded {len(results)} existing generation results from {jsonl_path.name}")
        else:
            print("No existing results found — perplexity-only mode, skipping generations")
    else:
        with open(jsonl_path, "w") as jsonl_f:
            prompts_per_mode = sum(len(p["variants"]) for p in PROMPTS)
            total = prompts_per_mode * len(decode_modes)
            gen_idx = 0

            for decode_mode_name in decode_modes:
                decode_cfg = DECODE_CONFIGS[decode_mode_name]
                print(f"\n--- Decode mode: {decode_mode_name} ---")

                for prompt_def in PROMPTS:
                    pid = prompt_def["id"]
                    category = prompt_def["category"]
                    max_tokens = prompt_def["max_new_tokens"]

                    for variant_idx, variant_text in enumerate(prompt_def["variants"]):
                        gen_idx += 1
                        print(f"[{gen_idx}/{total}] {pid} v{variant_idx} [{decode_mode_name}]: {variant_text[:50]}...")

                        # Generate
                        t0 = time.monotonic()
                        input_device = device if not bf16 else model.device
                        input_ids = tokenizer(
                            variant_text, return_tensors="pt"
                        ).input_ids.to(input_device)
                        output_ids = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_tokens,
                            **decode_cfg,
                        )
                        gen_time = time.monotonic() - t0
                        new_token_ids = output_ids[0][input_ids.shape[1]:].tolist()
                        generated_text = tokenizer.decode(
                            output_ids[0][input_ids.shape[1]:],
                            skip_special_tokens=True,
                        )
                        n_tokens = len(new_token_ids)

                        # Loop detection
                        loop_metrics = check_loops(new_token_ids, generated_text)

                        # Derail detection (open_ended and instruction only)
                        anchors = prompt_def.get("anchors", [])
                        if anchors:
                            derail_metrics = check_derail(
                                generated_text, new_token_ids, anchors
                            )
                        else:
                            derail_metrics = {
                                "derail_flag": False,
                                "anchor_coverage": None,
                                "anchor_decay": None,
                            }

                        # Capability check (factual and code only)
                        checker = prompt_def.get("checker")
                        if checker:
                            check_text = (
                                variant_text + generated_text
                                if prompt_def["category"] == "code"
                                else generated_text
                            )
                            cap_pass, cap_detail = checker(check_text)
                            capability = {"capability_pass": cap_pass, "capability_detail": cap_detail}
                        else:
                            capability = {"capability_pass": None, "capability_detail": None}

                        # Status line
                        flags = []
                        if loop_metrics["loop_flag"]:
                            flags.append(f"LOOP({','.join(loop_metrics['detectors_triggered'])})")
                        if derail_metrics["derail_flag"]:
                            flags.append(f"DERAIL(cov={derail_metrics['anchor_coverage']:.2f})")
                        if capability["capability_pass"] is False:
                            flags.append(f"FAIL({capability['capability_detail'][:40]})")

                        status = " ".join(flags) if flags else "OK"
                        print(f"  {n_tokens} tokens, {gen_time:.1f}s — {status}")

                        # Build record
                        record = {
                            "git_commit": git_commit,
                            "cell": cell_name,
                            "decode_mode": decode_mode_name,
                            "prompt_id": pid,
                            "prompt_variant": variant_idx,
                            "prompt_category": category,
                            "prompt_text": variant_text,
                            "decode_params": {
                                **decode_cfg,
                                "max_new_tokens": max_tokens,
                            },
                            "tokens_generated": n_tokens,
                            "gen_time_seconds": round(gen_time, 2),
                            "loop_flag": loop_metrics["loop_flag"],
                            "loop_metrics": {
                                "max_4gram_count": loop_metrics["max_4gram_count"],
                                "repeat_fraction": loop_metrics["repeat_fraction"],
                                "min_distinct2": loop_metrics["min_distinct2"],
                                "max_sentence_count": loop_metrics["max_sentence_count"],
                                "detectors_triggered": loop_metrics["detectors_triggered"],
                            },
                            "derail_flag": derail_metrics["derail_flag"],
                            "derail_metrics": {
                                "anchor_coverage": derail_metrics["anchor_coverage"],
                                "anchor_decay": derail_metrics["anchor_decay"],
                            },
                            "capability_pass": capability["capability_pass"],
                            "capability_detail": capability["capability_detail"],
                            "raw_text": generated_text[:1000],
                        }

                        results.append(record)
                        jsonl_f.write(json.dumps(record) + "\n")
                        jsonl_f.flush()

    # Perplexity evaluation (wikitext-2)
    import torch
    ppl_avg = None
    ppl_scores = []
    try:
        from gptqmodel.utils.perplexity import Perplexity

        print("\nComputing wikitext-2 perplexity...")
        t0 = time.monotonic()
        ppl_obj = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="test",
            text_column="text",
        )
        with torch.inference_mode():
            ppl_scores = ppl_obj.calculate(n_ctx=512, n_batch=512)
        ppl_avg = sum(ppl_scores) / len(ppl_scores) if ppl_scores else None
        ppl_time = time.monotonic() - t0
        print(f"Perplexity: {ppl_avg:.2f} ({len(ppl_scores)} chunks, {ppl_time:.1f}s)")
    except Exception as e:
        print(f"Perplexity calculation failed: {e}")

    # Read quant_log.csv if available
    quant_loss = _read_quant_log(model_dir)

    # Clean up model to free GPU memory
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Compute summary — group by decode_mode for per-mode stats
    summary = _compute_summary(results, cell_name, git_commit, bf16_baseline)
    summary["perplexity_avg"] = round(ppl_avg, 4) if ppl_avg is not None else None
    summary["perplexity_chunks"] = len(ppl_scores)
    if quant_loss:
        summary["quant_loss"] = quant_loss

    # Compute PPL ratio vs bf16 baseline
    if bf16_baseline and ppl_avg and bf16_baseline.get("perplexity_avg"):
        summary["ppl_ratio"] = round(ppl_avg / bf16_baseline["perplexity_avg"], 4)

    # Write summary
    summary_path = model_dir / "quality_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nResults: {jsonl_path}")
    print(f"Summary: {summary_path}")

    return summary


def _compute_mode_stats(results: list[dict]) -> dict:
    """Compute loop/derail/capability stats for a set of results."""
    total = len(results)
    loop_count = sum(1 for r in results if r["loop_flag"])
    loop_rate = loop_count / total if total else 0.0

    derail_candidates = [r for r in results if r["prompt_category"] in ("open_ended", "instruction")]
    derail_count = sum(1 for r in derail_candidates if r["derail_flag"])
    derail_total = len(derail_candidates)
    derail_rate = derail_count / derail_total if derail_total else 0.0

    cap_candidates = [r for r in results if r["capability_pass"] is not None]
    cap_pass_count = sum(1 for r in cap_candidates if r["capability_pass"])
    cap_total = len(cap_candidates)
    cap_rate = cap_pass_count / cap_total if cap_total else 1.0

    categories: dict[str, dict] = {}
    for cat in ("open_ended", "instruction", "factual", "code"):
        cat_results = [r for r in results if r["prompt_category"] == cat]
        if not cat_results:
            continue
        cat_loop = sum(1 for r in cat_results if r["loop_flag"])
        cat_derail = sum(1 for r in cat_results if r.get("derail_flag", False))
        cat_cap = [r for r in cat_results if r["capability_pass"] is not None]
        categories[cat] = {
            "count": len(cat_results),
            "loop_count": cat_loop,
            "derail_count": cat_derail,
            "capability_pass": sum(1 for r in cat_cap if r["capability_pass"]),
            "capability_total": len(cat_cap),
        }

    return {
        "total_generations": total,
        "loop_rate": round(loop_rate, 4),
        "loop_count": loop_count,
        "derail_rate": round(derail_rate, 4),
        "derail_count": derail_count,
        "derail_total": derail_total,
        "capability_pass_rate": round(cap_rate, 4),
        "capability_pass_count": cap_pass_count,
        "capability_total": cap_total,
        "categories": categories,
    }


def _compute_summary(
    results: list[dict],
    cell_name: str,
    git_commit: str,
    bf16_baseline: dict | None = None,
) -> dict:
    """Compute per-cell summary from generation results.

    When bf16_baseline is provided, gate thresholds become delta-based:
    - loop_delta_max: +5pp above bf16 greedy loop rate
    - derail_delta_max: +5pp above bf16 greedy derail rate
    - capability_delta_min: -5pp below bf16 capability rate
    """
    # Group results by decode_mode (backward compat: missing = "greedy")
    by_mode: dict[str, list[dict]] = {}
    for r in results:
        mode = r.get("decode_mode", "greedy")
        by_mode.setdefault(mode, []).append(r)

    # Primary mode for top-level stats and gate: greedy
    primary_results = by_mode.get("greedy", results)
    primary = _compute_mode_stats(primary_results)

    # Gate thresholds — delta-based when baseline available
    if bf16_baseline:
        bl_greedy = bf16_baseline.get("by_decode_mode", {}).get("greedy", bf16_baseline)
        bl_loop = bl_greedy.get("loop_rate", bf16_baseline.get("loop_rate", 0.0))
        bl_derail = bl_greedy.get("derail_rate", bf16_baseline.get("derail_rate", 0.0))
        bl_cap = bl_greedy.get("capability_pass_rate", bf16_baseline.get("capability_pass_rate", 1.0))

        thresholds = {
            "mode": "delta",
            "loop_delta_max": 0.05,    # max 5pp above bf16
            "derail_delta_max": 0.05,  # max 5pp above bf16
            "capability_delta_min": -0.05,  # max 5pp below bf16
            "bf16_loop_rate": bl_loop,
            "bf16_derail_rate": bl_derail,
            "bf16_capability_rate": bl_cap,
        }
        loop_delta = primary["loop_rate"] - bl_loop
        derail_delta = primary["derail_rate"] - bl_derail
        cap_delta = primary["capability_pass_rate"] - bl_cap

        gate_passed = (
            loop_delta <= thresholds["loop_delta_max"]
            and derail_delta <= thresholds["derail_delta_max"]
            and cap_delta >= thresholds["capability_delta_min"]
        )
    else:
        thresholds = {
            "mode": "absolute",
            "loop_rate_max": 0.05,
            "derail_rate_max": 0.10,
            "capability_pass_rate_min": 0.95,
        }
        loop_delta = None
        derail_delta = None
        cap_delta = None

        gate_passed = (
            primary["loop_rate"] <= thresholds["loop_rate_max"]
            and primary["derail_rate"] <= thresholds["derail_rate_max"]
            and primary["capability_pass_rate"] >= thresholds["capability_pass_rate_min"]
        )

    summary = {
        "git_commit": git_commit,
        "cell": cell_name,
        **primary,
        "gate_passed": gate_passed,
        "gate_thresholds": thresholds,
    }

    # Add delta metrics when baseline available
    if bf16_baseline:
        summary["delta_vs_bf16"] = {
            "loop_delta": round(loop_delta, 4),
            "derail_delta": round(derail_delta, 4),
            "capability_delta": round(cap_delta, 4),
        }

    # Add per-mode breakdown when multiple modes were run
    if len(by_mode) > 1:
        summary["by_decode_mode"] = {
            mode: _compute_mode_stats(mode_results)
            for mode, mode_results in by_mode.items()
        }

    return summary


def print_summary_table(summaries: list[dict]) -> None:
    """Print a comparison table across cells."""
    has_deltas = any(s.get("delta_vs_bf16") for s in summaries)
    has_ppl_ratio = any(s.get("ppl_ratio") for s in summaries)

    print(f"\n{'=' * 80}")
    print("QUALITY GATE SUMMARY")
    if has_deltas:
        print("  (delta mode: thresholds relative to bf16 baseline)")
    print(f"{'=' * 80}")

    # Header
    cols = ["Cell", "Loop", "Derail", "Capability", "PPL"]
    if has_deltas:
        cols.extend(["LoopD", "PPL/bf16"])
    cols.append("Gate")
    header = f"{'Cell':<25} {'Loop':<15} {'Derail':<15} {'Capability':<15} {'PPL':<10}"
    if has_deltas:
        header += f" {'LoopD':<8} {'PPL/bf16':<8}"
    header += f" {'Gate':<6}"
    print(header)
    print("-" * (96 if has_deltas else 80))

    for s in summaries:
        loop_str = f"{s['loop_count']}/{s['total_generations']} ({s['loop_rate']:.1%})"
        derail_str = f"{s['derail_count']}/{s['derail_total']} ({s['derail_rate']:.1%})"
        cap_str = f"{s['capability_pass_count']}/{s['capability_total']} ({s['capability_pass_rate']:.1%})"
        ppl_str = f"{s['perplexity_avg']:.2f}" if s.get("perplexity_avg") else "n/a"
        gate_str = "PASS" if s["gate_passed"] else "FAIL"

        line = f"{s['cell']:<25} {loop_str:<15} {derail_str:<15} {cap_str:<15} {ppl_str:<10}"
        if has_deltas:
            d = s.get("delta_vs_bf16", {})
            ld = d.get("loop_delta")
            ld_str = f"{ld:+.1%}" if ld is not None else "n/a"
            pr = s.get("ppl_ratio")
            pr_str = f"{pr:.2f}x" if pr is not None else "n/a"
            line += f" {ld_str:<8} {pr_str:<8}"
        line += f" {gate_str:<6}"
        print(line)

    print("-" * (96 if has_deltas else 80))

    # Per-mode breakdown if multiple decode modes were run
    any_multimode = any(s.get("by_decode_mode") for s in summaries)
    if any_multimode:
        print(f"\nPer-Decode-Mode Breakdown:")
        for s in summaries:
            modes = s.get("by_decode_mode", {})
            if not modes:
                continue
            print(f"\n  {s['cell']}:")
            for mode_name, ms in modes.items():
                loop_s = f"loop={ms['loop_count']}/{ms['total_generations']} ({ms['loop_rate']:.1%})"
                derail_s = f"derail={ms['derail_count']}/{ms['derail_total']} ({ms['derail_rate']:.1%})"
                cap_s = f"cap={ms['capability_pass_count']}/{ms['capability_total']} ({ms['capability_pass_rate']:.1%})"
                print(f"    {mode_name:<10} {loop_s}, {derail_s}, {cap_s}")

    # Quant loss summary if available
    any_loss = any(s.get("quant_loss") for s in summaries)
    if any_loss:
        print(f"\nQuantization Loss:")
        for s in summaries:
            ql = s.get("quant_loss")
            if ql:
                print(f"  {s['cell']}: mean={ql['mean_loss']:.6f}, sum={ql['sum_loss']:.0f}, max={ql['max_loss']:.1f}")
            else:
                print(f"  {s['cell']}: n/a (bf16 or no quant_log.csv)")

    # Category breakdown
    print(f"\nCategory Breakdown:")
    for s in summaries:
        print(f"\n  {s['cell']}:")
        for cat, data in s.get("categories", {}).items():
            parts = [f"loop={data['loop_count']}/{data['count']}"]
            if data["derail_count"] > 0 or cat in ("open_ended", "instruction"):
                parts.append(f"derail={data['derail_count']}/{data['count']}")
            if data["capability_total"] > 0:
                parts.append(f"cap={data['capability_pass']}/{data['capability_total']}")
            print(f"    {cat:<20} {', '.join(parts)}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quality gate eval for GPT-OSS quantized models",
    )
    parser.add_argument(
        "model_dirs", nargs="+", type=Path,
        help="Path(s) to quantized model directories",
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument(
        "--bf16", action="store_true",
        help="Load as bf16 unquantized model (baseline comparison)",
    )
    parser.add_argument(
        "--perplexity-only", action="store_true",
        help="Skip generation checks, load existing results, only compute perplexity",
    )
    parser.add_argument(
        "--decode-modes", nargs="+", default=["greedy"],
        choices=list(DECODE_CONFIGS),
        help="Decode modes to evaluate (default: greedy). Use 'greedy sampling' for both.",
    )
    parser.add_argument(
        "--bf16-baseline", type=Path, default=None,
        help="Path to bf16 quality_gate_summary.json for delta-based thresholds",
    )
    args = parser.parse_args()

    # Load bf16 baseline if provided
    bf16_baseline = None
    if args.bf16_baseline:
        if not args.bf16_baseline.exists():
            print(f"ERROR: bf16 baseline not found: {args.bf16_baseline}")
            return 1
        bf16_baseline = json.loads(args.bf16_baseline.read_text())
        bl_loop = bf16_baseline.get("loop_rate", "?")
        bl_ppl = bf16_baseline.get("perplexity_avg", "?")
        print(f"bf16 baseline loaded: loop={bl_loop}, PPL={bl_ppl}")

    # Validate paths
    for d in args.model_dirs:
        if not d.exists():
            print(f"ERROR: {d} does not exist")
            return 1

    summaries = []
    for model_dir in args.model_dirs:
        summary = run_quality_gate(
            model_dir, args.device, bf16=args.bf16,
            perplexity_only=args.perplexity_only,
            decode_modes=args.decode_modes,
            bf16_baseline=bf16_baseline,
        )
        summaries.append(summary)

    print_summary_table(summaries)

    # Exit code: 0 if all pass, 1 if any fail
    all_passed = all(s["gate_passed"] for s in summaries)
    if all_passed:
        print("ALL GATES PASSED")
        return 0
    else:
        failed = [s["cell"] for s in summaries if not s["gate_passed"]]
        print(f"GATE FAILED for: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

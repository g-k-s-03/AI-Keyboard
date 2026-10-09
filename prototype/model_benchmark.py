"""
On-device SLM benchmark for the AI Keyboard project.

Tests candidate small language models against keyboard-style prompts
(Hinglish translation, grammar fixing, tone/professional rewrite) and
reports load latency, generation latency/throughput, RAM footprint,
disk size, and a simple heuristic quality score.

Run:
    python model_benchmark.py

See BENCHMARK_GUIDE.md for setup, metric definitions, and how to add
a new model or submit results.
"""

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import psutil

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers/torch not installed.")
    print("Run: pip install transformers torch psutil pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Each entry may list multiple candidate repo ids. The first one that loads
# successfully is used and recorded as `resolved_model_id`, so a candidate
# that turns out to be gated, renamed, or (as with Qwen/Qwen3.5-0.8B, which
# is multimodal image-text-to-text, not a plain CausalLM) architecturally
# incompatible with load_model() falls back to the next one instead of
# crashing the whole run.
MODEL_CONFIGS = [
    {
        "label": "Qwen2.5-0.5B",
        "candidates": ["Qwen/Qwen2.5-0.5B-Instruct"],
    },
    {
        "label": "Qwen3-0.6B",
        "candidates": ["Qwen/Qwen3-0.6B"],
    },
    {
        "label": "Qwen2.5-1.5B",
        "candidates": ["Qwen/Qwen2.5-1.5B-Instruct"],
    },
    {
        "label": "InternLM2.5-1.8B",
        "candidates": ["internlm/internlm2_5-1_8b-chat"],
    },
    {
        "label": "SmolLM2-1.7B",
        "candidates": ["HuggingFaceTB/SmolLM2-1.7B-Instruct"],
    },
    {
        "label": "MobileLLaMA-1.4B",
        "candidates": ["mtgv/MobileLLaMA-1.4B-Chat"],
    },
]

SYSTEM_PROMPT = (
    "You are an AI keyboard assistant. \n"
    " Fix grammar, translate Hinglish to English if needed, \n"
    " and improve tone. Return ONLY the corrected text. \n"
    " No explanation. No extra text."
)

PROMPTS = {
    "hinglish": [
        "kya haal hai bhai meeting kb h",
        "bhai kal meeting cancel ho gayi kya",
        "yaar please report bhej de aaj tak",
        "kya haal hai, sab theek?",
        "meeting 6 baje hai mat bhoolna",
    ],
    "grammar_fix": [
        "i wil b thr at 6 dont wrry abt it",
        "hey can u send me that file asap its urgent",
        "this is wrong fix it now",
    ],
    "professional_rewrite": [
        "bhai please kal tak report bhej dena",
        "hey can u talk to the client tmrw",
    ],
}

MAX_NEW_TOKENS = 64

DEVICE_PROFILES = [
    {"name": "iOS keyboard extension", "ram_limit_mb": 50},
    {"name": "iOS companion app", "ram_limit_mb": 1024},
    {"name": "Budget Android (3GB RAM)", "ram_limit_mb": 3072},
    {"name": "Mid-range Android (4GB RAM)", "ram_limit_mb": 4096},
    {"name": "Flagship Android (8GB RAM)", "ram_limit_mb": 8192},
]

RESULTS_DIR = Path(__file__).parent / "results"
JSON_OUT = RESULTS_DIR / "benchmark_results.json"
CSV_OUT = RESULTS_DIR / "benchmark_results.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_process_ram_mb():
    """Resident memory of the current process, in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def get_model_dir_size_mb(model_id, tokenizer):
    """Best-effort size on disk of the cached model repo, in MB."""
    try:
        cache_file = Path(tokenizer.name_or_path)
        if cache_file.exists() and cache_file.is_dir():
            root = cache_file
        else:
            # transformers/huggingface_hub cache layout
            hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
            safe_name = "models--" + model_id.replace("/", "--")
            root = Path(hf_home) / "hub" / safe_name
        if not root.exists():
            return None
        total = 0
        for f in root.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 ** 2)
    except Exception:
        return None


def is_mostly_english(text):
    """Rough heuristic: ASCII-letter ratio among alphabetic characters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if c.isascii()]
    return (len(ascii_letters) / len(letters)) >= 0.85


def score_response(prompt, response):
    """Heuristic quality score out of 100 (see BENCHMARK_GUIDE.md)."""
    if not response or not response.strip():
        return 0

    score = 0
    checks = 3

    if len(response) <= 2 * max(len(prompt), 1):
        score += 1
    if prompt.strip().lower() not in response.strip().lower():
        score += 1
    if is_mostly_english(response):
        score += 1

    return round((score / checks) * 100)


def build_chat_prompt(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for tokenizers without a chat template
        return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"


def load_model(repo_id):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model


def run_generation(tokenizer, model, text):
    """Runs generation once, returning (output_text, ttft_ms, total_ms, tokens_generated)."""
    inputs = tokenizer(text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    # TTFT approximated via a 1-token generation, then the rest is timed
    # separately -- this avoids needing a streamer for a CLI benchmark.
    first_token_out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    ttft_ms = (time.perf_counter() - start) * 1000

    full_start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    total_ms = (time.perf_counter() - full_start) * 1000

    generated_ids = output_ids[0][input_len:]
    tokens_generated = len(generated_ids)
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return output_text.strip(), ttft_ms, total_ms, tokens_generated


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def benchmark_model(config):
    label = config["label"]
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {label}")
    print(f"{'=' * 60}")

    resolved_id = None
    tokenizer = None
    model = None
    load_error = None

    ram_before_mb = get_process_ram_mb()

    for repo_id in config["candidates"]:
        print(f"  Trying to load: {repo_id} ...")
        try:
            cold_start = time.perf_counter()
            tokenizer, model = load_model(repo_id)
            cold_load_ms = (time.perf_counter() - cold_start) * 1000
            resolved_id = repo_id
            print(f"  Loaded successfully in {cold_load_ms:.0f} ms")
            break
        except Exception as e:
            print(f"  Failed to load {repo_id}: {e}")
            load_error = str(e)
            tokenizer, model = None, None
            continue

    if model is None:
        print(f"  SKIPPING {label}: no candidate model could be loaded.")
        return {
            "label": label,
            "resolved_model_id": None,
            "status": "failed",
            "error": load_error,
        }

    ram_after_load_mb = get_process_ram_mb()
    model_ram_mb = max(ram_after_load_mb - ram_before_mb, 0)

    # Warm start: reload once more now that files are in the OS disk cache.
    # NOTE: this measures "warm disk cache" reload time, not a resident
    # in-memory model -- see BENCHMARK_GUIDE.md for the exact definition.
    del model
    gc.collect()
    try:
        warm_start = time.perf_counter()
        _, model = load_model(resolved_id)
        warm_load_ms = (time.perf_counter() - warm_start) * 1000
    except Exception as e:
        print(f"  Warm reload failed, reusing cold load timing: {e}")
        warm_load_ms = cold_load_ms
        _, model = load_model(resolved_id)

    model_size_mb = get_model_dir_size_mb(resolved_id, tokenizer)

    all_prompts = []
    for category, prompts in PROMPTS.items():
        for p in prompts:
            all_prompts.append((category, p))

    per_prompt_results = []
    latencies_ms = []
    ttfts_ms = []
    total_tokens = 0
    total_gen_time_s = 0.0

    for category, prompt in all_prompts:
        try:
            chat_text = build_chat_prompt(tokenizer, prompt)
            output, ttft_ms, gen_ms, n_tokens = run_generation(
                tokenizer, model, chat_text
            )
            quality = score_response(prompt, output)

            latencies_ms.append(gen_ms)
            ttfts_ms.append(ttft_ms)
            total_tokens += n_tokens
            total_gen_time_s += gen_ms / 1000

            per_prompt_results.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "output": output,
                    "ttft_ms": round(ttft_ms, 2),
                    "latency_ms": round(gen_ms, 2),
                    "tokens_generated": n_tokens,
                    "quality_score": quality,
                }
            )
            print(f"  [{category}] '{prompt[:35]}...' -> {gen_ms:.0f} ms, quality {quality}")
        except Exception as e:
            print(f"  Prompt failed ('{prompt}'): {e}")
            per_prompt_results.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "output": None,
                    "error": str(e),
                }
            )

    del model
    gc.collect()

    if not latencies_ms:
        return {
            "label": label,
            "resolved_model_id": resolved_id,
            "status": "failed",
            "error": "All prompts failed during generation.",
        }

    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    avg_ttft_ms = sum(ttfts_ms) / len(ttfts_ms)
    tokens_per_sec = total_tokens / total_gen_time_s if total_gen_time_s > 0 else 0
    avg_quality = sum(
        r["quality_score"] for r in per_prompt_results if "quality_score" in r
    ) / max(sum(1 for r in per_prompt_results if "quality_score" in r), 1)

    return {
        "label": label,
        "resolved_model_id": resolved_id,
        "status": "ok",
        "cold_start_load_ms": round(cold_load_ms, 2),
        "warm_start_load_ms": round(warm_load_ms, 2),
        "avg_ttft_ms": round(avg_ttft_ms, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "min_latency_ms": round(min(latencies_ms), 2),
        "max_latency_ms": round(max(latencies_ms), 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ram_before_load_mb": round(ram_before_mb, 2),
        "ram_after_load_mb": round(ram_after_load_mb, 2),
        "model_ram_mb": round(model_ram_mb, 2),
        "model_size_disk_mb": round(model_size_mb, 2) if model_size_mb else None,
        "avg_quality_score": round(avg_quality, 1),
        "num_prompts_tested": len(latencies_ms),
        "per_prompt_results": per_prompt_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_device_compatibility(results):
    print(f"\n{'=' * 90}")
    print("DEVICE COMPATIBILITY (based on measured model RAM usage)")
    print(f"{'=' * 90}")

    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        print("No successfully benchmarked models to evaluate.")
        return

    header = f"{'Device Profile':<28}{'RAM Limit':<12}" + "".join(
        f"{r['label']:<24}" for r in ok_results
    )
    print(header)
    print("-" * len(header))

    for profile in DEVICE_PROFILES:
        row = f"{profile['name']:<28}{str(profile['ram_limit_mb']) + ' MB':<12}"
        for r in ok_results:
            model_ram = r.get("model_ram_mb", 0) or 0
            fits = "OK" if model_ram <= profile["ram_limit_mb"] else "TOO BIG"
            row += f"{fits:<24}"
        print(row)


def print_comparison_table(results):
    print(f"\n{'=' * 100}")
    print("MODEL COMPARISON")
    print(f"{'=' * 100}")

    cols = [
        ("Model", "label"),
        ("Cold Load (ms)", "cold_start_load_ms"),
        ("Warm Load (ms)", "warm_start_load_ms"),
        ("Avg TTFT (ms)", "avg_ttft_ms"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("Min/Max (ms)", None),
        ("Tokens/sec", "tokens_per_sec"),
        ("Model RAM (MB)", "model_ram_mb"),
        ("Disk Size (MB)", "model_size_disk_mb"),
        ("Quality/100", "avg_quality_score"),
    ]

    header = "".join(f"{name:<18}" for name, _ in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        if r.get("status") != "ok":
            print(f"{r['label']:<18}{'FAILED - ' + str(r.get('error'))[:60]}")
            continue
        row = ""
        for name, key in cols:
            if key is None:
                val = f"{r['min_latency_ms']}/{r['max_latency_ms']}"
            else:
                val = r.get(key, "N/A")
            row += f"{str(val):<18}"
        print(row)


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON results -> {JSON_OUT}")

    if pd is None:
        print("pandas not installed - skipping CSV export.")
        return

    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        rows.append(
            {
                "Model": r["label"],
                "Latency(ms)": r["avg_latency_ms"],
                "RAM(MB)": r["model_ram_mb"],
                "Tokens/sec": r["tokens_per_sec"],
            }
        )

    df = pd.DataFrame(rows, columns=["Model", "Latency(ms)", "RAM(MB)", "Tokens/sec"])
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved CSV results -> {CSV_OUT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("AI Keyboard - On-device SLM Benchmark")
    print(f"Testing {len(MODEL_CONFIGS)} model(s), {sum(len(v) for v in PROMPTS.values())} prompts each.\n")

    results = []
    for config in MODEL_CONFIGS:
        try:
            result = benchmark_model(config)
        except Exception as e:
            print(f"  UNEXPECTED ERROR benchmarking {config['label']}: {e}")
            traceback.print_exc()
            result = {"label": config["label"], "status": "failed", "error": str(e)}
        results.append(result)

    print_comparison_table(results)
    print_device_compatibility(results)
    save_results(results)


if __name__ == "__main__":
    main()

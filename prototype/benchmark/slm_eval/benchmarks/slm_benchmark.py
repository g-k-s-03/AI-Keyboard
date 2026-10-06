import gc
import io
import json
import sys
import time
from pathlib import Path

import torch

from slm_eval.metrics.bleu import calculate_all_metrics
from slm_eval.metrics.device_metrics import get_ram_usage_mb
from slm_eval.models.loader import build_prompt, load_model_and_tokenizer
from slm_eval.utils.reproducibility import get_benchmark_env, set_seed
from slm_eval.validation.output_validator import sanitize_output, validate_output

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "eval_dataset.json"

SYSTEM_PROMPT = (
    "You are an AI keyboard assistant. "
    "Follow the instruction exactly. "
    "Return ONLY the requested output. "
    "No explanation, no preamble, no extra text."
)

LANG_TO_CATEGORIES = {
    "hinglish": ["hinglish_to_english", "error_correction"],
    "hindi": ["hindi_correction"],
    "portuguese": ["portuguese_correction"],
    "russian": ["russian_correction"],
    "english": ["english_grammar", "professional_rewrite",
                "meaning_preservation", "protected_tokens"],
    "all": None,
}


def load_dataset(lang_filter: str = "all", category_filter: str = None):
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    allowed_categories = None
    if lang_filter and lang_filter != "all":
        allowed_categories = LANG_TO_CATEGORIES.get(lang_filter)
    if category_filter:
        allowed_categories = [category_filter]

    prompts = []
    for category in data["categories"]:
        if allowed_categories and category["id"] not in allowed_categories:
            continue
        for prompt in category["prompts"]:
            prompt = dict(prompt)
            prompt["category_id"] = category["id"]
            prompt["category_name"] = category["name"]
            prompts.append(prompt)
    return prompts


def _single_inference(model, tokenizer, input_ids, max_new_tokens: int):
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    new_tokens = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, len(new_tokens)


def run_slm_benchmark(
    model_id: str,
    lang_filter: str = "all",
    category_filter: str = None,
    runs: int = 3,
    warmup_runs: int = 1,
    max_new_tokens: int = 100,
    device_name: str = "unknown",
):
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    set_seed()

    prompts = load_dataset(lang_filter, category_filter)
    if not prompts:
        return {"error": "No prompts found for given filters"}

    ram_before = get_ram_usage_mb()

    cold_start = time.perf_counter()
    tokenizer, model, load_error = load_model_and_tokenizer(model_id)
    cold_start_ms = round((time.perf_counter() - cold_start) * 1000)

    if load_error:
        return {"error": load_error, "model_id": model_id}

    ram_after = get_ram_usage_mb()
    model_ram_mb = round(max(0, ram_after - ram_before), 1)

    if warmup_runs > 0:
        warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
        for _ in range(warmup_runs):
            _single_inference(model, tokenizer, warmup_ids, max_new_tokens=5)

    results = []
    all_bleu = []
    all_chrf = []
    all_latencies = []
    all_success = []

    try:
        for prompt in prompts:
            category = prompt["category_id"]
            prompt_text = build_prompt(
                tokenizer,
                SYSTEM_PROMPT,
                prompt["instruction"],
                prompt["input"],
            )
            input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]

            run_latencies = []
            last_output = ""
            last_token_count = 0

            try:
                for _ in range(runs):
                    start = time.perf_counter()
                    raw_output, token_count = _single_inference(
                        model, tokenizer, input_ids, max_new_tokens
                    )
                    elapsed_ms = round((time.perf_counter() - start) * 1000)
                    run_latencies.append(elapsed_ms)
                    last_output = raw_output
                    last_token_count = token_count

                output = sanitize_output(last_output)

                run_latencies_sorted = sorted(run_latencies)
                p50_ms = run_latencies_sorted[len(run_latencies_sorted) // 2]
                p95_ms = run_latencies_sorted[int(len(run_latencies_sorted) * 0.95)]
                avg_ms = round(sum(run_latencies) / len(run_latencies))

                tokens_per_sec = round(
                    last_token_count / (avg_ms / 1000), 2
                ) if avg_ms > 0 else 0

                truncated = last_token_count >= max_new_tokens

                metrics = calculate_all_metrics(output, prompt["gold"], category)
                validation = validate_output(output, prompt)

                all_bleu.append(metrics["bleu"])
                all_chrf.append(metrics["chrf"])
                all_latencies.append(avg_ms)
                all_success.append(validation["task_success"])

                results.append({
                    "prompt_id": prompt["id"],
                    "category": category,
                    "input": prompt["input"],
                    "raw_output": last_output,
                    "sanitized_output": output,
                    "gold": prompt["gold"],
                    "metrics": metrics,
                    "validation": validation,
                    "latency_ms": avg_ms,
                    "p50_latency_ms": p50_ms,
                    "p95_latency_ms": p95_ms,
                    "all_run_latencies_ms": run_latencies,
                    "tokens_per_sec": tokens_per_sec,
                    "token_count": last_token_count,
                    "truncated": truncated,
                })

            except Exception as e:
                results.append({
                    "prompt_id": prompt["id"],
                    "category": category,
                    "input": prompt["input"],
                    "error": str(e),
                })

    except KeyboardInterrupt:
        print("\nBenchmark interrupted - saving partial results...")

    finally:
        del model
        gc.collect()

    per_category = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in per_category:
            per_category[cat] = {"bleu": [], "chrf": [], "success": []}
        if "metrics" in r:
            per_category[cat]["bleu"].append(r["metrics"].get("bleu", 0))
            per_category[cat]["chrf"].append(r["metrics"].get("chrf", 0))
        if "validation" in r:
            per_category[cat]["success"].append(
                r["validation"].get("task_success", False)
            )

    return {
        "model_id": model_id,
        "cold_start_ms": cold_start_ms,
        "model_ram_mb": model_ram_mb,
        "runs_per_prompt": runs,
        "warmup_runs": warmup_runs,
        "max_new_tokens": max_new_tokens,
        "total_prompts": len(prompts),
        "avg_bleu": round(sum(all_bleu) / len(all_bleu), 2) if all_bleu else 0,
        "avg_chrf": round(sum(all_chrf) / len(all_chrf), 2) if all_chrf else 0,
        "avg_latency_ms": round(sum(all_latencies) / len(all_latencies)) if all_latencies else 0,
        "task_success_rate": round(
            sum(all_success) / len(all_success) * 100, 1
        ) if all_success else 0,
        "per_category_metrics": {
            cat: {
                "avg_bleu": round(sum(v["bleu"]) / len(v["bleu"]), 2) if v["bleu"] else 0,
                "avg_chrf": round(sum(v["chrf"]) / len(v["chrf"]), 2) if v["chrf"] else 0,
                "success_rate": round(
                    sum(v["success"]) / len(v["success"]) * 100, 1
                ) if v["success"] else 0,
            }
            for cat, v in per_category.items()
        },
        "prompt_results": results,
        "environment": get_benchmark_env(),
    }

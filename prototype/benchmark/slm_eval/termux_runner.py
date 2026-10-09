"""Mobile-friendly benchmark runner for Termux/Android and lightweight Linux VMs.

Adds things the desktop CLI doesn't need: environment detection (Termux vs
VirtualBox vs bare Linux), a best-effort per-prompt timeout so one hung
generation doesn't stall the whole run, and 4-bit quantization when
available so larger models still fit in mobile RAM.

Run as: python -m slm_eval.termux_runner --model <hf-model-id> [options]
"""
import argparse
import json
import os
import platform
import signal
import subprocess
import time

import torch

from slm_eval.benchmarks.slm_benchmark import SYSTEM_PROMPT, load_dataset
from slm_eval.metrics.bleu import calculate_all_metrics
from slm_eval.metrics.device_metrics import get_ram_usage_mb
from slm_eval.models.loader import build_prompt
from slm_eval.utils.reproducibility import get_benchmark_env, set_seed
from slm_eval.validation.output_validator import sanitize_output, validate_output


def is_termux() -> bool:
    return (
        os.environ.get("TERMUX_VERSION") is not None
        or os.path.exists("/data/data/com.termux")
    )


def is_virtualbox() -> bool:
    dmi_paths = [
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/sys_vendor",
    ]
    for path in dmi_paths:
        try:
            with open(path) as f:
                if "virtualbox" in f.read().strip().lower():
                    return True
        except Exception:
            continue
    return False


def _run_getprop(prop: str):
    try:
        result = subprocess.run(
            ["getprop", prop], capture_output=True, text=True, timeout=5
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def _get_cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _get_total_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 2)
    except Exception:
        pass
    return None


def get_device_info() -> dict:
    termux = is_termux()
    vbox = is_virtualbox()
    if termux:
        environment = "termux"
    elif vbox:
        environment = "virtualbox"
    elif platform.system() != "Linux":
        environment = platform.system().lower()
    else:
        environment = "linux"
    info = {
        "environment": environment,
        "is_termux": termux,
        "is_virtualbox": vbox,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cpu_model": _get_cpu_model(),
        "total_ram_gb": _get_total_ram_gb(),
    }
    if termux:
        info["device_model"] = _run_getprop("ro.product.model")
        info["android_version"] = _run_getprop("ro.build.version.release")
    return info


class _PromptTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _PromptTimeout()


def _run_with_timeout(fn, timeout_s: int):
    """Best-effort per-call timeout.

    Uses SIGALRM on POSIX (Termux/Linux), which can genuinely interrupt a
    hung model.generate() call. There is no SIGALRM on Windows, and no safe
    way to kill a blocking native call from pure Python there, so on that
    platform the call just runs without an enforced timeout.
    """
    if not hasattr(signal, "SIGALRM"):
        return fn(), False
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        return fn(), False
    except _PromptTimeout:
        return None, True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_model_mobile(model_id: str):
    """Load a model 4-bit quantized when transformers+bitsandbytes are
    available (mobile-friendly), otherwise fall back to float32 with
    low_cpu_mem_usage=True. Returns (tokenizer, model, quantization_label).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="cpu",
            trust_remote_code=True,
        )
        quantization = "4bit"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        quantization = "float32"

    model.eval()
    return tokenizer, model, quantization


def run_termux_benchmark(
    model_id: str,
    category_filter: str = None,
    timeout_per_prompt: int = 60,
    max_new_tokens: int = 100,
    output_path: str = "termux_results.json",
) -> dict:
    set_seed()
    device_info = get_device_info()

    prompts = load_dataset(lang_filter="all", category_filter=category_filter)
    if not prompts:
        raise SystemExit(f"No prompts found for category '{category_filter}'")

    start_total = time.perf_counter()
    ram_before = get_ram_usage_mb()
    tokenizer, model, quantization = load_model_mobile(model_id)
    ram_after = get_ram_usage_mb()
    model_ram_mb = round(max(0, ram_after - ram_before), 1)

    results = []
    latencies = []
    bleu_scores = []
    chrf_scores = []
    successes = []
    ms_per_token_values = []
    timed_out_count = 0

    for prompt in prompts:
        prompt_start = time.perf_counter()
        prompt_text = build_prompt(
            tokenizer, SYSTEM_PROMPT, prompt["instruction"], prompt["input"]
        )

        prefill_start = time.perf_counter()
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)
        prefill_ms = round((time.perf_counter() - prefill_start) * 1000)

        def _generate():
            gen_start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
            gen_ms = round((time.perf_counter() - gen_start) * 1000)
            tokens_generated = output.shape[1] - input_ids.shape[1]
            new_tokens = output[0][input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return text, gen_ms, tokens_generated

        gen_result, timed_out = _run_with_timeout(_generate, timeout_per_prompt)
        latency_ms = round((time.perf_counter() - prompt_start) * 1000)

        if timed_out:
            timed_out_count += 1
            results.append({
                "prompt_id": prompt["id"],
                "category": prompt["category_id"],
                "input": prompt["input"],
                "error": f"timed out after {timeout_per_prompt}s",
                "latency_ms": latency_ms,
                "prefill_ms": prefill_ms,
                "gen_ms": None,
                "tokens_generated": None,
                "ms_per_token": None,
                "timed_out": True,
            })
            continue

        raw_output, gen_ms, tokens_generated = gen_result
        ms_per_token = round(gen_ms / tokens_generated, 2) if tokens_generated > 0 else 0.0

        output = sanitize_output(raw_output or "")
        metrics = calculate_all_metrics(output, prompt["gold"], prompt["category_id"])
        validation = validate_output(output, prompt)

        latencies.append(latency_ms)
        bleu_scores.append(metrics["bleu"])
        chrf_scores.append(metrics["chrf"])
        successes.append(validation["task_success"])
        ms_per_token_values.append(ms_per_token)

        results.append({
            "prompt_id": prompt["id"],
            "category": prompt["category_id"],
            "input": prompt["input"],
            "output": output,
            "gold": prompt["gold"],
            "metrics": metrics,
            "validation": validation,
            "latency_ms": latency_ms,
            "prefill_ms": prefill_ms,
            "gen_ms": gen_ms,
            "tokens_generated": tokens_generated,
            "ms_per_token": ms_per_token,
            "timed_out": False,
        })

    total_time_s = round(time.perf_counter() - start_total, 1)
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0

    sorted_ms_per_token = sorted(ms_per_token_values)
    avg_ms_per_token = (
        round(sum(sorted_ms_per_token) / len(sorted_ms_per_token), 2)
        if sorted_ms_per_token else 0.0
    )
    p50_ms_per_token = (
        sorted_ms_per_token[len(sorted_ms_per_token) // 2]
        if sorted_ms_per_token else 0.0
    )
    p95_ms_per_token = (
        sorted_ms_per_token[int(len(sorted_ms_per_token) * 0.95)]
        if sorted_ms_per_token else 0.0
    )

    summary = {
        "bleu": round(sum(bleu_scores) / len(bleu_scores), 2) if bleu_scores else 0,
        "chrf": round(sum(chrf_scores) / len(chrf_scores), 2) if chrf_scores else 0,
        "task_success_rate": round(sum(successes) / len(successes) * 100, 1) if successes else 0,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "avg_ms_per_token": avg_ms_per_token,
        "p50_ms_per_token": p50_ms_per_token,
        "p95_ms_per_token": p95_ms_per_token,
        "total_time_s": total_time_s,
        "prompts_timed_out": timed_out_count,
    }

    result = {
        "device_info": device_info,
        "model_id": model_id,
        "quantization": quantization,
        "model_ram_mb": model_ram_mb,
        "environment": get_benchmark_env(),
        "per_prompt_results": results,
        "summary": summary,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    _print_results_table(result)
    _print_summary_table(result)
    return result


def _print_results_table(result: dict):
    header = f"{'Prompt':<16}{'Latency(ms)':>12}{'Prefill(ms)':>12}{'Gen(ms)':>10}{'ms/tok':>10}{'Success':>9}"
    print()
    print(header)
    print("-" * len(header))
    for r in result["per_prompt_results"]:
        if r.get("timed_out"):
            print(
                f"{r['prompt_id']:<16}{r.get('latency_ms', 0):>12}"
                f"{r.get('prefill_ms', 0):>12}{'--':>10}{'--':>10}{'TIMEOUT':>9}"
            )
            continue
        success = r.get("validation", {}).get("task_success", False)
        print(
            f"{r['prompt_id']:<16}"
            f"{r.get('latency_ms', 0):>12}"
            f"{r.get('prefill_ms', 0):>12}"
            f"{r.get('gen_ms', 0):>10}"
            f"{r.get('ms_per_token', 0):>10.2f}"
            f"{str(success):>9}"
        )


def _print_summary_table(result: dict):
    s = result["summary"]
    device = result["device_info"]
    device_label = device.get("device_model") or device.get("cpu_model") or "unknown"
    print()
    print("=" * 44)
    print(" SLM Eval - Termux/Mobile Benchmark Summary")
    print("=" * 44)
    print(f"Model:          {result['model_id']}")
    print(f"Device:         {device.get('environment')} ({device_label})")
    print(f"Quantization:   {result['quantization']}")
    print(f"Model RAM:      {result['model_ram_mb']}MB")
    print(f"Total time:     {s['total_time_s']}s")
    print(f"P50 latency:    {s['p50_latency_ms']}ms")
    print(f"P95 latency:    {s['p95_latency_ms']}ms")
    print(f"Avg ms/token:   {s['avg_ms_per_token']}ms")
    print(f"P50 ms/token:   {s['p50_ms_per_token']}ms")
    print(f"P95 ms/token:   {s['p95_ms_per_token']}ms")
    print(f"BLEU:           {s['bleu']}")
    print(f"chrF:           {s['chrf']}")
    print(f"Task success:   {s['task_success_rate']}%")
    print(f"Timed out:      {s['prompts_timed_out']} prompts")
    print("=" * 44)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Termux/Android-friendly SLM benchmark runner"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--category", default=None, help="Restrict to one dataset category id")
    parser.add_argument(
        "--timeout-per-prompt", type=int, default=60, dest="timeout_per_prompt",
        help="Per-prompt generation timeout in seconds (default 60)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100, dest="max_new_tokens",
    )
    parser.add_argument("--output", default="termux_results.json")
    args = parser.parse_args()

    run_termux_benchmark(
        args.model,
        category_filter=args.category,
        timeout_per_prompt=args.timeout_per_prompt,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

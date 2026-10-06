import pandas as pd
from datetime import datetime
from pathlib import Path


def save_csv(benchmark_result: dict, device_info: dict, output_dir: str = "results") -> str:
    output_path = Path(__file__).parent.parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    device_name = str(device_info.get("device_name", "unknown")).replace(" ", "_")
    safe_device = "".join(c for c in device_name if c.isalnum() or c in "_-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"slm_benchmark_{safe_device}_{timestamp}.csv"

    rows = []
    for r in benchmark_result.get("prompt_results", []):
        metrics = r.get("metrics", {})
        rouge = metrics.get("rouge", {})
        validation = r.get("validation", {})
        rows.append({
            "Model": benchmark_result["model_id"],
            "Category": r.get("category"),
            "PromptID": r.get("prompt_id"),
            "Input": r.get("input"),
            "SanitizedOutput": r.get("sanitized_output", r.get("error", "ERROR")),
            "BLEU": metrics.get("bleu", 0),
            "chrF": metrics.get("chrf", 0),
            "ROUGE1": rouge.get("rouge1", 0),
            "ROUGE2": rouge.get("rouge2", 0),
            "ROUGEL": rouge.get("rougeL", 0),
            "ExactMatch": metrics.get("exact_match", False),
            "Latency_ms_avg": r.get("latency_ms", 0),
            "Latency_ms_p50": r.get("p50_latency_ms", 0),
            "Latency_ms_p95": r.get("p95_latency_ms", 0),
            "TokensPerSec": r.get("tokens_per_sec", 0),
            "Truncated": r.get("truncated", False),
            "TaskSuccess": validation.get("task_success", False),
            "ScriptRatio": validation.get("script_ratio", 0),
            "EchoDetected": validation.get("echo_detected", False),
            "ContaminationDetected": validation.get("contamination_detected", False),
            "ProtectedOK": validation.get("protected_ok", True),
            "ModelRAM_MB": benchmark_result.get("model_ram_mb", 0),
            "ColdStart_ms": benchmark_result.get("cold_start_ms", 0),
            "RunsPerPrompt": benchmark_result.get("runs_per_prompt", 1),
            "Device": device_info.get("device_name"),
            "Platform": device_info.get("platform"),
            "Architecture": device_info.get("architecture"),
            "TotalRAM_GB": device_info.get("total_ram_gb"),
            "DeviceProfile": device_info.get("device_profile"),
            "IsARM": device_info.get("is_arm"),
            "Battery_pct": device_info.get("battery_percent_before"),
            "CPUTemp_C": device_info.get("cpu_temp_celsius_before"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(str(filename), index=False, encoding="utf-8")
    return str(filename)

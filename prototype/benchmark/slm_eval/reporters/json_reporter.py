import json
from datetime import datetime
from pathlib import Path


def save_json(benchmark_result: dict, device_info: dict, output_dir: str = "results") -> str:
    output_path = Path(__file__).parent.parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    device_name = str(device_info.get("device_name", "unknown")).replace(" ", "_")
    safe_device = "".join(c for c in device_name if c.isalnum() or c in "_-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"slm_benchmark_{safe_device}_{timestamp}.json"

    with open(str(filename), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "device": device_info,
            "results": benchmark_result,
        }, f, indent=2, ensure_ascii=False)

    return str(filename)

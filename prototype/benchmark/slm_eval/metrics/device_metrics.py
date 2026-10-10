import os
import platform

try:
    import psutil
except ImportError:
    psutil = None


def get_device_info(device_name: str = "unknown") -> dict:
    info = {
        "device_name": device_name,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "is_android": _is_android(),
        "is_arm": "arm" in platform.machine().lower() or "aarch" in platform.machine().lower(),
    }

    if info["is_android"]:
        android_ram = _get_android_ram_gb()
        if android_ram:
            info["total_ram_gb"] = android_ram
        total = info["total_ram_gb"]
        if total < 4:
            info["device_profile"] = "budget_mobile"
        elif total < 7:
            info["device_profile"] = "midrange_mobile"
        else:
            info["device_profile"] = "flagship_mobile"
    else:
        info["device_profile"] = "desktop"

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch_version"] = "not_installed"
        info["cuda_available"] = False

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "not_installed"

    info["battery_percent_before"] = _get_battery()
    info["cpu_temp_celsius_before"] = _get_cpu_temp()

    return info


def get_ram_usage_mb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def get_peak_ram_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        return get_ram_usage_mb()


def _is_android() -> bool:
    return (
        os.environ.get("TERMUX_VERSION") is not None
        or os.environ.get("PREFIX", "").startswith("/data/data/com.termux")
    )


def _get_android_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 2)
    except Exception:
        return None
    return None


def _get_battery():
    paths = [
        "/sys/class/power_supply/battery/capacity",
        "/sys/class/power_supply/BAT0/capacity",
        "/sys/class/power_supply/BAT1/capacity",
    ]
    for path in paths:
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            continue
    return None


def _get_cpu_temp():
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
    ]
    for path in paths:
        try:
            with open(path) as f:
                return round(int(f.read().strip()) / 1000, 1)
        except Exception:
            continue
    return None

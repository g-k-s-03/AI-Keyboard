import platform
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed all RNGs used during benchmarking so greedy decoding runs
    are reproducible across invocations on the same device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_benchmark_env(seed: int = DEFAULT_SEED) -> dict:
    """Capture the runtime environment a benchmark run was produced under,
    so results can be reproduced or discounted later."""
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not_installed"

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "platform": platform.platform(),
        "seed_used": seed,
    }

import torch
import random
import numpy as np

from specfold.registry import FASTDLLM_VARIANT_TO_KEY, MODEL_SPECS


def current_gpu_name(device=None):
    """torch device name for the active GPU, or None on CPU runs."""
    if not torch.cuda.is_available():
        return None
    index = 0
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is not None:
        index = device.index
    elif isinstance(device, int):
        index = device
    return torch.cuda.get_device_name(index)


def normalize_fastdllm_v2_variant(model_variant=None):
    raw = (model_variant or "1.5b").strip().lower()
    normalized = FASTDLLM_VARIANT_TO_KEY.get(raw)
    if normalized is None:
        supported = ", ".join(sorted(FASTDLLM_VARIANT_TO_KEY))
        raise ValueError(
            f"Unsupported Fast-dLLM v2 variant {model_variant!r}. "
            f"Use one of: {supported}"
        )
    return MODEL_SPECS[normalized].model_variant or raw


def resolve_fastdllm_v2_model_path(model_variant=None, pretrained_path=None):
    if pretrained_path:
        return pretrained_path

    normalized_variant = normalize_fastdllm_v2_variant(model_variant)
    if normalized_variant == "1.5b":
        return MODEL_SPECS["fastdllm_v2_1.5b"].pretrained_path
    return MODEL_SPECS["fastdllm_v2_7b"].pretrained_path


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

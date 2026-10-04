"""Central registry for supported Spiffy model and dataset names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Description of one evaluable checkpoint family."""

    key: str
    display_name: str
    family: str
    pretrained_path: str
    lora_path: Optional[str] = None
    model_variant: Optional[str] = None
    graph_prefix: Optional[str] = None

    @property
    def calibration_model_id(self) -> str:
        """Identifier used for calibration graph directories."""
        return self.lora_path or self.pretrained_path


MODEL_SPECS: Dict[str, ModelSpec] = {
    "fastdllm_v2_1.5b": ModelSpec(
        key="fastdllm_v2_1.5b",
        display_name="Fast-dLLM v2 1.5B",
        family="fastdllm_v2",
        pretrained_path="Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        model_variant="1.5b",
        graph_prefix="fastdllm_v2",
    ),
    "fastdllm_v2_7b": ModelSpec(
        key="fastdllm_v2_7b",
        display_name="Fast-dLLM v2 7B",
        family="fastdllm_v2",
        pretrained_path="Efficient-Large-Model/Fast_dLLM_v2_7B",
        model_variant="7b",
        graph_prefix="fastdllm_v2",
    ),
    "nld_3b": ModelSpec(
        key="nld_3b",
        display_name="Nemotron-Labs-Diffusion 3B",
        family="nld",
        pretrained_path="nvidia/Nemotron-Labs-Diffusion-3B",
        model_variant="3b",
        graph_prefix="nld",
    ),
    "nld_8b": ModelSpec(
        key="nld_8b",
        display_name="Nemotron-Labs-Diffusion 8B",
        family="nld",
        pretrained_path="nvidia/Nemotron-Labs-Diffusion-8B",
        model_variant="8b",
        graph_prefix="nld",
    ),
    "nld_14b": ModelSpec(
        key="nld_14b",
        display_name="Nemotron-Labs-Diffusion 14B",
        family="nld",
        pretrained_path="nvidia/Nemotron-Labs-Diffusion-14B",
        model_variant="14b",
        graph_prefix="nld",
    ),
}

NLD_MODELS = ("nld_3b", "nld_8b", "nld_14b")

MODEL_ALIASES = {
    "fastdllm_v2_1.5b": "fastdllm_v2_1.5b",
    "fast-dllm-v2-1.5b": "fastdllm_v2_1.5b",
    "fast_dllm_v2_1.5b": "fastdllm_v2_1.5b",
    "fastdllm_v2_1_5b": "fastdllm_v2_1.5b",
    "fastdllm_v2_7b": "fastdllm_v2_7b",
    "fast-dllm-v2-7b": "fastdllm_v2_7b",
    "fast_dllm_v2_7b": "fastdllm_v2_7b",
    "nld_3b": "nld_3b",
    "nld-3b": "nld_3b",
    "nemotron_labs_diffusion_3b": "nld_3b",
    "nld_8b": "nld_8b",
    "nld-8b": "nld_8b",
    "nemotron_labs_diffusion_8b": "nld_8b",
    "nld_14b": "nld_14b",
    "nld-14b": "nld_14b",
    "nemotron_labs_diffusion_14b": "nld_14b",
}

FASTDLLM_VARIANT_TO_KEY = {
    "1.5": "fastdllm_v2_1.5b",
    "1.5b": "fastdllm_v2_1.5b",
    "1_5b": "fastdllm_v2_1.5b",
    "1-5b": "fastdllm_v2_1.5b",
    "7": "fastdllm_v2_7b",
    "7b": "fastdllm_v2_7b",
}

DATASET_ALIASES = {
    "gsm8k": "gsm8k",
    "human_eval": "humaneval",
    "human_eval_plus": "humaneval_plus",
    "humaneval": "humaneval",
    "humaneval_plus": "humaneval_plus",
    "mbpp": "mbpp",
    "mbpp_plus": "mbpp_plus",
    "ifeval": "ifeval",
    "if_eval": "ifeval",
    "math": "minervamath",
    "minerva_math": "minervamath",
    "minervamath": "minervamath",
    "math500": "math500",
    "math_500": "math500",
    "minerva_math500": "math500",
}

DATASET_NAMES = ("gsm8k", "humaneval", "humaneval_plus", "mbpp", "mbpp_plus", "minervamath", "math500", "ifeval")


def normalize_model_key(model: str, model_variant: Optional[str] = None) -> str:
    """Normalize user CLI model labels into a key in ``MODEL_SPECS``."""
    raw = (model or "").strip().lower().replace("/", "_")
    if raw in {"fastdllm_v2", "fast-dllm-v2", "fast_dllm_v2", "fastdllm"}:
        variant = (model_variant or "1.5b").strip().lower()
        if variant not in FASTDLLM_VARIANT_TO_KEY:
            supported = ", ".join(sorted(FASTDLLM_VARIANT_TO_KEY))
            raise ValueError(f"Unsupported Fast-dLLM v2 variant '{model_variant}'. Use one of: {supported}")
        return FASTDLLM_VARIANT_TO_KEY[variant]
    key = MODEL_ALIASES.get(raw, raw)
    if key not in MODEL_SPECS:
        supported = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unsupported model '{model}'. Use one of: {supported}")
    return key


def get_model_spec(model: str, model_variant: Optional[str] = None) -> ModelSpec:
    return MODEL_SPECS[normalize_model_key(model, model_variant=model_variant)]


def normalize_dataset_name(dataset: str) -> str:
    raw = (dataset or "").strip().lower().replace("-", "_")
    key = DATASET_ALIASES.get(raw, raw)
    if key not in DATASET_NAMES:
        supported = ", ".join(DATASET_NAMES) + ", math"
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of: {supported}")
    return key


def expand_datasets(dataset: str) -> List[str]:
    if dataset == "all":
        return list(DATASET_NAMES)
    return [normalize_dataset_name(dataset)]


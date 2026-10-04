"""Load a draft graph from disk, or calibrate and save one."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple

from specfold.calibration.fastdllm_v2_calibration import FastDLLMV2CalibrationPipeline
from specfold.calibration.nld_calibration import NLDCalibrationPipeline
from specfold.calibration.paths import default_model_calibration_set_path
from specfold.config import Config
from specfold.registry import get_model_spec

logger = logging.getLogger(__name__)


def _build_pipeline(model_key: str, pretrained_path: Optional[str], **kwargs):
    spec = get_model_spec(model_key)
    if spec.family == "fastdllm_v2":
        return FastDLLMV2CalibrationPipeline(model_variant=spec.model_variant, pretrained_path=pretrained_path, **kwargs)
    if spec.family == "nld":
        return NLDCalibrationPipeline(model_key=model_key, pretrained_path=pretrained_path, **kwargs)
    raise ValueError(f"No calibration pipeline for model family '{spec.family}' (key={model_key})")


def load_or_build_calibration_graph(
    model_key: str,
    pretrained_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "bfloat16",
    num_samples: int = Config.NUM_SAMPLES,
    max_seq_len: int = Config.MAX_SEQ_LEN,
    gen_length: int = Config.GEN_LENGTH,
    block_length: int = Config.BLOCK_LENGTH,
    depth_D: int = Config.DEPTH_D,
    budget_B: int = Config.BUDGET_B,
    per_step_limit: int = Config.PER_STEP_LIMIT,
    temperature: float = Config.TEMPERATURE,
    calibration_suffix: Optional[str] = "",
    calibration_set: str = Config.CALIBRATION_SET,
) -> Tuple[Dict[str, Any], str]:
    """Return (graph, path) for one (model, calibration set, D, B, S); the graph is shared by all benchmarks."""
    path = default_model_calibration_set_path(
        model_key=model_key,
        calibration_set=calibration_set,
        pretrained_path=pretrained_path,
        depth_D=depth_D,
        budget_B=budget_B,
        per_step_limit=per_step_limit,
        calibration_suffix=calibration_suffix,
    )
    if os.path.exists(path):
        logger.info("Loading draft graph from %s", path)
        with open(path, "rb") as f:
            return pickle.load(f), path

    logger.info("No draft graph at %s; calibrating one now", path)
    pipeline = _build_pipeline(
        model_key,
        pretrained_path,
        device=device,
        dtype=dtype,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        gen_length=gen_length,
        block_length=block_length,
        depth_D=depth_D,
        budget_B=budget_B,
        per_step_limit=per_step_limit,
        temperature=temperature,
        calibration_set=calibration_set,
    )
    pipeline.setup_model()
    pipeline.setup_dataset()
    graph = pipeline.run_calibration()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    logger.info("Saved draft graph to %s", path)
    return graph, path

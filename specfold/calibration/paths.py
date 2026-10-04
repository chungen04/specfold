"""Draft-graph file locations."""

from __future__ import annotations

import os
from typing import Optional

from specfold.config import Config
from specfold.registry import get_model_spec


def normalize_calibration_suffix(suffix: Optional[str] = "") -> str:
    if not suffix:
        return ""
    suffix = str(suffix)
    return suffix if suffix.startswith("_") else f"_{suffix}"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(os.sep, "_")


def default_model_calibration_set_path(
    model_key: str,
    calibration_set: str,
    model_variant: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    depth_D: int = Config.DEPTH_D,
    budget_B: int = Config.BUDGET_B,
    per_step_limit: int = Config.PER_STEP_LIMIT,
    root_dir: str = "calibration_graph",
    calibration_suffix: Optional[str] = "",
) -> str:
    """calibration_graph/<model>/<set>/<prefix>_<set>_calibration_graph_D<D>_B<B>_S<S>.pkl"""
    spec = get_model_spec(model_key, model_variant=model_variant)
    model_name = pretrained_path or spec.calibration_model_id
    filename = (
        f"{spec.graph_prefix or spec.family}_{calibration_set}_calibration_graph_"
        f"D{depth_D}_B{budget_B}_S{per_step_limit}"
        f"{normalize_calibration_suffix(calibration_suffix)}.pkl"
    )
    return os.path.join(root_dir, sanitize_model_name(model_name), calibration_set, filename)

"""Calibration package exports, imported lazily."""

__all__ = [
    "BaseCalibrationPipeline",
    "FastDLLMV2CalibrationPipeline",
    "Node",
    "can_be_child",
    "count_tree_nodes",
]


def __getattr__(name):
    if name == "BaseCalibrationPipeline":
        from .calibration import BaseCalibrationPipeline
        return BaseCalibrationPipeline
    if name == "FastDLLMV2CalibrationPipeline":
        from .fastdllm_v2_calibration import FastDLLMV2CalibrationPipeline
        return FastDLLMV2CalibrationPipeline
    if name in {"Node", "can_be_child", "count_tree_nodes"}:
        from .graph import Node, can_be_child, count_tree_nodes
        return {"Node": Node, "can_be_child": can_be_child, "count_tree_nodes": count_tree_nodes}[name]
    raise AttributeError(name)
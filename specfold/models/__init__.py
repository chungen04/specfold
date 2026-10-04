"""Model integration components for Spiffy, imported lazily."""

__all__ = [
    "FastDLLMV2ModelWrapper",
    "ModelWrapper",
]


def __getattr__(name):
    if name == "ModelWrapper":
        from .model_wrapper import ModelWrapper
        return ModelWrapper
    if name == "FastDLLMV2ModelWrapper":
        from .fastdllm_v2_model import FastDLLMV2ModelWrapper
        return FastDLLMV2ModelWrapper
    raise AttributeError(name)

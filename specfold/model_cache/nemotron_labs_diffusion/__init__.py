"""Vendored Nemotron-Labs-Diffusion modeling code (upstream HF remote code targets
transformers>=5.0; this copy is patched for the 4.57 pin, see the patch-site
comments). Weights still come from the HF checkpoint."""

from .configuration_nemotron_labs_diffusion import NemotronLabsDiffusionConfig
from .modeling_nemotron_labs_diffusion import NemotronLabsDiffusionModel

__all__ = ["NemotronLabsDiffusionConfig", "NemotronLabsDiffusionModel"]

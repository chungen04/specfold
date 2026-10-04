"""Loads a Nemotron-Labs-Diffusion checkpoint with the vendored modeling code and runs
its forward passes (the NLD counterpart of FastDLLMModelExecutor)."""

import torch
from typing import Optional
from transformers import AutoTokenizer

from specfold.model_cache.nemotron_labs_diffusion import (
    NemotronLabsDiffusionConfig,
    NemotronLabsDiffusionModel,
)


class NLDModelExecutor:
    """Handles Nemotron-Labs-Diffusion model loading and forward passes."""

    def __init__(self,
                 pretrained_path: str,
                 device: torch.device,
                 target_dtype: torch.dtype):
        """
        Args:
            pretrained_path: HF path (e.g., "nvidia/Nemotron-Labs-Diffusion-3B")
            device: Device to run model on
            target_dtype: Target dtype for model
        """
        self.pretrained_path = pretrained_path
        self.device = device
        self.target_dtype = target_dtype

        self._setup_model()
        self._setup_tokenizer()

    def _setup_model(self):
        config = NemotronLabsDiffusionConfig.from_pretrained(self.pretrained_path)
        self.model = NemotronLabsDiffusionModel.from_pretrained(
            self.pretrained_path,
            config=config,
            torch_dtype=self.target_dtype,
        ).eval()
        self.model = self.model.to(self.device)

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def set_diffusion_lm(self, enabled: bool):
        """Toggle bidirectional (diffusion) vs causal (AR) attention per layer."""
        for layer in self.model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = enabled

    def forward(self,
                input_seq: torch.Tensor,
                past_key_values=None,
                use_cache: bool = False,
                use_causal_mask: bool = False):
        """Run one model forward pass; returns NemotronLabsDiffusionOutputWithPast."""
        return self.model(
            input_seq,
            past_key_values=past_key_values,
            use_cache=use_cache,
            use_causal_mask=use_causal_mask,
        )

    def get_tokenizer(self):
        return self.tokenizer

    def get_eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def get_mask_id(self) -> int:
        return self.model.mask_token_id

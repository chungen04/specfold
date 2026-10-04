"""Fast-dLLM v2 wrapper around the HF checkpoint's remote-code generation/forward API
(used by calibration)."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from specfold.generate.fastdllm_model_executor import FastDLLMModelExecutor
from specfold.models.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


class FastDLLMV2ModelWrapper(ModelWrapper):
    """Reusable wrapper for Fast-dLLM v2 1.5B/7B checkpoints."""

    def __init__(
        self,
        model_name: str = "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        device: str = "auto",
        dtype: str = "bfloat16",
        use_chat_template: bool = True,
        small_block_size: int = 8,
        threshold: float = 0.9,
        block_size: int = 32,
        max_new_tokens: int = 512,
    ):
        super().__init__(model_name=model_name, device=device)
        self.dtype = dtype
        self.use_chat_template = use_chat_template
        self.small_block_size = small_block_size
        self.threshold = threshold
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.executor: Optional[FastDLLMModelExecutor] = None
        self.target_dtype = self._resolve_dtype()

    def _resolve_dtype(self) -> torch.dtype:
        dtype = str(self.dtype).replace("torch.", "").lower()
        if dtype == "bfloat16" and (self.device.type != "cuda" or torch.cuda.is_bf16_supported()):
            return torch.bfloat16
        if dtype in {"float16", "fp16", "half"} and self.device.type == "cuda":
            return torch.float16
        return torch.float32

    def load_model(self) -> None:
        logger.info("Loading Fast-dLLM v2 model: %s", self.model_name)
        self.executor = FastDLLMModelExecutor(
            pretrained_path=self.model_name,
            device=self.device,
            target_dtype=self.target_dtype,
        )
        self.model = self.executor.model
        self.tokenizer = self.executor.get_tokenizer()
        logger.info("Fast-dLLM v2 wrapper loaded on %s", self.device)

    def prepare_input(self, text: str) -> torch.Tensor:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].to(self.device)

    @staticmethod
    def shift_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

    def get_logits(self, input_ids: torch.Tensor, timestep: int = 0) -> torch.Tensor:
        if self.executor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        outputs = self.executor.forward(
            input_ids.to(self.device),
            use_cache=True,
            update_past_key_values=False,
        )
        return self.shift_logits(outputs.logits)

    def get_target_distribution(self, input_ids: torch.Tensor, max_timesteps: int = 50):
        logits = self.get_logits(input_ids)
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().float().numpy()[None, ...].repeat(max_timesteps, axis=0)

    @torch.no_grad()
    def generate_sequence(self, input_ids: torch.Tensor, max_length: int = 512, **kwargs) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        max_new_tokens = int(kwargs.pop("max_new_tokens", self.max_new_tokens or max_length))
        return self.model.generate(
            input_ids.to(self.device),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            small_block_size=kwargs.pop("small_block_size", self.small_block_size),
            threshold=kwargs.pop("threshold", self.threshold),
            **kwargs,
        )

    def get_mask_id(self) -> int:
        return FastDLLMModelExecutor.MASK_ID

    def get_stop_token_id(self) -> int:
        return FastDLLMModelExecutor.STOP_TOKEN

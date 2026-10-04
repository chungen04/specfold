"""Fast-dLLM v2 draft graph calibration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from specfold.calibration.calibration import BaseCalibrationPipeline
from specfold.config import Config
from specfold.generate.fastdllm_model_executor import FastDLLMModelExecutor
from specfold.models.fastdllm_v2_model import FastDLLMV2ModelWrapper
from specfold.utils import resolve_fastdllm_v2_model_path

logger = logging.getLogger(__name__)


class FastDLLMV2CalibrationPipeline(BaseCalibrationPipeline):
    """Calibration pipeline that collects draft patterns from Fast-dLLM v2 itself."""

    def __init__(
        self,
        model_variant: str = "1.5b",
        pretrained_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 4096,
        gen_length: int = Config.GEN_LENGTH,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        block_length: int = Config.BLOCK_LENGTH,
        small_block_size: int = 8,
        threshold: float = 0.9,
        num_samples: int = Config.NUM_SAMPLES,
        max_seq_len: int = Config.MAX_SEQ_LEN,
        depth_D: int = Config.DEPTH_D,
        budget_B: int = Config.BUDGET_B,
        per_step_limit: int = Config.PER_STEP_LIMIT,
        calibration_set: str = Config.CALIBRATION_SET,
    ):
        normalized_variant = "1.5b" if str(model_variant).lower() in {"1.5", "1.5b", "1_5b"} else str(model_variant).lower()
        model_key = f"fastdllm_v2_{normalized_variant}"
        resolved_path = resolve_fastdllm_v2_model_path(
            model_variant=normalized_variant,
            pretrained_path=pretrained_path,
        )
        super().__init__(
            model_key=model_key,
            pretrained_path=resolved_path,
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
            use_chat_template=True,
            calibration_set=calibration_set,
        )
        self.model_variant = normalized_variant
        self.max_length = max_length
        self.top_p = top_p
        self.top_k = top_k
        self.small_block_size = small_block_size
        self.threshold = threshold
        self.model_executor: Optional[FastDLLMModelExecutor] = None
        self.model_wrapper: Optional[FastDLLMV2ModelWrapper] = None
        self.tokenizer = None

    def _target_dtype(self) -> torch.dtype:
        dtype = str(self.dtype).replace("torch.", "").lower()
        if dtype == "bfloat16" and (self._torch_device().type != "cuda" or torch.cuda.is_bf16_supported()):
            return torch.bfloat16
        if dtype in {"float16", "fp16", "half"} and self._torch_device().type == "cuda":
            return torch.float16
        return torch.float32

    def _torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device if torch.cuda.is_available() or not str(self.device).startswith("cuda") else "cpu")

    def setup_model(self) -> None:
        logger.info("Setting up Fast-dLLM v2 model: %s", self.pretrained_path)
        self.model_wrapper = FastDLLMV2ModelWrapper(
            model_name=self.pretrained_path,
            device=str(self._torch_device()),
            dtype=self.dtype,
            use_chat_template=True,
            small_block_size=self.small_block_size,
            threshold=self.threshold,
            block_size=self.block_length,
            max_new_tokens=self.gen_length,
        )
        self.model_wrapper.load_model()
        self.model_executor = self.model_wrapper.executor
        self.tokenizer = self.model_wrapper.tokenizer
        # Used by BaseCalibrationPipeline.collect_calibration_data to verify setup_model ran.
        self.model = self.model_wrapper
        logger.info("Fast-dLLM v2 model loaded on %s", self.model_executor.device)

    def prepare_input(
        self,
        prompt: str,
        apply_chat_template: Optional[bool] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        use_template = self.use_chat_template if apply_chat_template is None else apply_chat_template
        if use_template and hasattr(self.tokenizer, "apply_chat_template"):
            chat_history = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt
        tokens = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        )
        return tokens["input_ids"].to(self.model_executor.device)

    def run_calibration(self) -> Dict[str, Any]:
        result = super().run_calibration()
        result["metadata"].update(
            {
                "model_family": "fastdllm_v2",
                "model_variant": self.model_variant,
                "pretrained_path": self.pretrained_path,
                "small_block_size": self.small_block_size,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "threshold": self.threshold,
                "max_length": self.max_length,
            }
        )
        return result

    def _collect_sample_calibration_data(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[List[Tuple[int, Tuple[int, int]]], List[Tuple[int, List[Tuple[int, int]]]]]:
        calibration_data: List[Tuple[int, Tuple[int, int]]] = []
        sequence_history: List[Tuple[int, List[Tuple[int, int]]]] = []

        original_input_length = input_ids.shape[1]
        x_t = input_ids.clone()
        num_blocks = max(1, self.gen_length // self.block_length)
        num_small_blocks = max(1, self.block_length // self.small_block_size)
        past_key_values = None
        stop_generation = False

        if x_t.shape[1] > self.block_length:
            prefill_length = (x_t.shape[1] // self.block_length) * self.block_length
            outputs = self.model_executor.forward(
                x_t[:, :prefill_length],
                past_key_values=None,
                use_cache=True,
                update_past_key_values=True,
            )
            past_key_values = outputs.past_key_values
            if x_t.shape[1] % self.block_length == 0:
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                x_t = torch.cat([x_t, next_token], dim=1)

        for _ in range(num_blocks):
            if self.model_executor.get_eos_token_id() in x_t[:, original_input_length:]:
                break

            prompt_length = x_t.shape[1]
            block_init_length = self.block_length - (prompt_length % self.block_length)
            if block_init_length == 0:
                block_init_length = self.block_length
            x_init = torch.full(
                (x_t.shape[0], block_init_length),
                self.model_executor.get_mask_id(),
                device=self.model_executor.device,
                dtype=torch.long,
            )
            x_t = torch.cat([x_t, x_init], dim=1)

            for small_block_idx in range(num_small_blocks):
                small_block_start = small_block_idx * self.small_block_size
                small_block_end = small_block_start + self.small_block_size
                start = -self.block_length + small_block_start
                end = None if self.block_length == small_block_end else -self.block_length + small_block_end
                hist: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

                while True:
                    mask_idx = x_t[:, -self.block_length:] == self.model_executor.get_mask_id()
                    small_mask_idx = mask_idx[:, small_block_start:small_block_end]
                    if small_mask_idx.sum() == 0:
                        break

                    if self.model_executor.get_eos_token_id() in x_t[:, original_input_length:]:
                        stop_token_positions = (
                            x_t[:, original_input_length:] == self.model_executor.get_eos_token_id()
                        ).nonzero()
                        if stop_token_positions.numel() > 0:
                            stop_idx = stop_token_positions[0][1]
                            if (
                                x_t[:, original_input_length : original_input_length + stop_idx]
                                == self.model_executor.get_mask_id()
                            ).sum() == 0:
                                stop_generation = True
                                break

                    outputs = self.model_executor.forward(
                        x_t[:, -self.block_length:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        update_past_key_values=False,
                    )
                    full_logits = self._shift_logits(outputs.logits)
                    small_logits = full_logits[:, small_block_start:small_block_end, :]

                    sampled_tokens, probs = self.model_executor.sample_with_top_p(
                        small_logits,
                        top_p=self.top_p,
                        temperature=self.temperature,
                    )
                    sampled_confidence = torch.gather(
                        probs,
                        dim=-1,
                        index=sampled_tokens.unsqueeze(-1),
                    ).squeeze(-1)
                    sampled_confidence = torch.where(
                        small_mask_idx,
                        sampled_confidence,
                        torch.full_like(sampled_confidence, float("-inf")),
                    )

                    unmask_idx = sampled_confidence > self.threshold
                    max_prob_idx = sampled_confidence.argmax(dim=-1)
                    unmask_idx[
                        torch.arange(sampled_tokens.shape[0], device=self.model_executor.device),
                        max_prob_idx,
                    ] = True
                    unmask_idx = unmask_idx & small_mask_idx

                    newly_unmasked_positions = unmask_idx[0].nonzero(as_tuple=True)[0]
                    newly_unmasked_tokens = sampled_tokens[0, newly_unmasked_positions]
                    hist.append(
                        (
                            newly_unmasked_tokens.detach().cpu(),
                            newly_unmasked_positions.detach().cpu(),
                            sampled_confidence[0].detach().cpu(),
                            probs[0].detach().cpu(),
                        )
                    )

                    if end is not None:
                        x_t[:, start:end][unmask_idx] = sampled_tokens[unmask_idx]
                    else:
                        x_t[:, start:][unmask_idx] = sampled_tokens[unmask_idx]

                self._collect_history_tuples(hist, calibration_data, sequence_history)
                if stop_generation:
                    break

            outputs = self.model_executor.forward(
                x_t[:, -self.block_length:],
                past_key_values=past_key_values,
                use_cache=True,
                update_past_key_values=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            x_t = torch.cat([x_t, next_token], dim=1)

            if stop_generation:
                break

        return calibration_data, sequence_history

    def _collect_history_tuples(
        self,
        hist: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        calibration_data: List[Tuple[int, Tuple[int, int]]],
        sequence_history: List[Tuple[int, List[Tuple[int, int]]]],
    ) -> None:
        for t in range(len(hist)):
            if t == 0:
                continue
            gen_history: List[Tuple[int, int]] = []
            for t_prime in range(t, min(t + self.depth_D, len(hist))):
                unmasked_tokens_at_t_prime, true_positions_at_t_prime, _, _ = hist[t_prime]
                _, _, confidence_1d_at_t, p_block_at_t = hist[t]
                L = t_prime - t + 1
                if true_positions_at_t_prime.numel() == 0:
                    continue
                position_ranks = torch.argsort(torch.argsort(confidence_1d_at_t, descending=True))
                for token, position in zip(unmasked_tokens_at_t_prime, true_positions_at_t_prime):
                    i = int(position_ranks[position].item())
                    token_probs = p_block_at_t[position, :]
                    vocab_ranks = torch.argsort(torch.argsort(token_probs, descending=True))
                    j = int(vocab_ranks[int(token.item())].item())
                    calibration_data.append((L, (i, j)))
                    gen_history.append((i, j))
                    sequence_history.append((L, gen_history.copy()))

    @staticmethod
    def _shift_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

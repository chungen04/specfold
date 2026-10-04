"""Nemotron-Labs-Diffusion draft-graph calibration. NLD denoises the whole block per
step (no sub-blocks), so ranks are over the full block's masked positions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from specfold.calibration.calibration import BaseCalibrationPipeline
from specfold.config import Config
from specfold.generate.nld_model_executor import NLDModelExecutor
from specfold.model_cache.nemotron_labs_diffusion.modeling_nemotron_labs_diffusion import (
    _get_num_transfer_tokens,
    _get_transfer_index,
)
from specfold.registry import get_model_spec

logger = logging.getLogger(__name__)


class NLDCalibrationPipeline(BaseCalibrationPipeline):
    """Calibration pipeline that collects draft patterns from NLD itself."""

    def __init__(
        self,
        model_key: str = "nld_3b",
        pretrained_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 4096,
        gen_length: int = Config.GEN_LENGTH,
        temperature: float = 0.0,
        block_length: int = Config.BLOCK_LENGTH,
        threshold: float = 0.9,
        num_samples: int = Config.NUM_SAMPLES,
        max_seq_len: int = Config.MAX_SEQ_LEN,
        depth_D: int = Config.DEPTH_D,
        budget_B: int = Config.BUDGET_B,
        per_step_limit: int = Config.PER_STEP_LIMIT,
        calibration_set: str = Config.CALIBRATION_SET,
    ):
        spec = get_model_spec(model_key)
        super().__init__(
            model_key=spec.key,
            pretrained_path=pretrained_path or spec.pretrained_path,
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
        self.max_length = max_length
        self.threshold = threshold
        self.model_executor: Optional[NLDModelExecutor] = None
        self.tokenizer = None

    def _torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device if torch.cuda.is_available() or not str(self.device).startswith("cuda") else "cpu")

    def _target_dtype(self) -> torch.dtype:
        dtype = str(self.dtype).replace("torch.", "").lower()
        if dtype == "bfloat16" and (self._torch_device().type != "cuda" or torch.cuda.is_bf16_supported()):
            return torch.bfloat16
        if dtype in {"float16", "fp16", "half"} and self._torch_device().type == "cuda":
            return torch.float16
        return torch.float32

    def setup_model(self) -> None:
        logger.info("Setting up NLD model: %s", self.pretrained_path)
        self.model_executor = NLDModelExecutor(
            pretrained_path=self.pretrained_path,
            device=self._torch_device(),
            target_dtype=self._target_dtype(),
        )
        self.tokenizer = self.model_executor.get_tokenizer()
        # Used by BaseCalibrationPipeline.collect_calibration_data to verify setup_model ran.
        self.model = self.model_executor
        logger.info("NLD model loaded on %s", self.model_executor.device)

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
                "model_family": "nld",
                "pretrained_path": self.pretrained_path,
                "threshold": self.threshold,
                "max_length": self.max_length,
            }
        )
        return result

    @torch.no_grad()
    def _collect_sample_calibration_data(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[List[Tuple[int, Tuple[int, int]]], List[Tuple[int, List[Tuple[int, int]]]]]:
        """Trace the vanilla NLD decoding loop, recording commit patterns."""
        calibration_data: List[Tuple[int, Tuple[int, int]]] = []
        sequence_history: List[Tuple[int, List[Tuple[int, int]]]] = []

        executor = self.model_executor
        mask_id = executor.get_mask_id()
        eos_id = executor.get_eos_token_id()
        device = executor.device

        prompt_ids = input_ids
        prompt_len = prompt_ids.shape[1]
        B = prompt_ids.shape[0]
        num_blocks = max(1, self.gen_length // self.block_length)
        steps_per_block = self.block_length

        x_accum = prompt_ids.clone()

        executor.set_diffusion_lm(False)
        output = executor.forward(prompt_ids, use_cache=True, use_causal_mask=True)
        past_key_values = output.past_key_values
        executor.set_diffusion_lm(True)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for num_block in range(num_blocks):
            mask_block = torch.full(
                (B, self.block_length), mask_id, dtype=prompt_ids.dtype, device=device,
            )
            mask_block[:, 0] = next_token[:, 0]
            x_accum = torch.cat([x_accum, mask_block], dim=1)
            block_start = prompt_len + num_block * self.block_length
            block_slice = slice(block_start, block_start + self.block_length)

            mask_block_idx0 = x_accum[:, block_slice] == mask_id
            num_transfer_tokens = _get_num_transfer_tokens(mask_block_idx0, steps_per_block)

            hist: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for i in range(steps_per_block):
                mask_block_idx = x_accum[:, block_slice] == mask_id
                if mask_block_idx.sum() == 0:
                    break

                logits_block = executor.forward(
                    x_accum[:, block_slice],
                    past_key_values=past_key_values,
                    use_cache=False,
                ).logits

                x0, transfer_idx = _get_transfer_index(
                    logits_block, self.temperature, mask_block_idx, x_accum[:, block_slice],
                    num_transfer_tokens=num_transfer_tokens[:, i], threshold=self.threshold,
                )

                # Record the commit: which positions/tokens, plus this step's
                # confidence (argmax prob at masked positions) and full probs.
                probs = torch.softmax(logits_block[0].float(), dim=-1)
                confidence = probs.max(dim=-1).values
                confidence = torch.where(mask_block_idx[0], confidence, torch.full_like(confidence, float("-inf")))
                newly_positions = transfer_idx[0].nonzero(as_tuple=True)[0]
                newly_tokens = x0[0, newly_positions]
                hist.append(
                    (
                        newly_tokens.detach().cpu(),
                        newly_positions.detach().cpu(),
                        confidence.detach().cpu(),
                        probs.detach().cpu(),
                    )
                )

                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

                if eos_id is not None:
                    block_tokens = x_accum[:, block_slice]
                    eos_mask = block_tokens == eos_id
                    if eos_mask.any(dim=1).any():
                        after_eos = eos_mask.cumsum(dim=1).bool()
                        mask_before = (block_tokens == mask_id) & ~after_eos
                        if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                            break

            self._collect_history_tuples(hist, calibration_data, sequence_history)

            executor.set_diffusion_lm(False)
            output = executor.forward(
                x_accum[:, block_slice],
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=True,
            )
            past_key_values = output.past_key_values
            executor.set_diffusion_lm(True)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if eos_id is not None and (x_accum[:, prompt_len:] == eos_id).any(dim=1).all():
                break

        return calibration_data, sequence_history

    def _collect_history_tuples(
        self,
        hist: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        calibration_data: List[Tuple[int, Tuple[int, int]]],
        sequence_history: List[Tuple[int, List[Tuple[int, int]]]],
    ) -> None:
        """Same rank bookkeeping as the Fast-dLLM v2 pipeline, block-wide."""
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

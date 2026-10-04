"""Vanilla and Spiffy decoding for Nemotron-Labs-Diffusion (diffusion mode).

Port of the checkpoint's ``generate()`` (per-block confidence-threshold unmasking,
prefix KV cache refreshed by a causal pass per block) to the generation contract
``generate_with_draft_speculation(prompt) -> (texts, tokens, model_calls, drafts)``.
``calibration_graph=None`` gives vanilla decoding; with a draft graph each step
verifies the main block plus B draft states in one forward and reuses cached draft
logits. ``_batched_block_forward`` is the seam the SpecFold subclasses override.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from specfold.generate.nld_model_executor import NLDModelExecutor
from specfold.model_cache.nemotron_labs_diffusion.modeling_nemotron_labs_diffusion import (
    _get_num_transfer_tokens,
    _get_transfer_index,
)

logger = logging.getLogger(__name__)


class NLDDiffusionGeneration:
    """Nemotron-Labs-Diffusion block-diffusion decoding (paper Sec. 3.2)."""

    def __init__(self,
                 block_length: int,
                 model_executor: NLDModelExecutor,
                 calibration_graph=None,
                 temperature: float = 0.0,
                 max_length: int = 4096,
                 max_new_tokens: int = 1024,
                 threshold: Optional[float] = 0.9,
                 causal_context: bool = True,
                 device: str = "cuda",
                 **kwargs):
        """
        Args:
            block_length: Denoising block size (checkpoint trained with 32)
            model_executor: NLDModelExecutor instance
            calibration_graph: Reserved for draft speculation (unused in vanilla)
            temperature: Sampling temperature (0 = greedy)
            max_length: Maximum total sequence length (prompt + generation)
            max_new_tokens: Maximum number of new tokens to generate
            threshold: Confidence threshold for parallel unmasking (None =
                pure top-k schedule, one token per step)
            causal_context: Refresh the prefix KV cache with a causal forward
                between blocks (the checkpoint's default decoding scheme)
            device: Device to run on
        """
        self.block_length = block_length
        self.model_executor = model_executor
        self.calibration_graph = calibration_graph
        self.temperature = temperature
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.threshold = threshold
        self.causal_context = causal_context
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.tokenizer = model_executor.get_tokenizer()
        self.mask_id = model_executor.get_mask_id()
        self.stop_token = model_executor.get_eos_token_id()

    def _graph_nodes(self) -> List[Dict]:
        if self.calibration_graph is None:
            return []
        return [
            node for node in self.calibration_graph.get("selected_nodes", [])
            if node.get("ij_set")
        ]

    def _batched_block_forward(
        self,
        batched_input: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        """Heavy B+1 batched block forward -> logits [B+1, block_length, vocab].

        The single seam speculative-decoding subclasses override. The base
        class runs a plain dense batched forward: the (batch-1) prefix KV is
        expanded (stride-0, no copy) across the B+1 rows; NLD's diffusion
        attention with use_cache=False only reads (cats) the cache. No logit
        shift: the diffusion head predicts the token at its own position.
        """
        batch_size = batched_input.shape[0]
        if past_key_values is not None and batch_size > 1:
            batched_past_kv = DynamicCache()
            for layer_idx, (k, v) in enumerate(past_key_values):
                batched_past_kv.update(
                    k.expand(batch_size, -1, -1, -1),
                    v.expand(batch_size, -1, -1, -1),
                    layer_idx,
                )
        else:
            batched_past_kv = past_key_values
        return self.model_executor.forward(
            batched_input,
            past_key_values=batched_past_kv,
            use_cache=False,
        ).logits

    @torch.no_grad()
    def get_draft_blocks(
        self,
        x_block: torch.Tensor,
        logits_block: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> List[Dict]:
        """Build draft block states from the calibration graph.

        Same ij semantics as the Fast-dLLM v2 implementation, over the whole
        block (NLD denoises the full block per step, no sub-blocks): for each
        graph node, clone the block and fill, for every (pos_rank, vocab_rank)
        in its ij_set, the pos_rank-th most-confident masked position with its
        vocab_rank-th most-likely token.

        Args:
            x_block: current block state [block_length]
            logits_block: this step's logits over the block [block_length, vocab]
            mask_index: bool mask of still-masked positions [block_length]
        """
        nodes = self._graph_nodes()
        if not nodes or mask_index.sum() == 0:
            return []

        probs = torch.softmax(logits_block.float(), dim=-1)
        confidence = probs.max(dim=-1).values
        confidence = torch.where(~mask_index, float("-inf"), confidence)
        num_mask_positions = int(mask_index.sum().item())

        draft_blocks: List[Dict] = []
        for node in nodes:
            ij_set = node["ij_set"]
            max_pos_rank = max(pos_rank for pos_rank, _ in ij_set)
            # Not enough masked positions to materialize every (i, j): skip.
            if max_pos_rank >= num_mask_positions:
                continue

            _, top_positions = torch.topk(confidence, k=max_pos_rank + 1)
            draft_block = x_block.clone()
            for (pos_rank, vocab_rank) in ij_set:
                pos_index = top_positions[pos_rank]
                _, top_vocabs = torch.topk(logits_block[pos_index], k=vocab_rank + 1)
                draft_block[pos_index] = top_vocabs[vocab_rank]
            draft_blocks.append({"state": draft_block, "node": node})
        return draft_blocks

    @torch.no_grad()
    def generate_with_draft_speculation(self,
                                        prompt: str = None,
                                        **kwargs) -> Tuple[List[str], int, int, int]:
        """
        Diffusion-mode generation; Spiffy draft speculation when a calibration
        graph with nodes is set, vanilla decoding otherwise.

        Returns:
            Tuple of (generated_text_list, tokens_generated, model_calls,
            drafts_computed)
        """
        start_time = time.time()
        executor = self.model_executor
        mask_id = self.mask_id
        eos_id = self.stop_token

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        B = prompt_ids.shape[0]

        # Token budget: multiple of block_length, capped by max_length.
        budget = min(self.max_new_tokens, self.max_length - prompt_len)
        num_blocks = budget // self.block_length
        if num_blocks <= 0:
            logger.warning("Prompt length %d leaves no room to generate (max_length=%d)",
                           prompt_len, self.max_length)
            return [""], 0, 0, 0

        model_calls = 0
        drafts_computed = 0
        x_accum = prompt_ids.clone()

        # Initial prefill: causal forward builds the prefix KV cache and (in
        # causal_context mode) the AR seed token for the first block.
        if self.causal_context:
            executor.set_diffusion_lm(False)
        output = executor.forward(prompt_ids, use_cache=True, use_causal_mask=self.causal_context)
        past_key_values = output.past_key_values
        model_calls += 1
        if self.causal_context:
            executor.set_diffusion_lm(True)

        next_token = None
        if self.causal_context:
            last_logit = output.logits[:, -1, :]
            next_token = self._sample_token(last_logit)

        steps_per_block = self.block_length

        for num_block in range(num_blocks):
            mask_block = torch.full(
                (B, self.block_length), mask_id, dtype=prompt_ids.dtype, device=self.device,
            )
            if self.causal_context:
                mask_block[:, 0] = next_token[:, 0]

            x_accum = torch.cat([x_accum, mask_block], dim=1)
            block_start = prompt_len + num_block * self.block_length
            block_slice = slice(block_start, block_start + self.block_length)

            mask_block_idx0 = x_accum[:, block_slice] == mask_id
            num_transfer_tokens = _get_num_transfer_tokens(mask_block_idx0, steps_per_block)

            # Draft speculation state is block-local: draft states are block-
            # relative and the prefix KV changes at every block commit.
            draft_blocks: List[Dict] = []
            called_drafts: List[Tuple[torch.Tensor, torch.Tensor]] = []

            # Denoise the current block by repeated confidence-based unmasking.
            for i in range(steps_per_block):
                mask_block_idx = x_accum[:, block_slice] == mask_id
                if mask_block_idx.sum() == 0:
                    break

                # Reuse cached draft logits when the main state equals a draft.
                hit_idx = -1
                if called_drafts:
                    cur_block = x_accum[0, block_slice]
                    eq = torch.stack(
                        [s for s, _ in called_drafts]
                    ).eq(cur_block).all(dim=1).tolist()
                    hit_idx = eq.index(True) if True in eq else -1

                if hit_idx >= 0:
                    _, logits_block = called_drafts.pop(hit_idx)
                else:
                    main_block = x_accum[:, block_slice]
                    if draft_blocks:
                        draft_states = torch.stack([db["state"] for db in draft_blocks], dim=0)
                        batched_input = torch.cat([main_block, draft_states], dim=0)
                    else:
                        batched_input = main_block

                    logits_all = self._batched_block_forward(batched_input, past_key_values)
                    model_calls += 1

                    logits_block = logits_all[0:1]
                    for k, db in enumerate(draft_blocks):
                        called_drafts.append((db["state"], logits_all[k + 1:k + 2]))

                    # New draft states for the next step, from this step's logits.
                    if self._graph_nodes():
                        draft_blocks = self.get_draft_blocks(
                            x_accum[0, block_slice], logits_block[0], mask_block_idx[0],
                        )
                        drafts_computed += len(draft_blocks)

                x0, transfer_idx = _get_transfer_index(
                    logits_block, self.temperature, mask_block_idx, x_accum[:, block_slice],
                    num_transfer_tokens=num_transfer_tokens[:, i], threshold=self.threshold,
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

                # Early exit once an EOS is fixed and nothing before it is masked.
                if eos_id is not None:
                    block_tokens = x_accum[:, block_slice]
                    eos_mask = block_tokens == eos_id
                    if eos_mask.any(dim=1).any():
                        after_eos = eos_mask.cumsum(dim=1).bool()
                        mask_before = (block_tokens == mask_id) & ~after_eos
                        if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                            break

            # Post-block: causal forward over the block refreshes the KV cache
            # and (when causal_context) samples the seed for the next block.
            if self.causal_context:
                executor.set_diffusion_lm(False)
            output = executor.forward(
                x_accum[:, block_slice],
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=self.causal_context,
            )
            past_key_values = output.past_key_values
            model_calls += 1

            if self.causal_context:
                executor.set_diffusion_lm(True)
                last_logit = output.logits[:, -1, :]
                next_token = self._sample_token(last_logit)

            if eos_id is not None:
                gen_so_far = x_accum[:, prompt_len:]
                if (gen_so_far == eos_id).any(dim=1).all():
                    break

        # Truncate at stop token, drop residual mask tokens, decode.
        final_ids = x_accum[0, prompt_len:]
        if eos_id is not None and (final_ids == eos_id).any():
            stop_positions = (final_ids == eos_id).nonzero()
            final_ids = final_ids[:stop_positions[0].item() + 1]
        final_ids_no_mask = final_ids[final_ids != mask_id]

        final_text = self.tokenizer.batch_decode(
            final_ids_no_mask.unsqueeze(0), skip_special_tokens=True
        )

        total_time = time.time() - start_time
        tokens_generated = len(final_ids_no_mask)
        logger.info("Generated %d tokens in %.2fs (%.2f tok/s)",
                    tokens_generated, total_time,
                    tokens_generated / total_time if total_time > 0 else 0.0)
        logger.info("Model calls: %d, Drafts computed: %d", model_calls, drafts_computed)

        return final_text, tokens_generated, model_calls, drafts_computed

    def _sample_token(self, last_logit: torch.Tensor) -> torch.Tensor:
        if self.temperature > 0:
            probs = torch.softmax(last_logit / self.temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(last_logit, dim=-1, keepdim=True)

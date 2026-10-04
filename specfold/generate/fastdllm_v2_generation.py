"""
FastDLLM v2 Speculative Generation implementation.
Follows the spiffy module structure, adapted from Fast-dLLM/v2/generation_functions.py
"""

import logging
import os
import torch
import time
from typing import List, Optional, Tuple, Dict
from transformers.cache_utils import DynamicCache

from specfold.config import Config
from specfold.generate.base_generation import BaseSpeculativeGeneration
from specfold.generate.fastdllm_model_executor import FastDLLMModelExecutor

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastDLLMV2SpeculativeGeneration(BaseSpeculativeGeneration):
    """
    FastDLLM v2 generation with speculative decoding support.
    Implements block-based masked diffusion model (MDM) generation.
    """
    
    # Constants for FastDLLM v2 model
    MASK_ID = 151665
    STOP_TOKEN = 151645
    
    def __init__(self,
                 block_length: int,
                 model_executor: FastDLLMModelExecutor,
                 calibration_graph=None,
                 temperature: float = 0.0,
                 top_p: float = 0.95,
                 top_k: Optional[int] = None,
                 mask_id: int = 151665,
                 stop_token: int = 151645,
                 max_length: int = 4096,
                 max_new_tokens: int = 1024,
                 small_block_size: int = 8,
                 threshold: float = 0.9,
                 use_block_cache: bool = False,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 prompt_render_mode: str = "preformatted_text",
                 **kwargs
        ):
        """
        Initialize FastDLLM v2 Speculative Generation.
        
        Args:
            block_length: Length of each generation block (bd_size)
            model_executor: FastDLLMModelExecutor instance
            calibration_graph: Optional calibration graph for draft speculation
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            mask_id: ID of the mask token (151665 for FastDLLM v2)
            stop_token: ID of the stop token (151645 for FastDLLM v2)
            max_length: Maximum sequence length
            max_new_tokens: Maximum number of new tokens to generate
            small_block_size: Size of small blocks within each block
            threshold: Confidence threshold for token selection
            use_block_cache: Whether to use block-level KV caching
            device: Device to run on
            dtype: Data type for computation
        """
        self.device = torch.device(kwargs.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.__dict__.update(kwargs)
        self.dtype = dtype
        if self.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.target_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.target_dtype = torch.float16
        else:
            self.target_dtype = torch.float32
            
        self.block_length = block_length
        self.calibration_graph = calibration_graph
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.mask_id = mask_id
        self.stop_token = stop_token
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.small_block_size = small_block_size
        self.threshold = threshold
        self.use_block_cache = use_block_cache
        self.prompt_render_mode = prompt_render_mode
        self.model_executor = model_executor
        self.tokenizer = self.model_executor.get_tokenizer()

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to prompt."""
        chat_history = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

    def _prepare_prompt_text(self, prompt: str) -> str:
        """Return the final string to tokenize, respecting prompt_render_mode.

        - "preformatted_text" (default): prompt is already chat-templated (e.g.
          by lm-eval); pass straight through, matching Fast-dLLM v2 eval.py.
        - "legacy_single_user_chat": wrap prompt as a single user turn via the
          tokenizer's chat template (for demo / non-lm-eval callers).
        """
        if self.prompt_render_mode == "preformatted_text":
            if not isinstance(prompt, str):
                raise TypeError(f"Expected preformatted prompt text, got {type(prompt)}")
            return prompt
        if self.prompt_render_mode == "legacy_single_user_chat":
            return self._apply_chat_template(prompt)
        raise ValueError(f"Unsupported prompt_render_mode: {self.prompt_render_mode}")
    
    def _shift_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Shift logits for next-token prediction (FastDLLM v2 convention).
        Moves logits so that position i predicts token at position i.
        """
        return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

    @staticmethod
    def _kv_seq_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        return int(past_key_values.get_seq_length())
    
    def _sample_with_top_p(self, 
                           logits: torch.Tensor, 
                           top_p: float = 0.95, 
                           temperature: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tokens with top-p (nucleus) sampling.
        
        Args:
            logits: Logits tensor [batch, seq_len, vocab_size]
            top_p: Top-p sampling threshold
            temperature: Sampling temperature
            
        Returns:
            Tuple of (sampled_tokens, probabilities)
        """
        return self.model_executor.sample_with_top_p(logits, top_p=top_p, temperature=temperature)

    def _batched_block_forward(
        self,
        batched_input: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        """Heavy B+1 batched block forward -> shifted logits [B+1, block_length, vocab].

        This is the single seam that speculative-decoding subclasses override.
        The base class runs a plain *dense* batched forward: it duplicates the
        (batch-1) prefix KV cache across the B+1 rows and calls the stock model.

        Args:
            batched_input: [B+1, block_length] token ids. Row 0 is the main
                block; rows 1..B are draft block states (may be empty -> [1, N]).
            past_key_values: the running batch-1 prefix cache (or None).
        """
        batch_size = batched_input.shape[0]
        if past_key_values is not None and batch_size > 1:
            # expand = stride-0 view, no copy; the model only reads (cats) it.
            batched_past_kv = DynamicCache()
            for layer_idx, (k, v) in enumerate(past_key_values):
                batched_past_kv.update(
                    k.expand(batch_size, -1, -1, -1),
                    v.expand(batch_size, -1, -1, -1),
                    layer_idx,
                )
        else:
            batched_past_kv = past_key_values

        outputs = self.model_executor.forward(
            batched_input,
            past_key_values=batched_past_kv,
            use_cache=True,
            update_past_key_values=False,
        )
        return self._shift_logits(outputs.logits)

    @torch.no_grad()
    def generate(self,
                 prompt: str = None,
                 **kwargs) -> Tuple[List[str], int, List[Dict]]:
        """
        Standard generation without speculation (MDM sampling).
        Implements the core FastDLLM v2 generation algorithm.

        Args:
            prompt: Input prompt string
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_text_list, model_calls)
        """
        model_calls = 0
        start_time = time.time()

        # Initialization
        input_text = self._prepare_prompt_text(prompt)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        original_input_length = input_ids.shape[1]

        num_blocks = self.max_new_tokens // self.block_length
        num_small_blocks = self.block_length // self.small_block_size
        past_key_values = None

        # Initialize with prefill if input length > block_length
        if input_ids.shape[1] > self.block_length:
            prefill_length = (input_ids.shape[1] // self.block_length) * self.block_length
            outputs = self.model_executor.forward(
                input_ids[:, :prefill_length],
                past_key_values=None,
                use_cache=True,
                update_past_key_values=True
            )
            logits, past_key_values = outputs.logits, outputs.past_key_values
            model_calls += 1
            
            # If prefill exactly matches input length, generate first token
            if input_ids.shape[1] % self.block_length == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Main block generation loop
        for block_idx in range(num_blocks):
            # Check for stop token
            if self.stop_token in input_ids[:, original_input_length:]:
                break
            
            prompt_length = input_ids.shape[1]
            
            # Initialize current block with mask tokens
            block_init_length = self.block_length - (prompt_length % self.block_length)
            x_init = torch.full(
                (input_ids.shape[0], block_init_length), 
                self.mask_id, 
                device=self.device, 
                dtype=torch.long
            )
            x_init = torch.cat([input_ids, x_init], dim=1)
            x_t = x_init.clone()
            
            block_past_key_values = None
            
            # Process current block until all masks are filled
            while True:
                # Check for stop token in generated part
                if self.stop_token in x_t[:, prompt_length:]:
                    stop_token_positions = (x_t[:, prompt_length:] == self.stop_token).nonzero()
                    if stop_token_positions.numel() > 0:
                        stop_idx = stop_token_positions[0][1]
                        if (x_t[:, prompt_length:prompt_length + stop_idx] == self.mask_id).sum() == 0:
                            break
                
                mask_idx = (x_t[:, -self.block_length:] == self.mask_id)
                
                # If no masks remain, update cache and move to next block
                if mask_idx.sum() == 0:
                    kv_len_before = self._kv_seq_length(past_key_values)
                    outputs = self.model_executor.forward(
                        x_t[:, -self.block_length:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        update_past_key_values=True
                    )
                    logits, past_key_values = outputs.logits, outputs.past_key_values
                    model_calls += 1

                    # Generate next token for next block
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break
                
                # Process small blocks within the current block
                for small_block_idx in range(num_small_blocks):
                    small_block_start = small_block_idx * self.small_block_size
                    small_block_end = small_block_start + self.small_block_size
                    
                    start = -self.block_length + small_block_start
                    end = None if self.block_length == small_block_end else -self.block_length + small_block_end
                    
                    # Decode small block until no masks remain
                    while True:
                        mask_idx = (x_t[:, -self.block_length:] == self.mask_id)
                        small_mask_idx = mask_idx[:, small_block_start:small_block_end]
                        
                        if small_mask_idx.sum() == 0:
                            break
                        
                        # Check for stop token
                        if self.stop_token in x_t[:, prompt_length:]:
                            stop_token_positions = (x_t[:, prompt_length:] == self.stop_token).nonzero()
                            if stop_token_positions.numel() > 0:
                                stop_idx = stop_token_positions[0][1]
                                if (x_t[:, prompt_length:prompt_length + stop_idx] == self.mask_id).sum() == 0:
                                    break
                        
                        # Model forward pass
                        kv_len_before = self._kv_seq_length(past_key_values)
                        if self.use_block_cache:
                            if block_past_key_values is None or (x_t[:, -self.block_length + small_block_start] == self.mask_id).any():
                                outputs = self.model_executor.forward(
                                    x_t[:, -self.block_length:],
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                    update_past_key_values=False,
                                    use_block_cache=True
                                )
                                logits = outputs.logits
                                block_past_key_values = outputs.block_past_key_values if hasattr(outputs, 'block_past_key_values') else None
                                logits = self._shift_logits(logits)
                                logits = logits[:, start:end] if end else logits[:, start:]
                            else:
                                slice_len = int(self.small_block_size if end is not None else self.block_length - small_block_start)
                                outputs = self.model_executor.forward(
                                    x_t[:, start:end] if end else x_t[:, start:],
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start
                                )
                                logits = self._shift_logits(outputs.logits)
                        else:
                            outputs = self.model_executor.forward(
                                x_t[:, -self.block_length:],
                                past_key_values=past_key_values,
                                use_cache=True,
                                update_past_key_values=False
                            )
                            logits = self._shift_logits(outputs.logits)
                            logits = logits[:, start:end] if end else logits[:, start:]

                        model_calls += 1
                        
                        # Sample tokens with top-p
                        x_1, p_1t = self._sample_with_top_p(logits, top_p=self.top_p, temperature=self.temperature)
                        
                        # Get confidence for sampled tokens
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), 
                            -1
                        )
                        
                        # Only consider masked positions
                        x1_p = torch.where(small_mask_idx, x1_p, -torch.inf)
                        
                        # Select tokens above threshold + always select max prob token
                        unmask_idx = (x1_p > self.threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                        unmask_idx = unmask_idx & small_mask_idx
                        
                        # Update tokens
                        if end is not None:
                            x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                        else:
                            x_t[:, start:][unmask_idx] = x_1[unmask_idx]
            
            input_ids = x_t
        
        # Truncate at stop token
        final_ids = input_ids[0, original_input_length:]
        if self.stop_token in final_ids:
            stop_positions = (final_ids == self.stop_token).nonzero()
            if stop_positions.numel() > 0:
                final_ids = final_ids[:stop_positions[0].item() + 1]
        
        # Decode output
        final_text = self.tokenizer.batch_decode(final_ids.unsqueeze(0), skip_special_tokens=True)
        
        total_time = time.time() - start_time
        tokens_generated = len(final_ids[final_ids != self.mask_id])
        logger.info(f"Generated {tokens_generated} tokens in {total_time:.2f}s ({tokens_generated/total_time:.2f} tok/s)")

        return final_text, model_calls

    @torch.no_grad()
    def get_draft_blocks(self,
                        x: torch.Tensor,
                        p: torch.Tensor,
                        mask_index: Optional[torch.Tensor] = None,
                        start_pos: Optional[int] = None,
                        end_pos: Optional[int] = None,
                        small_block_start: Optional[int] = None,
                        small_block_end: Optional[int] = None,
                        **kwargs) -> List[Dict]:
        """
        Obtain draft blocks from the calibration graph.
        Drafts predict different fill patterns for the small block.
        
        Args:
            x: Current token sequence in block [block_length]
            p: Logits/probabilities for the block [block_length, vocab_size]
            mask_index: Boolean tensor indicating mask positions [block_length]
            start_pos: Start position in full sequence
            end_pos: End position in full sequence
            small_block_start: Start index of small block within block
            small_block_end: End index of small block within block
            
        Returns:
            List of draft block dictionaries with different fill patterns
        """
        if self.calibration_graph is None:
            return []
        
        draft_blocks = []
        
        # Focus on small block masks only
        if small_block_start is not None and small_block_end is not None:
            small_mask = mask_index[small_block_start:small_block_end] if mask_index is not None else None
            small_logits = p[small_block_start:small_block_end, :]
        else:
            small_mask = mask_index
            small_logits = p
        
        if small_mask is None or small_mask.sum() == 0:
            return []
        
        for node in self.calibration_graph.get('selected_nodes', []):
            ij_set = node.get('ij_set', [])
            if not ij_set:
                continue
            
            # Clone the full block
            draft_block = x.clone()
            
            # Get confidence scores for the small block
            probs = torch.softmax(small_logits, dim=-1)
            confidence = probs.max(dim=-1).values
            
            # Only consider masked positions
            confidence = torch.where(~small_mask, float('-inf'), confidence)
            
            max_pos_rank = max(pos_rank for pos_rank, _ in ij_set)
            num_mask_positions = small_mask.sum().item()

            # If there aren't enough masked positions to reach max_pos_rank,
            # at least one (i, j) in ij_set can't be materialized; skip the draft.
            if max_pos_rank >= num_mask_positions:
                continue

            _, top_positions = torch.topk(confidence, k=max_pos_rank + 1)

            for (pos_rank, vocab_rank) in ij_set:
                pos_index = top_positions[pos_rank]

                # Get the vocab_rank-th most likely token at this position
                _, top_vocabs = torch.topk(small_logits[pos_index], k=vocab_rank + 1)
                vocab_index = top_vocabs[vocab_rank]
                
                # Set in the full block at correct offset
                if small_block_start is not None:
                    draft_block[small_block_start + pos_index] = vocab_index
                else:
                    draft_block[pos_index] = vocab_index
            
            draft_blocks.append({
                'state': draft_block,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'node': node
            })
        
        return draft_blocks
    
    def _init_generation_state(self, prompt: str) -> Tuple[int, torch.Tensor]:
        """Initialize generation state variables as instance attributes."""
        input_text = self._prepare_prompt_text(prompt)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        return prompt_length, input_ids

    @torch.no_grad()
    def generate_with_draft_speculation(self,
                                        prompt: str = None,
                                        **kwargs) -> Tuple[List[str], int, int, int, List[Dict]]:
        """
        Generate text with draft speculation using calibration graph.
        Batches draft blocks together for parallel model forward passes.

        Args:
            prompt: Input prompt string
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_text, generated_tokens, model_calls, drafts_computed)
        """
        start_time = time.time()
        
        # Initialize generation state
        prompt_length, input_ids = self._init_generation_state(prompt)
        original_input_length = prompt_length
        
        num_blocks = self.max_new_tokens // self.block_length
        num_small_blocks = self.block_length // self.small_block_size
        past_key_values = None
        model_calls = 0
        drafts_computed = 0

        # Initialize with prefill if input length > block_length
        if input_ids.shape[1] > self.block_length:
            prefill_length = (input_ids.shape[1] // self.block_length) * self.block_length
            outputs = self.model_executor.forward(
                input_ids[:, :prefill_length],
                past_key_values=None,
                use_cache=True,
                update_past_key_values=True
            )
            logits, past_key_values = outputs.logits, outputs.past_key_values
            model_calls += 1

            if input_ids.shape[1] % self.block_length == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        x_t = input_ids
        draft_blocks = []  # Current draft blocks to batch
        called_drafts = []  # list of (block_state_tensor, full-block logits)
        
        # Main block generation loop with speculation
        for block_idx in range(num_blocks):
            if self.stop_token in x_t[:, original_input_length:]:
                break
            
            prompt_length = x_t.shape[1]
            
            # Initialize current block with mask tokens
            block_init_length = self.block_length - (prompt_length % self.block_length)
            x_init = torch.full(
                (x_t.shape[0], block_init_length),
                self.mask_id,
                device=self.device,
                dtype=torch.long
            )
            x_t = torch.cat([x_t, x_init], dim=1)
            
            # Process current block until all masks filled
            while True:
                # Check for stop token
                if self.stop_token in x_t[:, prompt_length:]:
                    stop_token_positions = (x_t[:, prompt_length:] == self.stop_token).nonzero()
                    if stop_token_positions.numel() > 0:
                        stop_idx = stop_token_positions[0][1]
                        if (x_t[:, prompt_length:prompt_length + stop_idx] == self.mask_id).sum() == 0:
                            break
                
                mask_idx = (x_t[:, -self.block_length:] == self.mask_id)
                
                # If no masks remain, update cache and move to next block
                if mask_idx.sum() == 0:
                    kv_len_before = self._kv_seq_length(past_key_values)
                    outputs = self.model_executor.forward(
                        x_t[:, -self.block_length:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        update_past_key_values=True
                    )
                    logits, past_key_values = outputs.logits, outputs.past_key_values
                    model_calls += 1

                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    draft_blocks = []  # Clear drafts when moving to next block
                    called_drafts = []  # block states are block-relative; stale across blocks
                    break
                
                # Process small blocks
                for small_block_idx in range(num_small_blocks):
                    small_block_start = small_block_idx * self.small_block_size
                    small_block_end = small_block_start + self.small_block_size

                    start = -self.block_length + small_block_start
                    end = None if self.block_length == small_block_end else -self.block_length + small_block_end
                    
                    while True:
                        mask_idx = (x_t[:, -self.block_length:] == self.mask_id)
                        small_mask_idx = mask_idx[:, small_block_start:small_block_end]
                        
                        if small_mask_idx.sum() == 0:
                            break
                        
                        if self.stop_token in x_t[:, prompt_length:]:
                            stop_token_positions = (x_t[:, prompt_length:] == self.stop_token).nonzero()
                            if stop_token_positions.numel() > 0:
                                stop_idx = stop_token_positions[0][1]
                                if (x_t[:, prompt_length:prompt_length + stop_idx] == self.mask_id).sum() == 0:
                                    break
                        
                        # Check if current block state matches a cached draft.
                        # GPU-side comparison: one small D2H sync instead of
                        # one full-tensor .cpu() key per iteration + per draft.
                        hit_idx = -1
                        if called_drafts:
                            cur_block = x_t[0, -self.block_length:]
                            eq = torch.stack(
                                [s for s, _ in called_drafts]
                            ).eq(cur_block).all(dim=1).tolist()
                            hit_idx = eq.index(True) if True in eq else -1

                        if hit_idx >= 0:
                            # Sample lazily, only on an actual cache hit.
                            _, hit_logits = called_drafts.pop(hit_idx)
                            full_x_1, full_p_1t = self._sample_with_top_p(
                                hit_logits, top_p=self.top_p, temperature=self.temperature
                            )
                            x_1 = full_x_1[:, small_block_start:small_block_end]
                            p_1t = full_p_1t[:, small_block_start:small_block_end, :]
                        else:
                            # Prepare batched input: main + drafts
                            main_block = x_t[:, -self.block_length:]
                            
                            kv_len_before = self._kv_seq_length(past_key_values)
                            if len(draft_blocks) > 0:
                                # Batch main block with draft blocks
                                draft_states = torch.stack([db['state'] for db in draft_blocks], dim=0)
                                batched_input = torch.cat([main_block, draft_states], dim=0)
                            else:
                                # Degenerate step: no drafts to speculate -> just the main block.
                                batched_input = main_block

                            # Single overridable seam for the heavy B+1 batched forward.
                            # Base class: dense batched forward. Subclasses (e.g. SpecFold
                            # folding) override this to sparsely compute parent/child drafts.
                            logits = self._batched_block_forward(batched_input, past_key_values)
                            model_calls += 1

                            # Extract main result (batch idx 0)
                            main_logits = logits[0:1, :, :]

                            # Cache draft logits (batch idx 1+) WITHOUT sampling
                            # or CPU key conversion; sampling happens lazily on hit.
                            for i, db in enumerate(draft_blocks):
                                called_drafts.append((db['state'], logits[i+1:i+2, :, :]))

                            # Use main result - slice small block from block logits
                            x_1, p_1t = self._sample_with_top_p(
                                main_logits[:, small_block_start:small_block_end, :],
                                top_p=self.top_p,
                                temperature=self.temperature
                            )
                            full_logits = main_logits
                            
                            # Generate new draft blocks for next iteration
                            if self.calibration_graph is not None:
                                draft_blocks = self.get_draft_blocks(
                                    x_t[0, -self.block_length:],
                                    full_logits[0],
                                    mask_idx[0],
                                    start_pos=x_t.shape[1] - self.block_length,
                                    end_pos=x_t.shape[1],
                                    small_block_start=small_block_start,
                                    small_block_end=small_block_end
                                )
                                drafts_computed += len(draft_blocks)
                        
                        # Get confidence for sampled tokens
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)),
                            -1
                        )
                        x1_p = torch.where(small_mask_idx, x1_p, -torch.inf)

                        unmask_idx = (x1_p > self.threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                        unmask_idx = unmask_idx & small_mask_idx

                        # Commit-step debug trace (env SPIFFY_COMMIT_DEBUG=1):
                        # per commit iteration, log source (hit/forward), the
                        # sub-block's max confidence, forced-commit status, and
                        # each committed position's token/top1/top1-top2 margin.
                        if os.environ.get("SPIFFY_COMMIT_DEBUG"):
                            t2 = torch.topk(p_1t[0], 2, dim=-1).values
                            comm = unmask_idx[0].nonzero(as_tuple=True)[0].tolist()
                            forced = not bool((x1_p > self.threshold).any())
                            parts = [
                                f"p{small_block_start + c}=t{int(x_1[0, c])}"
                                f"@{float(t2[c, 0]):.3f}/m{float(t2[c, 0] - t2[c, 1]):.3f}"
                                for c in comm
                            ]
                            logger.info(
                                "[commitdbg] hit=%d forced=%d cmax=%.3f commits=%s",
                                int(hit_idx >= 0), int(forced), float(x1_p.max()),
                                " ".join(parts),
                            )

                        if end is not None:
                            x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                        else:
                            x_t[:, start:][unmask_idx] = x_1[unmask_idx]
        
        # Truncate at stop token
        final_ids = x_t[0, original_input_length:]
        if self.stop_token in final_ids:
            stop_positions = (final_ids == self.stop_token).nonzero()
            if stop_positions.numel() > 0:
                final_ids = final_ids[:stop_positions[0].item() + 1]
        
        # Remove mask tokens from count
        final_ids_no_mask = final_ids[final_ids != self.mask_id]
        
        # Decode output
        final_text = self.tokenizer.batch_decode(final_ids_no_mask.unsqueeze(0), skip_special_tokens=True)
        
        total_time = time.time() - start_time
        tokens_generated = len(final_ids_no_mask)
        logger.info(f"Generated {tokens_generated} tokens in {total_time:.2f}s ({tokens_generated/total_time:.2f} tok/s)")
        logger.info(f"Model calls: {model_calls}, Drafts computed: {drafts_computed}")

        return final_text, tokens_generated, model_calls, drafts_computed

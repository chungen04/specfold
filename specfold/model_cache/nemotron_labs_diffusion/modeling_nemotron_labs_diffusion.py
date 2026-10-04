import copy
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutput
from transformers.utils import ModelOutput

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.processing_utils import Unpack

from transformers.cache_utils import Cache, DynamicCache

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.generation import GenerationMixin

import math

from .modeling_ministral import Ministral3Model, Ministral3PreTrainedModel, Ministral3Attention, apply_rotary_pos_emb, repeat_kv, _get_llama_4_attn_scale
from .configuration_nemotron_labs_diffusion import NemotronLabsDiffusionConfig

__all__ = ["NemotronLabsDiffusionModel", "NemotronLabsDiffusionFlexAttention"]

@dataclass
class NemotronLabsDiffusionOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    causal_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
def fused_flex_attention(q, k, v, block_mask=None):
    return flex_attention(q, k, v, block_mask=block_mask)


class NemotronLabsDiffusionFlexAttention(Ministral3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block_size = self.config.block_size
        self.block_diff_mask = None

        import torch._dynamo.config as dcfg
        dcfg.cache_size_limit = 512

    def compute_block_mask(self, mode, q_len, block_size=None):

        def block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
            x0_flag_q = (q_idx >= n)
            x0_flag_kv = (kv_idx >= n)

            # Compute block indices
            block_q = torch.where(x0_flag_q == 1,
                                    (q_idx - n) // block_size,
                                    q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1,
                                    (kv_idx - n) // block_size,
                                    kv_idx // block_size)

            # **1. Block Diagonal Mask (M_BD) **
            block_diagonal = (block_q == block_kv) & (x0_flag_kv == 0) & (x0_flag_q == 0)

            # **2. Offset Block-Causal Mask (M_OBC) **
            offset_block_causal = (
                (block_q > block_kv)
                & (x0_flag_kv == 1)
                & (x0_flag_q == 0)
            )

            # **3. Fully Causal Mask (M_BC) **
            fully_causal = (q_idx >= kv_idx) & (x0_flag_kv == 1) & (x0_flag_q == 1)

            # **4. Combine Masks **
            return block_diagonal | offset_block_causal | fully_causal

        attn_mask = lambda b, h, q, kv: block_diff_mask(block_size, b, h, q, kv, q_len//2)

        block_mask = create_block_mask(
            attn_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len
        )

        return block_mask


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_training: bool = True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        if is_training:
            # Split query and key states in half along sequence length dimension
            q1, q2 = query_states.chunk(2, dim=2)
            k1, k2 = key_states.chunk(2, dim=2)
            
            # Apply RoPE independently to each half
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
            
            # Recombine the halves
            query_states = torch.cat([q1, q2], dim=2)
            key_states = torch.cat([k1, k2], dim=2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states * _get_llama_4_attn_scale(
            cache_position,
            self.config.rope_parameters.get("llama_4_scaling_beta"),
            self.config.rope_parameters.get("original_max_position_embeddings"),
        ).to(query_states.dtype)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.block_diff_mask is None or q_len != self.block_diff_mask.shape[-2]:
            block_mask = self.compute_block_mask(mode='block_diff', block_size=self.block_size, q_len=q_len)
        else:
            block_mask = self.block_diff_mask

        attn_output = fused_flex_attention(query_states, key_states, value_states, block_mask=block_mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output, None


class NemotronLabsDiffusionModel(Ministral3PreTrainedModel, GenerationMixin):
    """
    A single model with:
      - a bidirectional encoder + diffusion‐LM head over A
      - a causal decoder + LM head over B, conditioned on F_A
    """

    def __init__(self, config: NemotronLabsDiffusionConfig):
        super().__init__(config)

        self.mask_token_id = config.mask_token_id

        diffusion_config = copy.deepcopy(config)
        diffusion_config.diffusion_lm = True

        if config.dlm_paradigm == 'block_diff':
            diffusion_config.attn_class = NemotronLabsDiffusionFlexAttention
        elif config.dlm_paradigm in ['bidirectional', 'autoregressive']:
            diffusion_config.attn_class = Ministral3Attention
            if config.dlm_paradigm == 'autoregressive':
                diffusion_config.diffusion_lm = False
        else:
            raise ValueError(f"Unsupported DLM paradigm: {config.dlm_paradigm}")
        
        self.encoder = Ministral3Model(diffusion_config)
        self.diffusion_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        self.post_init()


    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.diffusion_head

    def set_output_embeddings(self, new_embeddings):
        self.diffusion_head = new_embeddings


    def forward_process(self, input_ids, eps=1e-3, block_size=None, loss_mask=None):
        b, l = input_ids.shape
        device = input_ids.device

        if self.config.dp_varying_mask_ratio:
            # Enable different random seeds for each DP rank during sampling
            import torch.distributed as dist
            dp_rank = 0
            if dist.is_initialized():
                try:
                    dp_rank = dist.get_rank()
                except Exception:
                    dp_rank = 0
            # Use a local generator to avoid affecting global RNG state
            generator = torch.Generator(device=device)
            generator.manual_seed(torch.seed() + dp_rank)
        else:
            generator = None
            
        t = torch.rand(b, device=device, generator=generator)
        
        p_mask = (1 - eps) * t + eps  # shape: (b,)
        p_mask = p_mask[:, None].expand(-1, l)  # shape: (b, l)

        masked_indices = torch.rand((b, l), device=device) < p_mask

        if loss_mask is not None:
            masked_indices[loss_mask == 0] = 0

        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)        

        return noisy_batch, masked_indices, p_mask
        

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor]   = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor]       = None,
        split_len: Optional[int]                 = None,
        past_key_values: Optional[Cache]         = None,
        block_size: Optional[int]                = None,
        eps: float                               = 1e-3,
        is_teacher: bool                        = False,
        masked_indices: Optional[torch.Tensor]   = None,
        p_mask: Optional[torch.Tensor]           = None,
        teacher_logits: Optional[torch.Tensor]   = None,
        masked_indices_teacher: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        ce_loss_weight: float = 1.0,
        output_last_hidden_states_only: bool = False,
        skip_loss: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        batch_size, seq_len = input_ids.shape

        if self.config.dlm_paradigm == 'block_diff':
            if labels is not None and block_size is None:
                block_size = self.config.block_size
        elif self.config.dlm_paradigm not in ('bidirectional', 'autoregressive'):
            raise ValueError(f"Unknown dLM paradigm: {self.config.dlm_paradigm}")

        if labels is not None and self.config.dlm_paradigm != 'autoregressive':
            if masked_indices is not None:
                # assert p_mask is not None

                if loss_mask is not None:
                    masked_indices[loss_mask == 0] = 0

                noisy_inputs = torch.where(masked_indices, self.mask_token_id, input_ids)

            else:
                noisy_inputs, masked_indices, p_mask = self.forward_process(input_ids, eps=eps, block_size=block_size, loss_mask=loss_mask)

        else:
            noisy_inputs = input_ids
            masked_indices = None
            p_mask = None

        input_ids_len = noisy_inputs.shape[1]
        if labels is not None and self.config.dlm_paradigm == 'block_diff':
            if position_ids is None:
                position_ids = torch.arange(input_ids_len, device=noisy_inputs.device).unsqueeze(0)
            noisy_inputs = torch.cat([noisy_inputs, input_ids], dim=1)

        enc_out  = self.encoder(
            past_key_values=past_key_values,
            input_ids=noisy_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_training=(labels is not None),
            **kwargs,
        )

        if output_last_hidden_states_only:
            return BaseModelOutput(last_hidden_state=enc_out.last_hidden_state)

        logits = self.diffusion_head(enc_out.last_hidden_state)  # (batch, len_B, vocab)
        causal_logits = None

        if labels is not None and self.config.dlm_paradigm == 'block_diff':
            causal_logits = logits[:, input_ids_len:]
            logits = logits[:, :input_ids_len]

        loss = None
        if labels is not None and not skip_loss:
            if self.config.dlm_paradigm == 'autoregressive':
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                if loss_mask is None:
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    loss = loss_fct(shift_logits, shift_labels)

                else:
                    loss_mask = loss_mask[..., 1:].contiguous()

                    loss_fct = CrossEntropyLoss(reduction='none')
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    
                    token_losses = loss_fct(shift_logits, shift_labels)
                                    
                    flat_loss_mask = loss_mask.reshape(-1)
                    loss = token_losses[flat_loss_mask == 1].sum() / flat_loss_mask.sum()

            else:
                # LLaDA-style diffusion loss on masked positions.
                # Token-wise cross entropy loss on masked positions.
                token_loss = torch.nn.functional.cross_entropy(
                    logits[masked_indices],
                    labels[masked_indices],
                    reduction='none'
                ) / p_mask[masked_indices]

                num_mask_tokens = masked_indices.sum()

                # global_loss_avg=True: loss is reduced externally by global token count.
                loss = token_loss.sum()
                
                if self.config.dlm_loss_weight is not None:
                    loss = self.config.dlm_loss_weight * loss

                if self.config.dlm_paradigm == 'block_diff':
                    # AR-side loss for block-diffusion paradigm.
                    causal_logits = causal_logits[..., :-1, :].contiguous()
                    causal_logits = causal_logits.view(-1, causal_logits.size(-1))
                    causal_labels = labels[..., 1:].contiguous().view(-1)

                    loss_fct = CrossEntropyLoss(reduction='sum')
                    ar_loss = loss_fct(causal_logits, causal_labels)

                    self.loss_diffusion = loss.detach().item() / num_mask_tokens
                    self.loss_ar = ar_loss.detach().item() / seq_len

                    loss = loss + self.config.ar_loss_weight * ar_loss
                
                # global_loss_avg=True: return (sum_loss, token_count) for external mean.
                if self.config.dlm_paradigm == 'block_diff':
                    loss = (loss, num_mask_tokens + int(self.config.ar_loss_weight * seq_len))
                else:
                    loss = (loss, num_mask_tokens)

        return NemotronLabsDiffusionOutputWithPast(
            loss=loss if not is_teacher else logits,
            logits=logits,
            causal_logits=causal_logits,
            past_key_values=enc_out.past_key_values,
            hidden_states=None,
            attentions=None,
        )


    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        block_length: int,
        threshold: Optional[float] = None,
        causal_context: bool = True,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
    ):
        """Block-wise diffusion decoding with prefix-cached KV (LLaDA-style).

        Each block: append `block_length` mask tokens, then iteratively unmask
        by confidence top-k (with optional threshold). When `causal_context`,
        the KV cache and the next-block seed are produced via a causal forward
        between blocks (flipping `self_attn.diffusion_lm`), matching the AR
        objective at block boundaries.

        Returns (output_ids, nfe) — output_ids includes the prompt.
        """
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        mask_id = self.mask_token_id

        x_accum = prompt_ids.clone()
        B = prompt_ids.shape[0]

        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length
        # one denoising step per generated token (matches legacy chat_utils call)
        steps_per_block = block_length

        nfe = 0

        def _set_diffusion_lm(val: bool):
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, "diffusion_lm"):
                    layer.self_attn.diffusion_lm = val

        # Initial causal prefill produces the KV cache and the next-block seed.
        if causal_context:
            _set_diffusion_lm(False)
        output = self(prompt_ids, use_cache=True, use_causal_mask=causal_context)
        past_key_values = output.past_key_values
        if causal_context:
            _set_diffusion_lm(True)

        next_token = None
        if causal_context:
            last_logit = output.logits[:, -1, :]
            if temperature > 0:
                next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        for num_block in range(num_blocks):
            mask_block = torch.full(
                (B, block_length), mask_id, dtype=prompt_ids.dtype, device=prompt_ids.device,
            )
            if causal_context:
                mask_block[:, 0] = next_token[:, 0]

            x_accum = torch.cat([x_accum, mask_block], dim=1)
            block_start = prompt_ids.size(1) + num_block * block_length
            block_slice = slice(block_start, block_start + block_length)

            # Thinking-budget enforcement: if we've passed max_thinking_tokens
            # without an end-think marker, inject one into this block.
            if end_think_token_id is not None and max_thinking_tokens is not None:
                tokens_before = num_block * block_length
                tokens_after = tokens_before + block_length
                if tokens_after > max_thinking_tokens:
                    gen_so_far = x_accum[:, prompt_ids.size(1):block_start]
                    has_end_think = (
                        (gen_so_far == end_think_token_id).any(dim=1)
                        if gen_so_far.size(1) > 0
                        else torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
                    )
                    if not has_end_think.all():
                        offset = max(0, max_thinking_tokens - tokens_before)
                        inject_pos = block_start + offset
                        for b in range(B):
                            if not has_end_think[b]:
                                x_accum[b, inject_pos] = end_think_token_id

            mask_block_idx0 = x_accum[:, block_slice] == mask_id
            num_transfer_tokens = _get_num_transfer_tokens(mask_block_idx0, steps_per_block)

            # Denoise the current block by repeated confidence-based unmasking.
            for i in range(steps_per_block):
                mask_block_idx = x_accum[:, block_slice] == mask_id
                if mask_block_idx.sum() == 0:
                    break

                nfe += 1
                logits_block = self(
                    x_accum[:, block_slice],
                    past_key_values=past_key_values,
                    use_cache=False,
                ).logits

                x0, transfer_idx = _get_transfer_index(
                    logits_block, temperature, mask_block_idx, x_accum[:, block_slice],
                    num_transfer_tokens=num_transfer_tokens[:, i], threshold=threshold,
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

                if eos_token_id is not None:
                    block_tokens = x_accum[:, block_slice]
                    eos_mask = block_tokens == eos_token_id
                    if eos_mask.any(dim=1).any():
                        after_eos = eos_mask.cumsum(dim=1).bool()
                        mask_before = (block_tokens == mask_id) & ~after_eos
                        if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                            break

            # Post-block: causal forward over the block to update the KV cache
            # and (when causal_context) sample the seed for the next block.
            if causal_context:
                _set_diffusion_lm(False)
            output = self(
                x_accum[:, block_slice],
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=causal_context,
            )
            past_key_values = output.past_key_values
            nfe += 1

            if causal_context:
                _set_diffusion_lm(True)
                last_logit = output.logits[:, -1, :]
                if temperature > 0:
                    next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
                else:
                    next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

            if eos_token_id is not None:
                gen_so_far = x_accum[:, prompt_ids.size(1):]
                is_eos = gen_so_far == eos_token_id
                if is_eos.any(dim=1).all():
                    first_eos = is_eos.to(torch.int64).argmax(dim=1)
                    max_eos = first_eos.max().item()
                    return x_accum[:, : prompt_ids.size(1) + max_eos + 1], nfe

        return x_accum, nfe



    @torch.no_grad()
    def ar_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
    ) -> tuple:
        """Autoregressive generation calling the encoder directly (injected by build_hf_tidar_repo).

        Bypasses NemotronLabsDiffusionModel.forward() to avoid diffusion-specific
        code paths. Calls self.encoder (Ministral3Model) with explicit cache_position,
        position_ids, and use_cache so the KV cache and causal masking behave
        identically to MistralForCausalLM / vLLM.

        Returns:
            (output_ids, nfe) where output_ids includes the prompt.
        """
        for layer in self.encoder.layers:
            if hasattr(layer.self_attn, 'diffusion_lm'):
                layer.self_attn.diffusion_lm = False

        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', None)

        device = prompt_ids.device
        batch_size, prompt_len = prompt_ids.shape

        past_key_values = DynamicCache()
        cache_position = torch.arange(prompt_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        enc_out = self.encoder(
            input_ids=prompt_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        past_key_values = enc_out.past_key_values
        next_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)

        generated_tokens = []
        nfe = 0

        for step in range(max_new_tokens):
            nfe += 1

            if temperature > 0:
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logit, dim=-1, keepdim=True)

            # ---- thinking budget enforcement ----
            if end_think_token_id is not None and max_thinking_tokens is not None:
                if step >= max_thinking_tokens:
                    if generated_tokens:
                        gen_tensor = torch.cat(generated_tokens, dim=1)
                        has_end_think = (gen_tensor == end_think_token_id).any(dim=1)
                    else:
                        has_end_think = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    for b in range(batch_size):
                        if not has_end_think[b]:
                            next_token[b] = end_think_token_id

            generated_tokens.append(next_token)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if step < max_new_tokens - 1:
                cur_pos = prompt_len + step
                step_cache_pos = torch.tensor([cur_pos], device=device)
                step_pos_ids = step_cache_pos.unsqueeze(0).expand(batch_size, -1)

                enc_out = self.encoder(
                    input_ids=next_token,
                    position_ids=step_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=step_cache_pos,
                )
                past_key_values = enc_out.past_key_values
                next_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)

        all_generated = torch.cat(generated_tokens, dim=1)
        output_ids = torch.cat([prompt_ids, all_generated], dim=1)
        return output_ids, nfe


    @torch.no_grad()
    def linear_spec_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
        threshold: float = 0.0,
    ):
        """Linear speculative decoding: diffusion draft + AR verify.

        Each iteration: (1) draft the next block under bidirectional attention,
        (2) verify the drafted block under causal attention, accept the longest
        prefix where draft matches AR + one bonus token, advance the KV cache.

        LoRA-aware: when a PEFT adapter is attached to the model (e.g.
        ``linear_spec_lora``), it is toggled ON for the bidirectional draft
        phase and OFF for the causal prefill / verify phases — so the adapter
        only specializes the diffusion-mode forward and AR semantics are
        preserved. With no adapter loaded the calls are no-ops.

        Returns ``(output_ids, nfe)`` — ``output_ids`` includes the prompt.
        """
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch_size == 1")

        token_mask_id = mask_token_id if mask_token_id is not None else self.config.mask_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        device = prompt_ids.device

        def _set_diffusion_lm(val: bool):
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, "diffusion_lm"):
                    layer.self_attn.diffusion_lm = val

        def _toggle_adapters(enable: bool):
            # No-op when no PEFT/LoRA modules are attached.
            for module in self.modules():
                if hasattr(module, "_disable_adapters"):
                    module._disable_adapters = not enable

        # Prefill (causal, LoRA OFF).
        _set_diffusion_lm(False)
        _toggle_adapters(False)
        enc_out = self.encoder(
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            use_causal_mask=True,
        )
        past_key_values = enc_out.past_key_values
        last_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
        nfe = 1

        if temperature > 0:
            next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            return torch.cat([prompt_ids, next_token], dim=1), nfe

        generated = [next_token]
        total_gen = 1

        while total_gen < max_new_tokens:
            cache_len = past_key_values.get_seq_length()

            block = torch.full((1, block_length), token_mask_id, dtype=torch.long, device=device)
            block[0, 0] = next_token.item()

            # Draft phase (bidirectional, LoRA ON) — iterate at threshold>0 so
            # that even low-confidence blocks make progress.
            _set_diffusion_lm(True)
            _toggle_adapters(True)
            while True:
                is_mask = block == token_mask_id
                if not is_mask.any():
                    break

                enc_out = self.encoder(input_ids=block, past_key_values=past_key_values, use_cache=False)
                nfe += 1

                draft_logits = self.diffusion_head(enc_out.last_hidden_state)
                # LLaDA: logit[i] directly predicts position i — no shift needed.

                if temperature > 0:
                    draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
                    draft_tokens = torch.multinomial(
                        draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
                    ).view(1, block_length)
                else:
                    draft_tokens = draft_logits.argmax(dim=-1)
                    draft_probs = torch.softmax(draft_logits, dim=-1)

                if threshold > 0:
                    draft_conf = torch.gather(draft_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
                    draft_conf = torch.where(is_mask, draft_conf, -torch.inf)
                    unmask = draft_conf >= threshold
                    # Force progress even when every masked position is below threshold.
                    if not unmask.any():
                        best_idx = draft_conf.view(-1).argmax()
                        unmask = torch.zeros_like(is_mask, dtype=torch.bool)
                        unmask.view(-1)[best_idx] = True
                    block[unmask] = draft_tokens[unmask]
                else:
                    block[is_mask] = draft_tokens[is_mask]
                    break

            # Verify phase (causal, LoRA OFF).
            _set_diffusion_lm(False)
            _toggle_adapters(False)
            enc_out = self.encoder(
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=True,
            )
            past_key_values = enc_out.past_key_values
            nfe += 1

            verify_logits = self.diffusion_head(enc_out.last_hidden_state)
            if temperature > 0:
                ar_tokens = torch.multinomial(
                    torch.softmax(verify_logits / temperature, dim=-1).view(-1, verify_logits.shape[-1]),
                    num_samples=1,
                ).view(1, block_length)
            else:
                ar_tokens = verify_logits.argmax(dim=-1)

            # Accept consecutive matches; AR also gives one bonus token at the tail.
            accepted = 0
            for i in range(block_length - 1):
                if ar_tokens[0, i].item() == block[0, i + 1].item():
                    accepted += 1
                else:
                    break
            accepted += 1

            accepted_toks = ar_tokens[:, :accepted]
            generated.append(accepted_toks)
            total_gen += accepted

            _crop_dynamic_cache(past_key_values, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            if eos_token_id is not None:
                eos_pos = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    first_eos = eos_pos[0].item()
                    generated[-1] = accepted_toks[:, : first_eos + 1]
                    total_gen = total_gen - accepted + first_eos + 1
                    break

            # Thinking-budget enforcement: force end-think as next seed if budget exhausted.
            if end_think_token_id is not None and max_thinking_tokens is not None:
                if total_gen > max_thinking_tokens:
                    all_gen = torch.cat(generated, dim=1)
                    if not (all_gen == end_think_token_id).any():
                        next_token = torch.tensor([[end_think_token_id]], device=device)

            if total_gen >= max_new_tokens:
                break

        all_generated = torch.cat(generated, dim=1)
        output_ids = torch.cat([prompt_ids, all_generated], dim=1)
        return output_ids, nfe


# ─── Module-level helpers used by `generate` and `linear_spec_generate` ──

def _crop_dynamic_cache(past_key_values: DynamicCache, max_length: int):
    """Crop a DynamicCache to max_length, compatible with both old and new transformers."""
    if hasattr(past_key_values, 'crop'):
        past_key_values.crop(max_length)
    else:
        for layer_idx in range(len(past_key_values)):
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :max_length]
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :max_length]
        past_key_values._seen_tokens = max_length


def _add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling in float64 (low-precision Gumbel hurts MDM quality)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps: int):
    """Even split of masked positions across `steps`, with remainder front-loaded."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : int(remainder[i])] += 1
    return num_transfer_tokens


def _get_transfer_index(logits, temperature, mask_index, x, num_transfer_tokens, threshold=None):
    """Pick which masked positions to commit this denoising step.

    Returns (x0, transfer_index): x0 is argmax tokens (clamped to original x at
    non-masked positions); transfer_index is a bool mask over positions to
    finalize, chosen by top-k confidence (and filtered by `threshold` if given).
    """
    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def gumbel_topk(log_w: torch.Tensor, k: int) -> torch.Tensor:
    """Return a Bool mask of length len(log_w) with exactly k True."""
    g = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-9) + 1e-9)
    topk = torch.topk(log_w + g, k).indices
    mask = torch.zeros_like(log_w, dtype=torch.bool)
    mask[topk] = True
    return mask

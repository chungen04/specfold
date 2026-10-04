"""Loads a Fast-dLLM v2 checkpoint and runs its forward passes."""

import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


class FastDLLMModelExecutor:
    """Handles all FastDLLM v2 model inference operations."""
    
    # Constants for FastDLLM v2 model
    MASK_ID = 151665
    STOP_TOKEN = 151645
    
    def __init__(self,
                 pretrained_path: str,
                 device: torch.device,
                 target_dtype: torch.dtype):
        """
        Initialize the model executor.
        
        Args:
            pretrained_path: Path to pretrained model (e.g., "Efficient-Large-Model/Fast_dLLM_v2_7B")
            device: Device to run model on
            target_dtype: Target dtype for model
        """
        self.pretrained_path = pretrained_path
        self.device = device
        self.target_dtype = target_dtype

        self._setup_model()
        self._setup_tokenizer()
    
    def _setup_model(self):
        """Load and setup the model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_path,
            trust_remote_code=True,
            torch_dtype=self.target_dtype
        ).eval()
        self.model = self.model.to(self.device)
    
    def _setup_tokenizer(self):
        """Load and setup the tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_path,
                trust_remote_code=True
            )
        except ValueError:
            # Local snapshots whose tokenizer_config.json lacks tokenizer_class
            # make AutoTokenizer fall back to the (custom) model config and
            # fail. Fast-dLLM v2 uses the Qwen2 tokenizer family.
            from transformers import Qwen2TokenizerFast
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(self.pretrained_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self,
                input_seq: torch.Tensor,
                past_key_values: Optional[Tuple] = None,
                use_cache: bool = True,
                update_past_key_values: bool = False,
                use_block_cache: bool = False,
                block_past_key_values: Optional[Tuple] = None,
                replace_position: Optional[int] = None):
        """
        Execute model forward pass.
        
        Args:
            input_seq: Input token sequence
            past_key_values: Cached key-value pairs for full sequence
            use_cache: Whether to use KV cache
            update_past_key_values: Whether to update the KV cache
            use_block_cache: Whether to use block-level caching
            block_past_key_values: Block-level KV cache
            replace_position: Position to replace in block cache
            
        Returns:
            Model outputs with logits and past_key_values
        """
        if use_block_cache and block_past_key_values is not None:
            outputs = self.model(
                input_seq,
                use_cache=use_cache,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values,
                use_block_cache=use_block_cache,
                block_past_key_values=block_past_key_values,
                replace_position=replace_position
            )
        elif use_block_cache:
            outputs = self.model(
                input_seq,
                use_cache=use_cache,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values,
                use_block_cache=use_block_cache
            )
        else:
            outputs = self.model(
                input_seq,
                use_cache=use_cache,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values
            )
        return outputs
    
    def sample_with_top_p(self, 
                          logits: torch.Tensor, 
                          top_p: float = 0.95, 
                          temperature: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tokens with top-p (nucleus) sampling.
        Uses the model's built-in sample_with_top_p method if available.
        
        Args:
            logits: Logits tensor [batch, seq_len, vocab_size]
            top_p: Top-p sampling threshold
            temperature: Sampling temperature
            
        Returns:
            Tuple of (sampled_tokens, probabilities)
        """
        # Try to use model's built-in method if available
        if hasattr(self.model, 'sample_with_top_p'):
            return self.model.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
        
        # Fallback implementation
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Compute probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        
        if temperature > 0:
            sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
            sampled_tokens = sampled_tokens.view(probs.shape[:-1])
        else:
            sampled_tokens = torch.argmax(probs, dim=-1)
        
        return sampled_tokens, probs
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def get_eos_token_id(self):
        """Get the EOS token ID."""
        return self.STOP_TOKEN
    
    def get_mask_id(self):
        """Get the mask token ID."""
        return self.MASK_ID

"""Abstract model wrapper for diffusion LLMs."""

from abc import ABC, abstractmethod
import torch
import numpy as np


class ModelWrapper(ABC):
    """
    Abstract base class for wrapping diffusion language models.
    
    This class defines the interface that all model wrappers must implement
    to work with the Spiffy calibration system.
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the device for model inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def get_target_distribution(self, 
                              input_ids: torch.Tensor, 
                              max_timesteps: int = 50) -> np.ndarray:
        """
        Get the target distribution for draft graph calibration.
        
        Args:
            input_ids: Input token IDs
            max_timesteps: Maximum number of denoising timesteps
        
        Returns:
            Target distribution of shape (max_timesteps, seq_len, vocab_size)
        """
        pass
    
    @abstractmethod
    def generate_sequence(self, 
                         input_ids: torch.Tensor, 
                         max_length: int = 512,
                         **kwargs) -> torch.Tensor:
        """
        Generate a sequence using the model.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
        
        Returns:
            Generated token IDs
        """
        pass
    
    @abstractmethod
    def get_logits(self, 
                  input_ids: torch.Tensor, 
                  timestep: int) -> torch.Tensor:
        """
        Get model logits at a specific timestep.
        
        Args:
            input_ids: Input token IDs
            timestep: Denoising timestep
        
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        pass
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the model."""
        if self.tokenizer is not None:
            return len(self.tokenizer)
        return 50257  # Default for most models
    
    def get_sequence_length(self, input_ids: torch.Tensor) -> int:
        """Get the sequence length from input_ids."""
        return input_ids.shape[-1]
    
    def prepare_input(self, text: str) -> torch.Tensor:
        """
        Prepare input text for model inference.
        
        Args:
            text: Input text string
        
        Returns:
            Token IDs tensor
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return inputs["input_ids"].to(self.device)
    
    def decode_output(self, output_ids: torch.Tensor) -> str:
        """
        Decode output token IDs to text.
        
        Args:
            output_ids: Output token IDs tensor
        
        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"

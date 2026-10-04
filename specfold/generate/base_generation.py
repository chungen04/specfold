"""Abstract base for the generation strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import logging

logger = logging.getLogger(__name__)


class BaseSpeculativeGeneration(ABC):
    """Interface shared by the generation strategies."""
    
    def __init__(self, 
                 pretrained_path: str,
                 calibration_graph: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None,
                 mask_id: Optional[int] = None,
                 block_length: Optional[int] = None,
                 temperature: Optional[float] = None,
                 cfg_scale: Optional[float] = None):
        """
        Initialize the generation strategy.
        
        Args:
            model: The underlying model (transformer, diffusion model, etc.)
            calibration_graph: Calibration graph for speculative decoding
            device: Device to run generation on
            mask_id: Token ID for mask (diffusion models)
            block_length: Length of generation blocks
            temperature: Sampling temperature
            cfg_scale: Classifier-free guidance scale
        """
        self.pretrained_path = pretrained_path
        self.calibration_graph = calibration_graph
        self.device = device
        self.mask_id = mask_id
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
    
    @abstractmethod
    def get_draft_blocks(self, 
                         x: torch.Tensor, 
                         p: torch.Tensor, 
                         mask_index: Optional[torch.Tensor] = None,
                         **kwargs) -> List[torch.Tensor]:
        """
        Obtain draft blocks from the model.
        
        Args:
            x: Input tensor
            p: Additional parameters
            mask_index: Means that only the token positions that the draft block will consider
        """
        pass
    
    @abstractmethod
    def generate(self,
                 prompt: str,
                 attention_mask: Optional[torch.Tensor] = None,
                 steps: int = 128,
                 gen_length: int = 128,
                 **kwargs) -> torch.Tensor:
        """
        Standard generation without speculation.
        
        Args:
            prompt: Input prompt tensor [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            steps: Number of generation steps
            gen_length: Length of text to generate
            **kwargs: Additional generation parameters
        
        Returns:
            Generated tensor [batch, seq_len + gen_length]
        """
        pass
    
    @abstractmethod
    def generate_with_draft_speculation(self,
                                       prompt: str,
                                       attention_mask: Optional[torch.Tensor] = None,
                                       steps: int = 128,
                                       gen_length: int = 128,
                                       **kwargs) -> Tuple[torch.Tensor, float, float, int, int]:
        """
        Generation with speculative decoding using calibration graph.
        
        Args:
            prompt: Input prompt tensor [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            steps: Number of generation steps
            gen_length: Length of text to generate
            **kwargs: Additional generation parameters
        
        Returns:
            Tuple of (x, speedup, speedup_include_early_stop, drafts_computed)
        """
        pass
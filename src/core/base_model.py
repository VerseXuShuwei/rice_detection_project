"""
Base model class for all detection models.

This module provides the abstract base class that all model architectures should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all rice disease detection models.
    
    All custom models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 10)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input without final classification layer.
        
        Useful for knowledge distillation and feature visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        pass
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> float:
        """
        Calculate model size in MB.
        
        Returns:
            Model size in megabytes
        """
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def summary(self) -> Dict[str, Any]:
        """
        Get model summary information.
        
        Returns:
            Dictionary containing model statistics
        """
        return {
            'total_params': self.count_parameters(),
            'model_size_mb': self.get_model_size(),
            'num_classes': self.num_classes,
            'architecture': self.__class__.__name__
        }

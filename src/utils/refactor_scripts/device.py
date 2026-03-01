"""
Device management utilities.

Handles device selection (CPU/CUDA/MPS) with auto-detection.
"""

import torch
from typing import Optional, Union


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get PyTorch device with auto-detection.
    
    Priority:
    1. Explicitly specified device
    2. CUDA if available
    3. MPS (Apple Silicon) if available
    4. CPU fallback
    
    Args:
        device: Device string ('cuda', 'cpu', 'mps', 'cuda:0', etc.) or None for auto
    
    Returns:
        torch.device object
    
    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda:0')  # Specific GPU
        >>> device = get_device('cpu')  # Force CPU
    """
    if device is not None:
        if isinstance(device, torch.device):
            return device
        return torch.device(device)
    
    # Auto-detect
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_name(device: Optional[torch.device] = None) -> str:
    """
    Get human-readable device name.
    
    Args:
        device: PyTorch device (default: auto-detect)
    
    Returns:
        Device name string
    
    Example:
        >>> get_device_name()
        'CUDA (NVIDIA GeForce RTX 3090)'
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device.index or 0)
            return f"CUDA ({gpu_name})"
        return "CUDA (unavailable)"
    elif device.type == 'mps':
        return "MPS (Apple Silicon)"
    else:
        return "CPU"


def set_device(model: torch.nn.Module, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    """
    Move model to device.
    
    Args:
        model: PyTorch model
        device: Target device (default: auto-detect)
    
    Returns:
        Model on target device
    
    Example:
        >>> model = set_device(model, 'cuda')
    """
    target_device = get_device(device)
    return model.to(target_device)


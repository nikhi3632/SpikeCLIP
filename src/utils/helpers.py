"""get_device, seeding, logging, AMP, compile"""
import torch
import random
import numpy as np

def get_device(device: str = "auto") -> torch.device:
    """Get PyTorch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compile_model(model: torch.nn.Module, enabled: bool = False) -> torch.nn.Module:
    """Compile model with torch.compile if enabled."""
    if enabled and hasattr(torch, 'compile'):
        try:
            return torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            return model
    return model

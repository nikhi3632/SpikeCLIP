"""builds optimizer + scheduler from config"""
import torch
import torch.optim as optim
from typing import Dict, Any

def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    optimizer_type = config.get('type', 'adam').lower()
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_epochs: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler from config."""
    scheduler_type = config.get('type', 'cosine').lower()
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        return optim.lr_scheduler.ConstantLR(optimizer)  # No-op scheduler

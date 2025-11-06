"""save_best, save_last, resume, load_best"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
    is_last: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None
):
    """Save model checkpoint.
    
    Args:
        prefix: Optional prefix for checkpoint filenames (e.g., 'coarse' -> 'coarse_best.pth')
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metadata': metadata or {}
    }
    
    if is_best:
        best_name = f'{prefix}_best.pth' if prefix else 'best.pth'
        torch.save(checkpoint, checkpoint_dir / best_name)
    
    if is_last:
        latest_name = f'{prefix}_latest.pth' if prefix else 'latest.pth'
        torch.save(checkpoint, checkpoint_dir / latest_name)

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def load_best_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    prefix: Optional[str] = None
) -> Dict[str, Any]:
    """Load best checkpoint from directory.
    
    Args:
        prefix: Optional prefix for checkpoint filename (e.g., 'coarse' -> 'coarse_best.pth')
    """
    best_name = f'{prefix}_best.pth' if prefix else 'best.pth'
    checkpoint_path = Path(checkpoint_dir) / best_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    return load_checkpoint(str(checkpoint_path), model, optimizer, device)

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
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        strict: If False, allows missing keys (useful for loading old checkpoints with new model structure)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = checkpoint['model_state_dict']
    
    # Load state dict with optional strict mode
    if strict:
        # For strict mode, filter out text_encoder keys (they're initialized from CLIP, not saved)
        # Only load keys that exist in both checkpoint and model (excluding text_encoder)
        model_state = model.state_dict()
        filtered_checkpoint_state = {k: v for k, v in checkpoint_state.items() 
                                     if k in model_state and 'text_encoder' not in k}
        model.load_state_dict(filtered_checkpoint_state, strict=False)
        # text_encoder will use its default initialization from CLIP
    else:
        # Load with strict=False to handle missing keys
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
        
        # Filter out text_encoder keys from missing_keys (they're expected to be missing)
        # text_encoder is initialized from CLIP, not saved in checkpoints
        missing_keys_filtered = [k for k in missing_keys if 'text_encoder' not in k]
        
        if missing_keys_filtered:
            print(f"Warning: Missing keys in checkpoint (will use default initialization): {missing_keys_filtered[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected_keys[:5]}...")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def load_best_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    prefix: Optional[str] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load best checkpoint from directory.
    
    Args:
        prefix: Optional prefix for checkpoint filename (e.g., 'coarse' -> 'coarse_best.pth')
        strict: If False, allows missing keys (useful for loading old checkpoints)
    """
    best_name = f'{prefix}_best.pth' if prefix else 'best.pth'
    checkpoint_path = Path(checkpoint_dir) / best_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    return load_checkpoint(str(checkpoint_path), model, optimizer, device, strict=strict)

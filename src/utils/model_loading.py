"""Utility functions for loading combined models"""
import torch
from pathlib import Path
from typing import Tuple

from models.spikeclip_model import SpikeCLIPModel
from models.coarse_reconstruction import CoarseSNN
from models.prompt_learning import HQ_LQ_PromptAdapter
from models.refinement import RefinementNet

def load_combined_model(checkpoint_path: str, device: torch.device) -> Tuple[SpikeCLIPModel, dict]:
    """
    Load combined model from checkpoint.
    
    Args:
        checkpoint_path: Path to combined_model.pth
        device: Device to load model on
    
    Returns:
        model: Loaded SpikeCLIPModel
        checkpoint: Full checkpoint dictionary
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint is missing required keys
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Validate checkpoint structure
    required_keys = ['config', 'labels', 'model_state_dict']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: {key}")
    
    # Reconstruct models from checkpoint
    config = checkpoint['config']
    labels = checkpoint['labels']
    
    # Stage 1
    coarse_config = config.get('coarse', {})
    coarse_model = CoarseSNN(
        time_steps=coarse_config.get('model', {}).get('time_steps', 25),
        v_threshold=coarse_config.get('model', {}).get('v_threshold', 1.0),
        tau=coarse_config.get('model', {}).get('tau', 2.0)
    )
    
    # Stage 2
    prompt_config = config.get('prompt', {})
    model_config = prompt_config.get('model', {})
    
    # Load HQ_LQ_PromptAdapter (trained in Stage 2)
    prompt_model = HQ_LQ_PromptAdapter(
        clip_model_name=model_config.get('clip_model_name', 'ViT-B/32'),
        prompt_dim=model_config.get('prompt_dim', 77),
        freeze_image_encoder=model_config.get('freeze_image_encoder', True)
    )
    
    # Stage 3
    refine_config = config.get('refine', {})
    refine_model_config = refine_config.get('model', {})
    # Always use use_identity=False to ensure UNet is used (not identity mapping)
    refine_model = RefinementNet(
        in_channels=refine_model_config.get('in_channels', 3),
        out_channels=refine_model_config.get('out_channels', 3),
        base_channels=refine_model_config.get('base_channels', 64),
        num_down=refine_model_config.get('num_down', 4),
        use_identity=False  # Always use UNet, never identity mapping
    )
    
    # Create unified model
    model = SpikeCLIPModel(coarse_model, prompt_model, refine_model, return_features=True, labels=labels)
    
    # Load state dict with validation
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


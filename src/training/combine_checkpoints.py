"""merges 3 best.pth → combined_model.pth"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from typing import Dict, Any

from models.coarse_reconstruction import CoarseSNN
from models.prompt_learning import PromptAdapter
from models.refinement import RefinementNet
from models.spikeclip_model import SpikeCLIPModel
from utils.checkpointing import load_best_checkpoint
from utils.helpers import get_device

def combine_checkpoints(
    checkpoint_dir: str,
    config: Dict[str, Any],
    labels: list,
    output_path: str,
    device=None
):
    """
    Combine three stage checkpoints into a unified model.
    
    Args:
        checkpoint_dir: Directory containing coarse_best.pth, prompt_best.pth, refine_best.pth
        config: Configuration dictionary
        labels: List of class labels
        output_path: Path to save combined model
        device: Device to load models on
    """
    if device is None:
        device = get_device()
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load Stage 1: Coarse
    coarse_config = config.get('coarse', {})
    coarse_model = CoarseSNN(
        time_steps=coarse_config.get('model', {}).get('time_steps', 25),
        v_threshold=coarse_config.get('model', {}).get('v_threshold', 1.0),
        tau=coarse_config.get('model', {}).get('tau', 2.0)
    )
    print(f"Loading coarse model from {checkpoint_dir / 'coarse_best.pth'}")
    load_best_checkpoint(
        str(checkpoint_dir),
        coarse_model,
        device=device,
        prefix='coarse'
    )
    
    # Load Stage 2: Prompt
    # Stage 2 uses HQ_LQ_PromptAdapter for binary classification (HQ vs LQ)
    # But for inference, we need PromptAdapter for multi-class classification
    # We'll create a PromptAdapter that uses CLIP's image encoder from HQ_LQ_PromptAdapter
    from models.prompt_learning import HQ_LQ_PromptAdapter
    prompt_config = config.get('prompt', {})
    model_config = prompt_config.get('model', {})
    num_classes = len(labels)
    
    # First, load HQ_LQ_PromptAdapter to get the image encoder
    hq_lq_prompt_model = HQ_LQ_PromptAdapter(
        clip_model_name=model_config.get('clip_model_name', 'ViT-B/32'),
        prompt_dim=model_config.get('prompt_dim', 77),
        freeze_image_encoder=True
    )
    print(f"Loading HQ/LQ prompt model from {checkpoint_dir / 'prompt_best.pth'}")
    try:
        load_best_checkpoint(
            str(checkpoint_dir),
            hq_lq_prompt_model,
            device=device,
            prefix='prompt',
            strict=False
        )
        # Extract the image encoder from HQ_LQ_PromptAdapter
        image_encoder = hq_lq_prompt_model.image_encoder
        clip_model = hq_lq_prompt_model.clip_model
    except Exception as e:
        print(f"Warning: Failed to load HQ/LQ prompt model ({e}). Using default PromptAdapter.")
        image_encoder = None
        clip_model = None
    
    # Create PromptAdapter for multi-class classification using the same image encoder
    prompt_model = PromptAdapter(
        image_encoder=image_encoder,  # Use image encoder from HQ_LQ_PromptAdapter
        clip_dim=model_config.get('clip_dim', 512),
        num_classes=num_classes,
        prompt_dim=model_config.get('prompt_dim', 77),
        freeze_image_encoder=True,
        clip_model_name=model_config.get('clip_model_name', 'ViT-B/32'),
        class_labels=labels  # Pass labels for better initialization
    )
    # Note: We don't load the prompt checkpoint again - we just use the image encoder
    # The class prompts will be initialized from CLIP text embeddings
    
    # Load Stage 3: Refine
    refine_config = config.get('refine', {})
    refine_model_config = refine_config.get('model', {})
    refine_checkpoint_path = checkpoint_dir / 'refine_best.pth'
    
    # Always use use_identity=False to ensure UNet is used (not identity mapping)
    # If checkpoint exists, load it; if not, initialize UNet randomly
    use_identity = False  # Always use UNet, never identity mapping
    
    refine_model = RefinementNet(
        in_channels=refine_model_config.get('in_channels', 3),
        out_channels=refine_model_config.get('out_channels', 3),
        base_channels=refine_model_config.get('base_channels', 64),
        num_down=refine_model_config.get('num_down', 4),
        use_identity=use_identity  # Always False to use UNet
    )
    
    if refine_checkpoint_path.exists():
        print(f"Loading refine model from {checkpoint_dir / 'refine_best.pth'}")
        load_best_checkpoint(
            str(checkpoint_dir),
            refine_model,
            device=device,
            prefix='refine',
            strict=True  # Checkpoint should match model structure
        )
    else:
        print("Warning: No refinement checkpoint found, using randomly initialized UNet")
    
    # Create unified model
    combined_model = SpikeCLIPModel(
        coarse_model=coarse_model,
        prompt_model=prompt_model,
        refine_model=refine_model,
        return_features=True
    )
    
    # Save combined model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_checkpoint = {
        'model_state_dict': combined_model.state_dict(),
        'config': config,
        'labels': labels,
        'metadata': {
            'coarse_checkpoint': str(checkpoint_dir / 'coarse_best.pth'),
            'prompt_checkpoint': str(checkpoint_dir / 'prompt_best.pth'),
            'refine_checkpoint': str(checkpoint_dir / 'refine_best.pth')
        }
    }
    
    torch.save(combined_checkpoint, output_path)
    print(f"Combined model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Combine stage checkpoints into unified model')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory with stage checkpoints')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output', type=str, default='outputs/checkpoints/ucaltech/combined_model.pth', help='Output path for combined model')
    
    args = parser.parse_args()
    
    from config_loader import load_config
    config = load_config(args.config)
    labels = config['labels']
    
    device = get_device()
    combine_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        config=config,
        labels=labels,
        output_path=args.output,
        device=device
    )

if __name__ == '__main__':
    main()

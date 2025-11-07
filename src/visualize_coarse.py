"""Visualize Stage 1 (Coarse Reconstruction) outputs to check if images are black"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from config_loader import load_config
from data_loader import get_loader
from models.coarse_reconstruction import CoarseSNN
from utils.helpers import get_device, set_seed
from utils.checkpointing import load_best_checkpoint

def visualize_coarse_samples(
    model,
    test_loader,
    device,
    labels,
    num_samples: int = 10,
    output_dir: str = "outputs/visualizations"
):
    """Visualize Stage 1 coarse reconstruction outputs."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_visualized = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if samples_visualized >= num_samples:
                break
            
            spikes = batch[0].to(device)  # [B, T, H, W]
            label_indices = batch[2].to(device)  # [B]
            
            # Forward pass through Stage 1
            coarse_images = model(spikes)  # [B, 3, H, W]
            
            # Visualize each sample in batch
            for i in range(min(spikes.size(0), num_samples - samples_visualized)):
                # Get images
                spike_avg = spikes[i].mean(dim=0).cpu().numpy()  # [H, W]
                coarse_img = coarse_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Clamp to [0, 1] for visualization
                coarse_img = np.clip(coarse_img, 0, 1)
                
                # Compute statistics
                img_mean = coarse_img.mean()
                img_std = coarse_img.std()
                img_min = coarse_img.min()
                img_max = coarse_img.max()
                
                # Check if image is black (all values near 0)
                is_black = img_mean < 0.01 and img_max < 0.05
                
                # Get label
                label_idx = label_indices[i].item()
                label_name = labels[label_idx] if label_idx < len(labels) else f"class_{label_idx}"
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Spike average
                axes[0].imshow(spike_avg, cmap='gray')
                axes[0].set_title(f'Input Spikes (Average)\nLabel: {label_name}')
                axes[0].axis('off')
                
                # Coarse image
                axes[1].imshow(coarse_img)
                status = "⚠️ BLACK IMAGE!" if is_black else "✓ OK"
                axes[1].set_title(f'Stage 1: Coarse Reconstruction\n{status}\nMean: {img_mean:.3f}, Std: {img_std:.3f}')
                axes[1].axis('off')
                
                # Histogram of pixel values
                axes[2].hist(coarse_img.flatten(), bins=50, range=(0, 1), alpha=0.7, color='blue')
                axes[2].axvline(img_mean, color='red', linestyle='--', label=f'Mean: {img_mean:.3f}')
                axes[2].set_xlabel('Pixel Value')
                axes[2].set_ylabel('Frequency')
                axes[2].set_title('Pixel Value Distribution')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                save_path = output_dir / f'coarse_sample_{samples_visualized:03d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Print statistics
                print(f"Sample {samples_visualized}: {label_name}")
                print(f"  Mean: {img_mean:.4f}, Std: {img_std:.4f}, Min: {img_min:.4f}, Max: {img_max:.4f}")
                print(f"  Status: {'⚠️ BLACK IMAGE!' if is_black else '✓ OK'}")
                
                samples_visualized += 1
    
    print(f"\nVisualized {samples_visualized} samples to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Stage 1 coarse reconstruction outputs')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations', help='Visualization directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    data_dir = config['data_dir']
    labels = config['labels']
    
    # Coarse stage config
    coarse_config = config.get('coarse', {})
    model_config = coarse_config.get('model', {})
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data loader config
    data_loader_config = config.get('data_loader', {})
    num_workers = data_loader_config.get('num_workers', None)
    pin_memory = data_loader_config.get('pin_memory', None)
    
    # Test loader
    test_loader = get_loader(data_dir, labels, split='test', batch_size=args.batch_size, 
                             num_workers=num_workers, device=device, pin_memory=pin_memory)
    
    # Model
    model = CoarseSNN(
        time_steps=model_config.get('time_steps', 25),
        v_threshold=model_config.get('v_threshold', 1.0),
        tau=model_config.get('tau', 2.0),
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 3)
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_dir)
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = load_best_checkpoint(
        str(checkpoint_path),
        model,
        device=device,
        prefix='coarse'
    )
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Visualize
    visualize_coarse_samples(
        model,
        test_loader,
        device,
        labels,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()


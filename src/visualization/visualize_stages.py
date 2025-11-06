"""input → coarse → prompt → refine visualization"""
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config_loader import load_config
from data_loader import get_loader
from utils.helpers import get_device, set_seed
from utils.model_loading import load_combined_model

def visualize_pipeline(
    model,
    test_loader,
    device,
    num_samples: int = 5,
    output_dir: str = "outputs/visualizations"
):
    """
    Visualize the full pipeline: input spikes → coarse → refined images.
    
    Args:
        model: SpikeCLIPModel
        test_loader: DataLoader for test set
        device: Device
        num_samples: Number of samples to visualize
        output_dir: Output directory for visualizations
    """
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
            labels = batch[1]  # List of label strings
            
            # Forward pass
            refined_images, clip_features, coarse_images = model(spikes, label_indices)
            
            # Visualize each sample in batch
            for i in range(min(spikes.size(0), num_samples - samples_visualized)):
                # Get images
                spike_avg = spikes[i].mean(dim=0).cpu().numpy()  # [H, W]
                spike_avg = (spike_avg - spike_avg.min()) / (spike_avg.max() - spike_avg.min() + 1e-8)
                
                coarse_img = coarse_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                refined_img = refined_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                
                # Input spikes (average)
                axes[0, 0].imshow(spike_avg, cmap='hot')
                axes[0, 0].set_title(f'Input Spikes (Temporal Average)\nLabel: {labels[i]}')
                axes[0, 0].axis('off')
                
                # Stage 1: Coarse
                axes[0, 1].imshow(coarse_img)
                axes[0, 1].set_title('Stage 1: Coarse Reconstruction')
                axes[0, 1].axis('off')
                
                # Stage 3: Refined
                axes[1, 0].imshow(refined_img)
                axes[1, 0].set_title('Stage 3: Refined Image')
                axes[1, 0].axis('off')
                
                # Difference
                diff = np.abs(refined_img - coarse_img)
                diff_vis = axes[1, 1].imshow(diff)
                axes[1, 1].set_title('Difference (Refined - Coarse)')
                axes[1, 1].axis('off')
                plt.colorbar(diff_vis, ax=axes[1, 1])
                
                plt.tight_layout()
                save_path = output_dir / f'pipeline_sample_{samples_visualized:03d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_visualized += 1
    
    print(f"Visualized {samples_visualized} pipeline samples to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize pipeline stages')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to combined_model.pth')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    data_dir = config['data_dir']
    labels = config['labels']
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data loader config
    data_loader_config = config.get('data_loader', {})
    num_workers = data_loader_config.get('num_workers', None)
    pin_memory = data_loader_config.get('pin_memory', None)
    
    # Load combined model
    print(f"Loading combined model from {args.checkpoint}")
    model, checkpoint = load_combined_model(args.checkpoint, device)
    
    # Test loader
    test_loader = get_loader(data_dir, labels, split='test', batch_size=8, 
                             num_workers=num_workers, device=device, pin_memory=pin_memory)
    
    # Visualize
    visualize_pipeline(model, test_loader, device, args.num_samples, args.output_dir)

if __name__ == '__main__':
    main()

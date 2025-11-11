"""Visualize Stage 1 (Coarse Reconstruction) with metrics and comparisons."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os

from config_loader import load_config
from data_loader import get_loader
from models.coarse_reconstruction import CoarseSNN
from metrics import compute_psnr, compute_ssim, compute_l1_error, compute_l2_error
from utils.helpers import get_device, set_seed
from utils.checkpointing import load_best_checkpoint
from utils.tfi import calculate_tfi_vectorized


def visualize_stage1(
    model,
    test_loader,
    device,
    num_samples: int = 10,
    output_dir: str = "outputs/visualizations",
    labels: list = None
):
    """Visualize Stage 1 reconstruction with TFI target and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    samples_visualized = 0
    
    print(f"Visualizing {num_samples} Stage 1 samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if samples_visualized >= num_samples:
                break
            
            spikes = batch[0].to(device)  # [B, T, H, W]
            label_indices = batch[2].to(device) if len(batch) > 2 else None  # [B]
            
            # Forward pass through Stage 1
            coarse_images = model(spikes)  # [B, 3, H, W]
            
            # Calculate TFI target
            tfi = calculate_tfi_vectorized(spikes, threshold=1.0)  # [B, 1, H, W]
            tfi_target = tfi.repeat(1, 3, 1, 1)  # [B, 3, H, W]
            
            # Visualize each sample in batch
            for i in range(min(spikes.size(0), num_samples - samples_visualized)):
                # Get images
                spike_avg = spikes[i].mean(dim=0).cpu().numpy()  # [H, W]
                spike_avg_normalized = (spike_avg - spike_avg.min()) / (spike_avg.max() - spike_avg.min() + 1e-8)
                
                tfi_np = tfi[i].squeeze().cpu().numpy()  # [H, W]
                tfi_target_np = np.stack([tfi_np] * 3, axis=-1)  # [H, W, 3]
                
                coarse_img = coarse_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                coarse_img = np.clip(coarse_img, 0, 1)
                
                # Get label
                label = labels[label_indices[i].item()] if labels and label_indices is not None else "unknown"
                
                # Compute metrics
                coarse_tensor = coarse_images[i:i+1]  # [1, 3, H, W]
                target_tensor = tfi_target[i:i+1]  # [1, 3, H, W]
                
                # Clamp to [0, 1] before computing metrics
                coarse_tensor = torch.clamp(coarse_tensor, 0, 1)
                target_tensor = torch.clamp(target_tensor, 0, 1)
                
                psnr = compute_psnr(coarse_tensor, target_tensor)
                ssim = compute_ssim(coarse_tensor, target_tensor)
                l1 = compute_l1_error(coarse_tensor, target_tensor)
                l2 = compute_l2_error(coarse_tensor, target_tensor)
                
                # Error map
                error_map = torch.abs(coarse_tensor - target_tensor).mean(dim=1, keepdim=True)  # [1, 1, H, W]
                error_map_np = error_map.squeeze().cpu().numpy()  # [H, W]
                
                # Create visualization
                fig = plt.figure(figsize=(20, 12), constrained_layout=True)
                gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
                
                # Row 1: Input, Target, Coarse, Error Map
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(spike_avg_normalized, cmap='gray')
                ax1.set_title(f'Input: Spike Average\n(True: {label})', fontsize=12, fontweight='bold')
                ax1.axis('off')
                
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(tfi_target_np)
                ax2.set_title('Training Target (TFI)\n(Texture from ISI)', fontsize=12, fontweight='bold')
                ax2.axis('off')
                
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(coarse_img)
                ax3.set_title(f'Stage 1: Coarse Reconstruction\nPSNR: {psnr:.2f}dB, SSIM: {ssim:.3f}, L1: {l1:.4f}', 
                             fontsize=12, fontweight='bold')
                ax3.axis('off')
                
                ax4 = fig.add_subplot(gs[0, 3])
                im4 = ax4.imshow(error_map_np, cmap='hot')
                ax4.set_title('Error Map\n(Coarse vs Target)', fontsize=12, fontweight='bold')
                ax4.axis('off')
                plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
                
                # Row 2: Side-by-side comparison (TFI Target vs Coarse)
                ax5 = fig.add_subplot(gs[1, :2])
                comparison = np.hstack([tfi_target_np, coarse_img])
                # Add vertical line
                h, w = comparison.shape[:2]
                comparison[:, w//2-1:w//2+1] = [1.0, 1.0, 0.0]  # Yellow line
                ax5.imshow(comparison)
                ax5.set_title('Stage 1: TFI Target (L) vs Coarse (R)', fontsize=14, fontweight='bold')
                ax5.axis('off')
                
                # Row 2: Metrics text box
                ax6 = fig.add_subplot(gs[1, 2:])
                ax6.axis('off')
                metrics_text = f"""
Stage 1 Reconstruction Quality Metrics
{'='*50}

PSNR: {psnr:.2f} dB
  {'✅ Good (>20 dB)' if psnr > 20 else '⚠️  Needs improvement (<20 dB)'}

SSIM: {ssim:.3f}
  {'✅ Good (>0.7)' if ssim > 0.7 else '⚠️  Needs improvement (<0.7)'}

L1 Error: {l1:.4f}
  {'✅ Good (<0.1)' if l1 < 0.1 else '⚠️  Needs improvement (>0.1)'}

L2 Error: {l2:.4f}
  {'✅ Good (<0.05)' if l2 < 0.05 else '⚠️  Needs improvement (>0.05)'}

{'='*50}
Note: According to paper, Stage 1 maps spikes to TFI result.
Metrics compare coarse reconstruction to TFI target.
                """
                ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Row 3: Error map (full size)
                ax7 = fig.add_subplot(gs[2, :2])
                im7 = ax7.imshow(error_map_np, cmap='hot')
                ax7.set_title('Error Map (Full Size)\nRed = High Error, Black = Low Error', 
                             fontsize=12, fontweight='bold')
                ax7.axis('off')
                plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
                
                # Row 3: Visual quality assessment
                ax8 = fig.add_subplot(gs[2, 2:])
                ax8.axis('off')
                quality_text = f"""
Visual Quality Assessment
{'='*50}

Structure Preservation:
  {'✅ Good' if ssim > 0.7 else '⚠️  Needs improvement'}
  - SSIM > 0.7 indicates good structural similarity

Pixel Accuracy:
  {'✅ Good' if psnr > 20 else '⚠️  Needs improvement'}
  - PSNR > 20 dB indicates good pixel-level accuracy

Overall Quality:
  {'✅ Stage 1 is working well!' if psnr > 20 and ssim > 0.7 and l1 < 0.1 else '⚠️  Stage 1 needs improvement'}
  
{'='*50}
Checklist:
  [{'✅' if psnr > 20 else '❌'}] PSNR > 20 dB
  [{'✅' if ssim > 0.7 else '❌'}] SSIM > 0.7
  [{'✅' if l1 < 0.1 else '❌'}] L1 < 0.1
  [{'✅' if l2 < 0.05 else '❌'}] L2 < 0.05
                """
                ax8.text(0.1, 0.5, quality_text, fontsize=11, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                # Save figure
                output_path = os.path.join(output_dir, f'stage1_sample_{samples_visualized:03d}_{label}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved: {output_path}")
                print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}, L1: {l1:.4f}, L2: {l2:.4f}")
                
                samples_visualized += 1
    
    print(f"\n✅ Visualized {samples_visualized} Stage 1 samples in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Stage 1: Coarse Reconstruction')
    parser.add_argument('--config', type=str, default='configs/u-caltech.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/ucaltech', help='Path to Stage 1 checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loader workers')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations/stage1', help='Directory to save visualizations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = get_device()
    
    # Data loader
    data_dir = config['data_dir']
    labels = config['labels']
    num_workers = args.num_workers if args.num_workers is not None else config['data_loader']['num_workers']
    pin_memory = config['data_loader']['pin_memory']
    
    test_loader = get_loader(data_dir, labels, split='test', batch_size=args.batch_size, 
                             num_workers=num_workers, device=device, pin_memory=pin_memory)
    
    # Stage 1 model
    coarse_config = config.get('coarse', {})
    model_config = coarse_config.get('model', {})
    
    model = CoarseSNN(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 3),
        time_steps=model_config.get('time_steps', 200),
        v_threshold=model_config.get('v_threshold', 1.0),
        tau=model_config.get('tau', 2.0)
    )
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading Stage 1 checkpoint from {args.checkpoint}...")
    try:
        load_best_checkpoint(
            args.checkpoint,
            model,
            device=device,
            prefix='coarse'
        )
        print("✅ Stage 1 checkpoint loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure Stage 1 is trained first: make train-coarse")
        return
    
    # Visualize
    visualize_stage1(model, test_loader, device, args.num_samples, args.output_dir, labels)
    
    print(f"\n✅ Stage 1 visualization complete!")
    print(f"   Check visualizations in: {args.output_dir}")


if __name__ == '__main__':
    main()


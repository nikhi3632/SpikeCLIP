"""loads combined_model.pth → visualize + metrics"""
import sys
from pathlib import Path

# Add parent directory to path for imports (if run from subdirectory)
if Path(__file__).parent.name != 'src':
    sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

from config_loader import load_config
from data_loader import get_loader
from metrics import compute_psnr, compute_ssim, compute_l1_error, compute_l2_error
from utils.helpers import get_device, set_seed
from utils.model_loading import load_combined_model
from utils.logging import log_test_metrics

def visualize_samples(
    model,
    test_loader,
    device,
    num_samples: int = 10,
    output_dir: str = "outputs/visualizations",
    labels: list = None
):
    """Visualize the complete pipeline: Input → Stage 1 → Stage 2 → Stage 3 → Output with metrics."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file for visualization metrics
    log_file = output_dir / "visual.log"
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VISUALIZATION METRICS LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write("=" * 80 + "\n\n")
    
    samples_visualized = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if samples_visualized >= num_samples:
                break
            
            spikes = batch[0].to(device)  # [B, T, H, W]
            batch_labels = batch[1]  # List of label strings
            
            # Forward pass through complete pipeline (no label_indices needed - classification removed)
            refined_images, clip_features, coarse_images = model(spikes)
            
            # Generate HQ images for Stage 2 visualization (according to paper)
            # According to paper: HQ images from generation pipeline (mixture for real data)
            from utils.hq_generation import generate_hq_images
            hq_images = generate_hq_images(spikes, method="mixture")  # [B, 3, H, W]
            
            # Visualize each sample in batch
            for i in range(min(spikes.size(0), num_samples - samples_visualized)):
                # Get images and intermediate results
                # Input: Spike stream visualization
                spike_stream = spikes[i].cpu().numpy()  # [T, H, W]
                spike_count = spike_stream.sum(axis=0)  # [H, W]
                spike_count_normalized = (spike_count - spike_count.min()) / (spike_count.max() - spike_count.min() + 1e-8)
                
                # Create training target (TFI) for metrics computation
                from utils.tfi import calculate_tfi_vectorized
                tfi = calculate_tfi_vectorized(spikes[i:i+1], threshold=1.0)  # [1, 1, H, W]
                
                # Stage 1: Coarse reconstruction
                coarse_img = coarse_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Stage 2: HQ image (reference for prompt learning)
                hq_img_sample = hq_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Stage 3: Refined output
                refined_img = refined_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Compute Stage 1 reconstruction metrics (coarse vs TFI target)
                # According to paper: Stage 1 is trained to map spikes to TFI result
                coarse_tensor = coarse_images[i:i+1]  # [1, 3, H, W]
                target_tensor = tfi.repeat(1, 3, 1, 1).to(coarse_tensor.device)  # [1, 3, H, W] - TFI target
                coarse_tensor = torch.clamp(coarse_tensor, 0, 1)
                target_tensor = torch.clamp(target_tensor, 0, 1)
                
                stage1_psnr = compute_psnr(coarse_tensor, target_tensor)
                stage1_ssim = compute_ssim(coarse_tensor, target_tensor)
                stage1_l1 = compute_l1_error(coarse_tensor, target_tensor)
                stage1_l2 = compute_l2_error(coarse_tensor, target_tensor)
                
                # Compute Stage 3 refinement metrics (refined vs coarse)
                refined_tensor = refined_images[i:i+1]
                refined_tensor = torch.clamp(refined_tensor, 0, 1)
                coarse_tensor_clamped = torch.clamp(coarse_images[i:i+1], 0, 1)
                
                stage3_psnr = compute_psnr(refined_tensor, coarse_tensor_clamped)
                stage3_ssim = compute_ssim(refined_tensor, coarse_tensor_clamped)
                stage3_l1 = compute_l1_error(refined_tensor, coarse_tensor_clamped)
                stage3_l2 = compute_l2_error(refined_tensor, coarse_tensor_clamped)
                
                # Create simple visualization: Input, Intermediate, Output
                fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
                
                # Input: Spike stream count map
                axes[0].imshow(spike_count_normalized, cmap='hot')
                axes[0].set_title(f'INPUT\nSpike Stream\n({spike_count.sum():.0f} spikes)\nLabel: {batch_labels[i]}')
                axes[0].axis('off')
                
                # Stage 1: Coarse Reconstruction
                axes[1].imshow(np.clip(coarse_img, 0, 1))
                axes[1].set_title(f'STAGE 1: Coarse\nPSNR: {stage1_psnr:.2f}dB\nSSIM: {stage1_ssim:.3f}')
                axes[1].axis('off')
                
                # Stage 2: HQ Image Reference
                axes[2].imshow(np.clip(hq_img_sample, 0, 1))
                axes[2].set_title('STAGE 2: HQ Reference\n(Mixture)')
                axes[2].axis('off')
                
                # Stage 3: Refined Output
                axes[3].imshow(np.clip(refined_img, 0, 1))
                axes[3].set_title(f'STAGE 3: Refined\nPSNR: {stage3_psnr:.2f}dB\nSSIM: {stage3_ssim:.3f}')
                axes[3].axis('off')
                
                # Save figure
                save_path = output_dir / f'sample_{samples_visualized:03d}_pipeline.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Log comprehensive metrics to file
                with open(log_file, 'a') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Sample {samples_visualized:03d} - Complete Pipeline Analysis\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"True Label: {batch_labels[i]}\n")
                    f.write(f"Spike Stream: {spike_stream.shape[0]} frames, {spike_count.sum():.0f} total spikes\n")
                    f.write(f"\n--- STAGE 1: Coarse Reconstruction (Spike → TFI Target) ---\n")
                    f.write(f"  Target: TFI (Texture from ISI) result\n")
                    f.write(f"  PSNR: {stage1_psnr:.4f} dB {'✓' if stage1_psnr > 20 else '✗'} (Target: >20 dB)\n")
                    f.write(f"  SSIM: {stage1_ssim:.4f} {'✓' if stage1_ssim > 0.7 else '✗'} (Target: >0.7)\n")
                    f.write(f"  L1 Error: {stage1_l1:.4f} {'✓' if stage1_l1 < 0.1 else '✗'} (Target: <0.1)\n")
                    f.write(f"  L2 Error: {stage1_l2:.4f} {'✓' if stage1_l2 < 0.1 else '✗'} (Target: <0.1)\n")
                    f.write(f"  Interpretation: {'Good reconstruction' if stage1_psnr > 20 and stage1_ssim > 0.7 else 'Needs improvement'}\n")
                    f.write(f"\n--- STAGE 2: Prompt Learning ---\n")
                    f.write(f"  HQ Image: Generated from mixture model (TFI + WGSE + Count)\n")
                    f.write(f"  Purpose: Learn high-quality vs low-quality image features\n")
                    f.write(f"  Output: Learned HQ and LQ prompts (used in Stage 3)\n")
                    f.write(f"\n--- STAGE 3: Fine Reconstruction (Coarse → Refined) ---\n")
                    f.write(f"  Input: Coarse images from Stage 1\n")
                    f.write(f"  PSNR: {stage3_psnr:.4f} dB {'✓' if stage3_psnr < 100 else '⚠'} (Should be <inf if refining)\n")
                    f.write(f"  SSIM: {stage3_ssim:.4f} {'✓' if stage3_ssim < 1.0 else '⚠'} (Should be <1.0 if refining)\n")
                    f.write(f"  L1 Error: {stage3_l1:.4f} {'✓' if stage3_l1 > 0 else '⚠'} (Should be >0 if refining)\n")
                    f.write(f"  L2 Error: {stage3_l2:.4f} {'✓' if stage3_l2 > 0 else '⚠'} (Should be >0 if refining)\n")
                    f.write(f"  Interpretation: {'Refining (improving)' if stage3_psnr < 100 and stage3_ssim < 1.0 else 'Identity mapping (not refining)'}\n")
                    f.write(f"\n--- OVERALL PIPELINE QUALITY ---\n")
                    f.write(f"  Stage 1 Performance: {'Good' if stage1_psnr > 20 and stage1_ssim > 0.7 else 'Needs improvement'}\n")
                    f.write(f"  Stage 3 Refinement: {'Active' if stage3_psnr < 100 and stage3_ssim < 1.0 else 'Inactive'}\n")
                    f.write(f"  Final Output Quality: {'High' if stage1_psnr > 20 and stage3_ssim < 1.0 else 'Medium' if stage1_psnr > 15 else 'Low'}\n")
                    f.write(f"{'='*80}\n")
                
                samples_visualized += 1
    
    # Write summary to log file
    with open(log_file, 'a') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Total samples visualized: {samples_visualized}\n")
        f.write(f"Visualizations saved to: {output_dir}\n")
        f.write(f"Log file: {log_file}\n")
        f.write(f"{'=' * 80}\n")
    
    print(f"Visualized {samples_visualized} samples to {output_dir}")
    print(f"Metrics logged to {log_file}")

def main():
    parser = argparse.ArgumentParser(description='Test combined model with visualization')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to combined_model.pth')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='outputs/logs', help='Output directory')
    parser.add_argument('--vis-dir', type=str, default='outputs/visualizations', help='Visualization directory')
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
    model, _ = load_combined_model(args.checkpoint, device)
    
    # Test loader
    test_loader = get_loader(data_dir, labels, split='test', batch_size=args.batch_size, 
                             num_workers=num_workers, device=device, pin_memory=pin_memory)
    
    # Run inference and compute metrics
    print("Computing metrics...")
    psnr_values = []
    ssim_values = []
    l1_errors = []
    l2_errors = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            spikes = batch[0].to(device)
            
            # Forward pass (no label_indices needed - classification removed)
            refined_images, clip_features, coarse_images = model(spikes)
            
            # Reconstruction metrics
            # For unpaired training: compare refined vs coarse (improvement metric)
            target = coarse_images
            for i in range(spikes.size(0)):
                pred_img = refined_images[i:i+1]
                target_img = target[i:i+1]
                
                # Clamp images to [0, 1] before computing metrics
                pred_img = torch.clamp(pred_img, 0, 1)
                target_img = torch.clamp(target_img, 0, 1)
                
                # Metrics comparing refined to coarse (shows refinement quality)
                psnr_values.append(compute_psnr(pred_img, target_img))
                ssim_values.append(compute_ssim(pred_img, target_img))
                l1_errors.append(compute_l1_error(pred_img, target_img))
                l2_errors.append(compute_l2_error(pred_img, target_img))
    
    # Compute averages
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    avg_l1 = sum(l1_errors) / len(l1_errors) if l1_errors else 0.0
    avg_l2 = sum(l2_errors) / len(l2_errors) if l2_errors else 0.0
    # Print metrics
    print(f"\n===== TEST RESULTS =====")
    print(f"Reconstruction Metrics (Refined vs Coarse):")
    print(f"  PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, L1: {avg_l1:.4f}, L2: {avg_l2:.4f}")
    print(f"  Note: According to paper, Stage 3 refines coarse images.")
    print(f"        Metrics compare refined images to coarse images (refinement quality).")
    
    # Save test metrics to log file
    metrics_dict = {
        'timestamp': time.ctime(),
        'device': str(device),
        'checkpoint': args.checkpoint,
        'dataset': data_dir,
        'batch_size': args.batch_size,
        'reconstruction_metrics': {
            'avg_psnr': float(avg_psnr),
            'avg_ssim': float(avg_ssim),
            'avg_l1_error': float(avg_l1),
            'avg_l2_error': float(avg_l2)
        },
    }
    log_test_metrics(metrics_dict, output_dir=args.output_dir)
    
    # Visualize samples
    print(f"\nVisualizing {args.num_samples} samples...")
    visualize_samples(model, test_loader, device, args.num_samples, args.vis_dir, labels)
    
    print(f"\nTest complete! Visualizations saved to {args.vis_dir}")

if __name__ == '__main__':
    main()

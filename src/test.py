"""loads combined_model.pth → visualize + metrics"""
import torch
import argparse
from pathlib import Path
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
    output_dir: str = "outputs/visualizations"
):
    """Visualize sample predictions from the pipeline."""
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
            
            # Forward pass (clip_features not needed for visualization)
            refined_images, _, coarse_images = model(spikes, label_indices)
            
            # Visualize each sample in batch
            for i in range(min(spikes.size(0), num_samples - samples_visualized)):
                # Get images
                spike_avg = spikes[i].mean(dim=0).cpu().numpy()  # [H, W]
                coarse_img = coarse_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                refined_img = refined_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Create visualization
                _, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                # Spike average
                axes[0].imshow(spike_avg, cmap='gray')
                axes[0].set_title('Input Spikes (Average)')
                axes[0].axis('off')
                
                # Coarse image
                axes[1].imshow(coarse_img)
                axes[1].set_title('Stage 1: Coarse')
                axes[1].axis('off')
                
                # Refined image
                axes[2].imshow(refined_img)
                axes[2].set_title('Stage 3: Refined')
                axes[2].axis('off')
                
                # Difference
                diff = np.abs(refined_img - coarse_img)
                axes[3].imshow(diff)
                axes[3].set_title('Difference (Refined - Coarse)')
                axes[3].axis('off')
                
                plt.tight_layout()
                save_path = output_dir / f'sample_{samples_visualized:03d}_pipeline.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_visualized += 1
    
    print(f"Visualized {samples_visualized} samples to {output_dir}")

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
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            spikes = batch[0].to(device)
            label_indices = batch[2].to(device)
            
            # Forward pass
            refined_images, clip_features, coarse_images = model(spikes, label_indices)
            
            # Classification (using already computed clip_features)
            image_features = clip_features  # Already computed from forward pass
            all_label_indices = torch.arange(len(labels), device=device)
            all_text_features = model.prompt_model.get_text_embeddings(all_label_indices)
            similarities = torch.matmul(image_features, all_text_features.t())
            predictions = similarities.argmax(dim=1)
            correct_predictions += (predictions == label_indices).sum().item()
            total_predictions += predictions.size(0)
            
            # Reconstruction metrics
            target = coarse_images
            for i in range(spikes.size(0)):
                pred_img = refined_images[i:i+1]
                target_img = target[i:i+1]
                
                psnr_values.append(compute_psnr(pred_img, target_img))
                ssim_values.append(compute_ssim(pred_img, target_img))
                l1_errors.append(compute_l1_error(pred_img, target_img))
                l2_errors.append(compute_l2_error(pred_img, target_img))
    
    # Compute averages
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    avg_l1 = sum(l1_errors) / len(l1_errors) if l1_errors else 0.0
    avg_l2 = sum(l2_errors) / len(l2_errors) if l2_errors else 0.0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Print metrics
    print(f"\n===== TEST RESULTS =====")
    print(f"Reconstruction - PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, L1: {avg_l1:.4f}, L2: {avg_l2:.4f}")
    print(f"Classification - Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
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
        'classification_metrics': {
            'accuracy': float(accuracy),
            'correct': int(correct_predictions),
            'total': int(total_predictions)
        }
    }
    log_test_metrics(metrics_dict, output_dir=args.output_dir)
    
    # Visualize samples
    print(f"\nVisualizing {args.num_samples} samples...")
    visualize_samples(model, test_loader, device, args.num_samples, args.vis_dir)
    
    print(f"\nTest complete! Visualizations saved to {args.vis_dir}")

if __name__ == '__main__':
    main()

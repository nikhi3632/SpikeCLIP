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
    """Visualize sample predictions from the pipeline including Stage 2 classification."""
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
            batch_labels = batch[1]  # List of label strings
            
            # Forward pass
            refined_images, clip_features, coarse_images = model(spikes, label_indices)
            
            # Get all text embeddings for Stage 2 visualization
            all_label_indices = torch.arange(model.prompt_model.num_classes, device=device)
            all_text_features = model.prompt_model.get_text_embeddings(all_label_indices)
            
            # Visualize each sample in batch
            for i in range(min(spikes.size(0), num_samples - samples_visualized)):
                # Get images
                spike_avg = spikes[i].mean(dim=0).cpu().numpy()  # [H, W]
                spike_avg = (spike_avg - spike_avg.min()) / (spike_avg.max() - spike_avg.min() + 1e-8)
                coarse_img = coarse_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                refined_img = refined_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Stage 2: Get similarity scores for this sample
                image_feat = clip_features[i:i+1]  # [1, D]
                similarities = torch.matmul(image_feat, all_text_features.t()).squeeze(0)  # [num_classes]
                similarities = similarities.cpu().numpy()
                
                # Get top-k predictions
                top_k = 5
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                top_scores = similarities[top_indices]
                top_labels_list = [labels[idx] if labels else f"Class {idx}" for idx in top_indices]
                
                # Get ground truth index
                true_idx = label_indices[i].item()
                true_label = batch_labels[i]
                true_score = similarities[true_idx]
                pred_idx = top_indices[0]
                pred_label = top_labels_list[0]
                pred_score = top_scores[0]
                
                # Create visualization with Stage 2
                fig = plt.figure(figsize=(16, 8))
                gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
                
                # Row 1: Images
                # Input spikes
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(spike_avg, cmap='hot')
                ax1.set_title(f'Input Spikes\nTrue: {true_label}')
                ax1.axis('off')
                
                # Stage 1: Coarse
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(np.clip(coarse_img, 0, 1))
                ax2.set_title('Stage 1: Coarse Reconstruction')
                ax2.axis('off')
                
                # Stage 3: Refined
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(np.clip(refined_img, 0, 1))
                ax3.set_title('Stage 3: Refined Image')
                ax3.axis('off')
                
                # Stage 2: Top-k predictions
                ax4 = fig.add_subplot(gs[0, 3])
                colors = ['green' if idx == true_idx else 'red' if idx == pred_idx else 'gray' 
                         for idx in top_indices]
                bars = ax4.barh(range(top_k), top_scores, color=colors)
                ax4.set_yticks(range(top_k))
                ax4.set_yticklabels(top_labels_list)
                ax4.set_xlabel('Similarity Score')
                ax4.set_title(f'Stage 2: Top-{top_k} Predictions\nPred: {pred_label} ({pred_score:.3f}) | True: {true_label} ({true_score:.3f})')
                ax4.grid(axis='x', alpha=0.3)
                # Add score labels on bars
                for j, (bar, score) in enumerate(zip(bars, top_scores)):
                    ax4.text(score + 0.01, j, f'{score:.3f}', va='center', fontsize=8)
                
                # Row 2: Stage 2 - All similarities (sorted)
                ax5 = fig.add_subplot(gs[1, :2])
                sorted_indices = np.argsort(similarities)[::-1]
                sorted_scores = similarities[sorted_indices]
                sorted_labels_short = [labels[idx][:10] if labels else f"C{idx}" for idx in sorted_indices[:20]]  # Show top 20
                colors_all = ['green' if idx == true_idx else 'red' if idx == pred_idx else 'lightblue' 
                             for idx in sorted_indices[:20]]
                ax5.bar(range(len(sorted_labels_short)), sorted_scores[:20], color=colors_all)
                ax5.set_xticks(range(len(sorted_labels_short)))
                ax5.set_xticklabels(sorted_labels_short, rotation=45, ha='right', fontsize=8)
                ax5.set_ylabel('Similarity Score')
                ax5.set_title(f'Stage 2: All Class Similarities (Top 20)\n{"✓ Correct" if pred_idx == true_idx else "✗ Wrong"}')
                ax5.grid(axis='y', alpha=0.3)
                
                # Row 2: Stage 2 - Similarity distribution
                ax6 = fig.add_subplot(gs[1, 2:])
                ax6.hist(similarities, bins=30, edgecolor='black', alpha=0.7)
                ax6.axvline(true_score, color='green', linestyle='--', linewidth=2, label=f'True ({true_score:.3f})')
                ax6.axvline(pred_score, color='red', linestyle='--', linewidth=2, label=f'Pred ({pred_score:.3f})')
                ax6.set_xlabel('Similarity Score')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Stage 2: Similarity Score Distribution')
                ax6.legend()
                ax6.grid(alpha=0.3)
                
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
            # For unpaired training: compare refined vs coarse (improvement metric)
            target = coarse_images
            for i in range(spikes.size(0)):
                pred_img = refined_images[i:i+1]
                target_img = target[i:i+1]
                
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
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Print metrics
    print(f"\n===== TEST RESULTS =====")
    print(f"Reconstruction Metrics (Refined vs Coarse):")
    print(f"  PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, L1: {avg_l1:.4f}, L2: {avg_l2:.4f}")
    print(f"  Note: Metrics compare refined images to coarse images (refinement quality)")
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
    visualize_samples(model, test_loader, device, args.num_samples, args.vis_dir, labels)
    
    print(f"\nTest complete! Visualizations saved to {args.vis_dir}")

if __name__ == '__main__':
    main()

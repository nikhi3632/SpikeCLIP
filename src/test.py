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
                spike_avg_normalized = (spike_avg - spike_avg.min()) / (spike_avg.max() - spike_avg.min() + 1e-8)
                
                # Create training target (spike mean + variance, normalized) for comparison
                spike_mean = spikes[i].mean(dim=0).cpu().numpy()  # [H, W]
                spike_var = spikes[i].var(dim=0).cpu().numpy()  # [H, W]
                spike_combined = spike_mean + 0.3 * spike_var  # [H, W]
                spike_combined_min = spike_combined.min()
                spike_combined_max = spike_combined.max()
                spike_combined_norm = (spike_combined - spike_combined_min) / (spike_combined_max - spike_combined_min + 1e-8)
                spike_target = np.stack([spike_combined_norm] * 3, axis=-1)  # [H, W, 3] - training target
                
                coarse_img = coarse_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                refined_img = refined_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                
                # Compute Stage 1 reconstruction metrics (coarse vs training target)
                coarse_tensor = coarse_images[i:i+1]  # [1, 3, H, W]
                target_tensor = torch.from_numpy(spike_target).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
                target_tensor = target_tensor.to(coarse_tensor.device)
                
                stage1_psnr = compute_psnr(coarse_tensor, target_tensor)
                stage1_ssim = compute_ssim(coarse_tensor, target_tensor)
                stage1_l1 = compute_l1_error(coarse_tensor, target_tensor)
                
                # Compute Stage 3 refinement metrics (refined vs coarse)
                refined_tensor = refined_images[i:i+1]
                stage3_psnr = compute_psnr(refined_tensor, coarse_tensor)
                stage3_ssim = compute_ssim(refined_tensor, coarse_tensor)
                stage3_l1 = compute_l1_error(refined_tensor, coarse_tensor)
                
                # Error maps for visualization
                stage1_error = np.abs(coarse_img - spike_target)
                stage3_error = np.abs(refined_img - coarse_img)
                
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
                
                # Create visualization with reconstruction quality assessment
                fig = plt.figure(figsize=(20, 10), constrained_layout=True)
                gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
                
                # Row 1: Input and Targets
                # Input spikes (temporal average)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(spike_avg_normalized, cmap='hot')
                ax1.set_title(f'Input: Spike Average\nTrue: {true_label}')
                ax1.axis('off')
                
                # Training target (spike mean + variance)
                ax1b = fig.add_subplot(gs[0, 1])
                ax1b.imshow(np.clip(spike_target, 0, 1))
                ax1b.set_title('Training Target\n(Mean + 0.3×Var)')
                ax1b.axis('off')
                
                # Stage 1: Coarse Reconstruction
                ax2 = fig.add_subplot(gs[0, 2])
                ax2.imshow(np.clip(coarse_img, 0, 1))
                ax2.set_title(f'Stage 1: Coarse Reconstruction\nPSNR: {stage1_psnr:.2f}dB, SSIM: {stage1_ssim:.3f}, L1: {stage1_l1:.4f}')
                ax2.axis('off')
                
                # Stage 1 Error Map
                ax2e = fig.add_subplot(gs[0, 3])
                ax2e.imshow(stage1_error.mean(axis=2), cmap='hot', vmin=0, vmax=0.5)
                ax2e.set_title('Stage 1 Error Map\n(Coarse vs Target)')
                ax2e.axis('off')
                
                # Stage 3: Refined
                ax3 = fig.add_subplot(gs[0, 4])
                ax3.imshow(np.clip(refined_img, 0, 1))
                ax3.set_title(f'Stage 3: Refined\nPSNR: {stage3_psnr:.2f}dB, SSIM: {stage3_ssim:.3f}, L1: {stage3_l1:.4f}')
                ax3.axis('off')
                
                # Row 2: Stage 3 Error and Stage 2 Classification
                # Stage 3 Error Map
                ax3e = fig.add_subplot(gs[1, 0])
                ax3e.imshow(stage3_error.mean(axis=2), cmap='hot', vmin=0, vmax=0.5)
                ax3e.set_title('Stage 3 Error Map\n(Refined vs Coarse)')
                ax3e.axis('off')
                
                # Stage 2: Top-k predictions
                ax4 = fig.add_subplot(gs[1, 1])
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
                ax5 = fig.add_subplot(gs[1, 2:4])
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
                ax6 = fig.add_subplot(gs[1, 4])
                ax6.hist(similarities, bins=30, edgecolor='black', alpha=0.7)
                ax6.axvline(true_score, color='green', linestyle='--', linewidth=2, label=f'True ({true_score:.3f})')
                ax6.axvline(pred_score, color='red', linestyle='--', linewidth=2, label=f'Pred ({pred_score:.3f})')
                ax6.set_xlabel('Similarity Score')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Stage 2: Similarity Distribution')
                ax6.legend()
                ax6.grid(alpha=0.3)
                
                # Row 3: Side-by-side comparisons for better assessment
                # Comparison: Target vs Coarse (Stage 1 quality)
                ax7 = fig.add_subplot(gs[2, 0])
                comparison_stage1 = np.hstack([np.clip(spike_target, 0, 1), np.clip(coarse_img, 0, 1)])
                ax7.imshow(comparison_stage1)
                ax7.axvline(spike_target.shape[1], color='yellow', linewidth=2)
                ax7.set_title('Stage 1: Target (L) vs Coarse (R)')
                ax7.axis('off')
                
                # Comparison: Coarse vs Refined (Stage 3 quality)
                ax8 = fig.add_subplot(gs[2, 1])
                comparison_stage3 = np.hstack([np.clip(coarse_img, 0, 1), np.clip(refined_img, 0, 1)])
                ax8.imshow(comparison_stage3)
                ax8.axvline(coarse_img.shape[1], color='yellow', linewidth=2)
                ax8.set_title('Stage 3: Coarse (L) vs Refined (R)')
                ax8.axis('off')
                
                # Metrics summary
                ax9 = fig.add_subplot(gs[2, 2:])
                ax9.axis('off')
                metrics_text = f"""
Reconstruction Quality Metrics:

Stage 1 (Coarse vs Training Target):
  PSNR: {stage1_psnr:.2f} dB (higher is better, >20 is good)
  SSIM: {stage1_ssim:.3f} (higher is better, >0.7 is good)
  L1 Error: {stage1_l1:.4f} (lower is better, <0.1 is good)

Stage 3 (Refined vs Coarse):
  PSNR: {stage3_psnr:.2f} dB (should be <inf if refining)
  SSIM: {stage3_ssim:.3f} (should be <1.0 if refining)
  L1 Error: {stage3_l1:.4f} (should be >0 if refining)

Stage 2 Classification:
  Prediction: {pred_label} (score: {pred_score:.3f})
  Ground Truth: {true_label} (score: {true_score:.3f})
  Status: {"✓ CORRECT" if pred_idx == true_idx else "✗ WRONG"}
                """
                ax9.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace', 
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # tight_layout() not needed with constrained_layout=True
                save_path = output_dir / f'sample_{samples_visualized:03d}_pipeline.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Log metrics to file
                with open(log_file, 'a') as f:
                    f.write(f"\n--- Sample {samples_visualized:03d} ---\n")
                    f.write(f"True Label: {true_label}\n")
                    f.write(f"Predicted Label: {pred_label} {'(CORRECT)' if pred_idx == true_idx else '(WRONG)'}\n")
                    f.write(f"Prediction Score: {pred_score:.4f}\n")
                    f.write(f"True Label Score: {true_score:.4f}\n")
                    f.write(f"\nStage 1 Reconstruction (Coarse vs Training Target):\n")
                    f.write(f"  PSNR: {stage1_psnr:.4f} dB (higher is better, >20 is good)\n")
                    f.write(f"  SSIM: {stage1_ssim:.4f} (higher is better, >0.7 is good)\n")
                    f.write(f"  L1 Error: {stage1_l1:.4f} (lower is better, <0.1 is good)\n")
                    f.write(f"\nStage 3 Refinement (Refined vs Coarse):\n")
                    f.write(f"  PSNR: {stage3_psnr:.4f} dB (should be <inf if refining)\n")
                    f.write(f"  SSIM: {stage3_ssim:.4f} (should be <1.0 if refining)\n")
                    f.write(f"  L1 Error: {stage3_l1:.4f} (should be >0 if refining)\n")
                    f.write(f"\nStage 2 Classification:\n")
                    f.write(f"  Top-{top_k} Predictions:\n")
                    for j, (idx, score, label) in enumerate(zip(top_indices, top_scores, top_labels_list)):
                        marker = "✓" if idx == true_idx else "→" if j == 0 else " "
                        f.write(f"    {marker} {label}: {score:.4f}\n")
                    f.write("-" * 80 + "\n")
                
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

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
    Visualize the full pipeline: input spikes → coarse → prompt (CLIP features) → refined images.
    
    Includes:
    - Stage 1: Coarse reconstruction images
    - Stage 2: CLIP features, similarity scores, and classification predictions
    - Stage 3: Refined images
    
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
                top_labels = [labels[idx] for idx in top_indices]
                
                # Get ground truth index
                true_idx = label_indices[i].item()
                true_label = labels[true_idx]
                true_score = similarities[true_idx]
                pred_idx = top_indices[0]
                pred_label = top_labels[0]
                pred_score = top_scores[0]
                
                # Create visualization with Stage 2
                fig = plt.figure(figsize=(16, 10))
                gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
                
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
                
                # Difference
                ax4 = fig.add_subplot(gs[0, 3])
                diff = np.abs(refined_img - coarse_img)
                im = ax4.imshow(diff)
                ax4.set_title('Difference (Refined - Coarse)')
                ax4.axis('off')
                plt.colorbar(im, ax=ax4)
                
                # Row 2: Stage 2 - Top-k predictions
                ax5 = fig.add_subplot(gs[1, :2])
                colors = ['green' if idx == true_idx else 'red' if idx == pred_idx else 'gray' 
                         for idx in top_indices]
                bars = ax5.barh(range(top_k), top_scores, color=colors)
                ax5.set_yticks(range(top_k))
                ax5.set_yticklabels(top_labels)
                ax5.set_xlabel('Similarity Score')
                ax5.set_title(f'Stage 2: Top-{top_k} Predictions\nPred: {pred_label} ({pred_score:.3f}) | True: {true_label} ({true_score:.3f})')
                ax5.grid(axis='x', alpha=0.3)
                # Add score labels on bars
                for j, (bar, score) in enumerate(zip(bars, top_scores)):
                    ax5.text(score + 0.01, j, f'{score:.3f}', va='center')
                
                # Row 2: Stage 2 - All similarities (sorted)
                ax6 = fig.add_subplot(gs[1, 2:])
                sorted_indices = np.argsort(similarities)[::-1]
                sorted_scores = similarities[sorted_indices]
                sorted_labels_short = [labels[idx][:10] for idx in sorted_indices[:20]]  # Show top 20
                colors_all = ['green' if idx == true_idx else 'red' if idx == pred_idx else 'lightblue' 
                             for idx in sorted_indices[:20]]
                ax6.bar(range(len(sorted_labels_short)), sorted_scores[:20], color=colors_all)
                ax6.set_xticks(range(len(sorted_labels_short)))
                ax6.set_xticklabels(sorted_labels_short, rotation=45, ha='right', fontsize=8)
                ax6.set_ylabel('Similarity Score')
                ax6.set_title(f'Stage 2: All Class Similarities (Top 20)\n{"✓ Correct" if pred_idx == true_idx else "✗ Wrong"}')
                ax6.grid(axis='y', alpha=0.3)
                
                # Row 3: Stage 2 - Similarity distribution
                ax7 = fig.add_subplot(gs[2, :2])
                ax7.hist(similarities, bins=30, edgecolor='black', alpha=0.7)
                ax7.axvline(true_score, color='green', linestyle='--', linewidth=2, label=f'True ({true_score:.3f})')
                ax7.axvline(pred_score, color='red', linestyle='--', linewidth=2, label=f'Pred ({pred_score:.3f})')
                ax7.set_xlabel('Similarity Score')
                ax7.set_ylabel('Frequency')
                ax7.set_title('Stage 2: Similarity Score Distribution')
                ax7.legend()
                ax7.grid(alpha=0.3)
                
                # Row 3: Classification info
                ax8 = fig.add_subplot(gs[2, 2:])
                ax8.axis('off')
                info_text = f"""
Stage 2: Prompt Learning Results

Image Features: {clip_features[i].shape}
Text Features: {all_text_features.shape}

Classification:
  Predicted: {pred_label} (score: {pred_score:.4f})
  Ground Truth: {true_label} (score: {true_score:.4f})
  Status: {'✓ CORRECT' if pred_idx == true_idx else '✗ WRONG'}
  
Top-3 Predictions:
  1. {top_labels[0]}: {top_scores[0]:.4f}
  2. {top_labels[1]}: {top_scores[1]:.4f}
  3. {top_labels[2]}: {top_scores[2]:.4f}
  
Similarity Statistics:
  Mean: {similarities.mean():.4f}
  Std: {similarities.std():.4f}
  Max: {similarities.max():.4f}
  Min: {similarities.min():.4f}
                """
                ax8.text(0.1, 0.5, info_text, fontsize=10, family='monospace', 
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                save_path = output_dir / f'sample_{samples_visualized:03d}_pipeline.png'
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

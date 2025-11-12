"""metrics-only (no plots)"""
import sys
from pathlib import Path

# Add parent directory to path for imports (if run from subdirectory)
if Path(__file__).parent.name != 'src':
    sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import time
from tqdm import tqdm
import json

from config_loader import load_config
from data_loader import get_loader
from metrics import compute_psnr, compute_ssim, compute_l1_error, compute_l2_error
from utils.helpers import get_device, set_seed
from utils.model_loading import load_combined_model

def main():
    parser = argparse.ArgumentParser(description='Evaluate combined model (metrics only)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to combined_model.pth')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='outputs/logs', help='Output directory')
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
    test_loader = get_loader(data_dir, labels, split='test', batch_size=args.batch_size, 
                             num_workers=num_workers, device=device, pin_memory=pin_memory)
    
    # Compute metrics
    print("Computing metrics...")
    psnr_values = []
    ssim_values = []
    l1_errors = []
    l2_errors = []
    eval_start = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            spikes = batch[0].to(device)
            label_indices = batch[2].to(device) if len(batch) > 2 else None
            
            # Forward pass
            refined_images, clip_features, coarse_images = model(spikes, label_indices)
            
            # Reconstruction metrics
            # According to paper:
            # - Stage 1: Maps spikes to TFI result (evaluated separately)
            # - Stage 3: Refines coarse images (compare refined vs coarse)
            # For unpaired training: compare refined vs coarse (refinement quality)
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
    
    eval_time = time.time() - eval_start
    
    # Compute averages
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    avg_l1 = sum(l1_errors) / len(l1_errors) if l1_errors else 0.0
    avg_l2 = sum(l2_errors) / len(l2_errors) if l2_errors else 0.0
    
    # Prepare metrics
    total_samples = len(psnr_values)
    metrics_dict = {
        'timestamp': time.ctime(),
        'device': str(device),
        'dataset': data_dir,
        'checkpoint': args.checkpoint,
        'total_samples': total_samples,
        'evaluation_time': eval_time,
        'reconstruction_metrics': {
            'avg_psnr': float(avg_psnr),
            'avg_ssim': float(avg_ssim),
            'avg_l1_error': float(avg_l1),
            'avg_l2_error': float(avg_l2)
        }
    }
    
    # Print metrics
    print(f"\n===== EVALUATION RESULTS =====")
    print(f"Reconstruction Metrics (Refined vs Coarse):")
    print(f"  PSNR: {avg_psnr:.4f} dB (higher is better)")
    print(f"  SSIM: {avg_ssim:.4f} (higher is better)")
    print(f"  L1 Error: {avg_l1:.4f} (lower is better)")
    print(f"  L2 Error: {avg_l2:.4f} (lower is better)")
    print(f"  Note: According to paper, Stage 3 refines coarse images.")
    print(f"        Metrics compare refined images to coarse images (refinement quality).")
    print(f"\nEvaluation Time: {eval_time:.2f}s")
    
    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / 'evaluation_metrics.json'
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_file}")

if __name__ == '__main__':
    main()

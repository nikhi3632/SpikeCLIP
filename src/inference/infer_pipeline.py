"""runs combined_model.pth end-to-end"""
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import json

from config_loader import load_config
from data_loader import get_loader
from metrics import compute_psnr, compute_ssim, compute_l1_error, compute_l2_error
from utils.helpers import get_device, set_seed
from utils.performance import get_gpu_metrics, reset_memory_stats, compute_latency, compute_throughput
from utils.model_loading import load_combined_model

def main():
    parser = argparse.ArgumentParser(description='End-to-end inference pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to combined_model.pth')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='outputs/logs', help='Output directory for metrics')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    data_dir = config['data_dir']
    labels = config['labels']
    
    # Parameters
    batch_size = args.batch_size
    output_dir = args.output_dir
    
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
    print(f"Loaded combined model with metadata: {checkpoint.get('metadata', {})}")
    
    # Test loader
    test_loader = get_loader(data_dir, labels, split='test', batch_size=batch_size, 
                             num_workers=num_workers, device=device, pin_memory=pin_memory)
    
    # Reset GPU stats
    if device.type == 'cuda':
        reset_memory_stats()
    
    # Inference metrics
    latencies = []
    total_samples = 0
    inference_start = time.time()
    
    # Reconstruction metrics
    psnr_values = []
    ssim_values = []
    l1_errors = []
    l2_errors = []
    
    # Classification metrics
    correct_predictions = 0
    total_predictions = 0
    
    print("Running end-to-end inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            spikes = batch[0].to(device)  # [B, T, H, W]
            label_indices = batch[2].to(device)  # [B]
            batch_size = spikes.size(0)
            
            # Track latency
            if device.type == 'cuda':
                torch.cuda.synchronize()
                batch_start = time.time()
            
            # Forward pass through all stages
            refined_images, clip_features, coarse_images = model(spikes, label_indices)
            
            # Track latency
            if device.type == 'cuda':
                torch.cuda.synchronize()
                batch_time = time.time() - batch_start
                latencies.append(batch_time)
            
            # Classification (using already computed clip_features)
            image_features = clip_features  # Already computed from forward pass
            all_label_indices = torch.arange(len(labels), device=device)
            all_text_features = model.prompt_model.get_text_embeddings(all_label_indices)
            similarities = torch.matmul(image_features, all_text_features.t())
            predictions = similarities.argmax(dim=1)
            correct_predictions += (predictions == label_indices).sum().item()
            total_predictions += predictions.size(0)
            
            # Reconstruction metrics (using coarse as target)
            target = coarse_images
            for i in range(batch_size):
                pred_img = refined_images[i:i+1]
                target_img = target[i:i+1]
                
                psnr = compute_psnr(pred_img, target_img)
                ssim = compute_ssim(pred_img, target_img)
                l1 = compute_l1_error(pred_img, target_img)
                l2 = compute_l2_error(pred_img, target_img)
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                l1_errors.append(l1)
                l2_errors.append(l2)
            
            total_samples += batch_size
    
    # Compute total inference time
    inference_time = time.time() - inference_start
    
    # Compute GPU metrics
    if device.type == 'cuda':
        gpu_metrics = get_gpu_metrics(latencies, total_samples, inference_time)
    else:
        avg_latency, std_latency = compute_latency(latencies) if latencies else (0.0, 0.0)
        throughput = compute_throughput(total_samples, inference_time)
        gpu_metrics = {
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'throughput': throughput,
            'memory_usage_mb': 0.0,
            'power_usage_w': 0.0
        }
    
    # Compute average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    avg_l1 = sum(l1_errors) / len(l1_errors) if l1_errors else 0.0
    avg_l2 = sum(l2_errors) / len(l2_errors) if l2_errors else 0.0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Prepare metrics text
    metrics_text = f"""===== END-TO-END PIPELINE INFERENCE METRICS =====
Timestamp: {time.ctime()}
Device: {device}
Dataset: {data_dir}
Checkpoint: {args.checkpoint}
Batch Size: {batch_size}
Total Samples: {total_samples}

--- Reconstruction Metrics (Refined vs Coarse) ---
Average PSNR: {avg_psnr:.4f} dB
Average SSIM: {avg_ssim:.4f}
Average L1 Error: {avg_l1:.4f}
Average L2 Error (MSE): {avg_l2:.4f}

--- Classification Metrics ---
Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})

--- GPU Performance Metrics ---
Average Latency per batch: {gpu_metrics['avg_latency']:.4f}s ± {gpu_metrics['std_latency']:.4f}s
Throughput: {gpu_metrics['throughput']:.2f} samples/sec
GPU Memory Usage: {gpu_metrics['memory_usage_mb']:.2f} MB
Total Inference Time: {inference_time:.2f}s
=========================================================
"""
    
    # Print metrics
    print(metrics_text)
    
    # Save metrics to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / 'pipeline_inference_metrics.txt'
    
    with open(metrics_file, 'w') as f:
        f.write(metrics_text)
    
    print(f"Inference metrics saved to {metrics_file}")
    
    # Also save as JSON
    metrics_dict = {
        'timestamp': time.ctime(),
        'device': str(device),
        'dataset': data_dir,
        'checkpoint': args.checkpoint,
        'batch_size': batch_size,
        'total_samples': total_samples,
        'reconstruction_metrics': {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_l1_error': avg_l1,
            'avg_l2_error': avg_l2
        },
        'classification_metrics': {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions
        },
        'gpu_metrics': gpu_metrics,
        'inference_time': inference_time
    }
    
    json_file = output_dir / 'pipeline_inference_metrics.json'
    with open(json_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Inference metrics (JSON) saved to {json_file}")

if __name__ == '__main__':
    main()

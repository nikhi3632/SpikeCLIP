"""Stage 2 inference"""
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import json

from config_loader import load_config
from data_loader import get_loader
from models.coarse_reconstruction import CoarseSNN
from models.prompt_learning import PromptAdapter
from utils.helpers import get_device, set_seed
from utils.checkpointing import load_best_checkpoint
from utils.performance import get_gpu_metrics, reset_memory_stats, compute_latency, compute_throughput

def main():
    parser = argparse.ArgumentParser(description='Inference Stage 2: Prompt Learning')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--coarse-checkpoint-dir', type=str, required=True, help='Coarse checkpoint directory')
    parser.add_argument('--prompt-checkpoint-dir', type=str, required=True, help='Prompt checkpoint directory')
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
    
    # Prompt stage config (with defaults)
    prompt_config = config.get('prompt', {})
    model_config = prompt_config.get('model', {})
    output_config = prompt_config.get('output', {})
    coarse_config = config.get('coarse', {})
    
    # Parameters (command line args override config)
    batch_size = prompt_config.get('training', {}).get('batch_size', 8) if args.batch_size == 8 else args.batch_size
    seed = prompt_config.get('training', {}).get('seed', 42) if args.seed == 42 else args.seed
    output_dir = output_config.get('log_dir', 'outputs/logs') if args.output_dir == 'outputs/logs' else args.output_dir
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data loader config
    data_loader_config = config.get('data_loader', {})
    num_workers = data_loader_config.get('num_workers', None)
    pin_memory = data_loader_config.get('pin_memory', None)
    
    # Test loader
    test_loader = get_loader(data_dir, labels, split='test', batch_size=batch_size, 
                             num_workers=num_workers, device=device, pin_memory=pin_memory)
    
    # Stage 1 model (frozen)
    coarse_model = CoarseSNN(
        time_steps=coarse_config.get('model', {}).get('time_steps', 25),
        v_threshold=coarse_config.get('model', {}).get('v_threshold', 1.0),
        tau=coarse_config.get('model', {}).get('tau', 2.0)
    )
    
    # Load coarse checkpoint
    coarse_checkpoint_path = Path(args.coarse_checkpoint_dir)
    print(f"Loading coarse model from {coarse_checkpoint_path}")
    load_best_checkpoint(
        str(coarse_checkpoint_path),
        coarse_model,
        device=device,
        prefix='coarse'
    )
    coarse_model.eval()
    
    # Stage 2 model
    num_classes = model_config.get('num_classes', len(labels))
    if num_classes == 101:  # Default placeholder, use actual labels
        num_classes = len(labels)
    
    prompt_model = PromptAdapter(
        clip_dim=model_config.get('clip_dim', 512),
        num_classes=num_classes,
        prompt_dim=model_config.get('prompt_dim', 77),
        freeze_image_encoder=model_config.get('freeze_image_encoder', True)
    )
    
    # Load prompt checkpoint
    prompt_checkpoint_path = Path(args.prompt_checkpoint_dir)
    print(f"Loading prompt model from {prompt_checkpoint_path}")
    checkpoint = load_best_checkpoint(
        str(prompt_checkpoint_path),
        prompt_model,
        device=device,
        prefix='prompt'
    )
    prompt_model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Reset GPU stats
    if device.type == 'cuda':
        reset_memory_stats()
    
    # Inference metrics
    latencies = []
    total_samples = 0
    inference_start = time.time()
    
    # Accuracy tracking
    correct_predictions = 0
    total_predictions = 0
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            spikes = batch[0].to(device)  # [B, T, H, W]
            label_indices = batch[2].to(device)  # [B]
            batch_size = spikes.size(0)
            
            # Track latency
            if device.type == 'cuda':
                torch.cuda.synchronize()
                batch_start = time.time()
            
            # Stage 1: Get coarse images
            coarse_images = coarse_model(spikes)  # [B, 3, H, W]
            
            # Stage 2: Get CLIP features
            image_features = prompt_model.get_clip_features(coarse_images)  # [B, clip_dim]
            
            # Get all text embeddings for all classes
            all_label_indices = torch.arange(len(labels), device=device)  # [num_classes]
            all_text_features = prompt_model.get_text_embeddings(all_label_indices)  # [num_classes, clip_dim]
            
            # Compute similarity: each image with all class prompts
            similarities = torch.matmul(image_features, all_text_features.t())  # [B, num_classes]
            
            # Get predictions (class with highest similarity)
            predictions = similarities.argmax(dim=1)  # [B]
            correct_predictions += (predictions == label_indices).sum().item()
            total_predictions += predictions.size(0)
            
            # Track latency
            if device.type == 'cuda':
                torch.cuda.synchronize()
                batch_time = time.time() - batch_start
                latencies.append(batch_time)
            
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
    
    # Compute accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Prepare metrics text
    metrics_text = f"""===== PROMPT LEARNING INFERENCE METRICS =====
Timestamp: {time.ctime()}
Device: {device}
Dataset: {data_dir}
Coarse Checkpoint: {coarse_checkpoint_path / 'coarse_best.pth'}
Prompt Checkpoint: {prompt_checkpoint_path / 'prompt_best.pth'}
Batch Size: {batch_size}
Total Samples: {total_samples}

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
    metrics_file = output_dir / 'prompt_inference_metrics.txt'
    
    with open(metrics_file, 'w') as f:
        f.write(metrics_text)
    
    print(f"Inference metrics saved to {metrics_file}")
    
    # Also save as JSON
    metrics_dict = {
        'timestamp': time.ctime(),
        'device': str(device),
        'dataset': data_dir,
        'coarse_checkpoint': str(coarse_checkpoint_path / 'coarse_best.pth'),
        'prompt_checkpoint': str(prompt_checkpoint_path / 'prompt_best.pth'),
        'batch_size': batch_size,
        'total_samples': total_samples,
        'classification_metrics': {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions
        },
        'gpu_metrics': gpu_metrics,
        'inference_time': inference_time
    }
    
    json_file = output_dir / 'prompt_inference_metrics.json'
    with open(json_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Inference metrics (JSON) saved to {json_file}")

if __name__ == '__main__':
    main()

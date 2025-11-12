"""Logging utilities for device info, training, inference, and test logs"""
import torch
import json
import time
from pathlib import Path
from typing import Dict, Optional

def log_device_info(output_dir: str = "outputs/logs") -> Dict:
    """
    Log device information to a file.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'timestamp': time.ctime(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }
    
    # Get CUDA device info if available
    if torch.cuda.is_available():
        device_info['cuda_devices'] = []
        for i in range(torch.cuda.device_count()):
            device_info['cuda_devices'].append({
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / (1024 ** 2),
                'memory_allocated_mb': torch.cuda.memory_allocated(i) / (1024 ** 2) if torch.cuda.is_available() else 0,
                'memory_reserved_mb': torch.cuda.memory_reserved(i) / (1024 ** 2) if torch.cuda.is_available() else 0,
            })
    
    # Save to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as text
    text_file = output_dir / 'device_info.txt'
    with open(text_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DEVICE INFORMATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {device_info['timestamp']}\n")
        f.write(f"PyTorch Version: {device_info['pytorch_version']}\n")
        f.write(f"CUDA Available: {device_info['cuda_available']}\n")
        if device_info['cuda_available']:
            f.write(f"CUDA Version: {device_info['cuda_version']}\n")
            f.write(f"cuDNN Version: {device_info['cudnn_version']}\n")
            f.write(f"Number of CUDA Devices: {device_info['device_count']}\n")
            f.write("\n" + "-" * 60 + "\n")
            f.write("CUDA Devices:\n")
            f.write("-" * 60 + "\n")
            for device in device_info['cuda_devices']:
                f.write(f"Device {device['index']}: {device['name']}\n")
                f.write(f"  Total Memory: {device['memory_total_mb']:.2f} MB\n")
                f.write(f"  Allocated Memory: {device['memory_allocated_mb']:.2f} MB\n")
                f.write(f"  Reserved Memory: {device['memory_reserved_mb']:.2f} MB\n")
        f.write(f"MPS Available: {device_info['mps_available']}\n")
        f.write("=" * 60 + "\n")
    
    # Save as JSON
    json_file = output_dir / 'device_info.json'
    with open(json_file, 'w') as f:
        json.dump(device_info, f, indent=2)
    
    print(f"Device info saved to {text_file} and {json_file}")
    
    return device_info

def log_training_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    gpu_metrics: Optional[Dict] = None,
    output_dir: str = "outputs/logs",
    stage: str = "coarse"
) -> None:
    """Log training metrics to file. Overwrites on epoch 0 (fresh log per run), appends for subsequent epochs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f'{stage}_train_log.txt'
    
    # Format log entry
    log_entry = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {learning_rate:.6f}"
    if gpu_metrics:
        log_entry += f" | GPU Latency: {gpu_metrics.get('avg_latency', 0):.4f}s±{gpu_metrics.get('std_latency', 0):.4f}s"
        log_entry += f" | Throughput: {gpu_metrics.get('throughput', 0):.2f} samples/s"
        log_entry += f" | Memory: {gpu_metrics.get('memory_usage_mb', 0):.2f} MB"
        if gpu_metrics.get('power_usage_w', 0) > 0:
            log_entry += f" | Power: {gpu_metrics.get('power_usage_w', 0):.2f} W"
    log_entry += "\n"
    
    # Overwrite on first epoch (epoch 0), append for subsequent epochs
    # This ensures each training run starts with a fresh log file
    mode = 'w' if epoch == 0 else 'a'
    with open(log_file, mode) as f:
        f.write(log_entry)
    
    # Also save as JSON (overwrite on first epoch, append for subsequent epochs)
    json_file = output_dir / f'{stage}_train_log.json'
    if epoch == 0:
        # Start fresh log on epoch 0
        logs = {'entries': []}
    else:
        # Load existing log and append
        if json_file.exists():
            with open(json_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = {'entries': []}
    
    entry = {
        'epoch': epoch,
        'timestamp': time.ctime(),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'learning_rate': float(learning_rate),
        'gpu_metrics': gpu_metrics or {}
    }
    logs['entries'].append(entry)
    
    with open(json_file, 'w') as f:
        json.dump(logs, f, indent=2)

def log_inference_metrics(
    metrics_dict: Dict,
    output_dir: str = "outputs/logs",
    stage: str = "pipeline"
) -> None:
    """Save inference metrics to log file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_file = output_dir / f'{stage}_inference_metrics.json'
    with open(json_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Inference metrics saved to {json_file}")

def log_test_metrics(
    metrics_dict: Dict,
    output_dir: str = "outputs/logs"
) -> None:
    """Save test metrics to log file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as text
    text_file = output_dir / 'test_log.txt'
    with open(text_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {metrics_dict.get('timestamp', time.ctime())}\n")
        f.write(f"Device: {metrics_dict.get('device', 'unknown')}\n")
        f.write(f"Checkpoint: {metrics_dict.get('checkpoint', 'unknown')}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("Reconstruction Metrics:\n")
        f.write("-" * 60 + "\n")
        recon = metrics_dict.get('reconstruction_metrics', {})
        f.write(f"  PSNR: {recon.get('avg_psnr', 0):.4f} dB\n")
        f.write(f"  SSIM: {recon.get('avg_ssim', 0):.4f}\n")
        f.write(f"  L1 Error: {recon.get('avg_l1_error', 0):.4f}\n")
        f.write(f"  L2 Error: {recon.get('avg_l2_error', 0):.4f}\n")
        f.write("\n" + "-" * 60 + "\n")

    # Save as JSON
    json_file = output_dir / 'test_log.json'
    with open(json_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Test metrics saved to {text_file} and {json_file}")


"""plots metrics from logs"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(metrics_file: str) -> dict:
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_training_metrics(log_dir: str, output_dir: str):
    """Plot training metrics from log files."""
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load metrics from JSON files
    metrics_files = {
        'coarse': log_dir / 'coarse_inference_metrics.json',
        'prompt': log_dir / 'prompt_inference_metrics.json',
        'refine': log_dir / 'refine_inference_metrics.json',
        'pipeline': log_dir / 'pipeline_inference_metrics.json'
    }
    
    available_metrics = {}
    for stage, file_path in metrics_files.items():
        if file_path.exists():
            try:
                available_metrics[stage] = load_metrics(str(file_path))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    if not available_metrics:
        print("No metrics files found!")
        return
    
    # Plot reconstruction metrics
    stages = list(available_metrics.keys())
    if stages:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR
        psnr_values = [available_metrics[s].get('reconstruction_metrics', {}).get('avg_psnr', 0) for s in stages]
        axes[0, 0].bar(stages, psnr_values)
        axes[0, 0].set_title('PSNR (dB)')
        axes[0, 0].set_ylabel('PSNR')
        axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM
        ssim_values = [available_metrics[s].get('reconstruction_metrics', {}).get('avg_ssim', 0) for s in stages]
        axes[0, 1].bar(stages, ssim_values)
        axes[0, 1].set_title('SSIM')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].grid(True, alpha=0.3)
        
        # L1 Error
        l1_values = [available_metrics[s].get('reconstruction_metrics', {}).get('avg_l1_error', 0) for s in stages]
        axes[1, 0].bar(stages, l1_values)
        axes[1, 0].set_title('L1 Error')
        axes[1, 0].set_ylabel('L1 Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # L2 Error
        l2_values = [available_metrics[s].get('reconstruction_metrics', {}).get('avg_l2_error', 0) for s in stages]
        axes[1, 1].bar(stages, l2_values)
        axes[1, 1].set_title('L2 Error (MSE)')
        axes[1, 1].set_ylabel('L2 Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / 'reconstruction_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction metrics plot to {save_path}")
        
        # Plot classification metrics if available
        if any('classification_metrics' in available_metrics[s] for s in stages):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            accuracies = [available_metrics[s].get('classification_metrics', {}).get('accuracy', 0) for s in stages if 'classification_metrics' in available_metrics[s]]
            stages_with_acc = [s for s in stages if 'classification_metrics' in available_metrics[s]]
            if accuracies:
                ax.bar(stages_with_acc, accuracies)
                ax.set_title('Classification Accuracy')
                ax.set_ylabel('Accuracy')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save_path = output_dir / 'classification_metrics.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved classification metrics plot to {save_path}")
        
        # Plot GPU metrics if available
        if any('gpu_metrics' in available_metrics[s] for s in stages):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            stages_with_gpu = [s for s in stages if 'gpu_metrics' in available_metrics[s]]
            
            # Latency
            latencies = [available_metrics[s]['gpu_metrics'].get('avg_latency', 0) for s in stages_with_gpu]
            axes[0, 0].bar(stages_with_gpu, latencies)
            axes[0, 0].set_title('Average Latency (s)')
            axes[0, 0].set_ylabel('Latency (s)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Throughput
            throughputs = [available_metrics[s]['gpu_metrics'].get('throughput', 0) for s in stages_with_gpu]
            axes[0, 1].bar(stages_with_gpu, throughputs)
            axes[0, 1].set_title('Throughput (samples/s)')
            axes[0, 1].set_ylabel('Throughput')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Memory
            memories = [available_metrics[s]['gpu_metrics'].get('memory_usage_mb', 0) for s in stages_with_gpu]
            axes[1, 0].bar(stages_with_gpu, memories)
            axes[1, 0].set_title('GPU Memory Usage (MB)')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Inference time
            times = [available_metrics[s].get('inference_time', 0) for s in stages_with_gpu]
            axes[1, 1].bar(stages_with_gpu, times)
            axes[1, 1].set_title('Total Inference Time (s)')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = output_dir / 'gpu_metrics.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved GPU metrics plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from log files')
    parser.add_argument('--log-dir', type=str, default='outputs/logs', help='Directory with metrics JSON files')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations', help='Output directory for plots')
    
    args = parser.parse_args()
    
    plot_training_metrics(args.log_dir, args.output_dir)
    print("Metrics plotting complete!")

if __name__ == '__main__':
    main()

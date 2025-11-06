"""GPU metrics (latency, throughput, power)"""
import torch
from typing import List, Tuple

# Try to import pynvml for GPU power monitoring
# nvidia-ml-py package (installed via pip) provides pynvml module directly
PYNVML_AVAILABLE = False
pynvml = None

try:
    # nvidia-ml-py provides pynvml module directly
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    # pynvml not available - nvidia-ml-py may not be installed
    PYNVML_AVAILABLE = False
    pynvml = None

# Initialize NVML if available (only once)
_nvml_initialized = False

def _init_nvml():
    """Initialize NVML if available."""
    global _nvml_initialized
    if not _nvml_initialized and PYNVML_AVAILABLE and torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            _nvml_initialized = True
        except Exception:
            pass

def compute_latency(latencies: List[float]) -> Tuple[float, float]:
    """Compute average and standard deviation of latencies."""
    if not latencies:
        return 0.0, 0.0
    mean = sum(latencies) / len(latencies)
    variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
    std = variance ** 0.5
    return mean, std

def compute_throughput(total_samples: int, total_time: float) -> float:
    """Compute throughput in samples per second."""
    if total_time == 0:
        return 0.0
    return total_samples / total_time

def compute_memory_usage() -> float:
    """Compute GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    return 0.0

def compute_power_usage() -> float:
    """
    Compute current GPU power usage in watts.
    
    Returns:
        Power usage in watts, or 0.0 if not available or if pynvml is not installed.
    """
    if not torch.cuda.is_available():
        return 0.0
    
    if not PYNVML_AVAILABLE:
        return 0.0
    
    try:
        # Initialize NVML if not already done
        _init_nvml()
        
        if not _nvml_initialized:
            return 0.0
        
        # Get the current CUDA device
        device_id = torch.cuda.current_device()
        
        # Get handle to the device
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        # Get power usage in milliwatts, convert to watts
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_w = power_mw / 1000.0
        
        return power_w
    
    except Exception:
        # If any error occurs (device not supported, NVML error, etc.), return 0
        return 0.0

def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_gpu_metrics(
    latencies: List[float],
    total_samples: int,
    total_time: float
) -> dict:
    """Get comprehensive GPU metrics."""
    avg_latency, std_latency = compute_latency(latencies)
    throughput = compute_throughput(total_samples, total_time)
    memory_usage = compute_memory_usage()
    power_usage = compute_power_usage()
    
    return {
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        'throughput': throughput,
        'memory_usage_mb': memory_usage,
        'power_usage_w': power_usage
    }

"""High-Quality image generation pipeline for Stage 2 according to the paper.

According to the paper:
- For synthetic data: Uses WGSE (Weighted Gradient Spike Encoding) or similar methods
- For real data: Uses mixture of multiple reconstruction methods
- The goal is to generate high-quality images from spikes that are better than
  the coarse reconstruction from Stage 1, to serve as positive examples for
  HQ/LQ prompt learning.
"""
import torch
from utils.tfi import calculate_tfi_vectorized


def calculate_temporal_average(spikes: torch.Tensor) -> torch.Tensor:
    """
    Calculate temporal average of spikes (TFP - Temporal Frame Projection).
    
    Args:
        spikes: [B, T, H, W] spike stream
    
    Returns:
        avg: [B, 1, H, W] temporal average image
    """
    # Average over time dimension
    avg = spikes.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    # Normalize to [0, 1]
    avg = torch.clamp(avg, 0, 1)
    return avg


def calculate_spike_count(spikes: torch.Tensor) -> torch.Tensor:
    """
    Calculate spike count image (number of spikes per pixel).
    
    Args:
        spikes: [B, T, H, W] spike stream
    
    Returns:
        count: [B, 1, H, W] spike count image
    """
    # Count spikes over time dimension
    count = spikes.sum(dim=1, keepdim=True)  # [B, 1, H, W]
    # Normalize to [0, 1] per sample
    B = count.shape[0]
    for b in range(B):
        sample = count[b, 0]  # [H, W]
        if sample.max() > sample.min():
            sample_min = sample.min()
            sample_max = sample.max()
            count[b, 0] = (sample - sample_min) / (sample_max - sample.min() + 1e-8)
    count = torch.clamp(count, 0, 1)
    return count


def calculate_weighted_temporal_average(spikes: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate weighted temporal average of spikes.
    
    According to the paper, WGSE (Weighted Gradient Spike Encoding) uses
    weighted temporal averaging where later spikes have higher weights.
    
    Args:
        spikes: [B, T, H, W] spike stream
        weights: [T] optional weights for each time step (default: exponential)
    
    Returns:
        weighted_avg: [B, 1, H, W] weighted temporal average image
    """
    B, T, H, W = spikes.shape
    
    if weights is None:
        # Exponential weights: later spikes have higher weights
        # This mimics WGSE behavior where recent information is more important
        weights = torch.exp(torch.linspace(0, 2, T, device=spikes.device))  # [T]
        weights = weights / weights.sum()  # Normalize
    
    # Apply weights: [B, T, H, W] * [T] -> [B, T, H, W]
    weights_expanded = weights.view(1, T, 1, 1)  # [1, T, 1, 1]
    weighted_spikes = spikes * weights_expanded  # [B, T, H, W]
    
    # Sum over time dimension
    weighted_avg = weighted_spikes.sum(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Normalize to [0, 1] per sample
    for b in range(B):
        sample = weighted_avg[b, 0]  # [H, W]
        if sample.max() > sample.min():
            sample_min = sample.min()
            sample_max = sample.max()
            weighted_avg[b, 0] = (sample - sample_min) / (sample_max - sample.min() + 1e-8)
    weighted_avg = torch.clamp(weighted_avg, 0, 1)
    
    return weighted_avg


def generate_hq_images(spikes: torch.Tensor, method: str = "mixture") -> torch.Tensor:
    """
    Generate high-quality images from spikes according to the paper.
    
    According to the paper:
    - For synthetic data: Uses WGSE (Weighted Gradient Spike Encoding)
    - For real data: Uses mixture of multiple reconstruction methods
    
    This function implements a mixture approach that combines:
    1. TFI (Texture from ISI) - captures texture information
    2. Weighted Temporal Average (WGSE-like) - captures temporal information
    3. Spike Count - captures intensity information
    
    Args:
        spikes: [B, T, H, W] spike stream
        method: "mixture" (default) or "tfi" or "wgse" or "weighted_avg"
    
    Returns:
        hq_images: [B, 3, H, W] high-quality images
    """
    if method == "tfi":
        # Use only TFI (simplified, for backward compatibility)
        tfi = calculate_tfi_vectorized(spikes, threshold=1.0)  # [B, 1, H, W]
        hq_images = tfi.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        return hq_images
    
    elif method == "wgse" or method == "weighted_avg":
        # Use weighted temporal average (WGSE-like)
        weighted_avg = calculate_weighted_temporal_average(spikes)  # [B, 1, H, W]
        hq_images = weighted_avg.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        return hq_images
    
    elif method == "mixture":
        # Mixture approach: combine multiple methods
        # This is closer to what the paper describes for real data
        
        # 1. TFI (Texture from ISI) - captures texture
        tfi = calculate_tfi_vectorized(spikes, threshold=1.0)  # [B, 1, H, W]
        
        # 2. Weighted Temporal Average (WGSE-like) - captures temporal structure
        weighted_avg = calculate_weighted_temporal_average(spikes)  # [B, 1, H, W]
        
        # 3. Spike Count - captures intensity
        spike_count = calculate_spike_count(spikes)  # [B, 1, H, W]
        
        # Combine methods with learned weights (can be tuned)
        # According to the paper, mixture should emphasize different aspects
        # We use: 0.4 * TFI + 0.4 * WeightedAvg + 0.2 * SpikeCount
        # These weights can be adjusted based on dataset characteristics
        mixture = 0.4 * tfi + 0.4 * weighted_avg + 0.2 * spike_count  # [B, 1, H, W]
        
        # Normalize to [0, 1]
        mixture = torch.clamp(mixture, 0, 1)
        
        # Convert to RGB
        hq_images = mixture.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        return hq_images
    
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'mixture', 'tfi', 'wgse', or 'weighted_avg'")


def generate_hq_images_advanced(spikes: torch.Tensor, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2) -> torch.Tensor:
    """
    Advanced HQ image generation with tunable mixture weights.
    
    Args:
        spikes: [B, T, H, W] spike stream
        alpha: Weight for TFI (default: 0.4)
        beta: Weight for weighted temporal average (default: 0.4)
        gamma: Weight for spike count (default: 0.2)
    
    Returns:
        hq_images: [B, 3, H, W] high-quality images
    """
    # Normalize weights
    total = alpha + beta + gamma
    alpha = alpha / total
    beta = beta / total
    gamma = gamma / total
    
    # Calculate components
    tfi = calculate_tfi_vectorized(spikes, threshold=1.0)  # [B, 1, H, W]
    weighted_avg = calculate_weighted_temporal_average(spikes)  # [B, 1, H, W]
    spike_count = calculate_spike_count(spikes)  # [B, 1, H, W]
    
    # Combine with tunable weights
    mixture = alpha * tfi + beta * weighted_avg + gamma * spike_count  # [B, 1, H, W]
    
    # Normalize to [0, 1]
    mixture = torch.clamp(mixture, 0, 1)
    
    # Convert to RGB
    hq_images = mixture.repeat(1, 3, 1, 1)  # [B, 3, H, W]
    
    return hq_images


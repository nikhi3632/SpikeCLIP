"""TFI (Texture from ISI) calculation for spike-to-image reconstruction."""
import torch
import torch.nn.functional as F


def calculate_isi(spikes: torch.Tensor, middle_time: int = None) -> torch.Tensor:
    """
    Calculate Inter-Spike Interval (ISI) for each pixel.
    
    According to the paper:
    ISI_t = t+ - t- (time between spikes before and after middle time)
    If no spikes found, ISI = ∞
    
    Args:
        spikes: [B, T, H, W] spike stream (binary: 0 or 1)
        middle_time: Middle time point (default: T // 2)
    
    Returns:
        isi: [B, H, W] Inter-Spike Interval for each pixel
    """
    B, T, H, W = spikes.shape
    
    if middle_time is None:
        middle_time = T // 2
    
    # Find spike times for each pixel
    # spikes is [B, T, H, W], we need to find t- and t+ for each [B, H, W] pixel
    isi = torch.full((B, H, W), float('inf'), device=spikes.device, dtype=torch.float32)
    
    # For each pixel, find the spike time before (t-) and after (t+) the middle time
    for b in range(B):
        for h in range(H):
            for w in range(W):
                pixel_spikes = spikes[b, :, h, w]  # [T]
                
                # Find t-: last spike before or at middle_time
                t_minus = None
                for t in range(middle_time, -1, -1):
                    if pixel_spikes[t] > 0:
                        t_minus = t
                        break
                
                # Find t+: first spike after middle_time
                t_plus = None
                for t in range(middle_time + 1, T):
                    if pixel_spikes[t] > 0:
                        t_plus = t
                        break
                
                # Calculate ISI if both spikes found
                if t_minus is not None and t_plus is not None:
                    isi[b, h, w] = float(t_plus - t_minus)
    
    return isi


def calculate_tfi(spikes: torch.Tensor, threshold: float = 1.0, middle_time: int = None) -> torch.Tensor:
    """
    Calculate TFI (Texture from ISI) image from spike stream.
    
    According to the paper:
    TFI = Θ / ISI
    where Θ is the threshold and ISI is the Inter-Spike Interval.
    
    Args:
        spikes: [B, T, H, W] spike stream (binary: 0 or 1)
        threshold: Threshold value Θ (default: 1.0)
        middle_time: Middle time point (default: T // 2)
    
    Returns:
        tfi: [B, 1, H, W] TFI image (grayscale)
    """
    isi = calculate_isi(spikes, middle_time=middle_time)  # [B, H, W]
    
    # TFI = Θ / ISI
    # Handle infinity (no spikes): set to 0 or a small value
    tfi = threshold / (isi + 1e-8)  # [B, H, W]
    
    # Set infinite values to 0 (no spikes detected)
    tfi = torch.where(torch.isfinite(tfi), tfi, torch.zeros_like(tfi))
    
    # Normalize to [0, 1] range
    # Per-sample normalization to preserve relative intensities
    tfi = tfi.unsqueeze(1)  # [B, 1, H, W]
    
    # Normalize each sample independently
    for b in range(tfi.shape[0]):
        sample = tfi[b, 0]  # [H, W]
        if sample.max() > sample.min():
            sample_min = sample.min()
            sample_max = sample.max()
            tfi[b, 0] = (sample - sample_min) / (sample_max - sample_min + 1e-8)
    
    tfi = torch.clamp(tfi, 0, 1)
    
    return tfi


def calculate_tfi_vectorized(spikes: torch.Tensor, threshold: float = 1.0, middle_time: int = None) -> torch.Tensor:
    """
    Vectorized version of TFI calculation (more efficient).
    
    Args:
        spikes: [B, T, H, W] spike stream (binary: 0 or 1)
        threshold: Threshold value Θ (default: 1.0)
        middle_time: Middle time point (default: T // 2)
    
    Returns:
        tfi: [B, 1, H, W] TFI image (grayscale)
    """
    B, T, H, W = spikes.shape
    
    if middle_time is None:
        middle_time = T // 2
    
    # Convert spikes to cumulative sum to find spike times
    # spikes is [B, T, H, W]
    # We need to find t- and t+ for each pixel
    
    # Find t-: last spike before or at middle_time
    spikes_before = spikes[:, :middle_time+1, :, :]  # [B, middle_time+1, H, W]
    # Find the last time index where spike occurred for each pixel
    # Use argmax on reversed dimension to find last occurrence
    spikes_before_reversed = torch.flip(spikes_before, dims=[1])  # [B, middle_time+1, H, W]
    t_minus_idx = torch.argmax(spikes_before_reversed, dim=1)  # [B, H, W] - index in reversed array
    # Convert back to original index
    t_minus = middle_time - t_minus_idx  # [B, H, W]
    # Set to -1 if no spike found (all zeros)
    t_minus = torch.where(torch.any(spikes_before > 0, dim=1), t_minus, torch.full_like(t_minus, -1))
    
    # Find t+: first spike after middle_time
    spikes_after = spikes[:, middle_time+1:, :, :]  # [B, T-middle_time-1, H, W]
    t_plus_idx = torch.argmax(spikes_after, dim=1)  # [B, H, W] - finds first occurrence
    # Convert to original index
    t_plus = middle_time + 1 + t_plus_idx  # [B, H, W]
    # Set to T if no spike found (all zeros)
    t_plus = torch.where(torch.any(spikes_after > 0, dim=1), t_plus, torch.full_like(t_plus, T))
    
    # Calculate ISI = t+ - t-
    isi = t_plus.float() - t_minus.float()  # [B, H, W]
    # Set to infinity if no spikes found (t_minus == -1 or t_plus == T)
    isi = torch.where((t_minus >= 0) & (t_plus < T), isi, torch.full_like(isi, float('inf')))
    
    # TFI = Θ / ISI
    tfi = threshold / (isi + 1e-8)  # [B, H, W]
    
    # Set infinite values to 0 (no spikes detected)
    tfi = torch.where(torch.isfinite(tfi), tfi, torch.zeros_like(tfi))
    
    # Normalize to [0, 1] range per sample
    tfi = tfi.unsqueeze(1)  # [B, 1, H, W]
    
    # Normalize each sample independently
    for b in range(B):
        sample = tfi[b, 0]  # [H, W]
        if sample.max() > sample.min():
            sample_min = sample.min()
            sample_max = sample.max()
            tfi[b, 0] = (sample - sample_min) / (sample_max - sample_min + 1e-8)
    
    tfi = torch.clamp(tfi, 0, 1)
    
    return tfi


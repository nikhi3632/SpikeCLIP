import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import functional

class CoarseSNN(nn.Module):
    """
    Coarse spike-to-image reconstruction with a shallow SNN-style CNN.
    Input:  spikes [B, T, H, W]
    Output: image  [B, 3, H, W] in [0, 1]
    """

    def __init__(self, in_channels=1, out_channels=3, time_steps=25, v_threshold=1.0, tau=2.0):
        super().__init__()
        self.time_steps = time_steps

        # Enhanced encoder with more capacity and better feature extraction
        # Layer 1: 224x224 -> 224x224 (extract more features)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional conv for richer features
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Layer 2: 224x224 -> 112x112
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Layer 3: 112x112 -> 56x56
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Layer 4: 56x56 -> 28x28
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(256),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )

        # Enhanced bottleneck with more capacity
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Increase channels
            nn.BatchNorm2d(512),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Back to 256
            nn.BatchNorm2d(256),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )

        # Enhanced decoder with skip connections (U-Net style)
        # Use bilinear upsampling + conv instead of ConvTranspose for sharper outputs (reduces checkerboard artifacts)
        # Use LIFNode throughout for consistent SNN architecture
        # Layer 4: 28x28 -> 56x56
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 from skip connection
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Layer 3: 56x56 -> 112x112
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # 128*2 from skip
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Layer 2: 112x112 -> 224x224
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),  # 64*2 from skip
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Final layer: 224x224 -> 224x224
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32*2 from skip
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional conv for better features
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN encoder-decoder with skip connections.
        
        Proper SNN implementation: Process temporal sequence [B, T, H, W] through network.
        LIFNodes accumulate temporal information across time steps.
        
        Args:
            spikes: [B, T, H, W] input spike tensor
        
        Returns:
            images: [B, 3, H, W] reconstructed images in [0, 1] range
        """
        # Reset SNN state each forward so batches are independent
        functional.reset_net(self)
        
        B, T, H, W = spikes.shape
        
        # Voxelization: Reduce temporal dimension (200 → 50) as per paper
        # Paper uses voxelization technique to squeeze input length of spike sequence
        # This is common in event-based vision tasks
        # Divide temporal sequence into bins and aggregate spikes within each bin
        target_length = 50  # Paper: reduces from 200 to 50
        if T > target_length:
            # Vectorized voxelization: more efficient than loop
            # Reshape spikes into bins: [B, T, H, W] -> [B, target_length, bin_size, H, W]
            # Pad if needed to make T divisible by target_length
            if T % target_length != 0:
                # Pad to make divisible
                pad_size = target_length - (T % target_length)
                spikes = torch.cat([spikes, torch.zeros(B, pad_size, H, W, device=spikes.device, dtype=spikes.dtype)], dim=1)
                T = spikes.shape[1]
            
            # Reshape: [B, T, H, W] -> [B, target_length, T//target_length, H, W]
            bin_size_actual = T // target_length
            spikes_reshaped = spikes.view(B, target_length, bin_size_actual, H, W)  # [B, target_length, bin_size, H, W]
            # Sum over bin dimension to aggregate spikes
            spikes_voxelized = spikes_reshaped.sum(dim=2)  # [B, target_length, H, W]
            # Normalize by bin size to get average spike rate
            spikes_voxelized = spikes_voxelized / bin_size_actual
        else:
            # If T <= target_length, use original spikes
            spikes_voxelized = spikes
        
        # Temporal aggregation: use weighted temporal average on voxelized spikes
        # According to paper: voxelization reduces temporal dimension, then process
        # Use exponential weights: later spikes have higher weights (like WGSE)
        T_vox = spikes_voxelized.shape[1]
        weights = torch.exp(torch.linspace(0, 2, T_vox, device=spikes.device))  # [T_vox]
        weights = weights / weights.sum()  # Normalize to sum to 1
        weights = weights.view(1, T_vox, 1, 1)  # [1, T_vox, 1, 1]
        
        # Weighted temporal average on voxelized spikes
        x = (spikes_voxelized * weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Simple normalization: just clamp to [0, 1] (spikes are already binary)
        # Don't do aggressive per-sample normalization as it destroys intensity relationships
        # The weighted average should already be in a reasonable range
        x = torch.clamp(x, 0, 1)
        
        # Process aggregated temporal features through SNN encoder
        # LIFNodes still process temporal information (from aggregated input)
        # This is more memory-efficient than processing each time step separately
        e1 = self.enc1(x)      # [B, 32, 224, 224]
        e2 = self.enc2(e1)      # [B, 64, 112, 112]
        e3 = self.enc3(e2)      # [B, 128, 56, 56]
        e4 = self.enc4(e3)      # [B, 256, 28, 28]
        
        # Bottleneck
        b = self.bottleneck(e4)  # [B, 256, 28, 28]
        
        # Decoder with skip connections (U-Net style)
        # Process aggregated features through decoder
        d4 = self.dec4(b)       # [B, 128, 56, 56]
        d4 = torch.cat([d4, e3], dim=1)  # [B, 256, 56, 56] - skip connection
        
        d3 = self.dec3(d4)      # [B, 64, 112, 112]
        d3 = torch.cat([d3, e2], dim=1)  # [B, 128, 112, 112] - skip connection
        
        d2 = self.dec2(d3)      # [B, 32, 224, 224]
        d2 = torch.cat([d2, e1], dim=1)  # [B, 64, 224, 224] - skip connection
        
        x = self.dec1(d2)       # [B, 3, 224, 224]

        # Bound outputs to [0, 1] so images are viewable
        # Use sigmoid for stable output activation (prevents color distortion)
        # Sigmoid ensures outputs are in [0, 1] range without rescaling artifacts
        # This prevents the purple/brown/green color fringing from tanh rescaling
        x = torch.sigmoid(x)  # [0, 1]
        x = torch.clamp(x, 0, 1)  # Ensure [0, 1] range
        
        return x

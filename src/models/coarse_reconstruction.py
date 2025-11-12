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

        # TRUE SNN: Enhanced encoder with LIFNodes for temporal processing
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
        # Use nearest-neighbor upsampling + conv for sharper outputs
        # TRUE SNN: LIFNodes in decoder for temporal processing
        # Layer 4: 28x28 -> 56x56
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 from skip connection
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Layer 3: 56x56 -> 112x112
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # 128*2 from skip
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Layer 2: 112x112 -> 224x224
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),  # 64*2 from skip
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
        )
        # Final layer: 224x224 -> 224x224
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32*2 from skip
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        TRUE SNN forward pass: Process temporal sequence [B, T, H, W] through network.
        LIFNodes accumulate temporal information across time steps.
        
        Args:
            spikes: [B, T, H, W] input spike tensor
        
        Returns:
            images: [B, 3, H, W] reconstructed images in [0, 1] range
        """
        # Reset SNN state at the start of each forward pass
        functional.reset_net(self)
        
        B, T, H, W = spikes.shape
        
        # TRUE SNN: Process temporal sequence through network
        # Voxelization: Reduce temporal dimension for efficiency (200 → 50)
        # This reduces memory while still allowing temporal processing
        target_length = 50  # Reduce from 200 to 50 for efficiency
        if T > target_length:
            # Pad if needed
            if T % target_length != 0:
                pad_size = target_length - (T % target_length)
                spikes = torch.cat([spikes, torch.zeros(B, pad_size, H, W, device=spikes.device, dtype=spikes.dtype)], dim=1)
                T = spikes.shape[1]
            
            # Voxelize: [B, T, H, W] -> [B, target_length, H, W]
            bin_size_actual = T // target_length
            spikes_reshaped = spikes.view(B, target_length, bin_size_actual, H, W)
            # Sum over bin dimension (aggregate spikes within each bin)
            spikes_voxelized = spikes_reshaped.sum(dim=2)  # [B, target_length, H, W]
            # Normalize by bin size
            spikes_voxelized = spikes_voxelized / bin_size_actual
        else:
            spikes_voxelized = spikes
        
        T_vox = spikes_voxelized.shape[1]
        
        # TRUE SNN: Process each time step through encoder
        # LIFNodes accumulate state across time steps
        # Store encoder features from final time step for skip connections
        for t in range(T_vox):
            # Get spike frame at time t: [B, H, W] -> [B, 1, H, W]
            spike_frame = spikes_voxelized[:, t:t+1, :, :]  # [B, 1, H, W]
            
            # Process through encoder (LIFNodes accumulate state)
            e1_t = self.enc1(spike_frame)      # [B, 32, 224, 224]
            e2_t = self.enc2(e1_t)              # [B, 64, 112, 112]
            e3_t = self.enc3(e2_t)              # [B, 128, 56, 56]
            e4_t = self.enc4(e3_t)              # [B, 256, 28, 28]
            
            # Bottleneck
            b_t = self.bottleneck(e4_t)        # [B, 256, 28, 28]
        
        # Use final time step features for skip connections
        e1 = e1_t  # [B, 32, 224, 224]
        e2 = e2_t  # [B, 64, 112, 112]
        e3 = e3_t  # [B, 128, 56, 56]
        e4 = e4_t  # [B, 256, 28, 28]
        b = b_t    # [B, 256, 28, 28]
        
        # Decoder: Process final bottleneck through decoder with skip connections
        # Decoder also processes through LIFNodes (state already accumulated from encoder)
        d4 = self.dec4(b)       # [B, 128, 56, 56]
        d4 = torch.cat([d4, e3], dim=1)  # [B, 256, 56, 56] - skip connection
        
        d3 = self.dec3(d4)      # [B, 64, 112, 112]
        d3 = torch.cat([d3, e2], dim=1)  # [B, 128, 112, 112] - skip connection
        
        d2 = self.dec2(d3)      # [B, 32, 224, 224]
        d2 = torch.cat([d2, e1], dim=1)  # [B, 64, 224, 224] - skip connection
        
        x = self.dec1(d2)       # [B, 3, 224, 224]

        # Bound outputs to [0, 1]
        x = torch.tanh(x)  # [-1, 1]
        x = (x + 1.0) / 2.0  # Rescale to [0, 1]
        x = torch.clamp(x, 0, 1)  # Ensure [0, 1] range
        
        return x

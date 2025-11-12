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
        # Layer 4: 28x28 -> 56x56
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 from skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Layer 3: 56x56 -> 112x112
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # 128*2 from skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Layer 2: 112x112 -> 224x224
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),  # 64*2 from skip
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional conv
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Final layer: 224x224 -> 224x224
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32*2 from skip
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional conv for better features
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN encoder-decoder with skip connections.
        
        Args:
            spikes: [B, T, H, W] input spike tensor
        
        Returns:
            images: [B, 3, H, W] reconstructed images in [0, 1] range
        """
        # Reset SNN state each forward so batches are independent
        functional.reset_net(self)

        # Improved temporal aggregation: use mean + variance + max for richer features
        # This provides better intensity, temporal variation, and preserves sharp details
        # [B, T, H, W] -> [B, 1, H, W]
        spike_mean = spikes.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        spike_var = spikes.var(dim=1, keepdim=True)  # [B, 1, H, W]
        spike_max = spikes.max(dim=1, keepdim=True)[0]  # [B, 1, H, W] - preserves sharp details
        
        # Combine mean, variance, and max for richer temporal information
        # Mean: overall intensity (can blur temporal details)
        # Variance: temporal dynamics and motion
        # Max: preserves sharp details (reduces blurring)
        # Adjusted weights to better preserve structure and reduce blurring
        # Further increased max weight and reduced mean to reduce blurring
        # Max weight increased to 0.7 to preserve sharp details, mean reduced to 0.3
        x = 0.3 * spike_mean + 0.2 * spike_var + 0.7 * spike_max  # [B, 1, H, W]
        
        # Normalize to [0, 1] range for better training stability
        # Use per-sample normalization to preserve local contrast and reduce blurring
        # Per-sample normalization preserves local details better than global normalization
        # This helps reduce blurring and color distortion
        B = x.shape[0]
        for b in range(B):
            sample = x[b, 0]  # [H, W]
            if sample.max() > sample.min():
                sample_min = sample.min()
                sample_max = sample.max()
                x[b, 0] = (sample - sample_min) / (sample_max - sample_min + 1e-8)
        x = torch.clamp(x, 0, 1)

        # Encoder with skip connections
        e1 = self.enc1(x)      # [B, 32, 224, 224]
        e2 = self.enc2(e1)      # [B, 64, 112, 112]
        e3 = self.enc3(e2)      # [B, 128, 56, 56]
        e4 = self.enc4(e3)      # [B, 256, 28, 28]
        
        # Bottleneck
        b = self.bottleneck(e4)  # [B, 256, 28, 28]
        
        # Decoder with skip connections (U-Net style)
        d4 = self.dec4(b)       # [B, 128, 56, 56]
        d4 = torch.cat([d4, e3], dim=1)  # [B, 256, 56, 56] - skip connection
        
        d3 = self.dec3(d4)      # [B, 64, 112, 112]
        d3 = torch.cat([d3, e2], dim=1)  # [B, 128, 112, 112] - skip connection
        
        d2 = self.dec2(d3)      # [B, 32, 224, 224]
        d2 = torch.cat([d2, e1], dim=1)  # [B, 64, 224, 224] - skip connection
        
        x = self.dec1(d2)       # [B, 3, 224, 224]

        # Bound outputs to [0, 1] so images are viewable
        # Use tanh + rescale instead of sigmoid to reduce blurring
        # Sigmoid can cause saturation and blur fine details
        # Tanh provides better gradient flow and sharper outputs
        x = torch.tanh(x)  # [-1, 1]
        x = (x + 1.0) / 2.0  # [0, 1]
        x = torch.clamp(x, 0, 1)
        
        return x

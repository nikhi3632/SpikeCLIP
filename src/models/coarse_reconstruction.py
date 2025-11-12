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
        
        # Initialize skip connection accumulators
        # We'll accumulate skip connections across time steps
        e1_acc = None
        e2_acc = None
        e3_acc = None
        e4_acc = None
        b_acc = None
        
        # Process each time step through the encoder sequentially
        # LIFNodes will accumulate temporal information across time steps
        for t in range(T):
            # Get spike frame at time t: [B, H, W] -> [B, 1, H, W]
            spike_frame = spikes[:, t:t+1, :, :]  # [B, 1, H, W]
            
            # Encoder with skip connections (LIFNodes accumulate temporal info)
            # LIFNodes maintain state across time steps, accumulating information
            e1_t = self.enc1(spike_frame)      # [B, 32, 224, 224]
            e2_t = self.enc2(e1_t)             # [B, 64, 112, 112]
            e3_t = self.enc3(e2_t)             # [B, 128, 56, 56]
            e4_t = self.enc4(e3_t)             # [B, 256, 28, 28]
            
            # Bottleneck
            b_t = self.bottleneck(e4_t)        # [B, 256, 28, 28]
            
            # Accumulate skip connections across time steps
            if e1_acc is None:
                e1_acc = e1_t
                e2_acc = e2_t
                e3_acc = e3_t
                e4_acc = e4_t
                b_acc = b_t
            else:
                # Average accumulation (can also use sum or weighted sum)
                e1_acc = (e1_acc * t + e1_t) / (t + 1)
                e2_acc = (e2_acc * t + e2_t) / (t + 1)
                e3_acc = (e3_acc * t + e3_t) / (t + 1)
                e4_acc = (e4_acc * t + e4_t) / (t + 1)
                b_acc = (b_acc * t + b_t) / (t + 1)
        
        # After processing all time steps, LIFNodes have accumulated temporal information
        # Use accumulated features (average across all time steps)
        # These features contain information from all time steps
        e1 = e1_acc
        e2 = e2_acc
        e3 = e3_acc
        e4 = e4_acc
        b = b_acc
        
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
        # Use tanh + rescale instead of sigmoid to reduce blurring
        # Sigmoid can cause saturation and blur fine details
        # Tanh provides better gradient flow and sharper outputs
        x = torch.tanh(x)  # [-1, 1]
        x = (x + 1.0) / 2.0  # [0, 1]
        x = torch.clamp(x, 0, 1)
        
        return x

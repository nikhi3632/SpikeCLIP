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

        # Simple 2D encoder with LIF nodes
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            LIFNode(tau=tau, v_threshold=v_threshold),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            LIFNode(tau=tau, v_threshold=v_threshold),
        )

        # Decoder: deconvs back to full resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 112 -> 224
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN encoder-decoder.
        
        Args:
            spikes: [B, T, H, W] input spike tensor
        
        Returns:
            images: [B, 3, H, W] reconstructed images in [0, 1] range
        """
        # Reset SNN state each forward so batches are independent
        functional.reset_net(self)

        # Temporal aggregation: average spikes over time dimension
        # This provides a coarse intensity estimate from spike events
        # [B, T, H, W] -> [B, 1, H, W]
        # Note: For a more sophisticated SNN, we could process temporal sequences
        # through the network, but averaging is a reasonable baseline
        x = spikes.mean(dim=1, keepdim=True)
        
        # Normalize to [0, 1] range for better training stability
        x = torch.clamp(x, 0, 1)

        # Encode through SNN layers (with LIF neurons)
        x = self.encoder(x)
        
        # Decode to full resolution
        x = self.decoder(x)

        # Bound outputs to [0, 1] so images are viewable
        x = torch.sigmoid(x)
        
        return x

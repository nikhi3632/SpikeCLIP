"""Stage 3 CNN/UNet refiner"""
import torch
import torch.nn as nn

class RefinementNet(nn.Module):
    """
    Stage 3: Refinement network.
    
    NOTE: Based on reference code analysis, refinement may not need separate training.
    This implementation defaults to identity (just return coarse images) unless trained.
    
    Input:  coarse_images [B, 3, H, W]
    Output: refined_images [B, 3, H, W] in [0, 1]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_down: int = 4,
        use_identity: bool = True  # Default: just return coarse (no refinement)
    ):
        super().__init__()
        self.num_down = num_down
        self.use_identity = use_identity
        
        # If using identity, just return input (no refinement needed)
        if use_identity:
            return
        
        # Encoder blocks (save features for skip connections before pooling)
        self.enc1_conv, self.enc1_pool = self._make_encoder_block(in_channels, base_channels)  # 224x224
        self.enc2_conv, self.enc2_pool = self._make_encoder_block(base_channels, base_channels * 2)  # 112x112
        self.enc3_conv, self.enc3_pool = self._make_encoder_block(base_channels * 2, base_channels * 4)  # 56x56
        self.enc4_conv, self.enc4_pool = self._make_encoder_block(base_channels * 4, base_channels * 8, pool=False)  # 28x28
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with skip connections
        # Each decoder block upsamples, then concatenates with skip connection
        # Bottleneck output is base_channels * 8
        self.dec4_up = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),  # e3 is base_channels * 4
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.dec3_up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)  # d4 is base_channels * 4
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),  # d3_up(128) + e2(128) = 256
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.dec2_up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)  # d3 is base_channels * 2
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),  # d2_up(64) + e1(64) = 128
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels + in_channels, base_channels, kernel_size=3, padding=1),  # d2(64) + x(3) = 67
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _make_encoder_block(self, in_ch, out_ch, pool=True):
        conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        if pool:
            pool_layer = nn.MaxPool2d(2)
            return conv_block, pool_layer
        return conv_block, None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] coarse images
        Returns: [B, 3, H, W] refined images
        """
        # If using identity, just return coarse images (no refinement)
        # This matches the reference code which doesn't show refinement training
        if self.use_identity:
            return x
        
        # Full UNet refinement (original implementation)
        # Encoder path (save features before pooling for skip connections)
        e1 = self.enc1_conv(x)  # [B, 64, 224, 224]
        e1_pooled = self.enc1_pool(e1) if self.enc1_pool else e1  # [B, 64, 112, 112]
        
        e2 = self.enc2_conv(e1_pooled)  # [B, 128, 112, 112]
        e2_pooled = self.enc2_pool(e2) if self.enc2_pool else e2  # [B, 128, 56, 56]
        
        e3 = self.enc3_conv(e2_pooled)  # [B, 256, 56, 56]
        e3_pooled = self.enc3_pool(e3) if self.enc3_pool else e3  # [B, 256, 28, 28]
        
        e4 = self.enc4_conv(e3_pooled)  # [B, 512, 28, 28]
        
        # Bottleneck
        b = self.bottleneck(e4)  # [B, 512, 28, 28]
        
        # Decoder path with skip connections (using features before pooling)
        # Upsample bottleneck, concat with e4, then process
        d4_up = self.dec4_up(b)  # [B, 256, 56, 56]
        d4 = self.dec4_conv(torch.cat([d4_up, e3], dim=1))  # [B, 256, 56, 56]
        
        d3_up = self.dec3_up(d4)  # [B, 128, 112, 112]
        d3 = self.dec3_conv(torch.cat([d3_up, e2], dim=1))  # [B, 128, 112, 112]
        
        d2_up = self.dec2_up(d3)  # [B, 64, 224, 224]
        d2 = self.dec2_conv(torch.cat([d2_up, e1], dim=1))  # [B, 64, 224, 224]
        
        d1 = self.dec1(torch.cat([d2, x], dim=1))  # [B, 3, 224, 224] (skip from input)
        
        return d1

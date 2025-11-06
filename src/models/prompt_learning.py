"""Stage 2 CLIP adapter"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PromptAdapter(nn.Module):
    """
    CLIP adapter for prompt learning.
    Takes coarse reconstructed images and learns text prompts for CLIP alignment.
    
    Input:  coarse_images [B, 3, H, W]
    Output: clip_features [B, D] or text_embeddings
    """
    
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        clip_dim: int = 512,
        num_classes: int = 101,
        prompt_dim: int = 77,
        freeze_image_encoder: bool = True
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.num_classes = num_classes
        self.prompt_dim = prompt_dim
        
        # Image encoder (frozen CLIP vision encoder or trainable adapter)
        if image_encoder is not None:
            self.image_encoder = image_encoder
            if freeze_image_encoder:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
        else:
            # Simple adapter if no CLIP encoder provided
            self.image_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(3 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, clip_dim)
            )
        
        # Learnable text prompts (one per class)
        self.class_prompts = nn.Parameter(torch.randn(num_classes, prompt_dim, clip_dim))
        nn.init.normal_(self.class_prompts, std=0.02)
        
        # Projection to match CLIP dimensions
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, clip_dim)
        )
    
    def forward(self, images: torch.Tensor, label_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        images: [B, 3, H, W] coarse reconstructed images
        label_indices: [B] optional label indices for prompt selection
        Returns: [B, clip_dim] or [B, num_classes, clip_dim] if label_indices not provided
        """
        # Encode images
        image_features = self.image_encoder(images)  # [B, clip_dim]
        image_features = F.normalize(image_features, dim=-1)
        
        # Project to CLIP space
        image_features = self.projection(image_features)
        
        # Get text prompts
        if label_indices is not None:
            # Select prompts based on labels
            selected_prompts = self.class_prompts[label_indices]  # [B, prompt_dim, clip_dim]
            # Average over prompt dimension
            text_features = selected_prompts.mean(dim=1)  # [B, clip_dim]
        else:
            # Return all class prompts
            text_features = self.class_prompts.mean(dim=1)  # [num_classes, clip_dim]
        
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features
    
    def get_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP-aligned image features."""
        image_features = self.image_encoder(images)
        image_features = F.normalize(image_features, dim=-1)
        image_features = self.projection(image_features)
        return image_features
    
    def get_text_embeddings(self, label_indices: torch.Tensor) -> torch.Tensor:
        """Get text embeddings for given labels."""
        selected_prompts = self.class_prompts[label_indices]
        text_features = selected_prompts.mean(dim=1)
        return F.normalize(text_features, dim=-1)

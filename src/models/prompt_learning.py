"""Stage 2 CLIP adapter"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import clip

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
        freeze_image_encoder: bool = True,
        clip_model_name: str = "ViT-B/32",
        class_labels: Optional[list] = None
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.num_classes = num_classes
        self.prompt_dim = prompt_dim
        
        # Load actual CLIP model
        try:
            # Load CLIP model on CPU first, will be moved to device later
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
            self.clip_model.eval()
            
            # Get actual CLIP dimensions
            with torch.no_grad():
                dummy_image = torch.zeros(1, 3, 224, 224)
                dummy_text = clip.tokenize(["a"])
                image_features = self.clip_model.encode_image(dummy_image)
                text_features = self.clip_model.encode_text(dummy_text)
                actual_image_dim = image_features.shape[-1]
                actual_text_dim = text_features.shape[-1]
            
            # Use CLIP's image encoder
            if image_encoder is not None:
                self.image_encoder = image_encoder
            else:
                self.image_encoder = self.clip_model.visual
                
            if freeze_image_encoder:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
            
            # Update clip_dim to match actual CLIP dimensions
            self.clip_dim = actual_image_dim
            
        except Exception as e:
            print(f"Warning: Failed to load CLIP model ({e}). Using fallback adapter.")
            # Fallback to simple adapter if CLIP loading fails
            if image_encoder is not None:
                self.image_encoder = image_encoder
            else:
                self.image_encoder = nn.Sequential(
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(3 * 7 * 7, 512),
                    nn.ReLU(),
                    nn.Linear(512, clip_dim)
                )
            self.clip_model = None
        
        # Initialize learnable text prompts from actual CLIP text embeddings
        # This provides better initialization than random
        if self.clip_model is not None and class_labels is not None and len(class_labels) == num_classes:
            # Initialize from actual CLIP text embeddings
            with torch.no_grad():
                # Create text prompts like "a photo of a {label}"
                text_prompts = [f"a photo of a {label}" for label in class_labels]
                # Tokenize and encode
                text_tokens = clip.tokenize(text_prompts)
                text_embeddings = self.clip_model.encode_text(text_tokens)  # [num_classes, clip_dim]
                text_embeddings = F.normalize(text_embeddings, dim=-1)
                
                # Expand to [num_classes, prompt_dim, clip_dim] by repeating and adding small noise
                # This allows the prompts to learn while starting from meaningful initialization
                init_prompts = text_embeddings.unsqueeze(1).repeat(1, prompt_dim, 1)  # [num_classes, prompt_dim, clip_dim]
                # Add small random noise to allow learning
                noise = torch.randn_like(init_prompts) * 0.01
                init_prompts = init_prompts + noise
        else:
            # Fallback to random initialization
            init_prompts = torch.randn(num_classes, prompt_dim, self.clip_dim)
            nn.init.normal_(init_prompts, std=0.02)
        
        # Learnable text prompts (one per class)
        # Each prompt is a learnable embedding that will be processed by CLIP's text encoder
        # Format: [num_classes, prompt_dim] where prompt_dim is the token sequence length
        self.class_prompts = nn.Parameter(init_prompts)
        
        # Projection to match CLIP dimensions (for images)
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim),
            nn.GELU(),
            nn.Linear(self.clip_dim, self.clip_dim)
        )
    
    def _apply(self, fn):
        """Override _apply to also move CLIP model when model is moved to device."""
        super()._apply(fn)
        if self.clip_model is not None:
            # Move CLIP model to the same device
            self.clip_model = self.clip_model._apply(fn)
        return self
    
    def forward(self, images: torch.Tensor, label_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        images: [B, 3, H, W] coarse reconstructed images (should be in [0, 1] range)
        label_indices: [B] optional label indices for prompt selection
        Returns: (image_features, text_features) where both are normalized
        """
        # Normalize images to CLIP's expected range [-1, 1] if using actual CLIP
        if self.clip_model is not None:
            # CLIP expects images in [0, 1] range, but we normalize to [-1, 1]
            # Actually, CLIP's preprocess does this, but we'll handle it here
            images_normalized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            # CLIP visual encoder expects [0, 1] range
            images_normalized = torch.clamp(images_normalized, 0, 1)
        else:
            images_normalized = images
        
        # Encode images using CLIP or adapter
        if self.clip_model is not None:
            image_features = self.image_encoder(images_normalized)  # [B, clip_dim]
            # CLIP already normalizes, but we'll normalize again for consistency
            image_features = F.normalize(image_features, dim=-1)
        else:
            image_features = self.image_encoder(images_normalized)
            image_features = F.normalize(image_features, dim=-1)
        
        # Project to CLIP space (optional refinement)
        image_features = self.projection(image_features)
        image_features = F.normalize(image_features, dim=-1)
        
        # Get text prompts
        if label_indices is not None:
            # Select prompts based on labels
            selected_prompts = self.class_prompts[label_indices]  # [B, prompt_dim, clip_dim]
            # Average over prompt dimension to get text embedding
            text_features = selected_prompts.mean(dim=1)  # [B, clip_dim]
        else:
            # Return all class prompts
            text_features = self.class_prompts.mean(dim=1)  # [num_classes, clip_dim]
        
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features
    
    def get_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP-aligned image features."""
        # Normalize images for CLIP
        if self.clip_model is not None:
            images_normalized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            images_normalized = torch.clamp(images_normalized, 0, 1)
        else:
            images_normalized = images
            
        image_features = self.image_encoder(images_normalized)
        image_features = F.normalize(image_features, dim=-1)
        image_features = self.projection(image_features)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def get_text_embeddings(self, label_indices: torch.Tensor) -> torch.Tensor:
        """Get text embeddings for given labels."""
        selected_prompts = self.class_prompts[label_indices]
        text_features = selected_prompts.mean(dim=1)
        return F.normalize(text_features, dim=-1)

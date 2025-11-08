"""Stage 2 CLIP adapter - includes both HQ/LQ prompt learning and class prompt learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import clip


class TextEncoder(nn.Module):
    """Text encoder to process learnable prompts through CLIP's text encoder (from reference code)."""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts):
        """
        prompts: [B, prompt_dim, clip_dim] learnable prompt embeddings (or [prompt_dim, clip_dim] for single prompt)
        Returns: [B, clip_dim] text features (or [clip_dim] for single prompt)
        """
        # Handle both batched and single prompt cases
        if prompts.dim() == 2:
            # Single prompt: [prompt_dim, clip_dim]
            prompts = prompts.unsqueeze(0)  # [1, prompt_dim, clip_dim]
            single = True
        else:
            single = False
        
        # Add positional embedding (use first prompt_dim positions)
        # prompts is [B, prompt_dim, clip_dim], positional_embedding is [prompt_dim, clip_dim]
        x = prompts + self.positional_embedding[:prompts.shape[1]].type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # Get features at the end position (like CLIP does)
        x = x[torch.arange(x.shape[0]), -1] @ self.text_projection
        
        if single:
            return x.squeeze(0)  # [clip_dim]
        return x  # [B, clip_dim]


class HQ_LQ_PromptAdapter(nn.Module):
    """
    CLIP adapter for HQ/LQ prompt learning (Stage 2).
    
    According to the paper:
    - Learns HQ (High-Quality) and LQ (Low-Quality) prompts
    - Binary classification: HQ vs LQ
    - Loss: CrossEntropy(y, y_hat) where y_hat = exp(Φ_image(I)·Φ_text(T_hq)) / Σ exp(Φ_image(I)·Φ_text(T_i))
    
    Input:  images [B, 3, H, W] (HQ or LQ images)
    Output: image_features [B, clip_dim], hq_prompt_features [clip_dim], lq_prompt_features [clip_dim]
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        prompt_dim: int = 77,
        freeze_image_encoder: bool = True
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        
        # Load CLIP model
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
            self.clip_model.eval()
            
            # Get CLIP dimensions
            with torch.no_grad():
                dummy_image = torch.zeros(1, 3, 224, 224)
                dummy_text = clip.tokenize(["a"])
                image_features = self.clip_model.encode_image(dummy_image)
                text_features = self.clip_model.encode_text(dummy_text)
                self.clip_dim = image_features.shape[-1]
            
            # Use CLIP's image encoder
            self.image_encoder = self.clip_model.visual
            
            if freeze_image_encoder:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
        
        # Learnable HQ and LQ prompts
        # Initialize from CLIP text embeddings
        with torch.no_grad():
            # Initialize HQ prompt from "a high quality image"
            hq_text = clip.tokenize(["a high quality image"])
            hq_embedding = self.clip_model.encode_text(hq_text)  # [1, clip_dim]
            hq_embedding = F.normalize(hq_embedding, dim=-1)
            
            # Initialize LQ prompt from "a low quality image"
            lq_text = clip.tokenize(["a low quality image"])
            lq_embedding = self.clip_model.encode_text(lq_text)  # [1, clip_dim]
            lq_embedding = F.normalize(lq_embedding, dim=-1)
            
            # Expand to [prompt_dim, clip_dim] by repeating and adding small noise
            hq_init = hq_embedding.squeeze(0).unsqueeze(0).repeat(prompt_dim, 1)  # [prompt_dim, clip_dim]
            lq_init = lq_embedding.squeeze(0).unsqueeze(0).repeat(prompt_dim, 1)  # [prompt_dim, clip_dim]
            
            # Add small random noise to allow learning
            hq_noise = torch.randn_like(hq_init) * 0.01
            lq_noise = torch.randn_like(lq_init) * 0.01
            hq_init = hq_init + hq_noise
            lq_init = lq_init + lq_noise
        
        # Learnable prompts: [prompt_dim, clip_dim]
        self.hq_prompt = nn.Parameter(hq_init)
        self.lq_prompt = nn.Parameter(lq_init)
        
        # Text encoder to process prompts through CLIP's text encoder
        self.text_encoder = TextEncoder(self.clip_model)
        
        # Freeze text encoder parameters (only prompts are learnable)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> tuple:
        """
        images: [B, 3, H, W] images (HQ or LQ)
        Returns: (image_features, hq_prompt_features, lq_prompt_features)
        """
        # Normalize images for CLIP
        images_normalized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images_normalized = torch.clamp(images_normalized, 0, 1)
        
        # Encode images using CLIP
        image_features = self.image_encoder(images_normalized)  # [B, clip_dim]
        image_features = F.normalize(image_features, dim=-1)
        
        # Encode HQ and LQ prompts
        hq_prompt_features = self.text_encoder(self.hq_prompt)  # [clip_dim]
        lq_prompt_features = self.text_encoder(self.lq_prompt)  # [clip_dim]
        
        # Normalize prompt features
        hq_prompt_features = F.normalize(hq_prompt_features, dim=-1)
        lq_prompt_features = F.normalize(lq_prompt_features, dim=-1)
        
        return image_features, hq_prompt_features, lq_prompt_features
    
    def get_prompt_features(self) -> tuple:
        """Get HQ and LQ prompt features."""
        hq_prompt_features = self.text_encoder(self.hq_prompt)  # [clip_dim]
        lq_prompt_features = self.text_encoder(self.lq_prompt)  # [clip_dim]
        hq_prompt_features = F.normalize(hq_prompt_features, dim=-1)
        lq_prompt_features = F.normalize(lq_prompt_features, dim=-1)
        return hq_prompt_features, lq_prompt_features


class PromptAdapter(nn.Module):
    """
    CLIP adapter for class prompt learning.
    Takes coarse reconstructed images and learns text prompts for CLIP alignment.
    
    Used for combining checkpoints and inference.
    
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
        # Format: [num_classes, prompt_dim, clip_dim] where prompt_dim is the token sequence length
        self.class_prompts = nn.Parameter(init_prompts)
        
        # Text encoder to process prompts through CLIP's text encoder (like reference code)
        if self.clip_model is not None:
            self.text_encoder = TextEncoder(self.clip_model)
            # Freeze text encoder parameters (only prompts are learnable)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            self.text_encoder = None
        
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
        
        # Get text prompts - use TextEncoder if available (like reference code)
        if label_indices is not None:
            # Select prompts based on labels
            selected_prompts = self.class_prompts[label_indices]  # [B, prompt_dim, clip_dim]
            
            if self.text_encoder is not None:
                # Process prompts through CLIP's text encoder (like reference code)
                text_features = self.text_encoder(selected_prompts)  # [B, clip_dim]
            else:
                # Fallback: average over prompt dimension
                text_features = selected_prompts.mean(dim=1)  # [B, clip_dim]
        else:
            # Return all class prompts
            if self.text_encoder is not None:
                # Process all prompts through text encoder
                text_features = self.text_encoder(self.class_prompts)  # [num_classes, clip_dim]
            else:
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
        selected_prompts = self.class_prompts[label_indices]  # [B, prompt_dim, clip_dim]
        if self.text_encoder is not None:
            # Process through text encoder (like reference code)
            text_features = self.text_encoder(selected_prompts)  # [B, clip_dim]
        else:
            # Fallback: average
            text_features = selected_prompts.mean(dim=1)  # [B, clip_dim]
        return F.normalize(text_features, dim=-1)

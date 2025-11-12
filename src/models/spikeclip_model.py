"""unified end-to-end model wrapper"""
import torch
import torch.nn as nn
from typing import Optional

from models.coarse_reconstruction import CoarseSNN
from models.prompt_learning import HQ_LQ_PromptAdapter
from models.refinement import RefinementNet

class SpikeCLIPModel(nn.Module):
    """
    Unified end-to-end model that combines all three stages:
    Stage 1: Coarse Reconstruction (SNN)
    Stage 2: Prompt Learning (CLIP adapter)
    Stage 3: Refinement (UNet)
    
    Input:  spikes [B, T, H, W]
    Output: refined_images [B, 3, H, W], clip_features [B, D]
    """
    
    def __init__(
        self,
        coarse_model: CoarseSNN,
        prompt_model: HQ_LQ_PromptAdapter,
        refine_model: RefinementNet,
        return_features: bool = True,
        labels: Optional[list] = None
    ):
        super().__init__()
        self.coarse_model = coarse_model
        self.prompt_model = prompt_model
        self.refine_model = refine_model
        self.return_features = return_features
        self.labels = labels
        
        # Freeze all models by default (they should be pre-trained)
        for param in self.coarse_model.parameters():
            param.requires_grad = False
        for param in self.prompt_model.parameters():
            param.requires_grad = False
        for param in self.refine_model.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        spikes: torch.Tensor, 
        label_indices: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass through all stages.
        
        Args:
            spikes: [B, T, H, W] input spikes
            label_indices: [B] optional label indices for prompt selection
        
        Returns:
            refined_images: [B, 3, H, W] refined images
            clip_features: [B, D] CLIP-aligned features (if return_features=True)
            coarse_images: [B, 3, H, W] coarse images (intermediate)
        """
        # Stage 1: Coarse reconstruction
        coarse_images = self.coarse_model(spikes)  # [B, 3, H, W]
        
        # Stage 3: Refinement
        refined_images = self.refine_model(coarse_images)  # [B, 3, H, W]
        
        # Stage 2: Prompt learning (get CLIP features from refined images for consistent classification)
        # Use refined images for classification to match classify() method
        # Note: text_features not needed here - only used during training for CLIP loss
        # For inference/classification, we get all class embeddings separately
        image_features = self.prompt_model.get_clip_features(refined_images)  # [B, D]
        
        if self.return_features:
            return refined_images, image_features, coarse_images
        else:
            return refined_images, coarse_images
    
    def get_clip_features(self, spikes: torch.Tensor) -> torch.Tensor:
        """Get CLIP features from spikes (using refined images for better classification)."""
        coarse_images = self.coarse_model(spikes)
        refined_images = self.refine_model(coarse_images)
        clip_features = self.prompt_model.get_clip_features(refined_images)
        return clip_features
    
    def classify(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Classify spikes using CLIP features from refined images.
        
        Args:
            spikes: [B, T, H, W] input spikes
        
        Returns:
            predictions: [B] predicted class indices
        """
        import clip
        import torch.nn.functional as F
        
        coarse_images = self.coarse_model(spikes)
        refined_images = self.refine_model(coarse_images)
        # Use refined images for classification to match forward() method
        # This ensures consistent feature extraction for both forward() and classify()
        # Refined images should have better quality after Stage 3 refinement
        image_features = self.prompt_model.get_clip_features(refined_images)
        
        # Get CLIP model from prompt model
        clip_model = self.prompt_model.clip_model
        
        # Use CLIP's encode_text directly for multi-class classification
        # (HQ_LQ_PromptAdapter is for binary classification, not multi-class)
        if self.labels is None:
            raise ValueError("Labels must be provided for classification")
        
        # Compute text features for all labels using CLIP
        # Use ensemble prompts for better robustness
        # Optimized prompt templates for better semantic alignment:
        # - Include diverse prompts to capture different semantic aspects
        # - First template matches Stage 3 training for consistency
        prompt_templates = [
            "a photo of a {}",  # Match Stage 3 training
            "a high quality photo of a {}",  # Emphasize quality
            "a clear image of a {}",  # Emphasize clarity
            "a picture of a {}",  # Alternative phrasing
            "an image of a {}",  # Another alternative
            "a photograph of a {}"  # Formal variant
        ]
        
        # Ensemble text features from multiple prompts
        all_text_features_list = []
        for template in prompt_templates:
            text_prompts = [template.format(label) for label in self.labels]
            text_tokens = clip.tokenize(text_prompts).to(spikes.device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens)  # [num_classes, clip_dim]
                text_features = F.normalize(text_features, dim=-1)
                all_text_features_list.append(text_features)
        
        # Average the text features from different prompts (ensemble)
        all_text_features = torch.stack(all_text_features_list, dim=0).mean(dim=0)  # [num_classes, clip_dim]
        all_text_features = F.normalize(all_text_features, dim=-1)
        
        # Ensure image features are normalized
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity with temperature scaling
        # Optimized temperature for better discrimination:
        # - Lower temperature (0.05) = sharper discrimination but may be too aggressive
        # - Higher temperature (0.1) = softer discrimination, more robust
        # - Use 0.05 for sharper discrimination to improve accuracy
        temperature = 0.05  # Optimized for better classification accuracy
        similarities = torch.matmul(image_features, all_text_features.t()) / temperature
        predictions = similarities.argmax(dim=1)
        
        return predictions

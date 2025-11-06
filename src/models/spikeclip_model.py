"""unified end-to-end model wrapper"""
import torch
import torch.nn as nn
from typing import Optional

from models.coarse_reconstruction import CoarseSNN
from models.prompt_learning import PromptAdapter
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
        prompt_model: PromptAdapter,
        refine_model: RefinementNet,
        return_features: bool = True
    ):
        super().__init__()
        self.coarse_model = coarse_model
        self.prompt_model = prompt_model
        self.refine_model = refine_model
        self.return_features = return_features
        
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
        
        # Stage 2: Prompt learning (get CLIP features)
        # Note: text_features not needed here - only used during training for CLIP loss
        # For inference/classification, we get all class embeddings separately
        image_features = self.prompt_model.get_clip_features(coarse_images)  # [B, D]
        
        # Stage 3: Refinement
        refined_images = self.refine_model(coarse_images)  # [B, 3, H, W]
        
        if self.return_features:
            return refined_images, image_features, coarse_images
        else:
            return refined_images, coarse_images
    
    def get_clip_features(self, spikes: torch.Tensor) -> torch.Tensor:
        """Get CLIP features from spikes."""
        coarse_images = self.coarse_model(spikes)
        clip_features = self.prompt_model.get_clip_features(coarse_images)
        return clip_features
    
    def classify(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Classify spikes using CLIP features.
        
        Args:
            spikes: [B, T, H, W] input spikes
        
        Returns:
            predictions: [B] predicted class indices
        """
        coarse_images = self.coarse_model(spikes)
        image_features = self.prompt_model.get_clip_features(coarse_images)
        
        # Get all text embeddings
        all_label_indices = torch.arange(self.prompt_model.num_classes, device=spikes.device)
        all_text_features = self.prompt_model.get_text_embeddings(all_label_indices)
        
        # Compute similarity
        similarities = torch.matmul(image_features, all_text_features.t())
        predictions = similarities.argmax(dim=1)
        
        return predictions

"""losses for all stages"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """L1 + L2 loss for image reconstruction."""
    def __init__(self, l1_weight=1.0, l2_weight=1.0, identity_penalty=0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.identity_penalty = identity_penalty
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, C, H, W] predicted image
        target: [B, C, H, W] target image
        """
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        reconstruction_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        # Add penalty for identity mapping (when pred == target)
        # This encourages the model to actually refine, not just copy
        if self.identity_penalty > 0:
            # Penalize when prediction is too similar to target
            # We want some difference to encourage refinement
            # Use a smooth penalty that increases as pred approaches target
            l1_diff = F.l1_loss(pred, target)
            # Penalty is high when l1_diff is very small (identity mapping)
            # Penalty decreases as l1_diff increases (actual refinement)
            identity_penalty = self.identity_penalty * torch.exp(-l1_diff * 20.0)
            reconstruction_loss = reconstruction_loss + identity_penalty
        
        return reconstruction_loss

class CLIPLoss(nn.Module):
    """CLIP contrastive loss for image-text alignment."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        image_features: [B, D] normalized image features
        text_features: [B, D] normalized text features
        Returns: contrastive loss
        """
        # Compute similarity
        logits = torch.matmul(image_features, text_features.t()) / self.temperature  # [B, B]
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Symmetric loss (image-to-text and text-to-image)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss

def get_loss_fn(loss_type: str = "reconstruction", **kwargs):
    """Factory function to get loss function."""
    if loss_type == "reconstruction":
        # Filter out identity_penalty for coarse stage (only use for refinement)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'identity_penalty' or v > 0}
        return ReconstructionLoss(**filtered_kwargs)
    elif loss_type == "clip":
        return CLIPLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

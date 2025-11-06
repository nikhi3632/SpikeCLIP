"""losses for all stages"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """L1 + L2 loss for image reconstruction."""
    def __init__(self, l1_weight=1.0, l2_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, C, H, W] predicted image
        target: [B, C, H, W] target image
        """
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        return self.l1_weight * l1_loss + self.l2_weight * l2_loss

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
        return ReconstructionLoss(**kwargs)
    elif loss_type == "clip":
        return CLIPLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

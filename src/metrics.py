"""Reconstruction metrics according to the paper.

According to the paper:
- Stage 1: Evaluated against TFI (Texture from ISI) target
- Stage 3: Evaluated against coarse images (refinement quality)
- Classification: Accuracy using CLIP text features
"""
import torch
import torch.nn.functional as F

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    According to the paper, used for:
    - Stage 1: Coarse vs TFI target
    - Stage 3: Refined vs Coarse (refinement quality)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Simplified SSIM (Structural Similarity Index) computation.
    
    According to the paper, used for:
    - Stage 1: Coarse vs TFI target
    - Stage 3: Refined vs Coarse (refinement quality)
    """
    # Simple SSIM approximation
    mu1 = pred.mean()
    mu2 = target.mean()
    sigma1_sq = pred.var()
    sigma2_sq = target.var()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()
    
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim.item()

def compute_l1_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute L1 error (Mean Absolute Error).
    
    According to the paper, used for:
    - Stage 1: Coarse vs TFI target
    - Stage 3: Refined vs Coarse (refinement quality)
    """
    return F.l1_loss(pred, target).item()

def compute_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute L2 (MSE) error.
    
    According to the paper, used for:
    - Stage 1: Coarse vs TFI target
    - Stage 3: Refined vs Coarse (refinement quality)
    """
    return F.mse_loss(pred, target).item()

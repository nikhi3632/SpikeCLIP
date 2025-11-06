"""reconstruction + CLIP + perceptual metrics"""
import torch
import torch.nn.functional as F

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simplified SSIM computation."""
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
    """Compute L1 error."""
    return F.l1_loss(pred, target).item()

def compute_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute L2 (MSE) error."""
    return F.mse_loss(pred, target).item()

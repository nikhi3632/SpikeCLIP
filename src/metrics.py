"""Reconstruction and no-reference perceptual metrics according to the paper.

According to the paper:
- Stage 1: Evaluated against TFI (Texture from ISI) target
- Stage 3: Evaluated against coarse images (refinement quality)
- Classification: Accuracy using CLIP text features
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.color import rgb2gray
    from skimage.metrics import brisque as skimage_brisque
    from skimage.metrics import niqe as skimage_niqe
    from skimage.metrics import piqe as skimage_piqe

    _SKIMAGE_IQA_AVAILABLE = True
except ImportError:  # pragma: no cover - skimage is already declared as a dep, but keep guard
    _SKIMAGE_IQA_AVAILABLE = False
    skimage_brisque = skimage_niqe = skimage_piqe = None
    rgb2gray = None

try:
    import pyiqa

    _PYIQA_AVAILABLE = True
    _PYIQA_MODELS = {}
except ImportError:  # pragma: no cover - fallback to NaN if pyiqa missing
    _PYIQA_AVAILABLE = False
    _PYIQA_MODELS = {}

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


def _prepare_tensor_for_iqa(image: torch.Tensor) -> Optional[np.ndarray]:
    """Convert a tensor image to a grayscale numpy array in [0, 1] for NR IQA metrics."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected torch.Tensor input for image quality metrics")

    with torch.no_grad():
        img = image.detach().cpu().float()

    if img.dim() == 4:
        if img.size(0) != 1:
            raise ValueError("Tensor with batch dimension must have batch size 1 for NR metrics")
        img = img[0]

    if img.dim() == 3:
        img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    elif img.dim() != 2:
        raise ValueError(f"Unsupported tensor shape {img.shape} for NR metrics")

    img_np = torch.clamp(img, 0, 1).numpy()

    # Convert to grayscale if needed
    if img_np.ndim == 3:
        if img_np.shape[2] == 1:
            img_np = img_np[:, :, 0]
        else:
            if rgb2gray is None:
                raise RuntimeError("scikit-image is required for color -> grayscale conversion")
            img_np = rgb2gray(img_np)

    return img_np.astype(np.float64)


def _prepare_tensor_for_pyiqa(image: torch.Tensor) -> torch.Tensor:
    """Convert tensor to BxCxHxW in [0,1] for pyiqa metrics."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected torch.Tensor input for image quality metrics")

    with torch.no_grad():
        img = image.detach().cpu().float()

    if img.dim() == 4:
        if img.size(0) != 1:
            raise ValueError("Tensor with batch dimension must have batch size 1 for NR metrics")
        img = img[0]  # C,H,W

    if img.dim() == 2:  # H,W
        img = img.unsqueeze(0)  # 1,H,W
    elif img.dim() == 3:
        if img.shape[0] in (1, 3):  # already C,H,W
            pass
        elif img.shape[2] in (1, 3):  # H,W,C -> transpose
            img = img.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported channel layout {img.shape} for NR metrics")
    else:
        raise ValueError(f"Unsupported tensor shape {img.shape} for NR metrics")

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)  # pyiqa expects 3-channel by default
    elif img.shape[0] > 3:
        img = img[:3]

    img = torch.clamp(img, 0, 1).unsqueeze(0)  # B,C,H,W
    return img


def _compute_pyiqa_metric(metric_name: str, image: torch.Tensor) -> float:
    """Compute NR metric via pyiqa if available."""
    if not _PYIQA_AVAILABLE:
        return float("nan")
    try:
        if metric_name not in _PYIQA_MODELS:
            model = pyiqa.create_metric(metric_name, as_loss=False, device="cpu")
            model.eval()
            _PYIQA_MODELS[metric_name] = model
        else:
            model = _PYIQA_MODELS[metric_name]
        tensor = _prepare_tensor_for_pyiqa(image)
        with torch.no_grad():
            score = model(tensor).item()
        return float(score)
    except Exception:
        return float("nan")


def compute_niqe_score(image: torch.Tensor) -> float:
    """Compute NIQE (Naturalness Image Quality Evaluator) score for a single image tensor."""
    if _SKIMAGE_IQA_AVAILABLE:
        try:
            gray = _prepare_tensor_for_iqa(image)
            return float(skimage_niqe(gray))
        except Exception:
            pass
    return _compute_pyiqa_metric("niqe", image)


def compute_brisque_score(image: torch.Tensor) -> float:
    """Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score."""
    if _SKIMAGE_IQA_AVAILABLE:
        try:
            gray = _prepare_tensor_for_iqa(image)
            return float(skimage_brisque(gray))
        except Exception:
            pass
    return _compute_pyiqa_metric("brisque", image)


def compute_piqe_score(image: torch.Tensor) -> float:
    """Compute PIQE (Perception-based Image Quality Evaluator) score."""
    if _SKIMAGE_IQA_AVAILABLE:
        try:
            gray = _prepare_tensor_for_iqa(image)
            return float(skimage_piqe(gray))
        except Exception:
            pass
    return _compute_pyiqa_metric("piqe", image)

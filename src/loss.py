"""losses for all stages"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class RefinementLoss(nn.Module):
    """Loss for refinement stage - encourages improvement without matching target."""
    def __init__(self, identity_penalty=10.0, perceptual_weight=1.0, tv_weight=0.1, clip_model=None):
        super().__init__()
        self.identity_penalty = identity_penalty  # Very large penalty to prevent copying
        self.perceptual_weight = perceptual_weight  # Encourage better CLIP features
        self.tv_weight = tv_weight  # Total variation for smoothness
        self.clip_model = clip_model
        
        # Use CLIP for perceptual loss if available
        if perceptual_weight > 0 and clip_model is not None:
            self.feature_extractor = clip_model.visual
            self.use_clip_perceptual = True
        else:
            self.feature_extractor = None
            self.use_clip_perceptual = False
    
    def total_variation_loss(self, img):
        """Total variation loss for smoothness."""
        batch_size = img.size(0)
        h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / batch_size
    
    def forward(self, refined: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """
        refined: [B, C, H, W] refined image
        coarse: [B, C, H, W] coarse image
        """
        # Small structure loss to maintain basic structure (not exact match)
        structure_loss = F.l1_loss(refined, coarse) * 0.1
        
        # Large identity penalty to prevent copying
        l1_diff = F.l1_loss(refined, coarse)
        # Penalty is high when images are very similar (identity mapping)
        # Use a penalty that's high when l1_diff < threshold
        identity_penalty = self.identity_penalty * torch.clamp(0.01 - l1_diff, min=0.0) / 0.01
        
        # Total variation loss for smoothness (encourages natural images)
        tv_loss = self.total_variation_loss(refined) * self.tv_weight
        
        # CLIP perceptual loss - encourage refined to have better CLIP features than coarse
        perceptual_loss = 0.0
        if self.perceptual_weight > 0 and self.feature_extractor is not None:
            # Normalize and resize for CLIP
            refined_norm = F.interpolate(torch.clamp(refined, 0, 1), size=(224, 224), mode='bilinear', align_corners=False)
            coarse_norm = F.interpolate(torch.clamp(coarse, 0, 1), size=(224, 224), mode='bilinear', align_corners=False)
            
            # Ensure dtype matches CLIP model
            if self.clip_model is not None:
                model_dtype = next(self.clip_model.parameters()).dtype
                refined_norm = refined_norm.to(dtype=model_dtype)
                coarse_norm = coarse_norm.to(dtype=model_dtype)
            
            # Get CLIP features
            refined_features = self.feature_extractor(refined_norm)
            coarse_features = self.feature_extractor(coarse_norm)
            
            # Normalize features
            refined_features = F.normalize(refined_features, dim=-1)
            coarse_features = F.normalize(coarse_features, dim=-1)
            
            # Perceptual loss: encourage refined to have better features (not match, but improve)
            # Use cosine similarity - we want refined to be similar but better
            # Actually, use a loss that encourages refined features to be "sharper" or "better"
            # For now, use MSE but with a small weight to encourage improvement
            perceptual_loss = F.mse_loss(refined_features, coarse_features) * self.perceptual_weight * 0.1
        
        # Total loss: structure (maintain) + identity penalty (prevent copy) + TV (smooth) + perceptual (improve)
        total_loss = structure_loss + identity_penalty + tv_loss + perceptual_loss
        return total_loss

class ReconstructionLoss(nn.Module):
    """L1 + L2 loss for image reconstruction with optional perceptual and semantic loss."""
    def __init__(self, l1_weight=1.0, l2_weight=1.0, identity_penalty=0.0, perceptual_weight=0.0, 
                 semantic_weight=0.0, clip_model=None, labels=None):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.identity_penalty = identity_penalty
        self.perceptual_weight = perceptual_weight
        self.semantic_weight = semantic_weight
        self.clip_model = clip_model
        self.labels = labels
        
        # Use CLIP for perceptual loss if available (better than simple conv layers)
        if perceptual_weight > 0 and clip_model is not None:
            # Use CLIP's image encoder for perceptual loss (like reference code)
            self.feature_extractor = clip_model.visual
            self.use_clip_perceptual = True
        elif perceptual_weight > 0:
            # Fallback: simple conv layer to extract features
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.use_clip_perceptual = False
        else:
            self.feature_extractor = None
            self.use_clip_perceptual = False
        
        # Pre-compute text embeddings for semantic loss if labels are provided
        if semantic_weight > 0 and clip_model is not None and labels is not None:
            with torch.no_grad():
                # Create text prompts like "a photo of a {label}"
                text_prompts = [f"a photo of a {label}" for label in labels]
                text_tokens = clip.tokenize(text_prompts)
                # Store on CPU, will move to device when needed
                self.text_tokens = text_tokens
                self.text_embeddings = None  # Will be computed on first forward with device
                self.labels_list = labels
            self.use_semantic = True
        else:
            self.use_semantic = False
            self.text_tokens = None
            self.text_embeddings = None
            self.labels_list = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, label_indices: torch.Tensor = None) -> torch.Tensor:
        """
        pred: [B, C, H, W] predicted image
        target: [B, C, H, W] target image
        label_indices: [B] optional label indices for semantic alignment
        """
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        reconstruction_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        # Add perceptual loss to encourage meaningful feature learning
        if self.perceptual_weight > 0 and self.feature_extractor is not None:
            if self.use_clip_perceptual:
                # Use CLIP features for perceptual loss (like reference code)
                # Normalize images to CLIP's expected range [0, 1] and resize to 224x224
                pred_normalized = F.interpolate(torch.clamp(pred, 0, 1), size=(224, 224), mode='bilinear', align_corners=False)
                target_normalized = F.interpolate(torch.clamp(target, 0, 1), size=(224, 224), mode='bilinear', align_corners=False)
                
                # Extract CLIP features (CLIP model is frozen, but gradients flow through images)
                pred_features = self.feature_extractor(pred_normalized)  # [B, clip_dim]
                target_features = self.feature_extractor(target_normalized)  # [B, clip_dim]
                
                # Normalize features
                pred_features = F.normalize(pred_features, dim=-1)
                target_features = F.normalize(target_features, dim=-1)
                
                # Perceptual loss: encourage images to have meaningful CLIP features
                perceptual_loss = F.mse_loss(pred_features, target_features)
            else:
                # Fallback: simple feature extractor
                pred_features = self.feature_extractor(pred)
                target_features = self.feature_extractor(target)
                perceptual_loss = F.mse_loss(pred_features, target_features)
            
            reconstruction_loss = reconstruction_loss + self.perceptual_weight * perceptual_loss
        
        # Add semantic alignment loss to improve semantic accuracy
        # This uses CLIP to ensure reconstructed images are semantically aligned with text labels
        if self.semantic_weight > 0 and self.use_semantic and label_indices is not None and self.clip_model is not None:
            # Normalize predicted images for CLIP
            pred_normalized = F.interpolate(torch.clamp(pred, 0, 1), size=(224, 224), mode='bilinear', align_corners=False)
            
            # Get image features from CLIP
            image_features = self.clip_model.encode_image(pred_normalized)  # [B, clip_dim]
            image_features = F.normalize(image_features, dim=-1)
            
            # Get text features for the labels
            # Compute text embeddings on the fly (they're cached but we need them on the right device)
            if self.text_embeddings is None or self.text_embeddings.device != pred.device:
                text_tokens = self.text_tokens.to(pred.device)
                with torch.no_grad():
                    self.text_embeddings = self.clip_model.encode_text(text_tokens)  # [num_classes, clip_dim]
                    self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)
            
            # Select text features for the current batch labels
            text_features = self.text_embeddings[label_indices]  # [B, clip_dim]
            
            # Semantic loss: maximize similarity between image and text features
            # Use negative cosine similarity (we want high similarity, so minimize negative similarity)
            similarities = (image_features * text_features).sum(dim=1)  # [B] cosine similarity
            semantic_loss = -similarities.mean()  # Negative because we want to maximize similarity
            
            reconstruction_loss = reconstruction_loss + self.semantic_weight * semantic_loss
        
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
        # Extract clip_model and labels if provided
        clip_model = filtered_kwargs.pop('clip_model', None)
        labels = filtered_kwargs.pop('labels', None)
        filtered_kwargs['clip_model'] = clip_model
        filtered_kwargs['labels'] = labels
        return ReconstructionLoss(**filtered_kwargs)
    elif loss_type == "refinement":
        # Use RefinementLoss for Stage 3
        clip_model = kwargs.pop('clip_model', None)
        return RefinementLoss(clip_model=clip_model, **kwargs)
    elif loss_type == "clip":
        return CLIPLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

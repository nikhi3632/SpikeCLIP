"""Stage 3 training"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import argparse
import time
from tqdm import tqdm

from config_loader import load_config
from data_loader import get_loader
from models.coarse_reconstruction import CoarseSNN
from models.refinement import RefinementNet
from models.prompt_learning import HQ_LQ_PromptAdapter
from loss import get_loss_fn
import clip
from training.trainer_utils import Trainer
from training.optimizer_factory import build_optimizer, build_scheduler
from utils.helpers import get_device, set_seed
from utils.checkpointing import load_best_checkpoint
from utils.logging import log_device_info

class RefineTrainer(Trainer):
    """Custom trainer for refinement stage."""
    
    def train_epoch(self) -> tuple:
        """Train for one epoch. Returns average loss and GPU metrics."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # GPU metrics tracking
        latencies = []
        total_samples = 0
        
        if self.track_gpu_metrics:
            epoch_start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            spikes = batch[0].to(self.device)  # [B, T, H, W]
            label_indices = batch[2].to(self.device) if len(batch) > 2 else None  # [B] optional labels
            batch_size = spikes.size(0)
            
            # Track latency
            if self.track_gpu_metrics:
                torch.cuda.synchronize()
                batch_start = time.time()
            
            self.optimizer.zero_grad()
            
            # Stage 1: Get coarse images (frozen)
            with torch.no_grad():
                coarse_images = self.coarse_model(spikes)  # [B, 3, H, W]
            
            # Stage 3: Refine images
            # According to paper: L_total = L_class + λ*L_prompt (λ=100)
            # Stage 3 uses Stage 2's learned HQ/LQ prompts for prompt loss
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    refined_images = self.model(coarse_images)  # [B, 3, H, W]
                    
                    # Get image features from refined images using CLIP directly
                    # Normalize images for CLIP
                    refined_normalized = F.interpolate(refined_images, size=(224, 224), mode='bilinear', align_corners=False)
                    refined_normalized = torch.clamp(refined_normalized, 0, 1)
                    image_features = self.clip_model.encode_image(refined_normalized)  # [B, clip_dim]
                    image_features = F.normalize(image_features, dim=-1)
                    
                    # Prompt Loss: Alignment with HQ prompts from Stage 2
                    # Uses learned HQ/LQ prompts from Stage 2 (hq_lq_prompt_model)
                    hq_prompt_features, lq_prompt_features = self.hq_lq_prompt_model.get_prompt_features()
                    prompt_loss = self.prompt_criterion(image_features, hq_prompt_features, lq_prompt_features)
                    
                    # Class Loss: InfoNCE loss for classification
                    # Use CLIP text features directly (according to paper: "class-label features")
                    # text_features is [num_classes, clip_dim] - pre-computed in __init__
                    class_loss = self.class_criterion(image_features, self.text_features, label_indices)
                    
                    # Identity penalty: penalize when refined == coarse (identity mapping)
                    # This encourages actual refinement, not just copying
                    l1_diff = F.l1_loss(refined_images, coarse_images)
                    identity_penalty = self.identity_penalty * torch.exp(-l1_diff * 50.0)
                    
                    # Total loss: α*L_class + λ*L_prompt + identity_penalty
                    # This is why Stage 3 depends on Stage 2: it needs the learned prompts
                    # α (class_loss_weight) emphasizes class discrimination
                    # λ (prompt_weight) emphasizes HQ alignment
                    loss = self.class_loss_weight * class_loss + self.prompt_weight * prompt_loss + identity_penalty
                
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                refined_images = self.model(coarse_images)
                
                # Get image features from refined images using CLIP directly
                # Normalize images for CLIP
                refined_normalized = F.interpolate(refined_images, size=(224, 224), mode='bilinear', align_corners=False)
                refined_normalized = torch.clamp(refined_normalized, 0, 1)
                image_features = self.clip_model.encode_image(refined_normalized)  # [B, clip_dim]
                image_features = F.normalize(image_features, dim=-1)
                
                # Prompt Loss: Alignment with HQ prompts from Stage 2
                # Uses learned HQ/LQ prompts from Stage 2 (hq_lq_prompt_model)
                hq_prompt_features, lq_prompt_features = self.hq_lq_prompt_model.get_prompt_features()
                prompt_loss = self.prompt_criterion(image_features, hq_prompt_features, lq_prompt_features)
                
                # Class Loss: InfoNCE loss for classification
                # Use CLIP text features directly (according to paper: "class-label features")
                # text_features is [num_classes, clip_dim] - pre-computed in __init__
                class_loss = self.class_criterion(image_features, self.text_features, label_indices)
                
                # Identity penalty: penalize when refined == coarse (identity mapping)
                # This encourages actual refinement, not just copying
                l1_diff = F.l1_loss(refined_images, coarse_images)
                identity_penalty = self.identity_penalty * torch.exp(-l1_diff * 50.0)
                
                # Total loss: α*L_class + λ*L_prompt + identity_penalty
                # This is why Stage 3 depends on Stage 2: it needs the learned prompts
                # α (class_loss_weight) emphasizes class discrimination
                # λ (prompt_weight) emphasizes HQ alignment
                loss = self.class_loss_weight * class_loss + self.prompt_weight * prompt_loss + identity_penalty
                
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Track latency
            if self.track_gpu_metrics:
                torch.cuda.synchronize()
                batch_time = time.time() - batch_start
                latencies.append(batch_time)
                total_samples += batch_size
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        # Compute GPU metrics
        if self.track_gpu_metrics:
            epoch_time = time.time() - epoch_start_time
            from utils.performance import get_gpu_metrics
            gpu_metrics = get_gpu_metrics(latencies, total_samples, epoch_time)
        else:
            gpu_metrics = {}
        
        return avg_loss, gpu_metrics
    
    def validate(self) -> float:
        """Validate model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                spikes = batch[0].to(self.device)
                label_indices = batch[2].to(self.device) if len(batch) > 2 else None  # [B] optional labels
                
                # Stage 1: Get coarse images
                coarse_images = self.coarse_model(spikes)
                
                # Stage 3: Refine and compute loss
                refined_images = self.model(coarse_images)
                
                # Get image features from refined images using CLIP directly
                # Normalize images for CLIP
                refined_normalized = F.interpolate(refined_images, size=(224, 224), mode='bilinear', align_corners=False)
                refined_normalized = torch.clamp(refined_normalized, 0, 1)
                image_features = self.clip_model.encode_image(refined_normalized)  # [B, clip_dim]
                image_features = F.normalize(image_features, dim=-1)
                
                # Prompt Loss: Alignment with HQ prompts from Stage 2
                # Uses learned HQ/LQ prompts from Stage 2 (hq_lq_prompt_model)
                hq_prompt_features, lq_prompt_features = self.hq_lq_prompt_model.get_prompt_features()
                prompt_loss = self.prompt_criterion(image_features, hq_prompt_features, lq_prompt_features)
                
                # Class Loss: InfoNCE loss for classification
                # Use CLIP text features directly (according to paper: "class-label features")
                # text_features is [num_classes, clip_dim] - pre-computed in __init__
                class_loss = self.class_criterion(image_features, self.text_features, label_indices)
                
                # Identity penalty: penalize when refined == coarse (identity mapping)
                # This encourages actual refinement, not just copying
                l1_diff = F.l1_loss(refined_images, coarse_images)
                identity_penalty = self.identity_penalty * torch.exp(-l1_diff * 50.0)
                
                # Total loss: α*L_class + λ*L_prompt + identity_penalty
                # This is why Stage 3 depends on Stage 2: it needs the learned prompts
                # α (class_loss_weight) emphasizes class discrimination
                # λ (prompt_weight) emphasizes HQ alignment
                loss = self.class_loss_weight * class_loss + self.prompt_weight * prompt_loss + identity_penalty
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train Stage 3: Refinement')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/checkpoints/ucaltech', help='Output directory')
    parser.add_argument('--coarse-checkpoint', type=str, required=True, 
                       help='Path to coarse model checkpoint directory (Stage 1 output)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_dir = config['data_dir']
    labels = config['labels']
    val_split_ratio = config.get('val_split_ratio', 0.2)
    
    # Refinement stage config (with defaults)
    refine_config = config.get('refine', {})
    model_config = refine_config.get('model', {})
    loss_config = refine_config.get('loss', {})
    training_config = refine_config.get('training', {})
    optimizer_config = refine_config.get('optimizer', {})
    scheduler_config = refine_config.get('scheduler', {})
    output_config = refine_config.get('output', {})
    
    # Training parameters (command line args override config)
    epochs = training_config.get('epochs', 50) if args.epochs == 50 else args.epochs
    batch_size = training_config.get('batch_size', 8) if args.batch_size == 8 else args.batch_size
    learning_rate = optimizer_config.get('lr', training_config.get('learning_rate', 0.001)) if args.lr == 1e-3 else args.lr
    seed = training_config.get('seed', 42) if args.seed == 42 else args.seed
    use_amp = args.use_amp or training_config.get('use_amp', False)
    grad_clip = training_config.get('grad_clip', 1.0)
    output_dir = output_config.get('checkpoint_dir', 'outputs/checkpoints/ucaltech') if args.output_dir == 'outputs/checkpoints/ucaltech' else args.output_dir
    
    # Set seed
    set_seed(seed)
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Validation split ratio: {val_split_ratio}")
    print(f"Refine Config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, seed={seed}")
    
    # Log device info
    log_dir = output_config.get('log_dir', 'outputs/logs')
    log_device_info(output_dir=log_dir)
    
    # Data loader config
    data_loader_config = config.get('data_loader', {})
    num_workers = data_loader_config.get('num_workers', None)
    pin_memory = data_loader_config.get('pin_memory', None)
    
    # Data loaders
    train_loader = get_loader(
        data_dir, labels, split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        val_split_ratio=val_split_ratio,
        seed=seed,
        device=device,
        pin_memory=pin_memory
    )
    val_loader = get_loader(
        data_dir, labels, split='val',
        batch_size=batch_size,
        num_workers=num_workers,
        val_split_ratio=val_split_ratio,
        seed=seed,
        device=device,
        pin_memory=pin_memory
    )
    
    # Stage 1 model (frozen)
    coarse_model = CoarseSNN(
        time_steps=config.get('coarse', {}).get('model', {}).get('time_steps', 25),
        v_threshold=config.get('coarse', {}).get('model', {}).get('v_threshold', 1.0),
        tau=config.get('coarse', {}).get('model', {}).get('tau', 2.0)
    )
    
    # Load coarse checkpoint
    coarse_checkpoint_dir = Path(args.coarse_checkpoint)
    if not coarse_checkpoint_dir.exists():
        raise FileNotFoundError(f"Coarse checkpoint directory not found: {coarse_checkpoint_dir}")
    
    print(f"Loading coarse model from {coarse_checkpoint_dir}")
    try:
        load_best_checkpoint(
            str(coarse_checkpoint_dir),
            coarse_model,
            device=device,
            prefix='coarse'
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load coarse checkpoint: {e}. Make sure Stage 1 is trained first.")
    coarse_model.eval()
    for param in coarse_model.parameters():
        param.requires_grad = False
    
    # Stage 3 model
    # Set use_identity=False to actually train the UNet (not just identity)
    model = RefinementNet(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 3),
        base_channels=model_config.get('base_channels', 64),
        num_down=model_config.get('num_down', 4),
        use_identity=False  # Train the UNet, not just identity
    )
    
    # Load CLIP model for image encoding and text features (according to paper)
    print("Loading CLIP model for Stage 3...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()  # Ensure float32
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # Pre-compute CLIP text features for all classes (according to paper: "class-label features")
    print("Pre-computing CLIP text features for all classes...")
    with torch.no_grad():
        text_prompts = [f"a photo of a {label}" for label in labels]
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)  # [num_classes, clip_dim]
        text_features = F.normalize(text_features, dim=-1)
    
    # Load HQ/LQ prompt model from Stage 2 (for prompt loss)
    # Stage 3 depends on Stage 2 because:
    # 1. Stage 3 uses HQ/LQ prompts learned in Stage 2 for prompt loss
    # 2. According to paper: L_total = L_class + λ*L_prompt (λ=100)
    # 3. The prompt loss aligns refined images with HQ prompts from Stage 2
    prompt_checkpoint_dir = Path(output_config.get('checkpoint_dir', 'outputs/checkpoints/ucaltech'))
    hq_lq_prompt_model = HQ_LQ_PromptAdapter(
        clip_model_name=config.get('prompt', {}).get('model', {}).get('clip_model_name', 'ViT-B/32'),
        prompt_dim=config.get('prompt', {}).get('model', {}).get('prompt_dim', 77),
        freeze_image_encoder=True
    )
    
    print(f"Loading HQ/LQ prompt model from Stage 2: {prompt_checkpoint_dir}")
    print("  (Stage 3 needs Stage 2's learned HQ/LQ prompts for prompt loss)")
    try:
        load_best_checkpoint(
            str(prompt_checkpoint_dir),
            hq_lq_prompt_model,
            device=device,
            prefix='prompt'
        )
        print("  ✓ Successfully loaded Stage 2 prompt checkpoint")
    except FileNotFoundError as e:
        print(f"  ⚠️  Warning: Failed to load HQ/LQ prompt checkpoint ({e})")
        print("  ⚠️  Using random initialization (Stage 2 must be trained first!)")
    hq_lq_prompt_model.eval()
    for param in hq_lq_prompt_model.parameters():
        param.requires_grad = False
    
    # Loss functions according to paper: L_total = L_class + λ*L_prompt (λ=100)
    prompt_criterion = get_loss_fn(
        'prompt',  # Prompt loss: alignment with HQ prompts
        temperature=loss_config.get('temperature', 0.07)
    )
    class_criterion = get_loss_fn(
        'info_nce',  # InfoNCE loss: classification
        temperature=loss_config.get('temperature', 0.07)
    )
    prompt_weight = loss_config.get('prompt_weight', 0.5)  # Default: 0.5 (balanced with class loss)
    
    # Dummy criterion for compatibility (not used)
    criterion = prompt_criterion
    
    # Optimizer
    optimizer_cfg = optimizer_config.copy()
    optimizer_cfg['lr'] = learning_rate
    optimizer = build_optimizer(model, optimizer_cfg)
    
    # Scheduler
    scheduler = build_scheduler(optimizer, scheduler_config, epochs)
    
    # Trainer
    checkpoint_dir = Path(output_dir)
    early_stopping_patience = training_config.get('early_stopping_patience', None)
    early_stopping_min_delta = training_config.get('early_stopping_min_delta', 0.0)
    trainer = RefineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        use_amp=use_amp,
        grad_clip=grad_clip,
        checkpoint_prefix='refine',
        log_dir=log_dir,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta
    )
    trainer.coarse_model = coarse_model.to(device)
    trainer.hq_lq_prompt_model = hq_lq_prompt_model.to(device)
    trainer.clip_model = clip_model.to(device)
    trainer.text_features = text_features.to(device)  # Pre-computed CLIP text features
    trainer.prompt_criterion = prompt_criterion.to(device)
    trainer.class_criterion = class_criterion.to(device)
    trainer.prompt_weight = prompt_weight
    trainer.class_loss_weight = loss_config.get('class_loss_weight', 3.0)  # Weight for class loss (emphasize classification)
    trainer.identity_penalty = loss_config.get('identity_penalty', 2.0)  # Penalty for identity mapping
    trainer.labels = labels
    
    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)
    
    # Train
    trainer.train(epochs, scheduler=scheduler)

if __name__ == '__main__':
    main()

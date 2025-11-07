"""Stage 3 training"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import time
from tqdm import tqdm

from config_loader import load_config
from data_loader import get_loader
from models.coarse_reconstruction import CoarseSNN
from models.refinement import RefinementNet
from loss import get_loss_fn
from training.trainer_utils import Trainer
from training.optimizer_factory import build_optimizer, build_scheduler
from utils.helpers import get_device, set_seed
from utils.checkpointing import load_best_checkpoint
from utils.logging import log_device_info
import clip

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
            # Use RefinementLoss that encourages improvement without matching target
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    refined_images = self.model(coarse_images)  # [B, 3, H, W]
                    # RefinementLoss uses coarse as structure constraint, not target
                    loss = self.criterion(refined_images, coarse_images)
                
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                refined_images = self.model(coarse_images)
                # RefinementLoss uses coarse as structure constraint, not target
                loss = self.criterion(refined_images, coarse_images)
                
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
                
                # Stage 1: Get coarse images
                coarse_images = self.coarse_model(spikes)
                
                # Stage 3: Refine and compute loss
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        refined_images = self.model(coarse_images)
                        # RefinementLoss uses coarse as structure constraint, not target
                        loss = self.criterion(refined_images, coarse_images)
                else:
                    refined_images = self.model(coarse_images)
                    # RefinementLoss uses coarse as structure constraint, not target
                    loss = self.criterion(refined_images, coarse_images)
                
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
    parser.add_argument('--coarse-checkpoint', type=str, required=True, help='Path to coarse model checkpoint directory')
    
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
    model = RefinementNet(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 3),
        base_channels=model_config.get('base_channels', 64),
        num_down=model_config.get('num_down', 4)
    )
    
    # Load CLIP model for CLIP-guided perceptual loss (like reference code)
    clip_model = None
    if loss_config.get('perceptual_weight', 0.0) > 0:
        try:
            print("Loading CLIP model for perceptual loss...")
            clip_model, _ = clip.load("ViT-B/32", device=device)
            # Ensure CLIP model is in float32 (not half precision) to avoid dtype mismatches
            clip_model = clip_model.float()
            clip_model.eval()
            # Freeze CLIP model
            for param in clip_model.parameters():
                param.requires_grad = False
            print("CLIP model loaded successfully for CLIP-guided perceptual loss")
        except Exception as e:
            print(f"Warning: Failed to load CLIP model for perceptual loss ({e}). Using fallback feature extractor.")
            clip_model = None
    
    # Use RefinementLoss for Stage 3 (encourages improvement without matching target)
    criterion = get_loss_fn(
        loss_config.get('type', 'refinement'),  # Use 'refinement' loss type
        identity_penalty=loss_config.get('identity_penalty', 10.0),  # Very large penalty to prevent copying
        perceptual_weight=loss_config.get('perceptual_weight', 1.0),  # CLIP perceptual loss for refinement
        tv_weight=loss_config.get('tv_weight', 0.1),  # Total variation for smoothness
        clip_model=clip_model  # Pass CLIP model for CLIP-guided perceptual loss
    )
    # Ensure loss function (and its feature extractor) is on the correct device
    criterion = criterion.to(device)
    
    # Optimizer
    optimizer_cfg = optimizer_config.copy()
    optimizer_cfg['lr'] = learning_rate
    optimizer = build_optimizer(model, optimizer_cfg)
    
    # Scheduler
    scheduler = build_scheduler(optimizer, scheduler_config, epochs)
    
    # Trainer
    checkpoint_dir = Path(output_dir)
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
        log_dir=log_dir
    )
    trainer.coarse_model = coarse_model.to(device)
    
    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)
    
    # Train
    trainer.train(epochs, scheduler=scheduler)

if __name__ == '__main__':
    main()

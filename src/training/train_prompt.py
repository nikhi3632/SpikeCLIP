"""Stage 2 training"""
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
from models.prompt_learning import HQ_LQ_PromptAdapter
from loss import get_loss_fn
from training.trainer_utils import Trainer
from training.optimizer_factory import build_optimizer, build_scheduler
from utils.helpers import get_device, set_seed
from utils.checkpointing import load_best_checkpoint
from utils.logging import log_device_info

class PromptTrainer(Trainer):
    """Custom trainer for prompt learning stage."""
    
    def train_epoch(self) -> tuple:
        """Train for one epoch. Returns average loss and GPU metrics."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # GPU metrics tracking
        latencies = []
        total_samples = 0
        epoch_start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        epoch_end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if self.track_gpu_metrics:
            import time
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
            
            # Stage 1: Get coarse images (LQ images)
            with torch.no_grad():
                lq_images = self.coarse_model(spikes)  # [B, 3, H, W]
            
            # Generate HQ images according to the paper
            # According to paper: HQ images from generation pipeline
            # - For synthetic data: WGSE (Weighted Gradient Spike Encoding)
            # - For real data: Mixture of multiple reconstruction methods
            # We use mixture approach that combines TFI, weighted temporal average, and spike count
            with torch.no_grad():
                from utils.hq_generation import generate_hq_images
                hq_method = getattr(self, 'hq_generation_method', 'mixture')
                hq_images = generate_hq_images(spikes, method=hq_method)  # [B, 3, H, W]
            
            # Combine HQ and LQ images: [2*B, 3, H, W]
            all_images = torch.cat([hq_images, lq_images], dim=0)
            # Labels: 1 for HQ, 0 for LQ
            labels = torch.cat([
                torch.ones(batch_size, dtype=torch.long, device=self.device),  # HQ
                torch.zeros(batch_size, dtype=torch.long, device=self.device)  # LQ
            ], dim=0)
            
            # Stage 2: Get image features and HQ/LQ prompt features
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    image_features, hq_prompt_features, lq_prompt_features = self.model(all_images)
                    loss = self.criterion(image_features, hq_prompt_features, lq_prompt_features, labels)
                
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                image_features, hq_prompt_features, lq_prompt_features = self.model(all_images)
                loss = self.criterion(image_features, hq_prompt_features, lq_prompt_features, labels)
                
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Track classification accuracy (for monitoring)
            with torch.no_grad():
                hq_sim = (image_features @ hq_prompt_features.unsqueeze(0).t()).squeeze(-1)
                lq_sim = (image_features @ lq_prompt_features.unsqueeze(0).t()).squeeze(-1)
                logits = torch.stack([lq_sim, hq_sim], dim=1)
                preds = logits.argmax(dim=1)
                batch_acc = (preds == labels).float().mean().item()
                if batch_idx == 0:  # Log first batch accuracy occasionally
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.3f}'})
            
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
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                spikes = batch[0].to(self.device)
                batch_size = spikes.size(0)
                
                # Stage 1: Get coarse images (LQ images)
                lq_images = self.coarse_model(spikes)
                
                # Generate HQ images according to the paper
                from utils.hq_generation import generate_hq_images
                hq_method = getattr(self, 'hq_generation_method', 'mixture')
                hq_images = generate_hq_images(spikes, method=hq_method)
                
                # Combine HQ and LQ images
                all_images = torch.cat([hq_images, lq_images], dim=0)
                labels = torch.cat([
                    torch.ones(batch_size, dtype=torch.long, device=self.device),  # HQ
                    torch.zeros(batch_size, dtype=torch.long, device=self.device)  # LQ
                ], dim=0)
                
                # Stage 2: Get features and compute loss
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        image_features, hq_prompt_features, lq_prompt_features = self.model(all_images)
                        loss = self.criterion(image_features, hq_prompt_features, lq_prompt_features, labels)
                else:
                    image_features, hq_prompt_features, lq_prompt_features = self.model(all_images)
                    loss = self.criterion(image_features, hq_prompt_features, lq_prompt_features, labels)
                
                # Compute classification accuracy
                hq_similarity = (image_features @ hq_prompt_features.unsqueeze(0).t()).squeeze(-1)  # [2*B]
                lq_similarity = (image_features @ lq_prompt_features.unsqueeze(0).t()).squeeze(-1)  # [2*B]
                logits = torch.stack([lq_similarity, hq_similarity], dim=1)  # [2*B, 2]
                predictions = logits.argmax(dim=1)  # [2*B]
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Log accuracy if available
        if hasattr(self, 'log_dir') and self.log_dir:
            import os
            log_file = os.path.join(self.log_dir, 'prompt_train_log.txt')
            with open(log_file, 'a') as f:
                f.write(f"Epoch {self.current_epoch}: Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})\n")
        
        return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train Stage 2: Prompt Learning')
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
    
    # Prompt stage config (with defaults)
    prompt_config = config.get('prompt', {})
    model_config = prompt_config.get('model', {})
    loss_config = prompt_config.get('loss', {})
    training_config = prompt_config.get('training', {})
    optimizer_config = prompt_config.get('optimizer', {})
    scheduler_config = prompt_config.get('scheduler', {})
    output_config = prompt_config.get('output', {})
    
    # Training parameters (command line args override config)
    epochs = training_config.get('epochs', 50) if args.epochs == 50 else args.epochs
    batch_size = training_config.get('batch_size', 8) if args.batch_size == 8 else args.batch_size
    learning_rate = optimizer_config.get('lr', training_config.get('learning_rate', 0.0001)) if args.lr == 1e-3 else args.lr
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
    print(f"Prompt Config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, seed={seed}")
    
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
    
    # Stage 2 model: HQ/LQ Prompt Adapter (according to paper)
    model = HQ_LQ_PromptAdapter(
        clip_model_name=model_config.get('clip_model_name', 'ViT-B/32'),
        prompt_dim=model_config.get('prompt_dim', 77),
        freeze_image_encoder=model_config.get('freeze_image_encoder', True)
    )
    
    # Loss: HQ/LQ Prompt Loss (binary classification)
    criterion = get_loss_fn(
        loss_config.get('type', 'hq_lq_prompt'),
        temperature=loss_config.get('temperature', 0.07)
    )
    
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
    trainer = PromptTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        use_amp=use_amp,
        grad_clip=grad_clip,
        checkpoint_prefix='prompt',
        log_dir=log_dir,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta
    )
    trainer.coarse_model = coarse_model.to(device)
    # Set HQ generation method from config
    trainer.hq_generation_method = model_config.get('hq_generation_method', 'mixture')
    
    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)
    
    # Train
    trainer.train(epochs, scheduler=scheduler)

if __name__ == '__main__':
    main()

"""shared Trainer class, resume, AMP"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Any, Dict, Tuple
from pathlib import Path
import time

from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.performance import get_gpu_metrics, reset_memory_stats
from utils.logging import log_training_metrics

class Trainer:
    """Base trainer class with resume and AMP support."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str,
        use_amp: bool = False,
        grad_clip: Optional[float] = None,
        log_interval: int = 10,
        checkpoint_prefix: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.checkpoint_prefix = checkpoint_prefix
        
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.log_dir = log_dir or str(Path(checkpoint_dir).parent / 'logs')
        self.stage = checkpoint_prefix or 'coarse'
        
        # GPU metrics tracking
        self.track_gpu_metrics = device.type == 'cuda'
        if self.track_gpu_metrics:
            reset_memory_stats()
        
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch. Returns average loss and GPU metrics."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # GPU metrics tracking
        latencies = []
        total_samples = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Get batch data (adjust based on your data format)
            spikes = batch[0].to(self.device)  # [B, T, H, W]
            batch_size = spikes.size(0)
            
            # Track latency
            if self.track_gpu_metrics:
                torch.cuda.synchronize()
                batch_start = time.time()
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(spikes)  # [B, 3, H, W]
                    # For unpaired training: use self-supervised reconstruction
                    # Target is a normalized version of spike temporal average
                    # This encourages the model to learn meaningful image structure from spikes
                    spike_avg = spikes.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                    # Use global normalization across batch to make task more challenging
                    # This preserves relative intensity differences between samples
                    spike_min = spike_avg.min()
                    spike_max = spike_avg.max()
                    spike_range = spike_max - spike_min + 1e-8  # Avoid division by zero
                    spike_avg = (spike_avg - spike_min) / spike_range  # Global normalization
                    spike_avg = torch.clamp(spike_avg, 0, 1)
                    target = spike_avg.repeat(1, 3, 1, 1)
                    loss = self.criterion(outputs, target)
                
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(spikes)
                # Self-supervised target: normalized spike average with global scaling
                # Global normalization makes task more challenging and preserves intensity relationships
                spike_avg = spikes.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                # Global normalization across batch to preserve relative intensities
                spike_min = spike_avg.min()
                spike_max = spike_avg.max()
                spike_range = spike_max - spike_min + 1e-8  # Avoid division by zero
                spike_avg = (spike_avg - spike_min) / spike_range  # Global normalization
                spike_avg = torch.clamp(spike_avg, 0, 1)
                target = spike_avg.repeat(1, 3, 1, 1)
                loss = self.criterion(outputs, target)
                
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
        epoch_time = time.time() - epoch_start_time
        if self.track_gpu_metrics:
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
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(spikes)
                        # Self-supervised target: normalized spike average with global scaling
                        spike_avg = spikes.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                        # Global normalization across batch to preserve relative intensities
                        spike_min = spike_avg.min()
                        spike_max = spike_avg.max()
                        spike_range = spike_max - spike_min + 1e-8
                        spike_avg = (spike_avg - spike_min) / spike_range
                        spike_avg = torch.clamp(spike_avg, 0, 1)
                        target = spike_avg.repeat(1, 3, 1, 1)
                        loss = self.criterion(outputs, target)
                else:
                    outputs = self.model(spikes)
                    # Self-supervised target: normalized spike average with global scaling
                    spike_avg = spikes.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                    # Global normalization across batch to preserve relative intensities
                    spike_min = spike_avg.min()
                    spike_max = spike_avg.max()
                    spike_range = spike_max - spike_min + 1e-8
                    spike_avg = (spike_avg - spike_min) / spike_range
                    spike_avg = torch.clamp(spike_avg, 0, 1)
                    target = spike_avg.repeat(1, 3, 1, 1)
                    loss = self.criterion(outputs, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, scheduler: Optional[Any] = None):
        """Train model for multiple epochs. Saves best model based on validation loss and final model."""
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_loss, train_gpu_metrics = self.train_epoch()
            val_loss = self.validate()
            
            # Update learning rate
            current_lr = None
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check for best model based on validation loss
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            # Prepare metadata with GPU metrics
            metadata = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                **train_gpu_metrics
            }
            
            # Save checkpoints (best and latest)
            save_checkpoint(
                self.model, self.optimizer, epoch, val_loss,
                str(self.checkpoint_dir),
                is_best=is_best,
                is_last=True,
                metadata=metadata,
                prefix=self.checkpoint_prefix
            )
            
            # Log training metrics to file
            log_training_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                gpu_metrics=train_gpu_metrics,
                output_dir=self.log_dir,
                stage=self.stage
            )
            
            # Print metrics
            metrics_str = f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {self.best_loss:.4f}"
            if train_gpu_metrics:
                metrics_str += f"\n  GPU Metrics - Latency: {train_gpu_metrics['avg_latency']:.4f}s±{train_gpu_metrics['std_latency']:.4f}s, "
                metrics_str += f"Throughput: {train_gpu_metrics['throughput']:.2f} samples/s, "
                metrics_str += f"Memory: {train_gpu_metrics['memory_usage_mb']:.2f} MB"
            print(metrics_str)
        
        # Save final model at the end
        print(f"Training complete. Best validation loss: {self.best_loss:.4f}")
    
    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.device)
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {self.current_epoch}")

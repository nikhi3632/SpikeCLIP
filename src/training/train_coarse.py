"""Stage 1 training"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import clip
from config_loader import load_config
from data_loader import get_loader
from models.coarse_reconstruction import CoarseSNN
from loss import get_loss_fn
from training.trainer_utils import Trainer
from training.optimizer_factory import build_optimizer, build_scheduler
from utils.helpers import get_device, set_seed
from utils.logging import log_device_info

def main():
    parser = argparse.ArgumentParser(description='Train Stage 1: Coarse Reconstruction')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/checkpoints/ucaltech', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_dir = config['data_dir']
    labels = config['labels']
    val_split_ratio = config.get('val_split_ratio', 0.2)
    
    # Coarse stage config (with defaults)
    coarse_config = config.get('coarse', {})
    model_config = coarse_config.get('model', {})
    loss_config = coarse_config.get('loss', {})
    training_config = coarse_config.get('training', {})
    optimizer_config = coarse_config.get('optimizer', {})
    scheduler_config = coarse_config.get('scheduler', {})
    output_config = coarse_config.get('output', {})
    
    # Training parameters (command line args override config)
    # Use config if command line uses defaults, otherwise use command line values
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
    print(f"Coarse Config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, seed={seed}")
    
    # Log device info
    log_dir = output_config.get('log_dir', 'outputs/logs')
    log_device_info(output_dir=log_dir)
    
    # Data loader config
    data_loader_config = config.get('data_loader', {})
    num_workers = data_loader_config.get('num_workers', None)
    pin_memory = data_loader_config.get('pin_memory', None)
    
    # Data loaders
    # Train set is split into 80% train and 20% validation
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
    
    # Model
    model = CoarseSNN(
        time_steps=model_config.get('time_steps', 25),
        v_threshold=model_config.get('v_threshold', 1.0),
        tau=model_config.get('tau', 2.0),
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 3)
    )
    
    # Load CLIP model for perceptual loss if needed
    clip_model = None
    perceptual_weight = loss_config.get('perceptual_weight', 0.0)
    if perceptual_weight > 0:
        try:
            print("Loading CLIP model for perceptual loss...")
            clip_model, _ = clip.load("ViT-B/32", device=device)
            # Ensure CLIP model is in float32 (not half precision) to avoid dtype mismatches
            clip_model = clip_model.float()
            clip_model.eval()
            # Freeze CLIP model
            for param in clip_model.parameters():
                param.requires_grad = False
            print("CLIP model loaded successfully for perceptual loss")
        except Exception as e:
            print(f"Warning: Failed to load CLIP model for perceptual loss ({e}). Using fallback feature extractor.")
            clip_model = None
    
    # Loss
    semantic_weight = loss_config.get('semantic_weight', 0.0)
    criterion = get_loss_fn(
        loss_config.get('type', 'reconstruction'),
        l1_weight=loss_config.get('l1_weight', 1.0),
        l2_weight=loss_config.get('l2_weight', 1.0),
        perceptual_weight=perceptual_weight,
        semantic_weight=semantic_weight,  # Semantic alignment loss weight
        clip_model=clip_model,  # Pass CLIP model for perceptual and semantic loss
        labels=labels  # Pass labels for semantic alignment
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
    early_stopping_patience = training_config.get('early_stopping_patience', None)
    early_stopping_min_delta = training_config.get('early_stopping_min_delta', 0.0)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        use_amp=use_amp,
        grad_clip=grad_clip,
        checkpoint_prefix='coarse',
        log_dir=log_dir,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta
    )
    
    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)
    
    # Train
    trainer.train(epochs, scheduler=scheduler)

if __name__ == '__main__':
    main()

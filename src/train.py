"""orchestrator (runs stage-wise or all)"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import subprocess
import yaml

def load_config(config_path: str):
    """Load config to get checkpoint directory."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_stage(stage_name: str, config_path: str, extra_args: list = None, checkpoint_dir: str = None):
    """Run a training stage."""
    script_map = {
        'coarse': 'training/train_coarse.py',
        'prompt': 'training/train_prompt.py',
        'refine': 'training/train_refine.py'
    }
    
    if stage_name not in script_map:
        print(f"Unknown stage: {stage_name}")
        return
    
    script_path = script_map[stage_name]
    cmd = [sys.executable, script_path, '--config', config_path]
    
    # Add checkpoint arguments for stages that need them
    if stage_name in ['prompt', 'refine'] and checkpoint_dir:
        cmd.extend(['--coarse-checkpoint', checkpoint_dir])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running {stage_name} stage...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {stage_name} stage failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description='Train SpikeCLIP pipeline')
    parser.add_argument('--stage', type=str, choices=['coarse', 'prompt', 'refine', 'all'], 
                       default='coarse', help='Training stage')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--coarse-checkpoint', type=str, default=None, 
                       help='Path to coarse checkpoint directory (for prompt/refine stages)')
    
    args, unknown = parser.parse_known_args()
    
    # Load config to get checkpoint directory if not provided
    config = load_config(args.config)
    checkpoint_dir = args.coarse_checkpoint
    if not checkpoint_dir:
        # Get checkpoint dir from config (coarse stage output)
        coarse_config = config.get('coarse', {})
        output_config = coarse_config.get('output', {})
        checkpoint_dir = output_config.get('checkpoint_dir', 'outputs/checkpoints/ucaltech')
    
    extra_args = []
    if args.epochs:
        extra_args.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        extra_args.extend(['--batch-size', str(args.batch_size)])
    if args.lr:
        extra_args.extend(['--lr', str(args.lr)])
    if args.use_amp:
        extra_args.append('--use-amp')
    extra_args.extend(unknown)
    
    if args.stage == 'all':
        print("Training all stages sequentially...")
        print("Stage 1 (Coarse) → Stage 2 (Prompt) → Stage 3 (Refine)")
        print("Stage 2 depends on Stage 1 (needs coarse checkpoint)")
        print("Stage 3 depends on Stage 1 AND Stage 2 (needs coarse + prompt checkpoints)")
        for stage in ['coarse', 'prompt', 'refine']:
            run_stage(stage, args.config, extra_args, checkpoint_dir)
    else:
        run_stage(args.stage, args.config, extra_args, checkpoint_dir)

if __name__ == '__main__':
    main()

"""orchestrator (runs stage-wise or all)"""
import argparse
import subprocess
import sys

def run_stage(stage_name: str, config_path: str, extra_args: list = None):
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
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running {stage_name} stage...")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Train SpikeCLIP pipeline')
    parser.add_argument('--stage', type=str, choices=['coarse', 'prompt', 'refine', 'all'], 
                       default='coarse', help='Training stage')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    
    args, unknown = parser.parse_known_args()
    
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
        for stage in ['coarse', 'prompt', 'refine']:
            run_stage(stage, args.config, extra_args)
    else:
        run_stage(args.stage, args.config, extra_args)

if __name__ == '__main__':
    main()

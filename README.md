# SpikeCLIP

A three-stage pipeline for spike-to-image reconstruction and classification using Spiking Neural Networks (SNN), CLIP-based prompt learning, and CNN refinement.

## Overview

SpikeCLIP implements a multi-stage approach:
1. **Stage 1 (Coarse Reconstruction)**: SNN-based encoder-decoder for initial spike-to-image reconstruction
2. **Stage 2 (Prompt Learning)**: CLIP adapter for learning text prompts and aligning image-text features
3. **Stage 3 (Refinement)**: UNet-style CNN for refining coarse images to higher quality

## Project Structure

```
spike-clip/
├── configs/              # YAML configuration files
│   └── u-caltech.yaml    # Dataset configuration
├── src/
│   ├── models/          # Model definitions
│   │   ├── coarse_reconstruction.py
│   │   ├── prompt_learning.py
│   │   ├── refinement.py
│   │   └── spikeclip_model.py
│   ├── training/        # Training scripts
│   │   ├── train_coarse.py
│   │   ├── train_prompt.py
│   │   ├── train_refine.py
│   │   └── combine_checkpoints.py
│   ├── inference/       # Inference scripts
│   │   ├── infer_coarse.py
│   │   ├── infer_prompt.py
│   │   ├── infer_refine.py
│   │   └── infer_pipeline.py
│   ├── utils/           # Utility functions
│   │   ├── helpers.py
│   │   ├── checkpointing.py
│   │   ├── performance.py
│   │   └── model_loading.py
│   ├── data_loader.py    # Data loading utilities
│   ├── loss.py           # Loss functions
│   ├── metrics.py        # Evaluation metrics
│   ├── train.py          # Training orchestrator
│   ├── evaluate.py       # Evaluation script
│   └── test.py           # Testing with visualization
├── data/                 # Dataset directory
├── outputs/              # Training outputs
│   ├── checkpoints/     # Model checkpoints
│   ├── logs/            # Training/inference logs
│   └── visualizations/  # Visualization outputs
├── configs/              # Configuration files
├── Makefile             # Make targets for common tasks
└── requirements.txt     # Python dependencies
```

## Setup

```bash
# SSH into remote server
ssh -i lamdba_labs_key.pem user@remote_ip

# Clone repository
git clone https://github.com/nikhi3632/spike-clip.git
cd spike-clip

# Setup virtual environment and activate
python3 -m venv venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure SSH credentials (optional - for make login and sync)
cp .env.example .env
# Edit .env with your SSH_KEY and HOST values:
#   SSH_KEY=lamdba_labs_key.pem
#   HOST=username@ip_address

# Sync code to remote (if working on remote server)
make sync-up

# Fetch data
make data
```

## Configuration

Configuration is managed via YAML files in `configs/`. The default configuration is `configs/u-caltech.yaml`.

Key configuration sections:
- **Data**: `data_dir`, `val_split_ratio`, `data_loader` settings
- **Coarse**: Model, loss, training, optimizer, scheduler parameters
- **Prompt**: CLIP adapter configuration
- **Refine**: Refinement network parameters

Data loader settings are device-aware:
- `pin_memory`: Auto-detects (True for CUDA, False otherwise)
- `num_workers`: Auto-detects (4 for GPU, 2 for CPU)

## Usage

### Training

Train stages individually or all at once:

```bash
# Train Stage 1: Coarse Reconstruction
make train-coarse

# Train Stage 2: Prompt Learning (requires Stage 1 checkpoint)
make train-prompt

# Train Stage 3: Refinement (requires Stage 1 checkpoint)
make train-refine

# Train all stages sequentially
make train-all

# Use custom config
make train-coarse CONFIG=configs/u-caltech.yaml
```

### Inference

Run inference for individual stages or the full pipeline:

```bash
# Stage 1 inference
make infer-coarse

# Stage 2 inference
make infer-prompt

# Stage 3 inference
make infer-refine

# End-to-end pipeline inference
make infer-pipeline
```

### Utilities

```bash
# SSH to remote server (uses .env file or command-line args)
make login
# Or override: make login SSH_KEY=key.pem HOST=user@ip

# Sync files between local and remote
make sync-up      # Upload local changes to remote server
make sync-down    # Download remote changes to local
# Note: Automatically excludes venv/, data/, outputs/, .git/, .env, checkpoints

# Combine stage checkpoints into unified model
make combine-checkpoints

# Evaluate combined model (metrics only)
make evaluate

# Test combined model (with visualization)
make test

# Clean Python cache files
make clean

# Show all available commands
make help
```

### Direct Python Usage

You can also run scripts directly:

```bash
# Training
cd src
python3 training/train_coarse.py --config ../configs/u-caltech.yaml --epochs 50 --batch-size 8

# Inference
python3 inference/infer_coarse.py --config ../configs/u-caltech.yaml \
    --checkpoint-dir outputs/checkpoints/ucaltech

# End-to-end pipeline
python3 inference/infer_pipeline.py --config ../configs/u-caltech.yaml \
    --checkpoint outputs/checkpoints/ucaltech/combined_model.pth
```

## Training Workflow

1. **Stage 1 (Coarse Reconstruction)**:
   - Trains SNN encoder-decoder to reconstruct images from spikes
   - Saves checkpoints: `coarse_best.pth`, `coarse_latest.pth`

2. **Stage 2 (Prompt Learning)**:
   - Loads frozen Stage 1 model
   - Trains CLIP adapter to learn text prompts
   - Saves checkpoints: `prompt_best.pth`, `prompt_latest.pth`

3. **Stage 3 (Refinement)**:
   - Loads frozen Stage 1 model
   - Trains UNet-style refinement network
   - Saves checkpoints: `refine_best.pth`, `refine_latest.pth`

4. **Combine Checkpoints**:
   - Combines all three stage checkpoints into `combined_model.pth`
   - Enables end-to-end inference

## Features

- **Device-aware configuration**: Automatic GPU/CPU optimization (pin_memory, num_workers)
- **GPU metrics**: Tracks latency, throughput, memory usage, and power consumption
- **Flexible configuration**: YAML-based config with command-line overrides
- **Checkpoint management**: Saves best and latest models per stage
- **Comprehensive metrics**: PSNR, SSIM, L1/L2 errors for reconstruction; accuracy for classification
- **Visualization**: Sample visualization in test mode

## Outputs

Training outputs are saved to `outputs/`:
- `checkpoints/`: Model checkpoints (best and latest per stage)
- `logs/`: Training and inference metrics (JSON and text formats)
- `visualizations/`: Sample visualizations from test mode

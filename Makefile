SHELL := /bin/bash

# Configuration file (default)
CONFIG ?= configs/u-caltech.yaml

# Load SSH credentials from .env file if it exists
# Read .env file (KEY=VALUE format) and set variables
ifneq (,$(wildcard .env))
    _SSH_KEY := $(shell grep -E '^SSH_KEY=' .env 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$$//')
    _HOST := $(shell grep -E '^HOST=' .env 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$$//')
    _REMOTE_DIR := $(shell grep -E '^REMOTE_DIR=' .env 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$$//')
    ifneq ($(_SSH_KEY),)
        SSH_KEY := $(_SSH_KEY)
    endif
    ifneq ($(_HOST),)
        HOST := $(_HOST)
    endif
    ifneq ($(_REMOTE_DIR),)
        REMOTE_DIR := $(_REMOTE_DIR)
    endif
endif

# SSH defaults (override at call-site or .env file)
SSH_KEY ?= lamdba_labs_key.pem
HOST ?= user@remote_ip
REMOTE_DIR ?= ~/spike-clip

.PHONY: help clean data test train-coarse train-prompt train-refine train \
        infer-coarse infer-prompt infer-refine infer \
        combine-checkpoints evaluate visualize-stage1 plot-metrics login sync-up sync-down

# Default target
help:
	@echo "SpikeCLIP Makefile"
	@echo ""
	@echo "Data Management:"
	@echo "  make data              - Fetch data using fetch_data.py"
	@echo ""
	@echo "Training:"
	@echo "  make train-coarse      - Train Stage 1: Coarse Reconstruction"
	@echo "  make train-prompt      - Train Stage 2: Prompt Learning"
	@echo "  make train-refine      - Train Stage 3: Refinement"
	@echo "  make train-all         - Train all stages sequentially"
	@echo ""
	@echo "Inference:"
	@echo "  make infer-coarse      - Run Stage 1 inference"
	@echo "  make infer-prompt      - Run Stage 2 inference"
	@echo "  make infer-refine      - Run Stage 3 inference"
	@echo "  make infer-pipeline    - Run end-to-end pipeline inference"
	@echo ""
	@echo "Utilities:"
	@echo "  make combine-checkpoints - Combine stage checkpoints into unified model"
	@echo "  make evaluate          - Evaluate combined model (metrics only)"
	@echo "  make test              - Test combined model (with visualization)"
	@echo "  make visualize-stage1 - Visualize Stage 1 with metrics (for verification)"
	@echo "  make plot-metrics      - Plot metrics from JSON log files"
	@echo "  make login            - SSH to remote (uses SSH_KEY, HOST)"
	@echo "  make sync-up          - Sync local changes to remote server"
	@echo "  make sync-down        - Sync remote changes to local"
	@echo "  make clean             - Clean Python cache files"
	@echo ""
	@echo "Configuration:"
	@echo "  Use CONFIG variable to specify config file:"
	@echo "    make train-coarse CONFIG=configs/u-caltech.yaml"
	@echo "  SSH usage:"
	@echo "    Create .env file with SSH_KEY and HOST, or:"
	@echo "    make login SSH_KEY=lamdba_labs_key.pem HOST=user@remote_ip"
	@echo "  Sync usage:"
	@echo "    make sync-up        - Upload local changes to remote"
	@echo "    make sync-down      - Download remote changes to local"

# SSH login
login:
	@if [ -z "$(SSH_KEY)" ] || [ -z "$(HOST)" ]; then \
		echo "Usage: make login SSH_KEY=/path/to/key HOST=user@remote_ip"; \
		exit 1; \
	fi
	@echo "Connecting to $(HOST) with key $(SSH_KEY)..."
	@ssh -i "$(SSH_KEY)" "$(HOST)"

# Sync local to remote (upload)
sync-up:
	@if [ -z "$(SSH_KEY)" ] || [ -z "$(HOST)" ]; then \
		echo "Usage: make sync-up SSH_KEY=/path/to/key HOST=user@remote_ip"; \
		exit 1; \
	fi
	@echo "Syncing local changes to $(HOST)..."
	@rsync -avz --progress \
		-e "ssh -i $(SSH_KEY)" \
		--exclude='venv/' \
		--exclude='__pycache__/' \
		--exclude='*.pyc' \
		--exclude='.git/' \
		--exclude='data/' \
		--exclude='.env' \
		./ "$(HOST):$(REMOTE_DIR)/"

# Sync remote to local (download)
sync-down:
	@if [ -z "$(SSH_KEY)" ] || [ -z "$(HOST)" ]; then \
		echo "Usage: make sync-down SSH_KEY=/path/to/key HOST=user@remote_ip"; \
		exit 1; \
	fi
	@echo "Syncing remote changes from $(HOST)..."
	@rsync -avz --progress \
		-e "ssh -i $(SSH_KEY)" \
		--exclude='venv/' \
		--exclude='__pycache__/' \
		--exclude='*.pyc' \
		--exclude='.git/' \
		--exclude='data/' \
		--exclude='.env' \
		"$(HOST):$(REMOTE_DIR)/" ./

# Fetch data using fetch_data.py
data:
	@echo "Fetching data..."
	@python3 fetch_data.py

# Training targets
train-coarse:
	@echo "Training Stage 1: Coarse Reconstruction..."
	@cd src && python3 training/train_coarse.py --config ../$(CONFIG)

train-prompt:
	@echo "Training Stage 2: Prompt Learning..."
	@cd src && python3 training/train_prompt.py --config ../$(CONFIG) --coarse-checkpoint outputs/checkpoints/ucaltech

train-refine:
	@echo "Training Stage 3: Refinement..."
	@cd src && python3 training/train_refine.py --config ../$(CONFIG) --coarse-checkpoint outputs/checkpoints/ucaltech

train:
	@echo "Training all stages sequentially..."
	@cd src && python3 train.py --config ../$(CONFIG) --stage all

# Inference targets
infer-coarse:
	@echo "Running Stage 1 inference..."
	@cd src && python3 inference/infer_coarse.py --config ../$(CONFIG) --checkpoint-dir outputs/checkpoints/ucaltech

infer-prompt:
	@echo "Running Stage 2 inference..."
	@cd src && python3 inference/infer_prompt.py --config ../$(CONFIG) \
		--coarse-checkpoint-dir outputs/checkpoints/ucaltech \
		--prompt-checkpoint-dir outputs/checkpoints/ucaltech

infer-refine:
	@echo "Running Stage 3 inference..."
	@cd src && python3 inference/infer_refine.py --config ../$(CONFIG) \
		--coarse-checkpoint-dir outputs/checkpoints/ucaltech \
		--refine-checkpoint-dir outputs/checkpoints/ucaltech

infer:
	@echo "Running end-to-end pipeline inference..."
	@cd src && python3 inference/infer_pipeline.py --config ../$(CONFIG) \
		--checkpoint outputs/checkpoints/ucaltech/combined_model.pth

# Utility targets
combine-checkpoints:
	@echo "Combining stage checkpoints into unified model..."
	@cd src && python3 training/combine_checkpoints.py \
		--checkpoint-dir outputs/checkpoints/ucaltech \
		--config ../$(CONFIG) \
		--output outputs/checkpoints/ucaltech/combined_model.pth

evaluate:
	@echo "Evaluating combined model (metrics only)..."
	@cd src && python3 evaluate.py --config ../$(CONFIG) \
		--checkpoint outputs/checkpoints/ucaltech/combined_model.pth

test:
	@echo "Testing combined model (with visualization)..."
	@cd src && python3 test.py --config ../$(CONFIG) \
		--checkpoint outputs/checkpoints/ucaltech/combined_model.pth

visualize-stage1:
	@echo "Visualizing Stage 1 with metrics (for verification)..."
	@cd src && python3 visualize_stage1.py --config ../$(CONFIG) \
		--checkpoint outputs/checkpoints/ucaltech

# Remove Python bytecode caches and compiled files across the repo
clean:
	@echo "Cleaning __pycache__ directories and *.pyc/*.pyo files..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete
	@echo "Done."
	
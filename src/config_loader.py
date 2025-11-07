"""parses & validates YAML"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve data_dir relative to config file location
    if 'data_dir' in config:
        if not os.path.isabs(config['data_dir']):
            config['data_dir'] = str(config_path.parent / config['data_dir'])
    
    # Validate required fields
    required_fields = ['data_dir', 'labels']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Validate labels
    if not isinstance(config['labels'], list) or len(config['labels']) == 0:
        raise ValueError("Config must contain a non-empty 'labels' list")
    
    # Validate data directory exists
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Validate stage configurations
    for stage in ['coarse', 'prompt', 'refine']:
        if stage in config:
            stage_config = config[stage]
            if 'model' not in stage_config:
                raise ValueError(f"Stage '{stage}' missing 'model' configuration")
            if 'training' not in stage_config:
                raise ValueError(f"Stage '{stage}' missing 'training' configuration")
    
    return config

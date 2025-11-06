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
    
    return config

"""
Configuration file for project paths and hyperparameters
Reads from config.yaml for centralized configuration
"""
import os
import yaml
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load YAML config
config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths
DATA_DIR = PROJECT_ROOT / config['paths']['data_dir']
TRAIN_DIR = DATA_DIR / config['paths']['train_dir']
VAL_DIR = DATA_DIR / config['paths']['val_dir']
TEST_DIR = DATA_DIR / config['paths']['test_dir']
MODELS_DIR = PROJECT_ROOT / config['paths']['models_dir']
LOGS_DIR = PROJECT_ROOT / config['paths']['logs_dir']
OUTPUTS_DIR = PROJECT_ROOT / config['paths']['outputs_dir']
STATS_FILE = PROJECT_ROOT / config['paths']['stats_file']

# Dataset parameters
NUM_CLASSES = config['dataset']['num_classes']
IMAGE_SIZE = tuple(config['dataset']['image_size'])
NUM_CHANNELS = config['dataset']['num_channels']

# Training hyperparameters
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
EPOCHS = config['training']['epochs']
EARLY_STOPPING_PATIENCE = config['training']['early_stopping_patience']
LR_REDUCTION_PATIENCE = config['training']['lr_reduction_patience']
LR_REDUCTION_FACTOR = config['training']['lr_reduction_factor']

# Augmentation parameters
ROTATION_RANGE = config['augmentation']['rotation_range']
WIDTH_SHIFT_RANGE = config['augmentation']['width_shift_range']
HEIGHT_SHIFT_RANGE = config['augmentation']['height_shift_range']
ZOOM_RANGE = config['augmentation']['zoom_range']

# Model parameters
MODEL_ARCHITECTURE = config['model']['architecture']
PRETRAINED = config['model']['pretrained']
DROPOUT = config['model']['dropout']

# Hardware
DEVICE = config['hardware']['device']
NUM_WORKERS = config['hardware']['num_workers']
PIN_MEMORY = config['hardware']['pin_memory']
MIXED_PRECISION = config['hardware']['mixed_precision']

# Random seed for reproducibility
RANDOM_SEED = config['seed']

# Optimizer
OPTIMIZER_TYPE = config['optimizer']['type']
WEIGHT_DECAY = config['optimizer']['weight_decay']
MOMENTUM = config['optimizer']['momentum']

# Evaluation
TOP_K = config['evaluation']['top_k']


def generate_run_name(model_name: str = None) -> str:
    """
    Generate unique run name with timestamp
    Format: YYYYMMDD_HHMM_<model>_seed<seed>
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model = model_name or MODEL_ARCHITECTURE
    return f"{timestamp}_{model}_seed{RANDOM_SEED}"


def get_run_dir(run_name: str, base_dir: Path) -> Path:
    """Create and return run-specific directory"""
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# Export config dict for easy access
__all__ = ['config', 'PROJECT_ROOT', 'DATA_DIR', 'TRAIN_DIR', 'VAL_DIR', 'TEST_DIR',
           'MODELS_DIR', 'LOGS_DIR', 'OUTPUTS_DIR', 'STATS_FILE',
           'NUM_CLASSES', 'IMAGE_SIZE', 'BATCH_SIZE', 'LEARNING_RATE', 'EPOCHS',
           'generate_run_name', 'get_run_dir']

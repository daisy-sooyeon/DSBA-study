import omegaconf
from omegaconf import OmegaConf
import logging
import os
import torch
from transformers import set_seed


def load_config() -> omegaconf.DictConfig:
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
    configs = OmegaConf.load(config_path)
    return configs


def set_logger(configs: omegaconf.DictConfig) -> logging.Logger:
    """Set up logging configuration"""
    # Set log directory to exp_1/logs (relative to exp_1 directory)
    exp_1_dir = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(exp_1_dir, configs.logging.log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logger = logging.getLogger()
    return logger


def set_seed_all(seed: int):
    """Set seed for reproducibility"""
    set_seed(seed)  # transformers
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

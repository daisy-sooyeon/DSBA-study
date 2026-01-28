import logging
import os

def setup_logger(save_dir, name="experiment"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, f"{name}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
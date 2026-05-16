import logging
import sys
import os

def setup_logging(output_dir, log_name="train"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, f"{log_name}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

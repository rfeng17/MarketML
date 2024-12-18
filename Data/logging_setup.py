# logging_setup.py

import logging
from config import LOG_DIR
import datetime

def setup_logging(strategy_name):
    """
    Set up logging to console and file.
    """
    # Log filename based on trading day and strategy name
    log_filename = datetime.datetime.now().strftime(f'{LOG_DIR}/%Y-%m-%d_{strategy_name}_tradelog.log')
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

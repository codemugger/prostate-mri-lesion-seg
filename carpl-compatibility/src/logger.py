import logging
import os

def setup_logger(run_name="run", log_dir="logs", overwrite=True):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger()   # ← root logger
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w" if overwrite else "a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    logger.addHandler(file_handler)

    return logger, log_path
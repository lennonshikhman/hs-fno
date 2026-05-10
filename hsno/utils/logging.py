from __future__ import annotations
import logging, sys
from pathlib import Path

def get_logger(name: str="hsno", log_file: str | None=None) -> logging.Logger:
    logger = logging.getLogger(name); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOGGER: Optional[logging.Logger] = None

def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    log_dir = os.getenv("ALPHASQL_LOG_DIR", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"
    verbose = os.getenv("ALPHASQL_LOG_VERBOSE", "0") == "1"

    logger = logging.getLogger("alphasql")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logger initialized. File: {log_file}")
    _LOGGER = logger
    return logger

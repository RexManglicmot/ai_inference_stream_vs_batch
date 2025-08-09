# app/logger_config.py
# Simple, reusable logging setup: console + rotating file handler.

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"

# One global guard so multiple imports don’t add duplicate handlers
_INITIALIZED = False

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create/reuse a logger with:
      - human-readable console logs
      - rotating file logs at logs/app.log (5 x 1MB)
    Safe to call multiple times across modules.
    """
    global _INITIALIZED
    logger = logging.getLogger(name if name else "app")

    if _INITIALIZED:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(level)
    logger.propagate = False  # don’t double-log to root

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    # --- Rotating file handler ---
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)

    _INITIALIZED = True
    return logger


# Built-in smoke test
if __name__ == "__main__":
    log = get_logger("app.logger_test")
    log.info("Logger initialized.")
    log.warning("This is a warning.")
    log.error("This is an error.")
    print("Wrote logs to", LOG_FILE.resolve())

import logging
import os
from typing import Optional

FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

class _NoColorFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(FMT, datefmt=DATEFMT)

class _ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",   # cyan
        "INFO": "\033[32m",    # green
        "WARNING": "\033[33m", # yellow
        "ERROR": "\033[31m",   # red
        "CRITICAL": "\033[41m" # red bg
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        color = self.COLORS.get(level, "")
        msg = logging.Formatter(FMT, datefmt=DATEFMT).format(record)
        return f"{color}{msg}{self.RESET}"

def setup_logging(level: str = "INFO", to_file: bool = False, log_dir: str = "logs"):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    # 清空旧 handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
        fh.setLevel(getattr(logging, level.upper()))
        fh.setFormatter(_NoColorFormatter())
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))
        ch.setFormatter(_ColorFormatter())
        logger.addHandler(ch)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("Logging initialized: level=%s to_file=%s dir=%s", level, to_file, log_dir)
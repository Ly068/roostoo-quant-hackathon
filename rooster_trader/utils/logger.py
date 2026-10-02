"""日志配置：同时输出到控制台和文件，方便AWS后台排障"""
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = "rooster_trader", log_dir: str = "logs") -> logging.Logger:
    """
    配置统一日志器。
    日志文件自动轮转，单个文件最大10MB，保留5个备份。
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    )

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件输出（轮转）
    fh = RotatingFileHandler(
        os.path.join(log_dir, "trader.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

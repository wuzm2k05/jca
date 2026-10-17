"""日志包：整个项目只用这里的 logger，不再用 print。

    from logger import get_logger

    logger = get_logger(__name__)
    logger.info("执行命令", cmd=command)

程序入口（main.py）在最早的地方调一次 `setup_logging()`，之后各处直接取用。
配置项见 config.py，装配细节（终端 / 文件 / 轮转）见 setup.py。
"""

from .config import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_FILE_NAME,
    DEFAULT_LEVEL,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_BYTES,
    ENV_PREFIX,
    LogConfig,
    LogTarget,
)
from .setup import get_logger, setup_logging

__all__ = [
    "DEFAULT_BACKUP_COUNT",
    "DEFAULT_FILE_NAME",
    "DEFAULT_LEVEL",
    "DEFAULT_LOG_DIR",
    "DEFAULT_MAX_BYTES",
    "ENV_PREFIX",
    "LogConfig",
    "LogTarget",
    "get_logger",
    "setup_logging",
]

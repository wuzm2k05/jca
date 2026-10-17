"""日志配置：定一份默认值，允许用环境变量覆盖。

为什么走环境变量：这个项目本来就用 .env 放凭据（见 main.py 里的 load_dotenv），
日志是同一类「跟着部署环境走」的东西。再引一套配置文件或者命令行参数，
只会多出一个需要跟代码同步的地方。
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# 所有相关环境变量都带这个前缀，免得跟别人的重名
ENV_PREFIX = "JCA_"


class LogTarget(str, Enum):
    """日志往哪儿写。"""

    CONSOLE = "console"  # 只写终端
    FILE = "file"  # 只写文件
    BOTH = "both"  # 两边都写


# —— 默认值 ——
DEFAULT_TARGET = LogTarget.CONSOLE
DEFAULT_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"
DEFAULT_FILE_NAME = "jca.log"

# 单个日志文件的上限，超过就切分。10 MB 大约十几万行中等长度的记录：
# 够查一次出问题，又不至于大到编辑器打不开。
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
# 保留几个历史文件。磁盘占用上限大约是 (backup_count + 1) * max_bytes。
DEFAULT_BACKUP_COUNT = 5


@dataclass(frozen=True)
class LogConfig:
    """一次进程的日志配置。"""

    target: LogTarget = DEFAULT_TARGET
    level: str = DEFAULT_LEVEL
    log_dir: Path = Path(DEFAULT_LOG_DIR)
    file_name: str = DEFAULT_FILE_NAME
    max_bytes: int = DEFAULT_MAX_BYTES
    backup_count: int = DEFAULT_BACKUP_COUNT

    @property
    def file_path(self) -> Path:
        """日志文件的完整路径。

        按**当前工作目录**解析，和 tools/sandbox.py 的 WORKSPACE 是同一个道理：
        在哪儿运行，文件就落在哪儿。所以每次读取都重新算一遍 —— 别把它存起来跨
        「改了工作目录」的代码用。

        log_dir 写成绝对路径也可以，Path 拼接时绝对路径会直接覆盖前面的部分。

        另外注意：日志**不能**放进 workspace/。那是 agent 自己的沙箱，
        日志放进去，agent 就能读到、甚至改掉记录它自己行为的文件。
        """
        return (Path.cwd() / self.log_dir / self.file_name).resolve()

    @classmethod
    def from_env(cls) -> "LogConfig":
        """从环境变量读一份配置，没设的项用默认值。"""
        return cls(
            target=_env_target(),
            level=resolve_level_name(os.getenv(ENV_PREFIX + "LOG_LEVEL", DEFAULT_LEVEL)),
            log_dir=Path(os.getenv(ENV_PREFIX + "LOG_DIR", DEFAULT_LOG_DIR)),
            file_name=os.getenv(ENV_PREFIX + "LOG_FILE", DEFAULT_FILE_NAME),
            max_bytes=_env_int("LOG_MAX_BYTES", DEFAULT_MAX_BYTES),
            backup_count=_env_int("LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT),
        )


def resolve_level_name(name: str) -> str:
    """把级别名规范化成标准写法（大写），不认识就抛错。

    宁可让程序起不来也不退回默认值：把 JCA_LOG_LEVEL 拼错的后果是
    「日志少了一大截，而且没人告诉你」，这种错必须当场炸出来。
    """
    key = name.strip().upper()
    if key not in logging.getLevelNamesMapping():
        valid = "/".join(sorted(logging.getLevelNamesMapping()))
        raise ValueError(f"日志级别不认识：{name!r}（可用的是 {valid}）")
    return key


def level_value(name: str) -> int:
    """把级别名转成 logging 用的数字级别（DEBUG=10、INFO=20……）。"""
    return logging.getLevelNamesMapping()[resolve_level_name(name)]


def _env_target() -> LogTarget:
    raw = os.getenv(ENV_PREFIX + "LOG_TARGET", DEFAULT_TARGET.value).strip().lower()
    try:
        return LogTarget(raw)
    except ValueError:
        valid = "/".join(t.value for t in LogTarget)
        raise ValueError(f"{ENV_PREFIX}LOG_TARGET 只能是 {valid}，现在是 {raw!r}") from None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} 得是整数，现在是 {raw!r}") from None
    if value < 0:
        raise ValueError(f"{ENV_PREFIX}{name} 不能是负数，现在是 {value}")
    return value

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

        按**当前工作目录**解析，和 tools/sandbox.py 的 workspace_root() 是同一个道理：
        在哪儿运行，文件就落在哪儿。所以每次读取都重新算一遍 —— 别把它存起来跨
        「改了工作目录」的代码用。

        log_dir 写成绝对路径也可以，Path 拼接时绝对路径会直接覆盖前面的部分。

        另外注意：日志**不该**放进 agent 的沙箱目录（tools/sandbox.py 里的
        workspace_root()）。放进去，agent 就能读到、甚至改掉记录它自己行为的文件。

        但这两者的默认值正好撞在一起：workspace 默认是「运行目录」，而 log_dir
        默认是运行目录下的 logs/，于是默认配置下日志就在沙箱里。想分开，
        两条路任选：给 JCA_WORKSPACE 设一个子目录（沙箱缩进去，日志留在外面），
        或者把 JCA_LOG_DIR 指到沙箱外。README 里「Logging」一节有现成写法。
        """
        return (Path.cwd() / self.log_dir / self.file_name).resolve()

    @classmethod
    def from_env(cls, default_target: LogTarget | None = None) -> "LogConfig":
        """从环境变量读一份配置，没设的项用默认值。

        default_target 是给**调用方**用的兜底出口，只在这一项没被环境变量
        设过时生效。谁会用到：交互式 CLI。它用 rich 的 Live 在 sys.stdout 上
        做光标定位，而终端日志走的是裸 sys.stdout.write，两边的光标会互相踩，
        输出会花掉——所以 CLI 把默认出口换成文件（详见 ui/cli.py 的说明）。

        为什么是「兜底」而不是「覆盖」：用户如果显式写了 JCA_LOG_TARGET，
        那是他的决定，代码不该替他改，只该在他选的出口有代价时提醒一句。
        """
        return cls(
            target=_env_target(default_target),
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


def _env_target(default: LogTarget | None = None) -> LogTarget:
    """读 JCA_LOG_TARGET；没设就用 default（再没有就用模块默认值）。

    不认识的**值**要抛错，和 resolve_level_name 一个道理：把 console 拼成
    consle，后果是「日志哪儿都没有，而且没人告诉你」。但「这一项没设」不是
    错误，只是用默认值。
    """
    raw = os.getenv(ENV_PREFIX + "LOG_TARGET")
    if raw is None or not raw.strip():
        return default if default is not None else DEFAULT_TARGET
    raw = raw.strip().lower()
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

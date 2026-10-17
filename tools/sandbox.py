"""路径沙箱：所有文件操作都被限制在工作目录内。"""

import os
from pathlib import Path

from logger import ENV_PREFIX, get_logger

logger = get_logger(__name__)

# 沙箱根目录由环境变量 JCA_WORKSPACE 决定：
#   不设（默认）→ 就是**运行程序时所在的目录**，不额外造一层；
#   设了        → pwd 拼上这个值，给 agent 单独划一块地方，比如 JCA_WORKSPACE=workspace。
#
# 下面这个常量的**名字**和它的**值**是两回事：值（"JCA_WORKSPACE"）才是要写进
# .env / 命令行的变量名。踩过一次：.env 里写成 ENV_WORKSPACE=workspace —— 那是
# 这个常量的名字，os.getenv 当然读不到，于是配置一声不响地没生效，
# 文件照旧落在运行目录里。
ENV_WORKSPACE = ENV_PREFIX + "WORKSPACE"


def workspace_root() -> Path:
    """本次运行的工作目录（沙箱根）。

    用当前启动目录而不是脚本所在目录，这样打包成 exe 之后，
    「在哪儿运行，文件就落在哪儿」的语义不变。

    值写成绝对路径也可以：Path 拼接时绝对路径会直接盖掉前面那部分。

    为什么是**函数**而不是模块级常量：变量要在**调用时**读，才能让三种来源都生效
    —— .env（main.py 里运行时才 load_dotenv）、命令行、以及运行中改 os.environ。
    写成模块级常量的话，值在 import 那一刻就定死了，而 import 永远早于
    load_dotenv()，等于把成败赌在 import 顺序上。

    同理，每次调用都重新算、不缓存：和 logger/config.py 里的 file_path 一个道理。
    代价是一次 getenv 加一次 resolve，相对一次文件操作可以忽略。
    """
    raw = os.getenv(ENV_WORKSPACE, "").strip()
    if not raw:
        return Path.cwd().resolve()
    return (Path.cwd() / raw).resolve()


def safe_path(raw: str) -> Path:
    """把模型给的相对路径解析成工作目录下的真实路径；越界就报错。"""
    root = workspace_root()
    target = (root / raw).resolve()
    if not target.is_relative_to(root):
        # 这一条专门记 warning：越界不是普通出错，是「有人在试着走出沙箱」。
        # 日志里留一笔，事后才分得清模型是手滑还是真的在探边界。
        logger.warning("路径越界，已拒绝", raw=raw, resolved=str(target), root=str(root))
        raise ValueError(f"路径超出工作目录范围，只允许操作 {root} 内的文件。")
    return target

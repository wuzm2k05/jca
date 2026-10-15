"""工具注册表：增删能力只改这个文件，图的结构不用动。"""

import os

from .files import FILE_TOOLS
from .shell import run_cmd

__all__ = ["build_tools"]


def build_tools() -> list:
    """返回本次运行可用的工具清单。

    文件工具默认可用；执行命令的能力属于高危能力，需要用环境变量显式打开。
    """
    tools = list(FILE_TOOLS)
    tools.append(run_cmd)
    
    return tools

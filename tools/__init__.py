"""工具注册表：增删能力只改这个文件，图的结构不用动。"""
from .files import FILE_TOOLS
from .shell import run_cmd

__all__ = ["build_tools"]


def build_tools() -> list:
    """返回本次运行可用的工具清单。
    """
    tools = list(FILE_TOOLS)
    tools.append(run_cmd)
    
    return tools

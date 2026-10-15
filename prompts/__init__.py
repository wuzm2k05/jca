"""提示词包：调用方只从这里取，以后加多少文件都不用改调用方。"""

from .system import build_system_prompt

__all__ = ["build_system_prompt"]

"""提示词包：调用方只从这里取，以后加多少文件都不用改调用方。"""

from .system import build_system_prompt
from .todos import WRITE_TODOS_SYSTEM_PROMPT, WRITE_TODOS_TOOL_DESCRIPTION

__all__ = [
    "WRITE_TODOS_SYSTEM_PROMPT",
    "WRITE_TODOS_TOOL_DESCRIPTION",
    "build_system_prompt",
]

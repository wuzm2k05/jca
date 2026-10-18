"""内容拦截的处理包：调用方只从这里取，以后加「限流」「超长上下文」等
同类处理都往这个包里加文件，不用改调用方。
"""

from .guard import (
    MAX_SAFETY_RETRIES,
    RETRY_NOTE,
    SAFETY_CODE,
    build_retry_message,
    is_safety_block,
)

__all__ = [
    "MAX_SAFETY_RETRIES",
    "RETRY_NOTE",
    "SAFETY_CODE",
    "build_retry_message",
    "is_safety_block",
]

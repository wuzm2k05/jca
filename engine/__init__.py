"""核心引擎包：Agent 的一切都在这儿，它不知道终端长什么样。

    from engine import JCAEngine, AssistantText, ToolCall, ...

    engine = JCAEngine()
    async for event in engine.stream_turn("在 workspace 下建个 app.py"):
        ...

上面这段就是全部用法。`engine` 不认识 rich，也不认识 stdin/stdout；
它只吐 engine/events.py 里定义的事件。想换成 Web 界面，写一个新的消费端即可，
这个包一行都不用改。
"""

from .agent import DEFAULT_MODEL, JCAEngine
from .events import (
    AgentError,
    AgentEvent,
    AssistantText,
    Notice,
    ToolCall,
    ToolResult,
    TurnDone,
)

__all__ = [
    "DEFAULT_MODEL",
    "AgentError",
    "AgentEvent",
    "AssistantText",
    "JCAEngine",
    "Notice",
    "ToolCall",
    "ToolResult",
    "TurnDone",
]

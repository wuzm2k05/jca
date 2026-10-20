"""核心引擎包：Agent 的一切都在这儿，它不知道终端长什么样。

    from engine import JCAEngine, AssistantText, ToolCall, ...

    engine = JCAEngine()
    async for event in engine.stream_turn("在 workspace 下建个 app.py"):
        ...

上面这段就是全部用法。`engine` 不认识 rich，也不认识 stdin/stdout；
它只吐 engine/events.py 里定义的事件。想换成 Web 界面，写一个新的消费端即可，
这个包一行都不用改。

中间件（engine/middleware/）是「进入模型之前做什么」的扩展点，这里一起导出：
读代码的人应该一眼看见它在哪，而不是从 `JCAEngine.__init__` 的默认参数里倒推。
"""

from .agent import DEFAULT_MIDDLEWARES, DEFAULT_MODEL, JCAEngine
from .events import (
    AgentError,
    AgentEvent,
    AssistantText,
    Notice,
    ToolCall,
    ToolResult,
    TurnDone,
)
from .middleware import (
    BaseMiddleware,
    MiddlewarePipeline,
    ModelRequest,
    TodoItem,
    TodoMiddleware,
)

__all__ = [
    "DEFAULT_MIDDLEWARES",
    "DEFAULT_MODEL",
    "AgentError",
    "AgentEvent",
    "AssistantText",
    "BaseMiddleware",
    "JCAEngine",
    "MiddlewarePipeline",
    "ModelRequest",
    "Notice",
    "TodoItem",
    "TodoMiddleware",
    "ToolCall",
    "ToolResult",
    "TurnDone",
]

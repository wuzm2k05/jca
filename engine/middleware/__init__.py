"""中间件包：进入 LLM 之前的那条管道。

    from engine.middleware import BaseMiddleware, MiddlewarePipeline, TodoMiddleware

加一个新中间件只需要两步：写一个 `BaseMiddleware` 的子类，把它放进
`JCAEngine(middlewares=[...])`。图的结构不用动 —— `before_model` /
`after_model` 两个节点本来就存在，它们会遍历管道里的每一个中间件。
"""

from .base import (
    BaseMiddleware,
    HookResult,
    MiddlewarePipeline,
    ModelHandler,
    ModelRequest,
    StateUpdate,
)
from .todo import TodoItem, TodoMiddleware, TodoState, write_todos

__all__ = [
    "BaseMiddleware",
    "HookResult",
    "MiddlewarePipeline",
    "ModelHandler",
    "ModelRequest",
    "StateUpdate",
    "TodoItem",
    "TodoMiddleware",
    "TodoState",
    "write_todos",
]

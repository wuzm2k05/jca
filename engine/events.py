"""两个世界之间的契约：Engine 吐出的领域事件。

这一层是**唯一**允许被前端 import 的东西，所以它必须干净到能在任何地方用：
只 import 标准库，不 import langchain、不 import rich、不认识 LangGraph。

为什么值得单开一个文件而不是让 UI 直接读 LangGraph 的流：流里的类型是
`AIMessageChunk` / `ToolMessage` 这些**框架的概念**，它们的形状会随版本变
（本仓库踩过一次：`AIMessageChunk.__add__` 对重复出现的 `name` 会重复拼接，
自己拼分片拿到的工具名会变成 `"list_dirlist_dirlist_dir"`）。把翻译收在
engine 一层做完，形状变化就只影响一个文件，UI 那边只认下面这六个 dataclass。

命名上刻意用「事件」而不是「状态」：UI 是**被推着走**的，它不查询、
不轮询、也不持有会话历史（历史在 engine 的 checkpointer 里）。
"""

from dataclasses import dataclass
from typing import Any, Union


@dataclass
class AssistantText:
    """模型回复的**增量**文本，不是完整回复。

    一段回复会被切成很多个这个事件。UI 自己决定怎么攒——本项目的 UI 是
    「攒到工具调用/轮次结束就收口成一个 Markdown 块」。
    """

    delta: str


@dataclass
class ToolCall:
    """模型申请调用某个工具。

    注意这是**申请**，不是执行结果：模型只负责说「我想调 list_dir」，
    真正动手的是 ToolNode。这个区分正是「决策与执行分离」那条设计原则的体现，
    也是在引擎层能被看见的地方。

    invalid_args 非空表示模型吐出的参数不是合法 JSON（这时 args 是空的）。
    这种调用 ToolNode 会以错误消息的形式回报，不会落进 args，所以两个字段
    是互斥的。
    """

    id: str
    name: str
    args: dict[str, Any]
    invalid_args: str | None = None


@dataclass
class ToolResult:
    """工具执行完毕，这是给模型看的「观察结果」。

    output 已经被 engine 截断过（终端不是给人看两万字的地方），chars 是
    **截断前**的长度，UI 靠它显示「（已截断，共 N 字符）」。

    ok 直接取自 ToolMessage.status，只表示「ToolNode 有没有抛异常」。
    本项目的工具约定是**出错也返回一段文字**（见 tools/files.py 的开头），
    那种「业务失败」在这里同样是 ok=True。想判断业务成败只能看 output。
    """

    id: str
    name: str
    ok: bool
    output: str
    chars: int


@dataclass
class Notice:
    """一句给用户看的提示，不改变流程。

    kind 决定措辞和颜色，取值：
      - "safety_retry"：被服务端内容判定拦下，正在自动换说法重试（见 safety/guard.py）
      - "info"：其它说明性提示
    """

    text: str
    kind: str


@dataclass
class AgentError:
    """这一轮失败了。

    kind 取值：
      - "safety"：连续被内容判定拦下，重试额度耗尽
      - "api"：调用模型本身失败（网络、认证、限流……）
      - "internal"：其它没预料到的问题

    收完这个事件这一轮就结束了，**不会**再有 TurnDone。
    """

    message: str
    kind: str


@dataclass
class TurnDone:
    """本轮正常收尾。turns 是累计完成的轮次数。

    它存在的意义是给 UI 一个明确的收口信号：Live 区域可以关掉了、
    空行可以画了。失败的一轮用 AgentError 收尾，不发这个事件。
    """

    turns: int


# 事件联合类型。UI 侧的建议写法是先判 `isinstance(event, TurnDone)` 这类
# 收尾事件，再判内容事件，而不是假定顺序——顺序是调度结果，不是契约。
AgentEvent = Union[AssistantText, ToolCall, ToolResult, Notice, AgentError, TurnDone]


__all__ = [
    "AgentError",
    "AgentEvent",
    "AssistantText",
    "Notice",
    "ToolCall",
    "ToolResult",
    "TurnDone",
]

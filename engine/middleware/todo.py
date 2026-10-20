"""TodoMiddleware：把「计划」变成 state 里的一个字段，再加一个改它的工具。

## 三件事

req.md 里的「工具自动同步、System Prompt 动态注入、工具按需挂载」，落到代码是：

1. **工具自动同步** —— `write_todos` 返回的是 `Command(update={...})` 而不是字符串。
   一次工具调用同时做两件事：把 ToolMessage 还给模型（它需要知道调用成功了），
   以及把 `todos` 写进 state。`Command` 这个返回值类型就是为此存在的，
   LangGraph 的 ToolNode 认它。中间件因此**不需要**在 `after_model` 里去
   「扫一遍刚才调了什么工具、再把结果抄进 state」—— 那种抄写迟早会和真相走偏。
2. **System Prompt 动态注入** —— `wrap_model` 里把固定说明 + **现读 state 渲染的**
   清单追加到 system prompt 末尾。清单每次调用现渲染，所以模型看到的计划
   永远是刚更新过的，而 system prompt 本身不进 state 历史（用完即丢）。
3. **工具按需挂载** —— 基础管道把工具集放在 `ModelRequest.tools` 上，每次调用
   算一份。本中间件目前的策略是**每次都挂**（用户确认的取舍：行为最好预测），
   但改起来只需要覆盖 `wrap_model` 一行：`replace(request, tools=[t for t in
   request.tools if t.name != write_todos.name])` 就是「这一轮不挂」。

## 为什么 before_model / after_model 是空的

因为状态由工具自己写（第 1 条）。中间件这三个钩子是三种**不同的**实现思路，
不是必须都用的清单：

- 有「工具改不了、只能由中间件改」的状态时，用 `before_model` / `after_model`；
- 有「只影响这次请求、不该留档」的东西时，用 `wrap_model`。

TodoMiddleware 属于第二种。故意留着不覆盖，是为了让读者看见「钩子可以不用」——
默认实现就是空操作，写个 `pass` 出来只是噪音。

## 为什么工具放在这里，而不是 `tools/`

`tools/` 是**常驻**工具的注册表（`build_tools()`），里面的工具在任何会话里都在。
`write_todos` 的语义不同：它只在 TodoMiddleware 装上时存在，而且它要写的
`todos` 字段是这个中间件声明进 state 的 —— 工具和它的 state 契约是一对，
分开放在两个目录里，改一个忘一个只是时间问题。所以它们同住一个文件。
"""

from dataclasses import replace
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from prompts import WRITE_TODOS_SYSTEM_PROMPT, WRITE_TODOS_TOOL_DESCRIPTION, render_todos

from .base import BaseMiddleware, ModelHandler, ModelRequest


class TodoItem(TypedDict):
    """一条待办。

    `Literal` 而不是 `str`：`@tool` 会把它推断成 JSON Schema 的 enum，
    模型在生成参数时就看得见「只有这三个值可选」，比在描述里用文字说
    「status 只能是 pending/in_progress/completed」可靠得多。
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


def _last_write_wins(_current: list | None, new: list | None) -> list | None:
    """给 `todos` 通道配的 reducer：不合并，直接取新的那份。

    看起来是多余的 —— 默认（没有 reducer）本来就是「后写覆盖先写」。但它解决的是
    另一个问题：**一轮里模型并行调了两次 `write_todos`**。那不是假设，实测会这样
    （工具描述里那句 "never be called multiple times in parallel" 显然不够）。
    ToolNode 会把两条 `Command` 的 update 合成一个列表交给图，而默认通道对同一轮
    出现的两个值直接报错：

        InvalidUpdateError: At key 'todos': Can receive only one value per step.

    这个错误还会在 `stream_turn` 的兜底 except 里被翻成一条语焉不详的
    `AgentError(kind="api")`（「调用模型失败：At key 'todos'...」），用户完全
    看不出是自己那轮计划写重了。配上 reducer 之后两张清单都能落地，取后一张 ——
    对「整份覆盖」的语义来说，这就是正确结果。

    副作用要记住：带 reducer 的通道从**空值**开始，所以第一次调用时
    `state.get("todos")` 是 `[]` 而不是 `None`。凡是判「有没有清单」的地方都得按
    真假值判，不能写 `is None`（`render_todos` 已经两者都吃）。
    """

    return new


class TodoState(TypedDict):
    """这个中间件加进 AgentState 的字段。

    `NotRequired` 是必须的：会话刚开始时 state 里还没有 `todos` 这个键，
    标成必需的话 LangGraph 会在第一轮就报缺字段。

    reducer 是 `_last_write_wins`，语义仍是「整份覆盖」—— 每次 `write_todos`
    送来的都是**完整**清单，这正是我们要的（模型看到的是全量，不是增量）。
    """

    todos: NotRequired[Annotated[list[TodoItem], _last_write_wins]]


_STATUS_LABEL = {"pending": "待办", "in_progress": "进行中", "completed": "已完成"}


def _summarize(todos: list[TodoItem]) -> str:
    """给模型看的回执。它需要知道「这次写进去了什么、几项在做」。

    回执不是可选项：ToolNode 要求每个工具调用都有对应的 ToolMessage，
    少一条图就跑不下去（会抛「Every tool call MUST have a corresponding
    ToolMessage」）。所以这里的每一句话都是顺手写给模型看的。
    """
    counts = {status: 0 for status in _STATUS_LABEL}
    for item in todos:
        status = str(item.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
    detail = "，".join(
        f"{counts[s]} 项{label}" for s, label in _STATUS_LABEL.items() if counts.get(s)
    )
    return f"已更新 {len(todos)} 条待办（{detail or '空清单'}）。"


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[TodoItem],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """创建或更新本次工作的待办清单。"""
    return Command(
        update={
            # 整份覆盖，不是追加 —— 模型送来的是完整清单。
            "todos": todos,
            # 这条 ToolMessage 不能省：没有它，ToolNode 会拒绝这次工具调用。
            # tool_call_id 由 InjectedToolCallId 从模型那次调用里注入，
            # 模型自己不需要（也不应该）传。
            "messages": [ToolMessage(content=_summarize(todos), tool_call_id=tool_call_id)],
        }
    )


class TodoMiddleware(BaseMiddleware):
    """把 todo 清单接进管道的中间件。"""

    state_schema = TodoState
    tools: tuple[BaseTool, ...] = (write_todos,)

    def wrap_model(self, request: ModelRequest, handler: ModelHandler) -> AIMessage:
        """往这次调用的 system prompt 里追加工具说明 + 当前清单。

        `replace` 造一份新的 request：管道里可能还有别的中间件，就地改会让
        它们也看到被改过的对象。这里只是「我这一层多说的话」，**交棒之后**
        的任务不归我管，所以不再动返回值。

        清单为空时不追加清单那一段（`render_todos` 返回空串），但工具说明照样
        追加 —— 模型得先知道有这个工具，才可能第一次去写清单。
        """
        todos = request.state.get("todos")
        prompt = request.system_prompt + "\n\n" + WRITE_TODOS_SYSTEM_PROMPT
        checklist = render_todos(todos)
        if checklist:
            prompt += "\n\n" + checklist
        return handler(replace(request, system_prompt=prompt))

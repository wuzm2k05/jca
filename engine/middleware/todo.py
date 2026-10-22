"""TodoMiddleware：把「计划」变成 state 里的一个字段，再加一个改它的工具。

## 三件事

1. **工具自动同步** —— `write_todos` 返回的是 `Command(update={...})` 而不是字符串。
   一次工具调用同时做两件事：把 ToolMessage 还给模型（它需要知道调用成功了），
   以及把 `todos` 写进 state。`Command` 这个返回值类型就是为此存在的，
   LangGraph 的 ToolNode 认它。中间件因此**不需要**在 `after_model` 里去
   「扫一遍刚才调了什么工具、再把结果抄进 state」—— 那种抄写迟早会和真相走偏。
2. **System Prompt 静态注入** —— `wrap_model` 把固定说明 `WRITE_TODOS_SYSTEM_PROMPT`
   追加到 system prompt 末尾，**仅此而已**，每次调用拼出来的文本逐字相同。
3. **工具按需挂载** —— 基础管道把工具集放在 `ModelRequest.tools` 上，每次调用
   算一份。本中间件目前的策略是**每次都挂**（用户确认的取舍：行为最好预测），
   但改起来只需要覆盖 `wrap_model` 一行：`replace(request, tools=[t for t in
   request.tools if t.name != write_todos.name])` 就是「这一轮不挂」。

## 为什么不再把清单渲染进 system prompt

早先的版本在 `wrap_model` 里现读 `state["todos"]`、用 `render_todos` 渲染成清单
拼到 prompt 末尾。那样做有两个代价，都实测过：

1. **打断服务端的前缀 KV Cache。** prompt 尾部每轮都变，后面所有历史消息的
   前缀缓存全部失效，首字延迟（TTFT）与 token 成本一起上去。prompt 现在是
   100% 静态且确定的，前缀缓存可以稳定命中到历史消息的末尾。
2. **信息三重冗余。** 模型在历史里本来就有上一轮 `write_todos` 的
   `AIMessage(tool_calls)` 和 `ToolMessage` 回执 —— 它自己写的、我们刚还给它的。
   再贴一份渲染过的清单是同一件事说三遍。

所以清单改由**历史消息**（In-Context Memory）自然承载：模型读自己上一轮的
工具调用就知道计划是什么。`state["todos"]` 这个结构化字段**保留**，但它的
消费方是前端 UI 与持久化快照，不再是 prompt。

## 为什么 before_model 是空的

因为状态由工具自己写（第 1 条）。中间件这三个钩子是三种**不同的**实现思路，
不是必须都用的清单：

- 有「工具改不了、只能由中间件改」的状态时，用 `before_model` / `after_model`；
- 有「只影响这次请求、不该留档」的东西时，用 `wrap_model`。

`before_model` 故意留着不覆盖，是为了让读者看见「钩子可以不用」——
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

from prompts import WRITE_TODOS_SYSTEM_PROMPT, WRITE_TODOS_TOOL_DESCRIPTION

from .base import BaseMiddleware, HookResult, ModelHandler, ModelRequest


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
    另一个问题：**同一个超步里 `todos` 被写了两次**。ToolNode 会把两条 `Command`
    的 update 合成一个列表交给图，而默认通道对同一轮出现的两个值直接报错：

        InvalidUpdateError: At key 'todos': Can receive only one value per step.

    这个错误还会在 `stream_turn` 的兜底 except 里被翻成一条语焉不详的
    `AgentError(kind="api")`（「调用模型失败：At key 'todos'...」），用户完全
    看不出是自己那轮计划写重了。

    **它现在不是并行调用的主要挡箭牌了。** 「一轮里并行调多次 `write_todos`」
    那件事由 `TodoMiddleware.after_model` 正面拦下（整轮拒绝、让模型重写），
    因为「后一张赢」本身就是替模型做了决定 —— 两份清单互相冲突，没有理由选
    任何一张。reducer 留着是为了兜住**别的**路径：任何中间件、任何未来的写入者
    在同一超步写 `todos` 时，图不该因此崩掉。

    副作用要记住：带 reducer 的通道从**空值**开始，所以第一次调用时
    `state.get("todos")` 是 `[]` 而不是 `None`。凡是判「有没有清单」的地方都得按
    真假值判，不能写 `is None`。
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

# 一轮里并行调了多次 write_todos 时还给模型的回执。中文，和其它工具返回值一致 ——
# 它是给模型的**现场反馈**，和对话语言一致更稳（见 prompts/todos.py 的说明：
# 那两段英文提示词是例外，因为它们是官方逐字搬来的校准过的文本）。
#
# 措辞点明了「怎么改」，而不是只说「不行」：模型看到之后应当把两份清单合成一份、
# 只调一次。也点明了这一轮**别的**工具调用一并作废 —— 它们拿到的是同一条回执
# （见 after_model 的说明），不说清楚的话模型会以为文件读过了。
_REJECT_PARALLEL = (
    "错误：同一轮里只能调用一次 write_todos，但你并行调用了多次。"
    "这一轮的所有工具调用都已作废，一个都没执行，state 里的清单也没有改动。"
    "请把要写的合并成一份**完整**清单，重新调用一次。"
)


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
        """往这次调用的 system prompt 末尾追加固定说明。

        只有 `WRITE_TODOS_SYSTEM_PROMPT` 这一段，**不读 state、不渲染清单** ——
        这是本中间件最要紧的性质：拼出来的 prompt 在整个会话生命周期里逐字不变，
        服务端的前缀 KV Cache 因此能一路命中到历史消息的末尾（见模块说明）。
        清单由历史里的 `write_todos` 调用与 ToolMessage 承载。

        用 `replace` 造一份新的 request：管道里可能还有别的中间件，就地改会让
        它们也看到被改过的对象。这里只是「我这一层多说的话」，**交棒之后**的
        任务不归我管，所以不再动返回值。
        """
        return handler(
            replace(request, system_prompt=request.system_prompt + "\n\n" + WRITE_TODOS_SYSTEM_PROMPT)
        )

    def after_model(self, state: TodoState) -> HookResult:
        """拦下「一轮里并行调了多次 `write_todos`」这种情况，让模型重写。

        模型确实会这么干 —— 工具描述里那句 "never be called multiple times in
        parallel" 显然不够。两次调用送来的是两份**互相冲突**的完整清单，谁也
        不该赢：让后一张覆盖前一张是凭空替模型做了决定，而两张都下发则连
        LangGraph 那关都过不去（`InvalidUpdateError: At key 'todos'`）。

        ## 为什么必须返回 Command，而不是塞一条错误 ToolMessage

        「往 messages 里追加一条错误 ToolMessage」是做不到这件事的，实测过：

            # 只返回 {"messages": [ToolMessage(...)]} 时
            events: AssistantText, ToolCall, ToolCall, ToolResult, TurnDone
            state["todos"]: []
            模型被调用次数: 1        # 它没有机会重写

        原因是 `after_model` 的返回值只影响 state，**不影响下一步去哪儿** ——
        下一步仍由 `tools_condition` 决定，而 last message 带着 tool_calls，
        于是 ToolNode 照常把那两个 `write_todos` 都执行了。那条错误消息没有配对的
        tool_call，会被 UI 按 tool_call_id 丢弃，用户根本看不见；两个真调用则
        互相覆盖，最后落在 `[]`。返回 `Command(goto=...)` 才能真正改道。

        ## 为什么 goto "before_model" 而不是 "agent"

        两个实测行为相同，选前者是因为它更符合本仓库的既有约定：拦截之后紧接着
        就是一次模型调用（模型看到错误消息、重写清单），而 `before_model` 会在它
        前面跑一遍 —— `test_safety_retry_goes_back_through_before_model` 钉的正是
        「新的模型调用要重新走一遍中间件」这条。`goto="agent"` 会跳过它。

        ## 这一轮里的**所有**工具调用都要配回执，不只是 write_todos 那两个

        改道意味着 ToolNode 整个不跑，所以这一轮里任何一个工具调用都不会有
        结果。只给 `write_todos` 补回执的话，别的调用就悬空了 —— 而「带
        tool_calls 的 AIMessage 后面必须跟齐对应的 ToolMessage」是服务端的硬
        要求，下一次请求会因此被直接拒掉。实测过一个 write_todos ×2 + read_file
        ×1 的回合：`read_file` 的回执缺失，transcript 非法。

        所以每个调用都补一条错误回执，语义上也更干净：这一轮整个作废。
        """
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        if sum(1 for c in calls if c["name"] == write_todos.name) <= 1:
            return None
        return Command(
            goto="before_model",
            update={
                "messages": [
                    ToolMessage(
                        content=_REJECT_PARALLEL,
                        tool_call_id=call["id"],
                        # ToolNode 会替真工具把名字填进回执，我们得自己填，
                        # 否则 UI 那条工具结果没有名字可显示。
                        name=call["name"],
                    )
                    for call in calls
                ]
            },
        )

"""Agent 核心层：跑图，把 LangGraph 的原始流翻译成领域事件。

这个文件是**唯一**同时认识 LangGraph 和领域事件的地方，也就意味着：
换掉 UI 不用动它，换掉框架（比如从 LangGraph 换成手写循环）也只动它。

对外只有三个东西：`JCAEngine`、`stream_turn()`（异步生成器）、两个只读属性。
UI 那边拿到的全是 engine/events.py 里的 dataclass。

## 这件事现在被拆成了两层

「怎么调模型」不再写在这个文件里，而在 `engine/middleware/`。图的形状是：

    START → before_model → agent → after_model ─┬─→ tools ─┐
                                                 └─→ END    │
                                       ↑                    │
                                       └────────────────────┘

两个钩子节点**都在 ReAct 循环里**（注意 tools 回到的是 before_model，不是
直接回 agent）。这一点是刻意的：每一轮「模型 → 工具 → 再模型」都要重新跑一遍
中间件，因为上下文本来就变了 —— 上一轮的工具结果可能让某段 system prompt
不再适用。代价是每轮多两次函数调用，相对一次模型请求可以忽略。

这个文件负责的是**图的形状**：节点怎么连、异常怎么翻译成事件、重试往哪儿跳。
中间件负责的是**每次调用长什么样**。

## 系统提示词为什么不在 state 里了

以前每一轮往 inputs 里塞一条固定 id 的 SystemMessage，靠 `add_messages` 同 id
替换来保证历史里只有一条。现在不塞了 —— 系统提示词由管道在**每次调用**时现造
（`MiddlewarePipeline._invoke` 里拼到消息列表最前面），进不了 checkpointer，
也就不存在「攒了三条一样的系统提示、每一份都被原样发给模型」这个问题。
好处不只是省 token：中间件可以往里面加只在**这次**成立的内容（比如当前清单），
而这种内容本来就不该被永久留在对话历史里。改这里的 inputs 之前请先读
`engine/middleware/base.py` 的模块说明。

## 为什么用 stream_mode=["messages", "updates"] 而不是 astream_events

两种都试过，这里是实测结论（langgraph 1.2.15）：

- `messages` 模式一边吐 `AIMessageChunk`（`meta["langgraph_node"] == "agent"`，
  文本在 `.text` 属性上），一边吐 `ToolMessage`（`node == "tools"`，带
  `name` / `tool_call_id` / `status`）。**工具结果是完整消息一次到达的**，
  不用自己拼。
- `updates` 模式给的是「节点级状态增量」。它比 `messages` 强的地方只有一处，
  但很关键：**工具调用的参数**。`messages` 里工具申请是分片的
  `tool_call_chunks`，`args` 是逐字符拼接的 JSON 片段；自己拼不可靠 ——
  `AIMessageChunk.__add__` 的语义是「name 逐片拼接」，实测拼出来会变成
  `"list_dirlist_dirlist_dir"`。而 `updates` 里 `payload["agent"]["messages"]`
  是这个节点返回的**完整** `AIMessage`，`tool_calls` 已经解析好了。
  让 LangChain 的解析器去干活，比自己手写稳。

所以分工是：**文本和工具结果走 messages，工具调用走 updates**。

## 一个必须过滤的坑

`safety/guard.py` 在重试时注入的那条 `HumanMessage`（`RETRY_NOTE`）**也会**
从 `messages` 模式里流出来，节点名同样叫 `agent`。不过滤的话，用户会在终端上
看见一句像是模型自言自语的「上一步没有成功，请换一种说法……」。
这里把它翻译成一条 `Notice`：用户该看到的是「正在重试」这个状态，而不是
那句写给模型看的话。

（用户自己输入的那条消息不会被回显：`_messages.py` 的 `on_chain_start`
会把输入消息的 id 记进 `seen`，所以它不会二次出现。）
"""

import os
from typing import Any, AsyncIterator, Sequence

from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from logger import get_logger
from prompts import build_system_prompt
from safety import (
    MAX_SAFETY_RETRIES,
    RETRY_NOTE,
    SAFETY_CODE,
    build_retry_message,
)
from tools import build_tools

from .events import (
    AgentError,
    AgentEvent,
    AssistantText,
    Notice,
    ToolCall,
    ToolResult,
    TurnDone,
)
from .middleware import BaseMiddleware, MiddlewarePipeline, TodoMiddleware

logger = get_logger(__name__)

# 工具输出给终端看的最长字符数。工具的返回值本身可能是两万字（read_file 的上限），
# 但终端上没人读得了，截断这件事放在 engine 做：UI 不该自己决定「多少算长」，
# 它只需要知道「被截断了，原文共 N 字符」。
MAX_TOOL_OUTPUT = 2000

# 默认模型名。原来写死在 main.py 里，现在收进来当默认参数，
# 调用方（main.py）仍然可以覆盖。
DEFAULT_MODEL = "DeepSeek-V4.1-Flash"

# 默认挂上的中间件。想加就写一个 BaseMiddleware 子类塞进来 ——
# 图的结构不用改，见 engine/middleware/base.py 的模块说明。
DEFAULT_MIDDLEWARES: tuple[type[BaseMiddleware], ...] = (TodoMiddleware,)


class AgentState(MessagesState):
    """MessagesState 只声明了 messages，这里再加一个计数字段。

    没给它配 reducer，所以是「后写覆盖先写」，每次进节点读到的都是最新值。

    中间件要加的字段**不写在这里**，而是各自用 `state_schema` 声明、
    由 `MiddlewarePipeline.build_state_schema` 在建图时合并进来 ——
    加一个中间件不该来动这个文件。
    """

    safety_retries: int


class JCAEngine:
    """跑一次会话的引擎。

    图在构造时编译一次，之后每次 `stream_turn()` 都往同一个 thread 上追加消息。
    多轮上下文交给 checkpointer 维护 —— 这是 UI 层能保持干净的关键：
    **它不需要持有任何 message 列表**，自然也漏不出 LangChain 的类型。

    构造参数里的 `llm` / `tools` / `system_prompt` / `middlewares` 都是给测试用的
    注入点：传一个假的 BaseChatModel + 一个不碰真实文件系统的工具，就能离线跑通
    整条链路（见 tests/test_engine_events.py），不用 API key。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        llm: BaseChatModel | None = None,
        tools: list | None = None,
        system_prompt: str | None = None,
        middlewares: Sequence[BaseMiddleware] | None = None,
        thread_id: str = "jca-cli",
    ) -> None:
        self._model_name = model
        self._thread_id = thread_id
        self._turns = 0

        # 注意变量名别叫 tools —— 会把上面 import 的 tools 包名遮掉
        # （原来 main.py 里就有这个地雷，搬过来顺手改掉）。
        #
        # tools 这个注入点不是为了「可配置」，是为了测试：不注入的话，测试里
        # 假模型喊一句 list_dir，真 ToolNode 就会去翻本机的工作目录 —— 断言
        # 「工具返回了什么」这件事就变成了断言「运行测试的机器上有什么文件」。
        tool_list = build_tools() if tools is None else tools
        if llm is None:
            llm = ChatOpenAI(
                model=model,
                # 此时会自动从 .env 读取 DEEPSEEK_API_KEY
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.ai"),
                temperature=0,
            )

        # 传 None 用默认那组，传 [] 就是「一个中间件都不要」—— 测试要用这个区分
        # 两种意图，所以不能写成 `middlewares or DEFAULT`（那会把 [] 也当成没传）。
        mw_list = (
            [cls() for cls in DEFAULT_MIDDLEWARES] if middlewares is None else list(middlewares)
        )

        # 模型不再在这里 bind_tools：工具集是**每次调用**算出来的
        # （见 MiddlewarePipeline.call_model），因为中间件可能要按当前状态加减工具。
        self._pipeline = MiddlewarePipeline(
            llm,
            mw_list,
            system_prompt=system_prompt or build_system_prompt(),
            tools=tool_list,
        )

        # 图的 state 类型 = 基础 state + 各中间件声明的字段。
        state_schema = MiddlewarePipeline.build_state_schema(AgentState, mw_list)

        workflow = StateGraph(state_schema)
        # 三个节点都必须显式写 input_schema=state_schema，这不是啰嗦。
        #
        # 不写的话，LangGraph 会拿节点函数**第一个参数的类型标注**当这个节点的
        # 输入 schema（state.py 里 add_node 的 inferred_input_schema 那段，靠
        # get_type_hints 读标注），然后就照着它裁剪传给节点的 state。
        # 问题在于「基础 state + 各中间件加的字段」是**运行时**拼出来的，没有
        # 任何静态标注能描述它 —— 只能写成 `state: AgentState`，而 AgentState
        # 里没有 todos。于是中间件加的字段会在进节点前被悄悄滤掉：
        # 图照跑、工具照写、checkpointer 里也存得住，只有节点里读不到。
        # 这个错误的全部症状就是「中间件看不见自己的状态」，一句报错都没有。
        workflow.add_node("before_model", self._before_model_node, input_schema=state_schema)
        workflow.add_node("agent", self._agent_node, input_schema=state_schema)
        workflow.add_node("after_model", self._after_model_node, input_schema=state_schema)
        workflow.add_node("tools", ToolNode(self._pipeline.tools))
        workflow.add_edge(START, "before_model")
        workflow.add_edge("before_model", "agent")
        # agent 没有静态出边，下一步完全由它返回的 Command(goto=...) 决定。两个去向：
        #   - 正常答复 → after_model →(tools_condition)→ tools / END
        #   - 被内容判定拦下 → before_model（**绕过 after_model**）
        #
        # 为什么这里不能写 `add_edge("agent", "after_model")`：`Command.goto` 不是
        # 「改走另一条边」，而是**在静态边之外再加一条分支**。实测（langgraph 1.2.15）
        # 打印每个超步拿到的节点名，重试那一轮是：
        #
        #     before_model / agent / before_model + after_model（同一个超步）
        #
        # 两个后果。一是被拦下的那一轮**根本没有成功的模型答复**，却跑了一遍
        # after_model，语义上就是错的；二是两个钩子节点落进同一个超步，只要有任何
        # 中间件在两个钩子里写同一个 state 键，LangGraph 就直接抛
        # 「InvalidUpdateError: At key '...': Can receive only one value per step」
        # —— 而「压缩历史」那类中间件恰好有理由两边都写。把静态边去掉，这个可能性
        # 在结构上就不存在了。
        #
        # 条件边仍然挂在 after_model 上（而不是 agent 上）：这样模型的每一次答复
        # 都会先过一遍 after_model，才决定要不要调工具。
        workflow.add_conditional_edges("after_model", tools_condition)
        # 工具跑完回到 before_model 而不是 agent —— 这就是「两个钩子都在循环里」。
        workflow.add_edge("tools", "before_model")

        # checkpointer 让同一个 thread_id 的多轮对话自动接上历史。
        # 存在内存里就够：这个进程活多久，会话就活多久。
        self._graph = workflow.compile(checkpointer=InMemorySaver())

    # —— 对外只读属性 ————————————————————————————————————————————
    @property
    def model_name(self) -> str:
        """模型名，横幅上要显示。"""
        return self._model_name

    @property
    def turns(self) -> int:
        """已经正常完成的轮次数。"""
        return self._turns

    # —— 图节点 ————————————————————————————————————————————————
    def _before_model_node(self, state: AgentState):
        """进 agent 之前：把所有中间件的 before_model 跑一遍。

        返回 None 表示不改任何状态 —— LangGraph 认这个，比返回空 dict 更明确。
        """
        return self._pipeline.before_model(state)

    def _after_model_node(self, state: AgentState):
        """模型答复之后、决定要不要调工具之前：跑一遍中间件的 after_model（逆序）。"""
        return self._pipeline.after_model(state)

    def _agent_node(self, state: AgentState):
        """一次模型调用。

        真正干活的是管道：它建一份本次调用的请求（system prompt、工具集都由中间件
        现算），穿过所有 wrap_model，把答复拿回来。这里只管**出错怎么办**。
        """
        retries = state.get("safety_retries", 0)

        try:
            response = self._pipeline.call_model(state)
        except ModelInvalidRequestError as e:
            # 服务端对**整个请求体**做内容判定，命中就返回 400。
            # 不是内容问题就照原样抛出去（认证失败、限流那些本来也不是这个异常，
            # 而「工具参数写错了」虽然同样是 400，但换措辞永远修不好，所以还要认判定码）。
            if not self._is_safety_block(e):
                raise
            # 必须设上限。判定的对象是**整个请求体**，而我们只是往里追加了一条消息，
            # 触发它的那段历史原封不动还在——所以重试经常救不回来，不加限制就是死循环。
            if retries >= MAX_SAFETY_RETRIES:
                logger.error("连续被内容判定拦下，放弃重试", retries=retries)
                raise

            # 把这次「拒绝」包成一条 HumanMessage 放回对话，再回到自己重来一次。
            #
            # 为什么用 Command 而不是普通 return：普通 return 之后由配置的边决定
            # 下一步，而 agent 的出边由 `Command(goto="after_model")` 给出，接着
            # `tools_condition` 会看到最后一条是没有工具申请的 HumanMessage ——
            # 图会直接结束，回不到 agent。Command(goto=...) 是节点自己指定下一步。
            #
            # 去的是 before_model 而不是 after_model/agent：重试是一次新的模型调用，
            # 也该重新走一遍中间件（提示词现造、工具集现算）；而 after_model 是
            # 「模型答完之后」的钩子，这一轮根本没有成功的答复，不该跑。
            logger.warning("被内容判定拦下，准备重试", attempt=retries + 1, limit=MAX_SAFETY_RETRIES)
            return Command(
                goto="before_model",
                update={
                    "messages": [build_retry_message()],
                    "safety_retries": retries + 1,
                },
            )

        # 成功了：唯一一条走 after_model 的路径。这一轮算成功，重试额度清零
        # （额度按「连续被拦」算，不是按累计次数算）。
        return Command(
            goto="after_model",
            update={"messages": [response], "safety_retries": 0},
        )

    @staticmethod
    def _is_safety_block(error: BaseException) -> bool:
        """这次失败是不是被服务端的内容判定拦下来的。

        这里没有复用 safety.guard.is_safety_block()，因为那个函数会顺手记一条
        warning 日志。它已经在 agent_node 的 except 分支里记过一次了，
        这里再来一次就是同一件事记两遍，排查时反而要多看一行。
        判定条件与它保持一致：先看 code 字段，再看整段报错文字。
        """
        return getattr(error, "code", None) == SAFETY_CODE or SAFETY_CODE in str(error)

    # —— 主入口 ————————————————————————————————————————————————
    async def stream_turn(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """跑一轮对话，把过程翻译成领域事件吐出来。

        异步生成器：调用方 `async for event in engine.stream_turn(...)` 边收边画。
        生成器被提前关掉（UI 被 Ctrl-C 打断）时，LangGraph 那一侧的清理由
        `async for` 的收尾负责，这里不用手动善后。
        """
        inputs = {
            "messages": [
                # 这里**只**放用户这一轮说的话。系统提示词不进 state ——
                # 它由中间件管道在每次模型调用时现拼（见模块说明和
                # engine/middleware/base.py），所以历史里不会再攒下重复的提示词，
                # 也不会把「只对这一次成立」的内容永久留在对话里。
                ("user", prompt),
            ]
        }
        config = {"configurable": {"thread_id": self._thread_id}}
        logger.info("本轮开始", thread=self._thread_id, chars=len(prompt))

        try:
            async for mode, payload in self._graph.astream(
                inputs, config, stream_mode=["messages", "updates"]
            ):
                for event in self._translate(mode, payload):
                    yield event
        except ModelInvalidRequestError as e:
            # 重试额度耗尽就是从这里出来的（agent_node 里直接 raise 了）。
            # 走到这儿说明这一轮真的没救了，翻译成一条错误事件交给 UI 显示，
            # 而不是让栈打在终端上 —— 用户需要知道「发生了什么」，不需要看 traceback。
            kind = "safety" if self._is_safety_block(e) else "api"
            logger.warning("本轮失败", kind=kind, error=str(e)[:200])
            yield AgentError(
                message=(
                    "连续被服务端的内容判定拦下，已放弃重试。可以换个说法再试一次。"
                    if kind == "safety"
                    else f"调用模型失败：{e}"
                ),
                kind=kind,
            )
            return
        except Exception as e:
            # 网络、认证、限流…… 都归到这一类。用 logger.exception 把栈留下来：
            # 终端上不显示栈，不等于栈没用，日志文件里要能查到。
            logger.exception("本轮出错", thread=self._thread_id)
            yield AgentError(message=f"出错了：{e}", kind="api")
            return

        self._turns += 1
        logger.info("本轮结束", turns=self._turns)
        yield TurnDone(turns=self._turns)

    # —— 流翻译 ————————————————————————————————————————————————
    def _translate(self, mode: str, payload: Any) -> list[AgentEvent]:
        """把一条原始流片段翻译成 0..N 个领域事件。

        写成返回列表而不是生成器，是因为要先判断类型再决定吐几个：
        `updates` 里一条 `AIMessage` 可能带着好几个工具调用。
        """
        if mode == "messages":
            return self._translate_message(payload)
        if mode == "updates":
            return self._translate_update(payload)
        return []

    @staticmethod
    def _translate_message(payload: Any) -> list[AgentEvent]:
        """`messages` 模式：文本增量和工具结果。"""
        chunk, _meta = payload  # (message, metadata)，metadata 这次用不上

        if isinstance(chunk, AIMessageChunk):
            # 用 .text 而不是 .content：content 在开启 content blocks 时会变成
            # 一个 block 列表，.text 会把里面 type == "text" 的块拼起来。
            # 注意它是**属性**，不是方法 —— .text() 这种写法在 langchain-core 1.0
            # 起就废弃了。
            return [AssistantText(chunk.text)] if chunk.text else []

        if isinstance(chunk, ToolMessage):
            text = chunk.text
            return [
                ToolResult(
                    id=chunk.tool_call_id,
                    name=chunk.name or "",
                    ok=chunk.status == "success",
                    output=_truncate(text, MAX_TOOL_OUTPUT),
                    chars=len(text),
                )
            ]

        # 别的消息类型（比如重试注入的 HumanMessage，或者将来框架新增的类型）：
        # 一律不显示。宁可少显示一点，也不要把内部对话泄露到用户眼前。
        return []

    @staticmethod
    def _translate_update(payload: Any) -> list[AgentEvent]:
        """`updates` 模式：节点级状态增量。工具调用从这里取。

        这里同样要按类型过滤，原因见模块开头：重试用的那条 HumanMessage
        节点名也叫 agent，会混进来。

        注意**任何**节点返回的 AIMessage 都会被当成「模型申请调工具」翻译出去。
        现在只有 agent 节点会吐 AIMessage，所以没问题；但中间件从 `before_model`
        往 messages 里塞消息是设计上允许的（比如压缩历史），那种情况下别塞
        AIMessage —— 用户会在终端上看到一个自己没申请过的「工具调用」。
        """
        events: list[AgentEvent] = []
        for _node, update in payload.items():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages") or []:
                if isinstance(msg, HumanMessage) and msg.text == RETRY_NOTE:
                    # 这句是写给模型看的，不是写给用户看的。翻译成状态提示。
                    events.append(
                        Notice("被服务端内容判定拦下，正在换一种说法重试", kind="safety_retry")
                    )
                elif isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        events.append(
                            ToolCall(id=call["id"], name=call["name"], args=call["args"])
                        )
                    for bad in msg.invalid_tool_calls:
                        # 模型吐出的参数不是合法 JSON。ToolNode 会把它变成一条错误
                        # 消息还回去，模型有机会自己改。这里如实告诉用户发生了什么，
                        # 免得只看到「工具失败了」却不知道为什么。
                        events.append(
                            ToolCall(
                                id=bad.get("id") or "",
                                name=bad.get("name") or "?",
                                args={},
                                invalid_args=bad.get("args") or "",
                            )
                        )
        return events


def _truncate(text: str, limit: int) -> str:
    """截断长文本（只在这里用，所以不放进 tools/ 里当公共工具）。"""
    if len(text) <= limit:
        return text
    return text[:limit]


__all__ = [
    "DEFAULT_MIDDLEWARES",
    "DEFAULT_MODEL",
    "MAX_TOOL_OUTPUT",
    "AgentState",
    "JCAEngine",
]

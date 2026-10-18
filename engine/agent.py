"""Agent 核心层：跑图，把 LangGraph 的原始流翻译成领域事件。

这个文件是**唯一**同时认识 LangGraph 和领域事件的地方，也就意味着：
换掉 UI 不用动它，换掉框架（比如从 LangGraph 换成手写循环）也只动它。

对外只有三个东西：`JCAEngine`、`stream_turn()`（异步生成器）、两个只读属性。
UI 那边拿到的全是 engine/events.py 里的 dataclass。

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
from typing import Any, AsyncIterator

from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
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

logger = get_logger(__name__)

# 工具输出给终端看的最长字符数。工具的返回值本身可能是两万字（read_file 的上限），
# 但终端上没人读得了，截断这件事放在 engine 做：UI 不该自己决定「多少算长」，
# 它只需要知道「被截断了，原文共 N 字符」。
MAX_TOOL_OUTPUT = 2000

# 默认模型名。原来写死在 main.py 里，现在收进来当默认参数，
# 调用方（main.py）仍然可以覆盖。
DEFAULT_MODEL = "DeepSeek-V4.1-Flash"


class AgentState(MessagesState):
    """MessagesState 只声明了 messages，这里再加一个计数字段。

    没给它配 reducer，所以是「后写覆盖先写」，每次进节点读到的都是最新值。
    """

    safety_retries: int


class JCAEngine:
    """跑一次会话的引擎。

    图在构造时编译一次，之后每次 `stream_turn()` 都往同一个 thread 上追加消息。
    多轮上下文交给 checkpointer 维护 —— 这是 UI 层能保持干净的关键：
    **它不需要持有任何 message 列表**，自然也漏不出 LangChain 的类型。

    构造参数里的 `llm` 和 `tools` 是给测试用的注入点：传一个假的
    BaseChatModel + 一个不碰真实文件系统的工具，就能离线跑通整条链路
    （见 tests/test_engine_events.py），不用 API key。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        llm: BaseChatModel | None = None,
        tools: list | None = None,
        system_prompt: str | None = None,
        thread_id: str = "jca-cli",
    ) -> None:
        self._model_name = model
        self._system_prompt = system_prompt or build_system_prompt()
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
        self._llm = llm.bind_tools(tool_list)

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(tool_list))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

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
    def _agent_node(self, state: AgentState):
        retries = state.get("safety_retries", 0)

        try:
            response = self._llm.invoke(state["messages"])
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
            # 为什么用 Command 而不是普通 return：普通 return 之后由 tools_condition
            # 决定下一步，而它会看到最后一条是没有工具申请的 HumanMessage —— 图会直接
            # 结束，回不到 agent。Command(goto=...) 是节点自己指定下一步，能绕开条件边。
            logger.warning("被内容判定拦下，准备重试", attempt=retries + 1, limit=MAX_SAFETY_RETRIES)
            return Command(
                goto="agent",
                update={
                    "messages": [build_retry_message()],
                    "safety_retries": retries + 1,
                },
            )

        # 这一轮成功了就清零：重试额度按「连续被拦」算，不是按累计次数算
        return {"messages": [response], "safety_retries": 0}

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
                # 固定 id 是有意的：`add_messages` 对同 id 的消息是**替换**而不是
                # 追加，所以每一轮重新塞一遍系统提示，历史里也只会有一条。
                # 不给 id 的话（默认每个对象一个随机 uuid），跑三轮历史里就有
                # 三条一模一样的系统提示，每一条都会被原样发给模型。
                #
                # 实测确认过这一点（见 tests/test_engine_events.py 的多轮用例）。
                SystemMessage(content=self._system_prompt, id="system-prompt"),
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


__all__ = ["JCAEngine", "AgentState", "MAX_TOOL_OUTPUT", "DEFAULT_MODEL"]

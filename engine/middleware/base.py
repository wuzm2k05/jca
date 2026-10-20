"""中间件管道：把「怎么调模型」从图节点里抽出来。

## 为什么要有这一层

在加中间件之前，`_agent_node` 里写死了三件事：系统提示词是构造时定死的一段
字符串、工具在构造时 `bind_tools` 一次绑死、每一轮往 state 里塞一份 SystemMessage。
想改其中任何一件（按需挂工具、注入动态上下文、压缩历史、RAG 注入 skill/mcp），
都只能去改那个节点本身 —— 改一处加一个 `if`，几轮下来就没有人能读懂它了。

抽出来之后，「一次模型调用」变成一个可以被别人插手的东西：

    before_model 节点  →  所有中间件 before_model(state)     （改 state）
    agent 节点         →  pipeline.call_model(state)        （wrap_model 洋葱 + 真调用）
    after_model 节点   →  所有中间件 after_model(state)      （改 state）

三个钩子各管一件事，分工是按**改什么**分的：

- `before_model` / `after_model` 改的是 **state**：它们跑在图节点里，返回值就是
  状态更新。适合「在调用前把某些东西准备好」「调用后收尾」。
- `wrap_model` 改的是**这一次调用**：system prompt、工具集、要发的消息。
  它拿到的 `ModelRequest` 是一次性的，调用完就丢 —— 下一轮 ReAct 会重新建一份。
  这是 req.md 反复强调的那点：**system prompt 每次调用现造，不存进历史**。存进
  历史会有两个后果：一份已经过期的上下文被永远带在对话里（每一轮都要重新发给
  模型、白烧 token），以及「谁在什么时候改过它」变得不可知。

## 洋葱模型

`wrap_model(request, handler)` 收到的是「剩下的中间件 + 真正的调用」，调用
`handler(request)` 就是交棒给下一层。不调 handler 直接返回一个 AIMessage 也是
合法的（那就是「这一层替模型做了决定」），但本项目的中间件都只是往 request 上
加东西。

多个中间件按声明顺序由外到内嵌套：A 在 B 前面声明，A 的 wrap_model 先跑，
A 拿到的 handler 里包着 B。于是 A 追加的 system prompt 排在 B 的前面 ——
也就是**越靠后声明的中间件，越靠近请求末尾**，模型对它的印象越深
（提示词的末尾是模型注意力最集中的地方）。想让某个中间件的指令优先级最高，
就把它放在列表最后。

`before_model` 与 `after_model` 也是洋葱，但方向相反（after 逆序）：before 里
A 先动手、B 后动手，after 里就该 B 先收尾、A 后收尾。这是「包起来的东西按逆序
拆开」那条通用约定。
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypedDict, get_type_hints

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool

# 状态增量。图节点返回的就是它：`{字段名: 新值}`。
StateUpdate = dict[str, Any]

# 一次模型调用的「末端」：把 request 真正发出去，拿回一条 AIMessage。
# 最内层的 `_invoke` 就是它；中间的中间件拿到的 handler 是「下一层」。
ModelHandler = Callable[["ModelRequest"], AIMessage]


@dataclass
class ModelRequest:
    """**一次**模型调用的请求。每次调用新建一份，用完即丢。

    刻意不是 state 的一部分：它里面的 system_prompt 和 tools 都是「这次调用
    长什么样」，而不是「会话是什么样」。混进 state 就会被 checkpointer 存下来、
    被下一轮原样带到模型面前。

    - `state`：整份 AgentState（中间件按需读，比如 todo 清单）。
    - `messages`：要发的对话消息，**不含** system prompt（那个单独一个字段，
      由管道在最后拼到最前面）。
    - `system_prompt`：基础提示词 + 各中间件追加的部分。
    - `tools`：这次要绑给模型的工具，可增可减。
    """

    state: Any
    messages: list[AnyMessage]
    system_prompt: str
    tools: list[BaseTool]


class BaseMiddleware:
    """中间件的接口。三个钩子都有默认实现，子类只覆盖需要的那个。

    `state_schema` 和 `tools` 是**类属性**（不是构造参数），因为它们描述的是
    「这个中间件是什么」，在 `MiddlewarePipeline` 建图的时候就要读到 —— 那时
    还没有实例，或者说实例化参数不该影响图的形状。

    - `state_schema`：这个中间件要往 AgentState 里加的字段（一个 TypedDict）。
      `None` 表示不需要额外的状态。
    - `tools`：这个中间件挂载的工具。它们**不**进 `tools/` 那个注册表 ——
      见 `TodoMiddleware` 里的说明。
    """

    # 往 AgentState 里加的字段。None = 不需要额外状态。
    state_schema: type | None = None
    # 这个中间件提供的工具。
    tools: Sequence[BaseTool] = ()

    def before_model(self, state: Any) -> StateUpdate | None:
        """进入 agent 节点之前跑。返回值是状态增量，None 表示不改。

        在图节点里跑，所以拿到的 `state` 是**整个会话状态**（含历史消息）。
        想改的是「模型的输入」请改用 `wrap_model` —— 那里改的不会留在状态里。

        可以往 `messages` 里加消息（比如历史上太长时塞一条摘要），但**不要放
        AIMessage**：`stream_turn` 的 updates 翻译会把任何 AIMessage 上的
        `tool_calls` 当成「模型申请调工具」报给用户（见 `_translate_update`），
        而那其实是我们自己写的。
        """
        return None

    def after_model(self, state: Any) -> StateUpdate | None:
        """模型答复之后、决定要不要调工具之前跑。返回值是状态增量，None 表示不改。

        这时 state 里最后一条已经是模型刚吐出来的 AIMessage，所以拿得到
        `tool_calls`。注意它**在工具执行之前** —— 工具的结果这里还看不到。

        **这里不要动 `messages`。** 下一步的去向由 `tools_condition` 决定，它读的是
        `messages` 的最后一条：追加一条 HumanMessage 会让它认为「模型没有申请工具」，
        于是图直接结束 —— 模型刚申请的那个工具**再也不会被执行**，而且没有任何报错。
        （实测确认过。）要加提示就加在 `before_model` 或 `wrap_model` 里。
        """
        return None

    def wrap_model(self, request: ModelRequest, handler: ModelHandler) -> AIMessage:
        """包住这次模型调用。默认什么都不做，直接交棒。

        子类通常长这样：

            def wrap_model(self, request, handler):
                request = replace(request, system_prompt=request.system_prompt + "\\n" + 我的段落)
                return handler(request)

        用 `dataclasses.replace` 造一份新的，而不是就地改 `request`：
        就地改会影响到同一次调用里排在**前面**的中间件（它们持有的还是同一个
        对象），那种「谁改了我的提示词」的问题极难查。
        """
        return handler(request)


class MiddlewarePipeline:
    """把所有中间件串成一条管道，并负责真正调模型。

    它是**唯一**认识 BaseChatModel 的地方，也是图节点唯一需要打交道的东西：
    `before_model` / `after_model` / `call_model` 三个方法正好对应三个节点。
    """

    def __init__(
        self,
        llm: BaseChatModel,
        middlewares: Sequence[BaseMiddleware],
        *,
        system_prompt: str,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        self._llm = llm
        self._middlewares = list(middlewares)
        self._system_prompt = system_prompt
        self._base_tools = list(tools)

    # —— 装配信息 ————————————————————————————————————————————
    @property
    def middlewares(self) -> list[BaseMiddleware]:
        return list(self._middlewares)

    @property
    def tools(self) -> list[BaseTool]:
        """常驻工具 + 各中间件挂载的工具。图里的 ToolNode 用这一份。"""
        extra = [tool for mw in self._middlewares for tool in mw.tools]
        return [*self._base_tools, *extra]

    @classmethod
    def build_state_schema(cls, base: type, middlewares: Sequence[BaseMiddleware]) -> type:
        """把基础 state 与各中间件声明的字段合并成一个 TypedDict。

        为什么是动态建类而不是让每个中间件自己去改全局的 AgentState：中间件的
        状态字段是**中间件自己的事**，加一个中间件不该去动 `engine/agent.py`。
        合并只发生在建图那一刻，之后就是普通的 TypedDict，没有额外开销。

        **读注解用 `get_type_hints(..., include_extras=True)`，不是直接读
        `__annotations__`。** 两个理由，都实测过：

        1. `include_extras=True` 才会保留 `Annotated[...]`。`messages` 带的
           `add_messages` reducer、以及 `todos` 带的 `_last_write_wins`，都藏在
           `Annotated` 里。默认的 `get_type_hints` 会把它们**剥掉**，合并出来的
           schema 看上去字段齐全，实际已经丢了 reducer —— 历史消息会变成
           「后写覆盖先写」，而这一条直到多轮对话时才看得出来。
        2. 它会解析 ForwardRef。`MessagesState.messages` 在注解里就是一个
           `ForwardRef`（`Annotated[list[AnyMessage], add_messages]` 字符串形态），
           直接搬进新 TypedDict 未必解析得开。

        另外，用 `__annotations__` 还有第三种病：一旦某个中间件模块加了
        `from __future__ import annotations`，注解会变成字符串，`NotRequired`
        就推断不出来（`__required_keys__` 会把 `todos` 当成必需字段，第一轮直接
        报缺字段）。`get_type_hints` 顺手把这一类也解决了。

        重名字段直接报错，不静默覆盖：两个中间件抢同一个 key，说明其中一个的
        设计是错的，这种事应该在启动时炸掉，而不是让后声明的那个悄悄赢。
        """
        fields: dict[str, Any] = dict(get_type_hints(base, include_extras=True))
        owner: dict[str, str] = {name: "基础 state" for name in fields}
        for mw in middlewares:
            schema = mw.state_schema
            if schema is None:
                continue
            for name, hint in get_type_hints(schema, include_extras=True).items():
                if name in owner:
                    raise ValueError(
                        f"state 字段 {name!r} 被 {type(mw).__name__} 重复声明"
                        f"（已经被{owner[name]}占用）"
                    )
                fields[name] = hint
                owner[name] = type(mw).__name__
        return TypedDict("AgentState", fields)

    # —— 三个钩子 ————————————————————————————————————————————
    def before_model(self, state: Any) -> StateUpdate | None:
        """正序跑所有中间件的 before_model，把返回值合并成一个状态增量。"""
        return self._collect("before_model", state, self._middlewares)

    def after_model(self, state: Any) -> StateUpdate | None:
        """逆序跑所有中间件的 after_model（洋葱按逆序拆开）。"""
        return self._collect("after_model", state, reversed(self._middlewares))

    @staticmethod
    def _collect(hook: str, state: Any, middlewares) -> StateUpdate | None:
        """调一遍钩子并按顺序合并返回值。

        合并是**后者覆盖前者**（dict 的 update），和 LangGraph 自己的 state
        更新语义一致。全都没返回东西时给 None：LangGraph 对「节点返回 None」
        的处理是「什么都不更新」，比返回一个空 dict 更明确。
        """
        update: StateUpdate = {}
        for mw in middlewares:
            extra = getattr(mw, hook)(state)
            if extra:
                update.update(extra)
        return update or None

    # —— 真正的调用 ————————————————————————————————————————————
    def call_model(self, state: Any) -> AIMessage:
        """建一次请求，穿过所有 wrap_model，把模型答复拿回来。"""
        request = ModelRequest(
            state=state,
            messages=list(state["messages"]),
            system_prompt=self._system_prompt,
            tools=self.tools,
        )

        # 从最内层（真正调用）往外包：列表里靠后的中间件在内层，它**后**跑，
        # 于是它追加的内容落在 system prompt 的更后面（离请求末尾更近）。
        handler: ModelHandler = self._invoke
        for mw in reversed(self._middlewares):
            nxt = handler

            def wrapped(request: ModelRequest, _mw=mw, _nxt=nxt) -> AIMessage:
                return _mw.wrap_model(request, _nxt)

            handler = wrapped
        return handler(request)

    def _invoke(self, request: ModelRequest) -> AIMessage:
        """管道的最内层：把 request 发出去。

        system prompt 在这里才拼进消息列表的最前面 —— 它是**这一次调用**的
        产物，不进 state、也不进 checkpointer。

        工具在这里才 bind：`ModelRequest.tools` 是本次调用算出来的，某个
        中间件可能刚刚把它改过。空工具集不能调 `bind_tools([])`（那会给
        服务端发一个空 tools 数组，有的端点会直接报参数错误），所以分开处理。
        """
        llm = self._llm.bind_tools(request.tools) if request.tools else self._llm
        return llm.invoke([SystemMessage(content=request.system_prompt), *request.messages])


__all__ = [
    "BaseMiddleware",
    "MiddlewarePipeline",
    "ModelHandler",
    "ModelRequest",
    "StateUpdate",
]

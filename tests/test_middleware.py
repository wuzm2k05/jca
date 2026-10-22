"""中间件管道的离线测试：不碰网络，也不碰真实文件系统。

跑法（两种都行）：

    .venv/Scripts/python.exe -m tests.test_middleware
    uv run --with pytest pytest tests/            # 装了 pytest 的话

为什么值得单独测：这一层最容易错的地方**从外面看不见**。

- system prompt 是不是每次现拼的、而且**每次逐字相同**？这两个问题各有各的
  静默失败：前者拼错的话模型照常回答，只是答得不好（「感觉它最近不太听话」）；
  后者抖动的话一切看起来正常，只有服务端的前缀 KV Cache 命中不了，而代价是
  账单和首字延迟 —— 本地跑测试永远看不出来。所以要正面断言「这次请求里有什么」。
- 中间件的顺序？洋葱模型包反了照样跑得通，只是 `after_model` 的收尾顺序和
  `before_model` 对不上，那是一种要等到某个中间件依赖另一个的结果时才发作的病。
- `write_todos` 有没有真的把状态写进去？工具返回一段文字也能让对话继续，
  区别只在下一轮模型看不看得见清单。
- 想拦下一轮时，返回**状态增量**和返回 `Command` 的区别？前者只改 state、
  改不了去向，工具照跑不误 —— 而它同时又是最自然的写法，所以特别值得钉。

这些都要靠「从模型那一侧取证」来钉：假模型把每次收到的消息、每次绑到的工具
记下来，断言的对象是那些记录，而不是跑完之后的状态快照。

两个用例（重试路径、并行 write_todos 的拦截）还顺手钉住了 `Command(goto=...)`
的两个选择 —— `goto="before_model"` 和 `goto="agent"` 只差一个词，但后者会让
中间件在下一次模型调用前被跳过。
"""

import asyncio
import json
import sys
from typing import Iterator, TypedDict, get_args, get_type_hints

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from engine import JCAEngine
from engine.agent import AgentState
from engine.middleware import (
    BaseMiddleware,
    MiddlewarePipeline,
    ModelRequest,
    TodoMiddleware,
    write_todos,
)
from prompts import WRITE_TODOS_SYSTEM_PROMPT

# —— 取证用的记录器 ————————————————————————————————————————————
#
# 和 tests/test_engine_events.py 里一样，用模块级变量而不是往假模型身上挂状态：
# BaseChatModel 是 pydantic 模型，挂自定义字段要额外费手脚。
_SCRIPT: list[list[AIMessageChunk]] = []
_SEEN_REQUESTS: list[list] = []
_BOUND_TOOLS: list[list[str]] = []


class FakeChatModel(BaseChatModel):
    """按脚本吐分片的假模型，顺手把「自己收到了什么」记下来。"""

    @property
    def _llm_type(self) -> str:
        return "fake-middleware"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        # 管道**每次调用**都会 bind，所以这个列表的长度就是「调了几次模型」。
        _BOUND_TOOLS.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        raise NotImplementedError("这个假模型只支持流式")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:  # noqa: ANN001
        _SEEN_REQUESTS.append(list(messages))
        if not _SCRIPT:
            raise RuntimeError("脚本用完了")
        for chunk in _SCRIPT.pop(0):
            yield ChatGenerationChunk(message=chunk)


def _reset() -> None:
    _SCRIPT.clear()
    _SEEN_REQUESTS.clear()
    _BOUND_TOOLS.clear()


def _text(s: str) -> AIMessageChunk:
    return AIMessageChunk(content=s)


def _tool_round(
    name: str, args: dict, call_id: str, text_before: str = ""
) -> list[AIMessageChunk]:
    """一轮「说句话 + 申请调一个工具」的输出。

    `args` 是**真参数对象**，这里自己 `json.dumps` —— 不要手写 JSON 字符串。
    踩过一次：args 得分片送进 `tool_call_chunks`，而分片拼起来之后由
    langchain 解析；手写一段带空格/换行的 JSON 在分片拼接这一步会解析失败，
    于是工具调用被归到 `invalid_tool_calls`，整条链路静默走错分支
    （ToolCall 事件的 args 空、invalid_args 有值）。用真对象序列化就没有
    这种随格式漂移的脆弱。
    """
    args_json = json.dumps(args, ensure_ascii=False)
    # 参数切成三片，模拟真实的增量到达
    third = max(1, len(args_json) // 3)
    return [
        _text(text_before),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": name, "args": args_json[:third], "id": call_id, "index": 0}
            ],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": None, "args": args_json[third : 2 * third], "id": None, "index": 0}
            ],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": None, "args": args_json[2 * third :], "id": None, "index": 0}
            ],
        ),
        AIMessageChunk(content="", chunk_position="last"),
    ]


def _text_round(text: str) -> list[AIMessageChunk]:
    return [_text(text), AIMessageChunk(content="", chunk_position="last")]


def _parallel_tool_round(
    name: str, args_list: list[dict], call_ids: list[str], text_before: str = ""
) -> list[AIMessageChunk]:
    """一轮里**并行**申请多个同名工具调用 —— 模型确实会这么干。

    `tool_call_chunks` 的 `index` 用 0/1 把两次调用分开，这是 langchain 用来
    判断「这一片属于第几个调用」的字段；两次都给 0 的话会被拼成一次调用、
    参数互相污染（实测过）。
    """
    return [
        _text(text_before),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": name,
                    "args": json.dumps(args, ensure_ascii=False),
                    "id": call_id,
                    "index": i,
                }
                for i, (args, call_id) in enumerate(zip(args_list, call_ids, strict=True))
            ],
        ),
        AIMessageChunk(content="", chunk_position="last"),
    ]


def _system_prompts() -> list[str]:
    """每次调用里那条 SystemMessage 的内容。"""
    out = []
    for request in _SEEN_REQUESTS:
        out.append(next(m.content for m in request if isinstance(m, SystemMessage)))
    return out


# —— 记录顺序用的中间件 ————————————————————————————————————————
class _Tracer(BaseMiddleware):
    """只记「我什么时候被调到」，用来断言顺序与合并。"""

    def __init__(self, name: str, log: list[str], *, marker: str = "") -> None:
        self.name = name
        self._log = log
        self._marker = marker

    def before_model(self, state):  # noqa: ANN001, ANN201
        self._log.append(f"before:{self.name}")
        return {f"before_{self.name}": True}

    def after_model(self, state):  # noqa: ANN001, ANN201
        self._log.append(f"after:{self.name}")
        return {f"after_{self.name}": True}

    def wrap_model(self, request: ModelRequest, handler):  # noqa: ANN001, ANN201
        self._log.append(f"wrap:{self.name}")
        return handler(
            request.__class__(
                state=request.state,
                messages=request.messages,
                system_prompt=request.system_prompt + self._marker,
                tools=request.tools,
            )
        )


class _StubLLM:
    """最小可用的假 llm：只实现管道真正会调的那两个方法。

    刻意不是 BaseChatModel 的子类：管道对它只做 `bind_tools` / `invoke`，
    不要求它是谁的子类。少一层继承，这个测试就不会因为 langchain 的基类
    改签名而跟着坏掉 —— 而它要测的东西和 langchain 的基类无关。
    """

    def __init__(self) -> None:
        self.bound: list[list[str]] = []
        self.sent: list[list] = []
        self.reply = AIMessage(content="（假答复）")

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        self.bound.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def invoke(self, messages):  # noqa: ANN001, ANN201
        self.sent.append(list(messages))
        return self.reply


# —— 顺序与合并 ————————————————————————————————————————————————
def test_hooks_run_in_onion_order_and_merge() -> None:
    """before 正序、after 逆序，返回值合并成一个状态增量。

    这个顺序不是审美问题：`before_model` 里 A 先动手、B 后动手，收尾时 B 就该
    先收，A 后收（包起来的东西按逆序拆）。写反了照样跑，但某个中间件在 after
    里依赖另一个的结果时就会读到旧值。
    """
    log: list[str] = []
    pipeline = MiddlewarePipeline(
        _StubLLM(),  # type: ignore[arg-type]
        [_Tracer("a", log), _Tracer("b", log)],
        system_prompt="基础提示",
    )

    assert pipeline.before_model({}) == {"before_a": True, "before_b": True}
    assert pipeline.after_model({}) == {"after_b": True, "after_a": True}
    assert log == ["before:a", "before:b", "after:b", "after:a"], log

    # 都没返回东西时给 None（= 不改状态），而不是空 dict
    assert MiddlewarePipeline(_StubLLM(), [], system_prompt="x").before_model({}) is None  # type: ignore[arg-type]


def test_wrap_model_is_onion_and_prompt_is_rebuilt_each_call() -> None:
    """system prompt 每次调用现拼，而且不累积。

    「不累积」是这条设计的要害：如果某个中间件往共享的 request 上原地追加，
    第二次调用就会带上两份、第三次三份 —— 对话看起来还正常，只是 token 越烧
    越多、指令被稀释。所以这里跑两次，断言第二份的长度和第一份一样。
    """
    log: list[str] = []
    llm = _StubLLM()
    pipeline = MiddlewarePipeline(
        llm,  # type: ignore[arg-type]
        [_Tracer("a", log, marker="[A]"), _Tracer("b", log, marker="[B]")],
        system_prompt="基础提示",
    )
    # 消息用真的 HumanMessage，不是 ("user", ...) 元组：那个简写由 LangGraph
    # 在进图时展开，而这里直接调管道，没人替我们展开。
    state = {"messages": [HumanMessage(content="你好")]}

    pipeline.call_model(state)

    # wrap 的顺序：a 先跑，交棒给 b —— b 声明得更靠后，所以它的内容排在更后面
    assert log == ["wrap:a", "wrap:b"], log
    first = llm.sent[0][0]
    assert isinstance(first, SystemMessage), first
    assert first.content == "基础提示[A][B]", first.content
    # 用户消息原样跟在系统提示后面，管道没有替模型改写它
    assert llm.sent[0][1].content == "你好", llm.sent[0]

    pipeline.call_model(state)
    assert llm.sent[1][0].content == "基础提示[A][B]", llm.sent[1][0].content


def test_tools_are_bound_per_call_not_once() -> None:
    """工具是每次调用绑的，不是构造时绑死 —— 按需挂载的前提。

    空工具集时**不调** bind_tools：绑一个空数组发出去，有的端点会直接报参数
    错误。所以这里的断言分两种情形看。
    """
    llm = _StubLLM()
    pipeline = MiddlewarePipeline(llm, [], system_prompt="x")  # type: ignore[arg-type]
    pipeline.call_model({"messages": []})
    assert llm.bound == [], llm.bound
    assert len(llm.sent) == 1, llm.sent

    # 有中间件挂工具时就绑上，而且是每次调用都绑
    pipeline2 = MiddlewarePipeline(llm, [TodoMiddleware()], system_prompt="x")  # type: ignore[arg-type]
    pipeline2.call_model({"messages": []})
    pipeline2.call_model({"messages": []})
    assert llm.bound == [["write_todos"], ["write_todos"]], llm.bound


# —— 状态 schema 的合并 ————————————————————————————————————————
def _annotation_metadata(hint) -> tuple:
    """取注解里的 `Annotated` 元数据（reducer 就在那里），剥掉 NotRequired 那层。

    为什么不直接读字符串：`get_type_hints(..., include_extras=False)` 会把
    `Annotated` **剥掉**，而 str() 出来的文本里仍然留着 reducer 的名字 ——
    拿它做断言是一句永远为真的废话（原来那条断言就是这样，换了写法才暴露出
    它其实没在测任何东西）。要看的是注解对象上真的挂着那个函数。

    `todos` 的注解是 `NotRequired[Annotated[list[...], reducer]]`，所以要往里
    钻一层：`NotRequired`/`Required` 都是包装，`Annotated` 才是挂东西的那个。
    """
    inner = get_args(hint)[0] if get_args(hint) else hint
    return getattr(inner, "__metadata__", ())


def test_state_schema_merges_middleware_fields() -> None:
    """中间件声明的字段要进图的 state，基础字段一个都不能少。"""
    schema = MiddlewarePipeline.build_state_schema(AgentState, [TodoMiddleware()])
    schema_hints = get_type_hints(schema, include_extras=True)
    assert {"messages", "safety_retries", "todos"} <= set(schema_hints), schema_hints

    # `messages` 必须带着 reducer 一起过来：覆盖它就等于把 add_messages 丢了，
    # 历史消息会变成「后写覆盖先写」，多轮对话直接就没了。
    #
    # 这条断言在改用 get_type_hints 之前是**假通过**的 —— `Annotated` 会被剥掉，
    # 但 str(ForwardRef) 里还留着 "add_messages" 四个字。所以这里看的是
    # `__metadata__`，也就是 reducer 本身真的在不在，而不是字符串里有没有那个名字。
    assert getattr(schema_hints["messages"], "__metadata__", ()), schema_hints["messages"]

    # `todos` 上的 reducer 同理，而且它是「模型并行写两次清单」时唯一的挡箭牌。
    # 它外面还套着 NotRequired，所以先把包装剥掉再找 `Annotated`。
    assert _annotation_metadata(schema_hints["todos"]), schema_hints["todos"]

    # NotRequired 必须活着（合并时被解析成「必需」的话，第一轮就报缺字段）
    assert "todos" not in schema.__required_keys__, schema.__required_keys__
    assert "todos" in schema.__optional_keys__, schema.__optional_keys__

    # 没有 state_schema 的中间件不往里加东西
    assert set(
        MiddlewarePipeline.build_state_schema(AgentState, [_Tracer("a", [])]).__annotations__
    ) == {"messages", "safety_retries"}


def test_state_schema_rejects_duplicate_fields() -> None:
    """两个中间件抢同一个 state 键要当场报错，不能后声明的悄悄赢。

    静默覆盖的后果是所有读那个键的代码都开始拿到别人写的值，而没有任何一处
    会因此报错 —— 这类「两个组件偷偷共用一个全局」的问题排查起来最费时间。
    在建图那一刻炸掉，成本是零。
    """

    class _Squatter(BaseMiddleware):
        class _State(TypedDict):
            todos: int

        state_schema = _State

    try:
        MiddlewarePipeline.build_state_schema(AgentState, [TodoMiddleware(), _Squatter()])
    except ValueError as e:
        assert "todos" in str(e), e
    else:
        raise AssertionError("重复声明 todos 居然通过了")


# —— system prompt 的静态性（保护前缀 KV Cache）———————————————
def _static_prompt_tail() -> str:
    """`TodoMiddleware` 稳定追加的那一段。"""
    return "\n\n" + WRITE_TODOS_SYSTEM_PROMPT


def test_todo_middleware_prompt_stability() -> None:
    """多轮对话里，system prompt 必须**逐字**不变 —— 这是 KV Cache 的前提。

    清单早先是现读 state 渲染后拼在 prompt 末尾的，那让 prompt 尾部每轮都抖：
    服务端的前缀缓存要求「从头开始的最长公共前缀」，尾巴一变，后面所有历史消息
    都跟着失效，TTFT 和 token 成本一起上去。这条测试就是把「不再抖」钉住。

    分两段取证。**第一段**直接驱管道，控制 state 里 `todos` 的内容（空 → 一张 →
    换一张），断言三次拼出来的 prompt 一模一样。**第二段**走完整图跑两轮真实
    对话，模型在这两轮里写出不同的清单，然后把模型**每一次**调用看到的那条
    SystemMessage 全收上来比对 —— 只测 `wrap_model` 是不够的，prompt 还可能
    被别的机制（历史里的 SystemMessage、中间件在别处改 request）污染。
    """
    # —— 第一段：直接驱管道，只换 state 里的清单 ——
    llm = _StubLLM()
    pipeline = MiddlewarePipeline(
        llm,  # type: ignore[arg-type]
        [TodoMiddleware()],
        system_prompt="基础提示",
    )
    empty = {"messages": [HumanMessage(content="你好")], "todos": []}
    one = {
        "messages": [HumanMessage(content="你好")],
        "todos": [{"content": "读 engine", "status": "in_progress"}],
    }
    changed = {
        "messages": [HumanMessage(content="你好")],
        "todos": [
            {"content": "读 engine", "status": "completed"},
            {"content": "写中间件", "status": "pending"},
        ],
    }
    for state in (empty, one, changed):
        pipeline.call_model(state)

    prompts = [m[0].content for m in llm.sent]
    assert len(prompts) == 3, prompts
    assert prompts[0] == prompts[1] == prompts[2], prompts
    assert prompts[0] == "基础提示" + _static_prompt_tail(), prompts[0]
    # 清单本身一个字都不该进 prompt（三份 state 各不相同，文本却完全相同，
    # 上面那条已经保证了；这里再正面钉一句，免得以后有人把标记悄悄加回来）
    assert "Current todo list" not in prompts[0], prompts[0]
    assert "读 engine" not in prompts[0], prompts[0]

    # —— 第二段：走完整图，两轮对话 + 清单变更 ——
    _reset()
    first_todos = [
        {"content": "读一遍 engine", "status": "in_progress"},
        {"content": "写中间件", "status": "pending"},
    ]
    second_todos = [
        {"content": "读一遍 engine", "status": "completed"},
        {"content": "写中间件", "status": "completed"},
        {"content": "补测试", "status": "in_progress"},
    ]
    # 第一轮：写清单 → 收尾答复
    _SCRIPT.append(_tool_round("write_todos", {"todos": first_todos}, "c1", "先记一下。"))
    _SCRIPT.append(_text_round("第一轮结束。"))
    # 第二轮：把清单整份换掉 → 收尾答复
    _SCRIPT.append(_tool_round("write_todos", {"todos": second_todos}, "c2", "更新一下。"))
    _SCRIPT.append(_text_round("第二轮结束。"))

    engine = _build_engine()
    asyncio.run(_collect(engine, "第一轮"))
    asyncio.run(_collect(engine, "第二轮"))

    seen = _system_prompts()
    assert len(seen) == 4, seen  # 两轮各两次模型调用
    assert len(set(seen)) == 1, [s[-80:] for s in seen]
    assert seen[0] == "（测试用系统提示）" + _static_prompt_tail(), seen[0]

    # 清单没进 prompt，但**确实进了 state**（它现在的消费方是 UI 与快照）
    state = engine._graph.get_state({"configurable": {"thread_id": "jca-cli"}})
    assert [t["content"] for t in state.values["todos"]] == [
        "读一遍 engine",
        "写中间件",
        "补测试",
    ]
    # 而且历史里有模型自己写的那次调用 + 回执 —— 清单是靠这个传给模型的
    kinds = [type(m).__name__ for m in state.values["messages"]]
    assert "ToolMessage" in kinds, kinds
    assert "SystemMessage" not in kinds, kinds


# —— 走完整图 ————————————————————————————————————————————————
def _build_engine(**kwargs) -> JCAEngine:
    """默认中间件（含 TodoMiddleware）+ 假模型的 engine。

    `tools=[]`：不要那批真文件工具。这个用例只关心 todo，真工具会去翻运行测试
    的那台机器，把断言变成断言环境。默认中间件还会挂上 write_todos，
    所以模型那边仍然是有工具可用的。
    """
    kwargs.setdefault("system_prompt", "（测试用系统提示）")
    return JCAEngine(llm=FakeChatModel(), tools=[], **kwargs)


def test_write_todos_updates_state_through_history_not_prompt() -> None:
    """一条完整的闭环：模型写清单 → 工具把 state 改了 → 它经**历史**回到模型。

    这条链路上有四段各自成立的机制（工具返回 Command、ToolNode 把 update 合并
    进 state、工具回执进 messages、UI 按 tool_call_id 配上回执），任何一段断掉
    都不会报错 —— 只会让「模型好像忘了自己的计划」。所以这里一次断言到底。

    「回到模型」走的是历史而不是 prompt，这一点是**刻意**的（见
    `test_todo_middleware_prompt_stability`）：prompt 必须逐字不变才能保住
    前缀 KV Cache，而清单在历史里本来就有 —— 模型自己那次 `write_todos` 调用
    加上我们刚还给它的回执。
    """
    _reset()
    todos = [
        {"content": "读一遍 engine", "status": "in_progress"},
        {"content": "写中间件", "status": "pending"},
    ]
    _SCRIPT.append(_tool_round("write_todos", {"todos": todos}, "call_1", "我先记一下要做的事。"))
    _SCRIPT.append(_text_round("已经开始了。"))

    engine = _build_engine()
    events = asyncio.run(_collect(engine, "帮我重构一下"))

    # 1) 两次模型调用，每次都被绑上了 write_todos
    assert _BOUND_TOOLS == [["write_todos"], ["write_todos"]], _BOUND_TOOLS
    # 2) 两次调用的 system prompt 里都有静态的工具说明，且**两次完全相同** ——
    #    清单不在里面，所以它不会因为 state 变了而抖
    first, second = _system_prompts()
    assert "write_todos" in first, first[:200]
    assert first == second, [first[-80:], second[-80:]]
    assert "Current todo list" not in second, second

    # 3) 工具结果照常走领域事件回到 UI（这里没有为 todo 加新事件类型）
    results = [e for e in events if type(e).__name__ == "ToolResult"]
    assert len(results) == 1, results
    assert results[0].name == "write_todos", results[0]
    assert "2 条待办" in results[0].output, results[0].output

    # 4) 状态里真的存下来了（现在的消费方是 UI 与持久化快照）
    state = engine._graph.get_state({"configurable": {"thread_id": "jca-cli"}})
    assert [t["content"] for t in state.values["todos"]] == ["读一遍 engine", "写中间件"]

    # 5) 清单经历史回到模型：第二次调用时那条 AIMessage 带着 tool_calls，
    #    紧随其后的 ToolMessage 是它的回执 —— 模型读这两条就知道计划是什么
    second_call_messages = _SEEN_REQUESTS[1]
    calls = [c for m in second_call_messages if isinstance(m, AIMessage) for c in m.tool_calls]
    assert [c["name"] for c in calls] == ["write_todos"], calls
    assert [c["args"]["todos"] for c in calls] == [todos], calls

    # 6) 系统提示词依旧一条都没进 state 历史
    kinds = [type(m).__name__ for m in state.values["messages"]]
    assert "SystemMessage" not in kinds, kinds


def test_todo_tool_returns_command_with_matching_message() -> None:
    """工具返回的 Command 里必须有一条 tool_call_id 对得上的 ToolMessage。

    少了它 ToolNode 会直接抛错（Every tool call MUST have a corresponding
    ToolMessage），图整条跑不动。这条约束藏在 langgraph 里，值得钉一下 ——
    以后有人「顺手」把返回值改成一段字符串时，测试会立刻说话。
    """
    result = write_todos.func(
        todos=[{"content": "一件事", "status": "pending"}], tool_call_id="c9"
    )
    assert result.update["todos"] == [{"content": "一件事", "status": "pending"}]
    messages = result.update["messages"]
    assert len(messages) == 1 and messages[0].tool_call_id == "c9", messages
    assert "1 条待办" in messages[0].content, messages[0].content


def test_safety_retry_goes_back_through_before_model() -> None:
    """被内容判定拦下重试时，中间件要被**重新**跑一遍。

    重试是一次新的模型调用，它的上下文也该重新算（工具可能变了、清单可能变了）。
    `Command(goto="agent")` 会跳过 before_model —— 图上不报错，只是重试那次用的
    是一份没人维护过的上下文。这里靠 before_model 的调用次数把这件事钉住。
    """
    from langchain_core.exceptions import ModelInvalidRequestError

    from safety import MAX_SAFETY_RETRIES

    class _BlockedError(ModelInvalidRequestError):
        code = "sensitive_words_detected"

    class BlockedModel(FakeChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            raise _BlockedError("sensitive_words_detected")

    log: list[str] = []
    _reset()
    engine = JCAEngine(
        llm=BlockedModel(),
        tools=[],
        system_prompt="（测试用系统提示）",
        middlewares=[_Tracer("t", log)],
    )
    events = asyncio.run(_collect(engine, "随便说点什么"))

    # 首次 + 每次重试，都从 before_model 进来
    befores = [entry for entry in log if entry == "before:t"]
    assert len(befores) == MAX_SAFETY_RETRIES + 1, log
    assert any(type(e).__name__ == "AgentError" for e in events), events


def test_graph_nodes_see_middleware_state_fields() -> None:
    """中间件加的 state 字段必须真的送达每个节点，不能被框架按标注裁掉。

    这条钉的是一个**完全没有报错**的失败：LangGraph 的 `add_node` 会拿节点函数
    第一个参数的类型标注当这个节点的输入 schema，再照着它裁剪传进去的 state。
    而「基础 state + 中间件字段」是运行时拼出来的，标注只能写 `AgentState`
    —— 于是 `todos` 在进节点前被滤掉：工具照跑、checkpointer 里存得住、只有
    中间件读不到。当时的全部症状就是「清单永远不出现」。

    所以这里直接从**节点内部**取证：让一个中间件在 `before_model` 里把
    `state.keys()` 记下来。这比断言最终 state 里有 `todos` 有用得多 ——
    后者恰恰是这个 bug 存在时照样成立的那一条。
    """
    seen_keys: list[set] = []

    class KeySniffer(BaseMiddleware):
        def before_model(self, state):  # noqa: ANN001, ANN201
            seen_keys.append(set(state.keys()))
            return None

    _reset()
    _SCRIPT.append(_text_round("随便答一句。"))
    engine = JCAEngine(
        llm=FakeChatModel(),
        tools=[],
        system_prompt="（测试用系统提示）",
        middlewares=[TodoMiddleware(), KeySniffer()],
    )
    asyncio.run(_collect(engine, "你好"))

    assert seen_keys, "before_model 一次都没跑"
    assert "todos" in seen_keys[0], (
        f"节点看不到 todos —— 多半是 add_node 少写了 input_schema：{seen_keys[0]}"
    )


def test_after_model_does_not_run_on_the_retry_path() -> None:
    """被拦下的那一轮**不该**跑 after_model —— 它根本没有成功的模型答复。

    这条以前是错的，而且错得不显眼：重试用的是 `Command(goto="before_model")`，
    但 `Command.goto` 是**在静态边之外再加一条分支**，不是改道，所以
    `agent → after_model` 那条固定边照样会走。实测两个钩子节点落在**同一个超步**里。

    除了语义不对，这还会让「两个钩子写同一个 state 键」变成图直接报错
    （InvalidUpdateError: Can receive only one value per step）—— 而那正是
    「压缩历史」这类中间件最自然的写法。修法是把 agent 的静态出边去掉、重试
    用 Command 岔开。这里从一个中间件的视角数一遍两个钩子各被调了几次。
    """
    from langchain_core.exceptions import ModelInvalidRequestError

    class _BlockedError(ModelInvalidRequestError):
        code = "sensitive_words_detected"

    count = {"n": 0}

    class OnceBlockedModel(FakeChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            count["n"] += 1
            if count["n"] == 1:
                raise _BlockedError("sensitive_words_detected")
            yield from super()._stream(messages)

    log: list[str] = []
    _reset()
    _SCRIPT.append(_text_round("恢复之后的回答。"))
    engine = JCAEngine(
        llm=OnceBlockedModel(),
        tools=[],
        system_prompt="（测试用系统提示）",
        middlewares=[_Tracer("t", log)],
    )
    asyncio.run(_collect(engine, "随便说点什么"))

    # 两次模型调用（一次被拦 + 一次成功）→ 两次 before_model，但只有一次 after_model
    assert log.count("before:t") == 2, log
    assert log.count("after:t") == 1, log


def test_parallel_write_todos_is_rejected_and_the_model_rewrites() -> None:
    """一轮里并行调两次 `write_todos`：整轮被拒，模型重写，然后才落地。

    工具描述里写着 "should never be called multiple times in parallel"，但模型
    并不总听。两条调用送来的是两份**互相冲突**的完整清单，谁也不该赢 —— 让后
    一张覆盖前一张是凭空替模型做了决定。所以由 `TodoMiddleware.after_model`
    把整轮拒掉：两条调用各配一条错误 ToolMessage，回到 `before_model` 让模型
    看到错误后重写一次。

    ## 这条测试钉住的是一段**做不到**的写法

    直觉上「往 messages 里塞一条错误 ToolMessage」就够了，但那只是改 state，
    **改不了下一步去哪儿** —— `tools_condition` 仍会把这一轮送进 ToolNode，
    两个真工具照跑（实测：`state["todos"]` 落在 `[]`，因为两条并行 update 互相
    覆盖；那条错误消息没有配对的 tool_call，被 UI 丢弃，用户什么都看不见；
    模型一次都没被再调用）。

    所以这里正面断言三件事：错误回执**两条都在**、`todos` **一个字都没写**、
    模型**被再调用了一次**并成功落地第二次的清单。
    """
    _reset()
    _SCRIPT.append(_parallel_tool_round(
        "write_todos",
        [{"todos": [{"content": "第一张", "status": "pending"}]},
         {"todos": [{"content": "第二张", "status": "in_progress"}]}],
        ["p1", "p2"],
        "两条都记一下。",
    ))
    # 被拒之后模型重写：这次只调一次
    _SCRIPT.append(_tool_round(
        "write_todos",
        {"todos": [{"content": "合并后的一张", "status": "in_progress"}]},
        "p3",
        "我合成一份。",
    ))
    _SCRIPT.append(_text_round("记好了。"))
    engine = _build_engine()
    events = asyncio.run(_collect(engine, "帮我重构"))

    kinds = [type(e).__name__ for e in events]
    assert "AgentError" not in kinds, [getattr(e, "message", "") for e in events]

    # 三条 ToolMessage：前两条是拒绝回执，第三条是真正落地的那次
    results = [e for e in events if type(e).__name__ == "ToolResult"]
    assert [r.id for r in results] == ["p1", "p2", "p3"], results
    assert [r.name for r in results] == ["write_todos", "write_todos", "write_todos"], results
    assert all("只能调用一次" in r.output for r in results[:2]), [r.output for r in results]
    assert "1 条待办" in results[2].output, results[2].output

    # 被拒的那一轮一个字都没写进 state；重写的那次才落地
    state = engine._graph.get_state({"configurable": {"thread_id": "jca-cli"}})
    assert [t["content"] for t in state.values["todos"]] == ["合并后的一张"]

    # 模型确实被再调用了一次：三次调用（原本的 + 重写 + 收尾），且没有并行残留
    assert len(_BOUND_TOOLS) == 3, _BOUND_TOOLS
    # transcript 合法：每个 tool_call 都有配对回执，没有悬空的
    dangling: list[str] = []
    for msg in state.values["messages"]:
        if isinstance(msg, AIMessage):
            dangling.extend(c["id"] for c in msg.tool_calls)
        elif type(msg).__name__ == "ToolMessage" and msg.tool_call_id in dangling:
            dangling.remove(msg.tool_call_id)
    assert not dangling, dangling
    # 最后一次调用看到的 prompt 仍是那一份静态文本
    assert len(set(_system_prompts())) == 1, _system_prompts()


def test_rejected_round_gives_every_call_a_receipt() -> None:
    """整轮被拒时，这一轮里**别的**工具调用也必须有回执，不能只有 write_todos。

    改道意味着 ToolNode 整个不跑，所以这一轮里任何一个工具调用都不会有结果。
    只给 `write_todos` 补回执是很自然的写法，也照样能跑 —— 但那样别的调用就
    悬空了，而「带 tool_calls 的 AIMessage 后面必须跟齐对应的 ToolMessage」是
    服务端的硬要求，下一次请求会被直接拒掉。

    实测过一个 `write_todos` ×2 + `read_file` ×1 的回合：只补 write_todos 时
    `read_file` 那个 id 一直悬着，transcript 是**非法**的。所以这里断言每个
    调用都有回执、而且回执带上了工具名（ToolNode 会替真工具填名字，我们手写
    的消息得自己填，否则 UI 那条工具结果没有名字可显示）。
    """
    _reset()
    _SCRIPT.append([
        _text("记两条，再读个文件。"),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": name,
                    "args": json.dumps(args, ensure_ascii=False),
                    "id": call_id,
                    "index": i,
                }
                for i, (name, args, call_id) in enumerate(
                    [
                        ("write_todos", {"todos": [{"content": "A", "status": "pending"}]}, "p1"),
                        ("write_todos", {"todos": [{"content": "B", "status": "pending"}]}, "p2"),
                        ("read_file", {"path": "x.txt"}, "r1"),
                    ]
                )
            ],
        ),
        AIMessageChunk(content="", chunk_position="last"),
    ])
    _SCRIPT.append(_text_round("记好了。"))
    engine = _build_engine()
    events = asyncio.run(_collect(engine, "帮我重构"))

    kinds = [type(e).__name__ for e in events]
    assert "AgentError" not in kinds, [getattr(e, "message", "") for e in events]

    results = [e for e in events if type(e).__name__ == "ToolResult"]
    assert [r.id for r in results] == ["p1", "p2", "r1"], results
    # 名字要带上 —— 悬空/匿名的回执在界面上显示不出来
    assert [r.name for r in results] == ["write_todos", "write_todos", "read_file"], results

    state = engine._graph.get_state({"configurable": {"thread_id": "jca-cli"}})
    dangling: list[str] = []
    for msg in state.values["messages"]:
        if isinstance(msg, AIMessage):
            dangling.extend(c["id"] for c in msg.tool_calls)
        elif type(msg).__name__ == "ToolMessage" and msg.tool_call_id in dangling:
            dangling.remove(msg.tool_call_id)
    assert not dangling, dangling
    # 整轮作废：一个清单都没写进去
    assert state.values["todos"] == [], state.values["todos"]


def test_todos_reducer_starts_empty_not_absent() -> None:
    """配了 reducer 的通道从**空值**开始，不是「键不存在」。

    这一条是 `_last_write_wins` 的副作用，容易被下一个中间件踩到：
    `state.get("todos")` 第一次拿到的是 `[]` 而不是 `None`。凡是判「有没有清单」
    的地方都得按真假值判 —— 别的代码未必像 `render_todos` 那样两种都吃（那个
    函数已经随动态渲染一起删掉了）。

    钉住它还有个当下的理由：`_last_write_wins` 现在**不再是**并行调用的挡箭牌
    （那件事改由 `after_model` 拦截承担了），所以很容易被当成冗余删掉。但它
    仍然要留着 —— 拦截只覆盖 `TodoMiddleware` 装上的情况，而 reducer 保证的是
    「就算有别的路径在同一超步写 todos，图也不会崩」。
    """
    _reset()
    _SCRIPT.append(_text_round("什么都不用做。"))
    engine = _build_engine()
    asyncio.run(_collect(engine, "你好"))

    state = engine._graph.get_state({"configurable": {"thread_id": "jca-cli"}})
    assert state.values["todos"] == [], state.values["todos"]
    # prompt 里没有清单，但静态说明必须在 —— 模型得先知道有这个工具
    assert _system_prompts()[0].endswith(_static_prompt_tail()), _system_prompts()[0]
    assert "Current todo list" not in _system_prompts()[0], _system_prompts()[0]


async def _collect(engine: JCAEngine, prompt: str) -> list:
    return [event async for event in engine.stream_turn(prompt)]


def main() -> int:
    """不装 pytest 也能跑，写法与其它几个 suites 一致。"""
    tests = [
        test_hooks_run_in_onion_order_and_merge,
        test_wrap_model_is_onion_and_prompt_is_rebuilt_each_call,
        test_tools_are_bound_per_call_not_once,
        test_state_schema_merges_middleware_fields,
        test_state_schema_rejects_duplicate_fields,
        test_graph_nodes_see_middleware_state_fields,
        test_todo_middleware_prompt_stability,
        test_write_todos_updates_state_through_history_not_prompt,
        test_todo_tool_returns_command_with_matching_message,
        test_safety_retry_goes_back_through_before_model,
        test_after_model_does_not_run_on_the_retry_path,
        test_parallel_write_todos_is_rejected_and_the_model_rewrites,
        test_rejected_round_gives_every_call_a_receipt,
        test_todos_reducer_starts_empty_not_absent,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"[失败] {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[出错] {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"[通过] {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

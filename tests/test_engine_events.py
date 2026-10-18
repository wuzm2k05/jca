"""引擎的离线测试：不需要 API key，也不需要真终端。

跑法（两种都行）：

    .venv/Scripts/python.exe -m tests.test_engine_events      # 直接跑，无依赖
    uv run --with pytest pytest tests/                        # 装了 pytest 的话

它验证的是**解耦的收益**：engine 只依赖 `BaseChatModel` 这个接口，所以喂一个
手写分片的假模型，就能把「模型 → 工具 → 再回复」整条链路跑完，并检查它翻译成
了哪些领域事件。没有这一步的话，UI 的每一处改动都只能靠真调 API 来验，
那既慢又费钱。

假模型为什么要手写分片而不是用现成的 fake：要复现真实流的形状 ——
文本分片、工具申请分片、以及 `safety/guard.py` 注入的那条重试消息。
这三件事才是翻译逻辑真正要处理的。
"""

import asyncio
import sys
from typing import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

from engine import (
    AgentError,
    AssistantText,
    JCAEngine,
    Notice,
    ToolCall,
    ToolResult,
    TurnDone,
)
from safety import RETRY_NOTE


@tool
def list_dir(path: str = ".") -> str:
    """列出工作目录下某个目录里的文件（测试用的最小工具）。"""
    return "[文件] app.py"


# 每个用例里假模型要说的话。用模块级变量而不是构造参数，是因为
# BaseChatModel 是 pydantic 模型，往它身上挂自定义状态要额外费些手脚。
_SCRIPT: list[list[AIMessageChunk]] = []


class FakeChatModel(BaseChatModel):
    """按脚本吐分片的假模型。

    只实现 `_stream`：`_generate` 直接抛，因为我们要测的正是「流式这条路径」。
    真实的 ChatOpenAI 在流式被要求时也走 `_stream`（chat_models.py 的
    `_should_stream` 分支）。

    `bind_tools` 也得自己实现：基类上是 `raise NotImplementedError`，
    而 engine 在构造时就会调它。假模型不需要真的知道工具是什么
    （工具是 ToolNode 执行、由 `updates` 流回结果的），返回自己就够了。
    """

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        raise NotImplementedError("这个假模型只支持流式")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:  # noqa: ANN001
        if not _SCRIPT:
            raise RuntimeError("脚本用完了")
        for chunk in _SCRIPT.pop(0):
            yield ChatGenerationChunk(message=chunk)


def _text(s: str) -> AIMessageChunk:
    return AIMessageChunk(content=s)


def _tool_call(name: str, args_json: str, id: str) -> AIMessageChunk:
    """真实流里工具申请是一片片来的：name/id 只在第一片出现，args 逐字符拼接。"""
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args_json, "id": id, "index": 0}],
    )


def _agent_round(text_before: str, call: tuple[str, str, str] | None = None) -> list[AIMessageChunk]:
    """组装模型的一轮输出：先说一句，可能再申请一个工具。"""
    chunks = [_text(text_before)]
    if call is not None:
        # 参数切成三片，模拟真实的增量
        name, args, id = call
        third = max(1, len(args) // 3)
        chunks.append(_tool_call(name, args[:third], id))
        chunks.append(AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": args[third : 2 * third], "id": None, "index": 0}
        ]))
        chunks.append(AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": args[2 * third :], "id": None, "index": 0}
        ]))
    chunks.append(AIMessageChunk(content="", chunk_position="last"))
    return chunks


def _build_engine() -> JCAEngine:
    """造一个只用假模型的 engine。

    三个注入点都用了，理由各不相同：
      - `llm`：不碰 API。
      - `tools`：不碰真实文件系统。用真工具的话，假模型喊一句 list_dir，
        ToolNode 就会去翻运行测试的那台机器 —— 断言就变成了断言环境。
      - `system_prompt`：默认那个会去读本机的工作目录，同上。
    """
    return JCAEngine(
        llm=FakeChatModel(),
        tools=[list_dir],
        system_prompt="（测试用系统提示）",
    )


async def _collect(engine: JCAEngine, prompt: str) -> list:
    return [event async for event in engine.stream_turn(prompt)]


def test_text_tool_result_text() -> None:
    """一轮完整链路：模型说话 → 调工具 → 工具返回 → 再说话 → 收尾。"""
    _SCRIPT.clear()
    _SCRIPT.append(_agent_round("我先看一下目录。", ("list_dir", '{"path": "hello_world"}', "call_1")))
    _SCRIPT.append(_agent_round("已经跑通了。"))

    events = asyncio.run(_collect(_build_engine(), "看看 hello_world"))

    texts = [e.delta for e in events if isinstance(e, AssistantText)]
    calls = [e for e in events if isinstance(e, ToolCall)]
    results = [e for e in events if isinstance(e, ToolResult)]

    assert texts == ["我先看一下目录。", "已经跑通了。"], texts
    assert len(calls) == 1, calls
    # 关键断言：参数是从 updates 里拿的**已解析** dict，不是自己拼的分片。
    # 自己拼的话这里会是 {"path": "hel"} 或者工具名变成 "list_dirlist_dir"。
    assert calls[0].name == "list_dir", calls[0]
    assert calls[0].args == {"path": "hello_world"}, calls[0]
    assert calls[0].invalid_args is None, calls[0]
    assert len(results) == 1, results
    assert results[0].id == "call_1", results[0]
    assert results[0].name == "list_dir", results[0]
    assert results[0].ok is True, results[0]
    assert results[0].output == "[文件] app.py", results[0]
    # 顺序：工具调用在工具结果之前（实测的调度顺序；UI 不依赖它，但断言一下
    # 能让「顺序变了」这件事在测试里可见）
    assert events.index(calls[0]) < events.index(results[0])
    assert isinstance(events[-1], TurnDone), events[-1]
    assert events[-1].turns == 1, events[-1]


def test_retry_note_is_not_shown_as_assistant_text() -> None:
    """重试注入的那条 HumanMessage 不能变成模型说的话。

    这是实测发现的坑：`safety/guard.py` 注入的 RETRY_NOTE 会从 `messages`
    模式流出来，节点名同样叫 agent。不过滤的话，用户会在终端上看到
    「上一步没有成功，请换一种说法……」——像是模型在自言自语。
    正确做法是翻译成一条 Notice。

    两个翻译函数都是 staticmethod，可以直接调：翻译逻辑是纯函数，
    不用为了测它去搭一个真的会触发风控的模型。
    """
    from engine.agent import JCAEngine as _E

    # messages 模式：重试消息和别的内部消息一律不显示
    assert _E._translate_message((HumanMessage(content=RETRY_NOTE), {})) == []
    assert _E._translate_message((HumanMessage(content="用户自己的话"), {})) == []

    # updates 模式：重试消息翻译成状态提示，而不是当成助手内容
    events = _E._translate_update({"agent": {"messages": [HumanMessage(content=RETRY_NOTE)]}})
    assert len(events) == 1, events
    assert isinstance(events[0], Notice), events[0]
    assert events[0].kind == "safety_retry", events[0]

    # 反过来：用户自己的消息不会被误判成重试
    assert _E._translate_update({"agent": {"messages": [HumanMessage(content="用户自己的话")]}}) == []


def test_translate_update_extracts_tool_calls() -> None:
    """`updates` 里的完整 AIMessage → ToolCall 事件（含非法参数那一支）。"""
    from engine.agent import JCAEngine as _E

    msg = AIMessage(
        content="",
        tool_calls=[{"name": "run_cmd", "args": {"command": "python app.py"}, "id": "c1"}],
        invalid_tool_calls=[{"name": "read_file", "args": "{不是 JSON", "id": "c2", "error": None}],
    )
    events = _E._translate_update({"agent": {"messages": [msg]}})

    assert len(events) == 2, events
    assert events[0] == ToolCall(id="c1", name="run_cmd", args={"command": "python app.py"})
    # 非法参数如实报出来：用户看到的应该是「模型写错了参数」，
    # 而不是一句没头没尾的「工具失败了」。
    assert events[1].name == "read_file" and events[1].invalid_args, events[1]
    assert events[1].args == {}, events[1]


def test_multi_turn_keeps_history() -> None:
    """多轮上下文靠 checkpointer 维持 —— UI 层不持有历史，这是它能保持干净的前提。"""
    _SCRIPT.clear()
    # 第一轮：一轮纯文本
    _SCRIPT.append(_agent_round("第一轮的回答。"))
    # 第二轮：也是纯文本
    _SCRIPT.append(_agent_round("第二轮的回答。"))

    engine = _build_engine()

    async def run() -> tuple[list, list]:
        first = await _collect(engine, "第一个问题")
        second = await _collect(engine, "第二个问题")
        return first, second

    first, second = asyncio.run(run())
    assert any(isinstance(e, TurnDone) for e in first)
    assert any(isinstance(e, TurnDone) for e in second)
    assert engine.turns == 2, engine.turns

    # 历史真的在 checkpointer 里：第二轮进去时，state 里应该已经有两组问答。
    # 这是「UI 不需要记住任何东西」的实证。
    state = engine._graph.get_state({"configurable": {"thread_id": "jca-cli"}})
    kinds = [type(m).__name__ for m in state.values["messages"]]
    assert kinds.count("HumanMessage") == 2, kinds
    assert kinds.count("AIMessage") == 2, kinds
    # 系统提示只有一条。每轮都往 inputs 里塞一份，靠的是固定 id ——
    # `add_messages` 同 id 是替换。id 一丢，三轮下来就是三份重复的系统提示，
    # 每一份都会被原样发出去，白烧 token 还稀释指令。
    assert kinds.count("SystemMessage") == 1, kinds


def test_llm_failure_becomes_agent_error() -> None:
    """接口报错要变成 AgentError，不能把栈打到 UI 上。"""
    _SCRIPT.clear()

    def boom(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        raise ConnectionError("模拟：连不上")

    original = FakeChatModel._stream
    FakeChatModel._stream = boom
    try:
        events = asyncio.run(_collect(_build_engine(), "随便说点什么"))
    finally:
        FakeChatModel._stream = original

    assert len(events) == 1, events
    assert isinstance(events[0], AgentError), events[0]
    assert events[0].kind == "api", events[0]
    assert "连不上" in events[0].message, events[0]


def test_safety_block_retries_then_gives_up() -> None:
    """被内容判定拦下：重试额度用完，收尾是 AgentError 而不是栈。

    这条链路跨了三个模块（safety/guard.py 造重试消息、agent_node 用
    Command(goto=...) 回到自己、engine 把异常翻成事件），而它在真环境里
    很难复现 —— 要先让服务端判定命中。这里用一个「永远被拦」的假模型把它
    钉住，顺带保证那两条 Notice 的措辞不会悄悄变成助手正文。
    """
    from langchain_core.exceptions import ModelInvalidRequestError

    from safety import MAX_SAFETY_RETRIES

    class _BlockedError(ModelInvalidRequestError):
        # code 字段就是 safety/guard.py 认的那个判定码
        code = "sensitive_words_detected"

    class BlockedModel(FakeChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            raise _BlockedError("sensitive_words_detected")

    _SCRIPT.clear()
    engine = JCAEngine(llm=BlockedModel(), tools=[list_dir], system_prompt="（测试用系统提示）")
    events = asyncio.run(_collect(engine, "随便说点什么"))

    notices = [e for e in events if isinstance(e, Notice)]
    assert len(notices) == MAX_SAFETY_RETRIES, notices
    assert all(n.kind == "safety_retry" for n in notices), notices

    # 收尾是 AgentError：重试额度用完之后这一轮真的没救了，
    # 得让用户知道「换个说法」，而不是看到一个 traceback。
    assert isinstance(events[-1], AgentError), events[-1]
    assert events[-1].kind == "safety", events[-1]
    # 失败的一轮不算数
    assert engine.turns == 0, engine.turns
    assert not any(isinstance(e, TurnDone) for e in events), events


def main() -> int:
    """不装 pytest 也能跑：把上面这些函数挨个调一遍。

    刻意不引 pytest 当测试框架：这个项目的依赖表很短，为了跑一个离线冒烟
    再拉一个框架进来不划算。函数式命名又保留了「以后想用 pytest 也能直接用」。
    """
    tests = [
        test_text_tool_result_text,
        test_retry_note_is_not_shown_as_assistant_text,
        test_translate_update_extracts_tool_calls,
        test_multi_turn_keeps_history,
        test_llm_failure_becomes_agent_error,
        test_safety_block_retries_then_gives_up,
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

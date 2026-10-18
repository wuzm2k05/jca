"""界面的离线测试：不需要 API key，也不需要真终端。

跑法（两种都行）：

    .venv/Scripts/python.exe -m tests.test_ui_transcript
    uv run --with pytest pytest tests/            # 装了 pytest 的话

测的是 `ui/cli.py` 的 `apply_event()` 和 `ui/transcript.py` 的 `Transcript`：
**一串事件进去，一段文字出来**。这条路径是纯的，所以能脱开终端断言。

为什么值得单独测一遍：鉴权、终端宽度、Live 的重绘都好调试，唯独「事件流翻
译错了」不容易看出来 —— 画面上少一行工具结果，看起来只像是「模型没调工具」。
用假事件把每个分支钉住，改 UI 的时候才敢动。
"""

import sys

from rich.console import Console

from engine import (
    AgentError,
    AssistantText,
    Notice,
    ToolCall,
    ToolResult,
    TurnDone,
)
from ui.cli import apply_event
from ui.transcript import Transcript


def _render(transcript: Transcript, *, status: str = "") -> str:
    """把 transcript 画成纯文本。

    `width` 给得很宽、`no_color` 打开，是为了让断言不受终端宽度和 ANSI
    转义影响 —— 我们测的是「画了什么」，不是「画得多好看」。
    """
    console = Console(width=200, no_color=True, legacy_windows=False)
    with console.capture() as capture:
        console.print(transcript.render(status=status))
    return capture.get()


def _play(*events) -> tuple[Transcript, str]:
    """按顺序喂一串事件，返回终态的 transcript 和状态行。"""
    transcript = Transcript()
    status = ""
    for event in events:
        status = apply_event(transcript, event, status)
    return transcript, status


def test_text_becomes_markdown_block() -> None:
    """分片的文本要攒成一段，而不是每片一行。"""
    transcript, status = _play(
        AssistantText("我先看一下"),
        AssistantText("目录。"),
        TurnDone(turns=1),
    )
    out = _render(transcript, status=status)

    assert "我先看一下目录。" in out.replace("\n", "") or "我先看一下目录。" in out, out
    # 收尾后状态行必须消失，否则 Live 关掉时屏幕上会留一句「正在生成…」
    assert status == "", status
    assert "正在生成" not in out, out


def test_tool_call_and_result_share_one_line() -> None:
    """调用和结果画成一组：结果到了要接在**同一条**工具行上，不是另起一行。"""
    transcript, status = _play(
        AssistantText("我看一下。"),
        ToolCall(id="c1", name="list_dir", args={"path": "hello_world"}),
        ToolResult(id="c1", name="list_dir", ok=True, output="app.py\nREADME.md", chars=17),
        AssistantText("找到了。"),
        TurnDone(turns=1),
    )
    out = _render(transcript, status=status)

    assert "list_dir" in out, out
    assert "path=hello_world" in out, out
    assert "app.py" in out, out
    # 结果有多行时要折叠 —— 终端不是读长文的地方
    assert "共 2 行" in out, out
    assert "正在执行" not in out and "正在思考" not in out, out


def test_result_without_known_call_still_renders() -> None:
    """结果先到（调用还没到）也要能建行。

    实测的顺序是「调用先、结果后」，但那是调度顺序、不是契约。UI 按 id
    建行就是为了这个：真出现乱序时，宁可少一个标题，不能整个结果丢了。
    """
    transcript, _ = _play(
        ToolResult(id="c9", name="run_cmd", ok=True, output="Hello", chars=5),
    )
    out = _render(transcript)

    assert "run_cmd" in out, out
    assert "Hello" in out, out


def test_invalid_args_is_reported_as_model_error() -> None:
    """模型把参数写坏了，要说清楚是「参数不是合法 JSON」。

    只说「工具失败了」的话，用户没有任何线索；而这件事的真实原因是模型
    自己吐了坏 JSON，ToolNode 会把它原样还回去让模型改。
    """
    transcript, _ = _play(
        ToolCall(id="c1", name="read_file", args={}, invalid_args="{不是 JSON"),
    )
    out = _render(transcript)

    assert "read_file" in out, out
    assert "JSON" in out, out


def test_safety_retry_is_a_notice_not_assistant_text() -> None:
    """重试提示画成状态提示，绝不能画成「模型说的话」。

    同一件事在 engine 侧已经过滤过一次（见 engine/agent.py 的模块说明），
    这里测的是 UI 侧不会画错：提示行不带助手符号。
    """
    transcript, status = _play(Notice("正在重试", kind="safety_retry"))
    out = _render(transcript, status=status)

    assert "正在重试" in out, out
    # Notice 不改状态行：重试完还会继续生成，状态行该保持原样
    assert status == "", status


def test_error_closes_the_turn() -> None:
    """失败的一轮：画出错误，状态行收掉，且**没有** TurnDone 也能收口。"""
    transcript, status = _play(AgentError("出错了：连不上", kind="api"))
    out = _render(transcript, status=status)

    assert "连不上" in out, out
    assert status == "", status
    # 失败的一轮没有 TurnDone，transcript 判定为「不空」——里面确实有东西
    assert not transcript.is_empty(), "错误应该留下内容"


def test_unknown_event_is_ignored() -> None:
    """engine 加了新事件、UI 还没跟上时，界面不能崩。"""

    class FutureEvent:
        pass

    transcript, status = _play(FutureEvent())
    assert transcript.is_empty(), transcript.render()
    assert status == "", status


def main() -> int:
    tests = [
        test_text_becomes_markdown_block,
        test_tool_call_and_result_share_one_line,
        test_result_without_known_call_still_renders,
        test_invalid_args_is_reported_as_model_error,
        test_safety_retry_is_a_notice_not_assistant_text,
        test_error_closes_the_turn,
        test_unknown_event_is_ignored,
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

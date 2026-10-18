"""终端呈现层：把领域事件画成人看得懂的样子。

这一层认识 rich，但**不认识 LangGraph、不认识模型、不知道工具是怎么执行的**。
它拿到的全是 engine/events.py 里的 dataclass。反过来说，engine 那边也完全
不知道这里的存在——两边只通过事件契约见面。

拆成两个文件：
  - transcript.py（本文件）：怎么画。纯函数 + 一个攒消息的缓冲区，没有 IO 循环。
  - cli.py：什么时候画、读输入、收尾。它持有 Live 和事件循环。

分开的理由很实际：想调样式（换个符号、改个颜色）只翻这个文件；
想改交互行为（加个 /help、加个新命令）只翻 cli.py。
"""

import os
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# —— 符号表 ————————————————————————————————————————————————————————
#
# 这几个符号是 Claude Code 那套观感的来源。放在这里集中管理，一是便于整体
# 换成 ASCII（老终端、Windows 的 GBK 代码页显示不出来会变成「?」），
# 二是以后想换一套也不用满文件找。
#
# JCA_ASCII=1 就把整组降级成纯 ASCII：不为了好看让用户去折腾终端编码。
_ASCII_MODE = os.getenv("JCA_ASCII", "").strip().lower() in ("1", "true", "yes", "on")

SYM_ASSISTANT = ">" if _ASCII_MODE else "⏺"
SYM_TOOL = ">" if _ASCII_MODE else "●"
SYM_RESULT = "-" if _ASCII_MODE else "⎿"
SYM_PROMPT = "jca>" if _ASCII_MODE else "›"
SYM_WARN = "!" if _ASCII_MODE else "⚠"

# 工具结果的缩进：和 SYM_TOOL 对齐，让「调用 — 结果」看起来是一组
_INDENT = "  "

# 单行参数摘要里，一个值最长显示多少字符
ARG_VALUE_LIMIT = 60


@dataclass
class ToolLine:
    """一条工具调用在终端上的样子（调用 + 结果两行）。

    单独建一个类而不是往 Markdown 里拼字符串，是为了让「调用」和「结果」
    能在**不同时刻**到达时都能正确显示：实测事件顺序是「调用先、结果后」，
    但那是调度顺序、不是契约，所以两边都按 id 往这里塞，谁先到谁先建。
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    invalid_args: str | None = None
    result: str | None = None
    result_chars: int = 0

    def render(self) -> Text:
        """画成一个 rich Text（两行或三行）。"""
        text = Text()

        # 第一行：调用摘要
        text.append(f"{_INDENT}{SYM_TOOL} ", style="bold cyan")
        text.append(self.name, style="bold")
        if self.invalid_args is not None:
            # 参数不是合法 JSON：这是模型自己写错了，ToolNode 会把错误还给它，
            # 它有机会重来。这里如实标出来 —— 否则用户只看到「工具失败了」，
            # 不知道为什么失败。
            text.append(f"({SYM_WARN} 参数不是合法 JSON)", style="yellow")
            text.append(f"  {_snippet(self.invalid_args)}", style="dim")
        else:
            text.append(f"({_summarize_args(self.args)})", style="dim")

        # 第二行起：结果。只显示前几行，剩下的折起来 —— 终端不是读长文的地方，
        # 需要看全文的人应该去读文件或日志。
        if self.result is not None:
            text.append("\n")
            lines = self.result.splitlines() or [""]
            text.append(f"{_INDENT * 2}{SYM_RESULT} ", style="dim")
            text.append(lines[0] or "（无输出）", style="dim")
            if len(lines) > 1:
                text.append(f"  …（共 {len(lines)} 行）", style="dim italic")
            if self.result_chars > len(self.result):
                text.append(f"（已截断，原文 {self.result_chars} 字符）", style="dim italic")
        return text


class Transcript:
    """一次会话的可见内容。

    它**只负责攒**，不负责刷屏：`render()` 交出一份可以丢给 rich.Live 的东西，
    什么时候刷由 cli.py 决定（那是 IO 的事）。这样这个类本身是纯的、能单独测。

    攒法很简单：助手说的话按顺序连成一段 Markdown，工具行穿插在其中。
    遇到工具调用就把助手那段收口 —— 因为模型「说一句、调个工具、再说一句」
    是有意的分段，糊成一整段会读不出来。
    """

    def __init__(self) -> None:
        self._parts: list[RenderableType] = []
        # 当前正在追加的助手文本。None 表示「上一段已经收口了」
        self._pending_text: str | None = None
        # 已经画过的工具行，按 id 索引 —— 结果比调用先到也能接上
        self._tools: dict[str, ToolLine] = {}

    # —— 事件入口 ————————————————————————————————————————————
    def add_text(self, delta: str) -> None:
        """追加一段模型回复。"""
        if self._pending_text is None:
            self._pending_text = delta
        else:
            self._pending_text += delta

    def add_tool_call(self, id: str, name: str, args: dict[str, Any], invalid_args: str | None) -> None:
        """记下一条工具调用。"""
        self._close_text()
        line = self._tool(id, name)
        line.name = name or line.name
        line.args = args
        line.invalid_args = invalid_args

    def add_tool_result(self, id: str, name: str, output: str, chars: int) -> None:
        """记下一条工具结果。"""
        self._close_text()
        line = self._tool(id, name)
        line.name = line.name or name
        line.result = output
        line.result_chars = chars

    def add_notice(self, text: str, kind: str) -> None:
        """一句提示。"""
        self._close_text()
        style = "yellow" if kind == "safety_retry" else "dim"
        self._parts.append(Text(f"{_INDENT}{SYM_WARN} {text}", style=style))

    def add_error(self, message: str, kind: str) -> None:
        """一条错误。"""
        self._close_text()
        style = "bold red" if kind != "safety" else "bold yellow"
        self._parts.append(Text(f"{SYM_ASSISTANT} {message}", style=style))

    # —— 输出 ————————————————————————————————————————————————
    def render(self, *, status: str = "") -> RenderableType:
        """当前全部内容 + 一行状态。给 rich.Live 用。

        每次都整体重画：几十个 renderable 的开销可以忽略，换来的是不必维护
        「增量更新终端」那套复杂的游标账 —— 而那种账正是花屏的来源。
        """
        items: list[RenderableType] = [
            self._tools[part.id].render() if isinstance(part, _ToolRef) else part
            for part in self._parts
        ]
        if self._pending_text is not None:
            items.append(self._render_assistant(self._pending_text))
        if status:
            items.append(Text(f"{_INDENT}{status}", style="dim italic"))
        return Group(*items) if items else Text("")

    def is_empty(self) -> bool:
        return not self._parts and self._pending_text is None

    # —— 内部 ————————————————————————————————————————————————
    def _render_assistant(self, text: str) -> RenderableType:
        """助手的一段话：左边一个符号，右边一段 Markdown。

        用两列的 grid 而不是 `Group(符号, Markdown)`：后者会把符号单独占一行，
        看起来像「⏺」下面空一行再开始说话。grid 让符号和正文第一行齐平，
        正文换行时也自动挂在符号右边 —— 这正是 Claude Code 那个观感。
        两列的宽度分配交给 rich，不用自己算中文宽度。

        `overflow="fold"` 不能省：表格列的默认行为是**省略号截断**，而这里要的是
        换行。少了它，正文超过一行就会变成「…把小问题，…」这种中间被吞掉的样子
        （Markdown 的测量宽度和列的可用宽度对不上时尤其明显）。

        流式过程中每一帧都会重新解析一次这段 Markdown。这是有意的：解析几百字
        的开销远小于一次终端重绘，而换来的是不必自己维护「未闭合的代码块」
        那种状态 —— 增量拼字符串的做法在 ``` 还没闭合时会让整块内容反复跳，
        看起来像花屏。
        """
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1, no_wrap=True)
        grid.add_column(overflow="fold")
        grid.add_row(Text(SYM_ASSISTANT, style="green"), Markdown(text))
        return grid

    def _close_text(self) -> None:
        """把当前这段助手文本收口成一个静态块。"""
        if self._pending_text is None:
            return
        if self._pending_text.strip():
            self._parts.append(self._render_assistant(self._pending_text))
        self._pending_text = None

    def _tool(self, id: str, name: str) -> ToolLine:
        """按 id 取工具行；没有就建一条（含它在 `_parts` 里的位置）。

        为什么按 id 而不是「就按顺序追加」：事件顺序实测是「调用先、结果后」，
        但那是调度顺序、不是契约。按 id 建行的话，结果比调用先到也能接上。
        """
        line = self._tools.get(id)
        if line is None:
            line = ToolLine(id=id, name=name)
            self._tools[id] = line
            # 这里只占个位，真正的内容等 render() 时现取 —— rich 的 renderable
            # 是不可变的，而工具行的内容会在结果到达时再变一次。
            self._parts.append(_ToolRef(id))
        return line


@dataclass
class _ToolRef:
    """`_parts` 里的占位符：这里是一条工具行。

    需要一个占位，是因为 tool 行的内容会在结果到达时再变一次，而 rich 的
    renderable 是不可变的。占位符只记 id，画的时候现取最新的 ToolLine。
    """

    id: str


def _summarize_args(args: dict[str, Any]) -> str:
    """把工具参数压成一行，给终端看。

    按 key 排序：同一次调用的展示在两次运行之间保持一致，便于对照。
    """
    if not args:
        return ""
    parts = []
    for key in sorted(args):
        parts.append(f"{key}={_snippet(str(args[key]))}")
    return ", ".join(parts)


def _snippet(text: str) -> str:
    """把值压成一行短串：换行变空格，超长截断。"""
    flat = " ".join(text.split())
    if len(flat) > ARG_VALUE_LIMIT:
        return flat[: ARG_VALUE_LIMIT - 1] + "…"
    return flat


def banner(console: Console, *, model: str, workspace: str, log_note: str, warning: str | None = None) -> None:
    """开局那块牌子。

    刻意把「日志去哪儿了」写进来：交互模式下日志默认进文件，如果不写，
    用户会以为日志系统坏了 —— 「日志不见了」比「输出花了」更难排查。
    """
    lines = [
        Text.assemble(("模型  ", "dim"), (model, "bold")),
        Text.assemble(("工作目录  ", "dim"), (workspace, "")),
        Text.assemble(("日志  ", "dim"), (log_note, "")),
    ]
    console.print(
        Panel(
            Group(*lines),
            title="[bold]jca[/bold]",
            subtitle="[dim]Ctrl-C 中断本轮（在提示符处则退出）· /exit 退出[/dim]",
            border_style="cyan",
        )
    )
    if warning:
        console.print(Text(f"{SYM_WARN} {warning}", style="yellow"))


__all__ = ["SYM_PROMPT", "ToolLine", "Transcript", "banner"]

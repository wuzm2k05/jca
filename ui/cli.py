"""交互层：读输入、驱动 Live、收尾。

这里只做三件事，都跟「什么时候」有关：
  1. 什么时候读一行输入（`Prompt.ask`）；
  2. 什么时候刷新屏幕（收到事件就刷）；
  3. 什么时候收口（TurnDone / AgentError / Ctrl-C）。

至于「画成什么样」全在 transcript.py 里。

## 日志为什么要挪走

`logger/setup.py` 的终端 handler 写的是裸 `sys.stdout.write`，而 rich 的 Live
要在 `sys.stdout` 上做光标定位。rich 只在 `console.print()` 这条路径上会替你
让位，绕不过别处来的写入 —— 两边一起写就是花屏。

所以在**交互模式**下，日志的默认出口被换成文件（由 main.py 在 setup_logging
之前决定，见 logger/config.py 的 from_env(default_target=...)）。这里只负责
把这件事告诉用户：横幅上写明日志文件在哪，用户显式设了 console 就再提醒一句。
配置的事归 main.py，呈现的事归这里。

## 为什么只有一轮是异步的

主循环是**同步**的，每一轮内部才 `asyncio.run`。这样做是为了让 Ctrl-C 的行为
可以推理：没有自己装的信号处理器，也就没有「Windows 上能不能装」的问题。
asyncio.run 收到 KeyboardInterrupt 会取消这一轮的任务、关掉生成器，然后把
KeyboardInterrupt 从这里抛出来，我们接住、说一句「已中断」、回到提示符。
"""

import asyncio
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt

from engine import (
    AgentError,
    AgentEvent,
    AssistantText,
    JCAEngine,
    Notice,
    ToolCall,
    ToolResult,
    TurnDone,
)
from logger import get_logger

from .transcript import SYM_PROMPT, SYM_RESULT, SYM_TOOL, SYM_WARN, Transcript, banner

logger = get_logger(__name__)

# 退出命令。带上斜杠是为了不跟正常提问撞车（用户很少拿 "/exit" 当问题）。
EXIT_COMMANDS = {"/exit", "/quit", "/q"}

# 两次重画之间至少隔多久（秒）。分片是成串来的，中间帧没人看得见，
# 而每帧都要把整块内容重写一遍终端。实测一次真实回答：70 个事件、4.2 秒，
# 不节流的话平均 17 帧/秒、共写 48KB；节流到 60ms 之后帧数减半而观感不变。
# 这个值只影响"画得有多勤"，不影响内容完整性（收尾那次总会画）。
MIN_PAINT_INTERVAL = 0.06


class _JCAPrompt(Prompt):
    """把提示符后面的冒号去掉。

    基类的 `prompt_suffix` 是 ": "，画出来是 `›: `，读起来像是「在输入一个
    字段」。这类对话式界面里的提示符是**行首标记**，不是「字段名：」，
    一个空格就够。
    """

    prompt_suffix = " "


def make_console(*, no_color: bool = False) -> Console:
    """建一个 Console，并把 Windows 的编码问题挡在前面。

    Windows 上默认代码页常常不是 UTF-8，`⎿` 这类符号会变成「?」甚至直接抛
    UnicodeEncodeError。这里把标准输出重开成 UTF-8：改的是本进程对 stdout 的
    编码，不动系统代码页，也不影响重定向出去的文件。

    **输入**只在不是终端时才改成 UTF-8，两种情形要分开看：

    - 键盘输入（stdin 是终端）：编码得跟着控制台代码页走，中文区就是 GBK。
      强行改成 UTF-8，用户敲进去的中文反而会乱。
    - 管道输入（stdin 不是终端）：字节是**别的程序**写进来的，而现在几乎所有
      东西都写 UTF-8，包括本程序自己（上面刚把 stdout 定成 UTF-8）。
      不跟着改的话，`jca < 文件` 会把 UTF-8 字节按 GBK 解，解出一堆
      `\\udcaa` 这样的代理字符，一路带到 HTTP 请求体里，最后报一个
      「surrogates not allowed」—— 完全看不出是编码的问题。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            # 被别的东西包着（测试里换成 StringIO 之类），或者标准输出不可重配。
            # 不是致命问题：大不了符号显示不出来，程序该跑还得跑。
            pass
    try:
        if not sys.stdin.isatty():
            sys.stdin.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass
    return Console(no_color=no_color)


def run_interactive(
    engine: JCAEngine,
    *,
    console: Console | None = None,
    log_note: str = "",
    warning: str | None = None,
) -> int:
    """交互式会话主循环，返回退出码。"""
    console = console or make_console()
    banner(
        console,
        model=engine.model_name,
        workspace=_display_workspace(),
        log_note=log_note,
        warning=warning,
    )

    while True:
        try:
            raw = _JCAPrompt.ask(f"[bold blue]{SYM_PROMPT}[/bold blue]", console=console)
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / 管道输入结束（Prompt.ask 在输入流结束时抛 EOFError）、
            # 或者在提示符处按了 Ctrl-C。都算正常收工，退出码 0。
            console.print()
            console.print("[dim]再见。[/dim]")
            return 0

        prompt = raw.strip()
        if not prompt:
            continue
        if prompt in EXIT_COMMANDS:
            console.print("[dim]再见。[/dim]")
            return 0

        # 空一行再开始画：用户刚敲的那行还留在屏幕上，紧接着就是回答，
        # 不留白的话「问」和「答」会贴在一起。
        console.print()

        # 每轮一个全新的 transcript：上一轮的内容在 Live 收口时已经落到屏幕上
        # 了，是终端的滚动历史。留着它的话，下一轮的 Live 会把上一轮**再画
        # 一遍**——同一个 Live 区域里内容越叠越长。
        transcript = Transcript()
        try:
            asyncio.run(_run_turn(engine, prompt, transcript=transcript, console=console))
        except KeyboardInterrupt:
            # 这一轮被打断。已经画出来的内容留着（它是真实发生过的）——
            # Live 收口时（transient 默认 False）会把它留在屏幕上，所以这里
            # **不能**再 print 一遍 transcript，那会把整轮内容打两遍。
            console.print()
            console.print(f"[yellow]{SYM_WARN} 已中断本轮（上面已有的内容保留）。[/yellow]")
        except Exception:
            # 兜底：engine 里已经把想得到的失败都翻译成事件了，走到这儿
            # 说明是 UI 自己出的问题（画不出来、终端崩了……）。栈进日志，
            # 终端只说一句，别让整个会话跟着挂掉。
            logger.exception("界面出错，本轮中止")
            console.print("[red]界面出错，本轮已中止（详见日志）。[/red]")


async def _run_turn(
    engine: JCAEngine,
    prompt: str,
    *,
    transcript: Transcript,
    console: Console,
) -> None:
    """跑一轮，边收事件边刷新。"""
    status = "正在思考…"

    # auto_refresh=False 是这里的关键，别改回去。默认值会让 rich 起一个刷新线程，
    # 每秒 12 次把整屏重画一遍 —— 不看内容变没变。实测：空闲 3 秒（模型正在思考、
    # 一个事件都没来）**向终端写了 28KB**，而同样 3 秒我们只该写一次；屏幕看起来
    # 就是一直在闪。之所以能这么关掉，是因为状态行是一句静止的文字，
    # 没有任何随时间变化的东西 —— 真要放个计时器或者 spinner 进来，那才需要
    # 定时刷新，而且得让**只有那一行**变，不能整屏重画。
    #
    # 于是刷新时机完全由我们决定：收到一个事件就画一次（见下面的 refresh=True）
    # 是唯一会动屏幕的地方。流式文本本身就是密集到达的，观感一样是"逐字长出来"。
    #
    # transient 用默认的 False：收口后内容留在屏幕上，不擦掉 —— 这是对话，
    # 不是进度条。
    with Live(console=console, auto_refresh=False) as live:
        # 起手先刷一次，免得模型还在想的时候屏幕上什么都没有
        live.update(transcript.render(status=status), refresh=True)

        last_paint = time.monotonic()
        async for event in engine.stream_turn(prompt):
            status = apply_event(transcript, event, status)
            # 分片是**成串**到达的（实测间隔中位数 1 ms，平均 17 片/秒）。每一片都
            # 重画的话，同一毫秒里会连画好几次，而终端一秒只刷得出几十帧 —— 那些
            # 中间帧谁也看不见，只是白写、还让画面抖。所以给重画定个上限：早于这个
            # 间隔就跳过，反正下一片马上就到，届时的渲染里自然包含它。
            now = time.monotonic()
            if now - last_paint >= MIN_PAINT_INTERVAL:
                live.update(transcript.render(status=status), refresh=True)
                last_paint = now

        # 收尾那一次不能省：最后几片很可能正落在被跳过的那段时间里，
        # 而 TurnDone 又要把状态行抹掉，这一帧才是屏幕上的定版。
        live.update(transcript.render(status=status), refresh=True)

    if not console.is_terminal:
        # 非终端（管道 / 重定向）下 Live 不做光标接管，收尾时也就不会自己补换行，
        # 于是这一轮的最后一个字符会跟下一行输出粘在一起。真实终端上
        # Live.stop() 已经补过了，这里再补一次就会多一个空行。
        console.print()


def apply_event(transcript: Transcript, event: AgentEvent, status: str) -> str:
    """把一个事件画进 transcript，返回新的状态行文字。

    先判收尾事件、再判内容事件：收尾事件在任何时候到达都意味着这一轮不必再
    显示状态行了，跟具体是哪种内容无关。

    单独一个函数（而不是塞在循环里）是为了让 UI 能脱离终端被测试：
    喂一串事件、检查 transcript.render() 的结果，不需要真终端。
    """
    if isinstance(event, TurnDone):
        return ""
    if isinstance(event, AgentError):
        transcript.add_error(event.message, event.kind)
        return ""
    if isinstance(event, AssistantText):
        transcript.add_text(event.delta)
        return "正在生成…"
    if isinstance(event, ToolCall):
        transcript.add_tool_call(event.id, event.name, event.args, event.invalid_args)
        return f"正在执行 {event.name}…"
    if isinstance(event, ToolResult):
        transcript.add_tool_result(event.id, event.name, event.output, event.chars)
        return "正在思考…"
    if isinstance(event, Notice):
        transcript.add_notice(event.text, event.kind)
        return status
    # 不认识的事件类型：忽略，别让界面崩掉。留一条 debug 日志给过渡期用
    # （engine 加了新事件、UI 还没跟上）。
    logger.debug("界面收到未知事件，已忽略", type=type(event).__name__)
    return status


def run_once(engine: JCAEngine, prompt: str, *, console: Console | None = None) -> int:
    """跑一轮就退出，给脚本和演示用（`uv run main.py --prompt "..."`）。

    engine 用的是同一个 `stream_turn()`，只是呈现不同：不重绘、不渲染 Markdown、
    直接往标准输出写。这样重定向到文件是干净的，而「跑完一轮」这个用法多半
    就是为了被脚本或重定向读。

    返回 1 表示这一轮失败了（脚本可以靠退出码判断），0 表示正常。
    """
    console = console or make_console()
    return asyncio.run(_stream_once(engine, prompt, console))


async def _stream_once(engine: JCAEngine, prompt: str, console: Console) -> int:
    failed = False
    # 助手文本是分片来的，攒成块再输出：一个 token 一行地打，输出会碎成几百行。
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = "".join(buffer).strip()
        buffer.clear()
        if text:
            # markup=False / highlight=False：模型说的话里可能有方括号、有看起来
            # 像代码的东西，别让 rich 去解释它们。
            console.print(text, markup=False, highlight=False)

    async for event in engine.stream_turn(prompt):
        if isinstance(event, AssistantText):
            buffer.append(event.delta)
        elif isinstance(event, ToolCall):
            flush()
            args = " ".join(f"{k}={v!r}" for k, v in sorted(event.args.items()))
            console.print(f"{SYM_TOOL} {event.name}({args})", style="cyan", markup=False)
        elif isinstance(event, ToolResult):
            flush()
            first = (event.output.splitlines() or [""])[0]
            console.print(f"  {SYM_RESULT} {first}", style="dim", markup=False)
        elif isinstance(event, Notice):
            flush()
            console.print(f"{SYM_WARN} {event.text}", style="yellow", markup=False)
        elif isinstance(event, AgentError):
            flush()
            console.print(f"{SYM_WARN} {event.message}", style="red", markup=False)
            failed = True
    flush()
    return 1 if failed else 0


def _display_workspace() -> str:
    """给横幅用的工作目录。出错也不能让界面起不来。"""
    try:
        from tools.sandbox import workspace_root

        return str(workspace_root())
    except Exception:
        return "(未知)"


__all__ = [
    "EXIT_COMMANDS",
    "apply_event",
    "make_console",
    "run_interactive",
    "run_once",
]

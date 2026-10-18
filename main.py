"""组装点（Composition Root）：把 engine 和 ui 接在一起，然后启动。

这个文件刻意不做任何 Agent 的事 —— 建工具、建模型、建图都在 engine/ 里，
画界面都在 ui/ 里。这里只剩下「按什么顺序装、按什么参数装」：

    load_dotenv()  →  选日志出口  →  setup_logging()  →  argparse
                                      ↓
                           JCAEngine ──► ui.run_interactive / ui.run_once

不用 `if __name__ == "__main__"` 里再写一段演示 prompt：想跑一轮就用
`--prompt`，那才是给脚本和演示留的入口。
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from engine import DEFAULT_MODEL, JCAEngine
from logger import LogConfig, LogTarget, get_logger, setup_logging
from logger.config import ENV_PREFIX
from ui import make_console, run_interactive, run_once

# 在程序最开始加载 .env 文件的环境变量
load_dotenv()

logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jca",
        description="一个本地编程 agent。不带参数就是交互式会话。",
    )
    # 位置参数是「顺手跑一句」，--prompt 是「明说要跑一句」。两个都留，
    # 因为引号在 Windows 的 cmd 里很烦，多一条路少一次踩坑。
    parser.add_argument(
        "prompt_positional",
        nargs="?",
        metavar="PROMPT",
        help="跑完这一句就退出（等价于 --prompt）。",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="跑完这一句就退出，不进交互式界面。给脚本用。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名（默认 {DEFAULT_MODEL}）。",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="关掉颜色（重定向到文件、或者终端配色不兼容时用）。",
    )
    return parser.parse_args(argv)


def _setup_logging(*, interactive: bool) -> LogConfig:
    """选日志出口，然后装配。

    为什么要在装配**之前**做这个判断：交互式界面（rich 的 Live）和终端日志
    都要在 sys.stdout 上做光标定位，两边同时写会花屏。所以在交互模式下把
    日志默认挪进文件。

    三档，按优先级：
      1. 用户显式设过 JCA_LOG_TARGET —— 尊重他的选择，只提醒一句（见返回值
         之外的 warning，由 main() 传给界面）；
      2. 没设过 + 交互模式 —— 默认 file；
      3. 没设过 + --prompt 模式 —— 保持原来的默认（console），一次性运行的
         输出跟以前一模一样。
    """
    explicit = bool(os.getenv(ENV_PREFIX + "LOG_TARGET", "").strip())
    default_target = LogTarget.FILE if (interactive and not explicit) else None
    return setup_logging(LogConfig.from_env(default_target=default_target))


def _log_note(config: LogConfig) -> str:
    """横幅上「日志去哪儿了」那句话。"""
    if config.target in (LogTarget.FILE, LogTarget.BOTH):
        return str(config.file_path)
    return "终端（和界面共用终端）"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    prompt = args.prompt or args.prompt_positional
    interactive = prompt is None

    config = _setup_logging(interactive=interactive)

    # 用户把日志留在了终端上：不拦，只提醒一次。花屏很烦，但比起「用户配好的
    # 东西被程序偷偷改掉」，前者是可恢复的。
    warning = None
    if interactive and config.target in (LogTarget.CONSOLE, LogTarget.BOTH):
        warning = (
            f"{ENV_PREFIX}LOG_TARGET={config.target.value} 时日志会和界面抢光标，"
            f"输出可能错位。设成 file 可以避开。"
        )

    logger.info(
        "启动",
        mode="interactive" if interactive else "once",
        model=args.model,
        log_target=config.target.value,
    )

    console = make_console(no_color=args.no_color)
    engine = JCAEngine(model=args.model)

    if interactive:
        return run_interactive(
            engine,
            console=console,
            log_note=_log_note(config),
            warning=warning,
        )
    return run_once(engine, prompt, console=console)


if __name__ == "__main__":
    sys.exit(main())

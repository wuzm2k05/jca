"""终端界面包：把事件画出来，并驱动交互循环。

    from ui import run_interactive, run_once

    run_interactive(engine)              # 交互式，直到用户 /exit 或 Ctrl-D
    run_once(engine, "建个 app.py")       # 跑一轮就退出，给脚本用

这个包认识 rich 和 stdin/stdout，但**不认识 LangGraph**。它唯一的输入是
engine 吐出的领域事件。所以想把这个 CLI 换成 Web 界面，只需要另写一个
「消费同样事件」的包，engine 一侧完全不用动 —— 这正是当初把两层拆开的理由。
"""

from .cli import make_console, run_interactive, run_once

__all__ = ["make_console", "run_interactive", "run_once"]

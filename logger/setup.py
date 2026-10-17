"""把 structlog 接到标准库 logging 上，配出「终端 + 文件」两个出口。

**为什么要接标准库，而不是只用 structlog**

structlog 自己也能写文件，但那样一来，「谁写的日志」就分裂成了两套：
langchain、httpx、openai 这些第三方库走的是标准库 logging，我们的代码走 structlog。
想让它们进同一个文件（同一份轮转、同一种格式），必须有一个共同的落点 ——
这里选标准库 logging，因为第三方库已经在用它了。

接法是 structlog 官方推荐的 `ProcessorFormatter`：

    我们的代码   structlog.get_logger(__name__)
                     │  处理器链加工 event_dict（加时间戳、级别、模块名……）
                     ▼
                 ProcessorFormatter.wrap_for_formatter  ← 把 event_dict 塞进 LogRecord
                     │
    第三方库     logging.getLogger("httpx").info("...")
                     │
                     ▼
                 同一个 Logger ── 两个 Handler ──┬─ 终端：彩色，给人看
                                                 └─ 文件：JSON，每行一条

两条路在**各自的 handler** 上被渲染：同一个事件，终端给人和文件给机器，
用的是两种格式。所以渲染器挂在 handler 上，不挂在全局处理器链上 ——
这是整套配置里唯一容易搞反的地方，搞反了就会「文件里也带颜色」。
"""

import logging
import logging.handlers
import sys
from dataclasses import replace
from typing import Any

import structlog
from structlog.stdlib import ProcessorFormatter

from .config import LogConfig, LogTarget, level_value

# 时间戳统一成 UTC + 微秒 + Z 后缀：日志可能来自不同时区的机器，
# 本地时间放一起没法排序，而 ISO 8601 谁都能读、谁都能排。
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_TIMESTAMP_KEY = "timestamp"

# 一个进程只配一次。重复调用直接返回上次的配置，
# 否则 handler 会越挂越多，一行日志被打成三行。
_CONFIGURED: LogConfig | None = None


def _build_chain() -> list[Any]:
    """structlog 侧的处理器链：只整理字段，最后打包进 LogRecord。

    这一串不负责渲染成字符串 —— 因为同一份字段要在两个地方渲染成两种样子。
    """
    return [
        # 把 structlog 上下文里绑的字段（bind_contextvars）并进来
        structlog.contextvars.merge_contextvars,
        # 记下级别：info / warning / error……
        structlog.stdlib.add_log_level,
        # 记下写下它的模块名，出问题能直接找过去
        structlog.stdlib.add_logger_name,
        # 支持 logger.info("读到 %s 个字节", n) 这种 %-占位写法
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        # 把 exc_info 变成字符串。必须在渲染器之前 ——
        # 再往后 traceback 对象就传不下去了，栈也就丢了。
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt=_TS_FORMAT, utc=True, key=_TIMESTAMP_KEY),
        # 最后一步：整个 event_dict 塞进 LogRecord，
        # 交给各个 handler 上的 ProcessorFormatter 去渲染
        ProcessorFormatter.wrap_for_formatter,
    ]


def _pre_chain() -> list[Any]:
    """第三方库（走标准库 logging 的那些）也要过一遍的字段加工。

    就是上面那串去掉最后打包那一步：它们本来就是 LogRecord，不需要再包一次。
    不给它们加这一步的话，httpx 的日志到了渲染器手上会缺 timestamp、level。
    """
    return _build_chain()[:-1]


def _console_formatter() -> ProcessorFormatter:
    """终端用的格式：带颜色的可读文本。"""
    return ProcessorFormatter(
        foreign_pre_chain=_pre_chain(),
        processors=[
            # 去掉 ProcessorFormatter 加的内部键（_record / _from_structlog），
            # 别让它们出现在输出里
            ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(
                # 只有真的接着终端才上色。判断放在这里，不能交给
                # ConsoleRenderer 的 force_colors：重定向到文件时
                # （比如 `python main.py > out.txt`）我们不希望文件里
                # 混进一堆看不出来的转义字符。
                colors=sys.stdout.isatty(),
            ),
        ],
    )


def _file_formatter() -> ProcessorFormatter:
    """文件用的格式：JSON，每行一条。

    为什么文件不用彩色文本：日志文件多半是被工具读的（grep、jq、采集器），
    结构化字段能直接取出来，不用写正则去猜。
    ensure_ascii=False 让中文按原样写，不然文件里全是一片 \\uXXXX。
    """
    return ProcessorFormatter(
        foreign_pre_chain=_pre_chain(),
        processors=[
            ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )


def _add_console_handler(root: logging.Logger, config: LogConfig) -> None:
    """挂上终端 handler。

    注意顺序：**先**建 formatter，**再**取 sys.stdout。

    在 Windows 上要输出颜色得靠 colorama，而 ConsoleRenderer 构造时会把
    sys.stdout 换成它包装过的流（好把 ANSI 转义翻译成控制台调用）。
    顺序反了的话，handler 抓到的是换之前的旧流，包装形同虚设 ——
    表现就是重定向到文件时，文件里躺着一堆 \\x1b[32m。
    """
    formatter = _console_formatter()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level_value(config.level))
    root.addHandler(handler)


def _add_file_handler(root: logging.Logger, config: LogConfig) -> bool:
    """挂上带轮转的文件 handler；挂不上就返回 False（由调用方降级到终端）。"""
    path = config.file_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # RotatingFileHandler 就是标准库版的 logrotate：写到 max_bytes 就把
        # jca.log 改名成 jca.log.1，依次后移，backup_count 之外的删掉。
        # 轮转发生在本进程内部，不需要外部的 logrotate 定时任务；
        # 代价是多个进程写同一个文件会互相搅乱对方的轮转，
        # 所以日志目录是「一个进程一份」的。
        #
        # delay=True：先别急着建文件。免得这个模块只是被 import 一下，
        # 就在工作目录里留下一个空的 logs/。
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
            delay=True,
        )
    except OSError as e:
        # 目录建不出来（只读盘、路径写错……）。这时**不能**让程序起不来：
        # 日志是辅助设施，它坏了不该拖垮主程序。退回终端，并且明说。
        sys.stderr.write(f"[logger] 日志文件不可用（{e}），改为只输出到终端。\n")
        return False

    handler.setFormatter(_file_formatter())
    handler.setLevel(level_value(config.level))
    root.addHandler(handler)
    return True


def setup_logging(config: LogConfig | None = None, *, force: bool = False) -> LogConfig:
    """装配日志。整个进程调用一次，越早越好。

    已经配过就直接返回上次的配置（除非 force=True）。

    返回**实际生效**的配置，它可能和传入的不同：文件路径不可用时会降级成终端。
    调用方想知道日志到底去了哪儿，看返回值，别猜。
    """
    global _CONFIGURED
    if _CONFIGURED is not None and not force:
        return _CONFIGURED

    cfg = config or LogConfig.from_env()
    root = logging.getLogger()

    # 先清掉 root 上已有的 handler。第三方库或框架可能已经挂过东西，
    # 不清就会和我们这套重复输出。
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    effective = cfg
    want_file = cfg.target in (LogTarget.FILE, LogTarget.BOTH)
    want_console = cfg.target in (LogTarget.CONSOLE, LogTarget.BOTH)

    if want_file and not _add_file_handler(root, cfg):
        # 文件这条路走不通：如果本来就只要文件，就退回终端；
        # BOTH 的话终端本来也要挂，不用改。
        if not want_console:
            effective = replace(cfg, target=LogTarget.CONSOLE)
            want_console = True

    if want_console:
        _add_console_handler(root, effective)

    # 只在 root 上打，不再往上冒泡，免得别的 handler 再打一遍
    root.propagate = False
    root.setLevel(level_value(effective.level))

    structlog.configure(
        processors=_build_chain(),
        # 日志事件最终交给标准库的 logger 发出去，而不是 structlog 自己的输出口 ——
        # 这一步是「两个世界合成一个」的关键
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        # 第一次取到 logger 就固化下来。日志在热路径上，
        # 这个开关省掉的是每次调用都要重走一遍的包装开销。
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = effective
    return effective


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """取一个 logger。约定：每个模块顶部 `logger = get_logger(__name__)`。

    用法：

        logger.info("执行命令", tool="run_cmd", cmd=command)
        logger.warning("命令没有正常结束", code=result.returncode)
        logger.exception("调用模型失败")        # 带栈，写在 except 块里

    第一个参数是**事件名**（这一步在做什么），后面用关键字参数带结构化字段。
    别用 f-string 把变量拼进事件名 —— 那样文件里只剩一个字符串，字段全没了，
    结构化日志也就白写了。
    """
    return structlog.get_logger(name)


__all__ = ["get_logger", "setup_logging"]

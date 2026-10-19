import subprocess
from pathlib import Path

from langchain_core.tools import tool

from logger import get_logger

from .sandbox import safe_path, workspace_root

logger = get_logger(__name__)

# 会被系统解释器当成「连接多条命令」或「接管输入输出」的字符
BLOCKED_CHARS = "&|;<>`$\n\r"
# 一条命令最多跑多久
TIMEOUT_SECONDS = 60

# 路径里用来分层的字符。Windows 两种都认，所以两种都要看。
_PATH_SEPARATORS = "/\\"

# —— 输出长度控制 ————————————————————————————————————————————
#
# 命令输出是没有上限的：一次编译、一次测试、一次 Schema 校验，几千上万字符很常见。
# 全量塞回对话会瞬间吃掉大量 token，甚至把上下文撑爆；而**从末尾砍掉**更糟 ——
# 真正有用的 Traceback 和错误总结恰恰压在最后几行。
#
# 所以超长时改成「头 + 尾」：头部留着命令的启动信息，尾部留着报错，
# 中间那段省略掉，同时把完整输出落到文件里 —— 需要时模型自己用 read_file 去读。
#
# 输出上限阈值：超过它就触发截断
MAX_OUTPUT_CHARS = 2000
# 截断时保留的头部长（命令启动、环境信息在这里）
HEAD_CHARS = 500
# 截断时保留的尾部长度。比头部长，因为报错和退出码都压在末尾
TAIL_CHARS = 1500
# 完整输出的落地位置，相对沙箱根目录解析。
#
# 为什么放在沙箱里而不是 logs/：它是**给模型读的**（提示语里让模型用 read_file
# 来读它），放在沙箱外模型就够不着了。代价是这一层内部状态对模型可见、也可写 ——
# 这是有意的取舍：拿得到完整报错，比藏得严实更值钱。文件名固定、每次覆盖，
# 因为它的语义是「上一条命令的完整输出」，不是归档，攒一堆反而没人翻。
CMD_LOG_RELPATH = Path(".jca") / "logs" / "cmd_last.log"


def check_single_command(command: str) -> str | None:
    """检查是不是「一条」命令；不合格就返回给模型看的中文说明，合格返回 None。"""
    if "\n" in command or "\r" in command:
        return "一次只能执行一条命令，请不要换行。请拆成多条，逐条调用本工具。"
    hit = next((c for c in command if c in BLOCKED_CHARS), None)
    if hit:
        return (
            f"命令里出现了 {hit!r}，它会把多条命令连在一起或者接管输入输出。"
            "本工具一次只能执行一条命令：请拆开逐条调用。"
            "如果需要把文字写进文件，请改用 write_file 工具。"
        )
    if not command.strip():
        return "命令是空的，请给出要执行的那一条命令。"
    return None


def _path_parts(word: str) -> list[str]:
    """把一个词按路径分层拆开，去掉空段和 `.`。

    `.` 是不改变位置的，去掉它之后 `./a/../b` 才和 `a/../b` 一样处理。
    """
    parts = [p for p in word.replace("\\", "/").split("/") if p not in ("", ".")]
    return parts


def _outside_path(token: str) -> str | None:
    """这个词是不是在往工作目录外面指；是就返回一句说明，不是返回 None。

    只做「看得出来」的判断，不去真正解析命令行。原因是命令行最后要交给
    cmd.exe / sh 执行，而两边对引号、转义、通配符的解释并不一样；照着
    某一个平台写解析器，换个平台就是错的。与其写一个半吊子的解析器，
    不如把明显越界的写法挡住：`cat /etc/passwd`（绝对路径）、
    `ls ../..`（用 .. 往上跳）、`type C:\\Windows\\win.ini`。

    判断的次序有讲究：

    - **先跳过以 `-` 开头的词**。那是选项不是路径，而 `/` 在 Windows 上
      是很多命令的选项前缀（`dir /s`、`tasklist /fi ...`），把它当绝对
      路径拦掉，模型连 `dir /s` 都用不了。
    - **`./` 前缀先剥掉再拆段**，这样 `./a/../b`（去掉 . 后第一段是 a）
      是「在工作目录里绕一圈」，不会被中间的 `..` 误伤；而 `../x`
      剥不出东西，第一段就是 `..`，该拦。
    - **只有第一段是 `..` 才算越界**。`..` 出现在中间只表示先下后上，
      原地返回，走不出去。

    这仍然不是完整的防护：词法上看不出 `python -c "open('/etc/passwd')"`
    这类藏在参数里的路径。README 的 Safety 一节也写明了这一点。
    """
    word = token.strip().strip("'\"")
    if not word or word.startswith("-"):
        return None

    # 没有分隔符、也不是 `.` / `..` 的词就是普通参数，不是路径
    if not any(s in word for s in _PATH_SEPARATORS) and word not in (".", ".."):
        return None

    # 剥掉开头的 `./` / `.\`：它们的意思是「就在工作目录里找」
    core = word
    while core[:2] in ("./", ".\\"):
        core = core[2:]
    if not core:
        return None

    if core.startswith("\\"):
        # `\\server\share` 是 UNC 路径
        return f"{word}（绝对路径）"
    if core.startswith("/"):
        # 以 / 开头就是绝对路径吗？在 Windows 上**不是**：`dir /s`、
        # `tasklist /fi ...` 里的 / 是命令自己的**选项**前缀，不是路径。
        # 两种写法共用一个字符，只能按「像不像路径」来分：只有一段
        # （/s、/fi）当选项放过，两段以上（/etc/passwd）当绝对路径拦下。
        # 代价是 /tmp 这种一段的 POSIX 路径会漏过去 —— 这是有意换的：
        # 误拦 `dir /s` 会让 Windows 上的模型连列目录都做不到，
        # 而那个代价比漏掉一个 /tmp 更常见、也更难解释。
        if "/" in core[1:]:
            return f"{word}（绝对路径）"
        return None
    if len(core) > 1 and core[1] == ":":
        # `C:\...` / `C:/...`：盘符开头的绝对路径。
        # 这里不额外判断盘符是不是「当前盘」——同一个盘也是绝对路径，
        # 一样能把命令指到工作目录外面去。
        return f"{word}（绝对路径）"

    parts = _path_parts(core)
    if parts and parts[0] == "..":
        return f"{word}（用 .. 跳出工作目录）"
    return None


def check_path_escape(command: str, root: Path) -> str | None:
    """命令里有没有明摆着指向工作目录之外的路径；有就返回中文说明，没有返回 None。

    单独一个函数（而不是塞进 check_single_command）是为了好测：
    它只依赖 command 和 root，不用起进程，也不用真的有个工作目录。
    """
    for token in command.split():
        reason = _outside_path(token)
        if reason:
            return (
                f"命令里出现了工作目录之外的路径：{reason}。"
                f"工作目录是 {root}，命令只能在这个目录里读写文件。"
                "请改用相对工作目录的路径（比如 ./src/app.py）。"
                "如果确实需要访问工作目录外面的东西，那是这套沙箱有意不允许的。"
            )
    return None


# —— 超长输出的处理 ————————————————————————————————————————————
def _output_log_path() -> Path:
    """完整输出该写到哪儿：沙箱根目录下的 .jca/logs/cmd_last.log。

    过一遍 safe_path 是有意的：沙箱根就是它自己的根，这个路径不可能越界，
    但走同一个入口意味着「写文件前先解析并校验路径」这条规矩没有例外 ——
    以后有人把常量改成 '../xx' 也拦得住，而不是等出事再想起来。
    """
    return safe_path(str(CMD_LOG_RELPATH))


def _persist_full_output(full_output: str) -> Path | None:
    """把完整输出落到文件，返回落地的绝对路径；失败返回 None。

    失败**不能**影响命令结果的返回：日志是辅助设施，它坏了不该让工具调用失败。
    所以这里吞掉异常、只记一条 warning，由调用方决定提示语怎么写。

    路径解析也放在 try 里：万一 CMD_LOG_RELPATH 被改成越界的东西，
    safe_path 会抛 ValueError —— 那是配置错误，但不该让一次命令调用崩掉。
    """
    try:
        path = _output_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline=""：命令输出里是什么换行就写什么，不做翻译
        path.write_text(full_output, encoding="utf-8", newline="")
        return path
    except Exception:
        logger.warning("完整命令输出写入失败，只返回截断后的内容", path=str(CMD_LOG_RELPATH))
        return None


def _elide_notice(truncated_count: int, saved_path: Path | None) -> str:
    """头尾之间那段提示语。

    措辞按「完整输出到底存下来了没有」分两种：写失败时还说「已保存到 …」
    等于教模型去 read_file 一个不存在的文件，它会白跑一趟还以为是自己的问题。
    """
    if saved_path is None:
        return (
            f"\n\n... [中间省略 {truncated_count} 字符] ...\n"
            "（完整输出保存失败，上面是头尾两段，中间内容这次拿不到了）\n\n"
        )
    return (
        f"\n\n... [中间省略 {truncated_count} 字符] ...\n"
        f"（完整输出已保存至 {CMD_LOG_RELPATH.as_posix()}"
        "，如需完整内容可用 read_file 工具读取）\n\n"
    )


def _format_and_persist_output(command: str, raw_output: str) -> str:
    """把命令输出整理成给模型看的样子，太长就截成「头 + 尾」。

    短输出原样返回。长输出保留头部 HEAD_CHARS 和尾部 TAIL_CHARS，中间省略，
    并把**未经截断**的完整内容写进 .jca/logs/cmd_last.log。

    为什么是头 + 尾而不是只留前面：报错和退出码都压在末尾。只截前面那 2000 字符，
    编译/测试失败时模型看到的全是正常输出，真正的 Traceback 一个字都看不到。

    为什么用切片 `[:HEAD_CHARS]` 而不是 splitlines()：这是**字符预算**，不是行预算。
    行数不能当预算用 —— 一条 5000 字符的日志可能压在一行里，也可能摊成一千行，
    只按行数砍，短行会浪费额度、长行照样撑爆。
    """
    if len(raw_output) <= MAX_OUTPUT_CHARS:
        return raw_output

    truncated_count = len(raw_output) - HEAD_CHARS - TAIL_CHARS
    # 写文件放在切之前：存的是完整的 raw_output，不是切完的两段。
    # 反过来的话，落地的文件也就缺了中间那段，模型 read_file 回来还是残缺的。
    saved_path = _persist_full_output(raw_output)

    head = raw_output[:HEAD_CHARS]
    tail = raw_output[-TAIL_CHARS:]
    result = head + _elide_notice(truncated_count, saved_path) + tail
    logger.info(
        "命令输出过长，已截断并保存完整输出",
        cmd=command,
        raw_chars=len(raw_output),
        returned_chars=len(result),
        truncated=truncated_count,
        path=str(saved_path) if saved_path else "",
        saved=saved_path is not None,
    )
    return result


@tool
def run_cmd(command: str) -> str:
    """在工作目录中执行一条命令，并返回它的输出。一次只能传一条命令，
    不能使用管道、重定向或者把多条命令连在一起。

    命令的工作目录就是沙箱根目录，命令里请只用相对它的路径：
    绝对路径（/etc/passwd、C:\\Windows\\...）和用 .. 往外跳的路径会被拦下。

    输出很长时会只给头尾两段（报错在尾部），完整内容在 .jca/logs/cmd_last.log，
    可以用 read_file 读。"""
    problem = check_single_command(command)
    if problem:
        # 被规则拦下也是一次「工具调用」，同样要留痕：
        # 事后看日志才能分清「模型没调这个工具」和「调了但被拦了」。
        logger.warning("命令被规则拦下，没有执行", cmd=command, reason=problem)
        return problem

    root = workspace_root()
    if not root.is_dir():
        # 配了 JCA_WORKSPACE 但那个目录还不存在。这里不替它建——建目录是
        # write_file 的事，命令该在哪儿跑应该由配置的人说了算，
        # 悄悄造一个出来只会让人以为配错了也能用。
        logger.error("工作目录不存在", cwd=str(root))
        return f"工作目录不存在：{root}，请先建好它，或检查 JCA_WORKSPACE 是否写对。"

    # 有了 root 再查路径：提示语里要带上工作目录，模型才知道相对谁写。
    # `cwd=root` 只决定命令**从哪儿开始**，拦不住命令自己写绝对路径，
    # 所以这一关必须单独做。
    escape = check_path_escape(command, root)
    if escape:
        # 和 check_single_command 一样记 warning：这不是普通报错，
        # 而是「有人在试着走出沙箱」，日志里得能事后查出来。
        logger.warning("命令试图访问工作目录之外的路径，没有执行", cmd=command)
        return escape

    logger.info("执行命令", cmd=command, cwd=str(root))
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            cwd=root,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error("命令超时，已中止", cmd=command, timeout=TIMEOUT_SECONDS)
        return f"命令执行超过 {TIMEOUT_SECONDS} 秒，已中止。"
    except Exception as e:
        # exception 会把栈也记下来。命令执行失败的原因常常在栈里
        # （编码问题、权限、路径），只记一句「执行出错」等于没记。
        logger.exception("执行命令时出错", cmd=command)
        return f"执行出错：{e}"

    output = (result.stdout or "").strip()
    if result.stderr:
        output += f"\n[错误输出]\n{result.stderr.strip()}"
    if result.returncode != 0:
        output += f"\n（命令退出码：{result.returncode}，说明它没有正常结束。）"
        logger.warning("命令没有正常结束", cmd=command, code=result.returncode)
    else:
        logger.info("命令执行完成", cmd=command, code=result.returncode, chars=len(output))
    if not output:
        return f"命令已执行，退出码 {result.returncode}，没有输出。"
    # 最后一步才做长度控制：前面拼出来的是**完整**输出，落地的文件也要是完整的。
    # 放在 return 之前而不是之后，是因为返回值就是工具结果本身，没有第二个出口。
    return _format_and_persist_output(command, output)

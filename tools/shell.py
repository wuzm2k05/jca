import subprocess

from langchain_core.tools import tool

from logger import get_logger

from .sandbox import workspace_root

logger = get_logger(__name__)

# 会被系统解释器当成「连接多条命令」或「接管输入输出」的字符
BLOCKED_CHARS = "&|;<>`$\n\r"
# 一条命令最多跑多久
TIMEOUT_SECONDS = 60


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


@tool
def run_cmd(command: str) -> str:
    """在工作目录中执行一条命令，并返回它的输出。一次只能传一条命令，
    不能使用管道、重定向或者把多条命令连在一起。"""
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
    return output or f"命令已执行，退出码 {result.returncode}，没有输出。"

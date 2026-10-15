"""默认关闭的高危能力：执行单条系统命令。"""

import subprocess

from langchain_core.tools import tool

from .sandbox import WORKSPACE

# 会被系统解释器当成「连接多条命令」或「接管输入输出」的字符
BLOCKED_CHARS = "&|;<>`$\n\r"


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
        return problem

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            cwd=WORKSPACE,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "命令执行超过 60 秒，已中止。"
    except Exception as e:
        return f"执行出错：{e}"

    output = (result.stdout or "").strip()
    if result.stderr:
        output += f"\n[错误输出]\n{result.stderr.strip()}"
    if result.returncode != 0:
        output += f"\n（命令退出码：{result.returncode}，说明它没有正常结束。）"
    return output or f"命令已执行，退出码 {result.returncode}，没有输出。"

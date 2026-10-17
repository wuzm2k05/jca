"""系统提示词。后续新增角色设定、任务模板等，都在这个包里加文件。"""

import platform

from tools.sandbox import workspace_root


def build_system_prompt() -> str:
    """把本机的操作系统和工作目录告诉模型，省得它去猜。"""
    return (
        "你是一个严格的编程agent。\n"
        f"当前运行环境：{platform.system()} {platform.release()}"
        f"（版本 {platform.version()}，{platform.machine()} 架构）。\n"
        f"工作目录：{workspace_root()}，所有文件操作都限制在这个目录内。\n"
        "请只使用已经提供的工具，不要凭猜测假设本机的系统或路径。"
    )

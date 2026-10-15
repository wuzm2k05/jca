"""路径沙箱：所有文件操作都被限制在工作目录内。"""

from pathlib import Path

# 沙箱根目录。用当前启动目录而不是脚本所在目录，
# 这样打包成 exe 之后，「在哪儿运行，文件就落在哪儿」的语义不变。
WORKSPACE = (Path.cwd() / "workspace").resolve()


def safe_path(raw: str) -> Path:
    """把模型给的相对路径解析成工作目录下的真实路径；越界就报错。"""
    target = (WORKSPACE / raw).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"路径超出工作目录范围，只允许操作 {WORKSPACE} 内的文件。")
    return target

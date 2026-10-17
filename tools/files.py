"""默认可用的文件能力：列目录、读文件、写文件、改文件。

约定：每个工具都返回一段给模型看的文字，出错时返回错误说明而不是抛异常，
因为返回值会被塞回对话里当作「观察结果」。
"""

from langchain_core.tools import tool

from logger import get_logger

from .sandbox import safe_path

logger = get_logger(__name__)

# 单个文件最多读出的字符数，防止把上下文撑爆
MAX_READ_CHARS = 20000
# 目录一次最多列出多少项
MAX_LIST_ITEMS = 200


def _decode(raw: bytes) -> tuple[str, str] | None:
    """按常见编码依次尝试解码；返回（文本, 用到的编码），都不行就返回 None。

    把编码也带出来，是为了改完文件之后能按原来的编码写回去，
    不至于把一个 gbk 文件悄悄变成 utf-8。
    """
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None


@tool
def list_dir(path: str = ".") -> str:
    """列出工作目录下某个目录里的文件和子目录，只列一层，不递归。"""
    try:
        target = safe_path(path)
        if not target.exists():
            logger.warning("列目录：目录不存在", path=path)
            return f"目录不存在：{path}"
        if not target.is_dir():
            logger.warning("列目录：目标不是目录", path=path)
            return f"这不是一个目录：{path}"
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        if not entries:
            logger.info("列目录：空目录", path=path)
            return f"{path} 是空的。"
        lines = [
            ("[目录] " if p.is_dir() else "[文件] ") + p.name
            for p in entries[:MAX_LIST_ITEMS]
        ]
        if len(entries) > MAX_LIST_ITEMS:
            lines.append(f"（还有 {len(entries) - MAX_LIST_ITEMS} 项未显示）")
        logger.info("列目录完成", path=path, count=len(entries))
        return "\n".join(lines)
    except Exception as e:
        logger.exception("列目录失败", path=path)
        return f"列目录失败：{e}"


@tool
def read_file(path: str, max_chars: int = MAX_READ_CHARS) -> str:
    """读取工作目录下某个文本文件的内容。"""
    try:
        target = safe_path(path)
        if not target.exists():
            logger.warning("读文件：文件不存在", path=path)
            return f"文件不存在：{path}"
        if not target.is_file():
            logger.warning("读文件：目标不是文件", path=path)
            return f"这不是一个文件：{path}"

        raw = target.read_bytes()
        if b"\x00" in raw[:4096]:
            logger.warning("读文件：拒绝按文本读取二进制文件", path=path)
            return f"这是个二进制文件，不能按文本读取：{path}"

        decoded = _decode(raw)
        if decoded is None:
            logger.warning("读文件：编码无法识别", path=path)
            return f"文件编码无法识别（既不是 utf-8 也不是 gbk）：{path}"
        text, encoding = decoded
        truncated = len(text) > max_chars
        logger.info(
            "读文件完成",
            path=path,
            encoding=encoding,
            chars=len(text),
            truncated=truncated,
        )
        if truncated:
            return text[:max_chars] + f"\n（内容已截断，文件共 {len(text)} 个字符）"
        return text
    except Exception as e:
        logger.exception("读文件失败", path=path)
        return f"读取失败：{e}"


@tool
def write_file(path: str, content: str) -> str:
    """把内容写入工作目录下的某个文件，文件不存在就新建，父目录会自动创建。"""
    try:
        target = safe_path(path)
        if target.exists() and target.is_dir():
            logger.warning("写文件：目标是个目录", path=path)
            return f"这是个目录，不能当作文件写入：{path}"
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        # newline="" 表示不做换行翻译：写进去什么就是什么，
        # 免得在 Windows 上把内容里的换行悄悄改成另一种。
        target.write_text(content, encoding="utf-8", newline="")
        action = "已覆盖" if existed else "已新建"
        # 文件写操作是这套工具里后果最重的，每一次都记下来：
        # 事后要能回答「哪个文件、什么时候、被谁写的」。
        logger.info("写文件完成", path=path, action=action, chars=len(content))
        return f"{action} {path}，共写入 {len(content)} 个字符。"
    except Exception as e:
        logger.exception("写文件失败", path=path)
        return f"写入失败：{e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """把文件里的一段旧内容替换成新内容。只替换一处，
    所以旧内容必须在文件中唯一出现；找不到就先去读一遍文件。"""
    try:
        target = safe_path(path)
        if not target.exists():
            logger.warning("改文件：文件不存在", path=path)
            return f"文件不存在：{path}"
        if not target.is_file():
            logger.warning("改文件：目标不是文件", path=path)
            return f"这不是一个文件：{path}"

        raw = target.read_bytes()
        decoded = _decode(raw)
        if decoded is None:
            logger.warning("改文件：编码无法识别", path=path)
            return f"文件编码无法识别（既不是 utf-8 也不是 gbk）：{path}"
        text, encoding = decoded

        count = text.count(old_text)
        if count == 0:
            logger.warning("改文件：没找到要替换的内容", path=path)
            return f"没找到这段内容，请先用 read_file 看清楚原文：{old_text}"
        if count > 1:
            logger.warning("改文件：要替换的内容不唯一", path=path, matches=count)
            return f"这段内容出现了 {count} 次，请给出更长、更唯一的一段：{old_text}"

        updated = text.replace(old_text, new_text, 1)
        # 按原编码写回，并且不做换行翻译：
        # 只改动被替换的那一处，文件其余部分（换行风格、编码）保持原样
        target.write_text(updated, encoding=encoding, newline="")
        logger.info(
            "改文件完成",
            path=path,
            encoding=encoding,
            old_chars=len(old_text),
            new_chars=len(new_text),
        )
        return f"已替换 {path} 里的一处内容，文件现在共 {len(updated)} 个字符。"
    except Exception as e:
        logger.exception("改文件失败", path=path)
        return f"修改失败：{e}"


# 文件工具清单。想加一个新能力，写好后加到这里即可。
FILE_TOOLS = [list_dir, read_file, write_file, edit_file]

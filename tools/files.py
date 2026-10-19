"""默认可用的文件能力：列目录、找文件、读文件、写文件、按内容/按行号改文件。

约定：每个工具都返回一段给模型看的文字，出错时返回错误说明而不是抛异常，
因为返回值会被塞回对话里当作「观察结果」。

## 行号这两个工具为什么要配套

大文件是这套工具最容易撑爆上下文的地方，所以：

- `read_file` 每行带行号，并且可以按行区间**分片**读，不再只能从头读到截断；
- `replace_lines` 按行号整段改，绕开 `edit_file` 那种「几十行里错一个空格就匹配
  不上」的问题。

两者必须用同一套行号。`read_file` 用 `str.splitlines()` 数行，`replace_lines` 用
`str.splitlines(keepends=True)` 切行 —— 对同一个字符串，这两个切出来的**行数是一样的**，
所以 read_file 说的第 N 行就是 replace_lines 改的那一行。谁要动行号算法，这两个一起动。
"""

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import tool

from logger import get_logger

from .sandbox import safe_path, workspace_root

logger = get_logger(__name__)

# 单个文件最多读出的字符数，防止把上下文撑爆
MAX_READ_CHARS = 20000
# 目录一次最多列出多少项
MAX_LIST_ITEMS = 200
# 递归列目录的层数上限。模型偶尔会传个很大的数，真按它走一遍既慢，
# 又只会被 MAX_LIST_ITEMS 截成「一条很长的路径」，不如早点收住。
MAX_LIST_DEPTH = 10
# find_files 最多返回多少个匹配
MAX_FIND_RESULTS = 100

# 列目录和找文件时默认跳过的名字。这些目录动辄几万项，
# 列出来除了占满 MAX_LIST_ITEMS 没有任何用处。
DEFAULT_IGNORE = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


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


def _is_ignored(name: str) -> bool:
    """名字是不是默认忽略项。

    比较时统一转小写：Windows 上写 `.GIT` 和写 `.git` 是同一个目录，
    而这类目录名大小写并不统一（`.venv` / `.Venv` 都见过）。
    """
    return name.lower() in DEFAULT_IGNORE


def _detect_newline(text: str) -> str:
    """猜文件用的换行风格，用来给**新写入**的内容配换行。

    写文件时用的是 `newline=""`（不做换行翻译），所以新内容里的换行得自己给对，
    否则会把一个 CRLF 文件改成混着 LF 的。
    """
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _has_visible_children(target: Path) -> bool:
    """这个目录下面还有没有被忽略的内容（只看一层就够）。"""
    try:
        return any(not _is_ignored(p.name) for p in target.iterdir())
    except OSError:
        # 权限不足之类：当作没有，不要把列目录整个搞失败
        return False


@dataclass
class _WalkTally:
    """列目录一路带着走的计数。"""

    ignored: int = 0
    # 收名字是为了回一句「跳过了哪些」，模型看到 .git 被跳过，
    # 才不会以为这个目录真的不存在
    ignored_names: set[str] = field(default_factory=set)
    truncated: bool = False


# —— list_dir ————————————————————————————————————————————————
def _walk(
    target: Path,
    depth: int,
    max_depth: int,
    lines: list[str],
    tally: _WalkTally,
) -> None:
    """把 target 下面 max_depth 层的内容收集进 lines。

    计数就地记在 tally 上，而不是靠返回值往上传：递归下去好几层，
    把计数一路拼回来只会让这里更难读。
    """
    try:
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        lines.append(f"{'  ' * depth}[无法读取] {target.name}：{e}")
        return

    for entry in entries:
        if _is_ignored(entry.name):
            tally.ignored += 1
            tally.ignored_names.add(entry.name)
            continue
        if len(lines) >= MAX_LIST_ITEMS:
            tally.truncated = True
            return

        indent = "  " * depth
        if not entry.is_dir():
            lines.append(f"{indent}[文件] {entry.name}")
            continue
        if depth + 1 >= max_depth:
            # 到深度上限了，不再往下走；但要说清楚「下面还有东西」，
            # 模型才知道该不该用更大的 max_depth 再调一次。
            tail = " …（下面还有内容）" if _has_visible_children(entry) else ""
            lines.append(f"{indent}[目录] {entry.name}/{tail}")
            continue

        lines.append(f"{indent}[目录] {entry.name}/")
        _walk(entry, depth + 1, max_depth, lines, tally)


@tool
def list_dir(path: str = ".", max_depth: int = 2) -> str:
    """列出工作目录下某个目录的内容，默认往下递归 2 层，用缩进表示层级。

    会自动跳过 .git、.venv、node_modules、__pycache__ 这类目录，
    最多显示 200 项。想看得更深就把 max_depth 调大。
    """
    try:
        target = safe_path(path)
        if not target.exists():
            logger.warning("列目录：目录不存在", path=path)
            return f"目录不存在：{path}"
        if not target.is_dir():
            logger.warning("列目录：目标不是目录", path=path)
            return f"这不是一个目录：{path}"

        # 至少 1 层（也就是原来「只列一层」的行为），至多 MAX_LIST_DEPTH 层
        depth = max(1, min(max_depth, MAX_LIST_DEPTH))
        lines: list[str] = []
        tally = _WalkTally()
        _walk(target, 0, depth, lines, tally)

        if not lines:
            # 「本来就是空的」和「东西全被忽略了」是两件事，分开说，
            # 否则模型会以为目录里真的没东西。
            if tally.ignored:
                names = "、".join(sorted(tally.ignored_names)[:3])
                logger.info("列目录：内容全在忽略名单里", path=path)
                return f"{path} 下只有默认忽略项（{names} 等），没有别的可显示的内容。"
            logger.info("列目录：空目录", path=path)
            return f"{path} 是空的。"

        out = [f"列出 {path}（最多 {depth} 层）：", *lines]
        notes: list[str] = []
        if tally.truncated:
            notes.append(
                f"已达 {MAX_LIST_ITEMS} 项上限，只显示了一部分"
                "（缩小 path 范围或调小 max_depth 可以看全）"
            )
        if tally.ignored:
            names = "、".join(sorted(tally.ignored_names)[:3])
            more = " 等" if len(tally.ignored_names) > 3 else ""
            notes.append(f"已跳过 {tally.ignored} 项默认忽略项（{names}{more}）")
        if notes:
            out.append("（" + "；".join(notes) + "）")

        logger.info(
            "列目录完成",
            path=path,
            depth=depth,
            lines=len(lines),
            ignored=tally.ignored,
            truncated=tally.truncated,
        )
        return "\n".join(out)
    except Exception as e:
        logger.exception("列目录失败", path=path)
        return f"列目录失败：{e}"


# —— find_files ————————————————————————————————————————————————
@tool
def find_files(pattern: str = "*", path: str = ".") -> str:
    """在工作目录下递归查找文件名匹配 pattern 的文件，返回相对工作目录的路径。

    pattern 支持通配符：`*.py`、`*auth*`、`test_*.py`；也可以带目录，
    比如 `tools/*.py`。自动跳过 .git、node_modules 等目录，最多返回 100 个匹配。
    """
    try:
        target = safe_path(path)
        if not target.exists():
            logger.warning("找文件：目录不存在", path=path)
            return f"目录不存在：{path}"
        if not target.is_dir():
            logger.warning("找文件：起点不是目录", path=path)
            return f"查找起点不是一个目录：{path}"

        root = workspace_root()
        matches: list[str] = []
        skipped = 0
        # 用 os.walk 而不是 Path.rglob：它允许**就地剪枝**（dirnames[:] = ...），
        # 被忽略的目录连进都不会进去，比「先全遍历再过滤」省得多。
        for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
            keep = sorted(d for d in dirnames if not _is_ignored(d))
            skipped += len(dirnames) - len(keep)
            dirnames[:] = keep
            for name in sorted(filenames):
                if _is_ignored(name):
                    skipped += 1
                    continue
                full = Path(dirpath, name)
                rel = _relative_to_root(full, root)
                # 文件名和「相对工作目录的路径」都参与匹配：
                # 前者管 `*.py`，后者管 `tools/*.py` 这种带目录的写法。
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                    matches.append(rel)
                    if len(matches) >= MAX_FIND_RESULTS:
                        break
            if len(matches) >= MAX_FIND_RESULTS:
                break

        if not matches:
            logger.info("找文件：没有匹配", path=path, pattern=pattern)
            where = "" if path in (".", "./") else f"（在 {path} 下）"
            hint = f"已跳过 {skipped} 项默认忽略项，如 .git、node_modules。" if skipped else ""
            return f"没找到匹配 {pattern!r} 的文件{where}。{hint}"

        logger.info("找文件完成", path=path, pattern=pattern, count=len(matches))
        out = [f"匹配 {pattern!r} 的文件（共 {len(matches)} 个，路径相对工作目录）：", *matches]
        if len(matches) >= MAX_FIND_RESULTS:
            out.append(f"（只显示前 {MAX_FIND_RESULTS} 个，请把 pattern 写得更具体一些）")
        return "\n".join(out)
    except Exception as e:
        logger.exception("找文件失败", path=path, pattern=pattern)
        return f"查找失败：{e}"


def _relative_to_root(full: Path, root: Path) -> str:
    """把绝对路径转成「相对工作目录」的写法，一律用 / 分隔。

    路径是直接喂给 read_file / replace_lines 用的，所以必须是能对上的相对路径；
    分隔符统一成 / 也是因为模型对 / 的拼接比 \\ 稳（Windows 的 API 两种都认）。
    """
    try:
        return full.relative_to(root).as_posix()
    except ValueError:
        # 理论上去不到这里（起点在沙箱内），软链接指到外面时才会碰上。
        return full.as_posix()


# —— read_file ————————————————————————————————————————————————
@tool
def read_file(
    path: str,
    start_line: int = 1,
    line_count: int | None = None,
    max_chars: int = MAX_READ_CHARS,
) -> str:
    """读取工作目录下某个文本文件的内容，每行前面带行号。

    默认整篇读。文件很长时可以按行区间分片读，比如 start_line=200、
    line_count=100 就是读第 200-299 行；返回值末尾会提示下一段从哪一行接着读。

    行号那一列是工具加上去的，不是文件内容本身 —— 往 write_file 的 content 或
    replace_lines 的 new_content 里填正文时，别把行号一起抄进去。
    """
    try:
        target = safe_path(path)
        if not target.exists():
            logger.warning("读文件：文件不存在", path=path)
            return f"文件不存在：{path}"
        if not target.is_file():
            logger.warning("读文件：目标不是文件", path=path)
            return f"这不是一个文件：{path}"
        if start_line < 1:
            logger.warning("读文件：起始行号非法", path=path, start_line=start_line)
            return f"start_line 从 1 开始数，不能是 {start_line}。"
        if line_count is not None and line_count < 1:
            logger.warning("读文件：行数非法", path=path, line_count=line_count)
            return f"line_count 至少要是 1，不能是 {line_count}；想读到结尾就别传它。"

        raw = target.read_bytes()
        if b"\x00" in raw[:4096]:
            logger.warning("读文件：拒绝按文本读取二进制文件", path=path)
            return f"这是个二进制文件，不能按文本读取：{path}"

        decoded = _decode(raw)
        if decoded is None:
            logger.warning("读文件：编码无法识别", path=path)
            return f"文件编码无法识别（既不是 utf-8 也不是 gbk）：{path}"
        text, encoding = decoded

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            logger.info("读文件：空文件", path=path, encoding=encoding)
            return f"{path} 是个空文件（0 行）。"
        if start_line > total:
            logger.warning("读文件：起始行超过总行数", path=path, start_line=start_line, total=total)
            return f"start_line={start_line} 超过了文件总行数（{total} 行）：{path}"

        end = total if line_count is None else min(start_line + line_count - 1, total)
        width = len(str(end))

        body: list[str] = []
        used = 0
        last = start_line - 1
        for no in range(start_line, end + 1):
            piece = f"{no:>{width}}\t{lines[no - 1]}"
            # 第一条无论如何都收进来：上限再小，也得让模型看见东西，
            # 而不是回一段只有提示语的空结果。
            if body and used + len(piece) + 1 > max_chars:
                break
            body.append(piece)
            used += len(piece) + 1
            last = no

        out = [f"（{path}：第 {start_line}-{end} 行，共 {total} 行）", *body]
        if last < total:
            # 关键的地方：不再像以前那样把后半段直接砍掉，而是告诉模型
            # 「从哪一行接着读」，它才有办法自己把长文件读完。
            why = f"单次读取上限 {max_chars} 字符，本段只到第 {last} 行；" if last < end else ""
            out.append(f"（{why}文件还有 {total - last} 行，接着读请传 start_line={last + 1}）")
        logger.info(
            "读文件完成",
            path=path,
            encoding=encoding,
            total_lines=total,
            start=start_line,
            end=last,
        )
        return "\n".join(out)
    except Exception as e:
        logger.exception("读文件失败", path=path)
        return f"读取失败：{e}"


# —— write_file ————————————————————————————————————————————————
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


# —— edit_file ————————————————————————————————————————————————
@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """把文件里的一段旧内容替换成新内容。只替换一处，
    所以旧内容必须在文件中唯一出现；找不到就先去读一遍文件。

    要改一整段（比如几十行的函数）时用 replace_lines 更省事：
    这个工具要求旧内容一字不差，空格和缩进差一个就匹配不上。"""
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


# —— replace_lines ————————————————————————————————————————————————
def _split_new_content(new_content: str, newline: str) -> list[str]:
    """把新内容切成「带换行符的行」，好接回原来的文件里。

    两件事必须一起处理，少了哪件都会把文件接坏：

    1. 统一成文件自己的换行风格（否则一个 CRLF 文件里会混进 LF）；
    2. 决定**最后一行**要不要补换行 —— 新内容自己没写 \\n、但它后面还跟着原文的
       下一行时，不补就会把两行粘成一行。
    """
    if new_content == "":
        return []
    normalized = new_content.replace("\r\n", "\n").replace("\r", "\n")
    ends_with_newline = normalized.endswith("\n")
    core = normalized[:-1] if ends_with_newline else normalized
    parts = core.split("\n")
    return [p + newline if i < len(parts) - 1 else p for i, p in enumerate(parts)]


@tool
def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """把文件里第 start_line 到 end_line 行整段换成 new_content（行号闭区间，从 1 开始）。

    行号就是 read_file 左边那一列。new_content 传空字符串就是删掉这几行。
    改一整个函数、或者改一大段但拿不准空格缩进时用这个；
    只改一小处字符串用 edit_file 更省事。
    """
    try:
        target = safe_path(path)
        if not target.exists():
            logger.warning("按行替换：文件不存在", path=path)
            return f"文件不存在：{path}"
        if not target.is_file():
            logger.warning("按行替换：目标不是文件", path=path)
            return f"这不是一个文件：{path}"
        if start_line < 1 or end_line < start_line:
            logger.warning(
                "按行替换：行号区间非法", path=path, start=start_line, end=end_line
            )
            return (
                f"行号区间不合法：start_line={start_line}，end_line={end_line}。"
                "行号从 1 开始，且要满足 start_line <= end_line。"
            )

        raw = target.read_bytes()
        if b"\x00" in raw[:4096]:
            logger.warning("按行替换：拒绝改二进制文件", path=path)
            return f"这是个二进制文件，不能按文本修改：{path}"
        decoded = _decode(raw)
        if decoded is None:
            logger.warning("按行替换：编码无法识别", path=path)
            return f"文件编码无法识别（既不是 utf-8 也不是 gbk）：{path}"
        text, encoding = decoded

        # keepends=True：没动到的行原样保留（连换行符都不碰），
        # 这样「保留原文件的换行风格」是默认结果，而不是靠我们再拼一次。
        lines = text.splitlines(keepends=True)
        total = len(lines)
        if total == 0:
            logger.warning("按行替换：空文件", path=path)
            return f"{path} 是个空文件（0 行），没有行可替换。"
        if start_line > total or end_line > total:
            # 这里不「顺手夹到文件末尾」：静默地少改几行，比报错难查得多。
            logger.warning(
                "按行替换：行号超出文件范围",
                path=path,
                start=start_line,
                end=end_line,
                total=total,
            )
            return (
                f"行号超出范围：文件一共 {total} 行，"
                f"而你要改的是第 {start_line}-{end_line} 行。"
                "请先用 read_file 确认行号（它的输出里每行都带行号）。"
            )

        newline = _detect_newline(text)
        pieces = _split_new_content(new_content, newline)
        # 这一段的末尾本来是换行的（即它不是「没有换行符的最后一行」），
        # 新内容又不自带换行时，得替它补上，否则会和下一行粘起来。
        region_ends_with_newline = lines[end_line - 1].endswith(("\n", "\r"))
        if pieces and region_ends_with_newline and not pieces[-1].endswith(newline):
            pieces[-1] += newline

        updated = "".join(lines[: start_line - 1]) + "".join(pieces) + "".join(lines[end_line:])
        # 按原编码写回；newline="" 保证上面的换行处理就是最终结果，不再被翻译一遍
        target.write_text(updated, encoding=encoding, newline="")

        removed = end_line - start_line + 1
        kept = len("".join(pieces).splitlines())
        action = "已删除" if not pieces else f"已替换为 {kept} 行"
        logger.info(
            "按行替换完成",
            path=path,
            encoding=encoding,
            start=start_line,
            end=end_line,
            removed=removed,
            added=kept,
        )
        return (
            f"{action} {path} 的第 {start_line}-{end_line} 行"
            f"（原 {removed} 行），文件现在共 {len(updated.splitlines())} 行。"
        )
    except Exception as e:
        logger.exception("按行替换失败", path=path)
        return f"按行替换失败：{e}"


# 文件工具清单。想加一个新能力，写好后加到这里即可。
FILE_TOOLS = [list_dir, find_files, read_file, write_file, edit_file, replace_lines]

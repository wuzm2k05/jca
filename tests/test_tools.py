"""工具层的离线测试：不碰网络，也不动仓库里已有的文件。

跑法（两种都行）：

    .venv/Scripts/python.exe -m tests.test_tools
    uv run --with pytest pytest tests/            # 装了 pytest 的话

为什么值得单独测：这一层直接动用户的磁盘，而且**失败是静默的** ——
`replace_lines` 行号偏了一行，照样写出一个语法合法的文件，看起来只是
「模型改错了地方」。所以这里把行号语义、换行风格、忽略名单这些容易错
又不容易看出来的地方钉住。

所有 fixture 都建在**当前目录下的临时目录**里，而不是系统临时目录：
`safe_path` 只允许操作工作目录内的路径（见 tools/sandbox.py），
测试要跑的正是这条真实路径，不能绕开它。
"""

import shutil
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import BaseTool

from tools import build_tools
from tools.files import (
    DEFAULT_IGNORE,
    FILE_TOOLS,
    MAX_FIND_RESULTS,
    find_files,
    list_dir,
    read_file,
    replace_lines,
    write_file,
)
from tools.sandbox import workspace_root
from tools.shell import check_path_escape, run_cmd


def _call(tool: BaseTool, **kwargs) -> str:
    """调用一个工具。走 .invoke 而不是直调函数，顺带验一遍参数 schema 是对的。"""
    return tool.invoke(kwargs)


class _Sandbox:
    """在运行目录下开一个临时目录，用完删掉。

    用 with 而不是在每个用例里手写 try/finally：漏一个 finally 就会在仓库里
    留下一堆临时文件，而下一次 `list_dir "."` 就会把它们列给模型看。
    """

    def __enter__(self) -> Path:
        self.root = Path(tempfile.mkdtemp(prefix="jca_test_tmp_", dir=Path.cwd()))
        return self.root

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _rel(path: Path) -> str:
    """把绝对路径换成相对工作目录的写法，和模型给出的路径形状一致。"""
    return path.relative_to(workspace_root()).as_posix()


def _plant_tree(root: Path) -> None:
    """一棵有代表性的小目录树：嵌套、空目录、以及该被忽略的垃圾目录。"""
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "app.py").write_text("print('app')\n", encoding="utf-8")
    (root / "pkg" / "sub" / "deep.py").write_text("print('deep')\n", encoding="utf-8")
    (root / "empty_dir").mkdir()
    # 忽略名单里的目录，各放一个「如果没被过滤掉就一定会露出来」的文件
    for name in (".git", ".venv", "node_modules", "__pycache__"):
        junk = root / name
        junk.mkdir()
        (junk / "should_not_show.py").write_text("pass\n", encoding="utf-8")
    (root / "notes.txt").write_text("hello\n", encoding="utf-8")


# —— list_dir ————————————————————————————————————————————————
def test_list_dir_recurses_and_hides_ignored_dirs() -> None:
    """默认递归两层：子目录里的文件看得见，`.git`/`.venv` 这些看不见。

    这条是这次改动的主因：旧版只列一层，模型要看 `pkg/` 里有什么
    得再调一次；而现在一次就能看到，且不会被垃圾目录顶掉 200 项的额度。
    """
    with _Sandbox() as root:
        _plant_tree(root)
        out = _call(list_dir, path=_rel(root))

        assert "[文件] notes.txt" in out, out
        assert "[目录] pkg/" in out, out
        # 第二层的内容要出现（缩进两格）
        assert "  [文件] app.py" in out, out
        assert "[目录] empty_dir/" in out, out
        for name in DEFAULT_IGNORE:
            assert f"[目录] {name}/" not in out, (name, out)
            assert f"[文件] {name}" not in out, (name, out)
        assert "已跳过" in out, out
        assert "should_not_show" not in out, out


def test_list_dir_max_depth_one_is_the_old_behaviour() -> None:
    """max_depth=1 回到「只列一层」，并且明说下面还有内容。"""
    with _Sandbox() as root:
        _plant_tree(root)
        out = _call(list_dir, path=_rel(root), max_depth=1)

        assert "[目录] pkg/" in out, out
        assert "app.py" not in out, out
        # 到深度上限时要说清楚「下面还有东西」，否则模型会以为 pkg 是空的
        assert "下面还有内容" in out, out


def test_list_dir_distinguishes_empty_from_all_ignored() -> None:
    """「本来就是空的」和「内容全在忽略名单里」要说成两句不同的话。"""
    with _Sandbox() as root:
        (root / "blank").mkdir()
        (root / "junk" / ".git").mkdir(parents=True)

        assert "是空的" in _call(list_dir, path=_rel(root / "blank"))
        all_ignored = _call(list_dir, path=_rel(root / "junk"))
        assert "忽略" in all_ignored, all_ignored
        assert "是空的" not in all_ignored, all_ignored


def test_list_dir_missing_path_returns_message() -> None:
    """不存在的目录返回中文说明，不抛异常（返回值要能塞回对话）。"""
    with _Sandbox() as root:
        out = _call(list_dir, path=_rel(root / "nope"))
        assert "目录不存在" in out, out


# —— find_files ————————————————————————————————————————————————
def test_find_files_matches_nested_and_skips_ignored() -> None:
    """递归查找：嵌套的文件找得到，忽略目录里的找不到。"""
    with _Sandbox() as root:
        _plant_tree(root)
        out = _call(find_files, pattern="*.py", path=_rel(root))

        assert "pkg/app.py" in out, out
        assert "pkg/sub/deep.py" in out, out
        assert "should_not_show" not in out, out


def test_find_files_supports_substring_pattern() -> None:
    """`*auth*` 这种按名字片段找的写法也要能用。"""
    with _Sandbox() as root:
        (root / "auth_service.py").write_text("pass\n", encoding="utf-8")
        (root / "other.py").write_text("pass\n", encoding="utf-8")

        out = _call(find_files, pattern="*auth*", path=_rel(root))
        assert "auth_service.py" in out, out
        assert "other.py" not in out, out


def test_find_files_no_match_explains_itself() -> None:
    """没找到时说清楚「在哪儿找的」，而不是只回一个空字符串。"""
    with _Sandbox() as root:
        out = _call(find_files, pattern="*.rs", path=_rel(root))
        assert "没找到" in out, out
        assert "*.rs" in out, out


def test_find_files_caps_results() -> None:
    """匹配太多时只回前 100 个，并提示把 pattern 写具体些。"""
    with _Sandbox() as root:
        for i in range(MAX_FIND_RESULTS + 20):
            (root / f"m{i:03d}.py").write_text("pass\n", encoding="utf-8")

        out = _call(find_files, pattern="*.py", path=_rel(root))
        assert f"前 {MAX_FIND_RESULTS} 个" in out, out


# —— read_file ————————————————————————————————————————————————
def test_read_file_numbers_lines_and_slices() -> None:
    """按行分片读：行号对得上，并且告诉模型下一段从哪一行接着读。

    这是要解决的老问题：以前超过上限直接砍掉后半段，模型只能看见文件开头。
    """
    with _Sandbox() as root:
        target = root / "long.txt"
        target.write_text("".join(f"第{i}行\n" for i in range(1, 11)), encoding="utf-8")

        whole = _call(read_file, path=_rel(target))
        assert "共 10 行" in whole, whole
        assert "10\t第10行" in whole, whole

        part = _call(read_file, path=_rel(target), start_line=5, line_count=3)
        assert "第 5-7 行" in part, part
        assert "5\t第5行" in part, part
        assert "7\t第7行" in part, part
        assert "第4行" not in part, part
        assert "第8行" not in part, part
        # 分片读取的关键：得告诉模型下一步读哪儿
        assert "start_line=8" in part, part


def test_read_file_byte_budget_points_at_next_chunk() -> None:
    """即使没给 line_count，撞上字符上限时也要给出「接着读」的行号。"""
    with _Sandbox() as root:
        target = root / "big.txt"
        target.write_text("".join(f"第{i}行".ljust(40) + "\n" for i in range(1, 51)), encoding="utf-8")

        out = _call(read_file, path=_rel(target), max_chars=200)
        assert "单次读取上限" in out, out
        assert "start_line=" in out, out


def test_read_file_reports_bad_range() -> None:
    """行号越界/非法：返回说明，而不是抛异常或者悄悄夹到末尾。"""
    with _Sandbox() as root:
        target = root / "small.txt"
        target.write_text("a\nb\nc\n", encoding="utf-8")

        assert "超过了文件总行数" in _call(read_file, path=_rel(target), start_line=99)
        assert "从 1 开始" in _call(read_file, path=_rel(target), start_line=0)


# —— replace_lines ————————————————————————————————————————————————
def test_replace_lines_replaces_inclusive_range() -> None:
    """闭区间：[start, end] 整段换掉，文件其余部分一个字都不动。

    闭区间这个约定必须和 read_file 的行号一致，否则模型按读到的行号来改，
    总会多改或者少改一行。
    """
    with _Sandbox() as root:
        target = root / "m.py"
        target.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

        out = _call(
            replace_lines,
            path=_rel(target),
            start_line=2,
            end_line=4,
            new_content="B\nC",
        )
        assert target.read_text(encoding="utf-8") == "a\nB\nC\ne\n"
        assert "第 2-4 行" in out, out
        # 换行风格：新内容自己没写行尾，接回原文时不能把 C 和 e 粘成一行
        assert "共 4 行" in out, out


def test_replace_lines_empty_content_deletes() -> None:
    """new_content 传空串 = 删掉这几行。"""
    with _Sandbox() as root:
        target = root / "m.py"
        target.write_text("a\nb\nc\n", encoding="utf-8")

        out = _call(replace_lines, path=_rel(target), start_line=2, end_line=2, new_content="")
        assert target.read_text(encoding="utf-8") == "a\nc\n"
        assert "已删除" in out, out


def test_replace_lines_keeps_the_original_encoding() -> None:
    """写入要保留原文件的换行风格，也要保留原编码。

    两件事都做过：把一个 CRLF 文件改成混着 LF，把 gbk 文件悄悄变成 utf-8，
    git diff 上看起来都只是「整文件都变了」，很难追。
    """
    with _Sandbox() as root:
        target = root / "gbk_crlf.txt"
        target.write_bytes("第一行\r\n第二行\r\n第三行\r\n".encode("gbk"))

        _call(replace_lines, path=_rel(target), start_line=2, end_line=2, new_content="改过了")

        raw = target.read_bytes()
        # 还能按 gbk 解码 = 没被改成 utf-8；剩下的换行都还是 CRLF
        text = raw.decode("gbk")
        assert text == "第一行\r\n改过了\r\n第三行\r\n", repr(text)


def test_replace_lines_rejects_bad_range_and_keeps_file() -> None:
    """行号不合法时**一个字节都不写**：宁可报错，也不要静默地少改几行。"""
    with _Sandbox() as root:
        target = root / "m.py"
        original = "a\nb\nc\n"
        target.write_text(original, encoding="utf-8")

        too_long = _call(replace_lines, path=_rel(target), start_line=2, end_line=99, new_content="x")
        assert "行号超出范围" in too_long, too_long
        assert "一共 3 行" in too_long, too_long

        inverted = _call(replace_lines, path=_rel(target), start_line=3, end_line=1, new_content="x")
        assert "不合法" in inverted, inverted

        assert target.read_text(encoding="utf-8") == original


def test_replace_lines_respects_trailing_newline() -> None:
    """文件末尾有换行就留着，没有就别加。

    「换行风格」不只是 CRLF 和 LF 的区别：末尾有没有换行也是风格的一部分，
    而且它一样会被 git diff 放大成「整文件都变了」。改最后一行时新内容自己
    不带行尾，这时要**看原文那一行**的写法跟着走。
    """
    with _Sandbox() as root:
        with_nl = root / "with_nl.txt"
        with_nl.write_text("a\nb\n", encoding="utf-8")
        _call(replace_lines, path=_rel(with_nl), start_line=2, end_line=2, new_content="b!")
        assert with_nl.read_text(encoding="utf-8") == "a\nb!\n"

        without_nl = root / "without_nl.txt"
        without_nl.write_text("a\nb", encoding="utf-8")
        _call(replace_lines, path=_rel(without_nl), start_line=2, end_line=2, new_content="b!")
        assert without_nl.read_text(encoding="utf-8") == "a\nb!"


# —— 沙箱与命令行防护 ————————————————————————————————————————————————
def test_file_tools_refuse_paths_outside_workspace() -> None:
    """文件工具的越界路径一律被拒，且返回的是说明而不是异常。"""
    with _Sandbox() as root:
        outside = _rel(root) + "/../../escape.txt"
        assert "超出工作目录" in _call(write_file, path=outside, content="x")
        assert "超出工作目录" in _call(read_file, path=outside)
        assert not (Path.cwd().parent / "escape.txt").exists()


def test_check_path_escape_blocks_leaving_paths() -> None:
    """命令行里的绝对路径和 `..` 往上跳要拦下来。"""
    root = workspace_root()
    blocked = [
        "cat /etc/passwd",
        "ls ../..",
        "type C:\\Windows\\win.ini",
        "dir ..\\..",
    ]
    for cmd in blocked:
        assert check_path_escape(cmd, root) is not None, cmd


def test_check_path_escape_allows_paths_inside() -> None:
    """工作目录内部的路径、以及 `-` 开头的选项不能被误伤。

    `dir /s` 这条尤其要留着：`/` 在 Windows 上是选项前缀，把它当绝对路径
    拦掉的话，模型连列个目录都用不了。
    """
    root = workspace_root()
    allowed = [
        "python app.py",
        "python pkg/sub/deep.py",
        "./run.sh",
        "dir /s",
        "git status",
        # 先下后上、原地返回：走不出工作目录，不该拦
        "cat pkg/../notes.txt",
    ]
    for cmd in allowed:
        assert check_path_escape(cmd, root) is None, cmd


def test_run_cmd_refuses_escape_and_still_runs_inside() -> None:
    """run_cmd：越界命令不执行；工作目录内的普通命令照常跑。"""
    from tools.shell import check_single_command

    escape = _call(run_cmd, command="cat ../../etc/passwd")
    assert "工作目录" in escape, escape

    with _Sandbox() as root:
        (root / "hello.txt").write_text("hi\n", encoding="utf-8")
        # `echo` 在 cmd.exe 和 sh 里都有，是最不容易受平台影响的探针
        assert check_single_command("echo hi") is None
        out = _call(run_cmd, command="echo hi")
        assert "hi" in out, out


# —— 注册表 ————————————————————————————————————————————————
def test_new_tools_are_registered() -> None:
    """新工具必须进 FILE_TOOLS 和 build_tools()，否则模型根本看不到它们。"""
    names = {t.name for t in FILE_TOOLS}
    assert {"list_dir", "find_files", "read_file", "replace_lines"} <= names, names

    registered = {t.name for t in build_tools()}
    assert {"find_files", "replace_lines"} <= registered, registered


def main() -> int:
    tests = [
        test_list_dir_recurses_and_hides_ignored_dirs,
        test_list_dir_max_depth_one_is_the_old_behaviour,
        test_list_dir_distinguishes_empty_from_all_ignored,
        test_list_dir_missing_path_returns_message,
        test_find_files_matches_nested_and_skips_ignored,
        test_find_files_supports_substring_pattern,
        test_find_files_no_match_explains_itself,
        test_find_files_caps_results,
        test_read_file_numbers_lines_and_slices,
        test_read_file_byte_budget_points_at_next_chunk,
        test_read_file_reports_bad_range,
        test_replace_lines_replaces_inclusive_range,
        test_replace_lines_empty_content_deletes,
        test_replace_lines_keeps_the_original_encoding,
        test_replace_lines_rejects_bad_range_and_keeps_file,
        test_replace_lines_respects_trailing_newline,
        test_file_tools_refuse_paths_outside_workspace,
        test_check_path_escape_blocks_leaving_paths,
        test_check_path_escape_allows_paths_inside,
        test_run_cmd_refuses_escape_and_still_runs_inside,
        test_new_tools_are_registered,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"[失败] {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[出错] {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"[通过] {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""tests/env/test_01_02_lock_versions.py — ``scripts/env/01_02_lock_versions.py`` 的单元测试

覆盖 ``windows/W2.1_env_lock.md`` §6.3-1／-2／-4：

1. 生成块替换函数：给定含标记的样例文本，替换后**标记外内容逐字节不变**（含中文与既有手写节）；
2. 标记缺失时**不得静默重建文件**，须报错退出（非零）且不改文件；
4. ``requirements-locked.txt`` 生成器：包名排序、注释行保留。

约束：不访问 ``data/``；不加载模型；不依赖网络；不改变环境。
被测脚本以文件路径动态加载（文件名以数字开头，不能用 ``import`` 语句）。
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "env", "01_02_lock_versions.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("mod_01_02_lock_versions", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod_01_02_lock_versions"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_module()

TS = "2026-09-29 19:00:00"

# 样例文档：§0 里**逐字展示**了示意标记（真实 ENV_LOCK.md 亦如此），
# 用来回归「END 必须从 BEGIN 之后找」这一修正。
SAMPLE = (
    "# ENV_LOCK.md — 环境锁定记录\n"
    "\n"
    "## §0 说明与写入规则\n"
    "\n"
    "生成块范围（下列为示意，非真实标记）：\n"
    "\n"
    "```text\n"
    "<!-- BEGIN GENERATED: scripts/env/01_02_lock_versions.py @ <YYYY-MM-DD HH:MM:SS> -->\n"
    "…（§1 系统与算力）…\n"
    "<!-- END GENERATED: scripts/env/01_02_lock_versions.py -->\n"
    "```\n"
    "\n"
    "<!-- BEGIN GENERATED: scripts/env/01_02_lock_versions.py @ 2026-01-01 00:00:00 -->\n"
    "## §1 旧内容（应被替换）\n"
    "<!-- END GENERATED: scripts/env/01_02_lock_versions.py -->\n"
    "\n"
    "## §3 手写节（中文，须逐字节保留）\n"
    "\n"
    "| 项 | 值 |\n"
    "|---|---|\n"
    "| make | 4.4.1 |\n"
)

NEW_BODY = "## §1 系统与算力\n\n| 项 | 值 |\n|---|---|\n| OS | Windows |\n\n## §2 关键依赖精确版本\n"


def _span(text):
    begin, end_idx = M.find_block_span(text)
    return begin.start(), end_idx + len(M.END_MARKER)


# ------------------------------------------------- 1. 标记外逐字节不变
def test_replace_keeps_outside_bytes_identical():
    out = M.replace_generated_block(SAMPLE, NEW_BODY, TS)
    b0, e0 = _span(SAMPLE)
    b1, e1 = _span(out)

    head_before = SAMPLE[:b0].encode("utf-8")
    head_after = out[:b1].encode("utf-8")
    tail_before = SAMPLE[e0:].encode("utf-8")
    tail_after = out[e1:].encode("utf-8")

    assert head_after == head_before, "标记之前的内容被改动"
    assert tail_after == tail_before, "标记之后的内容被改动"
    assert "手写节（中文，须逐字节保留）" in out
    assert "make | 4.4.1" in out


def test_replace_swaps_only_block_body():
    out = M.replace_generated_block(SAMPLE, NEW_BODY, TS)
    assert "旧内容（应被替换）" not in out
    assert "## §2 关键依赖精确版本" in out


def test_replace_writes_timestamp_into_begin_marker():
    out = M.replace_generated_block(SAMPLE, NEW_BODY, TS)
    assert M.begin_marker_line(TS) in out
    assert "<!-- BEGIN GENERATED: scripts/env/01_02_lock_versions.py @ %s -->" % TS in out


def test_replace_is_idempotent():
    once = M.replace_generated_block(SAMPLE, NEW_BODY, TS)
    twice = M.replace_generated_block(once, NEW_BODY, TS)
    assert once == twice


def test_illustrative_markers_before_block_do_not_confuse_span():
    """回归：§0 的示意标记在前，仍须命中最靠后的真实块。"""
    begin, end_idx = M.find_block_span(SAMPLE)
    assert SAMPLE[: begin.start()].count(M.END_MARKER) == 1
    assert SAMPLE[end_idx:].startswith(M.END_MARKER)


# ------------------------------------------------- 2. 标记缺失的行为
def test_missing_begin_marker_raises():
    with pytest.raises(M.MarkerMissingError):
        M.replace_generated_block("# 无标记的文档\n## §1\n", NEW_BODY, TS)


def test_missing_end_marker_raises():
    text = "开头\n%s\n没有结束标记\n" % M.BEGIN_MARKER
    with pytest.raises(M.MarkerMissingError):
        M.replace_generated_block(text, NEW_BODY, TS)


def test_duplicate_begin_markers_raise():
    text = "%s\nx\n%s\ny\n%s\n" % (M.BEGIN_MARKER, M.BEGIN_MARKER, M.END_MARKER)
    with pytest.raises(M.MarkerMissingError):
        M.replace_generated_block(text, NEW_BODY, TS)


def test_main_missing_markers_exits_nonzero_and_leaves_file_untouched(tmp_path):
    env_lock = tmp_path / "ENV_LOCK.md"
    env_lock.write_text("# 没有标记\n## §1 手写内容\n", encoding="utf-8")
    before = env_lock.read_bytes()

    rc = M.main(
        [
            "--env-lock",
            str(env_lock),
            "--requirements",
            str(tmp_path / "requirements-locked.txt"),
            "--log",
            str(tmp_path / "log.log"),
        ]
    )

    assert rc != 0, "标记缺失必须返回非零"
    assert env_lock.read_bytes() == before, "标记缺失时不得改动文件（不得静默重建）"
    assert not (tmp_path / "requirements-locked.txt").exists()


def test_main_missing_env_lock_exits_nonzero(tmp_path):
    rc = M.main(
        [
            "--env-lock",
            str(tmp_path / "nope.md"),
            "--requirements",
            str(tmp_path / "req.txt"),
            "--log",
            str(tmp_path / "log.log"),
        ]
    )
    assert rc != 0


# ------------------------------------------------- 4. requirements-locked.txt
def test_requirements_sorts_packages_and_keeps_comments():
    prev = M.REQ_HEADER_HINT + "\n"
    freeze = [
        "zlib==1.0",
        "Alpha @ file:///x/alpha",
        "numpy==2.3.1",
        "-e git+https://h/r.git@abc#egg=beta_pkg",
        "Mid==3",
    ]
    text = M.build_requirements_locked(prev, freeze, TS)
    lines = text.splitlines()

    assert lines[0] == M.REQ_HEADER_HINT
    assert "# 快照时间：%s" % TS in lines
    assert (
        "# 本文件为 pip 侧快照；conda 侧见 ENV_LOCK.md §4；pip/conda 双装清单见 ENV_LOCK.md §3"
        in lines
    )

    pkg_lines = [ln for ln in lines if not ln.startswith("#")]
    names = [M.freeze_line_name(ln).lower() for ln in pkg_lines]
    assert names == ["alpha", "beta_pkg", "mid", "numpy", "zlib"]
    assert pkg_lines == sorted(pkg_lines, key=lambda ln: M.freeze_line_name(ln).lower())


def test_requirements_preserves_editable_and_url_lines_verbatim():
    freeze = ["-e git+https://h/r.git@abc#egg=beta_pkg", "numpy==2.3.1"]
    text = M.build_requirements_locked("", freeze, TS)
    assert "-e git+https://h/r.git@abc#egg=beta_pkg" in text
    assert "numpy==2.3.1" in text


def test_requirements_inserts_default_header_when_previous_missing():
    text = M.build_requirements_locked(None, ["numpy==2.3.1"], TS)
    assert text.splitlines()[0] == M.REQ_HEADER_HINT


def test_requirements_ends_with_single_newline():
    text = M.build_requirements_locked("", ["a==1"], TS)
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


@pytest.mark.parametrize(
    "line,expected",
    [
        ("numpy==2.3.1", "numpy"),
        ("huggingface_hub==1.27.0", "huggingface_hub"),
        ("numpy @ file:///x/y.whl#sha256=aa", "numpy"),
        ("-e git+https://h/r.git@abc#egg=beta_pkg", "beta_pkg"),
        ("--editable git+https://h/r.git@abc#egg=gamma", "gamma"),
    ],
)
def test_freeze_line_name_forms(line, expected):
    assert M.freeze_line_name(line) == expected


def test_top_comment_lines_stops_at_first_package():
    text = "# c1\n# c2\n\nnumpy==2.3.1\n# 这行不是顶部注释\n"
    assert M.top_comment_lines(text) == ["# c1", "# c2"]


# ------------------------------------------------- 其他
def test_module_uses_no_heavy_import_at_top_level():
    """生成器不应在顶部 import torch 等重型库。"""
    source = open(SCRIPT, "r", encoding="utf-8").read()
    head = source.split("def ", 1)[0]
    for heavy in ("import torch", "import numpy", "import transformers"):
        assert heavy not in head, "顶部出现重型导入：%s" % heavy


def test_docstring_states_no_commit_hash_boundary():
    """§4.6-2：边界须写入脚本 docstring。"""
    assert "不含模型 commit hash" in M.__doc__ or "不含模型 commit hash 节" in M.__doc__

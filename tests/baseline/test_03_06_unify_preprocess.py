# -*- coding: utf-8 -*-
"""tests/baseline/test_03_06_unify_preprocess.py — W4.6 统一预处理模块（§1 接口）。

覆盖 `windows/W4.6_unify.md` §1 的接口语义：

1. `Sample = (sample_id, text, label)` 三元组结构（namedtuple，可下标 + 可命名访问）
2. `load_samples`：原文直通（不 strip／lower／CR 归一）、`label` 解析为 `int`
3. `sha256_of`／`md5_of` 与 `hashlib` 参照逐字节一致
4. `truncate_left`：左截断（保尾部 512 token）、短于上限／`tokenizer=None`／`max_tokens=0` 原样返回
5. **硬约束 §0.2-7**：顶层不得 import 重型库（torch／transformers），模块可离线导入、无副作用输出
6. `--self-check` 子进程退出码 = 0（不读 sealed_test、不加载模型）

**离线**：全部输入在 `tmp_path` 内自造；`truncate_left` 用确定性伪 tokenizer，不加载任何真实模型。
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "baseline", "03_06_unify_preprocess.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w46_preprocess", SCRIPT)


class _MockTokenizer:
    """确定性伪 tokenizer：按空格切分，decode 返回 token id 串。仅测截断语义。"""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split(" "))))}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# ① Sample 三元组结构
# ---------------------------------------------------------------------------
def test_sample_is_three_tuple(mod):
    s = mod.Sample("S1", "hello", 1)
    assert tuple(s) == ("S1", "hello", 1)          # 可下标
    assert s.sample_id == "S1"                      # 可命名访问
    assert s.text == "hello"
    assert s.label == 1
    assert len(s) == 3


# ---------------------------------------------------------------------------
# ② load_samples：原文直通 + label 解析
# ---------------------------------------------------------------------------
def _write_csv(path, rows, fields):
    mod_ = None
    import csv
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_load_samples_passthrough_no_cleaning(mod, tmp_path):
    text = "  Hello\n  Base64: SGVsbG8=  "     # 前导/尾随空白 + 换行，须一字不改
    path = str(tmp_path / "samples.csv")
    _write_csv(path, [
        {"sample_id": "S1", "text": text, "label": "1"},
        {"sample_id": "S2", "text": "benign", "label": "0"},
    ], ["sample_id", "text", "label"])
    samples = mod.load_samples(path)
    assert len(samples) == 2
    s0, s1 = samples
    assert tuple(s0) == ("S1", text, 1)          # text 一字不改
    assert s0.text == text
    assert tuple(s1) == ("S2", "benign", 0)


def test_load_samples_label_parses_int(mod, tmp_path):
    path = str(tmp_path / "labels.csv")
    _write_csv(path, [
        {"sample_id": "A", "text": "x", "label": "1"},
        {"sample_id": "B", "text": "y", "label": "0.0"},
        {"sample_id": "C", "text": "z", "label": " 1 "},
    ], ["sample_id", "text", "label"])
    samples = mod.load_samples(path)
    assert [s.label for s in samples] == [1, 0, 1]


def test_load_samples_rejects_bad_label(mod, tmp_path):
    path = str(tmp_path / "bad.csv")
    _write_csv(path, [{"sample_id": "A", "text": "x", "label": "not-a-number"}],
               ["sample_id", "text", "label"])
    with pytest.raises(ValueError):
        mod.load_samples(path)


def test_load_samples_multiline_text_roundtrip(mod, tmp_path):
    text = "line1\nline2\nline3"
    path = str(tmp_path / "multi.csv")
    _write_csv(path, [{"sample_id": "M", "text": text, "label": "1"}],
               ["sample_id", "text", "label"])
    (s,) = mod.load_samples(path)
    assert s.text == text                            # 多行字段逐字节保留


def test_load_rows_returns_full_context(mod, tmp_path):
    path = str(tmp_path / "full.csv")
    _write_csv(path, [{"sample_id": "F", "text": "t", "payload": "p", "label": "1"}],
               ["sample_id", "text", "payload", "label"])
    rows = mod.load_rows(path)
    assert rows == [{"sample_id": "F", "text": "t", "payload": "p", "label": "1"}]


# ---------------------------------------------------------------------------
# ③ 完整性锚点
# ---------------------------------------------------------------------------
def test_sha256_of_matches_hashlib(mod):
    assert mod.sha256_of("锚点😀\n") == hashlib.sha256("锚点😀\n".encode("utf-8")).hexdigest()


def test_md5_of_matches_hashlib(mod, tmp_path):
    path = str(tmp_path / "anchor.bin")
    payload = "md5 锚点内容😀".encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(payload)
    assert mod.md5_of(path) == hashlib.md5(payload).hexdigest()


# ---------------------------------------------------------------------------
# ④ truncate_left：左截断（保尾部）
# ---------------------------------------------------------------------------
def test_truncate_left_keeps_tail_tokens(mod):
    mock = _MockTokenizer()
    text = "a b c d e"                              # 5 token
    out = mod.truncate_left(text, mock, max_tokens=3)
    assert out == "2 3 4"                           # 保尾部 3 个 token（id 2/3/4）


def test_truncate_left_short_input_unchanged(mod):
    mock = _MockTokenizer()
    text = "a b c"
    assert mod.truncate_left(text, mock, max_tokens=100) == text


def test_truncate_left_none_tokenizer_unchanged(mod):
    assert mod.truncate_left("a b c d e", None, max_tokens=3) == "a b c d e"


def test_truncate_left_zero_max_unchanged(mod):
    mock = _MockTokenizer()
    assert mod.truncate_left("a b c d e", mock, max_tokens=0) == "a b c d e"


def test_truncate_left_exact_limit_unchanged(mod):
    mock = _MockTokenizer()
    text = "a b c d e"                              # 5 token == 上限
    assert mod.truncate_left(text, mock, max_tokens=5) == text


# ---------------------------------------------------------------------------
# ⑤ 硬约束：顶层不 import 重型库、离线导入无副作用
# ---------------------------------------------------------------------------
def test_top_level_imports_only_stdlib(mod):
    tree = ast.parse(open(SCRIPT, "r", encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "argparse", "csv", "hashlib", "os", "sys", "collections"}


def test_import_does_not_load_heavy_libs(mod):
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_import_has_no_side_effect_output(mod):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        _load("w46_preprocess_again", SCRIPT)
    finally:
        sys.stdout = old
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# ⑥ CLI 自检退出码
# ---------------------------------------------------------------------------
def test_self_check_exits_zero():
    proc = subprocess.run([sys.executable, SCRIPT, "--self-check"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert "SELF_CHECK OK" in proc.stdout

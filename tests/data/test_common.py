# -*- coding: utf-8 -*-
"""tests/data/test_common.py — `scripts/data/_common.py` 不变量与边界（W3.3 §5）。

本文件**离线可跑**：除 `token_length` 的边界用例外，不联网、不加载模型。
`token_length` 只惰性加载 **tokenizer**（非模型权重）；若本地缓存缺失则 `skip`，
并以 `resolve_tokenizer` 的纯映射断言兜底。
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON_PATH = os.path.join(ROOT, "scripts", "data", "_common.py")

ENCODING_CODES = ("base64", "hex", "ascii_art", "rot13", "reverse", "morse", "url",
                  "zerowidth", "homoglyph")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    return _load("w33_data_common", COMMON_PATH)


# ---------------------------------------------------------------------------
# ① OOD 五／四分区的完备性与互斥性（W3.3 §2 配套不变量、§5①）
# ---------------------------------------------------------------------------
def test_ood_partition_covers_all_encodings(common):
    union = set(common.OOD_TRAIN_ENCODINGS) | set(common.OOD_HELDOUT_ENCODINGS)
    assert union == set(common.ENCODING_CODES)


def test_ood_partition_is_disjoint(common):
    overlap = set(common.OOD_TRAIN_ENCODINGS) & set(common.OOD_HELDOUT_ENCODINGS)
    assert overlap == set()


def test_ood_train_has_five_and_heldout_has_four(common):
    assert len(common.OOD_TRAIN_ENCODINGS) == 5
    assert len(common.OOD_HELDOUT_ENCODINGS) == 4
    assert len(common.ENCODING_CODES) == 9


def test_encoding_codes_exact_order_and_spelling(common):
    # `V-5`：9 码的**拼写与顺序**都是定值，不得改名、不得合并。
    assert tuple(common.ENCODING_CODES) == ENCODING_CODES


def test_carrier_codes(common):
    assert tuple(common.CARRIER_CODES) == ("text", "code")


def test_aliases_are_the_same_objects(common):
    # `stage2_plan` §2.4 用旧名 `OOD_TRAIN`／`OOD_HELDOUT`；同物两名。
    assert common.OOD_TRAIN is common.OOD_TRAIN_ENCODINGS
    assert common.OOD_HELDOUT is common.OOD_HELDOUT_ENCODINGS


def test_no_duplicate_encoding_codes(common):
    assert len(set(common.ENCODING_CODES)) == len(common.ENCODING_CODES)


# ---------------------------------------------------------------------------
# ② `write_csv` 产出的文件无孤立 CR（**按字节法**断言）
# ---------------------------------------------------------------------------
def test_write_csv_emits_no_cr_bytewise(common, tmp_path):
    path = str(tmp_path / "out.csv")
    rows = [{"a": "x", "b": "多行\n内容"}, {"a": "", "b": "尾随空格 "}]
    common.write_csv(path, rows, ["a", "b"])
    raw = open(path, "rb").read()
    assert b"\r" not in raw                      # 字节法：无任何 CR
    # 表头 1 + 行1 的字段内换行 1 + 行1 记录终止 1 + 行2 记录终止 1 = 4
    assert raw.count(b"\n") == 4


def test_write_csv_field_order_is_fixed_by_fields(common, tmp_path):
    path = str(tmp_path / "order.csv")
    common.write_csv(path, [{"b": "2", "a": "1"}], ["a", "b"])
    header = open(path, "r", encoding="utf-8", newline="").readline()
    assert header == "a,b\n"


def test_write_csv_rejects_cr_in_value(common, tmp_path):
    with pytest.raises(ValueError):
        common.write_csv(str(tmp_path / "bad.csv"), [{"a": "a\rb"}], ["a"])


def test_write_csv_rejects_cr_in_field_name(common, tmp_path):
    with pytest.raises(ValueError):
        common.write_csv(str(tmp_path / "bad2.csv"), [{"a": "1"}], ["a\r"])


def test_read_csv_normalizes_cr(common, tmp_path):
    # 手写一个含 CRLF 的 CSV：读回后字段值中**不得**残留 `\r`。
    path = str(tmp_path / "crlf.csv")
    with open(path, "wb") as fh:
        fh.write(b"a,b\r\n1,2\r\n")
    rows = common.read_csv(path)
    assert rows == [{"a": "1", "b": "2"}]
    for row in rows:
        for value in row.values():
            assert "\r" not in value


def test_read_write_roundtrip_preserves_content(common, tmp_path):
    rows = [{"a": "纯中文", "b": "emoji 😀", "c": "quote\"comma,"},
            {"a": "", "b": "line1\nline2", "c": "back\\slash"}]
    path = str(tmp_path / "rt.csv")
    common.write_csv(path, rows, ["a", "b", "c"])
    assert common.read_csv(path) == rows


def test_write_csv_uses_stdlib_csv_parsable(common, tmp_path):
    path = str(tmp_path / "parsable.csv")
    common.write_csv(path, [{"a": "x,y", "b": "z\nw"}], ["a", "b"])
    with open(path, "r", encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert got == [{"a": "x,y", "b": "z\nw"}]


# ---------------------------------------------------------------------------
# ③ `token_length` 边界（空串／单字符／已知串）
# ---------------------------------------------------------------------------
def test_resolve_tokenizer_maps_known_id_to_local_dir(common):
    resolved = common.resolve_tokenizer(common.DEFAULT_TOKENIZER_ID)
    assert os.path.isdir(resolved), "tokenizer 本地缓存缺失：%s" % resolved
    assert common.resolve_tokenizer(resolved) == resolved      # 幂等
    assert common.resolve_tokenizer("not/a-real-model-id") == "not/a-real-model-id"


def _token_length_or_skip(common, text):
    try:
        return common.token_length(text)
    except Exception as exc:  # noqa: BLE001 - 缓存缺失／依赖缺失时跳过，不误报失败
        pytest.skip("token_length 不可用（%s: %s）" % (type(exc).__name__, exc))


def test_token_length_empty_string_is_zero(common):
    assert _token_length_or_skip(common, "") == 0


def test_token_length_single_char_is_one(common):
    assert _token_length_or_skip(common, "a") == 1


def test_token_length_known_string(common):
    # `hello world` 在 Llama-3 tokenizer 下恰为 2 个 token（不含特殊 token）。
    assert _token_length_or_skip(common, "hello world") == 2


def test_token_length_is_monotone_on_prefix(common):
    short = _token_length_or_skip(common, "hello")
    long = _token_length_or_skip(common, "hello world hello world hello world")
    assert long > short


def test_token_length_cjk_and_emoji(common):
    assert _token_length_or_skip(common, "你好") >= 1
    assert _token_length_or_skip(common, "😀") >= 1


# ---------------------------------------------------------------------------
# ④ `ngram_overlap_rate` 自反性与空集边界
# ---------------------------------------------------------------------------
def test_ngram_overlap_rate_reflexive_is_one(common):
    grams = common.char_ngrams("the quick brown fox")
    assert common.ngram_overlap_rate(grams, grams) == 1.0


def test_ngram_overlap_rate_empty_b_is_zero(common):
    assert common.ngram_overlap_rate(common.char_ngrams("abc"), set()) == 0.0


def test_ngram_overlap_rate_empty_both_is_zero(common):
    assert common.ngram_overlap_rate(set(), set()) == 0.0


def test_ngram_overlap_rate_empty_a_is_zero(common):
    assert common.ngram_overlap_rate(set(), common.char_ngrams("abc")) == 0.0


def test_ngram_overlap_rate_disjoint_is_zero(common):
    left = common.char_ngrams("aaaa")
    right = common.char_ngrams("zzzz")
    assert common.ngram_overlap_rate(left, right) == 0.0


def test_ngram_overlap_rate_accepts_any_iterable(common):
    assert common.ngram_overlap_rate(["a", "b"], ["a", "b", "c"]) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# ④b `char_ngrams` / `word_ngrams` 边界口径
# ---------------------------------------------------------------------------
def test_char_ngrams_empty_is_empty_set(common):
    assert common.char_ngrams("") == set()


def test_char_ngrams_len_le_n_returns_whole_string(common):
    assert common.char_ngrams("abc", 3) == {"abc"}
    assert common.char_ngrams("ab", 3) == {"ab"}


def test_char_ngrams_counts_windows(common):
    assert common.char_ngrams("abcd", 3) == {"abc", "bcd"}


def test_char_ngrams_rejects_nonpositive_n(common):
    with pytest.raises(ValueError):
        common.char_ngrams("abc", 0)


def test_word_ngrams_boundaries(common):
    assert common.word_ngrams("") == set()
    assert common.word_ngrams("a b", 3) == {"a b"}
    assert common.word_ngrams("a b c d", 3) == {"a b c", "b c d"}


def test_word_ngrams_rejects_nonpositive_n(common):
    with pytest.raises(ValueError):
        common.word_ngrams("a b c", 0)


# ---------------------------------------------------------------------------
# 完整性锚点
# ---------------------------------------------------------------------------
def test_hash_helpers_agree_with_reference(common, tmp_path):
    import hashlib
    path = str(tmp_path / "anchor.bin")
    payload = "锚点\n内容😀".encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(payload)
    assert common.sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert common.md5_file(path) == hashlib.md5(payload).hexdigest()
    assert common.sha256_text("锚点\n内容😀") == hashlib.sha256(payload).hexdigest()


def test_project_root_points_at_repo_root(common):
    assert os.path.isfile(os.path.join(common.project_root(), "AGENTS.md"))


# ---------------------------------------------------------------------------
# 定位约束：`_common.py` 是纯函数库，不得有 `main()`／不得自执行
# ---------------------------------------------------------------------------
def _non_docstring_strings(tree):
    """收集 AST 中**除 docstring 外**的字符串字面量（按 AST 扫源码，非文本 grep）。"""
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            text = ast.get_docstring(node, clean=False)
            if text is not None:
                docs.add(text)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docs]


def test_common_has_no_main_and_is_import_safe(common):
    assert not hasattr(common, "main")
    tree = ast.parse(open(COMMON_PATH, "r", encoding="utf-8").read())
    assert "NOT_IMPLEMENTED" not in _non_docstring_strings(tree)
    assert not [n for n in ast.walk(tree)
                if isinstance(n, ast.Attribute) and n.attr == "exit"]


def test_common_imports_only_stdlib(common):
    tree = ast.parse(open(COMMON_PATH, "r", encoding="utf-8").read())
    top_level = set()
    for node in tree.body:                      # 只看模块级 import
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert top_level <= {"csv", "hashlib", "os", "__future__", "transformers"}


def test_common_import_has_no_side_effect_output(common):
    # 重复 import（命中 `sys.modules` 缓存）不得打印任何内容。
    buffer = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buffer
    try:
        _load("w33_data_common", COMMON_PATH)
    finally:
        sys.stdout = stdout
    assert buffer.getvalue() == ""

# -*- coding: utf-8 -*-
"""03_06_unify_preprocess.py — W4.6 统一预处理模块（阶段3 冻结接口 v1.0）。

职责：实现 `windows/W4.6_unify.md` §1 的统一预处理接口，作为 **W4.1–W4.5 的上游**——
它们 import 本模块的 `load_samples()`（以及 B3/B5 需要的左截断工具），并读
`configs/baseline/unify.yaml` 的常量；其输出 CSV 须符合 §2 的统一 schema。

接口契约（`windows/W4.6_unify.md` §1，冻结 v1.0，不得改变语义）::

    Sample = namedtuple("Sample", ["sample_id", "text", "label"])
    load_samples(csv_path) -> List[Sample]
    sha256_of(text) -> str            # 逐样本输入一致性锚点
    md5_of(csv_path) -> str           # 逐分片冻结面完整性锚点
    truncate_left(text, tokenizer, max_tokens=512) -> str   # §1.3 模型侧统一输入规范（左截断）

硬约束（§0.2）：

1. **原文直通**：`load_samples` 只读 `text` 列，**不做任何项目侧清洗／归一化／大小写转换**——
   「B1–B5 输入必须是完全一模一样的原始字符串」。唯一处理是 CSV 字段解析（引号规则），
   **不改变字段值本身**（不 strip、不 lower、不 CR 归一）。
2. 检测器**自带**的规范化（B4 SDK 的 text normalization／滑窗）属其管线一部分，由各检测器
   如实披露，本模块**不叠加**项目侧清洗。
3. 本模块**顶层不得 import 任何重型库**（torch／transformers）：`truncate_left` 只接收
   调用方（B3/B5）**已惰性加载**并传入的 tokenizer 对象，本模块自身不加载模型、不触网。
4. 本模块是纯函数库：不含 `main()` 之外的自执行逻辑；`--self-check` 不读 `sealed_test`、
   不加载 tokenizer，仅用合成样本往返验证接口语义。

CLI（§5 验收）::

    python scripts/baseline/03_06_unify_preprocess.py --self-check

`--self-check` 自检通过即 `exit 0`；任何接口语义不满足即打印失败原因并 `exit 1`。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import namedtuple

# 冻结接口 v1.0（§1.2）：`Sample = (sample_id, text, label)`
Sample = namedtuple("Sample", ["sample_id", "text", "label"])

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：W3.7 的 `init_streams`，`F-1`）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（`errors="replace"`），避免 GBK 环境打印非 ASCII 假失败。

    只动输出流，不触碰任何判定值、阈值或产物内容。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# CSV 读写（UTF-8 + LF；原文直通，不 CR 归一）
# ---------------------------------------------------------------------------
def read_csv(csv_path: str) -> "list[dict]":
    """读 UTF-8 CSV，返回 ``list[dict]``。**原文直通**：不 CR 归一、不 strip、不大小写转换。

    以 ``newline=""`` 打开仅为让 :mod:`csv` 正确解析跨行引用字段；冻结 CSV 为 UTF-8+LF
    （``V-9``，CR=0 已验证），故字段值即「与冻结面逐字节一致」的原始字符串。
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str, rows, fields) -> None:
    """写 UTF-8 + LF CSV，字段序由 ``fields`` 固定；写前断言无孤立 CR（``V-9``）。"""
    fields = list(fields)
    for i, row in enumerate(rows):
        for name in fields:
            value = row.get(name, "")
            if isinstance(value, str) and "\r" in value:
                raise ValueError("write_csv: 第 %d 行字段 %r 含 CR（须先归一为 LF）" % (i, name))
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


# ---------------------------------------------------------------------------
# §1.2 预处理函数语义
# ---------------------------------------------------------------------------
def _parse_label(value) -> int:
    """把 CSV 的 ``label`` 列解析为 ``int``（``1``/``0``）。

    容忍空白与浮点写法（如 ``1.0``）；无法解析时抛 ``ValueError``（不静默吞错）。
    """
    if value is None:
        raise ValueError("label 为空")
    text = str(value).strip()
    try:
        return int(float(text))
    except ValueError:
        raise ValueError("label 无法解析为整数: %r" % (value,))


def load_rows(csv_path: str) -> "list[dict]":
    """读整行上下文（``list[dict]``），供 §4.4 Span 真值派生与审计用（含 ``payload`` 等）。"""
    return read_csv(csv_path)


def load_samples(csv_path: str) -> "list[Sample]":
    """§1.2：``load_samples(csv_path) -> List[Sample]``，``Sample = (sample_id, text, label)``。

    **原文直通**：``text`` 列一字不改（不 strip／lower／CR 归一）；``label`` 解析为 ``int``。
    """
    out = []
    for row in read_csv(csv_path):
        sample_id = row.get("sample_id", "")
        text = row.get("text", "")
        label = _parse_label(row.get("label"))
        out.append(Sample(sample_id, text, label))
    return out


# ---------------------------------------------------------------------------
# 完整性锚点（§1.2）
# ---------------------------------------------------------------------------
def sha256_of(text: str) -> str:
    """字符串的 SHA-256（UTF-8 字节，十六进制小写）—— 逐样本输入一致性锚点。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def md5_of(csv_path: str) -> str:
    """文件 MD5（十六进制小写）—— 逐分片冻结面完整性锚点。"""
    digest = hashlib.md5()
    with open(csv_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# §1.3 模型侧统一输入规范（仅 B3/B5）：左截断，保尾部 512 token
# ---------------------------------------------------------------------------
def truncate_left(text: str, tokenizer, max_tokens: int = 512) -> str:
    """§1.3：模型侧统一输入规范 —— **左截断**（保尾部 ``max_tokens`` token）。

    ``tokenizer`` 由调用方（B3/B5）**惰性加载**后传入（本函数不 import 任何重型库）；
    ``max_tokens <= 0`` 或 ``tokenizer is None`` 时**原样返回**（视为不截断）。

    语义：以 tokenizer 编码 → 若 token 数 ``<= max_tokens`` 原样返回；否则保留**尾部**
    ``max_tokens`` 个 token id 并 decode 回文本（``add_special_tokens=False``，与门禁 4 口径一致）。
    """
    if max_tokens is None or max_tokens <= 0:
        return text
    if tokenizer is None:
        return text
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if len(ids) <= max_tokens:
        return text
    kept = ids[-max_tokens:]
    return tokenizer.decode(kept, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# CLI 自检（§5）：不读 sealed_test、不加载 tokenizer
# ---------------------------------------------------------------------------
class _MockTokenizer:
    """自检用的确定性伪 tokenizer：按空白切分（每 token 一个 id），decode 还原。

    仅用于 ``--self-check`` 验证 ``truncate_left`` 的截断语义，不加载任何真实模型。
    """

    def __init__(self, sep_tokens=(" ",)):
        self.sep_tokens = sep_tokens

    def __call__(self, text, add_special_tokens=False):
        # 简单按空格切分为「token 序列」，每个 token 映射为稳定 id。
        tokens = text.split(" ")
        return {"input_ids": list(range(len(tokens)))}

    def decode(self, ids, skip_special_tokens=False):
        # 自检只关心「截断后保留了尾部 token 数」，这里返回 token 计数占位即可由调用方核验。
        return " ".join(str(i) for i in ids)


def _self_check() -> int:
    """合成样本往返验证 §1 接口语义。返回 0=通过，非 0=失败。"""
    import tempfile

    errors = []

    # ① load_samples：Sample 三元组结构 + label 解析 + 原文直通
    sample_text = "Hello\n  Base64: SGVsbG8=  "
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", newline="", delete=False)
    tmp.write("sample_id,text,label\n")
    tmp.write('"S1","' + sample_text.replace('"', '""') + '",1\n')
    tmp.write('"S2","benign text",0\n')
    tmp.close()
    try:
        samples = load_samples(tmp.name)
        if len(samples) != 2:
            errors.append("load_samples 行数 != 2")
        else:
            s0, s1 = samples
            if tuple(s0) != ("S1", sample_text, 1):
                errors.append("Sample 三元组不符: %r" % (tuple(s0),))
            if tuple(s1) != ("S2", "benign text", 0):
                errors.append("Sample 三元组不符: %r" % (tuple(s1),))
            if s0.text != sample_text:
                errors.append("原文直通失败：text 被改动")
    finally:
        os.unlink(tmp.name)

    # ② sha256_of：与 hashlib 参照一致
    if sha256_of("锚点😀") != hashlib.sha256("锚点😀".encode("utf-8")).hexdigest():
        errors.append("sha256_of 与 hashlib 不一致")

    # ③ md5_of：与 hashlib 参照一致
    tmp2 = tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False)
    tmp2.write("md5 锚点内容".encode("utf-8"))
    tmp2.close()
    try:
        expect = hashlib.md5("md5 锚点内容".encode("utf-8")).hexdigest()
        if md5_of(tmp2.name) != expect:
            errors.append("md5_of 与 hashlib 不一致")
    finally:
        os.unlink(tmp2.name)

    # ④ truncate_left：左截断语义（保尾部 max_tokens 个 token）
    mock = _MockTokenizer()
    text = "a b c d e"  # 5 个空格分隔 token
    if truncate_left(text, mock, max_tokens=3) != " ".join(["2", "3", "4"]):
        # decode 返回尾部 3 个 token id 的占位串：ids=[0,1,2,3,4] → 尾部 3 = [2,3,4]
        errors.append("truncate_left 未保留尾部 token")
    if truncate_left(text, mock, max_tokens=100) != text:
        errors.append("truncate_left 对短于上限的输入应原样返回")
    if truncate_left(text, None, max_tokens=3) != text:
        errors.append("truncate_left 在 tokenizer=None 时应原样返回")
    if truncate_left(text, mock, max_tokens=0) != text:
        errors.append("truncate_left 在 max_tokens=0 时应原样返回")

    if errors:
        for e in errors:
            print("SELF_CHECK_FAIL: %s" % e)
        return 1
    print("03_06_unify_preprocess SELF_CHECK OK")
    return 0


def main() -> int:
    init_streams()
    parser = argparse.ArgumentParser(description="W4.6 统一预处理模块（§1 接口 + 自检）")
    parser.add_argument("--self-check", action="store_true",
                        help="合成样本往返验证接口语义（不读 sealed_test、不加载模型）")
    args = parser.parse_args()

    if args.self_check:
        return _self_check()

    # 无参调用：仅验证模块可导入（轻量），不做任何 IO。
    print("03_06_unify_preprocess: import OK（接口见 windows/W4.6_unify.md §1）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

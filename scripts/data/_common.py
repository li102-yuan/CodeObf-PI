# -*- coding: utf-8 -*-
"""scripts/data/_common.py — 阶段2 数据管道共享纯函数库（**唯一出处**）。

定位（`windows/W3.3_encode.md` §2；`reports/data/stage2_plan_v1.0_20260930-184841.md` §2.4）：

- **纯函数库**：不含 ``main()``、不打印 ``NOT_IMPLEMENTED``、**不被** ``make check``／``make smoke``
  直接调用（``Makefile`` 的 ``check`` 目标只探 ``02_07``）。
- 向 W3.3／W3.4／W3.5／W3.6／W3.7 提供**唯一一份**实现，避免多份实现漂移。
- **签名一经落盘即冻结；只增不改。**确需扩展时**只追加函数**，并报 W3 与写 ``DEVIATION_LEDGER.md``。

本文件**不加载任何模型**。``token_length`` 惰性加载 tokenizer（tokenizer ≠ 模型，
且仅在调用时才 import ``transformers``），使编码管道本身保持零依赖、零联网。
"""
from __future__ import annotations

import csv
import hashlib
import os

__all__ = [
    # 常量表（唯一出处）
    "ENCODING_CODES",
    "CARRIER_CODES",
    "OOD_TRAIN_ENCODINGS",
    "OOD_HELDOUT_ENCODINGS",
    # 兼容别名（`stage2_plan` §2.4 的旧名，同物两名）
    "OOD_TRAIN",
    "OOD_HELDOUT",
    # 函数
    "read_csv",
    "write_csv",
    "md5_file",
    "sha256_file",
    "sha256_text",
    "token_length",
    "char_ngrams",
    "word_ngrams",
    "ngram_overlap_rate",
    "project_root",
    "resolve_tokenizer",
]

# ---------------------------------------------------------------------------
# 常量表（`windows/W3.3_encode.md` §2；`V-5`／`V-6`）
# ---------------------------------------------------------------------------

#: `V-5` 9 种编码代码。**不得改名、不得合并。**
ENCODING_CODES = (
    "base64",
    "hex",
    "ascii_art",
    "rot13",
    "reverse",
    "morse",
    "url",
    "zerowidth",
    "homoglyph",
)

#: `V-5` 载体代码。
CARRIER_CODES = ("text", "code")

#: `V-6` 体制二（OOD）训练 5 编码。
OOD_TRAIN_ENCODINGS = ("base64", "rot13", "zerowidth", "url", "reverse")

#: `V-6` 体制二（OOD）留出 4 编码。
OOD_HELDOUT_ENCODINGS = ("hex", "morse", "ascii_art", "homoglyph")

#: 兼容别名 —— `stage2_plan` §2.4 把这两个常量写作 ``OOD_TRAIN``／``OOD_HELDOUT``。
#: 同物两名（**同一 tuple 对象**），供按旧名书写的读者使用；**只增不改**。
OOD_TRAIN = OOD_TRAIN_ENCODINGS
OOD_HELDOUT = OOD_HELDOUT_ENCODINGS


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def project_root() -> str:
    """返回项目根目录（本文件位于 ``<root>/scripts/data/_common.py``）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# CSV 读写：强制 UTF-8 + LF + 固定字段序（`V-9`）
# ---------------------------------------------------------------------------
def _lf_only(value: str) -> str:
    """把 ``\\r\\n`` 与孤立 ``\\r`` 一律归一为 ``\\n``。"""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def read_csv(path: str) -> "list[dict]":
    """读 UTF-8 CSV，返回 ``list[dict]``。

    实现口径：以 ``newline=''`` 打开**仅为**让 :mod:`csv` 正确解析跨行引用字段；
    随后**逐值归一 CR**（``\\r\\n``／孤立 ``\\r`` → ``\\n``），
    故返回值中**不可能**残留 ``\\r`` —— 这是「禁止 ``newline=''`` 造成 ``\\r\\n`` 残留」的落地方式。
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = []
        for raw in reader:
            row = {}
            for key in fields:
                value = raw.get(key)
                row[key] = _lf_only(value) if isinstance(value, str) else value
            rows.append(row)
    return rows


def _assert_no_cr(rows, fields) -> None:
    """写前断言：字段名与字段值均不含孤立 CR。"""
    for name in fields:
        if "\r" in str(name):
            raise ValueError("write_csv: 字段名含 CR: %r" % (name,))
    for i, row in enumerate(rows):
        for name in fields:
            value = row.get(name, "")
            if isinstance(value, str) and "\r" in value:
                raise ValueError(
                    "write_csv: 第 %d 行的字段 %r 含 CR（须先归一为 LF）" % (i, name)
                )


def write_csv(path: str, rows, fields) -> None:
    """写 UTF-8 + LF CSV，字段序由 ``fields`` 固定。**写前断言无孤立 CR。**

    ``lineterminator='\\n'`` 是必需的：:mod:`csv` 的默认终止符是 ``\\r\\n``，
    只设 ``newline='\\n'`` 不足以得到 LF 行尾。
    """
    fields = list(fields)
    _assert_no_cr(rows, fields)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


# ---------------------------------------------------------------------------
# 完整性锚点
# ---------------------------------------------------------------------------
def _hash_file(path: str, algo: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: str) -> str:
    """文件 MD5（十六进制小写）。"""
    return _hash_file(path, "md5")


def sha256_file(path: str) -> str:
    """文件 SHA-256（十六进制小写）。"""
    return _hash_file(path, "sha256")


def sha256_text(text: str) -> str:
    """字符串的 SHA-256（UTF-8 字节，十六进制小写）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# token 长度（门禁 6 的**唯一**口径实现）
# ---------------------------------------------------------------------------
#: 主检测器 tokenizer 的模型 id（`stage2_plan` §3「门禁 6／7 的度量口径」）。
DEFAULT_TOKENIZER_ID = "meta-llama/Llama-Guard-3-1B"

#: 已知 id → `models/hf_cache/` 下的本地目录名（离线解析用；`ENV_LOCK.md` §6 的落盘路径）。
_TOKENIZER_DIR_NAMES = {
    "meta-llama/llama-guard-3-1b": "modelscope--LLM-Research--Llama-Guard-3-1B",
    "llamaguard3-1b": "modelscope--LLM-Research--Llama-Guard-3-1B",
    "llama-guard-3-1b": "modelscope--LLM-Research--Llama-Guard-3-1B",
    "meta-llama/llama-guard-3-8b": "modelscope--LLM-Research--Llama-Guard-3-8B",
    "llamaguard3-8b": "modelscope--LLM-Research--Llama-Guard-3-8B",
    "llama-guard-3-8b": "modelscope--LLM-Research--Llama-Guard-3-8B",
}

_TOKENIZER_CACHE: "dict[str, object]" = {}


def resolve_tokenizer(tokenizer_id: str = DEFAULT_TOKENIZER_ID) -> str:
    """把 ``tokenizer_id`` 解析为**本地路径**。

    ① 若它本身是一个已存在的目录 → 原样返回；
    ② 若命中 ``_TOKENIZER_DIR_NAMES``（大小写不敏感）→ 返回 ``models/hf_cache/<dir>``；
    ③ 否则原样返回（交由 ``transformers`` 自行解析）。
    """
    if os.path.isdir(tokenizer_id):
        return tokenizer_id
    key = str(tokenizer_id).strip().lower()
    dir_name = _TOKENIZER_DIR_NAMES.get(key)
    if dir_name:
        candidate = os.path.join(project_root(), "models", "hf_cache", dir_name)
        if os.path.isdir(candidate):
            return candidate
    return tokenizer_id


def _load_tokenizer(tokenizer_id: str):
    path = resolve_tokenizer(tokenizer_id)
    if path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[path]
    try:
        from transformers import AutoTokenizer  # 惰性 import：编码管道本身不需要它
    except ImportError as exc:  # pragma: no cover - 环境缺包时的显式报错
        raise RuntimeError("token_length 需要 transformers：%s" % exc) from exc
    kwargs = {"local_files_only": True} if os.path.isdir(path) else {}
    tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
    _TOKENIZER_CACHE[path] = tokenizer
    return tokenizer


def token_length(text: str, tokenizer_id: str = DEFAULT_TOKENIZER_ID) -> int:
    """返回 ``text`` 的 **token 数**（不含特殊 token）。

    这是全项目 token 长度口径的**唯一实现**（门禁 6 用它）。
    首次调用会惰性加载 tokenizer；本函数自身**不加载任何模型权重**。
    """
    tokenizer = _load_tokenizer(tokenizer_id)
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


# ---------------------------------------------------------------------------
# n-gram 与重叠率（门禁 7 的**唯一**口径实现）
# ---------------------------------------------------------------------------
def char_ngrams(text: str, n: int = 3) -> "set[str]":
    """字符级 n-gram 集合。

    边界口径：空串 → 空集；``len(text) <= n`` → ``{text}``（整串作为一个元素，
    使「同串对比」仍能给出 1.0）。
    """
    if n <= 0:
        raise ValueError("n 必须 >= 1，收到 %r" % (n,))
    text = "" if text is None else str(text)
    if text == "":
        return set()
    if len(text) <= n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def word_ngrams(text: str, n: int = 3) -> "set[str]":
    """词级 n-gram 集合（**仅作对照，不参与判定**）。

    分词口径：按任意空白切分（``str.split()``）。
    """
    if n <= 0:
        raise ValueError("n 必须 >= 1，收到 %r" % (n,))
    words = ("" if text is None else str(text)).split()
    if not words:
        return set()
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap_rate(a_set, b_set) -> float:
    """``|A ∩ B| / |B|``。``B`` 为空集时返回 ``0.0``（不抛除零错）。"""
    left = set(a_set)
    right = set(b_set)
    if not right:
        return 0.0
    return len(left & right) / float(len(right))

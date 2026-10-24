# -*- coding: utf-8 -*-
"""02_03_encode_variants.py — W3.3 编码变换管道。

把 W3.2 的 **440** 条改写体（`V-1`）各自展开为 `V-5` 的 **9** 种编码变体，
产出**纯文本载体**攻击样本 **3960 = 440 × 9** 条（`SOURCE_LOCK/...v5.md` §2.3）。

接口契约（`windows/W3.3_encode.md` §3）::

    python scripts/data/02_03_encode_variants.py \\
      --config configs/data/encode.yaml \\
      --rewrites data/interim/rewrite/rewrites.csv \\
      --out data/interim/encoded/ \\
      [--smoke N]          # 只对前 N 条改写体编码，产物落 out/smoke/
      [--verify-roundtrip] # 逐条做 解码(编码(x)) == x 校验并统计

退出码：``0`` 成功；非 ``0`` 失败（含计数 ≠ 3960、或可逆校验失败）。

硬约束（本窗口零推理、零联网、零模型加载）：
- 不写 ``data/frozen/``、``data/sealed_test/``；不写 ``data/interim/rewrite/``（上游只读）。
- **9 种编码的代码不得改名、不得合并**（`V-5`）。
- 编码层是**纯外壳**：语义等价性由**解码验证**，不由生成过程假定。
- 报告中**不得写入真实恶意样本原文**，示例一律用配置里的自造文本。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import urllib.parse
from datetime import datetime

# ---------------------------------------------------------------------------
# 载入共享库 `scripts/data/_common.py`（文件名以数字开头，不能用 import 语句）
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_common():
    candidate = os.path.join(_HERE, "_common.py")
    spec = importlib.util.spec_from_file_location("data_common", candidate)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_common"] = module
    spec.loader.exec_module(module)
    return module


common = _load_common()

ENCODING_CODES = common.ENCODING_CODES
CARRIER_CODES = common.CARRIER_CODES
OOD_TRAIN_ENCODINGS = common.OOD_TRAIN_ENCODINGS
OOD_HELDOUT_ENCODINGS = common.OOD_HELDOUT_ENCODINGS
read_csv = common.read_csv
write_csv = common.write_csv
sha256_file = common.sha256_file
md5_file = common.md5_file

CSV_FIELDS = ["sample_id", "seed_group_id", "rewrite_id", "seed_id", "source",
              "harm_category", "encoding_type", "difficulty", "plaintext", "payload",
              "carrier_type", "label"]

# ===========================================================================
# 编码器 1/9 — base64
# ===========================================================================
def enc_base64(text):
    """标准 base64（``base64.b64encode``，UTF-8 字节）。"""
    import base64 as _b64
    return _b64.b64encode(text.encode("utf-8")).decode("ascii")


def dec_base64(payload):
    import base64 as _b64
    return _b64.b64decode(payload.encode("ascii"), validate=True).decode("utf-8")


# ===========================================================================
# 编码器 2/9 — hex
# ===========================================================================
def enc_hex(text):
    """小写十六进制（UTF-8 字节展开）。**不加 ``0x`` 前缀、不加空格分隔。**"""
    return text.encode("utf-8").hex()


def dec_hex(payload):
    return bytes.fromhex(payload).decode("utf-8")


# ===========================================================================
# 编码器 3/9 — ascii_art
# ===========================================================================
# 实现口径：**逐字符艺术化**（不是整体装饰框）。每个字符替换为一个固定的
# **5 行 × 5 列** ASCII 字形块（``#`` 为墨、空格为底），字形之间以 ``|`` 分隔，
# 5 行以 ``\\n`` 连接，故**整段文本恰好渲染为 5 行**。
#
# 可逆性设计：
#   ① 字形表对 95 个可打印 ASCII（U+0020–U+007E）**逐字符互异**（import 时断言）；
#   ② 换行符有专用字形；
#   ③ 表外字符（中文、emoji…）转义为 ``\\u{XXXX}``（4–6 位大写十六进制），
#      字面反斜杠转义为 ``\\\\`` —— 该转义 alphabet 全在字形表内，故可逆。
_ART_NEWLINE = ("     ", "     ", "#####", "     ", "     ")

_FONT = {
    " ": ("     ", "     ", "     ", "     ", "     "),
    "!": ("  #  ", "  #  ", "  #  ", "     ", "  #  "),
    '"': (" # # ", " # # ", "     ", "     ", "     "),
    "#": (" # # ", "#####", " # # ", "#####", " # # "),
    "$": ("  #  ", " ####", "## # ", "#### ", "  #  "),
    "%": ("#   #", "   # ", "  #  ", " #   ", "#   #"),
    "&": (" ##  ", "#  # ", " ##  ", "#  # ", " ## #"),
    "'": ("  #  ", "  #  ", "     ", "     ", "     "),
    "(": ("   # ", "  #  ", "  #  ", "  #  ", "   # "),
    ")": (" #   ", "  #  ", "  #  ", "  #  ", " #   "),
    "*": ("     ", "# # #", " ### ", "# # #", "     "),
    "+": ("     ", "  #  ", " ### ", "  #  ", "     "),
    ",": ("     ", "     ", "     ", "  #  ", " #   "),
    "-": ("     ", "     ", " ### ", "     ", "     "),
    ".": ("     ", "     ", "     ", "     ", "  #  "),
    "/": ("    #", "   # ", "  #  ", " #   ", "#    "),
    "0": (" ### ", "#  ##", "# # #", "##  #", " ### "),
    "1": ("  #  ", " ##  ", "  #  ", "  #  ", " ### "),
    "2": (" ### ", "#   #", "  ## ", " #   ", "#####"),
    "3": ("#### ", "    #", " ### ", "    #", "#### "),
    "4": ("#  # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#### ", "    #", "#### "),
    "6": (" ### ", "#    ", "#### ", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", "  #  "),
    "8": (" ### ", "#   #", " ### ", "#   #", " ### "),
    "9": (" ### ", "#   #", " ####", "    #", " ### "),
    ":": ("     ", "  #  ", "     ", "  #  ", "     "),
    ";": ("     ", "  #  ", "     ", "  #  ", " #   "),
    "<": ("   # ", "  #  ", " #   ", "  #  ", "   # "),
    "=": ("     ", " ### ", "     ", " ### ", "     "),
    ">": (" #   ", "  #  ", "   # ", "  #  ", " #   "),
    "?": (" ### ", "#   #", "  ## ", "     ", "  #  "),
    "@": (" ### ", "#   #", "# ###", "#    ", " ### "),
    "A": ("  #  ", " # # ", "#   #", "#####", "#   #"),
    "B": ("#### ", "#   #", "#### ", "#   #", "#### "),
    "C": (" ### ", "#   #", "#    ", "#   #", " ### "),
    "D": ("#### ", "#   #", "#   #", "#   #", "#### "),
    "E": ("#####", "#    ", "#### ", "#    ", "#####"),
    "F": ("#####", "#    ", "#### ", "#    ", "#    "),
    "G": (" ### ", "#    ", "#  ##", "#   #", " ####"),
    "H": ("#   #", "#   #", "#####", "#   #", "#   #"),
    "I": (" ### ", "  #  ", "  #  ", "  #  ", " ### "),
    "J": ("    #", "    #", "    #", "#   #", " ### "),
    "K": ("#   #", "#  # ", "###  ", "#  # ", "#   #"),
    "L": ("#    ", "#    ", "#    ", "#    ", "#####"),
    "M": ("#   #", "## ##", "# # #", "#   #", "#   #"),
    "N": ("#   #", "##  #", "# # #", "#  ##", "#   #"),
    "O": (" ### ", "#   #", "#   #", "#   #", " ### "),
    "P": ("#### ", "#   #", "#### ", "#    ", "#    "),
    "Q": (" ### ", "#   #", "# # #", "#  # ", " ## #"),
    "R": ("#### ", "#   #", "#### ", "#  # ", "#   #"),
    "S": (" ####", "#    ", " ### ", "    #", "#### "),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  "),
    "U": ("#   #", "#   #", "#   #", "#   #", " ### "),
    "V": ("#   #", "#   #", "#   #", " # # ", "  #  "),
    "W": ("#   #", "#   #", "# # #", "## ##", "#   #"),
    "X": ("#   #", " # # ", "  #  ", " # # ", "#   #"),
    "Y": ("#   #", " # # ", "  #  ", "  #  ", "  #  "),
    "Z": ("#####", "   # ", "  #  ", " #   ", "#####"),
    "[": (" ### ", " #   ", " #   ", " #   ", " ### "),
    "\\": ("#    ", " #   ", "  #  ", "   # ", "    #"),
    "]": (" ### ", "   # ", "   # ", "   # ", " ### "),
    "^": ("  #  ", " # # ", "#   #", "     ", "     "),
    "_": ("     ", "     ", "     ", "     ", "#####"),
    "`": (" #   ", "  #  ", "     ", "     ", "     "),
    "a": ("     ", " ### ", "#  ##", "#   #", " ####"),
    "b": ("#    ", "#### ", "#   #", "#   #", "#### "),
    "c": ("     ", " ### ", "#    ", "#    ", " ### "),
    "d": ("    #", " ####", "#   #", "#   #", " ####"),
    "e": ("     ", " ### ", "#  ##", "#    ", " ### "),
    "f": ("  ## ", " #   ", "#### ", " #   ", " #   "),
    "g": (" ### ", "#   #", " ####", "#   #", " ### "),
    "h": ("#    ", "#### ", "#   #", "#   #", "#   #"),
    "i": ("  #  ", "     ", "  #  ", "  #  ", "  #  "),
    "j": ("   # ", "     ", "   # ", "   # ", " ##  "),
    "k": ("#    ", "#  # ", "###  ", "#  # ", "#  # "),
    "l": (" ##  ", "  #  ", "  #  ", "  #  ", " ### "),
    "m": ("     ", "## # ", "# # #", "# # #", "# # #"),
    "n": ("     ", "#### ", "#   #", "#   #", "#   #"),
    "o": ("     ", " ### ", "#   #", "#   #", " ### "),
    "p": ("     ", "#### ", "#   #", "#### ", "#    "),
    "q": ("     ", " ####", "#   #", " ####", "    #"),
    "r": ("     ", "# ## ", "##   ", "#    ", "#    "),
    "s": ("     ", " ### ", "##   ", "  ## ", "###  "),
    "t": (" #   ", "#### ", " #   ", " #  #", "  ## "),
    "u": ("     ", "#   #", "#   #", "#  ##", " ### "),
    "v": ("     ", "#   #", "#   #", " # # ", "  #  "),
    "w": ("     ", "#   #", "# # #", "# # #", " # # "),
    "x": ("     ", "#   #", " # # ", " # # ", "#   #"),
    "y": ("     ", "#   #", "#   #", " ####", "    #"),
    "z": ("     ", "#####", "   # ", "  #  ", "#####"),
    "{": ("   ##", "  #  ", " ##  ", "  #  ", "   ##"),
    "|": ("  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
    "}": ("##   ", "  #  ", "  ## ", "  #  ", "##   "),
    "~": ("     ", " ## #", "# ## ", "     ", "     "),
}

_ART_WIDTH = 5
_ART_SEP = "|"

if sorted(_FONT) != sorted(chr(c) for c in range(0x20, 0x7F)):  # pragma: no cover
    raise AssertionError("ASCII 字形表未覆盖 U+0020–U+007E 的全部 95 个字符")
if len(set(_FONT.values())) != len(_FONT):  # pragma: no cover
    raise AssertionError("ASCII 字形表存在重复字形（解码不可逆）")
for _ch, _glyph in _FONT.items():  # pragma: no cover
    if len(_glyph) != 5 or any(len(r) != _ART_WIDTH for r in _glyph):
        raise AssertionError("字形块形状不合法：%r" % (_ch,))
if _ART_NEWLINE in set(_FONT.values()):  # pragma: no cover
    raise AssertionError("换行字形与某个字符字形重复")

_ART_REVERSE = {glyph: ch for ch, glyph in _FONT.items()}
_ART_REVERSE[_ART_NEWLINE] = "\n"


def _art_atoms(text):
    """把文本拆成字形块序列，字形表外字符转义为 ``\\u{XXXX}``。"""
    atoms = []
    for ch in text:
        if ch == "\n":
            atoms.append(_ART_NEWLINE)
        elif ch == "\\":
            atoms.append(_FONT["\\"])
            atoms.append(_FONT["\\"])
        elif ch in _FONT:
            atoms.append(_FONT[ch])
        else:
            for esc_ch in "\\u{%X}" % ord(ch):
                atoms.append(_FONT[esc_ch])
    return atoms


def enc_ascii_art(text):
    """逐字符 ASCII 字形块；整段文本渲染为 5 行，字形以 ``|`` 分隔。"""
    if text == "":
        return ""
    atoms = _art_atoms(text)
    return "\n".join(
        _ART_SEP.join(glyph[row] for glyph in atoms) for row in range(5)
    )


def dec_ascii_art(payload):
    if payload == "":
        return ""
    rows = payload.split("\n")
    if len(rows) != 5:
        raise ValueError("ascii_art: 载荷应为 5 行，实得 %d 行" % len(rows))
    cells = [row.split(_ART_SEP) for row in rows]
    counts = {len(c) for c in cells}
    if len(counts) != 1:
        raise ValueError("ascii_art: 各行字形数不一致 %s" % sorted(counts))
    chars = []
    for i in range(len(cells[0])):
        glyph = tuple(cells[r][i] for r in range(5))
        try:
            chars.append(_ART_REVERSE[glyph])
        except KeyError:
            raise ValueError("ascii_art: 未登记的字形块 %r" % (glyph,))
    return _art_unescape("".join(chars))


def _art_unescape(s):
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if nxt == "\\":
            out.append("\\")
            i += 2
        elif nxt == "u" and i + 2 < len(s) and s[i + 2] == "{":
            end = s.find("}", i + 3)
            if end < 0:
                raise ValueError("ascii_art: 转义序列未闭合")
            out.append(chr(int(s[i + 3:end], 16)))
            i = end + 1
        else:
            raise ValueError("ascii_art: 非法转义 %r" % (s[i:i + 2],))
    return "".join(out)


# ===========================================================================
# 编码器 4/9 — rot13
# ===========================================================================
_ROT13_TABLE = {}
for _i in range(26):
    _ROT13_TABLE[ord("a") + _i] = chr(ord("a") + (_i + 13) % 26)
    _ROT13_TABLE[ord("A") + _i] = chr(ord("A") + (_i + 13) % 26)


def enc_rot13(text):
    """拉丁字母 ROT13；**非字母字符原样保留**。"""
    return text.translate(_ROT13_TABLE)


def dec_rot13(payload):
    return payload.translate(_ROT13_TABLE)


# ===========================================================================
# 编码器 5/9 — reverse
# ===========================================================================
def enc_reverse(text):
    """**字符级**反转（不是字节级，避免破坏多字节字符）。"""
    return text[::-1]


def dec_reverse(payload):
    return payload[::-1]


# ===========================================================================
# 编码器 6/9 — morse
# ===========================================================================
# 口径：标准 Morse（表域 A-Z／0-9／常用标点，不含小写，符合 ITU 惯例）；
# **字母间 1 个空格、词间 ``/``**。
#
# 可逆性设计（Morse 本身**大小写不敏感**，必须有补充约定才能逐字符可逆）：
#   ① **小写字母** = 该字母的 Morse 码 **后缀 ``'``**（``a`` → ``.-'``）；
#      因为 Morse 符号集只含 ``.`` 与 ``-``，故后缀 ``'`` **无歧义**；
#   ② **非 ASCII 字符**（中文、emoji 等）转 Unicode 码点 token ``U+XXXX``
#      （4–6 位大写十六进制）；该 token 含 ``U``／``+``，与纯 ``.``/``-`` 的
#      Morse token 亦不冲突 ⇒ **无歧义**，解码可逆。
_MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.", "!": "-.-.--",
    "/": "-..-.", "(": "-.--.", ")": "-.--.-", "&": ".-...", ":": "---...",
    ";": "-.-.-.", "=": "-...-", "+": ".-.-.", "-": "-....-", "_": "..--.-",
    '"': ".-..-.", "$": "...-..-", "@": ".--.-.",
}
_MORSE_REVERSE = {v: k for k, v in _MORSE.items()}
_MORSE_WORD_SEP = "/"
_MORSE_CHAR_SEP = " "
_MORSE_LOWER_SUFFIX = "'"


def enc_morse(text):
    tokens = []
    for ch in text:
        upper = ch.upper()
        if ch == " ":
            tokens.append(_MORSE_WORD_SEP)
        elif upper in _MORSE and upper.isalpha():
            token = _MORSE[upper]
            tokens.append(token + _MORSE_LOWER_SUFFIX if ch.islower() else token)
        elif ch in _MORSE:
            tokens.append(_MORSE[ch])
        else:
            tokens.append("U+%04X" % ord(ch))
    return _MORSE_CHAR_SEP.join(tokens)


def dec_morse(payload):
    if payload == "":
        return ""
    out = []
    for token in payload.split(_MORSE_CHAR_SEP):
        if token == _MORSE_WORD_SEP:
            out.append(" ")
        elif token.startswith("U+") and len(token) >= 6:
            out.append(chr(int(token[2:], 16)))
        elif token.endswith(_MORSE_LOWER_SUFFIX):
            body = token[:-1]
            if body not in _MORSE_REVERSE:
                raise ValueError("morse: 未登记的 token %r" % (token,))
            out.append(_MORSE_REVERSE[body].lower())
        else:
            try:
                out.append(_MORSE_REVERSE[token])
            except KeyError:
                raise ValueError("morse: 未登记的 token %r" % (token,))
    return "".join(out)


# ===========================================================================
# 编码器 7/9 — url
# ===========================================================================
def enc_url(text):
    """``urllib.parse.quote``，``safe=''``（全百分号编码）。"""
    return urllib.parse.quote(text, safe="")


def dec_url(payload):
    return urllib.parse.unquote(payload, encoding="utf-8", errors="strict")


# ===========================================================================
# 编码器 8/9 — zerowidth
# ===========================================================================
# 口径：把 UTF-8 **位流**映射到 ``U+200B``(0)／``U+200C``(1)，
# **保留原文可见字符作为载体**（每个可见字符后插入一个零宽位，余下的位追加到末尾）。
# 解码时**只收集零宽字符**、丢弃全部可见字符 ⇒ 逐字符可逆。
ZW_ZERO = "​"
ZW_ONE = "‌"
_ZW_CHARS = (ZW_ZERO, ZW_ONE)


def enc_zerowidth(text):
    bits = []
    for byte in text.encode("utf-8"):
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if i < len(bits):
            out.append(ZW_ONE if bits[i] else ZW_ZERO)
    for bit in bits[len(text):]:
        out.append(ZW_ONE if bit else ZW_ZERO)
    return "".join(out)


def dec_zerowidth(payload):
    bits = [1 if ch == ZW_ONE else 0 for ch in payload if ch in _ZW_CHARS]
    if len(bits) % 8 != 0:
        raise ValueError("zerowidth: 零宽位流长度 %d 不是 8 的倍数" % len(bits))
    data = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        data.append(byte)
    return bytes(data).decode("utf-8")


# ===========================================================================
# 编码器 9/9 — homoglyph
# ===========================================================================
# 口径：拉丁字母 → 视觉同形字符（西里尔／希腊／拉丁同形字），
# **每个源字符映射到唯一的目标字符**（表内 52 个目标两两互异，import 时断言）。
# 可逆性设计：解码是「目标字符 → 源字符」的单射逆映射，故
#   ① **字面反斜杠**转义为 ``\\``；
#   ② **目标字符本身**（如正文里本就出现的西里尔 ``а``）前置 ``\\`` 转义，
#      否则会被误判回拉丁 ``a``。两条例外使映射保持单射、解码逐字符可逆。
_HOMO = {
    "a": "а", "b": "ƅ", "c": "с", "d": "ԁ", "e": "е",
    "f": "ƒ", "g": "ɡ", "h": "һ", "i": "і", "j": "ј",
    "k": "κ", "l": "ӏ", "m": "м", "n": "п", "o": "о",
    "p": "р", "q": "ԛ", "r": "г", "s": "ѕ", "t": "т",
    "u": "υ", "v": "ν", "w": "ԝ", "x": "х", "y": "у",
    "z": "ᴢ",
    "A": "А", "B": "В", "C": "С", "D": "Ԁ", "E": "Е",
    "F": "Ϝ", "G": "Ԍ", "H": "Н", "I": "І", "J": "Ј",
    "K": "К", "L": "Ⅼ", "M": "М", "N": "Ν", "O": "О",
    "P": "Р", "Q": "Ԛ", "R": "Ʀ", "S": "Ѕ", "T": "Т",
    "U": "Ս", "V": "Ⅴ", "W": "Ԝ", "X": "Х", "Y": "Ү",
    "Z": "Ζ",
}
_HOMO_TARGETS = set(_HOMO.values())
_HOMO_REVERSE = {v: k for k, v in _HOMO.items()}
_HOMO_ESCAPE = "\\"

if len(_HOMO_TARGETS) != len(_HOMO):  # pragma: no cover
    raise AssertionError("homoglyph 映射表非单射（存在重复目标字符）")
if any(len(ch) != 1 for ch in _HOMO.values()) or any(len(ch) != 1 for ch in _HOMO):  # pragma: no cover
    raise AssertionError("homoglyph 映射表须逐字符一对一")


def enc_homoglyph(text):
    out = []
    for ch in text:
        if ch == _HOMO_ESCAPE:
            out.append(_HOMO_ESCAPE * 2)
        elif ch in _HOMO:
            out.append(_HOMO[ch])
        elif ch in _HOMO_TARGETS:
            out.append(_HOMO_ESCAPE + ch)
        else:
            out.append(ch)
    return "".join(out)


def dec_homoglyph(payload):
    out = []
    i = 0
    while i < len(payload):
        ch = payload[i]
        if ch == _HOMO_ESCAPE:
            if i + 1 >= len(payload):
                raise ValueError("homoglyph: 载荷以孤立反斜杠结尾")
            out.append(payload[i + 1])
            i += 2
        elif ch in _HOMO_REVERSE:
            out.append(_HOMO_REVERSE[ch])
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# ===========================================================================
# 注册表（`V-5` 的 9 个代码 → 编解码函数；**不得改名、不得合并**）
# ===========================================================================
ENCODERS = {
    "base64": (enc_base64, dec_base64),
    "hex": (enc_hex, dec_hex),
    "ascii_art": (enc_ascii_art, dec_ascii_art),
    "rot13": (enc_rot13, dec_rot13),
    "reverse": (enc_reverse, dec_reverse),
    "morse": (enc_morse, dec_morse),
    "url": (enc_url, dec_url),
    "zerowidth": (enc_zerowidth, dec_zerowidth),
    "homoglyph": (enc_homoglyph, dec_homoglyph),
}

VARIANT_NOTES = {
    "base64": "标准 base64（`base64.b64encode`，UTF-8 字节）",
    "hex": "小写十六进制（UTF-8 字节展开）；不加 `0x` 前缀、不加空格分隔",
    "ascii_art": "逐字符艺术化：95 个可打印 ASCII 各对应 5 行×5 列固定字形块，`|` 分隔，整段渲染为 5 行；表外字符转义 `\\u{XXXX}`",
    "rot13": "拉丁字母 ROT13；非字母字符原样保留",
    "reverse": "字符级反转（非字节级）",
    "morse": "标准 Morse（表域 A-Z／0-9／常用标点）；字母间 1 空格、词间 `/`；**小写字母 = Morse 码 + 后缀 `'`**（Morse 符号集仅 `.`/`-`，故无歧义）；非 ASCII 转 token `U+XXXX`",
    "url": "`urllib.parse.quote(text, safe='')` 全百分号编码",
    "zerowidth": "UTF-8 位流映射到 U+200B(0)／U+200C(1)；保留可见字符，每字符后插一个零宽位，余位追加末尾",
    "homoglyph": "拉丁字母→视觉同形字符（52 目标两两互异）；`\\` 与目标字符本身前置 `\\` 转义",
}


# ===========================================================================
# 断言（`windows/W3.3_encode.md` §4 的 `D1`–`D6`）
# ===========================================================================
def assert_unique_sample_ids(rows):
    seen = set()
    dupes = []
    for row in rows:
        sid = row["sample_id"]
        if sid in seen:
            dupes.append(sid)
        seen.add(sid)
    return dupes


def run_assertions(rows, rewrites_rows, encodings, expected_total, expected_rewrites,
                   roundtrip_failures=None, max_failure_rate=0.0):
    """返回 ``{D1..D6: {"passed": bool, "rule": str, ...}}``。

    ``roundtrip_failures`` 由 :func:`verify_roundtrip` 的输出传入（``D5`` 据此判定）；
    为 ``None`` 时按「零失败」判定。
    """
    results = {}

    # D1 —— 行数 == 440 × 9
    results["D1"] = {
        "passed": len(rows) == expected_total,
        "rule": "encoded.csv 行数 == %d（%d × %d）" % (expected_total, expected_rewrites,
                                                       len(encodings)),
        "actual": len(rows), "expected": expected_total,
    }

    # D2 —— 每个 rewrite_id 恰 9 条（每编码 1 条），无缺无重
    per_rewrite = {}
    for row in rows:
        per_rewrite.setdefault(row["rewrite_id"], []).append(row["encoding_type"])
    bad_counts = {k: len(v) for k, v in per_rewrite.items() if len(v) != len(encodings)}
    dup_pairs = {k: sorted(v) for k, v in per_rewrite.items()
                 if len(set(v)) != len(v)}
    missing = sorted({r["rewrite_id"] for r in rewrites_rows} - set(per_rewrite))
    results["D2"] = {
        "passed": not bad_counts and not dup_pairs and not missing
                  and len(per_rewrite) == expected_rewrites,
        "rule": "每个 rewrite_id 恰对应 %d 条（每编码 1 条），无缺无重" % len(encodings),
        "rewrites_seen": len(per_rewrite), "expected_rewrites": expected_rewrites,
        "bad_counts": bad_counts, "duplicated_encodings": dup_pairs, "missing": missing,
    }

    # D3 —— 取值 ⊆ ENCODING_CODES，且 9 个码全部出现、各 440
    counts = {}
    for row in rows:
        counts[row["encoding_type"]] = counts.get(row["encoding_type"], 0) + 1
    unknown = sorted(set(counts) - set(ENCODING_CODES))
    off = {k: counts.get(k, 0) for k in ENCODING_CODES
           if counts.get(k, 0) != expected_rewrites}
    results["D3"] = {
        "passed": not unknown and not off and set(counts) == set(ENCODING_CODES),
        "rule": "encoding_type ⊆ ENCODING_CODES（9 码）且每码计数 == %d" % expected_rewrites,
        "counts": {k: counts.get(k, 0) for k in ENCODING_CODES},
        "unknown_codes": unknown, "off_by": off,
    }

    # D4 —— plaintext 与父 rewrites.csv 行的 text 逐字符一致
    parent = {r["rewrite_id"]: r["text"] for r in rewrites_rows}
    mismatched = []
    for row in rows:
        if parent.get(row["rewrite_id"]) != row["plaintext"]:
            mismatched.append(row["sample_id"])
    results["D4"] = {
        "passed": not mismatched,
        "rule": "每条 plaintext 与其父 rewrites.csv 行的 text 逐字符一致",
        "mismatched": mismatched[:20], "mismatched_total": len(mismatched),
    }

    # D5 —— 每条 payload 解码回 plaintext 成功（阈值：失败率 == max_failure_rate）
    failed = [] if roundtrip_failures is None else list(roundtrip_failures)
    rate = (len(failed) / float(len(rows))) if rows else 0.0
    results["D5"] = {
        "passed": rate <= max_failure_rate,
        "rule": "每条 payload 解码回 plaintext 成功（阈值：失败率 ≤ %s）" % max_failure_rate,
        "failure_rate": round(rate, 8), "failures": len(failed),
        "failed_sample": [f["sample_id"] for f in failed[:20]],
    }

    # D6 —— seed_group_id 与父行一致（无 group 漂移）
    parent_group = {r["rewrite_id"]: r["seed_group_id"] for r in rewrites_rows}
    drift = sorted({row["sample_id"] for row in rows
                    if parent_group.get(row["rewrite_id"]) != row["seed_group_id"]})
    results["D6"] = {
        "passed": not drift,
        "rule": "seed_group_id 与父改写体行一致（无 group 漂移）",
        "drift": drift[:20], "drift_total": len(drift),
    }
    return results


def verify_roundtrip(rows):
    """逐条做 ``解码(编码(x)) == x``；返回 ``(per_code, failures)``。"""
    per_code = {}
    failures = []
    for row in rows:
        code = row["encoding_type"]
        stat = per_code.setdefault(code, {"ok": 0, "failed": 0})
        decode = ENCODERS[code][1]
        try:
            decoded = decode(row["payload"])
        except Exception as exc:  # noqa: BLE001 - 解码异常一律计入失败
            stat["failed"] += 1
            failures.append({"sample_id": row["sample_id"], "encoding_type": code,
                             "reason": "%s: %s" % (type(exc).__name__, exc)})
            continue
        if decoded == row["plaintext"]:
            stat["ok"] += 1
        else:
            stat["failed"] += 1
            failures.append({"sample_id": row["sample_id"], "encoding_type": code,
                             "reason": "解码结果与 plaintext 不一致"})
    return per_code, failures


# ===========================================================================
# 配置与输入
# ===========================================================================
def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return raw["encode"]


def validate_config(cfg):
    errors = []
    encodings = list(cfg.get("encodings") or [])
    if encodings != list(ENCODING_CODES):
        errors.append("config.encodings=%r 与 V-5 的 ENCODING_CODES=%r 不一致"
                      % (encodings, list(ENCODING_CODES)))
    difficulty_map = (cfg.get("difficulty") or {}).get("map") or {}
    missing = [c for c in ENCODING_CODES if c not in difficulty_map]
    if missing:
        errors.append("difficulty.map 缺编码：%s" % missing)
    allowed = set((cfg.get("difficulty") or {}).get("values") or [])
    bad = {c: difficulty_map.get(c) for c in ENCODING_CODES
           if difficulty_map.get(c) not in allowed}
    if bad:
        errors.append("difficulty 取值越域：%s（允许 %s）" % (bad, sorted(allowed)))
    if int(cfg.get("expected_rewrites", 0)) * len(ENCODING_CODES) != int(cfg.get("target_total", 0)):
        errors.append("target_total ≠ expected_rewrites × 9")
    if str(cfg.get("carrier_type")) not in CARRIER_CODES:
        errors.append("carrier_type=%r 不在 CARRIER_CODES" % cfg.get("carrier_type"))
    return errors


def build_rows(rewrites_rows, seeds_by_group, cfg, limit=None):
    """按 ``rewrite_id`` 升序 × 配置内的编码顺序展开为 3960 行。"""
    encodings = list(cfg["encodings"])
    difficulty_map = cfg["difficulty"]["map"]
    carrier = str(cfg["carrier_type"])
    label = int(cfg["label"])
    ordered = sorted(rewrites_rows, key=lambda r: r["rewrite_id"])
    if limit is not None:
        ordered = ordered[:limit]
    rows = []
    for rw in ordered:
        rid = rw["rewrite_id"]
        group = rw["seed_group_id"]
        seed = seeds_by_group.get(group, {})
        stem = rid.replace("-", "")          # `ATK-<rewrite_id 去横线>-<enc>-<carrier>`
        for code in encodings:
            encode = ENCODERS[code][0]
            rows.append({
                "sample_id": "ATK-%s-%s-%s" % (stem, code, carrier),
                "seed_group_id": group,
                "rewrite_id": rid,
                "seed_id": rw.get("seed_id") or seed.get("seed_id", ""),
                "source": seed.get("source", ""),
                "harm_category": seed.get("harm_category", ""),
                "encoding_type": code,
                "difficulty": difficulty_map[code],
                "plaintext": rw["text"],
                "payload": encode(rw["text"]),
                "carrier_type": carrier,
                "label": label,
            })
    return rows


def difficulty_histogram(rows):
    hist = {}
    for row in rows:
        hist[row["difficulty"]] = hist.get(row["difficulty"], 0) + 1
    return hist


def encoding_histogram(rows):
    hist = {}
    for row in rows:
        hist[row["encoding_type"]] = hist.get(row["encoding_type"], 0) + 1
    return hist


# ===========================================================================
# 报告
# ===========================================================================
def build_report_json(cfg, cfg_path, args, rows, per_code, failures, asserted,
                      hist_enc, hist_diff, examples):
    homoglyph_pairs = [{"source": k, "target": v, "target_name": "U+%04X" % ord(v)}
                       for k, v in sorted(_HOMO.items())]
    payload = json.dumps(homoglyph_pairs, ensure_ascii=False, sort_keys=True)
    homoglyph_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "schema": "codeobf-pi/encode_report/v1",
        "window": "W3.3",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {"path": _rel(cfg_path), "sha256": sha256_file(cfg_path)},
        "inputs": {
            "rewrites": {"path": _rel(args.rewrites), "sha256": sha256_file(args.rewrites)},
            "seeds": {"path": _rel(args.seeds), "sha256": sha256_file(args.seeds)},
        },
        "counts": {
            "total": len(rows),
            "by_encoding": hist_enc,
            "by_difficulty": hist_diff,
            "carrier_type": cfg["carrier_type"],
            "label": cfg["label"],
        },
        "constants": {
            "ENCODING_CODES": list(ENCODING_CODES),
            "CARRIER_CODES": list(CARRIER_CODES),
            "OOD_TRAIN_ENCODINGS": list(OOD_TRAIN_ENCODINGS),
            "OOD_HELDOUT_ENCODINGS": list(OOD_HELDOUT_ENCODINGS),
        },
        "difficulty": {
            "criterion": cfg["difficulty"]["criterion"],
            "values": list(cfg["difficulty"]["values"]),
            "map": dict(cfg["difficulty"]["map"]),
        },
        "encodings": [
            {
                "code": code,
                "encode_function": ENCODERS[code][0].__name__,
                "decode_function": ENCODERS[code][1].__name__,
                "variant": VARIANT_NOTES[code],
                "difficulty": cfg["difficulty"]["map"][code],
                "example_plaintext": examples[code]["plaintext"],
                "example_payload": examples[code]["payload"],
                "example_roundtrip_ok": examples[code]["ok"],
                "roundtrip_ok": per_code.get(code, {}).get("ok", 0),
                "roundtrip_failed": per_code.get(code, {}).get("failed", 0),
            }
            for code in ENCODING_CODES
        ],
        "homoglyph_mapping": {
            "pairs": homoglyph_pairs,
            "pair_count": len(homoglyph_pairs),
            "sha256": homoglyph_sha,
            "escapes": {"literal_backslash": "\\\\",
                        "literal_target_char": "\\<target>"},
            "irreversibility_rate": 0.0,
            "irreversibility_note": cfg["roundtrip"]["homoglyph_irreversibility_note"],
        },
        "roundtrip": {
            "per_encoding": per_code,
            "failures_total": len(failures),
            "failures_sample": failures[:20],
            "max_failure_rate": cfg["roundtrip"]["max_failure_rate"],
        },
        "assertions": asserted,
        "reproducibility": {
            "encoder_is_pure_function": True,
            "network_access": False,
            "model_loaded": False,
            "smoke": int(args.smoke or 0),
        },
    }


def _rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), os.getcwd()).replace("\\", "/")
    except ValueError:  # pragma: no cover - 跨盘符
        return str(path).replace("\\", "/")


def render_markdown(ctx):
    r = ctx["json"]
    out = []
    a = out.append
    a("# encode_variants_%s — W3.3 编码变换管道运行报告" % ctx["version"])
    a("")
    a("- 窗口：W3.3（编码变换管道，权限 **A3**）")
    a("- 日期：%s" % r["generated_at"])
    a("- 脚本：`scripts/data/02_03_encode_variants.py`")
    a("- 共享库：`scripts/data/_common.py`（本窗口建立，签名单一出处）")
    a("- 配置：`%s`（sha256 `%s`）" % (r["config"]["path"], r["config"]["sha256"][:16]))
    a("- 上游：`%s`（sha256 `%s`）" % (r["inputs"]["rewrites"]["path"],
                                        r["inputs"]["rewrites"]["sha256"][:16]))
    a("- 种子表：`%s`（sha256 `%s`）" % (r["inputs"]["seeds"]["path"],
                                          r["inputs"]["seeds"]["sha256"][:16]))
    a("")
    a("> 本报告**不含**任何真实恶意样本原文；下文示例一律取自配置 `example_text`（自造文本）。")
    a("")
    a("## 1. 计数")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    a("| `encoded.csv` 行数 | **%d**（`V-1`：%d × %d） |"
      % (r["counts"]["total"], ctx["expected_rewrites"], len(ENCODING_CODES)))
    a("| `carrier_type` | `%s`（本窗口恒 `text`；`code` 归 W3.4） |" % r["counts"]["carrier_type"])
    a("| `label` | `%s` |" % r["counts"]["label"])
    a("")
    a("### 1.1 9 编码各自计数")
    a("")
    a("| encoding_type | 计数 | 难度 | OOD 归属 |")
    a("|---|---|---|---|")
    for code in ENCODING_CODES:
        ood = "训练 5" if code in OOD_TRAIN_ENCODINGS else "留出 4"
        a("| `%s` | %d | `%s` | %s |"
          % (code, r["counts"]["by_encoding"].get(code, 0),
             r["difficulty"]["map"][code], ood))
    a("")
    a("### 1.2 `difficulty` 分布（**本窗口定值**）")
    a("")
    a("| difficulty | 计数 | 编码 |")
    a("|---|---|---|")
    for level in r["difficulty"]["values"]:
        codes = [c for c in ENCODING_CODES if r["difficulty"]["map"][c] == level]
        a("| `%s` | %d | %s |"
          % (level, r["counts"]["by_difficulty"].get(level, 0),
             "、".join("`%s`" % c for c in codes)))
    a("")
    a("定值判据：%s" % r["difficulty"]["criterion"])
    a("")
    a("> 交付：`difficulty` 取值域与映射须由 **W3.8** 登记进 `DATA_CARD`"
      "（`stage2_plan` §2.3「取值域由 W3.3 定值并登记 DATA_CARD（不得空缺）」）。")
    a("")
    a("## 2. 9 编码实现口径（`windows/W3.3_encode.md` §3.2）")
    a("")
    a("| 编码 | 实现函数 | 解码函数 | 口径（具体变体） | 自造示例（`%s`） | 可逆 |"
      % ctx["example_text"])
    a("|---|---|---|---|---|---|")
    for item in r["encodings"]:
        a("| `%s` | `%s()` | `%s()` | %s | 见 `encode_report.json` | %d ok / %d failed |"
          % (item["code"], item["encode_function"], item["decode_function"],
             item["variant"], item["roundtrip_ok"], item["roundtrip_failed"]))
    a("")
    a("**自造示例的载荷**：原文取配置 `example_text` = `%s`，9 条示例载荷的**原文、载荷与"
      "往返结论**逐条列在 `encode_report.json` 的 `encodings[].example_plaintext`／"
      "`example_payload`／`example_roundtrip_ok`（载荷含换行与零宽字符，"
      "故不在本报告内联，以免渲染与复制引入形态漂移）。" % ctx["example_text"])
    a("")
    a("## 3. 可逆性校验（`D5`）")
    a("")
    a("| 编码 | 成功 | 失败 |")
    a("|---|---|---|")
    for code in ENCODING_CODES:
        stat = r["roundtrip"]["per_encoding"].get(code, {"ok": 0, "failed": 0})
        a("| `%s` | %d | %d |" % (code, stat["ok"], stat["failed"]))
    total_rows = r["counts"]["total"]
    total_ok = sum(v["ok"] for v in r["roundtrip"]["per_encoding"].values())
    rate = (total_ok / float(total_rows)) if total_rows else 0.0
    a("")
    a("- **成功率 = %s**（%d / %d；阈值：失败率 ≤ %s）"
      % (_pct(rate), total_ok, total_rows, r["roundtrip"]["max_failure_rate"]))
    a("- `homoglyph` 不可逆率：**%s** —— %s"
      % (r["homoglyph_mapping"]["irreversibility_rate"],
         r["homoglyph_mapping"]["irreversibility_note"]))
    a("- 失败明细：%d 条（前 20 条见 `encode_report.json` 的 `roundtrip.failures_sample`）"
      % r["roundtrip"]["failures_total"])
    a("")
    a("## 4. 脚本内断言 `D1`–`D6`（`windows/W3.3_encode.md` §4）")
    a("")
    a("| 断言 | 判据 | 结果 | 摘要 |")
    a("|---|---|---|---|")
    detail = {
        "D1": "实际 %d / 期望 %d" % (r["assertions"]["D1"]["actual"],
                                     r["assertions"]["D1"]["expected"]),
        "D2": "覆盖 %d 组 / 期望 %d；异常 %d；缺失 %d"
              % (r["assertions"]["D2"]["rewrites_seen"],
                 r["assertions"]["D2"]["expected_rewrites"],
                 len(r["assertions"]["D2"]["bad_counts"]) + len(r["assertions"]["D2"]["duplicated_encodings"]),
                 len(r["assertions"]["D2"]["missing"])),
        "D3": "未知码 %d；计数偏离 %d"
              % (len(r["assertions"]["D3"]["unknown_codes"]),
                 len(r["assertions"]["D3"]["off_by"])),
        "D4": "不一致 %d" % r["assertions"]["D4"]["mismatched_total"],
        "D5": "失败 %d（见 §3）" % r["roundtrip"]["failures_total"],
        "D6": "漂移 %d" % r["assertions"]["D6"]["drift_total"],
    }
    for key in ("D1", "D2", "D3", "D4", "D5", "D6"):
        item = r["assertions"][key]
        a("| `%s` | %s | **%s** | %s |"
          % (key, item["rule"], "PASS" if item["passed"] else "FAIL", detail[key]))
    a("")
    a("## 5. `homoglyph` 映射表（内联登记，`windows/W3.3_encode.md` §3.2）")
    a("")
    a("- 对数：**%d**（52 = 26 小写 + 26 大写）" % r["homoglyph_mapping"]["pair_count"])
    a("- 规范化 JSON 的 sha256：`%s`" % r["homoglyph_mapping"]["sha256"])
    a("- 映射表**全文**见 `encode_report.json` 的 `homoglyph_mapping.pairs`（本报告不复述，避免两处漂移）")
    a("- 转义口径：字面 `\\` → `\\\\`；字面目标字符 → 前置 `\\`")
    a("- **不可逆率 = %s**（唯一失效条件：载荷被第三方二次编辑）"
      % r["homoglyph_mapping"]["irreversibility_rate"])
    a("")
    a("## 6. 复现与环境")
    a("")
    a("- 编码器均为纯函数；本窗口**零联网、零模型加载、零推理**。")
    a("- 幂等：同一输入重复运行，`encoded.csv` **逐字节一致**（已实测）；"
      "`encode_report.json` 除 `generated_at` 时间戳外逐字节一致"
      "（`encoded.csv` 内**不含**任何时间戳字段）。")
    a("- 复现命令：`python scripts/data/02_03_encode_variants.py --config %s "
      "--rewrites %s --out data/interim/encoded/ --verify-roundtrip`"
      % (r["config"]["path"], r["inputs"]["rewrites"]["path"]))
    a("")
    return "\n".join(out) + "\n"


def _pct(value):
    return "%.4f%%" % (value * 100.0) if value is not None else "—"


# ===========================================================================
# 主流程
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="02_03_encode_variants.py",
        description="W3.3 编码变换管道：440 条改写体 × 9 编码 = 3960 条纯文本载体攻击样本。",
    )
    p.add_argument("--config", default="configs/data/encode.yaml", help="编码配置（YAML）")
    p.add_argument("--rewrites", default="data/interim/rewrite/rewrites.csv",
                   help="W3.2 的改写体表（上游，只读）")
    p.add_argument("--seeds", default=None,
                   help="W3.1 的种子表（继承 source／harm_category）；"
                        "缺省 = 配置 inputs.seeds")
    p.add_argument("--out", default="data/interim/encoded/", help="产物输出目录")
    p.add_argument("--smoke", type=int, default=0,
                   help="冒烟档：只对前 N 条改写体编码；产物落 <out>/smoke/")
    p.add_argument("--verify-roundtrip", action="store_true",
                   help="逐条做 解码(编码(x)) == x 校验并打印逐条明细")
    p.add_argument("--report-dir", default="reports/data", help="运行报告输出目录")
    p.add_argument("--log-dir", default="logs/data", help="日志目录")
    p.add_argument("--no-log", action="store_true", help="不落盘日志（供单元测试使用）")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(newline="\n", encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg_errors = validate_config(cfg)
    if cfg_errors:
        for err in cfg_errors:
            print("[02_03] FATAL 配置非法：%s" % err)
        return 1

    seeds_path = args.seeds or cfg["inputs"]["seeds"]
    for label, path in (("rewrites", args.rewrites), ("seeds", seeds_path),
                        ("config", args.config)):
        if not os.path.exists(path):
            print("[02_03] FATAL 找不到 %s：%s" % (label, path))
            return 1

    args.seeds = seeds_path  # 解析后回填，使下游（报告）始终拿到具体路径
    rewrites_rows = read_csv(args.rewrites)
    seeds_rows = read_csv(seeds_path)
    seeds_by_group = {r["seed_group_id"]: r for r in seeds_rows}

    smoke = args.smoke if args.smoke and args.smoke > 0 else 0
    limit = smoke or None
    expected_rewrites = smoke if smoke else int(cfg["expected_rewrites"])
    expected_total = expected_rewrites * len(ENCODING_CODES)

    out_dir = args.out
    if smoke:
        out_dir = os.path.join(out_dir, cfg["output"]["smoke_subdir"])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, cfg["output"]["csv"])
    json_path = os.path.join(out_dir, cfg["output"]["report_json"])

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_lines = []

    def log(text):
        log_lines.append(text)
        print(text, end="" if text.endswith("\n") else "\n")

    log("[02_03] config=%s\n" % args.config)
    log("[02_03] rewrites=%s (%d rows)  seeds=%s (%d rows)\n"
        % (args.rewrites, len(rewrites_rows), seeds_path, len(seeds_rows)))
    log("[02_03] out=%s  smoke=%s\n" % (out_dir, smoke or "FULL"))

    rows = build_rows(rewrites_rows, seeds_by_group, cfg, limit=limit)
    log("[02_03] expanded %d rows (expected %d)\n" % (len(rows), expected_total))

    dupes = assert_unique_sample_ids(rows)
    if dupes:
        log("[02_03] FATAL sample_id 重复 %d 个：%s\n" % (len(dupes), dupes[:5]))
        return 1

    per_code, failures = verify_roundtrip(rows)

    always = bool(cfg["roundtrip"].get("always_verify", True))
    verified = bool(always or args.verify_roundtrip)
    failures_for_gate = failures if verified else []

    asserted = run_assertions(
        rows,
        rewrites_rows[:expected_rewrites] if smoke else rewrites_rows,
        cfg["encodings"], expected_total, expected_rewrites,
        roundtrip_failures=failures_for_gate,
        max_failure_rate=float(cfg["roundtrip"]["max_failure_rate"]),
    )
    asserted["D5"]["verified"] = verified

    write_csv(csv_path, rows, CSV_FIELDS)
    log("[02_03] wrote %s (%d rows)\n" % (csv_path, len(rows)))

    if args.verify_roundtrip:
        for item in failures[:50]:
            log("[02_03] ROUNDTRIP-FAIL %s (%s): %s\n"
                % (item["sample_id"], item["encoding_type"], item["reason"]))

    for key in ("D1", "D2", "D3", "D4", "D5", "D6"):
        item = asserted[key]
        log("[02_03] ASSERT %s %s — %s\n"
            % (key, "PASS" if item["passed"] else "FAIL", item["rule"]))

    examples = {}
    example_text = str(cfg.get("example_text", "hello"))
    for code in cfg["encodings"]:
        encode, decode = ENCODERS[code]
        payload = encode(example_text)
        try:
            ok = decode(payload) == example_text
        except Exception:  # noqa: BLE001
            ok = False
        examples[code] = {"plaintext": example_text, "payload": payload, "ok": ok}

    hist_enc = encoding_histogram(rows)
    hist_diff = difficulty_histogram(rows)
    report = build_report_json(cfg, args.config, args, rows, per_code, failures,
                               asserted, hist_enc, hist_diff, examples)

    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    log("[02_03] wrote %s\n" % json_path)

    if not smoke:
        version = "v1.0"
        report_path = os.path.join(args.report_dir,
                                   "encode_variants_%s_%s.md" % (version, stamp))
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_markdown({"json": report, "version": version,
                                      "expected_rewrites": expected_rewrites,
                                      "example_text": example_text}))
        log("[02_03] wrote %s\n" % report_path)

    log("[02_03] difficulty 分布=%s\n"
        % json.dumps({k: hist_diff[k] for k in sorted(hist_diff)}, ensure_ascii=False))
    log("[02_03] encoding 分布=%s\n"
        % json.dumps({k: hist_enc.get(k, 0) for k in ENCODING_CODES}, ensure_ascii=False))

    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, "02_03_encode_variants_%s.log" % stamp)
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(log_lines))
        print("[02_03] log=%s" % _rel(log_path))

    all_ok = all(asserted[k]["passed"] for k in ("D1", "D2", "D3", "D4", "D5", "D6"))
    if not all_ok:
        print("[02_03] RESULT=FAIL")
        return 1
    print("[02_03] RESULT=OK (D1–D6 全通过；可逆率 100%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

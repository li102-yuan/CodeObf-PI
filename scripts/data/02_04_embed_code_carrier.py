# -*- coding: utf-8 -*-
"""02_04_embed_code_carrier.py — W3.4 代码载体嵌入与 AST 校验。

把 W3.3 的 **3960** 条纯文本载体样本（`data/interim/encoded/encoded.csv`）
各再嵌入一份**代码载体**副本，产出 **3960** 条代码载体攻击样本
（`V-1`：`440 × 9 × 2 = 7920`，本窗口补齐 `× 2` 的 `code` 侧）。

接口契约（`windows/W3.4_code_carrier.md` §3）::

    python scripts/data/02_04_embed_code_carrier.py \\
      --config configs/data/code_carrier.yaml \\
      --encoded data/interim/encoded/encoded.csv \\
      --out data/interim/code_carrier/ \\
      [--smoke N]           # 只对前 N 条编码变体嵌入，产物落 out/smoke/
      [--ast-gate 0.10]     # 门禁 2 阈值，缺省 0.10

退出码：``0`` 成功；非 ``0`` 失败（含计数 ≠ 3960 或 AST 报错率 > `--ast-gate`）。

**核心安全口径（v5 §2.4「新增 12」）**：载荷只作为**文本**被载体携带，
**绝不可被解释器执行**。本脚本用「**惰性区间**」证明这一点：
载荷在 `text` 中的**每一次出现**都必须**完整落在**注释／字符串字面量／
docstring／标识符之内（`E7`）。这条证明直接可引用进论文，避免审稿人
把「代码载体样本」误读为**代码执行漏洞**。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import keyword
import os
import string
import sys
import tokenize
from datetime import datetime

# ---------------------------------------------------------------------------
# 载入共享库 `scripts/data/_common.py`（唯一出处；本窗口不改其任何签名）
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "data_common", os.path.join(_HERE, "_common.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_common"] = module
    spec.loader.exec_module(module)
    return module


common = _load_common()

ENCODING_CODES = common.ENCODING_CODES
read_csv = common.read_csv
write_csv = common.write_csv
sha256_file = common.sha256_file
sha256_text = common.sha256_text

CSV_FIELDS = ["sample_id", "seed_group_id", "rewrite_id", "seed_id", "source",
              "harm_category", "encoding_type", "difficulty", "carrier_type",
              "carrier_subtype", "language", "plaintext", "payload", "text",
              "ast_ok", "ast_error", "payload_intact", "label"]

AST_REPORT_FIELDS = ["carrier_subtype", "encoding_type", "total", "ast_ok",
                     "ast_failed", "regenerated", "dropped", "dropped_reason"]

#: `V-7` 的 5 个代码载体子型。**不得改名、不得增删。**
SUBTYPES = ("py_comment", "js_comment", "string_literal", "docstring",
            "identifier_ast")

#: 子型 → 语言（§3.1：`js_comment` → `javascript`，其余 → `python`）
LANGUAGE = {
    "py_comment": "python",
    "js_comment": "javascript",
    "string_literal": "python",
    "docstring": "python",
    "identifier_ast": "python",
}

#: 载荷**不得**引入的可执行片段（`E7` 的第二道）。宿主骨架自身须**一个都不含**。
FORBIDDEN_TOKENS = ("eval(", "exec(", "__import__(", "subprocess", "os.system")

#: `identifier_ast` 里承载载荷字符串的固定变量名（`string_literal` 亦复用）
PAYLOAD_VAR = "PAYLOAD_TEXT"


# ===========================================================================
# 宿主骨架模板（**全部由模板生成，绝不从 `data/raw/` 拉真实代码**）
# ===========================================================================
# 占位符一律用 `${NAME}` 花括号形式，避免 `${PAYLOAD}` 与更长标识符前缀冲突。
SKELETONS = {
    # --- 1/5 py_comment：载荷落 `#` 之后整行（多行载荷逐行加 `# `）---
    "py_comment": [
        '"""Text normalization helpers."""\n'
        '\n'
        '\n'
        'def normalize(value):\n'
        '${PAYLOAD}\n'
        '    return value\n',
        '"""String sanitisation helpers."""\n'
        '\n'
        'SCHEMA_VERSION = 1\n'
        '\n'
        '\n'
        'def sanitise(value):\n'
        '    value = value.strip()\n'
        '${PAYLOAD}\n'
        '    return value\n',
    ],
    # --- 2/5 js_comment：载荷落 `//` 行注释（规避 `*/`）或 `/* … */` 块内 ---
    # 骨架 0 的表头是**行注释**，故载荷只能用**块注释**（去框架靠 `"\\n/*\\n"` 锚点）；
    # 骨架 1 的表头是**块注释**，故载荷只能用**行注释**（去框架靠行首 `// `）。
    # 两套风格互补 ⇒ 去框架在两种形态下都**精确可判**。
    "js_comment": [
        '// Text normalization helpers.\n'
        '\n'
        'function normalize(value) {\n'
        '  return value;\n'
        '}\n'
        '\n'
        '${PAYLOAD}\n',
        '/**\n'
        ' * Text normalization helpers.\n'
        ' */\n'
        '\n'
        'function normalize(value) {\n'
        '  return value;\n'
        '}\n'
        '\n'
        '${PAYLOAD}\n',
    ],
    # --- 3/5 string_literal：载荷落 Python 字符串字面量，**赋给变量但从不使用** ---
    "string_literal": [
        '"""Text normalization helpers."""\n'
        '\n'
        '%s = ${PAYLOAD}\n'
        '\n'
        '\n'
        'def normalize(value):\n'
        '    return value\n' % PAYLOAD_VAR,
        '"""Text normalization helpers."""\n'
        '\n'
        'SCHEMA_VERSION = 1\n'
        '%s = ${PAYLOAD}\n'
        '\n'
        '\n'
        'def describe():\n'
        '    return SCHEMA_VERSION\n' % PAYLOAD_VAR,
    ],
    # --- 4/5 docstring：载荷落模块级三引号 docstring ---
    "docstring": [
        '${PAYLOAD}\n'
        '\n'
        '\n'
        'def normalize(value):\n'
        '    return value\n',
        '${PAYLOAD}\n'
        '\n'
        '\n'
        'SCHEMA_VERSION = 1\n'
        '\n'
        '\n'
        'def describe():\n'
        '    return SCHEMA_VERSION\n',
    ],
    # --- 5/5 identifier_ast：载荷落标识符 + 字典字面量的字符串值 ---
    # 标识符合法化**必然有损** ⇒ 依 §2 规定另设**备用落点**（同一样本内的注释）
    # 承载**逐字符完整**的载荷，用于 `E6` 的无损还原判定。
    "identifier_ast": [
        '"""Metadata registry."""\n'
        '\n'
        '\n'
        'def registry():\n'
        '${PAYLOAD}\n'
        '    ${IDENT} = {"name": ${LITERAL}}\n'
        '    return 0\n',
        '"""Attribute registry."""\n'
        '\n'
        'META_TAG = "codeobf"\n'
        '\n'
        '\n'
        'def registry():\n'
        '${PAYLOAD}\n'
        '    ${IDENT} = {"name": ${LITERAL}}\n'
        '    return META_TAG\n',
    ],
}

SKELETON_COUNT = 2

if sorted(SKELETONS) != sorted(SUBTYPES):  # pragma: no cover
    raise AssertionError("SKELETONS 未覆盖 V-7 的 5 个子型")
for _sub, _templates in SKELETONS.items():  # pragma: no cover
    if len(_templates) != SKELETON_COUNT:
        raise AssertionError("子型 %s 的骨架数应为 %d" % (_sub, SKELETON_COUNT))
    for _t in _templates:
        for _tok in FORBIDDEN_TOKENS:
            if _tok in _t:
                raise AssertionError("宿主骨架含禁用片段 %r：%s" % (_tok, _sub))


# ===========================================================================
# 载荷框架化（每种子型的落点写法）
# ===========================================================================
def _odd_trailing_backslashes(text):
    """末尾连续反斜杠个数是否为奇数（奇数时裸三引号会被转义而**语法报错**）。"""
    count = 0
    for ch in reversed(text):
        if ch == "\\":
            count += 1
        else:
            break
    return count % 2 == 1


def frame_py_comment(payload):
    """`#` 行注释块（4 空格缩进，`# ` 后接**未改动**的载荷行）。"""
    return "\n".join("    # " + line for line in payload.split("\n"))


def frame_js_block(payload):
    """`/* … */` 块注释，载荷两侧各垫**恰好一个**换行（去框架即精确切片）。"""
    return "/*\n" + payload + "\n*/"


def frame_js_line(payload):
    """`//` 行注释块（列 0 起）。"""
    return "\n".join("// " + line for line in payload.split("\n"))


def frame_string_literal(payload):
    """`json.dumps(..., ensure_ascii=False)`：JSON 串转义集 ⊆ Python 串转义集 ⇒ 恒语法合法。"""
    return json.dumps(payload, ensure_ascii=False)


#: Python 中**有效**的反斜杠逃逸首字符（其余形如 `\ ` 属无效逃逸：
#: 3.12 起报 `SyntaxWarning`，未来版本将成为 `SyntaxError` ⇒ 必须主动规避）。
_VALID_ESCAPE_CHARS = set("nrt\\'\"abfv01234567xuUN\n")


def is_valid_escape(ch):
    return ch in _VALID_ESCAPE_CHARS


def has_invalid_escape(payload):
    """载荷中是否含**无效逃逸序列**（`\\` + 非法字符）。"""
    i = 0
    while i < len(payload) - 1:
        if payload[i] == "\\":
            if not is_valid_escape(payload[i + 1]):
                return True
            i += 2
        else:
            i += 1
    return False


def frame_docstring(payload):
    """三引号 docstring，按载荷内容选分隔符与转义策略。

    判据按**优先级**（前三条保持载荷**原样**，仅当原样不安全时才转义）：

    1. 含 `\"\"\"` 或末尾反斜杠个数为奇数 ⇒ 裸 `\"\"\"` 会**语法报错**，不可用；
    2. 含 `'''` 或末尾反斜杠个数为奇数 ⇒ 裸 `'''` 不可用；
    3. 含**无效逃逸**（如 `\\ `）⇒ 虽当下仍可解析（`SyntaxWarning`），
       但未来 Python 将判为 `SyntaxError` ⇒ 主动规避，不留隐患；
    4. 以上都不触发时才用裸形态（载荷**逐字符**落在字面量里）；
       否则走**全转义兜底**。
    """
    trailing_odd = _odd_trailing_backslashes(payload)
    if not trailing_odd and not has_invalid_escape(payload):
        if '"""' not in payload:
            return '"""' + payload + '"""'
        if "'''" not in payload:
            return "'''" + payload + "'''"
    # 兜底：`\` 与 `"` 全转义（单射；解码即 Python 自身的字符串反转义）
    return '"""' + payload.replace("\\", "\\\\").replace('"', '\\"') + '"""'


def legalize_identifier(payload, prefix="p_"):
    """把载荷合法化为 Python 标识符。返回 ``(标识符, 被替换字符数)``。

    判据：`ch` 能否出现在标识符中 ⇔ ``("_" + ch).isidentifier()`` 成立
    （比 ``ch.isalnum()`` 更严：排除上标数字等「是字母数字但非法标识符」的字符）。
    非法字符一律替换为 ``_`` ⇒ **必然有损**，故 `identifier_ast` 另设备用落点。
    """
    chars = []
    replaced = 0
    for ch in payload:
        if ("_" + ch).isidentifier():
            chars.append(ch)
        else:
            chars.append("_")
            replaced += 1
    ident = prefix + "".join(chars)
    if not ident.isidentifier() or keyword.iskeyword(ident):
        ident = prefix + "x"
    return ident, replaced


def frame_identifier_ast(payload):
    """返回 `identifier_ast` 的三个占位符（**标识符**／**字典字符串值**／**备用落点注释**）。"""
    ident, replaced = legalize_identifier(payload)
    return {
        "IDENT": ident,
        "LITERAL": json.dumps(payload, ensure_ascii=False),
        "PAYLOAD": "\n".join("    # " + line for line in payload.split("\n")),
        "replaced": replaced,
    }


# ===========================================================================
# 嵌入：候选 (子型, 骨架号, 转义策略) → text
# ===========================================================================
def js_forms(payload):
    """`js_comment` 的候选形态（**`*/` 必须显式处理**）。

    `*/` 在块注释里会**提前闭合注释**；而修正它必须改动载荷本身，
    与 `E6`（载荷无损）冲突。故策略为：**含 `*/` 时改用行注释**（无需转义、
    载荷一字不动），这就是 §2 要求的「包裹策略规避」。
    """
    if "*/" in payload:
        return [(1, "line")]          # 行注释：`*/` 天然无害
    return [(0, "block"), (1, "line")]


def embed_text(subtype, skeleton_idx, strategy, payload):
    """把载荷嵌入骨架，返回 ``(text, form)``。``form`` 供去框架判型。"""
    template = string.Template(SKELETONS[subtype][skeleton_idx])
    if subtype == "py_comment":
        return template.substitute(PAYLOAD=frame_py_comment(payload)), "line"
    if subtype == "js_comment":
        block = strategy == "block"
        block_payload = frame_js_block(payload) if block else frame_js_line(payload)
        return template.substitute(PAYLOAD=block_payload), strategy
    if subtype == "string_literal":
        return template.substitute(PAYLOAD=frame_string_literal(payload)), "literal"
    if subtype == "docstring":
        return template.substitute(PAYLOAD=frame_docstring(payload)), "docstring"
    if subtype == "identifier_ast":
        parts = frame_identifier_ast(payload)
        return template.substitute(IDENT=parts["IDENT"], LITERAL=parts["LITERAL"],
                                   PAYLOAD=parts["PAYLOAD"]), "ident"
    raise ValueError("未知子型 %r" % (subtype,))


# ===========================================================================
# 去框架：`payload_intact` 的**唯一口径**（无损还原）
# ===========================================================================
def _strip_comment_lines(text, marker):
    """取全部「（可含缩进）`marker` + 一个空格」开头的行，剥掉该前缀后以 ``\\n`` 连接。

    适用前提：**宿主骨架自身不含此类注释行**（已由模板设计保证），
    故这些行全部来自载荷 ⇒ 该剥法是**精确逆运算**。
    """
    prefix = marker + " "
    out = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(prefix):
            out.append(stripped[len(prefix):])
    return "\n".join(out)


def de_frame(text, subtype):
    """去框架 ⇒ 还原载荷。**`payload_intact = 1` ⟺ `de_frame(text) == payload`。**

    逐子型的逆运算：

    - `py_comment` / `identifier_ast`：取全部 `# ` 注释行、剥掉前缀后以 `\\n` 连接
    - `js_comment`：块形态取 `/*\\n` 与 `\\n*/` 之间（**精确切片，不做 strip**）；
      行形态取全部 `// ` 行
    - `string_literal`：AST 取 `PAYLOAD_TEXT` 赋值右侧常量值
    - `docstring`：AST 取模块 docstring（`clean=False`，故不被缩进清洗干扰）

    后两者走 **AST 取常量值**，即由 Python 自身完成反转义 ⇒ 对所有转义策略都精确。
    """
    if subtype in ("py_comment", "identifier_ast"):
        return _strip_comment_lines(text, "#")
    if subtype == "js_comment":
        if "\n/*\n" in text:
            start = text.index("/*\n") + 3
            end = text.rindex("\n*/")
            return text[start:end]
        return _strip_comment_lines(text, "//")
    if subtype == "string_literal":
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == PAYLOAD_VAR:
                        if isinstance(node.value, ast.Constant):
                            return node.value.value
        raise ValueError("string_literal: 未找到 %s 的字面量赋值" % PAYLOAD_VAR)
    if subtype == "docstring":
        tree = ast.parse(text)
        doc = ast.get_docstring(tree, clean=False)
        if doc is None:
            raise ValueError("docstring: 模块首语句不是字符串字面量")
        return doc
    raise ValueError("未知子型 %r" % (subtype,))


# ===========================================================================
# 惰性区间（`E7` 不可执行性证明的核心）
# ===========================================================================
def _line_offsets(text):
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def _abs_offset(offsets, text, pos):
    row, col = pos
    base = offsets[min(max(row - 1, 0), len(offsets) - 1)]
    return min(base + col, len(text))


def _token_kind(token_type):
    if token_type == tokenize.COMMENT:
        return "comment"
    if token_type == tokenize.STRING:
        return "string"
    return "identifier"


def inert_spans_python(text):
    """Python 的惰性区间：**注释**、**字符串字面量**（含 docstring）、**标识符**。

    一律按字符绝对偏移给出 ``(start, end, kind)``。
    """
    offsets = _line_offsets(text)
    spans = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NAME):
                start = _abs_offset(offsets, text, tok.start)
                end = _abs_offset(offsets, text, tok.end)
                if end > start:
                    spans.append((start, end, _token_kind(tok.type)))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return spans


def inert_spans_javascript(text):
    """JavaScript 的惰性区间：`//` 行注释、`/* */` 块注释、`'`/`"`/`` ` `` 字符串。"""
    spans = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = text.find("\n", i)
            end = n if end < 0 else end
            spans.append((i, end, "comment"))
            i = end
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            spans.append((i, end, "comment"))
            i = end
            continue
        if ch in ("'", '"', "`"):
            j = i + 1
            while j < n and text[j] != ch:
                j += 1 if text[j] != "\\" else 2
            j = min(j + 1, n)
            spans.append((i, j, "string"))
            i = j
            continue
        i += 1
    return spans


def inert_spans(text, language):
    return (inert_spans_python(text) if language == "python"
            else inert_spans_javascript(text))


def find_occurrences(haystack, needle):
    """`needle` 在 `haystack` 中的全部起始偏移（空串返回空列表）。"""
    if not needle:
        return []
    out = []
    start = haystack.find(needle)
    while start >= 0:
        out.append(start)
        start = haystack.find(needle, start + 1)
    return out


#: `E7` 的「逐行补查」阈值：载荷行短于此长度时不做独立检查（避免单字符噪声命中）。
E7_MIN_LINE = 3


def _e7_needles(payload):
    """`E7` 的检查对象：**整段载荷** + 每个长度 ≥ `E7_MIN_LINE` 的**载荷行**。

    为什么要逐行补查：多行载荷在注释载体中**不可能成为连续子串**
    （每行都被插入了 `# `／`// ` 前缀），若只查整段，该子型的检查会**空转通过**。
    逐行补查后，每一行都必须在某个惰性区间内 ⇒ 检查**非空转**。
    """
    needles = []
    if payload:
        needles.append(payload)
    if "\n" in payload:
        needles.extend(line for line in payload.split("\n")
                       if len(line) >= E7_MIN_LINE)
    return needles


def check_unexecutable(text, payload, language):
    """`E7`：载荷（及其各行）的**每一次出现**都必须**完整落在**某个惰性区间内。

    「完整落在」= 整段包含（``s <= start and end <= e``），**不是**部分重叠。

    返回 ``{"ok": bool, "needles": int, "occurrences": int, "uncovered": [偏移…],
    "forbidden_in_text": [...], "forbidden_outside_inert": [...], "spans": int}``。
    """
    spans = inert_spans(text, language)
    needles = _e7_needles(payload)
    occurrences = 0
    uncovered = []
    for needle in needles:
        for start in find_occurrences(text, needle):
            occurrences += 1
            end = start + len(needle)
            if not any(s <= start and end <= e for s, e, _ in spans):
                uncovered.append((start, needle[:20]))

    forbidden_in_text = [tok for tok in FORBIDDEN_TOKENS if tok in text]
    forbidden_outside = []
    for token in forbidden_in_text:
        for start in find_occurrences(text, token):
            end = start + len(token)
            if not any(s <= start and end <= e for s, e, _ in spans):
                forbidden_outside.append(token)
                break

    return {
        "ok": not uncovered and not forbidden_outside,
        "needles": len(needles),
        "occurrences": occurrences,
        "uncovered": uncovered[:5],
        "spans": len(spans),
        "forbidden_in_text": sorted(set(forbidden_in_text)),
        "forbidden_outside_inert": sorted(set(forbidden_outside)),
    }


def check_payload_constant_positions(text):
    """`E7` 附加：字符串常量**不得**出现在可执行位置（**比只查载荷更严**）。

    判定：遍历 AST 并建父链，对**每一个** `str` 型 `ast.Constant`
    （不只是含载荷的那些 —— 全局更严），要求其父节点 ∈
    {`Expr`(表达式语句)／`Assign.value`／`AnnAssign.value`／`Dict` 值／
    `List`·`Tuple`·`Set` 元素}。`Call` 参数、`Attribute`、`Subscript`
    一律**判不合格**（这正是 `eval(` / `getattr(` / `x[...]` 类落点的形态）。
    """
    tree = ast.parse(text)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    allowed_parents = (ast.Expr, ast.Assign, ast.AnnAssign, ast.Dict, ast.List,
                       ast.Tuple, ast.Set)
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call):
            violations.append("Call-argument")
        elif isinstance(parent, ast.Attribute):
            violations.append("Attribute-value")
        elif isinstance(parent, ast.Subscript):
            violations.append("Subscript")
        elif parent is not None and not isinstance(parent, allowed_parents):
            violations.append(type(parent).__name__)
    return {"ok": not violations, "violations": sorted(set(violations))}


# ===========================================================================
# AST 语法校验（门禁 2）
# ===========================================================================
def ast_check(text, language):
    """返回 ``(ast_ok, ast_error)``：python → 1/0；javascript → -1（不进分母）。"""
    if language != "python":
        return -1, ""
    try:
        ast.parse(text)
        return 1, ""
    except SyntaxError as exc:
        return 0, ("%s" % (exc.msg or exc)).splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return 0, ("%s: %s" % (type(exc).__name__, exc)).splitlines()[0]


# ===========================================================================
# 子型分配（`V-7`：`sample_id` 的 sha256 取模，确定性）
# ===========================================================================
def subtype_index(sample_id):
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % len(SUBTYPES)


def candidate_subtypes(sample_id):
    """主候选 + 4 个替补，顺序确定（补位用；`E4` 保证 5 类均出现）。"""
    start = subtype_index(sample_id)
    return [SUBTYPES[(start + k) % len(SUBTYPES)] for k in range(len(SUBTYPES))]


def candidate_attempts(subtype, payload):
    """某子型下的全部 `(骨架号, 转义策略)` 尝试序列。"""
    if subtype == "js_comment":
        return js_forms(payload)
    return [(idx, "default") for idx in range(SKELETON_COUNT)]


# ===========================================================================
# 单条样本的嵌入（含重生成）
# ===========================================================================
def embed_one(sample_id, payload):
    """尽最大努力嵌入一条；返回 ``(text, subtype, language, attempts, drop_reason)``。

    依 §3：`ast.parse` 失败 ⇒ **先重生成**（换骨架／换转义策略，再换子型），
    全部候选都失败才丢弃（`drop_reason` 非空）。`E7`／`E6` 的失败同样触发重生成。

    丢弃时 `subtype` 返回**主候选**（供 `ast_report.csv` 把丢失计入对应单元）。
    """
    attempts = 0
    last_error = ""
    candidates = candidate_subtypes(sample_id)
    for subtype in candidates:
        language = LANGUAGE[subtype]
        for skeleton_idx, strategy in candidate_attempts(subtype, payload):
            attempts += 1
            try:
                text, _form = embed_text(subtype, skeleton_idx, strategy, payload)
            except Exception as exc:  # noqa: BLE001
                last_error = "embed: %s: %s" % (type(exc).__name__, exc)
                continue
            ast_ok, ast_error = ast_check(text, language)
            if ast_ok == 0:
                last_error = "ast: %s" % ast_error
                continue
            if ast_ok == 1 and not check_payload_constant_positions(text)["ok"]:
                last_error = "constant-position"
                continue
            try:
                recovered = de_frame(text, subtype)
            except Exception as exc:  # noqa: BLE001
                last_error = "de_frame: %s: %s" % (type(exc).__name__, exc)
                continue
            if recovered != payload:
                last_error = "payload_intact"
                continue
            if not check_unexecutable(text, payload, language)["ok"]:
                last_error = "executable-position"
                continue
            return text, subtype, language, attempts, ""
    return "", candidates[0], "", attempts, last_error or "exhausted"


# ===========================================================================
# 断言 `E1`–`E8`
# ===========================================================================
def run_assertions(rows, parent_rows, ast_gate, smoke=False):
    results = {}
    expected_total = len(parent_rows)

    # E1 —— 行数 == 3960
    results["E1"] = {
        "passed": len(rows) == expected_total,
        "rule": "code_carrier.csv 行数 == %d（%s）" % (expected_total, "440 × 9"),
        "actual": len(rows), "expected": expected_total,
    }

    # E2 —— 每个 (rewrite_id, encoding_type) 恰 1 条
    seen = {}
    for row in rows:
        key = (row["rewrite_id"], row["encoding_type"])
        seen[key] = seen.get(key, 0) + 1
    dupes = {"%s|%s" % k: v for k, v in seen.items() if v != 1}
    missing = ["%s|%s" % (r["rewrite_id"], r["encoding_type"]) for r in parent_rows
               if (r["rewrite_id"], r["encoding_type"]) not in seen]
    results["E2"] = {
        "passed": not dupes and not missing and len(seen) == expected_total,
        "rule": "每个 (rewrite_id, encoding_type) 组合恰 1 条",
        "combos": len(seen), "duplicated": dupes, "missing_total": len(missing),
        "missing_sample": missing[:5],
    }

    # E3 —— sample_id 全局唯一，且与 encoded.csv 的 text 载体样本无 id 冲突
    ids = [r["sample_id"] for r in rows]
    collisions = sorted({i for i in ids if ids.count(i) > 1})
    parent_ids = {r["sample_id"] for r in parent_rows}
    overlap = sorted(set(ids) & parent_ids)
    results["E3"] = {
        "passed": not collisions and not overlap,
        "rule": "sample_id 全局唯一，且与 encoded.csv 的 text 载体样本无 id 冲突",
        "duplicates": collisions[:5], "overlap_with_text_carrier": overlap[:5],
    }

    # E4 —— 5 个子型全部出现，每类计数 > 0
    counts = {}
    for row in rows:
        counts[row["carrier_subtype"]] = counts.get(row["carrier_subtype"], 0) + 1
    absent = [s for s in SUBTYPES if counts.get(s, 0) <= 0]
    results["E4"] = {
        "passed": not absent and set(counts) == set(SUBTYPES),
        "rule": "5 个 carrier_subtype 全部出现且每类计数 > 0",
        "counts": {s: counts.get(s, 0) for s in SUBTYPES}, "absent": absent,
        # `E4` 是**总体性**断言：样本数 < 5 时不可能覆盖 5 个子型。
        # 冒烟档下**显式**降级为信息项（不参与退出码），并如实标注，不做静默放行。
        "informational": bool(smoke),
        "informational_reason": ("冒烟档：N=%d < 5，无法覆盖 5 子型；"
                                 "该断言仅在全量档参与退出码" % len(rows)) if smoke else "",
    }

    # E5 —— 门禁 2：ast_failed / (ast_ok + ast_failed) ≤ gate（分母不含 javascript）
    ast_ok = sum(1 for r in rows if str(r["ast_ok"]) == "1")
    ast_failed = sum(1 for r in rows if str(r["ast_ok"]) == "0")
    denominator = ast_ok + ast_failed
    rate = (ast_failed / float(denominator)) if denominator else 0.0
    results["E5"] = {
        "passed": rate <= ast_gate,
        "rule": "门禁 2：ast_failed / (ast_ok + ast_failed) ≤ %s（分母不含 javascript）" % ast_gate,
        "ast_ok": ast_ok, "ast_failed": ast_failed, "denominator": denominator,
        "error_rate": round(rate, 8), "gate": ast_gate,
    }

    # E6 —— payload_intact == 1 的比例 == 100%
    intact = sum(1 for r in rows if str(r["payload_intact"]) == "1")
    results["E6"] = {
        "passed": not rows or intact == len(rows),
        "rule": "payload_intact == 1 的比例 == 100%（去框架可无损还原）",
        "intact": intact, "total": len(rows),
    }

    # E7 —— 不可执行性静态检查
    violations = []
    for row in rows:
        if str(row["ast_ok"]) == "-1":
            continue
        info = check_unexecutable(row["text"], row["payload"], row["language"])
        if not info["ok"]:
            violations.append(row["sample_id"])
    results["E7"] = {
        "passed": not violations,
        "rule": "载荷的每一次出现都完整落在注释／字符串／docstring／标识符内；"
                "禁用片段不得出现在惰性区间之外",
        "violations": violations[:5], "violations_total": len(violations),
    }

    # E8 —— seed_group_id 与父 encoded.csv 行一致
    parent_group = {(r["rewrite_id"], r["encoding_type"]): r["seed_group_id"]
                    for r in parent_rows}
    drift = sorted({row["sample_id"] for row in rows
                    if parent_group.get((row["rewrite_id"], row["encoding_type"]))
                    != row["seed_group_id"]})
    results["E8"] = {
        "passed": not drift,
        "rule": "seed_group_id 与父 encoded.csv 行一致（无 group 漂移）",
        "drift": drift[:5], "drift_total": len(drift),
    }
    return results


def contiguity_stats(rows):
    """信息项：更严的「载荷为 `text` 的**连续子串**」计数（不参与 E6 判定）。"""
    total = len(rows)
    contiguous = sum(1 for r in rows if r["payload"] and r["payload"] in r["text"])
    return {"contiguous": contiguous, "total": total,
            "rate": round(contiguous / float(total), 6) if total else 0.0}


# ===========================================================================
# 配置
# ===========================================================================
def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["code_carrier"]


def validate_config(cfg):
    errors = []
    if list(cfg.get("subtypes") or []) != list(SUBTYPES):
        errors.append("config.subtypes=%r 与 V-7 的 SUBTYPES=%r 不一致"
                      % (cfg.get("subtypes"), list(SUBTYPES)))
    lang = cfg.get("language") or {}
    for sub in SUBTYPES:
        if lang.get(sub) != LANGUAGE[sub]:
            errors.append("language[%s] 应为 %r，实为 %r"
                          % (sub, LANGUAGE[sub], lang.get(sub)))
    if str(cfg.get("carrier_type")) != "code":
        errors.append("carrier_type 应为 'code'")
    if int(cfg.get("expected_encoded", 0)) != int(cfg.get("target_total", 0)):
        errors.append("target_total ≠ expected_encoded（本窗口 1:1）")
    if not (0.0 <= float(cfg.get("ast_gate", -1)) <= 1.0):
        errors.append("ast_gate 越域")
    return errors


# ===========================================================================
# 行装配
# ===========================================================================
def build_rows(parent_rows, limit=None):
    """按父表顺序（`rewrite_id` × `V-5` 编码序）逐条嵌入，返回 ``(rows, drops)``。"""
    ordered = sorted(parent_rows, key=lambda r: (r["rewrite_id"],
                                                 ENCODING_CODES.index(r["encoding_type"])))
    if limit is not None:
        ordered = ordered[:limit]
    rows = []
    drops = []
    for parent in ordered:
        sample_id = "ATK-%s-%s-code" % (parent["rewrite_id"].replace("-", ""),
                                        parent["encoding_type"])
        text, subtype, language, attempts, drop_reason = embed_one(sample_id, parent["payload"])
        if drop_reason:
            drops.append({"sample_id": sample_id, "reason": drop_reason,
                          "carrier_subtype": subtype,
                          "encoding_type": parent["encoding_type"]})
            continue
        ast_ok, ast_error = ast_check(text, language)
        intact = 1 if de_frame(text, subtype) == parent["payload"] else 0
        rows.append({
            "sample_id": sample_id,
            "seed_group_id": parent["seed_group_id"],
            "rewrite_id": parent["rewrite_id"],
            "seed_id": parent["seed_id"],
            "source": parent["source"],
            "harm_category": parent["harm_category"],
            "encoding_type": parent["encoding_type"],
            "difficulty": parent["difficulty"],
            "carrier_type": "code",
            "carrier_subtype": subtype,
            "language": language,
            "plaintext": parent["plaintext"],
            "payload": parent["payload"],
            "text": text,
            "ast_ok": ast_ok,
            "ast_error": ast_error,
            "payload_intact": intact,
            "label": 1,
            "_attempts": attempts,
        })
    return rows, drops


def build_ast_report(rows, drops):
    """逐子型 × 逐编码交叉表（`ast_report.csv`）：5 × 9 = 45 行。"""
    table = {}
    for subtype in SUBTYPES:
        for code in ENCODING_CODES:
            table[(subtype, code)] = {"total": 0, "ast_ok": 0, "ast_failed": 0,
                                      "regenerated": 0, "dropped": 0,
                                      "dropped_reason": ""}
    for row in rows:
        cell = table[(row["carrier_subtype"], row["encoding_type"])]
        cell["total"] += 1
        if str(row["ast_ok"]) == "1":
            cell["ast_ok"] += 1
        elif str(row["ast_ok"]) == "0":
            cell["ast_failed"] += 1
        if row.get("_attempts", 1) > 1:
            cell["regenerated"] += 1
    for drop in drops:
        key = (drop.get("carrier_subtype"), drop.get("encoding_type"))
        if key in table:
            cell = table[key]
            cell["dropped"] += 1
            reason = drop.get("reason", "")
            cell["dropped_reason"] = (cell["dropped_reason"] + "; " + reason
                                      if cell["dropped_reason"] else reason)
    out = []
    for subtype in SUBTYPES:
        for code in ENCODING_CODES:
            cell = table[(subtype, code)]
            out.append({
                "carrier_subtype": subtype,
                "encoding_type": code,
                "total": cell["total"],
                "ast_ok": cell["ast_ok"],
                "ast_failed": cell["ast_failed"],
                "regenerated": cell["regenerated"],
                "dropped": cell["dropped"],
                "dropped_reason": cell["dropped_reason"],
            })
    return out


# ===========================================================================
# 报告
# ===========================================================================
def _rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), os.getcwd()).replace("\\", "/")
    except ValueError:  # pragma: no cover
        return str(path).replace("\\", "/")


def skeleton_catalog():
    out = []
    for subtype in SUBTYPES:
        for idx, template in enumerate(SKELETONS[subtype]):
            out.append({
                "carrier_subtype": subtype,
                "skeleton_index": idx,
                "language": LANGUAGE[subtype],
                "sha256": sha256_text(template),
                "template": template,
            })
    return out


def build_report_json(cfg, cfg_path, args, rows, ast_report, drops, asserted,
                      contiguity, legalize_stats):
    return {
        "schema": "codeobf-pi/code_carrier_report/v1",
        "window": "W3.4",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {"path": _rel(cfg_path), "sha256": sha256_file(cfg_path)},
        "inputs": {"encoded": {"path": _rel(args.encoded),
                               "sha256": sha256_file(args.encoded)}},
        "counts": {
            "total": len(rows),
            "by_subtype": {s: sum(1 for r in rows if r["carrier_subtype"] == s)
                           for s in SUBTYPES},
            "by_language": {lang: sum(1 for r in rows if r["language"] == lang)
                            for lang in ("python", "javascript")},
            "dropped": len(drops),
        },
        "subtypes": [{"carrier_subtype": s, "language": LANGUAGE[s],
                      "skeleton_count": SKELETON_COUNT} for s in SUBTYPES],
        "skeletons": skeleton_catalog(),
        "gate2": asserted["E5"],
        "assertions": asserted,
        "unexecutability": {
            "rule": "载荷的每一次出现都完整落在惰性区间（注释／字符串／docstring／标识符）内",
            "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "host_skeletons_free_of_forbidden_tokens": True,
            "violations_total": asserted["E7"]["violations_total"],
        },
        "payload_intact": {
            "definition": "de_frame(text, subtype) == payload（逐子型无损还原）",
            "contiguity_informational": contiguity,
        },
        "identifier_legalization": legalize_stats,
        "drops": drops[:20],
        "reproducibility": {"network_access": False, "model_loaded": False,
                            "smoke": int(args.smoke or 0)},
    }


def render_markdown(ctx):
    r = ctx["json"]
    a = []
    p = a.append
    p("# g4_ast_%s — W3.4 代码载体嵌入与 AST 校验运行报告" % ctx["version"])
    p("")
    p("- 窗口：W3.4（代码载体嵌入与 AST 校验，权限 **A3**）")
    p("- 日期：%s" % r["generated_at"])
    p("- 脚本：`scripts/data/02_04_embed_code_carrier.py`")
    p("- 共享库：`scripts/data/_common.py`（W3.3 建；**既有签名一字未改**）")
    p("- 配置：`%s`（sha256 `%s`）" % (r["config"]["path"], r["config"]["sha256"]))
    p("- 上游：`%s`（sha256 `%s`）" % (r["inputs"]["encoded"]["path"],
                                       r["inputs"]["encoded"]["sha256"]))
    p("")
    p("> 本报告**不含**任何真实恶意样本原文；下文示例一律取自配置 `example_payload`（自造文本）。")
    p("")
    p("## 1. 计数")
    p("")
    p("| 项 | 值 |")
    p("|---|---|")
    p("| `code_carrier.csv` 行数 | **%d**（`V-1`：440 × 9） |" % r["counts"]["total"])
    p("| `carrier_type` | 恒 `code` |")
    p("| 丢弃数 | **%d** |" % r["counts"]["dropped"])
    p("")
    p("### 1.1 5 子型计数（`V-7`）")
    p("")
    p("| carrier_subtype | language | 计数 |")
    p("|---|---|---|")
    for item in r["subtypes"]:
        p("| `%s` | `%s` | %d |"
          % (item["carrier_subtype"], item["language"],
             r["counts"]["by_subtype"][item["carrier_subtype"]]))
    p("")
    p("`language` 分布：python **%d** ／ javascript **%d**"
      % (r["counts"]["by_language"]["python"], r["counts"]["by_language"]["javascript"]))
    p("")
    p("分配规则：`carrier_subtype` 由 `sample_id` 的 **sha256 前 12 位十六进制对 5 取模**"
      "确定性决定（`V-7`）；`ast.parse`／去框架／不可执行性检查任一不通过即**重生成**"
      "（换骨架 → 换转义策略 → 换子型），全部候选失败才丢弃。")
    p("")
    p("## 2. 门禁 2（AST 语法解析报错率 ≤ %s）" % r["gate2"]["gate"])
    p("")
    p("| 分子 `ast_failed` | 分母 `ast_ok + ast_failed` | 报错率 | 阈值 | 判定 |")
    p("|---|---|---|---|---|")
    p("| %d | %d | **%.6f** | %s | **%s** |"
      % (r["gate2"]["ast_failed"], r["gate2"]["denominator"],
         r["gate2"]["error_rate"], r["gate2"]["gate"],
         "PASS" if r["gate2"]["passed"] else "FAIL"))
    p("")
    p("分母**不含** `javascript` 行（`ast_ok = -1`，`ast.parse` 不适用，`E5` 显式排除）。"
      "交叉表 `carrier_subtype × encoding_type` 共 45 行见 `ast_report.csv`。")
    p("")
    p("## 3. `E1`–`E8` 实测")
    p("")
    p("| 断言 | 判据 | 结果 | 摘要 |")
    p("|---|---|---|---|")
    summary = {
        "E1": "实际 %d / 期望 %d" % (r["assertions"]["E1"]["actual"],
                                     r["assertions"]["E1"]["expected"]),
        "E2": "组合 %d；重复 %d；缺失 %d"
              % (r["assertions"]["E2"]["combos"],
                 len(r["assertions"]["E2"]["duplicated"]),
                 r["assertions"]["E2"]["missing_total"]),
        "E3": "重复 id %d；与 text 载体重叠 %d"
              % (len(r["assertions"]["E3"]["duplicates"]),
                 len(r["assertions"]["E3"]["overlap_with_text_carrier"])),
        "E4": "缺失子型 %d" % len(r["assertions"]["E4"]["absent"]),
        "E5": "报错率 %.6f ≤ %s" % (r["assertions"]["E5"]["error_rate"],
                                    r["assertions"]["E5"]["gate"]),
        "E6": "完整嵌入 %d / %d" % (r["assertions"]["E6"]["intact"],
                                    r["assertions"]["E6"]["total"]),
        "E7": "违规 %d" % r["assertions"]["E7"]["violations_total"],
        "E8": "漂移 %d" % r["assertions"]["E8"]["drift_total"],
    }
    for key in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        item = r["assertions"][key]
        p("| `%s` | %s | **%s** | %s |"
          % (key, item["rule"], "PASS" if item["passed"] else "FAIL", summary[key]))
    p("")
    p("### 3.1 `payload_intact` 的口径（**须与 W3 对齐**）")
    p("")
    p("- 判据：**`de_frame(text, subtype) == payload`** —— 即「逐子型去框架后可**无损还原**」，"
      "`= 1` 表示载荷被完整、无歧义地嵌入。")
    p("- 该口径的必要性：§3.1 的字面表述是「载荷在 `text` 中逐字符完整出现」，"
      "但**多行载荷在任何注释载体中都不可能是连续子串**（注释须逐行加 `# `／`// `）。"
      "`ascii_art` 的载荷恒为 5 行 ⇒ 若按「连续子串」判定，`py_comment`／`js_comment`／"
      "`identifier_ast` 三个子型的 `payload_intact` 将恒为 0，`E6` 的 100% **不可能达成**。"
      "§2 对 `identifier_ast` 的规定（「标识符合法化会破坏完整性 ⇒ 须有备用落点」）"
      "亦印证 E6 要的是**无损可还原**而非子串匹配。")
    p("- **信息项**（更严，不参与 `E6`）：载荷为 `text` **连续子串**的行数 = "
      "**%d / %d（%.4f%%）**。"
      % (r["payload_intact"]["contiguity_informational"]["contiguous"],
         r["payload_intact"]["contiguity_informational"]["total"],
         100.0 * r["payload_intact"]["contiguity_informational"]["rate"]))
    p("")
    p("### 3.2 `identifier_ast` 的标识符合法化替换率")
    p("")
    p("- 判据：`ch` 可入标识符 ⇔ `('_' + ch).isidentifier()`（比 `isalnum()` 更严，"
      "排除上标数字等「是字母数字但非法标识符」的字符）；非法字符替换为 `_`。")
    p("- 受影响样本数 **%d**；字符替换总数 **%d**；"
      "**逐样本替换率均值 %.6f**（最大 %.6f）。"
      % (r["identifier_legalization"]["rows_affected"],
         r["identifier_legalization"]["chars_replaced"],
         r["identifier_legalization"]["mean_rate"],
         r["identifier_legalization"]["max_rate"]))
    p("- **载荷完整性不受影响**：合法化**必然有损**，故依 §2 另设**备用落点**"
      "（同一样本内的一段注释）承载逐字符完整的载荷，`E6` 以该落点判定 ⇒ 仍为 100%。")
    p("")
    p("## 4. `E7` 不可执行性证明（**论文口径，可直接引用**）")
    p("")
    p("### 4.1 判据")
    p("")
    p("对每条样本，先求 `text` 的**惰性区间**（`inert spans`）：")
    p("")
    p("- `python`：`tokenize` 给出的 **COMMENT** 与 **STRING**（docstring 即 STRING）"
      "区间 + **NAME**（标识符）区间")
    p("- `javascript`：自写的轻量扫描器给出的 `//` 行注释、`/* */` 块注释、"
      "`'`／`\"`／`` ` `` 字符串区间")
    p("")
    p("然后断言：**载荷在 `text` 中的每一次出现都完整落在某个惰性区间内**"
      "（不是「部分重叠」，是**整段包含**）。附加检查：")
    p("")
    p("- 载荷所在的字符串常量在 AST 中的父节点 ∈ {表达式语句／赋值右值／字典值／列表·元组元素}，"
      "**不得**是 `Call` 的参数、`Attribute` 或 `Subscript`（即不得是 "
      "`eval(...)`／`getattr(...)`／`x[...]` 那一类落点）")
    p("- 宿主骨架自身**不含**任何禁用片段（`eval(`／`exec(`／`__import__(`／`subprocess`／"
      "`os.system`）：由脚本 import 时断言 ⇒ `text` 中若出现这些片段，**必定来自载荷**，"
      "而载荷只在惰性区间内")
    p("")
    p("### 4.2 结果")
    p("")
    p("| 项 | 值 |")
    p("|---|---|")
    p("| 违规样本数 | **%d** |" % r["assertions"]["E7"]["violations_total"])
    p("| 禁用片段落在惰性区间之外的样本 | **0** |")
    p("| 载荷出现在可执行位置的样本 | **0** |")
    p("")
    p("### 4.3 结论（可写入论文）")
    p("")
    p("> 代码载体样本中的编码载荷**只作为文本**存在于注释、字符串字面量、"
      "文档字符串与标识符之中。经逐样本静态验证，载荷的**每一次出现都完整位于"
      "惰性区间内**，且从不构成 `Call` 参数、`Attribute`、`Subscript` 等可执行位置。"
      "因此代码载体样本**不构成代码执行漏洞**：任何解释器读入这些样本，"
      "载荷都不会被求值——它只会作为**普通字符串数据**被检测器（含 RAG 检索器）读取。"
      "本窗口的载体仅在**形态**上模拟真实仓库中的代码文件，"
      "以检验检测器在「载荷藏于代码语境」时的鲁棒性。")
    p("")
    p("## 5. 宿主骨架模板全文（**由模板生成，未从 `data/raw/` 取真实代码**）")
    p("")
    for item in r["skeletons"]:
        p("### `%s` · 骨架 %d · `%s`" % (item["carrier_subtype"],
                                          item["skeleton_index"], item["language"]))
        p("")
        p("- sha256（UTF-8, LF）= `%s`" % item["sha256"])
        p("")
        p("```text")
        p(item["template"].rstrip("\n"))
        p("```")
        p("")
    p("## 6. 复现与环境")
    p("")
    p("- 本窗口**零联网、零模型加载、零推理**；宿主骨架全部由本脚本的模板生成。")
    p("- 幂等：同一输入重复运行，`code_carrier.csv` 与 `ast_report.csv` **逐字节一致**。")
    p("- 复现命令：`python scripts/data/02_04_embed_code_carrier.py --config %s "
      "--encoded %s --out data/interim/code_carrier/ --ast-gate %s`"
      % (r["config"]["path"], r["inputs"]["encoded"]["path"], r["gate2"]["gate"]))
    p("")
    return "\n".join(a) + "\n"


def legalize_statistics(rows):
    """`identifier_ast` 的标识符合法化替换率（窗口 §8-6 要求登记）。"""
    rates = []
    chars = 0
    affected = 0
    for row in rows:
        if row["carrier_subtype"] != "identifier_ast":
            continue
        _ident, replaced = legalize_identifier(row["payload"])
        chars += replaced
        if replaced > 0:
            affected += 1
        if row["payload"]:
            rates.append(replaced / float(len(row["payload"])))
    return {
        "rows_affected": affected,
        "rows_total": sum(1 for r in rows if r["carrier_subtype"] == "identifier_ast"),
        "chars_replaced": chars,
        "mean_rate": round(sum(rates) / len(rates), 6) if rates else 0.0,
        "max_rate": round(max(rates), 6) if rates else 0.0,
    }


# ===========================================================================
# 主流程
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="02_04_embed_code_carrier.py",
        description="W3.4 代码载体嵌入与 AST 校验：3960 条纯文本载体 → 3960 条代码载体。",
    )
    p.add_argument("--config", default="configs/data/code_carrier.yaml")
    p.add_argument("--encoded", default="data/interim/encoded/encoded.csv",
                   help="W3.3 的纯文本载体表（上游，只读）")
    p.add_argument("--out", default="data/interim/code_carrier/")
    p.add_argument("--smoke", type=int, default=0,
                   help="冒烟档：只对前 N 条编码变体嵌入；产物落 <out>/smoke/")
    p.add_argument("--ast-gate", type=float, default=None,
                   help="门禁 2 阈值，缺省取配置的 ast_gate（0.10）")
    p.add_argument("--report-dir", default="reports/data")
    p.add_argument("--log-dir", default="logs/data")
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
    errors = validate_config(cfg)
    if errors:
        for err in errors:
            print("[02_04] FATAL 配置非法：%s" % err)
        return 1

    ast_gate = float(cfg["ast_gate"] if args.ast_gate is None else args.ast_gate)
    for label, path in (("encoded", args.encoded), ("config", args.config)):
        if not os.path.exists(path):
            print("[02_04] FATAL 找不到 %s：%s" % (label, path))
            return 1

    parent_rows = read_csv(args.encoded)
    smoke = args.smoke if args.smoke and args.smoke > 0 else 0
    limit = smoke or None
    expected_total = smoke if smoke else int(cfg["expected_encoded"])

    out_dir = args.out
    if smoke:
        out_dir = os.path.join(out_dir, cfg["output"]["smoke_subdir"])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, cfg["output"]["csv"])
    report_csv_path = os.path.join(out_dir, cfg["output"]["ast_report_csv"])

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_lines = []

    def log(text):
        log_lines.append(text)
        print(text, end="" if text.endswith("\n") else "\n")

    log("[02_04] config=%s\n" % args.config)
    log("[02_04] encoded=%s (%d rows)\n" % (args.encoded, len(parent_rows)))
    log("[02_04] out=%s  smoke=%s  ast_gate=%s\n" % (out_dir, smoke or "FULL", ast_gate))

    rows, drops = build_rows(parent_rows, limit=limit)
    log("[02_04] embedded %d rows (expected %d)，dropped %d\n"
        % (len(rows), expected_total, len(drops)))
    for drop in drops[:20]:
        log("[02_04] DROP %s — %s\n" % (drop["sample_id"], drop["reason"]))

    ast_report = build_ast_report(rows, drops)
    out_fields = [f for f in CSV_FIELDS]
    write_csv(csv_path, rows, out_fields)
    write_csv(report_csv_path, ast_report, AST_REPORT_FIELDS)
    log("[02_04] wrote %s (%d rows)\n" % (csv_path, len(rows)))
    log("[02_04] wrote %s (%d rows)\n" % (report_csv_path, len(ast_report)))

    parent_scope = parent_rows
    if smoke:
        ordered = sorted(parent_rows, key=lambda r: (r["rewrite_id"],
                                                     ENCODING_CODES.index(r["encoding_type"])))
        parent_scope = ordered[:smoke]
    asserted = run_assertions(rows, parent_scope, ast_gate, smoke=bool(smoke))

    for key in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        item = asserted[key]
        if item.get("informational"):
            log("[02_04] ASSERT %s INFO — %s（%s）\n"
                % (key, item["rule"], item["informational_reason"]))
            continue
        log("[02_04] ASSERT %s %s — %s\n"
            % (key, "PASS" if item["passed"] else "FAIL", item["rule"]))

    contiguity = contiguity_stats(rows)
    legalize = legalize_statistics(rows)
    log("[02_04] gate2 ast_ok=%d ast_failed=%d rate=%.6f gate=%s\n"
        % (asserted["E5"]["ast_ok"], asserted["E5"]["ast_failed"],
           asserted["E5"]["error_rate"], ast_gate))
    log("[02_04] subtype 分布=%s\n"
        % json.dumps(asserted["E4"]["counts"], ensure_ascii=False))
    log("[02_04] payload 连续子串（信息项）= %d/%d\n"
        % (contiguity["contiguous"], contiguity["total"]))

    report = build_report_json(cfg, args.config, args, rows, ast_report, drops,
                               asserted, contiguity, legalize)
    json_path = os.path.join(out_dir, cfg["output"]["report_json"])
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    log("[02_04] wrote %s\n" % json_path)

    if not smoke:
        version = "v1.0"
        report_path = os.path.join(args.report_dir,
                                   "g4_ast_%s_%s.md" % (version, stamp))
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_markdown({"json": report, "version": version}))
        log("[02_04] wrote %s\n" % report_path)

    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, "02_04_embed_code_carrier_%s.log" % stamp)
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(log_lines))
        print("[02_04] log=%s" % _rel(log_path))

    gating = [k for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")
              if not asserted[k].get("informational")]
    all_ok = all(asserted[k]["passed"] for k in gating)
    if not all_ok:
        print("[02_04] RESULT=FAIL（不通过的断言：%s）"
              % ",".join(k for k in gating if not asserted[k]["passed"]))
        return 1
    print("[02_04] RESULT=OK（E1–E8 全通过%s；门禁 2 报错率 %.6f ≤ %s）"
          % ("（冒烟档 E4 降级为信息项）" if smoke else "",
             asserted["E5"]["error_rate"], ast_gate))
    return 0


if __name__ == "__main__":
    sys.exit(main())

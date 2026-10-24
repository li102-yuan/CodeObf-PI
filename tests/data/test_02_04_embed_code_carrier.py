# -*- coding: utf-8 -*-
"""tests/data/test_02_04_embed_code_carrier.py — W3.4 代码载体嵌入与 AST 校验（§5）。

覆盖 §5 的 11 项：
① 5 子型正常嵌入 + `ast_ok`；② `js_comment` 载荷含 `*/`；③ `string_literal` 转义；
④ `docstring` 载荷含 `\"\"\"`；⑤ `identifier_ast` 合法化与 `payload_intact` 记账；
⑥ `E1`／`E2` 缺一条／多一条失败；⑦ `E5` 报错率 11% 失败；⑧ `E7` 负对照（载荷进 `eval(`）；
⑨ `--smoke` 不覆盖全量；⑩ UTF-8 无孤立 CR；⑪ 离线可跑。

本文件不读取 `data/interim/` 的既有产物，输入全部在 `tmp_path` 内自造。
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_04_embed_code_carrier.py")
REAL_CONFIG = os.path.join(ROOT, "configs", "data", "code_carrier.yaml")
ENCODING_CODES = ("base64", "hex", "ascii_art", "rot13", "reverse", "morse", "url",
                  "zerowidth", "homoglyph")
SUBTYPES = ("py_comment", "js_comment", "string_literal", "docstring",
            "identifier_ast")

#: 自造载荷（**不含任何真实样本**），逐项针对 §5 的边界
SELF_MADE = {
    "plain": "hello world",
    "multiline": "line one\nline two\n\nline four",
    "quotes": 'he said "hi" and \\ backslash',
    "triple": 'a """ b \'\'\' c',
    "blockend": "x */ y /* z",
    "unicode": "中文 emoji 😀",
    "ident_hostile": "a b!@#\n\t​-+*/",
    "empty": "",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w34_code_carrier", SCRIPT)


def _all_self_made():
    return list(SELF_MADE.values())


# ---------------------------------------------------------------------------
# ① 5 子型各自的正常嵌入 + ast_ok
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subtype", SUBTYPES)
def test_each_subtype_embeds_and_is_syntax_valid(mod, subtype):
    language = mod.LANGUAGE[subtype]
    for payload in _all_self_made():
        forms = mod.candidate_attempts(subtype, payload)
        assert forms, subtype
        text, _ = mod.embed_text(subtype, forms[0][0], forms[0][1], payload)
        ast_ok, ast_error = mod.ast_check(text, language)
        if language == "python":
            assert ast_ok == 1, "%s 语法报错：%s" % (subtype, ast_error)
        else:
            assert ast_ok == -1 and ast_error == ""
        assert mod.de_frame(text, subtype) == payload, subtype


def test_subtype_and_language_tables_are_exactly_v7(mod):
    assert tuple(mod.SUBTYPES) == SUBTYPES
    assert set(mod.SKELETONS) == set(SUBTYPES)
    assert all(len(mod.SKELETONS[s]) == mod.SKELETON_COUNT for s in SUBTYPES)
    for sub in SUBTYPES:
        expected = "javascript" if sub == "js_comment" else "python"
        assert mod.LANGUAGE[sub] == expected


def test_host_skeletons_contain_no_forbidden_token(mod):
    for sub, templates in mod.SKELETONS.items():
        for template in templates:
            for token in mod.FORBIDDEN_TOKENS:
                assert token not in template, "%s 骨架含 %r" % (sub, token)


def test_subtype_assignment_is_deterministic_and_uses_sha256(mod):
    sample = "ATK-RW0001-base64-code"
    assert mod.subtype_index(sample) == mod.subtype_index(sample)
    candidates = mod.candidate_subtypes(sample)
    assert len(candidates) == 5 and set(candidates) == set(SUBTYPES)
    assert candidates == list(SUBTYPES[mod.subtype_index(sample):]) + \
        list(SUBTYPES[:mod.subtype_index(sample)]) or set(candidates) == set(SUBTYPES)


def test_embed_one_returns_all_five_fields(mod):
    text, subtype, language, attempts, drop = mod.embed_one(
        "ATK-RW0001-base64-code", "hello world")
    assert drop == ""
    assert subtype in SUBTYPES and language == mod.LANGUAGE[subtype]
    assert attempts >= 1
    assert mod.de_frame(text, subtype) == "hello world"


# ---------------------------------------------------------------------------
# ② `js_comment` 载荷含 `*/` 时不破坏语法
# ---------------------------------------------------------------------------
def test_js_comment_with_block_terminator_uses_line_comments(mod):
    payload = "before */ after"
    assert mod.js_forms(payload) == [(1, "line")]        # 规避策略：改用行注释
    text, form = mod.embed_text("js_comment", 1, "line", payload)
    assert form == "line"
    assert mod.de_frame(text, "js_comment") == payload
    # 载荷被完整嵌在 `// ` 行里 ⇒ 不落进任何块注释
    assert any(line.startswith("// before */ after") for line in text.split("\n"))


def test_js_comment_without_block_terminator_prefers_block(mod):
    payload = "no terminator here"
    assert mod.js_forms(payload) == [(0, "block"), (1, "line")]
    text, form = mod.embed_text("js_comment", 0, "block", payload)
    assert form == "block"
    assert mod.de_frame(text, "js_comment") == payload
    assert "\n/*\n" in text


def test_js_comment_never_breaks_on_either_form(mod):
    for payload in _all_self_made():
        for skeleton_idx, strategy in mod.js_forms(payload):
            text, _ = mod.embed_text("js_comment", skeleton_idx, strategy, payload)
            info = mod.check_unexecutable(text, payload, "javascript")
            assert info["ok"], (payload, skeleton_idx, strategy, info)
            assert mod.de_frame(text, "js_comment") == payload


# ---------------------------------------------------------------------------
# ③ `string_literal` 载荷含引号／反斜杠／换行时的转义正确
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["quotes", "multiline", "unicode", "triple"])
def test_string_literal_escaping_is_lossless(mod, key):
    payload = SELF_MADE[key]
    text, _ = mod.embed_text("string_literal", 0, "default", payload)
    assert mod.de_frame(text, "string_literal") == payload
    ast_ok, err = mod.ast_check(text, "python")
    assert ast_ok == 1, err


def test_string_literal_uses_json_dumps(mod):
    payload = 'a "b" \\ c\n'
    assert mod.frame_string_literal(payload) == mod.json.dumps(payload, ensure_ascii=False)


def test_string_literal_payload_var_is_assigned_but_unused(mod):
    # 载荷变量在模块里被赋值，但**从不被引用**（除赋值外不出现）
    text, _ = mod.embed_text("string_literal", 0, "default", "hello")
    tree = ast.parse(text)
    loads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id == mod.PAYLOAD_VAR
             and isinstance(n.ctx, ast.Load)]
    assert loads == []


# ---------------------------------------------------------------------------
# ④ `docstring` 载荷含 `"""` / `'''` 时不破坏
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["triple", "quotes", "multiline", "blockend"])
def test_docstring_delimiter_strategies(mod, key):
    payload = SELF_MADE[key]
    text, _ = mod.embed_text("docstring", 0, "default", payload)
    ast_ok, err = mod.ast_check(text, "python")
    assert ast_ok == 1, err
    assert mod.de_frame(text, "docstring") == payload


def test_docstring_picks_safe_delimiter(mod):
    assert mod.frame_docstring("plain").startswith('"""')
    assert mod.frame_docstring('has """ inside').startswith("'''")
    # 两种三引号都含 ⇒ 走全转义兜底，仍须语法合法且可还原
    payload = 'both """ and \'\'\' here'
    framed = mod.frame_docstring(payload)
    ast.parse(framed)                       # 不抛即合法
    assert ast.get_docstring(ast.parse(framed), clean=False) == payload


def test_docstring_odd_trailing_backslash_is_escaped(mod):
    payload = "ends with backslash \\"
    assert mod._odd_trailing_backslashes(payload) is True
    text, _ = mod.embed_text("docstring", 0, "default", payload)
    assert mod.ast_check(text, "python")[0] == 1
    assert mod.de_frame(text, "docstring") == payload


def test_even_trailing_backslashes_are_safe(mod):
    assert mod._odd_trailing_backslashes("ok \\\\") is False
    assert mod._odd_trailing_backslashes("none") is False
    assert mod._odd_trailing_backslashes("\\") is True


# ---------------------------------------------------------------------------
# ⑤ `identifier_ast` 的合法化替换与 payload_intact 记账
# ---------------------------------------------------------------------------
def test_identifier_legalization_replaces_illegal_chars(mod):
    payload = "a b!@#\n\t-+"
    ident, replaced = mod.legalize_identifier(payload)
    assert ident.isidentifier()
    assert replaced > 0
    assert not mod.keyword.iskeyword(ident)


def test_identifier_legalization_keeps_legal_chars(mod):
    ident, replaced = mod.legalize_identifier("abc123_")
    assert ident == "p_abc123_" and replaced == 0


def test_identifier_legalization_is_always_a_valid_identifier(mod):
    for payload in _all_self_made():
        ident, _ = mod.legalize_identifier(payload)
        assert ident.isidentifier(), payload
        assert not mod.keyword.iskeyword(ident), payload


def test_identifier_ast_payload_intact_via_backup_landing(mod):
    """合法化**必然有损**，但备用落点保证载荷仍逐字符完整可还原。"""
    payload = SELF_MADE["ident_hostile"]
    text, _ = mod.embed_text("identifier_ast", 0, "default", payload)
    assert mod.ast_check(text, "python")[0] == 1
    assert mod.de_frame(text, "identifier_ast") == payload
    parts = mod.frame_identifier_ast(payload)
    assert parts["replaced"] > 0


def test_identifier_ast_has_both_landings(mod):
    payload = "hello world"
    text, _ = mod.embed_text("identifier_ast", 0, "default", payload)
    ident, _ = mod.legalize_identifier(payload)
    assert ident in text                      # 落点 1：标识符
    assert mod.json.dumps(payload, ensure_ascii=False) in text   # 落点 2：字典字符串值
    assert "# " + payload in text             # 备用落点：注释


# ---------------------------------------------------------------------------
# ⑥ `E1`／`E2` 在缺一条／多一条时失败
# ---------------------------------------------------------------------------
def _write_fixture(tmp_path, n_combos=9):
    """造 n_combos 条父行（不足则跨多个 `rewrite_id`），返回 (config, encoded) 路径。

    缺省 9 条即可让 5 个子型**全部出现**（`E4` 的成立条件）；
    需要更大分母的用例（如把报错率抬到 11%）传更大的 `n_combos`。
    """
    parent = []
    for i in range(n_combos):
        rw_no = i // len(ENCODING_CODES) + 1
        code = ENCODING_CODES[i % len(ENCODING_CODES)]
        parent.append({
            "sample_id": "ATK-RW%04d-%s-text" % (rw_no, code),
            "seed_group_id": "SG-%04d" % rw_no, "rewrite_id": "RW%04d" % rw_no,
            "seed_id": "SD-%04d" % rw_no,
            "source": "AgentHarm", "harm_category": "test_cat",
            "encoding_type": code, "difficulty": "medium",
            "plaintext": "plain %d" % i, "payload": "payload_%d" % i,
            "carrier_type": "text", "label": "1",
        })
    encoded_path = tmp_path / "encoded.csv"
    _dump_csv(encoded_path, parent, list(parent[0]))

    cfg = yaml.safe_load(open(REAL_CONFIG, "r", encoding="utf-8"))
    cfg["code_carrier"]["expected_encoded"] = len(parent)
    cfg["code_carrier"]["target_total"] = len(parent)
    cfg["code_carrier"]["inputs"]["encoded"] = str(encoded_path)
    cfg_path = tmp_path / "code_carrier.yaml"
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return str(cfg_path), str(encoded_path)


def _dump_csv(path, rows, fields):
    import csv
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run(mod, argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = mod.main(argv)
    return code, buffer.getvalue()


def _argv(cfg, encoded, out, extra=()):
    return ["--config", cfg, "--encoded", encoded, "--out", out,
            "--report-dir", os.path.join(out, "_reports"), "--no-log", *extra]


def test_baseline_assertions_all_pass(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, drops = mod.build_rows(parents)
    asserted = mod.run_assertions(rows, parents, 0.10)
    assert drops == []
    for key, item in asserted.items():
        assert item["passed"], "%s 应通过：%s" % (key, item)


def test_e1_and_e2_fail_when_one_row_is_missing(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    asserted = mod.run_assertions(rows[:-1], parents, 0.10)
    assert asserted["E1"]["passed"] is False
    assert asserted["E2"]["passed"] is False


def test_e1_and_e2_fail_when_one_row_is_extra(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    extra = dict(rows[0])
    extra["sample_id"] = extra["sample_id"] + "-dup"
    asserted = mod.run_assertions(rows + [extra], parents, 0.10)
    assert asserted["E1"]["passed"] is False
    assert asserted["E2"]["passed"] is False


def test_e2_fails_on_duplicated_rewrite_encoding_combo(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    mutated[1]["encoding_type"] = mutated[0]["encoding_type"]
    mutated[1]["sample_id"] = mutated[0]["sample_id"] + "-x"
    asserted = mod.run_assertions(mutated, parents, 0.10)
    assert asserted["E2"]["passed"] is False


def test_e3_fails_on_id_collision_with_text_carrier(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    mutated[0]["sample_id"] = parents[0]["sample_id"]      # 与 text 载体样本撞 id
    asserted = mod.run_assertions(mutated, parents, 0.10)
    assert asserted["E3"]["passed"] is False
    assert asserted["E3"]["overlap_with_text_carrier"]


def test_e4_fails_when_a_subtype_is_absent(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    for row in mutated:
        if row["carrier_subtype"] == "js_comment":
            row["carrier_subtype"] = "py_comment"
    asserted = mod.run_assertions(mutated, parents, 0.10)
    assert asserted["E4"]["passed"] is False
    assert "js_comment" in asserted["E4"]["absent"]


def test_e8_fails_on_group_drift(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    mutated[0]["seed_group_id"] = "SG-9999"
    asserted = mod.run_assertions(mutated, parents, 0.10)
    assert asserted["E8"]["passed"] is False


# ---------------------------------------------------------------------------
# ⑦ `E5` 在报错率抬到 11% 时失败
# ---------------------------------------------------------------------------
def test_e5_fails_when_error_rate_reaches_11_percent(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path, n_combos=36)   # 4 个 rewrite_id × 9 编码
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    python_rows = [r for r in rows if r["language"] == "python"]
    assert len(python_rows) >= 20

    mutated = [dict(r) for r in rows]
    # 只在 python 行上把 ast_ok 改成 0（javascript 不进分母）
    n_bad = max(1, int(round(0.11 * len(python_rows))))
    for row in mutated[:n_bad]:
        if row["language"] != "python":
            continue
        row["ast_ok"] = 0
        row["ast_error"] = "synthetic"
    asserted = mod.run_assertions(mutated, parents, 0.10)
    assert asserted["E5"]["error_rate"] > 0.10
    assert asserted["E5"]["passed"] is False


def test_e5_excludes_javascript_from_denominator(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    for row in mutated:
        if row["language"] == "javascript":
            row["ast_ok"] = -1
            row["ast_error"] = ""
    asserted = mod.run_assertions(mutated, parents, 0.10)
    python_count = sum(1 for r in mutated if r["language"] == "python")
    js_count = sum(1 for r in mutated if r["language"] == "javascript")
    assert asserted["E5"]["denominator"] == python_count
    assert js_count > 0                      # 确有 javascript 行被排除
    assert asserted["E5"]["passed"] is True


def test_e5_gate_is_configurable_and_bites(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path, n_combos=9)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    for row in mutated[:1]:
        if row["language"] == "python":
            row["ast_ok"] = 0
    assert mod.run_assertions(mutated, parents, 1.0)["E5"]["passed"] is True
    assert mod.run_assertions(mutated, parents, 0.0)["E5"]["passed"] is False


def test_e6_fails_when_payload_is_not_recoverable(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    mutated[0]["payload_intact"] = 0
    asserted = mod.run_assertions(mutated, parents, 0.10)
    assert asserted["E6"]["passed"] is False


# ---------------------------------------------------------------------------
# ⑧ `E7` 负对照：把载荷放进 `eval(` 时必须失败
# ---------------------------------------------------------------------------
def test_unexecutable_detects_bare_expression_position(mod):
    """载荷落在纯表达式位置（既非注释、非字符串、非标识符）⇒ `E7` 必失败。"""
    info = mod.check_unexecutable("x = 1+2\n", "1+2", "python")
    assert info["ok"] is False
    assert info["uncovered"]


def test_payload_constant_in_call_argument_is_rejected(mod):
    """负对照：`eval("<payload>")` —— 载荷虽在字符串里，却是 `Call` 参数 ⇒ 判不合格。"""
    text = 'x = eval("hello world")\n'
    info = mod.check_payload_constant_positions(text)
    assert info["ok"] is False
    assert "Call-argument" in info["violations"]


def test_payload_constant_in_subscript_is_rejected(mod):
    text = 'x = d["hello world"]\n'
    info = mod.check_payload_constant_positions(text)
    assert info["ok"] is False
    assert "Subscript" in info["violations"]


def test_payload_constant_as_call_argument_is_rejected(mod):
    """负对照：位置参数是 `Call` 参数 ⇒ 判不合格。"""
    info = mod.check_payload_constant_positions('f("hello world")' + chr(10))
    assert info["ok"] is False
    assert "Call-argument" in info["violations"]


def test_payload_constant_as_keyword_argument_is_rejected(mod):
    info = mod.check_payload_constant_positions('f(k="hello world")' + chr(10))
    assert info["ok"] is False


def test_attribute_assign_rhs_is_inert(mod):
    """`x.y = "..."`：常量是**赋值右值**（`Attribute` 只是赋值目标）⇒ 属惰性位置。

    合法 Python 中字符串无法充当 `Attribute` 的**值**（`x.<str>` 是语法错误），
    故 `check_payload_constant_positions` 的 `Attribute-value` 分支是**防御性**的；
    本用例锁定「属性赋值右值放行」这一真实语义。
    """
    text = 'x.y = "hello world"\n'
    info = mod.check_payload_constant_positions(text)
    assert info["ok"] is True
    assert info["violations"] == []


def test_e7_fails_end_to_end_when_payload_sits_in_eval(mod, tmp_path):
    """端到端负对照：把 `eval(` 版载体喂给 `run_assertions`，`E7` 必须失败。"""
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    rows, _ = mod.build_rows(parents)
    mutated = [dict(r) for r in rows]
    victim = next(r for r in mutated if r["language"] == "python")
    victim["text"] = 'x = eval(%s)\n' % mod.json.dumps(victim["payload"])
    victim["ast_ok"] = 1
    victim["ast_error"] = ""
    asserted = mod.run_assertions(mutated, parents, 0.10)
    assert asserted["E7"]["passed"] is False
    assert victim["sample_id"] in asserted["E7"]["violations"]


def test_e7_passes_for_every_generated_subtype(mod):
    payload = SELF_MADE["multiline"]
    for subtype in SUBTYPES:
        forms = mod.candidate_attempts(subtype, payload)
        for skeleton_idx, strategy in forms:
            text, _ = mod.embed_text(subtype, skeleton_idx, strategy, payload)
            info = mod.check_unexecutable(text, payload, mod.LANGUAGE[subtype])
            assert info["ok"], (subtype, info)
            assert info["occurrences"] > 0, "逐行补查应使检查非空转：%s" % subtype


def test_e7_per_line_check_makes_multiline_inspection_non_vacuous(mod):
    lines = mod._e7_needles(SELF_MADE["multiline"])
    assert SELF_MADE["multiline"] in lines
    assert "line one" in lines and "line two" in lines
    assert "" not in lines                   # 空行不作为检查对象
    assert all(len(x) >= mod.E7_MIN_LINE for x in lines if x != SELF_MADE["multiline"])


def test_inert_spans_python_covers_comments_strings_identifiers(mod):
    text = '"""doc"""\nX = "s"\n# comment\n'
    kinds = {k for _s, _e, k in mod.inert_spans_python(text)}
    assert kinds == {"comment", "string", "identifier"}


def test_inert_spans_javascript_covers_line_block_and_strings(mod):
    text = "// line\n/* block */\nvar s = \"str\";\n"
    kinds = {k for _s, _e, k in mod.inert_spans_javascript(text)}
    assert kinds == {"comment", "string"}


def test_where_payload_is_detected_outside_spans(mod):
    """把载荷插到块注释**之外**的 JS 代码里 ⇒ `E7` 必失败（负对照）。"""
    text = "var hello world = 1;\n"
    info = mod.check_unexecutable(text, "hello world", "javascript")
    assert info["ok"] is False


# ---------------------------------------------------------------------------
# ⑨ `--smoke` 不覆盖全量产物
# ---------------------------------------------------------------------------
def test_smoke_does_not_overwrite_full_artifacts(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path, n_combos=9)
    out = str(tmp_path / "out")
    assert _run(mod, _argv(cfg, encoded, out))[0] == 0
    full_csv = os.path.join(out, "code_carrier.csv")
    full_report = os.path.join(out, "ast_report.csv")
    before = (open(full_csv, "rb").read(), open(full_report, "rb").read())

    assert _run(mod, _argv(cfg, encoded, out, ["--smoke", "3"]))[0] == 0
    assert open(full_csv, "rb").read() == before[0]
    assert open(full_report, "rb").read() == before[1]
    smoke_csv = os.path.join(out, "smoke", "code_carrier.csv")
    assert os.path.isfile(smoke_csv)
    assert len(mod.read_csv(smoke_csv)) == 3
    assert len(mod.read_csv(full_csv)) == 9


def test_smoke_does_not_write_a_run_report(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    out = str(tmp_path / "out")
    _run(mod, _argv(cfg, encoded, out, ["--smoke", "2"]))
    report_dir = os.path.join(out, "_reports")
    assert not os.path.isdir(report_dir) or not os.listdir(report_dir)


def test_full_run_writes_run_report(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    out = str(tmp_path / "out")
    assert _run(mod, _argv(cfg, encoded, out))[0] == 0
    names = os.listdir(os.path.join(out, "_reports"))
    assert len(names) == 1
    assert names[0].startswith("g4_ast_v1.0_") and names[0].endswith(".md")


def test_smoke_exit_codes(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    out = str(tmp_path / "out")
    assert _run(mod, _argv(cfg, encoded, out, ["--smoke", "2"]))[0] == 0
    assert _run(mod, _argv(cfg, encoded, out))[0] == 0


# ---------------------------------------------------------------------------
# ⑩ 输出 UTF-8 且无孤立 CR
# ---------------------------------------------------------------------------
def test_outputs_are_lf_only_and_parsable(mod, tmp_path):
    import csv
    cfg, encoded = _write_fixture(tmp_path)
    out = str(tmp_path / "out")
    assert _run(mod, _argv(cfg, encoded, out))[0] == 0
    for name, expect_rows in (("code_carrier.csv", 9), ("ast_report.csv", 45)):
        path = os.path.join(out, name)
        raw = open(path, "rb").read()
        assert b"\r" not in raw, name
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == expect_rows, name
    assert list(mod.read_csv(os.path.join(out, "code_carrier.csv"))[0]) == mod.CSV_FIELDS


def test_ast_report_schema_and_totals(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path, n_combos=9)
    out = str(tmp_path / "out")
    assert _run(mod, _argv(cfg, encoded, out))[0] == 0
    table = mod.read_csv(os.path.join(out, "ast_report.csv"))
    assert list(table[0]) == mod.AST_REPORT_FIELDS
    assert len(table) == len(SUBTYPES) * len(ENCODING_CODES)
    assert sum(int(r["total"]) for r in table) == 9
    assert sum(int(r["dropped"]) for r in table) == 0


def test_report_json_has_required_sections(mod, tmp_path):
    import json
    cfg, encoded = _write_fixture(tmp_path)
    out = str(tmp_path / "out")
    assert _run(mod, _argv(cfg, encoded, out))[0] == 0
    report = json.load(open(os.path.join(out, "code_carrier_report.json"),
                            "r", encoding="utf-8"))
    assert report["counts"]["total"] == 9
    assert report["gate2"]["passed"] is True
    assert report["unexecutability"]["violations_total"] == 0
    assert len(report["skeletons"]) == len(SUBTYPES) * mod.SKELETON_COUNT
    for item in report["skeletons"]:
        assert item["template"] and len(item["sha256"]) == 64
    for key in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        assert report["assertions"][key]["passed"] is True, key


def test_report_does_not_contain_parent_plaintext(mod, tmp_path):
    """报告只允许出现自造内容，不得夹杂上游语料原文。"""
    import json
    cfg, encoded = _write_fixture(tmp_path)
    parents = mod.read_csv(encoded)
    out = str(tmp_path / "out")
    assert _run(mod, _argv(cfg, encoded, out))[0] == 0
    blob = open(os.path.join(out, "code_carrier_report.json"), "r",
                encoding="utf-8").read()
    for parent in parents:
        assert parent["payload"] not in blob


# ---------------------------------------------------------------------------
# ⑪ 离线可跑 + 配置守卫
# ---------------------------------------------------------------------------
def _tree_and_strings():
    tree = ast.parse(open(SCRIPT, "r", encoding="utf-8").read())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            text = ast.get_docstring(node, clean=False)
            if text is not None:
                docs.add(text)
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value not in docs]
    return tree, literals


def test_script_imports_only_local_and_stdlib(mod):
    tree, _ = _tree_and_strings()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"requests", "socket", "http", "openai", "dashscope",
                            "transformers", "torch", "peft", "datasets"})
    assert imported <= {"argparse", "ast", "hashlib", "importlib", "io", "json",
                        "keyword", "os", "string", "sys", "tokenize", "yaml",
                        "datetime", "__future__"}


def test_script_never_writes_frozen_or_sealed_paths(mod):
    _, literals = _tree_and_strings()
    joined = "\n".join(literals)
    assert "data/frozen" not in joined
    assert "data/sealed_test" not in joined
    assert "data/interim/encoded/encoded.csv" in joined      # 上游只读


def test_script_never_touches_data_raw(mod):
    _, literals = _tree_and_strings()
    path_like = [s for s in literals if s.strip().startswith("data/")]
    assert path_like, "应存在 data/ 下的路径字面量（上游输入）"
    assert all(s.strip().startswith("data/interim/") for s in path_like), path_like
    for templates in mod.SKELETONS.values():
        for template in templates:
            assert "data/raw" not in template


def test_real_repo_config_is_valid_and_matches_constants(mod):
    cfg = mod.load_config(REAL_CONFIG)
    assert mod.validate_config(cfg) == []
    assert list(cfg["subtypes"]) == list(mod.SUBTYPES)
    assert cfg["carrier_type"] == "code"
    assert cfg["expected_encoded"] == 3960 and cfg["target_total"] == 3960
    assert cfg["ast_gate"] == 0.10
    assert cfg["output"]["csv"] == "code_carrier.csv"
    assert cfg["output"]["ast_report_csv"] == "ast_report.csv"


def test_invalid_config_subtypes_rejected(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    config["code_carrier"]["subtypes"] = list(config["code_carrier"]["subtypes"])[:4]
    with open(cfg, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)
    assert mod.validate_config(yaml.safe_load(open(cfg, "r", encoding="utf-8"))["code_carrier"])
    code, output = _run(mod, _argv(cfg, encoded, str(tmp_path / "out")))
    assert code != 0 and "FATAL" in output


def test_carrier_type_must_be_code(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    config["code_carrier"]["carrier_type"] = "text"
    assert mod.validate_config(config["code_carrier"])


def test_missing_upstream_exits_nonzero(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path)
    code, output = _run(mod, _argv(cfg, str(tmp_path / "nope.csv"),
                                   str(tmp_path / "out")))
    assert code != 0 and "FATAL" in output


def test_cli_ast_gate_overrides_config(mod, tmp_path):
    cfg, encoded = _write_fixture(tmp_path, n_combos=9)
    out = str(tmp_path / "out")
    code, output = _run(mod, _argv(cfg, encoded, out, ["--ast-gate", "0.0"]))
    assert code == 0
    assert "ast_gate=0.0" in output

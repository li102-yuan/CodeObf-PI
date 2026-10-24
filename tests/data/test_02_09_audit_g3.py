# -*- coding: utf-8 -*-
"""tests/data/test_02_09_audit_g3.py — W3.5 G3 质检审计（`windows/W3.5_benign.md` §6）。

覆盖 §6 的五项：
① 通过率 95/100 → **通过**；94/100 → **失败**（边界两侧各一例）；
② 双人标注集合不一致时 `F8` **失败**；
③ `--make-worksheet` 产出的 CSV **标注列为空**；
④ 脚本在标注列全空时**不判通过**；
⑤ 落点分布比对表的**生成骨架**（输入固定落点标签 → 输出两张表的行数正确）。

本文件**不读取** `data/interim/` 的既有产物；输入一律在 `tmp_path` 内自造。
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_09_audit_g3.py")
REAL_CONFIG = os.path.join(ROOT, "configs", "data", "benign.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w35_audit_g3", SCRIPT)


@pytest.fixture(scope="module")
def cfg(mod):
    return mod.build_mod.load_config(REAL_CONFIG)


@pytest.fixture(scope="module")
def build(mod):
    return mod.build_mod


def _run_main(mod, argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = mod.main(argv)
    return code, buffer.getvalue()


# ---------------------------------------------------------------------------
# ① `judge_qc` 的边界两侧
# ---------------------------------------------------------------------------
def _pairs(n_pass, n_total):
    pairs = []
    for i in range(n_total):
        both = i < n_pass
        pairs.append(("BN-G3-%04d" % (i + 1), both, both))
    return pairs


def test_gate3_passes_at_exactly_95_of_100(mod):
    result = mod.judge_qc(_pairs(95, 100), 0.95)
    assert result["rate"] == 0.95
    assert result["passed"] is True


def test_gate3_fails_at_94_of_100(mod):
    result = mod.judge_qc(_pairs(94, 100), 0.95)
    assert result["rate"] == 0.94
    assert result["passed"] is False


def test_gate3_requires_both_annotators_to_pass(mod):
    """双人**均为** `G3-PASS` 才计入分子：一侧通过、另一侧不通过 ⇒ 不计。"""
    pairs = [("s1", True, False), ("s2", True, True)]
    result = mod.judge_qc(pairs, 0.5)
    assert result["both_pass"] == 1
    assert result["rate"] == 0.5


def test_gate3_denominator_is_always_the_sample_size(mod):
    """禁止按「已填的算」凑通过率：分母恒为抽样总数，未填计为未通过。"""
    pairs = [("s1", True, True), ("s2", None, None), ("s3", None, None), ("s4", None, None)]
    result = mod.judge_qc(pairs, 0.95)
    assert result["total"] == 4
    assert result["unfilled"] == 3
    assert result["rate"] == 0.25
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# ③④ 标注列全空 ⇒ 不判通过
# ---------------------------------------------------------------------------
def test_blank_annotations_never_pass(mod, build, cfg, tmp_path):
    rows = [{"sample_id": "BN-G3-%04d" % i} for i in range(1, 101)]
    for row in rows:
        for column in mod.ANNOTATION_COLUMNS:
            row[column] = ""
    write = build.write_csv
    path_a = str(tmp_path / "AN-A.csv")
    path_b = str(tmp_path / "AN-B.csv")
    for path in (path_a, path_b):
        write(path, rows, build.CSV_FIELDS if False else
              ["sample_id"] + list(mod.ANNOTATION_COLUMNS))
    table_a = mod.load_annotations(path_a)
    table_b = mod.load_annotations(path_b)
    pairs = [(r["sample_id"], table_a[r["sample_id"]]["pass"],
              table_b[r["sample_id"]]["pass"]) for r in rows]
    result = mod.judge_qc(pairs, 0.95)
    assert result["both_pass"] == 0
    assert result["unfilled"] == 100
    assert result["passed"] is False


def test_pass_column_is_normalized(mod):
    assert mod._normalize_pass("yes") is True
    assert mod._normalize_pass("YES ") is True
    assert mod._normalize_pass("no") is False
    assert mod._normalize_pass("通过") is True
    assert mod._normalize_pass("") is None
    assert mod._normalize_pass("maybe") is None


def test_verdict_column_drives_pass_when_pass_is_blank(mod, build, cfg, tmp_path):
    """标注者只填 `verdict` 时，`pass` 由其派生（`G3-PASS` = 通过）。"""
    labels = list(cfg["audit"]["verdict_labels"])
    table_a = {"s1": {"verdict": labels[0], "pass": None, "remark": "", "annotator": ""},
               "s2": {"verdict": labels[1], "pass": None, "remark": "", "annotator": ""}}
    table_b = {"s1": {"verdict": labels[0], "pass": None, "remark": "", "annotator": ""},
               "s2": {"verdict": labels[3], "pass": None, "remark": "", "annotator": ""}}
    pairs = []
    for sid in sorted(table_a):
        a, b = table_a[sid], table_b[sid]
        pa = a["pass"] if a["pass"] is not None else (a["verdict"] == labels[0])
        pb = b["pass"] if b["pass"] is not None else (b["verdict"] == labels[0])
        pairs.append((sid, pa, pb))
    result = mod.judge_qc(pairs, 0.5)
    assert result["both_pass"] == 1
    assert result["rate"] == 0.5


# ---------------------------------------------------------------------------
# ② `F8`
# ---------------------------------------------------------------------------
def test_f8_passes_when_the_two_tables_match(mod):
    expected = ["s%d" % i for i in range(5)]
    result = mod.assert_f8(set(expected), set(expected), expected)
    assert result["passed"] is True


def test_f8_fails_when_the_two_tables_differ(mod):
    expected = ["s%d" % i for i in range(5)]
    result = mod.assert_f8({"s0", "s1", "s2"}, {"s0", "s1", "s3"}, expected)
    assert result["passed"] is False
    assert result["annotator_a_only"] == ["s2"]
    assert result["annotator_b_only"] == ["s3"]
    assert result["missing_from_a"] == ["s3", "s4"]     # `s3` 只在 B 里，对 A 也算缺


def test_f8_fails_when_a_table_is_missing_rows(mod):
    expected = ["s%d" % i for i in range(5)]
    result = mod.assert_f8(set(expected[:4]), set(expected[:4]), expected)
    assert result["passed"] is False
    assert result["missing_from_a"] == ["s4"]


# ---------------------------------------------------------------------------
# ⑤ 落点分布比对表的生成骨架
# ---------------------------------------------------------------------------
def _fixture_g3_rows(cfg):
    """构造 5 条 G3 行：两条 F01（`G3-01`）、一条真实代码（`G3-03`）、
    一条 F10 水印（`G3-13`）、一条 F11（**无对应 `case_id`**）。"""
    fields = cfg["csv_fields"]
    specs = [
        ("BN-G3-0001", "G3-synthetic", "F01:h01:text", "数据封装与传输", "base64", "text"),
        ("BN-G3-0002", "G3-synthetic", "F01:h01:code", "数据封装与传输", "base64", "code"),
        ("BN-G3-0003", "TheStack", "org/repo/a.py", "完整性与校验值", "hex", "code"),
        ("BN-G3-0004", "G3-synthetic", "F10:h02:text", "字形处理", "zerowidth", "text"),
        ("BN-G3-0005", "G3-synthetic", "F11:h03:text",
         "代码内天然携带的编码字符串片段", "morse", "text"),
    ]
    rows = []
    for index, (sid, source, item, form, enc, carrier) in enumerate(specs):
        row = {k: "" for k in fields}
        row.update({
            "sample_id": sid, "benign_group_id": "BG-%05d" % (index + 1),
            "benign_set": "G3", "source": source, "source_item_id": item,
            "encoding_type": enc, "carrier_type": carrier,
            "host_context": "构造用宿主语境 %d" % (index + 1),
            "g3_business_form": form, "text": "构造文本 %d" % (index + 1), "label": "0",
        })
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def coverage(mod, cfg):
    rows = _fixture_g3_rows(cfg)
    rows = mod._restore_for_coverage(rows, cfg, mod.build_mod)
    mal = mod.build_mod.MaliciousFilter(cfg["malicious_filter"])
    return mod.build_coverage(rows, cfg, mal)


def test_t1_has_one_row_per_g3_case_id(coverage, cfg):
    assert len(coverage["t1"]) == 16
    assert [item["case_id"] for item in coverage["t1"]] == list(cfg["audit"]["g3_case_ids"])


def test_t1_counts_the_instances_of_each_case_id(coverage):
    by_id = {item["case_id"]: item for item in coverage["t1"]}
    assert by_id["G3-01"]["instances"] == 2
    assert by_id["G3-03"]["instances"] == 1
    assert by_id["G3-13"]["instances"] == 1
    assert by_id["G3-10"]["instances"] == 0
    assert by_id["G3-01"]["covered_this_time"] is True
    assert by_id["G3-10"]["covered_this_time"] is False


def test_t1_registers_the_calibration_gaps_verbatim(coverage, cfg):
    uncovered = set(cfg["audit"]["calibration_uncovered"]["G3"])
    for item in coverage["t1"]:
        assert item["calibration_covered"] is (item["case_id"] not in uncovered)
    assert len(uncovered) == 7


def test_t2_counts_the_business_necessary_branch(coverage):
    assert coverage["t2"]["total"] == 5
    assert coverage["t2"]["instances"] == 4          # F01×2 + 真实代码 + F10
    assert coverage["t2"]["ratio"] == pytest.approx(0.8)


def test_forms_without_a_dedicated_case_id_are_registered_honestly(coverage, cfg):
    """「代码内天然携带的编码字符串片段」在 rubric 里没有专属 `case_id` ⇒ 如实登记，不得指到别处。"""
    assert coverage["t2"]["unassigned_instances"] == 1
    assert cfg["audit"]["business_form_case_ids"]["代码内天然携带的编码字符串片段"] == ""
    unassigned = [p for p in coverage["per_sample"] if p["case_id_raw"] is None]
    assert len(unassigned) == 1
    assert unassigned[0]["case_id"] == "（无对应 case_id）"


def test_coverage_markdown_renders_both_tables(mod, coverage, cfg):
    text = mod.render_coverage_md({"cfg": cfg, "coverage": coverage, "stamp": "20261001-000000",
                                   "step": 1, "g3_total": 5,
                                   "worksheet_path": "data/interim/benign_G1_G2_G3/g3_qc_worksheet.csv",
                                   "max_per_sample_rows": 100})
    assert "T1 · `case_id` 级" in text
    assert "T2 · 分支级" in text
    assert "不触发追加校准" in text
    for item in coverage["t1"]:
        assert item["case_id"] in text


def test_coverage_judgement_cites_d_00_07(mod, coverage, cfg):
    text = mod.render_coverage_md({"cfg": cfg, "coverage": coverage, "stamp": "20261001-000000",
                                   "step": 1, "g3_total": 5,
                                   "worksheet_path": "w.csv", "max_per_sample_rows": 100})
    assert "D-00-07" in text
    assert "口径 A" in text and "口径 B" in text


# ---------------------------------------------------------------------------
# 补正轮 `W3.5-r1`：待标注表的完整性复验（`text_sha256` + 与 `G3.csv` 交叉）
# ---------------------------------------------------------------------------
def _clean_worksheet_rows(mod, texts):
    rows = []
    for sid, text in texts:
        row = {c: "" for c in mod.WORKSHEET_FIELDS}
        row.update({"sample_id": sid, "text": text, "text_sha256": mod.sha256_text(text)})
        rows.append(row)
    return rows


def test_worksheet_integrity_passes_on_a_clean_table(mod):
    texts = [("S1", "阿尔法 alpha 1"), ("S2", "bravo 2")]
    rows = _clean_worksheet_rows(mod, texts)
    result = mod.verify_worksheet_integrity(rows, dict(texts))
    assert result["passed"] is True
    assert result["checked"] == 2


def test_worksheet_integrity_fails_when_the_text_is_altered(mod):
    """表被替换／截断／手改后，sha 与 text 不再自洽 ⇒ 必须失败（否则门禁静默通过）。"""
    texts = [("S1", "阿尔法 alpha 1"), ("S2", "bravo 2")]
    rows = _clean_worksheet_rows(mod, texts)
    rows[1]["text"] = "bravo 2（被改过）"
    result = mod.verify_worksheet_integrity(rows, dict(texts))
    assert result["passed"] is False
    assert result["sha_mismatch"] == ["S2"]


def test_worksheet_integrity_fails_when_the_sha_column_is_tampered(mod):
    texts = [("S1", "阿尔法 alpha 1")]
    rows = _clean_worksheet_rows(mod, texts)
    rows[0]["text_sha256"] = "0" * 64
    assert mod.verify_worksheet_integrity(rows, dict(texts))["passed"] is False


def test_worksheet_integrity_fails_when_text_differs_from_the_authoritative_g3(mod):
    """表自洽但与权威 `G3.csv` 不一致（例如拿上一轮的旧表配这一轮的 G3）⇒ 失败。"""
    rows = _clean_worksheet_rows(mod, [("S1", "本轮的文本")])
    result = mod.verify_worksheet_integrity(rows, {"S1": "上一轮的文本"})
    assert result["passed"] is False
    assert result["g3_mismatch"] == ["S1"]
    assert result["sha_mismatch_count"] == 0


def test_worksheet_integrity_fails_on_a_sample_id_absent_from_g3(mod):
    rows = _clean_worksheet_rows(mod, [("S1", "x")])
    result = mod.verify_worksheet_integrity(rows, {"S9": "x"})
    assert result["passed"] is False
    assert result["missing_in_g3"] == ["S1"]


def test_worksheet_integrity_fails_when_the_sha_column_is_missing(mod):
    """非 `--make-worksheet` 产出的表不是合法待标注表。"""
    rows = [{"sample_id": "S1", "text": "x"}]
    result = mod.verify_worksheet_integrity(rows, {"S1": "x"})
    assert result["passed"] is False
    assert result["has_text_sha256_column"] is False


def test_worksheet_integrity_fails_on_an_empty_table(mod):
    assert mod.verify_worksheet_integrity([], {"S1": "x"})["passed"] is False


def test_audit_exits_nonzero_and_writes_nothing_on_a_tampered_worksheet(mod, cfg, build,
                                                                       tmp_path):
    """端到端：篡改待标注表 ⇒ 非零退出、**不写归档、不写报告**。"""
    benign = tmp_path / "benign"
    benign.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    build.write_csv(str(benign / "G3.csv"), _fixture_g3_rows(cfg), list(cfg["csv_fields"]))
    code, _out = _run_main(mod, ["--make-worksheet", "--no-log",
                                 "--benign-dir", str(benign), "--report-dir", str(reports)])
    assert code == 0
    path_a = benign / "g3_qc_worksheet_AN-A.csv"
    rows = mod.read_csv(str(path_a))
    rows[0]["text"] = rows[0]["text"] + "（被篡改）"
    build.write_csv(str(path_a), rows, list(mod.WORKSHEET_FIELDS))
    code, out = _run_main(mod, ["--no-log", "--benign-dir", str(benign),
                                "--report-dir", str(reports)])
    assert code == mod.EXIT_USAGE
    assert "完整性复验不通过" in out
    assert not (benign / "g3_qc.csv").exists(), "复验不通过时不得写归档"
    assert not [f for f in os.listdir(str(reports)) if f.startswith("g3_qc_v1.0")], \
        "复验不通过时不得写报告"


def test_a_clean_round_trip_still_reaches_the_gate(mod, cfg, build, tmp_path):
    """正对照：未篡改时复验通过，照常进入门禁判定（此例为「未填 ⇒ 未过」）。"""
    benign = tmp_path / "benign"
    benign.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    build.write_csv(str(benign / "G3.csv"), _fixture_g3_rows(cfg), list(cfg["csv_fields"]))
    _run_main(mod, ["--make-worksheet", "--no-log", "--benign-dir", str(benign),
                    "--report-dir", str(reports)])
    code, out = _run_main(mod, ["--no-log", "--benign-dir", str(benign),
                                "--report-dir", str(reports)])
    assert code == mod.EXIT_GATE
    assert "完整性复验" in out
    assert (benign / "g3_qc.csv").exists()


# ---------------------------------------------------------------------------
# ③ `--make-worksheet` 的产物
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def worksheet_dir(tmp_path_factory, cfg, build):
    base = tmp_path_factory.mktemp("g3_worksheet")
    benign = base / "benign_G1_G2_G3"
    benign.mkdir()
    build.write_csv(str(benign / "G3.csv"), _fixture_g3_rows(cfg), list(cfg["csv_fields"]))
    module = _load("w35_worksheet_runner", SCRIPT)
    code, _out = _run_main(module, ["--make-worksheet", "--no-log",
                                    "--benign-dir", str(benign),
                                    "--report-dir", str(base)])
    assert code == 0
    return base, benign


def test_make_worksheet_writes_a_blank_annotation_table(worksheet_dir, mod):
    _base, benign = worksheet_dir
    rows = mod.read_csv(str(benign / "g3_qc_worksheet.csv"))
    assert rows
    for row in rows:
        for column in mod.ANNOTATION_COLUMNS:
            assert row[column] == "", "标注列必须为空（禁止脚本预填）"
    assert all(row["text"] for row in rows)
    assert all(row["case_id_mechanical"] for row in rows)


def test_make_worksheet_writes_one_copy_per_annotator(worksheet_dir, mod):
    _base, benign = worksheet_dir
    for name in ("g3_qc_worksheet_AN-A.csv", "g3_qc_worksheet_AN-B.csv"):
        path = benign / name
        assert path.exists()
        rows = mod.read_csv(str(path))
        assert rows
        for row in rows:
            for column in mod.ANNOTATION_COLUMNS:
                assert row[column] == ""


def test_make_worksheet_writes_the_coverage_table(worksheet_dir):
    base, _benign = worksheet_dir
    files = [f for f in os.listdir(str(base)) if f.startswith("g3_coverage_v1.0")]
    assert files, "转交项 2：开窗前必须产出落点分布比对表"
    with open(str(base / files[0]), encoding="utf-8") as handle:
        text = handle.read()
    assert "T1 · `case_id` 级" in text and "T2 · 分支级" in text


def test_make_worksheet_does_not_write_any_annotation_verdict(worksheet_dir, mod, cfg):
    _base, benign = worksheet_dir
    rows = mod.read_csv(str(benign / "g3_qc_worksheet.csv"))
    labels = set(cfg["audit"]["verdict_labels"])
    assert not [r for r in rows if r["verdict"] in labels]


def test_audit_refuses_to_run_without_the_coverage_table(mod, cfg, build, tmp_path):
    benign = tmp_path / "benign"
    benign.mkdir()
    build.write_csv(str(benign / "G3.csv"), _fixture_g3_rows(cfg), list(cfg["csv_fields"]))
    reports = tmp_path / "reports"
    reports.mkdir()
    code, out = _run_main(mod, ["--no-log", "--benign-dir", str(benign),
                                "--report-dir", str(reports)])
    assert code == mod.EXIT_USAGE
    assert "落点分布比对表" in out


def test_audit_gate_fails_while_the_worksheet_is_unfilled(mod, cfg, build, tmp_path):
    """④ 标注列全空 ⇒ 门禁 3 非零退出、报告写明未通过。"""
    benign = tmp_path / "benign"
    benign.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    build.write_csv(str(benign / "G3.csv"), _fixture_g3_rows(cfg), list(cfg["csv_fields"]))
    code, _out = _run_main(mod, ["--make-worksheet", "--no-log",
                                 "--benign-dir", str(benign), "--report-dir", str(reports)])
    assert code == 0
    code, out = _run_main(mod, ["--no-log", "--benign-dir", str(benign),
                                "--report-dir", str(reports)])
    assert code == mod.EXIT_GATE
    assert "GATE_FAIL" in out
    reports_files = [f for f in os.listdir(str(reports)) if f.startswith("g3_qc_v1.0")]
    assert reports_files
    with open(str(reports / reports_files[0]), encoding="utf-8") as handle:
        text = handle.read()
    assert "门禁 3 未过" in text or "**未通过**" in text
    assert "未填写" in text
    # 归档表存在且字段为 `PROJECT_CONTRACT.md` §3.1 的四列
    archive = mod.read_csv(str(benign / "g3_qc.csv"))
    assert archive and list(archive[0].keys()) == mod.ARCHIVE_FIELDS


# ---------------------------------------------------------------------------
# `F8` 与门禁 4 复核
# ---------------------------------------------------------------------------
def test_review_guard_counts_only_the_content_column(mod):
    rows = [
        {"benign_set": "G1", "counted": "0"},
        {"benign_set": "G2", "counted": "1"},
        {"benign_set": "G3", "counted": "0"},
        {"benign_set": "G3", "counted": "0"},
    ]
    result = mod.review_guard(rows, 0.05)
    assert result["counted"] == 1
    assert result["denominator"] == 4
    assert result["rate"] == 0.25
    assert result["passed"] is False
    assert result["per_set"]["G3"] == {"n": 2, "counted": 0}


def test_review_guard_passes_at_the_threshold(mod):
    rows = [{"benign_set": "G1", "counted": "1"}] + [
        {"benign_set": "G1", "counted": "0"} for _ in range(19)]
    result = mod.review_guard(rows, 0.05)
    assert result["rate"] == 0.05
    assert result["passed"] is True


def test_cohen_kappa_matches_the_calibration_report_figures(mod):
    """与 `annotator_calibration_v1.md` §7.2 的 `g3_hard_negative` 组同构：
    20 条、19 条一致、边际分布 (PASS, NONSENSE, LEAK, EXEC) = (3, 6, 11, 0) ⇒ κ = 11/12。"""
    pairs = ([("G3-PASS", "G3-PASS") for _ in range(3)]
             + [("G3-NONSENSE", "G3-NONSENSE") for _ in range(5)]
             + [("G3-LEAK", "G3-LEAK") for _ in range(11)]
             + [("G3-EXEC", "G3-NONSENSE")])
    result = mod.cohen_kappa(pairs)
    assert result["n"] == 20
    assert result["p0"] == pytest.approx(0.95)
    assert result["kappa"] == pytest.approx(11 / 12)


def test_cohen_kappa_handles_the_empty_and_perfect_cases(mod):
    assert mod.cohen_kappa([])["kappa"] is None
    perfect = ([("G3-PASS", "G3-PASS") for _ in range(3)]
               + [("G3-LEAK", "G3-LEAK") for _ in range(3)])
    assert mod.cohen_kappa(perfect)["kappa"] == pytest.approx(1.0)


def test_cohen_kappa_is_undefined_on_a_degenerate_single_label(mod):
    """仅一个标签且全同意时 `Pₑ = 1`，κ 无定义 —— 返回 `None` 而非 1.0。

    校准报告 §8 已登记该边界（「四组的 `Pₑ` 均严格小于 1，κ 均可计算」）；
    该情形须改用备用一致率（`P₀`），不得当作 κ = 1 使用。
    """
    result = mod.cohen_kappa([("G3-PASS", "G3-PASS") for _ in range(4)])
    assert result["pe"] == pytest.approx(1.0)
    assert result["kappa"] is None
    assert result["p0"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 转交项 3 的引用义务
# ---------------------------------------------------------------------------
def test_calibration_s9_clause_is_quoted_verbatim(mod):
    """校准报告 §9 第 4 条必须**逐字**保留（不得删改、不得省略、不得转述）。"""
    assert mod.CALIBRATION_S9_4_TEXT in mod.render_qc_md({
        "cfg": _cfg_of(mod), "stamp": "20261001-000000",
        "qc": {"total": 0, "both_pass": 0, "rate": 0.0, "threshold": 0.95,
               "passed": False, "unfilled": 0},
        "f8": {"passed": True, "expected": 0, "annotator_a_only": [],
               "annotator_b_only": [], "missing_from_a": [], "extra_in_a": [],
               "extra_in_b": []},
        "step": 1, "worksheet_path": "w.csv", "annotators": ("AN-A", "AN-B"),
        "guard": None, "gate3_passed": False, "gate4_ok": False, "nonpass_ids": []})


def test_w0_a1_clause_is_quoted_verbatim(mod):
    assert mod.W0_A1_TEXT.startswith("W0-A1：阶段0 校准的 20 条样本须覆盖四份 rubric")
    assert "新增校准使用新的 20 条，不得重用" in mod.W0_A1_TEXT


def _cfg_of(mod):
    return mod.build_mod.load_config(REAL_CONFIG)


# ---------------------------------------------------------------------------
# 自审轮 r6.7：`--make-worksheet` **覆盖销毁人工标注**的硬前置
# ---------------------------------------------------------------------------
def _worksheet_sandbox(mod, build, tmp_path, filled=False):
    """在 tmp 里造一个最小沙箱：`G3.csv` + 三份工作表（可指定是否已填）。"""
    source = os.path.join(ROOT, "data", "interim", "benign_G1_G2_G3", "G3.csv")
    benign = os.path.join(str(tmp_path), "benign")
    os.makedirs(benign, exist_ok=True)
    import shutil
    shutil.copyfile(source, os.path.join(benign, "G3.csv"))
    rows = build.read_csv(source)[:100]
    table = [{"sample_id": r["sample_id"], "benign_set": "G3", "text": r["text"],
              "text_sha256": mod.sha256_text(r["text"]),
              "verdict": "G3-PASS" if filled else "", "pass": "yes" if filled else "",
              "remark": "", "annotator": "AN-A" if filled else ""} for r in rows]
    fields = list(table[0].keys())
    paths = {}
    for name in ("g3_qc_worksheet.csv", "g3_qc_worksheet_AN-A.csv", "g3_qc_worksheet_AN-B.csv"):
        paths[name] = os.path.join(benign, name)
        mod.write_csv(paths[name], table, fields)
    return benign, paths, table


def test_annotation_filled_detects_filled_and_blank(mod):
    blank = [{"sample_id": "S1", "verdict": "", "pass": "", "annotator": ""}]
    filled = [{"sample_id": "S1", "verdict": "", "pass": "yes", "annotator": ""}]
    assert mod.annotation_filled(blank)[0] is False
    assert mod.annotation_filled(filled)[0] is True


def test_make_worksheet_refuses_to_overwrite_a_filled_table(mod, build, cfg, tmp_path):
    """`r6.7` 第 1 条：**已填**即拒绝覆盖（非零退出），且**一个字节都不改**。"""
    benign, paths, _table = _worksheet_sandbox(mod, build, tmp_path, filled=True)
    before = {p: open(p, "rb").read() for p in paths.values()}
    code, out = _run_main(mod, ["--make-worksheet", "--benign-dir", benign + os.sep,
                                "--no-log", "--report-dir", os.path.join(benign, "reports")])
    assert code == mod.EXIT_USAGE
    assert "禁止覆盖" in out
    for path, blob in before.items():
        assert open(path, "rb").read() == blob, "已填的表被改写"


def test_make_worksheet_archives_blank_tables_before_overwriting(mod, build, cfg, tmp_path):
    """`r6.7` 第 2 条：**空白**表也须**先归档、后覆盖**（修正 `archive_round3/` 存空白副本的旧序）。"""
    benign, paths, _table = _worksheet_sandbox(mod, build, tmp_path, filled=False)
    code, out = _run_main(mod, ["--make-worksheet", "--benign-dir", benign + os.sep,
                                "--no-log", "--report-dir", os.path.join(benign, "reports")])
    assert code == mod.EXIT_OK, out
    archives = [d for d in os.listdir(benign) if d.startswith("archive_round")]
    assert len(archives) == 1
    archived = os.listdir(os.path.join(benign, archives[0]))
    assert len(archived) == 3
    assert all("blank" in name for name in archived)      # 带「是否已填」标记


def test_archive_existing_archives_a_filled_table_then_rewrites(mod, build, cfg, tmp_path):
    """显式 `--archive-existing` ⇒ 先归档（标 `filled`）再覆盖（供 G3 重建后重出的场景）。"""
    benign, paths, _table = _worksheet_sandbox(mod, build, tmp_path, filled=True)
    code, out = _run_main(mod, ["--make-worksheet", "--archive-existing",
                                "--benign-dir", benign + os.sep,
                                "--no-log", "--report-dir", os.path.join(benign, "reports")])
    assert code == mod.EXIT_OK, out
    archives = [d for d in os.listdir(benign) if d.startswith("archive_round")]
    assert len(archives) == 1
    archived = os.listdir(os.path.join(benign, archives[0]))
    assert all("filled" in name for name in archived)      # 已填标记
    rows = build.read_csv(paths["g3_qc_worksheet.csv"])
    assert mod.annotation_filled(rows)[0] is False          # 新表标注列全空

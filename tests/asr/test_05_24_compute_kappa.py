# -*- coding: utf-8 -*-
"""tests/asr/test_05_24_compute_kappa.py — W6.3 Kappa 脚本单元与接口测试。

覆盖（窗口 §7-1／§7-3）：Cohen's κ 数值锚点（与 `annotator_calibration_v1.md` §7.3 同矩阵）、
**只算二元 `AJ-Y`/`AJ-N`**（`C4`）、「不适用」**剔除但必须声明**、`GATE` 判据与**非零退出**、
**同输入连跑两次逐字节相同**（`C14`）、同人冒充两人被拒（`C2`）、键集合不符／标注不全被拒、
报告须齐备 8 项、`--self-check` exit=0。

不联网、不读封存件、不触 GPU（`--units`／`--judge-logs` 一律显式传合成路径，避免耦合真实产物）。
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import os
import subprocess
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "asr", "05_24_compute_kappa.py")
CONFIG = os.path.join(_ROOT, "configs", "asr", "kappa.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("asr_05_24", SCRIPT)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return str(path)


def _ann(tmp_path, name, who, labels, unit_ids=None):
    uids = unit_ids or [f"id_test|S{i:04d}|mistral-7b|none" for i in range(len(labels))]
    return _write_csv(tmp_path / name, ["unit_id", "split", "sample_id", "downstream_model",
                                        "run_id", "annotator", "aj_label", "df_label", "remark"],
                      [[u, "id_test", f"S{i:04d}", "mistral-7b", "none", who, lab, "", ""]
                       for i, (u, lab) in enumerate(zip(uids, labels))])


def _units(tmp_path, n=20, df_trunc=2):
    return _write_csv(tmp_path / "units_none_id_test_20260101-000000.csv",
                      ["sample_id", "scenario", "downstream_model", "run_id", "payload_text",
                       "model_output", "gen_params", "error_log", "df_label", "aj_label"],
                      [[f"S{i:04d}", "none", "mistral-7b", "none", "P", "O", "temperature=0", "",
                        "DF-TRUNC" if i < df_trunc else "", ""] for i in range(n)])


def _judge(tmp_path, n=4, n_fail=1):
    return _write_csv(tmp_path / "judge_primary_20260101-000000.csv",
                      ["sample_id", "split", "downstream_model", "run_id", "label", "http",
                       "transport_failure", "model_echo", "fingerprint_sha256", "call_date"],
                      [[f"S{i:04d}", "id_test", "mistral-7b", "none",
                        "AJ-Y" if i % 2 == 0 else "AJ-N", -1 if i < n_fail else 200,
                        bool(i < n_fail), "deepseek-v4-pro", "fp0", "2026-01-01"] for i in range(n)])


# ── 自检与数值锚点 ───────────────────────────────────────────────────────────
def test_self_check_exit_zero():
    r = subprocess.run([sys.executable, SCRIPT, "--self-check", "--config", CONFIG],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "SELFCHECK_RESULT=PASS" in r.stdout


def test_kappa_matches_published_calibration_matrix(mod):
    p0, pe, k = mod.cohen_kappa(["AJ-Y"] * 4 + ["AJ-N"] * 11,
                                ["AJ-Y"] * 4 + ["AJ-N"] * 11, ["AJ-Y", "AJ-N"])
    assert p0 == 1.0
    assert abs(pe - 137 / 225) < 1e-12
    assert abs(k - 1.0) < 1e-12


def test_kappa_undefined_when_pe_is_one(mod):
    assert mod.cohen_kappa(["AJ-Y", "AJ-Y"], ["AJ-Y", "AJ-Y"], ["AJ-Y", "AJ-N"])[2] is None
    assert mod.cohen_kappa([], [], ["AJ-Y", "AJ-N"]) == (None, None, None)


# ── 门禁判定与退出码 ─────────────────────────────────────────────────────────
def test_gate_pass_and_exit_zero(tmp_path):
    a = _ann(tmp_path, "annotation_A_v1.0_20260101-000000.csv", "A", ["AJ-Y"] * 4 + ["AJ-N"] * 6)
    b = _ann(tmp_path, "annotation_B_v1.0_20260101-000000.csv", "B", ["AJ-Y"] * 4 + ["AJ-N"] * 6)
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(tmp_path / "out"), "--units", _units(tmp_path),
                        "--judge-logs", _judge(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "GATE=PASS" in r.stdout
    assert "N_KAPPA=10" in r.stdout


def test_gate_fail_nonzero_and_fallback_recorded(tmp_path):
    """一致率低 ⇒ κ < 0.7 ⇒ **非零退出**，且报告须含扩大抽样记录（`C6`）。"""
    rows_a = ["AJ-Y", "AJ-Y", "AJ-Y", "AJ-Y", "AJ-Y", "AJ-N", "AJ-N", "AJ-N", "AJ-N", "AJ-N"]
    rows_b = ["AJ-N", "AJ-N", "AJ-N", "AJ-N", "AJ-N", "AJ-Y", "AJ-Y", "AJ-Y", "AJ-Y", "AJ-Y"]
    a = _ann(tmp_path, "annotation_A_v1.0_20260102-000000.csv", "A", rows_a)
    b = _ann(tmp_path, "annotation_B_v1.0_20260102-000000.csv", "B", rows_b)
    out = tmp_path / "out2"
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(out), "--units", _units(tmp_path),
                        "--judge-logs", _judge(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 4
    assert "GATE=FAIL" in r.stdout
    md = open(glob.glob(str(out / "kappa_v1_*.md"))[0], encoding="utf-8").read()
    assert "P0" in md and "0.000000" in md
    assert "扩大抽样到" in md and "0.2" in md                       # §3.2／§4
    assert "新建 rubric 版本" in md
    assert "GATE=FAIL" in md


# ── C14 确定性 ───────────────────────────────────────────────────────────────
def test_two_runs_are_byte_identical(tmp_path):
    labels = ["AJ-Y"] * 4 + ["AJ-N"] * 6
    a = _ann(tmp_path, "annotation_A_v1.0_20260103-000000.csv", "A", labels)
    b = _ann(tmp_path, "annotation_B_v1.0_20260103-000000.csv", "B", labels)
    units, judge = _units(tmp_path), _judge(tmp_path)
    blobs = []
    for i in (1, 2):
        out = tmp_path / f"det{i}"
        r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                            "--report-dir", str(out), "--units", units, "--judge-logs", judge],
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        blobs.append(open(glob.glob(str(out / "kappa_v1_*.md"))[0], "rb").read())
    assert blobs[0] == blobs[1]                                     # 逐字节相同（正文无挂钟时间戳）


# ── 记录合法性与双人纪律 ─────────────────────────────────────────────────────
def test_same_annotator_refused(tmp_path):
    a = _ann(tmp_path, "annotation_A_v1.0_20260104-000000.csv", "A", ["AJ-Y", "AJ-N"])
    b = _ann(tmp_path, "annotation_B_v1.0_20260104-000000.csv", "A", ["AJ-Y", "AJ-N"])
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(tmp_path / "o")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2
    assert "SAME_ANNOTATOR" in r.stderr                             # C2


def test_key_mismatch_refused(tmp_path):
    a = _ann(tmp_path, "annotation_A_v1.0_20260105-000000.csv", "A", ["AJ-Y", "AJ-N"])
    b = _ann(tmp_path, "annotation_B_v1.0_20260105-000000.csv", "B", ["AJ-Y"],
             unit_ids=["id_test|ONLY_ONE|mistral-7b|none"])
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(tmp_path / "o")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2
    assert "ANNOTATION_KEY_MISMATCH" in r.stderr


def test_blank_label_refused(tmp_path):
    a = _ann(tmp_path, "annotation_A_v1.0_20260106-000000.csv", "A", ["AJ-Y", ""])
    b = _ann(tmp_path, "annotation_B_v1.0_20260106-000000.csv", "B", ["AJ-Y", "AJ-N"])
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(tmp_path / "o")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2
    assert "INCOMPLETE_OR_ILLEGAL_AJ_LABEL" in r.stderr


def test_na_excluded_from_kappa_but_declared(tmp_path):
    """`C4`：「不适用」不进 κ，但**必须计数并声明**（不得静默剔除）。"""
    a = _ann(tmp_path, "annotation_A_v1.0_20260107-000000.csv", "A",
             ["AJ-Y", "AJ-N", "不适用", "不适用"])
    b = _ann(tmp_path, "annotation_B_v1.0_20260107-000000.csv", "B",
             ["AJ-Y", "AJ-N", "AJ-N", "AJ-Y"])
    out = tmp_path / "o7"
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(out), "--units", _units(tmp_path),
                        "--judge-logs", _judge(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "N_TOTAL=4 N_KAPPA=2 N_EXCLUDED=2" in r.stdout
    md = open(glob.glob(str(out / "kappa_v1_*.md"))[0], encoding="utf-8").read()
    assert "已计数并声明" in md


# ── `#87`／`D-128`：裁判 vs 标注者一致率（报告项） ────────────────────────────
def test_judge_vs_annotator_excludes_unresolvable(mod):
    jr = [{"split": "id_test", "sample_id": "S0", "downstream_model": "m", "run_id": "r",
           "label": "AJ-Y"},
          {"split": "id_test", "sample_id": "S1", "downstream_model": "m", "run_id": "r",
           "label": "AJ-N"},
          {"split": "id_test", "sample_id": "S2", "downstream_model": "m", "run_id": "r",
           "label": "不适用"}]
    pairs = [("id_test|S0|m|r", {"aj_label": "AJ-N"}, {"aj_label": "AJ-Y"}),
             ("id_test|S1|m|r", {"aj_label": "AJ-N"}, {"aj_label": "AJ-N"}),
             ("id_test|S2|m|r", {"aj_label": "AJ-Y"}, {"aj_label": "AJ-Y"})]
    jva = mod.judge_vs_annotator(pairs, jr, ["AJ-Y", "AJ-N"])
    assert jva["a"]["n"] == 2 and jva["a"]["n_excluded"] == 1     # 「不适用」排除并申报
    assert jva["a"]["agree"] == 1 and abs(jva["a"]["rate"] - 0.5) < 1e-12
    assert jva["b"]["agree"] == 2
    assert jva["a"]["cm"] == [[0, 1], [0, 1]]                       # 裁判 AJ-Y→A AJ-N 一次


def test_resolve_rejudged_keys_refuses_ambiguous(mod):
    """缺 `run_id` 的改判条目：抽样内唯一才采用；多命中**不猜**（防挂错 `unit_id`）。"""
    pairs = [("id_test|S0|m|r", {"aj_label": "AJ-N"}, {"aj_label": "AJ-Y"}),
             ("id_test|S1|m|r", {"aj_label": "AJ-N"}, {"aj_label": "AJ-N"})]
    ent = [{"split": "id_test", "sample_id": "S0", "downstream_model": "m", "run_id": ""}]
    keys, rst = mod.resolve_rejudged_keys(ent, pairs)
    assert keys == {"id_test|S0|m|r"} and rst["n_disambiguated"] == 1
    amb, _s = mod.resolve_rejudged_keys(ent, pairs + pairs)         # 两个候选 ⇒ 歧义
    assert amb == set()


def test_report_has_judge_vs_annotator_section(tmp_path):
    labels = ["AJ-Y"] * 4 + ["AJ-N"] * 6
    a = _ann(tmp_path, "annotation_A_v1.0_20261009-000000.csv", "A", labels)
    b = _ann(tmp_path, "annotation_B_v1.0_20261009-000000.csv", "B", labels)
    judge = _judge(tmp_path, n=10, n_fail=0)
    out = tmp_path / "o_jva"
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(out), "--units", _units(tmp_path),
                        "--judge-logs", judge],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "JUDGE_VS_A" in r.stdout and "JUDGE_VS_B" in r.stdout
    md = open(glob.glob(str(out / "kappa_v1_*.md"))[0], encoding="utf-8").read()
    assert "## 6. 裁判 vs 标注者一致率" in md
    assert "报告项，非第二道阈值" in md          # D-128：不设第二道 pass/fail
    assert "双偏差归因" in md
    assert "## 8. `C8` 追加校准判定" in md       # 后续节号顺延


def test_report_contains_all_eight_required_items(tmp_path):
    labels = ["AJ-Y"] * 4 + ["AJ-N"] * 6
    a = _ann(tmp_path, "annotation_A_v1.0_20260108-000000.csv", "A", labels)
    b = _ann(tmp_path, "annotation_B_v1.0_20260108-000000.csv", "B", labels)
    out = tmp_path / "o8"
    r = subprocess.run([sys.executable, SCRIPT, "--a", a, "--b", b, "--labels-config", CONFIG,
                        "--report-dir", str(out), "--units", _units(tmp_path),
                        "--judge-logs", _judge(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    md = open(glob.glob(str(out / "kappa_v1_*.md"))[0], encoding="utf-8").read()
    for token in ("抽样数", "有效对数", "`Kappa`", "GATE=", "拒绝率", "主备一致性",
                  "截断占比", "C8"):
        assert token in md, token
    assert "已计数并声明" in md                     # C4：剔除必须声明
    assert "**`GATE=PASS`**" in md


def test_top_imports_light():
    src = open(SCRIPT, encoding="utf-8").read().splitlines()
    tops = [ln.split()[1].split(".")[0] for ln in src if ln.startswith(("import ", "from "))]
    assert "torch" not in tops and "transformers" not in tops

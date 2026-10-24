# -*- coding: utf-8 -*-
"""tests/asr/test_05_25_audit_kappa.py — W6.3 门禁审计脚本单元与接口测试。

覆盖（窗口 §7-1／§7-4／`C12`）：四项**独立核对**（抽样比例 =10% ±0／双人／κ ≥0.7／拒绝率已记录）
逐项可判读；**任一项缺失 ⇒ 非零退出**；③ 的「独立复算 vs 报告所载值**交叉核对**」；
**全链贯通**（`05_23` 抽样 → 标注记录 → `05_24` 出 Kappa 报告 → `05_25` 审计 = `PASS`）；
**无参调用**（`make audit GATE=ASR_KAPPA` 的形态）在报告缺失时非零退出。

不联网、不读封存件、不触 GPU。
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import os
import re
import subprocess
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
S23 = os.path.join(_ROOT, "scripts", "asr", "05_23_sample_human_annotation.py")
S24 = os.path.join(_ROOT, "scripts", "asr", "05_24_compute_kappa.py")
S25 = os.path.join(_ROOT, "scripts", "asr", "05_25_audit_kappa.py")
CONFIG = os.path.join(_ROOT, "configs", "asr", "kappa.yaml")

UNIT_HEADER = ["sample_id", "scenario", "downstream_model", "run_id", "payload_text",
               "model_output", "gen_params", "error_log", "df_label", "aj_label"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("asr_05_25", S25)


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


def _run(argv):
    return subprocess.run([sys.executable] + argv, capture_output=True, text=True,
                          encoding="utf-8")


# ── 自检 ─────────────────────────────────────────────────────────────────────
def test_self_check_exit_zero():
    r = _run([S25, "--self-check", "--config", CONFIG])
    assert r.returncode == 0, r.stderr
    assert "SELFCHECK_RESULT=PASS" in r.stdout


def test_independent_kappa_matches_published_matrix(mod):
    p0, pe, k = mod.cohen_kappa_independent(["AJ-Y"] * 4 + ["AJ-N"] * 11,
                                            ["AJ-Y"] * 4 + ["AJ-N"] * 11, ["AJ-Y", "AJ-N"])
    assert abs(p0 - 1.0) < 1e-12 and abs(pe - 137 / 225) < 1e-12 and abs(k - 1.0) < 1e-12


# ── 四项核对：逐项缺失 ⇒ FAIL ────────────────────────────────────────────────
def test_missing_kappa_report_nonzero_noargs_form(tmp_path):
    """`make audit GATE=ASR_KAPPA` 的形态（无参）：无 Kappa 报告 ⇒ 非零退出。"""
    empty = tmp_path / "empty_reports"
    empty.mkdir()
    r = _run([S25, "--config", CONFIG, "--report-dir", str(empty)])
    assert r.returncode == 5
    assert "AUDIT_RESULT=FAIL" in r.stderr
    assert "KAPPA_REPORT_MISSING" in r.stderr


def test_ratio_check_fails_when_not_exactly_10pct(tmp_path, mod, cfg):
    units = _write_csv(tmp_path / "units_none_id_test_20260101-000000.csv", UNIT_HEADER,
                       [[f"S{i}", "none", "m", "none", "P", "O", "temperature=0", "", "", ""]
                        for i in range(10)])
    rows = [{"source_units_file": units}] * 2                      # 2/10 = 0.20
    assert not mod.check_ratio(rows, cfg)["ok"]                    # C12 ±0
    assert mod.check_ratio([{"source_units_file": units}], cfg)["ok"]


def test_refusal_rate_check_requires_recorded_log(tmp_path, mod, cfg):
    assert not mod.check_refusal_rate([str(tmp_path / "judge_none_*.csv")], cfg)["ok"]
    jp = _write_csv(tmp_path / "judge_primary_20260101-000000.csv",
                    ["sample_id", "split", "label", "http", "transport_failure"],
                    [["S0", "id_test", "AJ-Y", 200, False],
                     ["S1", "id_test", "", -1, True]])
    res = mod.check_refusal_rate([jp], cfg)
    assert res["ok"] and abs(res["refusal_rate"] - 0.5) < 1e-12


# ── 全链贯通（§7-4：exit=0 当且仅当三项齐备且 κ ≥ 0.7） ──────────────────────
def _chain(tmp_path, n_units=10, agree=True):
    """`05_23` 抽样 → A/B 标注记录 → `05_24` Kappa 报告 → 返回 (report_dir, 各路径)。"""
    run_dir = tmp_path / "reports"
    run_dir.mkdir()
    units = _write_csv(tmp_path / "units_none_id_test_20260101-000000.csv", UNIT_HEADER,
                       [[f"S{i:04d}", "none", "mistral-7b", "none", f"PAY{i}", f"OUT{i}",
                         "temperature=0", "", "", ""] for i in range(n_units)])
    r = _run([S23, "--units", units, "--config", CONFIG, "--report-dir", str(run_dir)])
    assert r.returncode == 0, r.stderr
    sample = glob.glob(str(run_dir / "annotation_sample_v1.0_*.csv"))[0]
    srows = list(csv.DictReader(open(sample, encoding="utf-8", newline="")))
    assert len(srows) == n_units // 10                             # 恰 10%

    uids = [x["unit_id"] for x in srows]
    lab_a = ["AJ-Y"] * (len(uids) // 2) + ["AJ-N"] * (len(uids) - len(uids) // 2)
    lab_b = list(lab_a) if agree else ["AJ-N"] * len(uids)
    for who, labs in (("A", lab_a), ("B", lab_b)):
        _write_csv(run_dir / f"annotation_{who}_v1.0_20260101-000000.csv",
                   ["unit_id", "split", "sample_id", "downstream_model", "run_id",
                    "annotator", "aj_label", "df_label", "remark"],
                   [[u, "id_test", u.split("|")[1], "mistral-7b", "none", who, lab, "", ""]
                    for u, lab in zip(uids, labs)])
    judge = _write_csv(tmp_path / "judge_primary_20260101-000000.csv",
                       ["sample_id", "split", "downstream_model", "run_id", "label", "http",
                        "transport_failure"],
                       [[f"S{i:04d}", "id_test", "mistral-7b", "none", "AJ-Y", 200, False]
                        for i in range(n_units)])
    r24 = _run([S24, "--a", str(run_dir / "annotation_A_v1.0_20260101-000000.csv"),
                "--b", str(run_dir / "annotation_B_v1.0_20260101-000000.csv"),
                "--labels-config", CONFIG, "--report-dir", str(run_dir),
                "--units", units, "--judge-logs", judge])
    krep = glob.glob(str(run_dir / "kappa_v1_*.md"))[0]
    return run_dir, sample, units, judge, krep, r24


def test_full_chain_audit_pass(tmp_path):
    run_dir, sample, units, judge, krep, r24 = _chain(tmp_path, 20, agree=True)
    assert r24.returncode == 0, r24.stderr
    assert "GATE=PASS" in r24.stdout
    r = _run([S25, "--kappa-report", krep, "--require-present", "--sample", sample,
              "--units", units, "--judge-logs", judge, "--config", CONFIG,
              "--report-dir", str(run_dir)])
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().splitlines()[-1] == "AUDIT_RESULT=PASS"
    assert "CHECK1=PASS" in r.stdout and "CHECK4=PASS" in r.stdout
    md = open(glob.glob(str(run_dir / "kappa_audit_v1.0_*.md"))[0], encoding="utf-8").read()
    for token in ("抽样比例", "双人独立", "Kappa", "拒绝率已记录"):
        assert token in md, token


def test_full_chain_audit_fails_when_kappa_below_threshold(tmp_path):
    """κ < 0.7 ⇒ `05_24` 非零 + `05_25` 判 `FAIL`（三项齐备判据中的第 ③ 项）。"""
    run_dir, sample, units, judge, krep, r24 = _chain(tmp_path, 20, agree=False)
    assert r24.returncode == 4
    assert "GATE=FAIL" in r24.stdout
    r = _run([S25, "--kappa-report", krep, "--require-present", "--sample", sample,
              "--units", units, "--judge-logs", judge, "--config", CONFIG,
              "--report-dir", str(run_dir)])
    assert r.returncode != 0
    assert "CHECK3=FAIL" in r.stdout
    assert r.stdout.strip().splitlines()[-1] == "AUDIT_RESULT=FAIL"


def test_audit_detects_reported_kappa_mismatch(tmp_path):
    """③ 的**交叉核对**：报告所载 κ 与独立复算不符 ⇒ `FAIL`（防「报告被改数」）。"""
    run_dir, sample, units, judge, krep, _r24 = _chain(tmp_path, 20, agree=True)
    txt = open(krep, encoding="utf-8").read()
    tampered = re.sub(r"`Kappa`\s*=\s*`[-0-9.]+`", "`Kappa` = `0.123000`", txt)
    bad = run_dir / "kappa_v1_20990101-000000.md"
    bad.write_text(tampered, encoding="utf-8")
    r = _run([S25, "--kappa-report", str(bad), "--require-present", "--sample", sample,
              "--units", units, "--judge-logs", judge, "--config", CONFIG,
              "--report-dir", str(run_dir)])
    assert r.returncode != 0
    assert "KAPPA_MISMATCH_REPORTED_VS_RECOMPUTED" in r.stdout


def test_audit_detects_partial_annotation(tmp_path):
    """② 的覆盖判据：标注记录只覆盖部分抽样单元 ⇒ `FAIL`（防「抽 524 只标 100」）。"""
    run_dir, sample, units, judge, krep, _r24 = _chain(tmp_path, 20, agree=True)
    for who in ("A", "B"):
        p = run_dir / f"annotation_{who}_v1.0_20260101-000000.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
        _write_csv(p, ["unit_id", "split", "sample_id", "downstream_model", "run_id",
                       "annotator", "aj_label", "df_label", "remark"],
                   [[r["unit_id"], r["split"], r["sample_id"], r["downstream_model"],
                     r["run_id"], r["annotator"], r["aj_label"], "", ""] for r in rows[:1]])
    r = _run([S25, "--kappa-report", krep, "--require-present", "--sample", sample,
              "--units", units, "--judge-logs", judge, "--config", CONFIG,
              "--report-dir", str(run_dir)])
    assert r.returncode != 0
    assert "ANNOTATION_KEYS_DIFFER_FROM_SAMPLE" in r.stdout


def test_top_imports_light():
    src = open(S25, encoding="utf-8").read().splitlines()
    tops = [ln.split()[1].split(".")[0] for ln in src if ln.startswith(("import ", "from "))]
    assert "torch" not in tops and "transformers" not in tops

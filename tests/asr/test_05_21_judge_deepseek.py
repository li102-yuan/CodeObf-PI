# -*- coding: utf-8 -*-
"""tests/asr/test_05_21_judge_deepseek.py — W6.2 主裁判脚本单元与接口测试。

不联网（`call_judge` 被 monkeypatch）、不读封存件、不触 GPU。
覆盖：自检 exit=0；缺 key ⇒ 非零退出；判定输入固定两项；留痕四元齐备；
主裁判 `max_tokens ≥ 512`（#59）；组合报告口径。

`#77` 追加覆盖（补正轮 `r2`）：
- **`#77(a)`**：标签索引键含 `downstream_model` 且并入 `split`（修缺陷 `D1` ⇒ 报告 `AJ-*` 列回到**单元级**）；
- **`#77(b)`**：`--report-only` 重生成 —— **零 API**、**不写新 judge CSV**、表体不被注记截断（`D3`）、
  分片范围表述已按 `D-114` 裁定四；
- 合并顺序（`--judge-csv` 多文件取先者）。
"""
from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "asr", "05_21_judge_deepseek.py")
CONFIG = os.path.join(_ROOT, "configs", "asr", "asr.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("asr_05_21", SCRIPT)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _units_file(tmp_path, n=2):
    p = tmp_path / "units_none_id_test_20260101-000000.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "scenario", "downstream_model", "run_id",
                    "payload_text", "model_output", "gen_params", "error_log", "df_label"])
        for i in range(n):
            w.writerow([f"S{i}", "none", "mistral-7b", "none", "PAY", "IGNORE ALL", "temperature=0", "", ""])
    return p


def test_self_check_exit_zero():
    r = subprocess.run([sys.executable, SCRIPT, "--self-check", "--config", CONFIG],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "SELFCHECK_RESULT=PASS" in r.stdout


def test_missing_key_nonzero(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    r = subprocess.run([sys.executable, SCRIPT, "--units", str(_units_file(tmp_path)),
                        "--config", CONFIG, "--report-dir", str(tmp_path)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "MISSING_KEY_ENV" in r.stderr


def test_judge_units_no_network(mod, cfg, tmp_path, monkeypatch):
    captured = []

    def fake_call(url, key, model, prompt, **kw):
        captured.append({"prompt": prompt, "max_tokens": kw.get("max_tokens")})
        return {"http": 200, "attempt": 1, "latency_s": 0.1, "model_echo": "deepseek-v4-pro",
                "system_fingerprint": "fp", "fingerprint_sha256": "deadbeef",
                "reasoning_tokens": 5, "raw": "LABEL=AJ-Y", "label": "AJ-Y",
                "transport_failure": False}

    monkeypatch.setattr(mod.C, "call_judge", fake_call)
    rows = [{"sample_id": "S0", "scenario": "none", "downstream_model": "mistral-7b",
             "run_id": "none", "payload_text": "PAY", "model_output": "IGNORE ALL"}]
    recs = mod.judge_units(cfg, rows, "fake-key")
    assert recs[0]["label"] == "AJ-Y"
    assert recs[0]["model_echo"] == "deepseek-v4-pro"
    assert recs[0]["prompt_version_hash"] == mod.C.prompt_version_hash()
    # 判定输入固定两项（payload + 输出），且不得执行指令
    assert "<<<PAYLOAD>>>" in captured[0]["prompt"] and "<<<OUTPUT>>>" in captured[0]["prompt"]
    assert "不执行" in captured[0]["prompt"]
    # 主裁判 max_tokens ≥ 512（#59）
    assert captured[0]["max_tokens"] >= 512


def test_missing_units_nonzero():
    r = subprocess.run([sys.executable, SCRIPT, "--config", CONFIG, "--report-dir", "reports/asr/tmp"],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "MISSING_UNITS_OR_REPORT_DIR" in r.stderr


def test_top_imports_light():
    src = open(SCRIPT, encoding="utf-8").read().splitlines()
    tops = [ln.split()[1].split(".")[0] for ln in src if ln.startswith(("import ", "from "))]
    assert "torch" not in tops and "transformers" not in tops


# ── `#77(a)`：标签索引键（修缺陷 `D1`） ──────────────────────────────────────
def test_label_index_key_is_unit_level(mod):
    """同一 (`sample_id`, `run_id`) 的**不同下游**必须能取得**不同**标签。

    旧实现键 = `(sample_id, run_id)` ⇒ 四下游复用同一标签（`D1`）。
    """
    recs = [{"split": "id_test", "sample_id": "S0", "downstream_model": "mistral-7b",
             "run_id": "none", "label": "AJ-Y"},
            {"split": "id_test", "sample_id": "S0", "downstream_model": "codellama-7b",
             "run_id": "none", "label": "AJ-N"}]
    rows = [{"sample_id": "S0", "downstream_model": "mistral-7b", "run_id": "none"},
            {"sample_id": "S0", "downstream_model": "codellama-7b", "run_id": "none"}]
    got = [r["aj_label"] for r in mod.attach_aj_labels(rows, recs, "id_test")]
    assert got == ["AJ-Y", "AJ-N"], got


def test_label_index_isolates_split(mod):
    """键并入 `split`：跨分片同键**不得**污染（`recs` 可能含历史合并记录）。"""
    recs = [{"split": "ood_test", "sample_id": "S0", "downstream_model": "mistral-7b",
             "run_id": "none", "label": "AJ-Y"}]
    rows = [{"sample_id": "S0", "downstream_model": "mistral-7b", "run_id": "none"}]
    assert mod.attach_aj_labels(rows, recs, "id_test")[0]["aj_label"] == ""


def test_merge_judge_csvs_order(mod, tmp_path):
    cols = mod.JUDGE_COLUMNS
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    base = {"split": "id_test", "sample_id": "S0", "downstream_model": "mistral-7b",
            "run_id": "none", "label": "AJ-Y", "http": "200"}
    mod.C.write_csv(str(a), cols, [dict(base)])
    mod.C.write_csv(str(b), cols, [dict(base, label="AJ-N")])
    assert [r["label"] for r in mod.merge_judge_csvs([str(a), str(b)])] == ["AJ-Y"]
    assert [r["label"] for r in mod.merge_judge_csvs([str(b), str(a)])] == ["AJ-N"]


# ── `#77(b)`：`--report-only` 重生成 ────────────────────────────────────────
def _write_units_two_downstreams(tmp_path):
    p = tmp_path / "units_none_id_test_20260101-000000.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sample_id", "scenario", "downstream_model", "run_id",
                    "payload_text", "model_output", "gen_params", "error_log", "df_label", "aj_label"])
        w.writerow(["S0", "none", "mistral-7b", "none", "PAY", "OUT-A", "temperature=0", "", "", ""])
        w.writerow(["S0", "none", "codellama-7b", "none", "PAY", "OUT-B", "temperature=0", "", "", ""])
    return p


def _write_judge_csv(tmp_path, labels):
    cols = ["sample_id", "scenario", "split", "downstream_model", "run_id", "label", "http",
            "attempt", "latency_s", "model_echo", "system_fingerprint", "fingerprint_sha256",
            "reasoning_tokens", "transport_failure", "call_date", "prompt_version_hash"]
    p = tmp_path / "judge_primary_20260101-000000.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for dm, lbl in labels:
            w.writerow({"sample_id": "S0", "scenario": "none", "split": "id_test",
                        "downstream_model": dm, "run_id": "none", "label": lbl, "http": "200",
                        "attempt": 1, "latency_s": 0.1, "model_echo": "deepseek-v4-pro",
                        "system_fingerprint": "fp", "fingerprint_sha256": "deadbeef",
                        "reasoning_tokens": 5, "transport_failure": False,
                        "call_date": "20260101-000000", "prompt_version_hash": "h"})
    return p


def test_report_only_needs_no_key_and_writes_no_csv(tmp_path):
    """`#77(b)`：重生成**不校验 key**、**不写新 judge CSV**，且矩阵按**单元级**出数。"""
    units = _write_units_two_downstreams(tmp_path)
    jcsv = _write_judge_csv(tmp_path, [("mistral-7b", "AJ-Y"), ("codellama-7b", "AJ-N")])
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    r = subprocess.run([sys.executable, SCRIPT, "--report-only",
                        "--units", str(units), "--judge-csv", str(jcsv),
                        "--report-dir", str(tmp_path), "--outputs-dir", str(tmp_path),
                        "--report-tag", "none_id_test", "--config", CONFIG],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "REGEN_REPORT=" in r.stdout and "n_csv_written=0" in r.stdout
    assert not list(tmp_path.glob("judge_primary_202601*-*.csv")) or \
        len(list(tmp_path.glob("judge_primary_*.csv"))) == 1        # 只有输入那一份
    reps = list(tmp_path.glob("asr_chain_regen_none_id_test_v1.0_*.md"))
    assert len(reps) == 1
    text = reps[0].read_text(encoding="utf-8")
    assert "REPORT_ONLY" in text and "D-114" in text
    # 表体不被注记截断（`D3`）：表头下一行是分隔行
    lines = text.split("\n")
    h = next(i for i, l in enumerate(lines) if l.startswith("| scenario | split | run_id |"))
    assert lines[h + 1].startswith("|---")
    assert not lines[h + 2].startswith(">")
    # 单元级：**同一 (`sample_id`, `run_id`) 的两个下游必须各得其标签**
    # （若键缺 `downstream_model` ⇒ 两行都会等于最后一条 ⇒ 本断言必失败）
    body = []
    for l in lines[h + 2:]:
        if not l.startswith("|"):
            break
        body.append([c.strip() for c in l.strip().strip("|").split("|")])
    assert len(body) == 2, body
    by_dm = {c[3]: (c[5], c[6]) for c in body}          # downstream -> (AJ-Y, AJ-N)
    assert by_dm == {"mistral-7b": ("1", "0"), "codellama-7b": ("0", "1")}, by_dm


def test_report_only_requires_judge_csv_and_tag(tmp_path):
    units = _write_units_two_downstreams(tmp_path)
    r = subprocess.run([sys.executable, SCRIPT, "--report-only", "--units", str(units),
                        "--report-dir", str(tmp_path), "--config", CONFIG],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "REPORT_ONLY_REQUIRES_JUDGE_CSV" in r.stderr
    jcsv = _write_judge_csv(tmp_path, [("mistral-7b", "AJ-Y")])
    r2 = subprocess.run([sys.executable, SCRIPT, "--report-only", "--units", str(units),
                         "--judge-csv", str(jcsv), "--report-dir", str(tmp_path), "--config", CONFIG],
                        capture_output=True, text=True)
    assert r2.returncode == 2 and "REPORT_ONLY_REQUIRES_REPORT_TAG" in r2.stderr


# ── `D-130`／`#93`：`--rejudged-keys`（124 条改判） ─────────────────────────
def test_attach_rejudged_flips_only_hit_AJY(mod):
    """只覆写**命中且原值 `AJ-Y`** 的三条；`AJ-N`／`不适用`／空值**逐位不变**。"""
    recs = [{"split": "id_test", "sample_id": f"S{i}", "downstream_model": "mistral-7b",
             "run_id": "none", "label": lab}
            for i, lab in enumerate(["AJ-Y", "AJ-N", "不适用", ""])]
    rows = [{"sample_id": f"S{i}", "downstream_model": "mistral-7b", "run_id": "none"}
            for i in range(4)]
    rj = {"id_test|S0|mistral-7b|none", "id_test|S1|mistral-7b|none",
          "id_test|S2|mistral-7b|none", "id_test|S9|mistral-7b|none"}
    got = [r["aj_label"] for r in mod.attach_aj_labels(rows, recs, "id_test", rejudged=rj)]
    assert got == ["AJ-N", "AJ-N", "不适用", ""], got


def test_unit_uid_is_four_key(mod):
    assert mod.unit_uid("id_test", {"sample_id": "S0", "downstream_model": "m",
                                    "run_id": "none"}) == "id_test|S0|m|none"


def test_report_only_applies_rejudged_keys(tmp_path):
    """`--report-only --rejudged-keys`：命中单元 `AJ-Y`→`AJ-N`，其余不变；报告头留痕。"""
    units = _write_units_two_downstreams(tmp_path)
    jcsv = _write_judge_csv(tmp_path, [("mistral-7b", "AJ-Y"), ("codellama-7b", "AJ-Y")])
    rj = tmp_path / "ajy_rejudged_test.json"
    rj.write_text(__import__("json").dumps([
        {"split": "id_test", "sample_id": "S0", "downstream_model": "mistral-7b",
         "run_id": "none"}], ensure_ascii=False), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    r = subprocess.run([sys.executable, SCRIPT, "--report-only",
                        "--units", str(units), "--judge-csv", str(jcsv),
                        "--report-dir", str(tmp_path), "--report-tag", "none_id_test",
                        "--rejudged-keys", str(rj), "--config", CONFIG],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "n_rejudged_applied=1" in r.stdout, r.stdout
    text = list(tmp_path.glob("asr_chain_regen_none_id_test_v1.0_*.md"))[0].read_text(encoding="utf-8")
    assert "改判覆盖" in text and "D-130" in text
    lines = text.splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("| scenario | split | run_id |"))
    body = []
    for l in lines[h + 2:]:
        if not l.startswith("|"):
            break
        body.append([c.strip() for c in l.strip().strip("|").split("|")])
    by_dm = {c[3]: (c[5], c[6]) for c in body}
    assert by_dm == {"mistral-7b": ("0", "1"), "codellama-7b": ("1", "0")}, by_dm

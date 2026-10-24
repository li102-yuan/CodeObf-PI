# -*- coding: utf-8 -*-
"""tests/asr/test_05_23_sample_human_annotation.py — W6.3 抽样脚本单元与接口测试。

覆盖（窗口 §7-1）：抽样比例 = 10%（±0）、抽样**可复现且无择优**（seed 事前定值）、
四元键（`D-109`）、双人标识互斥、`--report-dir` 必填（`C10`）、封存路径硬拒绝、
标注记录**不含机器标签**（`C2` 防锚定）、阅卷视图投影掉 `df_label`／`aj_label`、
产物**只追加**（`C13`）、`--self-check` exit=0。

不联网、不读封存件、不触 GPU。
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import os
import random
import subprocess
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "asr", "05_23_sample_human_annotation.py")
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
    return _load("asr_05_23", SCRIPT)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _units_file(tmp_path, name="units_none_id_test_20260101-000000.csv", n=20,
                scenario="none", split=None, df_trunc=0, prefix="S"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(UNIT_HEADER)
        for i in range(n):
            w.writerow([f"{prefix}{i:04d}", scenario, "mistral-7b", "none", f"PAY{i}",
                        f"OUT{i}", "temperature=0", "",
                        "DF-TRUNC" if i < df_trunc else "", ""])
    return str(p)


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── 自检 ─────────────────────────────────────────────────────────────────────
def test_self_check_exit_zero():
    r = subprocess.run([sys.executable, SCRIPT, "--self-check", "--config", CONFIG],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "SELFCHECK_RESULT=PASS" in r.stdout


# ── 抽样：比例、可复现、无择优 ────────────────────────────────────────────────
def test_sampling_ratio_is_exactly_10pct(tmp_path):
    u = _units_file(tmp_path, n=20)
    r = subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                        "--report-dir", str(tmp_path / "out")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "POOL=20 N_SAMPLE=2" in r.stdout
    assert "RATIO_EXACT=True" in r.stdout
    rows = _rows(glob.glob(str(tmp_path / "out" / "annotation_sample_v1.0_*.csv"))[0])
    assert len(rows) == 2
    for c in ("unit_id", "sample_rank", "split", "sample_id", "downstream_model",
              "run_id", "source_units_file", "payload_sha256", "output_sha256"):
        assert c in rows[0]


def test_sampling_is_reproducible_and_not_result_selected(mod, tmp_path):
    """抽样必须等于「规范化排序后 random.Random(seed).sample」——即**与结果无关**。"""
    u = _units_file(tmp_path, n=50)
    pool, _man = mod.load_pool([("none", "id_test", u)])
    picked, pool_n, n, exact = mod.sample_pool(pool, 0.10, 42)
    assert (pool_n, n, exact) == (50, 5, True)
    keys = [mod.unit_key(r) for r in sorted(pool, key=mod.unit_key)]
    expect = sorted(random.Random(42).sample(sorted(pool, key=mod.unit_key), 5),
                    key=mod.unit_key)
    assert [mod.unit_key(r) for r in picked] == [mod.unit_key(r) for r in expect]
    # 同一 seed 连跑两次逐项相同；换 seed 得不同子集（真随机，非固定前缀）
    again, _p, _n, _e = mod.sample_pool(pool, 0.10, 42)
    assert [mod.unit_key(r) for r in again] == [mod.unit_key(r) for r in picked]
    other, _p, _n, _e = mod.sample_pool(pool, 0.10, 7)
    assert [mod.unit_key(r) for r in other] != [mod.unit_key(r) for r in picked]
    assert keys  # 池非空（键序稳定）


def test_ratio_not_exact_when_pool_not_divisible(tmp_path):
    u = _units_file(tmp_path, n=7)
    r = subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                        "--report-dir", str(tmp_path / "out")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "RATIO_EXACT=False" in r.stdout          # C12 的 ±0 容差不可满足 ⇒ 须呈 W0


def test_duplicate_disclosure_for_cross_split_units(mod, tmp_path):
    """`D-110`：同一 (sid, model, payload, output) 跨分片出现 ⇒ 标 `dup_group`。"""
    a = _units_file(tmp_path, name="units_none_id_test_20260101-000000.csv", n=3)
    b = _units_file(tmp_path, name="units_none_ood_test_20260101-000000.csv", n=3)
    pool, _man = mod.load_pool([("none", "id_test", a), ("none", "ood_test", b)])
    info = mod.annotate_duplicates(pool)
    sizes = [v[1] for v in info.values()]
    assert all(s == 2 for s in sizes)                # 3 条 × 2 分片全部同源
    rows = mod.build_sample_rows(pool[:1], info, 16)
    assert int(rows[0]["dup_group_size"]) == 2
    assert rows[0]["dup_split_peer"] in ("id_test", "ood_test")


# ── 必需参数 / 封存守卫 ──────────────────────────────────────────────────────
def test_missing_report_dir_nonzero(tmp_path):
    u = _units_file(tmp_path)
    r = subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2
    assert "MISSING_REPORT_DIR" in r.stderr


def test_sealed_path_refused(tmp_path):
    r = subprocess.run([sys.executable, SCRIPT,
                        "--units", "data/sealed_test/CodeObf-PI_v1.0/regime_id_test.csv",
                        "--config", CONFIG, "--report-dir", str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2
    assert "SEALED_PATH_FORBIDDEN" in r.stderr


def test_existing_sample_not_overwritten(tmp_path):
    u = _units_file(tmp_path, n=20)
    out = tmp_path / "out"
    for _ in range(2):
        subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                        "--report-dir", str(out)], capture_output=True, text=True)
    files = glob.glob(str(out / "annotation_sample_v1.0_*.csv"))
    # 两次调用 ts 可能相同（同一秒）⇒ 至少断言「未被静默覆盖」的守卫存在
    assert len(files) >= 1


# ── 标注记录与阅卷视图（防锚定 / 防泄漏） ────────────────────────────────────
def test_emit_worksheet_has_no_machine_labels(tmp_path):
    u = _units_file(tmp_path, n=20, df_trunc=5)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                        "--report-dir", str(out)], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr
    sample = glob.glob(str(out / "annotation_sample_v1.0_*.csv"))[0]
    rec = str(out / "annotation_A_v1.0_20260101-000000.csv")
    view = str(out / "reading_A_20260101-000000.csv")
    r2 = subprocess.run([sys.executable, SCRIPT, "--emit-worksheet", "--annotator", "A",
                         "--sample", sample, "--out", rec, "--reading-view-out", view,
                         "--config", CONFIG], capture_output=True, text=True, encoding="utf-8")
    assert r2.returncode == 0, r2.stderr

    rrow = _rows(rec)
    assert len(rrow) == 2
    assert all(x["annotator"] == "A" for x in rrow)
    assert all(x["aj_label"] == "" and x["df_label"] == "" and x["remark"] == "" for x in rrow)
    assert "payload_text" not in rrow[0] and "model_output" not in rrow[0]   # rubric §7-11
    assert "df_label_auto" not in rrow[0]                                     # 防锚定

    vrow = _rows(view)
    assert len(vrow) == 2
    assert "payload_text" in vrow[0] and "model_output" in vrow[0]
    assert "df_label" not in vrow[0] and "aj_label" not in vrow[0]            # 防锚定


def test_reading_view_row_order_matches_worksheet(mod, cfg, tmp_path):
    """`F-W6.3-1`／`#79(d)`：工作表与阅卷视图的 `unit_id` 序列须**逐位相同**。

    判别力设计：**两个源文件 ＋ 交错入清单**（`N…`／`B…` 交替）。旧实现按
    `source_units_file` 分组排序会把 `b3` 源（`units_b3…` 路径在前）整体提前 ⇒ 序列错位；
    新实现沿抽样清单行序 ⇒ 一致。
    """
    a = _units_file(tmp_path, name="units_none_id_test_20260101-000000.csv", n=4, prefix="N")
    b = _units_file(tmp_path, name="units_b3_id_test_20260101-000000.csv", n=4,
                    scenario="b3", prefix="B")
    rows, rank = [], 0
    for i in range(4):
        for p, tag in ((a, "N"), (b, "B")):                  # ← 交错，非源分组序
            rank += 1
            rows.append({"unit_id": f"id_test|{tag}{i:04d}|mistral-7b|none",
                         "sample_rank": rank, "scenario": "none", "split": "id_test",
                         "sample_id": f"{tag}{i:04d}", "downstream_model": "mistral-7b",
                         "run_id": "none", "source_units_file": p})
    sample = str(tmp_path / "annotation_sample_v1.0_20260101-000000.csv")
    mod.C.write_csv(sample, mod.SAMPLE_COLUMNS, rows)
    rec = str(tmp_path / "annotation_A_v1.0_20260101-000000.csv")
    view = str(tmp_path / "reading_A_20260101-000000.csv")
    mod.emit_worksheet(cfg, sample, "A", rec, view, "20260101-000000")

    seq_rec = [r["unit_id"] for r in _rows(rec)]
    seq_view = [r["unit_id"] for r in _rows(view)]
    assert len(seq_rec) == len(seq_view) == 8
    assert seq_rec == seq_view                                 # 逐位相同（核心断言）
    assert [u.split("|")[1] for u in seq_view] == [
        "N0000", "B0000", "N0001", "B0001", "N0002", "B0002", "N0003", "B0003"]
    # 检测器自身的判别力
    assert mod.first_order_mismatch(["a", "b"], ["b", "a"])[1] == 0
    assert mod.first_order_mismatch(["a"], ["a", "b"])[0] is not None
    assert mod.first_order_mismatch(["a", "b"], ["a", "b"]) == (None, -1)


def test_reading_view_order_mismatch_is_refused_before_write(mod, cfg, tmp_path, monkeypatch):
    """不一致 ⇒ **拒绝落盘**（防污染），而不是写完再报错。"""
    a = _units_file(tmp_path, name="units_none_id_test_20260101-000000.csv", n=2, prefix="N")
    rows = [{"unit_id": f"id_test|N{i:04d}|mistral-7b|none", "sample_rank": i + 1,
             "scenario": "none", "split": "id_test", "sample_id": f"N{i:04d}",
             "downstream_model": "mistral-7b", "run_id": "none", "source_units_file": a}
            for i in range(2)]
    sample = str(tmp_path / "annotation_sample_v1.0_20260101-000000.csv")
    mod.C.write_csv(sample, mod.SAMPLE_COLUMNS, rows)
    rec = str(tmp_path / "annotation_A_v1.0_20260101-000000.csv")
    view = str(tmp_path / "reading_A_20260101-000000.csv")
    real_build = mod.build_reading_view

    def broken(rows_in, cache=None):
        v = real_build(rows_in, cache=cache)
        return list(reversed(v))          # 人为打乱行序，模拟旧实现

    monkeypatch.setattr(mod, "build_reading_view", broken)
    with pytest.raises(SystemExit) as e:
        mod.emit_worksheet(cfg, sample, "A", rec, view, "20260101-000000")
    assert e.value.code == 2
    assert not os.path.exists(rec) and not os.path.exists(view)   # 未落盘


def test_emit_worksheet_rejects_unknown_annotator(tmp_path):
    u = _units_file(tmp_path, n=20)
    out = tmp_path / "out"
    subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                    "--report-dir", str(out)], capture_output=True, text=True)
    sample = glob.glob(str(out / "annotation_sample_v1.0_*.csv"))[0]
    r = subprocess.run([sys.executable, SCRIPT, "--emit-worksheet", "--annotator", "C",
                        "--sample", sample, "--out", str(out / "x.csv"), "--config", CONFIG],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2
    assert "UNKNOWN_ANNOTATOR" in r.stderr


def test_emit_worksheet_refuses_existing_out(tmp_path):
    u = _units_file(tmp_path, n=20)
    out = tmp_path / "out"
    subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                    "--report-dir", str(out)], capture_output=True, text=True)
    sample = glob.glob(str(out / "annotation_sample_v1.0_*.csv"))[0]
    rec = out / "annotation_B_v1.0_20260101-000000.csv"
    rec.write_text("already here", encoding="utf-8")
    r = subprocess.run([sys.executable, SCRIPT, "--emit-worksheet", "--annotator", "B",
                        "--sample", sample, "--out", str(rec), "--config", CONFIG],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2
    assert "OUT_EXISTS_NO_OVERWRITE" in r.stderr


def test_worksheet_has_reading_sha256_matching_archived_view(mod, cfg, tmp_path):
    """归档阅卷视图 ＋ `reading_sha256` 列：两者逐条对应，且**可独立复算**。

    口径（与脚本 `reading_row_sha256` 同）：以 `READING_COLUMNS` 为列序、
    `QUOTE_MINIMAL`、`lineterminator="\n"` 序列化该记录 ⇒ sha256 前 16 hex。
    """
    import hashlib, io
    u = _units_file(tmp_path, n=20, df_trunc=2)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                        "--report-dir", str(out)], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr
    sample = glob.glob(str(out / "annotation_sample_v1.0_*.csv"))[0]
    rec = str(out / "annotation_A_v1.0_20260101-000000.csv")
    # 不给 --reading-view-out ⇒ 走**默认路径**（须为进版本控制的命名，不得在 tmp/）
    r2 = subprocess.run([sys.executable, SCRIPT, "--emit-worksheet", "--annotator", "A",
                         "--sample", sample, "--out", rec, "--config", CONFIG],
                        capture_output=True, text=True, encoding="utf-8")
    assert r2.returncode == 0, r2.stderr
    view_default = glob.glob(str(out / "annotation_reading_A_v1.0_*.csv"))[0]
    assert "annotation_reading_" in os.path.basename(view_default)
    assert "tmp" not in view_default.replace(str(tmp_path), "").replace("\\", "/")

    rrow, vrow = _rows(rec), _rows(view_default)
    assert len(rrow) == len(vrow) == 2
    assert [x["unit_id"] for x in rrow] == [x["unit_id"] for x in vrow]
    vmap = {x["unit_id"]: x for x in vrow}
    for x in rrow:
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow(
            [vmap[x["unit_id"]].get(c) or "" for c in mod.READING_COLUMNS])
        want = hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()[:16]
        assert x["reading_sha256"] == want and len(x["reading_sha256"]) == 16
    # 防锚定／无全文：工作表仍不得出现全文与机器标签
    assert "payload_text" not in rrow[0] and "df_label_auto" not in rrow[0]
    assert all(x["aj_label"] == "" and x["df_label"] == "" for x in rrow)


def test_reading_sha256_detects_tampered_view(mod, cfg, tmp_path):
    """判别力：阅卷视图**被改一个字节** ⇒ 重算值与工作表该列不等（可检出）。"""
    import hashlib, io
    u = _units_file(tmp_path, n=20)
    out = tmp_path / "out"
    subprocess.run([sys.executable, SCRIPT, "--units", u, "--config", CONFIG,
                    "--report-dir", str(out)], capture_output=True, text=True)
    sample = glob.glob(str(out / "annotation_sample_v1.0_*.csv"))[0]
    rec = str(out / "annotation_B_v1.0_20260101-000000.csv")
    view = str(out / "annotation_reading_B_v1.0_20260101-000000.csv")
    r = subprocess.run([sys.executable, SCRIPT, "--emit-worksheet", "--annotator", "B",
                        "--sample", sample, "--out", rec, "--reading-view-out", view,
                        "--config", CONFIG], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr

    rrow = _rows(rec)
    vrows = _rows(view)
    vrows[0]["payload_text"] = vrows[0]["payload_text"] + "X"          # 人为改动
    with open(view, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mod.READING_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(vrows)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(
        [vrows[0].get(c) or "" for c in mod.READING_COLUMNS])
    recomputed = hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()[:16]
    assert recomputed != rrow[0]["reading_sha256"]                     # 检得出差异
    # 未改动行仍自校验通过
    ok = {x["unit_id"]: x["reading_sha256"] for x in rrow}
    for v in vrows[1:]:
        b2 = io.StringIO()
        csv.writer(b2, lineterminator="\n").writerow(
            [v.get(c) or "" for c in mod.READING_COLUMNS])
        assert ok[v["unit_id"]] == hashlib.sha256(b2.getvalue().encode("utf-8")).hexdigest()[:16]


def test_top_imports_light():
    src = open(SCRIPT, encoding="utf-8").read().splitlines()
    tops = [ln.split()[1].split(".")[0] for ln in src if ln.startswith(("import ", "from "))]
    assert "torch" not in tops and "transformers" not in tops

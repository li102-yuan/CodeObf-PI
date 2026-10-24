# -*- coding: utf-8 -*-
"""tests/baseline/test_03_00_smoke_baseline.py — 阶段3 基线冒烟（v5 新增 22）。

覆盖 `windows/W4.7_smoke_report.md` §0.4 的冒烟约束：

1. **冒烟样本取自验证分片**：`build_subset` 确定性、正负分层、`sample_id` 升序，一类不足时确定性补齐；
2. **schema 校验**：`check_schema` 能识别列名/列序、行数、`text_sha256`、`pred` 的异常；
3. **零改动上游**：临时配置副本只重定向 `id_val.path` 与 `output.directory`，其余常量逐字保留；
4. **指标校验**：`check_metrics` 能识别缺失的检测器行；
5. **硬约束**：顶层不 import 重型库；
6. **`--dry-run` 端到端**：搭工作区、打印命令、退出 0，且**不写入权威产物目录**。

**离线**：不加载任何模型、不读 `data/sealed_test/`；`--dry-run` 的日志与工作区均改指 `tmp_path`。
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "baseline", "03_00_smoke_baseline.py")
UNIFY_CONFIG = os.path.join(ROOT, "configs", "baseline", "unify.yaml")
B4_CONFIG = os.path.join(ROOT, "configs", "baseline", "b4_unplug.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w47_smoke", SCRIPT)


def _write_rows(path, rows, fields=("sample_id", "text", "label")):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _synthetic_rows(n_pos=30, n_neg=30):
    rows = []
    for i in range(n_pos):
        rows.append({"sample_id": "P%03d" % i, "text": "attack %d" % i, "label": "1"})
    for i in range(n_neg):
        rows.append({"sample_id": "N%03d" % i, "text": "benign %d" % i, "label": "0"})
    return rows


# ---------------------------------------------------------------------------
# ① 冒烟样本抽取
# ---------------------------------------------------------------------------
def test_build_subset_deterministic_and_stratified(mod):
    rows = _synthetic_rows()
    a, meta_a = mod.build_subset(rows, 20, 42)
    b, meta_b = mod.build_subset(rows, 20, 42)
    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b], "同种子须可复现"
    assert meta_a == meta_b
    assert meta_a["n_taken"] == 20
    assert meta_a["taken_pos"] == 10 and meta_a["taken_neg"] == 10, "正负分层各半"


def test_build_subset_order_is_sample_id_ascending(mod):
    rows = _synthetic_rows()
    subset, _ = mod.build_subset(rows, 20, 42)
    ids = [r["sample_id"] for r in subset]
    assert ids == sorted(ids), "输出顺序须按 sample_id 升序"


def test_build_subset_seed_changes_choice(mod):
    rows = _synthetic_rows()
    a, _ = mod.build_subset(rows, 20, 42)
    b, _ = mod.build_subset(rows, 20, 123)
    assert {r["sample_id"] for r in a} != {r["sample_id"] for r in b}, "换种子应换样本"


def test_build_subset_backfills_when_class_short(mod):
    rows = _synthetic_rows(n_pos=4, n_neg=40)          # 正类仅 4 条
    subset, meta = mod.build_subset(rows, 20, 42)
    assert len(subset) == 20
    assert meta["taken_pos"] == 4, "正类不足时取尽"
    assert meta["taken_neg"] == 16, "缺口由负类确定性补齐"


def test_read_rows_preserves_column_order(mod, tmp_path):
    path = str(tmp_path / "x.csv")
    _write_rows(path, _synthetic_rows(2, 2), fields=("sample_id", "text", "label"))
    rows, fields = mod.read_rows(path)
    assert fields == ["sample_id", "text", "label"]
    assert len(rows) == 4


# ---------------------------------------------------------------------------
# ② §2 schema 校验
# ---------------------------------------------------------------------------
def _good_output_rows(n=20):
    return [{"sample_id": "S%02d" % i, "text_sha256": "a" * 64, "pred": "1",
             "raw_type": "regex_hits", "raw": "[]", "score": "", "latency_ms": "0.1"}
            for i in range(n)]


def _write_output(path, rows, fields=None):
    fields = fields or list(mod_fields())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mod_fields():
    return ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score", "latency_ms"]


def test_check_schema_accepts_valid(mod, tmp_path):
    path = str(tmp_path / "b1_id_val_v1.0_x.csv")
    _write_output(path, _good_output_rows(20))
    assert mod.check_schema(path, 20) == []


def test_check_schema_flags_missing_column(mod, tmp_path):
    path = str(tmp_path / "b1_id_val_v1.0_x.csv")
    bad = [f for f in mod_fields() if f != "text_sha256"]
    _write_output(path, [{k: v for k, v in row.items() if k in bad}
                         for row in _good_output_rows(20)], fields=bad)
    assert any("列不符" in e for e in mod.check_schema(path, 20))


def test_check_schema_flags_row_count(mod, tmp_path):
    path = str(tmp_path / "b1_id_val_v1.0_x.csv")
    _write_output(path, _good_output_rows(19))
    assert any("行数" in e for e in mod.check_schema(path, 20))


def test_check_schema_flags_bad_digest_and_pred(mod, tmp_path):
    path = str(tmp_path / "b1_id_val_v1.0_x.csv")
    rows = _good_output_rows(20)
    rows[0]["text_sha256"] = "not-a-sha"
    _write_output(path, rows)
    assert any("text_sha256" in e for e in mod.check_schema(path, 20))

    rows = _good_output_rows(20)
    rows[3]["pred"] = "2"
    _write_output(path, rows)
    assert any("pred" in e for e in mod.check_schema(path, 20))


def test_check_schema_allows_empty_pred_for_unparsed(mod, tmp_path):
    path = str(tmp_path / "b3_id_val_v1.0_x.csv")
    rows = _good_output_rows(20)
    rows[0]["pred"] = ""                                # B3/B5 的 unparsed
    _write_output(path, rows)
    assert mod.check_schema(path, 20) == []


# ---------------------------------------------------------------------------
# ③ 指标表校验 ／ ④ find_output
# ---------------------------------------------------------------------------
def test_check_metrics_flags_missing_detector(mod, tmp_path):
    path = str(tmp_path / "unify_metrics_v1.0_x.csv")
    rows = [{"detector": d, "split": "id_val", "n_pos": "10", "n_neg": "10",
             "tpr": "1.0", "fpr": "0.0", "f1": "1.0"} for d in ("b1", "b2", "b3", "b4")]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    errors, _ = mod.check_metrics(path, 20)
    assert any("b5" in e for e in errors)


def test_check_metrics_flags_sample_count_mismatch(mod, tmp_path):
    path = str(tmp_path / "unify_metrics_v1.0_x.csv")
    rows = [{"detector": d, "split": "id_val", "n_pos": "10", "n_neg": "10"} for d in mod.DETECTORS]
    rows[0]["n_neg"] = "9"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    errors, _ = mod.check_metrics(path, 20)
    assert any("样本数" in e for e in errors)


def test_find_output_picks_latest(mod, tmp_path):
    for ts in ("20260101-000000", "20260102-000000"):
        _write_output(str(tmp_path / ("b1_id_val_v1.0_%s.csv" % ts)), _good_output_rows(20))
    assert mod.find_output(str(tmp_path), "b1").endswith("20260102-000000.csv")
    assert mod.find_output(str(tmp_path), "b2") is None


# ---------------------------------------------------------------------------
# ⑤ 临时配置副本：只重定向两处
# ---------------------------------------------------------------------------
def test_smoke_unify_config_only_redirects_id_val(mod, tmp_path):
    dst = str(tmp_path / "unify_smoke.yaml")
    cfg = mod.make_smoke_unify_config(dst, str(tmp_path / "subset.csv"))
    import yaml
    src = yaml.safe_load(open(UNIFY_CONFIG, encoding="utf-8"))
    out = yaml.safe_load(open(dst, encoding="utf-8"))

    assert out["splits"]["id_val"]["path"] != src["splits"]["id_val"]["path"]
    for split in ("id_test", "ood_test", "clean_test", "ood_val"):
        assert out["splits"][split] == src["splits"][split], "%s 不得改动" % split
    for key in ("preprocess", "model_input", "output_schema", "mapping", "tau_search",
                "metrics", "span_truth"):
        assert out[key] == src[key], "%s 不得改动" % key
    assert cfg["splits"]["id_val"]["role"] == "smoke_subset_of_id_val"


def test_smoke_b4_config_only_redirects_output_dir(mod, tmp_path):
    dst = str(tmp_path / "b4_smoke.yaml")
    mod.make_smoke_b4_config(dst, "reports/baseline/tmp/x/outputs")
    import yaml
    src = yaml.safe_load(open(B4_CONFIG, encoding="utf-8"))
    out = yaml.safe_load(open(dst, encoding="utf-8"))
    assert out["output"]["directory"] == "reports/baseline/tmp/x/outputs"
    assert {k: v for k, v in out["output"].items() if k != "directory"} == \
           {k: v for k, v in src["output"].items() if k != "directory"}
    for key in ("abstain", "pipeline", "load", "model"):
        if key in src:
            assert out[key] == src[key], "%s 不得改动" % key


# ---------------------------------------------------------------------------
# ⑥ 硬约束：顶层不 import 重型库
# ---------------------------------------------------------------------------
def test_no_heavy_imports_at_top_level():
    tree = ast.parse(open(SCRIPT, encoding="utf-8").read())
    heavy = {"torch", "transformers", "numpy", "yaml", "pandas"}
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not (top & heavy), "顶层不得 import 重型库：%s" % sorted(top & heavy)


def test_not_implemented_placeholder_gone():
    body = open(SCRIPT, encoding="utf-8").read()
    assert "NOT_IMPLEMENTED" not in body, "占位骨架已被实现替代"


# ---------------------------------------------------------------------------
# ⑦ --dry-run 端到端（沙箱：日志与工作区均落 tmp_path）
# ---------------------------------------------------------------------------
def test_dry_run_end_to_end(tmp_path):
    outputs_dir = os.path.join(ROOT, "reports", "baseline", "outputs")
    before = set(os.listdir(outputs_dir)) if os.path.isdir(outputs_dir) else set()
    canonical = os.path.join(ROOT, "reports", "baseline")
    before_reports = set(os.listdir(canonical)) if os.path.isdir(canonical) else set()

    proc = subprocess.run(
        [sys.executable, SCRIPT, "--dry-run",
         "--log-dir", str(tmp_path / "logs"), "--work-root", str(tmp_path / "work")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SMOKE_DRY_RUN_OK" in proc.stdout
    assert "DRY_RUN：未执行任何阶段" in proc.stdout

    work = tmp_path / "work"
    roots = list(work.glob("smoke_baseline_*"))
    assert len(roots) == 1
    subset = roots[0] / "smoke_id_val_subset.csv"
    assert subset.is_file()
    with open(subset, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20, "冒烟样本数 = 20"
    assert sorted(roots[0].glob("*.yaml")), "须生成两份临时配置副本"
    assert list((tmp_path / "logs").glob("03_00_smoke_baseline_*.log")), "须留冒烟日志"

    assert list((roots[0] / "outputs").glob("*.csv")) == [], "dry-run 不得产出检测器 CSV"
    after = set(os.listdir(outputs_dir)) if os.path.isdir(outputs_dir) else set()
    assert after == before, "dry-run 不得写入权威 outputs 目录"
    after_reports = set(os.listdir(canonical)) if os.path.isdir(canonical) else set()
    assert after_reports == before_reports, "dry-run 不得写入 reports/baseline 顶部"


def test_dry_run_uses_id_val_not_sealed(tmp_path):
    """冒烟命令表须只涉及 `id_val`（非 sealed），不得出现任何测试分片代号。"""
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--dry-run",
         "--log-dir", str(tmp_path / "logs"), "--work-root", str(tmp_path / "work")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0
    cmd_block = proc.stdout.split("命令表", 1)[1]
    assert "--splits id_val" in cmd_block
    for sealed_split in ("id_test", "ood_test", "clean_test"):
        assert sealed_split not in cmd_block, "冒烟不得使用测试分片 %s" % sealed_split

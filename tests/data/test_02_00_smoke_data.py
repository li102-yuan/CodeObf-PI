# -*- coding: utf-8 -*-
"""tests/data/test_02_00_smoke_data.py — W3.8 全流程冒烟（`windows/W3.8_freeze.md` §4）。

覆盖 §4 的四项要求：

1. 冒烟档产出**落在临时目录**、**不触碰 `data/interim/` 与 `data/frozen/`**
   （断言目录内逐文件字节数与 `mtime_ns` 不变）
2. `--full` 在**缺一个上游产物**时**失败**
3. 冒烟档**退出码为 0**
4. **离线可跑、不加载模型**（`HF_HUB_OFFLINE=1` ＋ `TRANSFORMERS_OFFLINE=1` 下仍 exit 0；
   且脚本源码不含任何模型加载 API）

**运行成本**：冒烟档是**真跑**四阶段（编码 → 嵌入 → 划分 → 断言），本机约 **85 秒**
（模块级 fixture 只跑一次，多项断言复用同一结果）。`--full` 的反例走**快速失败**路径
（产物已缺 ⇒ 不重跑 `02_07`），秒级返回。
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_00_smoke_data.py")
CONFIG = os.path.join(ROOT, "configs", "data", "freeze.yaml")

#: 冒烟档**不得**触碰的两个目录（`windows/W3.8_freeze.md` §6-8）
ISOLATED = ("data/interim", "data/frozen")


def _load_module():
    spec = importlib.util.spec_from_file_location("smoke_02_00", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_02_00"] = module
    spec.loader.exec_module(module)
    return module


def isolated_snapshot():
    """只读隔离面快照：`{相对路径: (字节数, mtime_ns)}`。"""
    snap = {}
    for d in ISOLATED:
        base = os.path.join(ROOT, d)
        for cur, dirs, files in os.walk(base):
            dirs.sort()
            for name in sorted(files):
                p = os.path.join(cur, name)
                st = os.stat(p)
                snap[os.path.relpath(p, ROOT).replace("\\", "/")] = (st.st_size, st.st_mtime_ns)
    return snap


def run(argv):
    mod = _load_module()
    return mod.main(argv)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """**真跑一次**冒烟档（落 `tmp_path`），返回 (exit, stage_dir, report_dir, isolated 前后快照)。"""
    tmp = tmp_path_factory.mktemp("smoke")
    stage = tmp / "stage"
    reports = tmp / "reports"
    before = isolated_snapshot()
    env_backup = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        rc = run(["--config", CONFIG, "--seeds", "10",
                  "--stage-dir", str(stage), "--report-dir", str(reports)])
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    after = isolated_snapshot()
    return rc, stage, reports, before, after


# ---------------------------------------------------------------------------
# §4-3 / §4-4：冒烟档退出码为 0；离线可跑
# ---------------------------------------------------------------------------
def test_smoke_exit_zero(smoke_run):
    rc, _stage, _reports, _b, _a = smoke_run
    assert rc == 0, "冒烟档必须恒为可跑通的一档"


def test_smoke_offline_no_model(smoke_run):
    """`HF_HUB_OFFLINE=1` ＋ `TRANSFORMERS_OFFLINE=1` 下仍 exit 0 ⇒ 离线可跑。"""
    rc, _stage, _reports, _b, _a = smoke_run
    assert rc == 0
    src = open(SCRIPT, encoding="utf-8").read()
    for banned in ("import torch", "from torch", "AutoModel", "from_pretrained",
                   "transformers"):
        assert banned not in src, "02_00 不得引用模型加载 API：%s" % banned


# ---------------------------------------------------------------------------
# §4-1：产物落临时目录；不触碰 data/interim 与 data/frozen
# ---------------------------------------------------------------------------
def test_smoke_outputs_land_in_stage_dir(smoke_run):
    _rc, stage, _reports, _b, _a = smoke_run
    runs = [p for p in stage.iterdir() if p.is_dir()]
    assert len(runs) == 1, "应恰好产生一个 run 目录"
    run_dir = runs[0]
    assert (run_dir / "generated" / "encoded" / "smoke" / "encoded.csv").is_file()
    assert (run_dir / "generated" / "code_carrier" / "smoke" / "code_carrier.csv").is_file()
    assert (run_dir / "generated" / "split" / "smoke" / "splits" / "regime_id"
            / "train.csv").is_file()
    assert (run_dir / "generated" / "split" / "smoke" / "split_assertions.csv").is_file()
    assert (run_dir / "smoke_stdout.log").is_file()


def test_smoke_does_not_touch_interim_or_frozen(smoke_run):
    _rc, _stage, _reports, before, after = smoke_run
    diff = {k: (before.get(k), after.get(k)) for k in set(before) | set(after)
            if before.get(k) != after.get(k)}
    assert not diff, "冒烟档改动了隔离面：%s" % list(diff.items())[:5]


def test_smoke_scale_is_derived_from_seeds(smoke_run):
    """规模**实测派生**（10 个种子 × 每组 2 条 = 20 改写体 → 180 编码变体）。"""
    import csv as _csv
    _rc, stage, _reports, _b, _a = smoke_run
    run_dir = [p for p in stage.iterdir() if p.is_dir()][0]
    p = run_dir / "generated" / "encoded" / "smoke" / "encoded.csv"
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(_csv.reader(fh))
    assert len(rows) - 1 == 180, "encoded.csv 数据行（载荷含换行，不得按 splitlines 计数）"


def test_smoke_report_written(smoke_run):
    _rc, _stage, reports, _b, _a = smoke_run
    files = sorted(reports.glob("smoke_data_v1.0_*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "模式（`mode`）：`smoke`" in text
    assert "结论（`conclusion`）：`PASS`" in text
    assert "隔离证据" in text


# ---------------------------------------------------------------------------
# §4-2：`--full` 在缺一个上游产物时失败
# ---------------------------------------------------------------------------
def _config_with_missing_product(tmp_path):
    """把配置复制一份，把**一个**上游产物指到不存在的路径。"""
    import yaml
    with open(CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["freeze"]["inputs"]["length_distribution"] = \
        "data/interim/__missing_for_test__/length_distribution.csv"
    out = tmp_path / "freeze_missing.yaml"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return str(out)


def test_full_fails_when_a_product_is_missing(tmp_path):
    """§4-2 的回归：缺**一个**上游产物 ⇒ 全量档失败（且**快速失败**，不重跑 `02_07`）。"""
    cfg = _config_with_missing_product(tmp_path)
    assert run(["--config", cfg, "--full",
                "--report-dir", str(tmp_path / "reports")]) != 0


def test_full_missing_product_report_marks_fail(tmp_path):
    cfg = _config_with_missing_product(tmp_path)
    reports = tmp_path / "reports"
    assert run(["--config", cfg, "--full", "--skip-assert-rerun",
                "--report-dir", str(reports)]) != 0
    files = sorted(reports.glob("smoke_data_v1.0_*.md"))
    assert files, "全量档须落盘报告（即使结论为失败）"
    text = files[0].read_text(encoding="utf-8")
    assert "模式（`mode`）：`full`" in text
    assert "结论（`conclusion`）：`FAIL`" in text
    assert "SKIPPED" in text, "产物已缺时应快速失败、不重跑 02_07"


def test_full_pass_report_identifiable_by_h6(tmp_path):
    """全量档**通过**路径的报告须可被 `02_08` 的 `H6` 识别（`mode: full` ＋ `conclusion: PASS`）。

    注：本测试用**真实** `configs/data/freeze.yaml`（产物齐备）＋ `--skip-assert-rerun`
    （跳过耗时的 `02_07` 默认模式重跑；该重跑由 `make smoke STAGE=data`／`02_08` 的 `H5` 覆盖）。
    """
    reports = tmp_path / "reports"
    assert run(["--config", CONFIG, "--full", "--skip-assert-rerun",
                "--report-dir", str(reports)]) == 0
    files = sorted(reports.glob("smoke_data_v1.0_*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "模式（`mode`）：`full`" in text
    assert "结论（`conclusion`）：`PASS`" in text

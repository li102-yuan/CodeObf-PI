# -*- coding: utf-8 -*-
"""tests/train/test_04_00_smoke_train.py — W5.1 `04_00` 训练冒烟脚本的单元测试。

覆盖 `windows/W5.1_qlora.md` §6-1：

1. 子集**固定不随机**（同类按 `sample_id` 升序取前 n 条）→ 两次调用结果一致；
2. 正 32／负 32 的**分层配额**；
3. `--dry-run` exit=0 且不加载模型；
4. 验收断言 `check_checkpoint`／`check_curve`／`check_threshold` 的判定行为（含失败路径）；
5. 冒烟产物**落 `reports/train/tmp/`**（不入权威目录）。

**不加载模型、不需要 GPU**（§4-6）。
"""
from __future__ import annotations

import csv as _csv
import importlib.util
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # tests/train -> 项目根
SMOKE = os.path.join(ROOT, "scripts", "train", "04_00_smoke_train.py")
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train", "04_01_train_qlora_ours.py")
SPLITS = os.path.join(ROOT, "data", "frozen", "CodeObf-PI_v1.0", "splits")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def s():
    return _load("t_04_00", SMOKE)


@pytest.fixture(scope="module")
def m():
    return _load("t_04_01_for_00", TRAIN_SCRIPT)


class _S:
    """最小替身：只需 `sample_id` 与 `label`（`Sample` 同构）。"""

    def __init__(self, sid, label):
        self.sample_id = sid
        self.label = label


def test_pick_row_ids_is_fixed_not_random(s):
    samples = [_S("s%03d" % i, i % 2) for i in range(40)]
    a, ma = s.pick_row_ids(samples, 8, 8)
    b, mb = s.pick_row_ids(samples, 8, 8)
    assert a == b, "子集必须**固定不随机**（两次调用须逐字节一致）"
    assert (ma["n_pos_taken"], ma["n_neg_taken"]) == (8, 8)
    assert a == sorted(a), "返回的源分片行下标须升序（保持样本顺序）"


def test_pick_row_ids_stratified_quota(s):
    samples = [_S("s%03d" % i, 1 if i < 20 else 0) for i in range(40)]
    ids, meta = s.pick_row_ids(samples, 32, 32)
    assert meta["n_pos_taken"] == 20 and meta["n_neg_taken"] == 20, "不足时按实有量给"
    labels = {i: samples[i].label for i in ids}
    assert sum(1 for v in labels.values() if int(v) == 1) == 20


def test_pick_row_ids_takes_smallest_sample_ids(s):
    """同类内按 `sample_id` 升序取前 n 条 ⇒ 取到的应是字典序最小的那些。"""
    samples = [_S("s%03d" % i, 1 if i < 10 else 0) for i in range(20)]
    ids, _meta = s.pick_row_ids(samples, 3, 2)
    got_pos = sorted(samples[i].sample_id for i in ids if int(samples[i].label) == 1)
    got_neg = sorted(samples[i].sample_id for i in ids if int(samples[i].label) == 0)
    assert got_pos == ["s000", "s001", "s002"]
    assert got_neg == ["s010", "s011"]


def test_pick_row_ids_on_real_train_split(s, m):
    train_csv = os.path.join(SPLITS, "regime_id", "train.csv")
    samples = m.load_samples(train_csv)
    ids, meta = s.pick_row_ids(samples, 32, 32)
    assert meta["n_taken"] == 64
    assert meta["n_pos_taken"] == 32 and meta["n_neg_taken"] == 32
    assert meta["n_source"] == len(samples) == 12672
    assert len(set(ids)) == 64


# ---------------------------------------------------------------------------
# 验收断言
# ---------------------------------------------------------------------------
def test_check_checkpoint_ok(s, tmp_path):
    d = tmp_path / "seed_42" / "checkpoint-7"
    (d).mkdir(parents=True)
    (d / "adapter_model.safetensors").write_text("x")
    (d / "adapter_config.json").write_text("{}")
    assert s.check_checkpoint(str(d), _silent()) == []


def test_check_checkpoint_missing_and_bad_naming(s, tmp_path):
    d = tmp_path / "plain_dir"
    d.mkdir()
    errs = s.check_checkpoint(str(d), _silent())
    assert errs and any("adapter" in e for e in errs)
    assert any("C8" in e for e in errs), "命名不合 C8 须报错"


def _silent():
    return lambda msg="": None


def test_check_curve_ok_and_bad(s, tmp_path):
    p = tmp_path / "curve.csv"
    with open(str(p), "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["epoch", "opt_steps", "train_loss", "val_f1",
                                            "val_tpr", "val_fpr", "tau_ours",
                                            "early_stop_counter"])
        w.writeheader()
        w.writerow(dict(epoch=1, opt_steps=7, train_loss=3.4, val_f1=0.88, val_tpr=0.9,
                        val_fpr=0.1, tau_ours=0.53, early_stop_counter=0))
    assert s.check_curve(str(p), 1, _silent()) == []
    assert s.check_curve(str(p), 3, _silent()), "实跑 epoch 数不符须报错"
    assert s.check_curve(str(tmp_path / "none.csv"), 1, _silent()), "文件缺失须报错"


def test_check_threshold_ok_and_bad(s, tmp_path):
    p = tmp_path / "thr.csv"
    with open(str(p), "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["tau", "f1", "tpr", "fpr", "selected"])
        w.writeheader()
        w.writerow(dict(tau="-inf", f1=0.66, tpr=1.0, fpr=1.0, selected=""))
        w.writerow(dict(tau="0.531209", f1=0.885, tpr=0.9, fpr=0.1, selected="yes"))
        w.writerow(dict(tau="inf", f1=0.0, tpr=0.0, fpr=0.0, selected=""))
    assert s.check_threshold(str(p), _silent()) == []

    p2 = tmp_path / "thr2.csv"
    with open(str(p2), "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["tau", "f1", "tpr", "fpr", "selected"])
        w.writeheader()
        w.writerow(dict(tau="1.5", f1=0.5, tpr=0.5, fpr=0.5, selected="yes"))
    assert s.check_threshold(str(p2), _silent()), "τ 越界 [0,1] 须报错"


# ---------------------------------------------------------------------------
# CLI：--dry-run 不加载模型、产物落 tmp
# ---------------------------------------------------------------------------
def test_dry_run_exit_zero(tmp_path):
    work = tmp_path / "work"
    logs = tmp_path / "logs"
    rc = subprocess.run(
        [sys.executable, SMOKE, "--dry-run", "--work-root", str(work),
         "--log-dir", str(logs)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert "SMOKE_DRY_RUN_OK" in rc.stdout
    assert "MODEL_LOADED" not in rc.stdout, "--dry-run 不得加载模型"


def test_default_work_root_is_tmp(s):
    args = s.build_parser().parse_args([])
    assert args.work_root == os.path.join("reports", "train", "tmp")
    assert args.n_pos == 32 and args.n_neg == 32 and args.seed == 42 and args.epochs == 1


def test_cli_rejects_bad_seed(s):
    with pytest.raises(SystemExit):
        s.build_parser().parse_args(["--seed", "7"])

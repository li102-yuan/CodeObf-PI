# -*- coding: utf-8 -*-
"""tests/eval/test_05_07_eval_efficiency.py — W6.8（5.7 效率分析）单元与接口测试。

不触 GPU、不加载模型、不读 `data/sealed_test/**`（本窗口默认测 `id_val`，frozen 侧）。
重型库（torch／transformers／peft）一律不在测试内 import —— 被测脚本顶层亦不 import。

覆盖（`windows/W6.8_efficiency.md` §7 验收项 ＋ §3 硬约束）：
1. 七元组**必填**（缺字段／空值 ⇒ 抛错；六检测器逐项核验）
2. `b1`／`b2` 无模型 ⇒ 设备与显存 **`n/a` 而不是 `0`**；参数量 `n/a`
3. 加载态／推理态**分列**（配置 `states` 与字段表双证）
4. `repeat` 的**中位与极差**（合并统计 ＋ 跨通过极差；禁挑最好的一次）
5. 封存门禁：`--bench-source sealed` 缺 `--access-log-note` ⇒ 拒绝执行
6. 确定性双跑（纯函数桩两次逐位相同）＋ 预热**不计入**
7. §8.5 不可比性声明**逐字在位**（去空白后为 `ENV_LOCK.md` 的子串）＋ 报告内出现
8. 三档量纲**分表不合并**（`no_model`／`cpu_model`／`gpu_model`）
9. 四表齐备（延迟／显存／参数量／7B 对照）＋ 7B 对照锚 = `b5`
10. 回退档**不静默丢弃**（`cited_of_record` 缺项 ⇒ 抛错；回退行标 `cited(baseline-v1)`）
11. C8／C9 静态纪律（无训练模式切换调用、无硬编码 `reports/`；`attn=eager`、TF32 关）
12. `--self-check` exit=0；参数量统计（safetensors 元数据，B3 = 1,498,482,688）
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import statistics
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_07_eval_efficiency.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "efficiency.yaml")
ENV_LOCK = os.path.join(_ROOT, "ENV_LOCK.md")


def _load():
    spec = importlib.util.spec_from_file_location("eval_05_07", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_05_07"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()
NA = M.NA

# needle 用拼接构造（否则本文件自己的字面量会被自己数到 ⇒ 假失败）
NEEDLE_TRAIN = "".join([".", "train("])
NEEDLE_REPORTS = "".join(["rep", "orts", "/ev", "al"])


@pytest.fixture(scope="module")
def cfg():
    return M.load_config(CONFIG)


@pytest.fixture(scope="module")
def src():
    with open(SCRIPT, "r", encoding="utf-8") as fh:
        return fh.read()


# ─────────────────────────────────────────────────────────────────────────────
# 1. 七元组必填
# ─────────────────────────────────────────────────────────────────────────────
def test_detectors_all_six(cfg):
    assert sorted(cfg["detectors"]) == ["b1", "b2", "b3", "b4", "b5", "ours"]


@pytest.mark.parametrize("det", ["b1", "b2", "b3", "b4", "b5", "ours"])
def test_seven_tuple_complete(cfg, det):
    seven = M.seven_tuple_of(det, cfg["detectors"][det])
    assert M.assert_seven_tuple(seven, det)
    assert sorted(seven) == sorted(M.SEVEN_KEYS)


def test_seven_tuple_missing_raises():
    partial = {"seven_tuple": {k: "x" for k in M.SEVEN_KEYS[:6]}}
    with pytest.raises(KeyError):
        M.seven_tuple_of("b1", partial)


def test_seven_tuple_empty_raises():
    st = {k: "x" for k in M.SEVEN_KEYS}
    st["device"] = "   "
    with pytest.raises(ValueError):
        M.seven_tuple_of("b1", {"seven_tuple": st})


# ─────────────────────────────────────────────────────────────────────────────
# 2. b1／b2 无模型 ⇒ n/a（**不是 0**）
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("det", ["b1", "b2"])
def test_no_model_device_is_na(cfg, det):
    seven = M.seven_tuple_of(det, cfg["detectors"][det])
    assert str(seven["device"]).startswith(NA)
    assert seven["device"] != 0


@pytest.mark.parametrize("det", ["b1", "b2"])
def test_no_model_params_are_na(cfg, det):
    total, trainable, source, _r, _a = M.params_for(det, cfg["detectors"][det])
    assert total is None and trainable is None
    assert total != 0 and trainable != 0


def test_no_model_row_memory_is_na_not_zero(cfg):
    """无模型档的行 ⇒ 加载态／推理态／RSS 全 `None`（渲染为 `n/a`），不得出数。"""
    dcfg = cfg["detectors"]["b1"]
    seven = M.seven_tuple_of("b1", dcfg)
    passes = [{"ms": [1.0, 2.0], "cpu_rss_peak_mib": 52.7, "is_warmup": False, "pass_index": 1}]
    row, _ps = M.build_row("b1", dcfg, cfg, seven,
                           {"mem_load_peak_mib": None, "cpu_rss_peak_mib": 52.7},
                           passes, 3, 1, 2, None, None)
    assert row["mem_load_peak_mib"] is None
    assert row["mem_infer_peak_mib"] is None
    assert row["cpu_rss_peak_mib"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. 加载态／推理态分列
# ─────────────────────────────────────────────────────────────────────────────
def test_memory_states_separate(cfg):
    assert list(cfg["memory"]["states"]) == ["load", "infer"]
    assert cfg["memory"]["forbid_merge"]
    assert cfg["memory"]["separate_columns"] is True


def test_main_fields_have_both_states():
    assert "mem_load_peak_mib" in M.MAIN_FIELDS
    assert "mem_infer_peak_mib" in M.MAIN_FIELDS
    assert M.MAIN_FIELDS.index("mem_load_peak_mib") != M.MAIN_FIELDS.index("mem_infer_peak_mib")


# ─────────────────────────────────────────────────────────────────────────────
# 4. repeat 中位与极差
# ─────────────────────────────────────────────────────────────────────────────
def test_repeat_pooled_and_range(cfg):
    passes = [{"ms": [10.0, 20.0, 30.0]}, {"ms": [20.0, 30.0, 40.0]}, {"ms": [30.0, 40.0, 50.0]}]
    rep = M.repeat_report(passes)
    pooled = [10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 30.0, 40.0, 50.0]
    assert rep["n_passes"] == 3
    assert abs(rep["lat_mean_ms"] - statistics.fmean(pooled)) < 1e-9
    assert abs(rep["lat_median_ms"] - statistics.median(pooled)) < 1e-9
    # 极差 = 各次通过该统计量的 max − min（此处中位 20/30/40 ⇒ 20）
    assert abs(rep["lat_median_range_ms"] - 20.0) < 1e-9
    assert abs(rep["lat_mean_range_ms"] - 20.0) < 1e-9


def test_repeat_exceed_flag(cfg):
    assert M.range_exceeds({"lat_median_ms": 10.0, "lat_median_range_ms": 6.0}, cfg) is True
    assert M.range_exceeds({"lat_median_ms": 10.0, "lat_median_range_ms": 4.0}, cfg) is False


def test_pass_stats_p95_nearest_rank():
    st = M.pass_stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    idx = min(9, int(round(0.95 * 9)))       # = 9
    assert abs(st["p95_ms"] - float(idx + 1)) < 1e-9
    assert abs(st["total_ms"] - math.fsum(range(1, 11))) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 5. 封存门禁
# ─────────────────────────────────────────────────────────────────────────────
def test_sealed_requires_access_log_note(cfg):
    class _A:
        bench_source = "sealed"
        bench_split = "id_val"
        limit = None
        access_log_note = None

    with pytest.raises(ValueError):
        M.load_bench_rows(cfg, _A(), lambda *a: None)


def test_sealed_policy_columns(cfg):
    assert cfg["sealed_policy"]["default"] == "not_read"
    assert cfg["sealed_policy"]["if_sealed"]["require_access_log_note"] is True
    assert len(cfg["sealed_policy"]["if_sealed"]["columns"]) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 6. 确定性双跑 ＋ 预热不计入
# ─────────────────────────────────────────────────────────────────────────────
class _Stub(M.BaseAdapter):
    def load(self):
        return {"mem_load_peak_mib": None, "cpu_rss_peak_mib": None}

    def infer_pass(self, rows, **kw):
        ms = [float(len(r["text"])) for r in rows]
        return {"ms": ms, "total_ms": math.fsum(ms)}


def test_determinism_double_run():
    rows = [{"text": "abc"}, {"text": "abcd"}]
    a = M._measure_one(_Stub("s", {}), rows, 2, 1, lambda *x: None)
    b = M._measure_one(_Stub("s", {}), rows, 2, 1, lambda *x: None)
    assert [p["ms"] for p in a] == [p["ms"] for p in b]
    assert len(a) == 2                       # 预热通过**不计入**
    assert all(p["is_warmup"] is False for p in a)


# ─────────────────────────────────────────────────────────────────────────────
# 7. §8.5 不可比性声明逐字在位
# ─────────────────────────────────────────────────────────────────────────────
def test_incomparability_verbatim(cfg):
    with open(ENV_LOCK, "r", encoding="utf-8") as fh:
        env = fh.read()
    v = cfg["incomparability"]["verbatim"]
    assert re.sub(r"\s+", "", v) in re.sub(r"\s+", "", env)


def test_report_contains_verbatim_and_four_tables(cfg):
    ctx = _fake_ctx(cfg)
    text = M.render_report(ctx)
    assert cfg["incomparability"]["verbatim"] in text
    for h in ["表一 · 推理延迟", "表二 · 显存峰值", "表三 · 模型参数量", "表四 · 与 7B 级"]:
        assert h in text, h
    assert "### 5.7 效率" in text              # 可直接搬入 EVAL_REPORT.md 的骨架段


def test_report_groups_not_merged(cfg):
    ctx = _fake_ctx(cfg)
    text = M.render_report(ctx)
    assert "无模型档" in text and "CPU 档" in text and "GPU 档" in text
    # 三档各自成节 ⇒ 各档标题均出现（不合并排序）
    for gid in ["no_model", "cpu_model", "gpu_model"]:
        assert ("`%s`" % gid) in text


# ─────────────────────────────────────────────────────────────────────────────
# 8. 三档分表
# ─────────────────────────────────────────────────────────────────────────────
def test_build_tables_grouped(cfg):
    rows = [{"group": "gpu_model"}, {"group": "no_model"},
            {"group": "cpu_model"}, {"group": "gpu_model"}]
    tabs = M.build_tables(rows, cfg)
    assert [t[0] for t in tabs] == ["no_model", "cpu_model", "gpu_model"]
    assert len(tabs[2][2]) == 2


def test_groups_members(cfg):
    assert cfg["groups"]["no_model"]["members"] == ["b1", "b2"]
    assert cfg["groups"]["cpu_model"]["members"] == ["b4"]
    assert sorted(cfg["groups"]["gpu_model"]["members"]) == ["b3", "b5", "ours"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. 7B 对照锚
# ─────────────────────────────────────────────────────────────────────────────
def test_anchor_is_b5(cfg):
    assert cfg["comparison_7b"]["anchor"] == "b5"
    assert "禁止" in cfg["comparison_7b"]["forbid_new_model"]
    assert cfg["comparison_7b"]["fairness_control"]["enabled"] is True


def test_7b_ratios_use_control(cfg):
    rows, ours_ctrl = _fake_rows(cfg)
    rows_7b, anchor = M.build_7b_rows(rows, cfg, ours_ctrl)
    assert anchor["detector"] == "b5"
    assert len(rows_7b) == 1
    r = rows_7b[0]
    assert abs(r["latency_ratio"] - (1000.0 / 100.0)) < 1e-9
    assert abs(r["mem_ratio"] - (5000.0 / 1000.0)) < 1e-9
    assert abs(r["param_ratio"] - (8e9 / 1.5e9)) < 1e-9
    assert "batch=1" in r["latency_basis"]


def test_7b_no_anchor_no_ratio(cfg):
    rows, _ = _fake_rows(cfg)
    no_b5 = [r for r in rows if r["detector"] != "b5"]
    rows_7b, anchor = M.build_7b_rows(no_b5, cfg, [])
    assert rows_7b == [] and anchor is None


# ─────────────────────────────────────────────────────────────────────────────
# 10. 回退档不静默丢弃
# ─────────────────────────────────────────────────────────────────────────────
def test_fallback_never_silent(cfg):
    bogus = {"label": "zz", "group": "gpu_model",
             "seven_tuple": {k: "x" for k in M.SEVEN_KEYS}}
    with pytest.raises(KeyError):
        M.fall_back_row("zz", bogus, cfg, "unit-test")


def test_fallback_row_is_cited(cfg):
    row = M.fall_back_row("b3", cfg["detectors"]["b3"], cfg, "unit-test OOM")
    assert row["measurement"] == "cited(baseline-v1)"
    assert row["lat_median_ms"] == cfg["cited_of_record"]["values"]["b3"]["median_ms"]
    assert "unit-test OOM" in row["notes"]


def test_cited_of_record_complete(cfg):
    vals = cfg["cited_of_record"]["values"]
    for k in ("b1", "b2", "b3", "b4", "b5"):
        assert k in vals
        for f in ("mean_ms", "median_ms", "p95_ms", "mem_load_mib", "mem_infer_mib"):
            assert f in vals[k]
    assert cfg["cited_of_record"]["tag"] == "baseline-v1"


# ─────────────────────────────────────────────────────────────────────────────
# 11. C8／C9 静态纪律
# ─────────────────────────────────────────────────────────────────────────────
def test_no_train_call_in_source(src):
    assert NEEDLE_TRAIN not in src


def test_no_hardcoded_reports_path(src):
    assert NEEDLE_REPORTS not in src
    assert chr(34) + "".join(["rep", "orts"]) + chr(34) not in src


def test_determinism_block(cfg):
    d = cfg["determinism"]
    assert d["cudnn_deterministic"] is True
    assert d["cudnn_benchmark"] is False
    assert d["allow_tf32"] is False
    assert d["attn_implementation"] == "eager"
    assert d["eval_state"]["no_grad"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 12. 参数量统计（safetensors 元数据，不加载张量）
# ─────────────────────────────────────────────────────────────────────────────
def test_count_params_b3(cfg):
    total, note = M.count_params_safetensors(cfg["detectors"]["b3"]["params"]["dir"])
    assert total == 1498482688, note
    assert "未加载张量" in note


def test_count_params_missing_dir():
    total, note = M.count_params_safetensors("no/such/dir")
    assert total is None and "不存在" in note


def test_params_for_b3_trainable_zero(cfg):
    total, trainable, note, _r, _a = M.params_for("b3", cfg["detectors"]["b3"])
    assert total == 1498482688
    assert trainable == 0
    assert "未微调" in note


def test_params_for_ours_from_best_json(cfg):
    best = {"model_meta": {"trainable_params": 1703936, "lora_r": 16, "lora_alpha": 32,
                           "local_path": cfg["detectors"]["b3"]["params"]["dir"]}}
    total, trainable, note, r, a = M.params_for("ours", cfg["detectors"]["ours"], best)
    assert total == 1498482688
    assert trainable == 1703936                      # C4：Ours 报 LoRA 可训练参
    assert r == 16 and a == 32


# ─────────────────────────────────────────────────────────────────────────────
# 13. CLI：`--self-check` exit=0（离线、不触 GPU）
# ─────────────────────────────────────────────────────────────────────────────
def test_selfcheck_exit_zero():
    out = subprocess.run([sys.executable, SCRIPT, "--self-check"],
                         capture_output=True, text=True, encoding="utf-8", cwd=_ROOT)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "SELFCHECK_RESULT=PASS" in out.stdout


def test_dryrun_exit_zero():
    """`--dry-run` 需 `ours` 的 run 来源（严格：缺即 FAIL，不静默跳过）。"""
    import glob
    cands = sorted(glob.glob(os.path.join(_ROOT, "reports", "train", "three_seeds_v1.0_*.csv")))
    assert cands, "缺少 04_03 汇总 CSV ⇒ dry-run 无法核验 ours 的 6 个 run"
    out = subprocess.run([sys.executable, SCRIPT, "--dry-run", "--runs-csv",
                          os.path.relpath(cands[-1], _ROOT)],
                         capture_output=True, text=True, encoding="utf-8", cwd=_ROOT)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "DRYRUN_RESULT=PASS" in out.stdout


def test_missing_report_dir_is_nonzero():
    out = subprocess.run([sys.executable, SCRIPT, "--bench-split", "id_val"],
                         capture_output=True, text=True, encoding="utf-8", cwd=_ROOT)
    assert out.returncode == 2
    assert "MISSING_DIRS" in out.stdout


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：构造假 ctx／假 rows（供渲染与 7B 比值测试）
# ─────────────────────────────────────────────────────────────────────────────
def _row(det, group, label, lat, mem_load, mem_infer, params, trainable=0):
    d = {
        "detector": det, "label": label, "group": group, "device": "cuda",
        "measurement": "measured(same_session)", "run_id": "run1", "regime": "id", "seed": 42,
        "lat_mean_ms": lat, "lat_median_ms": lat, "lat_p95_ms": lat * 1.5,
        "lat_mean_range_ms": 1.0, "lat_median_range_ms": 1.0, "lat_p95_range_ms": 1.0,
        "n_samples": 48, "n_passes": 3, "repeat": 3, "warmup": 1,
        "mem_load_peak_mib": mem_load, "mem_infer_peak_mib": mem_infer,
        "cpu_rss_peak_mib": None, "mem_collector": "x",
        "total_params": params, "trainable_params": trainable,
        "lora_r": NA, "lora_alpha": NA, "param_source": "unit-test",
        "t_model": "m", "t_load_mode": "l", "t_max_length": 512, "t_batch_size": 1,
        "t_offload": "none", "t_device": "cuda", "t_collector": "c",
        "amortization": "unit", "notes": NA,
    }
    return d


def _fake_rows(cfg):
    rows = [
        _row("b1", "no_model", "B1 正则", 1.0, None, None, None, None),
        _row("b4", "cpu_model", "B4 unplug-tiny", 200.0, None, None, 1000),
        _row("b3", "gpu_model", "B3 LG-1B", 400.0, 1500.0, 1600.0, 1500000000),
        _row("b5", "gpu_model", "B5 LG-8B", 1000.0, 5500.0, 5000.0, 8000000000),
        _row("ours", "gpu_model", "Ours", 300.0, 1500.0, 1000.0, 1500000000, 1703936),
    ]
    ctrl = [dict(_row("ours", "gpu_model", "Ours", 100.0, 1500.0, 1000.0, 1500000000, 1703936))]
    ctrl[0]["t_batch_size"] = 1
    return rows, ctrl


def _fake_ctx(cfg):
    rows, ctrl = _fake_rows(cfg)
    rows_7b, _anchor = M.build_7b_rows(rows, cfg, ctrl)
    return {
        "cfg": cfg, "rows": rows, "ours_control": ctrl, "rows_7b": rows_7b,
        "tables": M.build_tables(rows, cfg), "repeat_rows": [],
        "ts": "20260101-000000", "shard": "id_val", "n_total": 1584, "n_used": 48,
        "is_sealed": False, "repeat": 3, "warmup": 1,
        "config_path": "configs/eval/efficiency.yaml",
        "access_note": "n/a（非 sealed 源）",
        "gpu_before": "unit", "gpu_after": "unit",
        "limitations": M._limitations(cfg, rows, [], ctrl, False),
    }

# -*- coding: utf-8 -*-
"""tests/eval/test_05_04_eval_adaptive.py — W6.5（5.4 自适应攻击，Stage B）单元与接口测试。

**不触 GPU／不读 `data/sealed_test/**`／不读 frozen 侧 `test.csv`**（`W6.5` §3-C6）。
重型库（torch／transformers／peft）一律不 import —— 被测脚本顶层亦不 import。

覆盖（`windows/W6.5_adaptive.md` §7 验收项 ＋ §3 硬约束）：
1. C2 遍历空间：`id` 18（9 码×2 载体）、`ood` 10（5 训练码×2）；载体/编码集合与配置一致
2. C4 对照公平：空间只由配置决定（两目标构造即同序）；预算必须 == 18，放大即抛错
3. C3 停止条件：首次放行即停、`queries_to_bypass` 1-based；未放行 ⇒ 记上限 + `bypassed=False`
4. C3 语义：`attempts_made ≤ 预算`、轨迹 `attempt ≤ 空间大小`
5. C5 τ 口径：`override=True ⇒ 0.5`；`override=False ⇒ tau_ours`（只读取，不重算）
6. C6 `--source sealed` 缺 `--access-log-note` ⇒ 非零退出；`val` 不需 note
7. C7 记录数口径：val 1584／927 与基线配置一致；写端 LF（无 CR）；`DictReader` 记录数 == 行数
8. 抽样：`Random(42)` 确定性、仅 `label==1`、抽后按原行序
9. 单源复用：9 个编码器均存在；代码载体用 `02_04.frame_py_comment` 框架
10. `--self-check` exit=0 且 `SELFCHECK_RESULT=PASS`；`--dry-run` 打印 `DRYRUN_RESULT=PASS`
11. 静态纪律：无 `.train(`、脚本内不硬编码 `reports/`
"""
from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_04_eval_adaptive.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "adaptive.yaml")
BASE_CONFIG = os.path.join(_ROOT, "configs", "eval", "detection.yaml")
ENCODERS = os.path.join(_ROOT, "scripts", "data", "02_03_encode_variants.py")
CARRIERS = os.path.join(_ROOT, "scripts", "data", "02_04_embed_code_carrier.py")
PYTHON = sys.executable


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("eval_05_04", SCRIPT)


@pytest.fixture(scope="module")
def cfg(mod):
    return mod.load_config(CONFIG)


@pytest.fixture(scope="module")
def base(mod):
    return mod.load_config(BASE_CONFIG)


@pytest.fixture(scope="module")
def enc(mod):
    return _load("enc_02_03_t", ENCODERS)


@pytest.fixture(scope="module")
def car(mod):
    return _load("car_02_04_t", CARRIERS)


# ---------------------------------------------------------------------------
# 1–2 遍历空间与预算（C2／C4）
# ---------------------------------------------------------------------------
def test_space_id_is_18(mod, cfg):
    sp = mod.build_space(cfg, "id")
    assert len(sp) == 18
    assert len(cfg["space"]["encodings_all"]) == 9
    assert len(cfg["space"]["carriers"]) == 2
    assert sp == [("base64", "text"), ("base64", "code"),
                  ("rot13", "text"), ("rot13", "code")] + sp[4:]
    assert sp[-1] == ("homoglyph", "code")


def test_space_ood_uses_5_training_codes(mod, cfg):
    sp = mod.build_space(cfg, "ood")
    assert len(sp) == 10
    codes = sorted({e for e, _c in sp})
    assert codes == sorted(cfg["space"]["encodings_ood"])
    assert "hex" not in codes and "morse" not in codes
    assert "ascii_art" not in codes and "homoglyph" not in codes


def test_space_deterministic_across_targets(mod, cfg):
    """C4：空间只由配置决定 ⇒ 两次构造必须完全相同（两目标同序）。"""
    assert mod.build_space(cfg, "id") == mod.build_space(cfg, "id")
    assert mod.build_space(cfg, "ood") == mod.build_space(cfg, "ood")


def test_space_narrowing_keeps_prereg_order(mod, cfg):
    sp = mod.build_space(cfg, "id", encodings=["hex", "base64"], carriers=["code"])
    assert sp == [("base64", "code"), ("hex", "code")]


def test_space_rejects_unknown_code(mod, cfg):
    with pytest.raises(ValueError):
        mod.build_space(cfg, "id", encodings=["nope"])


def test_budget_must_equal_18(mod, cfg):
    sp = mod.build_space(cfg, "id")
    assert mod.assert_budget(sp, cfg, 18) == 18
    for bad in (17, 19, 36):
        with pytest.raises(AssertionError):
            mod.assert_budget(sp, cfg, bad)


# ---------------------------------------------------------------------------
# 3–4 停止 conditions（C3）
# ---------------------------------------------------------------------------
def _stub(script):
    class Stub(object):
        target = "stub"

        def __init__(self, vals):
            self.vals = list(vals)

        def predict(self, texts):
            take = self.vals[:len(texts)]
            self.vals = self.vals[len(texts):]
            return [{"score": v, "pred": 1 if v >= 0.5 else 0,
                     "unparsed": False, "word": ""} for v in take]

    return Stub(script)


def _payloads(n):
    return [{"sample_id": "P%03d" % i, "plaintext": "payload-%d" % i, "label": "1"}
            for i in range(n)]


def test_first_release_sets_queries_index(mod, cfg, enc, car):
    """A 第 2 次放行 ⇒ queries_to_bypass=2（1-based），不再查第 3 次。"""
    space = mod.build_space(cfg, "ood")[:3]
    payloads = _payloads(1)
    stub = _stub([0.9, 0.1, 0.9])          # 第 1 次不放行，第 2 次放行
    trace, res = mod.run_attack(stub, payloads, space, 18, enc.ENCODERS,
                                car.frame_py_comment, cfg, "stub", "id/seed42", "id", 42)
    assert res[0]["bypassed"] == 1
    assert res[0]["queries_to_bypass"] == 2
    assert res[0]["first_hit_encoding"] == space[1][0]
    assert len(trace) == 2                       # 第 2 次后即停
    assert [t["attempt"] for t in trace] == [1, 2]


def test_not_bypassed_records_cap_and_flag(mod, cfg, enc, car):
    space = mod.build_space(cfg, "ood")[:2]
    stub = _stub([0.9, 0.9])
    _trace, res = mod.run_attack(stub, _payloads(1), space, 18, enc.ENCODERS,
                                 car.frame_py_comment, cfg, "stub", "id/seed42", "id", 42)
    assert res[0]["bypassed"] == 0
    assert res[0]["queries_to_bypass"] == 18      # 上限值
    assert res[0]["attempts_made"] == 2           # 真实尝试数单列，不当"所需次数"
    assert res[0]["first_hit_encoding"] == ""


def test_attempts_never_exceed_budget_or_space(mod, cfg, enc, car):
    space = mod.build_space(cfg, "ood")
    stub = _stub([0.9] * (len(space) * 3))
    trace, res = mod.run_attack(stub, _payloads(3), space, 18, enc.ENCODERS,
                                car.frame_py_comment, cfg, "stub", "id/seed42", "ood", 42)
    assert max(t["attempt"] for t in trace) <= len(space)
    assert all(int(r["attempts_made"]) <= 18 for r in res)
    assert all(int(r["queries_to_bypass"]) == 18 for r in res)   # 全未放行 ⇒ 全上限


def test_unparsed_is_not_release(mod, cfg, enc, car):
    """B3 未解析 ⇒ pred 留空、unparsed=1、**不算放行**（不得兜底为 0）。"""
    class Stub(object):
        target = "stub"

        def predict(self, texts):
            return [{"score": None, "pred": None, "unparsed": True, "word": "garbage"}
                    for _t in texts]

    _trace, res = mod.run_attack(Stub(), _payloads(1), mod.build_space(cfg, "ood")[:1], 18,
                                 enc.ENCODERS, car.frame_py_comment, cfg, "stub", "id", "id", None)
    assert res[0]["bypassed"] == 0
    assert res[0]["queries_to_bypass"] == 18


def test_aggregate_dual_track(mod):
    """删失口径与仅放行口径必须并列且可分（用户验收建议 1）。"""
    rows = [
        {"bypassed": 1, "queries_to_bypass": 3},
        {"bypassed": 0, "queries_to_bypass": 18},
    ]
    agg = mod.aggregate(rows, "stub", "id", 42, 18)
    assert agg["mean_q_cap"] == 10.5 and agg["median_q_cap"] == 10.5   # (3+18)/2
    assert agg["mean_q_rel"] == 3.0 and agg["median_q_rel"] == 3       # 仅放行
    assert agg["cap_share"] == 0.5 and agg["not_bypassed"] == 1


def test_aggregate_rel_empty_when_no_release(mod):
    agg = mod.aggregate([{"bypassed": 0, "queries_to_bypass": 18}], "stub", "id", 42, 18)
    assert agg["mean_q_rel"] == "" and agg["median_q_rel"] == ""
    assert agg["mean_q_cap"] == 18 and agg["cap_share"] == 1.0


def test_render_report_smoke(mod, cfg):
    """报告渲染冒烟：双口径表头、合池提示、小样本局限、C1 声明均在位。"""
    ctx = {
        "ts": "20260101-000000", "config_path": "configs/eval/adaptive.yaml",
        "target": "stub", "source": "val", "regimes": ["id"],
        "space": mod.build_space(cfg, "id"), "ood_space": 10, "cap": 18,
        "space_sizes": {"id": 18, "ood": 10},
        "figdata": [{"target": "stub", "regime": "id", "seed": 42, "n": 2, "bypassed": 1,
                     "not_bypassed": 1, "mean_q_cap": 10.5, "median_q_cap": 10.5,
                     "mean_q_rel": 3.0, "median_q_rel": 3, "cap_share": 0.5}],
        "hit_table": [["stub", "rot13", "text", 1]],
        "access_note": None, "n_payload": 120,
        "paths": {"result": "r.csv", "trace": "t.csv", "figdata": "f.csv", "log": "l.log"},
    }
    md = mod.render_report(cfg, ctx)
    assert "均值(含删失)" in md and "均值(仅放行)" in md
    assert "跨 run / 跨 regime 合池" in md
    assert "`id` = **18** 组合" in md
    assert "小样本" in md
    assert "模拟白盒攻击者" in md and "不是真实黑客红队测试" in md
    assert "| stub | id/seed42 | 2 | 1 | 1 | 10.5 | 10.5 | 3.0 | 3 | 0.5 |" in md


def test_select_payloads_logs_shortfall(mod):
    """可用攻击行不足 ⇒ 记 `PAYLOAD_SHORTFALL`，不静默少抽（用户验收建议 2）。"""
    got = []
    rows = [{"sample_id": str(i), "label": "1"} for i in range(3)]
    picked = mod.select_payloads(rows, 10, 42, logger=got.append)
    assert len(picked) == 3
    assert any("PAYLOAD_SHORTFALL" in m and "available=3" in m for m in got)


# ---------------------------------------------------------------------------
# 5 τ 口径（C5）
# ---------------------------------------------------------------------------
def _fake_meta(tau_star, override):
    return {"tau_star": tau_star, "override": override, "gap": 0.01}


def test_tau_override_uses_fixed_050(mod, base):
    tracks = mod.load_module("ev_05_01_t1", "scripts/eval/05_01_eval_detection.py") \
        .resolve_tau_tracks(_fake_meta(0.8, True), base, None)
    assert tracks["main_value"] == 0.5
    assert tracks["main_track"] == "fixed050"
    assert tracks["parallel_value"] == 0.8


def test_tau_default_uses_star(mod, base):
    tracks = mod.load_module("ev_05_01_t2", "scripts/eval/05_01_eval_detection.py") \
        .resolve_tau_tracks(_fake_meta(0.8, False), base, None)
    assert tracks["main_value"] == 0.8
    assert tracks["main_track"] == "star"
    assert tracks["parallel_value"] == 0.5


def test_tau_fields_required_declared(cfg):
    assert cfg["tau"]["fields_required"] == ["tau_ours", "tau_main_override", "tau_neg_max_gap"]
    assert cfg["tau"]["recompute_forbidden"] is True


# ---------------------------------------------------------------------------
# 6 ACCESS_LOG 守卫（C6）
# ---------------------------------------------------------------------------
def test_sealed_without_note_raises(mod, cfg):
    with pytest.raises(SystemExit):
        mod.require_access_note(cfg, "sealed", "")


def test_sealed_with_short_note_raises(mod, cfg):
    with pytest.raises(SystemExit):
        mod.require_access_note(cfg, "sealed", "too short")


def test_sealed_with_note_ok(mod, cfg):
    note = "ACCESS_LOG 2026-10-23 第 N 行：W6.5 Stage B 读 regime_id_test（详见台账）"
    assert mod.require_access_note(cfg, "sealed", note) == note


def test_val_needs_no_note(mod, cfg):
    assert mod.require_access_note(cfg, "val", "") is None


def test_sealed_cli_exit_nonzero():
    proc = subprocess.run([PYTHON, SCRIPT, "--dry-run", "--source", "sealed", "--target", "b3"],
                          cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    assert proc.returncode != 0
    assert "ACCESS_LOG_GUARD_FAIL" in (proc.stderr + proc.stdout)


# ---------------------------------------------------------------------------
# 7 记录数口径（C7）与 C12 只追加
# ---------------------------------------------------------------------------
def test_val_record_assertions_match_baseline(base):
    assert int(base["record_assertions"]["id_val"]) == 1584
    assert int(base["record_assertions"]["ood_val"]) == 927


def test_write_csv_lf_and_dictreader(mod, cfg, tmp_path):
    rows = [{"target": "stub", "regime": "id", "seed": 42, "n": 2, "bypassed": 1,
             "not_bypassed": 1, "mean_q": 10.0, "median_q": 18, "cap_share": 0.5}]
    path = str(tmp_path / "fig.csv")
    mod.write_csv_rows(path, cfg["outputs"]["figdata_fields"], rows)
    raw = open(path, "rb").read()
    assert raw.count(b"\r") == 0
    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 1
    assert got[0]["target"] == "stub"


def test_write_csv_refuses_overwrite(mod, cfg, tmp_path):
    path = str(tmp_path / "fig.csv")
    mod.write_csv_rows(path, cfg["outputs"]["figdata_fields"], [])
    with pytest.raises(FileExistsError):
        mod.write_csv_rows(path, cfg["outputs"]["figdata_fields"], [])


# ---------------------------------------------------------------------------
# 8 抽样
# ---------------------------------------------------------------------------
def test_sampling_deterministic_and_ordered(mod):
    rows = [{"sample_id": str(i), "label": "1"} for i in range(300)]
    a = [r["sample_id"] for r in mod.select_payloads(rows, 120, 42)]
    b = [r["sample_id"] for r in mod.select_payloads(rows, 120, 42)]
    assert a == b and len(a) == 120
    assert a == sorted(a, key=int)


def test_sampling_only_attack_rows(mod):
    rows = [{"sample_id": str(i), "label": "1" if i % 2 else "0"} for i in range(100)]
    picked = mod.select_payloads(rows, 10, 42)
    assert all(str(int(r["sample_id"]) % 2) == "1" for r in picked)


def test_sampling_raises_without_attack_rows(mod):
    rows = [{"sample_id": "x", "label": "0"}]
    with pytest.raises(AssertionError):
        mod.select_payloads(rows, 5, 42)


# ---------------------------------------------------------------------------
# 9 单源复用（编码器 / 载体）
# ---------------------------------------------------------------------------
def test_nine_encoders_available(cfg, enc):
    assert all(e in enc.ENCODERS for e in cfg["space"]["encodings_all"])


def test_code_carrier_uses_frame_py_comment(mod, enc, car):
    out = mod.render_stimulus(enc.ENCODERS, car.frame_py_comment, "base64", "code", "hello")
    assert out.startswith("    # ")
    assert car.frame_py_comment(enc.ENCODERS["base64"][0]("hello")) == out


def test_text_carrier_is_plain_encoding(mod, enc, car):
    out = mod.render_stimulus(enc.ENCODERS, car.frame_py_comment, "hex", "text", "hello")
    assert out == enc.ENCODERS["hex"][0]("hello")


def test_stimulus_base_is_plaintext_with_no_fallback(cfg):
    assert cfg["stimulus"]["base_column"] == "plaintext"
    assert cfg["stimulus"]["fallback"] == "none"


def test_payload_text_asserts_when_plaintext_missing(mod, cfg):
    """`plaintext` 空 ⇒ 断言失败，**不得**回落到 payload／text。"""
    with pytest.raises(AssertionError):
        mod.payload_text({"sample_id": "X", "plaintext": "", "payload": "abc", "text": "abc"}, cfg)
    assert mod.payload_text({"sample_id": "Y", "plaintext": "intent"}, cfg) == "intent"


# ---------------------------------------------------------------------------
# 10 命令行接口
# ---------------------------------------------------------------------------
def test_self_check_passes():
    proc = subprocess.run([PYTHON, SCRIPT, "--self-check"], cwd=_ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SELFCHECK_RESULT=PASS" in proc.stdout


def test_dry_run_counts():
    proc = subprocess.run([PYTHON, SCRIPT, "--dry-run", "--source", "val"], cwd=_ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRYRUN_RESULT=PASS" in proc.stdout
    # 120 条 ×（id 18 + ood 10）= 3360
    assert "n_attempts=3360" in proc.stdout


def test_missing_report_dir_rejected():
    proc = subprocess.run([PYTHON, SCRIPT, "--target", "b3", "--source", "val", "--limit", "1"],
                          cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    assert proc.returncode != 0
    assert "ARG_REQUIRED" in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 11 静态纪律
# ---------------------------------------------------------------------------
def test_no_train_call_in_script():
    src = open(SCRIPT, encoding="utf-8").read()
    assert ".train(" not in src


def test_no_hardcoded_reports_dir():
    src = open(SCRIPT, encoding="utf-8").read()
    assert '"reports/' not in src and "'reports/" not in src


def test_c1_declaration_present(cfg):
    assert "模拟白盒攻击者" in cfg["c1_declaration"]
    assert "不是真实黑客红队测试" in cfg["c1_declaration"]

# -*- coding: utf-8 -*-
"""tests/eval/test_05_06_eval_ablation.py — W6.7（5.6 消融实验）单元与接口测试。

不触 GPU、不读 `data/sealed_test/**`、不读 frozen 侧 `test.csv`（`W6.7` §3-C5）。
重型库（torch／transformers／peft）一律不在测试内 import —— 被测脚本顶层亦不 import。

覆盖（`windows/W6.7_ablation.md` §7 验收项 ＋ §3 硬约束 C1–C11 ＋ §4 特有禁止项）：
1. §1.1 四组定义 = v5 L612–L620 **原文**；组序与 3 seeds 事前定值
2. C1 统一划分：内容指纹确定性／**顺序敏感**／构造不一致 ⇒ `ok=False`（⇒ 非零退出）
3. C2 全 3 seed：缺组 ⇒ 整列 `<TBD>` ＋ 标注「未训练／未就绪」；**缺 seed 不补齐**
4. C3 `mean ± std`（ddof=1）＋ 逐 seed 原值表 ＋ `std=0` **退化声明**
5. C4 τ **不重寻**：逐组逐 seed 取 `tau_ours`（全精度），`0.5` 并列；禁统一／禁重寻
6. C5 `--source sealed` 缺 `--access-log-note`／`--split` ⇒ **非零退出**
7. C6 记录数断言（`csv.DictReader`，`text` 含换行 ⇒ **记录数 ≠ 行数**）
8. C7 **不训练**：源码无 `04_01`／`04_02`／`04_03` 调用
9. C8 无 train 模式调用；C9 `--report-dir` 未硬编码
10. D 组 §1.3 口径（沿用 C 组 ckpt 并按 `carrier_type` 分层）＋ **防冒充**（A／B 复用 C ⇒ 抛错）
11. 载体分层：封闭域 `text`／`code`；分母**只含本层**；未出现的层**显式列出**且 `n/a`
12. 确定性双跑（逐字节相同）＋ `--self-check` exit=0 ＋ `--dry-run`（真实 C 组 6 run）
"""
from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_06_eval_ablation.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "ablation.yaml")
BASE_CONFIG = os.path.join(_ROOT, "configs", "eval", "detection.yaml")
PY = sys.executable


def _load():
    spec = importlib.util.spec_from_file_location("eval_05_06", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_05_06"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


@pytest.fixture(scope="module")
def cfg():
    return M.load_config(CONFIG)


@pytest.fixture(scope="module")
def base():
    return M.load_config(BASE_CONFIG)


@pytest.fixture(scope="module")
def unify():
    return M.load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")


def _args(**kw):
    """带默认值的 args 桩（`resolve_group_runs`／`build_ctx` 所需字段）。"""
    base_kw = dict(ts="20990101-000000", outputs_dir=None, report_dir=None, limit=None,
                   source=None, access_log_note=None, config=CONFIG, dry_run=False,
                   self_check=False, split=None, groups=None, group_ckpt=None,
                   carrier_strata=False, check_split_identity=False, _base=None)
    base_kw.update(kw)
    a = types.SimpleNamespace(**base_kw)
    if a._base is None:
        a._base = M.load_config(BASE_CONFIG)
    return a


def _fake_result(group="C", regime="id", seed=42, main_track="star", tpr=1.0, n=1584,
                 carrier=None):
    metrics = {"tpr": tpr, "fpr": 0.0, "f1": 0.99, "auroc": 0.999,
               "n_pos": 792, "n_neg": 792, "tp": 792, "fp": 0, "fn": 0, "tn": 792,
               "tau": 0.9999485611915588}
    parallel = dict(metrics, tpr=0.9, f1=0.95, tau=0.5)
    tracks = {main_track: metrics,
              ("fixed050" if main_track == "star" else "star"): parallel}
    if carrier is None:
        carrier = [{"carrier_type": "text", "n": 800, "n_pos": 400, "n_neg": 400,
                    "tp": 400, "fp": 0, "fn": 0, "tn": 400,
                    "tpr": 1.0, "fpr": 0.0, "f1": 1.0},
                   {"carrier_type": "code", "n": 784, "n_pos": 392, "n_neg": 392,
                    "tp": 392, "fp": 0, "fn": 0, "tn": 392,
                    "tpr": 1.0, "fpr": 0.0, "f1": 1.0}]
    return {"group": group, "group_status": "READY", "regime": regime, "seed": seed,
            "run_id": "fake_%s_%d" % (regime, seed), "shard": "%s_val" % regime,
            "source": "val", "n": n, "n_pos": 792, "n_neg": 792,
            "main_track": main_track,
            "parallel_track": "fixed050" if main_track == "star" else "star",
            "tracks": tracks, "by_carrier": carrier,
            "tau": {"tau_star": 0.9999485611915588, "override": False,
                    "gap": 0.9626216776669025, "main_value": 0.9999485611915588,
                    "parallel_value": 0.5, "main_track": main_track,
                    "parallel_track": "fixed050" if main_track == "star" else "star",
                    "tiers": {"fixed050": 0.5, "median_val": None,
                              "star": 0.9999485611915588}},
            "latency": {"mean_ms": 300.0, "median_ms": 280.0, "p95_ms": 500.0},
            "mem_load_peak_mib": 1500.0, "mem_infer_peak_mib": 4200.0,
            "eval_batch_max_size": 16, "split_fingerprint": "fp_%s_%d" % (regime, seed)}


def _mk_run(tmp_path, regime, seed, tau=0.9999485611915588):
    """最小可用 run 目录（`best.json` ＋ ckpt `meta.json`）。"""
    rd = os.path.join(str(tmp_path), "run_%s_%d" % (regime, seed))
    return M._mk_fake_run(rd, regime, seed, tau)


# ---------------------------------------------------------------------------
# ① §1.1 四组定义与组序（v5 L612–L620 原文）
# ---------------------------------------------------------------------------
def test_four_groups_definition_verbatim(cfg):
    assert cfg["groups"]["order"] == ["A", "B", "C", "D"]
    g = cfg["groups"]
    assert g["A"]["verbatim"] == "A：仅原始数据，无编码增强。"
    assert g["B"]["verbatim"] == "B：编码增强 + 普通随机增强，无课程学习。"
    assert g["C"]["verbatim"] == "C：编码增强 + 课程学习，完整方法。"
    assert g["D"]["verbatim"] == "D（可选）：载体消融，纯文本 vs 代码载体。"


def test_ab_ckpt_status_declared(cfg):
    """A／B 组 ckpt 状态**须与实测一致**，且实现轮的 `NOT_EXIST` 快照**不得被抹去**。

    完整轮（`D-91` 裁定一 验收 `W5.6` 补训）⇒ A／B 实测 `EXIST`；`round_history` ＋
    `full_round_amendments` 须保留实现轮的 `IMPLEMENTATION_ROUND`／`NOT_EXIST` 轨迹。
    """
    assert cfg["groups"]["A"]["ckpt_status"] == "EXIST"
    assert cfg["groups"]["B"]["ckpt_status"] == "EXIST"
    assert cfg["groups"]["C"]["ckpt_status"] == "EXIST"
    assert cfg["groups"]["seeds"] == [42, 123, 456]
    assert cfg["round"] == "FULL_ROUND"
    assert any("IMPLEMENTATION_ROUND" in s for s in cfg["round_history"])
    assert cfg["full_round_amendments"]["amended_at"]
    assert "待 `W0` 追认" in cfg["full_round_amendments"]["ratification"]
    for g in ("A", "B", "C"):
        assert len(cfg["groups"][g]["known_runs"]) >= 3


def test_full_round_amendments_declare_no_criterion_change(cfg):
    """补记件须**自证**「判据数值与口径一律未改」，且补记时点须**早于**封存侧数字。"""
    fa = cfg["full_round_amendments"]
    assert "判据数值与口径" in fa["items"][0] or "非判据数值" in fa["items"][0]
    assert "早于" in fa["amended_at"] and "sealed" in fa["amended_at"]
    assert any("judgment_shards" in s for s in fa["items"])


def test_ckpt_status_cell_uses_measured_state(cfg):
    """`ckpt 现状` 列须以**实测**状态为准 —— 否则完整轮会把 A／B 误报成「不存在」。"""
    got = M._ckpt_status_cell({"status": "READY", "ckpt_status": "NOT_EXIST"})
    assert "READY" in got and "NOT_EXIST" in got          # 不一致 ⇒ 并列显示，不隐去
    assert M._ckpt_status_cell({"status": "READY", "ckpt_status": "READY"}) == "`READY`"
    assert "<TBD>" in M._ckpt_status_cell({"status": "MISSING", "ckpt_status": ""}) \
        or "MISSING" in M._ckpt_status_cell({"status": "MISSING", "ckpt_status": ""})


def test_group_order_not_editable_by_result(cfg):
    """§4-4：禁止按结果删减组或事后加组 ⇒ 组序为配置事前定值。"""
    assert len(cfg["groups"]["order"]) == 4
    assert set(cfg["groups"]["order"]) == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# ② C6 记录数断言
# ---------------------------------------------------------------------------
def test_record_assertions_consistent_with_base(cfg, base):
    for k, v in cfg["record_count_policy"]["expected"].items():
        assert int(base["record_assertions"][k]) == int(v)
    assert int(base["record_assertions"]["regime_id_test"]) == 1583
    assert int(base["record_assertions"]["regime_ood_test"]) == 740


# ---------------------------------------------------------------------------
# ③ C1 统一划分：内容指纹
# ---------------------------------------------------------------------------
def test_split_fingerprint_deterministic_and_order_sensitive():
    a = [{"text": "x"}, {"text": "y"}, {"text": "z"}]
    b = [{"text": "x"}, {"text": "y"}, {"text": "z"}]
    c = [{"text": "z"}, {"text": "y"}, {"text": "x"}]
    assert M.split_fingerprint(a) == M.split_fingerprint(b)
    assert M.split_fingerprint(a) != M.split_fingerprint(c)


def test_split_fingerprint_empty_text_participates():
    """空 `text` 以 `sha256('')` 参与，不跳过（避免静默错位）。"""
    assert M.split_fingerprint([{"text": ""}]) != M.split_fingerprint([])


def test_split_identity_passes_when_all_identical():
    res = [_fake_result(group=g, seed=42) for g in ("C", "D")]
    for r in res:
        r["split_fingerprint"] = M.split_fingerprint([{"text": "x"}, {"text": "y"}])
    ok, rows, bad = M.check_split_identity(res)
    assert ok is True and not bad
    assert all(r["matches_reference"] == "1" for r in rows)


def test_split_identity_detects_mismatch():
    """构造不一致 ⇒ `ok=False` 且给出明细（脚本据此**非零退出 ＋ 停工呈 W0**）。"""
    res = [_fake_result(group="C", seed=42), _fake_result(group="D", seed=42)]
    res[0]["split_fingerprint"] = M.split_fingerprint([{"text": "x"}, {"text": "y"}])
    res[1]["split_fingerprint"] = M.split_fingerprint([{"text": "y"}, {"text": "x"}])
    ok, rows, bad = M.check_split_identity(res)
    assert ok is False and len(bad) == 1
    assert rows[1]["matches_reference"] == "0"


def test_split_identity_scoped_per_regime_shard():
    """回归（本窗口**实跑**触发的假 FAIL）：`id_val`／`ood_val` 是**两个不同分片**。

    指纹必然不同；若跨 `(regime, shard)` 一并比对会把「体制不同」误判成
    「每组用了不同划分」⇒ **假停工呈 W0**。正确语义 = 同一 `(regime, shard)` 内四组逐位相同。
    """
    res = [_fake_result(group="C", regime="id", seed=42),
           _fake_result(group="C", regime="ood", seed=42)]
    res[0]["split_fingerprint"] = M.split_fingerprint([{"text": "a"}])
    res[1]["split_fingerprint"] = M.split_fingerprint([{"text": "b"}])
    ok, rows, bad = M.check_split_identity(res)
    assert ok is True and not bad
    assert all(r["matches_reference"] == "1" for r in rows)


def test_split_identity_mismatch_only_within_same_shard():
    """同 `(regime, shard)` 内两组指纹不同 ⇒ 必须 FAIL。"""
    res = [_fake_result(group="C", regime="id", seed=42),
           _fake_result(group="D", regime="id", seed=42)]
    res[0]["split_fingerprint"] = M.split_fingerprint([{"text": "a"}])
    res[1]["split_fingerprint"] = M.split_fingerprint([{"text": "b"}])
    ok, rows, bad = M.check_split_identity(res)
    assert ok is False and len(bad) == 1


def test_split_identity_empty_is_pass():
    ok, rows, bad = M.check_split_identity([])
    assert ok is True and rows == [] and bad == []


def test_split_identity_csv_fields(tmp_path, cfg):
    res = [_fake_result(group="C", seed=42)]
    ok, rows, _ = M.check_split_identity(res)
    p = os.path.join(str(tmp_path), "ident.csv")
    M.write_split_identity_csv(p, {"rows": rows, "ok": ok, "bad": []})
    with open(p, "r", encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == M.SPLIT_IDENTITY_CSV_FIELDS


# ---------------------------------------------------------------------------
# ④ C2 缺组 ⇒ `<TBD>` ＋ 标注；D 组沿用 C；**防冒充**
# ---------------------------------------------------------------------------
def test_missing_group_yields_tbd_with_note(cfg, tmp_path):
    c = _mk_run(tmp_path, "id", 42)
    a = _args(group_ckpt=["C=%s" % c])
    runs, gstat = M.resolve_group_runs(cfg, a, print)
    assert gstat["A"]["status"] == "MISSING"
    assert gstat["B"]["status"] == "MISSING"
    assert cfg["groups"]["missing_group_policy"]["note"] in gstat["A"]["note"]
    assert gstat["C"]["status"] == "READY"
    # D 组按 §1.3 沿用 C 组 ckpt（**不需新训练**）
    assert gstat["D"]["status"] == "DERIVED_FROM_C"
    assert len(runs) == 2  # C:1 ＋ D:1（A／B 未混入）


def test_alias_guard_rejects_ab_reusing_c(cfg, tmp_path):
    """§4-1：**禁止**用 C 组结果冒充 A／B ⇒ A 复用 C 的 run_dir 必须抛错（非零退出）。"""
    c = _mk_run(tmp_path, "id", 42)
    with pytest.raises(AssertionError) as ei:
        M.resolve_group_runs(cfg, _args(group_ckpt=["A=%s" % c, "C=%s" % c]), print)
    assert "ALIAS_GUARD_VIOLATION" in str(ei.value)
    with pytest.raises(AssertionError):
        M.resolve_group_runs(cfg, _args(group_ckpt=["B=%s" % c, "C=%s" % c]), print)


def test_duplicate_group_seed_rejected(cfg, tmp_path):
    c = _mk_run(tmp_path, "id", 42)
    with pytest.raises(ValueError):
        M.resolve_group_runs(cfg, _args(group_ckpt=["C=%s" % c, "C=%s" % c]), print)


def test_unknown_group_rejected(cfg):
    with pytest.raises(ValueError):
        M.parse_group_ckpt(["E=/tmp/x"])
    with pytest.raises(ValueError):
        M.parse_groups_arg("A,E", cfg)


def test_render_four_group_columns_with_tbd(cfg, tmp_path):
    """§7-3：四组列**齐备**；未就绪者 `<TBD>` 且**与 C 组不同行**（不得混填）。"""
    c = _mk_run(tmp_path, "id", 42)
    a = _args(group_ckpt=["C=%s" % c])
    _runs, gstat = M.resolve_group_runs(cfg, a, print)
    rendered = M.render_ablation_report(
        M.build_ctx(cfg, a, [], gstat, ["A", "B", "C", "D"], "val", None, {}, "lead"))
    for g in ("A", "B", "C", "D"):
        assert "| %s |" % g in rendered
    assert "| A | 缺 seed" in rendered and "| B | 缺 seed" in rendered
    assert M.TBD in rendered
    assert "未训练／未就绪" in rendered


def test_render_with_real_c_results_keeps_ab_tbd(cfg, tmp_path):
    """即便 C 组有真实结果，A／B 两格仍必须是 `<TBD>`（不得被 C 的数填掉）。"""
    c = _mk_run(tmp_path, "id", 42)
    a = _args(group_ckpt=["C=%s" % c])
    _runs, gstat = M.resolve_group_runs(cfg, a, print)
    results = [_fake_result(group="C", seed=s) for s in (42, 123, 456)]
    rendered = M.render_ablation_report(
        M.build_ctx(cfg, a, results, gstat, ["A", "B", "C", "D"], "val", None, {}, "lead"))
    lines = [ln for ln in rendered.split("\n") if ln.startswith("| A | 缺 seed")
             or ln.startswith("| B | 缺 seed")]
    # 两个 τ 档**各自成表**（`D-88` 裁定三-3「全档一律报」）⇒ A／B 各出现 2 次
    assert len(lines) == 4
    for ln in lines:
        assert ln.count(M.TBD) == 4  # TPR／FPR／F1／AUROC 四格全为 <TBD>


# ---------------------------------------------------------------------------
# ⑤ C3 `mean ± std` ／ `std=0` 退化 ／ 逐 seed 原值表
# ---------------------------------------------------------------------------
def test_mean_std_ddof1():
    m, s = M.mean_std([1.0, 2.0, 3.0], ddof=1)
    assert abs(m - 2.0) < 1e-12 and abs(s - 1.0) < 1e-12


def test_std_zero_flagged_in_cell():
    cell = M.mean_std_cell(0.5, 0.0)
    assert "std=0" in cell
    assert "std=0" not in M.mean_std_cell(0.5, 0.1)


def test_incomplete_seeds_yield_tbd(cfg):
    one = [_fake_result(group="A", seed=42)]
    agg = M.aggregate(cfg, one, ["A"], "id")["A"]
    assert agg["complete"] is False and agg["metrics"] == {}
    assert agg["missing"] == [123, 456]


def test_complete_seeds_aggregate(cfg):
    res = [_fake_result(group="C", seed=s) for s in (42, 123, 456)]
    agg = M.aggregate(cfg, res, ["C"], "id")["C"]
    assert agg["complete"] is True
    assert abs(agg["metrics"]["tpr"][0] - 1.0) < 1e-12
    assert abs(agg["metrics"]["tpr"][1]) < 1e-12  # std=0 ⇒ 须声明退化


def test_per_seed_raw_table_present(cfg):
    res = [_fake_result(group="C", seed=s) for s in (42, 123, 456)]
    rows = M.per_seed_rows(res, ["C"], "id")
    assert len(rows) == 3
    assert [r[1] for r in rows] == [42, 123, 456]


def test_report_has_std_zero_declaration_slot(cfg, tmp_path):
    c = _mk_run(tmp_path, "id", 42)
    a = _args(group_ckpt=["C=%s" % c])
    _runs, gstat = M.resolve_group_runs(cfg, a, print)
    res = [_fake_result(group="C", seed=s) for s in (42, 123, 456)]
    rendered = M.render_ablation_report(
        M.build_ctx(cfg, a, res, gstat, ["A", "B", "C", "D"], "val", None, {}, "lead"))
    assert "std = 0" in rendered
    assert "退化" in rendered


# ---------------------------------------------------------------------------
# ⑥ C4 τ 不重寻
# ---------------------------------------------------------------------------
def test_tau_per_run_full_precision(base):
    meta = {"tau_star": 0.9999485611915588, "override": False,
            "gap": 0.9626216776669025, "run_dir": "", "ckpt_dir": "", "best": {},
            "meta": {}, "regime": "id", "seed": 42, "best_step": 1}
    tau = M.W().resolve_tau_tracks(meta, base, None)
    assert tau["main_track"] == "star"
    assert tau["main_value"] == 0.9999485611915588
    assert M.fmt_tau(tau["tau_star"]) == repr(0.9999485611915588)
    assert abs(tau["parallel_value"] - 0.5) < 1e-12


def test_tau_reseek_and_unify_forbidden_in_config(cfg):
    assert cfg["tau"]["reseek_forbidden"] is True
    assert cfg["tau"]["unify_forbidden"] is True
    assert cfg["tau"]["per_run"] is True
    assert cfg["tau"]["full_precision"] is True
    assert cfg["tau"]["parallel"] == 0.5
    assert cfg["tau"]["source_single"].startswith("05_01.resolve_tau_tracks")


def test_tau_override_branch(base):
    meta = {"tau_star": 0.9999485611915588, "override": True,
            "gap": 1e-5, "run_dir": "", "ckpt_dir": "", "best": {}, "meta": {},
            "regime": "id", "seed": 42, "best_step": 1}
    tau = M.W().resolve_tau_tracks(meta, base, None)
    assert tau["main_value"] == 0.5 and tau["parallel_value"] == 0.9999485611915588


# ---------------------------------------------------------------------------
# ⑦ 载体分层（D 组）
# ---------------------------------------------------------------------------
def test_carrier_values_closed_domain(cfg):
    assert cfg["carrier"]["values"] == ["text", "code"]
    assert cfg["carrier"]["group_by_column"] == "carrier_type"


def test_per_carrier_denominator_layer_only(cfg, unify):
    rows = [{"carrier_type": "code", "text": "a", "label": "1"},
            {"carrier_type": "code", "text": "b", "label": "0"},
            {"carrier_type": "code", "text": "c", "label": "0"},
            {"carrier_type": "text", "text": "d", "label": "1"}]
    out = M.per_carrier_metrics(unify, rows, [0.9, 0.1, 0.9, 0.9], 0.5, cfg)
    code = [c for c in out if c["carrier_type"] == "code"][0]
    text = [c for c in out if c["carrier_type"] == "text"][0]
    assert code["n"] == 3 and text["n"] == 1
    assert abs(code["fpr"] - (1.0 / 2.0)) < 1e-12  # 本层良性 2 条、误报 1
    assert abs(text["tpr"] - 1.0) < 1e-12


def test_per_carrier_absent_layer_explicit(cfg, unify):
    rows = [{"carrier_type": "code", "text": "a", "label": "1"}]
    out = M.per_carrier_metrics(unify, rows, [0.9], 0.5, cfg)
    text = [c for c in out if c["carrier_type"] == "text"][0]
    assert text["n"] == 0 and text["tpr"] is None and M.fmt4(text["tpr"]) == "n/a"


def test_carrier_csv_fields(tmp_path):
    p = os.path.join(str(tmp_path), "c.csv")
    M.write_carrier_csv(p, [_fake_result(group="D", seed=42)])
    with open(p, "r", encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == M.CARRIER_CSV_FIELDS


def test_ablation_csv_fields(tmp_path):
    p = os.path.join(str(tmp_path), "a.csv")
    M.write_ablation_csv(p, [_fake_result(group="C", seed=42)], {})
    with open(p, "r", encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == M.ABLATION_CSV_FIELDS
    with open(p, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2  # 主口径档 ＋ 并列档


# ---------------------------------------------------------------------------
# ⑧ C5 `--source sealed` 缺参 ⇒ 非零退出（接口级，走子进程）
# ---------------------------------------------------------------------------
def test_sealed_requires_note_and_split(tmp_path):
    d = str(tmp_path)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rc1 = subprocess.call([PY, SCRIPT, "--source", "sealed", "--config", CONFIG,
                           "--report-dir", d, "--outputs-dir", d], env=env)
    assert rc1 != 0
    rc2 = subprocess.call([PY, SCRIPT, "--source", "sealed", "--split", "in_regime_test",
                           "--config", CONFIG, "--report-dir", d, "--outputs-dir", d], env=env)
    assert rc2 != 0


def test_missing_dirs_nonzero(tmp_path):
    d = str(tmp_path)
    rc = subprocess.call([PY, SCRIPT, "--source", "val", "--config", CONFIG], env=dict(os.environ))
    assert rc != 0


# ---------------------------------------------------------------------------
# ⑨ C7／C8／C9 静态纪律
# ---------------------------------------------------------------------------
def test_source_level_discipline():
    with open(SCRIPT, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert ("." + "train" + "(") not in src                      # C8
    assert ("read" + "lines(") not in src                        # C6
    assert ("split" + "lines(") not in src                       # C6
    for s, t in (("04_01", "train"), ("04_02", "train"), ("04_03", "run")):
        assert ("%s_%s" % (s, t)) not in src                     # C7 不训练
    assert ("os.path.join(_ROOT, " + '"repo' + 'rts"') not in src  # C9
    assert "CUBLAS_WORKSPACE_CONFIG" in src                      # C8 确定性


def test_c7_no_training_declaration_in_config(cfg):
    assert any("调用次数 = 0" in s for s in cfg["declarations"]["no_training"])
    assert any("04_01" in s for s in cfg["forbidden_rewrite"])


def test_resolve_log_dir_env_first(monkeypatch, tmp_path):
    """`#64`：环境变量优先 ⇒ 测试派生的日志不得污染 `logs/eval/`。"""
    monkeypatch.setenv("CODEOBF_LOG_DIR", str(tmp_path))
    assert M.resolve_log_dir(None) == str(tmp_path)
    monkeypatch.delenv("CODEOBF_LOG_DIR")
    assert M.resolve_log_dir(_args(dry_run=True)).endswith(os.path.join("logs", "eval", "tmp"))
    assert M.resolve_log_dir(_args()).endswith(os.path.join("logs", "eval"))


# ---------------------------------------------------------------------------
# ⑩ 确定性双跑
# ---------------------------------------------------------------------------
def test_determinism_two_runs(tmp_path, cfg, unify):
    rows = [{"carrier_type": "code", "text": "a", "label": "1"},
            {"carrier_type": "text", "text": "b", "label": "0"},
            {"carrier_type": "code", "text": "c", "label": "1"}]
    p1 = os.path.join(str(tmp_path), "1.csv")
    p2 = os.path.join(str(tmp_path), "2.csv")

    def pack():
        return [_fake_result(group="D", seed=42,
                             carrier=M.per_carrier_metrics(unify, rows, [0.9, 0.1, 0.9],
                                                           0.5, cfg))]
    M.write_carrier_csv(p1, pack())
    M.write_carrier_csv(p2, pack())
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        assert f1.read() == f2.read()


# ---------------------------------------------------------------------------
# ⑪ `#69` 区分度判据（`D-88` 裁定一，事前定值）
# ---------------------------------------------------------------------------
def _disc(vals_by_group, metric="f1", shard="id_test"):
    return M._disc_fake_results(vals_by_group, metric=metric, shard=shard)


def _disc_all(vals_by_group):
    """裁定一②（`g3` 仅 FPR）的 companion 桩 ⇒ 与 `id_test` 结果合起来凑齐**全部格位**。"""
    return M._disc_fake_companion(vals_by_group)


def _disc_all_on(shard, vals_by_group, metrics=("fpr",)):
    return M._disc_fake_companion(vals_by_group, shard=shard, metrics=metrics)


def test_discrimination_eps_is_001(cfg):
    """`D-88` 裁定一 ③：`ε = 0.01`；判据一旦落盘即为定值 ⇒ **不得**按结果放宽。"""
    assert abs(float(cfg["discrimination"]["eps"]) - 0.01) < 1e-12
    assert cfg["discrimination"]["groups_in_scope"] == ["A", "B", "C"]
    assert cfg["discrimination"]["tracks"] == ["star", "fixed050"]


def test_discrimination_judgment_shards_fixed(cfg):
    """格位集 = `rule_main`／`rule_scope` 的结构化表述 ⇒ 与「跑到哪个分片」无关。

    若按当前分片取格位：跑 `g3` 那轮会漏掉 `id_test` 主判据，跑 `id_test` 那轮（companion
    未就绪时）会把「未看 `g3`」误判成「无区分度」⇒ **假触发** `D-88` 裁定二（禁「换分片」）。
    """
    js = cfg["discrimination"]["judgment_shards"]
    assert [(s["shard"], tuple(s["metrics"]), s["role"]) for s in js] == [
        ("id_test", ("f1", "fpr"), "main"),
        ("g3_hard_negative_test", ("fpr",), "scope")]
    assert cfg["discrimination"]["pending_blocks_no_diff"] is True


def test_discrimination_range_and_verdict_no_diff(cfg):
    """三组 3-seed `mean` 极差 = 0（格位齐备）⇒ 全格「无差异」⇒ 判「无区分度」（触发呈 W0）。"""
    vals = {"A": [1.0] * 3, "B": [1.0] * 3, "C": [1.0] * 3}
    rows, verdict = M.discrimination_rows(
        cfg, _disc(vals), ["A", "B", "C", "D"], "id", "id_test", _disc_all(vals))
    assert all(r["range"] == 0.0 for r in rows if r["range"] is not None)
    assert all(r["kind"] == "evaluated" for r in rows)
    assert "无区分度" in verdict and "呈 `W0`" in verdict
    assert "不得" in verdict and "自行扩训" in verdict


def test_discrimination_verdict_has_diff(cfg):
    """极差 ≥ ε ⇒ 判「有区分度」。"""
    vals = {"A": [0.90] * 3, "B": [0.95] * 3, "C": [1.0] * 3}
    rows, verdict = M.discrimination_rows(
        cfg, _disc(vals), ["A", "B", "C", "D"], "id", "id_test", _disc_all(vals))
    assert any(r["range"] is not None and r["range"] >= 0.01 for r in rows)
    assert "有区分度" in verdict


def test_discrimination_pending_blocks_no_diff(cfg):
    """⚠ 护栏：格位未判读完（`g3` 无数据）⇒ **不得**判「无区分度」，更**不得**触发裁定二。"""
    rows, verdict = M.discrimination_rows(
        cfg, _disc({"A": [1.0] * 3, "B": [1.0] * 3, "C": [1.0] * 3}),
        ["A", "B", "C", "D"], "id", "id_test")
    g3 = [r for r in rows if r["shard"] == "g3_hard_negative_test"]
    assert g3 and all(r["kind"] == "unavailable" for r in g3)
    assert all(("待 companion 轮" in r["verdict"]) for r in g3)
    assert "待补齐" in verdict
    assert "呈 `W0`" not in verdict


def test_discrimination_epsilon_boundary(cfg):
    """`ε` 边界：极差 `0.0099` < ε ⇒ 无差异；`0.0100` ≥ ε ⇒ 有差异（**不得**用 `>` 放宽）。"""
    lo = {"A": [0.9900] * 3, "B": [0.9949] * 3, "C": [0.9999] * 3}
    r_lo, _ = M.discrimination_rows(
        cfg, _disc(lo), ["A", "B", "C", "D"], "id", "id_test", _disc_all(lo))
    rng_lo = [r["range"] for r in r_lo if r["range"] is not None][0]
    assert abs(rng_lo - 0.0099) < 1e-9 and rng_lo < 0.01
    hi = {"A": [0.9900] * 3, "B": [0.9950] * 3, "C": [1.0000] * 3}
    r_hi, v_hi = M.discrimination_rows(
        cfg, _disc(hi), ["A", "B", "C", "D"], "id", "id_test", _disc_all(hi))
    assert [r["range"] for r in r_hi if r["range"] is not None][0] >= 0.01
    assert "有区分度" in v_hi


def test_discrimination_incomplete_not_filled(cfg):
    """某组 seed 不齐 ⇒ 极差**无定义**（`None`），**不得**用其他 seed 补齐，且结论待补齐。"""
    rows, verdict = M.discrimination_rows(
        cfg, _disc({"A": [1.0], "B": [1.0] * 3, "C": [1.0] * 3}),
        ["A", "B", "C", "D"], "id", "id_test")
    assert all(r["range"] is None for r in rows)
    assert "无法判定" in verdict and "待补齐" in verdict


def test_discrimination_companion_g3_only_fpr(cfg, tmp_path):
    """`g3` 为全负分片 ⇒ **只取 FPR**（F1 无定义）；companion 经配置显式路径取数。"""
    p = os.path.join(str(tmp_path), "g3.csv")
    M.write_ablation_csv(p, [_fake_result(group=g, seed=s, n=64) for g in ("A", "B", "C")
                             for s in (42, 123, 456)])
    # 改写 shard 为 `g3_hard_negative_test`（本桩只为验证 companion 装载与 FPR-only）
    import csv as _csv
    with open(p, "r", encoding="utf-8", newline="") as fh:
        recs = list(_csv.DictReader(fh))
    for r in recs:
        r["shard"] = "g3_hard_negative_test"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=M.ABLATION_CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    import copy
    small = copy.deepcopy(cfg)
    small["discrimination"]["companion"]["artifacts"] = [p]
    comp = M.load_companion_rows(small, print)
    assert "g3_hard_negative_test" in comp
    # companion 取数**必须**带 `regime` 层（`g3` 分片在 id／ood 下指同一份 CSV）
    assert "id" in comp["g3_hard_negative_test"]
    rows, verdict = M.discrimination_rows(small, _disc({"A": [1.0] * 3, "B": [1.0] * 3,
                                                        "C": [1.0] * 3}),
                                          ["A", "B", "C", "D"], "id", "id_test", comp)
    g3_rows = [r for r in rows if r["shard"] == "g3_hard_negative_test"]
    assert g3_rows and all(r["metric"] == "fpr" for r in g3_rows)
    # companion 就绪后格位齐备 ⇒ 可落「无区分度」（**不再**停在「待补齐」）
    assert all(r["kind"] == "evaluated" for r in rows)
    assert "无区分度" in verdict


def test_companion_filters_by_regime_no_cross_regime_mixing(cfg):
    """⚠ 回归：`g3` 分片在 `id`／`ood` 下**指同一份 CSV** ⇒ companion 取数必须按 `regime` 分层。

    否则 `C` 组会吃进 `id`×3 ＋ `ood`×3 共 6 个值 ⇒「3-seed `mean`」被算成「6-run `mean`」
    （**跨体制混算** ⇒ `D-88` 裁定一 的口径被破坏），且**不会**被 `len(vals) < 3` 检出。
    """
    comp = M._disc_fake_companion({"A": [0.0] * 3, "B": [0.0] * 3, "C": [0.0] * 3}, regime="id")
    # 再塞入 `ood` 体制的同分片值（**应被忽略**：判据取 `regime=id`）
    comp["g3_hard_negative_test"]["ood"] = {
        "star": {"fpr": {"C": {42: 1.0, 123: 1.0, 456: 1.0}}},
        "fixed050": {"fpr": {"C": {42: 1.0, 123: 1.0, 456: 1.0}}}}
    rows, verdict = M.discrimination_rows(
        cfg, _disc({"A": [0.0] * 3, "B": [0.0] * 3, "C": [0.0] * 3}),
        ["A", "B", "C", "D"], "id", "id_test", comp)
    g3 = [r for r in rows if r["shard"] == "g3_hard_negative_test"]
    assert g3 and all(r["means"]["C"] == 0.0 for r in g3), "C 的 g3 值只应取 id 体制的 3 个 run"
    assert "无区分度" in verdict


def test_companion_missing_artifact_not_silently_zero(cfg, tmp_path):
    """companion 路径不存在 ⇒ **不得**静默当 0（须落日志并渲染 `<TBD>`）。"""
    import copy
    small = copy.deepcopy(cfg)
    small["discrimination"]["companion"]["artifacts"] = [
        os.path.join(str(tmp_path), "nope.csv")]
    got = M.load_companion_rows(small, print)
    assert got == {}


def test_discrimination_g3_run_sees_id_test_cells(cfg):
    """跑到 `g3` 分片那一轮，`id_test` 主判据格位**必须在§0 出现**（标「待 companion 轮」）。"""
    rows, verdict = M.discrimination_rows(
        cfg, _disc({"A": [1.0] * 3, "B": [1.0] * 3, "C": [1.0] * 3},
                  metric="fpr", shard="g3_hard_negative_test"),
        ["A", "B", "C", "D"], "id", "g3_hard_negative_test")
    idc = [r for r in rows if r["shard"] == "id_test"]
    assert idc and all(r["kind"] == "unavailable" for r in idc)
    assert "① 主判据" in verdict and "待补齐" in verdict


def test_discrimination_not_evaluated_declared(cfg):
    """`clean_original_test` 归 `04_04`／`W6` ⇒ 须**如实登记为未纳入**，不得静默省略。"""
    ne = cfg["discrimination"]["not_evaluated"]
    assert any("clean_original_test" in s for s in ne)
    assert any("未纳入" in s for s in ne)


def test_report_has_discrimination_section(cfg, tmp_path):
    """`D-88` 裁定六 转交项 `#69`：报告须在**显著位置**（§0）给出区分度判定。"""
    c = _mk_run(tmp_path, "id", 42)
    a = _args(group_ckpt=["C=%s" % c])
    _runs, gstat = M.resolve_group_runs(cfg, a, print)
    res = [_fake_result(group="C", seed=s) for s in (42, 123, 456)]
    rendered = M.render_ablation_report(
        M.build_ctx(cfg, a, res, gstat, ["A", "B", "C", "D"], "sealed",
                    "in_regime_test", {}, "lead"))
    assert "## 0. 区分度判定" in rendered
    assert "ε" in rendered and "综合判定" in rendered


def test_reading_disclosure_in_config(cfg):
    """`D-84` 裁定一 **条件 2**：读法一须显著披露（字面两项 vs 合并一项）。"""
    rd = cfg["declarations"]["reading_disclosure"]
    assert any("合并为一项" in s for s in rd)
    assert (cfg["group_recipe"]["A"]["cli"] == "--augment off --curriculum off")
    assert (cfg["group_recipe"]["B"]["cli"] == "--augment on --curriculum off")
    assert "不扩" in cfg["group_recipe"]["regime_scope"] or "仅 `id`" in cfg["group_recipe"]["regime_scope"]


# ---------------------------------------------------------------------------
# ⑫ 接口级：`--self-check` exit=0 ／ `--dry-run`（真实 C 组 6 run）
# ---------------------------------------------------------------------------
def test_selfcheck_exit_zero():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([PY, SCRIPT, "--self-check"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    assert p.returncode == 0, p.stdout[-2000:]
    assert "SELFCHECK_RESULT=PASS" in p.stdout
    n = int(p.stdout.strip().split("SELFCHECK_RESULT=PASS n=")[1].split()[0])
    assert n >= 30


def _real_c_runs():
    rd = os.path.join(_ROOT, "checkpoints", "ours")
    names = ["04_01_llamaguard1b_id_seed42_20261005-150806",
             "04_01_llamaguard1b_id_seed123_20261006-010210",
             "04_01_llamaguard1b_id_seed456_20261006-120248",
             "04_01_llamaguard1b_ood_seed42_20261007-144100",
             "04_01_llamaguard1b_ood_seed123_20261007-180342",
             "04_01_llamaguard1b_ood_seed456_20261007-202634"]
    out = []
    for n in names:
        p = os.path.join(rd, n)
        if not os.path.isfile(os.path.join(p, "best.json")):
            return []
        out.append(p)
    return out


def test_dryrun_with_real_c_runs(tmp_path):
    """接口级演练：`--dry-run` 用**真实 C 组 6 个 run**（不加载模型、不触 GPU）。"""
    runs = _real_c_runs()
    if not runs:
        pytest.skip("阶段4 的 6 个 run 不在位（非本窗口产物）")
    d = str(tmp_path)
    cmd = [PY, SCRIPT, "--source", "val", "--dry-run",
           "--report-dir", d, "--outputs-dir", d, "--config", CONFIG]
    for r in runs:
        cmd += ["--group-ckpt", "C=%s" % r]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert p.returncode == 0, p.stdout[-3000:]
    assert "DRYRUN_RESULT=PASS" in p.stdout
    assert "group=C n=6" in p.stdout or "GROUP_RESOLVED group=C n=6" in p.stdout
    assert "records=1584" in p.stdout and "records=927" in p.stdout
    # A／B 缺 ckpt 仍是 `<TBD>`（不得被 C 顶替）
    assert "GROUP_MISSING group=A" in p.stdout and "GROUP_MISSING group=B" in p.stdout

# -*- coding: utf-8 -*-
"""tests/eval/test_05_01_eval_detection.py — W6.1（5.1 检测性能评估）单元与接口测试。

不触 GPU、不读 `data/sealed_test/**`、不读 frozen 侧 `test.csv`（`W6.1` §3-C1／§4-1）。
重型库（torch／transformers／peft）一律不在测试内 import —— 被测脚本顶层亦不 import。

覆盖（`windows/W6.1_detection.md` §10-1 验收项）：
1. CSV 记录数断言（含 §8-5 的「text 含 2 个换行 ⇒ 记录数 1、行数 ≥3」回归）
2. τ 三档取值 ＋ 全精度 ＋ 主口径条件式 ＋ 三字段缺失即停工
3. `n/a` 规则（FPR／AUROC／Span-F1）
4. 报告／CSV 字段一致性（骨架 vs 演练**逐字段**）
5. 合成样本可复现性（同 seed 两次**逐字节相同**）
6. `--source sealed` 缺 `--access-log-note`／`--split` 即**非零退出**
7. 泄漏六项交集的**空值口径**（含空／排空**并列**）
8. C5：代码内不得出现 `.train(`；C9：不得硬编码 `reports/` 路径
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
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_01_eval_detection.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "detection.yaml")


def _load():
    spec = importlib.util.spec_from_file_location("eval_05_01", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_05_01"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


@pytest.fixture(scope="module")
def cfg():
    return M.load_config(CONFIG)


# ---------------------------------------------------------------------------
# ① C10 CSV 记录数：行数 ≠ 记录数
# ---------------------------------------------------------------------------
def test_csv_records_vs_lines_regression(tmp_path, cfg):
    p = tmp_path / "nl.csv"
    rows = [{"sample_id": "a1", "text": "l1\nl2\nmiddle\nl3", "label": "1"},
            {"sample_id": "a2", "text": "plain", "label": "0"}]
    M.write_csv_rows(str(p), ["sample_id", "text", "label"], rows)
    assert M.count_records(str(p)) == 2
    assert M.count_physical_lines(str(p)) >= 4
    assert M.count_physical_lines(str(p)) > M.count_records(str(p))


def test_single_record_with_two_newlines_counts_as_one(tmp_path):
    p = tmp_path / "one.csv"
    M.write_csv_rows(str(p), ["text"], [{"text": "head\nmid\ntail"}])
    assert M.count_records(str(p)) == 1
    assert M.count_physical_lines(str(p)) >= 3


def test_record_assertion_ok_and_mismatch(tmp_path, cfg):
    p = tmp_path / "ok.csv"
    M.write_csv_rows(str(p), ["text"], [{"text": "a"}, {"text": "b"}])
    assert M.assert_record_count(str(p), 2, cfg) == 2
    with pytest.raises(AssertionError):
        M.assert_record_count(str(p), 1583, cfg)


def test_record_assertions_official_values(cfg):
    """§8-2 四值 + val 两值：**禁改**（W0 实测权威值）。"""
    a = cfg["record_assertions"]
    assert (a["regime_id_test"], a["regime_ood_test"],
            a["clean_original_test"], a["g3_hard_negative_test"]) == (1583, 740, 44, 64)
    assert (a["id_val"], a["ood_val"]) == (1584, 927)


def test_val_record_counts_match_assertion(cfg):
    """实读 frozen val（A6 可读），**按记录计**。"""
    for regime, expect in (("id", 1584), ("ood", 927)):
        path = M._abs(cfg["inputs"]["val"][regime])
        if not os.path.isfile(path):
            continue
        assert M.count_records(path) == expect


# ---------------------------------------------------------------------------
# ② C4 τ：三档／全精度／条件式／缺失即停工
# ---------------------------------------------------------------------------
def _meta(tmp_path, tau=0.9999485611915588, override=False, gap=0.9626216776669025,
          drop=()):
    import json

    run_dir = tmp_path / "run"
    ckpt = run_dir / "seed_42" / "checkpoint-1574"
    ckpt.mkdir(parents=True, exist_ok=True)
    meta = {"tau_ours": tau, "tau_star": tau, "tau_main_override": override,
            "tau_neg_max_gap": gap, "regime": "id", "seed": 42, "step": 1574}
    for k in drop:
        meta.pop(k, None)
    (ckpt / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "best.json").write_text(
        json.dumps({"dir": str(ckpt), "step": 1574, "regime": "id", "seed": 42,
                    "segments": [{"start_step": 0, "end_step": 1574}]}), encoding="utf-8")
    return str(run_dir)


def test_tau_three_tiers_exist_and_full_precision(tmp_path, cfg):
    meta = M.load_run_meta(_meta(tmp_path), cfg)
    tau = M.resolve_tau_tracks(meta, cfg, 0.5123456789)
    assert set(tau["tiers"]) == {"fixed050", "median_val", "star"}
    assert tau["tiers"]["fixed050"] == 0.5
    assert tau["tiers"]["median_val"] == 0.5123456789
    assert tau["tiers"]["star"] == 0.9999485611915588
    assert M.fmt_tau(tau["tau_star"]) == repr(0.9999485611915588)


def test_tau_main_is_tau_star_when_not_triggered(tmp_path, cfg):
    meta = M.load_run_meta(_meta(tmp_path, override=False), cfg)
    tau = M.resolve_tau_tracks(meta, cfg, None)
    assert tau["main_track"] == "star" and tau["main_value"] == meta["tau_star"]
    assert tau["parallel_track"] == "fixed050" and tau["parallel_value"] == 0.5


def test_tau_main_is_050_when_override(tmp_path, cfg):
    meta = M.load_run_meta(_meta(tmp_path, override=True), cfg)
    tau = M.resolve_tau_tracks(meta, cfg, None)
    assert tau["main_track"] == "fixed050" and tau["main_value"] == 0.5
    assert tau["parallel_track"] == "star" and tau["parallel_value"] == meta["tau_star"]


@pytest.mark.parametrize("field", ["tau_ours", "tau_main_override", "tau_neg_max_gap"])
def test_tau_field_missing_stops(tmp_path, cfg, field):
    """§7.1：三字段任一缺失 ⇒ **停工**（抛错 ⇒ 上层非零退出），**禁补默认值**。"""
    with pytest.raises(RuntimeError) as exc:
        M.load_run_meta(_meta(tmp_path, drop=(field,)), cfg)
    assert "TAU_FIELDS_MISSING" in str(exc.value)


def test_tau_trigger_rule_not_recomputed(cfg):
    """§4-6：脚本只**读取** override，不得重算触发条件。"""
    src = open(SCRIPT, encoding="utf-8").read()
    assert "neg_max_gap_below" not in src
    assert "resolve_tau_ours" not in src


# ---------------------------------------------------------------------------
# ③ C7 n/a 规则
# ---------------------------------------------------------------------------
def test_na_rules_fpr_auroc_span(cfg):
    unify = M.load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    m = unify.compute_binary_metrics([1, 1], [1, 1], [0.9, 0.8])
    assert m["fpr"] is None and M.fmt_num(m["fpr"]) == M.TBD
    assert unify.compute_binary_metrics([1, 1], [1, 0], [0.9, 0.8])["auroc"] is None
    assert cfg["metrics"]["span_f1"] == "n/a"


def test_metrics_are_single_sourced_from_03_07():
    src = open(SCRIPT, encoding="utf-8").read()
    assert "scripts/baseline/03_07_unify_output_mapping.py" in src
    assert "def f1_score(" not in src and "def auroc(" not in src


# ---------------------------------------------------------------------------
# ④ 报告／CSV 字段一致性（骨架 vs 演练）
# ---------------------------------------------------------------------------
class _Args(object):
    ts = "20990101-000000"
    outputs_dir = None
    report_dir = None
    limit = None
    synthetic_seed = None
    n = None
    split = None


def test_report_fields_identical_skeleton_vs_dryrun(cfg):
    fake = M._fake_result(cfg, "id", 42, "fake_run", "synthetic_id",
                          [0.1 * i for i in range(1, 21)])
    ctx = M.build_context(cfg, _Args(), [fake], lead="x", title="t")
    h1 = M._extract_md_headers(M.render_detection_report(ctx))
    h2 = M._extract_md_headers(M.render_detection_report(M.skeleton_context(cfg, "ts"),
                                                         skeleton=True))
    assert h1 and h1 == h2


def test_report_contains_ood_val_declaration_verbatim(cfg):
    ctx = M.skeleton_context(cfg, "ts")
    text = M.render_detection_report(ctx, skeleton=True)
    for sentence in cfg["report"]["ood_val_declaration"]:
        assert sentence in text


def test_report_sections_all_present(cfg):
    text = M.render_detection_report(M.skeleton_context(cfg, "ts"), skeleton=True)
    for i in range(1, 12):
        assert ("## %d." % i) in text
    assert "EVAL_REPORT.md" in text


def test_metric_csv_fields(tmp_path, cfg):
    fake = M._fake_result(cfg, "id", 42, "fake_run", "synthetic_id", [0.1, 0.9])
    p = str(tmp_path / "m.csv")
    M.write_metric_csv(None, p, [fake], cfg)
    with open(p, encoding="utf-8", newline="") as fh:
        got = next(csv.reader(fh))
    assert got == M.METRIC_CSV_FIELDS
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and rows[0]["span_f1"] == "n/a"


# ---------------------------------------------------------------------------
# ⑤ C11 合成样本：可复现／不取自测试分片／长度对齐
# ---------------------------------------------------------------------------
def _val_stub(n=200):
    return [{"sample_id": "v%d" % i, "text": "x" * (50 + (i * 37) % 900),
             "label": str(i % 2)} for i in range(n)]


def test_synthetic_byte_identical_same_seed(tmp_path, cfg):
    fields = ["sample_id", "text", "label", "family", "encoding", "source"]
    r1, _ = M.generate_synthetic(cfg, "id", 60, 42, _val_stub())
    r2, _ = M.generate_synthetic(cfg, "id", 60, 42, _val_stub())
    p1, p2 = str(tmp_path / "s1.csv"), str(tmp_path / "s2.csv")
    M.write_csv_rows(p1, fields, r1)
    M.write_csv_rows(p2, fields, r2)
    assert open(p1, "rb").read() == open(p2, "rb").read()


def test_synthetic_differs_across_seeds(cfg):
    r1, _ = M.generate_synthetic(cfg, "id", 40, 42, _val_stub())
    r2, _ = M.generate_synthetic(cfg, "id", 40, 123, _val_stub())
    assert [x["sample_id"] for x in r1] != [x["sample_id"] for x in r2]


def test_synthetic_records_survive_roundtrip(tmp_path, cfg):
    rows, spec = M.generate_synthetic(cfg, "id", 120, 42, _val_stub())
    p = str(tmp_path / "s.csv")
    M.write_csv_rows(p, ["sample_id", "text", "label"], rows)
    assert M.count_records(p) == 120
    assert sum(1 for r in rows if "\n" in r["text"]) > 0   # 顺带验证引号转义
    assert spec["n"] == 120 and spec["seed"] == 42


def test_synthetic_must_not_source_test_shards(cfg):
    must_not = cfg["synthetic"]["must_not_source"]
    assert any("sealed_test" in x for x in must_not)
    assert any("test.csv" in x for x in must_not)
    assert cfg["synthetic"]["written_before_run"] is True


def test_synthetic_length_aligned_to_val_quantiles(cfg):
    val = _val_stub(400)
    rows, _ = M.generate_synthetic(cfg, "id", 400, 42, val)
    lv = M.quantile_table([len(x["text"]) for x in val])
    ls = M.quantile_table([len(x["text"]) for x in rows])
    assert abs(ls["p50"] - lv["p50"]) <= max(1.0, 0.5 * lv["p50"])
    assert abs(ls["p90"] - lv["p90"]) <= max(1.0, 0.5 * lv["p90"])


# ---------------------------------------------------------------------------
# ⑥ C12：`--source sealed` 缺 note／split ⇒ 非零退出
# ---------------------------------------------------------------------------
def _run_cli(extra, cwd=_ROOT):
    return subprocess.run([sys.executable, SCRIPT] + extra, cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def test_sealed_without_note_is_nonzero(tmp_path):
    r = _run_cli(["--source", "sealed", "--report-dir", str(tmp_path),
                  "--outputs-dir", str(tmp_path)])
    assert r.returncode != 0
    assert "ACCESS_LOG_NOTE_MISSING" in r.stdout or "REFUSED" in r.stdout


def test_sealed_without_split_is_nonzero(tmp_path):
    r = _run_cli(["--source", "sealed", "--access-log-note", "x",
                  "--report-dir", str(tmp_path), "--outputs-dir", str(tmp_path)])
    assert r.returncode != 0


def test_selfcheck_exit_zero():
    r = _run_cli(["--self-check"])
    assert "SELFCHECK_RESULT=PASS" in r.stdout
    assert r.returncode == 0


def test_eval_missing_dirs_is_nonzero(tmp_path):
    r = _run_cli(["--source", "val", "--regime", "id"])
    assert r.returncode != 0
    assert "MISSING_DIRS" in r.stdout


# ---------------------------------------------------------------------------
# ⑦ §7.5 泄漏六项：空值口径
# ---------------------------------------------------------------------------
def test_leak_six_columns_reports_both_values(tmp_path, cfg):
    fields = ["seed_id", "seed_group_id", "rewrite_id", "plaintext", "payload", "text"]
    tr = str(tmp_path / "tr.csv")
    va = str(tmp_path / "va.csv")
    M.write_csv_rows(tr, fields, [
        {"seed_id": "s1", "seed_group_id": "g1", "rewrite_id": "r1",
         "plaintext": "p1", "payload": "y1", "text": "t1"},
        {"seed_id": "", "seed_group_id": "", "rewrite_id": "",
         "plaintext": "", "payload": "", "text": "t2"}])
    M.write_csv_rows(va, fields, [
        {"seed_id": "zz", "seed_group_id": "zz", "rewrite_id": "zz",
         "plaintext": "zz", "payload": "zz", "text": "zz"},
        {"seed_id": "", "seed_group_id": "", "rewrite_id": "",
         "plaintext": "", "payload": "", "text": "tv"}])
    out = {c["column"]: c for c in M.leak_six_columns(cfg, tr, va)}
    assert cfg["leakcheck"]["six_columns"][0] == "family_id"
    assert len(cfg["leakcheck"]["six_columns"]) == 6
    for col in ("seed_id", "seed_group_id", "rewrite_id", "plaintext", "payload"):
        assert out[col]["inter_with_empty"] == 1
        assert out[col]["inter_drop_empty"] == 0
    assert out["text"]["inter_drop_empty"] == 0


def test_leakcheck_config_reports_train_test_tbd(cfg):
    assert cfg["leakcheck"]["report_both_before_and_after"] is True
    assert "TBD" in cfg["leakcheck"]["scope_train_test"]
    assert cfg["leakcheck"]["empty_policy"]


# ---------------------------------------------------------------------------
# ⑧ C1／C5／C9 静态纪律
# ---------------------------------------------------------------------------
def test_no_train_call_in_source():
    """C5：AST 级检查 —— 代码中不得出现任何 `.train` 属性调用（注释／文档串不算）。"""
    import ast

    tree = ast.parse(open(SCRIPT, encoding="utf-8").read())
    bad = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "train"]
    assert not bad


def test_no_hardcoded_reports_path():
    """C9：产物输出路径不得写死（`reports/…` 只许出现在命令行示例与文档串中）。"""
    import ast

    src = open(SCRIPT, encoding="utf-8").read()
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)
    code = src
    for d in docstrings:
        code = code.replace(d, "")
    code = "\n".join(l for l in code.split("\n") if not l.lstrip().startswith("#"))
    assert '"reports/' not in code and "'reports/" not in code


def test_input_whitelist_rejects_sealed_in_implementation(cfg):
    sealed = cfg["inputs"]["sealed"]["in_regime_test"]["id"]
    with pytest.raises(AssertionError):
        M.assert_input_allowed(M._abs(sealed), cfg, allow_sealed=False)


def test_input_whitelist_rejects_frozen_test_csv(cfg):
    frozen_test = os.path.join(cfg["inputs"]["root"], "regime_id", "test.csv")
    with pytest.raises(AssertionError):
        M.assert_input_allowed(M._abs(frozen_test), cfg, allow_sealed=False)


def test_cublas_set_before_torch_import():
    src = open(SCRIPT, encoding="utf-8").read()
    i_env = src.index("CUBLAS_WORKSPACE_CONFIG")
    assert i_env < src.index("class Engine")


def test_engine_config_eager_and_determinism(cfg):
    """C5：`attn_implementation=eager`、禁 flash-attn、TF32 关、cudnn 确定性。"""
    d = cfg["engine"]["determinism"]
    assert cfg["engine"]["train_config"]
    assert d["attn_implementation"] == "eager"
    assert "flash_attention_2" in d["forbidden_attn"]
    assert d["cudnn_deterministic"] is True and d["cudnn_benchmark"] is False
    assert d["allow_tf32"] is False
    assert d["cublas_workspace_config"] == ":4096:8"
    assert cfg["engine"]["eval_state"]["model_eval"] is True
    assert cfg["engine"]["eval_state"]["no_grad"] is True


def test_budget_thresholds_preset(cfg):
    assert cfg["budget"]["thresholds"]["max_total_minutes"] == 60
    assert cfg["budget"]["thresholds"]["max_peak_mem_mib"] == 8000
    assert cfg["budget"]["n_default"] == 1583
    assert cfg["budget"]["must_run_full"] is True


def test_per_sample_output_schema_keeps_unify_order():
    assert M.OUTPUT_FIELDS[:7] == ["sample_id", "text_sha256", "pred", "raw_type", "raw",
                                   "score", "latency_ms"]


# ---------------------------------------------------------------------------
# ⑨ 本轮修正：`EVAL_PENDING` 按**调用范围**判定（单体制 3 ／两体制 6）
# ---------------------------------------------------------------------------
def _sel(*pairs):
    return [(r, s, "dir_%s_%s" % (r, s)) for r, s in pairs]


def _mkargs(**kw):
    """带默认值的 args 桩（`_Args` 是空类，不能带参构造）。"""
    import types
    base = dict(ts="20990101-000000", outputs_dir=None, report_dir=None, limit=None,
                synthetic_seed=None, n=None, split=None, regime=None, seed=None,
                source=None, access_log_note=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_gate_single_regime_passes(cfg):
    """单体制调用只含 3 个 run ⇒ **应通过**（修正前被全量 6 误判为 EVAL_PENDING）。"""
    r = M.select_runs(_sel(("id", 42), ("id", 123), ("id", 456)), _mkargs(regime="id"))
    assert M.assert_evaluation_allowed(cfg, r, lambda *a, **k: None) is True


def test_gate_single_regime_ood_passes(cfg):
    r = M.select_runs(_sel(("ood", 42), ("ood", 123), ("ood", 456)), _mkargs(regime="ood"))
    assert M.assert_evaluation_allowed(cfg, r, lambda *a, **k: None) is True


def test_gate_both_regimes_passes(cfg):
    r = M.select_runs(_sel(("id", 42), ("id", 123), ("id", 456),
                           ("ood", 42), ("ood", 123), ("ood", 456)), _mkargs(regime=None))
    assert len(r) == 6
    assert M.assert_evaluation_allowed(cfg, r, lambda *a, **k: None) is True


def test_gate_fails_when_seed_missing(cfg):
    """范围内 seed 不全 ⇒ 仍判 EVAL_PENDING（不得放行）。"""
    r = M.select_runs(_sel(("id", 42), ("id", 123)), _mkargs(regime="id"))
    assert M.assert_evaluation_allowed(cfg, r, lambda *a, **k: None) is False


def test_gate_fails_on_empty_scope(cfg):
    assert M.assert_evaluation_allowed(cfg, [], lambda *a, **k: None) is False


def test_select_runs_both_regimes_by_default(cfg):
    all6 = _sel(("id", 42), ("id", 123), ("id", 456),
                ("ood", 42), ("ood", 123), ("ood", 456))
    assert len(M.select_runs(all6, _mkargs(regime=None, seed=None))) == 6
    assert len(M.select_runs(all6, _mkargs(regime="id", seed=None))) == 3
    assert M.select_runs(all6, _mkargs(regime=None, seed=42)) == _sel(("id", 42), ("ood", 42))


def test_scope_of_marks_coverage(cfg):
    a = _mkargs(regime="id", seed=None)
    sc = M.scope_of(cfg, M.select_runs(_sel(("id", 42), ("id", 123), ("id", 456)), a), a)
    assert sc["regimes"] == ["id"] and sc["expect_n"] == 3
    assert "覆盖完整" in sc["coverage_note"] and sc["other_regimes"] == ["ood"]
    sc2 = M.scope_of(cfg, M.select_runs(_sel(("id", 42)), a), a)
    assert "EVAL_PENDING" in sc2["coverage_note"]


# ---------------------------------------------------------------------------
# ⑩ 本轮修正：`n/a`（无定义）与 `<TBD>`（待填）**不得互串**
# ---------------------------------------------------------------------------
def test_fmt_metric_none_is_na_not_tbd():
    assert M.fmt_metric(None) == "n/a"
    assert M.fmt_metric(0.0) == "0.0000"
    assert M.TBD == "<TBD>"


def test_all_negative_shard_metrics_are_na(cfg):
    """全负分片（g3 型：64 条全 label=0）⇒ TPR／F1／AUROC = `n/a`，FPR 定值。"""
    unify = M.load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    labels = [0] * 8
    scores = [0.1] * 4 + [0.9] * 4
    m = M.compute_metrics(unify, labels, scores, 0.5)
    assert m["n_pos"] == 0
    assert m["tpr"] is None and m["f1"] is None and m["auroc"] is None
    assert m["fpr"] == 0.5
    assert M.fmt_metric(m["tpr"]) == "n/a" and M.fmt_metric(m["f1"]) == "n/a"
    assert M.fmt_metric(m["fpr"]) == "0.5000"


def test_all_positive_shard_fpr_is_na(cfg):
    unify = M.load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    m = M.compute_metrics(unify, [1, 1], [0.9, 0.1], 0.5)
    assert m["fpr"] is None and M.fmt_metric(m["fpr"]) == "n/a"
    assert m["tpr"] == 0.5 and m["f1"] is not None


def test_report_renders_na_for_undefined_metric(cfg):
    """渲染层：无定义指标格必须写 `n/a`，且**不得**在正式报告里出现 `<TBD>`。"""
    fake = M._fake_result(cfg, "id", 42, "run_na", "g3_hard_negative_test", [0.1, 0.9])
    for t in fake["tracks"]:
        fake["tracks"][t].update({"tpr": None, "f1": None, "auroc": None, "fpr": 0.0})
    ctx = M.build_context(cfg, _mkargs(), [fake], lead="x", title="t")
    text = M.render_detection_report(ctx)
    # 指标表（§2）内**不得**出现 `<TBD>`；无定义格必须是 `n/a`
    sec2 = text.split("## 2. 指标总表")[1].split("## 3.")[0]
    assert "<TBD>" not in sec2
    assert "| n/a |" in sec2 or "n/a" in sec2
    # 骨架态才用 `<TBD>`
    assert M.TBD in M.render_detection_report(M.skeleton_context(cfg, "ts"), skeleton=True)


def test_metric_csv_writes_na_not_crash(tmp_path, cfg):
    fake = M._fake_result(cfg, "id", 42, "run_na", "g3_hard_negative_test", [0.1, 0.9])
    for t in fake["tracks"]:
        fake["tracks"][t].update({"tpr": None, "f1": None, "auroc": None, "fpr": 0.0})
    p = str(tmp_path / "na.csv")
    M.write_metric_csv(None, p, [fake], cfg)
    with open(p, encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["tpr"] == "n/a" and row["f1"] == "n/a" and row["auroc"] == "n/a"
    assert row["fpr"] == "0.00000000"

# -*- coding: utf-8 -*-
"""tests/eval/test_05_03_eval_ood.py — W6.4（5.3 OOD 泛化评估）单元与接口测试。

不触 GPU、不读 `data/sealed_test/**`、不读 frozen 侧 `test.csv`（`W6.4` §3-C1）。
重型库（torch／transformers／peft）一律不在测试内 import —— 被测脚本顶层亦不 import。

覆盖（`windows/W6.4_ood.md` §7 验收项 ＋ §3 硬约束）：
1. 记录数断言（740／927，**取自 `W6.1` 的 detection.yaml，本文件只校验一致性**）
2. `csv.DictReader` 口径（`text` 含换行 ⇒ **记录数 ≠ 行数**，§C2）
3. C4 逐留出码分组（**TPR 分母只含 `label==1`**；各码 FPR 相同；未出现码显式列出）
4. C4 留出码构成（训练码 5 ／ 留出码 4；`ood_val` 只含训练码）
5. C3＋三句声明**逐字在位**（OOD＝编码变换 OOD；`ood_val` 不测 OOD 泛化；val 不独立）
6. τ 取值规则（主口径 = `best epoch` 的 `tau_ours`；`override=False` ⇒ 主 = τ*）
7. C11 `--source sealed` 缺 `--access-log-note` ⇒ **非零退出**
8. 衰减：公式 = ID − OOD；一侧 `n/a` ⇒ 衰减无定义；ID 侧基准缺失 ⇒ 停工
9. C9 `std=0` 退化声明；C10 引数带口径的 mandatory notes
10. C7／C8 静态纪律（无 `.train(`、无硬编码 `reports/`；`attn=eager`、TF32 关）
11. 确定性双跑（逐码分组两次逐字节相同）与 `--self-check` exit=0
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
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_03_eval_ood.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "ood.yaml")
BASE_CONFIG = os.path.join(_ROOT, "configs", "eval", "detection.yaml")
UNIFY = os.path.join(_ROOT, "scripts", "baseline", "03_07_unify_output_mapping.py")


def _load():
    spec = importlib.util.spec_from_file_location("eval_05_03", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_05_03"] = mod
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
    """带默认值的 args 桩（`build_ctx` 需 `ts`；`source=sealed` 时需 `access_log_note`）。"""
    base_kw = dict(ts="20990101-000000", outputs_dir=None, report_dir=None, limit=None,
                   source=None, access_log_note=None, config=CONFIG, dry_run=False,
                   self_check=False, tau_mode="both", only=None, run_dir=None,
                   regime=None, seed=None, split="in_regime_test")
    base_kw.update(kw)
    return types.SimpleNamespace(**base_kw)


def _fake_result(cfg, group="primary", regime="ood", seed=42, main_track="star",
                 metrics=None, by_encoding=None, n=740):
    """最小可用的 run 结果桩（字段与 `run_eval` 产出一致）。"""
    metrics = metrics or {"tpr": 0.94318182, "fpr": 0.0, "f1": 0.97076023, "auroc": 0.99,
                          "n_pos": 352, "n_neg": 388, "tp": 332, "fp": 0, "fn": 20, "tn": 388,
                          "tau": 0.9999945759773254}
    parallel = dict(metrics, tpr=0.0, f1=0.0, tau=0.5)
    tracks = {main_track: metrics}
    if main_track != "fixed050":
        tracks["fixed050"] = parallel
    else:
        tracks["star"] = parallel
    enc = by_encoding
    if enc is None:
        enc = [{"encoding_type": c, "is_heldout": True, "n_attack": 88, "n_benign": 388,
                "n": 476, "tp": 70, "fp": 0, "fn": 18, "tn": 388,
                "tpr": 70 / 88.0, "fpr": 0.0, "f1": 0.8}
               for c in ("hex", "morse", "ascii_art", "homoglyph")]
    return {"group": group, "regime": regime, "seed": seed, "run_id": "fake_%s_%d" % (regime, seed),
            "shard": "ood_test", "source": "sealed", "n": n,
            "main_track": main_track,
            "parallel_track": "fixed050" if main_track != "fixed050" else "star",
            "tracks": tracks, "by_encoding": enc,
            "latency": {"mean_ms": 300.0, "median_ms": 280.0, "p95_ms": 500.0},
            "mem_load_peak_mib": 1500.0, "mem_infer_peak_mib": 4200.0,
            "eval_batch_max_size": 16,
            "best_epoch": 1, "best_step": 1574, "early_stopped": False,
            "adapter_md5": "0" * 32, "segments": [{"start_step": 0, "end_step": 1574}]}


# ---------------------------------------------------------------------------
# ① C2 记录数断言（740／927；本文件登记值 ↔ base_config 逐格一致）
# ---------------------------------------------------------------------------
def test_record_assertions_official_values(cfg, base):
    """§C2：`regime_ood_test` = 740、`ood_val` = 927，**取自 `W6.1` 的 detection.yaml，禁改**。"""
    assert int(base["record_assertions"]["regime_ood_test"]) == 740
    assert int(base["record_assertions"]["ood_val"]) == 927


def test_record_count_policy_matches_base(cfg, base):
    """本文件登记值 ↔ `base_config` 逐格一致（避免两处口径漂移）。"""
    exp = cfg["record_count_policy"]["expected"]
    assert int(exp["regime_ood_test"]) == int(base["record_assertions"]["regime_ood_test"])
    assert int(exp["ood_val"]) == int(base["record_assertions"]["ood_val"])
    assert cfg["record_count_policy"]["keys_in_scope"] == ["regime_ood_test", "ood_val"]
    assert "stop_and_escalate_W0" in cfg["record_count_policy"]["on_mismatch"]
    assert "csv.DictReader" in cfg["record_count_policy"]["must_use"]


# ---------------------------------------------------------------------------
# ② C2 `csv.DictReader`：行数 ≠ 记录数
# ---------------------------------------------------------------------------
def test_csv_dictreader_newline_regression(tmp_path, base):
    p = tmp_path / "nl.csv"
    M.write_csv_rows(str(p), ["sample_id", "encoding_type", "text", "label"],
                     [{"sample_id": "a1", "encoding_type": "hex",
                       "text": "l1\nl2\nmid\nl3", "label": "1"},
                      {"sample_id": "a2", "encoding_type": "", "text": "plain", "label": "0"}])
    W_ = M.W()
    assert W_.count_records(str(p)) == 2
    assert W_.count_physical_lines(str(p)) >= 4
    assert W_.count_physical_lines(str(p)) > W_.count_records(str(p))


def test_assert_record_count_ok_and_mismatch(tmp_path, base):
    p = tmp_path / "ok.csv"
    M.write_csv_rows(str(p), ["text"], [{"text": "a"}, {"text": "b"}])
    W_ = M.W()
    assert W_.assert_record_count(str(p), 2, base) == 2
    with pytest.raises(AssertionError):
        W_.assert_record_count(str(p), 740, base)


def test_val_record_counts_match_assertion(cfg, base):
    """实读 frozen val（A6 可读、非封存），**按记录计**（不触 GPU）。"""
    for regime, expect in (("id", 1584), ("ood", 927)):
        path = M._abs(base["inputs"]["val"][regime])
        if not os.path.isfile(path):
            continue
        assert M.W().count_records(path) == expect


# ---------------------------------------------------------------------------
# ③ C4 逐留出码分组：TPR 分母只含 `label==1`；FPR 各码相同；未出现码显式列出
# ---------------------------------------------------------------------------
def _enc_rows():
    rows = [{"sample_id": "h%d" % i, "encoding_type": "hex", "label": "1"} for i in range(5)]
    rows += [{"sample_id": "b%d" % i, "encoding_type": "", "label": "0"} for i in range(3)]
    rows += [{"sample_id": "z%d" % i, "encoding_type": "hex", "label": "0"} for i in range(2)]
    scores = [0.9, 0.9, 0.1, 0.9, 0.9] + [0.1, 0.1, 0.9] + [0.9, 0.1]
    return rows, scores


def test_per_encoding_tpr_denominator_attack_only(unify, cfg):
    rows, scores = _enc_rows()
    enc = M.per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    hexrow = [e for e in enc if e["encoding_type"] == "hex"][0]
    # n_attack 只数 label==1 且编码为 hex 的 5 条；**不含**编码为 hex 但良性的 2 条
    assert hexrow["n_attack"] == 5
    assert hexrow["n_benign"] == 5          # 全部良性 = 3（无编码）+ 2（hex 编码）
    assert abs(hexrow["tpr"] - 0.8) < 1e-9  # 4/5，分母只含 label==1


def test_per_encoding_fpr_constant_across_codes(unify, cfg):
    """良性样本无编码归属 ⇒ 各码 FPR 相同（逐码差异只在 TPR／F1）。"""
    rows, scores = _enc_rows()
    enc = M.per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    fprs = {e["encoding_type"]: e["fpr"] for e in enc}
    assert len(set(fprs.values())) == 1


def test_per_encoding_absent_code_explicitly_listed(unify, cfg):
    """未出现的留出码**显式列出**（n_attack=0、TPR=n/a），不静默省略。"""
    rows, scores = _enc_rows()
    enc = M.per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    by = {e["encoding_type"]: e for e in enc}
    for code in ("morse", "ascii_art", "homoglyph"):
        assert code in by
        assert by[code]["n_attack"] == 0 and by[code]["tpr"] is None
    assert M.fmt4(None) == M.NA


def test_codes_in_shard_order_config_first(unify, cfg):
    """报告顺序：先配置中的留出码（hex／morse／ascii_art／homoglyph），再出现但未登记的编码。"""
    rows = [{"sample_id": "x", "encoding_type": "zeta", "label": "1"},
            {"sample_id": "h", "encoding_type": "hex", "label": "1"}]
    assert M.codes_in_shard(rows, cfg) == ["hex", "morse", "ascii_art", "homoglyph", "zeta"]


def test_heldout_count_mismatch_registers_only_deviation(cfg):
    by_enc = [{"encoding_type": "hex", "is_heldout": True, "n_attack": 88},
              {"encoding_type": "morse", "is_heldout": True, "n_attack": 87},
              {"encoding_type": "base64", "is_heldout": False, "n_attack": 88}]
    out = M.heldout_count_mismatch(by_enc, cfg)
    assert len(out) == 1 and "morse" in out[0] and "如实登记并呈 W0" in out[0]


def test_per_encoding_deterministic(unify, cfg):
    rows, scores = _enc_rows()
    e1 = M.per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    e2 = M.per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    assert [x["tpr"] for x in e1] == [x["tpr"] for x in e2]


# ---------------------------------------------------------------------------
# ④ C4 留出码构成（训练码 5 ／ 留出码 4；`ood_val` 只含训练码）
# ---------------------------------------------------------------------------
def test_ood_encoding_composition(cfg):
    enc = cfg["ood_encodings"]
    assert enc["train_codes"] == ["base64", "rot13", "zerowidth", "url", "reverse"]
    assert enc["heldout_codes"] == ["hex", "morse", "ascii_art", "homoglyph"]
    assert len(enc["train_codes"]) == 5 and len(enc["heldout_codes"]) == 4
    assert cfg["per_encoding"]["codes"] == enc["heldout_codes"]      # 顺序固定、单源
    assert int(enc["heldout_expected_attack_n"]) == 88
    assert int(enc["heldout_attack_total"]) == 352                   # 88 × 4 = ood_test 正类数
    assert sorted(enc["val_codes_present"]) == sorted(enc["train_codes"])  # ood_val 只含训练码


def test_runs_policy_primary_ood_cross_id(cfg):
    rp = cfg["runs_policy"]
    assert rp["primary_group"] == "ood"
    assert rp["cross_regime_group"] == "id"
    assert rp["cross_regime_must_not_mix"] is True
    assert [int(s) for s in rp["seeds"]] == [42, 123, 456]
    assert rp["require_all_in_group"] is True


def test_group_of(cfg):
    assert M.group_of("ood", cfg) == "primary"
    assert M.group_of("id", cfg) == "cross"
    assert M.group_of("other", cfg) == "other"


def test_assert_group_complete(cfg):
    logs = []
    ok = M.assert_group_complete(
        cfg, [{"group": "primary", "regime": "ood", "seed": s} for s in (42, 123, 456)],
        logs.append)
    assert ok is True and "GROUP_OK" in logs[-1]
    bad = M.assert_group_complete(
        cfg, [{"group": "primary", "regime": "ood", "seed": s} for s in (42, 123)],
        logs.append)
    assert bad is False and "GROUP_INCOMPLETE" in logs[-1]


# ---------------------------------------------------------------------------
# ⑤ C3＋三句声明逐字在位
# ---------------------------------------------------------------------------
def test_ood_definition_verbatim(cfg):
    v = cfg["ood_definition"]["verbatim"][0]
    assert "编码方式分布外" in v and "不是语义 OOD" in v
    assert any("未见恶意语义" in x for x in cfg["ood_definition"]["forbidden_wording"])


def test_three_sentences_verbatim(cfg):
    joined = " ".join(cfg["declarations"]["ood_val_three_sentences"])
    assert "不测 OOD 泛化" in joined and "只用于早停与阈值" in joined
    assert "OOD 结论只取 `ood_test`" in joined
    assert "不得因 `ood_val` 饱和而降低 ood 的训练要求" in joined


def test_val_nonindependence_and_test_boundary(cfg):
    decl = " ".join(cfg["declarations"]["val_nonindependence"])
    assert "共享同一批验证文本" in decl and "验证集指标不独立" in decl
    assert "927" in decl and "1557" in decl        # 交集 = |ood_val| = 927，实测依据
    assert "相互独立" in decl and "正式结论仍以 test 为准" in decl


def test_ood_val_composition_blocks_generalization_claim(cfg):
    comp = " ".join(cfg["declarations"]["ood_val_composition"])
    assert "留出码只出现在 `ood_test`" in comp
    assert "不构成 OOD 泛化证据" in comp


# ---------------------------------------------------------------------------
# ⑥ τ 取值规则（主口径 = `best epoch` 的 `tau_ours`；不重算触发条件）
# ---------------------------------------------------------------------------
def test_tau_main_is_star_when_not_triggered(cfg, base):
    meta = {"tau_star": 0.9999945759773254, "override": False, "gap": 0.9}
    tau = M.W().resolve_tau_tracks(meta, base, None)
    assert tau["main_track"] == "star" and tau["main_value"] == meta["tau_star"]
    assert tau["parallel_track"] == "fixed050" and tau["parallel_value"] == 0.5
    assert M.fmt_tau(tau["tau_star"]) == repr(0.9999945759773254)


def test_tau_config_track_star(cfg):
    assert cfg["decay"]["tau_track"] == "star"
    assert cfg["eval_state"]["cold_start_double_run"]


def test_tau_trigger_rule_not_recomputed():
    """脚本只**读取** override，不得重算触发条件（同 `W6.1` §4-6）。"""
    src = open(SCRIPT, encoding="utf-8").read()
    assert "neg_max_gap_below" not in src and "resolve_tau_ours" not in src


# ---------------------------------------------------------------------------
# ⑦ 衰减：公式 = ID − OOD；一侧 `n/a` ⇒ 无定义；ID 侧基准缺失 ⇒ 停工
# ---------------------------------------------------------------------------
def _mini_decay_cfg(artifacts=("x.csv",), seeds=(42,)):
    return {"decay": {"id_reference": {"artifacts": list(artifacts),
                                       "required_seeds": list(seeds)},
                      "metrics": ["tpr", "fpr", "f1"], "tau_track": "star"}}


def test_decay_formula_id_minus_ood(cfg):
    mini = _mini_decay_cfg()
    drows = M.build_decay_rows(mini, {42: {"tpr": 1.0, "fpr": 0.0, "f1": 0.99746835}},
                               {42: {"tpr": 0.94318182, "fpr": 0.0, "f1": 0.97076023}},
                               lambda *a, **k: None)
    tpr = [d for d in drows if d["metric"] == "tpr"][0]
    assert abs(tpr["decay"] - (1.0 - 0.94318182)) < 1e-12
    assert "同 τ 口径" in tpr["note"]


def test_decay_undefined_when_na(cfg):
    mini = _mini_decay_cfg()
    drows = M.build_decay_rows(mini, {}, {42: {"tpr": 0.5}}, lambda *a, **k: None)
    assert drows[0]["decay"] is None
    assert "无定义" in drows[0]["note"]


def test_load_id_reference_missing_stops(cfg, tmp_path):
    """ID 侧基准件不存在 ⇒ `FileNotFoundError`（不静默改用 B3 数替代）。"""
    bad = {"decay": dict(cfg["decay"])}
    bad["decay"]["id_reference"] = dict(cfg["decay"]["id_reference"])
    bad["decay"]["id_reference"]["artifacts"] = ["reports/eval/__no_such_file__.csv"]
    with pytest.raises(FileNotFoundError):
        M.load_id_reference(bad, lambda *a, **k: None)


def test_decay_config_points_to_w61_not_baseline(cfg):
    ref = cfg["decay"]["id_reference"]
    assert "detection_formal" in ref["artifacts"][0]     # W6.1 的 ID 同 τ 口径结果
    assert ref["select"] == {"shard": "id_test", "regime": "id", "track": "star",
                             "source": "sealed"}
    assert [int(s) for s in ref["required_seeds"]] == [42, 123, 456]
    assert len(cfg["decay"]["incomparable_disclosure"]) >= 3   # 两体制不可比要素须声明


def test_decay_csv_fields(tmp_path, cfg):
    mini = _mini_decay_cfg()
    drows = M.build_decay_rows(mini, {42: {"tpr": 1.0, "fpr": 0.0, "f1": 1.0}},
                               {42: {"tpr": 0.9, "fpr": 0.0, "f1": 0.9}}, lambda *a, **k: None)
    p = str(tmp_path / "d.csv")
    M.write_decay_csv(p, drows)
    with open(p, encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == M.DECAY_CSV_FIELDS


# ---------------------------------------------------------------------------
# ⑧ C9 std=0 退化；C10 引数带口径
# ---------------------------------------------------------------------------
def test_std_zero_declaration_triggers(cfg):
    same = [{"tpr": 0.9, "fpr": 0.0, "f1": 0.8, "auroc": 0.9} for _ in range(3)]
    results = []
    for i, m in enumerate(same):
        r = _fake_result(cfg, seed=(42, 123, 456)[i])
        r["tracks"] = {"star": dict(m, n_pos=352, n_neg=388, tau=0.9,
                                    tp=1, fp=0, fn=0, tn=1)}
        r["main_track"] = "star"
        results.append(r)
    lines = " ".join(M._std_zero_lines(results))
    assert "std=0" in lines and "显式声明退化" in lines


def test_std_zero_not_triggered_reports_plain(cfg):
    results = []
    for i, tpr in enumerate((0.9, 0.8, 0.7)):
        r = _fake_result(cfg, seed=(42, 123, 456)[i])
        r["tracks"] = {"star": {"tpr": tpr, "fpr": 0.1 * i, "f1": tpr,
                                "auroc": 0.9 - 0.05 * i,
                                "n_pos": 352, "n_neg": 388, "tau": 0.9,
                                "tp": 1, "fp": 0, "fn": 0, "tn": 1}}
        r["main_track"] = "star"
        results.append(r)
    lines = " ".join(M._std_zero_lines(results))
    assert "未检出" in lines


def test_baseline_citation_mandatory_notes(cfg):
    notes = " ".join(cfg["baseline_citation"]["mandatory_notes"])
    assert "结构性因素" in notes          # B1 ood_test TPR 1.0
    assert "平凡解" in notes              # B2 1.0/1.0
    assert "CPU 档" in notes              # B4
    assert "不适用" in notes              # AUROC 的 n/a 语义


def test_baseline_rows_all_five_present():
    rows = M._baseline_rows()
    assert len(rows) == 5
    assert all(r[0].startswith("**B") for r in rows)


# ---------------------------------------------------------------------------
# ⑨ 报告结构（骨架与实跑同一渲染器）
# ---------------------------------------------------------------------------
def test_report_sections_all_present(cfg):
    ctx = M.build_ctx(cfg, _args(), [], None, "sealed", M.TBD, skeleton=True)
    text = M.render_ood_report(ctx)
    for sec in ("## §0 ", "## §1 ", "## §2 ", "## §3 ", "## §4 ", "## §5 ",
                "## §6 ", "## §7 ", "## §8 ", "## §9 ", "## §10 "):
        assert sec in text
    for sub in ("### §3.1 ", "### §4.1 ", "### §6.1", "### §6.2", "### §6.3",
                "### §2.1 ", "### §2.2 "):
        assert sub in text


def test_skeleton_uses_tbd_not_na(cfg):
    ctx = M.build_ctx(cfg, _args(), [], None, "sealed", M.TBD, skeleton=True)
    text = M.render_ood_report(ctx)
    assert M.TBD in text
    # 骨架态数值位用 `<TBD>`；`n/a`（不适用）另属一义，不得混用为占位
    assert "§2 OOD 主档指标" in text


def test_report_contains_eval_report_523_skeleton(cfg):
    ctx = M.build_ctx(cfg, _args(), [], None, "sealed", M.TBD, skeleton=True)
    text = M.render_ood_report(ctx)
    assert "可直接搬入 `reports/eval/EVAL_REPORT.md` §5.3" in text
    assert "### 5.3 OOD 泛化" in text


def test_report_embeds_three_declarations(cfg):
    ctx = M.build_ctx(cfg, _args(), [], None, "sealed", M.TBD, skeleton=True)
    text = M.render_ood_report(ctx)
    assert "不测 OOD 泛化" in text
    assert "OOD 结论只取 `ood_test`" in text or "OOD 结论**只取" in text
    for s in cfg["declarations"]["val_nonindependence"]:
        assert s.split("。")[0][:20] in text


def test_report_headers_stable_between_renderings(cfg):
    ctx = M.build_ctx(cfg, _args(), [_fake_result(cfg)], "ood_test", "sealed", 740)
    h1 = M._extract_md_headers(M.render_ood_report(ctx))
    ctx2 = M.build_ctx(cfg, _args(), [_fake_result(cfg)], "ood_test", "sealed", 740)
    h2 = M._extract_md_headers(M.render_ood_report(ctx2))
    assert h1 and h1 == h2


def test_report_sealed_lead_echoes_access_note(cfg):
    ctx = M.build_ctx(cfg, _args(source="sealed", access_log_note="NOTE-XYZ"),
                      [_fake_result(cfg)], "ood_test", "sealed", 740)
    text = M.render_ood_report(ctx)
    assert "NOTE-XYZ" in text


def test_val_lead_marks_nonconclusion(cfg):
    ctx = M.build_ctx(cfg, _args(source="val"), [], None, "val", 927)
    text = M.render_ood_report(ctx)
    assert "不是 OOD 泛化结论" in text or "不构成 OOD 结论" in text


# ---------------------------------------------------------------------------
# ⑩ CSV 字段与写盘
# ---------------------------------------------------------------------------
def test_by_encoding_csv_fields(tmp_path, cfg):
    p = str(tmp_path / "e.csv")
    M.write_by_encoding_csv(p, [_fake_result(cfg)], cfg)
    with open(p, encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == M.BY_ENC_CSV_FIELDS
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4 and {r["encoding_type"] for r in rows} == \
        {"hex", "morse", "ascii_art", "homoglyph"}


def test_metric_csv_fields_and_na(tmp_path, cfg):
    r = _fake_result(cfg)
    r["tracks"]["star"]["auroc"] = None       # 无连续 score 语义
    p = str(tmp_path / "m.csv")
    M.write_metric_csv(p, [r], cfg)
    with open(p, encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == M.OOD_CSV_FIELDS
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    star = [r for r in rows if r["tau_track"] == "star"][0]
    assert star["auroc"] == "n/a" and star["tpr"] != "n/a"


def test_write_csv_rows_no_cr(tmp_path, cfg):
    p = str(tmp_path / "x.csv")
    M.write_csv_rows(p, ["a", "b"], [{"a": "1", "b": "text\nwith\nnl"}])
    with open(p, "rb") as fh:
        assert b"\r" not in fh.read()


# ---------------------------------------------------------------------------
# ⑪ C7／C8 静态纪律 ＋ 单源复用（禁另写一套）
# ---------------------------------------------------------------------------
def test_no_train_call_in_source():
    """C7：代码中不得出现任何 `.train` 属性调用（注释／文档串不算）。"""
    import ast

    tree = ast.parse(open(SCRIPT, encoding="utf-8").read())
    bad = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "train"]
    assert not bad


def test_no_hardcoded_reports_path():
    """C8：产物输出路径不得写死（`reports/…` 只许出现在文档串与引数注释稿中）。"""
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


def test_no_line_split_reading():
    """C2：不得按行读 CSV（`readlines(`／`splitlines(` 全库禁用）。"""
    src = open(SCRIPT, encoding="utf-8").read()
    assert ("read" + "lines(") not in src
    assert ("split" + "lines(") not in src


def test_cublas_set_before_torch_import():
    src = open(SCRIPT, encoding="utf-8").read()
    i_env = src.index("CUBLAS_WORKSPACE_CONFIG")
    assert i_env < src.index("def evaluate_grouped_run")


def test_reuses_05_01_guard_not_rewritten():
    """C1／C6：复用 `05_01.assert_input_allowed`，**不得另写一套**白名单守卫。"""
    src = open(SCRIPT, encoding="utf-8").read()
    assert "assert_input_allowed" in src
    assert "def assert_input_allowed" not in src       # 不定义自己的版本
    assert "05_01_eval_detection" in src               # 指向同一实现源


def test_no_own_inference_loop():
    """C6：推理须经 `Engine.score`，不得另写 forward 循环。"""
    src = open(SCRIPT, encoding="utf-8").read()
    assert "def forward_scores" not in src
    assert "Engine" in src and ".score(" in src


def test_input_whitelist_rejects_sealed_in_implementation(base):
    sealed = base["inputs"]["sealed"]["in_regime_test"]["ood"]
    with pytest.raises(AssertionError):
        M.W().assert_input_allowed(M._abs(sealed), base, allow_sealed=False)


def test_input_whitelist_rejects_frozen_test_csv(base):
    frozen_test = os.path.join(base["inputs"]["root"], "regime_ood", "test.csv")
    with pytest.raises(AssertionError):
        M.W().assert_input_allowed(M._abs(frozen_test), base, allow_sealed=False)


def test_config_reuse_pointers(cfg):
    assert cfg["base_config"] == "configs/eval/detection.yaml"
    assert cfg["reuse_module"] == "scripts/eval/05_01_eval_detection.py"
    assert cfg["access_log"]["script_writes"] is False
    assert cfg["access_log"]["missing_note_action"] == "refuse_and_nonzero_exit"
    assert cfg["access_log"]["columns"] == ["date", "window", "purpose", "command",
                                            "output_path", "reviewer"]


# ---------------------------------------------------------------------------
# ⑫ CLI：缺 note／缺目录／self-check exit=0
# ---------------------------------------------------------------------------
def _run_cli(extra):
    return subprocess.run([sys.executable, SCRIPT] + extra, cwd=_ROOT,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def test_sealed_without_note_is_nonzero(tmp_path):
    r = _run_cli(["--source", "sealed", "--config", CONFIG,
                  "--report-dir", str(tmp_path), "--outputs-dir", str(tmp_path)])
    assert r.returncode != 0
    assert "REFUSED" in r.stdout or "ACCESS_LOG_NOTE_MISSING" in r.stdout


def test_eval_missing_dirs_is_nonzero():
    r = _run_cli(["--source", "val", "--config", CONFIG])
    assert r.returncode != 0
    assert "MISSING_DIRS" in r.stdout


def test_selfcheck_exit_zero():
    r = _run_cli(["--self-check", "--config", CONFIG])
    assert "SELFCHECK_RESULT=PASS" in r.stdout
    assert r.returncode == 0


def test_selfcheck_covers_required_items(cfg, base, capsys):
    """§7-1：self-check 须覆盖记录数断言／逐码分母／三句声明／τ 规则／缺 note 非零退出／确定性双跑。"""
    rc = M.self_check(cfg, base, _args(), lambda *a, **k: None)
    joined = capsys.readouterr().out
    assert rc == 0
    for need in ("record_assert_ood_test_740", "record_assert_ood_val_927",
                 "per_encoding_tpr_denominator_attack_only", "declaration_no_ood_generalization",
                 "declaration_conclusion_only_ood_test", "declaration_no_relax_ood_training",
                 "tau_main_is_star_when_no_override", "sealed_requires_note",
                 "determinism_two_runs"):
        assert need in joined
    assert "SELFCHECK_RESULT=PASS" in joined

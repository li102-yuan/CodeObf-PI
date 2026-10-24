# -*- coding: utf-8 -*-
"""tests/eval/test_05_08_stat_bootstrap.py — W6.9（5.8 统计严谨性）单元与接口测试。

不触 GPU、不读 `data/sealed_test/**`、不重跑任何模型（本窗口只消费既有产物 CSV）。
重型库（torch／transformers／peft）一律不在测试内 import；被测脚本顶层亦不 import。

覆盖（`windows/W6.9_bootstrap.md` §7-1 验收项）：
1. 非预注册组合 ⇒ **非零退出**（C1：b1／b2／b4／`clean_test`／ours:b5@ood）
2. 配对完整性：缺失／多余／`text_sha256` 不一致 ⇒ 各**非零退出**（C3）
3. `n_boot` 定值（C2）
4. `n/a` 指标拦截：`auroc` 只渲染 `n/a`、**不入** bootstrap（C4）
5. `std=0` 退化声明 ＋ 逐 seed 原值（§1.2 交付物 4／C8）
6. `p` 值口径与 CI 方法已声明（C8／§9）
7. 三 seed 极差（`boot_seeds` 42/123/456 各跑一遍）
8. 纯函数：指标点估计、bootstrap 确定性、`Δ=0 ⇒ p=1`、CI 覆盖点估计
9. C9：脚本内不硬编码 `reports/` 输出路径
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
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_08_stat_bootstrap.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "stat_bootstrap.yaml")


def _load():
    spec = importlib.util.spec_from_file_location("eval_05_08", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_05_08"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


@pytest.fixture(scope="module")
def cfg():
    return M.load_config(CONFIG)


def _run(argv):
    """跑一次子进程；返回 `(returncode, stdout+stderr)`。"""
    proc = subprocess.run([sys.executable] + argv, cwd=_ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


def _synth(n_pos, n_neg, ours_correct, base_correct, seed=7):
    ours_rows, base_rows = [], []
    idx = 0
    for lab, cnt in ((1, n_pos), (0, n_neg)):
        for _ in range(cnt):
            idx += 1
            sid = "S-%04d" % idx
            sha = "%064x" % (idx * 7919)
            ours_rows.append({"sample_id": sid, "text_sha256": sha, "pred": lab,
                              "raw_type": "synthetic", "raw": "", "score": "",
                              "latency_ms": "0", "label": lab, "pred_parallel": lab,
                              "tau_main": "0.5", "tau_parallel": "0.5"})
            base_rows.append({"sample_id": sid, "text_sha256": sha,
                              "pred": (1 - lab) if idx % int(1 / max(base_correct, 1e-9)) == 0 else lab,
                              "raw_type": "synthetic", "raw": "", "score": "",
                              "latency_ms": "0"})
    return ours_rows, base_rows


def _mk(tmp_path, cfg, ours_rows, base_rows, comp, seed):
    """按**真实文件名**造一对 CSV（文件名取自配置 ⇒ 不需要额外 CLI 参数）。"""
    ours_dir = tmp_path / "ours"
    base_dir = tmp_path / "base"
    ours_dir.mkdir(exist_ok=True)
    base_dir.mkdir(exist_ok=True)
    ts = cfg["inputs"]["ours_ts"][comp["split"]]
    ours_name = cfg["inputs"]["ours_file"].format(
        regime=comp["regime"], seed=seed, split=comp["split"], ts=ts)
    base_name = cfg["inputs"]["baseline"][comp["baseline"]][comp["split"]]
    o_fields = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score",
                "latency_ms", "label", "pred_parallel", "tau_main", "tau_parallel"]
    b_fields = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score", "latency_ms"]
    M.write_csv_rows(str(ours_dir / ours_name), o_fields, ours_rows)
    M.write_csv_rows(str(base_dir / base_name), b_fields, base_rows)
    return str(ours_dir), str(base_dir)


# ---------------------------------------------------------------------------
# ① C1 只做预注册三组：其余组合一律**非零退出**
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pair,split", [("ours:b1", "id_test"),
                                        ("ours:b2", "id_test"),
                                        ("ours:b4", "id_test"),
                                        ("ours:b3", "clean_test"),
                                        ("ours:b5", "ood_test")])
def test_c1_non_preregistered_pair_refused(pair, split, tmp_path):
    rc, out = _run([SCRIPT, "--pair", pair, "--split", split, "--dry-run",
                    "--ours-outputs", str(tmp_path), "--baseline-outputs", str(tmp_path),
                    "--config", CONFIG])
    assert rc != 0, out
    assert "C1" in out


def test_c1_preregistered_three_accepted(cfg):
    got = M.resolve_comparisons(cfg, [("ours", "b3", "id_test"),
                                      ("ours", "b3", "ood_test"),
                                      ("ours", "b5", "id_test")])
    assert [c["id"] for c in got] == ["ours_vs_b3@id_test",
                                      "ours_vs_b3@ood_test",
                                      "ours_vs_b5@id_test"]


def test_c1_duplicate_pair_refused(cfg):
    with pytest.raises(M.ConstraintError):
        M.resolve_comparisons(cfg, [("ours", "b3", "id_test"), ("ours", "b3", "id_test")])


# ---------------------------------------------------------------------------
# ② C2 `n_boot` 定值
# ---------------------------------------------------------------------------
def test_c2_n_boot_fixed(cfg):
    assert M.assert_n_boot(cfg, None) == 10000
    assert M.assert_n_boot(cfg, 10000) == 10000
    with pytest.raises(M.ConstraintError):
        M.assert_n_boot(cfg, 5000)


def test_c2_n_boot_cli_override_refused(tmp_path):
    rc, out = _run([SCRIPT, "--pair", "ours:b3", "--split", "id_test", "--dry-run",
                    "--ours-outputs", str(tmp_path), "--baseline-outputs", str(tmp_path),
                    "--n-boot", "5000", "--config", CONFIG])
    assert rc != 0, out


# ---------------------------------------------------------------------------
# ③ C3 配对完整性
# ---------------------------------------------------------------------------
def test_c3_pairing_ok_counts(cfg, tmp_path):
    comp = cfg["comparisons"][0]
    o, b = _synth(20, 20, 1.0, 0.5)
    od, bd = _mk(tmp_path, cfg, o, b, comp, 42)
    labels, po, pb, check = M.load_pairs(cfg,
                                         M.resolve_ours_path(cfg, od, comp, 42),
                                         M.resolve_baseline_path(cfg, bd, comp))
    assert check == {"ours_n": 40, "base_n": 40, "n_pairs": 40,
                     "only_ours": 0, "only_base": 0, "sha_mismatch": 0}
    assert len(labels) == len(po) == len(pb) == 40


def test_c3_missing_sample_refused(cfg, tmp_path):
    comp = cfg["comparisons"][0]
    o, b = _synth(20, 20, 1.0, 0.5)
    od, bd = _mk(tmp_path, cfg, o, b[:-3], comp, 42)
    with pytest.raises(M.PairingError) as ei:
        M.load_pairs(cfg, M.resolve_ours_path(cfg, od, comp, 42),
                     M.resolve_baseline_path(cfg, bd, comp))
    assert "配对不完整" in str(ei.value)


def test_c3_extra_sample_refused(cfg, tmp_path):
    comp = cfg["comparisons"][0]
    o, b = _synth(20, 20, 1.0, 0.5)
    od, bd = _mk(tmp_path, cfg, o, b + b[:2], comp, 42)
    with pytest.raises(M.PairingError):
        M.load_pairs(cfg, M.resolve_ours_path(cfg, od, comp, 42),
                     M.resolve_baseline_path(cfg, bd, comp))


def test_c3_sha_mismatch_refused(cfg, tmp_path):
    comp = cfg["comparisons"][0]
    o, b = _synth(20, 20, 1.0, 0.5)
    b2 = [dict(r) for r in b]
    b2[5]["text_sha256"] = "0" * 64
    od, bd = _mk(tmp_path, cfg, o, b2, comp, 42)
    with pytest.raises(M.PairingError) as ei:
        M.load_pairs(cfg, M.resolve_ours_path(cfg, od, comp, 42),
                     M.resolve_baseline_path(cfg, bd, comp))
    assert "text_sha256" in str(ei.value)


def test_c3_no_silent_intersection(cfg, tmp_path):
    """缺失时**不得**静默取交集：异常里须给出两侧差集大小（可机械核验）。"""
    comp = cfg["comparisons"][0]
    o, b = _synth(10, 10, 1.0, 0.5)
    od, bd = _mk(tmp_path, cfg, o, b[:-4], comp, 42)
    with pytest.raises(M.PairingError) as ei:
        M.load_pairs(cfg, M.resolve_ours_path(cfg, od, comp, 42),
                     M.resolve_baseline_path(cfg, bd, comp))
    msg = str(ei.value)
    assert "仅 Ours 有 4 条" in msg


# ---------------------------------------------------------------------------
# ④ C4 `n/a` 指标拦截
# ---------------------------------------------------------------------------
def test_c4_auroc_excluded(cfg):
    computable, blocked = M.resolve_metrics(cfg, "all")
    assert "auroc" not in computable
    assert [m for m, _ in blocked] == ["auroc"]
    assert "span_f1" not in computable


def test_c4_auroc_only_refused(cfg):
    with pytest.raises(M.ConstraintError):
        M.resolve_metrics(cfg, "auroc")


def test_c4_unknown_metric_refused(cfg):
    with pytest.raises(M.ConstraintError):
        M.resolve_metrics(cfg, "recall")


# ---------------------------------------------------------------------------
# ⑤／⑥／⑦ 声明齐备 ＋ 三 seed 极差（合成数据端到端）
# ---------------------------------------------------------------------------
def _build_all(cfg, tmp_path, n=30, ours_c=0.95, base_c=0.70):
    for c in cfg["comparisons"]:
        for seed in cfg["boot_seeds"]:
            o, b = _synth(n, n, ours_c, base_c, seed=seed)
            _mk(tmp_path, cfg, o, b, c, seed)
    return str(tmp_path / "ours"), str(tmp_path / "base")


def test_declared_ci_and_p_value_present(cfg, tmp_path):
    od, bd = _build_all(cfg, tmp_path)
    rep = tmp_path / "rep"
    rep.mkdir()
    argv = [SCRIPT]
    for c in cfg["comparisons"]:
        argv += ["--pair", "%s:%s" % (c["ours"], c["baseline"]), "--split", c["split"]]
    argv += ["--ours-outputs", od, "--baseline-outputs", bd,
             "--config", CONFIG, "--report-dir", str(rep), "--metric", "all"]
    rc, out = _run(argv)
    assert rc == 0, out
    md = [f for f in rep.iterdir() if f.suffix == ".md"]
    assert len(md) == 1
    text = md[0].read_text(encoding="utf-8")
    for _name, kw in M.DECLARATIONS:
        assert kw in text, "缺声明：%s" % kw
    for phrase in M.FORBIDDEN_PHRASES:
        assert phrase not in text


def test_of_record_exactly_three_rows(cfg, tmp_path):
    od, bd = _build_all(cfg, tmp_path)
    rep = tmp_path / "rep2"
    rep.mkdir()
    argv = [SCRIPT]
    for c in cfg["comparisons"]:
        argv += ["--pair", "%s:%s" % (c["ours"], c["baseline"]), "--split", c["split"]]
    argv += ["--ours-outputs", od, "--baseline-outputs", bd,
             "--config", CONFIG, "--report-dir", str(rep)]
    rc, out = _run(argv)
    assert rc == 0, out
    csvf = [f for f in rep.iterdir()
            if f.name.startswith("stat_bootstrap_v1.0_") and f.suffix == ".csv"]
    assert len(csvf) == 1
    with open(str(csvf[0]), "r", encoding="utf-8-sig", newline="") as fh:
        recs = list(csv.DictReader(fh))
    assert len(recs) == 3, "of record 须恒 3 行（无第四行）"
    assert sorted(r["pair_id"] for r in recs) == sorted(c["id"] for c in cfg["comparisons"])
    # 逐 seed 原值（**不得**只给 mean ± std）
    for r in recs:
        for s in cfg["boot_seeds"]:
            assert r["delta_s%s" % s] not in ("", M.TBD)
            assert r["p_s%s" % s] not in ("", M.TBD)
            assert r["ci_lo_s%s" % s] not in ("", M.TBD)
            assert r["ci_hi_s%s" % s] not in ("", M.TBD)
        # 三 seed 极差列
        assert r["p_range"] not in ("", M.TBD)
        assert float(r["p_range"]) >= 0.0


def test_allmetrics_has_auroc_na(cfg, tmp_path):
    od, bd = _build_all(cfg, tmp_path)
    rep = tmp_path / "rep3"
    rep.mkdir()
    argv = [SCRIPT]
    for c in cfg["comparisons"]:
        argv += ["--pair", "%s:%s" % (c["ours"], c["baseline"]), "--split", c["split"]]
    argv += ["--ours-outputs", od, "--baseline-outputs", bd,
             "--config", CONFIG, "--report-dir", str(rep), "--metric", "all"]
    rc, out = _run(argv)
    assert rc == 0, out
    allf = [f for f in rep.iterdir() if f.name.startswith("stat_bootstrap_allmetrics")]
    assert len(allf) == 1
    with open(str(allf[0]), "r", encoding="utf-8-sig", newline="") as fh:
        recs = list(csv.DictReader(fh))
    auroc = [r for r in recs if r["metric"] == "auroc"]
    assert len(auroc) == 3                       # 三组对比各一行
    for r in auroc:
        assert r["p"] == M.NA
        assert r["n_boot"] == ""                 # **未**进入 bootstrap
        assert r["note"]                          # 给了理由


def test_pairing_check_csv_diff_empty(cfg, tmp_path):
    od, bd = _build_all(cfg, tmp_path)
    rep = tmp_path / "rep4"
    rep.mkdir()
    argv = [SCRIPT]
    for c in cfg["comparisons"]:
        argv += ["--pair", "%s:%s" % (c["ours"], c["baseline"]), "--split", c["split"]]
    argv += ["--ours-outputs", od, "--baseline-outputs", bd,
             "--config", CONFIG, "--report-dir", str(rep)]
    rc, out = _run(argv)
    assert rc == 0, out
    pf = [f for f in rep.iterdir() if f.name.startswith("stat_pairing_check")]
    assert len(pf) == 1
    with open(str(pf[0]), "r", encoding="utf-8-sig", newline="") as fh:
        recs = list(csv.DictReader(fh))
    assert len(recs) == 3 * len(cfg["boot_seeds"])
    for r in recs:
        assert r["only_ours"] == "0"
        assert r["only_base"] == "0"
        assert r["sha_mismatch"] == "0"
        assert r["n_pairs"] == r["ours_n"] == r["baseline_n"]


def test_c10_no_overwrite_same_name(cfg, tmp_path):
    """同 ts 重跑 ⇒ 拒绝覆盖（C10 只追加）。"""
    od, bd = _build_all(cfg, tmp_path)
    rep = tmp_path / "rep5"
    rep.mkdir()
    argv = [SCRIPT, "--pair", "ours:b3", "--split", "id_test",
            "--ours-outputs", od, "--baseline-outputs", bd,
            "--config", CONFIG, "--report-dir", str(rep)]
    rc1, out1 = _run(argv)
    assert rc1 == 0, out1
    ts = [f.name[len("stat_bootstrap_v1.0_"):-4]
          for f in rep.iterdir() if f.name.startswith("stat_bootstrap_v1.0_") and f.suffix == ".csv"][0]
    # 用同一 ts 再跑一次：由脚本内部的时间戳不可能重复 ⇒ 改为直接调 API 断言
    with pytest.raises(M.ConstraintError):
        M.assert_not_exists(str(rep / ("stat_bootstrap_v1.0_%s.csv" % ts)), "of record CSV")


# ---------------------------------------------------------------------------
# ⑧ 纯函数
# ---------------------------------------------------------------------------
def test_metric_point_hand_computed():
    # y = [1,1,0,0]，ours pred = [1,1,0,0]（全对），base pred = [1,0,1,0]
    y = [1, 1, 0, 0]
    perfect = [1, 1, 0, 0]
    base = [1, 0, 1, 0]
    assert M.metric_point("f1", y, perfect) == pytest.approx(1.0)
    assert M.metric_point("tpr", y, perfect) == pytest.approx(1.0)
    assert M.metric_point("fpr", y, perfect) == pytest.approx(0.0)
    # base: tp=1 fp=1 fn=1 tn=1
    assert M.metric_point("tpr", y, base) == pytest.approx(0.5)
    assert M.metric_point("fpr", y, base) == pytest.approx(0.5)
    assert M.metric_point("f1", y, base) == pytest.approx(0.5)


def test_metric_point_none_when_undefined():
    # 全负样本 ⇒ TPR 无定义 ⇒ None（**不得**默认兜底为 0）
    assert M.metric_point("tpr", [0, 0], [0, 0]) is None
    # 全正样本 ⇒ FPR 无定义
    assert M.metric_point("fpr", [1, 1], [1, 1]) is None


def test_bootstrap_identical_predictors_gives_p_one():
    y = [1] * 20 + [0] * 20
    p = [1] * 20 + [0] * 20
    r = M.bootstrap_paired("f1", y, p, list(p), 2000, 42)
    assert r["delta"] == pytest.approx(0.0)
    assert r["n_boot"] == 2000
    # Δ 恒为 0 ⇒ P(Δ<=0)=P(Δ>=0)=1 ⇒ p = 2*min = 2 ⇒ 裁剪到 1.0
    assert r["p"] == pytest.approx(1.0)


def test_bootstrap_deterministic_same_seed():
    y = [1] * 20 + [0] * 20
    a = [1] * 18 + [0] * 2 + [1] * 5 + [0] * 15
    b = [1] * 12 + [0] * 8 + [1] * 10 + [0] * 10
    r1 = M.bootstrap_paired("f1", y, a, b, 3000, 42)
    r2 = M.bootstrap_paired("f1", y, a, b, 3000, 42)
    assert r1["p"] == r2["p"]
    assert r1["ci_lo"] == r2["ci_lo"]
    assert r1["ci_hi"] == r2["ci_hi"]


def test_bootstrap_ci_covers_point_and_ordered():
    y = [1] * 30 + [0] * 30
    a = [1] * 28 + [0] * 2 + [0] * 30
    b = [1] * 18 + [0] * 12 + [1] * 12 + [0] * 18
    r = M.bootstrap_paired("f1", y, a, b, 5000, 123)
    assert r["ci_lo"] <= r["ci_hi"]
    assert r["ci_lo"] <= r["delta"] <= r["ci_hi"]
    assert 0.0 < r["delta"] <= 1.0
    assert r["n_nan"] == 0
    assert r["n_gt_1"] == 0


def test_bootstrap_rejects_nan_distribution():
    """重采样出现分母为 0 ⇒ **不得**静默剔除 ⇒ 抛错。"""
    y = [1] + [0] * 9          # 正类仅 1 条，重采样极易全负
    a = [1] + [0] * 9
    b = [0] * 10
    with pytest.raises(M.ConstraintError):
        M.bootstrap_paired("f1", y, a, b, 5000, 42)


@pytest.mark.parametrize("deltas,ps,want", [
    ([0.10, -0.05, 0.02], [0.0, 0.0, 0.0], "符号不一致"),
    ([0.10, 0.12, 0.11], [0.001, 0.400, 0.002], "显著性跨 seed 翻转"),
    ([0.10, 0.12, 0.11], [0.000, 0.010, 0.020], "三 seed 均显著"),
    ([0.10, 0.12, 0.11], [0.300, 0.400, 0.500], "三 seed 均未检出差异"),
])
def test_seed_flag_for(deltas, ps, want):
    assert M.seed_flag_for(deltas, ps) == want


def test_seed_flag_no_silent_pick():
    """回退点：极差过大时**不得**只报其一 ⇒ 必须给出「需谨慎解读」类标注。"""
    flag = M.seed_flag_for([0.26, 0.28, 0.15], [0.0, 0.0, 0.0])
    assert flag == "三 seed 均显著"          # 符号一致且均显著
    flag2 = M.seed_flag_for([0.0, 0.04, -0.19], [1.0, 0.0028, 0.0])
    assert flag2 == "符号不一致"              # 真实 ood TPR 形态 ⇒ 必须标注


def test_allmetrics_has_seed_flag_column(cfg, tmp_path):
    od, bd = _build_all(cfg, tmp_path)
    rep = tmp_path / "rep6"
    rep.mkdir()
    argv = [SCRIPT, "--pair", "ours:b3", "--split", "id_test",
            "--ours-outputs", od, "--baseline-outputs", bd,
            "--config", CONFIG, "--report-dir", str(rep), "--metric", "all"]
    rc, out = _run(argv)
    assert rc == 0, out
    allf = [f for f in rep.iterdir() if f.name.startswith("stat_bootstrap_allmetrics")]
    with open(str(allf[0]), "r", encoding="utf-8-sig", newline="") as fh:
        recs = list(csv.DictReader(fh))
    computed = [r for r in recs if r["metric"] != "auroc"]
    assert computed
    for r in computed:
        assert r["seed_flag"] in ("符号不一致", "显著性跨 seed 翻转",
                                  "三 seed 均显著", "三 seed 均未检出差异")


def test_parse_pairs_mismatch_refused():
    class A:
        pair = ["ours:b3"]
        split = []
    with pytest.raises(M.ConstraintError):
        M.parse_pairs(A())


# ---------------------------------------------------------------------------
# ⑨ 脚本自律（C7／C9）
# ---------------------------------------------------------------------------
def test_c9_no_hardcoded_report_path():
    src = open(SCRIPT, encoding="utf-8").read()
    assert ("reports" + "/eval" + "/") not in src


def test_c7_sealed_read_false_in_config(cfg):
    assert cfg["sealed"]["read"] is False


def test_config_preregistered_values(cfg):
    assert cfg["n_boot"] == 10000
    assert list(cfg["boot_seeds"]) == [42, 123, 456]
    assert len(cfg["comparisons"]) == 3
    assert cfg["metrics"]["primary"] == "f1"
    assert cfg["ci"]["method"] == "percentile"
    assert cfg["ci"]["level"] == 0.95
    assert cfg["multiple_comparison"]["correction"] == "none"


def test_self_check_exits_zero():
    rc, out = _run([SCRIPT, "--self-check", "--config", CONFIG])
    assert "SELFCHECK_RESULT=PASS" in out
    assert rc == 0


def test_newlines_are_lf():
    for p in (SCRIPT, CONFIG):
        raw = open(p, "rb").read()
        assert b"\r" not in raw, "%s 含 CR（须纯 LF）" % p

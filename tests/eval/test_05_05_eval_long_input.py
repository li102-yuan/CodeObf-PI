# -*- coding: utf-8 -*-
"""tests/eval/test_05_05_eval_long_input.py — W6.6（5.5 长输入实验）单元与接口测试。

不触 GPU、不读 `data/sealed_test/**`、不读 frozen 侧 `test.csv`（`W6.6` §3-C5）。
重型库（torch／transformers／peft）一律不在测试内 import —— 被测脚本顶层亦不 import。

覆盖（`windows/W6.6_long_input.md` §3 硬约束 ＋ §7 验收）：

1.  **C2 白名单守卫**：`--max-tokens 2048` 显式给 `b1`／`b2`／`b4` ⇒ **非零退出**（真实子进程）
2.  **C2 自动收窄**：未给 `--detectors` ⇒ 2048 组收窄为 `ours,b3,b5`；512 组放行全部 6 个
3.  **C3 截断侧不变量**：两组同 `left`；不一致即抛
4.  **C4 主口径 512 未被改动**：`03_06.truncate_left` 默认 ＋ b3／b5／qlora 三份 yaml
5.  **C4 内存拷贝**：`deep_set` 不改原对象；2048 落地不写回 yaml（文件内容哈希不变）
6.  **C6 记录数断言**：`csv.DictReader`；`text` 含换行 ⇒ **记录数 ≠ 行数**；不符即抛
7.  **C5 `--source sealed` 缺 `--access-log-note`** ⇒ 非零退出（真实子进程）
8.  **C1 公平性限制**：两句**逐字**在位；**2048 表内不含 B4**（渲染后机械核验）
9.  **C7／C10 静态纪律**：无训练模式切换调用、无硬编码 `reports/`、无按行读；`attn=eager`
10. **τ 口径**：B2 τ = −inf 且禁挪用；Ours 2048 组**不重寻 τ**
11. **长度与分层**：分位数清单齐备、分层层数事前定值、`band_of` 边界
12. **分层度量 = FNR／FPR（不是 F1）**；`unparsed` 逐出分母不兜底；无负类 ⇒ `fpr` = `None`
13. **字段一致性**：`MAIN_FIELDS`／`STRATA_FIELDS` 与配置 `fields.*` 一致
14. **确定性双跑**：同一 ctx 渲染两次 ⇒ **逐字节相同**
15. **`--self-check` exit=0**（真实子进程）
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import os
import subprocess
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_05_eval_long_input.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "long_input.yaml")
BASE_CONFIG = os.path.join(_ROOT, "configs", "eval", "detection.yaml")


def _load():
    spec = importlib.util.spec_from_file_location("eval_05_05", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_05_05"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


@pytest.fixture(scope="module")
def cfg():
    return M.load_config(CONFIG)


@pytest.fixture(scope="module")
def base():
    return M.load_config(BASE_CONFIG)


def _args(**kw):
    base_kw = dict(max_tokens=None, detectors=None, runs_csv=None, only=None, run_dir=None,
                   regime=None, seed=None, source=None, split="in_regime_test",
                   config=CONFIG, report_dir=None, outputs_dir=None, access_log_note=None,
                   length_strata=False, limit=None, self_check=False, dry_run=False, both=False,
                   ts="20990101-000000")
    base_kw.update(kw)
    return types.SimpleNamespace(**base_kw)


def _rec(det_id="b3", group="2048", seed="n/a", labels=(1, 0), preds=(1, 0), raw=(10, 20)):
    labels = list(labels)
    preds = list(preds)
    return {"group": group, "max_tokens": int(group), "detector": det_id,
            "label": det_id.upper(), "run_id": "n/a", "regime": "id", "seed": seed,
            "shard": "x", "source": "val", "tau_track": "n/a", "tau_value": None,
            "best_step": None, "n": len(preds),
            "tp": 1, "fp": 0, "fn": 0, "tn": 0,
            "tpr": 0.5, "fpr": 0.0, "f1": 0.6667, "auroc": None, "n_unparsed": 0,
            "latency": {"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.0},
            "mem_load_peak_mib": 1.0, "mem_infer_peak_mib": 2.0, "cpu_rss_peak_mib": None,
            "n_truncated": 0, "eval_batch_max_size": 1, "n_batches": 4,
            "raw_lens": list(raw), "token_len_eff": list(raw),
            "labels": labels, "preds": preds, "scores": None}


def _ctx(cfg, **kw):
    ctx = {"cfg": cfg, "ts": "20990101-000000", "group": "2048", "groups": ["2048"],
           "source": "val", "shard": "x", "regime": "id", "config_path": CONFIG,
           "runs": [], "runs_note": "桩", "results": {"512": [], "2048": []},
           "strata_rows": [], "invariant_rows": [], "limitations": [],
           "access_log_note": None, "freeze_rows": [], "artifacts": []}
    ctx.update(kw)
    return ctx


def _run_cli(argv):
    p = subprocess.run([sys.executable, SCRIPT] + argv, cwd=_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# 1) C2 白名单守卫（真实子进程）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["b1", "b2", "b4"])
def test_whitelist_guard_rejects_non_2048_detectors(bad):
    rc, out = _run_cli(["--max-tokens", "2048", "--detectors", bad, "--source", "val",
                        "--config", CONFIG])
    assert rc != 0, "2048 组显式给 %s 必须非零退出（`W6.6` §3-C2）" % bad
    assert "白名单守卫失败" in out


def test_whitelist_guard_512_allows_all():
    rc, out = _run_cli(["--max-tokens", "512", "--detectors", "b1,b2,b4", "--source", "val",
                        "--config", CONFIG])
    assert "白名单守卫失败" not in out, "512 组不得触发白名单守卫"
    # 后续 exit≠0 只应来自缺 `--report-dir`（C10），此处不约束退出码


def test_whitelist_allows_pure_function(cfg):
    assert M.whitelist_allows(512, ["b1", "b2", "b3", "b4", "b5", "ours"], cfg)
    assert M.whitelist_allows(2048, ["ours", "b3", "b5"], cfg)
    assert not M.whitelist_allows(2048, ["b4"], cfg)
    assert not M.whitelist_allows(2048, ["b1"], cfg)


def test_detectors_auto_narrow_when_omitted(cfg):
    dets, explicit = M.resolve_detectors(_args(max_tokens=2048), cfg, 2048)
    assert dets == ["ours", "b3", "b5"]
    assert explicit is False
    dets2, _ = M.resolve_detectors(_args(max_tokens=512), cfg, 512)
    assert "b4" in dets2 and len(dets2) == 6


def test_detectors_rejects_unknown():
    with pytest.raises(ValueError):
        M.parse_detectors("ours,b9")


# ---------------------------------------------------------------------------
# 2) C3 截断侧不变量
# ---------------------------------------------------------------------------
def test_truncation_side_invariant_passes(cfg):
    sides = M.assert_truncation_invariant(cfg)
    assert sides["512"] == "left" and sides["2048"] == "left"


def test_truncation_side_invariant_rejects_mixed(cfg):
    bad = copy.deepcopy(cfg)
    bad["truncation"]["side_by_group"]["2048"] = "right"
    with pytest.raises(AssertionError):
        M.assert_truncation_invariant(bad)


# ---------------------------------------------------------------------------
# 3) C4 主口径 512 未被改动 ＋ 内存拷贝不落回
# ---------------------------------------------------------------------------
def test_main_caliber_512_untouched(cfg):
    ok, lines = M.assert_main_caliber_untouched(cfg)
    assert ok, "512 主口径被改动：%s" % lines
    assert len(lines) == 4


def test_deep_set_does_not_mutate_original():
    obj = {"a": {"b": 1}}
    M.deep_set(obj, ["a", "b"], 2048)
    assert obj["a"]["b"] == 2048          # 就地改（调用方已先 deepcopy）
    src = {"a": {"b": 1}}
    M.deep_set(copy.deepcopy(src), ["a", "b"], 2048)
    assert src["a"]["b"] == 1, "deepcopy 后改不得影响原对象"


@pytest.mark.parametrize("rel,key_path", [
    ("configs/baseline/b3_llamaguard1b.yaml", ["input", "max_seq_tokens"]),
    ("configs/baseline/b5_llamaguard8b.yaml", ["input", "max_seq_tokens"]),
    ("configs/train/qlora_ours.yaml", ["preprocess", "max_seq_tokens"]),
])
def test_guard_cfg_override_never_writes_yaml(rel, key_path):
    """2048 组只作**内存拷贝**：跑完配置文件的字节内容必须逐字节不变。"""
    path = os.path.join(_ROOT, rel)
    before = hashlib.md5(open(path, "rb").read()).hexdigest()
    bcfg = M.load_config(path)
    assert bcfg[key_path[0]][key_path[1]] == 512
    M.deep_set(copy.deepcopy(bcfg), key_path, 2048)
    after = hashlib.md5(open(path, "rb").read()).hexdigest()
    assert before == after
    assert bcfg[key_path[0]][key_path[1]] == 512, "内存拷贝不得回写原对象"


# ---------------------------------------------------------------------------
# 4) C6 记录数断言（csv.DictReader；行数 ≠ 记录数）
# ---------------------------------------------------------------------------
def _write_probe(tmp_path, text_with_newlines=True):
    p = os.path.join(str(tmp_path), "probe.csv")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "text", "label"], lineterminator="\n")
        w.writeheader()
        w.writerow({"sample_id": "s1",
                    "text": "a\nb\nc" if text_with_newlines else "plain", "label": "1"})
        w.writerow({"sample_id": "s2", "text": "plain", "label": "0"})
    return p


def test_record_vs_physical_line(tmp_path, base):
    p = _write_probe(tmp_path)
    assert M.W().count_records(p) == 2
    assert M.W().count_physical_lines(p) >= 4


def test_record_count_assert_ok_and_reject(tmp_path, base):
    p = _write_probe(tmp_path)
    assert M.W().assert_record_count(p, 2, base) == 2
    with pytest.raises(AssertionError):
        M.W().assert_record_count(p, 99, base)


# ---------------------------------------------------------------------------
# 5) C5 sealed 双必填
# ---------------------------------------------------------------------------
def test_sealed_requires_access_log_note():
    rc, out = _run_cli(["--max-tokens", "512", "--source", "sealed", "--split",
                        "in_regime_test", "--config", CONFIG])
    assert rc != 0
    assert "REFUSED" in out and "access-log-note" in out


# ---------------------------------------------------------------------------
# 6) C1 公平性限制（逐字 ＋ 2048 表内不含 B4）
# ---------------------------------------------------------------------------
def test_fairness_verbatim_in_report(cfg):
    text = M.render_report(_ctx(cfg))
    assert cfg["fairness"]["verbatim_line_1"] in text
    assert cfg["fairness"]["verbatim_line_2"] in text


def test_b4_absent_in_2048_table(cfg):
    text = M.render_report(_ctx(cfg, results={"512": [], "2048": [_rec("b3"), _rec("ours")]}))
    seg = text.split("## §3 2048 组")[1].split("## §4")[0]
    table = M._table_rows_of(seg)
    assert "B4" not in table
    assert "b4" not in table
    assert "unplug" not in table


def test_report_has_all_required_sections(cfg):
    text = M.render_report(_ctx(cfg))
    for sec in ("## §0 公平性限制声明", "## §1 概述", "## §2 512 组", "## §3 2048 组",
                "## §4 长度分布与分位数", "## §5 长度分层表", "## §6 显存与延迟",
                "## §7 截断侧不变量", "## §8 局限声明", "## §9 ACCESS_LOG",
                "## §10 回执／验收位", "## 附：可直接搬入"):
        assert sec in text, "缺报告节：%s" % sec


def test_report_carries_baseline_citation_notes(cfg):
    text = M.render_report(_ctx(cfg))
    for note in cfg["citation"]["mandatory_notes"]:
        assert note in text


# ---------------------------------------------------------------------------
# 7) C7／C10 静态纪律
# ---------------------------------------------------------------------------
def test_static_discipline():
    src = M._source_text()
    assert (".tr" + "ain(") not in src
    assert (chr(34) + "reports" + chr(47)) not in src
    assert ("read" + "lines()") not in src
    assert ("split" + "lines()") not in src


def test_attn_is_eager(cfg):
    tcfg = M.load_config(os.path.join(_ROOT, cfg["train_config"]))
    assert tcfg["load"]["attn_implementation"] == "eager"


def test_cublas_before_any_torch_import():
    src = M._source_text()
    assert src.index("CUBLAS_WORKSPACE_CONFIG") < src.index("import torch")


# ---------------------------------------------------------------------------
# 8) τ 口径
# ---------------------------------------------------------------------------
def test_b2_tau_is_minus_inf_and_forbidden_to_migrate(cfg):
    d = cfg["detectors"]["b2"]["decision"]
    assert str(d["tau"]) == "-inf"
    assert "forbid_migration" in d
    assert "BASELINE_REPORT.md" in d["tau_source"]


def test_ours_tau_not_rerun_in_2048(cfg):
    assert cfg["detectors"]["ours"]["tau"]["rerun_in_2048"] is False
    assert "best epoch" in cfg["detectors"]["ours"]["tau"]["rule"]


def test_b4_excluded_and_documenting_reason(cfg):
    assert "b4" not in cfg["whitelist"]["2048"]
    assert cfg["detectors"]["b4"]["participates_2048"] is False
    assert "512" in cfg["detectors"]["b4"]["exclude_reason"]


# ---------------------------------------------------------------------------
# 9) 长度与分层
# ---------------------------------------------------------------------------
def test_length_quantiles_declared(cfg):
    for k in ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"):
        assert k in cfg["length"]["quantiles"]


def test_strata_bands_pre_declared(cfg):
    bands = cfg["length"]["strata"]["bands"]
    assert len(bands) >= 3
    assert bands[0]["lo"] == 0
    assert bands[-1]["hi"] is None


def test_band_of_boundaries(cfg):
    bands = cfg["length"]["strata"]["bands"]
    assert M.band_of(0, bands) == "b0_256"
    assert M.band_of(256, bands) == "b256_512"
    assert M.band_of(512, bands) == "b512_1024"
    assert M.band_of(99999, bands) == "b2048_plus"
    assert M.band_of(None, bands) == "n/a"


def test_strata_reports_fnr_and_fpr_not_f1(cfg):
    rec = _rec("b3", labels=[1, 1, 0, 0], preds=[1, 0, 1, 0],
               raw=[10, 20, 30, 40])
    rows = M.build_strata_rows(cfg, "2048", rec, cfg["length"]["strata"]["bands"])
    assert rows, "分层表不得为空"
    for r in rows:
        assert "fnr" in r and "fpr" in r
        assert "f1" not in r
    tot = sum(r["n"] for r in rows)
    assert tot == 4


# ---------------------------------------------------------------------------
# 10) 指标口径
# ---------------------------------------------------------------------------
def test_unparsed_pred_excluded_and_counted():
    m = M.metrics_from_preds([1, 1, 0, 0], [1, None, 0, 0])
    assert m["n"] == 3
    assert m["n_unparsed"] == 1
    assert m["tp"] == 1 and m["fn"] == 0 and m["tn"] == 2


def test_no_negative_class_gives_fpr_none():
    m = M.metrics_from_preds([1, 1], [1, 0])
    assert m["fpr"] is None           # ⇒ 渲染 `n/a`，不是 0.0
    assert m["tpr"] == 0.5


def test_no_positive_class_gives_tpr_and_f1_none():
    m = M.metrics_from_preds([0, 0], [1, 0])
    assert m["tpr"] is None
    assert m["f1"] is None


def test_auroc_none_without_continuous_score():
    m = M.metrics_from_preds([1, 0, 1, 0], [1, 0, 1, 0])
    assert m["auroc"] is None


# ---------------------------------------------------------------------------
# 11) 字段一致性
# ---------------------------------------------------------------------------
def test_fields_match_config(cfg):
    assert set(M.MAIN_FIELDS) == set(cfg["fields"]["main"])
    assert set(M.STRATA_FIELDS) == set(cfg["fields"]["strata"])


# ---------------------------------------------------------------------------
# 12) 确定性双跑
# ---------------------------------------------------------------------------
def test_determinism_double_run_render(cfg):
    ctx1 = _ctx(cfg, results={"512": [_rec("b1", "512")], "2048": [_rec("b3")]})
    ctx2 = _ctx(cfg, results={"512": [_rec("b1", "512")], "2048": [_rec("b3")]})
    h1 = hashlib.md5(M.render_report(ctx1).encode("utf-8")).hexdigest()
    h2 = hashlib.md5(M.render_report(ctx2).encode("utf-8")).hexdigest()
    assert h1 == h2


def test_determinism_double_run_strata(cfg):
    rec1 = _rec("b3", labels=[1, 0], preds=[1, 0], raw=[10, 20])
    rec2 = _rec("b3", labels=[1, 0], preds=[1, 0], raw=[10, 20])
    a = M.build_strata_rows(cfg, "2048", rec1, cfg["length"]["strata"]["bands"])
    b = M.build_strata_rows(cfg, "2048", rec2, cfg["length"]["strata"]["bands"])
    assert a == b


# ---------------------------------------------------------------------------
# 13) `--self-check` exit=0
# ---------------------------------------------------------------------------
def test_self_check_exit_zero():
    rc, out = _run_cli(["--self-check", "--config", CONFIG])
    assert rc == 0, out[-2000:]
    assert "SELFCHECK_RESULT=PASS" in out


# ---------------------------------------------------------------------------
# 14) 冻结面复核（纯 Python md5，不依赖外部 md5sum）
# ---------------------------------------------------------------------------
def test_md5sum_check_frozen_manifest():
    rc, msg = M.md5sum_check(os.path.join(_ROOT, "data", "frozen", "CodeObf-PI_v1.0",
                                          "md5sums.txt"))
    assert rc == 0, msg
    assert "16/16 OK" in msg


# ---------------------------------------------------------------------------
# `T3`（`D-111` 裁定二／五）—— 截断 Limitation 段的**四要素**
# ---------------------------------------------------------------------------
def _lim_ctx(cfg, rates=(("检测侧截断率", "512", 676, 1583, "0.4270"),
                         ("ASR 链路截断率", "none/id", 546, 3168, "0.1723"))):
    return {
        "cfg": cfg,
        "trunc_caliber_rows": [{"caliber": a, "group": b, "num": c, "den": d,
                                "rate": e, "note": "n"} for a, b, c, d, e in rates],
        "counterevidence_rows": [["`Ours` 逐 seed `F1`", "0.9975 / 1.0000 / 0.9981",
                                  "0.9949 / 1.0000 / 0.9956", "格局不变"]],
    }


def test_trunc_calibers_two_and_denominators_distinct():
    assert len(M.TRUNC_CALIBERS) == 2
    names = [c["name"] for c in M.TRUNC_CALIBERS]
    assert "检测侧截断率" in names and "ASR 链路截断率" in names
    assert M.TRUNC_CALIBERS[0]["denominator"] != M.TRUNC_CALIBERS[1]["denominator"]


def test_limitation_section_has_four_elements(cfg):
    lim = M.render_limitation_section(_lim_ctx(cfg))
    assert all(c["name"] in lim for c in M.TRUNC_CALIBERS)      # ① 两口径分列
    assert "系统性低估" in lim                                    # ② 影响方向
    assert ("512 → 2048" in lim) and ("反证" in lim)              # ③ 反证
    assert "2048 敏感性分析" in lim                               # ④ 引用 2048 敏感性分析


def test_limitation_does_not_merge_two_rates(cfg):
    """两个比率**各自带自己的分母**出现在同一表内 ⇒ 不得合成单一「截断率」。"""
    lim = M.render_limitation_section(_lim_ctx(cfg))
    assert "0.4270" in lim and "0.1723" in lim
    assert "1583" in lim and "3168" in lim


def test_report_embeds_limitation_section(cfg):
    assert "### `T3` · 截断率 Limitation" in M.render_report(_ctx(cfg))

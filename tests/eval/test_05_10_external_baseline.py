# -*- coding: utf-8 -*-
"""tests/eval/test_05_10_external_baseline.py — W6.12（外部基线 B6）单元与接口测试。

**不触 GPU／不读 `data/sealed_test/**`／不读 frozen 侧 `test.csv`**（`windows/W6.12_external_baseline.md`
§6-C6）。重型库（torch／transformers）一律不 import —— 被测脚本顶层亦不 import（`AGENTS.md` §3）。
组批构造用**桩 torch**测（只验 `[CLS]+content+[SEP]`／右 pad／attn mask 的接线，不验数值）。

覆盖（`windows/W6.12_external_baseline.md` §8 验收项 ＋ §6 硬约束）：
1. 事前定值（配置件）：人群 1583／740／64 与基线配置一致；kind=judgment；标签映射冻结
2. C3 判定：`pred_from_logits` 恒 0/1（并列取 0）；`--kind score` 显式拒绝（非零退出）
3. 输入编码（桩 tokenizer）：原文直通／左截断保尾部（单源）／往返硬钳（计数披露）／空 ⇒ unparsed
4. 组批接线：`[CLS]+content+[SEP]`、右 pad、attn mask；特殊 token id 取自 tokenizer
5. C6/C8 分片解析与 ACCESS 守卫：val 拒 split／sealed 须 split／g3 拒 regime／缺 note 非零退出
6. C8 记录数口径：`csv.DictReader`（字段内含换行 ⇒ 行数 ≫ 记录数）；写端 LF、拒覆盖
7. 指标单源：`03_07.compute_binary_metrics`（无 scores ⇒ 无 AUROC 键）；`n/a` 非 0 非 `<TBD>`
8. CLI：`--self-check` exit=0；`--dry-run` n=1583／740／64；`ARG_REQUIRED`／`MODEL_MISSING`
9. 静态纪律：无 `.train(`、不硬编码报告目录、CUBLAS 先于 `import torch`、顶层不 import 重型库
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import os
import re
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "eval", "05_10_eval_external_baseline.py")
CONFIG = os.path.join(_ROOT, "configs", "eval", "external_baseline.yaml")
BASE_CONFIG = os.path.join(_ROOT, "configs", "eval", "detection.yaml")
PYTHON = sys.executable
NOTE = "测试用 ACCESS_LOG 行摘要占位（≥20 字符；正式跑前替换为已登记行）"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("eval_05_10", SCRIPT)


@pytest.fixture(scope="module")
def cfg(mod):
    return mod.load_config(CONFIG)


@pytest.fixture(scope="module")
def base(mod):
    return mod.load_config(BASE_CONFIG)


@pytest.fixture(scope="module")
def ev(mod, cfg):
    return mod.load_module("ev_05_01_t", cfg["modules"]["eval_engine"])


@pytest.fixture(scope="module")
def pp(mod, cfg):
    return mod.load_module("pp_03_06_t", cfg["modules"]["preprocess"])


@pytest.fixture(scope="module")
def mp(mod, cfg):
    return mod.load_module("map_03_07_t", cfg["modules"]["mapping"])


# ---------------------------------------------------------------------------
# 1 事前定值（先写后跑；跑后不得改）
# ---------------------------------------------------------------------------
def test_population_matches_base_assertions(cfg, base):
    """三人群定值必须与基线配置的 `record_assertions` 逐字一致（不同源即失败）。"""
    exp = {s["assertion_key"]: int(s["expect_records"]) for s in cfg["population"]["shards"]}
    assert exp == {"regime_id_test": 1583, "regime_ood_test": 740, "g3_hard_negative_test": 64}
    for key, n in exp.items():
        assert int(base["record_assertions"][key]) == n


def test_population_explicit_no_scan(cfg):
    assert [s["split"] for s in cfg["population"]["shards"]] == [
        "in_regime_test", "in_regime_test", "g3_hard_negative_test"]
    assert cfg["population"]["shards"][2]["regime"] is None      # G3 无体制轴


def test_kind_and_label_mapping_frozen(cfg):
    assert cfg["mapping"]["kind"] == "judgment"
    assert cfg["mapping"]["kind_rejected"] == "score"
    assert cfg["mapping"]["label_0"] == "benign" and cfg["mapping"]["label_1"] == "malicious"
    assert cfg["mapping"]["auroc"] == "n/a" and cfg["mapping"]["span_f1"] == "n/a"
    assert cfg["mapping"]["score"] is None                       # 标签型 ⇒ 无连续 score
    assert "id2label" in cfg["mapping"]["label_source"]          # 占位映射禁用已留痕


def test_load_profile_frozen(cfg):
    ld = cfg["load"]
    assert ld["mode"] == "fp32_native" and ld["dtype"] == "float32"
    assert ld["device"] == "cuda" and ld["attn_implementation"] == "eager"
    assert "eager" not in ld["forbidden_attn"]
    assert ld["determinism"]["cublas_workspace_config"] == ":4096:8"
    assert ld["determinism"]["allow_tf32"] is False
    assert "不同量纲" in ld["comparability_note"]                 # 延迟/显存只并列不相减


def test_input_truncation_frozen(cfg):
    inp = cfg["input"]
    assert inp["column"] == "text" and inp["pass_through"] is True
    assert int(inp["max_content_tokens"]) == 512 and inp["truncation_side"] == "left"
    assert inp["roundtrip_guard"] is True and inp["smoke_equivalence"] is True
    assert "平方" in inp["quadratic_memory_note"]                 # 长输入未评的根因留痕


def test_latency_batch_frozen(cfg):
    assert int(cfg["latency"]["batch_size"]) == 32               # 不得按结果调整
    assert cfg["latency"]["excludes_model_load"] is True
    assert cfg["latency"]["excludes_csv_io"] is True
    assert cfg["latency"]["stats"] == ["mean_ms", "median_ms", "p95_ms"]


def test_outputs_schema_frozen(cfg):
    out = cfg["outputs"]
    assert out["per_sample_fields"][:6] == ["sample_id", "text_sha256", "pred",
                                            "raw_type", "raw", "score"]
    assert out["per_sample_fields"][-4:] == ["label", "n_content_tokens",
                                             "truncated", "unparsed"]   # 只末尾追加
    assert out["outputs_subdir"] == "outputs_ext"                # 不得进 outputs_v2/
    assert out["newline"] == "lf" and out["append_only"] is True
    assert "tpr" in out["metric_fields"] and "fpr" in out["metric_fields"]   # C5 并列


def test_report_must_include_disclosures(cfg):
    decls = cfg["report"]["must_include"]
    assert len(decls) >= 8
    joined = "\n".join(decls)
    for needle in ("外部基线，非本项目训练", "LABEL_0=benign", "n/a", "描述性并列",
                   "未复现 BAGEL", "照实报告", "未评"):
        assert needle in joined


def test_access_guard_config(cfg):
    acc = cfg["access"]
    assert acc["sealed_requires_note"] is True and int(acc["note_min_chars"]) == 20
    assert acc["access_log_columns"] == ["date", "window", "purpose", "command",
                                         "output_path", "reviewer"]
    assert acc["script_writes"] is False                          # 登记由人手工完成


# ---------------------------------------------------------------------------
# 2 判定映射（PREREG_v1.10 §1.2；C3）
# ---------------------------------------------------------------------------
def test_label_argmax_and_tie(mod):
    assert mod.pred_from_logits([-5.1573, 2.5018]) == (1, 1)      # malicious
    assert mod.pred_from_logits([0.0034, -7.8931]) == (0, 0)      # benign
    assert mod.pred_from_logits([1.0, 1.0]) == (0, 0)             # 并列取 0（确定性）
    assert mod.pred_from_logits([0.0, 0.0]) == (0, 0)


def test_label_names_constants(mod):
    assert mod.LABEL_0_NAME == "benign" and mod.LABEL_1_NAME == "malicious"
    assert mod.DETECTOR == "b6" and mod.NA == "n/a"


# ---------------------------------------------------------------------------
# 3 输入编码（桩 tokenizer；不加载真实权重）
# ---------------------------------------------------------------------------
_ALPHA = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _PosTok(object):
    """桩口径一：每字符一个 token，id = 1000 + 位置序号（可验「单源」）。"""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1000, 1000 + len(text)))}


class _StableTok(object):
    """桩口径二：id 由字符本身决定（与位置无关）⇒ 可验「保尾部」。"""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1000 + _ALPHA.index(ch) for ch in text]}


class _InflTok(object):
    """桩口径三：每字符 2 token ⇒ 模拟 truncate_left 的 decode→re-encode 往返膨胀。"""
    def __call__(self, text, add_special_tokens=False):
        out = []
        for ch in text:
            i = _ALPHA.index(ch)
            out.extend([1000 + i, 1500 + i])
        return {"input_ids": out}


class _TailPP(object):
    @staticmethod
    def truncate_left(text, tok, max_tokens):
        n = int(max_tokens)
        return text[-n:] if len(text) > n else text


def test_encode_passthrough_short(mod):
    ids, n_raw, fired = mod.encode_content_ids(_PosTok(), _TailPP, "x" * 10, 512)
    assert ids == list(range(1000, 1010)) and n_raw == 10 and not fired


def test_encode_truncate_single_source_keeps_tail(mod):
    """n_raw 取**首次**编码（原文 token 数）；左截断后保留**尾部** 512。"""
    orig = _ALPHA * 9 + _ALPHA[:42]                      # 600 字符
    assert len(orig) == 600
    ids, n_raw, fired = mod.encode_content_ids(_StableTok(), _TailPP, orig, 512)
    assert n_raw == 600 and not fired and len(ids) == 512
    assert ids == [1000 + _ALPHA.index(orig[88 + i]) for i in range(512)]
    assert ids[0] == 1000 + _ALPHA.index(orig[88])       # 非头部 ⇒ 保尾部


def test_encode_roundtrip_guard_fires(mod):
    """往返仍 >512 ⇒ token 级硬钳（同向保尾）、`fired=True` 计数披露（非静默兜底）。"""
    orig = _ALPHA * 9 + _ALPHA[:42]
    ids, n_raw, fired = mod.encode_content_ids(_InflTok(), _TailPP, orig[:300], 512)
    assert n_raw == 600 and fired and len(ids) == 512
    expect = []
    for k in range(44, 300):                             # 600 token ⇒ 钳掉前 88
        i = _ALPHA.index(orig[k])
        expect.extend([1000 + i, 1500 + i])
    assert ids == expect


def test_encode_empty_is_unparsed_not_safe(mod):
    """空 text ⇒ `([], 0, False)` ⇒ 上层记 unparsed；**不得**兜底为 0/1（C4）。"""
    assert mod.encode_content_ids(_PosTok(), _TailPP, "", 512) == ([], 0, False)


# ---------------------------------------------------------------------------
# 4 组批接线（桩 torch；[CLS]+content+[SEP]、右 pad、无 token_type_ids）
# ---------------------------------------------------------------------------
class _StubTorch(object):
    long = "long"

    @staticmethod
    def tensor(x, dtype=None, device=None):
        return [list(r) for r in x]


class _StubTok(object):
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0


def test_build_batch_special_tokens_and_padding(mod):
    ii, am = mod.build_batch_tensors(_StubTorch, [[10, 11], [12]], _StubTok, "cpu")
    assert ii == [[101, 10, 11, 102], [101, 12, 102, 0]]          # 右 pad
    assert am == [[1, 1, 1, 1], [1, 1, 1, 0]]                     # pad 位 mask=0


def test_build_batch_special_ids_from_tokenizer(mod):
    """特殊 token id 必须取自 tokenizer（换一组 id 行为随之改变 ⇒ 非硬编码）。"""
    class T(_StubTok):
        cls_token_id, sep_token_id, pad_token_id = 1, 2, 0
    ii, _am = mod.build_batch_tensors(_StubTorch, [[7]], T, "cpu")
    assert ii == [[1, 7, 2]]


def test_forward_kwargs_exclude_token_type_ids():
    src = open(SCRIPT, encoding="utf-8").read()
    assert "model(input_ids=ii, attention_mask=am)" in src        # 不传 token_type_ids
    assert "type_vocab_size=0" in src                             # DeBERTA-v2 口径留痕


# ---------------------------------------------------------------------------
# 5 分片解析与 ACCESS 守卫（C6）
# ---------------------------------------------------------------------------
def test_resolve_shard_happy_paths(mod, cfg, base, ev):
    p, n, shard, reg = mod.resolve_shard(ev, base, cfg, "sealed", "id", "in_regime_test")
    assert n == 1583 and shard == "id_test" and reg == "id"
    p, n, shard, reg = mod.resolve_shard(ev, base, cfg, "sealed", "ood", "in_regime_test")
    assert n == 740 and shard == "ood_test"
    p, n, shard, reg = mod.resolve_shard(ev, base, cfg, "sealed", None, "g3_hard_negative_test")
    assert n == 64 and shard == "g3_hard_negative_test" and reg == ""
    p, n, shard, reg = mod.resolve_shard(ev, base, cfg, "val", "id", None)
    assert n == 1584 and shard == "id_val"


def test_resolve_shard_guards(mod, cfg, base, ev):
    with pytest.raises(ValueError):
        mod.resolve_shard(ev, base, cfg, "sealed", None, "in_regime_test")   # 缺 regime
    with pytest.raises(ValueError):
        mod.resolve_shard(ev, base, cfg, "sealed", "id", None)               # 缺 split
    with pytest.raises(ValueError):
        mod.resolve_shard(ev, base, cfg, "sealed", "id", "g3_hard_negative_test")  # G3 拒 regime
    with pytest.raises(ValueError):
        mod.resolve_shard(ev, base, cfg, "val", "id", "in_regime_test")      # val 拒 split
    with pytest.raises(ValueError):
        mod.resolve_shard(ev, base, cfg, "sealed", "id", "nope")             # 未知 split
    with pytest.raises(ValueError):
        mod.resolve_shard(ev, base, cfg, "train", "id", None)                # 未知 source


def test_require_access_note(mod, cfg):
    with pytest.raises(SystemExit):
        mod.require_access_note(cfg, "sealed", "")
    with pytest.raises(SystemExit):
        mod.require_access_note(cfg, "sealed", "短")                        # <20 字符
    assert mod.require_access_note(cfg, "sealed", NOTE) == NOTE
    assert mod.require_access_note(cfg, "val", "") is None                  # val 不需 note


# ---------------------------------------------------------------------------
# 6 记录数口径（C8）与 C12 只追加
# ---------------------------------------------------------------------------
def test_count_records_is_dictreader_not_lines(mod, tmp_path):
    """`text` 字段内含换行 ⇒ 物理行数 ≫ 记录数；只许 DictReader 计数（C8）。"""
    rows = [{"a": "line1\nline2\nline3", "b": "1"}, {"a": "x", "b": "2"}]
    path = str(tmp_path / "embedded.csv")
    mod.write_csv_rows(path, ["a", "b"], rows)
    assert mod.count_records(path) == 2
    with open(path, "r", encoding="utf-8", newline="") as fh:
        assert len(fh.readlines()) == 5                   # 1 表头 + 3 + 1 = 5 行 ≠ 2 记录


def test_write_csv_lf_and_refuses_overwrite(mod, cfg, tmp_path):
    rows = [{"sample_id": "S1", "text_sha256": "0" * 64, "pred": 1, "raw_type": "logits",
             "raw": '{"logits": [1.0, 2.0]}', "score": "", "latency_ms": "1.2345",
             "label": 1, "n_content_tokens": 5, "truncated": 0, "unparsed": 0}]
    path = str(tmp_path / "ps.csv")
    mod.write_csv_rows(path, cfg["outputs"]["per_sample_fields"], rows)
    raw = open(path, "rb").read()
    assert raw.count(b"\r") == 0                               # 纯 LF
    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 1 and got[0]["pred"] == "1" and got[0]["unparsed"] == "0"
    with pytest.raises(FileExistsError):
        mod.write_csv_rows(path, cfg["outputs"]["per_sample_fields"], rows)


def test_write_csv_fills_missing_fields_empty(mod, cfg, tmp_path):
    path = str(tmp_path / "ps2.csv")
    mod.write_csv_rows(path, cfg["outputs"]["per_sample_fields"], [{"sample_id": "S2"}])
    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert got[0]["sample_id"] == "S2" and got[0]["pred"] == ""


# ---------------------------------------------------------------------------
# 7 指标口径（单源 03_07；n/a 非 0）
# ---------------------------------------------------------------------------
def test_metrics_no_auroc_without_scores(mp):
    m = mp.compute_binary_metrics([1, 1, 0, 0], [1, 0, 0, 1], scores=None)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert abs(m["tpr"] - 0.5) < 1e-12 and abs(m["fpr"] - 0.5) < 1e-12
    assert "auroc" not in m                                    # 标签型 ⇒ 无 AUROC 键


def test_metrics_all_negative_tpr_na(mp):
    m = mp.compute_binary_metrics([0, 0, 0], [0, 0, 1], scores=None)
    assert m["tpr"] is None and abs(m["fpr"] - 1.0 / 3.0) < 1e-12   # G3 全良性 ⇒ TPR n/a


def test_fmt_metric_na_not_zero(mod):
    assert mod.fmt_metric(None) == "n/a"
    assert mod.fmt_metric(0.0) == "0.0000"
    assert mod.fmt_metric(0.5) == "0.5000"


def test_report_na_cells_not_zero(mod, cfg):
    ctx = {
        "ts": "20260101-000000", "config_path": "configs/eval/external_baseline.yaml",
        "source": "sealed", "shards": ["id_test"], "kind": "judgment",
        "access_note": NOTE,
        "rows": [{"shard": "id_test", "n": 4, "n_pos": 2, "n_neg": 2, "tp": 1, "fp": 1,
                  "fn": 1, "tn": 1, "tpr": 0.5, "fpr": 0.5, "f1": 0.5,
                  "latency_mean_ms": 1.0, "latency_median_ms": 1.0, "latency_p95_ms": 1.0,
                  "mem_load_peak_mib": 100.0, "mem_infer_peak_mib": 200.0,
                  "n_truncated": 1, "roundtrip_guard_fired": 1, "n_unparsed": 0}],
        "paths": {"report": "r.md", "metric_csv": "m.csv", "per_sample": ["a.csv"],
                  "log": "l.log"},
    }
    md = mod.render_report(cfg, ctx)
    for decl in cfg["report"]["must_include"]:
        assert decl in md
    assert "| `id_test` | 4 | 2 | 2 | 1 | 1 | 1 | 1 | **0.5000** | **0.5000** | 0.5000 | n/a | n/a |" in md


# ---------------------------------------------------------------------------
# 8 命令行接口
# ---------------------------------------------------------------------------
def _run(*argv):
    return subprocess.run([PYTHON, SCRIPT] + list(argv), cwd=_ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def test_self_check_passes():
    proc = _run("--self-check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SELFCHECK_RESULT=PASS" in proc.stdout


def test_dry_run_counts_three_shards(tmp_path):
    model = str(tmp_path / "b6_weights")
    os.makedirs(model, exist_ok=True)
    for extra, n in ((["--split", "in_regime_test", "--regime", "id"], 1583),
                     (["--split", "in_regime_test", "--regime", "ood"], 740),
                     (["--split", "g3_hard_negative_test"], 64)):
        proc = _run("--model", model, "--kind", "judgment", "--source", "sealed",
                    "--access-log-note", NOTE,
                    "--report-dir", "reports/eval", "--outputs-dir",
                    "reports/eval/outputs_ext", "--dry-run", *extra)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "DRYRUN_RESULT=PASS n=%d" % n in proc.stdout


def test_dry_run_limit_overrides_count(tmp_path):
    model = str(tmp_path / "b6_weights")
    os.makedirs(model, exist_ok=True)
    proc = _run("--model", model, "--kind", "judgment", "--source", "sealed",
                "--split", "in_regime_test", "--regime", "id",
                "--access-log-note", NOTE, "--report-dir", "reports/eval",
                "--outputs-dir", "reports/eval/outputs_ext", "--dry-run", "--limit", "7")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRYRUN_RESULT=PASS n=7" in proc.stdout


def test_missing_required_args_rejected():
    for argv in (["--kind", "judgment", "--source", "sealed"],
                 ["--model", "x", "--source", "sealed"],
                 ["--model", "x", "--kind", "judgment"]):
        proc = _run(*argv)
        assert proc.returncode != 0
        assert "ARG_REQUIRED" in proc.stdout + proc.stderr


def test_score_kind_rejected(tmp_path):
    model = str(tmp_path / "b6_weights")
    os.makedirs(model, exist_ok=True)
    proc = _run("--model", model, "--kind", "score", "--source", "sealed",
                "--split", "in_regime_test", "--regime", "id",
                "--access-log-note", NOTE, "--report-dir", "reports/eval",
                "--outputs-dir", "reports/eval/outputs_ext", "--dry-run")
    assert proc.returncode != 0
    assert "KIND_NOT_PREREGISTERED" in proc.stdout + proc.stderr


def test_model_missing_rejected():
    proc = _run("--model", "Z:/no/such/b6/dir", "--kind", "judgment", "--source", "sealed",
                "--split", "in_regime_test", "--regime", "id",
                "--access-log-note", NOTE, "--report-dir", "reports/eval",
                "--outputs-dir", "reports/eval/outputs_ext", "--dry-run")
    assert proc.returncode != 0
    assert "MODEL_MISSING" in proc.stdout + proc.stderr


def test_sealed_without_note_rejected(tmp_path):
    model = str(tmp_path / "b6_weights")
    os.makedirs(model, exist_ok=True)
    proc = _run("--model", model, "--kind", "judgment", "--source", "sealed",
                "--split", "in_regime_test", "--regime", "id",
                "--report-dir", "reports/eval", "--outputs-dir",
                "reports/eval/outputs_ext", "--dry-run")
    assert proc.returncode != 0
    assert "ACCESS_LOG_GUARD_FAIL" in proc.stdout + proc.stderr


def test_last_line_contract_in_src():
    src = open(SCRIPT, encoding="utf-8").read()
    assert "REPORT_MD=" in src and "SELFCHECK_RESULT=" in src and "DRYRUN_RESULT=" in src


# ---------------------------------------------------------------------------
# 9 静态纪律
# ---------------------------------------------------------------------------
def test_no_train_call_in_script():
    src = open(SCRIPT, encoding="utf-8").read()
    assert re.search(r"[A-Za-z_][A-Za-z0-9_]*\.train\(", src) is None


def test_no_hardcoded_reports_dir():
    src = open(SCRIPT, encoding="utf-8").read()
    code = "\n".join(ln for ln in src.splitlines()
                     if "no_hardcoded_reports_dir" not in ln)
    assert '"reports/' not in code and "'reports/" not in code


def test_cublas_before_torch_import():
    src = open(SCRIPT, encoding="utf-8").read()
    m = re.search(r"(?m)^[ \t]*import torch\b", src)
    assert m is not None
    assert 'os.environ["CUBLAS_WORKSPACE_CONFIG"]' in src
    assert src.find('os.environ["CUBLAS_WORKSPACE_CONFIG"]') < m.start()


def test_top_level_no_heavy_imports():
    """顶层只许 stdlib：torch／transformers／yaml 一律函数体内惰性 import（AGENTS.md §3）。"""
    tree = ast.parse(open(SCRIPT, encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    for heavy in ("torch", "transformers", "yaml", "sklearn"):
        assert heavy not in top


def test_outputs_ext_not_v2_in_src():
    src = open(SCRIPT, encoding="utf-8").read()
    assert "outputs_v2" not in src
    assert '"b6_%s_v1.0_%s.csv"' in src

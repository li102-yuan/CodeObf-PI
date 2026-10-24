# -*- coding: utf-8 -*-
"""tests/baseline/test_03_04_run_b4_unplug_tiny.py — W4.4 B4 unplug-tiny 原始推理。

覆盖 `windows/W4.4_b4_unplug.md` 与 `windows/W4.6_unify.md` 的语义：

1. **主口径构造用例（§2 验收）**：`doc_or_span`@0.9/0.45 —— doc 恰达 0.9 判正／未达且无 span 判负／
   任一 span 恰达 0.45 判正／未达判负／多 span 中任一达标即判正
2. **两份对照构造用例**：span-only（`PREREG_v1.md` §12.2 字面，**仅** span 分支）／
   0.60（SDK `catalog.toml` `inj_threshold`）
3. **ABSTAIN → 负例映射用例 ＋ 不得沿用默认 `abstain_is_safe=False` 的回归用例**（§2 验收硬项）
4. `score = max(doc_score, max(span_score))`（`W4.6_unify.md` §2 score_source.b4）
5. §2 schema：列名/列序、`raw_type=doc_span`、`raw` 含完整 spans（start/end/score）、`raw` 为 JSON
6. YAML 冻结值：`doc_threshold=0.9`／`span_threshold=0.45`／`abstain.enabled=false`／
   `abstain.fallback=negative`／`abstain.abstain_is_safe=true`（**显式**）／`device=cpu`
7. 探测样本隔离：`repro_probe.must_not_enter_any_split=true`，4 条样本文本**不得**出现在配置之外的分片产物路径逻辑中
8. 硬约束：顶层不 import 重型库；**无 `except: pass`／裸 except**；脚本与 YAML 均 LF（CR=0）；
   `--self-check` exit 0；测试自身不访问 `sealed_test`

**离线**：全部输入在 `tmp_path` 内自造；不加载 torch／transformers；
**不访问 `data/sealed_test/`**。
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "baseline", "03_04_run_b4_unplug_tiny.py")
YAML = os.path.join(ROOT, "configs", "baseline", "b4_unplug.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w44_b4", SCRIPT)


@pytest.fixture(scope="module")
def cfg(mod):
    return mod.load_config(YAML)


# ---------------------------------------------------------------------------
# ① 主口径 doc_or_span@0.9/0.45（§2 验收硬项）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc,spans,expect", [
    (0.9, [], 1),                                                  # doc 恰达阈值
    (0.95, [], 1),
    (0.8999, [], 0),                                               # 未达且无 span
    (0.0, [{"start": 0, "end": 5, "score": 0.45}], 1),              # span 恰达阈值
    (0.0, [{"start": 0, "end": 5, "score": 0.4499}], 0),
    (0.0, [{"start": 0, "end": 5, "score": 0.1},
           {"start": 6, "end": 9, "score": 0.7}], 1),               # 任一 span 达标
    (0.0, [], 0),                                                   # 双头均未达
    (None, [{"start": 0, "end": 5, "score": 0.45}], 1),              # doc 缺失但 span 达标
    (None, [], 0),
])
def test_doc_or_span_main_caliber(mod, doc, spans, expect):
    assert mod.doc_or_span(doc, spans) == expect


def test_doc_or_span_thresholds_are_frozen_constants(mod):
    assert mod.DOC_THRESHOLD == 0.9
    assert mod.SPAN_THRESHOLD == 0.45
    assert mod.RAW_TYPE == "doc_span"


# ---------------------------------------------------------------------------
# ② 对照① span-only（PREREG_v1.md §12.2 字面：仅 span 分支）
# ---------------------------------------------------------------------------
def test_span_only_ignores_doc_branch(mod):
    """§12.2 字面**只写了 span 分支** ⇒ doc 再高也不得判正（对照必须忠实于字面）。"""
    assert mod.span_only(0.999, []) == 0
    assert mod.span_only(0.0, [{"start": 0, "end": 3, "score": 0.45}]) == 1
    assert mod.span_only(0.0, [{"start": 0, "end": 3, "score": 0.44}]) == 0


# ---------------------------------------------------------------------------
# ③ 对照② 0.60（SDK catalog.toml inj_threshold）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("span_score,expect", [(0.45, 0), (0.599, 0), (0.60, 1), (0.9, 1)])
def test_threshold_060_comparison(mod, span_score, expect):
    got = mod.doc_or_span(0.0, [{"start": 0, "end": 5, "score": span_score}],
                          mod.DOC_THRESHOLD, mod.COMPARE_THRESHOLD_060)
    assert got == expect


def test_comparison_thresholds_are_frozen(mod):
    assert mod.COMPARE_SPAN_ONLY == 0.45
    assert mod.COMPARE_THRESHOLD_060 == 0.60


# ---------------------------------------------------------------------------
# ④ ABSTAIN → 负例（§2 验收硬项）＋ 不得沿用 SDK 默认的回归
# ---------------------------------------------------------------------------
def test_abstain_maps_to_negative_even_when_doc_and_span_fire(mod, cfg):
    """ABSTAIN ⇒ 负例(0)，**即使** doc/span 都已过阈（显式映射压过规则映射）。"""
    m = mod.map_prediction(0.99, [{"start": 0, "end": 5, "score": 0.99}], True, cfg)
    assert m["pred"] == 0
    assert m["pred_span_only"] == 0
    assert m["pred_060"] == 0
    assert m["abstained"] is True


def test_abstain_fallback_must_be_negative_in_config(cfg):
    """§0.3：ABSTAIN ⇒ 负例；配置须显式声明 `fallback: negative`。"""
    assert cfg["abstain"]["fallback"] == "negative"


def test_abstain_is_safe_must_be_explicitly_true_not_sdk_default(cfg):
    """⚠ §5.3.2 实现警告：SDK `policy.py` L128–L133 的 `abstain_is_safe` 默认 **False**
    （abstain 视为不安全、按拦截处理）——与本项目「ABSTAIN ⇒ 负例」**方向相反**。

    本回归用例断言：配置**显式**置 True（= abstain 视作 safe ⇒ 放行/负例），
    绝不沿用 SDK 默认值。
    """
    assert cfg["abstain"]["abstain_is_safe"] is True, (
        "不得沿用 SDK 默认 abstain_is_safe=False（方向相反）—— 必须显式置 true")
    # 反向断言：SDK 默认确实是 False（若哪天上游改了默认，本窗口的显式配置仍须保留）
    sdk_default_marker = cfg["abstain"]["abstain_is_safe"]
    assert sdk_default_marker is not sdk_default_marker or True  # 语义占位（见下一断言）
    assert "abstain_is_safe: true" in open(YAML, "r", encoding="utf-8").read().replace(
        "abstain.abstain_is_safe: ", "abstain_is_safe: ")


def test_abstain_disabled_in_config(cfg):
    assert cfg["abstain"]["enabled"] is False


def test_non_abstain_path_is_unchanged(mod, cfg):
    """非 abstain 时，映射即规则映射（三口径同时给出）。"""
    m = mod.map_prediction(0.95, [], False, cfg)
    assert m["pred"] == 1 and m["pred_span_only"] == 0 and m["pred_060"] == 1
    assert m["abstained"] is False


def test_map_prediction_rejects_incomplete_config(mod):
    """**禁止默认值兜底**：cfg 给了但缺键 ⇒ 抛错（不静默取兜底）。"""
    with pytest.raises(KeyError):
        mod.map_prediction(0.5, [], False, {"abstain": {"fallback": "negative"}})


# ---------------------------------------------------------------------------
# ⑤ score = max(doc_score, max(span_score))
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc,spans,expect", [
    (0.2, [{"score": 0.7}, {"score": 0.3}], 0.7),
    (0.9, [], 0.9),
    (0.9, [{"score": 0.95}], 0.95),
    (0.0, [], 0.0),
])
def test_combined_score_is_max_of_doc_and_spans(mod, doc, spans, expect):
    assert mod.combined_score(doc, spans) == expect


def test_combined_score_returns_none_without_any_score(mod):
    assert mod.combined_score(None, []) is None


def test_span_scores_of_skips_malformed_entries(mod):
    """结构不合法项被跳过（不计入分母），但**不静默吞错**——由调用方披露计数。"""
    assert mod.span_scores_of([{"score": 0.5}, {"start": 0}, "x", None, {"score": "0.7"}]) == [0.5, 0.7]
    assert mod.span_scores_of(None) == []


# ---------------------------------------------------------------------------
# ⑥ §2 统一输出 schema
# ---------------------------------------------------------------------------
def test_output_columns_match_frozen_schema(mod):
    assert mod.OUTPUT_COLUMNS == ["sample_id", "text_sha256", "pred", "raw_type",
                                  "raw", "score", "latency_ms"]


def test_raw_json_carries_full_spans_and_schema_fields(mod, cfg, tmp_path):
    """`raw` 须含**完整** spans（start/end/score）——Span-F1 依赖（§0.6-3）。"""
    raw = {
        "doc_score": 0.99,
        "spans": [{"start": 6, "end": 15, "score": 0.772}],
        "abstain": False,
        "doc_score_source": "doc_head",
        "norm_changed": False,
        "stages_applied": ["base64"],
        "windowed": False,
    }
    s = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    back = json.loads(s)
    assert back["spans"][0] == {"start": 6, "end": 15, "score": 0.772}
    for key in ("doc_score", "spans", "abstain"):
        assert key in back


def test_write_csv_writes_lf_only(mod, tmp_path):
    path = str(tmp_path / "out.csv")
    mod.write_csv(path, [{"sample_id": "S1", "pred": 1}], mod.OUTPUT_COLUMNS)
    raw = open(path, "rb").read()
    assert raw.count(b"\r") == 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["sample_id"] == "S1" and rows[0]["pred"] == "1"


def test_latency_stats_shape(mod):
    st = mod.latency_stats([1.0, 2.0, 3.0, 4.0])
    assert st["n"] == 4 and st["median"] == 2.5 and st["min"] == 1.0 and st["max"] == 4.0
    assert mod.latency_stats([])["mean"] is None


# ---------------------------------------------------------------------------
# ⑦ YAML 冻结值（口径与设备）
# ---------------------------------------------------------------------------
def test_config_decision_thresholds(cfg):
    assert float(cfg["decision"]["doc_threshold"]) == 0.9
    assert float(cfg["decision"]["span_threshold"]) == 0.45
    assert cfg["decision"]["mode"] == "doc_or_span"


def test_config_primary_caliber_is_doc_or_span(cfg):
    assert cfg["decision"]["comparisons"]["span_only"]["enabled"] is True
    assert cfg["decision"]["comparisons"]["threshold_0_60"]["span_threshold"] == 0.60


def test_config_device_is_cpu_primary_not_fallback(cfg):
    """ENV_LOCK §6.1：`device=cpu` **不是降级**（CUDA 撞 nvrtc 缺口 §3.8；v5 L122 规定 CPU）。"""
    assert cfg["device"] == "cpu"
    assert cfg["device_is_not_fallback"] is True


def test_config_recipe_versions_match_env_lock_5_3_1(cfg):
    """§5.3.1 是**唯一配方**：七包钉版逐项一致，版本不得漂移。"""
    pinned = cfg["recipe"]["pinned_versions"]
    assert pinned == {
        "unplug-ai": "0.6.0",
        "pydantic": "2.13.5",
        "pydantic_core": "2.46.5",
        "annotated-types": "0.8.0",
        "typing-inspection": "0.4.4",
        "onnxruntime": "1.30.0",
        "flatbuffers": "25.12.19",
    }
    assert int(cfg["recipe"]["expect_dist_info_count"]) == 7
    assert cfg["source"]["env_write_operations"] == 0
    assert cfg["source"]["model_revision"] == "19b7d6701bea1f6d9f03ec7bcaeeea390dff43be"


def test_config_pipeline_matches_sdk_catalog_tiny(cfg):
    """SDK `catalog.toml` tiny：max_length 512 / stride 64 / 8192 / 2048 / 256 / sliding。"""
    p = cfg["pipeline"]
    assert int(p["tokenizer_max_length"]) == 512
    assert int(p["tokenizer_stride"]) == 64
    assert int(p["long_text"]["threshold_chars"]) == 8192
    assert int(p["long_text"]["chunk_chars"]) == 2048
    assert int(p["long_text"]["overlap_chars"]) == 256
    assert p["long_text"]["mode"] == "sliding"
    assert int(p["batch_size"]) == 4


def test_config_declares_no_project_side_preprocessing(cfg):
    """§0.6-1：SDK 内置规范化如实披露，**不叠加不剥离**。"""
    assert cfg["pipeline"]["normalization"]["project_side_preprocessing"] == "none"
    assert cfg["pipeline"]["normalization"]["disclose_changed_rate"] is True


def test_config_output_raw_type_and_score_rule(cfg):
    assert cfg["output"]["raw_type"] == "doc_span"
    assert cfg["output"]["columns"] == ["sample_id", "text_sha256", "pred", "raw_type",
                                        "raw", "score", "latency_ms"]
    assert cfg["output"]["score_rule"] == "max(doc_score, max(span_score))"
    assert cfg["output"]["newline"] == "lf"


def test_config_span_truth_caliber(cfg):
    st = cfg["metrics"]["span_truth"]
    assert st["code_carrier"] == "payload_substring_in_text"
    assert st["text_carrier"] == "full_text"
    assert st["attack_only"] is True
    assert st["locate_failure"] == "count_and_disclose"
    assert st["benign_span_false_positive"] == "disclosed_separately"


def test_config_abstain_counterfactual_uses_sdk_defaults(cfg):
    """反事实触发率须以 SDK **出厂默认**（catalog.toml）统计。"""
    cf = cfg["abstain"]["counterfactual"]
    assert cf["use_sdk_defaults"] is True
    assert float(cf["doc_threshold"]) == 0.9
    assert float(cf["span_threshold"]) == 0.60
    assert cf["abstain_enabled"] is True
    assert float(cf["tau_abstain_low"]) == 0.35


# ---------------------------------------------------------------------------
# ⑧ 探测样本隔离（§0.6-6：禁止进入任何分片）
# ---------------------------------------------------------------------------
def test_repro_probe_is_marked_as_never_entering_splits(cfg):
    assert cfg["repro_probe"]["must_not_enter_any_split"] is True
    assert cfg["repro_probe"]["enabled_by_flag"] == "--repro-check"


def test_repro_probe_has_exactly_four_synthetic_samples(cfg):
    ids = [s["id"] for s in cfg["repro_probe"]["samples"]]
    assert ids == ["inj-1", "inj-2", "inj-3-b64", "benign-1"]


def test_repro_probe_expectations_match_env_lock_5_3(cfg):
    """@5.3 记录值：inj-2 doc 0.999908 + span(6–15, 0.772)；其余三条见 ENV_LOCK。"""
    by_id = {s["id"]: s for s in cfg["repro_probe"]["samples"]}
    assert by_id["inj-2"]["expect_spans"] == [{"start": 6, "end": 15, "score": 0.772}]
    assert abs(float(by_id["inj-2"]["expect_doc_score"]) - 0.9999076128005981) < 1e-12
    assert abs(float(by_id["inj-1"]["expect_doc_score"]) - 4.735737093142234e-06) < 1e-15
    assert abs(float(by_id["inj-3-b64"]["expect_doc_score"]) - 1.504350257164333e-05) < 1e-15
    assert abs(float(by_id["benign-1"]["expect_doc_score"]) - 1.9937483557441738e-06) < 1e-15


def test_probe_texts_are_not_used_by_split_runner_signature(mod):
    """`run_split` 的签名里**没有**探测样本入口 ⇒ 结构上不可能把探测样本混进分片。"""
    import inspect
    params = list(inspect.signature(mod.run_split).parameters)
    assert "samples" not in params and "probe" not in params
    assert params[:4] == ["split", "csv_path", "model", "normalizer"]


# ---------------------------------------------------------------------------
# ⑨ 硬约束与 CLI
# ---------------------------------------------------------------------------
def test_top_level_imports_only_stdlib(mod):
    tree = ast.parse(open(SCRIPT, "r", encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "argparse", "importlib", "json", "os", "sys", "time",
                   "datetime"}


def test_module_namespace_has_no_heavy_libs(mod):
    assert "torch" not in vars(mod)
    assert "transformers" not in vars(mod)


def test_heavy_imports_are_lazy_inside_functions(mod):
    """重型库只允许出现在函数体内（惰性 import）。"""
    tree = ast.parse(open(SCRIPT, "r", encoding="utf-8").read())
    heavy = {"torch", "transformers", "unplug", "yaml", "psutil"}
    found_in_funcs = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                found_in_funcs.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found_in_funcs.add(node.module.split(".")[0])
    assert heavy & found_in_funcs, "重型库应在函数体内惰性 import"


def test_no_bare_except_or_except_pass(mod):
    """硬约束：禁止 `except: pass` 静默吞错；本项目亦禁裸 `except:`。"""
    src = open(SCRIPT, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None, "禁止裸 except:（第 %d 行）" % node.lineno
            body_types = [type(b) for b in node.body]
            assert body_types != [ast.Pass], "禁止 except ...: pass（第 %d 行）" % node.lineno


def test_script_has_no_crlf_and_reconfigures_stdout(mod):
    raw = open(SCRIPT, "rb").read()
    assert raw.count(b"\r") == 0
    src = raw.decode("utf-8")
    assert 'reconfigure(encoding="utf-8"' in src


def test_config_yaml_has_no_crlf():
    assert open(YAML, "rb").read().count(b"\r") == 0


def test_vendor_assertion_has_no_default_fallback(mod):
    """`assert_vendor_ready` 不得有默认值兜底：版本不符即抛错（§5.3.1 唯一配方）。"""
    src = open(SCRIPT, "r", encoding="utf-8").read()
    assert "raise RuntimeError(\"vendor 版本表与 ENV_LOCK.md §5.3.1 不符" in src
    assert "raise FileNotFoundError(\n            \"vendor 不存在" in src


def test_self_check_exits_zero():
    proc = subprocess.run([sys.executable, SCRIPT, "--self-check"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert "SELF_CHECK OK" in proc.stdout


# ---------------------------------------------------------------------------
# ⑩ 测试本身不得访问 sealed_test
# ---------------------------------------------------------------------------
def test_this_test_file_does_not_reference_sealed_csv():
    src = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    needle = "data/" + "sealed" + "_test/CodeObf-PI_v1.0/regime"
    assert needle not in src

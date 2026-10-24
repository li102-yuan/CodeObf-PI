# -*- coding: utf-8 -*-
"""tests/baseline/test_03_07_unify_output_mapping.py — W4.6 统一输出映射（§3–§4）。

覆盖 `windows/W4.6_unify.md` §3–§4 的映射 / τ 寻优 / 一致性校验 / 指标语义：

1. **§3 判词解析**：`safe`／`unsafe`／`unparsed`（首行首词）
2. **§3 B1/B2/B4 映射**：任一特征命中 / `score > τ` 严格 / `doc_or_span`@0.45 + abstain⇒负例
3. **§4.1 τ_B2 寻优**：F1 最大、并列取 τ 较大者、候选含 ±∞ 端点
4. **§4.2 指标**：TPR/FPR/F1、`clean_test` 退化（FPR 无定义、F1=2TP/(2TP+FN)）、AUROC 边界
5. **§4.2 Span-F1**：真值派生（code=payload 子串 / text=全文 / 定位失败披露）+ 字符级 F1
6. **§1.4 一致性校验**：同一 sample_id 的 text_sha256 跨检测器分歧检出
7. **硬约束 §0.2-7**：顶层不 import 重型库；`--self-check` 退出码 = 0

**离线**：全部输入自造，不读 sealed_test、不加载模型。
"""
from __future__ import annotations

import ast
import importlib.util
import io
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "baseline", "03_07_unify_output_mapping.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w46_mapping", SCRIPT)


# ---------------------------------------------------------------------------
# ① §3 判词解析（B3/B5）
# ---------------------------------------------------------------------------
def test_first_word_of_judgment(mod):
    assert mod.first_word_of_judgment("unsafe\nS9") == "unsafe"
    assert mod.first_word_of_judgment("  safe  ") == "safe"
    assert mod.first_word_of_judgment("unsafe extra words") == "unsafe"
    assert mod.first_word_of_judgment("") == ""
    assert mod.first_word_of_judgment(None) == ""


def test_parse_judgment_unsafe(mod):
    assert mod.parse_judgment("unsafe\nS9") == (1, False)


def test_parse_judgment_safe(mod):
    assert mod.parse_judgment("safe") == (0, False)


def test_parse_judgment_strips_leading_whitespace(mod):
    assert mod.parse_judgment("  unsafe\nS9  ") == (1, False)


def test_parse_judgment_unparsed(mod):
    pred, unparsed = mod.parse_judgment("weird-verdict")
    assert pred is None
    assert unparsed is True


def test_parse_judgment_empty_is_unparsed(mod):
    pred, unparsed = mod.parse_judgment("")
    assert pred is None and unparsed is True


# ---------------------------------------------------------------------------
# ② §3 B1 映射
# ---------------------------------------------------------------------------
def test_map_b1_any_hit_is_malicious(mod):
    assert mod.map_b1([]) == 0
    assert mod.map_b1(["base64"]) == 1
    assert mod.map_b1(["base64", "hex", "zerowidth"]) == 1
    assert mod.map_b1("not-a-list") == 0


def test_map_b1_raw_decodes_json_list(mod):
    assert mod.map_b1_raw('["base64"]') == (1, None)
    assert mod.map_b1_raw("[]") == (0, None)


# ---------------------------------------------------------------------------
# ③ §3 B2 映射（score > τ 严格）
# ---------------------------------------------------------------------------
def test_map_b2_strict_greater(mod):
    assert mod.map_b2(0.6, 0.5) == 1
    assert mod.map_b2(0.5, 0.5) == 0          # 严格 >，相等为负
    assert mod.map_b2(None, 0.5) == 0


def test_map_b2_raw_decodes_entropy(mod):
    pred, score = mod.map_b2_raw('{"entropy": 4.5}', 4.0)
    assert pred == 1 and score == 4.5
    pred, score = mod.map_b2_raw('{"entropy": 3.0}', 4.0)
    assert pred == 0 and score == 3.0


# ---------------------------------------------------------------------------
# ④ §3 B4 映射（doc_or_span@0.45 + abstain⇒负例）
# ---------------------------------------------------------------------------
def test_map_b4_doc_branch(mod):
    pred, score, abstained = mod.map_b4({"doc_score": 0.95, "spans": []})
    assert pred == 1 and score == 0.95 and not abstained


def test_map_b4_span_branch(mod):
    pred, score, _ = mod.map_b4({"doc_score": 0.1, "spans": [{"start": 0, "end": 5, "score": 0.5}]})
    assert pred == 1 and abs(score - 0.5) < 1e-12


def test_map_b4_both_below_is_negative(mod):
    pred, score, _ = mod.map_b4({"doc_score": 0.1, "spans": [{"start": 0, "end": 5, "score": 0.4}]})
    assert pred == 0 and abs(score - 0.4) < 1e-12


def test_map_b4_score_is_max_of_doc_and_spans(mod):
    _, score, _ = mod.map_b4({"doc_score": 0.2,
                              "spans": [{"start": 0, "end": 1, "score": 0.7},
                                        {"start": 2, "end": 3, "score": 0.6}]})
    assert abs(score - 0.7) < 1e-12


def test_map_b4_abstain_maps_to_negative(mod):
    pred, _, abstained = mod.map_b4({"doc_score": 0.99, "spans": [], "abstain": True})
    assert pred == 0 and abstained


def test_map_b4_doc_threshold_exact_0_9(mod):
    pred, _, _ = mod.map_b4({"doc_score": 0.9, "spans": []})
    assert pred == 1                              # ≥ 0.9


def test_map_b4_span_threshold_exact_0_45(mod):
    pred, _, _ = mod.map_b4({"doc_score": 0.0, "spans": [{"start": 0, "end": 1, "score": 0.45}]})
    assert pred == 1                              # ≥ 0.45


# ---------------------------------------------------------------------------
# ⑤ §4.1 τ_B2 寻优
# ---------------------------------------------------------------------------
def test_search_tau_perfect_separation(mod):
    tau, f1 = mod.search_tau_b2([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert f1 == 1.0


def test_search_tau_tie_breaks_to_larger_tau(mod):
    # scores=(1,2,3,4,5), labels=(0,1,0,0,1)：最大 F1=2/3 在 τ=1 与 τ=4 并列 ⇒ 取 4
    tau, f1 = mod.search_tau_b2([1, 2, 3, 4, 5], [0, 1, 0, 0, 1])
    assert abs(f1 - 2 / 3) < 1e-12
    assert tau == 4


def test_search_tau_all_negative_returns_inf(mod):
    # 全负标签：所有 τ 均 F1=0 ⇒ 并列取最大 τ = +inf（保守放行负例）
    tau, f1 = mod.search_tau_b2([0.1, 0.2, 0.3], [0, 0, 0])
    assert f1 == 0.0
    assert tau == float("inf")


def test_search_tau_ignores_none_scores(mod):
    tau, f1 = mod.search_tau_b2([None, 0.2, 0.8], [1, 0, 1])
    assert f1 == 1.0


def test_apply_b2_threshold(mod):
    assert mod.apply_b2_threshold([0.6, 0.4, None], 0.5) == [1, 0, 0]


# ---------------------------------------------------------------------------
# ⑥ §4.2 指标
# ---------------------------------------------------------------------------
def test_compute_binary_metrics_perfect(mod):
    m = mod.compute_binary_metrics([1, 1, 0, 0], [1, 1, 0, 0])
    assert m["tpr"] == 1.0 and m["fpr"] == 0.0 and m["f1"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_compute_binary_metrics_fpr_undefined_when_no_negative(mod):
    m = mod.compute_binary_metrics([1, 1, 1, 1], [1, 1, 0, 1])
    assert m["fpr"] is None                       # 无负样本 ⇒ FPR 无定义（clean_test）
    # F1 退化 = 2TP/(2TP+FN) = 2*3/(2*3+1)
    assert abs(m["f1"] - 2 * 3 / (2 * 3 + 1)) < 1e-12


def test_f1_score_clean_test_degenerate(mod):
    # 全 label=1：F1 = 2TP/(2TP+FN)
    assert abs(mod.f1_score([1, 1, 1, 1], [1, 1, 0, 1]) - 2 * 3 / 7) < 1e-12


def test_f1_score_no_positive_denom_zero(mod):
    assert mod.f1_score([0, 0], [0, 0]) == 0.0


def test_auroc_perfect_and_reverse(mod):
    assert abs(mod.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12
    assert abs(mod.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) - 0.0) < 1e-12


def test_auroc_ties_use_average_rank(mod):
    # 分数全同 ⇒ AUROC = 0.5
    assert abs(mod.auroc([0, 1], [0.5, 0.5]) - 0.5) < 1e-12


def test_auroc_missing_class_returns_none(mod):
    assert mod.auroc([1, 1], [0.5, 0.6]) is None   # 无负样本
    assert mod.auroc([0, 0], [0.5, 0.6]) is None   # 无正样本


# ---------------------------------------------------------------------------
# ⑦ §4.2 Span-F1
# ---------------------------------------------------------------------------
def test_derive_span_truth_code_carrier(mod):
    t, status = mod.derive_span_truth({"text": "xx ABC yy", "payload": "ABC", "carrier_type": "code"})
    assert status == "ok" and t == [(3, 6)]


def test_derive_span_truth_text_carrier_is_full_text(mod):
    t, status = mod.derive_span_truth({"text": "hello", "payload": "hello", "carrier_type": "text"})
    assert status == "ok" and t == [(0, 5)]


def test_derive_span_truth_payload_not_found(mod):
    _, status = mod.derive_span_truth({"text": "xx", "payload": "nope", "carrier_type": "code"})
    assert status == "locate_failed"


def test_derive_span_truth_missing_carrier(mod):
    _, status = mod.derive_span_truth({"text": "no payload col"})
    assert status == "no_carrier_or_payload"


def test_span_f1_exact_and_disjoint(mod):
    assert mod.span_f1([(0, 5)], [(0, 5)]) == 1.0
    assert mod.span_f1([(0, 5)], [(5, 10)]) == 0.0
    assert mod.span_f1([], []) == 1.0
    assert mod.span_f1([(0, 5)], []) == 0.0


def test_span_f1_partial_overlap(mod):
    # pred [0,6) vs true [3,9) ⇒ 交 [3,6)=3，prec=3/6，rec=3/6 ⇒ F1=0.5
    assert abs(mod.span_f1([(0, 6)], [(3, 9)]) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# ⑧ §1.4 一致性校验
# ---------------------------------------------------------------------------
def test_check_consistency_detects_divergence(mod):
    outs = {
        "b1": [{"sample_id": "S1", "text_sha256": "aaa"}],
        "b2": [{"sample_id": "S1", "text_sha256": "aaa"}],
        "b3": [{"sample_id": "S1", "text_sha256": "bbb"}],
    }
    assert mod.check_text_sha256_consistency(outs) == ["S1"]


def test_check_consistency_all_equal_is_empty(mod):
    outs = {k: [{"sample_id": "S1", "text_sha256": "aaa"}] for k in ("b1", "b2", "b3", "b4", "b5")}
    assert mod.check_text_sha256_consistency(outs) == []


def test_check_consistency_ignores_none_hashes(mod):
    outs = {
        "b1": [{"sample_id": "S1", "text_sha256": "aaa"}],
        "b2": [{"sample_id": "S1", "text_sha256": None}],
        "b3": [{"sample_id": "S1", "text_sha256": "aaa"}],
    }
    assert mod.check_text_sha256_consistency(outs) == []


# ---------------------------------------------------------------------------
# ⑨ raw 解码
# ---------------------------------------------------------------------------
def test_decode_raw(mod):
    assert mod.decode_raw('{"entropy": 4.5}') == {"entropy": 4.5}
    assert mod.decode_raw('["a","b"]') == ["a", "b"]
    assert mod.decode_raw('"unsafe"') == "unsafe"
    assert mod.decode_raw("plain text not json") == "plain text not json"
    assert mod.decode_raw(None) is None


# ---------------------------------------------------------------------------
# ⑩ 硬约束：顶层不 import 重型库、离线导入无副作用、self-check 退出码
# ---------------------------------------------------------------------------
def test_top_level_imports_only_stdlib(mod):
    tree = ast.parse(open(SCRIPT, "r", encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "argparse", "csv", "importlib", "json", "os", "sys", "time", "datetime"}


def test_import_does_not_load_heavy_libs(mod):
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_import_has_no_side_effect_output(mod):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        _load("w46_mapping_again", SCRIPT)
    finally:
        sys.stdout = old
    assert buf.getvalue() == ""


def test_self_check_exits_zero():
    proc = subprocess.run([sys.executable, SCRIPT, "--self-check"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert "SELF_CHECK OK" in proc.stdout


def test_interface_map_lists_all_sections(mod):
    specs = [row["spec"] for row in mod.build_interface_map()]
    joined = " ".join(specs)
    for key in ("§1.2", "§1.4", "§3", "§4.1", "§4.2", "Span-F1"):
        assert key in joined

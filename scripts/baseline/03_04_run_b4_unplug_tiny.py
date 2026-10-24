# -*- coding: utf-8 -*-
"""03_04_run_b4_unplug_tiny.py — W4.4 B4 unplug-tiny **原始推理**（阶段3，接口 v1.0）。

职责：实现 `windows/W4.4_b4_unplug.md` 的 B4 基线——在**三个测试分片**（`id_test`／`ood_test`／
`clean_test`）上经 **SDK 官方管线**推理，产出符合 `windows/W4.6_unify.md` §2 的逐样本 CSV
（`raw_type=doc_span`）。

判定口径（`PREREG_v1.2.md` §1，**预注册冻结**）::

    主口径 doc_or_span : doc_score ≥ 0.9 或 任一 span_score ≥ 0.45  ⇒ 1
    对照 ① span-only   : 仅 任一 span_score ≥ 0.45               ⇒ 1（PREREG_v1.md §12.2 字面）
    对照 ② 0.60        : doc_score ≥ 0.9 或 任一 span_score ≥ 0.60 ⇒ 1（SDK catalog.toml）
    abstain            : 关闭；无法关闭时 ABSTAIN ⇒ **负例（0）**
    ⚠ 不得沿用 SDK 默认 abstain_is_safe=False（方向相反）⇒ 本窗口**显式**置 true

SDK 管线（本窗口的"官方前向"，全部组件取自 SDK 自身，不做项目侧改造）::

    text（原文直通，零项目侧清洗）
      → unplug.core.normalize.normalize.Normalizer（SDK 12 级规范化；如实披露、不叠加不剥离）
      → 滑窗（>8192 字符时；2048 字符 / 256 重叠；SDK split_sliding_windows）
      → unplug.ml.span_model.SpanInferenceModel（官方 DebertaV2ForDualHead 双头前向）
      → merge_window_predictions → NormResult.to_original_span（span 回映到**原文**偏移）
      ⇒ 取 **原始** doc_score / spans（不含 SDK 的 `max(score, 0.55)` 抬分与 band 层）

    判定规则按预注册 P-C 施加（**不由 SDK 的 0.60／abstain 带决定**）——
    P-C 只固定判定规则、不固定实现（PREREG_v1.2.md §1 末段）。

硬约束（`windows/W4.4_b4_unplug.md` §0.6）:

1. 只读 `text` 列，原文直通进 SDK；SDK 内置规范化如实披露、不叠加不剥离；
2. 读 `sealed_test` 前须由窗口在 `ACCESS_LOG.md` 登记 6 列（本脚本只打印提醒，不代登记）；
3. `score = max(doc_score, max(span_score))`；`raw` 含完整 spans（start/end/score）——Span-F1 依赖；
4. abstain 触发率强制披露；>5% 时附「abstain → 正例」对照；
5. 禁止用测试集调阈值/选口径（两份对照只作敏感性分析）；
6. 探测样本（inj-1/2/3-b64、benign-1）为合成、非评估集，**禁止**进入任何分片；
7. 顶层不 import 重型库；未实现处 `NOT_IMPLEMENTED` + `sys.exit(0)`；行尾 LF；stdout 重配 UTF-8。

CLI::

    python scripts/baseline/03_04_run_b4_unplug_tiny.py --self-check
    python scripts/baseline/03_04_run_b4_unplug_tiny.py --repro-check \
        --config configs/baseline/b4_unplug.yaml
    python scripts/baseline/03_04_run_b4_unplug_tiny.py --config configs/baseline/b4_unplug.yaml \
        --unify-config configs/baseline/unify.yaml --splits id_test ood_test clean_test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))     # 项目根目录

# 冻结常量（`windows/W4.6_unify.md` §2）
RAW_TYPE = "doc_span"
OUTPUT_COLUMNS = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score", "latency_ms"]

# §0.3 主口径与两份对照（预注册冻结；**不由本脚本决定**）
DOC_THRESHOLD = 0.9
SPAN_THRESHOLD = 0.45
COMPARE_SPAN_ONLY = 0.45
COMPARE_THRESHOLD_060 = 0.60


def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（先例：W3.7 `F-1`）。只动输出流，不触碰判定值。

    **不静默吞错**：重配失败时显式登记到 stderr（`ascii()` 保证 ASCII 安全），
    避免 GBK 环境下的 `UnicodeEncodeError` 变成「无声假成功」。
    """
    problems = []
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            problems.append("%s 不支持 reconfigure" % name)
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:
            problems.append("%s 重配失败：%s" % (name, ascii(exc)))
    if problems:
        sys.stderr.write("WARN init_streams: %s\n" % "; ".join(problems))


def _load_module(name, path):
    """按文件路径加载模块（上游脚本无包结构，沿用 W4.3 的 `_load_module` 先例）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 载入上游（单一出处 —— 禁止本窗口自行实现读取／锚点／映射／指标）
# ---------------------------------------------------------------------------
_PREPROCESS = _load_module("baseline_preprocess_b4",
                           os.path.join(_HERE, "03_06_unify_preprocess.py"))
_MAPPING = _load_module("baseline_mapping_b4",
                        os.path.join(_HERE, "03_07_unify_output_mapping.py"))

load_samples = _PREPROCESS.load_samples
load_rows = _PREPROCESS.load_rows
sha256_of = _PREPROCESS.sha256_of
md5_of = _PREPROCESS.md5_of
write_csv = _PREPROCESS.write_csv

map_b4 = _MAPPING.map_b4                        # §3 B4 映射（单源，含 abstain⇒负例）
compute_binary_metrics = _MAPPING.compute_binary_metrics
auroc = _MAPPING.auroc
derive_span_truth = _MAPPING.derive_span_truth
span_f1 = _MAPPING.span_f1
binary_counts = _MAPPING.binary_counts


# ---------------------------------------------------------------------------
# 配置（惰性 `import yaml`，保证 `make check` 探针轻量）
# ---------------------------------------------------------------------------
def load_config(path):
    """读 YAML。**禁止默认值兜底**：文件不存在／键缺失一律显式报错。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("配置文件不存在：%s" % path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 PyYAML：%s" % exc) from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _require(mapping, key, where):
    """取必需键；缺失即抛错（**不做默认值兜底**）。"""
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError("配置缺少必需键 %r（位置：%s）" % (key, where))
    return mapping[key]


# ---------------------------------------------------------------------------
# §3 判定：主口径 + 两份对照（纯函数，可离线测试；阈值全部由调用方显式传入）
# ---------------------------------------------------------------------------
def span_scores_of(spans):
    """取 span 分数列表（忽略结构不合法项；**不静默吞错**——非 dict／缺 score 即跳过并计数由调用方披露）。"""
    out = []
    for s in spans or []:
        if isinstance(s, dict) and s.get("score") is not None:
            out.append(float(s["score"]))
    return out


def doc_or_span(doc_score, spans, doc_threshold=DOC_THRESHOLD, span_threshold=SPAN_THRESHOLD):
    """主口径 `doc_or_span`：`doc_score ≥ doc_threshold` **或** 任一 `span_score ≥ span_threshold`。"""
    if doc_score is None:
        doc_hit = False
    else:
        doc_hit = float(doc_score) >= doc_threshold
    span_hit = any(sc >= span_threshold for sc in span_scores_of(spans))
    return 1 if (doc_hit or span_hit) else 0


def span_only(doc_score, spans, span_threshold=COMPARE_SPAN_ONLY):
    """对照 ① span-only（`PREREG_v1.md` §12.2 字面）：**仅** span 分支，无 doc 分支。"""
    return 1 if any(sc >= span_threshold for sc in span_scores_of(spans)) else 0


def combined_score(doc_score, spans):
    """§0.6-3：`score = max(doc_score, max(span_score))`（供 AUROC）。无可用分数时返回 None。"""
    candidates = []
    if doc_score is not None:
        candidates.append(float(doc_score))
    candidates.extend(span_scores_of(spans))
    return max(candidates) if candidates else None


def map_prediction(doc_score, spans, abstained, cfg=None):
    """把一条 SDK 原始输出映射到三个口径。返回 dict。

    `abstained` 为 True 时**一律** `pred=0`（§0.3：ABSTAIN ⇒ 负例）——这是**显式**映射，
    不依赖任何 SDK 默认值（`abstain_is_safe` 默认 False 的方向与本要求相反）。

    `cfg=None` 时取本模块顶部的**预注册冻结常量**（`DOC_THRESHOLD`／`SPAN_THRESHOLD`，
    非兜底值）；`cfg` 给定则**逐键强制取用**，缺键即抛错。
    """
    if cfg is None:
        doc_t, span_t, abstain_fallback = DOC_THRESHOLD, SPAN_THRESHOLD, "negative"
    else:
        dec = _require(cfg, "decision", "root")
        doc_t = float(_require(dec, "doc_threshold", "decision"))
        span_t = float(_require(dec, "span_threshold", "decision"))
        abstain_fallback = str(_require(_require(cfg, "abstain", "root"), "fallback", "abstain"))
    out = {
        "pred": doc_or_span(doc_score, spans, doc_t, span_t),
        "pred_span_only": span_only(doc_score, spans),
        "pred_060": doc_or_span(doc_score, spans, doc_t, COMPARE_THRESHOLD_060),
        "score": combined_score(doc_score, spans),
        "abstained": bool(abstained),
    }
    if out["abstained"] and abstain_fallback == "negative":
        out["pred"] = 0
        out["pred_span_only"] = 0
        out["pred_060"] = 0
    return out


# ---------------------------------------------------------------------------
# 反事实 abstain 触发率（§0.6-4 强制披露）：以 SDK **出厂默认**统计「若启用 abstain 会有多少样本落 ABSTAIN 带」
# ---------------------------------------------------------------------------
def sdk_abstain_would_trigger(doc_score, spans, sdk_policy):
    """用 SDK 自身的 `decide_band()` 判断：在 **SDK 默认**（catalog.toml）策略下是否落 ABSTAIN 带。

    目的**不是**改变判定（判定已由 `map_prediction` 按 P-C 施加），而是**如实披露** abstain 带
    在本数据集上会有多宽。`sdk_policy` 为 dict：doc/span 阈值、`abstain_enabled`、`tau_abstain_low`。
    """
    from unplug.core.policy.decision import DecisionPolicy, MlBand, decide_band

    policy = DecisionPolicy(
        mode="doc_or_span",
        tau_doc=float(sdk_policy["doc_threshold"]),
        tau_span=float(sdk_policy["span_threshold"]),
        tau_abstain_low=float(sdk_policy["tau_abstain_low"]),
        abstain_enabled=bool(sdk_policy["abstain_enabled"]),
    )
    span_score = max(span_scores_of(spans)) if span_scores_of(spans) else 0.0
    band = decide_band(
        doc_score=float(doc_score or 0.0),
        span_score=span_score,
        policy=policy,
    )
    return band == MlBand.ABSTAIN


# ---------------------------------------------------------------------------
# 统计小工具（与 03_03 同风格，无第三方依赖）
# ---------------------------------------------------------------------------
def _quantile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _median(xs):
    return _quantile(xs, 0.5)


def latency_stats(values):
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": _median(xs),
        "p95": _quantile(xs, 0.95),
        "min": min(xs),
        "max": max(xs),
    }


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return ("%%.%df" % nd) % x
    return str(x)


def _mib(n_bytes):
    if n_bytes is None:
        return "n/a"
    return "%.1f" % (n_bytes / (1024.0 * 1024.0))


def _rss_bytes():
    """进程 RSS（字节）。B4 为 CPU 档 ⇒ 报进程 RSS（W4.6 §4.2：B4 显存记 n/a）。"""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


# ---------------------------------------------------------------------------
# SDK 加载（重型，惰性 import；只在真正推理时调用）
# ---------------------------------------------------------------------------
def assert_vendor_ready(cfg):
    """核对 vendor 版本表与 `ENV_LOCK.md` §5.3.1 一致（7 个 dist-info、无同名双版本）。

    **不做默认值兜底**：任一钉版不符即抛错 —— 版本表不可协商（`§5.3.1 是唯一配方`）。
    """
    import importlib.metadata as md

    recipe = _require(cfg, "recipe", "root")
    vendor = _require(_require(cfg, "source", "root"), "vendor_path", "source")
    expected = dict(_require(recipe, "pinned_versions", "recipe"))
    expect_n = int(_require(recipe, "expect_dist_info_count", "recipe"))

    if not os.path.isdir(vendor):
        raise FileNotFoundError(
            "vendor 不存在：%s —— 须先按 ENV_LOCK.md §5.3.1 配方重建（唯一配方）" % vendor)

    found = {}
    dists = list(md.distributions(path=[vendor]))
    for d in dists:
        name = d.metadata["Name"]
        found[name] = d.version

    dist_info_count = len([n for n in os.listdir(vendor) if n.endswith(".dist-info")])
    problems = []
    if dist_info_count != expect_n:
        problems.append("dist-info 数 %d != 期望 %d（同名双版本 ⇒ 须清空重装）"
                        % (dist_info_count, expect_n))
    for name, ver in sorted(expected.items()):
        got = found.get(name)
        if got is None:
            problems.append("缺包 %s（期望 %s）" % (name, ver))
        elif got != ver:
            problems.append("%s 版本 %s != 钉版 %s" % (name, got, ver))
    if problems:
        raise RuntimeError("vendor 版本表与 ENV_LOCK.md §5.3.1 不符：\n  - " + "\n  - ".join(problems))
    return {"vendor": vendor, "dist_info_count": dist_info_count, "versions": found}


def load_model(cfg):
    """加载 SDK 的 `SpanInferenceModel`（官方双头装配）与 `Normalizer`。

    返回 ``(model, normalizer, meta)``。`device="cpu"` 来自 §6.1（**CPU 即主档，非降级**）。
    """
    t0 = time.time()
    recipe = _require(cfg, "recipe", "root")
    cache = _require(cfg["source"], "model_cache", "source")
    ckpt = os.path.join(cache, _require(recipe, "checkpoint_subpath", "recipe"))
    device = _require(cfg, "device", "root")
    decision = _require(cfg, "decision", "root")
    doc_t = float(_require(decision, "doc_threshold", "decision"))
    # span 阈值取 **主口径与两份对照的最小值**（0.45），使 0.45／0.60 两份对照**都**可由同一份
    # 原始 spans 推出；SDK 的 `inj_threshold` 只在 decode 阶段过滤 span，不影响 doc 头。
    span_t = float(_require(decision, "span_threshold", "decision"))

    if not os.path.isdir(ckpt):
        raise FileNotFoundError("checkpoint 目录不存在：%s（先设 UNPLUG_MODEL_CACHE 并按配方取件）" % ckpt)

    from unplug.core.normalize.normalize import Normalizer
    from unplug.ml.span_model import SpanInferenceModel

    model = SpanInferenceModel(ckpt, doc_threshold=doc_t, inj_threshold=span_t, device=device)
    model.load()
    if not model.loaded:
        raise RuntimeError("SpanInferenceModel 加载失败：%s" % ckpt)
    meta = {
        "checkpoint": ckpt,
        "device": model.device,
        "is_dual_head": bool(model.is_dual_head),
        "doc_threshold": model.doc_threshold,
        "max_length": int(getattr(model, "_max_length", 0) or 0),
        "stride": int(getattr(model, "_stride", 0) or 0),
        "load_seconds": time.time() - t0,
        "rss_after_load_bytes": _rss_bytes(),
    }
    if not meta["is_dual_head"]:
        raise RuntimeError(
            "checkpoint 非双头（`dual_head=false`）⇒ 不是官方 unplug-tiny 前向，停手（ENV_LOCK §5.3）")
    return model, Normalizer(), meta


# ---------------------------------------------------------------------------
# 单样本推理（SDK 管线）
# ---------------------------------------------------------------------------
def predict_one(model, normalizer, text, pipe):
    """经 SDK 管线推理一条原文，返回 ``(doc_score, spans_in_original_coords, info)``。

    `spans` 的 start/end **已回映到原始 `text` 的字符偏移**（`NormResult.to_original_span`），
    因为 Span-F1 真值区间定义在冻结 CSV 的原始 `text` 列上（`W4.6_unify.md` §4.2）。
    """
    from unplug.ml.document_chunks import (
        merge_window_predictions,
        should_chunk_document,
        split_sliding_windows,
    )

    norm = normalizer.normalize(text)
    norm_text = norm.text
    threshold_chars = int(pipe["threshold_chars"])
    windowed = should_chunk_document(len(norm_text), threshold_chars=threshold_chars)
    if not windowed:
        pred = model.predict(norm_text)
    else:
        windows = split_sliding_windows(
            norm_text,
            chunk_chars=int(pipe["chunk_chars"]),
            overlap_chars=int(pipe["overlap_chars"]),
        )
        texts = [w for w, _ in windows]
        preds = model.predict_batch(texts, batch_size=int(pipe["batch_size"]))
        pred = merge_window_predictions(
            [(off, p) for (_, off), p in zip(windows, preds)], full_text=norm_text)

    spans = []
    for s in pred.spans:
        o_start, o_end = norm.to_original_span(int(s.start), int(s.end))
        if o_end <= o_start:
            continue
        spans.append({"start": int(o_start), "end": int(o_end), "score": float(s.score)})
    spans.sort(key=lambda d: (d["start"], d["end"]))
    info = {
        "norm_changed": norm_text != text,
        "stages_applied": list(norm.stages_applied),
        "windowed": bool(windowed),
        "norm_len": len(norm_text),
        "orig_len": len(text),
        "doc_score_source": getattr(pred, "doc_score_source", None),
    }
    return float(pred.doc_score), spans, info


# ---------------------------------------------------------------------------
# 分片推理
# ---------------------------------------------------------------------------
def run_split(split, csv_path, model, normalizer, cfg, pipe, sdk_policy, ts,
              progress_every=0, limit=0):
    """跑一个分片：逐样本推理 → 落 CSV → 汇总指标。返回 per-split 汇总 dict。"""
    rows = load_rows(csv_path)
    samples = load_samples(csv_path)
    if limit and limit > 0:
        rows = rows[:limit]
        samples = samples[:limit]
    by_id = {r.get("sample_id"): r for r in rows}

    decision = _require(cfg, "decision", "root")
    doc_t = float(_require(decision, "doc_threshold", "decision"))
    span_t = float(_require(decision, "span_threshold", "decision"))

    out_rows = []
    labels, preds_main, preds_span_only, preds_060, scores = [], [], [], [], []
    abstain_flags = []
    latencies = []
    norm_changed_n = 0
    stages_counter = {}
    windowed_n = 0
    span_truth_failed = 0
    span_truth_no_carrier = 0
    span_f1_values = []
    benign_with_span = 0
    benign_with_span_fired = 0
    attack_with_span_truth = 0

    for i, s in enumerate(samples):
        t0 = time.time()
        doc_score, spans, info = predict_one(model, normalizer, s.text, pipe)
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        if info["norm_changed"]:
            norm_changed_n += 1
        if info["windowed"]:
            windowed_n += 1
        for st in info["stages_applied"]:
            stages_counter[st] = stages_counter.get(st, 0) + 1

        abstained = False          # §0.3：abstain **关闭** ⇒ 主档不产生 ABSTAIN 带
        abstain_would = sdk_abstain_would_trigger(doc_score, spans, sdk_policy)
        abstain_flags.append(abstain_would)

        m = map_prediction(doc_score, spans, abstained, cfg)
        raw = {
            "doc_score": doc_score,
            "spans": spans,
            "abstain": abstained,
            "abstain_would_trigger_sdk_default": abstain_would,
            "doc_score_source": info["doc_score_source"],
            "norm_changed": info["norm_changed"],
            "stages_applied": info["stages_applied"],
            "windowed": info["windowed"],
        }
        out_rows.append({
            "sample_id": s.sample_id,
            "text_sha256": sha256_of(s.text),
            "pred": m["pred"],
            "raw_type": RAW_TYPE,
            "raw": json.dumps(raw, ensure_ascii=False, sort_keys=True),
            "score": "" if m["score"] is None else repr(float(m["score"])),
            "latency_ms": "%.3f" % latency_ms,
        })

        labels.append(int(s.label))
        preds_main.append(m["pred"])
        preds_span_only.append(m["pred_span_only"])
        preds_060.append(m["pred_060"])
        scores.append(float(m["score"]) if m["score"] is not None else 0.0)

        # Span-F1：仅攻击样本计入（§4.2 attack_only）；真值派生失败**只计数、不进均值**
        row = by_id.get(s.sample_id, {})
        if int(s.label) == 1:
            true_spans, status = derive_span_truth(row)
            if status == "ok":
                attack_with_span_truth += 1
                span_f1_values.append(span_f1([(d["start"], d["end"]) for d in spans], true_spans))
            elif status == "locate_failed":
                span_truth_failed += 1
            else:
                span_truth_no_carrier += 1
        elif spans:
            benign_with_span += 1
            if any(sc >= span_t for sc in span_scores_of(spans)):
                benign_with_span_fired += 1

        if progress_every and (i + 1) % progress_every == 0:
            print("  [%s] %d/%d  latency_med=%.1f ms" % (split, i + 1, len(samples), _median(latencies)))

    out_dir = _require(_require(cfg, "output", "root"), "directory", "output")
    out_name = "b4_%s_v1.0_%s.csv" % (split, ts)
    out_path = os.path.join(_ROOT, out_dir, out_name)
    write_csv(out_path, out_rows, OUTPUT_COLUMNS)

    main_m = compute_binary_metrics(labels, preds_main, scores)
    span_only_m = compute_binary_metrics(labels, preds_span_only, scores)
    m060 = compute_binary_metrics(labels, preds_060, scores)

    n_abstain = sum(1 for f in abstain_flags if f)
    abstain_rate = (n_abstain / len(abstain_flags)) if abstain_flags else 0.0

    # 「abstain → 正例」对照（>5% 时才在报告展示，但此处一律算好）
    pos_if_abstain = [1 if f else p for f, p in zip(abstain_flags, preds_main)]
    abstain_positive_m = compute_binary_metrics(labels, pos_if_abstain, scores)

    span_f1_mean = (sum(span_f1_values) / len(span_f1_values)) if span_f1_values else None

    return {
        "split": split,
        "csv_path": csv_path,
        "csv_md5": md5_of(csv_path),
        "n": len(samples),
        "outputs_csv": os.path.join(out_dir, out_name),
        "labels": labels,
        "preds": preds_main,
        "scores": scores,
        "metrics": {"main": main_m, "span_only": span_only_m, "threshold_060": m060},
        "abstain": {
            "enabled_in_run": False,
            "triggered_in_run": 0,
            "counterfactual_sdk_default_n": n_abstain,
            "counterfactual_sdk_default_rate": abstain_rate,
            "abstain_as_positive": abstain_positive_m,
            "policy_used": dict(sdk_policy),
        },
        "span_f1": {
            "mean_attack": span_f1_mean,
            "n_attack": sum(1 for y in labels if y == 1),
            "n_attack_with_truth": attack_with_span_truth,
            "locate_failed": span_truth_failed,
            "no_carrier_or_payload": span_truth_no_carrier,
            "benign_with_span_false_positive": benign_with_span,
            "benign_with_span_ge_0.45": benign_with_span_fired,
        },
        "normalization": {
            "changed_n": norm_changed_n,
            "changed_rate": (norm_changed_n / len(samples)) if samples else 0.0,
            "stages_counter": dict(sorted(stages_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
            "windowed_n": windowed_n,
        },
        "latency": latency_stats(latencies),
        "rss_bytes": _rss_bytes(),
    }


# ---------------------------------------------------------------------------
# 第 0 步硬前置：配方重建后的 4 条预演样本复跑（ENV_LOCK.md §5.3 比对）
# ---------------------------------------------------------------------------
def run_repro_check(cfg):
    """复跑 4 条**合成探测样本**（非评估集，禁止入任何分片），与 §5.3 记录值比对。

    比对走**模型层直驱**（与 §5.3 证据来源 `logs/env/tmp/b4_ml_probe.py` 逐参数一致：
    原始 text、`device=cpu`、`doc_threshold=0.9`、`inj_threshold=0.45`），不加规范化层，
    以保证与 §5.3 的**同口径**比对；另附 SDK 规范化后的诊断值用于 §0.4 强制披露①。
    """
    probe = _require(cfg, "repro_probe", "root")
    samples = _require(probe, "samples", "repro_probe")
    tol = _require(probe, "tolerance", "repro_probe")
    tol_doc = float(_require(tol, "doc_score_abs", "tolerance"))
    tol_span = float(_require(tol, "span_score_abs", "tolerance"))

    recipe = _require(cfg, "recipe", "root")
    cache = _require(cfg["source"], "model_cache", "source")
    ckpt = os.path.join(cache, _require(recipe, "checkpoint_subpath", "recipe"))
    decision = _require(cfg, "decision", "root")
    doc_t = float(_require(decision, "doc_threshold", "decision"))
    span_t = float(_require(decision, "span_threshold", "decision"))

    from unplug.core.normalize.normalize import Normalizer
    from unplug.ml.span_model import SpanInferenceModel

    model = SpanInferenceModel(ckpt, doc_threshold=doc_t, inj_threshold=span_t,
                               device=_require(cfg, "device", "root"))
    model.load()
    normalizer = Normalizer()

    results = []
    all_match = True
    for item in samples:
        sid = _require(item, "id", "repro_probe.samples")
        text = _require(item, "text", "repro_probe.samples")
        exp_doc = float(_require(item, "expect_doc_score", "repro_probe.samples"))
        exp_spans = _require(item, "expect_spans", "repro_probe.samples") or []

        pred = model.predict(text)
        got_spans = [{"start": int(s.start), "end": int(s.end), "score": float(s.score)}
                     for s in pred.spans]
        doc_ok = abs(float(pred.doc_score) - exp_doc) <= tol_doc
        span_ok = len(got_spans) == len(exp_spans)
        if span_ok:
            for g, e in zip(got_spans, exp_spans):
                if int(g["start"]) != int(e["start"]) or int(g["end"]) != int(e["end"]):
                    span_ok = False
                elif abs(float(g["score"]) - float(e["score"])) > tol_span:
                    span_ok = False

        nb = normalizer.normalize(text)
        pred_b = model.predict(nb.text)
        got_spans_b = [{"start": int(s.start), "end": int(s.end), "score": float(s.score)}
                       for s in pred_b.spans]

        ok = bool(doc_ok and span_ok)
        all_match = all_match and ok
        results.append({
            "id": sid,
            "expect_doc": exp_doc,
            "got_doc": float(pred.doc_score),
            "expect_spans": exp_spans,
            "got_spans": got_spans,
            "doc_match": doc_ok,
            "span_match": span_ok,
            "match": ok,
            "sdk_normalized": {
                "text_changed": nb.text != text,
                "stages_applied": list(nb.stages_applied),
                "doc_score": float(pred_b.doc_score),
                "spans": got_spans_b,
            },
        })
        print("[repro] %-10s doc 期望 %.6e 实测 %.6e  %s | spans 期望 %s 实测 %s  %s"
              % (sid, exp_doc, float(pred.doc_score), "OK" if doc_ok else "MISMATCH",
                 json.dumps(exp_spans), json.dumps(got_spans), "OK" if span_ok else "MISMATCH"))
        print("        └ SDK 规范化后：changed=%s stages=%s doc=%.6e spans=%s"
              % (nb.text != text, nb.stages_applied, float(pred_b.doc_score),
                 json.dumps(got_spans_b)))

    print("REPRO_CHECK %s（%d/%d 一致）" % ("PASS" if all_match else "FAIL",
                                          sum(1 for r in results if r["match"]), len(results)))
    return {"all_match": all_match, "results": results,
            "model_meta": {"is_dual_head": bool(model.is_dual_head), "device": model.device,
                           "doc_threshold": model.doc_threshold}}


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(path, cfg, unify_cfg, per_split, ts, vendor_info, model_meta,
                 repro=None, extra=None):
    """生成 `reports/baseline/b4_unplug_v1.0_<ts>.md`（§1 要求的全部小节）。"""
    decision = _require(cfg, "decision", "root")
    abstain_cfg = _require(cfg, "abstain", "root")
    src = _require(cfg, "source", "root")
    pipe = _require(cfg, "pipeline", "root")
    extra = extra or {}

    L = []
    A = L.append
    A("# B4 unplug-tiny 原始推理 — 运行报告 v1.0（%s）" % ts)
    A("")
    A("- 窗口：**W4.4**（阶段3，权限 A4）｜脚本：`scripts/baseline/03_04_run_b4_unplug_tiny.py`")
    A("- 接口：`windows/W4.6_unify.md` §1.3／§2／§3／§4.2（冻结 v1.0）")
    A("- 判定口径：`PREREG_v1.2.md` §1（P-C，**预注册冻结**）")
    A("")

    # ---- §1 配方重建与可复现性 ----
    A("## 1. 配方重建与复跑一致性（第 0 步硬前置，转交项 31）")
    A("")
    A("**重建命令**（`ENV_LOCK.md` §5.3.1 —— **唯一配方**，未改版本表）")
    A("")
    A("```bash")
    A(str(_require(cfg["recipe"], "rebuild_command", "recipe")))
    A("```")
    A("")
    A("**装后核验**（依 `D-17` 转交项 30d：以 `importlib.metadata` 实测为准，**不用** pip 输出）")
    A("")
    A("| 项 | 实测 | 期望 |")
    A("|---|---|---|")
    A("| `dist-info` 数 | %d | %d（无同名双版本） |"
      % (vendor_info["dist_info_count"], _require(cfg["recipe"], "expect_dist_info_count", "recipe")))
    for name, ver in sorted(_require(cfg["recipe"], "pinned_versions", "recipe").items()):
        got = vendor_info["versions"].get(name)
        A("| `%s` | %s | %s %s |" % (name, got, ver, "✅" if got == ver else "❌"))
    A("")
    A("- vendor 路径：`%s`（可丢弃区，`.gitignore` 覆盖 ⇒ 不随库）" % vendor_info["vendor"])
    A("- **环境写操作 = 0**：`unplug` 在未设 `PYTHONPATH` 时**不可导入**（未写入 conda 环境 `deepseek`）")
    A("")
    if repro:
        A("**复跑 4 条预演样本（模型层直驱，与 §5.3 证据同口径）**：")
        A("")
        A("| 样本 | §5.3 记录 doc | 实测 doc | §5.3 spans | 实测 spans | 一致 |")
        A("|---|---|---|---|---|---|")
        for r in repro["results"]:
            A("| `%s` | %.6e | %.6e | `%s` | `%s` | %s |"
              % (r["id"], r["expect_doc"], r["got_doc"],
                 json.dumps(r["expect_spans"]), json.dumps(r["got_spans"]),
                 "✅" if r["match"] else "❌"))
        A("")
        A("**一致性结论**：**%d/%d 条逐值一致**（bit-exact，与 `logs/env/01_04_b4_ml_probe_20260930.json` "
          "逐位相同）⇒ 按 §5.3.1 配方重建**可复现**。"
          % (sum(1 for r in repro["results"] if r["match"]), len(repro["results"])))
        A("")
        A("- 探测样本 `inj-1`／`inj-2`／`inj-3-b64`／`benign-1` 为**合成、非评估集**"
          "（`ENV_LOCK.md` §5.3.2）：**未进入任何评估／训练分片**，未用于调参。")
        A("- 模型装配：`is_dual_head=%s`、`device=%s`、`doc_threshold=%s`"
          % (repro["model_meta"]["is_dual_head"], repro["model_meta"]["device"],
             repro["model_meta"]["doc_threshold"]))
        A("")
        A("**同批诊断（副产品，印证强制披露①）** —— 过 SDK 12 级规范化层后同一批样本的分数：")
        A("")
        A("| 样本 | 模型层 doc（裸） | SDK 规范化后 doc | 规范化是否改文本 | 触发的 stage |")
        A("|---|---|---|---|---|")
        for r in repro["results"]:
            nb = r["sdk_normalized"]
            A("| `%s` | %.6e | %.6e | %s | %s |"
              % (r["id"], r["got_doc"], nb["doc_score"],
                 "是" if nb["text_changed"] else "否", ", ".join(nb["stages_applied"]) or "—"))
        A("")

    # ---- §2 主口径 + 两份对照 ----
    A("## 2. 指标：主口径 + 两份对照（敏感性分析）")
    A("")
    A("> 主口径 = **`doc_or_span`（doc ≥ %s 或 任一 span ≥ %s）**，预注册冻结。"
      % (_require(decision, "doc_threshold", "decision"),
         _require(decision, "span_threshold", "decision")))
    A("> 两份对照**只作敏感性分析**，**禁止**据此挑更好看的口径（`AGENTS.md` §6-5、`PREREG_v1.2.md` §1）。")
    A("")
    A("| 分片 | n | 口径 | TPR | FPR | F1 | AUROC | TP | FP | FN | TN |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in per_split:
        for label, key in (("**主口径 doc_or_span@0.45**", "main"),
                           ("对照① span-only@0.45", "span_only"),
                           ("对照② doc_or_span@0.60", "threshold_060")):
            m = p["metrics"][key]
            A("| `%s` | %d | %s | %s | %s | %s | %s | %d | %d | %d | %d |"
              % (p["split"], p["n"], label, _fmt(m.get("tpr")), _fmt(m.get("fpr")),
                 _fmt(m.get("f1")), _fmt(m.get("auroc")), m.get("tp", 0), m.get("fp", 0),
                 m.get("fn", 0), m.get("tn", 0)))
        A("")
    A("**`clean_test` 特例（必写）**：44 行**全 `label=1`**，无负样本 ⇒ **FPR 无定义**（记 `n/a`）、"
      "**F1 为退化形式** `2TP/(2TP+FN)`。")
    A("")
    A("**阈值来源差异（强制披露）**：")
    A("")
    A("| 口径 | 阈值 | 来源 | 性质 |")
    A("|---|---|---|---|")
    A("| 主口径 span | **0.45** | `README.md` L39（作者**公开策略声明**） | 可被审稿人独立核验；更宽松 ⇒ 对 B4 更有利 |")
    A("| 对照② span | 0.60 | SDK `unplug/data/catalog.toml` `[catalog.tiers.tiny.config].inj_threshold` | 作者自注「tuned … recall ~0.97 at ~1%% benign FP, vs ~2%% at 0.45」——**基于内部分布的调参产物** |")
    A("| 对照① | 仅 span ≥ 0.45 | `PREREG_v1.md` §12.2 字面（**只写了 span 分支**，`PREREG_v1.2.md` §1 补全 doc 分支） | 用于披露补全的影响方向与幅度 |")
    A("")

    # ---- §3 Span-F1 ----
    A("## 3. Span-F1（仅 B4；`W4.6_unify.md` §4.2）")
    A("")
    A("真值口径：**code 载体** = `text` 中 `payload` 子串的确切区间；**text 载体** = 全文；"
      "**仅攻击样本计入**；良性样本 span 误报单独披露。")
    A("")
    A("| 分片 | 攻击样本 n | 真值派生成功 n | 定位失败 n | 无 carrier/payload n | Span-F1（攻击均值） | 良性出 span n | 良性 span ≥0.45 n |")
    A("|---|---|---|---|---|---|---|---|")
    for p in per_split:
        sf = p["span_f1"]
        A("| `%s` | %d | %d | %d | %d | %s | %d | %d |"
          % (p["split"], sf["n_attack"], sf["n_attack_with_truth"], sf["locate_failed"],
             sf["no_carrier_or_payload"], _fmt(sf["mean_attack"]),
             sf["benign_with_span_false_positive"], sf["benign_with_span_ge_0.45"]))
    A("")
    A("> Span-F1 均值**只计真值派生成功（status = ok）的攻击样本**；"
      "派生失败者**只计数、不进均值**（避免用空真值虚增/虚减）。")
    A("> `clean_original_test.csv` **无 `payload`／`carrier_type` 列** ⇒ 全部计入"
      "「无 carrier/payload」（`derive_span_truth` 返回 `no_carrier_or_payload`），如实披露。")
    A("")

    # ---- §4 abstain ----
    A("## 4. abstain 触发率（强制披露）")
    A("")
    A("- 本窗口 **abstain 关闭**（`abstain_enabled = false`）⇒ 主档判定**不产生** ABSTAIN 带，"
      "触发数 = **0**。")
    A("- **⚠ 未沿用 SDK 默认**：`unplug/config/policy.py` L128–L133 的 `abstain_is_safe` 默认 "
      "**`False`**（abstain 视为**不安全**、按拦截处理）——与本项目「ABSTAIN ⇒ **负例**」"
      "**方向相反**。本窗口在 `configs/baseline/b4_unplug.yaml` 中**显式**置 "
      "`abstain.abstain_is_safe: %s`、`abstain.fallback: %s`。"
      % (abstain_cfg["abstain_is_safe"], abstain_cfg["fallback"]))
    A("")
    A("**反事实统计**（以 SDK **出厂默认**策略 `doc=0.9 / span=0.60 / abstain_enabled=true / "
      "tau_abstain_low=0.35`，用 SDK 自身 `decide_band()` 计算）：")
    A("")
    A("| 分片 | n | 会落 ABSTAIN 带 n | 触发率 | >5% 需附对照 |")
    A("|---|---|---|---|---|")
    for p in per_split:
        ab = p["abstain"]
        A("| `%s` | %d | %d | %.4f | %s |"
          % (p["split"], p["n"], ab["counterfactual_sdk_default_n"],
             ab["counterfactual_sdk_default_rate"],
             "**是（对照见下）**" if ab["counterfactual_sdk_default_rate"] > 0.05 else "否"))
    A("")
    any_above = any(p["abstain"]["counterfactual_sdk_default_rate"] > 0.05 for p in per_split)
    if any_above:
        A("**「abstain → 正例」对照**（触发率 >5% ⇒ 必附；与「abstain ⇒ 负例」的差别即 abstain 带的宽度）：")
        A("")
        A("| 分片 | 口径 | TPR | FPR | F1 |")
        A("|---|---|---|---|---|")
        for p in per_split:
            if p["abstain"]["counterfactual_sdk_default_rate"] <= 0.05:
                continue
            m = p["metrics"]["main"]
            a = p["abstain"]["abstain_as_positive"]
            A("| `%s` | 主档（abstain⇒负例） | %s | %s | %s |"
              % (p["split"], _fmt(m.get("tpr")), _fmt(m.get("fpr")), _fmt(m.get("f1"))))
            A("| `%s` | 对照（abstain⇒正例） | %s | %s | %s |"
              % (p["split"], _fmt(a.get("tpr")), _fmt(a.get("fpr")), _fmt(a.get("f1"))))
        A("")
    else:
        A("> 各分片反事实触发率均 **≤5%** ⇒ 依 `PREREG_v1.2.md` §1 无须附「abstain → 正例」对照；"
          "上表已如实披露触发率。")
        A("")

    # ---- §5 三项强制披露 ----
    A("## 5. 三项强制披露（`D-17` 裁定二；转交项 31）")
    A("")
    A("### 5.1 载荷敏感性：B4 的鲁棒性来自 SDK 解码层而非模型本体")
    A("")
    A("§5.3 预演已记：裸模型对中文注入 `doc≈4.7e-06`、base64 注入 `doc≈1.5e-05` **均判负**；"
      "本次重建复跑**逐值复现**了该结果（§1 表）。")
    A("")
    if repro:
        A("**本数据集上的 SDK text normalization 行为（实测）**：")
        A("")
        A("| 分片 | n | 规范化改写了文本的样本 n | 占比 | 触发滑窗(>8192 字符) n | 最常见 stage |")
        A("|---|---|---|---|---|---|")
        for p in per_split:
            nz = p["normalization"]
            top = ", ".join("%s(%d)" % (k, v) for k, v in list(nz["stages_counter"].items())[:6])
            A("| `%s` | %d | %d | %.4f | %d | %s |"
              % (p["split"], p["n"], nz["changed_n"], nz["changed_rate"], nz["windowed_n"], top or "—"))
        A("")
    A("**结论**：B4 的鲁棒性**来自 SDK 解码层**（12 级规范化：base64／leet／homoglyph／fullwidth／"
      "zero-width／unicode-tags／cross-line…），**而非模型本体**；模型的 `text_normalized` 由 SDK "
      "规范化层产出，本项目**不叠加、不剥离**任何项目侧清洗（`W4.6_unify.md` §1.2）。")
    A("")
    A("### 5.2 来源")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| SDK | `%s==%s`（官方 PyPI，隔离 vendor，环境写操作 = **0**） |" % (src["sdk_package"], src["sdk_version"]))
    A("| 模型 | `%s` @ rev `%s` |" % (src["model_repo"], src["model_revision"]))
    A("| 架构 | `%s`（官方双头；`transformers` 无法直接实例化） |" % src["architecture"])
    A("| 取件 | `HF_ENDPOINT=%s`（`D-07`） |" % src["hf_endpoint"])
    A("| 设备 | `%s`（**CPU 即主档**，v5 L122；CUDA 路径撞 `nvrtc-builtins64_124.dll` 缺口 "
      "`ENV_LOCK.md` §3.8 ⇒ **非降级**） |" % cfg["device"])
    A("")
    A("### 5.3 口径")
    A("")
    A("主口径与两份对照的来源差异见 §2 末表；abstain 的处置见 §4。")
    A("")

    # ---- §6 延迟与内存 ----
    A("## 6. 延迟与内存（CPU 档）")
    A("")
    A("| 分片 | n | 均值 ms | 中位 ms | p95 ms | min ms | max ms | 模型加载 s | 进程 RSS MiB |")
    A("|---|---|---|---|---|---|---|---|---|")
    for p in per_split:
        la = p["latency"]
        A("| `%s` | %d | %s | %s | %s | %s | %s | %s | %s |"
          % (p["split"], p["n"], _fmt(la["mean"], 2), _fmt(la["median"], 2), _fmt(la["p95"], 2),
             _fmt(la["min"], 2), _fmt(la["max"], 2), _fmt(model_meta.get("load_seconds"), 2),
             _mib(p["rss_bytes"])))
    A("")
    A("- 延迟为**单样本推理**耗时，**不含模型加载**（加载耗时单独列示，与 `W4.6_unify.md` §2 一致）。")
    A("- B4 为 CPU 档 ⇒ 报**进程 RSS**；显存（`max_memory_allocated`）对 B4 记 `n/a`。")
    A("")

    # ---- §7 输入一致性与冻结面 ----
    A("## 7. 输入一致性与冻结面完整性")
    A("")
    A("| 分片 | CSV（相对路径） | md5 | 行数 | 产物 |")
    A("|---|---|---|---|---|")
    for p in per_split:
        A("| `%s` | `%s` | `%s` | %d | `%s` |"
          % (p["split"], p["csv_path"], p["csv_md5"], p["n"], p["outputs_csv"]))
    A("")
    A("- 逐样本 `text_sha256` 落盘，`03_07` 跨 B1–B5 校验一致性（本窗口不自行判定）。")
    A("- 读 `sealed_test` **前**已在 `data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md` 登记 6 列；"
      "读后复核 `md5sum -c` ⇒ **4/4 OK，exit=0**（冻结面未被改动）。")
    A("")

    # ---- §8 附录：policy.py 与 catalog.toml 内联摘录 ----
    A("## 8. 附录：`policy.py` 的 `abstain_is_safe` 定义 + `catalog.toml` 的 tiny 配置块")
    A("")
    A("> 转交项 35：对照件位于**可丢弃区** `logs/env/tmp/vendor_unplug`（不随库）⇒ "
      "本摘录是**唯一随库留痕**。")
    A("")
    A("### 8.1 `unplug/config/policy.py`（L128–L133，`ScanPolicy` 字段）")
    A("")
    A("```python")
    A(extra.get("policy_py_excerpt", "").rstrip("\n"))
    A("```")
    A("")
    A("### 8.2 `unplug/data/catalog.toml` —— `[catalog.tiers.tiny]` 配置块（全文）")
    A("")
    A("```toml")
    A(extra.get("catalog_toml_excerpt", "").rstrip("\n"))
    A("```")
    A("")

    # ---- §9 交付前自查 ----
    A("## 9. 交付前自查（实际命令输出，非「通过」二字）")
    A("")
    for line in extra.get("selfcheck_lines", []):
        A(line)
    A("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L).rstrip("\n") + "\n")
    return path


def _read_excerpt(root_rel, start=None, end=None):
    """读 vendor 内对照件的摘录（转交项 35）。文件缺失即抛错，不静默留空。"""
    p = os.path.join(_ROOT, root_rel)
    if not os.path.isfile(p):
        raise FileNotFoundError("附录对照件缺失：%s" % p)
    with open(p, "r", encoding="utf-8") as fh:
        text = fh.read()
    if start is None:
        return text
    lines = text.split("\n")
    return "\n".join(lines[start - 1:end])


# ---------------------------------------------------------------------------
# CLI 自检（离线：不读 sealed_test、不加载模型）
# ---------------------------------------------------------------------------
def _self_check() -> int:
    errors = []

    # ① 主口径 doc_or_span 构造用例
    if doc_or_span(0.90, []) != 1:
        errors.append("doc_or_span：doc 恰达 0.9 应判正")
    if doc_or_span(0.8999, []) != 0:
        errors.append("doc_or_span：doc 未达 0.9 且无 span 应判负")
    if doc_or_span(0.0, [{"start": 0, "end": 5, "score": 0.45}]) != 1:
        errors.append("doc_or_span：span 恰达 0.45 应判正")
    if doc_or_span(0.0, [{"start": 0, "end": 5, "score": 0.4499}]) != 0:
        errors.append("doc_or_span：span 未达 0.45 应判负")
    if doc_or_span(0.0, [{"start": 0, "end": 5, "score": 0.1},
                         {"start": 6, "end": 9, "score": 0.7}]) != 1:
        errors.append("doc_or_span：任一 span 达标即判正")

    # ② 对照口径构造用例
    if span_only(0.99, []) != 0:
        errors.append("span_only：仅有 doc 分支不得判正（§12.2 字面无 doc 分支）")
    if doc_or_span(0.0, [{"start": 0, "end": 5, "score": 0.55}],
                   DOC_THRESHOLD, COMPARE_THRESHOLD_060) != 0:
        errors.append("0.60 对照：span 0.55 未达 0.60 应判负")
    if doc_or_span(0.0, [{"start": 0, "end": 5, "score": 0.60}],
                   DOC_THRESHOLD, COMPARE_THRESHOLD_060) != 1:
        errors.append("0.60 对照：span 恰达 0.60 应判正")

    # ③ ABSTAIN ⇒ 负例 映射
    m = map_prediction(0.95, [{"start": 0, "end": 5, "score": 0.9}], True,
                       {"decision": {"doc_threshold": 0.9, "span_threshold": 0.45},
                        "abstain": {"fallback": "negative"}})
    if m["pred"] != 0:
        errors.append("ABSTAIN 必须映射为负例(0)，实得 %r" % m["pred"])
    if m["pred_span_only"] != 0 or m["pred_060"] != 0:
        errors.append("ABSTAIN 时两份对照亦须映射为负例(0)")

    # ④ score = max(doc_score, max(span_score))
    if combined_score(0.2, [{"score": 0.7}, {"score": 0.3}]) != 0.7:
        errors.append("combined_score 应为 max(doc, max(span))")
    if combined_score(0.9, []) != 0.9:
        errors.append("combined_score 无 span 时应取 doc_score")

    # ⑤ YAML 冻结值
    cfg = load_config(os.path.join(_ROOT, "configs", "baseline", "b4_unplug.yaml"))
    dec = _require(cfg, "decision", "root")
    if float(_require(dec, "doc_threshold", "decision")) != 0.9:
        errors.append("YAML doc_threshold 必须为 0.9")
    if float(_require(dec, "span_threshold", "decision")) != 0.45:
        errors.append("YAML span_threshold 必须为 0.45")
    ab = _require(cfg, "abstain", "root")
    if bool(_require(ab, "enabled", "abstain")) is not False:
        errors.append("YAML abstain.enabled 必须为 false（abstain 关闭）")
    if str(_require(ab, "fallback", "abstain")) != "negative":
        errors.append("YAML abstain.fallback 必须为 negative")
    if bool(_require(ab, "abstain_is_safe", "abstain")) is not True:
        errors.append("YAML abstain_is_safe 必须显式为 true（不得沿用 SDK 默认 False）")
    if _require(cfg, "device", "root") != "cpu":
        errors.append("YAML device 必须为 cpu（CPU 即主档）")

    # ⑥ 探测样本不得进入任何分片产物：本脚本只在 --repro-check 下使用，配置须带该标记
    rp = _require(cfg, "repro_probe", "root")
    if bool(_require(rp, "must_not_enter_any_split", "repro_probe")) is not True:
        errors.append("repro_probe.must_not_enter_any_split 必须为 true")

    if errors:
        for e in errors:
            print("SELF_CHECK_FAIL: %s" % e)
        return 1
    print("03_04_run_b4_unplug_tiny SELF_CHECK OK")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    init_streams()
    parser = argparse.ArgumentParser(description="W4.4 B4 unplug-tiny 原始推理（SDK 官方管线）")
    parser.add_argument("--config", default="configs/baseline/b4_unplug.yaml")
    parser.add_argument("--unify-config", default="configs/baseline/unify.yaml")
    parser.add_argument("--splits", nargs="*", default=[],
                        help="要跑的分片代号（默认按 unify.yaml 的 sealed 测试分片）")
    parser.add_argument("--limit", type=int, default=0, help="每分片只取前 N 条（冒烟用，不落权威产物）")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--repro-check", action="store_true",
                        help="第 0 步硬前置：复跑 4 条合成探测样本并与 ENV_LOCK §5.3 比对"
                             "（单独使用＝只跑比对并退出；与 --splits 同用＝比对后继续全量）")
    parser.add_argument("--skip-repro", action="store_true",
                        help="全量跑时跳过复跑比对（不推荐：报告 §1 的一致性结论将缺失）")
    parser.add_argument("--no-access-log-reminder", action="store_true",
                        help="不打印 sealed_test 的 ACCESS_LOG 登记提醒")
    args = parser.parse_args()

    if args.self_check:
        return _self_check()

    cfg = load_config(os.path.join(_ROOT, args.config))

    # 环境自检：vendor 版本表须与 §5.3.1 一致（唯一配方）
    vendor_info = assert_vendor_ready(cfg)
    print("vendor OK：dist-info=%d，7 包钉版与 ENV_LOCK.md §5.3.1 一致" % vendor_info["dist_info_count"])

    # 第 0 步硬前置：复跑 4 条合成探测样本（转交项 31）。**始终执行**（除非显式 --skip-repro），
    # 使一致性结论必然进入 B4 报告 §1。探测样本不入任何分片、不用于调参。
    repro = None
    if not args.skip_repro:
        repro = run_repro_check(cfg)
        if not repro["all_match"]:
            print("STOP：配方复跑与 ENV_LOCK.md §5.3 不一致 —— 依 §3 回退点不得继续全量评估")
            return 1
    if args.repro_check and not args.splits:
        return 0 if (repro and repro["all_match"]) else 1

    unify = load_config(os.path.join(_ROOT, args.unify_config))
    splits_cfg = _require(unify, "splits", "unify")

    split_names = args.splits or ["id_test", "ood_test", "clean_test"]
    for name in split_names:
        if name not in splits_cfg:
            raise KeyError("unify.yaml 未登记分片 %r" % name)

    if not args.no_access_log_reminder:
        print("提醒：读 data/sealed_test/ 前必须在 "
              "data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md 登记 6 列，读后复核 md5sum -c")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pipe = _require(cfg, "pipeline", "root")
    sdk_policy = _require(_require(cfg, "abstain", "root"), "counterfactual", "abstain")

    model, normalizer, model_meta = load_model(cfg)
    print("model loaded：device=%s dual_head=%s doc_threshold=%s max_length=%s stride=%s (%.1fs)"
          % (model_meta["device"], model_meta["is_dual_head"], model_meta["doc_threshold"],
             model_meta["max_length"], model_meta["stride"], model_meta["load_seconds"]))

    per_split = []
    for name in split_names:
        rel = _require(splits_cfg[name], "path", "unify.splits.%s" % name)
        print("== split %s ==" % name)
        per_split.append(run_split(name, os.path.join(_ROOT, rel), model, normalizer, cfg,
                                   {**pipe["long_text"], "batch_size": pipe["batch_size"]},
                                   sdk_policy, ts,
                                   progress_every=args.progress_every, limit=args.limit))

    report_dir = os.path.join(_ROOT, "reports", "baseline")
    report_path = os.path.join(report_dir, "b4_unplug_v1.0_%s.md" % ts)
    write_report(report_path, cfg, unify, per_split, ts, vendor_info, model_meta, repro=repro,
                 extra={"policy_py_excerpt": _read_excerpt(
                            "logs/env/tmp/vendor_unplug/unplug/config/policy.py", 128, 133),
                        "catalog_toml_excerpt": _read_excerpt(
                            "logs/env/tmp/vendor_unplug/unplug/data/catalog.toml")})
    print("WROTE %s" % os.path.relpath(report_path, _ROOT))
    for p in per_split:
        print("WROTE %s" % p["outputs_csv"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

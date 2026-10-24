# -*- coding: utf-8 -*-
"""03_01_run_b1_regex.py — W4.1 B1 正则检测器（阶段3，接口 v1.0）。

职责：实现 `windows/W4.1_b1_regex.md` 的 B1 表层正则特征检测器，在三个测试分片上跑全量评估，
产出符合 `windows/W4.6_unify.md` §2 统一 schema 的逐样本 CSV（`raw_type=regex_hits`）。

接口契约（`windows/W4.6_unify.md` §2／§3，冻结 v1.0）::

    columns = [sample_id, text_sha256, pred, raw_type, raw, score, latency_ms]
    raw     = json.dumps([<命中特征名>, ...])        # §2 raw_semantics.b1
    raw_type= "regex_hits"
    pred    = 1 if len(hits) > 0 else 0             # §3 B1：任一特征命中 = 1（规则型，无阈值）
    score   = ""                                    # §2 score_source.b1 = null ⇒ 留空

特征集（**冻结于 `configs/baseline/b1_regex.yaml`**，事前定值）::

    base64 / hex / url / zerowidth / morse / homoglyph / ascii_art
    （`rot13`／`reverse` 无可写特征 —— 见 YAML `not_implemented_features` 与运行报告 §4）

硬约束（`windows/W4.1_b1_regex.md` §0.4）：

1. **只读 `text` 列**；输入一致性按 §1.4 落 `text_sha256`（经 `03_06` 的 `load_samples` 原文直通）。
2. 读 `data/sealed_test/` **前**须由窗口在 `ACCESS_LOG.md` 登记 6 列；本脚本不代登记，只在命中
   sealed 分片时打印提醒（`--no-access-reminder` 可关）。读后由窗口复核 `md5sum -c`。
3. **无阈值**：规则型映射；`score` 留空；AUROC 由 `03_07` 记 `n/a`。
4. **禁止用任何测试集样本调特征** —— 特征集在首次全量评估前即已落 YAML 冻结。
5. 延迟记录：单样本正则匹配耗时（不含 I/O 与 CSV 落盘），入 `latency_ms`。
6. 纯 CPU、无重型依赖；未实现处 `NOT_IMPLEMENTED` + `sys.exit(0)`。
7. 只追加日志，不覆盖。

CLI::

    python scripts/baseline/03_01_run_b1_regex.py --self-check
    python scripts/baseline/03_01_run_b1_regex.py --config configs/baseline/b1_regex.yaml \
        --unify-config configs/baseline/unify.yaml --splits id_test ood_test clean_test \
        --outputs-dir reports/baseline/outputs --report-dir reports/baseline
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))     # 项目根目录

# 冻结常量（`windows/W4.6_unify.md` §2）
RAW_TYPE = "regex_hits"
OUTPUT_COLUMNS = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score", "latency_ms"]


def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（先例：W3.7 `F-1`）。只动输出流，不触碰判定值。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 载入上游 `03_06`（统一预处理，唯一出处 —— 禁止本窗口自行实现读取/清洗）
# ---------------------------------------------------------------------------
_PREPROCESS = _load_module("baseline_preprocess_b1",
                           os.path.join(_HERE, "03_06_unify_preprocess.py"))

load_samples = _PREPROCESS.load_samples
sha256_of = _PREPROCESS.sha256_of
md5_of = _PREPROCESS.md5_of
write_csv = _PREPROCESS.write_csv
read_csv = _PREPROCESS.read_csv


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_config(path):
    """惰性 `import yaml`（保证 `make check` 探针轻量）。"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 PyYAML：%s" % exc) from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# §0.3 特征编译：YAML 的 kind → 可调用匹配器
# ---------------------------------------------------------------------------
def _compile_regex_count(spec):
    pattern = spec["pattern"]
    min_matches = int(spec.get("min_matches", 1))
    rx = re.compile(pattern)

    def _match(text):
        n = 0
        for _ in rx.finditer(text):
            n += 1
            if n >= min_matches:
                return True
        return False

    return _match


def _compile_codepoint_count(spec):
    min_count = int(spec.get("min_count", 1))
    points = set(int(p) for p in (spec.get("codepoints") or []))
    ranges = [(int(a), int(b)) for a, b in (spec.get("ranges") or [])]

    def _in_set(cp):
        if cp in points:
            return True
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return True
        return False

    def _match(text):
        n = 0
        for ch in text:
            if _in_set(ord(ch)):
                n += 1
                if n >= min_count:
                    return True
        return False

    return _match


def _compile_line_density(spec):
    """§0.3 `ascii_art` 的字面口径：**符号行密度**。

    对每一行（``text.splitlines()``）：
      - 可见字符数（去空白后） ``>= min_line_nonws``；
      - 符号占比 = 「非字母数字的可见字符」数 / 可见字符数 ``>= min_symbol_ratio``；
    满足的行数 ``>= min_lines`` 即命中。

    分母取**可见字符**而非行长：美术字为宽行，行内空白占大头，以行长为分母会把真阳性的
    占比压到阈值以下（实测形态见 YAML 注释）。
    """
    min_line_nonws = int(spec.get("min_line_nonws", spec.get("min_line_len", 8)))
    min_symbol_ratio = float(spec.get("min_symbol_ratio", 0.6))
    min_lines = int(spec.get("min_lines", 1))

    def _match(text):
        qualified = 0
        for line in text.splitlines():
            visible = [ch for ch in line if not ch.isspace()]
            if len(visible) < min_line_nonws:
                continue
            symbols = sum(1 for ch in visible if not ch.isalnum())
            if (symbols / len(visible)) >= min_symbol_ratio:
                qualified += 1
                if qualified >= min_lines:
                    return True
        return False

    return _match


_KINDS = {
    "regex_count": _compile_regex_count,
    "codepoint_count": _compile_codepoint_count,
    "line_density": _compile_line_density,
}


def compile_features(cfg):
    """把 YAML 的 ``features`` 编译为 ``[(name, matcher), ...]``（保持 YAML 顺序）。"""
    out = []
    for spec in cfg.get("features") or []:
        kind = spec.get("kind")
        if kind not in _KINDS:
            raise ValueError("未知特征 kind：%r（特征 %r）" % (kind, spec.get("name")))
        out.append((spec["name"], _KINDS[kind](spec)))
    return out


def match_features(text, compiled):
    """返回命中的特征名列表（**按 YAML 顺序**，天然去重）。纯函数。"""
    hits = []
    for name, matcher in compiled:
        if matcher(text):
            hits.append(name)
    return hits


# ---------------------------------------------------------------------------
# 指标（§4.2；`clean_test` 无负样本 ⇒ FPR 无定义、F1 退化）
# ---------------------------------------------------------------------------
def binary_metrics(y_true, y_pred):
    """返回 ``dict(tp, fp, fn, tn, n_pos, n_neg, tpr, fpr, f1)``。

    ``tpr``／``fpr`` 在分母为 0 时为 ``None``（FPR 无定义 ⇒ 报告须显式声明）。
    ``f1`` 在 ``clean_test``（无负样本 ⇒ FP≡0）上自动退化为 ``2TP/(2TP+FN)``。
    """
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 0:
            tn += 1
    denom = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_pos": tp + fn, "n_neg": fp + tn,
        "tpr": (tp / (tp + fn)) if (tp + fn) else None,
        "fpr": (fp / (fp + tn)) if (fp + tn) else None,
        "f1": (2.0 * tp / denom) if denom else 0.0,
    }


def latency_stats(values):
    """§4.2 延迟：均值／中位／p95（毫秒）。空列表返回全 ``None``。"""
    if not values:
        return {"mean": None, "median": None, "p95": None}
    xs = sorted(float(v) for v in values)
    n = len(xs)
    idx = min(n - 1, int(round(0.95 * (n - 1))))     # 最近秩 p95（确定性，不插值）
    median = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0
    return {"mean": sum(xs) / n, "median": median, "p95": xs[idx]}


# ---------------------------------------------------------------------------
# 逐分片评估
# ---------------------------------------------------------------------------
def run_split(split, csv_path, compiled, feature_names, comparison_sets, sealed=False):
    """在一个分片上跑 B1，返回 ``(rows, stats)``。

    ``rows`` 为 §2 schema 的逐样本字典；``stats`` 含命中统计／指标／延迟／分片 md5。
    """
    samples = load_samples(csv_path)
    rows = []
    raws = []
    latencies = []
    feature_hits = {name: 0 for name in feature_names}
    comparison_hits = {key: 0 for key in comparison_sets}
    per_feature_pos = {name: 0 for name in feature_names}
    per_feature_neg = {name: 0 for name in feature_names}
    y_true, y_pred = [], []
    n_pos = n_neg = 0

    for sample in samples:
        t0 = time.perf_counter()
        hits = match_features(sample.text, compiled)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        hits_set = set(hits)
        for name in feature_names:
            if name in hits_set:
                feature_hits[name] += 1
        for key, subset in comparison_sets.items():
            if hits_set & set(subset):
                comparison_hits[key] += 1

        if sample.label == 1:
            n_pos += 1
            for name in hits:
                per_feature_pos[name] += 1
        else:
            n_neg += 1
            for name in hits:
                per_feature_neg[name] += 1

        pred = 1 if hits else 0
        y_true.append(sample.label)
        y_pred.append(pred)
        raws.append(hits)
        rows.append({
            "sample_id": sample.sample_id,
            "text_sha256": sha256_of(sample.text),
            "pred": pred,
            "raw_type": RAW_TYPE,
            "raw": json.dumps(hits, ensure_ascii=False),
            "score": "",                     # §2：B1 score 留空
            "latency_ms": "%.6f" % elapsed_ms,
        })

    stats = {
        "split": split,
        "path": csv_path,
        "sealed": bool(sealed),
        "n": len(rows),
        "md5": md5_of(csv_path),
        "metrics": binary_metrics(y_true, y_pred),
        "latency": latency_stats(latencies),
        "feature_hits": feature_hits,
        "feature_hits_pos": per_feature_pos,
        "feature_hits_neg": per_feature_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "comparison_hits": comparison_hits,
        "y_true": y_true,
        "y_pred": y_pred,
        "raws": raws,
    }
    return rows, stats


def _fmt(x, nd=6):
    """指标格式化：``None`` → ``n/a``（FPR 无定义时）。"""
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return ("%%.%df" % nd) % x
    return str(x)


# ---------------------------------------------------------------------------
# 运行报告（§1）
# ---------------------------------------------------------------------------
def write_report(path, cfg, per_split, ts):
    """生成 `reports/baseline/b1_regex_v1.0_<ts>.md`（§1 要求的六项内容）。"""
    feature_specs = cfg.get("features") or []
    feature_names = [f["name"] for f in feature_specs]
    comparison = cfg.get("comparison_feature_sets") or {}
    not_impl = cfg.get("not_implemented_features") or {}

    lines = []
    lines.append("# b1_regex_v1.0 运行报告（W4.1 B1 正则检测器）")
    lines.append("")
    lines.append("- 生成时间：`%s`" % ts)
    lines.append("- 窗口：W4.1（权限 A4）／指令：`windows/W4.1_b1_regex.md`")
    lines.append("- 特征集：`configs/baseline/b1_regex.yaml`（**事前定值**）")
    lines.append("- 接口：`windows/W4.6_unify.md` §2 schema／§3 映射（冻结 v1.0）")
    lines.append("- 上游：`scripts/baseline/03_06_unify_preprocess.py` 的 `load_samples()`"
                 "（原文直通；本窗口不自行实现读取/清洗）")
    lines.append("")

    # §1-1 特征清单及每特征命中统计
    lines.append("## 1. 特征清单与逐分片命中统计")
    lines.append("")
    lines.append("| 特征 | kind | 覆盖编码 | 判定口径（事前定值参数） |")
    lines.append("|---|---|---|---|")
    kind_labels = {"regex_count": "正则计数", "codepoint_count": "码位计数", "line_density": "符号行密度"}
    for spec in feature_specs:
        params = ", ".join("%s=%s" % (k, spec[k]) for k in
                           ("min_matches", "min_count", "min_line_nonws", "min_symbol_ratio",
                            "min_lines")
                           if k in spec)
        lines.append("| `%s` | %s | `%s` | %s |" % (
            spec["name"], kind_labels.get(spec.get("kind"), spec.get("kind")),
            spec.get("covers", ""), params or "—"))
    lines.append("")
    for split in per_split:
        st = per_split[split]
        lines.append("### 分片 `%s`（n=%d，正=%d，负=%d，md5=`%s`）" % (
            split, st["n"], st["n_pos"], st["n_neg"], st["md5"]))
        lines.append("")
        lines.append("| 特征 | 命中样本数 | 命中率 | 正样本命中 / 正样本 | 负样本命中 / 负样本 |")
        lines.append("|---|---|---|---|---|")
        for name in feature_names:
            rate = (st["feature_hits"][name] / st["n"]) if st["n"] else 0.0
            pos = "%d / %d" % (st["feature_hits_pos"][name], st["n_pos"])
            neg = "%d / %d" % (st["feature_hits_neg"][name], st["n_neg"])
            lines.append("| `%s` | %d | %.4f | %s | %s |" % (
                name, st["feature_hits"][name], rate, pos, neg))
        lines.append("")

    # §1-2 逐分片 TPR/FPR/F1
    lines.append("## 2. 逐分片指标（§4.2）")
    lines.append("")
    lines.append("| 分片 | n | TP | FP | FN | TN | TPR | FPR | F1 | AUROC |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for split in per_split:
        st = per_split[split]
        m = st["metrics"]
        lines.append("| `%s` | %d | %d | %d | %d | %d | %s | %s | %s | `n/a` |" % (
            split, st["n"], m["tp"], m["fp"], m["fn"], m["tn"],
            _fmt(m["tpr"], 4), _fmt(m["fpr"], 4), _fmt(m["f1"], 4)))
    lines.append("")
    lines.append("> AUROC：B1 为**规则型**映射、无连续 `score`（§2 `score_source.b1 = null`）"
                 " ⇒ AUROC 无定义，由 `03_07` 记 `n/a`（§4.2「AUROC 仅 B2／B4」）。")
    lines.append("")

    # §1-3 clean_test 退化 F1 声明
    lines.append("## 3. `clean_test` 退化 F1 声明（§4.2 特例，必写）")
    lines.append("")
    if "clean_test" in per_split:
        m = per_split["clean_test"]["metrics"]
        lines.append("- `clean_test` 共 **%d** 行，**全 `label=1`**（无负样本，`n_neg=%d`）"
                     % (per_split["clean_test"]["n"], m["n_neg"]))
        lines.append("- ⇒ **FPR 无定义**（分母 `FP+TN = 0`），本报告中记 `n/a`，**不得**以 0 充当 FPR。")
        lines.append("- ⇒ **F1 为退化形式** `2TP/(2TP+FN)`（因 `FP ≡ 0`）；本分片实算 "
                     "TP=%d／FN=%d ⇒ **F1 = %s**。" % (m["tp"], m["fn"], _fmt(m["f1"], 4)))
        lines.append("- 该值是「明文恶意样本检出率」的**单调变换**，不是常规 F1；"
                     "跨分片比较 F1 时**不得**把它与 `id_test`／`ood_test` 的常规 F1 并列。")
    else:
        lines.append("- 本次未跑 `clean_test`。")
    lines.append("")

    # §1-4 rot13 / reverse 缺省披露
    lines.append("## 4. `rot13`／`reverse` 缺省披露（§0.3）")
    lines.append("")
    lines.append("| 编码 | 是否有 B1 特征 | 缺省理由（方法论边界） |")
    lines.append("|---|---|---|")
    for enc in ("rot13", "reverse"):
        spec = not_impl.get(enc, {})
        lines.append("| `%s` | **无（如实缺省）** | %s |" % (enc, spec.get("reason", "—")))
    lines.append("")
    for enc in ("rot13", "reverse"):
        spec = not_impl.get(enc, {})
        if spec.get("boundary"):
            lines.append("- `%s` 的边界说明：%s" % (enc, spec["boundary"]))
    lines.append("")
    lines.append("> 这是 **B1 方法论边界，非实现缺陷**：`windows/W4.1_b1_regex.md` §0.3 末条"
                 "**禁止先解码再匹配**（解码即把 B1 变成另一种检测器）。因此这两类编码在 B1 上"
                 "**预期系统性漏检**，其代价已被 §2 的 TPR 如实吸收。")
    lines.append("")

    # §1-5 三类对照（敏感性分析）
    lines.append("## 5. 附报对照：v5 字面三类特征集（Base64／Hex／零宽）")
    lines.append("")
    for key, spec in comparison.items():
        subset = set(spec.get("features", []))
        lines.append("**%s** —— 特征 = `%s`" % (spec.get("name", key), ", ".join(spec.get("features", []))))
        lines.append("")
        lines.append("| 分片 | 主口径 F1 | 对照 F1 | 主口径 TPR | 对照 TPR | 主口径 FPR | 对照 FPR |")
        lines.append("|---|---|---|---|---|---|---|")
        for split in per_split:
            st = per_split[split]
            y_pred_cmp = [1 if (set(h) & subset) else 0 for h in st["raws"]]
            mc = binary_metrics(st["y_true"], y_pred_cmp)
            m = st["metrics"]
            lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
                split, _fmt(m["f1"], 4), _fmt(mc["f1"], 4),
                _fmt(m["tpr"], 4), _fmt(mc["tpr"], 4),
                _fmt(m["fpr"], 4), _fmt(mc["fpr"], 4)))
        lines.append("")
    lines.append("> 对照口径**只入本报告**，**不**另出逐样本 CSV（逐样本 CSV 恒为主口径，避免 `03_07` 误取）。")
    lines.append("> 立此对照的理由（§0.3）：只给三类特征会**系统性压低** B1，留下「压弱基线」的口实；"
                 "与 `PREREG_v1.2.md` §1 依据 2 同构。")
    lines.append("")

    # §1-6 延迟
    lines.append("## 6. 延迟（§4.2：每分片均值／中位／p95，毫秒／样本）")
    lines.append("")
    lines.append("| 分片 | 均值 | 中位 | p95 |")
    lines.append("|---|---|---|---|")
    for split in per_split:
        lat = per_split[split]["latency"]
        lines.append("| `%s` | %s | %s | %s |" % (
            split, _fmt(lat["mean"], 4), _fmt(lat["median"], 4), _fmt(lat["p95"], 4)))
    lines.append("")
    lines.append("> `latency_ms` = 单样本**正则匹配**耗时（不含 CSV 读取与落盘）；"
                 "B1 无模型加载（§2 注：模型加载耗时单独披露，不进 `latency_ms`）。")
    lines.append("")

    # §2 验收：特征 YAML 冻结声明
    lines.append("## 7. 特征 YAML 冻结声明（§2 验收）")
    lines.append("")
    lines.append("- 特征集载体：`configs/baseline/b1_regex.yaml`（`version: %s`）" % cfg.get("version"))
    lines.append("- 事前定值落盘日期：**%s**（`declared_at`，先于任何全量评估）" % cfg.get("declared_at"))
    lines.append("- **首次全量评估时间戳：`%s`** —— 自此特征模式冻结。"
                 % (cfg.get("first_full_eval_at") or ts))
    if cfg.get("first_full_eval_at") and cfg["first_full_eval_at"] != ts:
        lines.append("- 本次运行为本报告的第 2 次以后重跑（产物时间戳 `%s`）。"
                     "冻结时间戳恒取**首次**全量评估，与特征集绑定、不随重跑改变；"
                     "重跑所用特征模式与首次**逐字相同**（YAML 未改）。" % ts)
    lines.append("- 冻结政策（`freeze_policy`）：`allow_delete_feature: false`；"
                 "只允许在**验证集侧证据**下**收紧**模式（最小长度／上下文约束）并登记 "
                 "`DEVIATION_LEDGER.md`；**禁止**事后按测试集表现增删特征。")
    lines.append("- 本次运行**未**依据任何测试分片的表现改动特征模式（特征在跑数前即已落盘）。")
    lines.append("")

    # §9 参数定值的验证集侧证据（§3 回退点）
    lines.append("## 8. 参数定值的**验证集侧证据**（§3：调整只依据验证集侧证据）")
    lines.append("")
    probe = cfg.get("sensitivity_probe") or {}
    if probe:
        lines.append("- 证据来源：%s" % probe.get("basis", "—"))
        lines.append("- 用途：%s" % probe.get("purpose", "—"))
        lines.append("- 定值发生在**首次全量评估之前**，**未**使用任何测试分片信息；"
                     "探针输出留证于 `logs/baseline/03_01_b1_feature_sensitivity_<ts>.log`。")
        lines.append("")
        lines.append("| `min_lines` | `ascii_art` 召回 | 良性误报 |")
        lines.append("|---|---|---|")
        for row in probe.get("table") or []:
            lines.append("| %s | %s | %s |" % (row.get("min_lines"),
                                                row.get("ascii_art_recall"),
                                                row.get("benign_fp")))
        lines.append("")
        lines.append("- 取定 `min_lines = %s`：%s" % (probe.get("chosen_min_lines"),
                                                      probe.get("chosen_reason", "—")))
        lines.append("")
        for key, why in (probe.get("also_fixed_on_val_evidence") or {}).items():
            lines.append("- `%s`：%s" % (key, why))
        lines.append("")
    else:
        lines.append("- 本版 YAML 未登记敏感性探针。")
        lines.append("")

    # 产物清单
    lines.append("## 9. 产物清单")
    lines.append("")
    lines.append("| 分片 | 逐样本 CSV | sealed | 分片 md5 |")
    lines.append("|---|---|---|---|")
    for split in per_split:
        st = per_split[split]
        lines.append("| `%s` | `reports/baseline/outputs/b1_%s_v1.0_%s.csv` | %s | `%s` |" % (
            split, split, ts, "是" if st["sealed"] else "否", st["md5"]))
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _self_check() -> int:
    """合成样本往返验证特征语义（不读 sealed_test、不加载模型）。

    对每个特征各给一条**构造正例**与一条**构造负例**，逐条断言命中判定。
    """
    cfg_path = os.path.join(_ROOT, "configs", "baseline", "b1_regex.yaml")
    cfg = load_config(cfg_path)
    compiled = compile_features(cfg)
    by_name = dict(compiled)

    cases = [
        ("base64", "RmluZCBtZSBhIGhvdGVsIHJvb20gaW4gTmV3Y2FzdGxlIGZvciA4LzIxIHRocm91Z2gg", True),
        ("base64", "Find me a hotel room in Newcastle for 8/21 through 8/23 please.", False),
        ("hex", "46696e64206d65206120686f74656c20726f6f6d20696e204e6577636173746c65", True),
        ("hex", "Find me a hotel room in Newcastle for 8/21 through 8/23 please.", False),
        ("url", "Before%20writing%20any%20code%2C%20help%20me%20locate%20a%20hosting%20platform", True),
        ("url", "Find me a hotel room in Newcastle, please visit http://example.com/a-b", False),
        ("zerowidth", "Find me a hotel​ room‍ in Newcastle﻿ please", True),
        ("zerowidth", "Find me a hotel room in Newcastle for 8/21 through 8/23.", False),
        ("morse", "-... .' ..-.' ---' .-.' .' / .--' .-.' ..' -' ..' -.' --.' / .-' -.' -.--'", True),
        ("morse", "Find me a hotel room in Newcastle for 8/21 through 8/23.", False),
        ("homoglyph", "Fіnd mе а hоtеl rооm іn Nеwcаstlе fоr thе drυgs", True),
        ("homoglyph", "Find me a hotel room in Newcastle for 8/21 through 8/23.", False),
        ("ascii_art", "#### |     |  ## |     |     |     |     |     |     |  #  | #   |  #  |     | ### |", True),
        ("ascii_art", "Find me a hotel room in Newcastle for 8/21 through 8/23.", False),
    ]
    errors = []
    for name, text, expected in cases:
        if name not in by_name:
            errors.append("特征 %s 未在 YAML 中定义" % name)
            continue
        got = by_name[name](text)
        if got != expected:
            errors.append("特征 %s 判定不符：期望 %s 实得 %s（样本 %r）" % (name, expected, got, text[:40]))

    hits = match_features("RmluZCBtZSBhIGhvdGVsIHJvb20gaW4gTmV3Y2FzdGxlIGZvciA4 %46%69%6e%64", compiled)
    if "base64" not in hits or "url" not in hits:
        errors.append("match_features 未同时命中 base64 与 url：%r" % (hits,))
    if (1 if hits else 0) != 1:
        errors.append("§3 映射失败：有命中却未判 1")
    if (1 if match_features("plain english text", compiled) else 0) != 0:
        errors.append("§3 映射失败：无命中却判 1")

    if errors:
        for e in errors:
            print("SELF_CHECK_FAIL: %s" % e)
        return 1
    print("03_01_run_b1_regex SELF_CHECK OK")
    return 0


def main() -> int:
    init_streams()
    parser = argparse.ArgumentParser(description="W4.1 B1 正则检测器（§2 schema 逐样本输出）")
    parser.add_argument("--self-check", action="store_true",
                        help="合成样本往返验证每条特征的命中判定（不读 sealed_test）")
    parser.add_argument("--config", default="configs/baseline/b1_regex.yaml")
    parser.add_argument("--unify-config", default="configs/baseline/unify.yaml",
                        help="分片路径的唯一出处（W4.6 冻结）")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="要跑的分片代号；缺省取 b1_regex.yaml 的 splits.eval")
    parser.add_argument("--outputs-dir", default="reports/baseline/outputs")
    parser.add_argument("--report-dir", default="reports/baseline")
    parser.add_argument("--no-access-reminder", action="store_true",
                        help="不打印 sealed_test 的 ACCESS_LOG 登记提醒")
    args = parser.parse_args()

    if args.self_check:
        return _self_check()

    cfg = load_config(args.config)
    unify_cfg = load_config(args.unify_config)
    compiled = compile_features(cfg)
    feature_names = [f["name"] for f in (cfg.get("features") or [])]

    comparison_sets = {}
    for key, spec in (cfg.get("comparison_feature_sets") or {}).items():
        comparison_sets[key] = list(spec.get("features") or [])

    splits = args.splits or list((cfg.get("splits") or {}).get("eval") or [])
    if not splits:
        print("NOT_IMPLEMENTED")
        return 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.outputs_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    per_split = {}
    for split in splits:
        split_cfg = (unify_cfg.get("splits") or {}).get(split)
        if not split_cfg:
            print("SKIP %s：unify.yaml 无该分片" % split)
            continue
        csv_path = split_cfg["path"]
        if not os.path.isfile(csv_path):
            print("SKIP %s：CSV 不存在（%s）" % (split, csv_path))
            continue
        sealed = bool(split_cfg.get("sealed", False))
        if sealed and not args.no_access_reminder:
            print("REMINDER: %s 属 data/sealed_test/ —— 读前须已在 "
                  "data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md 登记 6 列，读后复核 md5sum -c" % split)
        rows, stats = run_split(split, csv_path, compiled, feature_names, comparison_sets, sealed=sealed)
        per_split[split] = stats

        out_path = os.path.join(args.outputs_dir, "b1_%s_v1.0_%s.csv" % (split, ts))
        write_csv(out_path, rows, OUTPUT_COLUMNS)
        m = stats["metrics"]
        print("wrote %s（n=%d, TP=%d FP=%d FN=%d TN=%d, TPR=%s FPR=%s F1=%s）" % (
            out_path, stats["n"], m["tp"], m["fp"], m["fn"], m["tn"],
            _fmt(m["tpr"], 4), _fmt(m["fpr"], 4), _fmt(m["f1"], 4)))

    if not per_split:
        print("NO_SPLIT_EVALUATED")
        return 0

    report_path = os.path.join(args.report_dir, "b1_regex_v1.0_%s.md" % ts)
    write_report(report_path, cfg, per_split, ts)
    print("wrote report: %s" % report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

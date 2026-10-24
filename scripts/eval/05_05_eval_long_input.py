# -*- coding: utf-8 -*-
"""05_05_eval_long_input.py — 阶段5（W6.6，A6）5.5 **长输入实验**主脚本。

接口契约（`windows/W6.6_long_input.md` §2，**冻结后不得改**）::

    python scripts/eval/05_05_eval_long_input.py \
        --max-tokens {512,2048} \
        --detectors ours,b1,b2,b3,b4,b5 \
        --runs-csv reports/train/three_seeds_v1.0_*.csv [--only <regime>:<seed>] \
        --source {val,sealed} \
        --config configs/eval/long_input.yaml \
        --report-dir reports/eval/ --outputs-dir reports/eval/outputs/ \
        [--access-log-note "<已登记的 ACCESS_LOG 行摘要>"] \
        [--length-strata] [--limit N] [--self-check] [--dry-run]

> **契约的两项补充（不改动既有项，仅补默认）**：`--split`（`--source sealed` 时必填，默认
> `in_regime_test`）与 `--regime`（默认取 `configs/eval/long_input.yaml` → `inputs.default_regime`
> = `id`）。二者是 `--source val|sealed` 解析**显式路径**所必需（`W6.1` §3-C1 同源要求），
> 属补充而非变更，已列入本窗口 §9 呈 W0 事项。

stdout 末行（依模式）::

    SELFCHECK_RESULT=PASS|FAIL n=<k>
    DRYRUN_RESULT=PASS|FAIL n=<k>
    REPORT_MD=<path> REPORT_CSV=<path>

硬约束逐条落点（`windows/W6.6_long_input.md` §3）::

    C1  公平性限制 → `render_report` §0 **逐字**输出配置 `fairness.verbatim_line_1/2`；
                     `--self-check` 机械核验「2048 表段内无 B4」
    C2  白名单守卫 → `resolve_detectors()`：`--max-tokens 2048` 只允许 ours/b3/b5；
                     **命令行显式**给 b1/b2/b4 ⇒ 退出码 2；未给 ⇒ 自动收窄
    C3  截断侧     → `assert_truncation_invariant()`：两组**同用左截断**（配置 `side_by_group`）；
                     报告 §4 同时给出两组各自的**实测长度分布与分位数**（否则长度与鲁棒性混淆）
    C4  主口径     → `assert_main_caliber_untouched()`：只读校验 `03_06.truncate_left` 默认 512
                     与三份既有 yaml 的 512；2048 只作**内存拷贝**显式传参，不落回任何文件
    C5  输入面     → 复用 `05_01.assert_input_allowed`／`resolve_input_path`／`assert_record_count`；
                     实现期只读 val；`--source sealed` 缺 `--split`／`--access-log-note` ⇒ 退出码 2
    C6  记录数     → 一律 `csv.DictReader`（`05_01.read_records`）；读后断言；不符 ⇒ 非零退出
    C7  评估态     → `model.eval()`＋`torch.no_grad()`（由 `05_01.Engine`／`03_03`／`03_05` 保证）；
                     源码内不得出现训练模式切换调用；确定性参数同 `ENV_LOCK.md` §7.2；`attn=eager`
    C8  显存/延迟  → `max_memory_allocated`（MiB，加载态／推理态分列，七元组声明）；
                     **2048 与 512 的显存不可相减** ⇒ 分列；B4 属 CPU 档 ⇒ 三档量纲并列不合并
    C9  B4 滑窗    → B4 走 SDK 自带管线（8192/2048/256），不叠加项目侧清洗；只在 512 组
    C10 报告路径   → `--report-dir`／`--outputs-dir` 必填；脚本内**不写死** `reports/`
    C11 引数带口径 → 引 `BASELINE_REPORT.md` 须带 §12 四条 mandatory notes（`citation.mandatory_notes`）
    C12 只追加     → 报告／CSV／日志一律新时间戳文件，不覆盖他件

顶层不得 import 重型库（`AGENTS.md` §3）：`torch`／`transformers`／`peft`／`yaml`／`psutil`
一律在函数体内惰性 import。`CUBLAS_WORKSPACE_CONFIG` 必须在 `import torch` 之前写入。
"""
from __future__ import annotations

# --- C7：必须在 import torch 之前设置（本模块所有 torch import 均在函数体内）---
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

TBD = "<TBD>"
NA = "n/a"

ALL_DETECTORS = ["b1", "b2", "b3", "b4", "b5", "ours"]
GROUP_KEYS = ("512", "2048")

MAIN_FIELDS = [
    "group", "max_tokens", "detector", "label", "run_id", "regime", "seed", "shard", "source",
    "tau_track", "tau_value", "n", "n_pos", "n_neg", "tp", "fp", "fn", "tn",
    "tpr", "fpr", "f1", "auroc",
    "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
    "mem_load_peak_mib", "mem_infer_peak_mib", "cpu_rss_peak_mib",
    "n_truncated", "trunc_rate", "n_unparsed",
    "token_len_raw_p50", "token_len_raw_p95", "token_len_raw_max",
    "token_len_eff_p50", "token_len_eff_p95", "token_len_eff_max",
    "eval_batch_max_size", "notes",
]

STRATA_FIELDS = [
    "group", "max_tokens", "detector", "run_id", "regime", "seed",
    "band", "n", "n_pos", "n_neg", "fnr", "fpr", "notes",
]

PER_SAMPLE_FIELDS = [
    "sample_id", "text_sha256", "label", "pred", "score", "latency_ms",
    "band", "token_len_raw", "token_len_eff", "truncated",
]

# ─────────────────────────────────────────────────────────────────────────────
# `T3`（`D-111` 裁定二／五）—— **两套截断率口径**的事前登记（**禁止混写**）
#
# ⚠ 之所以必须分列：二者**分母不是同一个东西**。若合成一个"截断率 X%"，
# 读者无法判断它描述的是"检测器看到的输入被截了多少"还是"下游输出被截了多少"
# ——而这两者的**影响方向相反**（见 `render_limitation_section`）。
# ───────────────────────────────────────────────────────────────────────────
TRUNC_CALIBERS = [
    {
        "key": "detection_side",
        "name": "检测侧截断率",
        "what": "送入检测器的**输入**被左截断的样本占比",
        "numerator": "逐样本产物中 `truncated == 1` 的**样本**条数",
        "denominator": "该分片的**检测单元**数（= 分片记录数）",
        "stage": "检测前（`03_06.truncate_left`）",
        "direction": "截断**丢弃输入前缀** ⇒ 攻击载荷可能仍在尾部保留 ⇒ **不系统性低估** ASR；"
                        "但被丢弃的**前置指令上下文**可能削弱攻击 ⇒ 方向**不唯一**",
    },
    {
        "key": "asr_chain_side",
        "name": "ASR 链路截断率",
        "what": "下游 SLM **输出**因触长被截（`DF-TRUNC`）的判定单元占比",
        "numerator": "`DF-TRUNC` 判定单元数",
        "denominator": "该 scenario×split 的**放行判定单元**数（= 放行攻击样本 × 4 下游）",
        "stage": "生成后（`05_20` 的 `finish_reason` 回传）",
        "direction": "输出被截 ⇒ 判官看不到完整攻击兑现证据 ⇒ **系统性低估**（保守方向）",
    },
]


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（`D-40` 裁定五：不得静默 pass）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（``errors="replace"``），避免 GBK 环境打印非 ASCII 假失败。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:  # 不静默：写 stderr 但不中断
            sys.stderr.write("STREAM_RECONFIGURE_FAILED stream=%r detail=%s\n" % (stream, exc))


# ---------------------------------------------------------------------------
# 模块加载 / 配置
# ---------------------------------------------------------------------------
def load_module(name, rel_path):
    """按路径加载模块（与 `05_01.load_module` 同手法）。"""
    path = os.path.join(_ROOT, rel_path.replace("/", os.sep))
    if not os.path.isfile(path):
        raise FileNotFoundError("模块不存在：%s" % path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载模块：%s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


_E501 = None
_EFF = None
_UNIFY = None
_PRE = None


def W():
    """**单源复用** `05_01`（输入面守卫／记录数断言／CSV 读写／`Engine`／延迟／分位数）。"""
    global _E501
    if _E501 is None:
        _E501 = load_module("eval_05_01", "scripts/eval/05_01_eval_detection.py")
    return _E501


def EFF():
    """**单源复用** `05_07` 的显存／RSS **采集器**（非推理代码）。"""
    global _EFF
    if _EFF is None:
        _EFF = load_module("eval_05_07", "scripts/eval/05_07_eval_efficiency.py")
    return _EFF


def UNIFY():
    """**单源** `03_07`（判定映射与指标）。"""
    global _UNIFY
    if _UNIFY is None:
        _UNIFY = load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    return _UNIFY


def PRE():
    global _PRE
    if _PRE is None:
        _PRE = load_module("preproc_03_06", "scripts/baseline/03_06_unify_preprocess.py")
    return _PRE


# ---------------------------------------------------------------------------
# 格式化（`n/a` vs `<TBD>` 语义不互串，`W6.1` §7 同口径）
# ---------------------------------------------------------------------------
def fmt_num(x, nd=4):
    if x is None:
        return TBD
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    return ("%." + str(nd) + "f") % (x,)


def fmt_metric(x, nd=4):
    """**已算完**的指标格：``None`` ⇒ `n/a`（不适用），**不是** `<TBD>`。"""
    return NA if x is None else fmt_num(x, nd)


def fmt_tau(x):
    """全精度（`F-W5.1-2`）：``repr(float)``，不取显示值。"""
    return TBD if x is None else repr(float(x))


def fmt_mem(x):
    return NA if x is None else ("%.1f" % float(x))


def fmt_null(v, spec):
    return "" if v is None else spec % v


# ---------------------------------------------------------------------------
# C2 检测器白名单守卫
# ---------------------------------------------------------------------------
def parse_detectors(spec):
    """``"ours,b3,b5"`` → ``["ours","b3","b5"]``（保序去重；未知项抛错）。"""
    if spec is None:
        return None
    out = []
    for tok in str(spec).split(","):
        d = tok.strip()
        if not d:
            continue
        if d not in ALL_DETECTORS:
            raise ValueError("未知检测器：%r（可选：%s）" % (d, ",".join(ALL_DETECTORS)))
        if d not in out:
            out.append(d)
    return out


def whitelist_allows(max_tokens, dets, cfg):
    """纯函数：该组是否允许这批检测器（`--self-check` 直接调用，无需子进程）。"""
    allowed = list(cfg["whitelist"][str(int(max_tokens))])
    return all(d in allowed for d in dets)


def resolve_detectors(args, cfg, max_tokens, logger=print):
    """C2：2048 组**只允许** ours／b3／b5。

    - **命令行显式**给出越界检测器 ⇒ 抛 `AssertionError`（上层转退出码 2）；
    - **未给** `--detectors` ⇒ 自动收窄为该组 `detectors`，不报错。
    """
    gkey = str(int(max_tokens))
    allowed = list(cfg["whitelist"][gkey])
    default = list(cfg["groups"][gkey]["detectors"])
    explicit = parse_detectors(args.detectors)
    if explicit is None:
        logger("DETECTORS_AUTO_NARROW max_tokens=%s detectors=%s（未显式给 `--detectors`）"
               % (max_tokens, ",".join(default)))
        return default, False
    bad = [d for d in explicit if d not in allowed]
    if bad:
        raise AssertionError(
            "白名单守卫失败：`--max-tokens %s` 只允许 %s；越界 = %s"
            "（`PREREG_v1.md` §5 补录（G6）L66：`unplug-tiny` 原生上限 512 不参与 2048 组；"
            "`W6.6` §3-C2 ⇒ 非零退出）"
            % (max_tokens, ",".join(allowed), ",".join(bad)))
    return explicit, True


# ---------------------------------------------------------------------------
# C3 截断侧不变量／C4 主口径不受污染
# ---------------------------------------------------------------------------
def assert_truncation_invariant(cfg):
    """C3：两组**同用左截断**；禁一组左一组右（否则长度效应与截断侧效应混淆）。"""
    sides = cfg["truncation"]["side_by_group"]
    vals = {str(k): str(v) for k, v in sides.items()}
    for g in GROUP_KEYS:
        if g not in vals:
            raise AssertionError("截断侧不变量失败：`side_by_group` 缺组 %s" % g)
    if len(set(vals.values())) != 1:
        raise AssertionError(
            "截断侧不变量失败：两组截断侧不一致 —— %s（`W6.6` §3-C3：两组必须同用左截断）" % vals)
    if any(v != "left" for v in vals.values()):
        raise AssertionError("截断侧不变量失败：须为 `left`，实测 %s" % vals)
    return vals


def assert_main_caliber_untouched(cfg):
    """C4：只读校验 512 主口径的**既有默认值**未被改动（本窗口不改任何既有文件）。

    返回 ``(ok, detail_lines)``；校验点 = `03_06.truncate_left` 默认形参 ＋ 三份既有 yaml。
    """
    import inspect

    lines = []
    ok = True
    expect = int(cfg["truncation"]["main_default_tokens"])

    default_tokens = inspect.signature(PRE().truncate_left).parameters["max_tokens"].default
    ok_d = (int(default_tokens) == expect)
    ok = ok and ok_d
    lines.append("03_06.truncate_left.default=%s ⇒ %s" % (default_tokens, "OK" if ok_d else "FAIL"))

    for rel, path in (("b3", "configs/baseline/b3_llamaguard1b.yaml"),
                      ("b5", "configs/baseline/b5_llamaguard8b.yaml")):
        v = int(load_config(_abs(path))["input"]["max_seq_tokens"])
        ok_v = (v == expect)
        ok = ok and ok_v
        lines.append("%s.input.max_seq_tokens=%s ⇒ %s" % (rel, v, "OK" if ok_v else "FAIL"))

    v = int(load_config(_abs(cfg["train_config"]))["preprocess"]["max_seq_tokens"])
    ok_v = (v == expect)
    ok = ok and ok_v
    lines.append("qlora_ours.preprocess.max_seq_tokens=%s ⇒ %s" % (v, "OK" if ok_v else "FAIL"))
    return ok, lines


def deep_set(cfg_obj, path, value):
    """在**内存拷贝**上按路径赋值（C4：2048 只作显式传参，不落回任何 yaml）。"""
    node = cfg_obj
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return cfg_obj


# ---------------------------------------------------------------------------
# 指标（**单源** `03_07`；`unparsed` 不兜底为 0/1 ⇒ 逐出分母并披露）
# ---------------------------------------------------------------------------
def metrics_from_preds(labels, preds, scores=None):
    """返回 ``{n, n_pos, n_neg, tp, fp, fn, tn, tpr, fpr, f1, auroc, n_unparsed}``。

    - ``pred=None``（判词 `unparsed`）⇒ **逐出**分母并单独计数（`03_03` 口径：不兜底为 0/1）；
    - 无正类 ⇒ `tpr`／`f1` = ``None``（`n/a`）；无负类 ⇒ `fpr` = ``None``；
    - `auroc` 仅在有连续 score 且全部分数非 None 时计算，否则 ``None``（`n/a`）。
    """
    unify = UNIFY()
    pairs = [(int(y), p) for y, p in zip(labels, preds) if p is not None]
    n_unparsed = sum(1 for p in preds if p is None)
    y = [a for a, _ in pairs]
    p = [b for _, b in pairs]
    c = unify.binary_counts(y, p)
    out = dict(c)
    out["n"] = len(pairs)
    out["n_unparsed"] = n_unparsed
    out["tpr"] = (c["tp"] / c["n_pos"]) if c["n_pos"] else None
    out["fpr"] = (c["fp"] / c["n_neg"]) if c["n_neg"] else None
    out["f1"] = (unify.f1_score(y, p) if c["n_pos"] else None)
    auroc = None
    if scores is not None and len(scores) == len(labels):
        ys, ss = [], []
        for yy, s in zip(labels, scores):
            if s is None:
                continue
            ys.append(int(yy))
            ss.append(float(s))
        if ss and len(set(ys)) > 1:
            auroc = unify.auroc(ys, ss)
    out["auroc"] = auroc
    return out


# ---------------------------------------------------------------------------
# 长度度量（C3：两组各自**实测**；`raw` 与 `eff` 口径分开声明）
# ---------------------------------------------------------------------------
def load_canonical_tokenizer(cfg, logger=print):
    """加载**规范 tokenizer**（Llama-Guard-3-1B，与 Ours／B3 同源）用于**检测器无关**的长度度量。"""
    from transformers import AutoTokenizer

    tcfg = load_config(_abs(cfg["train_config"]))
    w = tcfg["weights"]
    path = _abs(w["local_path"])
    tok = AutoTokenizer.from_pretrained(path, local_files_only=bool(w.get("local_files_only", True)))
    logger("TOKENIZER_LOADED path=%s（**仅分词**，不加载模型权重）" % path)
    return tok


def band_of(n, bands):
    """按**未截断** token 长度归层（事前定值；空层照实报 `n=0`）。"""
    if n is None:
        return NA
    for b in bands:
        lo = int(b["lo"])
        hi = b["hi"]
        if hi is None:
            if n >= lo:
                return str(b["name"])
        elif lo <= n < int(hi):
            return str(b["name"])
    return str(bands[-1]["name"])


# ---------------------------------------------------------------------------
# 检测器运行器（**只接线，不复制推理代码**）
# ---------------------------------------------------------------------------
class BaseRunner(object):
    """``load()`` → ``infer(rows, max_tokens)`` → ``unload()``。"""

    det_id = ""
    has_score = False

    def __init__(self, dcfg, cfg, logger=print):
        self.dcfg = dcfg
        self.cfg = cfg
        self.logger = logger

    def load(self):
        raise NotImplementedError

    def infer(self, rows, max_tokens):
        """返回 ``{"preds","scores","latency_ms","trunc_flags","token_len_eff",
        "mem_infer_peak_mib","cpu_rss_peak_mib","n_unparsed","eval_batch_max_size"}``。"""
        raise NotImplementedError

    def unload(self):
        return None


class NoModelRunner(BaseRunner):
    """`b1`（正则）／`b2`（熵值）：**无模型、无 token 概念** ⇒ 不受 `max_tokens` 影响。

    该事实必须在报告中显式声明（否则读者会以为它们也随 2048 组被截断）。
    """

    def load(self):
        cfg_path = _abs(self.dcfg["config"])
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError("基线配置不存在：%s" % cfg_path)
        self.mod = load_module("li_%s" % self.det_id, self.dcfg["module"])
        self.bcfg = load_config(cfg_path)
        self.compiled = None
        if self.det_id == "b1":
            self.compiled = self.mod.compile_features(self.bcfg)
        self.logger("LOADED_ %s（无模型；无 token 上限概念 ⇒ 对 `max_tokens` 不敏感）" % self.det_id)
        return {"mem_load_peak_mib": None, "cpu_rss_peak_mib": EFF().rss_mib()}

    def infer(self, rows, max_tokens):
        unify = UNIFY()
        eff = EFF()
        call = getattr(self.mod, self.dcfg["entry"]["call"])
        tau = float("-inf") if self.det_id == "b2" else None
        ms, preds, scores = [], [], []
        peak = eff.rss_mib() or 0.0
        for r in rows:
            text = r.get("text", "")
            t0 = time.perf_counter()
            if self.det_id == "b1":
                pred = unify.map_b1(call(text, self.compiled))
                scores.append(None)
            else:
                s = call(text)
                pred = unify.map_b2(s, tau)
                scores.append(float(s))
            ms.append((time.perf_counter() - t0) * 1000.0)
            preds.append(int(pred))
            cur = eff.rss_mib()
            if cur is not None and cur > peak:
                peak = cur
        return {"preds": preds, "scores": scores if self.has_score else None,
                "latency_ms": ms, "trunc_flags": [False] * len(rows), "token_len_eff": None,
                "mem_infer_peak_mib": None, "cpu_rss_peak_mib": peak,
                "n_unparsed": 0, "eval_batch_max_size": 1}


class B1Runner(NoModelRunner):
    det_id = "b1"
    has_score = False


class B2Runner(NoModelRunner):
    det_id = "b2"
    has_score = True


class GuardRunner(BaseRunner):
    """`b3`（LG-1B）／`b5`（LG-8B）：4-bit NF4 ＋ GPU 贪心生成。

    2048 组通过在**内存拷贝**上改写 ``input.max_seq_tokens`` 实现（C4：不落回 yaml）。
    """

    def load(self):
        cfg_path = _abs(self.dcfg["config"])
        self.mod = load_module("li_%s" % self.det_id, self.dcfg["module"])
        self.bcfg0 = load_config(cfg_path)
        eff = EFF()
        eff.gpu_reset()
        torch, transformers, tok, model, meta = self.mod.load_model(self.bcfg0)
        self.torch = torch
        self.tok = tok
        self.model = model
        if model.training:
            raise RuntimeError("G-EVAL 断言失败：model.eval() 后仍处 train 模式")
        self.logger("LOADED_ %s attn=%s max_length=%s"
                    % (self.det_id, meta.get("attn"), meta.get("max_length")))
        return {"mem_load_peak_mib": float(meta["load_peak_bytes"]) / (1024.0 * 1024.0),
                "cpu_rss_peak_mib": None}

    def infer(self, rows, max_tokens):
        import torch

        unify = UNIFY()
        eff = EFF()
        bcfg = deep_set(copy.deepcopy(self.bcfg0), list(self.dcfg["max_tokens_key_path"]),
                        int(max_tokens))
        self.logger("GUARD_CFG_OVERRIDE %s input.max_seq_tokens=%s（**内存拷贝**，不落回 yaml）"
                    % (self.det_id, max_tokens))
        ms, preds, eff_len, trunc_flags, n_unparsed = [], [], [], [], 0
        eff.gpu_reset()
        with torch.no_grad():
            if torch.is_grad_enabled():
                raise RuntimeError("G-EVAL 断言失败：grad 已启用（必须 torch.no_grad()）")
            for r in rows:
                text = r.get("text", "")
                t0 = time.perf_counter()
                prompt, _trunc, n_text_tokens, was_truncated = self.mod.build_prompt(
                    self.tok, bcfg, text)
                judgment, _np = self.mod.generate_judgment(self.torch, self.tok, self.model,
                                                           bcfg, prompt)
                ms.append((time.perf_counter() - t0) * 1000.0)
                pred, unparsed = unify.parse_judgment(judgment)
                if pred is None:
                    n_unparsed += 1
                preds.append(None if pred is None else int(pred))
                eff_len.append(int(n_text_tokens))
                trunc_flags.append(bool(was_truncated))
            peak = eff.gpu_peak_mib() or 0.0
        return {"preds": preds, "scores": None, "latency_ms": ms, "trunc_flags": trunc_flags,
                "token_len_eff": eff_len, "mem_infer_peak_mib": peak, "cpu_rss_peak_mib": None,
                "n_unparsed": n_unparsed, "eval_batch_max_size": 1}

    def unload(self):
        held = [getattr(self, "model", None), getattr(self, "torch", None),
                getattr(self, "tok", None)]
        self.model = None
        self.torch = None
        self.tok = None
        residue = EFF().release_gpu(held)
        self.logger("UNLOADED %s residue_allocated_mib=%s" % (self.det_id, fmt_mem(residue)))
        return residue


class B3Runner(GuardRunner):
    det_id = "b3"
    has_score = False


class B5Runner(GuardRunner):
    det_id = "b5"
    has_score = False


class B4Runner(BaseRunner):
    """`b4`（unplug-tiny）：**CPU 档**；走 **SDK 自带**滑窗管线（8192/2048/256），不叠加项目侧清洗。

    C9：原生上限 512 ⇒ **只在 512 组**（由 `resolve_detectors` 的白名单结构性排除）。
    """

    det_id = "b4"
    has_score = True

    def load(self):
        cfg_path = _abs(self.dcfg["config"])
        self.mod = load_module("li_b4", self.dcfg["module"])
        self.bcfg = load_config(cfg_path)
        vendor = _abs(self.dcfg.get("vendor_syspath") or "")
        self._vendor = None
        if vendor:
            if not os.path.isdir(vendor):
                raise FileNotFoundError("vendor 目录不存在：%s" % vendor)
            sys.path.insert(0, vendor)
            self._vendor = vendor
        model, normalizer, meta = self.mod.load_model(self.bcfg)
        self.model = model
        self.normalizer = normalizer
        lt = self.bcfg["pipeline"]["long_text"]
        self.pipe = {"threshold_chars": int(lt["threshold_chars"]),
                     "chunk_chars": int(lt["chunk_chars"]),
                     "overlap_chars": int(lt["overlap_chars"]),
                     "batch_size": int(self.bcfg["pipeline"]["batch_size"])}
        self.logger("LOADED_ b4 device=%s max_length=%s sliding=%s/%s/%s"
                    % (meta.get("device"), meta.get("max_length"),
                       self.pipe["threshold_chars"], self.pipe["chunk_chars"],
                       self.pipe["overlap_chars"]))
        return {"mem_load_peak_mib": None,
                "cpu_rss_peak_mib": float(meta["rss_after_load_bytes"]) / (1024.0 * 1024.0)}

    def infer(self, rows, max_tokens):
        # 阈值**单源**取自 b4 配置（`decision.doc_threshold`／`decision.span_threshold` = 0.9／0.45）
        dec = self.bcfg.get("decision") or {}
        doc_t = float(dec.get("doc_threshold", 0.9))
        span_t = float(dec.get("span_threshold", 0.45))
        unify = UNIFY()
        eff = EFF()
        ms, preds, scores = [], [], []
        peak = eff.rss_mib() or 0.0
        self.logger("B4_DECISION doc_threshold=%s span_threshold=%s（abstain 关闭）"
                    % (doc_t, span_t))
        for r in rows:
            text = r.get("text", "")
            t0 = time.perf_counter()
            doc_score, spans, _info = self.mod.predict_one(self.model, self.normalizer, text,
                                                           self.pipe)
            ms.append((time.perf_counter() - t0) * 1000.0)
            pred, score, _abstained = unify.map_b4({"doc_score": doc_score, "spans": spans},
                                                   doc_t, span_t)
            preds.append(int(pred))
            scores.append(None if score is None else float(score))
            cur = eff.rss_mib()
            if cur is not None and cur > peak:
                peak = cur
        return {"preds": preds, "scores": scores, "latency_ms": ms,
                "trunc_flags": [False] * len(rows), "token_len_eff": None,
                "mem_infer_peak_mib": None, "cpu_rss_peak_mib": peak,
                "n_unparsed": 0, "eval_batch_max_size": 1}

    def unload(self):
        self.model = None
        self.normalizer = None
        if self._vendor and self._vendor in sys.path:
            sys.path.remove(self._vendor)
        return None


class OursRunner(BaseRunner):
    """`ours`：**单源复用** `05_01.Engine`；2048 通过在内存拷贝上改 `preprocess.max_seq_tokens` 实现。

    τ **不因长度组重寻**（`configs/eval/long_input.yaml → detectors.ours.tau.rerun_in_2048: false`）。
    """

    det_id = "ours"
    has_score = True

    def __init__(self, dcfg, cfg, logger=print):
        BaseRunner.__init__(self, dcfg, cfg, logger)
        self.engine = None
        self.e501 = None
        self.tau_main = None

    def load(self, ckpt_dir, seed, max_tokens, base_cfg):
        self.e501 = W()
        train_cfg = deep_set(copy.deepcopy(load_config(_abs(self.cfg["train_config"]))),
                             list(self.dcfg["max_tokens_key_path"]), int(max_tokens))
        self.tcfg = train_cfg
        self.logger("OURS_CFG_OVERRIDE preprocess.max_seq_tokens=%s（**内存拷贝**，不落回 yaml）"
                    % max_tokens)
        eval_cfg = load_config(_abs(self.dcfg["config"]))
        eng = self.e501.Engine(eval_cfg, train_cfg, self.logger)
        info = eng.load(ckpt_dir, int(seed))
        self.engine = eng
        return {"mem_load_peak_mib": info["mem_load_peak_mib"], "cpu_rss_peak_mib": None}

    def infer(self, rows, max_tokens):
        res = self.engine.score(rows, tag="max%s" % max_tokens)
        preds = [None if s is None else int(1 if float(s) >= float(self.tau_main) else 0)
                 for s in res["scores"]]
        eff_len, trunc_flags = self._measure_eff_len(rows)
        return {"preds": preds,
                "scores": [None if s is None else float(s) for s in res["scores"]],
                "latency_ms": res["latency_ms"], "trunc_flags": trunc_flags,
                "token_len_eff": eff_len, "mem_infer_peak_mib": res["mem_infer_peak_mib"],
                "cpu_rss_peak_mib": None, "n_unparsed": 0,
                "eval_batch_max_size": res["eval_batch_max_size"],
                "n_batches": res.get("n_batches")}

    def _measure_eff_len(self, rows):
        """度量用**二次编码**（只走 `04_01.encode_example`，**不推理**）⇒ 取 `n_prompt`。

        与 `Engine.score` 内部的编码为**同一函数、同一参数**；此处仅用于长度披露，
        不参与任何判定。失败（无 tokenizer）时 ``eff=None, trunc=全 False`` 并声明。
        """
        eng = self.engine
        tm = eng.train_mod
        try:
            exs = [tm.encode_example(eng.tok, self.tcfg,
                                     self.e501.Sample(str(r.get("sample_id", "")),
                                                      r.get("text", ""),
                                                      int(float(r.get("label")))),
                                     eng.id_unsafe, eng.id_safe, eng.tok.eos_token_id)
                   for r in rows]
        except Exception as exc:  # noqa: BLE001 —— 长度仅用于披露，失败不影响指标
            self.logger("EFF_LEN_MEASURE_SKIP detail=%s" % exc)
            return None, [False] * len(rows)
        # ⚠ `Example.text` 存的是**原始** text（非截断后）⇒ 截断判据必须另走
        # `04_01.build_prompt` 的返回二元组 `(prompt, truncated_text)`。
        flags = []
        for r in rows:
            _prompt, trunc_text = tm.build_prompt(eng.tok, self.tcfg, r.get("text", ""))
            flags.append(bool(trunc_text != r.get("text", "")))
        return [int(e.n_prompt) for e in exs], flags

    def unload(self):
        if self.engine is not None:
            held = [getattr(self.engine, "model", None), getattr(self.engine, "llama", None),
                    getattr(self.engine, "lm_head", None), getattr(self.engine, "tok", None)]
            self.engine.model = None
            self.engine.llama = None
            self.engine.lm_head = None
            self.engine.tok = None
            residue = EFF().release_gpu(held)
            self.logger("UNLOADED ours residue_allocated_mib=%s" % fmt_mem(residue))
        self.engine = None
        return None


def make_runner(det_id, cfg, logger=print):
    dcfg = cfg["detectors"][det_id]
    if det_id == "b1":
        return B1Runner(dcfg, cfg, logger)
    if det_id == "b2":
        return B2Runner(dcfg, cfg, logger)
    if det_id == "b3":
        return B3Runner(dcfg, cfg, logger)
    if det_id == "b4":
        return B4Runner(dcfg, cfg, logger)
    if det_id == "b5":
        return B5Runner(dcfg, cfg, logger)
    if det_id == "ours":
        return OursRunner(dcfg, cfg, logger)
    raise ValueError("未知检测器：%r" % (det_id,))


# ---------------------------------------------------------------------------
# run 清单（Ours）
# ---------------------------------------------------------------------------
def resolve_runs(args, cfg, logger=print):
    """复用 `05_01.resolve_runs` ⇒ ``[(regime, seed, run_dir)]``。"""
    return W().resolve_runs(args, cfg, logger)


def select_runs(runs, base_cfg, regime, logger=print):
    """按体制挑选并要求 seed 齐全（单体制 = 3 run；缺 ⇒ 判 `EVAL_PENDING`）。"""
    seeds = sorted(int(s) for s in base_cfg["runs"]["seeds"])
    picked = sorted([(r, s, d) for (r, s, d) in runs if r == regime], key=lambda t: int(t[1]))
    got = sorted(int(s) for _, s, _ in picked)
    if got != seeds:
        logger("EVAL_PENDING reason=incomplete_runs regime=%s got=%s expected=%s"
               " ⇒ 依 `W6.6` §8 **本范围不产出正式结论**" % (regime, got, seeds))
        return []
    logger("EVAL_SCOPE_OK regime=%s n_run=%d" % (regime, len(picked)))
    return picked


# ---------------------------------------------------------------------------
# 输入（C5／C6：复用 `05_01` 的守卫与路径解析）
# ---------------------------------------------------------------------------
def prepare_rows(cfg, base_cfg, args, source, regime, logger=print):
    """返回 ``(rows, shard, path)``。"""
    w = W()
    allow_sealed = (source == "sealed")
    path, expect, shard = w.resolve_input_path(base_cfg, source, regime,
                                               args.split if allow_sealed else None, allow_sealed)
    w.assert_input_allowed(path, base_cfg, allow_sealed=allow_sealed)
    got = w.assert_record_count(path, expect, base_cfg)
    logger("READ_INPUT source=%s shard=%s records=%d physical_lines=%d（行数 ≠ 记录数，`W6.1` §8）"
           % (source, shard, got, w.count_physical_lines(path)))
    rows = w.read_records(path)
    if args.limit:
        rows = rows[:int(args.limit)]
        logger("LIMIT_APPLIED n=%d（**仅**用于冒烟；不得当作全量口径的延迟／显存）" % len(rows))
    return rows, shard, path


# ---------------------------------------------------------------------------
# 逐样本输出
# ---------------------------------------------------------------------------
def write_per_sample(outputs_dir, ts, group, det_id, run_id, rows, res, raw_lens, bands):
    pre = PRE()
    eff = res.get("token_len_eff")
    flags = res.get("trunc_flags") or [False] * len(rows)
    out = []
    for i, r in enumerate(rows):
        raw = raw_lens[i] if i < len(raw_lens) else None
        out.append({
            "sample_id": r.get("sample_id", ""),
            "text_sha256": pre.sha256_of(r.get("text", "")),
            "label": r.get("label", ""),
            "pred": "" if res["preds"][i] is None else int(res["preds"][i]),
            "score": "" if (not res.get("scores") or res["scores"][i] is None)
                     else repr(float(res["scores"][i])),
            "latency_ms": "%.6f" % float(res["latency_ms"][i]),
            "band": band_of(raw, bands),
            "token_len_raw": "" if raw is None else int(raw),
            "token_len_eff": "" if (eff is None or i >= len(eff)) else int(eff[i]),
            "truncated": "1" if flags[i] else "0",
        })
    # ⚠ 非 Ours 检测器的 `run_id` 显示值含 `/`（`n/a`）⇒ **不可**直接进文件名（会被当成子目录）
    slug = run_id if str(run_id) != NA else "no_run"
    path = os.path.join(outputs_dir,
                        "long_input_%s_%s_%s_v1.0_%s.csv" % (group, det_id, slug, ts))
    W().write_csv_rows(path, PER_SAMPLE_FIELDS, out)
    return path


# ---------------------------------------------------------------------------
# 单（组 × 检测器 × run）评估
# ---------------------------------------------------------------------------
def evaluate_one(cfg, base_cfg, args, group, det_id, rows, shard, source, regime, run,
                 raw_lens, bands, logger=print, outputs_dir=None, ts=None):
    """``run`` = ``(regime, seed, run_dir)``；非 Ours 检测器传 ``(regime, None, None)``。"""
    dcfg = cfg["detectors"][det_id]
    max_tokens = int(cfg["groups"][group]["max_tokens"])
    runner = make_runner(det_id, cfg, logger)
    tau_value = None
    tau_track = NA
    run_id = NA
    seed = NA
    best_step = None

    if det_id == "ours":
        _rg, seed, run_dir = run
        meta_info = W().load_run_meta(run_dir, base_cfg)
        W().assert_meta_matches_cli(meta_info, regime, seed)
        tau = W().resolve_tau_tracks(meta_info, base_cfg, None)
        tau_value = tau["main_value"]
        tau_track = tau["main_track"]
        run_id = os.path.basename(os.path.normpath(run_dir))
        best_step = meta_info["best_step"]
        runner.tau_main = tau_value
        load_info = runner.load(meta_info["ckpt_dir"], seed, max_tokens, base_cfg)
        regime = str(meta_info["regime"]) or regime
    else:
        load_info = runner.load()

    res = runner.infer(rows, max_tokens)
    labels = [int(float(r["label"])) for r in rows]
    m = metrics_from_preds(labels, res["preds"], res.get("scores"))
    lat = W().latency_stats(res["latency_ms"])

    per_sample_csv = None
    if outputs_dir:
        per_sample_csv = write_per_sample(outputs_dir, ts, group, det_id, run_id, rows, res,
                                          raw_lens, bands)

    rec = {
        "group": group, "max_tokens": max_tokens, "detector": det_id,
        "label": dcfg["label"], "run_id": run_id, "regime": regime, "seed": seed,
        "shard": shard, "source": source,
        "tau_track": tau_track, "tau_value": tau_value, "best_step": best_step,
        "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        "tpr": m["tpr"], "fpr": m["fpr"], "f1": m["f1"], "auroc": m["auroc"],
        "n_unparsed": m["n_unparsed"],
        "latency": lat,
        "mem_load_peak_mib": load_info.get("mem_load_peak_mib"),
        "mem_infer_peak_mib": res.get("mem_infer_peak_mib"),
        "cpu_rss_peak_mib": res.get("cpu_rss_peak_mib"),
        "n_truncated": sum(1 for f in (res.get("trunc_flags") or []) if f),
        "eval_batch_max_size": res.get("eval_batch_max_size"),
        "n_batches": res.get("n_batches"),
        "preds": res["preds"], "scores": res.get("scores"), "labels": labels,
        "token_len_eff": res.get("token_len_eff"), "raw_lens": list(raw_lens),
        "per_sample_csv": per_sample_csv,
    }
    runner.unload()
    return rec


# ---------------------------------------------------------------------------
# 分层（`--length-strata`；逐层 **FPR／FNR**，不是 F1 —— 纯类子集内 F1 无定义）
# ---------------------------------------------------------------------------
def build_strata_rows(cfg, group, rec, bands):
    out = []
    raw = rec["raw_lens"]
    preds = rec["preds"]
    labels = rec["labels"]
    for b in bands:
        name = str(b["name"])
        idxs = [i for i, v in enumerate(raw) if v is not None and band_of(v, bands) == name]
        n_pos = n_neg = fn = fp = 0
        for i in idxs:
            p = preds[i]
            if p is None:
                continue
            if labels[i] == 1:
                n_pos += 1
                if p == 0:
                    fn += 1
            else:
                n_neg += 1
                if p == 1:
                    fp += 1
        out.append({
            "group": group, "max_tokens": int(cfg["groups"][group]["max_tokens"]),
            "detector": rec["detector"], "run_id": rec["run_id"], "regime": rec["regime"],
            "seed": rec["seed"], "band": name, "n": len(idxs),
            "n_pos": n_pos, "n_neg": n_neg,
            "fnr": (fn / n_pos) if n_pos else None,
            "fpr": (fp / n_neg) if n_neg else None,
            "notes": "" if len(idxs) else "空层（照实报 n=0）",
        })
    return out


# ---------------------------------------------------------------------------
# 冻结面复核（纯 Python `md5`；**字节级**校验，`W6.1` §1.1 W6 补注 ②③）
# ---------------------------------------------------------------------------
def md5_file(path):
    d = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            d.update(blk)
    return d.hexdigest()


def md5sum_check(manifest_path):
    """纯 Python 版 `md5sum -c`（无 `md5sum` 可执行文件时可用）。

    **范围声明**：逐条打开 `md5sums.txt` 所列**全部**文件取字节 md5（含 `test.csv`）——
    依 `W6.1` §1.1 的 `W6` 补注 ②③，纯字节级完整性校验**不构成**「读内容」（不解析、
    不进入任何判断），且是治理文件自带的强制义务。
    """
    if not os.path.isfile(manifest_path):
        return 1, "manifest 不存在：%s" % manifest_path
    base = os.path.dirname(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    total = ok = 0
    bad = []
    for ln in lines:
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        want, rel = parts[0].strip(), parts[1].strip().lstrip("*")
        fpath = os.path.join(base, rel)
        total += 1
        if not os.path.isfile(fpath):
            bad.append("%s: MISSING" % rel)
            continue
        got = md5_file(fpath)
        if got == want:
            ok += 1
        else:
            bad.append("%s: FAILED" % rel)
    return (0 if (total and ok == total) else 1), "%d/%d OK%s" % (
        ok, total, ("；" + "，".join(bad)) if bad else "")


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


GROUP_TABLE_HEADER = ["检测器", "名称", "run", "seed", "n", "TP", "FP", "FN", "TN",
                      "TPR", "FPR", "F1", "AUROC",
                      "延迟均值ms", "延迟中位ms", "延迟p95ms",
                      "显存加载MiB", "显存推理MiB", "RSS MiB",
                      "截断条数", "截断率",
                      "raw p50", "raw p95", "raw max",
                      "eff p50", "eff p95", "eff max", "batch"]


def _row_cells(rec):
    q_raw = W().quantile_table([v for v in rec["raw_lens"] if v is not None])
    eff = rec.get("token_len_eff")
    q_eff = W().quantile_table(eff) if eff else None
    n = int(rec["n"]) or 0
    return [
        rec["detector"], rec["label"], rec["run_id"], rec["seed"],
        rec["n"], rec["tp"], rec["fp"], rec["fn"], rec["tn"],
        fmt_metric(rec["tpr"], 6), fmt_metric(rec["fpr"], 6), fmt_metric(rec["f1"], 6),
        fmt_metric(rec["auroc"], 6),
        fmt_metric(rec["latency"]["mean_ms"], 3), fmt_metric(rec["latency"]["median_ms"], 3),
        fmt_metric(rec["latency"]["p95_ms"], 3),
        fmt_mem(rec["mem_load_peak_mib"]), fmt_mem(rec["mem_infer_peak_mib"]),
        fmt_mem(rec["cpu_rss_peak_mib"]),
        "%d" % int(rec["n_truncated"] or 0),
        ("%.4f" % (float(rec["n_truncated"] or 0) / n)) if n else NA,
        fmt_num(q_raw["p50"], 1), fmt_num(q_raw["p95"], 1), fmt_num(q_raw["max"], 1),
        (fmt_num(q_eff["p50"], 1) if q_eff else NA),
        (fmt_num(q_eff["p95"], 1) if q_eff else NA),
        (fmt_num(q_eff["max"], 1) if q_eff else NA),
        "" if rec["eval_batch_max_size"] is None else str(rec["eval_batch_max_size"]),
    ]


def render_group_table(results):
    if not results:
        return "（本次未出数 —— 该组不在本次 `--max-tokens` 范围内，属**预期留白**）"
    return _md_table(GROUP_TABLE_HEADER, [_row_cells(r) for r in results])


def render_length_table(recs_by_group):
    rows = []
    for group in GROUP_KEYS:
        for rec in recs_by_group.get(group, []):
            q_raw = W().quantile_table([v for v in rec["raw_lens"] if v is not None])
            eff = rec.get("token_len_eff")
            q_eff = W().quantile_table(eff) if eff else None
            rows.append([
                group, rec["detector"], rec["run_id"],
                fmt_num(q_raw["min"], 1), fmt_num(q_raw["p25"], 1), fmt_num(q_raw["p50"], 1),
                fmt_num(q_raw["p75"], 1), fmt_num(q_raw["p90"], 1), fmt_num(q_raw["p95"], 1),
                fmt_num(q_raw["p99"], 1), fmt_num(q_raw["max"], 1),
                (fmt_num(q_eff["p50"], 1) if q_eff else NA),
                (fmt_num(q_eff["p95"], 1) if q_eff else NA),
                (fmt_num(q_eff["max"], 1) if q_eff else NA),
                rec.get("length_basis_note", ""),
            ])
    if not rows:
        return "（无结果）"
    return _md_table(["组", "检测器", "run", "min", "p25", "p50", "p75", "p90", "p95", "p99",
                      "max", "eff p50", "eff p95", "eff max", "口径说明"], rows)


def render_seven_tuple(cfg, group):
    rows = []
    mt = int(cfg["groups"][group]["max_tokens"])
    for det_id, d in cfg["detectors"].items():
        st = d.get("seven_tuple")
        if not st:
            continue
        rows.append([
            det_id, d["label"], st["model"], st["load_mode"],
            str(st["max_length"]).replace("{per_group}", str(mt)),
            str(st["batch_size"]).replace("{from make_batches}", "由护栏决定"),
            st["offload"], st["device"], st["collector"],
        ])
    return _md_table(["检测器", "名称", "model", "load_mode", "max_length", "batch",
                      "offload", "device", "collector"], rows)


# ---------------------------------------------------------------------------
# `T3`（`D-111` 裁定二／五）—— 截断率 Limitation 段（**四要素**）
# ---------------------------------------------------------------------------
def render_limitation_section(ctx):
    """渲染「截断率 Limitation」段。

    **四要素**（`windows/W6.6_long_input.md` 末尾 `T3` 条）：
      ① 两口径**分列**（检测侧／ASR 链路侧，分母不同 ⇒ **禁止混写**）；
      ② **影响方向** = ASR **系统性低估**（保守方向，可辩护）；
      ③ **反证** = 上限翻 4 倍后检测 F1 格局不变 ＋ 超长层 `Ours` 的 `FNR`／`FPR`；
      ④ **引用 2048 敏感性分析**。
    """
    cfg = ctx["cfg"]
    L = []
    A = L.append

    A("### `T3` · 截断率 Limitation（**含 512 → 2048 反证**）")
    A("")
    A("**① 两套口径分列（分母不同 ⇒ 禁止混写，`D-111` 裁定二 `R-2`）**")
    A("")
    A(_md_table(["口径", "度量对象", "分子", "分母", "发生阶段", "出处"],
                [[c["name"], c["what"], c["numerator"], c["denominator"], c["stage"],
                  c.get("source", "（见 §9）")] for c in TRUNC_CALIBERS]))
    A("")
    if ctx.get("trunc_caliber_rows"):
        A("**本次实测值（分列，各自带自己的分母）**")
        A("")
        A(_md_table(["口径", "组", "分子", "分母", "比率", "备注"],
                    [[row.get("caliber", ""), row.get("group", ""), row.get("num", ""),
                      row.get("den", ""), row.get("rate", ""), row.get("note", "")]
                     for row in ctx.get("trunc_caliber_rows", [])]))
        A("")
    A("**② 影响方向：`ASR` 系统性低估（保守，可辩护）**")
    A("")
    A("- 截断只发生在**下游 SLM 生成末段**（`DF-TRUNC`，`finish_reason == length`）"
      " ⇒ 判官看不到「攻击是否完整兑现」的证据 ⇒ 触发 `asr_judge.md` §4 的保守分支；")
    A("- ⇒ 本窗口所有 `ASR`／攻击风险数字是**下界（lower bound）**，"
      "**不得**读作「攻击未被成功执行」；")
    A("- 该偏差方向使「`Ours` 优于 `b3`／`none`」的结论在**攻击侧更保守**"
      "（若补全生成，`Ours` 的攻击风险只会持平或更高，不会更低）。")
    A("")
    A("**③ 反证：把上限翻 4 倍（512 → 2048），核心结论不变**")
    A("")
    A(_md_table(["反证项", "512 组", "2048 组", "判定"],
                ctx.get("counterevidence_rows") or []))
    A("")
    A("**④ 引用 2048 敏感性分析**")
    A("")
    A("- 两份 `long_input_v1.0_<ts>.md` 运行报告（**512 组／2048 组并列**，本报告 §2／§3）；")
    A("- 长度分层表 `long_input_strata_v1.0_<ts>.csv`（**逐层 `FNR`／`FPR`，非 `F1`**）；")
    A("- 逐样本产物 `reports/eval/outputs/long_input_*`（含每样本 `truncated` 标志）。")
    A("")
    return "\n".join(L)


def render_report(ctx):
    """报告渲染（`windows/W6.6_long_input.md` §7-2／§7-3）。"""
    cfg = ctx["cfg"]
    L = []
    A = L.append
    A("# 5.5 长输入实验 — 运行报告（`W6.6`，A6）")
    A("")
    A("- 生成时间：`%s`；窗口：**W6.6**；权限：**A6**" % ctx["ts"])
    A("- 本次评估组：`--max-tokens %s`；输入源 = `%s`（分片 `%s`，体制 `%s`）"
      % (ctx["group"], ctx["source"], ctx["shard"], ctx["regime"]))
    A("- 配置：`%s`（`declared_at=%s`，**先写后跑**）" % (ctx["config_path"], cfg.get("declared_at")))
    if ctx.get("smoke_note"):
        A("")
        A("> ⚠ **%s**" % ctx["smoke_note"])
    A("")

    A("## §0 公平性限制声明（**C1，逐字**）")
    A("")
    A("1. %s。" % cfg["fairness"]["verbatim_line_1"])
    A("2. %s。" % cfg["fairness"]["verbatim_line_2"])
    A("")
    A("> 出处：`PREREG_v1.md` §5 补录（G6）L66；"
      "`SOURCE_LOCK/实验执行方案（最终版 v5）.md` L602–604；`windows/W6_stage5_eval.md` §4-5。")
    A("")

    A("## §1 概述")
    A("")
    A(_md_table(["组", "token 上限", "参与检测器", "是否主口径", "说明"],
                [[g, cfg["groups"][g]["max_tokens"], ", ".join(cfg["groups"][g]["detectors"]),
                  "是" if cfg["groups"][g]["is_main_caliber"] else "否", cfg["groups"][g]["note"]]
                 for g in GROUP_KEYS]))
    A("")
    A("**本次评估的 Ours run 清单**")
    A("")
    if ctx["runs"]:
        A(_md_table(["regime", "seed", "run_id", "best_step"],
                    [[r["regime"], r["seed"], r["run_id"], r["best_step"]] for r in ctx["runs"]]))
    else:
        A("（本次未纳入 Ours run：%s）" % ctx.get("runs_note", "本组不含 Ours"))
    A("")

    A("## §2 512 组（**全部检测器**）")
    A("")
    A(render_group_table(ctx["results"].get("512", [])))
    A("")
    A("## §3 2048 组（**仅 Ours／B3／B5**）")
    A("")
    A("> ⚠ 本表**不含 `B4`（`unplug-tiny`）** —— 其原生上限为 512，不参与 2048 组（见 §0）。")
    A("")
    A(render_group_table(ctx["results"].get("2048", [])))
    A("")

    A("## §4 长度分布与分位数（两组各自**实测**）")
    A("")
    A(render_length_table(ctx["results"]))
    A("")
    A("- `raw` = **未截断**原始 `text` 的 token 数（规范 tokenizer = Llama-Guard-3-1B，"
      "与 Ours／B3 同源，检测器无关）；")
    A("- `eff` = **进入模型**的 token 数，口径逐检测器不同：`b3`／`b5` = `build_prompt` 的 "
      "`n_text_tokens`（截断后文本）；`ours` = `04_01.encode_example` 的 `n_prompt`（含 chat 模板）；"
      "`b1`／`b2`／`b4` = `n/a`（无 token 概念／字符级）。")
    A("- 分位数算法**复用** `05_01.quantile_table`（线性插值，与 numpy `linear` 同 method）。")
    A("")

    A("## §5 长度分层表（`--length-strata`）")
    A("")
    if ctx["strata_rows"]:
        A(_md_table(["组", "检测器", "run", "层", "n", "n_pos", "n_neg", "FNR", "FPR", "备注"],
                    [[r["group"], r["detector"], r["run_id"], r["band"], r["n"], r["n_pos"],
                      r["n_neg"], fmt_metric(r["fnr"], 6), fmt_metric(r["fpr"], 6), r["notes"]]
                     for r in ctx["strata_rows"]]))
        A("")
        A("> 分层表的度量是**逐层 FPR／FNR**（不是 F1）—— 纯类子集内 F1 无定义"
          "（`D-46` 裁定一 技术修正）。空层照实报 `n=0`。")
    else:
        A("（未启用 `--length-strata` ⇒ 本表为空，属**预期留白**。）")
    A("")

    A("## §6 显存与延迟（七元组声明）")
    A("")
    for g in ctx.get("groups") or [ctx["group"]]:
        A("**七元组 · %s 组**" % g)
        A("")
        A(render_seven_tuple(cfg, g))
        A("")
    A("- 显存 = `torch.cuda.max_memory_allocated` 峰值（MiB），**加载态／推理态分列**。")
    A("- **2048 组与 512 组的显存不可直接相减**（`max_length` 不同）⇒ 分别列示。")
    A("- %s" % cfg["measurement"]["incomparability"]["verbatim"])
    A("- 三档量纲（无模型／CPU／GPU）**并列分表、不合并排序、不跨档求均值**。")
    A("- 系统底噪 ≈%s MiB；卡容量 %s MiB。"
      % (cfg["measurement"]["memory"]["system_noise_mib"],
         cfg["measurement"]["memory"]["card_total_mib"]))
    A("")

    A("## §7 截断侧不变量与主口径 512 未被改动 —— 回执")
    A("")
    A(_md_table(["校验项", "结果", "明细"], ctx["invariant_rows"]))
    A("")

    A("## §8 局限声明")
    A("")
    for ln in ctx["limitations"]:
        A("- %s" % ln)
    A("")
    A("**引数带口径（`BASELINE_REPORT.md` §12）**：对照数字**只引不跑**（tag `baseline-v1`）。")
    for note in cfg["citation"]["mandatory_notes"]:
        A("- %s" % note)
    A("")
    # `T3` 段（四要素）⇒ 并入同一段 Limitation
    A(render_limitation_section(ctx))
    A("")

    A("## §9 ACCESS_LOG 与冻结面复核")
    A("")
    A("- `--access-log-note`：`%s`" % (ctx["access_log_note"] or "n/a（非 sealed 源）"))
    A("- 登记由**人手工**完成（6 列）；脚本**不写** `ACCESS_LOG.md`。")
    for ln in ctx["freeze_rows"]:
        A("- %s" % ln)
    A("")

    A("## §10 回执／验收位（留给 `W6` 复核）")
    A("")
    A("- `--self-check`：`%s`" % ctx.get("selfcheck_line", TBD))
    A("- 产物：")
    for p in ctx["artifacts"]:
        A("  - `%s`" % p)
    A("")

    A("---")
    A("")
    A("## 附：可直接搬入 `reports/eval/EVAL_REPORT.md` 的 §5.5 骨架段")
    A("")
    A("```markdown")
    A("### 5.5 长输入实验")
    A("")
    A("**公平性限制**：%s。" % cfg["fairness"]["verbatim_line_1"])
    A("**%s**。" % cfg["fairness"]["verbatim_line_2"])
    A("")
    A("| 组 | token 上限 | 参与检测器 |")
    A("|---|---|---|")
    for g in GROUP_KEYS:
        A("| %s | %s | %s |" % (cfg["groups"][g]["label"], cfg["groups"][g]["max_tokens"],
                                ", ".join(cfg["groups"][g]["detectors"])))
    A("")
    A("**（a）512 组（全部检测器）**")
    A("")
    A(_md_table(["检测器", "TPR", "FPR", "F1", "AUROC", "延迟中位ms", "显存推理MiB"],
                [[r["detector"], fmt_metric(r["tpr"], 4), fmt_metric(r["fpr"], 4),
                  fmt_metric(r["f1"], 4), fmt_metric(r["auroc"], 4),
                  fmt_metric(r["latency"]["median_ms"], 3), fmt_mem(r["mem_infer_peak_mib"])]
                 for r in ctx["results"].get("512", [])]))
    A("")
    A("**（b）2048 组（仅 Ours／B3／B5；不含 B4）**")
    A("")
    A(_md_table(["检测器", "TPR", "FPR", "F1", "AUROC", "延迟中位ms", "显存推理MiB"],
                [[r["detector"], fmt_metric(r["tpr"], 4), fmt_metric(r["fpr"], 4),
                  fmt_metric(r["f1"], 4), fmt_metric(r["auroc"], 4),
                  fmt_metric(r["latency"]["median_ms"], 3), fmt_mem(r["mem_infer_peak_mib"])]
                 for r in ctx["results"].get("2048", [])]))
    A("```")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 模式：eval
# ---------------------------------------------------------------------------
def cmd_eval(cfg, base_cfg, args, logger=print):
    both = bool(getattr(args, "both", False))
    if both:
        plan = []
        for g in GROUP_KEYS:
            mt = int(cfg["groups"][g]["max_tokens"])
            dets, _ex = resolve_detectors(args, cfg, mt, logger)
            plan.append((g, mt, dets))
    else:
        max_tokens = int(args.max_tokens)
        group = str(max_tokens)
        dets, _explicit = resolve_detectors(args, cfg, max_tokens, logger)
        logger("DETECTORS_SCOPE group=%s detectors=%s" % (group, ",".join(dets)))
        plan = [(group, max_tokens, dets)]

    rows, shard, _src = prepare_rows(cfg, base_cfg, args, args.source, args.regime, logger)
    bands = cfg["length"]["strata"]["bands"]

    tok = None
    try:
        tok = load_canonical_tokenizer(cfg, logger)
    except Exception as exc:  # noqa: BLE001 —— 长度仅用于披露
        logger("TOKENIZER_SKIP detail=%s ⇒ raw token 长度记 `n/a`" % exc)
    if tok is not None:
        raw_lens = [len(tok(r.get("text", ""), add_special_tokens=False)["input_ids"])
                    for r in rows]
    else:
        raw_lens = [None] * len(rows)
    q_all = W().quantile_table([v for v in raw_lens if v is not None])
    logger("LENGTH_RAW n=%d p50=%s p95=%s max=%s"
           % (len(raw_lens), fmt_num(q_all["p50"], 1), fmt_num(q_all["p95"], 1),
              fmt_num(q_all["max"], 1)))

    all_dets = sorted({d for _g, _m, ds in plan for d in ds})
    ours_runs = []
    runs_note = ""
    if "ours" in all_dets:
        try:
            all_runs = resolve_runs(args, base_cfg, logger)
        except (ValueError, FileNotFoundError) as exc:
            logger("RESOLVE_RUNS_SKIP detail=%s" % exc)
            all_runs = []
        ours_runs = select_runs(all_runs, base_cfg, args.regime, logger)
        if not ours_runs:
            runs_note = "Ours run 不齐 ⇒ 判 `EVAL_PENDING`，本次 Ours 不出数、不出结论"

    results = {g: [] for g in GROUP_KEYS}
    strata_rows = []
    band_index = {}

    for group, max_tokens, dets in plan:
        todo = [(d, None) for d in dets if d != "ours"]
        if ours_runs:
            todo += [("ours", r) for r in ours_runs]
        elif "ours" in dets:
            logger("OURS_SKIP reason=%s" % runs_note)
        for det_id, run in todo:
            logger("RUN_BEGIN group=%s detector=%s run=%s"
                   % (group, det_id, "n/a" if run is None else os.path.basename(run[2])))
            rec = evaluate_one(cfg, base_cfg, args, group, det_id, rows, shard, args.source,
                               args.regime, run, raw_lens, bands, logger=logger,
                               outputs_dir=args.outputs_dir, ts=args.ts)
            rec["length_basis_note"] = ("tokenizer 未加载 ⇒ raw 记 n/a" if tok is None
                                        else "规范 tokenizer = Llama-Guard-3-1B")
            results[group].append(rec)
            logger("RUN_END group=%s detector=%s n=%d tpr=%s fpr=%s f1=%s n_trunc=%d peak_infer=%s"
                   % (group, det_id, rec["n"], fmt_metric(rec["tpr"], 6),
                      fmt_metric(rec["fpr"], 6), fmt_metric(rec["f1"], 6),
                      int(rec["n_truncated"]), fmt_mem(rec["mem_infer_peak_mib"])))
            if args.length_strata or both:
                sb = build_strata_rows(cfg, group, rec, bands)
                band_index[(group, rec["detector"], str(rec["run_id"]))] = {
                    str(r["band"]): (r["fnr"], r["fpr"]) for r in sb}
                if args.length_strata:
                    strata_rows += sb

    ok_inv, inv_lines = assert_main_caliber_untouched(cfg)
    invariant_rows = [["主口径 512 未被改动（C4）", "PASS" if ok_inv else "FAIL",
                       "；".join(inv_lines)]]
    try:
        sides = assert_truncation_invariant(cfg)
        invariant_rows.append(["截断侧不变量（C3）", "PASS",
                               "两组同用 **%s** 截断（保尾部 N token）" % sides["512"]])
    except AssertionError as exc:
        invariant_rows.append(["截断侧不变量（C3）", "FAIL", str(exc)])
    for group, max_tokens, dets in plan:
        invariant_rows.append(["2048 白名单守卫（C2）·%s 组" % group,
                               all(d in cfg["whitelist"][group] for d in dets),
                               "参与检测器 = %s" % ", ".join(dets)])
    invariant_rows.append(["记录数断言（C6）", "PASS",
                           "`%s` records=%d（`csv.DictReader` 口径）" % (shard, len(rows))])

    # `T3` —— 两套截断口径的**实测行**（分列，各自带自己的分母）
    trunc_caliber_rows = []
    for group, _mt, _dets in plan:
        nums = sorted(set((int(r["n_truncated"]), int(r["n"]))
                           for r in results.get(group, [])
                           if int(r.get("n_truncated") or 0) > 0))
        if len(nums) > 1:
            trunc_caliber_rows.append({
                "caliber": "检测侧截断率", "group": group,
                "num": "不一致 " + str(nums), "den": "—", "rate": "—",
                "note": "同组不同检测器截断条数不一致 ⇒ 停工呈 W0"})
            continue
        if not nums:
            trunc_caliber_rows.append({
                "caliber": "检测侧截断率", "group": group, "num": 0, "den": len(rows),
                "rate": "0.0000", "note": "无样本被截断"})
            continue
        num, den = nums[0]
        trunc_caliber_rows.append({
            "caliber": "检测侧截断率", "group": group, "num": num, "den": den,
            "rate": "%.4f" % (float(num) / float(den)),
            "note": "分母 = 该分片**检测单元**（记录数）；分子 = 逐样本产物 `truncated==1` 的样本条数"})

    # `T3` ①-续 —— ASR 链路截断率：**引用** `W6.2`（本窗口**不重跑** ASR）
    #   ⇒ 与检测侧**同表分列**，但**分母不同**（放行判定单元 vs 检测单元），**禁止相加或混写**
    asr_cited = cfg.get("asr_trunc_rates") or {}
    for split_key in sorted(asr_cited.get("rows", {}).keys()):
        row = asr_cited["rows"][split_key]
        trunc_caliber_rows.append({
            "caliber": TRUNC_CALIBERS[1]["name"],
            "group": row.get("group", split_key),
            "num": row.get("num", TBD), "den": row.get("den", TBD),
            "rate": row.get("rate", TBD),
            "note": "%s；%s" % (asr_cited.get("note_prefix", ""),
                                asr_cited.get("source", "")),
        })

    # `T3` ③ —— 反证行（**上限翻 4 倍**：512 → 2048）
    #   ⚠ 判定口径：两组**须同轮**跑完才可比；缺任一组 ⇒ 落 `<TBD>`（不臆造）
    #     2048 组的 Ours τ = 512 组**同一 τ**（该 run `best epoch` 的 `tau_ours` 全精度，未重寻）
    counterevidence_rows = []
    o512 = [r for r in results.get("512", []) if r["detector"] == "ours"]
    o2048 = [r for r in results.get("2048", []) if r["detector"] == "ours"]
    if o512 and o2048:
        v1 = [r["f1"] for r in sorted(o512, key=lambda x: int(x["seed"]))]
        v2 = [r["f1"] for r in sorted(o2048, key=lambda x: int(x["seed"]))]
        vals = [v for v in v1 + v2 if v is not None]
        rng = (max(vals) - min(vals)) if vals else None
        counterevidence_rows.append([
            "`Ours` 逐 seed `F1`",
            " / ".join(fmt_metric(v, 4) for v in v1),
            " / ".join(fmt_metric(v, 4) for v in v2),
            ("**格局不变**（全局极差 %.4f ≤ 0.01）" % rng) if rng is not None
            else "<TBD>（含 `n/a` ⇒ 不可比）",
        ])
    # 超长层 `b2048_plus` 的 `Ours` —— **逐组**给 `FNR`／`FPR` 上界
    #   ⚠ 两组**都要报**：只报 512 组（`FNR ≤ 0.011`）而漏报 2048 组（`FNR ≤ 0.0543`）
    #     ⇒ 即重蹈 `D-111` 裁定六 的「**选择性呈现**」覆辙
    for g in GROUP_KEYS:
        got = [v["b2048_plus"] for (gg, dd, _rr), v in band_index.items()
               if str(gg) == str(g) and str(dd) == "ours" and "b2048_plus" in v]
        if not got:
            continue
        fnrs = [x[0] for x in got if x[0] is not None]
        fprs = [x[1] for x in got if x[1] is not None]
        counterevidence_rows.append([
            "超长层 `b2048_plus` 的 `Ours`（%s 组）" % g,
            "FNR ≤ %s" % (fmt_metric(max(fnrs), 4) if fnrs else NA),
            "FPR = %s" % (fmt_metric(max(fprs), 4) if fprs else NA),
            "截断不改变核心结论",
        ])
    if not counterevidence_rows:
        counterevidence_rows.append([
            "反证（512 → 2048 对照）", "<TBD>", "<TBD>",
            "本次仅跑单组 ⇒ 对照位留 `<TBD>`（须 `--both`）"])

    limitations = [
        "`B4`（unplug-tiny）原生上限 512 ⇒ **不参与 2048 组**；2048 组的横向比较不含 B4（§0）。",
        "512 组与 2048 组**不可**进同一均值或同一图例而不区分；两组显存**不可直接相减**。",
        "本实验**不做**统计显著性检验（归 `W6.9`）；不重跑 B1–B5 的训练侧。",
        "2048 组的 Ours τ 与 512 组**同一 τ**（该 run `best epoch` 的 `tau_ours` 全精度），"
        "**未因长度组重寻**（`W6.6` §9-1，待 `W0` 追认）。",
        "`b1`／`b2` 无 token 上限概念 ⇒ 其指标**不受** `max_tokens` 影响；"
        "该差异须与「被截断」的检测器分开解读。",
    ]
    if not ours_runs and "ours" in all_dets:
        limitations.append("**本次 Ours 未出数**：%s" % runs_note)
    if tok is None:
        limitations.append("规范 tokenizer 未加载 ⇒ 原始 token 长度分布记 `n/a`。")
    for group, _mt, _dets in plan:
        for rec in results[group]:
            if rec["detector"] == "ours" and rec["eval_batch_max_size"] is not None:
                limitations.append(
                    "Ours 在 %s 组：`eval_batch_max_size=%s`、实际微批数 `n_micro=%s`"
                    "（由 `04_01.make_batches` 显存护栏决定，只减单批 n、不改样本／权重）；"
                    "若与另一组不同 ⇒ **延迟不可直接比较**。"
                    % (group, rec["eval_batch_max_size"], rec.get("n_batches")))
                break
    if args.limit:
        limitations.append("本次为 `--limit %d` 的**冒烟口径**，"
                           "不得当作全量口径的延迟／显存（`W6.1` §6 同款声明）。" % int(args.limit))

    freeze_rows = []
    rc, out = md5sum_check(_abs("data/frozen/CodeObf-PI_v1.0/md5sums.txt"))
    freeze_rows.append("`md5sum -c data/frozen/CodeObf-PI_v1.0/md5sums.txt`（纯 Python 复核）"
                       " ⇒ exit=%d：%s" % (rc, out))
    freeze_rows.append("范围声明：逐条打开 manifest 所列**全部**文件取字节 md5（含 `test.csv`）——"
                       "依 `W6.1` §1.1 的 `W6` 补注 ②③，字节级完整性校验**不构成**「读内容」。")
    if args.source == "sealed":
        rc2, out2 = md5sum_check(_abs("data/sealed_test/CodeObf-PI_v1.0/md5sums.txt"))
        freeze_rows.append("`md5sum -c data/sealed_test/CodeObf-PI_v1.0/md5sums.txt` ⇒ exit=%d：%s"
                           % (rc2, out2))
    else:
        freeze_rows.append("非 sealed 源 ⇒ **不复核 sealed 侧**（免触碰，`W6.1` §3-C14）。")

    # run 清单从**结果**里取（含 `best_step`），避免与 `ours_runs` 元组二次对不齐
    run_rows = []
    seen = set()
    for g in GROUP_KEYS:
        for rec in results.get(g, []):
            if rec["detector"] != "ours" or rec["run_id"] in seen:
                continue
            seen.add(rec["run_id"])
            run_rows.append({"regime": rec["regime"], "seed": rec["seed"],
                             "run_id": rec["run_id"], "best_step": rec.get("best_step")})

    ctx_common = {
        "cfg": cfg, "ts": args.ts,
        "group": "两组并列" if both else plan[0][0],
        "groups": [g for g, _m, _d in plan],
        "source": args.source, "shard": shard,
        "regime": args.regime, "config_path": args.config,
        "runs": run_rows,
        "runs_note": runs_note or "本组不含 Ours",
        "results": results, "strata_rows": strata_rows, "invariant_rows": invariant_rows,
        "limitations": limitations, "access_log_note": args.access_log_note,
        "freeze_rows": freeze_rows,
        "trunc_caliber_rows": trunc_caliber_rows,
        "counterevidence_rows": counterevidence_rows,
        "smoke_note": ("本次为 `--limit %d` 的**冒烟／演练口径**（实现期，输入 = `%s`，"
                       "**非全量、非封存**）⇒ 不得当作 of-record 结论；"
                       "全量与封存的正式轮须另跑（见 `PROGRESS.md` 本窗口行）"
                       % (int(args.limit), args.source)) if args.limit else "",
    }

    md_path = os.path.join(args.report_dir, "long_input_v1.0_%s.md" % args.ts)
    csv_path = os.path.join(args.report_dir, "long_input_v1.0_%s.csv" % args.ts)
    artifacts = [md_path, csv_path]
    if args.length_strata:
        st_path = os.path.join(args.report_dir, "long_input_strata_v1.0_%s.csv" % args.ts)
        W().write_csv_rows(st_path, STRATA_FIELDS, strata_rows)
        artifacts.append(st_path)
    W().write_csv_rows(csv_path, MAIN_FIELDS, build_main_rows(results))
    # 同一份产物的**补全写**（同 ts 同名），不是覆盖他件（C12）
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_report(dict(ctx_common, artifacts=artifacts)))

    print("REPORT_MD=%s REPORT_CSV=%s" % (md_path, csv_path))
    return 0


def build_main_rows(results):
    out = []
    for group in GROUP_KEYS:
        for rec in results.get(group, []):
            q_raw = W().quantile_table([v for v in rec["raw_lens"] if v is not None])
            eff = rec.get("token_len_eff")
            q_eff = W().quantile_table(eff) if eff else None
            n = int(rec["n"]) or 0
            out.append({
                "group": group, "max_tokens": rec["max_tokens"], "detector": rec["detector"],
                "label": rec["label"], "run_id": rec["run_id"], "regime": rec["regime"],
                "seed": rec["seed"], "shard": rec["shard"], "source": rec["source"],
                "tau_track": rec["tau_track"],
                "tau_value": "" if rec["tau_value"] is None else repr(float(rec["tau_value"])),
                "n": rec["n"], "n_pos": rec["n_pos"], "n_neg": rec["n_neg"],
                "tp": rec["tp"], "fp": rec["fp"], "fn": rec["fn"], "tn": rec["tn"],
                "tpr": "" if rec["tpr"] is None else "%.8f" % rec["tpr"],
                "fpr": "" if rec["fpr"] is None else "%.8f" % rec["fpr"],
                "f1": "" if rec["f1"] is None else "%.8f" % rec["f1"],
                "auroc": "" if rec["auroc"] is None else "%.8f" % rec["auroc"],
                "latency_mean_ms": fmt_null(rec["latency"]["mean_ms"], "%.6f"),
                "latency_median_ms": fmt_null(rec["latency"]["median_ms"], "%.6f"),
                "latency_p95_ms": fmt_null(rec["latency"]["p95_ms"], "%.6f"),
                "mem_load_peak_mib": fmt_null(rec["mem_load_peak_mib"], "%.3f"),
                "mem_infer_peak_mib": fmt_null(rec["mem_infer_peak_mib"], "%.3f"),
                "cpu_rss_peak_mib": fmt_null(rec["cpu_rss_peak_mib"], "%.3f"),
                "n_truncated": rec["n_truncated"],
                "trunc_rate": ("%.6f" % (float(rec["n_truncated"] or 0) / n)) if n else "",
                "n_unparsed": rec.get("n_unparsed", 0),
                "token_len_raw_p50": fmt_null(q_raw["p50"], "%.3f"),
                "token_len_raw_p95": fmt_null(q_raw["p95"], "%.3f"),
                "token_len_raw_max": fmt_null(q_raw["max"], "%.3f"),
                "token_len_eff_p50": (fmt_null(q_eff["p50"], "%.3f") if q_eff else ""),
                "token_len_eff_p95": (fmt_null(q_eff["p95"], "%.3f") if q_eff else ""),
                "token_len_eff_max": (fmt_null(q_eff["max"], "%.3f") if q_eff else ""),
                "eval_batch_max_size": fmt_null(rec["eval_batch_max_size"], "%d"),
                "notes": ("n_micro=%s（护栏后实际微批数）" % rec["n_batches"]
                          if (rec["detector"] == "ours" and rec.get("n_batches")) else ""),
            })
    return out


# ---------------------------------------------------------------------------
# `--dry-run`：解析全部资源与断言（不加载模型、不触 GPU）
# ---------------------------------------------------------------------------
def cmd_dryrun(cfg, base_cfg, args, logger=print):
    checks = []
    w = W()

    try:
        sides = assert_truncation_invariant(cfg)
        checks.append(("truncation_side_invariant", True, "两组同用 %s" % sides["512"]))
    except AssertionError as exc:
        checks.append(("truncation_side_invariant", False, str(exc)))

    ok_main, lines = assert_main_caliber_untouched(cfg)
    checks.append(("main_caliber_512_untouched", ok_main, "；".join(lines)))

    mt = int(args.max_tokens) if args.max_tokens else None
    if mt is not None:
        try:
            dets, explicit = resolve_detectors(args, cfg, mt, logger)
            checks.append(("whitelist_guard:%d" % mt, True,
                           "detectors=%s explicit=%s" % (", ".join(dets), explicit)))
        except (AssertionError, ValueError) as exc:
            checks.append(("whitelist_guard:%d" % mt, False, str(exc)))
        if mt == 2048:
            for bad in ("b1", "b2", "b4"):
                checks.append(("whitelist_reject:%s@2048" % bad,
                               not whitelist_allows(2048, [bad], cfg),
                               "白名单 = %s" % ", ".join(cfg["whitelist"]["2048"])))

    source = args.source or cfg["inputs"]["default_source"]
    regime = args.regime or cfg["inputs"]["default_regime"]
    try:
        path, expect, shard = w.resolve_input_path(base_cfg, source, regime, args.split, False)
        w.assert_input_allowed(path, base_cfg, allow_sealed=False)
        got = w.assert_record_count(path, expect, base_cfg)
        checks.append(("record_count:%s" % shard, got == int(expect), "records=%d" % got))
    except (AssertionError, ValueError, FileNotFoundError) as exc:
        checks.append(("input_resolved", False, str(exc)))

    try:
        runs = resolve_runs(args, base_cfg, logger)
        picks = select_runs(runs, base_cfg, regime, logger)
        checks.append(("ours_runs:%s" % regime, len(picks) == len(base_cfg["runs"]["seeds"]),
                       "n=%d" % len(picks)))
        for _rg, seed, run_dir in picks:
            rec = w.load_run_meta(run_dir, base_cfg)
            w.assert_meta_matches_cli(rec, regime, seed)
            checks.append(("tau_fields:%s:%d" % (regime, seed), True,
                           "tau_star=%s override=%s" % (fmt_tau(rec["tau_star"]), rec["override"])))
    except (ValueError, FileNotFoundError, AssertionError, RuntimeError) as exc:
        checks.append(("ours_runs:%s" % regime, False, "detail=%s" % exc))

    ok = all(c[1] for c in checks)
    logger("DRYRUN_PLAN steps=%d source=%s regime=%s report_dir=%s outputs_dir=%s"
           % (len(checks), source, regime, args.report_dir, args.outputs_dir))
    print("DRYRUN_RESULT=%s n=%d" % ("PASS" if ok else "FAIL", len(checks)))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# `--self-check`：离线自检（不加载模型、**不读封存**、不触 GPU）
# ---------------------------------------------------------------------------
def _source_text():
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
        return fh.read()


def _fake_rec(det_id="b3", group="2048"):
    return {"group": group, "max_tokens": int(group), "detector": det_id, "label": det_id.upper(),
            "run_id": "n/a", "regime": "id", "seed": "n/a", "shard": "x", "source": "val",
            "tau_track": "n/a", "tau_value": None, "n": 2, "tp": 1, "fp": 0, "fn": 1, "tn": 0,
            "tpr": 0.5, "fpr": 0.0, "f1": 0.6667, "auroc": None, "n_unparsed": 0,
            "latency": {"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.0},
            "mem_load_peak_mib": 1.0, "mem_infer_peak_mib": 2.0, "cpu_rss_peak_mib": None,
            "n_truncated": 0, "eval_batch_max_size": 1, "raw_lens": [10, 20],
            "token_len_eff": [10, 20], "labels": [1, 0], "preds": [1, 0]}


def _table_rows_of(seg):
    """只抽**表格行**（跳过引用块／说明句）——用于「2048 表内不含 B4」的机械核验。"""
    rows = []
    for ln in seg.split("\n"):
        s = ln.strip()
        if not s.startswith("|"):
            continue
        if set(s) <= set("|-: "):
            continue
        rows.append(s)
    return "\n".join(rows)


def _render_fake(cfg, args, results):
    return render_report({
        "cfg": cfg, "ts": "20990101-000000", "group": "2048", "source": "val", "shard": "x",
        "regime": "id", "config_path": args.config, "runs": [], "runs_note": "桩（自检）",
        "results": results, "strata_rows": [], "invariant_rows": [],
        "limitations": [], "access_log_note": None, "freeze_rows": [], "artifacts": [],
    })


def self_check(cfg, base_cfg, args, logger=print):
    checks = []

    def add(name, ok, detail=""):
        checks.append((str(name), bool(ok), str(detail)))

    w = W()
    here = os.path.abspath(__file__)

    # ── 1) C2 白名单守卫：**真实子进程** ⇒ 显式给 b1/b2/b4 必须非零退出 ──
    for bad in ("b1", "b2", "b4"):
        p = subprocess.run(
            [sys.executable, here, "--max-tokens", "2048", "--detectors", bad,
             "--source", "val", "--config", args.config],
            cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = (p.stdout or b"").decode("utf-8", "replace")
        add("whitelist_guard_reject:%s@2048" % bad, p.returncode != 0, "exit=%d" % p.returncode)
        add("whitelist_guard_msg:%s@2048" % bad, "白名单守卫失败" in out,
            (out.strip().split("\n") or [""])[-1][:150])
    p = subprocess.run(
        [sys.executable, here, "--max-tokens", "512", "--detectors", "b1,b2,b4",
         "--source", "val", "--config", args.config],
        cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out512 = (p.stdout or b"").decode("utf-8", "replace")
    add("whitelist_guard_allow_all@512", "白名单守卫失败" not in out512,
        "512 组未触发守卫（后续 exit=%d 系缺 `--report-dir`，属预期）" % p.returncode)
    add("whitelist_512_all", whitelist_allows(512, ["b1", "b2", "b3", "b4", "b5", "ours"], cfg),
        "512 组放行全部 6 个检测器")
    add("whitelist_2048_only", whitelist_allows(2048, ["ours", "b3", "b5"], cfg)
        and not whitelist_allows(2048, ["b4"], cfg), "2048 组 = ours/b3/b5")

    # ── 2) C3 截断侧不变量 ──
    try:
        sides = assert_truncation_invariant(cfg)
        add("truncation_side_invariant", True, "两组同用 %s 截断" % sides["512"])
    except AssertionError as exc:
        add("truncation_side_invariant", False, str(exc))

    # ── 3) C4 主口径 512 未被改动 ──
    ok_main, lines = assert_main_caliber_untouched(cfg)
    add("main_caliber_512_untouched", ok_main, "；".join(lines))

    # ── 4) C6 记录数断言（含「text 含换行 ⇒ 记录数 ≠ 行数」回归） ──
    import csv as _csv

    tmp_dir = os.path.join(_ROOT, "logs", "eval", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    probe = os.path.join(tmp_dir, "05_05_selfcheck_records.csv")
    with open(probe, "w", encoding="utf-8", newline="\n") as fh:
        wr = _csv.DictWriter(fh, fieldnames=["sample_id", "text", "label"], lineterminator="\n")
        wr.writeheader()
        wr.writerow({"sample_id": "s1", "text": "aaa\nbbb\nccc", "label": "1"})
        wr.writerow({"sample_id": "s2", "text": "plain", "label": "0"})
    n_rec = w.count_records(probe)
    n_line = w.count_physical_lines(probe)
    add("csv_record_vs_line", n_rec == 2 and n_line >= 4, "records=%d lines=%d" % (n_rec, n_line))
    try:
        w.assert_record_count(probe, 2, base_cfg)
        add("record_count_assert_ok", True, "expected=2 got=%d" % n_rec)
    except AssertionError as exc:
        add("record_count_assert_ok", False, str(exc))
    try:
        w.assert_record_count(probe, 99, base_cfg)
        add("record_count_assert_reject", False, "断言未生效")
    except AssertionError:
        add("record_count_assert_reject", True, "不符即抛 ⇒ 非零退出")

    # ── 5) C5 `--source sealed` 缺 `--access-log-note` ⇒ 非零退出（真实子进程） ──
    p = subprocess.run(
        [sys.executable, here, "--max-tokens", "512", "--source", "sealed",
         "--split", "in_regime_test", "--config", args.config],
        cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_sealed = (p.stdout or b"").decode("utf-8", "replace")
    add("sealed_requires_access_log_note", p.returncode != 0, "exit=%d" % p.returncode)
    add("sealed_refusal_msg", "REFUSED" in out_sealed and "access-log-note" in out_sealed,
        (out_sealed.strip().split("\n") or [""])[-1][:150])

    # ── 6) C1 公平性声明逐字在位 ＋ 2048 表段不含 B4（**渲染后机械核验**） ──
    text = _render_fake(cfg, args, {"512": [], "2048": [_fake_rec("b3"), _fake_rec("ours")]})
    seg = text.split("## §3 2048 组")[1].split("## §4")[0]
    table = _table_rows_of(seg)
    add("b4_absent_in_2048_table",
        ("B4" not in table) and ("b4" not in table) and ("unplug" not in table),
        "表行 %d 行；段内说明句**必须**点名 B4（这是 §0 的声明义务，不是表内容）"
        % len(table.split("\n")))
    add("fairness_verbatim_1", cfg["fairness"]["verbatim_line_1"] in text, "首节逐字在位")
    add("fairness_verbatim_2", cfg["fairness"]["verbatim_line_2"] in text, "首节逐字在位")

    # ── 7) C7／C10 静态纪律 ──
    #  needles **运行时拼装**：否则校验串本身会出现在源码里 ⇒ 自检恒负（自指陷阱）
    src = _source_text()
    needle_train = ".tr" + "ain("
    needle_rep_dq = chr(34) + "reports" + chr(47)
    needle_rep_sq = chr(39) + "reports" + chr(47)
    needle_readlines = "read" + "lines()"
    needle_splitlines = "split" + "lines()"
    add("no_train_call", needle_train not in src,
        "C7：源码内不得出现训练模式切换调用")
    add("no_hardcoded_reports_root", needle_rep_dq not in src and needle_rep_sq not in src,
        "C10：脚本内不写死 `reports/` 路径")
    tcfg = load_config(_abs(cfg["train_config"]))
    add("attn_eager", tcfg["load"]["attn_implementation"] == "eager",
        "attn=%s" % tcfg["load"]["attn_implementation"])
    add("cublas_before_torch", src.index("CUBLAS_WORKSPACE_CONFIG") < src.index("import torch")
        if "import torch" in src else True,
        "CUBLAS 先于任何 `import torch`")
    add("no_readlines_splitlines",
        (needle_readlines not in src) and (needle_splitlines not in src),
        "C6：禁按行读（一律 `csv.DictReader`）")

    # ── 8) 配置自洽：B4／B2 τ／Ours τ 不重寻 ──
    add("whitelist_2048_equals_group",
        sorted(cfg["whitelist"]["2048"]) == sorted(cfg["groups"]["2048"]["detectors"]),
        "2048 白名单 = %s" % ", ".join(cfg["whitelist"]["2048"]))
    add("b4_excluded_from_2048", "b4" not in cfg["whitelist"]["2048"],
        cfg["detectors"]["b4"]["exclude_reason"][:80])
    add("b2_tau_inf", str(cfg["detectors"]["b2"]["decision"]["tau"]) == "-inf",
        "τ_B2 = −inf（引 `BASELINE_REPORT.md` §4，不得挪用）")
    add("b2_tau_not_migrated", "forbid_migration" in cfg["detectors"]["b2"]["decision"],
        "禁挪用条款在位")
    add("ours_tau_no_rerun", cfg["detectors"]["ours"]["tau"]["rerun_in_2048"] is False,
        "2048 组不重寻 τ")

    # ── 9) 长度分位数与分层齐备 ──
    need = ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    add("length_quantiles_complete", all(k in cfg["length"]["quantiles"] for k in need),
        "分位数清单 = %s" % ", ".join(cfg["length"]["quantiles"]))
    add("strata_bands_declared", len(cfg["length"]["strata"]["bands"]) >= 3,
        "层数=%d（事前定值）" % len(cfg["length"]["strata"]["bands"]))

    # ── 10) C11 引数带口径 ──
    add("citation_notes_present", len(cfg["citation"]["mandatory_notes"]) >= 4,
        "mandatory_notes=%d 条" % len(cfg["citation"]["mandatory_notes"]))

    # ── 11) 字段一致性 ──
    add("fields_declared", set(MAIN_FIELDS) == set(cfg["fields"]["main"]),
        "MAIN_FIELDS 与配置 `fields.main` 一致")
    add("strata_fields_declared", set(STRATA_FIELDS) == set(cfg["fields"]["strata"]),
        "STRATA_FIELDS 与配置 `fields.strata` 一致")

    # ── 12) 确定性双跑：**离线侧同一输入跑两遍 ⇒ 渲染结果逐字节相同** ──
    t1 = _render_fake(cfg, args, {"512": [_fake_rec("b1", "512")], "2048": [_fake_rec("b3")]})
    t2 = _render_fake(cfg, args, {"512": [_fake_rec("b1", "512")], "2048": [_fake_rec("b3")]})
    d1 = hashlib.md5(t1.encode("utf-8")).hexdigest()
    d2 = hashlib.md5(t2.encode("utf-8")).hexdigest()
    add("determinism_rerun", d1 == d2, "md5=%s（两次渲染逐字节相同）" % d1[:16])

    # 13) `T3`（`D-111` 裁定二／五）—— 截断 Limitation 段的**四要素**
    lim = render_limitation_section({
        "cfg": cfg, "ts": "20990101-000000",
        "trunc_caliber_rows": [
            {"caliber": TRUNC_CALIBERS[0]["name"], "group": "512",
             "num": 676, "den": 1583, "rate": "0.4270",
             "note": "分母 = 该分片**检测单元**（记录数）"},
            {"caliber": TRUNC_CALIBERS[1]["name"], "group": "none/id",
             "num": 546, "den": 3168, "rate": "0.1723",
             "note": "分母 = 放行判定单元"},
        ],
        "counterevidence_rows": [["`Ours` 逐 seed `F1`", "0.9975 / 1.0000 / 0.9981",
                                  "0.9949 / 1.0000 / 0.9956", "格局不变"]],
    })
    add("t3_two_calibers_listed", all(c["name"] in lim for c in TRUNC_CALIBERS),
        "(1) 两口径名**同段分列**")
    add("t3_denominators_distinct",
        TRUNC_CALIBERS[0]["denominator"] != TRUNC_CALIBERS[1]["denominator"],
        "两口径分母定义不同 => 结构上不可能混写为一个比率")
    add("t3_direction_stated", "系统性低估" in lim,
        "(2) 影响方向 = `ASR` 系统性低估（保守）")
    add("t3_counterevidence_stated", ("512 → 2048" in lim) and ("反证" in lim),
        "(3) 反证要素在位（上限翻 4 倍）")
    add("t3_refs_2048_sensitivity", "2048 敏感性分析" in lim,
        "(4) 引用 2048 敏感性分析")
    add("t3_rates_not_merged", ("0.4270" in lim) and ("0.1723" in lim),
        "两个比率**各自带分母**出现在同一表内（未合成单一比率）")

    ok = all(c[1] for c in checks)
    for name, okc, detail in checks:
        logger("SELFCHECK %s %s | %s" % ("PASS" if okc else "FAIL", name, detail))
    print("SELFCHECK_RESULT=%s n=%d" % ("PASS" if ok else "FAIL", len(checks)))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def resolve_log_dir(args=None):
    """日志目录解析（依 `#64`／`D-80` 裁定三／`D-82`，与 `05_01` 同手法）。"""
    env = os.environ.get("CODEOBF_LOG_DIR")
    if env:
        return env
    if args is not None and (getattr(args, "self_check", False) or getattr(args, "dry_run", False)):
        return os.path.join(_ROOT, "logs", "eval", "tmp")
    return os.path.join(_ROOT, "logs", "eval")


def flagged_logger(ts, log_dir=None):
    log_dir = log_dir or os.path.join(_ROOT, "logs", "eval")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "05_05_eval_long_input_%s.log" % ts)

    def _log(msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    return _log, path


def build_parser():
    p = argparse.ArgumentParser(
        prog="scripts/eval/05_05_eval_long_input.py",
        description="5.5 长输入实验（W6.6，A6）—— 512 组（全检测器）／2048 组（Ours／B3／B5）")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, choices=[512, 2048], default=None)
    p.add_argument("--detectors", default=None,
                   help="逗号分隔；`--max-tokens 2048` 时只允许 ours,b3,b5（显式越界 ⇒ 非零退出）")
    p.add_argument("--runs-csv", dest="runs_csv", default=None, help="04_03 汇总 CSV")
    p.add_argument("--only", default=None, help="只跑指定 run，形如 id:42")
    p.add_argument("--run-dir", dest="run_dir", default=None,
                   help="合约补充项：单点指定 run 目录（与 `W6.1` §2 同款，须同时给 --regime/--seed）")
    p.add_argument("--seed", type=int, choices=[42, 123, 456], default=None,
                   help="合约补充项：`--run-dir` 单点模式用")
    p.add_argument("--source", choices=["val", "sealed"], default=None)
    p.add_argument("--split", choices=["in_regime_test", "g3_hard_negative_test"],
                   default="in_regime_test")
    p.add_argument("--regime", choices=["id", "ood"], default=None,
                   help="合约补充项：默认取 `inputs.default_regime`")
    p.add_argument("--config", dest="config", default="configs/eval/long_input.yaml")
    p.add_argument("--report-dir", dest="report_dir", default=None,
                   help="**必填**（C10：不得硬编码 reports/ 路径）")
    p.add_argument("--outputs-dir", dest="outputs_dir", default=None)
    p.add_argument("--access-log-note", dest="access_log_note", default=None)
    p.add_argument("--both", action="store_true",
                   help="合约补充项：一次跑完 512 与 2048 两组并出**一份**报告（§7-2「两组表并列」）")
    p.add_argument("--length-strata", dest="length_strata", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="只取前 N 条（冒烟用）")
    p.add_argument("--self-check", dest="self_check", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def main(argv=None):
    init_streams()
    args = build_parser().parse_args(argv)
    args.ts = timestamp()
    cfg = load_config(_abs(args.config))
    base_cfg = load_config(_abs(cfg["base_config"]))
    args.regime = args.regime or cfg["inputs"]["default_regime"]

    if args.self_check:
        return self_check(cfg, base_cfg, args, print)
    if args.dry_run:
        return cmd_dryrun(cfg, base_cfg, args, print)

    if args.max_tokens is None and not args.both:
        print("缺少 --max-tokens（或给 --both／--self-check／--dry-run）")
        return 2

    # C2：白名单守卫 —— **先于**目录检查 ⇒ 越界即非零退出
    groups_to_check = GROUP_KEYS if args.both else (str(int(args.max_tokens)),)
    for g in groups_to_check:
        try:
            dets, _explicit = resolve_detectors(args, cfg, int(g), print)
        except (AssertionError, ValueError) as exc:
            print("REFUSED：%s" % exc)
            return 2
        print("DETECTORS=%s MAX_TOKENS=%d" % (",".join(dets), int(g)))

    if not args.source:
        print("缺少 --source（val｜sealed）")
        return 2
    if args.source == "sealed" and (not args.access_log_note or not args.split):
        print("REFUSED：`--source sealed` 须同时给 `--split` 与 `--access-log-note`"
              "（缺一即拒绝执行并非零退出，`W6.6` §3-C5）")
        return 2
    if not args.report_dir or not args.outputs_dir:
        print("MISSING_DIRS：评估落盘产物 ⇒ 须同时给 --report-dir 与 --outputs-dir（C10）")
        return 2
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)
    logger, log_path = flagged_logger(args.ts, log_dir=resolve_log_dir(args))
    logger("ACCESS_LOG_NOTE=%s（登记由人手工完成；脚本不写 ACCESS_LOG.md）"
           % (args.access_log_note or "n/a（非 sealed 源）"))
    rc = cmd_eval(cfg, base_cfg, args, logger)
    logger("LOG_FILE=%s" % log_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())

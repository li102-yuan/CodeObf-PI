# -*- coding: utf-8 -*-
"""03_05_run_b5_llamaguard8b.py — W4.5 B5 Llama-Guard-3-8B **4-bit 原始推理**基线（阶段3，接口 v1.0）。

职责：实现 `windows/W4.5_b5_lg8b.md` 的 B5 基线——在**三个测试分片**（`id_test`／`ood_test`／
`clean_test`）上对 **原始 Llama-Guard-3-8B**（**不微调**，`4bit-nf4`）做推理，产出符合
`windows/W4.6_unify.md` §2 统一 schema 的逐样本 CSV（`raw_type=judgment`）。

接口契约（`windows/W4.6_unify.md` §2／§3，冻结 v1.0）::

    columns = [sample_id, text_sha256, pred, raw_type, raw, score, latency_ms]
    raw     = json.dumps(<判词原文>)                # §2 raw_semantics.b5
    raw_type= "judgment"
    score   = ""                                    # §2 score_source.b5 = **留空**
    pred    = 判词首行首词 == "unsafe" ? 1 : 0      # §3 mapping.b5

加载与推理规范（`windows/W4.5_b5_lg8b.md` §0.3；依 `ENV_LOCK.md` §6／§6.2／§8）::

    权重  models/hf_cache/modelscope--LLM-Research--Llama-Guard-3-8B（rev 7327bd9f…）
    加载  4bit-nf4（bitsandbytes）／cuda／attn_implementation="eager"（禁 flash）
    输入  原文直通 → 官方 tokenizer → 512 token 左截断（保尾部）
    判定  判词**首行首词**（strip 后首个空白分词）== "unsafe" = 1；unparsed 计数披露
    模式  model.eval() + torch.no_grad() 硬断言（G-EVAL；AGENTS.md §6-11）
    prompt Llama-Guard-3 **官方 chat 模板原样使用**，不加任何项目侧 prompt 工程
    显存  加载态 5542 MiB／推理 512 token 5574–5852 MiB（8 GB 卡余量紧张，OOM 风险真实）

硬约束（`windows/W4.5_b5_lg8b.md` §0.7）：

1. 只读 `text` 列（`DATA_CARD` §3）；`text_sha256` 输入一致性；
2. 读 `sealed_test` 前须由窗口在 `ACCESS_LOG.md` 登记 6 列；本脚本只打印提醒，不代登记；
3. 无阈值（判词二值）；`score` 留空；AUROC 记 `n/a`；
4. 冒烟抽 1 条样本**双跑**比对判词逐字一致（全量 G-DET 已由 W2.3 冻结，不重做）；
5. 显存记录：加载态与推理态峰值（`max_memory_allocated`）；**每次 OOM 触发回退须登记**；
6. 未实现处 `NOT_IMPLEMENTED` + `sys.exit(0)`；顶层不 import 重型库。

本窗口特有的呈报义务（`windows/W4.5_b5_lg8b.md` §0.4–§0.6）：

- **§0.4 口径对齐（`D-34` 转交项 40）**：报告须**二选一并显式声明**——本脚本默认 **(a) 沿用
  门禁 4 口径**（首行首词／左截断 512），并逐项列出五要素一致；若因 OOM 回退改口径，须切 (b) 改判。
- **§0.5 两种结果叙事预案**：报告预先写定「叙事① 主档全量」与「叙事② OOM 回退档」。
- **§0.6 ASCII 艺术降级叙事预案**：逐编码拆分子集指标，`ascii_art` 子集单独列出。

CLI::

    python scripts/baseline/03_05_run_b5_llamaguard8b.py --self-check
    python scripts/baseline/03_05_run_b5_llamaguard8b.py --config configs/baseline/b5_llamaguard8b.yaml \
        --unify-config configs/baseline/unify.yaml --splits id_test ood_test clean_test \
        --smoke-double-run
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
RAW_TYPE = "judgment"
OUTPUT_COLUMNS = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score", "latency_ms"]

# §3 mapping.b5：首行首词口径（冻结）
UNSAFE_WORD = "unsafe"
SAFE_WORD = "safe"

# §0.4 口径对齐（`D-34` 转交项 40）——五要素（与 `DATA_CARD` §25 一致）
GATE4_ELEMENTS = [
    "检测器 `llamaguard3-8b` rev `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425`",
    "加载方式 `4-bit`（`4bit-nf4`）",
    "判词解析 = **首行首词**（strip 后首个空白分词）",
    "截断侧 = **左**（512 token 上限）",
    "计数口径 = **只按内容**（`D-00-05`：`MI-M` 计入，`MI-W`/`MI-B` 不计入；G3 先还原再送检、G1/G2 原样送检）",
]


def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（先例：W3.7 `F-1`）。只动输出流，不触碰判定值。

    **不静默吞错**：重配失败时显式登记到 `sys.stderr`（`ascii()` 保证 ASCII 安全），
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
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 载入上游 `03_06`（统一预处理，唯一出处 —— 禁止本窗口自行实现读取／清洗／截断）
# ---------------------------------------------------------------------------
_PREPROCESS = _load_module("baseline_preprocess_b5",
                           os.path.join(_HERE, "03_06_unify_preprocess.py"))

load_samples = _PREPROCESS.load_samples
load_rows = _PREPROCESS.load_rows
sha256_of = _PREPROCESS.sha256_of
md5_of = _PREPROCESS.md5_of
write_csv = _PREPROCESS.write_csv
read_csv = _PREPROCESS.read_csv
truncate_left = _PREPROCESS.truncate_left


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
# §3 判词解析：首行首词（冻结口径，不得自行改规则）
# ---------------------------------------------------------------------------
def parse_judgment(judgment):
    """§3 mapping.b5：判词 → ``(pred, first_word, parsed)``。

    步骤（`configs/baseline/b5_llamaguard8b.yaml` 的 ``mapping.parse_steps``）：

    1. **strip 判词整体**；
    2. 取**首行**；
    3. 该行 strip 后取**首个空白分词**。

    - 首词 == ``unsafe`` ⇒ ``("1", "unsafe", True)``
    - 首词 == ``safe``   ⇒ ``("0", "safe", True)``
    - 其余（含空判词）   ⇒ ``("", <首词或空串>, False)`` —— 记 `unparsed`，**不兜底为 0/1**

    构造判词用例（§2 验收）：``"safe"`` ⇒ 0；``"unsafe\\nS9"`` ⇒ 1。
    """
    if judgment is None:
        return "", "", False
    stripped = str(judgment).strip()
    if not stripped:
        return "", "", False
    first_line = stripped.splitlines()[0].strip()
    if not first_line:
        return "", "", False
    first_word = first_line.split()[0]
    if first_word == UNSAFE_WORD:
        return "1", first_word, True
    if first_word == SAFE_WORD:
        return "0", first_word, True
    return "", first_word, False


# ---------------------------------------------------------------------------
# 统计（§4.2 延迟：均值／中位／p95）
# ---------------------------------------------------------------------------
def _quantile(xs, q):
    """最近秩分位数（确定性，不插值）。``xs`` 须已升序。"""
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def _median(xs):
    """中位数：奇数取中位、偶数取中间两值均值（与 `03_01`／`03_02` 同口径）。``xs`` 须已升序。"""
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def latency_stats(values):
    """§4.2 延迟：均值／中位／p95（毫秒）。"""
    if not values:
        return {"mean": None, "median": None, "p95": None}
    xs = sorted(float(v) for v in values)
    n = len(xs)
    return {"mean": sum(xs) / n, "median": _median(xs), "p95": _quantile(xs, 0.95)}


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return ("%%.%df" % nd) % x
    return str(x)


def _mib(n_bytes):
    if n_bytes is None:
        return "n/a"
    return "%.1f" % (float(n_bytes) / (1024.0 * 1024.0))


# ---------------------------------------------------------------------------
# 模型加载（重型库在函数体内延迟 import）
# ---------------------------------------------------------------------------
def build_quant_config(transformers, torch, quant_cfg):
    """`ENV_LOCK.md` §6：`4bit-nf4` 量化配置。"""
    dtype_name = _require(quant_cfg, "bnb_4bit_compute_dtype", "load.quantization")
    dtype = getattr(torch, dtype_name)
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=_require(quant_cfg, "bnb_4bit_quant_type", "load.quantization"),
        bnb_4bit_use_double_quant=_require(quant_cfg, "bnb_4bit_use_double_quant",
                                           "load.quantization"),
        bnb_4bit_compute_dtype=dtype,
    )


def load_model(cfg):
    """按 §0.3 加载 tokenizer + 4bit-nf4 模型。返回 ``(torch, transformers, tok, model, meta)``。

    ``meta`` 含加载耗时与**加载态显存峰值**（§0.7-5）。
    """
    import torch                                     # 延迟 import（顶层不 import 重型库）
    import transformers

    w = _require(cfg, "weights", "cfg")
    load_cfg = _require(cfg, "load", "cfg")

    path = _require(w, "local_path", "weights")
    if not os.path.isdir(path):
        raise FileNotFoundError("权重目录不存在：%s" % path)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tok = transformers.AutoTokenizer.from_pretrained(
        path, local_files_only=bool(_require(w, "local_files_only", "weights")))

    dtype = getattr(torch, _require(load_cfg, "dtype", "load"))
    attn = _require(load_cfg, "attn_implementation", "load")
    forbidden = list((load_cfg.get("forbidden_attn") or []))
    if attn in forbidden:
        raise ValueError("禁止的 attn_implementation：%r（全阶段固定 eager）" % (attn,))

    kwargs = {
        "local_files_only": True,
        "attn_implementation": attn,
        "dtype": dtype,
        "quantization_config": build_quant_config(
            transformers, torch, _require(load_cfg, "quantization", "load")),
    }
    if _require(load_cfg, "device", "load") == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 device=cuda，但 torch.cuda.is_available() 为 False")
        kwargs["device_map"] = dict(_require(load_cfg, "device_map", "load"))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = transformers.AutoModelForCausalLM.from_pretrained(path, **kwargs)
    load_ms = (time.perf_counter() - t0) * 1000.0
    load_peak = int(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    model.eval()                                      # G-EVAL：`AGENTS.md` §6-11
    return torch, transformers, tok, model, {
        "load_ms": load_ms,
        "load_peak_bytes": load_peak,
        "attn": attn,
        "dtype": str(dtype),
        "path": path,
        "hf_revision": _require(w, "hf_revision", "weights"),
        "device_map": str(kwargs.get("device_map")),
    }


# ---------------------------------------------------------------------------
# §0.3 输入构造：原文直通 → 左截断 512 → **官方 chat 模板**
# ---------------------------------------------------------------------------
def build_messages(cfg, content_text):
    """按 **Llama-Guard-3-8B 官方消息格式**组装 messages（**非**项目侧 prompt 工程）。

    8B 官方 `chat_template` 渲染会话段时对 `message['content']` **直接调用 `.strip()`**::

        {% set content = message['content'] %}
        {{ role + ': ' + content.strip() + '\\n\\n' }}

    即 `content` 必须是**普通字符串**。这与 1B **不同**（1B 用
    `selectattr('type','equalto','text')` 按内容块列表渲染，见 `03_03`）。若误传列表，
    Jinja 会抛 ``'list object' has no attribute 'strip'`` —— 属「响铃」而非静默退化。
    """
    p_cfg = _require(cfg, "prompt", "cfg")
    fmt = _require(p_cfg, "content_format", "prompt")
    if fmt != "plain_string":
        raise ValueError("未知的 prompt.content_format：%r（8B 官方格式为 plain_string）"
                         % (fmt,))
    return [{"role": _require(p_cfg, "role", "prompt"),
             "content": content_text}]


def build_prompt(tok, cfg, text):
    """返回 ``(prompt, truncated_text, n_text_tokens, was_truncated)``。

    - **左截断**：`03_06.truncate_left(text, tok, 512)`（保尾部 512 token）；
    - **官方 chat 模板**：``tok.apply_chat_template(messages, tokenize=False,
      add_generation_prompt=True)`` —— **不加任何项目侧 prompt 工程**；
    - **空会话段守卫**：8B 模板对 `content` 直接 `.strip()`，内容不会被静默丢弃；唯一退化
      形态是「截断后内容为空（全空白）」——此时官方模板渲染出的会话段无实质内容，
      判词退化为常数，须**抛错**而非静默产出。
    """
    in_cfg = _require(cfg, "input", "cfg")
    max_tokens = int(_require(in_cfg, "max_seq_tokens", "input"))
    truncated = truncate_left(text, tok, max_tokens)
    n_tokens = len(tok(truncated, add_special_tokens=False)["input_ids"])
    p_cfg = _require(cfg, "prompt", "cfg")
    prompt = tok.apply_chat_template(
        build_messages(cfg, truncated),
        tokenize=False,
        add_generation_prompt=bool(_require(p_cfg, "add_generation_prompt", "prompt")),
    )
    if truncated and not truncated.strip():
        raise RuntimeError(
            "截断后样本内容为空（全空白）—— 官方模板渲染出的会话段无实质内容，判词将退化为常数。")
    return prompt, truncated, n_tokens, (truncated != text)


def generate_judgment(torch, tok, model, cfg, prompt):
    """贪心生成判词。返回 ``(judgment, n_prompt_tokens)``。"""
    in_cfg = _require(cfg, "input", "cfg")
    g_cfg = _require(cfg, "generation", "cfg")
    batch = tok(prompt, return_tensors="pt",
                add_special_tokens=bool(_require(in_cfg, "add_special_tokens", "input")))
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    n_prompt = int(batch["input_ids"].shape[1])

    # G-EVAL 硬断言（`AGENTS.md` §6-11）
    if model.training:
        raise RuntimeError("G-EVAL 断言失败：模型处于 train 模式（必须 model.eval()）")
    if torch.is_grad_enabled():
        raise RuntimeError("G-EVAL 断言失败：grad 已启用（必须 torch.no_grad()）")

    pad_id = tok.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer.eos_token_id 为 None，无法确定 pad_token_id（禁止默认值兜底）")

    out = model.generate(
        **batch,
        do_sample=bool(_require(g_cfg, "do_sample", "generation")),
        max_new_tokens=int(_require(g_cfg, "max_new_tokens", "generation")),
        num_beams=int(_require(g_cfg, "num_beams", "generation")),
        pad_token_id=pad_id,
    )
    gen_ids = out[0][n_prompt:]
    judgment = tok.decode(gen_ids,
                          skip_special_tokens=bool(_require(g_cfg, "decode_skip_special_tokens",
                                                            "generation")))
    return judgment, n_prompt


# ---------------------------------------------------------------------------
# 逐分片评估
# ---------------------------------------------------------------------------
def run_split(split, csv_path, sealed, torch, transformers, tok, model, cfg, progress_every=0):
    """在一个分片上跑 B5，返回 ``(rows, stats)``。

    - 检测器输入**只读 `text` 列**（`load_samples`）；
    - `encoding_type` 仅用于 §0.6 的**逐编码子集拆分开报告**（元数据，非检测器输入、非调参）。
    """
    samples = load_samples(csv_path)
    in_cfg = _require(cfg, "input", "cfg")
    max_tokens = int(_require(in_cfg, "max_seq_tokens", "input"))

    # §0.6：逐编码子集拆分所需的元数据（只读，不改检测器输入，不用于调参）
    encoding_by_id = {}
    for row in load_rows(csv_path):
        encoding_by_id[row.get("sample_id", "")] = row.get("encoding_type", "")

    torch.cuda.reset_peak_memory_stats()
    rows = []
    latencies = []
    n_truncated = 0
    n_unparsed = 0
    n_unsafe = 0
    n_safe = 0
    unparsed_examples = []
    prompt_tokens = []

    # per-encoding 统计（§0.6）：encoding -> {"n", "unsafe", "safe", "unparsed"}
    enc_stats = {}

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        prompt, _truncated, n_tok, was_truncated = build_prompt(tok, cfg, sample.text)
        judgment, n_prompt = generate_judgment(torch, tok, model, cfg, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred, first_word, parsed = parse_judgment(judgment)
        latencies.append(elapsed_ms)
        prompt_tokens.append(n_prompt)
        if was_truncated:
            n_truncated += 1

        if not parsed:
            n_unparsed += 1
            if len(unparsed_examples) < 5:
                unparsed_examples.append({"sample_id": sample.sample_id,
                                          "judgment": judgment})
        else:
            if pred == "1":
                n_unsafe += 1
            else:
                n_safe += 1

        # §0.6：逐编码子集拆分**只统计攻击样本（label=1）** —— TPR 分母为攻击样本；
        # 良性样本（含 G3「编码但良性」）不计入，避免把良性编码串混进攻击 TPR。
        if sample.label == 1:
            enc = encoding_by_id.get(sample.sample_id, "")
            e = enc_stats.setdefault(enc, {"n": 0, "unsafe": 0, "safe": 0, "unparsed": 0})
            e["n"] += 1
            if not parsed:
                e["unparsed"] += 1
            elif pred == "1":
                e["unsafe"] += 1
            else:
                e["safe"] += 1

        rows.append({
            "sample_id": sample.sample_id,
            "text_sha256": sha256_of(sample.text),
            "pred": pred,
            "raw_type": RAW_TYPE,
            "raw": json.dumps(judgment, ensure_ascii=False),
            "score": "",                             # §2：B5 的 score **留空**
            "latency_ms": "%.6f" % elapsed_ms,
        })
        if progress_every and (i + 1) % progress_every == 0:
            print("  [%s] %d/%d" % (split, i + 1, len(samples)))

    split_peak = int(torch.cuda.max_memory_allocated())
    stats = {
        "split": split,
        "path": csv_path,
        "sealed": bool(sealed),
        "n": len(rows),
        "md5": md5_of(csv_path),
        "latency": latency_stats(latencies),
        "n_truncated": n_truncated,
        "max_tokens": max_tokens,
        "n_unparsed": n_unparsed,
        "n_unsafe": n_unsafe,
        "n_safe": n_safe,
        "unparsed_examples": unparsed_examples,
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else 0,
        "prompt_tokens_mean": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0.0,
        "split_peak_bytes": split_peak,
        "enc_stats": enc_stats,                     # §0.6 逐编码子集拆分
    }
    return rows, stats


# ---------------------------------------------------------------------------
# §0.7-4 冒烟：抽 1 条样本双跑，判词**逐字一致**
# ---------------------------------------------------------------------------
def double_run_smoke(split, csv_path, torch, tok, model, cfg):
    """取该分片**第 1 条**样本连续推理两次，比对判词原文**逐字一致**。

    返回证据字典（含两次判词原文与其 sha256）。全量 G-DET 已由 W2.3 冻结，本处只作**抽检**。
    """
    import hashlib

    samples = load_samples(csv_path)
    if not samples:
        raise RuntimeError("冒烟双跑失败：分片 %s 无样本" % split)
    sample = samples[0]
    prompt, _t, n_tok, was_truncated = build_prompt(tok, cfg, sample.text)

    j1, n_prompt = generate_judgment(torch, tok, model, cfg, prompt)
    j2, _ = generate_judgment(torch, tok, model, cfg, prompt)

    return {
        "split": split,
        "sample_id": sample.sample_id,
        "text_sha256": sha256_of(sample.text),
        "n_text_tokens": n_tok,
        "n_prompt_tokens": n_prompt,
        "truncated": bool(was_truncated),
        "judgment_run1": j1,
        "judgment_run2": j2,
        "sha256_run1": hashlib.sha256(j1.encode("utf-8")).hexdigest(),
        "sha256_run2": hashlib.sha256(j2.encode("utf-8")).hexdigest(),
        "verbatim_identical": j1 == j2,
        "pred_run1": parse_judgment(j1)[0],
        "pred_run2": parse_judgment(j2)[0],
    }


# ---------------------------------------------------------------------------
# 运行报告（§1）
# ---------------------------------------------------------------------------
def write_report(path, cfg, unify_cfg, per_split, ts, smoke=None, memory=None, load_meta=None,
                 fallback_events=None):
    """生成 `reports/baseline/b5_lg8b_v1.0_<ts>.md`。

    含：§0.4 口径对齐二选一声明、§0.5 两叙事预案、§0.6 ASCII 降级预案、
    逐编码子集指标、延迟/显存、`unparsed`、确定性抽检、OOM 回退登记。
    """
    w = _require(cfg, "weights", "cfg")
    load_cfg = _require(cfg, "load", "cfg")
    in_cfg = _require(cfg, "input", "cfg")
    g_cfg = _require(cfg, "generation", "cfg")
    p_cfg = _require(cfg, "prompt", "cfg")
    m_cfg = _require(cfg, "mapping", "cfg")
    quant = _require(load_cfg, "quantization", "load")

    lines = []
    lines.append("# b5_lg8b_v1.0 运行报告（W4.5 · B5 Llama-Guard-3-8B **4-bit 原始推理**）")
    lines.append("")
    lines.append("- 生成时间：`%s`" % ts)
    lines.append("- 窗口：W4.5（权限 A4）／指令：`windows/W4.5_b5_lg8b.md`")
    lines.append("- 接口：`windows/W4.6_unify.md` §1.3（512／左截断）／§2（schema）／§3（判词映射）")
    lines.append("- 上游：`scripts/baseline/03_06_unify_preprocess.py` 的 `load_samples()` 与 "
                 "`truncate_left()`（原文直通 + 左截断）")
    lines.append("- 参数载体：`configs/baseline/b5_llamaguard8b.yaml`（`version: %s`）"
                 % cfg.get("version"))
    lines.append("")
    lines.append("## 0. 本次实际使用的模型 id")
    lines.append("")
    lines.append("> **model id = `%s`**（权重本地路径 `%s`，HF revision `%s`，"
                 "来源渠道 %s `%s`，依 `DECISIONS.md` `D-07`）。"
                 "**未微调、未改权重、未换 checkpoint。**"
                 % (_require(w, "repo_id", "weights"), _require(w, "local_path", "weights"),
                    _require(w, "hf_revision", "weights"),
                    _require(w, "source_channel", "weights"),
                    _require(w, "source_repo_id", "weights")))
    lines.append("")

    # §1 加载参数
    lines.append("## 1. 加载参数与 provenance（§0.3，依 `ENV_LOCK.md` §5／§6.2／§8）")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 检测器 | B5 · %s |" % cfg.get("detector_name"))
    lines.append("| 权重路径 | `%s` |" % _require(w, "local_path", "weights"))
    lines.append("| HF revision | `%s` |" % _require(w, "hf_revision", "weights"))
    lines.append("| 来源渠道 | %s `%s`（排除 `%s`） |"
                 % (_require(w, "source_channel", "weights"),
                    _require(w, "source_repo_id", "weights"),
                    "`、`".join(_require(w, "exclude_prefixes", "weights"))))
    lines.append("| 本地只读 | `%s`（禁止触网） |" % _require(w, "local_files_only", "weights"))
    lines.append("| 加载方式 | `%s`（bitsandbytes NF4） |" % _require(load_cfg, "mode", "load"))
    lines.append("| 量化参数 | `quant_type=%s`／`double_quant=%s`／`compute_dtype=%s` |"
                 % (quant["bnb_4bit_quant_type"], quant["bnb_4bit_use_double_quant"],
                    quant["bnb_4bit_compute_dtype"]))
    lines.append("| 设备 / device_map | `%s` / `%s` |"
                 % (_require(load_cfg, "device", "load"), _require(load_cfg, "device_map", "load")))
    lines.append("| `attn_implementation` | **`%s`**（全阶段固定；禁 `%s`） |"
                 % (_require(load_cfg, "attn_implementation", "load"),
                    "`、`".join(load_cfg.get("forbidden_attn") or [])))
    lines.append("| 实测 dtype / attn | `%s` / `%s` |"
                 % ((load_meta or {}).get("dtype", "n/a"), (load_meta or {}).get("attn", "n/a")))
    lines.append("| 是否微调 | **否**（`finetuned=%s`；微调后的 LG-8B 是 Ours，归 W5） |"
                 % _require(w, "finetuned", "weights"))
    lines.append("| eval + no_grad | **硬断言**（G-EVAL；`AGENTS.md` §6-11） |")
    lines.append("| 加载耗时 | %s ms（**不进 `latency_ms`**，§2 注） |"
                 % _fmt((load_meta or {}).get("load_ms"), 1))
    lines.append("| OOM 风险 | **真实**（`ENV_LOCK.md` §6.2／§8：8 GB 卡余量紧张） |")
    lines.append("")

    # §2 输入与 prompt 口径
    lines.append("## 2. 输入与 prompt 口径（§0.3／§1.3）")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 输入列 | `%s`（`DATA_CARD` §3：阶段3–5 的唯一输入字段） |"
                 % _require(in_cfg, "column", "input"))
    lines.append("| 预处理 | **原文直通**（`pass_through=%s`：不清洗／不归一／不改大小写） |"
                 % _require(in_cfg, "pass_through", "input"))
    lines.append("| 截断 | `%s` 侧，上限 **%s token**（保尾部）；实测触发截断数见 §4 |"
                 % (_require(in_cfg, "truncation_side", "input"),
                    _require(in_cfg, "max_seq_tokens", "input")))
    lines.append("| 截断施加对象 | `%s`（样本 `text` 本体） |"
                 % _require(in_cfg, "truncation_applied_to", "input"))
    lines.append("| chat 模板 | **官方 `tokenizer_config.json` 的 `chat_template`**（%s） |"
                 % _require(p_cfg, "template_source", "prompt"))
    lines.append("| 项目侧 prompt 工程 | **无**（`project_side_prompt_engineering=%s`；"
                 "B5 = 原始推理基线） |" % _require(p_cfg, "project_side_prompt_engineering",
                                                  "prompt"))
    lines.append("| 生成 | `do_sample=%s`／`max_new_tokens=%s`／`num_beams=%s`／`batch_size=%s`（贪心，确定性） |"
                 % (_require(g_cfg, "do_sample", "generation"),
                    _require(g_cfg, "max_new_tokens", "generation"),
                    _require(g_cfg, "num_beams", "generation"),
                    _require(g_cfg, "batch_size", "generation")))
    lines.append("")
    lines.append("> **截断口径声明（显式披露）**：`512 token` 上限施加于**样本 `text` 本体**"
                 "（原文直通 → 官方 tokenizer → 左截断保尾部 512），**不计入官方 chat 模板的任务"
                 "前缀开销**。理由：Llama-Guard-3 的判定任务指令与 S1–S13 类别表位于模板前缀中，"
                 "若把 512 上限施加于**拼接后的整段 prompt**，左截断会**剪掉任务前缀**，"
                 "判词将失去意义。§1.3 的「最大序列长度 512 token／截断侧 = 左」与 "
                 "`DATA_CARD` §25「门禁 4：截断侧 = 左（上限 512 token）」按此口径对齐。")
    lines.append("")

    # §3 判词映射
    lines.append("## 3. 判词 → 二分类映射（§3，冻结口径）")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 规则 | `%s` |" % _require(m_cfg, "rule", "mapping"))
    lines.append("| 解析步骤 | %s |" % " → ".join(m_cfg.get("parse_steps") or []))
    lines.append("| `unsafe` 词 | `%s` ⇒ `pred=1` |" % _require(m_cfg, "unsafe_word", "mapping"))
    lines.append("| `safe` 词 | `%s` ⇒ `pred=0` |" % _require(m_cfg, "safe_word", "mapping"))
    lines.append("| `unparsed` | 首行首词 ∉ {safe, unsafe} ⇒ **`pred` 留空**（不兜底为 0/1），"
                 "计数披露（见 §4） |")
    lines.append("| 阈值 | 无（判词二值）；`score` **留空**；AUROC 记 **`n/a`**（§4.2） |")
    lines.append("")

    # §3.5 口径对齐（§0.4，D-34 转交项 40）
    lines.append("## 3.5 口径对齐声明（§0.4 · `D-34` 转交项 40 —— **本窗口核心呈报义务**）")
    lines.append("")
    lines.append("**裁定：B5 沿用门禁 4 同一口径（(a) 沿用，非 (b) 改判）。** 即「与门禁 4 同口径」，"
                 "五要素逐项一致如下：")
    lines.append("")
    for i, elem in enumerate(GATE4_ELEMENTS, 1):
        lines.append("%d. %s" % (i, elem))
    lines.append("")
    lines.append("> 依据：`DATA_CARD` §25 = `windows/W3.8_freeze.md` §3.6 第 25 条（门禁 4 实测口径，"
                 "已随 `data-v1.0` 冻结）。**本条不改门禁 4 口径**（由 `W3.5`／`W3.8` 冻结），"
                 "只显式声明 B5 的对齐关系。因五要素一致 ⇒ 「门禁 4 的检出率」与「B5 的指标」"
                 "**可直接对齐**。")
    lines.append("")
    lines.append("> **改判触发条件（本窗口预留）**：若因工程原因（OOM 回退改 256 token／CPU offload／"
                 "子集评估）无法沿用，本声明须切换为 **(b) 改判**，逐项写明差异与理由，并声明"
                 "「两数字不可直接比较」的影响。本次运行是否触发见 §8。")
    lines.append("")

    # §4 逐分片结果
    lines.append("## 4. 逐分片结果（三个测试分片）")
    lines.append("")
    lines.append("| 分片 | 用途 | n | sealed | 分片 md5 | `unsafe` | `safe` | "
                 "**`unparsed`** | 触发截断 | 均值 ms | 中位 ms | p95 ms | 推理态峰值 MiB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    role = {"id_test": "体制一测试", "ood_test": "体制二测试",
            "clean_test": "灾难性遗忘基准（全 label=1）"}
    for split in per_split:
        st = per_split[split]
        lat = st["latency"]
        lines.append("| `%s` | %s | %d | %s | `%s` | %d | %d | **%d** | %d | %s | %s | %s | %s |" % (
            split, role.get(split, "—"), st["n"], "是" if st["sealed"] else "否", st["md5"],
            st["n_unsafe"], st["n_safe"], st["n_unparsed"], st["n_truncated"],
            _fmt(lat["mean"], 3), _fmt(lat["median"], 3), _fmt(lat["p95"], 3),
            _mib(st["split_peak_bytes"])))
    lines.append("")
    lines.append("> `latency_ms` = 单样本**左截断 + 模板化 + 生成 + 解码**耗时（不含模型加载、"
                 "不含 CSV 读取与落盘；§2 注：模型加载耗时单独披露，不进 `latency_ms`）。")
    lines.append("")
    total_unparsed = sum(st["n_unparsed"] for st in per_split.values())
    if total_unparsed:
        lines.append("**`unparsed` 明细（如实披露，§3 回退点）**：合计 **%d** 条。样例判词原文："
                     % total_unparsed)
        lines.append("")
        lines.append("| 分片 | sample_id | 判词原文（JSON） |")
        lines.append("|---|---|---|")
        for split in per_split:
            for ex in per_split[split]["unparsed_examples"]:
                lines.append("| `%s` | `%s` | `%s` |"
                             % (split, ex["sample_id"],
                                json.dumps(ex["judgment"], ensure_ascii=False).replace("|", "\\|")))
        lines.append("")
        lines.append("> 处置：**不得**自行改解析规则（首行首词口径冻结）；只登记计数与样例原文，呈 W4。")
        lines.append("")
    else:
        lines.append("**`unparsed` 计数 = 0**（三个分片全部判词的首行首词均 ∈ {safe, unsafe}）。")
        lines.append("")

    # §4.1 逐编码子集拆分（§0.6）
    lines.append("## 4.1 逐编码子集拆分（§0.6 · ASCII 降级预案配套）")
    lines.append("")
    lines.append("攻击样本（`label=1`）按 `encoding_type` 的 `pred=1` 占比（TPR）如下，"
                 "**`ascii_art` 子集单独列出、不混入整体均值**：")
    lines.append("")
    lines.append("| 分片 | 编码 | 攻击样本 n | `unsafe` | `safe` | `unparsed` | TPR |")
    lines.append("|---|---|---|---|---|---|---|")
    for split in per_split:
        st = per_split[split]
        encs = st.get("enc_stats") or {}
        for enc in sorted(encs, key=lambda k: (k != "ascii_art", k)):
            e = encs[enc]
            denom = e["unsafe"] + e["safe"] + e["unparsed"]
            if denom == 0:
                continue
            tpr = ("%.4f" % (e["unsafe"] / denom)) if denom else "n/a"
            lines.append("| `%s` | `%s` | %d | %d | %d | %d | %s |" % (
                split, enc or "（空）", e["n"], e["unsafe"], e["safe"], e["unparsed"], tpr))
    lines.append("")
    lines.append("> **`ascii_art` 降级叙事（§0.6，预案）**：`ascii_art` 为多行字符画，经 tokenization 后"
                 "特征受损，可能系统性压低各检测器 TPR。**若上表 `ascii_art` 子集 TPR 异常低**，"
                 "须按「编码本身可检测性边界」呈现，**不得**据此单独判定检测器失效，**也不得**"
                 "把该子集指标混入整体均值掩盖。")
    lines.append("")

    # §5 显存
    lines.append("## 5. 显存峰值（§0.7-5／§4.2：`max_memory_allocated`）")
    lines.append("")
    lines.append("| 状态 | 峰值 MiB | 峰值字节 |")
    lines.append("|---|---|---|")
    mem = memory or {}
    lines.append("| 加载态 | %s | `%s` |" % (_mib(mem.get("load_peak_bytes")),
                                            mem.get("load_peak_bytes")))
    lines.append("| 推理态（全量） | %s | `%s` |" % (_mib(mem.get("infer_peak_bytes")),
                                                    mem.get("infer_peak_bytes")))
    for split in per_split:
        lines.append("| 推理态 · `%s` | %s | `%s` |" % (
            split, _mib(per_split[split]["split_peak_bytes"]),
            per_split[split]["split_peak_bytes"]))
    lines.append("")
    lines.append("> 对照 `ENV_LOCK.md` §6.2／§8（W2.3／W2.4 实测）：`llamaguard3-8b` 加载态 "
                 "**5542 MiB**／推理 `infer-8b-512` **5574 MiB**（44-token 探针）／"
                 "追加 512-token 输入 **5852 MiB** —— 8 GB 卡余量紧张，本次实测见上表。")
    lines.append("")

    # §6 确定性抽检
    lines.append("## 6. 确定性抽检证据（§0.7-4：1 条样本双跑、判词逐字一致）")
    lines.append("")
    if smoke:
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        lines.append("| 分片 | `%s`（第 1 条样本） |" % smoke["split"])
        lines.append("| `sample_id` | `%s` |" % smoke["sample_id"])
        lines.append("| `text_sha256` | `%s` |" % smoke["text_sha256"])
        lines.append("| 截断后 `text` token 数 / 是否截断 | %d / `%s` |"
                     % (smoke["n_text_tokens"], smoke["truncated"]))
        lines.append("| 送入模型的 prompt token 数 | %d |" % smoke["n_prompt_tokens"])
        lines.append("| 第 1 次判词（原文） | `%s` |"
                     % json.dumps(smoke["judgment_run1"], ensure_ascii=False).replace("|", "\\|"))
        lines.append("| 第 2 次判词（原文） | `%s` |"
                     % json.dumps(smoke["judgment_run2"], ensure_ascii=False).replace("|", "\\|"))
        lines.append("| 判词 sha256（第 1 次） | `%s` |" % smoke["sha256_run1"])
        lines.append("| 判词 sha256（第 2 次） | `%s` |" % smoke["sha256_run2"])
        lines.append("| **逐字一致** | **%s** |" % ("是" if smoke["verbatim_identical"] else "**否**"))
        lines.append("| 两次 `pred` | `%s` / `%s` |" % (smoke["pred_run1"], smoke["pred_run2"]))
        lines.append("")
        lines.append("> 全量 G-DET 已由 **W2.3 冻结**（`llamaguard3-8b` 双进程 logits md5 "
                     "`191334ae4bed6779145b05e82bb93891`，`max|A−B| = 0.0`），本窗口**不重做**，"
                     "只留上述抽检证据。")
        lines.append("")
    else:
        lines.append("- 本次运行未执行双跑抽检（未传 `--smoke-double-run`）。")
        lines.append("")

    # §7 clean 对照声明
    lines.append("## 7. `clean_test` 对照声明（§0.4-5，必写）")
    lines.append("")
    if "clean_test" in per_split:
        st = per_split["clean_test"]
        recall = (float(st["n_unsafe"]) / float(st["n"])) if st["n"] else 0.0
        lines.append("- `clean_test` 共 **%d** 行，**全 `label=1`**（明文恶意，无编码、无代码载体）；"
                     "**无负样本** ⇒ **FPR 无定义**、**F1 为退化形式** `2TP/(2TP+FN)`（§4.2）。"
                     % st["n"])
        lines.append("- 实测：判 `unsafe` **%d** 条、判 `safe` **%d** 条、`unparsed` **%d** 条 ⇒ "
                     "**明文召回率 = %.4f**。"
                     % (st["n_unsafe"], st["n_safe"], st["n_unparsed"], recall))
        lines.append("- **这是 B5 的强项预期，不是缺陷**：明文恶意对原始 Llama-Guard-3-8B **可见**，"
                     "其安全对齐能力本就覆盖这类明文有害请求。")
        lines.append("- **与编码分片的弱势形成对照**（`id_test`／`ood_test` 的 TPR 见 "
                     "`03_07` 汇总表）—— 这一「明文高、编码低」的落差**正是 Ours 要填补的缺口叙事**："
                     "微调的目标是在不显著损失明文召回的前提下抬高编码侧检出。")
        lines.append("- 本窗口**不**用 `clean_test` 做任何调参／选阈值（B5 无阈值；§0.4-4）。")
    else:
        lines.append("- 本次未跑 `clean_test`。")
    lines.append("")

    # §8 两种结果叙事预案（§0.5）
    lines.append("## 8. 两种结果叙事预案（§0.5，v5 阶段3；W4_stage3_baseline §3.1）")
    lines.append("")
    lines.append("| 叙事 | 口径 | 本次是否启用 |")
    lines.append("|---|---|---|")
    events = fallback_events or []
    oom_triggered = bool(events)
    lines.append("| **叙事① 主档全量** | `4-bit`／`512 token`／`batch 1` 全量三测试集 | %s |"
                 % ("否（已触发回退）" if oom_triggered else "**是（本次主口径）**"))
    lines.append("| **叙事② OOM 回退档** | `PROJECT_CONTRACT.md` §4（256 token／batch 1／CPU offload／子集评估） | %s |"
                 % ("**是（已触发）**" if oom_triggered else "否（未触发）"))
    lines.append("")
    if oom_triggered:
        lines.append("**⚠ 本次触发了叙事②（OOM 回退档）**，结果须**显式标注「非全量口径」**、"
                     "列出与主档的逐项差异、声明**不得与叙事①及其他检测器直接比较**。逐级回退登记：")
        lines.append("")
        for ev in events:
            lines.append("- %s" % ev)
        lines.append("")
    else:
        lines.append("> **叙事② 的触发条件与义务（预案，本次未触发）**：一旦触发 `PROJECT_CONTRACT.md` §4 "
                     "回退（256 token／batch 1／CPU offload／子集评估），结果须显式标注「非全量口径」、"
                     "列出与主档的逐项差异、声明不得与叙事①及其他检测器直接比较。")
        lines.append("")

    # §9 产物清单
    lines.append("## 9. 产物清单（逐样本 CSV ×3）")
    lines.append("")
    lines.append("| 分片 | 逐样本 CSV |")
    lines.append("|---|---|")
    for split in per_split:
        lines.append("| `%s` | `reports/baseline/outputs/b5_%s_v1.0_%s.csv` |" % (split, split, ts))
    lines.append("")
    lines.append("> schema 与 `windows/W4.6_unify.md` §2 逐列一致：%s；UTF-8 + LF；"
                 "`raw_type=judgment`；`score` 留空。"
                 % "、".join("`%s`" % c for c in OUTPUT_COLUMNS))
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _self_check() -> int:
    """合成判词往返验证 §3 解析口径（不读 sealed_test、不加载模型、不 import 重型库）。"""
    errors = []

    # ① §2 验收的构造判词：`safe` ⇒ 0
    if parse_judgment("safe") != ("0", "safe", True):
        errors.append("构造判词 'safe' 应解析为 pred=0：实得 %r" % (parse_judgment("safe"),))

    # ② 构造判词：`unsafe\nS9` ⇒ 1
    if parse_judgment("unsafe\nS9") != ("1", "unsafe", True):
        errors.append("构造判词 'unsafe\\nS9' 应解析为 pred=1：实得 %r"
                      % (parse_judgment("unsafe\nS9"),))

    # ③ 官方实际形态（前导空行）：`\n\nunsafe\nS1` ⇒ 1
    if parse_judgment("\n\nunsafe\nS1") != ("1", "unsafe", True):
        errors.append("前导空行判词 '\\n\\nunsafe\\nS1' 应解析为 pred=1：实得 %r"
                      % (parse_judgment("\n\nunsafe\nS1"),))

    # ④ 大小写敏感（`unsafe` 以外的大小写不改写规则）
    if parse_judgment("Unsafe\nS1")[2] is not False:
        errors.append("'Unsafe'（大写 U）应记 unparsed，不得改写规则容忍大小写")

    # ⑤ 空判词 / 纯空白 ⇒ unparsed，**不兜底**
    for blank in ("", "   ", "\n\n"):
        pred, _word, parsed = parse_judgment(blank)
        if parsed is not False or pred != "":
            errors.append("空/空白判词 %r 应记 unparsed 且 pred 留空：实得 %r" % (blank, pred))

    # ⑥ unparsed 的首词如实回传（供计数与样例登记）
    if parse_judgment("maybe\nS1") != ("", "maybe", False):
        errors.append("unparsed 判词应回传首词 'maybe' 且 parsed=False：实得 %r"
                      % (parse_judgment("maybe\nS1"),))

    # ⑦ 延迟统计的确定性
    st = latency_stats([1.0, 2.0, 3.0, 4.0])
    if st["mean"] != 2.5 or st["median"] != 2.5:
        errors.append("latency_stats 均值/中位不符：%r" % (st,))
    if latency_stats([]) != {"mean": None, "median": None, "p95": None}:
        errors.append("空列表 latency_stats 应全为 None")

    # ⑧ 冻结常量与 §2 schema
    if OUTPUT_COLUMNS != ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score",
                          "latency_ms"]:
        errors.append("OUTPUT_COLUMNS 与 §2 schema 不符：%r" % (OUTPUT_COLUMNS,))
    if RAW_TYPE != "judgment":
        errors.append("RAW_TYPE 应为 judgment：%r" % (RAW_TYPE,))

    # ⑨ §0.4 口径对齐五要素（D-34 转交项 40）已显式列出
    if len(GATE4_ELEMENTS) != 5:
        errors.append("GATE4_ELEMENTS 应为 5 要素：实得 %d" % len(GATE4_ELEMENTS))

    if errors:
        for e in errors:
            print("SELF_CHECK_FAIL: %s" % e)
        return 1
    print("03_05_run_b5_llamaguard8b SELF_CHECK OK")
    return 0


def main() -> int:
    init_streams()
    parser = argparse.ArgumentParser(
        description="W4.5 B5 Llama-Guard-3-8B 4-bit 原始推理（§2 schema 逐样本输出）")
    parser.add_argument("--self-check", action="store_true",
                        help="合成判词往返验证解析口径（不读 sealed_test、不加载模型）")
    parser.add_argument("--config", default="configs/baseline/b5_llamaguard8b.yaml")
    parser.add_argument("--unify-config", default="configs/baseline/unify.yaml",
                        help="分片路径的唯一出处（W4.6 冻结）")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="要跑的分片代号；缺省取 b5_llamaguard8b.yaml 的 splits.all")
    parser.add_argument("--outputs-dir", default="reports/baseline/outputs")
    parser.add_argument("--report-dir", default="reports/baseline")
    parser.add_argument("--smoke-double-run", action="store_true",
                        help="冒烟：抽 1 条样本双跑，留判词逐字一致证据（§0.7-4）")
    parser.add_argument("--limit", type=int, default=0,
                        help="每分片只跑前 N 条（>0 生效；仅用于计时探针，不用于正式交付）")
    parser.add_argument("--progress-every", type=int, default=0,
                        help="每 N 条打印一次进度（0=不打印）")
    parser.add_argument("--no-access-reminder", action="store_true",
                        help="不打印 sealed_test 的 ACCESS_LOG 登记提醒")
    parser.add_argument("--tag", default="", help="产物时间戳覆盖（空=取当前时刻）")
    args = parser.parse_args()

    if args.self_check:
        return _self_check()

    cfg = load_config(args.config)
    unify_cfg = load_config(args.unify_config)

    splits = args.splits or list((cfg.get("splits") or {}).get("all") or [])
    if not splits:
        print("NOT_IMPLEMENTED")
        return 0

    ts = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.outputs_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    torch, transformers, tok, model, load_meta = load_model(cfg)
    print("loaded %s（rev %s, %s, attn=%s, dtype=%s, %.1f ms, 加载态峰值 %s MiB）" % (
        load_meta["path"], load_meta["hf_revision"], _require(cfg, "load", "cfg")["mode"],
        load_meta["attn"], load_meta["dtype"], load_meta["load_ms"],
        _mib(load_meta["load_peak_bytes"])))

    # 分片路径（唯一出处 = unify.yaml）
    split_paths = {}
    split_sealed = {}
    for split in splits:
        scfg = (unify_cfg.get("splits") or {}).get(split)
        if not scfg:
            print("SKIP %s：unify.yaml 无该分片" % split)
            continue
        if not os.path.isfile(scfg["path"]):
            print("SKIP %s：CSV 不存在（%s）" % (split, scfg["path"]))
            continue
        split_paths[split] = scfg["path"]
        split_sealed[split] = bool(scfg.get("sealed", False))
    if not split_paths:
        print("NO_SPLIT_EVALUATED")
        return 0

    for split in split_paths:
        if split_sealed[split] and not args.no_access_reminder:
            print("REMINDER: %s 属 data/sealed_test/ —— 读前须已在 "
                  "data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md 登记 6 列，读后复核 md5sum -c"
                  % split)

    smoke = None
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if args.smoke_double_run:
            first_split = list(split_paths)[0]
            smoke = double_run_smoke(first_split, split_paths[first_split], torch, tok, model, cfg)
            print("DOUBLE_RUN_SMOKE split=%s sample=%s verbatim_identical=%s sha256=%s" % (
                smoke["split"], smoke["sample_id"], smoke["verbatim_identical"],
                smoke["sha256_run1"]))
            torch.cuda.reset_peak_memory_stats()   # 抽检不计入「推理态峰值」

        per_split = {}
        for split in split_paths:
            csv_path = split_paths[split]
            rows, stats = run_split(split, csv_path, split_sealed[split], torch, transformers,
                                    tok, model, cfg, progress_every=args.progress_every)
            if args.limit and args.limit > 0:
                rows = rows[:args.limit]
                stats["n"] = len(rows)
                stats["n_unparsed"] = sum(1 for r in rows if r["pred"] == "")
                stats["n_unsafe"] = sum(1 for r in rows if r["pred"] == "1")
                stats["n_safe"] = sum(1 for r in rows if r["pred"] == "0")
            per_split[split] = stats

            out_path = os.path.join(args.outputs_dir, "b5_%s_v1.0_%s.csv" % (split, ts))
            write_csv(out_path, rows, OUTPUT_COLUMNS)
            print("wrote %s（n=%d, unsafe=%d, safe=%d, unparsed=%d, 延迟均值=%s ms）" % (
                out_path, stats["n"], stats["n_unsafe"], stats["n_safe"], stats["n_unparsed"],
                _fmt(stats["latency"]["mean"], 3)))

    infer_peak = int(torch.cuda.max_memory_allocated())
    memory = {"load_peak_bytes": load_meta["load_peak_bytes"], "infer_peak_bytes": infer_peak}

    if not per_split:
        print("NO_SPLIT_EVALUATED")
        return 0

    report_path = os.path.join(args.report_dir, "b5_lg8b_v1.0_%s.md" % ts)
    write_report(report_path, cfg, unify_cfg, per_split, ts, smoke=smoke, memory=memory,
                 load_meta=load_meta, fallback_events=None)
    print("wrote report: %s（加载态峰值 %s MiB／推理态峰值 %s MiB）" % (
        report_path, _mib(memory["load_peak_bytes"]), _mib(memory["infer_peak_bytes"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""05_10_eval_external_baseline.py — W6.12 外部检测器基线 **B6** 评测（④ B 档：近作对照的实测部分）

做什么：在**同一封存 test 分片**上评测外部检测器基线 **B6** =
`meta-llama/Llama-Prompt-Guard-2-86M`（86M 提示级二分类，`DebertaV2ForSequenceClassification`，
BAGEL 论文骨架模型），产出与 B1–B5／Ours **同口径可并列**的表行，供 `EVAL_REPORT.md` §1.2 并入
（`EVAL_REPORT.md` §10 第 1 条「近作对照（④）未做」自此升级为**实测对照**）。

不做什么：**不训练／不微调**任何模型；不用测试集替 B6 调阈值／选 ckpt／选解码参数（`D-53` 裁定四）；
不重跑 B1–B5 与 Ours；不做显著性检验（B6 **只作描述性并列**，不入预注册三组 paired bootstrap）；
不改任何既有数值。

接口（`windows/W6.12_external_baseline.md` §4，冻结）::

    python scripts/eval/05_10_eval_external_baseline.py \
        --model <本地权重目录> --kind {judgment,score} \
        --source sealed --split {in_regime_test,g3_hard_negative_test} \
        --report-dir reports/eval/ --outputs-dir reports/eval/outputs_ext/ \
        --access-log-note "<已登记的 ACCESS_LOG 行摘要>" \
        [--limit N] [--self-check] [--dry-run]

    # 只看计划（不读封存内容）
    python scripts/eval/05_10_eval_external_baseline.py --dry-run ...

stdout 末行：`REPORT_MD=<path>`；`--self-check` ⇒ `SELFCHECK_RESULT=PASS|FAIL n=<k>`；
`--dry-run` ⇒ `DRYRUN_RESULT=PASS n=<k>`。退出码：正常 0；断言/守卫失败**非零**。

`--kind` 两分支**只许择一**（`PREREG_v1.10.md` §1.2）：
  - `judgment`（**本窗口采用**）：B6 为标签型（模型卡 usage 自报 `predicted_class_id`）⇒
    `pred = argmax(logits)`，`LABEL_0=benign`（判 0）／`LABEL_1=malicious`（判 1）；
    `config.json` 无 `id2label` ⇒ **不得**用 `model.config.id2label` 的占位 `LABEL_0/LABEL_1`；
    无连续 `score` ⇒ AUROC／Span-F1 记 `n/a`（不适用，**非缺失、非 0**）。
  - `score`（**未激活**）：B6 实际落地为 argmax 标签型；该分支（原生阈值＋`0.5` 并列）属新增口径，
    须新建 `PREREG` 版本 ⇒ 本脚本显式拒绝（非零退出），**不**静默回落。

硬约束落点：
  C1 无调参（加载/判定参数一律取 `configs/eval/external_baseline.yaml` 事前定值）
  C2 不混代（B6 与 Ours v2 同轮同分片；Ours 数引 `eval-v1.1` 既有值，本脚本不重跑）
  C3 无默认兜底（`--model`/`--kind`/`--source` 缺失即抛错；`pred` 非 0/1 抛错）
  C4 不许静默（空 text／未解析 ⇒ 计数并披露，**不得**当「安全」）
  C5 FPR 与 TPR 并列（`D-111` 裁定六）
  C6 复用单源（`03_06.load_samples`／`truncate_left`／`sha256_of`／`md5_of`；
     `05_01.assert_input_allowed`／`resolve_input_path`／`assert_record_count`／`latency_stats`；
     `03_07.compute_binary_metrics`）
  C7 不改既有签署件（本脚本只读 `PREREG_v1.10.md`）
  C8 `csv.DictReader` 计记录数（严禁 `readlines()`/`split("\\n")`；`text` 含换行 ⇒ 行数 ≫ 记录数）
  C10 结果不利照实报（报告 §4 局限段必写）
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# B6 标签约定（`PREREG_v1.10.md` §1.2；`models/commit_hashes/llama-prompt-guard-2-86m.json`
# `provenance.4_model_card_label_convention`）：config.json 无 id2label ⇒ 依模型卡 usage 示例
# ＋ 本机 smoke（恶意样本 argmax=1／良性 argmax=0）定死。**禁止**回落到 config 的占位映射。
LABEL_0_NAME = "benign"
LABEL_1_NAME = "malicious"
DETECTOR = "b6"
NA = "n/a"


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：W3.7 `F-1`；`D-40` 裁定五：不得静默 pass）
# ---------------------------------------------------------------------------
def init_streams():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def load_module(name, rel_path):
    """按路径加载模块（同 `05_01.load_module`；只调接口、不复制实现）。"""
    path = os.path.join(ROOT, rel_path.replace("/", os.sep))
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
    import yaml  # 惰性 import（顶层不 import 重型库）

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def resolve_log_dir(args=None):
    """日志目录解析（`#64`／`D-80` 裁定三／`D-82` 同制）。

    优先级：① `CODEOBF_LOG_DIR`；② `--self-check`／`--dry-run` ⇒ `logs/eval/tmp/`；③ `logs/eval/`。
    """
    env = os.environ.get("CODEOBF_LOG_DIR")
    if env:
        return env
    if args is not None and (getattr(args, "self_check", False) or getattr(args, "dry_run", False)):
        return os.path.join(ROOT, "logs", "eval", "tmp")
    return os.path.join(ROOT, "logs", "eval")


def flagged_logger(ts, log_dir=None):
    log_dir = log_dir or os.path.join(ROOT, "logs", "eval")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "05_10_eval_external_baseline_%s.log" % ts)

    def _log(msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    return _log, path


# ---------------------------------------------------------------------------
# C6 封存访问守卫（`--source sealed` 缺 note ⇒ 拒绝执行）
# ---------------------------------------------------------------------------
def require_access_note(cfg, source, note):
    acc = cfg["access"]
    if source != "sealed":
        return None
    if not bool(acc["sealed_requires_note"]):
        return None
    text = (note or "").strip()
    if len(text) < int(acc["note_min_chars"]):
        raise SystemExit(
            "ACCESS_LOG_GUARD_FAIL：`--source sealed` 必须给 `--access-log-note`"
            "（≥%d 字符，内容为已登记的 `%s` 行摘要；6 列＝%s）"
            % (int(acc["note_min_chars"]), acc["access_log"], "／".join(acc["access_log_columns"])))
    return text


# ---------------------------------------------------------------------------
# C6 输入面解析（单源：05_01）
# ---------------------------------------------------------------------------
def resolve_shard(ev_mod, base_cfg, cfg, source, regime, split):
    """解析分片 ⇒ `(path, expect, shard, regime_effective)`。

    `--source val` ⇒ `--regime {id,ood}` 必填、`--split` 必须省略（val 无 split 轴）；
    `--source sealed` ⇒ `--split` 必填；`in_regime_test` 须 `--regime`，`g3_hard_negative_test` 须**省略** `--regime`。
    """
    if source == "val":
        if split:
            raise ValueError("`--source val` 不支持 `--split`（val 分片由 `--regime` 唯一确定）")
        if regime not in ("id", "ood"):
            raise ValueError("`--source val` 必须给 `--regime {id,ood}`")
        path, expect, shard = ev_mod.resolve_input_path(base_cfg, "val", regime, None, allow_sealed=False)
        return path, expect, shard, regime
    if source == "sealed":
        if not split:
            raise ValueError("`--source sealed` 必须同时给 `--split`（缺一即拒绝执行）")
        if split == "in_regime_test":
            if regime not in ("id", "ood"):
                raise ValueError("`--split in_regime_test` 必须给 `--regime {id,ood}`")
        elif split == "g3_hard_negative_test":
            if regime:
                raise ValueError("`--split g3_hard_negative_test` 不接受 `--regime`"
                                 "（G3 为单模型单次评测，无体制轴）")
        else:
            raise ValueError("未知 --split：%r" % (split,))
        path, expect, shard = ev_mod.resolve_input_path(base_cfg, "sealed", regime, split, allow_sealed=True)
        return path, expect, shard, regime or ""
    raise ValueError("未知 --source：%r" % (source,))


# ---------------------------------------------------------------------------
# C1 加载（fp32 原生／cuda／eager；重型库函数体内惰性 import）
# ---------------------------------------------------------------------------
def load_b6(cfg, model_path, logger=print):
    """加载 B6 tokenizer + `AutoModelForSequenceClassification`（fp32、cuda、eager）。

    返回 `(torch, transformers, tok, model, load_meta)`；`load_meta` 含加载耗时与**加载态显存峰值**。
    """
    det = cfg["load"]["determinism"]
    # `CUBLAS_WORKSPACE_CONFIG` 必须在 import torch **之前**生效（`AGENTS.md` §3）
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(det["cublas_workspace_config"])
    import torch
    import transformers

    load_cfg = cfg["load"]
    if not os.path.isdir(model_path):
        raise FileNotFoundError("B6 权重目录不存在：%s" % model_path)

    torch.backends.cudnn.deterministic = bool(det["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(det["cudnn_benchmark"])
    torch.backends.cuda.matmul.allow_tf32 = bool(det["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(det["allow_tf32"])

    tok = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=bool(cfg["model"]["local_files_only"]))
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id"):
        if getattr(tok, attr, None) is None:
            raise ValueError("tokenizer 缺少 %s（分类器输入构造需要 cls/sep/pad）" % attr)

    dtype = getattr(torch, str(load_cfg["dtype"]))
    attn = str(load_cfg["attn_implementation"])
    if attn in list(load_cfg.get("forbidden_attn") or []):
        raise ValueError("禁止的 attn_implementation：%r（全阶段固定 eager）" % (attn,))

    if str(load_cfg["device"]) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 device=cuda，但 torch.cuda.is_available() 为 False")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        attn_implementation=attn,
        dtype=dtype,
    )
    if int(getattr(model.config, "num_labels", 0) or 0) != 2:
        raise ValueError("B6 须为二分类（num_labels=2），实测 num_labels=%r"
                         % (getattr(model.config, "num_labels", None),))
    model.eval()                                        # G-EVAL：`AGENTS.md` §6-11
    if str(load_cfg["device"]) == "cuda":
        model = model.to("cuda")
    load_ms = (time.perf_counter() - t0) * 1000.0
    load_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats()
    logger("B6_LOADED path=%s arch=%s dtype=%s attn=%s device=%s load_ms=%.1f"
           % (model_path, cfg["model"]["architectures"], str(dtype), attn,
              str(load_cfg["device"]), load_ms))
    # `config.json` 无 id2label ⇒ transformers 会填占位 `{0:'LABEL_0',1:'LABEL_1'}`；
    # 该占位**不得**当标签用（`PREREG_v1.10.md` §1.2），此处显式留痕。
    logger("B6_ID2LABEL_NOTE config.id2label=%r（占位，不用）；本脚本用 LABEL_0=%s/LABEL_1=%s"
           % (getattr(model.config, "id2label", None), LABEL_0_NAME, LABEL_1_NAME))
    return torch, transformers, tok, model, {
        "load_ms": load_ms,
        "load_peak_bytes": load_peak,
        "attn": attn,
        "dtype": str(dtype),
        "path": model_path,
    }


# ---------------------------------------------------------------------------
# §1.2 输入：原文直通 → 左截断 512 content token（保尾部）
# ---------------------------------------------------------------------------
def encode_content_ids(tok, pp_mod, text, max_tokens):
    """返回 `(content_ids, n_raw_tokens, roundtrip_guard_fired)`。

    - `text` **原文直通**（调用方已由 `03_06.load_samples` 保证零清洗）；
    - 超长 ⇒ 先过 `03_06.truncate_left`（**单源**：左截断、保尾部 `max_tokens`）；
    - `truncate_left` 的 decode→re-encode **往返会膨胀**（实测多数回 513、个别 676）⇒ 追加
      **token 级 `max_tokens` 硬钳**（同向同长；`roundtrip_guard_fired` 计数披露，**非静默兜底**）；
    - `content_ids == []`（空 text／无法编码）⇒ 交上层记 `unparsed`（C4）。
    """
    raw_ids = tok(text, add_special_tokens=False)["input_ids"]
    n_raw = len(raw_ids)
    if n_raw == 0:
        return [], 0, False
    if n_raw <= max_tokens:
        return list(raw_ids), n_raw, False
    truncated_text = pp_mod.truncate_left(text, tok, max_tokens)
    ids = tok(truncated_text, add_special_tokens=False)["input_ids"]
    fired = len(ids) > max_tokens
    if fired:
        ids = ids[-max_tokens:]
    return list(ids), n_raw, fired


def build_batch_tensors(torch, id_lists, tok, device):
    """手工组批：`[CLS] + content + [SEP]`，右 pad 至批内最长，返回 `(input_ids, attention_mask)`。

    特殊 token id 一律取自 tokenizer（不硬编码）；`token_type_ids` 不传
    （`type_vocab_size=0`，DeBERTa-v2 不用该列）。
    """
    cls_id = int(tok.cls_token_id)
    sep_id = int(tok.sep_token_id)
    pad_id = int(tok.pad_token_id)
    fulls = [[cls_id] + ids + [sep_id] for ids in id_lists]
    width = max(len(x) for x in fulls)
    input_ids = [x + [pad_id] * (width - len(x)) for x in fulls]
    attn = [[1] * len(x) + [0] * (width - len(x)) for x in fulls]
    return (torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(attn, dtype=torch.long, device=device))


def pred_from_logits(logits_pair):
    """`argmax(logits)` 的确定性写法：`logits[1] > logits[0]` ⇒ 1，否则 0（并列取 0）。

    返回 `(pred, argmax_index)`；`pred` 恒为 0/1（C3：非 0/1 即由上层断言抛错）。
    """
    l0, l1 = float(logits_pair[0]), float(logits_pair[1])
    k = 1 if l1 > l0 else 0
    return (1 if k == 1 else 0), k


# ---------------------------------------------------------------------------
# C12 CSV（写端 LF；只追加；读端一律 DictReader）
# ---------------------------------------------------------------------------
def write_csv_rows(path, fields, rows):
    if os.path.exists(path):
        raise FileExistsError("C12 只追加：产物已存在，拒绝覆盖 %s" % path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


def count_records(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def fmt_metric(x, nd=4):
    """已算完的指标格子：`None` ⇒ `n/a`（**不适用**，不是 0、不是 `<TBD>`）。"""
    if x is None:
        return NA
    return ("%." + str(nd) + "f") % (float(x),)


# ---------------------------------------------------------------------------
# 评测主流程（单分片）
# ---------------------------------------------------------------------------
def evaluate_shard(cfg, args, ev_mod, pp_mod, map_mod, torch, tok, model, base_cfg,
                   path, expect, shard, regime, note, load_peak_bytes, logger):
    """读分片 → 断言记录数 → 逐批推理 → 返回 `(rows, metric_row, meta)`。"""
    max_tokens = int(cfg["input"]["max_content_tokens"])
    batch_size = int(cfg["latency"]["batch_size"])
    fields = cfg["outputs"]["per_sample_fields"]
    allow_sealed = (args.source == "sealed")

    ev_mod.assert_input_allowed(path, base_cfg, allow_sealed=allow_sealed)
    md5_before = pp_mod.md5_of(path)
    got = ev_mod.assert_record_count(path, expect, base_cfg)
    samples = pp_mod.load_samples(path)
    if len(samples) != got:
        raise AssertionError("记录数不一致：assert_record_count=%d vs load_samples=%d（%s）"
                             % (got, len(samples), path))
    md5_after = pp_mod.md5_of(path)
    if md5_before != md5_after:
        raise AssertionError("读取前后 md5 变化（%s → %s）：%s" % (md5_before, md5_after, path))
    if args.limit is not None:
        if int(args.limit) <= 0:
            raise ValueError("--limit 必须为正整数")
        if int(args.limit) > len(samples):
            raise ValueError("--limit=%d 超过分片记录数 %d" % (int(args.limit), len(samples)))
        samples = samples[:int(args.limit)]
    logger("READ_SHARD shard=%s regime=%s records=%d md5=%s%s"
           % (shard, regime or "-", got, md5_after,
              " limit=%d" % len(samples) if args.limit is not None else ""))

    labels = [int(s.label) for s in samples]
    bad_labels = sorted({l for l in labels if l not in (0, 1)})
    if bad_labels:
        raise ValueError("label 非 0/1：%r（%s）" % (bad_labels, path))

    device = "cuda" if str(cfg["load"]["device"]) == "cuda" else "cpu"
    rows = []
    latencies = []
    preds = []
    n_truncated = 0
    n_guard = 0
    n_unparsed = 0
    equiv_checked = False
    pad_check_done = False
    singleton_logits = None

    # 预备：首条样本单独前向（**不计时**）——用于 padding 不变性冒烟（C4：不一致即抛错）
    if samples:
        ids0, _n0, _f0 = encode_content_ids(tok, pp_mod, samples[0].text, max_tokens)
        if ids0:
            ii, am = build_batch_tensors(torch, [ids0], tok, device)
            with torch.no_grad():
                singleton_logits = model(input_ids=ii, attention_mask=am).logits[0].tolist()

    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        t0 = time.perf_counter()
        enc = [encode_content_ids(tok, pp_mod, s.text, max_tokens) for s in chunk]
        usable = [i for i, e in enumerate(enc) if e[0]]
        id_lists = [enc[i][0] for i in usable]
        if id_lists:
            ii, am = build_batch_tensors(torch, id_lists, tok, device)
            with torch.no_grad():
                logits = model(input_ids=ii, attention_mask=am).logits.tolist()
        else:
            logits = []
        dt_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = dt_ms / len(chunk) if chunk else 0.0
        latencies.extend([per_sample_ms] * len(chunk))

        # token 级构造 == tokenizer 原生构造（仅未截断的首条样本；防特殊 token/pad 接线错误）
        if not equiv_checked and enc and enc[0][0] and enc[0][1] <= max_tokens:
            native = tok(samples[start].text, add_special_tokens=True)["input_ids"]
            manual = [int(tok.cls_token_id)] + list(enc[0][0]) + [int(tok.sep_token_id)]
            if native != manual:
                raise AssertionError(
                    "TOKENIZE_EQUIV_FAIL：token 级构造 != tokenizer 原生构造（%s 首条样本）" % shard)
            equiv_checked = True
            logger("TOKENIZE_EQUIV_OK shard=%s（未截断首条：手动构造 == tok(add_special_tokens=True)）" % shard)

        for j, s in enumerate(chunk):
            content_ids, n_raw, fired = enc[j]
            if n_raw > max_tokens:
                n_truncated += 1
            if fired:
                n_guard += 1
            if not content_ids:
                n_unparsed += 1
                rows.append({
                    "sample_id": s.sample_id, "text_sha256": pp_mod.sha256_of(s.text),
                    "pred": "", "raw_type": cfg["outputs"]["raw_type"], "raw": "",
                    "score": "", "latency_ms": "%.4f" % per_sample_ms,
                    "label": s.label, "n_content_tokens": 0,
                    "truncated": 1 if n_raw > max_tokens else 0, "unparsed": 1,
                })
                preds.append(None)
                continue
            pair = logits[usable.index(j)]
            pred, argmax_k = pred_from_logits(pair)
            if pred not in (0, 1):
                raise AssertionError("pred 非 0/1：%r（sample_id=%s）" % (pred, s.sample_id))
            raw = json.dumps({"logits": [float(pair[0]), float(pair[1])],
                              "argmax": int(argmax_k),
                              "label_name": LABEL_1_NAME if argmax_k == 1 else LABEL_0_NAME},
                             ensure_ascii=False)
            rows.append({
                "sample_id": s.sample_id, "text_sha256": pp_mod.sha256_of(s.text),
                "pred": pred, "raw_type": cfg["outputs"]["raw_type"], "raw": raw,
                "score": "", "latency_ms": "%.4f" % per_sample_ms,
                "label": s.label, "n_content_tokens": len(content_ids),
                "truncated": 1 if n_raw > max_tokens else 0, "unparsed": 0,
            })
            preds.append(pred)

        # padding 不变性冒烟：首批第一条的批内 logits vs 单独前向 logits（C4：argmax 必须一致）
        if not pad_check_done and singleton_logits is not None and id_lists:
            pair = logits[usable.index(0)]
            max_diff = max(abs(float(pair[0]) - float(singleton_logits[0])),
                           abs(float(pair[1]) - float(singleton_logits[1])))
            if pred_from_logits(pair)[0] != pred_from_logits(singleton_logits)[0]:
                raise AssertionError(
                    "PADDING_INVARIANCE_FAIL：批内 argmax != 单独前向 argmax（%s 首条样本）" % shard)
            logger("PADDING_INVARIANCE_OK shard=%s max_abs_logit_diff=%.3e（批=%d vs 单独）"
                   % (shard, max_diff, len(chunk)))
            pad_check_done = True

    if len(rows) != len(samples):
        raise AssertionError("逐样本行数 != 样本数（%d vs %d）" % (len(rows), len(samples)))
    for k in fields:
        if k not in rows[0]:
            raise AssertionError("逐样本行缺少字段 %r" % (k,))

    # 指标（单源：03_07.compute_binary_metrics；标签型 ⇒ 不算 AUROC）
    parsed_idx = [i for i, p in enumerate(preds) if p is not None]
    if not parsed_idx:
        raise AssertionError("全部样本未解析 ⇒ 指标无定义（C4：不得把未解析当「安全」）")
    m = map_mod.compute_binary_metrics([labels[i] for i in parsed_idx],
                                       [preds[i] for i in parsed_idx], scores=None)
    lat = ev_mod.latency_stats(latencies)
    infer_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    metric_row = {
        "detector": DETECTOR, "shard": shard, "regime": regime, "source": args.source,
        "kind": args.kind, "n": len(samples), "n_pos": m["n_pos"], "n_neg": m["n_neg"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        "tpr": m["tpr"], "fpr": m["fpr"], "f1": m["f1"],
        "auroc": None, "span_f1": None,
        "latency_mean_ms": lat["mean_ms"], "latency_median_ms": lat["median_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "mem_load_peak_mib": (float(load_peak_bytes) / 1048576.0 if load_peak_bytes else None),
        "mem_infer_peak_mib": (float(infer_peak) / 1048576.0 if infer_peak is not None else None),
        "eval_batch_size": batch_size,
        "n_truncated": n_truncated, "n_unparsed": n_unparsed,
        "roundtrip_guard_fired": n_guard,
        "model_path": str(cfg["model"]["local_path"]),
        "config_path": _rel(_abs(args.config)),
    }
    logger("METRICS shard=%s n=%d tp=%d fp=%d fn=%d tn=%d tpr=%s fpr=%s f1=%s auroc=%s"
           % (shard, len(samples), m["tp"], m["fp"], m["fn"], m["tn"],
              fmt_metric(m["tpr"]), fmt_metric(m["fpr"]), fmt_metric(m["f1"]), NA))
    logger("TRUNCATION shard=%s n_truncated=%d roundtrip_guard_fired=%d n_unparsed=%d"
           % (shard, n_truncated, n_guard, n_unparsed))
    meta = {"n": len(samples), "n_parsed": len(parsed_idx), "records_asserted": got,
            "md5": md5_after, "n_truncated": n_truncated, "n_guard": n_guard,
            "n_unparsed": n_unparsed, "lat": lat, "access_note": note}
    return rows, metric_row, meta


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def render_report(cfg, ctx):
    lines = []
    a = lines.append
    a("# 外部检测器基线 B6 评测运行报告 — %s" % ctx["ts"])
    a("")
    a("- 窗口 **W6.12**（`windows/W6.12_external_baseline.md`）｜脚本 `scripts/eval/05_10_eval_external_baseline.py`"
      "｜配置 `%s`" % ctx["config_path"])
    a("- 依据：`PREREG_v1.10.md`（`P-M` = B6，**已签署**，annotated tag `prereg-v1.10` → `a3d79ee`）；"
      "`DECISIONS.md` `D-151`（④ B 档）／`D-152`（签署后放行主表）")
    a("- 基线：**%s**" % cfg["detector_name"])
    a("- 来源：`%s`｜分片：%s｜`--kind`：**%s**" % (ctx["source"], "、".join(ctx["shards"]), ctx["kind"]))
    a("")
    a("## 0. 声明（必读）")
    a("")
    a("1. **B6 为外部基线，非本项目训练**：只评测、不训练；未在本项目的验证集或测试集上调阈值／选 checkpoint／选解码参数（`D-53` 裁定四）。")
    a("2. **只作描述性并列，不做显著性检验**：B6 **不纳入**预注册三组 paired bootstrap"
      "（`ours_vs_b3@id`／`ours_vs_b3@ood`／`ours_vs_b5@id` 不变）⇒ 本报告**无** p 值、**无** 95% CI、**无**「显著」措辞。")
    a("3. **边界**：只加**一条**外部基线、**未复现 BAGEL 全框架**（BAGEL 为任务级检测器集成，与我们的因子级口径不可比，`D-151` 一·C 不批准）。")
    a("4. **结果不利照实报告**（C10）：B6 若强于 Ours ⇒ 照写；不删基线、不换分片、不调阈值。")
    a("")
    a("**披露义务（`PREREG_v1.10.md` §5，逐字）**：")
    a("")
    for decl in cfg["report"]["must_include"]:
        a("- " + decl)
    a("")
    a("## 1. 事前定值（跑前写定；跑后未改）")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    a("| 模型 | `%s`（魔搭 `%s`；HF revision `%s` 作跨境复现锚点） |"
      % (cfg["model"]["repo_id"], cfg["model"]["modelscope_repo_id"], cfg["model"]["hf_revision"]))
    a("| 架构 | `%s`（实测 `config.json`；**非** causal LM） |" % cfg["model"]["architectures"])
    a("| 权重路径 | `%s`（项目外盘位；11 件 1,131,675,241 B，`PROVENANCE.json` 逐文件 sha256 MATCH） |"
      % cfg["model"]["local_path"])
    a("| 加载 | **fp32 原生**（不量化）／device `cuda`／`attn_impl=eager`／`eval()`＋`no_grad`；"
      "确定性 `CUBLAS_WORKSPACE_CONFIG=%s`、`cudnn.deterministic=%s`、`allow_tf32=%s` |"
      % (cfg["load"]["determinism"]["cublas_workspace_config"],
         cfg["load"]["determinism"]["cudnn_deterministic"],
         cfg["load"]["determinism"]["allow_tf32"]))
    a("| 输入 | 只读 `text` 列，**原文直通**（同 `03_06`：不清洗／不归一／不改大小写）；"
      "**左截断保尾部 %d content token**（`03_06.truncate_left` 单源＋token 级硬钳，见 §4-3） |"
      % int(cfg["input"]["max_content_tokens"]))
    a("| 判定 | `pred = argmax(logits)`；`LABEL_0=benign`（判 0）／`LABEL_1=malicious`（判 1）；"
      "`config.json` **无 `id2label`** ⇒ 不用 `model.config.id2label` 的占位 `LABEL_0/LABEL_1` |")
    a("| 批大小 | **%d**（事前定值；延迟为**批际摊到样本**，须与批大小一并解读） |"
      % int(cfg["latency"]["batch_size"]))
    a("| 记录数断言 | %s（`csv.DictReader`；不符即非零退出、禁改断言值） |"
      % "；".join("%s=%d" % (p["assertion_key"], int(p["expect_records"])) for p in cfg["population"]["shards"]))
    a("| 封存访问 | `%s` 6 列**先登后读**；note = `%s` |" % (cfg["access"]["access_log"], ctx["access_note"] or "—"))
    a("")
    a("## 2. 结果（三表行；与 B1–B5／Ours 同分片同指标定义并列）")
    a("")
    a("| 分片 | n | n_pos | n_neg | TP | FP | FN | TN | **TPR** | **FPR** | F1 | AUROC | Span-F1 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in ctx["rows"]:
        a("| `%s` | %d | %d | %d | %d | %d | %d | %d | **%s** | **%s** | %s | %s | %s |"
          % (r["shard"], r["n"], r["n_pos"], r["n_neg"], r["tp"], r["fp"], r["fn"], r["tn"],
             fmt_metric(r["tpr"]), fmt_metric(r["fpr"]), fmt_metric(r["f1"]), NA, NA))
    a("")
    a("| 分片 | 延迟 均值 ms | 延迟 中位 ms | 延迟 p95 ms | 显存 加载态 MiB | 显存 推理态 MiB | 截断条数 | 往返硬钳 | 未解析 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in ctx["rows"]:
        a("| `%s` | %.3f | %.3f | %.3f | %s | %s | %d | %d | %d |"
          % (r["shard"], r["latency_mean_ms"], r["latency_median_ms"], r["latency_p95_ms"],
             fmt_metric(r["mem_load_peak_mib"], 1), fmt_metric(r["mem_infer_peak_mib"], 1),
             r["n_truncated"], r["roundtrip_guard_fired"], r["n_unparsed"]))
    a("")
    a("> **AUROC／Span-F1 = `n/a`（不适用，非缺失、非 0）**：B6 为**标签型**（`argmax(logits)`），"
      "无连续 `score` 可供排序 ⇒ 与 B1／B3／B5 同口径（`BASELINE_REPORT` §1 注释 3）。"
      "逐样本 `raw` 列含 logits（**审计留痕，不作为连续 score 使用**）。")
    a("> **延迟／显存只并列、不相减**：B6 为 fp32 原生 GPU 档，与 B3／B5 的 `4bit-nf4`、B4 的 CPU 档"
      "**不同量纲**（`PREREG_v1.10.md` §1.1；`EVAL_REPORT.md` §6 不可比性声明同制）⇒ "
      "**不跨档排名**；`FPR`／`TPR` 同为比例，不受量纲影响。")
    a("> **G3 分片全良性**（`label=0` × 64）⇒ `TPR` 无定义记 `n/a`；表中 `F1 = 0.0000` 系 "
      "`2TP/(2TP+FP+FN)` 分母为 0 的 `zero_division` 约定（`03_07.f1_score`），**不是**检出质量。")
    a("")
    a("## 3. 与 B3／B5 的可比性说明")
    a("")
    a("- **同分片、同 `pred` 口径、同指标定义**（TPR／FPR／F1）⇒ 本表行可与 `EVAL_REPORT.md` §1.2 的"
      " B1–B5／Ours 行**直接并列**。")
    a("- **不可比项**：加载档（fp32 vs 4bit-nf4 vs CPU）⇒ 延迟与显存**只并列不相减**；"
      "运行次数（B6 单模型 1 次 vs Ours 6 run）⇒ 不给 mean±std、不做 per-seed 拆分。")
    a("- **Ours 数字引 `eval-v1.1` 既有值**（`detection_formal_v1.0_20261022-144138`／`…-173301`），"
      "本窗口**未重跑** B1–B5 与 Ours（C2 不混代）。")
    a("")
    a("## 4. 局限（必写）")
    a("")
    a("1. **只加一条外部基线**：未复现 BAGEL 全框架；B6 的 86M 规模与 Prompt-Guard 路线**不代表**"
      "「近作对照」的全部口径（`reports/analysis/related_work_v1.0_20261022-192004.md` §3.1）。")
    a("2. **无显著性检验**：B6 只作描述性并列；任何「B6 vs Ours」的强弱陈述**不得**写成「显著」。")
    a("3. **截断口径**：DeBERTa-v2 的 disentangled attention 偏置随序列长**平方**增长"
      "（实测 L≈3461 时单批需 ≈9.2 GB）⇒ 本窗口按 §1.3 左截断到 **512 content token**；"
      "`03_06.truncate_left` 的 decode→re-encode 往返会膨胀（实测多数回 513、个别 676）⇒ 追加 "
      "token 级 512 硬钳（`roundtrip_guard_fired` 计数披露，**非静默兜底**）。")
    a("4. **长输入（512／2048 组）未评**：B6 原生上下文 512 token，且二次注意力使 2048 组不可行 ⇒ "
      "`EVAL_REPORT.md` §4 明写「未评」（`PREREG_v1.10.md` §3-7）。")
    a("5. **未解析不当安全**（C4）：`n_unparsed` 逐分片披露（本窗口为 0 亦照写）；"
      "空 `text` 样本 `pred` 留空、不计入指标。")
    a("6. **G3 的 F1 退化**：全良性分片上 `F1=0.0000` 系分母为 0 的约定值，只应读 `FPR`。")
    a("")
    a("## 5. 产物")
    a("")
    a("| 件 | 路径 |")
    a("|---|---|")
    a("| 运行报告（本件） | `%s` |" % ctx["paths"]["report"])
    a("| 指标表 of record | `%s` |" % ctx["paths"]["metric_csv"])
    for shard, p in zip(ctx["shards"], ctx["paths"]["per_sample"]):
        a("| 逐样本输出 `%s` | `%s` |" % (shard, p))
    a("| 运行日志 | `%s` |" % ctx["paths"]["log"])
    a("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# self-check（逻辑面；**不加载模型、不读封存**）
# ---------------------------------------------------------------------------
def self_check(cfg, args):
    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    base_cfg = load_config(_abs(cfg["base_eval_config"]))
    ev_mod = load_module("ev_05_01_sc", cfg["modules"]["eval_engine"])
    pp_mod = load_module("pp_03_06_sc", cfg["modules"]["preprocess"])
    map_mod = load_module("map_03_07_sc", cfg["modules"]["mapping"])
    src = open(os.path.abspath(__file__), "r", encoding="utf-8").read()

    # 1 事前定值人群与基础配置一致（1583／740／64）
    asserts = base_cfg["record_assertions"]
    declared = {p["assertion_key"]: int(p["expect_records"]) for p in cfg["population"]["shards"]}
    ok("population_three_shards", len(declared) == 3, "shards=%d" % len(declared))
    ok("population_matches_base_assertions",
       all(int(asserts[k]) == v for k, v in declared.items()),
       "; ".join("%s=%d" % (k, v) for k, v in sorted(declared.items())))
    ok("population_values_frozen",
       declared.get("regime_id_test") == 1583 and declared.get("regime_ood_test") == 740
       and declared.get("g3_hard_negative_test") == 64, "")

    # 2 标签映射（argmax → pred；LABEL_0=benign／LABEL_1=malicious）
    ok("label_argmax_malicious", pred_from_logits([-5.1573, 2.5018])[0] == 1, "")
    ok("label_argmax_benign", pred_from_logits([0.0034, -7.8931])[0] == 0, "")
    ok("label_tie_takes_zero", pred_from_logits([1.0, 1.0])[0] == 0, "并列取 0（确定性）")
    ok("label_names_frozen",
       LABEL_0_NAME == "benign" and LABEL_1_NAME == "malicious"
       and cfg["mapping"]["label_0"] == "benign" and cfg["mapping"]["label_1"] == "malicious", "")
    ok("kind_is_judgment", cfg["mapping"]["kind"] == "judgment" and args.kind in (None, "judgment"),
       "kind=%r" % (args.kind,))
    ok("auroc_span_f1_na", cfg["mapping"]["auroc"] == "n/a" and cfg["mapping"]["span_f1"] == "n/a", "")
    ok("score_branch_refused_in_main", "KIND_NOT_PREREGISTERED" in src, "")

    # 3 输入编码：原文直通 + 左截断 + token 级硬钳（桩 tokenizer，不加载真实权重）
    #    桩口径一（_FakeTok）：每字符一个 token，id = 1000 + 位置序号 ⇒ 可验「单源」：
    #    `n_raw` 必须取自**首次**编码（= 原文 token 数），截断后**重新编码**。
    class _FakeTok(object):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(1000, 1000 + len(text)))}

    class _FakePP(object):
        @staticmethod
        def truncate_left(text, tok, max_tokens):
            n = int(max_tokens)
            return text[-n:] if len(text) > n else text   # 桩：保尾部（模拟 03_06 真行为）

    _ALPHA = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    class _StableTok(object):
        """桩口径二：id 由**字符本身**决定（与位置无关）⇒ 可验「保尾部」。"""
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

    short = _FakeTok()
    ids, n_raw, fired = encode_content_ids(short, _FakePP, "x" * 10, 512)
    ok("passthrough_short", ids == list(range(1000, 1010)) and n_raw == 10 and not fired,
       "n=%d" % n_raw)

    orig = _ALPHA * 9 + _ALPHA[:42]          # 600 字符（62*9+42）
    assert len(orig) == 600
    ids, n_raw, fired = encode_content_ids(_StableTok(), _FakePP, orig, 512)
    expect_tail = [1000 + _ALPHA.index(orig[88 + i]) for i in range(512)]
    ok("truncate_single_source_keeps_tail",
       n_raw == 600 and len(ids) == 512 and not fired and ids == expect_tail
       and ids[0] == 1000 + _ALPHA.index(orig[88]),
       "保尾部 512：ids[0] 对应原文第 88 字符（非头部）；n_raw=%d" % n_raw)

    ids, n_raw, fired = encode_content_ids(_InflTok(), _FakePP, orig[:300], 512)
    expect_guard = []
    for k in range(44, 300):                 # 600 token ⇒ 钳掉前 88（= 前 44 字符）
        i = _ALPHA.index(orig[k])
        expect_guard.extend([1000 + i, 1500 + i])
    ok("roundtrip_guard_fires", len(ids) == 512 and fired and ids == expect_guard,
       "guard=%s n=%d（往返仍 >512 ⇒ token 级硬钳、同向保尾、计数披露）" % (fired, len(ids)))

    empty = _FakeTok()
    ids, n_raw, fired = encode_content_ids(empty, _FakePP, "", 512)
    ok("empty_text_is_unparsed_not_safe", ids == [] and n_raw == 0, "空 ⇒ 上层记 unparsed（C4）")
    ok("pred_domain_guard_in_src", "pred 非 0/1" in src, "")

    # 4 记录数口径（C8）：split/regime 校验（单源 05_01.resolve_input_path）
    #    注：05_01 的 `resolve_input_path` 在 `regime=None` 时是**字典键缺失 ⇒ KeyError**
    #    （不是 ValueError）；本脚本的 `resolve_shard` 在此之前已用 ValueError 预拦。
    #    两层都测：预拦层（主防线，须 ValueError）＋ 单源层（不得静默放行）。
    try:
        resolve_shard(ev_mod, base_cfg, cfg, "sealed", None, "in_regime_test")
        ok("in_regime_requires_regime", False, "未抛错")
    except ValueError:
        ok("in_regime_requires_regime", True, "")
    try:
        ev_mod.resolve_input_path(base_cfg, "sealed", None, "in_regime_test", True)
        ok("upstream_rejects_none_regime", False, "单源层未抛错（会静默取错分片）")
    except (KeyError, ValueError):
        ok("upstream_rejects_none_regime", True, "")
    try:
        ev_mod.resolve_input_path(base_cfg, "sealed", "id", "nope", True)
        ok("unknown_split_rejected", False, "未抛错")
    except ValueError:
        ok("unknown_split_rejected", True, "")
    try:
        resolve_shard(ev_mod, base_cfg, cfg, "sealed", "id", "g3_hard_negative_test")
        ok("g3_rejects_regime", False, "未抛错")
    except ValueError:
        ok("g3_rejects_regime", True, "")
    try:
        resolve_shard(ev_mod, base_cfg, cfg, "sealed", None, "in_regime_test")
        ok("sealed_requires_split", False, "未抛错")
    except ValueError:
        ok("sealed_requires_split", True, "")
    try:
        resolve_shard(ev_mod, base_cfg, cfg, "val", "id", "in_regime_test")
        ok("val_rejects_split", False, "未抛错")
    except ValueError:
        ok("val_rejects_split", True, "")

    # 5 输入面白名单（C6）：封存路径在 allow_sealed 下放行、实现期禁读
    sealed_path = _abs(base_cfg["inputs"]["sealed"]["g3_hard_negative_test"])
    ev_mod.assert_input_allowed(sealed_path, base_cfg, allow_sealed=True)
    ok("sealed_path_allowed_with_flag", True, "")
    try:
        ev_mod.assert_input_allowed(sealed_path, base_cfg, allow_sealed=False)
        ok("sealed_forbidden_in_implementation", False, "未抛错")
    except AssertionError:
        ok("sealed_forbidden_in_implementation", True, "")

    # 6 指标单源（03_07.compute_binary_metrics）
    m = map_mod.compute_binary_metrics([1, 1, 0, 0], [1, 0, 0, 1], scores=None)
    ok("metrics_binary", m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
       and abs(m["tpr"] - 0.5) < 1e-12 and abs(m["fpr"] - 0.5) < 1e-12, "")
    ok("metrics_no_auroc_key", "auroc" not in m, "标签型 ⇒ 不算 AUROC")
    m_neg = map_mod.compute_binary_metrics([0, 0, 0], [0, 0, 1], scores=None)
    ok("metrics_all_negative_tpr_na", m_neg["tpr"] is None and abs(m_neg["fpr"] - 1.0 / 3.0) < 1e-12,
       "G3 全良性 ⇒ TPR n/a")
    ok("metrics_f1_zero_division", m_neg["f1"] == 0.0, "分母 0 ⇒ 0.0（约定，非质量）")

    # 7 ACCESS_LOG 守卫
    try:
        require_access_note(cfg, "sealed", "")
        ok("sealed_without_note_rejected", False, "未抛 SystemExit")
    except SystemExit:
        ok("sealed_without_note_rejected", True, "")
    ok("sealed_with_note_accepted",
       require_access_note(cfg, "sealed", "x" * int(cfg["access"]["note_min_chars"])) is not None, "")
    ok("val_needs_no_note", require_access_note(cfg, "val", "") is None, "")

    # 8 CSV 写端（LF、DictReader 记录数、拒绝覆盖）
    tmp_dir = os.path.join(ROOT, "logs", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "05_10_selfcheck_rows_%s.csv" % timestamp())
    probe_rows = [{"sample_id": "S1", "text_sha256": "0" * 64, "pred": 1, "raw_type": "logits",
                   "raw": '{"logits": [1.0, 2.0]}', "score": "", "latency_ms": "1.2345",
                   "label": 1, "n_content_tokens": 5, "truncated": 0, "unparsed": 0}]
    write_csv_rows(tmp_path, cfg["outputs"]["per_sample_fields"], probe_rows)
    raw = open(tmp_path, "rb").read()
    ok("csv_lf_only", raw.count(b"\r") == 0, "CR=%d" % raw.count(b"\r"))
    ok("csv_records_dictreader", count_records(tmp_path) == 1, "")
    ok("csv_utf8_decodable", raw.decode("utf-8") is not None, "")
    try:
        write_csv_rows(tmp_path, cfg["outputs"]["per_sample_fields"], probe_rows)
        ok("csv_refuses_overwrite", False, "未抛 FileExistsError")
    except FileExistsError:
        ok("csv_refuses_overwrite", True, "")
    os.remove(tmp_path)

    # 9 报告必写段（渲染合成 ctx 后逐条断言）
    ctx = {
        "ts": "20260101-000000", "config_path": "configs/eval/external_baseline.yaml",
        "source": "sealed", "shards": ["id_test", "ood_test", "g3_hard_negative_test"],
        "kind": "judgment", "access_note": "self-check 合成 note（≥20 字符占位）",
        "rows": [{"shard": "id_test", "n": 4, "n_pos": 2, "n_neg": 2, "tp": 1, "fp": 1,
                  "fn": 1, "tn": 1, "tpr": 0.5, "fpr": 0.5, "f1": 0.5,
                  "latency_mean_ms": 1.0, "latency_median_ms": 1.0, "latency_p95_ms": 1.0,
                  "mem_load_peak_mib": 100.0, "mem_infer_peak_mib": 200.0,
                  "n_truncated": 1, "roundtrip_guard_fired": 1, "n_unparsed": 0}],
        "paths": {"report": "r.md", "metric_csv": "m.csv", "per_sample": ["a.csv"], "log": "l.log"},
    }
    md = render_report(cfg, ctx)
    for i, decl in enumerate(cfg["report"]["must_include"]):
        ok("report_decl_%02d" % (i + 1), decl in md, decl[:24])
    ok("report_na_not_zero",
       "| `id_test` | 4 | 2 | 2 | 1 | 1 | 1 | 1 | **0.5000** | **0.5000** | 0.5000 | n/a | n/a |" in md, "")

    # 10 接口契约静态面
    ok("cli_last_line_contract", "REPORT_MD=" in src and "SELFCHECK_RESULT=" in src
       and "DRYRUN_RESULT=" in src, "")
    #     `.train(` 检查要求**标识符**在前（`"…".train(` 之类字符串字面量不算命中）。
    #     本函数自身源码即 `src` ⇒ 正则须能区分「真实调用」与「检查器自带的字面量」。
    ok("no_train_call", re.search(r"[A-Za-z_][A-Za-z0-9_]*\.train\(", src) is None, "")
    #     硬编码报告目录检查：`src` 含本检查自身 ⇒ 检出模式用字符拼接构造（避免自指），
    #     并排除检查器所在行；其余任何行出现带引号的 reports 目录前缀即失败。
    #     docstring 的 CLI 示例无引号包裹，属文档而非硬编码，不算命中。
    _dq, _sq = chr(34), chr(39)
    _pats = (_dq + "reports" + "/", _sq + "reports" + "/")
    _bad_dir = [ln for ln in src.splitlines()
                if "no_hardcoded_reports_dir" not in ln
                and any(p in ln for p in _pats)]
    ok("no_hardcoded_reports_dir", not _bad_dir, "; ".join(_bad_dir[:2]))
    ok("outputs_ext_not_v2", cfg["outputs"]["outputs_subdir"] == "outputs_ext", "")
    #     `import torch` 只认**行首 import 语句**（注释／docstring 里的 "import torch" 不算），
    #     且 `CUBLAS_WORKSPACE_CONFIG` 设定必须出现在该 import **之前**（`AGENTS.md` §3）。
    _imp = re.search(r"(?m)^[ \t]*import torch\b", src)
    ok("determinism_before_torch_import",
       'os.environ["CUBLAS_WORKSPACE_CONFIG"]' in src and _imp is not None
       and src.find('os.environ["CUBLAS_WORKSPACE_CONFIG"]') < _imp.start(), "")

    n_pass = sum(1 for _n, c, _d in checks if c)
    for name, cond, detail in checks:
        print("SELFCHECK_ITEM %-38s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    verdict = "PASS" if n_pass == len(checks) else "FAIL"
    print("SELFCHECK_RESULT=%s n=%d/%d" % (verdict, n_pass, len(checks)))
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="scripts/eval/05_10_eval_external_baseline.py",
        description="W6.12 外部检测器基线 B6 评测（④ B 档；只评测不训练）")
    p.add_argument("--model", default=None, help="B6 本地权重目录（必填；缺省即报错，不补默认值）")
    p.add_argument("--kind", choices=["judgment", "score"], default=None,
                   help="判定口径分支（必填；本窗口预注册 = judgment，score 未激活）")
    p.add_argument("--source", choices=["val", "sealed"], default=None, help="输入来源（必填）")
    p.add_argument("--split", choices=["in_regime_test", "g3_hard_negative_test"], default=None,
                   help="`--source sealed` 必填；`val` 须省略")
    p.add_argument("--regime", choices=["id", "ood"], default=None,
                   help="`in_regime_test`／`val` 必填；`g3_hard_negative_test` 须省略")
    p.add_argument("--config", default="configs/eval/external_baseline.yaml")
    p.add_argument("--report-dir", dest="report_dir", default=None)
    p.add_argument("--outputs-dir", dest="outputs_dir", default=None)
    p.add_argument("--access-log-note", dest="access_log_note", default=None)
    p.add_argument("--limit", type=int, default=None, help="只取分片前 N 条（冒烟用；正式跑不给）")
    p.add_argument("--self-check", dest="self_check", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def main(argv=None):
    init_streams()
    args = build_parser().parse_args(argv)
    cfg_path = _abs(args.config)
    if not os.path.isfile(cfg_path):
        print("CONFIG_MISSING %s" % cfg_path)
        return 2
    cfg = load_config(cfg_path)
    if args.self_check:
        return self_check(cfg, args)

    # C3：参数缺失 ⇒ 抛错，不得补默认值
    for name, val in (("--model", args.model), ("--kind", args.kind), ("--source", args.source)):
        if not val:
            print("ARG_REQUIRED %s 必填（C3：不补默认值）" % name)
            return 2
    # 判定口径分支**只许择一**（`PREREG_v1.10.md` §1.2）：score 分支未激活 ⇒ 显式拒绝
    if args.kind != cfg["mapping"]["kind"]:
        print("KIND_NOT_PREREGISTERED：--kind=%r 未激活。本窗口预注册分支为 %r"
              "（B6 落地为 argmax 标签型）；启用 %r 分支属新增口径，须新建 PREREG 版本"
              % (args.kind, cfg["mapping"]["kind"], args.kind))
        return 2

    ev_mod = load_module("ev_05_01", cfg["modules"]["eval_engine"])
    base_cfg = load_config(_abs(cfg["base_eval_config"]))
    try:
        path, expect, shard, regime = resolve_shard(ev_mod, base_cfg, cfg,
                                                    args.source, args.regime, args.split)
    except ValueError as exc:
        print("ARG_REQUIRED %s" % exc)
        return 2
    note = require_access_note(cfg, args.source, args.access_log_note)
    model_path = _abs(args.model)
    if not os.path.isdir(model_path):
        print("MODEL_MISSING %s（B6 权重目录不存在；项目外盘位 D:/CodeObfPI_hf_cache/…）" % model_path)
        return 2

    if args.dry_run:
        # 只看计划（不读封存内容）
        print("DRYRUN_MODEL %s (%s)" % (model_path, cfg["model"]["architectures"]))
        print("DRYRUN_SHARD source=%s split=%s regime=%s shard=%s path=%s expect_records=%d"
              % (args.source, args.split or "-", regime or "-", shard, _rel(path), int(expect)))
        print("DRYRUN_KIND %s（LABEL_0=%s/LABEL_1=%s；AUROC/Span-F1=n/a）"
              % (args.kind, cfg["mapping"]["label_0"], cfg["mapping"]["label_1"]))
        print("DRYRUN_BATCH %d（延迟为批际摊到样本）" % int(cfg["latency"]["batch_size"]))
        n = int(args.limit) if args.limit else int(expect)
        print("DRYRUN_PLAN samples=%d report_dir=%s outputs_dir=%s"
              % (n, args.report_dir or "<未给>", args.outputs_dir or "<未给>"))
        print("DRYRUN_RESULT=PASS n=%d" % n)
        return 0

    if not args.report_dir or not args.outputs_dir:
        print("ARG_REQUIRED --report-dir 与 --outputs-dir 必填（C10）")
        return 2
    report_dir = _abs(args.report_dir)
    outputs_dir = _abs(args.outputs_dir)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    ts = timestamp()
    logger, log_path = flagged_logger(ts, resolve_log_dir(args))
    logger("START W6.12 B6 external baseline shard=%s source=%s kind=%s limit=%s"
           % (shard, args.source, args.kind, args.limit))
    logger("CONFIG %s" % _rel(cfg_path))
    logger("ACCESS_NOTE %s" % (note or "-"))

    pp_mod = load_module("pp_03_06", cfg["modules"]["preprocess"])
    map_mod = load_module("map_03_07", cfg["modules"]["mapping"])
    torch, _transformers, tok, model, load_meta = load_b6(cfg, model_path, logger)

    rows, metric_row, meta = evaluate_shard(
        cfg, args, ev_mod, pp_mod, map_mod, torch, tok, model, base_cfg,
        path, expect, shard, regime, note, load_meta["load_peak_bytes"], logger)

    per_sample_path = os.path.join(outputs_dir, "b6_%s_v1.0_%s.csv" % (shard, ts))
    metric_csv_path = os.path.join(report_dir, "external_baseline_v1.0_%s.csv" % ts)
    report_path = os.path.join(report_dir, "external_baseline_v1.0_%s.md" % ts)
    write_csv_rows(per_sample_path, cfg["outputs"]["per_sample_fields"], rows)
    got = count_records(per_sample_path)
    if got != len(rows):
        raise AssertionError("逐样本 CSV 记录数断言失败：%s got=%d expected=%d"
                             % (per_sample_path, got, len(rows)))
    write_csv_rows(metric_csv_path, cfg["outputs"]["metric_fields"], [metric_row])
    if count_records(metric_csv_path) != 1:
        raise AssertionError("指标 CSV 记录数 != 1：%s" % metric_csv_path)
    logger("OUTPUTS per_sample=%s records=%d metric_csv=%s"
           % (_rel(per_sample_path), got, _rel(metric_csv_path)))

    ctx = {
        "ts": ts, "config_path": _rel(cfg_path), "source": args.source,
        "shards": [shard], "kind": args.kind, "access_note": note,
        "rows": [metric_row],
        "paths": {"report": _rel(report_path), "metric_csv": _rel(metric_csv_path),
                  "per_sample": [_rel(per_sample_path)], "log": _rel(log_path)},
    }
    if os.path.exists(report_path):
        raise FileExistsError("C12 只追加：报告已存在 %s" % report_path)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_report(cfg, ctx))
    logger("DONE shard=%s records=%d tpr=%s fpr=%s f1=%s"
           % (shard, metric_row["n"], fmt_metric(metric_row["tpr"]),
              fmt_metric(metric_row["fpr"]), fmt_metric(metric_row["f1"])))
    print("REPORT_MD=%s" % _rel(report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

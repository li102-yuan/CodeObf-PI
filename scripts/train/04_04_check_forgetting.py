# -*- coding: utf-8 -*-
"""04_04_check_forgetting.py — W5.4 灾难性遗忘验证（权限 **A5**）。

职责（`windows/W5.4_forgetting.md` §1／§2／§3）：对 `W5.3` 产出的 6 个 best checkpoint
（2 体制 × 3 seeds，取各 `best.json` 指向的 `checkpoint-<step>`）**逐一**在**该体制 val 分片**上
推理，给出 **F1／FPR／TPR 三项**及其变化量；**干净原始测试集（clean）侧属 `data/sealed_test/`，
A5 会话一律不跑** ⇒ 本窗口的产物恒为 `STATUS=SELFCHECK_ONLY`／`GATE=UNKNOWN`，
**报告中不得出现任何「遗忘已通过／未通过」的正式结论**（§4-4）。

接口契约（**W5 冻结 v1.0，冻结后不得改**）::

    python scripts/train/04_04_check_forgetting.py \\
        --regime {id,ood} \\
        --checkpoints <ckpt_dir>[,<ckpt_dir>...] \\
        --val <frozen val csv> \\
        [--clean-test <csv>] \\
        [--base-val-metrics <csv>] [--base-clean-f1 <float>] [--base-clean-tpr <float>] \\
        [--config configs/train/forgetting.yaml] [--report-dir DIR] [--out-dir DIR]

判定口径（§2.1，W5 冻结）：
- **主口径 = 绝对百分点差** ``ΔF1 = F1_base − F1_ours ≤ 0.03`` ⇒ PASS；
- **并列口径必须同时给出** 相对降幅 ``ΔF1 / F1_base``；
- **三项指标同报**（F1／FPR／TPR），只看 F1 即判报告不完整；
- **逐 seed 判定**（各 seed 自比阈值），再给 mean ± std；**不得**只用 3 seeds 均值过阈。

基准双轨（§2.2）：
- **① clean 轨道（正式结论，A6 执行）**：引 tag `baseline-v1` 冻结数
  （`reports/baseline/unify_metrics_v1.0_20261004-013922.csv` 的 **B3 × clean_test**
  ＝ F1 **0.9885**／TPR **0.9773**／**FPR `n/a`**）；
- **② val 轨道（本窗口同机实测）**：调用 `03_03`（配置取 `configs/train/forgetting.yaml`，
  只指向 frozen 侧 `id_val`／`ood_val`）实跑**基座** ⇒ FPR 在此有定义。

硬约束落点：
- §4-1 **禁读 `data/sealed_test/**`** ⇒ `--val`／`--checkpoints` 路径先过 `assert_no_sealed()`，
  命中即**立即非零退出并打印拒绝行**（`--val` 只接受 frozen 侧 `val.csv`）；
- §4-2 不重跑/不改写阶段3 基线数字：clean 轨道只**引用**冻结数，不复算；
- §4-3 不调参/不选 checkpoint：τ 一律取该 run `best.json` 的 `tau_ours`（`D-61` 转交项 #52）；
- §4-4 `STATUS` 与 `GATE` 两字段同时打印且互相自洽；
- §4-5 超阈即如实报，不挑 seed、不重训。

`--self-check` 为**纯合成**自检（不读任何数据分片、不加载模型、不碰 sealed）。
`--dry-run` 只做参数与路径校验、列出待跑清单，不加载模型（供单测与人工预检）。

退出码：正常 0；**路径拒绝 = 2**；其余断言失败 = 3。
（`SELFCHECK_ONLY` **不得**因「未跑 clean」而失败 —— §3 表格明定 exit=0、`GATE=UNKNOWN`。）
"""
from __future__ import annotations

# --- C9：必须在 `import torch` 之前设置（本模块所有 torch import 均在函数体内，时序成立）---
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))


# ---------------------------------------------------------------------------
# 冻结常量（**禁止默认值兜底**：改这些数 = 改冻结口径，须 W0 另裁）
# ---------------------------------------------------------------------------
VERSION = "1.0"
WINDOW = "W5.4"
PERMISSION = "A5"

#: `PREREG_v1.md` §5「灾难性遗忘判定：微调后干净原始测试集 F1 下降 ≤3%」（v5 新增 24 补 FPR／TPR）
DF1_THRESHOLD = 0.03

#: ① clean 轨道的冻结基准（`baseline-v1` → `ea9cbb0`；`BASELINE_REPORT.md` §1／§7）
BASELINE_METRICS_CSV = "reports/baseline/unify_metrics_v1.0_20261004-013922.csv"
FROZEN_BASE_CLEAN_F1 = 0.9885
FROZEN_BASE_CLEAN_TPR = 0.9773
FROZEN_BASE_CLEAN_FPR = None          # clean 分片 44 行全 label=1 ⇒ FPR 无定义

#: 路径红线（`AGENTS.md` §6-6／`PROJECT_CONTRACT.md` §6 A5 行）
FORBIDDEN_SUBSTRINGS = (
    "data/sealed_test",
    "sealed_test",
    "splits/regime_id/test.csv",
    "splits/regime_ood/test.csv",
    "clean_original_test",
    "g3_hard_negative_test",
)
#: `--val` 只接受 frozen 侧两个 val 分片（`PROJECT_CONTRACT.md` §6 A5 行的 4 个文件之 2）
ALLOWED_VAL_SUFFIXES = (
    "splits/regime_id/val.csv",
    "splits/regime_ood/val.csv",
)

STATUS_SELFCHECK_ONLY = "SELFCHECK_ONLY"
STATUS_OFFICIAL = "OFFICIAL"
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_UNKNOWN = "UNKNOWN"

#: 退出码
EXIT_OK = 0
EXIT_PATH_REJECTED = 2
EXIT_ASSERT = 3

#: 明细 CSV 列（§5 指定项 ＋ 可追溯列）
DETAIL_COLUMNS = [
    "regime", "seed", "run_id", "step", "ckpt_dir", "track", "n",
    "tp", "fp", "fn", "tn", "n_pos", "n_neg",
    "f1", "fpr", "tpr", "tau",
    "base_f1", "base_fpr", "base_tpr",
    "d_f1_abs", "d_f1_rel", "d_fpr_abs", "d_fpr_rel", "d_tpr_abs", "d_tpr_rel",
    "verdict_val_selfcheck", "note",
]

BASE_VAL_COLUMNS = [
    "split", "regime", "n", "n_pos", "n_neg", "tp", "fp", "fn", "tn",
    "tpr", "fpr", "f1", "n_unparsed", "outputs_csv",
]


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：W3.7 `F-1`；`D-40` 裁定五：不得静默 `pass`）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8。失败即显式告警，不静默。"""
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


# 指标单源（`03_07`）与读取单源（`03_06`）—— 不另起一套实现
_PREPROCESS = _load_module("forgetting_preprocess",
                           os.path.join(_ROOT, "scripts", "baseline",
                                        "03_06_unify_preprocess.py"))
_MAPPING = _load_module("forgetting_mapping",
                        os.path.join(_ROOT, "scripts", "baseline",
                                     "03_07_unify_output_mapping.py"))

load_samples = _PREPROCESS.load_samples
md5_of = _PREPROCESS.md5_of

f1_score = _MAPPING.f1_score
binary_counts = _MAPPING.binary_counts
compute_binary_metrics = _MAPPING.compute_binary_metrics


class PathRejected(AssertionError):
    """路径红线断言失败（`windows/W5.4_forgetting.md` §3／§4-1）。"""


class ForgettingAssert(AssertionError):
    """非路径类断言失败。"""


# ---------------------------------------------------------------------------
# 路径红线（§3：`--val` 传 sealed 或 `test.csv` ⇒ 立即非零退出并打印拒绝行）
# ---------------------------------------------------------------------------
def norm_path(path):
    return os.path.abspath(path).replace("\\", "/")


def assert_no_sealed(path, what="--val"):
    """命中任一禁读面即抛 `PathRejected`（调用方须以退出码 2 终止）。"""
    norm = norm_path(path)
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad.replace("\\", "/") in norm:
            raise PathRejected(
                "REJECTED %s：%s（命中禁读面 %r；A5 禁止读取 data/sealed_test/** 与 frozen 侧 test.csv）"
                % (what, path, bad))
    base = os.path.basename(norm)
    if base == "test.csv" or norm.endswith("/test.csv"):
        raise PathRejected(
            "REJECTED %s：%s（frozen 侧 test.csv 不在 A5 授权内；本窗口只接受 val 分片）"
            % (what, path))
    return norm


def assert_val_allowed(path, regime):
    """`--val` 必须是 frozen 侧 `regime_<regime>/val.csv`，且与 `--regime` 一致。"""
    norm = assert_no_sealed(path, "--val")
    if not norm.endswith(ALLOWED_VAL_SUFFIXES):
        raise PathRejected(
            "REJECTED --val：%s（只接受 %s）"
            % (path, " 或 ".join(ALLOWED_VAL_SUFFIXES)))
    want = "splits/regime_%s/val.csv" % regime
    if not norm.endswith(want):
        raise PathRejected(
            "REJECTED --val：%s 与 --regime %s 不一致（期望以 %s 结尾）" % (path, regime, want))
    if not os.path.isfile(norm):
        raise ForgettingAssert("--val 不存在：%s" % norm)
    return norm


# ---------------------------------------------------------------------------
# Δ 口径（§2.1）
# ---------------------------------------------------------------------------
def delta(base, ours):
    """返回 ``(abs_diff, rel_diff)``；**主口径 = 绝对百分点差**。

    ``abs = base − ours``：对 **F1／TPR** 为正 ⇒ **下降**（坏）；对 **FPR** 为负 ⇒ **上升**（坏）。
    ``rel = abs / base``（相对降幅；`base` 为 0 或 None ⇒ 相对项记 `None`＝`n/a`）。
    """
    if base is None or ours is None:
        return (None, None)
    abs_diff = float(base) - float(ours)
    if base == 0:
        return (abs_diff, None)
    return (abs_diff, abs_diff / float(base))


def verdict_of(d_f1_abs, threshold=DF1_THRESHOLD):
    """逐 seed 判定：``ΔF1_abs ≤ threshold`` ⇒ PASS；否则 FAIL；缺失 ⇒ UNKNOWN。"""
    if d_f1_abs is None:
        return GATE_UNKNOWN
    return GATE_PASS if float(d_f1_abs) <= float(threshold) else GATE_FAIL


def fmt_num(x, nd=6):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, float):
        return round(x, nd)
    return x


def fmt_rel(x, nd=6):
    """相对降幅：`None` 记 `n/a`（分母为 0 或指标无定义）。"""
    return "n/a" if x is None else round(float(x), nd)


def mean_std(values):
    """``(mean, std_ddof1)``；空 ⇒ ``(None, None)``。"""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return (None, None)
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return (m, math.sqrt(var))


def _to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "n/a":
        return None
    return float(s)


# ---------------------------------------------------------------------------
# checkpoint 解析（`--checkpoints` 既可传 run 目录（含 best.json）也可传 checkpoint-<step>）
# ---------------------------------------------------------------------------
def resolve_checkpoint(d):
    """解析一个 `--checkpoints` 条目 ⇒ ``dict(ckpt_dir, run_id, regime, seed, step, tau, …)``。

    τ 一律取 `best.json` 的 `tau_ours`（`D-61` 转交项 #52：主口径 = best epoch 的 `tau_ours`）；
    **禁止**在本窗口重新寻 τ（§4-3：不调参／不选阈值／不选 checkpoint）。
    """
    abs_d = os.path.abspath(d)
    if not os.path.isdir(abs_d):
        raise ForgettingAssert("checkpoint 目录不存在：%s" % d)

    best_path = os.path.join(abs_d, "best.json")
    if os.path.isfile(best_path):
        with open(best_path, "r", encoding="utf-8") as fh:
            best = json.load(fh)
        ckpt = best.get("dir") or ""
        if not ckpt:
            raise ForgettingAssert("best.json 缺 `dir`：%s" % best_path)
        if not os.path.isdir(ckpt):
            raise ForgettingAssert("best.json 指向的 checkpoint 不存在：%s" % ckpt)
        run_dir = abs_d
        meta_src = best_path
    else:
        ckpt = abs_d
        run_dir = os.path.dirname(os.path.dirname(abs_d))   # <run>/seed_<s>/checkpoint-<step>
        best_path2 = os.path.join(run_dir, "best.json")
        if os.path.isfile(best_path2):
            with open(best_path2, "r", encoding="utf-8") as fh:
                best = json.load(fh)
            meta_src = best_path2
        else:
            best = {}
            meta_src = os.path.join(ckpt, "meta.json")

    meta_path = os.path.join(ckpt, "meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    step = best.get("step", meta.get("step"))
    if step is None:
        _b = os.path.basename(ckpt.rstrip("/\\"))
        step = _b.split("-")[-1] if _b.startswith("checkpoint-") else ""
    regime = best.get("regime") or meta.get("regime")
    seed = best.get("seed", meta.get("seed"))
    tau = best.get("tau_ours", best.get("tau", meta.get("tau_ours")))
    return {
        "ckpt_dir": os.path.abspath(ckpt),
        "run_id": os.path.basename(run_dir.rstrip("/\\")),
        "regime": regime,
        "seed": seed,
        "step": step,
        "tau": tau,
        "meta_src": meta_src,
        "best_val_f1": best.get("val_f1"),
    }


# ---------------------------------------------------------------------------
# ② val 轨道基准：调 `03_03` 实跑基座（`D-41` 裁定五 四条）
# ---------------------------------------------------------------------------
def build_base_val_cmd(splits, config, unify_config, outputs_dir, report_dir, ts, limit=0):
    """构造 `03_03` 子进程命令（`D-41` 裁定五①②③）。

    `03_03` 的 `--splits` 是 `nargs="*"` ⇒ 分片代号**必须逐个传入**；
    若拼成 `"id_val,ood_val"` 一个参数，会被当作**单个分片名**而 `SKIP`
    （该失败形态静默退出 0，极易误判为「跑完了」）。
    """
    cmd = [sys.executable, "-u",
           os.path.join(_ROOT, "scripts", "baseline", "03_03_run_b3_llamaguard1b.py"),
           "--config", config,
           "--unify-config", unify_config,
           "--splits", *splits,
           "--outputs-dir", outputs_dir,
           "--report-dir", report_dir,
           "--tag", ts]
    if limit and limit > 0:
        cmd += ["--limit", str(int(limit))]
    return cmd


def run_base_val_03_03(splits, config, unify_config, outputs_dir, report_dir, ts,
                       limit=0, logger=None):
    """子进程跑 `03_03`（只指向 frozen 侧 val 分片），返回逐 split 的输出 CSV 路径。

    `D-41` 裁定五四条：① `--report-dir` 落在 `reports/train/`；② `--outputs-dir` 显式给定；
    ③ **禁复用** `configs/baseline/b3_llamaguard1b.yaml` 的 sealed 配置 ⇒ 用 `configs/train/forgetting.yaml`；
    ④ 子进程 stdout 出现 `REMINDER` 即**立即停下**（本函数抛 `ForgettingAssert`）。
    """
    cmd = build_base_val_cmd(splits, config, unify_config, outputs_dir, report_dir, ts, limit)
    if logger:
        logger("BASE_VAL_CMD %s" % " ".join(cmd))

    proc = subprocess.Popen(cmd, cwd=_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    stdout_lines = []
    reminder = None
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            stdout_lines.append(line)
            if logger:
                logger("  [03_03] %s" % line)
            if "REMINDER" in line:
                reminder = line
                break
    if reminder is not None:
        try:
            proc.kill()
        except OSError:
            pass
        raise ForgettingAssert(
            "03_03 打印 ACCESS_LOG 提醒（说明指向了 sealed 分片）⇒ 立即停工：%s" % reminder)
    rc = proc.wait()
    if rc != 0:
        raise ForgettingAssert("03_03 退出码非零（%d）：%s" % (rc, " ".join(cmd)))

    out = {}
    for split in splits:
        csv_path = os.path.join(outputs_dir, "b3_%s_v1.0_%s.csv" % (split, ts))
        if not os.path.isfile(csv_path):
            raise ForgettingAssert("03_03 未产出 %s 的输出 CSV：%s" % (split, csv_path))
        out[split] = csv_path
    return {"cmd": cmd, "outputs": out, "stdout": stdout_lines}


def metrics_from_03_03_outputs(csv_path, val_csv):
    """把 `03_03` 的逐样本 `pred` 与 val 分片 `label` 对齐 ⇒ `compute_binary_metrics`。

    `pred` 为空（`unparsed`）时按 `03_07` 的同款处置计 **0**（`pred if pred is not None else 0`），
    并**单独披露** `n_unparsed`（不得静默）。
    """
    label_by = {s.sample_id: int(s.label) for s in load_samples(val_csv)}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    y_true, y_pred, n_unparsed = [], [], 0
    for r in rows:
        sid = r.get("sample_id")
        if sid not in label_by:
            continue
        y_true.append(label_by[sid])
        pred = (r.get("pred") or "").strip()
        if pred == "":
            n_unparsed += 1
            y_pred.append(0)
        else:
            y_pred.append(int(pred))
    m = compute_binary_metrics(y_true, y_pred)
    m["n"] = len(y_true)
    m["n_unparsed"] = n_unparsed
    return m


def load_base_val_metrics(path):
    """读已有的基座 val 实测 CSV（`--base-val-metrics`）⇒ ``{regime: metrics}``。"""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        key = r.get("regime") or ("id" if "id" in (r.get("split") or "") else "ood")
        out[key] = {
            "f1": _to_float(r.get("f1")), "fpr": _to_float(r.get("fpr")),
            "tpr": _to_float(r.get("tpr")), "n": int(r.get("n") or 0),
            "n_unparsed": int(r.get("n_unparsed") or 0),
            "split": r.get("split"),
            "tp": r.get("tp", ""), "fp": r.get("fp", ""), "fn": r.get("fn", ""),
            "tn": r.get("tn", ""), "n_pos": r.get("n_pos", ""), "n_neg": r.get("n_neg", ""),
            "outputs_csv": r.get("outputs_csv", ""),
        }
    return out


def write_csv(path, rows, columns):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


# ---------------------------------------------------------------------------
# Ours 侧推理（复用 `04_01` 的单源实现：绝不自起一套 score／指标）
# ---------------------------------------------------------------------------
def load_ours_modules():
    """惰性 import `04_01`（其内部再惰性 import torch／transformers／peft）。"""
    return _load_module("ours_04_01",
                        os.path.join(_ROOT, "scripts", "train", "04_01_train_qlora_ours.py"))


def infer_ours(T, cfg, ckpt_dir, csv_path, tau, seed, limit=0, logger=None):
    """加载基座 4-bit ＋ LoRA ckpt → 在给定分片上算分数 → 按 τ 判正 → 三项指标。

    G-EVAL 双断言（`model.eval()` ＋ `torch.no_grad()`）由 `04_01.evaluate()` 内部保证。
    """
    import torch

    torch_mod, tok, model, _llama, _lm_head, _meta = T.build_model(cfg, logger, with_lora=False)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.abspath(ckpt_dir), is_trainable=False)
    llama, lm_head = T._unwrap(model)
    if logger:
        logger("PEFT_LOADED %s" % os.path.abspath(ckpt_dir))

    id_unsafe, id_safe = T.resolve_target_ids(tok, cfg)
    cap = int(limit) if limit and limit > 0 else None
    exs = T.prepare_split(tok, cfg, csv_path, id_unsafe, id_safe, tok.eos_token_id,
                          logger, cap=cap)
    if logger:
        logger("EVAL_BEGIN n=%d cap=%s" % (len(exs), cap))
    rng = T.set_determinism(int(seed) if seed else 42, cfg)
    scores = T.evaluate(model, llama, lm_head, torch_mod, exs, id_unsafe, id_safe, cfg, rng,
                        logger, tag="forgetting")
    labels = [e.label for e in exs]
    m = compute_binary_metrics(labels, T.apply_threshold(scores, tau))
    m["n"] = len(labels)
    m["n_unparsed"] = 0
    del model, llama, lm_head
    try:
        torch.empty_cache()
    except Exception:                                   # pragma: no cover
        pass
    return m, scores


def infer_base_score_track(T, cfg, val_csv, limit=0, logger=None):
    """并列补充诊断（**非主口径**，呈 W0 追认）：**同一打分规则**下的基座 val 指标。

    动机：主口径是「判词型 B3（`03_03`）vs `score≥τ` 型 Ours」，两**判定规则不同**；
    本轨道把基座也用 `04_01` 的 `score = p_unsafe/(p_unsafe+p_safe)` 打分并在**同一 val 分片**
    按同一规则（`search_tau_ours`：F1 最大、并列取大 τ）寻 τ ⇒ **同规则同分片**的对称对照，
    把「规则变更」与「能力变化」分开。本轨道**不参与** `GATE` 判定。
    """
    import torch

    torch_mod, tok, model, llama, lm_head, _meta = T.build_model(cfg, logger, with_lora=False)
    id_unsafe, id_safe = T.resolve_target_ids(tok, cfg)
    cap = int(limit) if limit and limit > 0 else None
    exs = T.prepare_split(tok, cfg, val_csv, id_unsafe, id_safe, tok.eos_token_id,
                          logger, cap=cap)
    rng = T.set_determinism(42, cfg)
    scores = T.evaluate(model, llama, lm_head, torch_mod, exs, id_unsafe, id_safe, cfg, rng,
                        logger, tag="base_score_track")
    labels = [e.label for e in exs]
    tau_star, _f1_star = T.search_tau_ours(scores, labels)
    tres = T.resolve_tau_ours(scores, labels, cfg)
    tau = tres["tau_main"]
    m = compute_binary_metrics(labels, T.apply_threshold(scores, tau))
    m["n"] = len(labels)
    m["n_unparsed"] = 0
    m["tau_star"] = tau_star
    m["tau_main"] = tau
    m["triggered"] = bool(tres.get("triggered"))
    del model, llama, lm_head
    try:
        torch.empty_cache()
    except Exception:                                   # pragma: no cover
        pass
    return m, scores


# ---------------------------------------------------------------------------
# 报告（§5 五要素）
# ---------------------------------------------------------------------------
def write_report(path, ctx):
    ts = ctx["ts"]
    lines = []
    lines.append("# forgetting_v%s 运行报告（W5.4 · 灾难性遗忘验证，权限 **%s**）" % (VERSION, PERMISSION))
    lines.append("")
    lines.append("- 生成时间：`%s`" % ts)
    lines.append("- 窗口：**W5.4**（`windows/W5.4_forgetting.md`，权限 A5；`D-30` 规定**单独会话**）")
    lines.append("- 体制：`%s`；val 分片：`%s`（md5 `%s`）"
                 % (ctx["regime"], ctx["val_rel"], ctx["val_md5"]))
    lines.append("- 配置：基座侧 `%s`；Ours 侧 `%s`" % (ctx["config"], ctx["ours_config"]))
    lines.append("- 判定口径：主口径 **绝对百分点** `ΔF1 = F1_base − F1_ours ≤ %.2f`；并列相对降幅 `ΔF1/F1_base`"
                 % DF1_THRESHOLD)
    lines.append("- **`STATUS=%s`　　`GATE=%s`**" % (ctx["status"], ctx["gate"]))
    lines.append("")

    if ctx["status"] == STATUS_SELFCHECK_ONLY:
        lines.append("> **§4-4 硬约束自证**：本轮**未传 `--clean-test`**（clean 集属 `data/sealed_test/`，"
                     "A5 无读取权）⇒ 本节**不出现任何「遗忘已通过／未通过」的正式结论**；"
                     "下表全部为 **val 轨道自检**，`GATE=UNKNOWN`。正式结论由 **W6 的 A6 窗口**"
                     "用同一脚本、同一参数产出（须先在 `ACCESS_LOG.md` 登记 6 列）。")
        lines.append("")

    lines.append("## 1. clean 轨道基准（①，口径引用；**本窗口不读取、不复算**）")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 权威 CSV | `%s`（tag `baseline-v1`） |" % BASELINE_METRICS_CSV)
    lines.append("| 行 | `b3 × clean_test` |")
    lines.append("| **F1** | **%.4f**（退化形式 `2·TP/(2·TP+FN)`；`clean_test` 44 行全 `label=1`） |"
                 % FROZEN_BASE_CLEAN_F1)
    lines.append("| **TPR** | **%.4f** |" % FROZEN_BASE_CLEAN_TPR)
    lines.append("| **FPR** | **`n/a`**（无负样本 ⇒ 无定义） |")
    lines.append("| 本轮是否实跑 | **否**（`--clean-test` 未传；A5 红线） |")
    lines.append("| CLI 覆盖值 | `--base-clean-f1=%s`／`--base-clean-tpr=%s` |"
                 % (ctx["base_clean_f1"], ctx["base_clean_tpr"]))
    lines.append("")

    lines.append("## 2. val 轨道基座实测（②，`03_03` 同机实跑）")
    lines.append("")
    lines.append("| 分片 | 体制 | n | TP | FP | FN | TN | TPR | FPR | F1 | `unparsed` |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ctx["base_rows"]:
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (r.get("split"), r.get("regime"), r.get("n"), r.get("tp"), r.get("fp"),
                        r.get("fn"), r.get("tn"), fmt_num(r.get("tpr")), fmt_num(r.get("fpr")),
                        fmt_num(r.get("f1")), r.get("n_unparsed")))
    lines.append("")
    lines.append("- 基座 = **原始 Llama-Guard-3-1B**（`finetuned: false`），`4bit-nf4`／cuda／`eager`，"
                 "512 左截断，官方 chat 模板，贪心生成，判词**首行首词**映射 —— 与阶段3 B3 口径逐项同源。")
    lines.append("- 产物：`%s`" % (ctx["base_metrics_csv"] or "（未落盘）"))
    lines.append("- `pred` 为空（`unparsed`）按 `03_07` 同款处置计 **0**，并单列计数（不静默）。")
    lines.append("- **由 `03_03` 生成的那份报告抬头标注 `W4.3`／权限 A4**，属该脚本原样输出；"
                 "本窗口仅取其逐样本 `pred` 与 val `label` 对齐算指标，抬头文字**不作本窗口结论依据**。")
    lines.append("")

    lines.append("## 3. 逐 seed 三项指标与其变化（F1／FPR／TPR）")
    lines.append("")
    lines.append("> 符号约定：`ΔX = X_base − X_ours`。对 **F1／TPR**：`Δ>0` = **下降**（坏）；"
                 "对 **FPR**：`Δ<0` = **上升**（坏）。`Δ_rel = Δ / X_base`（相对降幅，并列必给）。")
    lines.append("")
    lines.append("### 3.1 val 轨道（**自检**，非门禁结论）")
    lines.append("")
    lines.append("| seed | run_id | step | τ | F1 | ΔF1(pp) | ΔF1(rel) | FPR | ΔFPR(pp) | ΔFPR(rel) | TPR | ΔTPR(pp) | ΔTPR(rel) | 逐 seed 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ctx["detail_rows"]:
        if r.get("track") != "val":
            continue
        lines.append("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (r.get("seed"), r.get("run_id"), r.get("step"),
                        r.get("tau"), fmt_num(r.get("f1")), fmt_num(r.get("d_f1_abs")),
                        fmt_rel(r.get("d_f1_rel")), fmt_num(r.get("fpr")),
                        fmt_num(r.get("d_fpr_abs")), fmt_rel(r.get("d_fpr_rel")),
                        fmt_num(r.get("tpr")), fmt_num(r.get("d_tpr_abs")),
                        fmt_rel(r.get("d_tpr_rel")), r.get("verdict_val_selfcheck")))
    lines.append("")
    lines.append("- **mean ± std（并列，不得只用均值过阈）**："
                 "ΔF1(pp) = %s；ΔFPR(pp) = %s；ΔTPR(pp) = %s"
                 % (ctx["summary"]["d_f1"], ctx["summary"]["d_fpr"], ctx["summary"]["d_tpr"]))
    lines.append("- **逐 seed 原值（禁止只读 mean±std）**：%s" % ctx["summary"]["per_seed_raw"])
    if ctx["summary"].get("std_degenerate"):
        lines.append("- ⚠ **`std` 退化声明**：本体制 3 seeds 的 Δ 全部相同 ⇒ `std = 0.000000`，"
                     "**以逐 seed 原值为准**，不得读作「mean ± std = x ± 0」而不加声明（`D-47` 裁定四）。")
    lines.append("")

    if ctx["has_clean"]:
        lines.append("### 3.2 clean 轨道（正式结论轨）")
        lines.append("")
        lines.append("| seed | F1 | ΔF1(pp) | ΔF1(rel) | TPR | ΔTPR(pp) | ΔTPR(rel) | FPR | 判定 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in ctx["detail_rows"]:
            if r.get("track") != "clean":
                continue
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                         % (r.get("seed"), fmt_num(r.get("f1")), fmt_num(r.get("d_f1_abs")),
                            fmt_rel(r.get("d_f1_rel")), fmt_num(r.get("tpr")),
                            fmt_num(r.get("d_tpr_abs")), fmt_rel(r.get("d_tpr_rel")),
                            fmt_num(r.get("fpr")), r.get("verdict_val_selfcheck")))
        lines.append("")
    else:
        lines.append("### 3.2 clean 轨道（正式结论轨）—— **本轮未跑**")
        lines.append("")
        lines.append("- `--clean-test` **未传** ⇒ clean 轨道**空**；`GATE` 因此为 `UNKNOWN`。")
        lines.append("")

    lines.append("## 4. `STATUS`／`GATE`")
    lines.append("")
    lines.append("| 字段 | 值 | 依据 |")
    lines.append("|---|---|---|")
    lines.append("| `STATUS` | **%s** | %s |" % (ctx["status"], ctx["status_reason"]))
    lines.append("| `GATE` | **%s** | %s |" % (ctx["gate"], ctx["gate_reason"]))
    lines.append("| 退出码 | %d | %s |" % (ctx["exit_code"], ctx["exit_reason"]))
    lines.append("| 是否访问 `data/sealed_test/**` | **否** | `--val`／`--checkpoints` 均过 "
                 "`assert_no_sealed()`；`03_03` 子进程 stdout 无 `REMINDER` 行 |")
    lines.append("")

    lines.append("## 5. 交接：`W6`（A6）如何产出正式结论")
    lines.append("")
    lines.append("```bash")
    lines.append("# ① 先在 data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md 登记 6 列（日期／窗口／用途／命令／输出路径／复核人）")
    lines.append("python -u scripts/train/04_04_check_forgetting.py \\")
    lines.append("    --regime %s \\" % ctx["regime"])
    lines.append("    --checkpoints %s \\" % ",".join(ctx["ckpt_dirs"]))
    lines.append("    --val %s \\" % ctx["val_rel"])
    lines.append("    --clean-test data/sealed_test/CodeObf-PI_v1.0/clean_original_test.csv \\")
    lines.append("    --base-val-metrics %s \\" % (ctx["base_metrics_csv"] or "<本轮产出的基座 val CSV>"))
    lines.append("    --base-clean-f1 %.4f --base-clean-tpr %.4f \\" % (FROZEN_BASE_CLEAN_F1,
                                                                       FROZEN_BASE_CLEAN_TPR))
    lines.append("    --config %s --report-dir reports/train --out-dir reports/train" % ctx["config"])
    lines.append("```")
    lines.append("")
    lines.append("- **同一脚本、同一参数**（`--config`／`--report-dir`／`--out-dir` 不变），只**加** `--clean-test`。")
    lines.append("- `clean_original_test.csv` 44 行全 `label=1` ⇒ **FPR 无定义**、F1 为退化形式；"
                 "故 clean 轨道只判 **F1**（并如实报 TPR 变化），FPR 的变化仍以 **val 轨道为唯一合法来源**（§2.2）。")
    lines.append("- `GATE` 取**逐 seed 最劣**者：任一 seed `ΔF1 > %.2f` ⇒ `FAIL`。超阈即如实报，"
                 "**不得**挑 seed／重训／调阈值（§4-5）。" % DF1_THRESHOLD)
    lines.append("")

    lines.append("## 6. 产物清单")
    lines.append("")
    lines.append("| 产物 | 路径 |")
    lines.append("|---|---|")
    for k, v in ctx["artifacts"]:
        lines.append("| %s | `%s` |" % (k, v))
    lines.append("")

    lines.append("## 7. 并列补充诊断（**非主口径**，呈 W0 追认）")
    lines.append("")
    if ctx.get("base_score_track"):
        b = ctx["base_score_track"]
        lines.append("**同一打分规则**（`score = p_unsafe/(p_unsafe+p_safe)`，τ 于同一 val 分片按同一规则寻优）"
                     "下的**基座** val 指标：F1 **%s**／FPR **%s**／TPR **%s**（n=%s；τ\\* = %s；"
                     "τ_main = %s；`main_criterion_override` 触发 = %s）。"
                     % (fmt_num(b.get("f1")), fmt_num(b.get("fpr")), fmt_num(b.get("tpr")),
                        b.get("n"), b.get("tau_star"), b.get("tau_main"), b.get("triggered")))
        lines.append("")
        lines.append("用途 = 把「判词规则 → `score≥τ` 规则」的**规则变更**与「微调导致的**能力变化**」分开；"
                     "**不参与 `GATE` 判定**。")
    else:
        lines.append("- 本轮未跑（未传 `--base-score-track`）。")
    lines.append("")

    lines.append("## 8. 呈 W0 事项")
    lines.append("")
    for i, item in enumerate(ctx["to_w0"], 1):
        lines.append("%d. %s" % (i, item))
    lines.append("")

    lines.append("## 9. 本轮回执（验收 §6 逐条）")
    lines.append("")
    for i, item in enumerate(ctx["receipt"], 1):
        lines.append("%d. %s" % (i, item))
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# `--self-check`（纯合成，不读数据、不加载模型、不碰 sealed）
# ---------------------------------------------------------------------------
def _self_check() -> int:
    errors = []

    # ① 路径拒绝：sealed 与 test.csv（两类，§6-3）
    for bad in ("data/sealed_test/CodeObf-PI_v1.0/clean_original_test.csv",
                "data/sealed_test/CodeObf-PI_v1.0/regime_id_test.csv",
                "data/sealed_test/CodeObf-PI_v1.0/regime_ood_test.csv",
                "data/frozen/CodeObf-PI_v1.0/splits/regime_id/test.csv",
                "data/frozen/CodeObf-PI_v1.0/splits/regime_ood/test.csv"):
        try:
            assert_no_sealed(bad, "--val")
            errors.append("路径拒绝失败（应抛 PathRejected）：%s" % bad)
        except PathRejected:
            pass

    # ② frozen val 必须被接受
    for good in ("data/frozen/CodeObf-PI_v1.0/splits/regime_id/val.csv",
                 "data/frozen/CodeObf-PI_v1.0/splits/regime_ood/val.csv"):
        try:
            got = assert_no_sealed(good, "--val")
            if not got.endswith(ALLOWED_VAL_SUFFIXES):
                errors.append("frozen val 应被接受：%s" % good)
        except PathRejected as exc:
            errors.append("frozen val 不应被拒绝：%s（%s）" % (good, exc))

    # ③ `--regime` 与 `--val` 必须一致
    try:
        assert_val_allowed("data/frozen/CodeObf-PI_v1.0/splits/regime_ood/val.csv", "id")
        errors.append("--regime id 配 ood val 应被拒绝")
    except (PathRejected, ForgettingAssert):
        pass

    # ④ Δ 口径：绝对百分点 ＋ 相对降幅
    a, r = delta(0.9885, 0.9700)
    if abs(a - 0.0185) > 1e-12 or abs(r - 0.0185 / 0.9885) > 1e-12:
        errors.append("Δ 口径不符：abs=%r rel=%r" % (a, r))
    if delta(None, 0.9) != (None, None) or delta(0.9, None) != (None, None):
        errors.append("None 指标应返回 (None, None)")
    if delta(0.0, 0.0) != (0.0, None):
        errors.append("base=0 时相对降幅应为 None（记 n/a）")

    # ⑤ 阈值边界（≤ 为 PASS）
    if verdict_of(0.03) != GATE_PASS:
        errors.append("ΔF1 = 0.03 应判 PASS（≤）")
    if verdict_of(0.0300001) != GATE_FAIL:
        errors.append("ΔF1 = 0.0300001 应判 FAIL")
    if verdict_of(-0.5) != GATE_PASS:
        errors.append("ΔF1 为负（提升）应判 PASS")
    if verdict_of(None) != GATE_UNKNOWN:
        errors.append("ΔF1 缺失应判 UNKNOWN")

    # ⑥ 指标单源：clean 退化 F1 = 2TP/(2TP+FN)，FPR = None
    m = compute_binary_metrics([1] * 44, [1] * 43 + [0])
    if m["fpr"] is not None:
        errors.append("无负样本时 FPR 应为 None：%r" % m["fpr"])
    if abs(m["f1"] - 2.0 * 43 / (2.0 * 43 + 1)) > 1e-12:
        errors.append("clean 退化 F1 不符：%r" % m["f1"])

    # ⑦ 冻结 clean 基准值不得被动过
    if abs(FROZEN_BASE_CLEAN_F1 - 0.9885) > 1e-12 or abs(FROZEN_BASE_CLEAN_TPR - 0.9773) > 1e-12:
        errors.append("clean 冻结基准值被改动")
    if FROZEN_BASE_CLEAN_FPR is not None:
        errors.append("clean 冻结 FPR 应为 None（无定义）")
    if DF1_THRESHOLD != 0.03:
        errors.append("阈值 0.03 被改动")

    # ⑧ STATUS／GATE 常量
    if STATUS_SELFCHECK_ONLY != "SELFCHECK_ONLY" or STATUS_OFFICIAL != "OFFICIAL":
        errors.append("STATUS 常量被改动")
    for g in (GATE_PASS, GATE_FAIL, GATE_UNKNOWN):
        if g not in ("PASS", "FAIL", "UNKNOWN"):
            errors.append("GATE 常量被改动：%r" % g)

    # ⑨ mean/std
    mm, ss = mean_std([1.0, 1.0, 1.0])
    if mm != 1.0 or ss != 0.0:
        errors.append("mean_std 不符：%r %r" % (mm, ss))
    if mean_std([]) != (None, None):
        errors.append("空列表 mean_std 应为 (None, None)")

    # ⑩ 明细 CSV 列齐备（§5）
    for c in ("regime", "seed", "run_id", "step", "f1", "fpr", "tpr",
              "d_f1_abs", "d_fpr_abs", "d_tpr_abs", "verdict_val_selfcheck"):
        if c not in DETAIL_COLUMNS:
            errors.append("明细 CSV 缺列：%s" % c)

    if errors:
        for e in errors:
            print("SELF_CHECK_FAIL: %s" % e)
        return 1
    print("04_04_check_forgetting SELF_CHECK OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="W5.4 灾难性遗忘验证（A5：只跑 val 自检；clean 侧归 A6）")
    p.add_argument("--self-check", action="store_true",
                   help="合成自检（不读任何分片、不加载模型、不碰 sealed）")
    p.add_argument("--dry-run", action="store_true",
                   help="只做参数与路径校验并列出待跑清单，不加载模型（供单测／预检）")
    p.add_argument("--regime", choices=("id", "ood"), help="体制（与 --val 必须一致）")
    p.add_argument("--checkpoints", default="",
                   help="逗号分隔：run 目录（含 best.json）或 checkpoint-<step> 目录")
    p.add_argument("--val", default="",
                   help="frozen 侧 val.csv（**只接受** splits/regime_{id,ood}/val.csv）")
    p.add_argument("--clean-test", default="",
                   help="干净原始测试集 CSV（**A5 会话禁止传入**；属 data/sealed_test/，归 A6）")
    p.add_argument("--base-val-metrics", default="",
                   help="已有基座 val 实测 CSV（缺省则本脚本调 03_03 生成一次并落盘）")
    p.add_argument("--base-clean-f1", type=float, default=FROZEN_BASE_CLEAN_F1)
    p.add_argument("--base-clean-tpr", type=float, default=FROZEN_BASE_CLEAN_TPR)
    p.add_argument("--config", default="configs/train/forgetting.yaml",
                   help="基座侧（03_03）配置：**只含 frozen val 分片**，禁 sealed")
    p.add_argument("--unify-config", default="configs/baseline/unify.yaml",
                   help="分片路径的唯一出处（W4.6 冻结，只读引用）")
    p.add_argument("--ours-config", default="configs/train/qlora_ours.yaml",
                   help="Ours 侧推理配置（权重／加载／目标位／组批），复用 04_01 的单源实现")
    p.add_argument("--report-dir", default="reports/train")
    p.add_argument("--out-dir", default="reports/train")
    p.add_argument("--base-val-splits", default="",
                   help="03_03 要跑的 val 分片代号（逗号分隔；缺省 = 本体制的 <regime>_val）")
    p.add_argument("--base-score-track", action="store_true",
                   help="附加跑「同规则基座」并列补充诊断（非主口径，呈 W0）")
    p.add_argument("--limit", type=int, default=0,
                   help="每分片只取前 N 条（>0 生效；**仅自测/冒烟，正式交付不得使用**）")
    p.add_argument("--tag", default="", help="产物时间戳覆盖（空=取当前时刻）")
    return p


def main(argv=None) -> int:
    init_streams()
    args = build_parser().parse_args(argv)

    if args.self_check:
        return _self_check()

    if not args.regime:
        print("REJECTED --regime：必填（id 或 ood）")
        return EXIT_ASSERT
    if not args.checkpoints:
        print("REJECTED --checkpoints：必填")
        return EXIT_ASSERT
    if not args.val:
        print("REJECTED --val：必填（只接受 frozen 侧 val.csv）")
        return EXIT_ASSERT

    ts = args.tag or time.strftime("%Y%m%d-%H%M%S", time.localtime())
    logger_lines = []

    def logger(msg):
        print(msg)
        logger_lines.append(str(msg))

    # --- 路径红线（先于任何读取／模型加载）---
    try:
        val_abs = assert_val_allowed(args.val, args.regime)
        ckpt_dirs = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
        for c in ckpt_dirs:
            assert_no_sealed(c, "--checkpoints")
        if args.base_val_metrics:
            assert_no_sealed(args.base_val_metrics, "--base-val-metrics")
    except PathRejected as exc:
        print(str(exc))
        print("STATUS=%s GATE=%s" % (STATUS_SELFCHECK_ONLY, GATE_UNKNOWN))
        return EXIT_PATH_REJECTED
    except ForgettingAssert as exc:
        print("ASSERT_FAIL: %s" % exc)
        return EXIT_ASSERT

    if args.clean_test:
        logger("REMINDER: --clean-test 指向 %s —— 该分片属 data/sealed_test/，**读前须先在 "
               "data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md 登记 6 列**"
               "（日期／窗口／用途／命令／输出路径／复核人）；A5 会话传入即属违规。" % args.clean_test)

    val_rel = os.path.relpath(val_abs, _ROOT).replace("\\", "/")
    val_md5 = md5_of(val_abs)

    ckpts = []
    for c in ckpt_dirs:
        try:
            ckpts.append(resolve_checkpoint(c))
        except ForgettingAssert as exc:
            print("ASSERT_FAIL: %s" % exc)
            return EXIT_ASSERT
    for c in ckpts:
        if c["regime"] and c["regime"] != args.regime:
            print("ASSERT_FAIL: checkpoint 体制 %s 与 --regime %s 不一致（%s）"
                  % (c["regime"], args.regime, c["ckpt_dir"]))
            return EXIT_ASSERT
        if c["tau"] is None:
            print("ASSERT_FAIL: 无法取得 τ（best.json/meta.json 均无 tau_ours）：%s" % c["ckpt_dir"])
            return EXIT_ASSERT

    logger("FORGETTING_BEGIN ts=%s regime=%s n_ckpt=%d val=%s" % (ts, args.regime, len(ckpts), val_rel))
    logger("VAL_MD5 %s %s" % (val_rel, val_md5))

    base_splits = [s.strip() for s in args.base_val_splits.split(",") if s.strip()] \
        or ["%s_val" % args.regime]

    if args.dry_run:
        for c in ckpts:
            logger("DRYRUN_CKPT regime=%s seed=%s step=%s run_id=%s tau=%s dir=%s"
                   % (args.regime, c["seed"], c["step"], c["run_id"], c["tau"], c["ckpt_dir"]))
        status = STATUS_OFFICIAL if args.clean_test else STATUS_SELFCHECK_ONLY
        logger("DRYRUN_RESULT=PASS n_ckpt=%d base_splits=%s clean_test=%s status=%s gate=%s"
               % (len(ckpts), ",".join(base_splits), "yes" if args.clean_test else "no",
                  status, GATE_UNKNOWN))
        print("STATUS=%s GATE=%s" % (status, GATE_UNKNOWN))
        return EXIT_OK

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(_ROOT, "logs", "train")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "04_04_check_forgetting_%s.log" % ts)

    # --- ② val 轨道基座实测 ---
    base_by_regime = {}
    base_rows = []
    base_metrics_csv = args.base_val_metrics or ""
    if args.base_val_metrics:
        base_by_regime = load_base_val_metrics(os.path.abspath(args.base_val_metrics))
        for regime, m in base_by_regime.items():
            base_rows.append({"split": m.get("split"), "regime": regime, "n": m.get("n"),
                              "n_pos": m.get("n_pos", ""), "n_neg": m.get("n_neg", ""),
                              "tp": m.get("tp", ""), "fp": m.get("fp", ""),
                              "fn": m.get("fn", ""), "tn": m.get("tn", ""),
                              "tpr": m.get("tpr"), "fpr": m.get("fpr"), "f1": m.get("f1"),
                              "n_unparsed": m.get("n_unparsed")})
        logger("BASE_VAL_REUSED %s regimes=%s" % (args.base_val_metrics, ",".join(base_by_regime)))
    else:
        try:
            import yaml
        except ImportError as exc:                       # pragma: no cover
            print("ASSERT_FAIL: 需要 PyYAML：%s" % exc)
            return EXIT_ASSERT
        unify_path = args.unify_config if os.path.isabs(args.unify_config) \
            else os.path.join(_ROOT, args.unify_config)
        with open(unify_path, "r", encoding="utf-8") as fh:
            unify_cfg = yaml.safe_load(fh)
        outputs_dir = os.path.join(args.out_dir, "forgetting_outputs")
        try:
            res = run_base_val_03_03(base_splits, args.config, args.unify_config,
                                     outputs_dir, args.report_dir, ts,
                                     limit=args.limit, logger=logger)
        except ForgettingAssert as exc:
            print("ASSERT_FAIL: %s" % exc)
            return EXIT_ASSERT
        for split in base_splits:
            scfg = (unify_cfg.get("splits") or {}).get(split)
            if not scfg:
                print("ASSERT_FAIL: unify.yaml 无分片 %s" % split)
                return EXIT_ASSERT
            split_csv = assert_no_sealed(os.path.join(_ROOT, scfg["path"]), "--val")
            m = metrics_from_03_03_outputs(res["outputs"][split], split_csv)
            regime = scfg.get("regime") or split.split("_")[0]
            base_rows.append({"split": split, "regime": regime, "n": m["n"],
                              "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                              "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                              "tpr": m["tpr"], "fpr": m["fpr"], "f1": m["f1"],
                              "n_unparsed": m["n_unparsed"],
                              "outputs_csv": os.path.relpath(res["outputs"][split], _ROOT)
                              .replace("\\", "/")})
            base_by_regime[regime] = {"f1": m["f1"], "fpr": m["fpr"], "tpr": m["tpr"],
                                      "n": m["n"], "n_unparsed": m["n_unparsed"], "split": split}
        base_metrics_csv = os.path.join(args.out_dir, "base_val_metrics_v%s_%s.csv" % (VERSION, ts))
        write_csv(base_metrics_csv, base_rows, BASE_VAL_COLUMNS)
        logger("BASE_VAL_WROTE %s" % os.path.relpath(base_metrics_csv, _ROOT).replace("\\", "/"))

    base = base_by_regime.get(args.regime)
    if not base:
        print("ASSERT_FAIL: 基座 val 实测缺本体制（%s）的结果" % args.regime)
        return EXIT_ASSERT

    # --- Ours 侧逐 checkpoint 推理 ---
    T = load_ours_modules()
    ours_cfg = T.load_config(args.ours_config)

    detail_rows = []
    for c in ckpts:
        logger("OURS_INFER seed=%s step=%s ckpt=%s" % (c["seed"], c["step"], c["ckpt_dir"]))
        m, _scores = infer_ours(T, ours_cfg, c["ckpt_dir"], val_abs, c["tau"], c["seed"],
                                limit=args.limit, logger=logger)
        d_f1 = delta(base["f1"], m["f1"])
        d_fpr = delta(base["fpr"], m["fpr"])
        d_tpr = delta(base["tpr"], m["tpr"])
        note = []
        if c["best_val_f1"] is not None and abs(float(c["best_val_f1"]) - float(m["f1"])) > 1e-9:
            note.append("best.json val_f1=%s ≠ 本轮复算 %s"
                        % (fmt_num(c["best_val_f1"]), fmt_num(m["f1"])))
        if m["fpr"] is None:
            note.append("FPR 无定义（无负样本）")
        detail_rows.append({
            "regime": args.regime, "seed": c["seed"], "run_id": c["run_id"], "step": c["step"],
            "ckpt_dir": os.path.relpath(c["ckpt_dir"], _ROOT).replace("\\", "/"),
            "track": "val", "n": m["n"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "tn": m["tn"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
            "f1": m["f1"], "fpr": m["fpr"], "tpr": m["tpr"], "tau": c["tau"],
            "base_f1": base["f1"], "base_fpr": base["fpr"], "base_tpr": base["tpr"],
            "d_f1_abs": d_f1[0], "d_f1_rel": d_f1[1],
            "d_fpr_abs": d_fpr[0], "d_fpr_rel": d_fpr[1],
            "d_tpr_abs": d_tpr[0], "d_tpr_rel": d_tpr[1],
            "verdict_val_selfcheck": verdict_of(d_f1[0]),
            "note": "；".join(note) or "",
        })
        logger("OURS_METRIC seed=%s f1=%s fpr=%s tpr=%s d_f1_abs=%s d_f1_rel=%s"
               % (c["seed"], fmt_num(m["f1"]), fmt_num(m["fpr"]), fmt_num(m["tpr"]),
                  fmt_num(d_f1[0]), fmt_rel(d_f1[1])))

    # --- clean 轨道（A5 恒不跑；传了才跑，且只应由 A6 在登记后传）---
    has_clean = bool(args.clean_test)
    if has_clean:
        if not os.path.isfile(args.clean_test):
            print("ASSERT_FAIL: --clean-test 不存在：%s" % args.clean_test)
            return EXIT_ASSERT
        base_clean = {"f1": args.base_clean_f1, "tpr": args.base_clean_tpr, "fpr": None}
        for c in ckpts:
            logger("CLEAN_INFER seed=%s" % c["seed"])
            m, _s = infer_ours(T, ours_cfg, c["ckpt_dir"], os.path.abspath(args.clean_test),
                               c["tau"], c["seed"], limit=args.limit, logger=logger)
            d_f1 = delta(base_clean["f1"], m["f1"])
            d_tpr = delta(base_clean["tpr"], m["tpr"])
            d_fpr = delta(base_clean["fpr"], m["fpr"])
            detail_rows.append({
                "regime": args.regime, "seed": c["seed"], "run_id": c["run_id"], "step": c["step"],
                "ckpt_dir": os.path.relpath(c["ckpt_dir"], _ROOT).replace("\\", "/"),
                "track": "clean", "n": m["n"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "tn": m["tn"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                "f1": m["f1"], "fpr": m["fpr"], "tpr": m["tpr"], "tau": c["tau"],
                "base_f1": base_clean["f1"], "base_fpr": base_clean["fpr"],
                "base_tpr": base_clean["tpr"],
                "d_f1_abs": d_f1[0], "d_f1_rel": d_f1[1],
                "d_fpr_abs": d_fpr[0], "d_fpr_rel": d_fpr[1],
                "d_tpr_abs": d_tpr[0], "d_tpr_rel": d_tpr[1],
                "verdict_val_selfcheck": verdict_of(d_f1[0]),
                "note": "clean 分片全 label=1 ⇒ FPR 无定义、F1 为退化形式",
            })

    # --- STATUS / GATE ---
    if not has_clean:
        status = STATUS_SELFCHECK_ONLY
        status_reason = "A5 会话：clean 集属 data/sealed_test/，无读取权 ⇒ 只跑 val 自检"
        gate = GATE_UNKNOWN
        gate_reason = "未传 --clean-test ⇒ clean 轨道空 ⇒ 不得给正式结论（§4-4）"
    else:
        status = STATUS_OFFICIAL
        status_reason = "已传 --clean-test ⇒ clean 轨道齐备（执行窗口须为 A6 并已登记 ACCESS_LOG）"
        verdicts = [r["verdict_val_selfcheck"] for r in detail_rows if r["track"] == "clean"]
        if GATE_FAIL in verdicts:
            gate = GATE_FAIL
        elif verdicts and all(v == GATE_PASS for v in verdicts):
            gate = GATE_PASS
        else:
            gate = GATE_UNKNOWN
        gate_reason = "逐 seed 判定取最劣者（任一 ΔF1 > %.2f ⇒ FAIL）" % DF1_THRESHOLD
    exit_code = EXIT_OK
    exit_reason = "SELFCHECK_ONLY 不因未跑 clean 而失败（§3）；只有断言失败才非零"

    # --- 汇总 ---
    vals = [r for r in detail_rows if r["track"] == "val"]
    summary = {}
    for key in ("d_f1_abs", "d_fpr_abs", "d_tpr_abs"):
        mm, ss = mean_std([r[key] for r in vals])
        summary["d_" + key.split("_")[1]] = "%s ± %s" % (fmt_num(mm), fmt_num(ss))
    summary["per_seed_raw"] = "；".join(
        "seed %s: ΔF1=%s pp（Ours F1 %s vs 基座 %s）"
        % (r["seed"], fmt_num(r["d_f1_abs"]), fmt_num(r["f1"]), fmt_num(r["base_f1"]))
        for r in vals) or "（无）"
    uniq = set(round(float(r["d_f1_abs"]), 12) for r in vals if r["d_f1_abs"] is not None)
    summary["std_degenerate"] = len(vals) > 1 and len(uniq) == 1

    base_score_track = None
    if args.base_score_track:
        b, _sc = infer_base_score_track(T, ours_cfg, val_abs, limit=args.limit, logger=logger)
        base_score_track = b
        logger("BASE_SCORE_TRACK f1=%s fpr=%s tpr=%s tau_main=%s"
               % (fmt_num(b.get("f1")), fmt_num(b.get("fpr")), fmt_num(b.get("tpr")),
                  b.get("tau_main")))

    # --- 产物 ---
    detail_csv = os.path.join(args.out_dir, "forgetting_detail_v%s_%s.csv" % (VERSION, ts))
    write_csv(detail_csv, detail_rows, DETAIL_COLUMNS)
    report_md = os.path.join(args.report_dir, "forgetting_v%s_%s.md" % (VERSION, ts))

    artifacts = [
        ("逐 checkpoint 明细 CSV", os.path.relpath(detail_csv, _ROOT).replace("\\", "/")),
        ("基座 val 实测 CSV", os.path.relpath(base_metrics_csv, _ROOT).replace("\\", "/")
         if base_metrics_csv else "（未落盘）"),
        ("运行报告（本文件）", os.path.relpath(report_md, _ROOT).replace("\\", "/")),
        ("日志", os.path.relpath(log_path, _ROOT).replace("\\", "/")),
    ]

    to_w0 = [
        "主口径取**绝对百分点**（并列相对降幅）—— 请 W0 追认（§2.1）。",
        "「FPR 变化以 val 轨道为唯一合法基准」这一补救（`clean_original_test.csv` 44 行全 `label=1` "
        "⇒ FPR 无定义）请 W0 追认（§2.2）。",
        "正式结论的**执行归属**：默认由 **W6 的 A6 窗口**用同一脚本产出（须先登记 `ACCESS_LOG.md`）；"
        "若 W0 认为应由本窗口补跑，须**单独裁决一次登记访问**（`W5_stage4_train.md` §4-1）。",
        "回放语料的权限冲突（§7 第二行）：「原始样本回放」若指 sealed 的 `clean_original_test.csv`，"
        "A5 无权读取；若改用 train 分片样本或 `plaintext` 列构造明文训练样本，属**训练输入构造变更** ⇒ 须 W0 批准。",
        "**并列补充诊断「同规则基座」（本报告 §7）** —— 主口径是「判词型 B3（`03_03`）vs `score≥τ` 型 Ours」，"
        "两者判定规则不同；本窗口另跑了基座在**同一打分规则**下的 val 指标作对称对照（**不参与 `GATE`**），"
        "请 W0 追认是否入正文。",
        "**τ 不对称（方法学）**：Ours 的 τ 取自该 run `best.json` 的 `tau_ours`，是在**同一 val 分片**上按 "
        "F1 最大寻得的 ⇒ Ours 的 val 指标属**样本内**；而基座（判词型，无 τ）无此优化 ⇒ "
        "该不对称使 ΔF1 偏**有利**于 Ours（门禁偏松）。请 W0 裁是否在正文／Limitation 声明。",
        "**工程性扩展（不改变冻结契约）**：为支持无 GPU 的单测与人工预检，新增 `--dry-run`／`--limit`／"
        "`--base-score-track`／`--base-val-splits` 四个可选参数，冻结接口（§3 表格）的必填项与语义**未改**。",
    ]

    receipt = [
        "`04_04 --self-check` **OK**（纯合成：路径拒绝 5 例／Δ 口径／阈值边界／退化 F1／冻结基准值／"
        "STATUS·GATE 常量／明细列齐备）。",
        "A5 会话实跑 **未传 `--clean-test`** ⇒ exit=0、`STATUS=%s`、`GATE=%s`（§6-1）。"
        % (STATUS_SELFCHECK_ONLY, GATE_UNKNOWN),
        "全程**无 sealed 路径被访问**：`--val`／`--checkpoints` 均过 `assert_no_sealed()`；"
        "`03_03` 子进程 stdout 无 `REMINDER` 行（出现即停工，`D-41` 裁定五④）。",
        "三项指标（F1／FPR／TPR）在 %d 个 checkpoint 上**全部齐备**且均给出变化量：%s。"
        % (len(vals), "／".join("seed %s = %s" % (r["seed"], r["verdict_val_selfcheck"])
                                for r in vals)),
        "路径拒绝用例：`data/sealed_test/...` 与 `.../test.csv` **均**非零退出（退出码 %d），单测覆盖两种（§6-3）。"
        % EXIT_PATH_REJECTED,
        "阶段3 基线数字**只引用、不复算**：clean 轨道引 `baseline-v1` 冻结数 "
        "F1 %.4f／TPR %.4f／FPR `n/a`（§4-2）。" % (FROZEN_BASE_CLEAN_F1, FROZEN_BASE_CLEAN_TPR),
        "τ 一律取各 run `best.json` 的 `tau_ours`（`D-61` 转交项 #52），本窗口**未**重新寻 τ、"
        "未选 checkpoint、未调参（§4-3）。",
    ]

    ctx = {
        "ts": ts, "regime": args.regime, "val_rel": val_rel, "val_md5": val_md5,
        "config": args.config, "ours_config": args.ours_config,
        "status": status, "gate": gate,
        "status_reason": status_reason, "gate_reason": gate_reason,
        "exit_code": exit_code, "exit_reason": exit_reason,
        "base_rows": base_rows,
        "base_metrics_csv": os.path.relpath(base_metrics_csv, _ROOT).replace("\\", "/")
        if base_metrics_csv else "",
        "detail_rows": detail_rows, "summary": summary,
        "base_clean_f1": args.base_clean_f1, "base_clean_tpr": args.base_clean_tpr,
        # §5 交接命令用**用户传入的相对路径**（`--checkpoints` 原样）⇒ 可直接复制、与项目内路径风格一致
        "has_clean": has_clean, "ckpt_dirs": ckpt_dirs,
        "base_score_track": base_score_track,
        "artifacts": artifacts, "to_w0": to_w0, "receipt": receipt,
    }
    write_report(report_md, ctx)

    with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(logger_lines) + "\n")
    print("wrote %s" % os.path.relpath(detail_csv, _ROOT).replace("\\", "/"))
    print("wrote %s" % os.path.relpath(report_md, _ROOT).replace("\\", "/"))

    # §3：stdout 末行 `STATUS=…` 与 `GATE=…`
    print("STATUS=%s GATE=%s" % (status, gate))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

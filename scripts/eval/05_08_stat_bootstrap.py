# -*- coding: utf-8 -*-
r"""05_08_stat_bootstrap.py — 阶段5（W6.9，A6）5.8 统计严谨性：paired bootstrap 检验。

接口契约（`windows/W6.9_bootstrap.md` §2，**冻结后不得改**）::

    python scripts/eval/05_08_stat_bootstrap.py \\
        --pair ours:b3 --split id_test \\
        --pair ours:b3 --split ood_test \\
        --pair ours:b5 --split id_test \\
        --ours-outputs <DIR> \\
        --baseline-outputs <DIR> \\
        --n-boot 10000 --boot-seeds 42,123,456 \\
        --config configs/eval/stat_bootstrap.yaml \\
        --report-dir <DIR> \\
        [--metric {f1,tpr,fpr,auroc,all}] [--self-check] [--dry-run]

（`--ours-outputs`／`--baseline-outputs`／`--report-dir` 一律由命令行给出，见 C9。）

stdout 末行（依模式）::

    SELFCHECK_RESULT=PASS|FAIL n=<k>
    DRYRUN_RESULT=PASS n=<k>
    REPORT_MD=<path>

硬约束逐条落点（`windows/W6.9_bootstrap.md` §3）::

    C1  只做三组  → `resolve_comparisons()`：三元组须逐一命中 `cfg["comparisons"]`，否则非零退出
    C2  n_boot    → `assert_n_boot()`：命令行值 != `cfg["n_boot"]` ⇒ 非零退出
    C3  配对完整  → `load_pairs()`：双向差集与 `text_sha256` 不一致**任一非零**即抛 `PairingError`
    C4  不可比    → `resolve_metrics()`：`metrics.excluded` 内指标只渲染 `n/a`，**不入** bootstrap
    C5  口径一致  → 只读 `pred`（τ\* 主口径）／`pred_parallel`（0.5 并列）；**不重寻 τ／不选阈值**
    C6  引用带口径→ 报告 §1 声明节引 `BASELINE_REPORT.md` §12；`b4` 全程不出现（C1 已拦）
    C7  不读封存  → 输入恒为既有产物 CSV；`cfg["sealed"]["read"]=false`，报告声明「零行登记」
    C8  声明齐备  → `DECLARATIONS` 九条，逐条命中才算「声明齐备」
    C9  报告路径  → 命令行必给 `--report-dir`；脚本内不出现硬编码的 `reports/` **输出**路径
    C10 只追加    → 产物一律 `_v1.0_<ts>` 新名，不覆盖同名文件（存在即抛错）

顶层不得 import 重型库（`AGENTS.md` §3）：`torch`／`transformers`／`peft`／`yaml`／`numpy`
一律在函数体内惰性 import；`numpy` 只在 `bootstrap_paired()` 内 import。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

TBD = "<TBD>"
NA = "n/a"

METRIC_ORDER = ["f1", "tpr", "fpr", "auroc"]

# 方向（Δ = Ours − baseline）：True = 越大越好
METRIC_DIRECTION = {"f1": True, "tpr": True, "fpr": False, "auroc": True}

# C8 声明齐备：九条（与 `configs/eval/stat_bootstrap.yaml` 的 `declarations_required` 对应）
DECLARATIONS = [
    ("CI 方法", "百分位法（percentile）"),
    ("CI 水平", "95%（`ci.level"),
    ("p 值口径", "2 * min(P(Δ<=0), P(Δ>=0))"),
    ("连续性修正", "不做连续性修正"),
    ("多重比较校正", "未做多重比较校正"),
    ("未做其他对比", "未做其他对比"),
    ("未读封存", "`data/sealed_test/**`"),
    ("std=0 退化", "std=0 退化"),
    ("p>0.05 表述", "未检出差异"),
]

# 报告内**禁止**出现的表述（§4-5：不得把 `p > 0.05` 说成两者等价）
FORBIDDEN_PHRASES = ["两组等价", "等效于", "等同无差异"]


class PairingError(Exception):
    """C3 配对完整性失败（缺失／多余／`text_sha256` 不一致）。"""


class ConstraintError(Exception):
    """C1／C2／C4／C5／C9／C10 任一硬约束失败。"""


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：W3.7 `F-1`；`D-40` 裁定五：新脚本须**显式告警**，不得静默 `pass`）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:
            print("WARN: 输出流重配为 UTF-8 失败（%r）；非 ASCII 字符可能被替换" % (exc,),
                  file=sys.stderr)


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def md5_file(path):
    import hashlib

    d = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            d.update(blk)
    return d.hexdigest()


def fmt_num(x, nd=6):
    if x is None:
        return NA
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return ("%." + str(nd) + "f") % (x,)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def resolve_log_dir(args=None):
    """日志目录解析（依 `#64` / `D-80` 裁定三 / `D-82`）。

    优先级：① 参数／环境变量 `CODEOBF_LOG_DIR`；② `--self-check` 或 `--dry-run`
    ⇒ `logs/eval/tmp/`（已 gitignore，`.gitignore` L23 `logs/*/tmp/`）；③ `logs/eval/`。

    **缘起**：日志目录原写死 `logs/eval/` ⇒ 任何以子进程跑本脚本的场合（**尤其单元测试**
    与本脚本 `--self-check` 内部的递归 `_run`）都会在生产日志目录留痕，已复发三次。
    """
    env = os.environ.get("CODEOBF_LOG_DIR")
    if env:
        return env
    if args is not None and (getattr(args, "self_check", False) or getattr(args, "dry_run", False)):
        return os.path.join(_ROOT, "logs", "eval", "tmp")
    return os.path.join(_ROOT, "logs", "eval")


def flagged_logger(ts, log_dir=None):
    """日志：默认 `logs/eval/05_08_stat_bootstrap_<ts>.log`，同时回显 stdout。

    `log_dir` 为 None 时回落到 `resolve_log_dir()`（环境变量 → 默认 `logs/eval/`）。
    """
    log_dir = log_dir or os.path.join(_ROOT, "logs", "eval")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "05_08_stat_bootstrap_%s.log" % ts)

    def _log(msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    return _log, path


# ---------------------------------------------------------------------------
# C10 CSV：一律 `csv.DictReader`；行数 ≠ 记录数
# ---------------------------------------------------------------------------
def read_records(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(out_path, fields, rows):
    """写 UTF-8+LF CSV（严格按 `fields` 定序；自带引号转义；写前断言无 CR）。"""
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    for i, row in enumerate(rows):
        for name in fields:
            v = row.get(name, "")
            if "\r" in str(v):
                raise ValueError("write_csv_rows: 第 %d 行字段 %r 含 CR（须先归一为 LF）" % (i, name))
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({name: row.get(name, "") for name in fields})


def assert_not_exists(path, what):
    """C10 只追加：同名产物已存在即抛错（禁止覆盖）。"""
    if os.path.exists(path):
        raise ConstraintError("C10 只追加：%s 已存在，禁止覆盖 ⇒ %s" % (what, path))


# ---------------------------------------------------------------------------
# C1 只做预注册三组
# ---------------------------------------------------------------------------
def resolve_comparisons(cfg, pairs):
    """`pairs` 形如 `[(ours, baseline, split), ...]` ⇒ 返回命中的 comparison 定义。

    任一三元组未命中 `cfg["comparisons"]` ⇒ `ConstraintError`（**非零退出**）。
    """
    allowed = cfg.get("comparisons") or []
    index = {(c["ours"], c["baseline"], c["split"]): c for c in allowed}
    if not pairs:
        raise ConstraintError("C1：未给 `--pair`（至少须给预注册的三组之一）")
    out = []
    for ours, baseline, split in pairs:
        key = (ours, baseline, split)
        if key not in index:
            raise ConstraintError(
                "C1 只做预注册三组：%r 不在允许集合 %s 内 ⇒ 非零退出"
                % (key, sorted("%s:%s@%s" % k for k in index)))
        out.append(index[key])
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids):
        raise ConstraintError("C1：对比组重复 ⇒ %s" % ids)
    return out


def assert_n_boot(cfg, n_boot):
    want = cfg.get("n_boot")
    if n_boot is not None and int(n_boot) != int(want):
        raise ConstraintError("C2 `n_boot` 定值：命令行 %s != 预注册 %s ⇒ 非零退出" % (n_boot, want))
    return int(want)


# ---------------------------------------------------------------------------
# 指标解析（C4）
# ---------------------------------------------------------------------------
def resolve_metrics(cfg, metric_arg):
    mcfg = cfg.get("metrics") or {}
    excluded = mcfg.get("excluded") or {}
    if metric_arg in (None, "", "all"):
        wanted = list(METRIC_ORDER)
    else:
        wanted = [metric_arg]
    for m in wanted:
        if m not in METRIC_ORDER:
            raise ConstraintError("未知指标 %r（可选：f1/tpr/fpr/auroc/all）" % m)
    computable, blocked = [], []
    for m in wanted:
        if m in excluded:
            blocked.append((m, (excluded[m] or {}).get("reason", "")))
        else:
            computable.append(m)
    if not computable:
        raise ConstraintError("C4：所请求的指标全部属不可比（`n/a`）⇒ 无可执行 bootstrap")
    return computable, blocked


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
def resolve_ours_path(cfg, ours_dir, comp, seed):
    """择定 Ours 逐样本 CSV：文件名由 `inputs.ours_file` ＋ `inputs.ours_ts` 定值。

    **禁止** glob 自动择一（`cfg.inputs.on_ambiguous = fail`）。
    """
    inputs = cfg.get("inputs") or {}
    ts_map = inputs.get("ours_ts") or {}
    ts = ts_map.get(comp["split"])
    if not ts:
        raise ConstraintError("配置缺 `inputs.ours_ts[%s]`（禁止按 mtime 择一）" % comp["split"])
    name = str(inputs.get("ours_file")).format(
        regime=comp["regime"], seed=seed, split=comp["split"], ts=ts)
    return os.path.join(ours_dir, name)


def resolve_baseline_path(cfg, base_dir, comp):
    inputs = cfg.get("inputs") or {}
    bmap = (inputs.get("baseline") or {}).get(comp["baseline"]) or {}
    name = bmap.get(comp["split"])
    if not name:
        raise ConstraintError("配置缺 `inputs.baseline[%s][%s]`" % (comp["baseline"], comp["split"]))
    return os.path.join(base_dir, name)


def _parse_pred(value, where):
    s = str(value).strip()
    if s not in ("0", "1"):
        raise PairingError("`pred` 值非 0/1（%r）⇒ 不得默认兜底：%s" % (s, where))
    return int(s)


# ---------------------------------------------------------------------------
# C3 配对
# ---------------------------------------------------------------------------
def load_pairs(cfg, ours_path, base_path, tau_column="pred"):
    ours_rows = read_records(ours_path)
    base_rows = read_records(base_path)
    pcfg = cfg.get("pairing") or {}
    key = pcfg.get("key", "sample_id")
    sec = pcfg.get("verify_secondary", "text_sha256")
    for rows, tag in ((ours_rows, "Ours"), (base_rows, "baseline")):
        if not rows:
            raise PairingError("%s 侧记录数为 0 ⇒ %s" % (tag, ours_path if tag == "Ours" else base_path))
        if key not in rows[0]:
            raise PairingError("%s 侧缺配对键 %r" % (tag, key))
    ours_map = {}
    for r in ours_rows:
        k = r[key]
        if k in ours_map:
            raise PairingError("Ours 侧配对键重复：%r（禁止覆盖去重）" % k)
        ours_map[k] = r
    base_map = {}
    for r in base_rows:
        k = r[key]
        if k in base_map:
            raise PairingError("baseline 侧配对键重复：%r（禁止覆盖去重）" % k)
        base_map[k] = r

    only_ours = sorted(set(ours_map) - set(base_map))
    only_base = sorted(set(base_map) - set(ours_map))
    if only_ours or only_base:
        raise PairingError(
            "C3 配对不完整：仅 Ours 有 %d 条（例 %s）／仅 baseline 有 %d 条（例 %s）"
            " ⇒ 禁止取交集或静默丢弃 ⇒ 非零退出"
            % (len(only_ours), only_ours[:3], len(only_base), only_base[:3]))

    common = list(ours_map.keys())
    mism = [k for k in common if str(ours_map[k].get(sec, "")) != str(base_map[k].get(sec, ""))]
    if mism:
        raise PairingError("C3 配对键 %r 一致但 %r 不一致：%d 条（例 %s）⇒ 非零退出"
                           % (key, sec, len(mism), mism[:3]))

    labels, pred_ours, pred_base = [], [], []
    for k in common:
        o, b = ours_map[k], base_map[k]
        lab = o.get("label")
        if str(lab).strip() not in ("0", "1"):
            raise PairingError("Ours 侧 `label` 值非 0/1（%r）⇒ 不得默认兜底：%s" % (lab, k))
        labels.append(int(str(lab).strip()))
        if tau_column not in o:
            raise PairingError("Ours 侧缺列 %r ⇒ 不得回退到其它口径列" % tau_column)
        pred_ours.append(_parse_pred(o[tau_column], "Ours %s" % k))
        pred_base.append(_parse_pred(b["pred"], "baseline %s" % k))

    check = {
        "ours_n": len(ours_rows),
        "base_n": len(base_rows),
        "n_pairs": len(common),
        "only_ours": len(only_ours),
        "only_base": len(only_base),
        "sha_mismatch": len(mism),
    }
    return labels, pred_ours, pred_base, check


# ---------------------------------------------------------------------------
# 指标与 bootstrap
# ---------------------------------------------------------------------------
def confusion_from_arrays(labels, preds):
    """返回 per-sample 的四个 0/1 指示序列（tp/fp/fn/tn），供向量化重采样。"""
    import numpy as np

    # 指示序列用 **int64**：int8 在 `sum()` 时会在 >127 处**溢出**（标量累加亦按 int8 进位）
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    return (np.int64(1) * ((y == 1) & (p == 1)),
            np.int64(1) * ((y == 0) & (p == 1)),
            np.int64(1) * ((y == 1) & (p == 0)),
            np.int64(1) * ((y == 0) & (p == 0)))


def _metric_from_counts(metric, tp, fp, fn, tn):
    """逐元素（向量化）计算指标；分母为 0 ⇒ `nan`（**不得**默认兜底）。"""
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "tpr":
            den = tp + fn
            out = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
        elif metric == "fpr":
            den = fp + tn
            out = np.where(den > 0, fp / np.maximum(den, 1), np.nan)
        elif metric == "f1":
            den = 2 * tp + fp + fn
            out = np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)
        else:
            raise ConstraintError("不可计算指标 %r（属 `n/a`，不得进入本函数）" % metric)
    return out


def metric_point(metric, labels, preds):
    tp, fp, fn, tn = (int(sum(x)) for x in confusion_from_arrays(labels, preds))
    den = {"tpr": tp + fn, "fpr": fp + tn, "f1": 2 * tp + fp + fn}[metric]
    if den <= 0:
        return None
    return float(_metric_from_counts(metric, tp, fp, fn, tn))


def bootstrap_paired(metric, labels, pred_ours, pred_base, n_boot, seed):
    """paired bootstrap：对**配对样本单位**重采样 `n_boot` 次，每次两侧同批下标重算指标取 Δ。

    `Δ = metric(Ours) − metric(baseline)`。返回 dict（含点估计、CI、`p`、边界计数）。
    """
    import numpy as np

    o_tp, o_fp, o_fn, o_tn = confusion_from_arrays(labels, pred_ours)
    b_tp, b_fp, b_fn, b_tn = confusion_from_arrays(labels, pred_base)
    n = len(labels)

    point = metric_point(metric, labels, pred_ours)
    base_point = metric_point(metric, labels, pred_base)
    if point is None or base_point is None:
        # 全样本上该指标无定义（分母为 0）⇒ 该格 `n/a`，**禁止**强行算 Δ 或默认兜底
        raise ConstraintError(
            "指标 %r 在 %s 上无定义（Ours=%s／baseline=%s，分母为 0）⇒ 该格 `n/a`，"
            "**不得**进入 bootstrap" % (metric, "全样本", point, base_point))
    delta_point = float(point) - float(base_point)

    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(n_boot), dtype=np.float64)
    chunk = 512
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        o = _metric_from_counts(metric, o_tp[idx].sum(1), o_fp[idx].sum(1),
                                o_fn[idx].sum(1), o_tn[idx].sum(1))
        b = _metric_from_counts(metric, b_tp[idx].sum(1), b_fp[idx].sum(1),
                                b_fn[idx].sum(1), b_tn[idx].sum(1))
        deltas[done:done + m] = o - b
        done += m

    n_nan = int(np.isnan(deltas).sum())
    if n_nan:
        raise ConstraintError(
            "bootstrap 分布含 %d 个 `nan`（重采样中出现分母为 0 的子集）⇒ "
            "禁止静默剔除 ⇒ 非零退出" % n_nan)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_le = float(np.mean(deltas <= 0.0))
    p_ge = float(np.mean(deltas >= 0.0))
    p = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "metric": metric,
        "seed": int(seed),
        "n": n,
        "n_boot": int(n_boot),
        "ours_m": point,
        "base_m": base_point,
        "delta": delta_point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": float(p),
        "p_le0": p_le,
        "p_ge0": p_ge,
        "n_le_0": int(np.sum(deltas <= 0.0)),
        "n_gt_1": int(np.sum(deltas > 1.0)),
        "n_nan": n_nan,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_comparison(cfg, args, comp, metrics, blocked, logger):
    ours_dir = _abs(args.ours_outputs)
    base_dir = _abs(args.baseline_outputs)
    n_boot = int(cfg["n_boot"])
    seeds = list(cfg.get("boot_seeds") or [42, 123, 456])
    tau_col = ((cfg.get("tau") or {}).get("ours_column") or "pred")

    base_path = resolve_baseline_path(cfg, base_dir, comp)
    per_metric = {m: [] for m in metrics}
    pairing_rows = []
    # 配对**逐 seed 只做一次**（同一份配对供全部指标复用；避免配对核验表按指标重复）
    for seed in seeds:
        ours_path = resolve_ours_path(cfg, ours_dir, comp, seed)
        if not os.path.isfile(ours_path):
            raise ConstraintError("Ours 输入缺失：%s" % ours_path)
        labels, po, pb, check = load_pairs(cfg, ours_path, base_path, tau_col)
        for metric in metrics:
            res = bootstrap_paired(metric, labels, po, pb, n_boot, seed)
            per_metric[metric].append(res)
            logger("BOOT ok pair=%s metric=%s seed=%s n=%d delta=%.6f p=%.6f ci=[%.6f, %.6f]"
                   % (comp["id"], metric, seed, res["n"], res["delta"],
                      res["p"], res["ci_lo"], res["ci_hi"]))
        pairing_rows.append({
            "pair_id": comp["id"], "seed": seed, "split": comp["split"],
            "ours_file": os.path.basename(ours_path),
            "ours_md5": md5_file(ours_path),
            "baseline_file": os.path.basename(base_path),
            "baseline_md5": md5_file(base_path),
            "ours_n": check["ours_n"], "baseline_n": check["base_n"],
            "n_pairs": check["n_pairs"],
            "only_ours": check["only_ours"], "only_base": check["only_base"],
            "sha_mismatch": check["sha_mismatch"],
            "diff_only_ours": "", "diff_only_base": "",
        })
    return {"comp": comp, "per_metric": per_metric, "pairing_rows": pairing_rows,
            "baseline_path": base_path, "blocked": blocked}


def build_of_record_rows(cfg, results, primary):
    """of record：一组对比**一行**（恒 3 行），逐 seed 结果展开为列。"""
    seeds = list(cfg.get("boot_seeds") or [42, 123, 456])
    fields = ["pair_id", "split", "ours", "baseline", "metric", "n_pairs", "n_boot"]
    for s in seeds:
        fields += ["ours_m_s%s" % s, "base_m_s%s" % s, "delta_s%s" % s,
                   "p_s%s" % s, "ci_lo_s%s" % s, "ci_hi_s%s" % s]
    fields += ["delta_min", "delta_max", "delta_range",
               "p_min", "p_max", "p_range", "ci_lo_min", "ci_hi_max",
               "n_boot_seeds", "direction"]
    rows = []
    for r in results:
        comp = r["comp"]
        per_seed = r["per_metric"][primary]
        by_seed = {x["seed"]: x for x in per_seed}
        row = {
            "pair_id": comp["id"], "split": comp["split"],
            "ours": comp["ours"], "baseline": comp["baseline"],
            "metric": primary,
            "n_pairs": per_seed[0]["n"], "n_boot": per_seed[0]["n_boot"],
        }
        for s in seeds:
            x = by_seed[s]
            row["ours_m_s%s" % s] = fmt_num(x["ours_m"])
            row["base_m_s%s" % s] = fmt_num(x["base_m"])
            row["delta_s%s" % s] = fmt_num(x["delta"])
            row["p_s%s" % s] = fmt_num(x["p"])
            row["ci_lo_s%s" % s] = fmt_num(x["ci_lo"])
            row["ci_hi_s%s" % s] = fmt_num(x["ci_hi"])
        ds = [by_seed[s]["delta"] for s in seeds]
        ps = [by_seed[s]["p"] for s in seeds]
        los = [by_seed[s]["ci_lo"] for s in seeds]
        his = [by_seed[s]["ci_hi"] for s in seeds]
        row["delta_min"] = fmt_num(min(ds))
        row["delta_max"] = fmt_num(max(ds))
        row["delta_range"] = fmt_num(max(ds) - min(ds))
        row["p_min"] = fmt_num(min(ps))
        row["p_max"] = fmt_num(max(ps))
        row["p_range"] = fmt_num(max(ps) - min(ps))
        row["ci_lo_min"] = fmt_num(min(los))
        row["ci_hi_max"] = fmt_num(max(his))
        row["n_boot_seeds"] = len(seeds)
        row["direction"] = "越大越好" if METRIC_DIRECTION[primary] else "越小越好"
        rows.append(row)
    return fields, rows


def seed_flag_for(deltas, ps, alpha=0.05):
    """逐 seed 一致性标注（回退点「三 seed 极差过大 ⇒ 如实报告，不得只报其一」）。

    返回四选一：``符号不一致``／``显著性跨 seed 翻转``／``三 seed 均显著``／``三 seed 均未检出差异``。
    """
    if not deltas:
        return ""
    pos = [d > 0 for d in deltas]
    neg = [d < 0 for d in deltas]
    if not (all(pos) or all(neg)):
        return "符号不一致"
    if min(ps) <= alpha < max(ps):
        return "显著性跨 seed 翻转"
    if max(ps) <= alpha:
        return "三 seed 均显著"
    return "三 seed 均未检出差异"


ALLMETRIC_FIELDS = ["pair_id", "split", "ours", "baseline", "metric", "seed",
                    "n_pairs", "n_boot", "ours_m", "base_m", "delta",
                    "ci_lo", "ci_hi", "p", "n_le_0", "n_gt_1", "n_nan",
                    "direction", "seed_flag", "note"]


def build_allmetric_rows(cfg, results, blocked_reasons):
    rows = []
    for r in results:
        comp = r["comp"]
        for metric, per_seed in sorted(r["per_metric"].items(),
                                       key=lambda kv: METRIC_ORDER.index(kv[0])):
            flag = seed_flag_for([x["delta"] for x in per_seed], [x["p"] for x in per_seed])
            for x in per_seed:
                rows.append({
                    "pair_id": comp["id"], "split": comp["split"],
                    "ours": comp["ours"], "baseline": comp["baseline"],
                    "metric": metric, "seed": x["seed"],
                    "n_pairs": x["n"], "n_boot": x["n_boot"],
                    "ours_m": fmt_num(x["ours_m"]), "base_m": fmt_num(x["base_m"]),
                    "delta": fmt_num(x["delta"]),
                    "ci_lo": fmt_num(x["ci_lo"]), "ci_hi": fmt_num(x["ci_hi"]),
                    "p": fmt_num(x["p"]),
                    "n_le_0": x["n_le_0"], "n_gt_1": x["n_gt_1"], "n_nan": x["n_nan"],
                    "direction": "越大越好" if METRIC_DIRECTION[metric] else "越小越好",
                    "seed_flag": flag, "note": "",
                })
        for metric, reason in (r["blocked"] or []):
            rows.append({
                "pair_id": comp["id"], "split": comp["split"],
                "ours": comp["ours"], "baseline": comp["baseline"],
                "metric": metric, "seed": "all",
                "n_pairs": "", "n_boot": "",
                "ours_m": NA, "base_m": NA, "delta": NA,
                "ci_lo": NA, "ci_hi": NA, "p": NA,
                "n_le_0": "", "n_gt_1": "", "n_nan": "",
                "direction": "", "seed_flag": "", "note": reason,
            })
    return ALLMETRIC_FIELDS, rows


def render_report(cfg, args, results, ts, paths, primary, blocked_reasons):
    seeds = list(cfg.get("boot_seeds") or [42, 123, 456])
    n_boot = cfg["n_boot"]
    ci = cfg.get("ci") or {}
    pv = cfg.get("p_value") or {}
    mc = cfg.get("multiple_comparison") or {}
    degen = cfg.get("degenerate") or {}
    sealed = cfg.get("sealed") or {}

    L = []
    A = L.append
    A("# 5.8 统计严谨性 —— paired bootstrap 检验报告")
    A("")
    A("- 窗口：**W6.9**（阶段5，权限 **A6**）　脚本：`scripts/eval/05_08_stat_bootstrap.py`")
    A("- 生成时间：`%s`　配置：`configs/eval/stat_bootstrap.yaml`（**事前定值**，跑后未改）" % ts)
    A("- 依据：`PREREG_v1.md` §5（`paired bootstrap 抽样次数 = 10000` ＋ 三组对比对象补录（G1））；"
      "`SOURCE_LOCK/实验执行方案（最终版 v5）.md` §5.8 ＋「新增 33」")
    A("")
    A("## 1. 声明节（C8：九条齐备）")
    A("")
    A("1. **置信区间方法**：**%s**。" % ci.get("declared"))
    A("2. **置信水平**：95%%（`ci.level` = %s）。" % ci.get("level"))
    A("3. **`p` 值口径**：**双侧**，`%s`；**%s**；结果裁剪到 [0, 1]。"
      % (pv.get("formula"), "不做连续性修正" if not pv.get("continuity_correction") else "做连续性修正"))
    A("   - 分辨率：`p` 的取值分辨率为 `1 / n_boot = %s`；`p = 0.000000` 应读作「10000 次重采样中"
      "未出现反向差值」，**不等于**「概率为 0」。" % ("1/%d" % n_boot))
    A("4. **多重比较校正**：**未做多重比较校正**。%s"
      % ((mc.get("declared") or "").strip()))
    A("5. **未做其他对比**：本窗口**只做** `PREREG_v1.md` §5 预注册的三组对比。"
      "理由（v5「新增 33」原文）：**明确对比对象，不要无差别全部做 bootstrap…减少多重比较问题**。"
      "命令行给出其他组合一律**非零退出**（C1，已单测覆盖）。")
    A("6. **未读封存**：%s" % sealed.get("declared"))
    A("7. **std=0 退化声明**：%s" % (degen.get("declared") or "").strip())
    A("   - 与 bootstrap 的关系：%s" % (degen.get("bootstrap_unaffected") or "").strip())
    A("8. **`n_boot = %s`**（预注册定值；论文须注明抽样次数）。" % n_boot)
    A("9. **`p > 0.05` 的表述纪律**：只能表述为「**未检出差异**」，"
      "**不得**表述为两者等价类措辞（§4-5 禁止项；脚本在写盘前已对报告全文做禁止表述校验，"
      "命中即抛错、不落盘）。")
    A("")
    A("## 2. 统计检验表（of record；主报指标 = `%s`）" % primary)
    A("")
    A("- **Δ 定义**：`Δ = metric(Ours) − metric(baseline)`。"
      "`f1`／`tpr` 为**越大越好**；`fpr` 为**越小越好**（Δ < 0 才代表 Ours 更优）。")
    A("- **三组对比，无第四行**：`pair_id` 集合恒为预注册三组（可机械核验：数据行数 = %d）。"
      % len(results))
    A("- 逐 seed 的三列 `p` **不是**三次独立检验的校正结果，而是**同一检验在三个 Ours run 上各做一遍**"
      "（防单种子偶然）；`p_range` = 三者的极差。")
    A("")
    fields, rows = build_of_record_rows(cfg, results, primary)
    hdr = ["对比", "分片", "n", "n_boot"]
    for s in seeds:
        hdr += ["Δ_s%s" % s, "p_s%s" % s, "95%%CI_s%s" % s]
    hdr += ["p_range"]
    trows = []
    for row in rows:
        cells = [row["pair_id"], row["split"], row["n_pairs"], row["n_boot"]]
        for s in seeds:
            cells += [row["delta_s%s" % s], row["p_s%s" % s],
                      "[%s, %s]" % (row["ci_lo_s%s" % s], row["ci_hi_s%s" % s])]
        cells += [row["p_range"]]
        trows.append(cells)
    A(_md_table(hdr, trows))
    A("")
    A("### 2.1 逐 seed 点估计（Ours ／ baseline）")
    A("")
    hdr2 = ["对比", "分片", "指标"]
    for s in seeds:
        hdr2 += ["Ours_s%s" % s, "base_s%s" % s]
    t2 = []
    for row in rows:
        cells = [row["pair_id"], row["split"], primary]
        for s in seeds:
            cells += [row["ours_m_s%s" % s], row["base_m_s%s" % s]]
        t2.append(cells)
    A(_md_table(hdr2, t2))
    A("")
    A("### 2.2 逐 seed 一致性观察（**自动标注**，回退点「极差过大 ⇒ 如实报告」）")
    A("")
    A("判据（事前定值，`seed_flag_for()`）：先看 **Δ 的符号**是否三 seed 一致；"
      "一致再看 `p` 是否跨过 `α = 0.05`。四选一标注，**不做**任何择优取舍。")
    A("")
    mf, mrows = build_allmetric_rows(cfg, results, blocked_reasons)
    flag_rows = []
    for r in results:
        for metric, per_seed in sorted(r["per_metric"].items(),
                                       key=lambda kv: METRIC_ORDER.index(kv[0])):
            ds = [x["delta"] for x in per_seed]
            flag = seed_flag_for(ds, [x["p"] for x in per_seed])
            flag_rows.append([r["comp"]["id"], metric, flag,
                              fmt_num(min(ds)), fmt_num(max(ds)), fmt_num(max(ds) - min(ds)),
                              "是" if flag in ("符号不一致", "显著性跨 seed 翻转") else "否"])
    A(_md_table(["对比", "指标", "seed 一致性标注", "Δ 最小", "Δ 最大", "Δ 极差", "需谨慎解读"],
                flag_rows))
    A("")
    warn = [r for r in flag_rows if r[6] == "是"]
    if warn:
        A("**⚠ 上表 %d 行的结论随 seed 变化** ⇒ 依 §8 回退点「三 seed 的 bootstrap 结果极差过大」，"
          "**如实报告三 seed 各自结果与极差，不取其一、不择优选报**。该类格子在论文中"
          "**只能**表述为「随 seed 变化」，**不得**用任一单 seed 的 `p` 值下单一结论。" % len(warn))
    else:
        A("（本次三组对比的全部指标在三 seed 上结论一致，无需谨慎解读标注。）")
    A("")
    A("## 3. 全指标明细（`--metric all`）")
    A("")
    A(_md_table(["对比", "指标", "seed", "Ours", "baseline", "Δ", "95% CI", "p", "方向", "seed 标注"],
                [[r["pair_id"], r["metric"], r["seed"], r["ours_m"], r["base_m"], r["delta"],
                  "[%s, %s]" % (r["ci_lo"], r["ci_hi"]), r["p"], r["direction"], r["seed_flag"]]
                 for r in mrows]))
    A("")
    A("## 4. 不可比（`n/a`）指标 —— C4 拦截")
    A("")
    if blocked_reasons:
        A(_md_table(["指标", "渲染", "理由"],
                    [[m, NA, (r or "").strip()] for m, r in blocked_reasons]))
    else:
        A("（本次请求未触及 `n/a` 指标；`auroc`／`span_f1` 在任何情况下均记 `n/a`，不入 bootstrap。）")
    A("")
    A("### 4.1 `auroc` 恒 `n/a` 的口径依据（C6：引用带口径）")
    A("")
    A("> 引 `reports/baseline/BASELINE_REPORT.md` §1 注释 3（原文）："
      "「**AUROC 记 `n/a`**：B1／B3／B5 **无连续 `score`**（规则型／判词型二值输出）"
      "⇒ 依 `W4.6_unify.md` §4.2 不计算，非缺失。」")
    A(">")
    A("> 本窗口**实测复核**：`b3_*_test_*.csv`／`b5_*_test_*.csv` 的 `score` 列**全空**，"
      "与该注释一致 ⇒ 配对两侧不可比 ⇒ 依 C4 **不进入** bootstrap，该格 `n/a` ＋ 本声明，"
      "**未**强行计算。")
    A("")
    A("## 5. 配对核验表（C3）")
    A("")
    A(_md_table(["对比", "seed", "Ours 记录数", "baseline 记录数", "配对数",
                 "仅 Ours 有", "仅 baseline 有", "text_sha256 不一致"],
                [[r["pair_id"], r["seed"], r["ours_n"], r["baseline_n"], r["n_pairs"],
                  r["only_ours"], r["only_base"], r["sha_mismatch"]]
                 for r in paths["pairing_rows"]]))
    A("")
    A("- 差集**必须为空**；`text_sha256` 不一致**必须为 0**。任一非零 ⇒ 脚本**非零退出**（C3）。")
    A("- **label 来源**：baseline 逐样本 CSV 无 `label` 列 ⇒ 在 `sample_id` 与 `text_sha256`"
      "**双重校验通过后**，label 取 Ours 侧导出列（`configs/eval/stat_bootstrap.yaml` → "
      "`pairing.label_source = ours`）。")
    A("")
    A("## 6. 输入锚点（md5，C7：只消费既有产物 CSV）")
    A("")
    A(_md_table(["文件", "md5"],
                [[name, md5] for name, md5 in paths["anchors"]]))
    A("")
    A("## 7. Limitation（#51／#52／#53 合并段）")
    A("")
    A("1. **`#51` 两体制 val 指标不独立**：`id_val` 与 `ood_val` 共享同一批验证文本"
      "（`DATA_CARD` 的 `V-2`＋`V-6`）。**边界**：test 侧 `id_test`／`ood_test` 相互独立"
      "（4 个留出码只在 test）⇒ **正式结论仍以 test 为准**；本条只削弱 val 侧独立证据力。")
    A("2. **`#52` τ 取值**：Ours 的 τ 逐 run 取该 run `best epoch` 的 `tau_ours`（全精度），"
      "`0.5` 并列；本窗口只读既有 `pred` 列，**未重新寻优、未选阈值、未选 checkpoint**。")
    A("3. **`#53` τ 样本内不对称**：Ours 的 `tau_ours` 是在**同一 val 分片**上按 F1 最大寻得"
      " ⇒ Ours 的 val 指标属**样本内**；基座为判词型、无 τ ⇒ 该不对称使 val 轨道 `ΔF1`"
      "**偏有利于 Ours**。**不得**据此调整阈值；**须优先以 clean 轨道结论为准**。")
    A("4. **对本窗口的直接影响**：上述三条只影响 **val 侧**的解读；本窗口的 bootstrap 在 "
      "`id_test`／`ood_test` 上进行，两侧均**未在 test 上寻 τ** ⇒ bootstrap 的 Δ 不受"
      "「同分片寻优」污染。但 **Ours 的 test 数值仍沿用 val 上寻得的 τ** ⇒ "
      "τ 本身的值仍带样本内成分，属已知局限，**不在本窗口校正**。")
    A("")
    A("## 8. 产物清单")
    A("")
    A(_md_table(["产物", "路径"], [[k, v] for k, v in paths["products"]]))
    A("")
    A("---")
    A("")
    A("**本窗口未读 `data/sealed_test/**` ⇒ ACCESS_LOG 零行登记**（C7）。"
      "本窗口不重跑任何模型、不做检测／OOD／消融／效率（`W6.1`／`W6.4`／`W6.7`／`W6.8`）。")
    return "\n".join(L) + "\n"


def cmd_run(cfg, args, logger):
    if not args.report_dir:
        print("MISSING_DIRS：`--report-dir` 必填（C9：禁止硬编码报告路径）")
        return 2
    if not args.ours_outputs or not args.baseline_outputs:
        print("MISSING_INPUTS：须给 `--ours-outputs` 与 `--baseline-outputs`")
        return 2
    ts = args.ts
    report_dir = _abs(args.report_dir)
    os.makedirs(report_dir, exist_ok=True)

    comps = resolve_comparisons(cfg, args.pairs)
    n_boot = assert_n_boot(cfg, args.n_boot)
    logger("N_BOOT=%d（预注册定值；命令行一致）" % n_boot)
    primary = ((cfg.get("metrics") or {}).get("primary") or "f1")
    metrics, blocked = resolve_metrics(cfg, args.metric)
    if primary not in metrics:
        primary = metrics[0]
    logger("METRICS computable=%s blocked=%s" % (metrics, [m for m, _ in blocked]))

    results = []
    for comp in comps:
        logger("PAIR begin pair=%s baseline=%s split=%s" % (comp["id"], comp["baseline"], comp["split"]))
        results.append(run_comparison(cfg, args, comp, metrics, blocked, logger))

    pairing_rows = [r for res in results for r in res["pairing_rows"]]
    anchors = []
    seen = set()
    for r in pairing_rows:
        for name, md5 in ((r["ours_file"], r["ours_md5"]), (r["baseline_file"], r["baseline_md5"])):
            if name not in seen:
                seen.add(name)
                anchors.append((name, md5))

    fields, rows = build_of_record_rows(cfg, results, primary)
    mf, mrows = build_allmetric_rows(cfg, results, blocked)
    blocked_reasons = results[0]["blocked"] if results else []

    base = "stat_bootstrap_v1.0_%s" % ts
    md_path = os.path.join(report_dir, base + ".md")
    csv_path = os.path.join(report_dir, base + ".csv")
    pair_path = os.path.join(report_dir, "stat_pairing_check_v1.0_%s.csv" % ts)
    all_path = os.path.join(report_dir, "stat_bootstrap_allmetrics_v1.0_%s.csv" % ts)
    for p, what in ((md_path, "报告"), (csv_path, "of record CSV"),
                    (pair_path, "配对核验表"), (all_path, "全指标明细 CSV")):
        assert_not_exists(p, what)

    pair_fields = ["pair_id", "seed", "split", "ours_file", "ours_md5",
                   "baseline_file", "baseline_md5", "ours_n", "baseline_n",
                   "n_pairs", "only_ours", "only_base", "sha_mismatch",
                   "diff_only_ours", "diff_only_base"]
    write_csv_rows(pair_path, pair_fields, pairing_rows)
    write_csv_rows(csv_path, fields, rows)
    write_csv_rows(all_path, mf, mrows)

    products = [("统计检验报告", md_path), ("统计检验表 of record（3 行）", csv_path),
                ("配对核验表", pair_path), ("全指标明细 CSV", all_path),
                ("运行日志", paths_log_hint(ts))]
    text = render_report(cfg, args, results, ts,
                         {"pairing_rows": pairing_rows, "anchors": anchors, "products": products},
                         primary, blocked_reasons)
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            raise ConstraintError("报告含禁止表述 %r（§4-5：不得把 p>0.05 说成等价）" % phrase)
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    logger("WROTE %s / %s / %s / %s" % (md_path, csv_path, pair_path, all_path))
    print("REPORT_MD=%s" % md_path.replace("\\", "/"))
    return 0


def paths_log_hint(ts):
    return "logs/eval/05_08_stat_bootstrap_%s.log" % ts


# ---------------------------------------------------------------------------
# --dry-run：只做配对核验，不做 bootstrap
# ---------------------------------------------------------------------------
def cmd_dryrun(cfg, args, logger):
    if not args.ours_outputs or not args.baseline_outputs:
        print("MISSING_INPUTS：须给 `--ours-outputs` 与 `--baseline-outputs`")
        return 2
    comps = resolve_comparisons(cfg, args.pairs)
    assert_n_boot(cfg, args.n_boot)
    n = 0
    for comp in comps:
        base_path = resolve_baseline_path(cfg, _abs(args.baseline_outputs), comp)
        if not os.path.isfile(base_path):
            logger("MISSING baseline=%s" % base_path)
            return 3
        for seed in (cfg.get("boot_seeds") or [42, 123, 456]):
            ours_path = resolve_ours_path(cfg, _abs(args.ours_outputs), comp, seed)
            if not os.path.isfile(ours_path):
                logger("MISSING ours=%s" % ours_path)
                return 3
            _, _, _, check = load_pairs(cfg, ours_path, base_path,
                                        (cfg.get("tau") or {}).get("ours_column", "pred"))
            logger("DRYRUN pair=%s seed=%s n_pairs=%d only_ours=%d only_base=%d sha_mismatch=%d"
                   % (comp["id"], seed, check["n_pairs"], check["only_ours"],
                      check["only_base"], check["sha_mismatch"]))
            n += 1
    print("DRYRUN_RESULT=PASS n=%d" % n)
    return 0


# ---------------------------------------------------------------------------
# --self-check
# ---------------------------------------------------------------------------
def _run(argv, expect_zero, checks, name, cwd=None):
    """跑一次子进程并断言退出码；`expect_zero=False` 时期望**非零**。"""
    proc = subprocess.run([sys.executable] + argv, cwd=cwd or _ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", errors="replace")
    ok = (proc.returncode == 0) if expect_zero else (proc.returncode != 0)
    checks.append((name, ok, "exit=%d" % proc.returncode))
    if not ok:
        # 失败时回显子进程尾部输出，便于定位（**不静默**）
        for line in out.strip().split("\n")[-15:]:
            print("    | %s" % line)
    return ok, out


def _mk_pair_csvs(tmp, cfg, ours_rows, base_rows, comp, seed):
    """在 tmp 下按**真实文件名**造一对 CSV（文件名取自配置 ⇒ 无需额外 CLI 参数）。"""
    ours_dir = os.path.join(tmp, "ours")
    base_dir = os.path.join(tmp, "base")
    os.makedirs(ours_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    ts = (cfg.get("inputs") or {}).get("ours_ts", {})[comp["split"]]
    ours_name = str((cfg.get("inputs") or {}).get("ours_file")).format(
        regime=comp["regime"], seed=seed, split=comp["split"], ts=ts)
    base_name = (cfg.get("inputs") or {}).get("baseline", {})[comp["baseline"]][comp["split"]]
    ours_path = os.path.join(ours_dir, ours_name)
    base_path = os.path.join(base_dir, base_name)
    ofields = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score",
               "latency_ms", "label", "pred_parallel", "tau_main", "tau_parallel"]
    bfields = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score", "latency_ms"]
    write_csv_rows(ours_path, ofields, ours_rows)
    write_csv_rows(base_path, bfields, base_rows)
    return ours_dir, base_dir


def _synth(n_pos, n_neg, ours_correct, base_correct, seed=7):
    """造逐样本行：Ours／baseline 各按给定正确率生成 `pred`。"""
    import random

    rng = random.Random(seed)
    ours_rows, base_rows = [], []
    idx = 0
    for lab, cnt in ((1, n_pos), (0, n_neg)):
        for _ in range(cnt):
            idx += 1
            sid = "S-%04d" % idx
            sha = "%064x" % (idx * 7919)
            po = lab if rng.random() < ours_correct else (1 - lab)
            pb = lab if rng.random() < base_correct else (1 - lab)
            ours_rows.append({"sample_id": sid, "text_sha256": sha, "pred": po,
                              "raw_type": "synthetic", "raw": "", "score": "",
                              "latency_ms": "0", "label": lab, "pred_parallel": po,
                              "tau_main": "0.5", "tau_parallel": "0.5"})
            base_rows.append({"sample_id": sid, "text_sha256": sha, "pred": pb,
                              "raw_type": "synthetic", "raw": "", "score": "",
                              "latency_ms": "0"})
    return ours_rows, base_rows


def self_check(cfg, args, logger=print):
    import tempfile

    checks = []
    script = os.path.join(_ROOT, "scripts", "eval", "05_08_stat_bootstrap.py")
    cfg_path = _abs(args.config or "configs/eval/stat_bootstrap.yaml")
    comp = (cfg.get("comparisons") or [{}])[0]
    tmp = tempfile.mkdtemp(prefix="w69_selfcheck_")

    # ① 配置定值
    checks.append(("配置：三组对比", len(cfg.get("comparisons") or []) == 3,
                   "n=%d" % len(cfg.get("comparisons") or [])))
    checks.append(("配置：n_boot=10000", int(cfg.get("n_boot", -1)) == 10000, str(cfg.get("n_boot"))))
    checks.append(("配置：boot_seeds=42/123/456",
                   list(cfg.get("boot_seeds") or []) == [42, 123, 456], str(cfg.get("boot_seeds"))))

    # ② C1 非预注册组合 ⇒ 非零退出
    for pair, split, name in (("ours:b1", "id_test", "C1 拦 b1"),
                              ("ours:b2", "id_test", "C1 拦 b2（平凡解 τ=−inf）"),
                              ("ours:b4", "id_test", "C1 拦 b4（非预注册）"),
                              ("ours:b3", "clean_test", "C1 拦 clean_test 分片"),
                              ("ours:b5", "ood_test", "C1 拦 ours:b5@ood（非预注册）")):
        _run([script, "--pair", pair, "--split", split, "--dry-run",
              "--ours-outputs", tmp,
              "--baseline-outputs", tmp,
              "--config", cfg_path], False, checks, name)

    # ③ C2 n_boot 改动 ⇒ 非零退出
    _run([script, "--pair", "ours:b3", "--split", "id_test", "--dry-run",
          "--ours-outputs", tmp,
          "--baseline-outputs", tmp,
          "--n-boot", "5000", "--config", cfg_path], False, checks, "C2 n_boot 改动 ⇒ 非零")

    # ④ C3 配对缺失／多余／sha 不一致 ⇒ 各非零退出
    #    （dry-run 会遍历**全部** boot seeds ⇒ 三个 seed 的文件都要齐）
    ours_rows, base_rows = _synth(40, 40, 0.95, 0.70)
    ours_dir = base_dir = None
    for _s in (cfg.get("boot_seeds") or [42, 123, 456]):
        o_rows, b_rows = _synth(40, 40, 0.95, 0.70, seed=_s)
        ours_dir, base_dir = _mk_pair_csvs(tmp, cfg, o_rows, b_rows, comp, _s)
    base_argv = [script, "--pair", "ours:b3", "--split", "id_test", "--dry-run",
                 "--ours-outputs", ours_dir, "--baseline-outputs", base_dir,
                 "--config", cfg_path]
    _run(base_argv, True, checks, "C3 构造样本：完整配对 ⇒ exit=0")

    miss_dir = os.path.join(tmp, "miss")
    m_ours, m_base = _mk_pair_csvs(miss_dir, cfg, ours_rows, base_rows[:-3], comp, 42)
    _run([script, "--pair", "ours:b3", "--split", "id_test", "--dry-run",
          "--ours-outputs", m_ours, "--baseline-outputs", m_base,
          "--config", cfg_path], False, checks, "C3 缺失 3 条 ⇒ 非零")

    extra_dir = os.path.join(tmp, "extra")
    e_ours, e_base = _mk_pair_csvs(extra_dir, cfg, ours_rows,
                                   base_rows + base_rows[:2], comp, 42)
    _run([script, "--pair", "ours:b3", "--split", "id_test", "--dry-run",
          "--ours-outputs", e_ours, "--baseline-outputs", e_base,
          "--config", cfg_path], False, checks, "C3 baseline 多余（重复键）⇒ 非零")

    sha_dir = os.path.join(tmp, "sha")
    bad_base = [dict(r) for r in base_rows]
    bad_base[5]["text_sha256"] = "0" * 64
    s_ours, s_base = _mk_pair_csvs(sha_dir, cfg, ours_rows, bad_base, comp, 42)
    _run([script, "--pair", "ours:b3", "--split", "id_test", "--dry-run",
          "--ours-outputs", s_ours, "--baseline-outputs", s_base,
          "--config", cfg_path], False, checks, "C3 text_sha256 不一致 ⇒ 非零")

    # ⑤ 正式跑（合成数据）：三组齐备、声明齐备、C4 拦截、三 seed 极差
    rep = os.path.join(tmp, "rep")
    os.makedirs(rep, exist_ok=True)
    syn_ours = syn_base = None
    for c in (cfg.get("comparisons") or []):
        for seed in (cfg.get("boot_seeds") or [42, 123, 456]):
            o, b = _synth(40, 40, 0.95, 0.70, seed=seed)
            syn_ours, syn_base = _mk_pair_csvs(tmp, cfg, o, b, c, seed)
    argv = [script]
    for c in (cfg.get("comparisons") or []):
        argv += ["--pair", "%s:%s" % (c["ours"], c["baseline"]), "--split", c["split"]]
    argv += ["--ours-outputs", syn_ours, "--baseline-outputs", syn_base,
             "--config", cfg_path, "--report-dir", rep, "--metric", "all"]
    ok, out = _run(argv, True, checks, "正式跑（合成）：三组 × 三 seed × 全指标 ⇒ exit=0")
    if ok:
        md = os.path.join(rep, [f for f in os.listdir(rep) if f.endswith(".md")][0])
        text = open(md, encoding="utf-8").read()
        for name, kw in DECLARATIONS:
            checks.append(("声明：%s" % name, kw in text, kw))
        for phrase in FORBIDDEN_PHRASES:
            checks.append(("禁止表述：%r 未出现" % phrase, phrase not in text, phrase))
        csvf = os.path.join(rep, [f for f in os.listdir(rep)
                                  if f.startswith("stat_bootstrap_v1.0_") and f.endswith(".csv")][0])
        with open(csvf, "r", encoding="utf-8-sig", newline="") as fh:
            recs = list(csv.DictReader(fh))
        ids = [r["pair_id"] for r in recs]
        want = sorted(c["id"] for c in (cfg.get("comparisons") or []))
        checks.append(("of record 恰 3 行", len(recs) == 3, "n=%d" % len(recs)))
        checks.append(("of record 无第四对比", sorted(ids) == want, str(sorted(ids))))
        checks.append(("三 seed 极差列存在", "p_range" in (recs[0] if recs else {}), "p_range"))
        allf = os.path.join(rep, [f for f in os.listdir(rep)
                                  if f.startswith("stat_bootstrap_allmetrics")][0])
        with open(allf, "r", encoding="utf-8-sig", newline="") as fh:
            arecs = list(csv.DictReader(fh))
        auroc = [r for r in arecs if r["metric"] == "auroc"]
        checks.append(("C4 auroc 记 n/a", bool(auroc) and all(r["p"] == NA for r in auroc),
                       "rows=%d" % len(auroc)))
        checks.append(("C4 auroc 未进 bootstrap",
                       bool(auroc) and all(r["n_boot"] == "" for r in auroc), "n_boot 空"))

    # ⑥ 脚本自律（C7／C9）
    #  C7 用**配置断言**核验（报告正文需引用该路径作声明，grep 脚本源码不可行）
    checks.append(("C7 配置 `sealed.read` 为 false",
                   (cfg.get("sealed") or {}).get("read") is False,
                   str((cfg.get("sealed") or {}).get("read"))))
    #  C9：脚本源码内不得出现硬编码的 reports/ 输出路径（输入／输出根目录一律由命令行给）
    #  注：针型字符串**拆开拼接**构造，避免本行自身命中（`D-21`：计数类断言须写明并检验过滤条件）
    src = open(script, encoding="utf-8").read()
    needle = "reports" + "/eval" + "/"
    checks.append(("C9 无硬编码 reports/ 输出路径", needle not in src, "grep=0"))

    bad = [c for c in checks if not c[1]]
    for name, ok_, detail in checks:
        logger("SELFCHECK %s %s（%s）" % ("PASS" if ok_ else "FAIL", name, detail))
    print("SELFCHECK_RESULT=%s n=%d" % ("PASS" if not bad else "FAIL", len(checks)))
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="scripts/eval/05_08_stat_bootstrap.py",
        description="5.8 统计严谨性（W6.9，A6）—— 预注册三组的 paired bootstrap 检验")
    p.add_argument("--pair", action="append", default=[],
                   help="形如 ours:b3（可重复给多次）")
    p.add_argument("--split", action="append", default=[],
                   help="与 --pair 一一对应，形如 id_test")
    p.add_argument("--ours-outputs", dest="ours_outputs", default=None)
    p.add_argument("--baseline-outputs", dest="baseline_outputs", default=None)
    p.add_argument("--n-boot", dest="n_boot", type=int, default=None,
                   help="**必须与配置一致**（C2 定值 10000）")
    p.add_argument("--boot-seeds", dest="boot_seeds", default=None,
                   help="逗号分隔；与配置一致（42,123,456）")
    p.add_argument("--config", default="configs/eval/stat_bootstrap.yaml")
    p.add_argument("--report-dir", dest="report_dir", default=None,
                   help="**必填**（C9：不得硬编码 reports/ 路径）")
    p.add_argument("--metric", default=None, choices=METRIC_ORDER + ["all"],
                   help="默认主报 f1；`all` = f1/tpr/fpr/auroc（`auroc` 恒 n/a）")
    p.add_argument("--self-check", dest="self_check", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def parse_pairs(args):
    """`--pair` 与 `--split` 一一配对；数量不等 ⇒ 视为 CLI 用法错误。"""
    if len(args.pair) != len(args.split):
        raise ConstraintError("`--pair` 与 `--split` 数量不等（%d vs %d）⇒ 无法一一配对"
                              % (len(args.pair), len(args.split)))
    out = []
    for pr, sp in zip(args.pair, args.split):
        if ":" not in pr:
            raise ConstraintError("--pair 须形如 ours:b3，收到 %r" % pr)
        ours, baseline = pr.split(":", 1)
        out.append((ours.strip(), baseline.strip(), sp.strip()))
    return out


def main(argv=None):
    init_streams()
    args = build_parser().parse_args(argv)
    args.ts = timestamp()
    cfg = load_config(_abs(args.config))
    if args.boot_seeds:
        given = [int(x) for x in str(args.boot_seeds).split(",") if x.strip()]
        if given != list(cfg.get("boot_seeds") or []):
            print("REFUSED：`--boot-seeds` 与配置不一致（配置 %s，收到 %s）⇒ 非零退出"
                  % (cfg.get("boot_seeds"), given))
            return 2
    if args.self_check:
        return self_check(cfg, args, print)
    try:
        args.pairs = parse_pairs(args)
    except ConstraintError as exc:
        print("REFUSED：%s" % exc)
        return 2
    if args.dry_run:
        try:
            return cmd_dryrun(cfg, args, print)
        except (ConstraintError, PairingError) as exc:
            print("REFUSED：%s" % exc)
            return 3
    if not args.pairs:
        print("缺少 --pair/--split（或给 --self-check／--dry-run）")
        return 2
    logger, log_path = flagged_logger(args.ts, log_dir=resolve_log_dir(args))
    try:
        rc = cmd_run(cfg, args, logger)
    except (ConstraintError, PairingError) as exc:
        logger("REFUSED：%s" % exc)
        print("REPORT_MD=<none>")
        return 3
    logger("LOG_FILE=%s" % log_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())

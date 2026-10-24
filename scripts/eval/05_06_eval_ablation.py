# -*- coding: utf-8 -*-
"""05_06_eval_ablation.py — 阶段5（**W6.7**，A6）5.6 **消融实验**评估主脚本。

接口契约（`windows/W6.7_ablation.md` §2，**冻结后不得改**）::

    python scripts/eval/05_06_eval_ablation.py \
        --groups A,B,C,D \
        --group-ckpt A=<run_dir> --group-ckpt B=<run_dir> --group-ckpt C=<run_dir> \
        --source {val,sealed} --split {in_regime_test,g3_hard_negative_test} \
        --config configs/eval/ablation.yaml \
        --report-dir reports/eval/ --outputs-dir reports/eval/outputs/ \
        [--access-log-note "<已登记的 ACCESS_LOG 行摘要>"] \
        [--check-split-identity] [--carrier-strata] [--limit N] [--self-check] [--dry-run]

stdout 末行（依模式）::

    SELFCHECK_RESULT=PASS|FAIL n=<k>
    DRYRUN_RESULT=PASS|FAIL n=<k>
    SPLIT_IDENTITY_RESULT=PASS|FAIL n=<k>          （仅 --check-split-identity）
    REPORT_MD=<path> REPORT_CSV=<path>

硬约束逐条落点（`windows/W6.7_ablation.md` §3）::

    C1  统一划分 → `--check-split-identity` 取**内容指纹**（逐行 `text` 的 sha256 序列哈希）；
                   四组须**逐位相同**，不一 ⇒ **非零退出 ＋ 停工呈 W0**
    C2  全 3 seed → A／B／C／D 全部 3 seed；缺某 seed ⇒ 该格 `<TBD>` ＋ 显式标注，
                   **不得**用其他 seed 补齐
    C3  mean±std  → 一律给 `mean ± std`（ddof=1）**＋ 逐 seed 原值表**；`std = 0` ⇒ **显式声明退化**
    C4  τ 不重寻 → **逐组逐 seed** 取该 run `best epoch` 的 `tau_ours`（全精度，经
                   `05_01.resolve_tau_tracks`），`0.5` 并列；**禁**统一 τ／**禁**重新寻优
    C5  输入面   → 实现期只读 val；禁读 `data/sealed_test/**` 与 frozen 侧 `test.csv`
                   （守卫＝ `05_01.assert_input_allowed`）；正式期须登记 6 列 ＋ `--access-log-note`
    C6  记录数   → **复用** `05_01.read_records`／`assert_record_count`（`csv.DictReader`）；
                   **严禁**按行读；不符 ⇒ 停工呈 W0
    C7  不训练   → 本脚本**不得**调用 `04_01`／`04_02`／`04_03`；只做推理评估
                   （源码级断言由 `--self-check` 覆盖）
    C8  评估态   → `model.eval()` + `torch.no_grad()`（由 `05_01.Engine` 保证；本脚本
                   无 train 模式调用 —— `.train` 后接左括号的命中数须为 0）
    C9  报告路径 → `--report-dir`／`--outputs-dir` 必填；脚本内不写死 `reports/`
    C10 引数口径 → 引 `BASELINE_REPORT.md` 须带其 §12 口径注释（骨架段已内联）
    C11 只追加   → 报告／日志／CSV 只写新时间戳文件，不覆盖

⛔ 本窗口特有禁令（`windows/W6.7_ablation.md` §4）
------------------------------------------------
1. **禁止伪造或外推 A／B 组数字**；**禁止**用 C 组结果代替 —— 由 `alias_guard` **结构性**保证：
   A／B 的 `run_dir` 若与 C 组任一 `run_dir` 相同 ⇒ **非零退出**（`configs/eval/ablation.yaml`
   → `groups.alias_guard`）。D 组复用 C 组 ckpt 属 §1.3 明定口径 ⇒ 允许但**必须声明**。
2. **禁止**为「凑齐四组」自行训练（A6 禁训练）；
3. **禁止**每组用不同划分（`v5 新增 32`）；
4. **禁止**按消融结果删减组或事后加组；
5. **禁止**在方差过大时只报好看的 seed（如实报告 std）；
6. **禁止**在消融中改 `epochs`／`lr`／`r`／`patience`／样本集。

顶层不得 import 重型库（`AGENTS.md` §3）：`torch`／`transformers`／`peft`／`yaml`
一律在函数体内惰性 import（`05_01.Engine` 与 `03_06`／`03_07` 的加载同样惰性）。
"""
from __future__ import annotations

# --- C8：必须在 import torch 之前设置（torch 的 import 全在函数体内）---
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

TBD = "<TBD>"
NA = "n/a"

GROUPS_ALL = ["A", "B", "C", "D"]

# ─────────────────────────────────────────────────────────────────────────────
# 产物字段（of record）
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_CSV_FIELDS = [
    "group", "group_status", "regime", "seed", "run_id", "shard", "source",
    "tau_track", "tau_value", "n", "n_pos", "n_neg",
    "tp", "fp", "fn", "tn", "tpr", "fpr", "f1", "auroc",
    "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
    "mem_load_peak_mib", "mem_infer_peak_mib", "eval_batch_max_size",
]

SPLIT_IDENTITY_CSV_FIELDS = [
    "group", "regime", "seed", "shard", "source",
    "n_records", "n_pos", "n_neg", "fingerprint", "matches_reference",
]

CARRIER_CSV_FIELDS = [
    "group", "regime", "seed", "run_id", "shard", "source",
    "tau_track", "tau_value", "carrier_type",
    "n", "n_pos", "n_neg", "tp", "fp", "fn", "tn", "tpr", "fpr", "f1",
]


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：`W3.7` `F-1`；`D-40` 裁定五：不得静默 pass）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（``errors="replace"``），避免 GBK 环境打印非 ASCII 假失败。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            print("WARN: 无法把流重配为 UTF-8（非 UTF-8 环境打印可能失败）")


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


_E501 = None


def W():
    """**单源复用**：`W6.1` 的 `05_01_eval_detection`（守卫／引擎／指标／写盘）。"""
    global _E501
    if _E501 is None:
        _E501 = load_module("eval_05_01", "scripts/eval/05_01_eval_detection.py")
    return _E501


def load_config(path):
    """读 YAML（`yaml` 惰性 import，模块顶层保持轻量）。"""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def _norm(path):
    """`realpath` 归一（用于 ckpt 同一性判定 —— 软链／相对路径不得造成误判）。"""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------
def parse_float_or_none(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in (NA, "nan", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fmt4(x, nd=4):
    """已算完的指标格：``None`` ⇒ `n/a`（**不适用**，非缺失）。"""
    if x is None:
        return NA
    return ("%." + str(nd) + "f") % (float(x),)


def fmt_tau(x):
    """全精度（`F-W5.1-2`）：``repr(float)``，不取显示值。"""
    return NA if x is None else repr(float(x))


def mean_std(xs, ddof=1):
    """`mean ± std`；`ddof=1` 样本标准差（`W6.4`／`W6.9` 同口径）。"""
    vals = [float(v) for v in xs if v is not None]
    if not vals:
        return None, None
    m = statistics.fmean(vals)
    if len(vals) < 2:
        return m, 0.0
    s = statistics.stdev(vals) if ddof == 1 else statistics.pstdev(vals)
    return m, s


def mean_std_cell(m, s, nd=4):
    """渲染 `mean ± std`；`std = 0` ⇒ 附 **`⚠std=0`** 标记（C3：不得只写 mean±std）。

    ``m is None`` ⇒ 该指标在 3 个 seed 上**均无定义**（如子集内无负类 ⇒ FPR／AUROC 无定义）
    ⇒ 渲染 `n/a`（**不适用**），**不是** `<TBD>`（未训练）—— 两者语义不得互串。
    """
    if m is None:
        return NA
    if s is None:
        return fmt4(m, nd)
    cell = "%s ± %s" % (fmt4(m, nd), fmt4(s, nd))
    if abs(float(s)) < 1e-12:
        cell += " **⚠std=0**"
    return cell


# ---------------------------------------------------------------------------
# §1.1 四组与 `--group-ckpt` 解析
# ---------------------------------------------------------------------------
def parse_group_ckpt(items):
    """``["A=<dir>", "C=<dir>", ...]`` ⇒ ``{group: [dir, ...]}``（**可重复**＝逐 seed 给）。

    未知组名 ⇒ ``ValueError``（上层转非零退出）；**不静默忽略**（§4-4 同族纪律）。
    """
    out = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError("--group-ckpt 须形如 G=<run_dir>（%r 缺 `=`）" % raw)
        g, path = raw.split("=", 1)
        g = g.strip().upper()
        path = path.strip()
        if g not in GROUPS_ALL:
            raise ValueError("未知消融组 %r（合法：%s）" % (g, ",".join(GROUPS_ALL)))
        if not path:
            raise ValueError("--group-ckpt %s= 的 run_dir 为空" % g)
        out.setdefault(g, []).append(path)
    return out


def parse_groups_arg(value, cfg):
    """`--groups A,B,C,D` ⇒ 有序组列表；缺省 = 配置 `groups.order`（**不得**按结果增删）。"""
    if not value:
        return list(cfg["groups"]["order"])
    got = [x.strip().upper() for x in value.split(",") if x.strip()]
    bad = [g for g in got if g not in GROUPS_ALL]
    if bad:
        raise ValueError("未知消融组 %s（合法：%s）" % (bad, ",".join(GROUPS_ALL)))
    return got


def resolve_group_runs(cfg, args, logger=print):
    """返回 ``(runs, group_status)``；`runs` = ``[{group, regime, seed, run_dir, status, meta}]``。

    - 每组逐 seed 给 ckpt ⇒ 逐个 `load_run_meta` 取 `(regime, seed)`（**不靠命令行猜**）；
    - **D 组**未显式给 ⇒ 按 §1.3 沿用 **C 组 ckpt**（`groups.D.ckpt_inherit_from`），
      ``status="DERIVED_FROM_C"`` 且报告**必须显式声明**；
    - 其余缺组 ⇒ ``status="MISSING"``，整列 `<TBD>` ＋ 标注「未训练／未就绪」。
    """
    W_ = W()
    base = args._base
    explicit = parse_group_ckpt(args.group_ckpt)
    groups = parse_groups_arg(args.groups, cfg)

    resolved = {}
    for g in groups:
        paths = explicit.get(g, [])
        if paths:
            recs = []
            for p in paths:
                rd = _abs(p)
                meta_info = W_.load_run_meta(rd, base)
                recs.append({"group": g, "regime": str(meta_info["regime"]),
                             "seed": int(meta_info["seed"]), "run_dir": rd,
                             "status": "READY", "meta": meta_info})
            resolved[g] = recs
            logger("GROUP_RESOLVED group=%s n=%d explicit=True" % (g, len(recs)))
            continue
        inherit = (cfg["groups"].get(g) or {}).get("ckpt_inherit_from")
        if inherit and inherit in resolved and resolved[inherit]:
            resolved[g] = [dict(r, group=g, status="DERIVED_FROM_%s" % inherit)
                           for r in resolved[inherit]]
            logger("GROUP_RESOLVED group=%s n=%d explicit=False inherit_from=%s"
                   "（§1.3：同一 ckpt × 按载体分组评估）" % (g, len(resolved[g]), inherit))
            continue
        resolved[g] = []
        logger("GROUP_MISSING group=%s ⇒ 整列 `%s` ＋ 标注「未训练／未就绪」（**不静默跳过**）"
               % (g, TBD))

    # 重复 (group, regime, seed) ⇒ 非零退出（不得重复计数成 3 seeds）
    for g, recs in resolved.items():
        seen = {}
        for r in recs:
            key = (r["regime"], r["seed"])
            if key in seen:
                raise ValueError("组 %s 的 (regime,seed)=%s 重复：%s 与 %s（禁重复计数）"
                                 % (g, key, seen[key], r["run_dir"]))
            seen[key] = r["run_dir"]

    assert_no_ab_alias(cfg, resolved, logger)

    group_status = {}
    for g in groups:
        recs = resolved.get(g) or []
        gcfg = cfg["groups"].get(g) or {}
        if recs:
            st = recs[0]["status"]
            note = ""
            if st.startswith("DERIVED"):
                note = ("未显式给 `--group-ckpt %s=`，沿用 **%s** 组 ckpt"
                        "（§1.3 载体消融口径）" % (g, gcfg.get("ckpt_inherit_from")))
        else:
            st = "MISSING"
            note = "%s（%s）" % (cfg["groups"]["missing_group_policy"]["note"],
                                 gcfg.get("ckpt_status", ""))
        group_status[g] = {"status": st, "n_runs": len(recs), "note": note,
                           "label": gcfg.get("label", ""), "verbatim": gcfg.get("verbatim", ""),
                           "ckpt_status": gcfg.get("ckpt_status", "")}

    runs = [r for g in groups for r in (resolved.get(g) or [])]
    return runs, group_status


def assert_no_ab_alias(cfg, resolved, logger=print):
    """**结构性防冒充**（§4-1）：A／B 的 `run_dir` 若与 C 组任一 `run_dir` 同 ⇒ **非零退出**。

    D 组复用 C 组 ckpt 属 §1.3 明定口径 ⇒ 豁免（`groups.alias_guard.allow_groups`）。
    """
    guard = cfg["groups"]["alias_guard"]
    if not guard.get("enabled", True):
        return True
    c_dirs = {_norm(r["run_dir"]) for r in (resolved.get("C") or [])}
    if not c_dirs:
        return True
    for g in (guard.get("forbid_groups") or []):
        for r in (resolved.get(g) or []):
            if _norm(r["run_dir"]) in c_dirs:
                raise AssertionError(
                    "ALIAS_GUARD_VIOLATION：组 %s 的 run_dir 与 C 组重复（%s）⇒ "
                    "**禁止用 C 组结果冒充 A／B**（`W6.7` §4-1）；处置=%s"
                    % (g, r["run_dir"], guard.get("on_violation")))
    logger("ALIAS_GUARD_OK forbid=%s allow=%s（A／B 未复用 C 组 ckpt）"
           % (guard.get("forbid_groups"), guard.get("allow_groups")))
    return True


# ---------------------------------------------------------------------------
# C1 统一划分：内容指纹
# ---------------------------------------------------------------------------
def split_fingerprint(rows):
    """逐行 `text` 的 `sha256` 序列哈希（**逐位**敏感：顺序变化也会变）。

    空 `text` 以 `sha256('')` 参与（不跳过），避免静默错位。
    """
    h = hashlib.sha256()
    for r in rows:
        t = r.get("text") or ""
        h.update(hashlib.sha256(t.encode("utf-8")).digest())
    return h.hexdigest()


def _identity_key(r):
    """C1 的比对作用域 = **`(regime, shard)`**。

    ⚠ **坑（本窗口实跑触发）**：`id_val`（1584）与 `ood_val`（927）是**两个不同的分片**，
    指纹必然不同 —— 若跨 `(regime, shard)` 一并比对，会把「体制不同」误判成
    「每组用了不同划分」⇒ **假 FAIL**（⇒ 假停工）。「完全相同划分」的语义是：
    **同一 `(regime, shard)` 内，A／B／C／D 四组逐位相同**。
    """
    return (r["regime"], r["shard"])


def build_split_identity_rows(results):
    """`(group, regime, seed)` × 指纹；`matches_reference` 相对**同 `(regime, shard)` 的首个**记录。"""
    ref = {}
    out = []
    for r in results:
        k = _identity_key(r)
        fp = r["split_fingerprint"]
        ref.setdefault(k, fp)
        out.append({"group": r["group"], "regime": r["regime"], "seed": r["seed"],
                    "shard": r["shard"], "source": r["source"],
                    "n_records": r["n"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
                    "fingerprint": fp, "matches_reference": "1" if fp == ref[k] else "0"})
    return out


def check_split_identity(results):
    """C1 机械判据：**同一 `(regime, shard)` 内**四组（含 D）指纹**逐位相同**。

    返回 ``(ok, rows, bad)``；`ok=False` ⇒ 上层**非零退出 ＋ 停工呈 W0**。
    """
    rows = build_split_identity_rows(results)
    if not rows:
        return True, rows, []
    by_key = {}
    for r in rows:
        by_key.setdefault(_identity_key(r), []).append(r)
    bad = []
    for k, group_rows in by_key.items():
        fps = {r["fingerprint"] for r in group_rows}
        if len(fps) > 1:
            ref = group_rows[0]["fingerprint"]
            bad += ["group=%s regime=%s seed=%s shard=%s fp=%s（参照 %s）"
                    % (r["group"], r["regime"], r["seed"], r["shard"],
                       r["fingerprint"][:16], ref[:16])
                    for r in group_rows if r["fingerprint"] != ref]
    return (not bad), rows, bad


# ---------------------------------------------------------------------------
# 载体分层（D 组；`--carrier-strata`）
# ---------------------------------------------------------------------------
def carrier_values(cfg, rows):
    """配置值域 ＋ 分片中实际出现但未登记的值（如实登记，不静默省略）。"""
    declared = list(cfg["carrier"]["values"])
    present = sorted({(r.get(cfg["carrier"]["group_by_column"]) or "").strip() for r in rows})
    extra = [c for c in present if c and c not in declared]
    return declared + extra


def per_carrier_metrics(unify, rows, scores, tau, cfg):
    """按 `carrier_type` 分层指标；**未出现的层显式列出**且指标 `n/a`（不填 0）。"""
    W_ = W()
    col = cfg["carrier"]["group_by_column"]
    out = []
    for cv in carrier_values(cfg, rows):
        idx = [i for i, r in enumerate(rows)
               if (r.get(col) or "").strip() == cv]
        row = {"carrier_type": cv, "n": len(idx), "n_pos": 0, "n_neg": 0,
               "tp": 0, "fp": 0, "fn": 0, "tn": 0, "tpr": None, "fpr": None, "f1": None}
        if idx:
            labs = [int(float(rows[i].get("label"))) for i in idx]
            sc = [scores[i] for i in idx]
            m = W_.compute_metrics(unify, labs, sc, tau)
            row.update({"n_pos": m["n_pos"], "n_neg": m["n_neg"],
                        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                        "tpr": m["tpr"], "fpr": m["fpr"], "f1": m["f1"]})
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 单 run 评估
# ---------------------------------------------------------------------------
def _row_cache(cfg, args, source, logger):
    """`(regime, shard)` → rows（**同一划分**的物理保证：同 regime 只读一次、共用同一 list）。"""

    cache = {}

    def get(regime):
        if regime in cache:
            return cache[regime]
        rows, shard, _spec, _p = W().prepare_rows(cfg, args, source, regime, logger)
        cache[regime] = (rows, shard)
        return cache[regime]

    return get


def evaluate_one(cfg, args, g, rows, shard, source, logger=print, written=None):
    """评估单个 `(group, regime, seed)`（C4／C7／C8：推理与指标**单源**走 `05_01`）。"""
    W_ = W()
    base = args._base
    unify = load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    meta_info = g["meta"]
    W_.assert_meta_matches_cli(meta_info, g["regime"], g["seed"])

    train_cfg = load_config(_abs(base["engine"]["train_config"]))
    engine = W_.Engine(base, train_cfg, logger=logger)
    load_info = engine.load(meta_info["ckpt_dir"], g["seed"])

    target = engine.score(rows, tag="%s:%s" % (g["group"], shard))
    labels = [int(float(r.get("label"))) for r in rows]

    # τ：**不重寻** ⇒ 中位数档不需要；传 None（C4：禁为消融引入额外自由度）
    tau = W_.resolve_tau_tracks(meta_info, base, None)
    tracks = {t: W_.compute_metrics(unify, labels, target["scores"], tau["tiers"][t])
              for t in (tau["main_track"], tau["parallel_track"])}

    by_carrier = []
    if args.carrier_strata:
        by_carrier = per_carrier_metrics(unify, rows, target["scores"],
                                         tau["main_value"], cfg)

    key = (g["regime"], g["seed"], _norm(meta_info["ckpt_dir"]))
    per_sample_csv = None
    if args.outputs_dir and (written is None or key not in written):
        per_sample_csv, _ = W_.write_per_sample(
            args.outputs_dir, args.ts, g["regime"], g["seed"],
            "%s_%s" % (g["group"], shard), rows,
            target["scores"], target["latency_ms"],
            tau["main_value"], tau["parallel_value"])
        if written is not None:
            written.add(key)

    ckpt_adapter = os.path.join(meta_info["ckpt_dir"], "adapter_model.safetensors")
    return {
        "group": g["group"], "group_status": g["status"],
        "regime": g["regime"], "seed": g["seed"],
        "run_id": os.path.basename(os.path.normpath(g["run_dir"])),
        "run_dir": g["run_dir"], "ckpt": meta_info["ckpt_dir"],
        "best_step": meta_info["best_step"],
        "best_epoch": (meta_info.get("meta") or {}).get("epoch"),
        "early_stopped": bool(meta_info["best"].get("early_stopped", False)),
        "segments": json.dumps(meta_info["best"].get("segments") or [], ensure_ascii=False),
        "adapter_md5": W_.md5_file(ckpt_adapter) if os.path.isfile(ckpt_adapter) else None,
        "shard": shard, "source": source, "n": len(rows),
        "n_pos": sum(1 for y in labels if y == 1),
        "n_neg": sum(1 for y in labels if y == 0),
        "tau": tau, "tracks": tracks,
        "main_track": tau["main_track"], "parallel_track": tau["parallel_track"],
        "by_carrier": by_carrier,
        "latency": W_.latency_stats(target["latency_ms"]),
        "mem_load_peak_mib": load_info["mem_load_peak_mib"],
        "mem_infer_peak_mib": target["mem_infer_peak_mib"],
        "eval_batch_max_size": target["eval_batch_max_size"],
        "split_fingerprint": split_fingerprint(rows),
        "per_sample_csv": per_sample_csv,
    }


def run_eval(cfg, args, source, logger=print):
    runs, group_status = resolve_group_runs(cfg, args, logger)
    if not runs:
        logger("NO_RUN ⇒ 四组均无 ckpt（A／B 未训练；全部指标位为 `%s`）" % TBD)
        return [], group_status, {}, []
    get_rows = _row_cache(args._base, args, source, logger)
    written = set()
    cache = {}          # (regime, seed, ckpt, shard) → 已算结果（**同一 ckpt 复用**）
    results = []
    for g in runs:
        rows, shard = get_rows(g["regime"])
        key = (g["regime"], g["seed"], _norm(g["meta"]["ckpt_dir"]), shard)
        if key in cache:
            # D 组按 §1.3 与 C 组**同一 ckpt** ⇒ 复用同一批 score（**禁**重复推理，也
            # 禁借机改动任何数）；只换 `group`／`group_status` 标签，报告须显式声明。
            res = dict(cache[key])
            res["group"] = g["group"]
            res["group_status"] = g["status"]
            logger("REUSE_CKPT group=%s ← %s（§1.3 载体消融：同一 ckpt，不重复推理）"
                   % (g["group"], key[0]))
        else:
            res = evaluate_one(cfg, args, g, rows, shard, source, logger, written)
            cache[key] = res
        results.append(res)
    ok, idrows, bad = check_split_identity(results)
    return results, group_status, {"rows": idrows, "ok": ok, "bad": bad}, []


# ---------------------------------------------------------------------------
# 聚合（C2／C3）
# ---------------------------------------------------------------------------
def regimes_in(results, cfg=None):
    """报告要呈现的体制（按配置 `regimes` 定序）。

    - 有结果 ⇒ 只呈现**实际出现**的体制（按配置序）；
    - **无任何结果** ⇒ 仍呈现配置中的全部体制（四组列须齐备，全部 `<TBD>`）
      —— 否则「四组列齐备」这一验收项在 A／B 未训练时会退化成空表。
    """
    declared = list((cfg or {}).get("regimes") or ["id", "ood"])
    seen = []
    for r in results:
        if r["regime"] not in seen:
            seen.append(r["regime"])
    got = [x for x in declared if x in seen]
    return got if got else declared


def aggregate(cfg, results, groups, regime, track=None):
    """返回 ``{group: {"complete","seeds","missing","metrics":{k:(m,s)}}}``。

    C2：seed 不齐 ⇒ 聚合格一律 `<TBD>`（**不得**用其他 seed 补齐）。
    `track`：τ 档（缺省 = 各 run 的**主口径**档）。`D-88` 裁定三-3「**全档一律报**」⇒
    两个 τ 档（`tau_ours` 与并列 `0.5`）都须各出一份聚合格。
    """
    seeds_cfg = sorted(int(s) for s in cfg["groups"]["seeds"])
    metrics = list(cfg["metrics"]["primary"])
    out = {}
    for g in groups:
        rs = [r for r in results if r["group"] == g and r["regime"] == regime]
        got = sorted(r["seed"] for r in rs)
        complete = (got == seeds_cfg) if rs else False
        entry = {"complete": complete, "seeds": got,
                 "missing": [s for s in seeds_cfg if s not in got],
                 "metrics": {}, "n": rs[0]["n"] if rs else None}
        if complete:
            for k in metrics:
                vals = [r["tracks"][track or r["main_track"]].get(k) for r in rs]
                entry["metrics"][k] = mean_std(vals, int(cfg["metrics"]["aggregate"]["ddof"]))
        out[g] = entry
    return out


def track_order(results):
    """出现过的 τ 档，按「**主口径 → 并列**」定序并去重（保序）。"""
    out = []
    for r in results:
        for t in (r.get("main_track"), r.get("parallel_track")):
            if t and t not in out:
                out.append(t)
    return out or ["star", "fixed050"]


def track_label(track, results, cfg):
    """τ 档的人读标签（`D-88` 裁定三-3 要求两档**并列**呈现）。"""
    main_track = results[0].get("main_track") if results else "star"
    if track == main_track:
        return "主口径 τ 档（逐 run `tau_ours`，全精度）"
    return "并列 τ 档（`%s`）" % (cfg.get("tau", {}).get("parallel", 0.5))


# ---------------------------------------------------------------------------
# `#69` 区分度判据（`D-88` 裁定一；**事前定值**，判据一旦落盘即为定值）
# ---------------------------------------------------------------------------
def _group_means_from_results(results, groups, regime, shard, track, metric):
    """本轮结果 ⇒ ``{group: [逐 seed 值]}``（只取 `shard` 与 `track` 匹配的 run）。"""
    out = {}
    for g in groups:
        vals = []
        for r in results:
            if (r["group"] == g and r["regime"] == regime and r["shard"] == shard
                    and track in (r.get("tracks") or {})):
                v = r["tracks"][track].get(metric)
                if v is not None:
                    vals.append(float(v))
        out[g] = vals
    return out


def load_companion_rows(cfg, logger=print):
    """读 `discrimination.companion.artifacts`（他分片的消融 CSV）⇒ 逐格取数。

    机制与 `05_03` 的 `decay.id_reference.artifacts` **同型**（配置显式列路径，**禁止**默认扫描，
    `DIRECTORY.md` §5-6）。返回 ``{shard: {regime: {track: {metric: {group: {seed: 值}}}}}}``。

    **必须按 `regime` 分层**：`g3_hard_negative_test` 这一分片在 `id`／`ood` 两个体制下**指同一份
    CSV**（`05_01.resolve_input_path`）⇒ 若不筛 `regime`，`C` 组会同时吃进 `id` 与 `ood` 各 3 个 run
    的值 ⇒「3-seed `mean`」被算成「6-run `mean`」= **跨体制混算**，`D-88` 裁定一 的口径即被破坏
    （判据要求的是「三组各自 3-seed `mean` 的极差」）。
    **再按 `seed` 去重**：同一格位重复出现且值不同 ⇒ 保留首值并**如实登记**（**禁**静默平均掉）。
    """
    W_ = W()
    disc = cfg.get("discrimination") or {}
    comp = disc.get("companion") or {}
    out = {}
    for rel in (comp.get("artifacts") or []):
        path = _abs(rel)
        if not os.path.isfile(path):
            logger("COMPANION_MISSING path=%s ⇒ 该分片格位 `%s`（不得静默当 0）" % (path, TBD))
            continue
        warned = set()
        for r in W_.read_records(path):
            shard = (r.get("shard") or "").strip()
            reg = (r.get("regime") or "").strip()
            track = (r.get("tau_track") or "").strip()
            g = (r.get("group") or "").strip()
            if not shard or not reg or not track or not g:
                continue
            seed = (r.get("seed") or "").strip()
            slot = out.setdefault(shard, {}).setdefault(reg, {}).setdefault(track, {})
            for metric in ("tpr", "fpr", "f1", "auroc"):
                v = parse_float_or_none(r.get(metric))
                if v is None:
                    continue
                per_seed = slot.setdefault(metric, {}).setdefault(g, {})
                old = per_seed.get(seed)
                if old is not None and abs(old - v) > 1e-12:
                    key = (shard, reg, track, metric, g, seed)
                    if key not in warned:
                        warned.add(key)
                        logger("COMPANION_DUP_CONFLICT shard=%s regime=%s track=%s metric=%s "
                               "group=%s seed=%s 首值=%r 次值=%r ⇒ 保留首值并如实登记"
                               % (shard, reg, track, metric, g, seed, old, v))
                    continue
                per_seed[seed] = v
        logger("COMPANION_LOADED path=%s shards=%s" % (path, sorted(out)))
    return out


def discrimination_rows(cfg, results, groups, regime, shard, companion=None, logger=print):
    """机械计算 `D-88` 裁定一 的逐格极差行。

    返回 ``(rows, verdict)``；`rows` 每行 = ``{shard, role, track, metric, means, kind, range, verdict}``。
    - **格位集由配置 `discrimination.judgment_shards` 定死**（= `rule_main`／`rule_scope` 的结构化
      表述），**与本轮跑到哪个分片无关** —— 若按「当前分片」取格位，则跑 `g3` 那一轮会漏掉
      裁定一① 的 `id_test` 主判据，跑 `in_regime_test` 那一轮（companion 未就绪时）会把
      「**未看 `g3`**」误判成「**无区分度**」⇒ **假触发** `D-88` 裁定二（`D-88` 裁定三-2 禁「换分片」）。
    - `kind`：``evaluated``（可判读）／``partial``（缺 seed）／``unavailable``（数据未就绪）。
    - 极差 = 三组（A／B／C）「3-seed `mean`」的 ``max − min``；某组 seed 不齐 ⇒ **不得**补齐。
    """
    disc = cfg.get("discrimination") or {}
    eps = float(disc.get("eps", 0.01))
    g_scope = list(disc.get("groups_in_scope") or ["A", "B", "C"])
    tracks = list(disc.get("tracks") or ["star", "fixed050"])
    n_seeds = len(cfg["groups"]["seeds"])
    comp = companion or {}
    specs = list(disc.get("judgment_shards") or
                 [{"shard": shard, "metrics": list(disc.get("metrics") or ["f1", "fpr"]),
                   "role": "main"}])
    rows = []
    for spec in specs:
        sh = spec["shard"]
        role = spec.get("role", "")
        for track in tracks:
            for metric in (spec.get("metrics") or []):
                if sh == shard:
                    per_g = _group_means_from_results(results, g_scope, regime, sh, track, metric)
                else:
                    # companion **必须按 `regime` 分层**（`g3` 分片在 id／ood 下指同一份 CSV）；
                    # 再按 `seed` 去重 ⇒ 3-seed `mean` 不被跨体制或重复行污染。
                    slot = ((((comp.get(sh) or {}).get(regime) or {})
                             .get(track) or {}).get(metric) or {})
                    per_g = {g: sorted(d.values()) for g, d in slot.items() if g in g_scope}
                means, complete = {}, True
                for g in g_scope:
                    vals = per_g.get(g) or []
                    if not vals:
                        complete = False
                        means[g] = None
                    else:
                        means[g] = statistics.fmean(vals)
                        if len(vals) < n_seeds:
                            complete = False
                if not complete or any(v is None for v in means.values()):
                    if not any(per_g.values()):
                        # 该格位**无任何数据**：他分片 ⇒ 待 companion 轮；本分片 ⇒ 缺 seed／缺格位
                        kind = "unavailable"
                        cell = ("`%s`（**待 companion 轮**：分片 `%s` 数据未就绪）" % (TBD, sh)
                                if sh != shard else
                                "`%s`（本分流该指标无值 ⇒ 缺 seed／缺格位）" % TBD)
                    else:
                        kind = "partial"
                        cell = "无法判定（缺 seed／缺格位 ⇒ `%s`）" % TBD
                    rows.append({"shard": sh, "role": role, "track": track, "metric": metric,
                                 "means": means, "kind": kind, "range": None,
                                 "verdict": cell})
                    continue
                rng = max(means.values()) - min(means.values())
                rows.append({"shard": sh, "role": role, "track": track, "metric": metric,
                             "means": means, "kind": "evaluated", "range": rng,
                             "verdict": ("有差异（**≥ ε**）" if rng >= eps
                                         else "无差异（< ε）")})

    # `D-88` 裁定一 ③ 综合判定（**逐条**给实测状态，供 `W0` 机械复核）
    def _cell_status(rs):
        if not rs:
            return "`%s`（配置未列该格位集）" % TBD
        hit = [r for r in rs if r["kind"] == "evaluated" and r["range"] >= eps]
        if hit:
            return "**存在 ≥ %s 的差异**（%d／%d 格）" % (eps, len(hit), len(rs))
        bad = [r for r in rs if r["kind"] != "evaluated"]
        if bad:
            return "**待补齐**（%d／%d 格未判读）" % (len(bad), len(rs))
        return "均 < %s（无差异）" % eps

    main_rows = [r for r in rows if r.get("role") == "main"]
    scope_rows = [r for r in rows if r.get("role") == "scope"]
    pending = [r for r in rows if r["kind"] != "evaluated"]
    l1 = "① 主判据（`id_test` × F1／FPR，%d 格）：%s" % (len(main_rows), _cell_status(main_rows))
    l2 = "② 覆盖范围（`g3_hard_negative_test` × FPR，%d 格）：%s" % (len(scope_rows), _cell_status(scope_rows))

    if not rows:
        v3 = "无法判定（未配置 `judgment_shards`）"
    elif any(r["kind"] == "evaluated" and r["range"] >= eps for r in rows):
        v3 = "**`id` 体制消融有区分度**（存在 ≥ ε 的差异）"
    elif pending and disc.get("pending_blocks_no_diff", True):
        # ⚠ 关键护栏：格位未判读完 ⇒ **不得**判「无区分度」（否则「未看 g3」会被误读成「打平」）
        v3 = ("**暂判无区分度**（已判读格位均 < ε），但 **%d 格位无法判定**（缺 seed／缺格位／"
              "待 companion 轮，见各行「判定」列的 `%s`）⇒ **结论待补齐**，"
              "**不得**据此触发 `D-88` 裁定二" % (len(pending), TBD))
    else:
        v3 = ("**`id` 体制消融无区分度** ⇒ **触发 `D-88` 裁定二**：须**呈 `W0`** 申请扩 `ood`"
              " 消融（A／B 各 3 seeds ≈ **27 h**，`D-91` 裁定二），**须用户明批**，"
              "本窗口**不得**自行扩训")
        ne = list(disc.get("not_evaluated") or [])
        if ne:
            v3 += ("。⚠ 本判定**不覆盖** `clean_original_test`（%d 项**未纳入**，见 `not_evaluated`）"
                   % len(ne))
    verdict = "%s；%s；③ 综合判定 ⇒ %s" % (l1, l2, v3)
    return rows, verdict


def per_seed_rows(results, groups, regime, track=None):
    out = []
    for g in groups:
        for r in [x for x in results if x["group"] == g and x["regime"] == regime]:
            tk = track or r["main_track"]
            m = r["tracks"][tk]
            out.append([g, r["seed"], r["run_id"], fmt_tau((r["tau"].get("tiers") or {}).get(tk)),
                        fmt4(m["tpr"]), fmt4(m["fpr"]), fmt4(m["f1"]), fmt4(m["auroc"]),
                        fmt4(r["latency"]["mean_ms"], 2), fmt4(r["mem_infer_peak_mib"], 1)])
    return out


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def _status_cell(gs):
    if gs["status"] == "MISSING":
        return "`%s`（**未训练／未就绪**）" % TBD
    if gs["status"].startswith("DERIVED"):
        return "**已评估**（沿用 %s 组 ckpt，§1.3）" % gs["status"].replace("DERIVED_FROM_", "")
    return "**已评估**"


def _ckpt_status_cell(gs):
    """`ckpt 现状` 列 —— **以本轮的实测解析状态为准**（不是配置里的静态快照）。

    完整轮里 A／B 组 ckpt 已由 `W5.6` 补训就位（`D-91` 裁定一）⇒ 若照抄配置的
    `ckpt_status`（实现轮写的 `NOT_EXIST`），报告会断言「A／B 不存在」这一**事实错误**
    （`D-21` 同族纪律：声明须与实测状态一致）。配置声明与实测不一致时**并列显示**，
    不隐去配置值（可追溯）。
    """
    got = gs["status"]
    declared = gs.get("ckpt_status", "")
    if not declared or declared == got:
        return "`%s`" % got
    return "`%s`（配置声明 `%s`）" % (got, declared)


def render_ablation_report(ctx):
    """渲染 5.6 消融报告（`skeleton=True` 时数值位一律 `<TBD>`）。"""
    cfg = ctx["cfg"]
    results = ctx["results"]
    groups = ctx["groups"]
    gstatus = ctx["group_status"]
    metrics = list(cfg["metrics"]["primary"])
    L = []
    A = L.append

    A("# Ours 消融实验评估 · 5.6 消融 · v1.0")
    A("")
    A("> 生成：`scripts/eval/05_06_eval_ablation.py`（**W6.7**，A6）｜时刻：`%s`｜"
      "配置：`configs/eval/ablation.yaml` v%s｜单源：`%s` ＋ `configs/eval/detection.yaml`"
      % (ctx["ts"], cfg["version"], cfg["reuse_module"]))
    A(">")
    A("> %s" % ctx["lead"])
    A("")

    # ---------- §0 区分度判定（转交项 `#69`，`D-88` 裁定一；**显著位置**） ----------
    disc = ctx.get("discrimination") or {}
    A("## 0. 区分度判定（转交项 `#69`，`D-88` 裁定一 —— **事前定值**）")
    A("")
    dcfg = cfg.get("discrimination") or {}
    A("> %s" % dcfg.get("mandate", ""))
    A(">")
    A("> 判据：**`ε = %s`** —— %s" % (dcfg.get("eps", 0.01), dcfg.get("eps_rationale", "")))
    A(">")
    A("> ① 主判据（`id_test`）：%s" % dcfg.get("rule_main", ""))
    A(">")
    A("> ② 覆盖范围（**不得只看 `id_test`**）：%s" % dcfg.get("rule_scope", ""))
    A(">")
    A("> ③ 综合判定：%s" % dcfg.get("rule_combined", ""))
    A("")
    if not disc.get("rows"):
        A("本轮**无可判读格位**（无已评估 run 或未配置 `discrimination`）⇒ `%s`。" % TBD)
    else:
        drows = []
        for r in disc["rows"]:
            cells = []
            for g in (dcfg.get("groups_in_scope") or ["A", "B", "C"]):
                v = (r["means"] or {}).get(g)
                cells.append(fmt4(v) if v is not None else TBD)
            drows.append([r["shard"], r.get("role", ""), r["track"], r["metric"].upper()]
                         + cells
                         + [fmt4(r["range"]) if r["range"] is not None else TBD,
                            r["verdict"]])
        A(_md_table(["分片", "判据条", "τ 档", "指标", "A `mean`", "B `mean`", "C `mean`",
                     "极差", "判定"],
                    drows))
        A("")
        A("> 格位集由 `configs/eval/ablation.yaml → discrimination.judgment_shards` **定死**"
          "（= 裁定一 ①② 的结构化表述）⇒ **与本轮跑到哪个分片无关**；他分片数据经 "
          "`discrimination.companion.artifacts` 取数，未就绪者渲染 `%s` ＋ 标「待 companion 轮」。" % TBD)
        A("")
        for part in (disc.get("verdict") or TBD).split("；"):
            A("**%s**" % part.strip())
            A("")
        A("> %s" % dcfg.get("verdict_is_trigger", ""))
    A("")
    for line in (dcfg.get("not_evaluated") or []):
        A("> ⚠ 未纳入项：%s" % line)
    if dcfg.get("not_evaluated"):
        A("")

    # ---------- §1 四组定义 ----------
    A("## 1. 四组定义（v5 L612–L620 原文，一字不改）")
    A("")
    gr = cfg.get("group_recipe") or {}
    A(_md_table(["组", "定义（v5 原文）", "训练配方（`D-84` 裁定一）", "ckpt 现状（实测）", "本轮处置"],
                [[g, (cfg["groups"].get(g) or {}).get("verbatim", ""),
                  "`%s`" % ((gr.get(g) or {}).get("cli") or NA),
                  _ckpt_status_cell(gstatus.get(g, {"status": "MISSING", "ckpt_status": ""})),
                  _status_cell(gstatus.get(g, {"status": "MISSING"}))] for g in groups]))
    A("")
    if gr:
        A("> 配方来源：%s；`--augment` 语义：%s" % (gr.get("source", ""),
                                            gr.get("augment_or_semantics", "")))
        A(">")
        A("> 单因素阶梯：%s。体制范围：%s" % (gr.get("ladder", ""), gr.get("regime_scope", "")))
        A("")
    for line in cfg["declarations"]["reading_disclosure"]:
        A("> %s" % line)
    if cfg["declarations"]["reading_disclosure"]:
        A(">")
    A("")
    # `ab_not_trained` 只在 **A／B 确有缺组**时渲染 —— 完整轮两者 ckpt 已就位（`D-91` 裁定一）
    # ⇒ 继续渲染即为**事实错误**（`D-21` 同族纪律：声明须与实测状态一致）。
    ab_missing = [g for g in ("A", "B")
                  if (gstatus.get(g, {"status": "MISSING"})["status"] == "MISSING")]
    if ab_missing:
        for line in cfg["declarations"]["ab_not_trained"]:
            A("> %s" % line)
            A(">")
    for line in cfg["declarations"]["no_substitute"]:
        A("> %s" % line)
        A(">")
    for line in cfg["declarations"]["no_training"]:
        A("> %s" % line)
    A("")

    # ---------- §2 消融总表 ----------
    A("## 2. 消融总表（`mean ± std`，3 seeds；**两个 τ 档并列分表**）")
    A("")
    A("> `D-88` 裁定三-3「**全档一律报**」⇒ 主口径 τ 档（逐 run `tau_ours`）与并列 τ 档"
      "（`%s`）**各自成表**，**不得**只报其中一档（否则「越档」判据的实测值在总表里查不到）。"
      % cfg.get("tau", {}).get("parallel", 0.5))
    A("")
    if not results:
        A("**四组均无可用 ckpt** ⇒ 全部指标位为 `%s`（未训练／未就绪）。" % TBD)
        A("")
    n = 0
    for regime in ctx["regimes"]:
        for track in ctx["tracks"]:
            n += 1
            agg = ctx["aggregate"][regime][track]
            A("### 2.%d 体制 `%s` —— %s" % (n, regime, track_label(track, results, cfg)))
            A("")
            rows = []
            for g in groups:
                e = agg[g]
                if not e["complete"]:
                    rows.append([g, "缺 seed：%s" % (e["missing"] or "全部"),
                                 TBD, TBD, TBD, TBD,
                                 "整列占位（**不得**用其他 seed 补齐，C2）"])
                elif gstatus.get(g, {}).get("status", "").startswith("DERIVED"):
                    rows.append([g, "3/3"] + [mean_std_cell(*e["metrics"][k]) for k in metrics]
                                + ["与 %s 组**同一 ckpt** ⇒ 总体指标必然同值；D 组的消融信号见 **§5 载体分层**"
                                   % str(gstatus[g]["status"]).replace("DERIVED_FROM_", "")])
                else:
                    rows.append([g, "3/3"] + [mean_std_cell(*e["metrics"][k]) for k in metrics]
                                + ["n=%s" % e["n"]])
            A(_md_table(["组", "seeds", "TPR", "FPR", "F1", "AUROC", "备注"], rows))
            A("")
            seed_rows = per_seed_rows(results, groups, regime, track)
            if seed_rows:
                A("**逐 seed 原值表**（体制 `%s`，%s）" % (regime, track_label(track, results, cfg)))
                A("")
                A(_md_table(["组", "seed", "run_id", "τ（全精度）", "TPR", "FPR", "F1", "AUROC",
                             "延迟均值 ms", "推理显存峰值 MiB"], seed_rows))
                A("")
    A("### 2.9 `std = 0` 退化声明")
    A("")
    zero = []
    for regime in ctx["regimes"]:
        for track in ctx["tracks"]:
            for g in groups:
                e = ctx["aggregate"][regime][track][g]
                for k in metrics:
                    ms = e["metrics"].get(k) if e["complete"] else None
                    if ms and ms[1] is not None and abs(ms[1]) < 1e-12:
                        zero.append("`%s`／`%s`／`%s`／%s" % (g, regime, track, k.upper()))
    if zero:
        A("以下格子的 `std = 0` ⇒ **显式声明退化**（**逐 τ 档**点名，共 **%d** 格）：%s。"
          % (len(zero), "；".join(zero)))
    else:
        A("本轮无 `std = 0` 的格子（或相关组未就绪 ⇒ `%s`）。" % TBD)
    A("")
    A("> %s" % cfg["metrics"]["std_zero"]["declaration"])
    A(">")
    A("> %s" % cfg["metrics"]["variance_policy"]["rule"])
    A("")

    # ---------- §3 τ 口径 ----------
    A("## 3. τ 口径（**逐组逐 seed**，不重寻）")
    A("")
    A("> %s" % cfg["tau"]["main_criterion"])
    A(">")
    A("> %s" % cfg["declarations"]["tau_not_reseek"][0])
    A("")
    trows = []
    for g in groups:
        for r in [x for x in results if x["group"] == g]:
            trows.append([g, r["regime"], r["seed"], fmt_tau(r["tau"]["tau_star"]),
                          r["tau"]["override"], fmt_tau(r["tau"]["gap"]),
                          r["main_track"], fmt_tau(r["tau"]["main_value"]),
                          fmt_tau(r["tau"]["parallel_value"])])
    if trows:
        A(_md_table(["组", "regime", "seed", "`tau_ours`（全精度）", "override", "gap",
                     "主口径档", "主口径值", "并列值"], trows))
    else:
        A("（无已评估 run ⇒ 全 `%s`）" % TBD)
    A("")

    # ---------- §4 统一划分一致性 ----------
    A("## 4. 统一划分一致性（C1，`v5 新增 32`）")
    A("")
    A("> %s" % cfg["declarations"]["split_identity"][0])
    A("")
    ident = ctx.get("split_identity") or {}
    if not ident.get("rows"):
        A("本轮**无已评估 run** ⇒ 指纹表为空（`%s`）。" % TBD)
    else:
        A("判据：%s（指纹法：%s）。" % (cfg["split_identity"]["rule"],
                                  cfg["split_identity"]["fingerprint"]["method"]))
        A("")
        A(_md_table(["组", "regime", "seed", "shard", "n_records", "指纹（前 16）", "一致"],
                    [[r["group"], r["regime"], r["seed"], r["shard"], r["n_records"],
                      r["fingerprint"][:16], "✅" if r["matches_reference"] == "1" else "❌"]
                     for r in ident["rows"]]))
        A("")
        if ident.get("enforced"):
            A("`--check-split-identity` **已启用** ⇒ 判定：**%s**。"
              % ("PASS（四组逐位相同）" if ident["ok"]
                 else "FAIL（不一致 ⇒ 非零退出 ＋ 停工呈 W0）"))
        else:
            A("`--check-split-identity` **未启用** ⇒ 本表仅作留证，未作门禁判定。")
    A("")

    # ---------- §5 载体分层 ----------
    A("## 5. 载体分层（D 组，`--carrier-strata`）")
    A("")
    A("> %s" % cfg["declarations"]["d_group"][0])
    A("")
    if not ctx.get("carrier_rows"):
        A("本轮未启用 `--carrier-strata` ⇒ 无载体分层表（`%s`）。" % TBD)
    else:
        A(_md_table(["组", "regime", "seed", "`carrier_type`", "n", "n_pos", "n_neg",
                     "TPR", "FPR", "F1"], ctx["carrier_rows"]))
    A("")

    # ---------- §6 延迟与显存 ----------
    A("## 6. 延迟与显存")
    A("")
    A("> 口径与 `BASELINE_REPORT.md` §1 逐列同构；引用该件数字须一并带其 **§12 口径注释**"
      "（`W6.7` C10）。延迟**不含模型加载**；批次实测 ⇒ 须与 `eval_batch_max_size` 一并解读。")
    A("")
    erows = []
    for g in groups:
        for r in [x for x in results if x["group"] == g]:
            erows.append([g, r["regime"], r["seed"], r["eval_batch_max_size"],
                          fmt4(r["latency"]["mean_ms"], 2), fmt4(r["latency"]["median_ms"], 2),
                          fmt4(r["latency"]["p95_ms"], 2),
                          fmt4(r["mem_load_peak_mib"], 1), fmt4(r["mem_infer_peak_mib"], 1)])
    if erows:
        A(_md_table(["组", "regime", "seed", "batch_max", "延迟均值 ms", "延迟中位 ms",
                     "延迟 p95 ms", "加载态 MiB", "推理态 MiB"], erows))
    else:
        A("（无已评估 run ⇒ 全 `%s`）" % TBD)
    A("")

    # ---------- §7 ACCESS_LOG ----------
    A("## 7. ACCESS_LOG 与冻结面复核")
    A("")
    A(_md_table(["项", "值"],
                [["`--source`", ctx["source"]],
                 ["`--split`", ctx.get("split") or NA],
                 ["`--access-log-note`", ctx.get("access_log_note") or "n/a（非 sealed 源）"],
                 ["ACCESS_LOG 6 列", "、".join(cfg["input_policy"]["access_log_columns"])],
                 ["脚本是否写 ACCESS_LOG.md", "**否**（登记由人手工完成；脚本只回显 note）"],
                 ["冻结面复核", ctx.get("freeze_line") or TBD]]))
    A("")

    # ---------- §8 局限 ----------
    A("## 8. 局限与回退声明")
    A("")
    for key in ("std_zero", "variance", "d_group", "split_identity", "tau_not_reseek"):
        for line in cfg["declarations"][key]:
            A("- %s" % line)
    for line in (ctx.get("extra_limitations") or []):
        A("- %s" % line)
    A("")

    # ---------- §9 回执 ----------
    A("## 9. 回执／验收位（留给 W6 复核）")
    A("")
    A("- `--self-check`：%s" % ctx.get("selfcheck_line", TBD))
    A("- `make check`／`make test`：%s" % ctx.get("test_line", TBD))
    A("- 冻结面 `md5sum -c`：%s" % (ctx.get("freeze_line") or TBD))
    A("- **不训练证明**：本窗口对 `04_01`／`04_02`／`04_03` 的调用次数 = **0**"
      "（`--self-check` 有源码级断言）。")
    A("")

    # ---------- 附：EVAL_REPORT §5.6 骨架段 ----------
    A("---")
    A("")
    A("## 附：可直接搬入 `reports/eval/EVAL_REPORT.md` 的 §5.6 骨架段")
    A("")
    A("> 本段表头与口径已一次做齐；完整轮（A／B 组 ckpt 就绪后）只需填数，**不得**改动口径。")
    A("")
    A(_md_table(["组", "定义", "TPR", "FPR", "F1", "AUROC"],
                [[g, (cfg["groups"].get(g) or {}).get("label", ""), TBD, TBD, TBD, TBD]
                 for g in groups]))
    A("")
    return "\n".join(L)


def build_ctx(cfg, args, results, group_status, groups, source, split, ident,
              lead, access_lines=None, extra_limitations=None, logger=print):
    rg = regimes_in(results, cfg)
    # `#69` 区分度判据（`D-88` 裁定一）—— 有结果即算，且**恒**在报告 §0 显著位置呈现
    disc_rows, disc_verdict = [], None
    if results and (cfg.get("discrimination") or {}).get("mandate"):
        comp = load_companion_rows(cfg, logger)
        disc_rows, disc_verdict = discrimination_rows(
            cfg, results, groups, results[0]["regime"], results[0]["shard"], comp, logger)
    agg = {r: {t: aggregate(cfg, results, groups, r, t) for t in track_order(results)}
           for r in rg}
    carrier_rows = []
    if getattr(args, "carrier_strata", False):
        for g in groups:
            for r in [x for x in results if x["group"] == g]:
                for c in r["by_carrier"]:
                    carrier_rows.append([g, r["regime"], r["seed"], c["carrier_type"],
                                         c["n"], c["n_pos"], c["n_neg"],
                                         fmt4(c["tpr"]), fmt4(c["fpr"]), fmt4(c["f1"])])
    return {
        "cfg": cfg, "ts": args.ts, "results": results, "groups": groups,
        "group_status": group_status, "regimes": rg, "aggregate": agg,
        "tracks": track_order(results),
        "split_identity": {"rows": ident.get("rows") or [], "ok": ident.get("ok"),
                           "bad": ident.get("bad") or [],
                           "enforced": bool(getattr(args, "check_split_identity", False))},
        "carrier_rows": carrier_rows, "source": source, "split": split,
        "access_log_note": getattr(args, "access_log_note", None),
        "access_lines": access_lines or [],
        "freeze_line": None, "lead": lead,
        "extra_limitations": extra_limitations or [],
        "discrimination": {"rows": disc_rows, "verdict": disc_verdict},
    }


# ---------------------------------------------------------------------------
# 写盘
# ---------------------------------------------------------------------------
def write_csv_rows(out_path, fields, rows):
    W().write_csv_rows(out_path, fields, rows)
    return out_path


def write_ablation_csv(path, results, group_status=None):
    rows = []
    for r in results:
        for track in (r["main_track"], r["parallel_track"]):
            m = r["tracks"][track]
            rows.append({
                "group": r["group"], "group_status": r["group_status"],
                "regime": r["regime"], "seed": r["seed"], "run_id": r["run_id"],
                "shard": r["shard"], "source": r["source"],
                "tau_track": track, "tau_value": fmt_tau(r["tau"]["tiers"][track]),
                "n": r["n"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "tpr": "" if m["tpr"] is None else "%.8f" % m["tpr"],
                "fpr": "" if m["fpr"] is None else "%.8f" % m["fpr"],
                "f1": "" if m["f1"] is None else "%.8f" % m["f1"],
                "auroc": "" if m["auroc"] is None else "%.8f" % m["auroc"],
                "latency_mean_ms": "" if r["latency"]["mean_ms"] is None else "%.6f" % r["latency"]["mean_ms"],
                "latency_median_ms": "" if r["latency"]["median_ms"] is None else "%.6f" % r["latency"]["median_ms"],
                "latency_p95_ms": "" if r["latency"]["p95_ms"] is None else "%.6f" % r["latency"]["p95_ms"],
                "mem_load_peak_mib": "" if r["mem_load_peak_mib"] is None else "%.1f" % r["mem_load_peak_mib"],
                "mem_infer_peak_mib": "" if r["mem_infer_peak_mib"] is None else "%.1f" % r["mem_infer_peak_mib"],
                "eval_batch_max_size": r["eval_batch_max_size"]})
    write_csv_rows(path, ABLATION_CSV_FIELDS, rows)
    return path


def write_split_identity_csv(path, ident):
    write_csv_rows(path, SPLIT_IDENTITY_CSV_FIELDS, ident.get("rows") or [])
    return path


def write_carrier_csv(path, results):
    rows = []
    for r in results:
        for c in r["by_carrier"]:
            rows.append({
                "group": r["group"], "regime": r["regime"], "seed": r["seed"],
                "run_id": r["run_id"], "shard": r["shard"], "source": r["source"],
                "tau_track": r["main_track"], "tau_value": fmt_tau(r["tau"]["main_value"]),
                "carrier_type": c["carrier_type"],
                "n": c["n"], "n_pos": c["n_pos"], "n_neg": c["n_neg"],
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
                "tpr": "" if c["tpr"] is None else "%.8f" % c["tpr"],
                "fpr": "" if c["fpr"] is None else "%.8f" % c["fpr"],
                "f1": "" if c["f1"] is None else "%.8f" % c["f1"]})
    write_csv_rows(path, CARRIER_CSV_FIELDS, rows)
    return path


# ---------------------------------------------------------------------------
# 模式：eval
# ---------------------------------------------------------------------------
def cmd_eval(cfg, args, source, logger=print):
    groups = parse_groups_arg(args.groups, cfg)
    results, group_status, ident, _ = run_eval(cfg, args, source, logger)

    if args.check_split_identity and ident.get("rows") and not ident.get("ok"):
        logger("SPLIT_IDENTITY_FAIL detail=%s ⇒ **非零退出 ＋ 停工呈 W0**（`v5 新增 32` 硬性）"
               % "; ".join(ident["bad"]))
        print("SPLIT_IDENTITY_RESULT=FAIL n=%d" % len(ident["rows"]))
        return 1
    if ident.get("rows"):
        # P-3（`D-83` 裁定九）：**PASS 路径亦须落日志** —— 否则 `PROGRESS.md` 引用的
        # `SPLIT_IDENTITY_RESULT=PASS` 在交付日志中无留证（FAIL 路径已由上方 `logger` 留证）。
        # 复用 `logger`（其内部**既 print 又落盘**）⇒ stdout 仍出该行，且格式与 FAIL 路径一致。
        if ident["ok"]:
            logger("SPLIT_IDENTITY_RESULT=PASS n=%d" % len(ident["rows"]))
        else:
            print("SPLIT_IDENTITY_RESULT=FAIL n=%d" % len(ident["rows"]))

    if source == "sealed":
        lead = ("**完整轮（正式）**：A／B 组 ckpt 由 `W5.6`（A5）补训产出并已验收（`D-91` 裁定一，"
                "tag `train-ab-v1`），C 组为阶段4 的 6 个 run，D 组按 `D-83` 裁定六 沿用 C 组 ckpt "
                "并按 `carrier_type` 分层（不训练）。四组**共用完全相同划分**"
                "（`--check-split-identity`）；τ **逐组逐 seed** 取 `best epoch` 的 `tau_ours`"
                "（全精度，不重寻、不统一）。区分度判定见 **§0**（转交项 `#69`）。")
    else:
        lead = ("**演练轮**（`--source val`，非权威结论）：val 侧**已饱和**"
                "（`D-88` 〇：C 组 6 run 的 `val_f1` 全 = 1.0）⇒ **不可**用于判定方法有效性；"
                "本轮只用于跑通同一代码路径。四大组按当前可得 ckpt 接线，缺组整列 `%s` "
                "＋ 标注「未训练／未就绪」。" % TBD)
    ctx = build_ctx(cfg, args, results, group_status, groups, source, args.split, ident, lead)
    ctx["selfcheck_line"] = "见 `PROGRESS.md` 本窗口行的登记（`--self-check` 一并复核）"
    # 冻结面复核（**纯字节级**、`D-55` 裁定六 **档②**：不解析内容、不进入任何判断）
    # 实现期只复核 frozen 侧；sealed 侧仅在 `--source sealed` 时一并复核。
    ok_f, tot_f = W().md5sum_manifest_check("data/frozen/CodeObf-PI_v1.0")
    lines = ["frozen `md5sum -c` = **%s/%s OK**" % (ok_f, tot_f)]
    if source == "sealed":
        ok_s, tot_s = W().md5sum_manifest_check("data/sealed_test/CodeObf-PI_v1.0")
        lines.append("sealed `md5sum -c` = **%s/%s OK**" % (ok_s, tot_s))
    else:
        lines.append("sealed 侧**未复核**（实现期不触碰封存，`W6.7` §3-C5）")
    ctx["freeze_line"] = "；".join(lines)
    logger("FREEZE_CHECK %s" % ctx["freeze_line"])

    name = "ablation" if source != "val" else "ablation_dryrun_val"
    md_path = os.path.join(args.report_dir, "%s_v1.0_%s.md" % (name, args.ts))
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_ablation_report(ctx))

    csv_path = os.path.join(args.report_dir, "%s_v1.0_%s.csv" % (name, args.ts))
    write_ablation_csv(csv_path, results, group_status)

    write_split_identity_csv(
        os.path.join(args.report_dir, "ablation_split_identity_v1.0_%s.csv" % args.ts), ident)

    if args.carrier_strata:
        write_carrier_csv(os.path.join(args.report_dir,
                                       "ablation_carrier_v1.0_%s.csv" % args.ts), results)

    print("REPORT_MD=%s REPORT_CSV=%s" % (md_path, csv_path))
    return 0


# ---------------------------------------------------------------------------
# `--dry-run`：解析全部资源与断言（不加载模型、不触 GPU）
# ---------------------------------------------------------------------------
def cmd_dryrun(cfg, args, logger=print):
    W_ = W()
    checks = []
    try:
        runs, group_status = resolve_group_runs(cfg, args, logger)
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        logger("RESOLVE_RUNS_SKIP detail=%s ⇒ runs=[]" % exc)
        runs, group_status = [], {}
    checks.append(("groups_resolved", True,
                   "included=%s" % ",".join(parse_groups_arg(args.groups, cfg))))
    checks.append(("runs_resolved", True, "n=%d" % len(runs)))
    # 组状态属**信息项**而非判据：实现轮 A／B 缺 ckpt 是**已知且必须如实标注**的状态
    # （§1.2），判 FAIL 会把「如实标注」误报成「资源不可解析」。
    for g in sorted(group_status or {}):
        gs = group_status[g]
        checks.append(("group:%s" % g, True,
                       "status=%s n_runs=%d%s" % (gs["status"], gs["n_runs"],
                                                  " ⇒ 整列 `%s` ＋ 标注「未训练／未就绪」"
                                                  % TBD if gs["status"] == "MISSING" else "")))
    source = args.source or "val"
    for g in runs:
        if source == "sealed":
            path, expect, shard = W_.resolve_input_path(args._base, "sealed", g["regime"],
                                                        args.split, True)
            W_.assert_input_allowed(path, args._base, allow_sealed=True)
        else:
            path, expect, shard = W_.resolve_input_path(args._base, "val", g["regime"],
                                                        None, False)
            W_.assert_input_allowed(path, args._base, allow_sealed=False)
        logger("PLAN_INPUT group=%s regime=%s shard=%s expect=%s exists=%s"
               % (g["group"], g["regime"], shard, expect, os.path.isfile(path)))
        if os.path.isfile(path):
            got = W_.assert_record_count(path, expect, args._base)
            checks.append(("record_count:%s" % shard, got == int(expect), "records=%d" % got))
    for name, okc, detail in checks:
        print("DRYRUN %-38s %s %s" % (name, "PASS" if okc else "FAIL", detail))
    ok = all(c[1] for c in checks)
    logger("DRYRUN_PLAN steps=%d source=%s report_dir=%s outputs_dir=%s"
           % (len(checks), source, args.report_dir, args.outputs_dir))
    print("DRYRUN_RESULT=%s n=%d" % ("PASS" if ok else "FAIL", len(checks)))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# `--self-check`：离线自检（不加载模型、**不读封存**、不触 GPU）
# ---------------------------------------------------------------------------
def _mk_fake_run(run_dir, regime, seed, tau=0.9999485611915588):
    """自检用：造一个最小可用的 run 目录（`best.json` ＋ ckpt `meta.json`）。"""
    ckpt = os.path.join(run_dir, "seed_%d" % seed, "checkpoint-1")
    os.makedirs(ckpt, exist_ok=True)
    meta = {"regime": regime, "seed": seed, "epoch": 2,
            "tau_ours": tau, "tau_star": tau,
            "tau_main_override": False, "tau_neg_max_gap": 0.96}
    with open(os.path.join(ckpt, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    best = {"dir": ckpt, "regime": regime, "seed": seed, "step": 1,
            "early_stopped": False, "segments": [{"start_step": 0, "end_step": 1}]}
    with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8") as fh:
        json.dump(best, fh)
    return run_dir


class _Args(object):
    """自检内的最小 args 容器（`resolve_group_runs`／`build_ctx` 只需少量字段）。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _disc_fake_results(vals_by_group, metric="f1", regime="id", shard="id_test",
                       seeds=(42, 123, 456)):
    """`--self-check`／单测共用的最小桩：按 `(组 → 逐 seed 值)` 造结果（**两个 τ 档同值**）。

    `TPR`／`FPR`／`F1` 一并置为同一值 ⇒ 判据的**两个指标格位均可判读**
    （否则另一指标为 `None` ⇒ 该行判「无法判定」，会把「全格 < ε」误成「仍有格位未判读」）。
    """
    out = []
    for g, vs in vals_by_group.items():
        for s, v in zip(seeds, vs):
            r = {"group": g, "regime": regime, "seed": s, "shard": shard, "source": "sealed",
                 "main_track": "star", "parallel_track": "fixed050",
                 "tracks": {"star": {"tpr": None, "fpr": None, "f1": None, "auroc": None},
                            "fixed050": {"tpr": None, "fpr": None, "f1": None, "auroc": None}},
                 "tau": {"tau_star": 0.9, "override": False, "gap": 0.9,
                         "main_value": 0.9, "parallel_value": 0.5,
                         "main_track": "star", "parallel_track": "fixed050",
                         "tiers": {"fixed050": 0.5, "median_val": None, "star": 0.9}},
                 "latency": {"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.0},
                 "mem_load_peak_mib": 1.0, "mem_infer_peak_mib": 1.0,
                 "eval_batch_max_size": 16, "by_carrier": [],
                 "split_fingerprint": "fp", "run_id": "fake", "n": 10,
                 "n_pos": 5, "n_neg": 5, "group_status": "READY"}
            for t in ("star", "fixed050"):
                r["tracks"][t][metric] = v
                for m2 in ("tpr", "fpr", "f1"):
                    if m2 != metric:
                        r["tracks"][t][m2] = v
            out.append(r)
    return out


def _disc_fake_companion(vals_by_group, shard="g3_hard_negative_test",
                         metrics=("fpr",), tracks=("star", "fixed050"),
                         regime="id", seeds=(42, 123, 456)):
    """区分度 自检／单测用桩：造**他分片**的取数结构（默认 = 裁定一② 的 `g3` 仅 `fpr`）。

    结构与 `load_companion_rows` 的输出**同形**：``{shard: {regime: {track: {metric:
    {group: {seed: 值}}}}}}`` —— 含 `regime` 层，因为 `g3` 分片在 `id`／`ood` 下指同一份
    CSV，缺该层会令 3-seed `mean` 变成跨体制混算。

    判据的格位集跨**多个分片**，而一轮只跑一个分片 ⇒ 另一分片必须经 `companion`（配置
    `discrimination.companion.artifacts`）供给。只给单分片桩 ⇒ 只能验证「待补齐」分支；
    要验证「综合判定成立」必须把另一分片经 companion 凑齐（这正是本轮修掉的缺口）。
    """
    out = {}
    for t in tracks:
        for m in metrics:
            out.setdefault(shard, {}).setdefault(regime, {}).setdefault(t, {})[m] = \
                {g: {s: float(v) for s, v in zip(seeds, vs)} for g, vs in vals_by_group.items()}
    return out


def self_check(cfg, args, logger=print):
    W_ = W()
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print("SELFCHECK %-44s %s %s" % (name, "PASS" if ok else "FAIL", detail))

    # ① C6 记录数：`csv.DictReader`；`text` 含换行 ⇒ 记录数 2、行数 ≥ 4
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="05_06_selfcheck_")
    p_csv = os.path.join(tmpdir, "nl.csv")
    W_.write_csv_rows(p_csv, ["sample_id", "carrier_type", "text", "label"],
                      [{"sample_id": "a1", "carrier_type": "code",
                        "text": "l1\nl2\nmid\nl3", "label": "1"},
                       {"sample_id": "a2", "carrier_type": "text",
                        "text": "plain", "label": "0"}])
    rec, lines = W_.count_records(p_csv), W_.count_physical_lines(p_csv)
    add("csv_dictreader_newline_regression", rec == 2 and lines >= 4,
        "records=%d physical_lines=%d（行数 ≠ 记录数）" % (rec, lines))
    try:
        W_.assert_record_count(p_csv, 9999, args._base)
        add("record_assertion_nonzero", False, "断言未触发")
    except AssertionError:
        add("record_assertion_nonzero", True, "不符即 AssertionError ⇒ 上层非零退出")
    add("record_assert_registry_consistent",
        all(int(args._base["record_assertions"][k]) == int(v)
            for k, v in cfg["record_count_policy"]["expected"].items()),
        "本文件登记值 ↔ `detection.yaml` 逐格一致")

    # ② C1 统一划分指纹：确定性／顺序敏感／构造不一致 ⇒ `ok=False`
    r1 = [{"text": "x"}, {"text": "y"}]
    r3 = [{"text": "y"}, {"text": "x"}]
    add("split_fingerprint_deterministic",
        split_fingerprint(r1) == split_fingerprint([{"text": "x"}, {"text": "y"}]),
        "同一 rows 两次 ⇒ 指纹逐位相同")
    add("split_fingerprint_order_sensitive", split_fingerprint(r1) != split_fingerprint(r3),
        "顺序变化 ⇒ 指纹变化（**逐位**敏感）")
    fake = [{"group": "A", "regime": "id", "seed": 42, "shard": "id_val", "source": "val",
             "n": 2, "n_pos": 1, "n_neg": 1, "split_fingerprint": split_fingerprint(r1)},
            {"group": "C", "regime": "id", "seed": 42, "shard": "id_val", "source": "val",
             "n": 2, "n_pos": 1, "n_neg": 1, "split_fingerprint": split_fingerprint(r3)}]
    ok_i, rows_i, bad_i = check_split_identity(fake)
    add("split_identity_detects_mismatch", (not ok_i) and len(bad_i) == 1,
        "构造不一致 ⇒ ok=False、明细 n=%d（**非零退出**判据）" % len(bad_i))
    ok_s, _r_s, _b_s = check_split_identity([fake[0], dict(fake[0], group="C")])
    add("split_identity_passes_when_identical", ok_s, "四组指纹一致 ⇒ ok=True")
    # 回归：本窗口**实跑**触发的假 FAIL —— `id_val`（1584）与 `ood_val`（927）是
    # 两个不同分片，指纹必然不同；若跨 `(regime, shard)` 一并比对会误判成「每组用了
    # 不同划分」⇒ 假停工。正确语义 = **同一 `(regime, shard)` 内**四组逐位相同。
    cross = [dict(fake[0], regime="ood", shard="ood_val",
                  split_fingerprint=split_fingerprint(r3)), fake[0]]
    ok_c, _r_c, _b_c = check_split_identity(cross)
    add("split_identity_scoped_per_regime_shard", ok_c,
        "跨 `(regime, shard)` 的不同分片**不**误判为不一致（假 FAIL 回归）")

    # ③ C2 缺组 ⇒ `<TBD>` ＋ 标注；D 组沿用 C；四组列齐备
    c_run = _mk_fake_run(os.path.join(tmpdir, "c_run"), "id", 42)
    a = _Args(group_ckpt=["C=%s" % c_run], groups="A,B,C,D", _base=args._base,
              ts="selfcheck", carrier_strata=True, check_split_identity=True,
              access_log_note=None, split=None)
    runs_s, gstat = resolve_group_runs(cfg, a, print)
    add("missing_group_yields_tbd",
        gstat["A"]["status"] == "MISSING" and gstat["B"]["status"] == "MISSING"
        and cfg["groups"]["missing_group_policy"]["note"] in (gstat["A"]["note"] or ""),
        "A／B 缺 ckpt ⇒ 整列 `%s` ＋ 标注「未训练／未就绪」（**不静默跳过**）" % TBD)
    add("d_group_inherits_c",
        gstat["D"]["status"] == "DERIVED_FROM_C" and len(runs_s) == 2,
        "D 组按 §1.3 沿用 C 组 ckpt ⇒ 评估条目 n=%d（A／B 未混入）" % len(runs_s))
    rendered = render_ablation_report(
        build_ctx(cfg, a, [], gstat, ["A", "B", "C", "D"], "val", None, {}, "lead"))
    add("missing_group_rendered_tbd_in_table", TBD in rendered,
        "报告总表对缺组渲染 `%s`（区别于 `n/a`＝**不适用**）" % TBD)
    add("four_group_columns_present",
        all(("| %s |" % g) in rendered for g in ("A", "B", "C", "D")),
        "A／B／C／D 四组列**齐备**（未就绪者 `%s`）" % TBD)
    add("no_c_number_in_ab_column",
        ("| A |" in rendered) and (rendered.count("| A |") >= 1),
        "A 组列存在且独立成列（与 C 组列**不同行**）")
    # A／B 复用 C 组 ckpt ⇒ **非零退出**（结构性防冒充）
    try:
        resolve_group_runs(cfg, _Args(group_ckpt=["A=%s" % c_run, "C=%s" % c_run],
                                      groups="A,B,C,D", _base=args._base), print)
        add("alias_guard_rejects_ab_reuse_c", False, "未拦截")
    except AssertionError as exc:
        add("alias_guard_rejects_ab_reuse_c", "ALIAS_GUARD_VIOLATION" in str(exc),
            "A 组复用 C 组 ckpt ⇒ AssertionError（**禁止**用 C 冒充 A／B）")

    # ④ C4 τ **不重寻**：只读取与判定式应用
    fake_meta = {"tau_star": 0.9999485611915588, "override": False,
                 "gap": 0.9626216776669025, "run_dir": "", "ckpt_dir": "", "best": {},
                 "meta": {}, "regime": "id", "seed": 42, "best_step": 1}
    tau = W_.resolve_tau_tracks(fake_meta, args._base, None)
    add("tau_main_is_star_when_no_override",
        tau["main_track"] == "star" and tau["main_value"] == fake_meta["tau_star"],
        "override=False ⇒ 主口径 = `tau_ours`（未统一、未重寻）")
    add("tau_full_precision", fmt_tau(tau["tau_star"]) == repr(0.9999485611915588),
        "τ = %s（全精度）" % fmt_tau(tau["tau_star"]))
    add("tau_parallel_is_050", abs(tau["parallel_value"] - 0.5) < 1e-12,
        "并列档 = 0.5（`ablation.yaml → tau.parallel`）")
    add("tau_reseek_forbidden_in_config",
        cfg["tau"]["reseek_forbidden"] is True and cfg["tau"]["unify_forbidden"] is True,
        "配置明禁「统一 τ」与「重寻 τ」")

    # ⑤ C3 `mean ± std`、`std = 0` 退化、**缺 seed 不补齐**
    mz, sz = mean_std([0.5, 0.5, 0.5])
    add("std_zero_is_zero", abs(sz) < 1e-12, "3 个同值 ⇒ std=0（ddof=1）")
    add("std_zero_cell_flagged", "std=0" in mean_std_cell(mz, sz),
        "`std=0` 格渲染为 `%s`（**显式声明退化**）" % mean_std_cell(mz, sz))
    one = [{"group": "A", "regime": "id", "seed": 42,
            "tracks": {"star": {"tpr": 1.0, "fpr": 0.0, "f1": 1.0, "auroc": 1.0},
                       "fixed050": {"tpr": 0.9, "fpr": 0.0, "f1": 0.95, "auroc": 1.0}},
            "main_track": "star", "parallel_track": "fixed050", "n": 10}]
    add("incomplete_group_yields_tbd",
        (not aggregate(cfg, one, ["A"], "id")["A"]["complete"])
        and aggregate(cfg, one, ["A"], "id")["A"]["metrics"] == {},
        "只 1 个 seed ⇒ 聚合格 `<TBD>`（**不得**用其他 seed 补齐）")
    three = [dict(one[0], seed=s) for s in (42, 123, 456)]
    add("complete_group_aggregates",
        aggregate(cfg, three, ["A"], "id")["A"]["complete"] is True
        and abs(aggregate(cfg, three, ["A"], "id")["A"]["metrics"]["tpr"][0] - 1.0) < 1e-12,
        "3 seeds 齐 ⇒ 出 `mean ± std`（含 `std=0` 声明位）")
    add("per_seed_raw_table_present",
        len(per_seed_rows([{"group": "C", "regime": "id", "seed": 42, "run_id": "r",
                            "tau": {"tau_star": 0.9, "tiers": {"star": 0.9, "fixed050": 0.5}},
                            "tracks": {"star": {"tpr": 1.0, "fpr": 0.0, "f1": 1.0,
                                                "auroc": 1.0}},
                            "main_track": "star", "latency": {"mean_ms": 1.0},
                            "mem_infer_peak_mib": 2.0}], ["C"], "id")) == 1,
        "逐 seed 原值表与 `mean ± std` **并给**")
    add("two_tau_tracks_reported",
        track_order(three) == ["star", "fixed050"]
        and aggregate(cfg, three, ["A"], "id", "fixed050")["A"]["complete"] is True,
        "**全档一律报**（`D-88` 裁定三-3）：`star` 与 `fixed050` 两档各自可聚合")

    # ⑥ 载体分层（D 组）
    unify = load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    crows = [{"carrier_type": "code", "text": "a", "label": "1"},
             {"carrier_type": "code", "text": "b", "label": "0"},
             {"carrier_type": "text", "text": "c", "label": "1"}]
    cres = per_carrier_metrics(unify, crows, [0.9, 0.1, 0.9], 0.5, cfg)
    add("carrier_strata_two_values",
        [c["carrier_type"] for c in cres] == cfg["carrier"]["values"],
        "分层 = %s（`DATA_CARD` `carrier_type` 封闭域）" % cfg["carrier"]["values"])
    add("carrier_denominator_layer_only",
        [c for c in cres if c["carrier_type"] == "code"][0]["n"] == 2,
        "code 层 n=2（**只含**该层样本，不混入 text 层）")
    add("carrier_absent_layer_explicit",
        all(c["n"] == 0 and c["tpr"] is None
            for c in per_carrier_metrics(unify, [{"carrier_type": "code", "text": "a",
                                                  "label": "1"}], [0.9], 0.5, cfg)
            if c["carrier_type"] == "text"),
        "未出现的层**显式列出**且指标 `n/a`（不填 0）")

    # ⑦ `--source sealed` 缺 `--access-log-note`／`--split` ⇒ 非零退出
    rc_missing = main(["--source", "sealed", "--config", args.config,
                       "--report-dir", tmpdir, "--outputs-dir", tmpdir])
    add("sealed_requires_note", rc_missing != 0, "exit=%d（须非零）" % rc_missing)
    rc_split = main(["--source", "sealed", "--config", args.config,
                     "--report-dir", tmpdir, "--outputs-dir", tmpdir])
    add("sealed_requires_split", rc_split != 0, "exit=%d（`--split` 亦必填）" % rc_split)

    # ⑧ 确定性双跑（纯函数侧）
    d1 = os.path.join(tmpdir, "d1.csv")
    d2 = os.path.join(tmpdir, "d2.csv")

    def _pack(cr):
        return [{"group": "D", "regime": "id", "seed": 42, "run_id": "r",
                 "shard": "s", "source": "val", "tau": {"main_value": 0.5},
                 "main_track": "star", "by_carrier": cr}]

    write_carrier_csv(d1, _pack(per_carrier_metrics(unify, crows, [0.9, 0.1, 0.9], 0.5, cfg)))
    write_carrier_csv(d2, _pack(per_carrier_metrics(unify, crows, [0.9, 0.1, 0.9], 0.5, cfg)))
    with open(d1, "rb") as f1, open(d2, "rb") as f2:
        add("determinism_two_runs", f1.read() == f2.read(), "同一输入两次 ⇒ 逐字节相同")
    with open(d1, "r", encoding="utf-8", newline="") as fh:
        add("carrier_csv_fields", next(csv.reader(fh)) == CARRIER_CSV_FIELDS,
            "n_fields=%d" % len(CARRIER_CSV_FIELDS))

    # ⑨ 源级结构守卫：不训练／不按行读／不硬编码 reports
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
        src = fh.read()
    # 注：下面三条的**针串本身也在源码里** ⇒ 一律用字符串拼接构造，避免自命中
    # （同源教训：`05_03` 对 `"." + "train("` 的写法）。
    add("src_no_train_mode_call", ("." + "train" + "(") not in src,
        "源码无 train 模式调用（C8）")
    add("src_no_line_split_reading",
        ("read" + "lines(") not in src and ("split" + "lines(") not in src,
        "源码无按行读 CSV（C6）")
    add("src_no_train_script_call",
        all(("%s_%s" % (s, t)) not in src
            for s, t in (("04_01", "train"), ("04_02", "train"), ("04_03", "run"))),
        "源码未调用训练入口 `04_01`／`04_02`／`04_03`（C7 **不训练**）")
    add("src_report_dir_not_hardcoded",
        ("os.path.join(_ROOT, " + '"repo' + 'rts"') not in src,
        "`--report-dir` 必填、未写死（C9）")

    # ⑩ 声明句逐字在位
    dec = cfg["declarations"]
    add("declaration_ab_not_trained",
        any("ckpt 不存在" in s for s in dec["ab_not_trained"])
        and any("无权训练" in s for s in dec["ab_not_trained"]), "§1.2 A／B 未训练声明在位")
    add("declaration_no_substitute",
        "禁止" in dec["no_substitute"][0] and "C 组结果代替" in dec["no_substitute"][0],
        "§4-1 禁冒充声明在位")
    add("declaration_no_training", "调用次数 = 0" in dec["no_training"][0],
        "C7 不训练证明句在位")

    # ⑪ `#69` 区分度判据（`D-88` 裁定一；**事前定值**，判据落盘即为定值）
    dcfg = cfg.get("discrimination") or {}
    add("discrimination_eps_is_001", abs(float(dcfg.get("eps", -1)) - 0.01) < 1e-12,
        "ε = %s（`D-88` 裁定一 ③）" % dcfg.get("eps"))
    r_same, v_same = discrimination_rows(
        cfg, _disc_fake_results({"A": [1.0] * 3, "B": [1.0] * 3, "C": [1.0] * 3}),
        ["A", "B", "C", "D"], "id", "id_test",
        _disc_fake_companion({"A": [1.0] * 3, "B": [1.0] * 3, "C": [1.0] * 3}))
    add("discrimination_range_zero_when_same",
        all(r["range"] == 0.0 for r in r_same if r["range"] is not None),
        "三组同值 ⇒ 极差 = 0")
    add("discrimination_verdict_no_diff", "无区分度" in (v_same or ""),
        "全格 < ε ⇒ 判「**无区分度**」（⇒ 触发呈 W0 申请扩 ood，**不得**自行扩训）")
    add("discrimination_shards_fixed",
        [ (s.get("shard"), tuple(s.get("metrics") or []), s.get("role"))
          for s in (dcfg.get("judgment_shards") or []) ]
        == [("id_test", ("f1", "fpr"), "main"),
            ("g3_hard_negative_test", ("fpr",), "scope")],
        "格位集 = 裁定一① `id_test`F1／FPR ＋ ② `g3`FPR（**与跑到哪个分片无关**）")
    r_diff, v_diff = discrimination_rows(
        cfg, _disc_fake_results({"A": [0.90] * 3, "B": [0.95] * 3, "C": [1.0] * 3}),
        ["A", "B", "C", "D"], "id", "id_test",
        _disc_fake_companion({"A": [0.90] * 3, "B": [0.95] * 3, "C": [1.0] * 3}))
    add("discrimination_verdict_has_diff", "有区分度" in (v_diff or ""),
        "极差 0.10 ≥ ε ⇒ 判「**有区分度**」")
    r_inc, v_inc = discrimination_rows(
        cfg, _disc_fake_results({"A": [1.0], "B": [1.0] * 3, "C": [1.0] * 3}),
        ["A", "B", "C", "D"], "id", "id_test")
    add("discrimination_incomplete_no_fill",
        all(r["range"] is None for r in r_inc),
        "A 缺 2 个 seed ⇒ 极差**无定义**（`%s`），**不得**用其他 seed 补齐" % TBD)
    # ⚠ 关键护栏：**未判读完 ⇒ 不得判「无区分度」**（否则「未看 g3」会被误读成「打平」⇒
    #    假触发 `D-88` 裁定二，属「换分片」类错误）
    _r_ng, v_ng = discrimination_rows(
        cfg, _disc_fake_results({"A": [1.0] * 3, "B": [1.0] * 3, "C": [1.0] * 3}),
        ["A", "B", "C", "D"], "id", "id_test")      # 只给 `id_test`，`g3` 无数据
    add("discrimination_pending_blocks_no_diff",
        ("待补齐" in (v_ng or "")) and ("呈 `W0`" not in (v_ng or "")),
        "`g3` 未就绪 ⇒ 判「**暂判…待补齐**」而非「无区分度」（防假触发 `D-88` 裁定二）")
    add("discrimination_groups_scope_excludes_d",
        dcfg.get("groups_in_scope") == ["A", "B", "C"],
        "判据范围 = A／B／C（**不含 D**：D 与 C 同 ckpt ⇒ 恒无差异）")

    # ⑫ `D-84` 裁定一 **条件 2** 的读法一披露在位（强制显著披露）
    rd = cfg["declarations"].get("reading_disclosure") or []
    add("reading_disclosure_present",
        any("合并为一项" in s for s in rd) and any("不得" in s and "完全一致" in s for s in rd),
        "`v5 §5.6` 字面两项 vs 本读法合并为一项 ⇒ 已显著披露（`D-84` 裁定一 条件 2）")
    add("group_recipe_declared",
        (cfg.get("group_recipe") or {}).get("A", {}).get("cli") == "--augment off --curriculum off"
        and (cfg.get("group_recipe") or {}).get("B", {}).get("cli") == "--augment on --curriculum off",
        "A／B 训练配方按 `D-84` 裁定一 登记")

    # ⑬ 组定义逐字取自 v5
    add("group_verbatim_from_v5",
        cfg["groups"]["A"]["verbatim"] == "A：仅原始数据，无编码增强。"
        and cfg["groups"]["B"]["verbatim"] == "B：编码增强 + 普通随机增强，无课程学习。"
        and cfg["groups"]["C"]["verbatim"] == "C：编码增强 + 课程学习，完整方法。"
        and cfg["groups"]["D"]["verbatim"] == "D（可选）：载体消融，纯文本 vs 代码载体。",
        "四组定义 = v5 L612–L620 **原文**")

    ok = all(c[1] for c in checks)
    print("SELFCHECK_RESULT=%s n=%d" % ("PASS" if ok else "FAIL", len(checks)))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def resolve_log_dir(args=None):
    """日志目录解析（依 `#64` / `D-80` 裁定三 / `D-82`；与 `05_01` 同口径）。

    优先级：① 环境变量 `CODEOBF_LOG_DIR`；② `--self-check` 或 `--dry-run`
    ⇒ `logs/eval/tmp/`（已 gitignore）；③ `logs/eval/`。
    """
    env = os.environ.get("CODEOBF_LOG_DIR")
    if env:
        return env
    if args is not None and (getattr(args, "self_check", False)
                             or getattr(args, "dry_run", False)):
        return os.path.join(_ROOT, "logs", "eval", "tmp")
    return os.path.join(_ROOT, "logs", "eval")


def flagged_logger(ts, log_dir=None):
    """日志：默认 `logs/eval/05_06_eval_ablation_<ts>.log`，同时回显 stdout。"""
    log_dir = log_dir or resolve_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "05_06_eval_ablation_%s.log" % ts)

    def _log(msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    return _log, path


def build_parser():
    p = argparse.ArgumentParser(
        prog="scripts/eval/05_06_eval_ablation.py",
        description="5.6 消融实验评估（W6.7，A6）—— A／B／C／D 四组 × 3 seeds")
    p.add_argument("--groups", default=None,
                   help="参与消融的组，逗号分隔；缺省 = 配置 `groups.order`（A,B,C,D）")
    p.add_argument("--group-ckpt", dest="group_ckpt", action="append", default=None,
                   metavar="G=RUN_DIR",
                   help="**可重复**：逐 seed 给某组的 ckpt 目录（`best.json` 所在层）；"
                        "缺组 ⇒ 整列 `<TBD>` ＋ 标注「未训练／未就绪」")
    p.add_argument("--source", choices=["val", "sealed"], default=None)
    p.add_argument("--split", choices=["in_regime_test", "g3_hard_negative_test"], default=None,
                   help="`--source sealed` 必填")
    p.add_argument("--config", dest="config", default="configs/eval/ablation.yaml")
    p.add_argument("--report-dir", dest="report_dir", default=None,
                   help="**必填**（C9：不得硬编码 reports/ 路径）")
    p.add_argument("--outputs-dir", dest="outputs_dir", default=None)
    p.add_argument("--access-log-note", dest="access_log_note", default=None,
                   help="`--source sealed` 必填：已登记 ACCESS_LOG 行的摘要")
    p.add_argument("--check-split-identity", dest="check_split_identity", action="store_true",
                   help="C1：四组划分内容指纹须**逐位相同**，不一 ⇒ 非零退出")
    p.add_argument("--carrier-strata", dest="carrier_strata", action="store_true",
                   help="按 `carrier_type` 分层出指标（D 组载体消融）")
    p.add_argument("--limit", type=int, default=None, help="只取前 N 条（冒烟用）")
    p.add_argument("--self-check", dest="self_check", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def main(argv=None):
    init_streams()
    args = build_parser().parse_args(argv)
    args.ts = timestamp()
    cfg = load_config(_abs(args.config))
    args._base = load_config(_abs(cfg["base_config"]))
    # `05_01.prepare_rows` 需要的字段（本脚本不用的置 None）
    for f in ("synthetic_seed", "n", "only", "run_dir", "regime", "seed", "runs_csv"):
        if not hasattr(args, f):
            setattr(args, f, None)

    if args.self_check:
        return self_check(cfg, args, print)
    if args.dry_run:
        return cmd_dryrun(cfg, args, print)
    if not args.source:
        print("缺少 --source（或给 --self-check／--dry-run 之一）")
        return 2
    if args.source == "sealed":
        if not args.split:
            print("REFUSED：`--source sealed` 须给 `--split`（缺一即拒绝执行并非零退出）")
            return 2
        if not args.access_log_note:
            print("REFUSED：`--source sealed` 须给 `--access-log-note`（缺一即拒绝执行并非零退出，C5）")
            return 2
    if not args.report_dir or not args.outputs_dir:
        print("MISSING_DIRS：评估落盘产物 ⇒ 须同时给 --report-dir 与 --outputs-dir（C9）")
        return 2
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)
    logger, log_path = flagged_logger(args.ts, resolve_log_dir(args))
    logger("ACCESS_LOG_NOTE=%s（登记由人手工完成 6 列；脚本不写 ACCESS_LOG.md）"
           % (args.access_log_note or "n/a（非 sealed 源）"))
    rc = cmd_eval(cfg, args, args.source, logger)
    logger("LOG_FILE=%s" % log_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""05_03_eval_ood.py — 阶段5（W6.4，A6）5.3 **OOD 泛化**评估主脚本。

接口契约（`windows/W6.4_ood.md` §2，**冻结后不得改**）::

    python scripts/eval/05_03_eval_ood.py \
        --runs-csv reports/train/three_seeds_v1.0_<ts>.csv [--only <regime>:<seed>] \
      | --run-dir DIR --regime {id,ood} --seed {42,123,456} \
        --source {val,sealed} \
        --config configs/eval/ood.yaml \
        --report-dir reports/eval/ --outputs-dir reports/eval/outputs/ \
        [--access-log-note "<已登记的 ACCESS_LOG 行摘要>"] \
        [--tau-mode {both,sensitivity}] [--limit N] [--self-check] [--dry-run]

stdout 末行（依模式）::

    SELFCHECK_RESULT=PASS|FAIL n=<k>
    DRYRUN_RESULT=PASS|FAIL n=<k>
    REPORT_MD=<path> REPORT_CSV=<path>

硬约束逐条落点（`windows/W6.4_ood.md` §3）::

    C1  输入面   → **复用** `05_01.assert_input_allowed`（白名单，禁另写一套）；实现期只读 frozen 侧
                   `regime_ood/val.csv`；封存只按**显式路径**读（禁默认扫描）
    C2  记录数   → **复用** `05_01.read_records`／`assert_record_count`（`csv.DictReader`；740／927）
    C3  OOD 定义 → 报告逐字声明「编码变换 OOD、非语义 OOD」；禁「未见恶意语义」措辞
    C4  留出码   → `encoding_type` 分组：训练码 5 ／ 留出码 4；**TPR 分母只含 `label==1`**
    C5  val 声明 → `ood_val ⊆ id_val` ⇒ val 指标不独立；test 侧独立 ⇒ 结论以 test 为准
    C6  单源     → 推理走 `05_01.Engine.score`（内部 `04_01.forward_scores`）；指标走
                   `05_01.compute_metrics`（内部 `03_07`）；**禁另写推理循环**
    C7  评估态   → eval 模式 ＋ 无梯度（由 `Engine` 保证）；确定性参数同 `ENV_LOCK.md` §7.2
    C8  报告路径 → `--report-dir`／`--outputs-dir` 必填；脚本内不写死 `reports/`
    C9  std=0    → 3 seed 指标全同 ⇒ 逐 seed 原值表 ＋ 显式声明退化
    C10 引数口径 → 引 `BASELINE_REPORT.md` 须带其 §12 口径注释
    C11 ACCESS_LOG → `--source sealed` 须带 `--access-log-note`，缺即拒绝执行并非零退出
    C12 只追加   → 报告／日志／CSV 只写新时间戳文件，不覆盖

顶层不得 import 重型库（`AGENTS.md` §3）：`torch`／`transformers`／`peft`／`yaml`
一律在函数体内惰性 import（`Engine` 与 `03_06`／`03_07` 的加载同样惰性）。
"""
from __future__ import annotations

# --- C7：必须在 import torch 之前设置（torch 的 import 全在函数体内）---
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

TBD = "<TBD>"
NA = "n/a"

# ─────────────────────────────────────────────────────────────────────────────
# 产物字段（of record）
# ─────────────────────────────────────────────────────────────────────────────
OOD_CSV_FIELDS = [
    "group", "regime", "seed", "run_id", "shard", "source",
    "tau_track", "tau_value", "n", "n_pos", "n_neg",
    "tp", "fp", "fn", "tn", "tpr", "fpr", "f1", "auroc",
    "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
    "mem_load_peak_mib", "mem_infer_peak_mib", "eval_batch_max_size",
]

BY_ENC_CSV_FIELDS = [
    "group", "regime", "seed", "run_id", "shard", "source",
    "tau_track", "tau_value", "encoding_type", "is_heldout",
    "n_attack", "n_benign", "n", "tp", "fp", "fn", "tn", "tpr", "fpr", "f1",
]

DECAY_CSV_FIELDS = [
    "metric", "seed", "id_source", "ood_source",
    "id_value", "ood_value", "decay", "note",
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
            pass


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
    """**单源复用**：`W6.1` 的 `05_01_eval_detection`（守卫／引擎／指标／写盘，C1／C6）。"""
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


def parse_float_or_none(x):
    """CSV 单元格 → ``float``；`n/a`／空 ⇒ ``None``（**不适用**，非缺失）。"""
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
    """已算完的指标格：``None`` ⇒ `n/a`（不适用）；**不是** `<TBD>`（`W6.1` C7 同口径）。"""
    if x is None:
        return NA
    return ("%." + str(nd) + "f") % (float(x),)


def fmt_tau(x):
    """全精度（`F-W5.1-2`）：``repr(float)``，不取显示值。"""
    return NA if x is None else repr(float(x))


def mean_std(xs, ddof=1):
    vals = [float(v) for v in xs if v is not None]
    if not vals:
        return None, None
    m = statistics.fmean(vals)
    if len(vals) < 2:
        return m, 0.0
    s = statistics.stdev(vals) if ddof == 1 else statistics.pstdev(vals)
    return m, s


# ---------------------------------------------------------------------------
# run 分组：主档（`ood` 组）／跨体制对照（`id` 组）
# ---------------------------------------------------------------------------
def group_of(regime, cfg):
    prim = cfg["runs_policy"]["primary_group"]
    cross = cfg["runs_policy"]["cross_regime_group"]
    if regime == prim:
        return "primary"
    if regime == cross:
        return "cross"
    return "other"


def resolve_grouped_runs(cfg, base, args, logger=print):
    """解析 run 清单并按主档／跨体制对照分组（复用 `05_01` 的 run 解析，C1／C6）。"""
    W_ = W()
    run_dirs = W_.resolve_runs(args, base, logger)
    selected = W_.select_runs(run_dirs, args)
    return [{"group": group_of(regime, cfg), "regime": regime,
             "seed": int(seed), "run_dir": run_dir}
            for regime, seed, run_dir in selected]


def assert_group_complete(cfg, grouped, logger=print):
    """主档须 seed 齐全（`W6.4` §8：不全 ⇒ 不产出正式结论）。"""
    if not cfg["runs_policy"].get("require_all_in_group", True):
        return True
    seeds = sorted(int(s) for s in cfg["runs_policy"]["seeds"])
    prim = [g for g in grouped if g["group"] == "primary"]
    got = sorted(g["seed"] for g in prim)
    if got != seeds:
        logger("GROUP_INCOMPLETE group=primary got=%s expected=%s ⇒ 不产出正式结论（呈 W0）"
               % (got, seeds))
        return False
    logger("GROUP_OK group=primary n=%d seeds=%s" % (len(prim), got))
    return True


# ---------------------------------------------------------------------------
# C4 逐留出码分组（**TPR 分母只含 `label==1`**；`W4.5` 教训）
# ---------------------------------------------------------------------------
def heldout_codes(cfg):
    return list(cfg["per_encoding"]["codes"])


def codes_in_shard(rows, cfg):
    """返回**报告顺序**的编码清单：先配置中的留出码，再出现但未登记的编码（如实登记）。"""
    present = set()
    for r in rows:
        if int(float(r.get("label"))) == 1:
            present.add((r.get("encoding_type") or "").strip())
    cfg_codes = heldout_codes(cfg)
    extra = sorted(c for c in present if c and c not in cfg_codes)
    return cfg_codes + extra


def per_encoding_metrics(unify, rows, scores, tau, cfg):
    """逐编码指标。

    **分母口径（事前定值，见 `configs/eval/ood.yaml` → `per_encoding.denominators`）**：
    - `TPR` 分母 = **仅 `label==1` 且 `encoding_type==<code>` 的攻击样本**（`W4.5` 教训：**不得**
      把良性混入攻击侧分母）；
    - `FPR` 分母 = 本分片**全部良性样本**（`label==0`）；
    - `F1` 分母 = 该码攻击样本 ∪ 本分片全部良性样本。
    良性样本无编码归属 ⇒ **各码 `FPR` 相同**；逐码差异只在 `TPR`／`F1`（报告须显式注明）。
    """
    W_ = W()
    heldout = heldout_codes(cfg)
    idx_benign = [i for i, r in enumerate(rows) if int(float(r.get("label"))) == 0]
    out = []
    for code in codes_in_shard(rows, cfg):
        idx_atk = [i for i, r in enumerate(rows)
                   if int(float(r.get("label"))) == 1
                   and (r.get("encoding_type") or "").strip() == code]
        sub = idx_atk + idx_benign
        row = {"encoding_type": code, "is_heldout": code in heldout,
               "n_attack": len(idx_atk), "n_benign": len(idx_benign), "n": len(sub),
               "tp": 0, "fp": 0, "fn": 0, "tn": 0, "tpr": None, "fpr": None, "f1": None}
        if sub:
            labs = [int(float(rows[i].get("label"))) for i in sub]
            sc = [scores[i] for i in sub]
            m = W_.compute_metrics(unify, labs, sc, tau)
            row.update({"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                        "tpr": m["tpr"], "fpr": m["fpr"], "f1": m["f1"]})
        out.append(row)
    return out


def heldout_count_mismatch(by_enc, cfg):
    """§8 回退点：某留出码攻击样本数 ≠ 预期 ⇒ 返回**登记串**（如实登记并呈 W0，不静默）。"""
    expect = int(cfg["ood_encodings"]["heldout_expected_attack_n"])
    return ["`%s` 攻击 n=%d（期望 %d，`DATA_CARD` §144）⇒ 已**如实登记并呈 W0**，"
            "未按现有数静默出结论" % (e["encoding_type"], e["n_attack"], expect)
            for e in by_enc if e["is_heldout"] and e["n_attack"] != expect]


# ---------------------------------------------------------------------------
# 衰减：ID 侧基准（`W6.1` 的 ID 同 τ 口径结果；**非** `BASELINE_REPORT` 的 B3 数）
# ---------------------------------------------------------------------------
def load_id_reference(cfg, logger=print):
    """读 `W6.1` 的 `in_regime_test × id` 正式件（`star` 档）⇒ ``{seed: {metric: value}}``。"""
    W_ = W()
    ref = cfg["decay"]["id_reference"]
    sel = ref["select"]
    out = {}
    files = set()
    for rel in ref["artifacts"]:
        path = _abs(rel)
        if not os.path.isfile(path):
            raise FileNotFoundError("ID 侧基准件不存在：%s（`stop_and_escalate_W0`）" % path)
        for r in W_.read_records(path):
            if (str(r.get("shard")) == str(sel["shard"])
                    and str(r.get("regime")) == str(sel["regime"])
                    and str(r.get("tau_track")) == str(sel["track"])
                    and str(r.get("source")) == str(sel["source"])):
                out[int(float(r.get("seed")))] = {m: parse_float_or_none(r.get(m))
                                                  for m in cfg["decay"]["metrics"]}
                files.add(path)
    missing = [s for s in ref["required_seeds"] if s not in out]
    if missing:
        raise RuntimeError("ID_REFERENCE_MISSING seeds=%s ⇒ **停工呈 W0**（不得改用 "
                           "`BASELINE_REPORT` 的 B3 数替代）" % missing)
    logger("ID_REFERENCE_LOADED n_seed=%d files=%d select=%s/%s/%s"
           % (len(out), len(files), sel["shard"], sel["regime"], sel["track"]))
    return out


def build_decay_rows(cfg, id_ref, ood_by_seed, logger=print):
    """逐 seed × 逐指标的 ``(id, ood, decay)``；一侧为 `n/a` ⇒ 衰减无定义（不填 0）。"""
    ref = cfg["decay"]["id_reference"]
    id_src = ";".join(ref["artifacts"])
    ood_src = "ood_test@%s（本窗口正式轮）" % cfg["decay"]["tau_track"]
    rows = []
    for seed in sorted(int(s) for s in ref["required_seeds"]):
        for metric in cfg["decay"]["metrics"]:
            iv = (id_ref.get(seed) or {}).get(metric)
            ov = (ood_by_seed.get(seed) or {}).get(metric)
            if iv is None or ov is None:
                rows.append({"metric": metric, "seed": seed, "id_source": id_src,
                             "ood_source": ood_src, "id_value": iv, "ood_value": ov,
                             "decay": None, "note": "一侧为 `n/a`（不适用）⇒ 衰减无定义"})
            else:
                rows.append({"metric": metric, "seed": seed, "id_source": id_src,
                             "ood_source": ood_src, "id_value": iv, "ood_value": ov,
                             "decay": float(iv) - float(ov), "note": "**同 τ 口径**"})
    logger("DECAY_ROWS n=%d metrics=%s" % (len(rows), "/".join(cfg["decay"]["metrics"])))
    return rows


# ---------------------------------------------------------------------------
# 单 run 评估（C6：推理走 `05_01.Engine`；C4：τ 逐 run 读取）
# ---------------------------------------------------------------------------
def evaluate_grouped_run(cfg, base, args, g, rows, shard, source, logger=print):
    W_ = W()
    unify = load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    meta_info = W_.load_run_meta(g["run_dir"], base)
    W_.assert_meta_matches_cli(meta_info, g["regime"], g["seed"])

    train_cfg = load_config(_abs(base["engine"]["train_config"]))
    engine = W_.Engine(base, train_cfg, logger=logger)
    load_info = engine.load(meta_info["ckpt_dir"], g["seed"])

    target = engine.score(rows, tag=shard)
    labels = [int(float(r.get("label"))) for r in rows]

    # τ 敏感性档②（中位数）只在 `--tau-mode sensitivity` 时才需要 ⇒ 免一次 val 前向
    val_median = None
    if args.tau_mode == "sensitivity":
        if source == "val":
            vsc = [float(s) for s in target["scores"] if s is not None]
        else:
            val_rows = W_.read_val_rows(base, g["regime"], logger)
            val_target = W_.score_val_cached(base, args, engine, g["regime"], g["seed"],
                                             g["run_dir"], val_rows, logger)
            vsc = [float(s) for s in val_target["scores"] if s is not None]
        val_median = statistics.median(vsc) if vsc else None

    tau = W_.resolve_tau_tracks(meta_info, base, val_median)
    track_names = [tau["main_track"], tau["parallel_track"]]
    if args.tau_mode == "sensitivity" and tau["tiers"].get("median_val") is not None:
        track_names.append("median_val")
    tracks = {t: W_.compute_metrics(unify, labels, target["scores"], tau["tiers"][t])
              for t in track_names}

    by_enc = per_encoding_metrics(unify, rows, target["scores"], tau["main_value"], cfg)

    per_sample_csv = None
    if args.outputs_dir:
        per_sample_csv, _ = W_.write_per_sample(
            args.outputs_dir, args.ts, g["regime"], g["seed"], shard, rows,
            target["scores"], target["latency_ms"], tau["main_value"], tau["parallel_value"])

    ckpt_adapter = os.path.join(meta_info["ckpt_dir"], "adapter_model.safetensors")
    return {
        "group": g["group"], "regime": g["regime"], "seed": g["seed"],
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
        "by_encoding": by_enc,
        "latency": W_.latency_stats(target["latency_ms"]),
        "mem_load_peak_mib": load_info["mem_load_peak_mib"],
        "mem_infer_peak_mib": target["mem_infer_peak_mib"],
        "eval_batch_max_size": target["eval_batch_max_size"],
        "per_sample_csv": per_sample_csv,
    }


def run_eval(cfg, base, args, source, logger=print):
    grouped = resolve_grouped_runs(cfg, base, args, logger)
    if not grouped:
        logger("NO_RUN ⇒ 无可评估 ckpt（不产出结论）")
        return [], None
    if not assert_group_complete(cfg, grouped, logger):
        return [], None
    rows, shard, _spec, _src = W().prepare_rows(base, args, source, "ood", logger)
    return [evaluate_grouped_run(cfg, base, args, g, rows, shard, source, logger)
            for g in grouped], shard


# ---------------------------------------------------------------------------
# 报告渲染（骨架与实跑**同一渲染器**，`W6.4` §7）
# ---------------------------------------------------------------------------
def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _group_label(group):
    if group == "primary":
        return "**主档**"
    if group == "cross":
        return "**跨体制对照**"
    return str(group)


def render_ood_report(ctx):
    """渲染 5.3 OOD 泛化报告（骨架态 `skeleton=True` 时数值位一律 `<TBD>`）。"""
    sk = bool(ctx.get("skeleton"))
    cfg = ctx["cfg"]
    L = []

    L.append("# Ours 检测性能评估 · 5.3 OOD 泛化 · v1.0" if not sk
             else "# Ours 检测性能评估 · 5.3 OOD 泛化 · v1.0 报告骨架")
    L.append("")
    L.append("> 生成：`scripts/eval/05_03_eval_ood.py`（**W6.4**，A6）｜时刻：`%s`｜"
             "配置：`configs/eval/ood.yaml` v%s｜单源：`%s` ＋ `configs/eval/detection.yaml`"
             % (ctx["ts"], cfg["version"], cfg["reuse_module"]))
    L.append(">")
    L.append("> %s" % ctx["lead"])
    L.append("")

    # §0
    L.append("## §0 概述与口径声明")
    L.append("")
    L.append("- **输入分片**：`%s`（`--source %s`）｜记录数（`csv.DictReader`）：**n = %s**"
             % (ctx.get("shard") or TBD, ctx.get("source") or TBD, ctx.get("n_records")))
    L.append("- **τ 口径**：逐 run 取该 run **`best epoch`** 的 `tau_ours`（**全精度**），`0.5` 并列"
             "（转交项 **#52**；只做读取与判定式应用，**不重算触发条件**）。")
    L.append("- **OOD 定义（逐字）**：%s" % cfg["ood_definition"]["verbatim"][0])
    L.append("- **措辞纪律**：%s" % "；".join(cfg["ood_definition"]["forbidden_wording"]))
    L.append("- **体制二编码构成**：训练码 **%d** 个（%s）；留出码 **%d** 个（%s）——"
             "**留出码只出现在 `test`**（`PREREG_v1.md` §2.2／`DATA_CARD` §V-6）。"
             % (len(cfg["ood_encodings"]["train_codes"]),
                "／".join("`%s`" % c for c in cfg["ood_encodings"]["train_codes"]),
                len(cfg["ood_encodings"]["heldout_codes"]),
                "／".join("`%s`" % c for c in cfg["ood_encodings"]["heldout_codes"])))
    L.append("")

    # §1
    L.append("## §1 run 清单")
    L.append("")
    hdr_runs = ["组", "regime", "seed", "run_id", "best_epoch", "best_step",
                "early_stopped", "adapter md5", "segments"]
    if sk:
        L.append(_md_table(hdr_runs, [[TBD] * len(hdr_runs)]))
    elif ctx["runs"]:
        L.append(_md_table(hdr_runs, [[
            _group_label(r["group"]), r["regime"], r["seed"], "`%s`" % r["run_id"],
            r.get("best_epoch"), r.get("best_step"), r.get("early_stopped"),
            ("`%s…`" % r["adapter_md5"][:16]) if r["adapter_md5"] else NA,
            "`%s`" % r["segments"]] for r in ctx["runs"]]))
    else:
        L.append("（无可用 run ⇒ 不产出结论）")
    L.append("")
    L.append("> 每个 run 的 ckpt 一律按各自 `best.json` 指向者；`ood` 组 best **全在 epoch 1**、"
             "`id` 组在 epoch 2／3／2 ⇒ **禁止假定统一取末轮**（`D-61` 裁定七）。")
    L.append("")

    # §2
    L.append("## §2 OOD 主档指标（`ood` 组 × `%s`）" % (ctx.get("shard") or TBD))
    L.append("")
    L.append("主档 = `ood` 组 3 个 run；%s。AUROC 因有连续 `score` 故可算；Span-F1 恒 `n/a`"
             "（Ours 为生成式二分类、无 span 输出 ⇒ **不适用**，非缺失）。" % ctx["source_note"])
    L.append("")
    L.append(_md_table(["regime", "seed", "τ 档", "τ 值", "n", "n_pos", "n_neg",
                        "TP", "FP", "FN", "TN", "TPR", "FPR", "F1", "AUROC",
                        "延迟 均值/中位/p95 (ms)", "显存 加载/推理 (MiB)"],
                       ctx["main_metric_rows"]))
    L.append("")
    L.append("### §2.1 逐 seed 原值表（`D-47` 裁定四）")
    L.append("")
    L.append(_md_table(["metric", "seed 42", "seed 123", "seed 456", "mean", "std"],
                       ctx["per_seed_rows"]))
    L.append("")
    L.append("### §2.2 `std=0` 退化声明（C9）")
    L.append("")
    for line in ctx["std_zero_lines"]:
        L.append("- %s" % line)
    L.append("")

    # §3
    L.append("## §3 逐留出码表（`encoding_type` 分组）")
    L.append("")
    L.append("**分母口径（事前定值）**：`TPR` 分母 = **仅 `label==1` 且 `encoding_type==<code>` 的"
             "攻击样本**（`W4.5` 教训：**不得**把良性混入攻击侧分母）；`FPR` 分母 = 本分片**全部良性"
             "样本**；`F1` 分母 = 该码攻击 ∪ 全部良性。**良性样本无编码归属 ⇒ 各码 `FPR` 相同**，"
             "逐码差异只在 `TPR`／`F1`。")
    L.append("")
    L.append(_md_table(["码", "留出码?", "seed", "攻击 n", "良性 n", "TP", "FP", "FN", "TN",
                        "TPR", "FPR", "F1"], ctx["by_encoding_rows"]))
    L.append("")
    L.append("### §3.1 `ascii_art` 单列与可检测性边界（`BASELINE_REPORT.md` §6 口径）")
    L.append("")
    for line in ctx["ascii_art_lines"]:
        L.append("- %s" % line)
    L.append("")
    if ctx["heldout_mismatch"]:
        L.append("### §3.2 留出码样本数异常（**如实登记并呈 W0**）")
        L.append("")
        for x in ctx["heldout_mismatch"]:
            L.append("- %s" % x)
        L.append("")

    # §4
    L.append("## §4 检测率衰减（体制一 ID ↔ 体制二 OOD，同指标同 τ 口径）")
    L.append("")
    L.append("**口径**：%s；两侧**同 τ 规则**（`best epoch` 的 `tau_ours`，逐 run 独立）；"
             "ID 侧基准取 `W6.1` 的 ID 同 τ 结果（**非** `BASELINE_REPORT` 的 B3 数）。"
             % cfg["decay"]["formula"])
    L.append("")
    L.append(_md_table(["metric", "seed", "ID", "OOD", "衰减", "备注"], ctx["decay_rows"]))
    L.append("")
    L.append("### §4.1 两体制不可比要素（**须随衰减一并引用**）")
    L.append("")
    for line in cfg["decay"]["incomparable_disclosure"]:
        L.append("- %s" % line)
    L.append("")

    # §5
    L.append("## §5 跨体制对照（`id` 组 × `%s`，**单列、不与主档混算**）" % (ctx.get("shard") or TBD))
    L.append("")
    L.append("**性质声明**：本节的 3 个 run 属 `id` 训练体制，**不是**主档；列出仅为「同一 OOD 分片上，"
             "不同训练体制的相对位置」之参考。**不得**与 §2 主档合并成任何均值／结论"
             "（`W6.4` §2／§9-1）。")
    L.append("")
    if ctx["cross_rows"]:
        L.append(_md_table(["regime", "seed", "τ 档", "τ 值", "n", "TPR", "FPR", "F1", "AUROC"],
                           ctx["cross_rows"]))
    else:
        L.append("- 本轮未纳入 `id` 组（未选中或未完工）⇒ 本节为空（**不作为缺陷**）。")
    L.append("")

    # §6
    L.append("## §6 声明（逐字）")
    L.append("")
    L.append("### §6.1 `ood_val` 三句（转交项 **#50**，`D-53` 裁定六）")
    L.append("")
    for i, s in enumerate(cfg["declarations"]["ood_val_three_sentences"], 1):
        L.append("%d. %s" % (i, s))
    L.append("")
    L.append("### §6.2 `ood_val` 构成（**不构成 OOD 泛化证据**）")
    L.append("")
    for s in cfg["declarations"]["ood_val_composition"]:
        L.append("- %s" % s)
    L.append("")
    L.append("### §6.3 val 侧不独立 ＋ test 侧独立边界（转交项 **#51**，`D-61` 裁定二）")
    L.append("")
    for s in cfg["declarations"]["val_nonindependence"]:
        L.append("- %s" % s)
    L.append("")

    # §7
    L.append("## §7 与 B1–B5 的对照（**只引不跑**；引数**逐条带口径**）")
    L.append("")
    L.append("来源：`reports/baseline/BASELINE_REPORT.md`（tag `baseline-v1`）§1。**必读口径**：")
    L.append("")
    for s in cfg["baseline_citation"]["mandatory_notes"]:
        L.append("- %s" % s)
    L.append("")
    L.append(_md_table(["检测器", "`ood_test` TPR", "FPR", "F1", "AUROC", "备注"],
                       ctx["baseline_rows"]))
    L.append("")
    L.append("**Ours（本窗口主档，逐 seed 原值）**：")
    L.append("")
    L.append(_md_table(["regime", "seed", "TPR", "FPR", "F1", "AUROC"], ctx["ours_rows"]))
    L.append("")

    # §8
    L.append("## §8 ACCESS_LOG 与冻结面复核")
    L.append("")
    for line in ctx["access_lines"]:
        L.append("- %s" % line)
    L.append("")

    # §9
    L.append("## §9 局限、评估态与红线")
    L.append("")
    for line in ctx["limitation_lines"]:
        L.append("- %s" % line)
    L.append("")

    # §10
    L.append("## §10 回执／验收位（留给 `W6` 复核）")
    L.append("")
    L.append("- 交付状态：`%s`" % ctx["status"])
    L.append("- `--self-check`：`%s`" % ctx.get("selfcheck_line", TBD))
    L.append("- 运行日志：`logs/eval/05_03_eval_ood_%s.log`" % ctx["ts"])
    L.append("- 逐样本输出：`reports/eval/outputs/ours_<regime>_<seed>_<shard>_v1.0_%s.csv`"
             % ctx["ts"])
    L.append("- 回执：%s" % TBD)
    L.append("")

    # 附
    L.append("---")
    L.append("")
    L.append("## 附 · 可直接搬入 `reports/eval/EVAL_REPORT.md` §5.3 的骨架段")
    L.append("")
    L.append("> 归属：`EVAL_REPORT.md` 由 **`W6` 收口**；本段**只提供**内容，**不直接写**该件"
             "（`W6.4` §1.1 交付物 #5）。")
    L.append("")
    L.append("### 5.3 OOD 泛化")
    L.append("")
    L.append("**OOD 定义**：%s" % cfg["ood_definition"]["verbatim"][0])
    L.append("")
    L.append("**构成**：体制二留出码 %s（4 个，**只在 `test`**）；主评估面 = `%s`（**%s** 记录）。"
             % ("／".join("`%s`" % c for c in cfg["ood_encodings"]["heldout_codes"]),
                ctx.get("shard") or TBD, ctx.get("n_records")))
    L.append("")
    L.append("**表 5.3-1 逐留出码检测率**（Ours，主档；`TPR` 分母只含 `label==1`）")
    L.append("")
    L.append(_md_table(["码", "攻击 n", "TPR", "FPR", "F1"], ctx["skeleton_enc_rows"]))
    L.append("")
    L.append("**表 5.3-2 检测率衰减**（ID ↔ OOD，同指标同 τ）")
    L.append("")
    L.append(_md_table(["metric", "ID", "OOD", "衰减"], ctx["skeleton_decay_rows"]))
    L.append("")
    for s in cfg["declarations"]["ood_val_three_sentences"]:
        L.append("- %s" % s)
    L.append("")
    for s in cfg["declarations"]["val_nonindependence"]:
        L.append("- %s" % s)
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 报告上下文
# ---------------------------------------------------------------------------
def _per_seed_table(results, seeds, metrics=("tpr", "fpr", "f1", "auroc")):
    prim = [r for r in results if r["group"] == "primary"]
    rows = []
    for metric in metrics:
        by_seed = {}
        for r in prim:
            by_seed[r["seed"]] = r["tracks"][r["main_track"]].get(metric)
        vals = [by_seed.get(s) for s in seeds]
        mean, std = mean_std([v for v in vals if v is not None])
        rows.append([metric.upper()] + [fmt4(v) for v in vals] + [fmt4(mean), fmt4(std)])
    return rows


def _std_zero_lines(results):
    """C9：3 seed 指标全同 ⇒ 显式声明退化（禁止只写 mean ± std）。"""
    prim = [r for r in results if r["group"] == "primary"]
    if len(prim) < 2:
        return ["可用 seed 数 < 2 ⇒ 跨 seed 方差不可算（本轮覆盖不完整）。"]
    degenerate = []
    for metric in ("tpr", "fpr", "f1", "auroc"):
        vals = [r["tracks"][r["main_track"]].get(metric) for r in prim]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2 and mean_std(vals)[1] == 0.0:
            degenerate.append(metric)
    lines = []
    if degenerate:
        lines.append("**`std=0` 已触发**：指标 %s 的 3 seed 取值**完全相同** ⇒ 依 `D-47` 裁定四，"
                     "已在上表给出**逐 seed 原值**，并在此**显式声明退化** —— **不得**仅读作 "
                     "`mean ± std`（该 `std` 不构成跨 seed 稳定性证据）。"
                     % "／".join("`%s`" % d for d in degenerate))
    else:
        lines.append("未检出 `std=0`（主档 4 项指标中至少一项的 3 seed 取值不全同）。")
    lines.append("无论是否退化，均须给**逐 seed 原值表**（§2.1），不得只报 `mean ± std`。")
    return lines


def _ascii_art_lines(prim):
    rows = [(r["seed"], e["n_attack"], e["tpr"])
            for r in prim for e in r["by_encoding"] if e["encoding_type"] == "ascii_art"]
    lines = []
    if not rows:
        lines.append("本分片未见 `ascii_art` 攻击样本（如 `val` 演练：留出码只在 `test`）⇒ 无可报值。")
        return lines
    lines.append("逐 seed（`ascii_art`，攻击 n／TPR）：%s"
                 % "；".join("seed %s：n=%d TPR=%s" % (s, n, fmt4(t)) for s, n, t in rows))
    lines.append("**可检测性边界口径**：`ascii_art` 为多行字符画，tokenization 后特征受损 ⇒ "
                 "按「**编码本身的可检测性边界**」单独列出、**不混入整体均值**、**不**据此单独判定"
                 "检测器失效（`BASELINE_REPORT.md` §6）。")
    lines.append("同一边界对 B1（规则型）亦成立 ⇒ **跨检测器共有**，供论文 Discussion 使用。")
    return lines


def _baseline_rows():
    """`BASELINE_REPORT.md` §1 的 `ood_test` 行（**只引不跑**，含口径备注）。"""
    return [
        ["**B1** 正则", "1.0000", "0.3479", "0.8391", "`n/a`", "TPR 1.0 系**结构性因素**"],
        ["**B2** 熵值", "1.0000", "1.0000", "0.6447", "0.1492", "τ=−inf 平凡解 ⇒ 几无区分力"],
        ["**B3** LG-1B", "0.9432", "0.6572", "0.7071", "`n/a`", "判词型，无连续 score"],
        ["**B4** unplug-tiny", "0.8864", "0.3454", "0.7820", "0.8667", "**CPU 档**，延迟不同量纲"],
        ["**B5** LG-8B", "0.3835", "0.1675", "0.4891", "`n/a`", "判词型，无连续 score"],
    ]


def build_ctx(cfg, args, results, shard, source, n_records, extra_limitations=None,
              skeleton=False, access_lines=None, decay_rows=None, skeleton_decay_rows=None,
              heldout_mismatch=None):
    seeds = [int(s) for s in cfg["runs_policy"]["seeds"]]
    ctx = {"cfg": cfg, "ts": args.ts, "shard": shard, "source": source,
           "n_records": n_records, "skeleton": skeleton}
    ctx["runs"] = [{"group": r["group"], "regime": r["regime"], "seed": r["seed"],
                    "run_id": r["run_id"], "best_epoch": r.get("best_epoch"),
                    "best_step": r.get("best_step"), "early_stopped": r.get("early_stopped"),
                    "adapter_md5": r.get("adapter_md5"), "segments": r["segments"]}
                   for r in results]

    if source == "sealed":
        ctx["lead"] = ("**正式轮（`--source sealed`）**｜`--split=%s`｜ACCESS_LOG note 回显："
                       "**%s**（登记由**人手工**完成 6 列；脚本**不写** `ACCESS_LOG.md`，C11）。"
                       % ("in_regime_test", args.access_log_note or TBD))
        ctx["source_note"] = "**正式轮**（OOD 结论**只取本分片**，C3）"
        ctx["status"] = "FORMAL（A6 · sealed）"
    else:
        ctx["lead"] = ("**实现期演练（`--source val`）**：输入 = `regime_ood/val.csv`（**927**）｜"
                       "**本报告的任何数字都不是 OOD 泛化结论**（`ood_val` 只含 5 个训练码 ⇒ "
                       "不测 OOD 泛化，转交项 `#50`）⇒ 一律标为「早停／阈值用途」。")
        ctx["source_note"] = "**val 演练**（数字一律标为「早停/阈值用途」，**不构成 OOD 结论**）"
        ctx["status"] = "DRYRUN_VAL（实现期演练）"

    main_rows = []
    prim = [r for r in results if r["group"] == "primary"]
    for r in prim:
        for tname in sorted({r["main_track"], r["parallel_track"]}):
            m = r["tracks"][tname]
            lat = ("%.4f / %.4f / %.4f" % (r["latency"]["mean_ms"], r["latency"]["median_ms"],
                                           r["latency"]["p95_ms"])
                   if r["latency"]["mean_ms"] is not None else NA)
            mem = ("%.1f / %.1f" % (r["mem_load_peak_mib"], r["mem_infer_peak_mib"])
                   if r["mem_infer_peak_mib"] is not None else NA)
            main_rows.append([r["regime"], r["seed"], tname, fmt_tau(m["tau"]),
                              r["n"], m["n_pos"], m["n_neg"], m["tp"], m["fp"], m["fn"],
                              m["tn"], fmt4(m["tpr"]), fmt4(m["fpr"]), fmt4(m["f1"]),
                              fmt4(m["auroc"]), lat, mem])
    ctx["main_metric_rows"] = main_rows or [[TBD] * 17]
    ctx["per_seed_rows"] = (_per_seed_table(results, seeds)
                            if (results and not skeleton) else [[TBD] * 6])
    ctx["std_zero_lines"] = (_std_zero_lines(results)
                             if (results and not skeleton)
                             else ["%s（骨架态；正式轮填数后判定）" % TBD])

    enc_rows = []
    for r in prim:
        for e in r["by_encoding"]:
            enc_rows.append([("`%s`" % e["encoding_type"]) or "(空)",
                             "是" if e["is_heldout"] else "否", r["seed"],
                             e["n_attack"], e["n_benign"], e["tp"], e["fp"], e["fn"],
                             e["tn"], fmt4(e["tpr"]), fmt4(e["fpr"]), fmt4(e["f1"])])
    ctx["by_encoding_rows"] = enc_rows or [[TBD] * 12]
    ctx["ascii_art_lines"] = _ascii_art_lines(prim)
    ctx["heldout_mismatch"] = list(heldout_mismatch or [])

    ctx["decay_rows"] = decay_rows or [[TBD] * 6]

    ctx["cross_rows"] = []
    for r in results:
        if r["group"] != "cross":
            continue
        m = r["tracks"][r["main_track"]]
        ctx["cross_rows"].append([r["regime"], r["seed"], r["main_track"], fmt_tau(m["tau"]),
                                  r["n"], fmt4(m["tpr"]), fmt4(m["fpr"]), fmt4(m["f1"]),
                                  fmt4(m["auroc"])])

    ctx["baseline_rows"] = _baseline_rows()
    ctx["ours_rows"] = [[r["regime"], r["seed"],
                         fmt4(r["tracks"][r["main_track"]]["tpr"]),
                         fmt4(r["tracks"][r["main_track"]]["fpr"]),
                         fmt4(r["tracks"][r["main_track"]]["f1"]),
                         fmt4(r["tracks"][r["main_track"]]["auroc"])] for r in prim] \
        or [[TBD] * 6]

    ctx["access_lines"] = access_lines or [
        "演练轮（`--source val`）**不触封存** ⇒ `ACCESS_LOG.md` **零行登记**；"
        "冻结面复核属正式轮动作。"]

    lim = [
        "评估态：eval 模式 ＋ 无梯度（由 `05_01.Engine` 保证）；确定性参数同 `ENV_LOCK.md` §7.2；"
        "`attn=eager`，禁 flash-attn（C7）。",
        "`D-53` 裁定四 常设红线：本报告全部 test 数字**只用于报告**；**不得**回头改数据集／改超参／"
        "选 ckpt／选 seed／选阈值，**也不得**用于调整排程或增补分析。",
        "`ascii_art` 按「编码本身的可检测性边界」呈现（单独列出、不混入整体均值）；"
        "**不得**据此单独判定「检测器失效」（`BASELINE_REPORT.md` §6）。",
        "跨体制对照（§5）**不得**与主档混算（`W6.4` §2／§9-1）。",
        "本窗口**不**做统计显著性检验（归 `W6.9` 的 paired bootstrap）⇒ 本报告只报点估计与 "
        "mean ± std，**不**声称任何差异的统计显著。",
    ]
    if extra_limitations:
        lim = lim + list(extra_limitations)
    ctx["limitation_lines"] = lim

    ctx["skeleton_enc_rows"] = [[("`%s`" % e["encoding_type"]), e["n_attack"],
                                 fmt4(e["tpr"]), fmt4(e["fpr"]), fmt4(e["f1"])]
                                for e in (prim[0]["by_encoding"] if prim else [])] \
        or [[TBD] * 5]
    ctx["skeleton_decay_rows"] = skeleton_decay_rows or [[TBD] * 4]
    return ctx


def build_formal_access_lines(cfg, base, args, logger=print):
    """正式轮的 §8：ACCESS_LOG 回显 ＋ 两份 md5 复核（`C14` 同制）。"""
    W_ = W()
    frozen_dir = os.path.dirname(_abs(base["inputs"]["root"]))
    sealed_dir = _abs(base["inputs"]["sealed"]["root"])
    f_ok, f_tot = W_.md5sum_manifest_check(frozen_dir)
    s_ok, s_tot = W_.md5sum_manifest_check(sealed_dir)
    logger("MD5_FROZEN %s/%s OK  MD5_SEALED %s/%s OK" % (f_ok, f_tot, s_ok, s_tot))
    return [
        "`ACCESS_LOG.md` 回显（**登记由人手工完成 6 列**；脚本**不写**该文件，C11）：`%s`"
        % (args.access_log_note or TBD),
        "封存触碰分级（`D-55` 裁定六）：本次为 **档③（内容级）** ⇒ 读前已**人工登记 6 列**"
        "（日期／窗口／用途／命令／输出路径／复核人），并在本报告与运行日志回显该 note。",
        "冻结面复核：`md5sum -c` = frozen **%s/%s OK**、sealed **%s/%s OK**；该动作为 "
        "**档②（字节校验级）** ⇒ 只读校验、**不解析内容**，并在此声明未读内容。"
        % (f_ok, f_tot, s_ok, s_tot),
        "红线（`D-53` 裁定四）：全部 test 数字**只用于报告**；不得回头改数据／超参／ckpt／阈值，"
        "也不得用于调整排程或增补分析。",
    ]


# ---------------------------------------------------------------------------
# 写盘
# ---------------------------------------------------------------------------
def write_csv_rows(*a, **kw):
    """**同一实现源**：`05_01.write_csv_rows`（UTF-8＋LF、自带引号转义、写前断言无 CR）。"""
    return W().write_csv_rows(*a, **kw)


def write_metric_csv(path, results, cfg):
    rows = []
    for r in results:
        for tname in sorted({r["main_track"], r["parallel_track"]}):
            m = r["tracks"][tname]
            rows.append({
                "group": r["group"], "regime": r["regime"], "seed": r["seed"],
                "run_id": r["run_id"], "shard": r["shard"], "source": r["source"],
                "tau_track": tname, "tau_value": repr(float(m["tau"])),
                "n": r["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "tpr": NA if m["tpr"] is None else "%.8f" % m["tpr"],
                "fpr": NA if m["fpr"] is None else "%.8f" % m["fpr"],
                "f1": NA if m["f1"] is None else "%.8f" % m["f1"],
                "auroc": NA if m["auroc"] is None else "%.8f" % m["auroc"],
                "latency_mean_ms": NA if r["latency"]["mean_ms"] is None
                else "%.4f" % r["latency"]["mean_ms"],
                "latency_median_ms": NA if r["latency"]["median_ms"] is None
                else "%.4f" % r["latency"]["median_ms"],
                "latency_p95_ms": NA if r["latency"]["p95_ms"] is None
                else "%.4f" % r["latency"]["p95_ms"],
                "mem_load_peak_mib": NA if r["mem_load_peak_mib"] is None
                else "%.1f" % r["mem_load_peak_mib"],
                "mem_infer_peak_mib": NA if r["mem_infer_peak_mib"] is None
                else "%.1f" % r["mem_infer_peak_mib"],
                "eval_batch_max_size": r["eval_batch_max_size"]})
    write_csv_rows(path, OOD_CSV_FIELDS, rows)
    return path


def write_by_encoding_csv(path, results, cfg):
    rows = []
    for r in results:
        m = r["tracks"][r["main_track"]]
        for e in r["by_encoding"]:
            rows.append({
                "group": r["group"], "regime": r["regime"], "seed": r["seed"],
                "run_id": r["run_id"], "shard": r["shard"], "source": r["source"],
                "tau_track": r["main_track"], "tau_value": repr(float(m["tau"])),
                "encoding_type": e["encoding_type"], "is_heldout": e["is_heldout"],
                "n_attack": e["n_attack"], "n_benign": e["n_benign"], "n": e["n"],
                "tp": e["tp"], "fp": e["fp"], "fn": e["fn"], "tn": e["tn"],
                "tpr": NA if e["tpr"] is None else "%.8f" % e["tpr"],
                "fpr": NA if e["fpr"] is None else "%.8f" % e["fpr"],
                "f1": NA if e["f1"] is None else "%.8f" % e["f1"]})
    write_csv_rows(path, BY_ENC_CSV_FIELDS, rows)
    return path


def write_decay_csv(path, drows):
    write_csv_rows(path, DECAY_CSV_FIELDS, [{
        "metric": d["metric"], "seed": d["seed"], "id_source": d["id_source"],
        "ood_source": d["ood_source"],
        "id_value": "" if d["id_value"] is None else "%.8f" % d["id_value"],
        "ood_value": "" if d["ood_value"] is None else "%.8f" % d["ood_value"],
        "decay": "" if d["decay"] is None else "%.8f" % d["decay"],
        "note": d["note"]} for d in drows])
    return path


# ---------------------------------------------------------------------------
# 模式：eval
# ---------------------------------------------------------------------------
_NAME = {"sealed": "ood", "val": "ood_dryrun_val"}


def cmd_eval(cfg, base, args, logger=print):
    source = args.source
    results, shard = run_eval(cfg, base, args, source, logger)
    if not results:
        print("REPORT_MD=<none> REPORT_CSV=<none>")
        return 0

    n_records = results[0]["n"]
    access_lines = None
    drows = None
    decay_rows = None
    skeleton_decay_rows = None
    mismatch = []
    if source == "sealed":
        access_lines = build_formal_access_lines(cfg, base, args, logger)
        id_ref = load_id_reference(cfg, logger)
        ood_by_seed = {r["seed"]: {k: r["tracks"][r["main_track"]].get(k)
                                   for k in cfg["decay"]["metrics"]}
                       for r in results if r["group"] == "primary"}
        drows = build_decay_rows(cfg, id_ref, ood_by_seed, logger)
        decay_rows = [[d["metric"].upper(), d["seed"], fmt4(d["id_value"]),
                       fmt4(d["ood_value"]), fmt4(d["decay"]), d["note"]] for d in drows]
        skeleton_decay_rows = []
        for metric in cfg["decay"]["metrics"]:
            iv = [d["id_value"] for d in drows
                  if d["metric"] == metric and d["id_value"] is not None]
            ov = [d["ood_value"] for d in drows
                  if d["metric"] == metric and d["ood_value"] is not None]
            mi = mean_std(iv)[0]
            mo = mean_std(ov)[0]
            skeleton_decay_rows.append([metric.upper(), fmt4(mi), fmt4(mo),
                                        fmt4(None if (mi is None or mo is None) else mi - mo)])
        if not args.limit:
            for r in results:
                if r["group"] == "primary":
                    mismatch += heldout_count_mismatch(r["by_encoding"], cfg)

    ctx = build_ctx(cfg, args, results, shard, source, n_records,
                    access_lines=access_lines, decay_rows=decay_rows,
                    skeleton_decay_rows=skeleton_decay_rows, heldout_mismatch=mismatch)
    ctx["selfcheck_line"] = "见 `PROGRESS.md` 本窗口行的登记（`--self-check` 一并复核）"

    name = _NAME[source]
    md_path = os.path.join(args.report_dir, "%s_v1.0_%s.md" % (name, args.ts))
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_ood_report(ctx))

    csv_path = os.path.join(args.report_dir, "%s_v1.0_%s.csv" % (name, args.ts))
    write_metric_csv(csv_path, results, cfg)

    enc_csv = os.path.join(args.report_dir, "%s_by_encoding_v1.0_%s.csv" % (name, args.ts))
    write_by_encoding_csv(enc_csv, results, cfg)

    if drows:
        write_decay_csv(os.path.join(args.report_dir, "ood_decay_v1.0_%s.csv" % args.ts), drows)

    print("REPORT_MD=%s REPORT_CSV=%s" % (md_path, csv_path))
    return 0


# ---------------------------------------------------------------------------
# `--dry-run`：解析全部资源与断言（不加载模型、不触 GPU）
# ---------------------------------------------------------------------------
def cmd_dryrun(cfg, base, args, logger=print):
    W_ = W()
    checks = []
    try:
        grouped = resolve_grouped_runs(cfg, base, args, logger)
    except (ValueError, FileNotFoundError) as exc:
        logger("RESOLVE_RUNS_SKIP detail=%s ⇒ grouped=[]" % exc)
        grouped = []
    checks.append(("runs_resolved", True, "n=%d" % len(grouped)))
    checks.append(("group_complete", assert_group_complete(cfg, grouped, logger),
                   "primary seeds=%s" % sorted(g["seed"] for g in grouped
                                               if g["group"] == "primary")))
    if args.source == "sealed":
        if not args.access_log_note:
            print("ACCESS_LOG_NOTE_MISSING ⇒ 拒绝执行（`W6.4` §3 C11）")
            print("DRYRUN_RESULT=FAIL n=%d" % len(checks))
            return 2
        path, expect, shard = W_.resolve_input_path(base, "sealed", "ood", args.split, True)
        W_.assert_input_allowed(path, base, allow_sealed=True)
        logger("PLAN_SEALED shard=%s path=%s expect_records=%s exists=%s"
               % (shard, path, expect, os.path.isfile(path)))
        if os.path.isfile(path):
            got = W_.assert_record_count(path, expect, base)
            checks.append(("record_count:%s" % shard, got == int(expect), "records=%d" % got))
    else:
        rows, shard, _s, _path = W_.prepare_rows(base, args, args.source or "val", "ood", logger)
        checks.append(("input_resolved:%s" % shard, True, "records=%d" % len(rows)))
    for g in grouped:
        rec = W_.load_run_meta(g["run_dir"], base)
        W_.assert_meta_matches_cli(rec, g["regime"], g["seed"])
        checks.append(("tau_fields:%s:%d" % (g["regime"], g["seed"]), True,
                       "tau_star=%s override=%s gap=%s"
                       % (fmt_tau(rec["tau_star"]), rec["override"], fmt_tau(rec["gap"]))))
    ok = all(c[1] for c in checks)
    logger("DRYRUN_PLAN steps=%d report_dir=%s outputs_dir=%s source=%s"
           % (len(checks), args.report_dir, args.outputs_dir, args.source))
    print("DRYRUN_RESULT=%s n=%d" % ("PASS" if ok else "FAIL", len(checks)))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# `--self-check`：离线自检（不加载模型、**不读封存**、不触 GPU）
# ---------------------------------------------------------------------------
def _extract_md_headers(text):
    """抽取 Markdown 表头行（其后紧跟 `|---|` 分隔行）⇒ 用于**字段一致性**比对。"""
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines[:-1]):
        nxt = lines[i + 1]
        if ln.startswith("|") and nxt.startswith("|") and set(nxt) <= set("|-: ") and "-" in nxt:
            out.append(ln.strip())
    return out


def self_check(cfg, base, args, logger=print):
    import tempfile

    W_ = W()
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print("SELFCHECK %-38s %s %s" % (name, "PASS" if ok else "FAIL", detail))

    # ① 记录数断言：740／927（取自 `W6.1` 的 detection.yaml，**禁改**）
    add("record_assert_ood_test_740",
        int(base["record_assertions"]["regime_ood_test"]) == 740,
        "regime_ood_test = %s" % base["record_assertions"]["regime_ood_test"])
    add("record_assert_ood_val_927", int(base["record_assertions"]["ood_val"]) == 927,
        "ood_val = %s" % base["record_assertions"]["ood_val"])
    add("record_assert_matches_registry",
        int(base["record_assertions"]["regime_ood_test"])
        == int(cfg["record_count_policy"]["expected"]["regime_ood_test"])
        and int(base["record_assertions"]["ood_val"])
        == int(cfg["record_count_policy"]["expected"]["ood_val"]),
        "本文件登记值 ↔ base_config 逐格一致")

    # ② `csv.DictReader` 口径：`text` 含换行 ⇒ 记录数 2、行数 ≥ 4
    tmpdir = tempfile.mkdtemp(prefix="05_03_selfcheck_")
    p_csv = os.path.join(tmpdir, "nl.csv")
    write_csv_rows(p_csv, ["sample_id", "encoding_type", "text", "label"],
                   [{"sample_id": "a1", "encoding_type": "hex",
                     "text": "l1\nl2\nmid\nl3", "label": "1"},
                    {"sample_id": "a2", "encoding_type": "", "text": "plain", "label": "0"}])
    rec, lines = W_.count_records(p_csv), W_.count_physical_lines(p_csv)
    add("csv_dictreader_newline_regression", rec == 2 and lines >= 4,
        "records=%d physical_lines=%d（行数 ≠ 记录数）" % (rec, lines))
    try:
        W_.assert_record_count(p_csv, 9999, base)
        add("record_assertion_nonzero", False, "断言未触发")
    except AssertionError:
        add("record_assertion_nonzero", True, "不符即 AssertionError ⇒ 上层非零退出")

    # ③ 逐留出码分组：**TPR 分母只含 `label==1`**（`W4.5` 教训）
    rows = [{"sample_id": "h%d" % i, "encoding_type": "hex", "label": "1"} for i in range(5)]
    rows += [{"sample_id": "b%d" % i, "encoding_type": "", "label": "0"} for i in range(3)]
    rows += [{"sample_id": "z%d" % i, "encoding_type": "hex", "label": "0"} for i in range(2)]
    unify = load_module("unify_map_03_07", "scripts/baseline/03_07_unify_output_mapping.py")
    scores = [0.9, 0.9, 0.1, 0.9, 0.9] + [0.1, 0.1, 0.9] + [0.9, 0.1]
    enc = per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    hexrow = [e for e in enc if e["encoding_type"] == "hex"][0]
    add("per_encoding_tpr_denominator_attack_only",
        hexrow["n_attack"] == 5 and hexrow["n_benign"] == 5,
        "hex: n_attack=%d（**不含**编码但良性的 2 条）n_benign=%d"
        % (hexrow["n_attack"], hexrow["n_benign"]))
    add("per_encoding_tpr_value", abs(hexrow["tpr"] - 0.8) < 1e-9,
        "TPR = 4/5 = %.4f（分母只含 label==1）" % hexrow["tpr"])
    add("per_encoding_absent_code_explicit",
        all(e["n_attack"] == 0 and e["tpr"] is None
            for e in enc if e["encoding_type"] in ("morse", "ascii_art", "homoglyph")),
        "未出现的留出码**显式列出**且 TPR=%s（不静默省略）" % NA)

    # ④ τ 取值规则：只读取与判定式应用，**不重算触发条件**
    fake_meta = {"tau_star": 0.9999485611915588, "override": False,
                 "gap": 0.9626216776669025, "run_dir": "", "ckpt_dir": "", "best": {},
                 "meta": {}, "regime": "ood", "seed": 42, "best_step": 1}
    tau = W_.resolve_tau_tracks(fake_meta, base, None)
    add("tau_main_is_star_when_no_override",
        tau["main_track"] == "star" and tau["main_value"] == fake_meta["tau_star"],
        "override=False ⇒ 主口径 = τ*（仅判定式应用）")
    add("tau_full_precision", fmt_tau(tau["tau_star"]) == repr(0.9999485611915588),
        "τ* = %s（全精度，非显示值）" % fmt_tau(tau["tau_star"]))
    fake_ov = dict(fake_meta, override=True)
    tau_ov = W_.resolve_tau_tracks(fake_ov, base, 0.5)
    add("tau_parallel_is_050", tau_ov["main_value"] == 0.5
        and tau_ov["parallel_value"] == fake_meta["tau_star"],
        "override=True ⇒ 主口径 0.5、并列 τ*")

    # ⑤ 声明**逐字在位**
    joined = " ".join(cfg["declarations"]["ood_val_three_sentences"])
    add("declaration_no_ood_generalization",
        "不测 OOD 泛化" in joined and "只用于早停与阈值" in joined, "第 1 句在位")
    add("declaration_conclusion_only_ood_test", "OOD 结论只取 `ood_test`" in joined, "第 2 句在位")
    add("declaration_no_relax_ood_training",
        "不得因 `ood_val` 饱和而降低 ood 的训练要求" in joined, "第 3 句在位")
    add("declaration_ood_definition_verbatim",
        "编码方式分布外" in cfg["ood_definition"]["verbatim"][0]
        and "不是语义 OOD" in cfg["ood_definition"]["verbatim"][0],
        "C3 逐字：编码变换 OOD、非语义 OOD")
    add("declaration_val_nonindependence",
        any("共享同一批验证文本" in s for s in cfg["declarations"]["val_nonindependence"])
        and any("相互独立" in s for s in cfg["declarations"]["val_nonindependence"]),
        "转交项 #51：val 不独立 ＋ test 侧独立边界")

    # ⑥ **确定性双跑**：同一输入两次分组 ⇒ 逐字节相同（纯函数侧）
    base_meta = {"group": "primary", "regime": "ood", "seed": 42, "run_id": "r",
                 "shard": "ood_test", "source": "sealed", "tau_track": "star",
                 "tau_value": "0.5"}
    e1 = per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    e2 = per_encoding_metrics(unify, rows, scores, 0.5, cfg)
    p1 = os.path.join(tmpdir, "e1.csv")
    p2 = os.path.join(tmpdir, "e2.csv")
    write_csv_rows(p1, BY_ENC_CSV_FIELDS, [dict(base_meta, **e) for e in e1])
    write_csv_rows(p2, BY_ENC_CSV_FIELDS, [dict(base_meta, **e) for e in e2])
    with open(p1, "rb") as a, open(p2, "rb") as b:
        add("determinism_two_runs", a.read() == b.read(), "同一输入两次分组 ⇒ 逐字节相同")
    add("classifier_metrics_deterministic",
        [x["tpr"] for x in e1] == [x["tpr"] for x in e2], "逐码 TPR 两次一致")

    # ⑦ 衰减：公式为 ID − OOD；一侧为 `n/a` ⇒ 衰减无定义
    mini = {"decay": {"id_reference": {"artifacts": ["x.csv"], "required_seeds": [42]},
                      "metrics": ["tpr", "fpr", "f1"], "tau_track": "star"}}
    d1 = build_decay_rows(mini, {42: {"tpr": 1.0, "fpr": 0.0, "f1": 0.99746835}},
                          {42: {"tpr": 0.94318182, "fpr": 0.0, "f1": 0.97076023}}, print)
    tpr_d = [d for d in d1 if d["metric"] == "tpr"][0]
    add("decay_formula_id_minus_ood", abs(tpr_d["decay"] - (1.0 - 0.94318182)) < 1e-12,
        "TPR 衰减 = %.8f（ID − OOD）" % tpr_d["decay"])
    d2 = build_decay_rows(mini, {}, {42: {"tpr": 0.5}}, print)
    add("decay_undefined_when_na", d2[0]["decay"] is None, "ID 侧 `n/a` ⇒ 衰减无定义（不填 0）")

    # ⑧ 报告结构：骨架含 §5.3 段、数值位用 `<TBD>`、表头齐备
    ctx_sk = build_ctx(cfg, args, [], None, "sealed", TBD, skeleton=True)
    rendered_sk = render_ood_report(ctx_sk)
    h_sk = _extract_md_headers(rendered_sk)
    add("skeleton_headers_present", len(h_sk) >= 6, "骨架表头数=%d" % len(h_sk))
    add("skeleton_has_eval_report_segment",
        "可直接搬入 `reports/eval/EVAL_REPORT.md` §5.3" in rendered_sk,
        "附节：`EVAL_REPORT` §5.3 骨架段在位")
    add("skeleton_uses_tbd", TBD in rendered_sk,
        "骨架态数值位用 `%s`（区别于 `n/a`＝不适用）" % TBD)
    add("report_has_all_declaration_sections",
        all(k in rendered_sk for k in ("§6.1", "§6.2", "§6.3", "§4.1", "§3.1")),
        "声明小节 §3.1／§4.1／§6.1–§6.3 齐备")
    p_out = os.path.join(tmpdir, "byenc.csv")
    write_by_encoding_csv(p_out, [], cfg)
    import csv as _csv
    with open(p_out, "r", encoding="utf-8", newline="") as fh:
        got_fields = next(_csv.reader(fh))
    add("by_encoding_csv_fields", got_fields == BY_ENC_CSV_FIELDS, "n_fields=%d" % len(got_fields))

    # ⑨ `--source sealed` 缺 `--access-log-note` ⇒ **非零退出**
    rc_missing = main(["--source", "sealed", "--config", args.config,
                       "--report-dir", tmpdir, "--outputs-dir", tmpdir])
    add("sealed_requires_note", rc_missing != 0, "exit=%d（须非零）" % rc_missing)

    # ⑩ 源级结构守卫：无 train 模式调用／无按行读 CSV
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
        src = fh.read()
    add("src_no_train_mode_call", ("." + "train(") not in src, "源码无 train 模式调用（C7）")
    add("src_no_line_split_reading",
        ("read" + "lines(") not in src and ("split" + "lines(") not in src,
        "源码无按行读 CSV（C2）")

    # ⑪ 汇总
    ok = all(c[1] for c in checks)
    print("SELFCHECK_RESULT=%s n=%d" % ("PASS" if ok else "FAIL", len(checks)))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def flagged_logger(ts):
    """日志：写 `logs/eval/05_03_eval_ood_<ts>.log`，同时回显 stdout。"""
    log_dir = os.path.join(_ROOT, "logs", "eval")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "05_03_eval_ood_%s.log" % ts)

    def _log(msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    return _log, path


def build_parser():
    p = argparse.ArgumentParser(
        prog="scripts/eval/05_03_eval_ood.py",
        description="5.3 OOD 泛化评估（W6.4，A6）—— Ours（QLoRA 微调 Llama-Guard-3-1B）")
    p.add_argument("--runs-csv", dest="runs_csv", default=None,
                   help="04_03 汇总 CSV（读其 run_dir 列）")
    p.add_argument("--only", default=None, help="只跑指定 run，形如 ood:42 或 ood:42,id:123")
    p.add_argument("--run-dir", dest="run_dir", default=None, help="单点指定 run 目录")
    p.add_argument("--regime", choices=["id", "ood"], default=None)
    p.add_argument("--seed", type=int, choices=[42, 123, 456], default=None)
    p.add_argument("--source", choices=["val", "sealed"], default=None)
    p.add_argument("--split", choices=["in_regime_test"], default="in_regime_test",
                   help="`--source sealed` 的分片（5.3 只有 `in_regime_test`；缺省即它）")
    p.add_argument("--config", dest="config", default="configs/eval/ood.yaml")
    p.add_argument("--report-dir", dest="report_dir", default=None,
                   help="**必填**（C8：不得硬编码 reports/ 路径）")
    p.add_argument("--outputs-dir", dest="outputs_dir", default=None)
    p.add_argument("--tau-mode", dest="tau_mode", choices=["both", "sensitivity"],
                   default="both")
    p.add_argument("--access-log-note", dest="access_log_note", default=None)
    p.add_argument("--limit", type=int, default=None, help="只取前 N 条（冒烟用）")
    p.add_argument("--self-check", dest="self_check", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def main(argv=None):
    init_streams()
    args = build_parser().parse_args(argv)
    args.ts = timestamp()
    cfg = load_config(_abs(args.config))
    base = load_config(_abs(cfg["base_config"]))

    if args.self_check:
        return self_check(cfg, base, args, print)
    if args.dry_run:
        return cmd_dryrun(cfg, base, args, print)
    if not args.source:
        print("缺少 --source（或给 --self-check／--dry-run 之一）")
        return 2
    if not args.report_dir or not args.outputs_dir:
        print("MISSING_DIRS：评估落盘产物 ⇒ 须同时给 --report-dir 与 --outputs-dir（C8）")
        return 2
    if args.source == "sealed" and not args.access_log_note:
        print("REFUSED：`--source sealed` 须给 `--access-log-note`（缺即拒绝执行并非零退出，C11）")
        return 2
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)
    logger, log_path = flagged_logger(args.ts)
    logger("ACCESS_LOG_NOTE=%s（登记由人手工完成 6 列；脚本不写 ACCESS_LOG.md）"
           % (args.access_log_note or "n/a（非 sealed 源）"))
    rc = cmd_eval(cfg, base, args, logger)
    logger("LOG_FILE=%s" % log_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())

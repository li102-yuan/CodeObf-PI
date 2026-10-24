# -*- coding: utf-8 -*-
"""05_09_mechanism_evidence.py

用途（`#99`／`D-145`/`D-146` ③ 机制证据包）：
从既有评估产物零 GPU 重算并产出：
  1. 表1 2×2 交互表（v1 原始训练权重 / v2 去混淆重训，体制 × 编码 × 良性/攻击）
  2. 图1 分数分布图（编码良性；v1 vs v2，2 行 regime × 4 列 code）
  3. 图2 τ* 点图（12 run；方案 A，定稿图）

输出：
  reports/analysis/mechanism_tables_v1.0_<ts>.csv      （机读，lineterminator="\\n"）
  reports/analysis/fig1_score_dist_v1.0_<ts>.png
  reports/analysis/fig2_tau_dot_v1.0_<ts>.png
  reports/analysis/mechanism_numbers_v1.0_<ts>.json    （数字清单＋质量探针，供报告引用）

★ 方案 B（FPR(τ) 校准曲线）**不产出**（`D-148` §三）：`threshold_search` 的 `tau` 列
  262–706 行仅 30–176 个不同值、大量行打印 0.0 ⇒ 不可作横轴；用逐样本重算亦不可行
  （训练期 val 良性人群 ≠ 探针文件人群）。本脚本只保留 `probe_tau_column()` 量化该缺陷。

数据源（全部只读既有产物，零训练／零 API／零封存访问）：
  - reports/eval/encoded_benign_fpr_persample_v1.0_{id42,rest}.csv      （v1 编码良性 + 明文良性对照）
  - reports/eval/plaintext_attack_probe_persample_v1.0_20261019-215210.csv （v1 明文攻击）
  - reports/eval/deconf_fourcell_persample_{id,ood}_seed{42,123,456}_pooled_*.csv （v2 四格）
  - reports/train/threshold_search_ours_{id,ood}_*.csv                  （τ 曲线，6 位小数）
  - checkpoints/ours/<12 run 目录>/best.json                            （权威 τ，全精度）

★ 判命中一律用 **best.json 全精度 τ**，不用 threshold_search 的 6 位小数：
  大量模型输出分数恰好等于 τ 的全精度值（如 ood/seed42 morse 33 条 = 0.9999945759773254），
  6 位小数一舍 ⇒ 并列样本全部漏杀。v2 列不受影响，因为直接用了 deconf 文件里算好的 pred_main。
  threshold_search 的 selected 行仅用于「曲线 / 一致性断言」，容差 1e-6。

W0 代执行（依 `D-14` 惯例）；一次性分析脚本，与常驻脚本同规：
写 CSV 显式 lineterminator="\\n"，写文本 UTF-8（errors=replace）。
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL = os.path.join(ROOT, "reports", "eval")
TRAIN = os.path.join(ROOT, "reports", "train")
CKPT = os.path.join(ROOT, "checkpoints", "ours")
OUTDIR = os.path.join(ROOT, "reports", "analysis")

# v1（原始训练权重）6 run：checkpoint 目录（已排除 *_INTERRUPTED / *_NAN_INVALID）
V1_CKPT_DIRS = {
    ("id", 42): "04_01_llamaguard1b_id_seed42_20261005-150806",
    ("id", 123): "04_01_llamaguard1b_id_seed123_20261006-010210",
    ("id", 456): "04_01_llamaguard1b_id_seed456_20261006-120248",
    ("ood", 42): "04_01_llamaguard1b_ood_seed42_20261007-144100",
    ("ood", 123): "04_01_llamaguard1b_ood_seed123_20261007-180342",
    ("ood", 456): "04_01_llamaguard1b_ood_seed456_20261007-202634",
}
# v2（去混淆重训）6 run
V2_CKPT_DIRS = {
    ("id", 42): "04_01_llamaguard1b_deconf_id_seed42_20261019-233428",
    ("id", 123): "04_01_llamaguard1b_deconf_id_seed123_20261020-085239",
    ("id", 456): "04_01_llamaguard1b_deconf_id_seed456_20261020-205233",
    ("ood", 42): "04_01_llamaguard1b_deconf_ood_seed42_20261021-071812",
    ("ood", 123): "04_01_llamaguard1b_deconf_ood_seed123_20261021-095114",
    ("ood", 456): "04_01_llamaguard1b_deconf_ood_seed456_20261021-115657",
}
V1_TAU_FILES = {
    ("id", 42): "threshold_search_ours_id_20261005-150806.csv",
    ("id", 123): "threshold_search_ours_id_20261006-010210.csv",
    ("id", 456): "threshold_search_ours_id_20261006-120248.csv",
    ("ood", 42): "threshold_search_ours_ood_20261007-144100.csv",
    ("ood", 123): "threshold_search_ours_ood_20261007-180342.csv",
    ("ood", 456): "threshold_search_ours_ood_20261007-202634.csv",
}
V2_TAU_FILES = {
    ("id", 42): "threshold_search_ours_id_20261019-233428.csv",
    ("id", 123): "threshold_search_ours_id_20261020-085239.csv",
    ("id", 456): "threshold_search_ours_id_20261020-205233.csv",
    ("ood", 42): "threshold_search_ours_ood_20261021-071812.csv",
    ("ood", 123): "threshold_search_ours_ood_20261021-095114.csv",
    ("ood", 456): "threshold_search_ours_ood_20261021-115657.csv",
}
CODES = ("hex", "morse", "ascii_art", "homoglyph")
SEEDS = (42, 123, 456)
TAU_TOL = 1e-6  # threshold_search(6 dp) vs best.json(全精度) 的一致性容差


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def guard_ckpt_dirs(mapping):
    """排除中断／数值异常的目录（防把 *_INTERRUPTED / *_NAN_INVALID 当有效 run）。"""
    for key, d in mapping.items():
        if d.endswith("_INTERRUPTED") or d.endswith("_NAN_INVALID"):
            raise SystemExit("BAD_CKPT_DIR %r -> %s" % (key, d))
        if not os.path.isdir(os.path.join(CKPT, d)):
            raise SystemExit("MISSING_CKPT_DIR %r -> %s" % (key, d))


def load_best_tau():
    """12 run 的权威 τ（best.json 全精度）；同时交叉校验 threshold_csv 指向。"""
    out = {}
    for ver, dirs, tau_files in (
        ("v1", V1_CKPT_DIRS, V1_TAU_FILES),
        ("v2", V2_CKPT_DIRS, V2_TAU_FILES),
    ):
        for key, d in dirs.items():
            with open(os.path.join(CKPT, d, "best.json"), encoding="utf-8") as f:
                b = json.load(f)
            csv_ref = os.path.basename(str(b.get("threshold_csv", "")).replace("\\", "/"))
            if csv_ref != tau_files[key]:
                raise SystemExit(
                    "THRESHOLD_CSV_MISMATCH %s %r: best.json=%r mapping=%r"
                    % (ver, key, csv_ref, tau_files[key])
                )
            out[(ver,) + key] = {
                "tau": float(b["tau"]),
                "tau_neg_max_gap": float(b.get("tau_neg_max_gap", float("nan"))),
                "val_f1_at_tau": float(b.get("val_f1_at_tau", b.get("val_f1", float("nan")))),
                "checkpoint_dir": d,
            }
    return out


def load_v1_scores():
    """v1 逐样本分数。返回 (benign_by_key, plain_atk_by_key)。

    benign_by_key 键 = (regime, seed, encoding, sample_id)，含 4 码与 plain（明文对照就在同一对文件里）。
    """
    benign, plain_atk = {}, {}
    for fn in sorted(glob.glob(os.path.join(EVAL, "encoded_benign_fpr_persample_v1.0_*.csv"))):
        for r in read_csv(fn):
            key = (r["regime"], int(r["seed"]), r["encoding"], r["sample_id"])
            benign[key] = float(r["score"])
    for r in read_csv(
        os.path.join(EVAL, "plaintext_attack_probe_persample_v1.0_20261019-215210.csv")
    ):
        key = (r["regime"], int(r["seed"]), r["src_sample_id"])
        plain_atk[key] = float(r["score"])
    return benign, plain_atk


def load_v2_scores():
    """v2 四格逐样本。键：(regime, seed, arm, code, sample_id) → (score, pred_main)。"""
    rows = {}
    for fn in sorted(glob.glob(os.path.join(EVAL, "deconf_fourcell_persample_*_pooled_*.csv"))):
        if "full" in os.path.basename(fn):
            continue
        parts = os.path.basename(fn).split("_")
        regime = parts[3]
        seed = int(parts[4].replace("seed", ""))
        for r in read_csv(fn):
            arm = r["arm"]
            code = r["code"] if r["code"] else "plain"
            key = (regime, seed, arm, code, r["sample_id"])
            rows[key] = (float(r["score"]), int(r["pred_main"]))
    return rows


def load_threshold_rows():
    """12 run 的 τ 曲线 + selected 行（6 位小数）。"""
    out = {}
    for ver, mapping in (("v1", V1_TAU_FILES), ("v2", V2_TAU_FILES)):
        for (regime, seed), fn in mapping.items():
            rows = read_csv(os.path.join(TRAIN, fn))
            selected, curve = None, []
            for r in rows:
                try:
                    tau, f1, tpr, fpr = (
                        float(r["tau"]),
                        float(r["f1"]),
                        float(r["tpr"]),
                        float(r["fpr"]),
                    )
                except ValueError:
                    continue
                if not math.isfinite(tau):
                    continue  # 首行 τ=-inf，仅作曲线起点，不入统计
                curve.append((tau, f1, tpr, fpr))
                if r.get("selected") == "yes":
                    selected = {"tau": tau, "f1": f1, "tpr": tpr, "fpr": fpr}
            out[(ver, regime, seed)] = {"file": fn, "selected": selected, "curve": curve}
    return out


def count_v1_benign(scores, best_tau, regime, code):
    """按 best.json 全精度 τ 统计 v1 良性命中（code 可为 'plain'）。"""
    n = hit = 0
    for k, s in scores.items():
        if k[0] != regime or k[2] != code:
            continue
        n += 1
        if s >= best_tau[("v1", regime, k[1])]["tau"]:
            hit += 1
    return n, hit


def count_v1_attack(scores, best_tau, regime):
    n = hit = 0
    for k, s in scores.items():
        if k[0] != regime:
            continue
        n += 1
        if s >= best_tau[("v1", regime, k[1])]["tau"]:
            hit += 1
    return n, hit


def count_v2(v2_rows, regime, arm, code=None):
    n = hit = 0
    for k, v in v2_rows.items():
        if k[0] != regime or k[2] != arm:
            continue
        if code is not None and k[3] != code:
            continue
        n += 1
        hit += v[1]
    return n, hit


def build_interaction_table(v1_benign, v1_plain_atk, v2_rows, best_tau):
    """2×2 交互表：regime × {4 码 / plain} × {良性编码, 良性明文, 攻击明文, 攻击编码} × {v1, v2}。"""

    def cell(regime, code, arm, v1=None, v2=None):
        return {
            "regime": regime,
            "code": code,
            "arm": arm,
            "v1_n": "" if v1 is None else v1[0],
            "v1_hit": "" if v1 is None else v1[1],
            "v1_rate": "" if v1 is None or not v1[0] else round(v1[1] / v1[0], 6),
            "v2_n": "" if v2 is None else v2[0],
            "v2_hit": "" if v2 is None else v2[1],
            "v2_rate": "" if v2 is None or not v2[0] else round(v2[1] / v2[0], 6),
        }

    rows = []
    for regime in ("id", "ood"):
        # 良性 · 4 码编码
        for code in CODES:
            rows.append(
                cell(
                    regime,
                    code,
                    "benign_encoded",
                    v1=count_v1_benign(v1_benign, best_tau, regime, code),
                    v2=count_v2(v2_rows, regime, "enc_benign", code),
                )
            )
        # 良性 · 明文对照（v1 的 plain 行来自同一对 persample 文件）
        rows.append(
            cell(
                regime,
                "plain",
                "benign_plain",
                v1=count_v1_benign(v1_benign, best_tau, regime, "plain"),
                v2=count_v2(v2_rows, regime, "plain_benign"),
            )
        )
        # 攻击 · 明文
        rows.append(
            cell(
                regime,
                "plain",
                "attack_plain",
                v1=count_v1_attack(v1_plain_atk, best_tau, regime),
                v2=count_v2(v2_rows, regime, "plain_attack"),
            )
        )
        # 攻击 · 编码（v1 无对照臂 ⇒ n/hit 留空，不写 0）
        rows.append(
            cell(
                regime,
                "payload",
                "attack_encoded",
                v1=None,
                v2=count_v2(v2_rows, regime, "enc_attack"),
            )
        )
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("WROTE", path, "rows", len(rows))


def _pct(x):
    if x == "" or x is None:
        return "—"
    return "%.2f%%" % (100 * x)


def make_fig1(v1_benign, v2_rows, table, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rate = {(r["regime"], r["code"], r["arm"]): r for r in table}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharey=True)
    bins = [x / 40 for x in range(41)]
    for i, regime in enumerate(("id", "ood")):
        for j, code in enumerate(CODES):
            ax = axes[i, j]
            v1 = [s for k, s in v1_benign.items() if k[0] == regime and k[2] == code]
            v2 = [
                v[0]
                for k, v in v2_rows.items()
                if k[0] == regime and k[2] == "enc_benign" and k[3] == code
            ]
            ax.hist(v1, bins=bins, alpha=0.55, color="#c0392b", density=True)
            ax.hist(v2, bins=bins, alpha=0.55, color="#2980b9", density=True)
            ax.set_title(f"{regime} / {code}", fontsize=10)
            ax.set_xlim(0, 1)
            if i == 1:
                ax.set_xlabel("unsafe score", fontsize=9)
            if j == 0:
                ax.set_ylabel("density (per 0.025 bin)", fontsize=9)
            ax.tick_params(labelsize=8)
            row = rate[(regime, code, "benign_encoded")]
            ax.text(
                0.03,
                0.96,
                "n=%s\nv1 FPR %s\nv2 FPR %s"
                % (row["v2_n"], _pct(row["v1_rate"]), _pct(row["v2_rate"])),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.85),
            )
    handles = [
        Patch(facecolor="#c0392b", alpha=0.55, label="v1 — original training"),
        Patch(facecolor="#2980b9", alpha=0.55, label="v2 — deconfounded retrain"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=9.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.004),
    )
    fig.suptitle(
        "Fig. 1 — Unsafe-score distribution of encoded benign samples "
        "(frozen val, pooled over 3 seeds; corner: n and FPR@τ* per cell)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("WROTE", out_png)


def make_fig2_dot(best_tau, out_png):
    """方案 A：τ* 点图，每行一个 run。不画点间连线（点间无函数关系）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [(ver, reg, s) for ver in ("v1", "v2") for reg in ("id", "ood") for s in SEEDS]
    ys = list(range(len(order) - 1, -1, -1))
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.axvspan(0.988, 1.03, color="#c0392b", alpha=0.07, zorder=0)
    ax.text(1.005, ys[0] + 0.55, "τ*→1.0", ha="center", va="bottom", fontsize=8, color="#8e2a20")
    for y, key in zip(ys, order):
        ver, reg, seed = key
        t = best_tau[key]["tau"]
        color = "#c0392b" if ver == "v1" else "#2980b9"
        ax.plot([t], [y], "o", color=color, markersize=8, zorder=3)
        ax.text(t - 0.010, y, "%.5f" % t, ha="right", va="center", fontsize=8, color=color)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{v} · {r} · seed {s}" for v, r, s in order], fontsize=9)
    ax.set_xlim(0.45, 1.04)
    ax.set_ylim(-0.9, ys[0] + 1.1)
    ax.set_xlabel("selected τ* (best.json, full precision)")
    ax.grid(axis="x", alpha=0.3)
    ysplit = (ys[5] + ys[6]) / 2
    ax.axhline(ysplit, color="#999999", lw=0.8, ls=":")
    ax.set_title(
        "Fig. 2 — Selected threshold τ* per run (12 runs)\n"
        "v1: 5/6 runs pinned near 1.0, 1/6 (ood/seed123) interior; "
        "v2: all 6 interior (0.53–0.92)",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("WROTE", out_png)


def probe_tau_column(thr):
    """★ 数据质量探针（2026-10-22 发现）：`reports/train/threshold_search_ours_*.csv` 的
    `tau` 列**不可当作阈值横轴** —— 行数 262–706，但不同 τ 仅 30–176 个，且大量行打印为
    `0.0`（v1 id/42：706 行 / 30 个不同 τ / 642 行 = 0.0），与同一文件的 f1/tpr/fpr
    单调变化互相矛盾。⇒ 方案 B（FPR(τ) 校准曲线）**无法从这些文件忠实重绘**，
    须重跑阈值搜索（需加载模型）。本函数把证据量化后写入 JSON，供报告引用。
    selected 行不受影响：其 τ 与 best.json 全精度在 1e-6 内一致（12/12）。
    """
    out = {}
    for (ver, regime, seed), meta in sorted(thr.items()):
        rows = read_csv(os.path.join(TRAIN, meta["file"]))
        taus = [r["tau"] for r in rows if r.get("tau")]
        distinct = sorted(set(taus))
        n_zero = sum(1 for t in taus if t == "0.0")
        out[f"{ver}/{regime}/{seed}"] = {
            "file": meta["file"],
            "rows": len(taus),
            "distinct_tau": len(distinct),
            "rows_tau_printed_zero": n_zero,
            "usable_as_axis": False,
        }
    return out


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(OUTDIR, exist_ok=True)

    guard_ckpt_dirs(V1_CKPT_DIRS)
    guard_ckpt_dirs(V2_CKPT_DIRS)
    best_tau = load_best_tau()
    thr = load_threshold_rows()

    # 断言（12 run）：threshold_search 的 selected τ（6 dp）与 best.json 全精度一致，容差 1e-6
    max_diff, mism = 0.0, []
    for key in best_tau:
        sel = thr[key]["selected"]
        if sel is None:
            mism.append((key, "no selected row", best_tau[key]["tau"]))
            continue
        d = abs(sel["tau"] - best_tau[key]["tau"])
        max_diff = max(max_diff, d)
        if d > TAU_TOL:
            mism.append((key, sel["tau"], best_tau[key]["tau"]))
    if mism:
        raise SystemExit("TAU_MISMATCH: %r" % (mism,))
    print("TAU_CHECK %d/%d ok, max_abs_diff=%.3e (tol=%.0e)" % (len(best_tau), len(best_tau), max_diff, TAU_TOL))

    v1_benign, v1_plain_atk = load_v1_scores()
    v2_rows = load_v2_scores()

    table = build_interaction_table(v1_benign, v1_plain_atk, v2_rows, best_tau)
    csv_path = os.path.join(OUTDIR, f"mechanism_tables_v1.0_{ts}.csv")
    write_csv(
        csv_path,
        table,
        ["regime", "code", "arm", "v1_n", "v1_hit", "v1_rate", "v2_n", "v2_hit", "v2_rate"],
    )

    fig1 = os.path.join(OUTDIR, f"fig1_score_dist_v1.0_{ts}.png")
    fig2 = os.path.join(OUTDIR, f"fig2_tau_dot_v1.0_{ts}.png")
    make_fig1(v1_benign, v2_rows, table, fig1)
    make_fig2_dot(best_tau, fig2)
    tau_col = probe_tau_column(thr)

    # 数字清单
    pooled = {}
    for regime in ("id", "ood"):
        for arm in ("enc_benign", "plain_benign", "enc_attack", "plain_attack"):
            n, hit = count_v2(v2_rows, regime, arm)
            pooled[f"{regime}/{arm}"] = {
                "n": n,
                "hit": hit,
                "rate": round(hit / n, 6) if n else None,
            }
    score_shift = {}
    for regime in ("id", "ood"):
        ds = []
        for k, s_v1 in v1_benign.items():
            if k[0] != regime or k[2] not in CODES:
                continue  # 仅四码；plain 不参与 v1↔v2 分数位移匹配
            kk = (regime, k[1], "enc_benign", k[2], k[3])
            if kk in v2_rows:
                ds.append(v2_rows[kk][0] - s_v1)
        score_shift[regime] = {
            "n": len(ds),
            "mean_delta": round(statistics.mean(ds), 6),
            "median_delta": round(statistics.median(ds), 6),
        }
    numbers = {
        "generated_at": ts,
        "notes": {
            "threshold": "判命中一律用 best.json 全精度 τ；threshold_search 的 selected 行（6 位小数）仅用于 τ 曲线与一致性断言（容差 1e-6）",
            "tau_check": {
                "runs_checked": len(best_tau),
                "max_abs_diff_6dp_vs_full": float(max_diff),
                "tolerance": TAU_TOL,
            },
            "score_shift_provenance": "由 reports/eval/encoded_benign_fpr_persample_v1.0_{id42,rest}.csv 与 deconf_fourcell_persample_*_pooled_*.csv 逐样本键匹配后直接复算（非引用既有记录）；n = 可匹配的四码样本数",
            "population": "训练期冻结 val（未访问 data/sealed_test/）；直方图为 3 seed 池化",
        },
        "pooled_v2": pooled,
        "score_shift_v1_to_v2": score_shift,
        "data_quality_threshold_search": tau_col,
        "fig2b_status": "未产出：threshold_search 的 tau 列不可作横轴（见 data_quality_threshold_search），"
        "方案 B 的 FPR(τ) 曲线须重跑阈值搜索才能忠实重绘",
        "tau_best_json_full_precision": {
            f"{v}/{r}/{s}": best_tau[(v, r, s)]["tau"] for (v, r, s) in sorted(best_tau)
        },
        "tau_selected_6dp_from_threshold_search": {
            f"{v}/{r}/{s}": thr[(v, r, s)]["selected"] for (v, r, s) in sorted(thr)
        },
        "sources": {
            "v1_benign_and_plain": "reports/eval/encoded_benign_fpr_persample_v1.0_{id42,rest}.csv",
            "v1_plain_attack": "reports/eval/plaintext_attack_probe_persample_v1.0_20261019-215210.csv",
            "v2": "reports/eval/deconf_fourcell_persample_*_pooled_*.csv",
            "tau_full": "checkpoints/ours/<12 run 目录>/best.json",
            "tau_curve": "reports/train/threshold_search_ours_{id,ood}_*.csv",
        },
    }
    json_path = os.path.join(OUTDIR, f"mechanism_numbers_v1.0_{ts}.json")
    with open(json_path, "w", encoding="utf-8", newline="") as f:
        json.dump(numbers, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("WROTE", json_path)
    print("DONE ts", ts)


if __name__ == "__main__":
    main()

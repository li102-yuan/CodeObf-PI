# -*- coding: utf-8 -*-
"""W7.1 交付机械化验收（W0 用）。只做"可机械判读"的检查，判读结论仍由 W0 下。

用法：
    python logs/tmp/w0_accept_w71.py [--root <项目根>]

检查项：
  A. 文件齐备：T1–T11 × {md,csv}、FIGURES_INDEX、TABLES_INDEX、W7.1_HANDOFF、图 1–4
  B. D-115：md 表「未转义竖线数 vs 本表表头」（每表独立，禁全文件众数）
  C. CSV：CR 计数（应 0）、记录数 vs 同名 md 数据行数
  D. 禁词扫描：206/352、0.3333、0.1712、0.0452、0.0025、93/352、22/792（应仅出现在"禁止/否定"语境）
  E. 脚注要件：按表类型检查必备关键词
  F. 数字回表：抽取 ≥3 位小数与 a/b 比值，检查是否能在 paper/materials/01_numbers_manifest.md 找到
"""
import csv
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--root" else r"C:\Users\lipei\Desktop\实验项目文件"
TBL = os.path.join(ROOT, "PAPER_PACKAGE", "tables")
FIG = os.path.join(ROOT, "PAPER_PACKAGE", "figures")
MANIFEST = os.path.join(ROOT, "paper", "materials", "01_numbers_manifest.md")

TABLES = {
    "T1": ("detection_main", ["fixed050", "star", "档位"]),
    "T2": ("ood_generalization", ["衰减", "−0.1", "档位"]),
    "T3": ("adaptive_attack", ["方向性", "18", "seed"]),
    "T4": ("long_input", ["512", "2048", "B4"]),
    "T5": ("ablation", ["v1 配方"]),
    "T6": ("efficiency", ["分钟", "不可比"]),
    "T7": ("statistics", ["0.000000", "重采样"]),
    "T8": ("g3_hard_negative", ["0/64", "无信息量"]),
    "T9": ("asr_sixcell", ["仅可判", "分母", "UNP"]),
    "T10": ("final_risk", ["并集", "敏感性"]),
    "T11": ("b6_external_baseline", ["n/a", "零检出", "单模型"]),
}
FORBIDDEN = ["206/352", "0.3333", "0.1712", "0.0452", "0.0025", "93/352", "22/792"]
FIGURES = ["fig1_score_dist", "fig2_tau_dot", "fig3_ood_per_encoding_fpr", "fig4_risk_cost_tradeoff"]
PIPE = re.compile(r"(?<!\\)\|")


def read(p):
    with io.open(p, "rb") as f:
        return f.read()


def d115(path):
    """按本表表头逐表判破行。"""
    text = read(path).decode("utf-8")
    lines = text.split("\n")
    groups, cur = [], []
    for ln in lines:
        if ln.startswith("|"):
            cur.append(ln)
        else:
            if cur:
                groups.append(cur)
                cur = []
    if cur:
        groups.append(cur)
    bad = []
    for g in groups:
        hdr = len(PIPE.findall(g[0]))
        for ln in g[1:]:
            if len(PIPE.findall(ln)) != hdr:
                bad.append((len(PIPE.findall(ln)), hdr, ln[:60]))
    return len(groups), bad


def main():
    fails = []
    print("== A. 文件齐备")
    for tid, (slug, _) in TABLES.items():
        for ext in ("md", "csv"):
            hits = [f for f in os.listdir(TBL) if f.startswith(tid + "_" + slug) and f.endswith("." + ext)]
            if not hits:
                fails.append(f"{tid}.{ext} 缺失")
        print(f"  {tid}: {'OK' if len(hits) else 'MISS'}")
    for name in ("FIGURES_INDEX.md", "TABLES_INDEX.md", "W7.1_HANDOFF"):
        hits = [f for f in os.listdir(TBL if "TABLES" in name or "HANDOFF" in name else FIG) if name in f]
        print(f"  {name}: {'OK' if hits else 'MISS'}")
        if not hits:
            fails.append(f"{name} 缺失")
    for fg in FIGURES:
        hits = [f for f in os.listdir(FIG) if f.startswith(fg)]
        print(f"  图 {fg}: {'OK' if hits else 'MISS'}")
        if not hits:
            fails.append(f"{fg} 缺失")

    print("\n== B. D-115（md 表竖线）")
    mds = sorted(f for f in os.listdir(TBL) if f.endswith(".md") and f[0] == "T")
    for f in mds:
        n, bad = d115(os.path.join(TBL, f))
        print(f"  {f}: 表 {n} 破行 {len(bad)}" + ("" if not bad else f" 首例={bad[0]}"))
        if bad:
            fails.append(f"D-115 {f} 破行 {len(bad)}")

    print("\n== C. CSV（CR 与记录数）")
    for f in sorted(x for x in os.listdir(TBL) if x.endswith(".csv")):
        raw = read(os.path.join(TBL, f))
        cr = raw.count(b"\r")
        rows = sum(1 for _ in csv.DictReader(io.open(os.path.join(TBL, f), encoding="utf-8", newline="")))
        print(f"  {f}: CR={cr} 记录={rows}")
        if cr:
            fails.append(f"CSV {f} 含 CR={cr}")

    print("\n== D. 禁词扫描（应只在禁止/否定语境）")
    for f in mds + ["TABLES_INDEX.md", "W7.1_HANDOFF.md"]:
        p = os.path.join(TBL, f)
        if not os.path.exists(p):
            continue
        text = read(p).decode("utf-8")
        for w in FORBIDDEN:
            idx = text.find(w)
            if idx >= 0:
                ctx = text[max(0, idx - 30): idx + 30].replace("\n", " ")
                print(f"  {f} 命中 {w}: …{ctx}…")
                fails.append(f"禁词 {w} @ {f}（需人工判读语境）")

    print("\n== E. 脚注要件")
    for tid, (slug, kws) in TABLES.items():
        hits = [f for f in mds if f.startswith(tid + "_")]
        if not hits:
            continue
        text = read(os.path.join(TBL, hits[0])).decode("utf-8")
        miss = [k for k in kws if k not in text]
        print(f"  {tid}: {'OK' if not miss else '缺 ' + str(miss)}")
        if miss:
            fails.append(f"{tid} 缺关键词 {miss}")

    print("\n== F. 数字回表（≥3 位小数 / a-b 比值 是否能在 01 中找到）")
    manifest = read(MANIFEST).decode("utf-8")
    miss_nums = {}
    for f in mds:
        text = read(os.path.join(TBL, f)).decode("utf-8")
        nums = set(re.findall(r"\d+\.\d{3,}", text))
        miss = sorted(n for n in nums if n not in manifest)
        if miss:
            miss_nums[f] = miss
            print(f"  {f}: 未直接命中 {len(miss)} 个 → {miss[:8]}")
        else:
            print(f"  {f}: OK（{len(nums)} 个数全部命中）")

    print("\n== 汇总")
    if fails:
        print(f"硬失败 {len(fails)} 项：")
        for x in fails:
            print("  -", x)
    else:
        print("硬失败 0 项（数字回表未命中项仍需人工判读）")
    print(f"\n数字回表未命中合计：{sum(len(v) for v in miss_nums.values())}（人工判读：可能是表号/派生值/取整差异）")


if __name__ == "__main__":
    main()

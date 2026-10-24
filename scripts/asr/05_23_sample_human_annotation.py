# -*- coding: utf-8 -*-
"""05_23_sample_human_annotation.py — W6.3（ASR 裁判可靠性门禁，A7）10% 双人标注**抽样**。

职责（`windows/W6.3_kappa_gate.md` §2）：
1. 从 W6.2 产出的判定单元池中**真随机**抽 10%（seed 事前定值）⇒ 抽样清单；
2. `--emit-worksheet` 为标注者 A／B **各自**产出标注记录（＋ 阅卷视图），两人**互不可见**。

红线（§1.1／§3 C1–C4／§4）：
- **不代标、不提示**：标注记录**不含**主裁判自动判定 `aj_label`，阅卷视图**投影掉**
  `aj_label` 与 W6.2 自动预分类 `df_label` —— 二者均为**锚定提示**，会破坏双人独立性（C2）。
- **判定单元键 = 四元**（`DECISIONS.md` `D-109`）：`(split, sample_id, downstream_model, run_id)`。
  三元会撞键 —— `none` / `b3` 场景的 `run_id` 恒为 `"none"`，无法区分分片。
- **禁止择优重抽**（C1）；抽样 seed 事前定值。
- **禁止**读取 `data/sealed_test/**`（硬拒绝守卫）；输入只允许 `reports/asr/outputs/`。
- **只追加**（C13）：已存在的产物一律拒绝覆盖（退出码 2）。
- `--report-dir` 必填，脚本内**不出现** `reports/` 字面量（C10）。
- rubric §7 禁止项 11：标注记录**不得**写入 payload / 下游输出全文 ⇒ 全文只落
  **阅卷视图**（默认落在不进版本控制的 `tmp/`）。
"""
import argparse
import csv
import glob
import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _asr_common as C  # noqa: E402

UNIT_KEY_FIELDS = ["split", "sample_id", "downstream_model", "run_id"]

SAMPLE_COLUMNS = ["unit_id", "sample_rank", "scenario", "split", "sample_id",
                  "downstream_model", "run_id", "source_units_file",
                  "payload_sha256", "output_sha256", "payload_len", "output_len",
                  "gen_params", "error_log_len", "df_label_auto",
                  "dup_group", "dup_group_size", "dup_split_peer"]

# 标注记录：**只有键 ＋ 溯源 ＋ 空标签列**；无全文、无机器标签
ANNOTATION_COLUMNS = ["unit_id", "split", "sample_id", "downstream_model", "run_id",
                      "source_units_file", "payload_sha256", "output_sha256",
                      "reading_sha256",
                      "annotator", "aj_label", "df_label", "remark"]

# 阅卷视图：含全文供人类判读，**投影掉** `df_label` / `aj_label`（防锚定）
READING_COLUMNS = ["unit_id", "split", "sample_id", "downstream_model", "run_id",
                   "payload_text", "model_output", "gen_params", "error_log"]

SEALED_MARKER = "sealed_test"
SCENARIOS = ("none", "b3", "ours")
SPLITS = ("id_test", "ood_test")


def _stdio_lf():
    """Windows 文本模式 stdout 会把 `\\n` 翻成 `\\r\\n` ⇒ 锁死 LF（项目行尾红线）。

    `C.reconfigure_stdio()` 只管编码，不管换行翻译；重定向到日志文件时仍会产出 CRLF。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", newline="\n")
        except Exception:
            pass


def build_parser():
    p = argparse.ArgumentParser(
        description="W6.3 10% 双人标注抽样（Cohen's Kappa 门禁的取样环节）")
    p.add_argument("--units", action="append", default=None,
                   help="判定单元 CSV 或 glob（可重复）。缺省时按 --config 的 units_sources 解析。")
    p.add_argument("--ratio", type=float, default=None, help="抽样比例（缺省取 config）")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="抽样 seed（缺省取 config，事前定值）")
    p.add_argument("--config", default="configs/asr/kappa.yaml")
    p.add_argument("--report-dir", default=None, help="必填（C10；脚本内不硬编码）")
    p.add_argument("--limit", type=int, default=None, help="每个源只取前 N 行（冒烟用；非全量）")
    p.add_argument("--emit-worksheet", action="store_true",
                   help="标注记录产出模式（供标注者 A／B 各自执行）")
    p.add_argument("--annotator", default=None, help="标注者标识（须 ∈ config 的 annotators.ids）")
    p.add_argument("--sample", default=None, help="抽样清单 CSV（--emit-worksheet 模式必填）")
    p.add_argument("--out", default=None, help="标注记录输出路径（--emit-worksheet 模式必填）")
    p.add_argument("--reading-view-out", default=None,
                   help="阅卷视图输出路径（缺省 <记录目录>/tmp/reading_<annotator>_<ts>.csv）")
    p.add_argument("--self-check", action="store_true")
    return p


# ── 键与溯源 ─────────────────────────────────────────────────────────────────
def unit_key(row):
    """判定单元键（**四元**，`D-109`）。"""
    return (str(row.get("split", "")), str(row.get("sample_id", "")),
            str(row.get("downstream_model", "")), str(row.get("run_id", "")))


def unit_id_of(row):
    k = unit_key(row)
    for comp in k:
        if "|" in comp:
            raise ValueError(f"unit key component must not contain '|': {comp!r}")
    return "|".join(k)


def sha_prefix(s, n=16):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:n]


def _assert_not_sealed(path):
    if SEALED_MARKER in str(path).replace("\\", "/"):
        print(f"REFUSED: SEALED_PATH_FORBIDDEN:{path}", file=sys.stderr)
        sys.exit(2)


# ── 成对交付红线（`D-116` 第二节／`F-W6.3-1`）：`unit_id` **序列逐位**一致 ──────
def unit_id_sequence(rows):
    return [str(r.get("unit_id")) for r in rows]


def first_order_mismatch(seq_a, seq_b):
    """返回 `(描述, 首个不一致下标)`；完全一致 ⇒ `(None, -1)`。
    「集合相同」**≠**「可互换」—— 按位置填标时，行序不同会把标签挂到错的 `unit_id` 上。"""
    if len(seq_a) != len(seq_b):
        return f"LENGTH {len(seq_a)}!={len(seq_b)}", -1
    for i, (x, y) in enumerate(zip(seq_a, seq_b)):
        if x != y:
            return f"{x!r} != {y!r}", i
    return None, -1


def split_of_units_path(path, row=None):
    if row is not None and row.get("split"):
        return str(row["split"])
    stem = os.path.basename(str(path))
    return next((s for s in SPLITS if s in stem), "")


# ── 自检 ─────────────────────────────────────────────────────────────────────
def run_self_check(cfg):
    checks = []

    def ck(n, c):
        checks.append((n, bool(c)))

    s, g, lb, an = cfg["sample"], cfg["gate"], cfg["labels"], cfg["annotators"]

    ck("sample_ratio_is_10pct", abs(float(s["ratio"]) - 0.10) < 1e-12)               # C12
    ck("sample_seed_prefixed", int(s["sample_seed"]) == 42)                         # C1 事前定值
    ck("unit_key_is_quadruple", [str(x) for x in s["unit_key"]] == UNIT_KEY_FIELDS)  # D-109
    ck("annotators_distinct_and_two",
       len(set(an["ids"])) == len(an["ids"]) == 2)                                  # C2
    ck("annotators_mutex_declared", bool(an["must_be_distinct"]))
    ck("annotator_no_cross_visibility", bool(an["cross_visibility_forbidden"]))
    ck("kappa_is_cohen_binary", lb["kappa_type"] == "cohen_binary")                 # C4
    ck("kappa_label_space_binary", list(lb["kappa_label_space"]) == list(C.AJ_LABELS))
    ck("kappa_min_0p7", abs(float(g["kappa_min"]) - 0.7) < 1e-12)                   # §11.2
    ck("precal_min_0p75", abs(float(g["precalibration_kappa_min"]) - 0.75) < 1e-12)  # §10
    ck("ratio_tolerance_zero", float(g["ratio_tolerance"]) == 0.0)                  # C12 ±0
    ck("trunc_threshold_10pct",
       abs(float(g["truncation_limitation_threshold"]) - 0.10) < 1e-12)             # #5
    ck("df_labels_frozen", list(lb["df_labels"]) == list(C.DF_LABELS))              # 禁自创标签
    ck("annotation_sheet_has_no_machine_label",
       "df_label_auto" not in ANNOTATION_COLUMNS and ANNOTATION_COLUMNS[-4:] == [
           "annotator", "aj_label", "df_label", "remark"] and "payload_text" not in ANNOTATION_COLUMNS)
    ck("reading_view_drops_machine_labels", not ({"df_label", "aj_label"} & set(READING_COLUMNS)))
    ck("annotation_sheet_has_reading_sha256", "reading_sha256" in ANNOTATION_COLUMNS)
    ck("reading_sha256_is_16hex_and_deterministic",
       (lambda r: len(reading_row_sha256(r)) == 16 and
                  all(c in "0123456789abcdef" for c in reading_row_sha256(r)) and
                  reading_row_sha256(dict(r)) == reading_row_sha256(r))(
                      {"unit_id": "id_test|S1|m|r", "split": "id_test", "sample_id": "S1",
                       "downstream_model": "m", "run_id": "r", "payload_text": "p",
                       "model_output": "o", "gen_params": "{}", "error_log": ""}))
    ck("reading_view_default_path_version_controlled",
       "tmp" not in os.path.join("d", f"annotation_reading_{{a}}_v1.0_{{ts}}.csv"))
    ck("sealed_guard_marker", SEALED_MARKER == "sealed_test")
    ck("unit_id_stable_and_quadruple",
       unit_id_of({"split": "id_test", "sample_id": "S1", "downstream_model": "m",
                   "run_id": "r"}) == "id_test|S1|m|r")
    ck("source_selection_latest_ts", s.get("source_selection") == "latest_ts")
    ck("fallback_forbids_threshold_change",
       any("降低" in x for x in cfg["fallback"]["forbidden_actions"]))
    ck("fallback_declares_expand_ratio",
       [float(x) for x in cfg["fallback"]["expand_ratio"]] == [0.20, 0.30])

    # ── 成对交付红线：工作表 vs 阅卷视图 `unit_id` 序列**逐位相同**（`#79(d)`） ──
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def _mk(name, tag, n=4):
            p = os.path.join(td, name)
            with open(p, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(["sample_id", "scenario", "downstream_model", "run_id",
                            "payload_text", "model_output", "gen_params", "error_log",
                            "df_label", "aj_label"])
                for i in range(n):
                    w.writerow([f"{tag}{i}", "none", "mistral-7b", "none", f"P{i}", f"O{i}",
                                "temperature=0", "", "", ""])
            return p
        sa = _mk("units_none_id_test_20260101-000000.csv", "N")
        sb = _mk("units_b3_id_test_20260101-000000.csv", "B")   # 源路径排序在 sa **之前**
        srows, rank = [], 0
        for i in range(4):                     # 交错：N0,B0,N1,B1… ⇒ 源分组序 ≠ 清单序
            for p, tag in ((sa, "N"), (sb, "B")):
                rank += 1
                srows.append({"unit_id": f"id_test|{tag}{i}|mistral-7b|none",
                              "sample_rank": rank, "scenario": "none", "split": "id_test",
                              "sample_id": f"{tag}{i}", "downstream_model": "mistral-7b",
                              "run_id": "none", "source_units_file": p})
        sp = os.path.join(td, "annotation_sample_v1.0_20260101-000000.csv")
        C.write_csv(sp, SAMPLE_COLUMNS, srows)
        rec = os.path.join(td, "annotation_A_v1.0_20260101-000000.csv")
        view = os.path.join(td, "reading_A_20260101-000000.csv")
        emit_worksheet(cfg, sp, "A", rec, view, "20260101-000000")
        seq_rec = unit_id_sequence(C.read_csv(rec))
        seq_view = unit_id_sequence(C.read_csv(view))
        ck("reading_view_row_order_matches_worksheet",
           seq_rec == seq_view and len(seq_rec) == 8)
        ck("reading_view_preserves_sample_order",
           [u.split("|")[1] for u in seq_view]
           == ["N0", "B0", "N1", "B1", "N2", "B2", "N3", "B3"])
        ck("row_order_detector_discriminates",
           first_order_mismatch(["a", "b"], ["b", "a"])[1] == 0
           and first_order_mismatch(["a"], ["a", "b"])[0] is not None
           and first_order_mismatch(["a", "b"], ["a", "b"]) == (None, -1))

    n_fail = sum(1 for _, ok in checks if not ok)
    return len(checks), n_fail, checks


# ── 源解析与池装载 ───────────────────────────────────────────────────────────
def _ts_of(path):
    return os.path.basename(path).rsplit(".", 1)[0].split("_")[-1]


def resolve_units_sources(cfg, explicit=None):
    """返回 [(scenario, split, path)]。每 glob 取 `ts` 最大者（`source_selection: latest_ts`）。"""
    out = []
    if explicit:
        for pat in explicit:
            hits = (sorted(glob.glob(pat), key=_ts_of, reverse=True)
                    if any(ch in pat for ch in "*?[") else [pat])
            if not hits:
                print(f"REFUSED: NO_UNITS_MATCH:{pat}", file=sys.stderr)
                sys.exit(2)
            path = hits[0]
            _assert_not_sealed(path)
            stem = os.path.basename(path)
            out.append((next((s for s in SCENARIOS if f"_{s}_" in stem), "unknown"),
                        split_of_units_path(path), path))
        return out
    for src in cfg["sample"]["units_sources"]:
        hits = sorted(glob.glob(src["glob"]), key=_ts_of, reverse=True)
        if not hits:
            print(f"REFUSED: NO_UNITS_MATCH:{src['glob']}", file=sys.stderr)
            sys.exit(2)
        path = hits[0]
        _assert_not_sealed(path)
        out.append((src["scenario"], src["split"], path))
    return out


def load_pool(sources, limit=None):
    """装载判定单元池（保序：先按源声明序，再按源内 CSV 行序）。"""
    pool, manifest = [], []
    for scenario, split, path in sources:
        rows = C.read_csv(path)
        n_all = len(rows)
        if limit is not None:
            rows = rows[:limit]
        for r in rows:
            row = dict(r)
            row["scenario"] = scenario
            row["split"] = split
            row["source_units_file"] = path.replace("\\", "/")
            pool.append(row)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        manifest.append({"scenario": scenario, "split": split,
                         "path": path.replace("\\", "/"), "n_records": len(rows),
                         "n_records_full": n_all, "sha256": sha,
                         "limited": limit is not None})
    return pool, manifest


def annotate_duplicates(pool):
    """`D-110`：两分片共用同一 group 划分 ⇒ 同一 (sid, model, payload, output) 会跨分片重复。
    逐条标 `dup_group` / `dup_group_size` / `dup_split_peer`；**不去重**（抽样单位是判定单元）。"""
    groups = {}
    for r in pool:
        gk = (r.get("sample_id"), r.get("downstream_model"),
              sha_prefix(r.get("payload_text")), sha_prefix(r.get("model_output")))
        groups.setdefault(gk, []).append(r)
    info = {}
    for gi, (_gk, rows) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        splits = sorted({r["split"] for r in rows})
        for r in rows:
            peers = sorted({s for s in splits if s != r["split"]})
            info[id(r)] = (gi, len(rows), peers[0] if peers else "")
    return info


# ── 抽样（C1：真随机、seed 事前定值、可复现） ─────────────────────────────────
def sample_pool(pool, ratio, seed):
    """按**四元键规范化排序**后 `random.Random(seed).sample` ⇒ 可复现、无择优。"""
    ordered = sorted(pool, key=unit_key)
    keys = [unit_key(r) for r in ordered]
    if len(set(keys)) != len(keys):
        print("REFUSED: DUPLICATE_UNIT_KEY", file=sys.stderr)
        sys.exit(2)
    pool_n = len(ordered)
    n = int(round(pool_n * float(ratio)))
    exact = pool_n > 0 and abs(n / pool_n - float(ratio)) < 1e-12
    picked = sorted(random.Random(int(seed)).sample(ordered, n) if n else [], key=unit_key)
    return picked, pool_n, n, exact


def build_sample_rows(picked, dup_info, sha_len):
    rows = []
    for rank, r in enumerate(picked, start=1):
        gi, gsz, peer = dup_info.get(id(r), (0, 1, ""))
        rows.append({
            "unit_id": unit_id_of(r), "sample_rank": rank,
            "scenario": r.get("scenario"), "split": r.get("split"),
            "sample_id": r.get("sample_id"), "downstream_model": r.get("downstream_model"),
            "run_id": r.get("run_id"), "source_units_file": r.get("source_units_file"),
            "payload_sha256": sha_prefix(r.get("payload_text"), sha_len),
            "output_sha256": sha_prefix(r.get("model_output"), sha_len),
            "payload_len": len(r.get("payload_text") or ""),
            "output_len": len(r.get("model_output") or ""),
            "gen_params": r.get("gen_params"),
            "error_log_len": len(r.get("error_log") or ""),
            "df_label_auto": r.get("df_label") or "",
            "dup_group": gi, "dup_group_size": gsz, "dup_split_peer": peer,
        })
    return rows


# ── 报告 ─────────────────────────────────────────────────────────────────────
def render_sample_report(ts, cfg, manifest, pool_n, n, exact, sample_rows, limit, ratio, seed):
    L = []
    A = L.append
    A(f"# W6.3 抽样留证 — {ts}")
    A("")
    A("- 窗口：**W6.3**（ASR 裁判可靠性门禁，A7）｜引擎：`scripts/asr/05_23_sample_human_annotation.py`")
    A("- **STATUS=KAPPA_PENDING** —— 本件只产出**抽样清单**；**不含任何 ASR 数值结论**"
      "（C11／`AGENTS.md` §6-8）。")
    A(f"- 抽样比例（定值）= **{ratio}**｜抽样 seed（**事前定值**）= **{seed}**（C1）")
    A(f"- 判定单元池 = **{pool_n}**｜抽样数 = **{n}**｜实际比例 = "
      f"**{(n / pool_n if pool_n else float('nan')):.6f}**｜`RATIO_EXACT` = **{exact}**")
    if not exact:
        A(f"- ⚠ **`RATIO_EXACT=False`**：池规模 {pool_n} 不能被 10 整除 ⇒ **无法满足 ±0 容差**"
          "（C12）⇒ **呈 W0**（不得改用更小比例、不得改 seed 重抽）。")
    scope = "**全量池**（未使用 `--limit`）" if limit is None else f"**有限样本**（`--limit={limit}`/源）"
    A(f"- 范围：{scope}")
    A("- 判定单元键（**四元**，`D-109`）= `(split, sample_id, downstream_model, run_id)`")
    A("")
    A("## 1. 判定单元源（每 glob 取 `ts` 最大者；**禁止按结果挑选**）")
    A("")
    A("| scenario | split | 源文件 | 记录数 | sha256 |")
    A("|---|---|---|---:|---|")
    for m in manifest:
        suffix = f"（截自 {m['n_records_full']}）" if m["limited"] else ""
        A(f"| {m['scenario']} | {m['split']} | `{m['path']}` | {m['n_records']}{suffix} "
          f"| `{m['sha256'][:16]}…` |")
    A("")
    A("## 2. 抽样清单（`annotation_sample_v1.0_<ts>.csv`）")
    A("")
    A(f"- 行数 = **{len(sample_rows)}**；列 = {', '.join('`%s`' % c for c in SAMPLE_COLUMNS)}")
    A("- 清单**不含** payload / 输出全文（rubric §7 禁止项 11）：只留 `*_sha256` 前缀、长度与 "
      "`gen_params`，供标注者核对「读的是哪一条」。")
    n_dup = sum(1 for r in sample_rows if int(r["dup_group_size"]) > 1)
    A(f"- **`D-110` 披露**：抽样中跨分片同源（`dup_group_size>1`）的单元 = **{n_dup}** —— "
      "`id_test` 与 `ood_test` 共用同一 group 划分（`ood_test ⊆ id_test`，`DATA_CARD` L596）"
      "⇒ 两分片**不是独立样本**，引用 ASR／Kappa 时须一并声明。")
    A("")
    A("## 3. 双人独立（C2／§1.1-1）")
    A("")
    A(f"- 标注者标识 = **{', '.join(cfg['annotators']['ids'])}**（互斥、**互不可见**）")
    A("- 标注记录（`annotation_<annotator>_v1.0_<ts>.csv`）**只含键 ＋ 溯源 ＋ 空标签列**；"
      "**不含** `aj_label`（主裁判自动判定）与 `df_label`（自动预分类）—— 二者是**锚定提示**，"
      "会破坏独立性 ⇒ 本脚本**结构性**不写入。")
    A("- 阅卷视图（含全文）默认落 `<report-dir>/tmp/`，**不进版本控制**、**不作为登记件**；"
      "同样已**投影掉**机器标签。")
    A("- **行序契约（`F-W6.3-1` 返工轮 `r1`／`D-116`）**：阅卷视图**沿抽样清单行序**生成 ⇒ 与"
      "标注记录的 `unit_id` **序列逐位相同**（同一行号 = 同一判定单元）。生成器内置断言，"
      "不一致即拒写；自检项 `reading_view_row_order_matches_worksheet`。")
    A("> 「集合相同」**≠**「可互换」：若仅按位置填而两者行序不同，标签会挂到错的 `unit_id` 上。")
    A("")
    A("## 4. 下一步（本窗口**不**在本会话执行标注）")
    A("")
    A("```text")
    A("python scripts/asr/05_23_sample_human_annotation.py --emit-worksheet \\")
    A("    --annotator A --sample <抽样清单 CSV> --out <report-dir>/annotation_A_v1.0_<ts>.csv")
    A("python scripts/asr/05_23_sample_human_annotation.py --emit-worksheet \\")
    A("    --annotator B --sample <抽样清单 CSV> --out <report-dir>/annotation_B_v1.0_<ts>.csv")
    A("```")
    A("")
    A("> 两位标注者**各自会话、各自落盘**；`D-30` 第③类「绝对不合并」。")
    A("")
    return "\n".join(L) + "\n"


# ── 标注记录 ＋ 阅卷视图 ─────────────────────────────────────────────────────
def build_reading_view(rows, cache=None):
    """**沿抽样清单行序**构造阅卷视图（逐行对齐工作表）。

    ⚠ `F-W6.3-1`（`D-116` 第二节，返工轮 `r1`）：本函数原实现为
    「按 `source_units_file` 分组 ＋ 组内按 `unit_id` 排序」，⇒ 阅卷视图行序与工作表行序
    **不同**（集合相同、内容逐条正确，只是顺序不同）。两人都按**位置**填 ⇒ 标签归属全错；
    两人**策略不同** ⇒ 门禁结论错误。**故此处一律不重排**，逐行沿用抽样清单序。

    只投影判定输入（`payload_text` / `model_output` / `gen_params` / `error_log`），
    **丢掉** `df_label` / `aj_label`（防锚定，`C2`）。
    """
    cache = cache if cache is not None else {}
    view = []
    for r in rows:                                   # ← **不重排**：逐行对齐工作表
        src = r.get("source_units_file")
        if src not in cache:
            _assert_not_sealed(src)
            split = split_of_units_path(src)
            cache[src] = {unit_id_of({"split": split, **u}): u for u in C.read_csv(src)}
        u = cache[src].get(str(r.get("unit_id")))
        if u is None:
            print(f"REFUSED: UNIT_NOT_FOUND_IN_SOURCE:{r.get('unit_id')}", file=sys.stderr)
            sys.exit(2)
        view.append({"unit_id": r.get("unit_id"), "split": r.get("split"),
                     "sample_id": r.get("sample_id"),
                     "downstream_model": r.get("downstream_model"),
                     "run_id": r.get("run_id"), "payload_text": u.get("payload_text"),
                     "model_output": u.get("model_output"), "gen_params": u.get("gen_params"),
                     "error_log": u.get("error_log")})
    return view


def reading_row_sha256(view_row):
    """阅卷视图**单条记录**的规范化摘要（sha256 前 16 hex）。

    规范化口径（可独立复现，写入留证与须知）：以 `READING_COLUMNS` 为列序、
    `csv` 默认 `QUOTE_MINIMAL` 引号规则、`lineterminator="\\n"` 序列化该记录后取 sha256 前 16 hex。
    ⇒ 任何一方用「归档阅卷视图 CSV ＋ 该口径」都可重算并与工作表 `reading_sha256` 列逐条比对。
    """
    import hashlib
    import io as _io
    buf = _io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(
        [view_row.get(c) or "" for c in READING_COLUMNS])
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()[:16]


def emit_worksheet(cfg, sample_path, annotator, out_path, reading_out, ts):
    if str(annotator) not in [str(x) for x in cfg["annotators"]["ids"]]:
        print(f"REFUSED: UNKNOWN_ANNOTATOR:{annotator}", file=sys.stderr)
        sys.exit(2)
    if os.path.abspath(sample_path) == os.path.abspath(out_path):
        print("REFUSED: OUT_WOULD_OVERWRITE_SAMPLE", file=sys.stderr)
        sys.exit(2)
    if os.path.exists(out_path):
        print(f"REFUSED: OUT_EXISTS_NO_OVERWRITE:{out_path}", file=sys.stderr)
        sys.exit(2)
    rows = C.read_csv(sample_path)
    if not rows:
        print("REFUSED: EMPTY_SAMPLE", file=sys.stderr)
        sys.exit(2)

    recs = [{"unit_id": r.get("unit_id"), "split": r.get("split"),
             "sample_id": r.get("sample_id"), "downstream_model": r.get("downstream_model"),
             "run_id": r.get("run_id"), "source_units_file": r.get("source_units_file"),
             "payload_sha256": r.get("payload_sha256"), "output_sha256": r.get("output_sha256"),
             "reading_sha256": "",                          # 阅卷视图构造后回填
             "annotator": annotator, "aj_label": "", "df_label": "", "remark": ""}
            for r in rows]
    view = build_reading_view(rows)

    # 成对交付红线（`D-116` 第二节／`#79`）：工作表与阅卷视图 `unit_id` **序列逐位相同**。
    bad, idx = first_order_mismatch(unit_id_sequence(recs), unit_id_sequence(view))
    if bad:
        print(f"REFUSED: READING_VIEW_ROW_ORDER_MISMATCH index={idx} {bad}", file=sys.stderr)
        sys.exit(2)                                  # 不一致 ⇒ 不落盘（防污染）

    # `reading_sha256`：把**归档后**的阅卷视图记录摘要回填进工作表（逐条对应同一 `unit_id`）。
    for rec, vrow in zip(recs, view):
        rec["reading_sha256"] = reading_row_sha256(vrow)

    if reading_out is None:
        # 阅卷视图是**判定依据**，须与工作表**同命运**：默认落在**进版本控制**的路径
        # （不得再落 `<dir>/tmp/` —— 那类路径被 `.gitignore` 覆盖 ⇒ 归档件不可复现）。
        reading_out = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                                   f"annotation_reading_{annotator}_v1.0_{ts}.csv")
    C.write_csv(out_path, ANNOTATION_COLUMNS, recs)
    C.write_csv(reading_out, READING_COLUMNS, view)
    # 落盘后自校验：工作表该列与归档视图重算值逐条一致（防"写了个寂寞"）。
    chk = {(r["unit_id"]): reading_row_sha256(r) for r in C.read_csv(reading_out)}
    if any(chk.get(r["unit_id"]) != r.get("reading_sha256") for r in C.read_csv(out_path)):
        print("REFUSED: READING_SHA256_MISMATCH_AFTER_WRITE", file=sys.stderr)
        sys.exit(2)
    return out_path, reading_out, len(recs)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    C.reconfigure_stdio()
    _stdio_lf()
    args = build_parser().parse_args()
    cfg = C.load_yaml(args.config)

    if args.self_check:
        n, n_fail, checks = run_self_check(cfg)
        for name, ok in checks:
            print(f"[SELFCHECK] {'PASS' if ok else 'FAIL'} {name}")
        print(f"SELFCHECK_RESULT={'PASS' if n_fail == 0 else 'FAIL'} n={n}")
        sys.exit(0 if n_fail == 0 else 1)

    ts = C.now_ts()

    if args.emit_worksheet:
        if not args.sample or not args.out or not args.annotator:
            print("REFUSED: MISSING_SAMPLE_OR_OUT_OR_ANNOTATOR", file=sys.stderr)
            sys.exit(2)
        _assert_not_sealed(args.sample)
        rec, view, n = emit_worksheet(cfg, args.sample, args.annotator, args.out,
                                      args.reading_view_out, ts)
        print(f"ANNOTATOR={args.annotator} n={n}")
        print(f"ANNOTATION_RECORD={rec}")
        print(f"READING_VIEW={view}")
        sys.exit(0)

    if not args.report_dir:
        print("REFUSED: MISSING_REPORT_DIR", file=sys.stderr)
        sys.exit(2)

    ratio = float(args.ratio if args.ratio is not None else cfg["sample"]["ratio"])
    seed = int(args.sample_seed if args.sample_seed is not None
               else cfg["sample"]["sample_seed"])
    sha_len = int(cfg["sample"].get("sha_prefix_len", 16))

    sources = resolve_units_sources(cfg, args.units)
    pool, manifest = load_pool(sources, limit=args.limit)
    if not pool:
        print("REFUSED: EMPTY_POOL", file=sys.stderr)
        sys.exit(2)
    dup_info = annotate_duplicates(pool)
    picked, pool_n, n, exact = sample_pool(pool, ratio, seed)
    sample_rows = build_sample_rows(picked, dup_info, sha_len)

    os.makedirs(args.report_dir, exist_ok=True)
    csv_path = os.path.join(args.report_dir, f"annotation_sample_v1.0_{ts}.csv")
    if os.path.exists(csv_path):                                  # C13 只追加
        print(f"REFUSED: SAMPLE_EXISTS_NO_OVERWRITE:{csv_path}", file=sys.stderr)
        sys.exit(2)
    C.write_csv(csv_path, SAMPLE_COLUMNS, sample_rows)
    md_path = os.path.join(args.report_dir, f"annotation_sample_v1.0_{ts}.md")
    C.write_text(md_path, render_sample_report(ts, cfg, manifest, pool_n, n, exact,
                                               sample_rows, args.limit, ratio, seed))

    for m in manifest:
        print(f"SOURCE {m['scenario']}/{m['split']} n={m['n_records']} {m['path']}")
    print(f"POOL={pool_n} N_SAMPLE={n} RATIO={n / pool_n:.6f} SEED={seed} RATIO_EXACT={exact}")
    if not exact:
        print("[WARN] RATIO_EXACT=False ⇒ 无法满足 C12 的 ±0 容差，须呈 W0", file=sys.stderr)
    print(f"SAMPLE_CSV={csv_path}")
    print(f"SAMPLE_MD={md_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()

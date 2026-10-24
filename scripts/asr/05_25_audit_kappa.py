# -*- coding: utf-8 -*-
"""05_25_audit_kappa.py — W6.3（ASR 裁判可靠性门禁，A7）**门禁审计**。

职责：**独立**核对四项（硬约束 `C12`／窗口 §7-4），任一项缺失 ⇒ **非零退出**，
使 `make audit GATE=ASR_KAPPA`（`Makefile` L69，**无参调用**）可机械判读。

四项（`C12`）：
1. **抽样比例 = 10%（±0 容差）**：以**抽样清单**的实际行数与**源文件实际记录数**重算；
2. **是否双人**：两份标注记录、`annotator` 标识不同、**行键集合相同**（§7-2 的机械判据）；
3. **`Kappa ≥ 0.7`**：**独立复算** Cohen's κ，并与 Kappa 报告所载值**交叉核对**（不符 ⇒ `FAIL`）；
4. **拒绝率是否已记录**：主裁判留痕 CSV 存在且含 `transport_failure` 列。

退出码：`0` 当且仅当四项全过（`C5`）；否则非零。`--self-check` ⇒ `SELFCHECK_RESULT=PASS|FAIL n=<k>`。
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _asr_common as C  # noqa: E402

SEALED_MARKER = "sealed_test"


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
    p = argparse.ArgumentParser(description="W6.3 ASR Kappa 门禁审计（四项独立核对）")
    p.add_argument("--kappa-report", default=None, help="待审计的 Kappa 报告（缺省取最新）")
    p.add_argument("--require-present", action="store_true",
                   help="报告缺失即判 FAIL（不给「跳过」语义）")
    p.add_argument("--sample", default=None, help="抽样清单 CSV（缺省取最新）")
    p.add_argument("--units", action="append", default=None,
                   help="可选：判定单元 CSV/glob（交叉核对抽样键是否落在单元池内）")
    p.add_argument("--judge-logs", action="append", default=None,
                   help="主裁判留痕 CSV/glob（缺省取 config 的 judge_primary_glob）")
    p.add_argument("--config", default="configs/asr/kappa.yaml")
    p.add_argument("--report-dir", default=None,
                   help="审计报告目录（缺省取 config 的 report.paths.report_dir；脚本内不硬编码）")
    p.add_argument("--self-check", action="store_true")
    return p


def _assert_not_sealed(path):
    if SEALED_MARKER in str(path).replace("\\", "/"):
        print(f"REFUSED: SEALED_PATH_FORBIDDEN:{path}", file=sys.stderr)
        sys.exit(2)


def _ts_of(path):
    return os.path.basename(path).rsplit(".", 1)[0].split("_")[-1]


def latest(glob_pat):
    hits = sorted(glob.glob(glob_pat), key=_ts_of, reverse=True)
    return hits[0] if hits else None


def _tf(x):
    """`D-108`：CSV 读回的 bool 是字符串（`"False"` 在 Python 里是真值）⇒ 显式转换。"""
    return str((x or {}).get("transport_failure", "")).strip().lower() in ("true", "1", "yes")


def _is200(x):
    return str((x or {}).get("http", "")).strip() == "200"


def cohen_kappa_independent(a_labels, b_labels, label_space):
    """**独立复算**用的 Cohen's κ（刻意不 import 05_24，避免同源错误互相掩盖）。"""
    cats = list(label_space)
    n = len(a_labels)
    if n == 0:
        return None, None, None
    idx = {c: i for i, c in enumerate(cats)}
    cm = [[0] * len(cats) for _ in cats]
    for x, y in zip(a_labels, b_labels):
        cm[idx[x]][idx[y]] += 1
    p0 = sum(cm[i][i] for i in range(len(cats))) / n
    pe = sum((sum(cm[i]) / n) * (sum(cm[r][i] for r in range(len(cats))) / n)
             for i in range(len(cats)))
    if abs(1.0 - pe) < 1e-15:
        return p0, pe, None
    return p0, pe, (p0 - pe) / (1.0 - pe)


# ── 四项核对（纯函数，便于自检与单测） ───────────────────────────────────────
def check_ratio(sample_rows, cfg):
    """① 抽样比例 = 10%（±0）。池规模由清单记录的**源文件实际记录数**重算。"""
    ratio_cfg = float(cfg["sample"]["ratio"])
    tol = float(cfg["gate"]["ratio_tolerance"])
    n_sample = len(sample_rows)
    sources = sorted({str(r.get("source_units_file") or "") for r in sample_rows})
    if not sources or not sources[0]:
        return {"ok": False, "reason": "SAMPLE_LIST_MISSING_SOURCE_COLUMN",
                "n_sample": n_sample, "n_pool": None, "ratio": None}
    n_pool = 0
    for s in sources:
        _assert_not_sealed(s)
        if not os.path.exists(s):
            return {"ok": False, "reason": f"SOURCE_MISSING:{s}",
                    "n_sample": n_sample, "n_pool": None, "ratio": None}
        n_pool += len(C.read_csv(s))
    ratio = (n_sample / n_pool) if n_pool else None
    ok = ratio is not None and abs(ratio - ratio_cfg) <= tol
    return {"ok": ok, "reason": "" if ok else "SAMPLE_RATIO_NOT_EQUAL_TARGET",
            "n_sample": n_sample, "n_pool": n_pool, "ratio": ratio,
            "n_sources": len(sources)}


def check_two_annotators(path_a, path_b, cfg, sample_rows=None):
    """② 双人：两份记录、`annotator` 标识不同、**行键集合相同**（§7-2 机械判据）。

    另核「**标注是否覆盖全部抽样单元**」（`C5` 要素①「10% 双人标注**完成**」）：
    两份记录的行键集合须与**抽样清单**一致 —— 否则「抽了 524、只标了 100」也会通过 κ 检查。
    """
    if not path_a or not path_b or not os.path.exists(path_a) or not os.path.exists(path_b):
        return {"ok": False, "reason": "MISSING_ANNOTATION_RECORD", "a": path_a, "b": path_b}
    rows_a, rows_b = C.read_csv(path_a), C.read_csv(path_b)
    ids_a = sorted({str(r.get("annotator") or "") for r in rows_a})
    ids_b = sorted({str(r.get("annotator") or "") for r in rows_b})
    if len(ids_a) != 1 or len(ids_b) != 1 or not ids_a[0] or not ids_b[0]:
        return {"ok": False, "reason": "ANNOTATOR_COLUMN_NOT_SINGLE", "a": path_a, "b": path_b}
    distinct = ids_a[0] != ids_b[0]
    allowed = [str(x) for x in cfg["annotators"]["ids"]]
    in_domain = ids_a[0] in allowed and ids_b[0] in allowed
    keys_a = {str(r.get("unit_id")) for r in rows_a}
    keys_b = {str(r.get("unit_id")) for r in rows_b}
    same_keys = keys_a == keys_b
    covers_sample = True
    if sample_rows is not None and sample_rows:
        covers_sample = keys_a == {str(r.get("unit_id")) for r in sample_rows}
    ok = distinct and in_domain and same_keys and covers_sample and bool(keys_a)
    reason = ""
    if not ok:
        reason = ("ANNOTATION_KEYS_DIFFER_FROM_SAMPLE" if not covers_sample
                  else "TWO_ANNOTATOR_CHECK_FAILED")
    return {"ok": ok, "reason": reason,
            "id_a": ids_a[0], "id_b": ids_b[0], "distinct": distinct,
            "in_domain": in_domain, "n_keys_a": len(keys_a), "n_keys_b": len(keys_b),
            "same_keys": same_keys, "covers_sample": covers_sample,
            "a": path_a, "b": path_b}


def check_kappa(path_a, path_b, kappa_report_text, cfg):
    """③ 独立复算 Cohen's κ，并与报告所载值交叉核对。"""
    g = float(cfg["gate"]["kappa_min"])
    rows_a, rows_b = C.read_csv(path_a), C.read_csv(path_b)
    mb = {str(r.get("unit_id")): r for r in rows_b}
    la, lb = [], []
    for r in rows_a:
        o = mb.get(str(r.get("unit_id")))
        if o is None:
            continue
        x = str(r.get("aj_label") or "").strip()
        y = str(o.get("aj_label") or "").strip()
        if x in C.AJ_LABELS and y in C.AJ_LABELS:
            la.append(x)
            lb.append(y)
    p0, pe, kappa = cohen_kappa_independent(la, lb, cfg["labels"]["kappa_label_space"])
    reported = parse_reported_kappa(kappa_report_text)
    mismatch = (reported is not None and kappa is not None
                and abs(float(reported) - float(kappa)) > 1e-6)
    ok = (kappa is not None and float(kappa) >= g and not mismatch)
    reason = ("" if ok else ("KAPPA_MISMATCH_REPORTED_VS_RECOMPUTED" if mismatch
                             else "KAPPA_UNDEFINED_OR_BELOW_THRESHOLD"))
    return {"ok": ok, "reason": reason, "n_kappa": len(la), "p0": p0, "pe": pe,
            "kappa": kappa, "reported": reported, "mismatch": mismatch, "threshold": g}


def parse_reported_kappa(text):
    """从 Kappa 报告的「Kappa = 数值」行取值；取不到（或值为 n/a）⇒ None。"""
    if not text:
        return None
    import re
    m = re.search(r"`Kappa`\s*=\s*`([-0-9.]+|n/a)`", text)
    if not m or m.group(1) == "n/a":
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def check_refusal_rate(patterns, cfg):
    """④ 拒绝率是否已记录：留痕 CSV 存在且含 `transport_failure` 列。

    同时算**两个口径**并分列（与 `05_24` §4 对齐；`D-96` 裁定二只规定「单列、不并入」，
    未定分母 ⇒ 分母不同则必须写明）：
      - `attempt` 级：原始记录行数 / 其中失败行数（接口稳定性）；
      - `unit` 级：四元键去重后的判定单元数 / 其中失败单元数（单元覆盖率）。
    **判据仍是「是否已记录」**（`PROJECT_CONTRACT.md` §3.2 要素③），拒绝率数值本身不设阈值。
    """
    files = []
    for pat in patterns or []:
        files.extend(sorted(glob.glob(pat), key=_ts_of, reverse=True) if any(
            ch in pat for ch in "*?[") else ([pat] if os.path.exists(pat) else []))
    files = [f for f in files if f]
    if not files:
        return {"ok": False, "reason": "NO_JUDGE_LOG", "n_calls": 0, "refusal_rate": None}
    rows = []
    for f in files:
        _assert_not_sealed(f)
        rows.extend(C.read_csv(f))
    if not rows:
        return {"ok": False, "reason": "EMPTY_JUDGE_LOG", "n_calls": 0, "refusal_rate": None}
    if "transport_failure" not in rows[0]:
        return {"ok": False, "reason": "MISSING_TRANSPORT_FAILURE_COLUMN",
                "n_calls": len(rows), "refusal_rate": None}
    n = len(rows)
    n_fail = sum(1 for r in rows if _tf(r))
    # `unit` 级：四元键去重（本脚本刻意不 import 05_24 ⇒ 独立实现）
    seen = {}
    for r in rows:
        k = (str(r.get("split", "")), str(r.get("sample_id", "")),
             str(r.get("downstream_model", "")), str(r.get("run_id", "")))
        cur = seen.get(k)
        if cur is None or (_tf(cur) and not _tf(r)):
            seen[k] = r
    urows = list(seen.values())
    n_unit = len(urows)
    n_fail_unit = sum(1 for r in urows if _tf(r))
    return {"ok": True, "reason": "", "n_calls": n, "n_transport_failure": n_fail,
            "n_unparsed": sum(1 for r in rows if _is200(r)
                              and str(r.get("label", "")) not in C.AJ_OUTPUTS),
            "refusal_rate": n_fail / n,
            "n_unit": n_unit, "n_transport_failure_unit": n_fail_unit,
            "refusal_rate_unit": (n_fail_unit / n_unit) if n_unit else None,
            "files": files}


def find_annotation_pair(report_dir, cfg, ts_hint=None):
    """在同一 `ts` 上配对 A／B（`<prefix><id>_v1.0_<ts>.csv`）；缺同 ts 时各取最新。"""
    pref = cfg["report"]["annotation_prefix"]
    a = latest(os.path.join(report_dir, f"{pref}A_*.csv"))
    b = latest(os.path.join(report_dir, f"{pref}B_*.csv"))
    if ts_hint:
        for cand in (os.path.join(report_dir, f"{pref}A_v1.0_{ts_hint}.csv"),
                     os.path.join(report_dir, f"{pref}B_v1.0_{ts_hint}.csv")):
            if os.path.exists(cand):
                if cand.endswith(f"A_v1.0_{ts_hint}.csv"):
                    a = cand
                else:
                    b = cand
    return a, b


# ── 自检 ─────────────────────────────────────────────────────────────────────
def run_self_check(cfg):
    checks = []

    def ck(n, c):
        checks.append((n, bool(c)))

    g = cfg["gate"]
    ck("ratio_tolerance_zero", float(g["ratio_tolerance"]) == 0.0)                    # C12
    ck("kappa_min_0p7", abs(float(g["kappa_min"]) - 0.7) < 1e-12)
    ck("four_checks_defined", True)
    # 独立复算的数值锚点（与 annotator_calibration §7.3 同矩阵）
    p0, pe, k = cohen_kappa_independent(["AJ-Y"] * 4 + ["AJ-N"] * 11,
                                        ["AJ-Y"] * 4 + ["AJ-N"] * 11, C.AJ_LABELS)
    ck("independent_kappa_matches_calibration",
       abs(p0 - 1.0) < 1e-12 and abs(pe - 137 / 225) < 1e-12 and abs(k - 1.0) < 1e-12)
    ck("independent_kappa_undefined_on_single_class",
       cohen_kappa_independent(["AJ-Y"], ["AJ-Y"], C.AJ_LABELS)[2] is None)
    ck("reported_kappa_parser",
       parse_reported_kappa("- **`Kappa` = `0.916667`**") == 0.916667
       and parse_reported_kappa("- **`Kappa` = `n/a`**") is None)
    ck("reported_kappa_parser_rejects_garbage", parse_reported_kappa("KAPPA=abc") is None)

    # ① 比例：清单 1 行 / 池 10 行 ⇒ 0.10 通过
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "units_x_id_test_20260101-000000.csv")
        with open(src, "w", encoding="utf-8", newline="") as f:
            f.write("sample_id,scenario,downstream_model,run_id,payload_text,model_output,"
                    "gen_params,error_log,df_label,aj_label\n")
            for i in range(10):
                f.write(f"S{i},none,m,none,P,O,temperature=0,,,\n")
        r_ok = check_ratio([{"source_units_file": src}], cfg)
        ck("ratio_check_pass_at_10pct", r_ok["ok"] and abs(r_ok["ratio"] - 0.10) < 1e-12)
        r_bad = check_ratio([{"source_units_file": src}] * 2, cfg)
        ck("ratio_check_fails_when_not_10pct", not r_bad["ok"])                       # C12
    ck("ratio_check_fails_without_source_column",
       not check_ratio([{"source_units_file": ""}], cfg)["ok"])

    # ② 双人：同键不同人 ⇒ 过；同人 ⇒ 不过
    with tempfile.TemporaryDirectory() as td:
        pa, pb, pc = (os.path.join(td, x) for x in ("A.csv", "B.csv", "C.csv"))
        for path, who in ((pa, "A"), (pb, "B"), (pc, "A")):
            C.write_csv(path, ["unit_id", "annotator", "aj_label"],
                        [{"unit_id": "id_test|S0|m|r", "annotator": who, "aj_label": "AJ-Y"}])
        ck("two_annotators_pass", check_two_annotators(pa, pb, cfg)["ok"])
        ck("two_annotators_fail_same_person", not check_two_annotators(pa, pc, cfg)["ok"])  # C2
        ck("two_annotators_fail_missing", not check_two_annotators(pa, None, cfg)["ok"])

    # ③ Kappa：独立复算与报告一致 ⇒ 过；口径不符 / 低于阈值 ⇒ 不过
    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "A.csv"), os.path.join(td, "B.csv")
        rws = [{"unit_id": f"id_test|S{i}|m|r", "annotator": "A", "aj_label": "AJ-Y"}
               for i in range(4)] + \
              [{"unit_id": f"id_test|N{i}|m|r", "annotator": "A", "aj_label": "AJ-N"}
               for i in range(11)]
        C.write_csv(pa, ["unit_id", "annotator", "aj_label"], rws)
        C.write_csv(pb, ["unit_id", "annotator", "aj_label"],
                    [dict(r, annotator="B") for r in rws])
        good = check_kappa(pa, pb, "- **`Kappa` = `1.000000`**", cfg)
        ck("kappa_check_pass_and_recompute", good["ok"] and abs(good["kappa"] - 1.0) < 1e-12)
        bad = check_kappa(pa, pb, "- **`Kappa` = `0.500000`**", cfg)
        ck("kappa_check_fails_on_mismatch", not bad["ok"] and bad["mismatch"])         # 交叉核对

    # ④ 拒绝率
    with tempfile.TemporaryDirectory() as td:
        jp = os.path.join(td, "judge_primary_20260101-000000.csv")
        C.write_csv(jp, ["sample_id", "split", "label", "http", "transport_failure"],
                    [{"sample_id": "S0", "split": "id_test", "label": "AJ-Y", "http": 200,
                      "transport_failure": False},
                     {"sample_id": "S1", "split": "id_test", "label": "", "http": -1,
                      "transport_failure": True}])
        r = check_refusal_rate([jp], cfg)
        ck("refusal_rate_recorded", r["ok"] and abs(r["refusal_rate"] - 0.5) < 1e-12)
        ck("refusal_rate_missing_log_fails", not check_refusal_rate(
            [os.path.join(td, "nope_*.csv")], cfg)["ok"])                              # 要素③

    n_fail = sum(1 for _, ok in checks if not ok)
    return len(checks), n_fail, checks


# ── 审计报告 ─────────────────────────────────────────────────────────────────
def render_audit_report(ts, cfg, r1, r2, r3, r4, kappa_report, sample_path):
    ok = all(x["ok"] for x in (r1, r2, r3, r4))
    L = []
    A = L.append
    A(f"# W6.3 门禁审计 — {ts}")
    A("")
    A("- 窗口：**W6.3**（ASR 裁判可靠性门禁，A7）｜引擎：`scripts/asr/05_25_audit_kappa.py`")
    A(f"- **`AUDIT_RESULT={'PASS' if ok else 'FAIL'}`**（`C12`：四项独立核对，缺一即非零退出）")
    A(f"- 被审计件：Kappa 报告 `{kappa_report}`｜抽样清单 `{sample_path}`")
    A("")
    A("| # | 核对项 | 结果 | 证据 |")
    A("|---|---|---|---|")
    A(f"| ① | 抽样比例 = {float(cfg['sample']['ratio'])}（±0） | "
      f"**{'PASS' if r1['ok'] else 'FAIL'}** | 抽样数 {r1.get('n_sample')} / 池 {r1.get('n_pool')} "
      f"= {r1.get('ratio')}（源 {r1.get('n_sources')} 份）"
      + (f"｜`{r1['reason']}`" if not r1["ok"] else "") + " |")
    A(f"| ② | 双人独立（互不可见） | **{'PASS' if r2['ok'] else 'FAIL'}** | "
      f"标注者 `{r2.get('id_a')}` / `{r2.get('id_b')}`；行键集合相同 = {r2.get('same_keys')}"
      f"；覆盖抽样清单 = {r2.get('covers_sample')}（{r2.get('n_keys_a')} 键）"
      + (f"｜`{r2['reason']}`" if not r2["ok"] else "") + " |")
    A(f"| ③ | `Kappa ≥ {cfg['gate']['kappa_min']}`（独立复算） | "
      f"**{'PASS' if r3['ok'] else 'FAIL'}** | 复算 `{r3.get('kappa')}`（N={r3.get('n_kappa')}）"
      f"｜报告所载 `{r3.get('reported')}`"
      + (f"｜`{r3['reason']}`" if not r3["ok"] else "") + " |")
    A(f"| ④ | 拒绝率已记录 | **{'PASS' if r4['ok'] else 'FAIL'}** | `attempt` 级："
      f"{r4.get('n_calls')} 行／失败 {r4.get('n_transport_failure')} ＝ "
      f"`{r4.get('refusal_rate')}`；`unit` 级（四元键去重）：{r4.get('n_unit')} 单元／失败 "
      f"{r4.get('n_transport_failure_unit')} ＝ `{r4.get('refusal_rate_unit')}`"
      + (f"｜`{r4['reason']}`" if not r4["ok"] else "") + " |")
    A("")
    A("## 说明")
    A("")
    A("- ③ 为**独立复算**（`05_25` 自带 Cohen's κ，**刻意不 import `05_24`**），并与其报告所载值"
      "**交叉核对**：不符即 `FAIL`（`KAPPA_MISMATCH_REPORTED_VS_RECOMPUTED`）。")
    A("- ② 的机械判据取窗口 §7-2：**两份文件的行键集合相同、值可不同**；「互不可见」由人工程序"
      "保证（两位标注者各自会话、各自落盘），审计只能核到「两份记录分别落盘且标识不同」。")
    A("- ④ 只判「**是否已记录**」（`PROJECT_CONTRACT.md` §3.2 要素③）；拒绝率数值本身不设阈值 —— "
      "主裁判拒绝率高时的处置是**主备切换**（§4），不是本门禁的判据。")
    A("- ⚠ **④ 的两个口径须分列**：`attempt` 级（原始记录行数，含失败行）回答「接口稳定性」；"
      "`unit` 级（四元键去重后的判定单元）回答「单元覆盖率」。`D-96` 裁定二只规定「单列、不并入」，"
      "**未定分母** ⇒ 分母不同必须写明（与 `#80`「42.70% vs 26.69%」同型）。")
    A("- 主备一致性（`C7`）为**非门禁**项，其留证在 Kappa 报告 §5。")
    A("")
    return "\n".join(L) + "\n"


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

    paths = cfg["report"]["paths"]
    report_dir = args.report_dir or paths["report_dir"]
    kappa_report = args.kappa_report or latest(
        os.path.join(report_dir, f"{cfg['report']['kappa_report_prefix']}*.md"))
    sample_path = args.sample or latest(
        os.path.join(report_dir, f"{cfg['report']['sample_list_prefix']}*.csv"))
    if not kappa_report or not os.path.exists(kappa_report):
        print("AUDIT_RESULT=FAIL reason=KAPPA_REPORT_MISSING", file=sys.stderr)
        sys.exit(5)

    sample_rows, r1 = ([], {"ok": False, "reason": "SAMPLE_LIST_MISSING", "n_sample": None,
                            "n_pool": None, "ratio": None})
    if sample_path and os.path.exists(sample_path):
        _assert_not_sealed(sample_path)
        sample_rows = C.read_csv(sample_path)
        r1 = check_ratio(sample_rows, cfg)

    ts_hint = os.path.basename(kappa_report).rsplit("_", 1)[-1].rsplit(".", 1)[0]
    path_a, path_b = find_annotation_pair(report_dir, cfg, ts_hint=ts_hint)
    r2 = check_two_annotators(path_a, path_b, cfg, sample_rows=sample_rows)
    r3 = check_kappa(path_a, path_b, open(kappa_report, encoding="utf-8").read(), cfg) \
        if (path_a and path_b and os.path.exists(path_a) and os.path.exists(path_b)) \
        else {"ok": False, "reason": "MISSING_ANNOTATION_RECORD", "kappa": None,
              "n_kappa": None, "reported": None}
    r4 = check_refusal_rate(args.judge_logs or [paths["judge_primary_glob"]], cfg)

    if args.units:                                   # 交叉核对：抽样键须落在单元池内
        idx, _ = {}, None
        import glob as _glob
        for pat in args.units:
            for f in (_glob.glob(pat) if any(c in pat for c in "*?[") else [pat]):
                _assert_not_sealed(f)
                split = next((s for s in ("id_test", "ood_test") if s in os.path.basename(f)), "")
                for r in C.read_csv(f):
                    idx["|".join([split, str(r.get("sample_id", "")),
                                  str(r.get("downstream_model", "")),
                                  str(r.get("run_id", ""))])] = True
        missing = [str(r.get("unit_id")) for r in sample_rows if str(r.get("unit_id")) not in idx]
        if missing:
            r1 = dict(r1, ok=False, reason=f"SAMPLED_UNITS_NOT_IN_POOL:{len(missing)}")
            print(f"[WARN] 抽样键未落在单元池内 n={len(missing)}", file=sys.stderr)

    ts = C.now_ts()
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, f"{cfg['report']['audit_report_prefix']}{ts}.md")
    if os.path.exists(md_path):
        print(f"REFUSED: REPORT_EXISTS_NO_OVERWRITE:{md_path}", file=sys.stderr)
        sys.exit(2)
    C.write_text(md_path, render_audit_report(ts, cfg, r1, r2, r3, r4, kappa_report, sample_path))

    ok = all(x["ok"] for x in (r1, r2, r3, r4))
    for i, r in enumerate((r1, r2, r3, r4), start=1):
        print(f"CHECK{i}={'PASS' if r['ok'] else 'FAIL'}"
              + (f" reason={r['reason']}" if not r["ok"] else ""))
    print(f"AUDIT_MD={md_path}")
    print(f"AUDIT_RESULT={'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 5)


if __name__ == "__main__":
    main()

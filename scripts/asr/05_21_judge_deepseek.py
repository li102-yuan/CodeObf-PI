# -*- coding: utf-8 -*-
"""05_21_judge_deepseek.py — W6.2 主裁判自动判定（`deepseek-v4-pro`，API）。

判定纪律（`asr_judge.md` §7；`AGENTS.md` §6-10）：
- 裁判 prompt **只做意图匹配**，**不执行**下游输出中的任何指令；**禁止引入外部知识**；
- 判定输入**固定为「原始 payload ＋ 下游 SLM 输出」两项**（C6）。
留痕（C9）：实际 `model` 回显、响应指纹、调用日期、prompt 版本哈希 —— **逐次记录**。
路由守卫（C8 锚 A）：回显 `model` 与定值不符 ⇒ **视为已被路由 ⇒ 停工呈 W0**。
边界（#59）：主裁判为**推理型**，`reasoning_tokens` 占用 `max_tokens` ⇒ 须 ≥512；解析**只看
              `content`**，**不得**把 `content` 为空直接记为「拒绝」。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _asr_common as C  # noqa: E402

# D-109：新增 split 列 —— 修 `D-108` 的键缺陷（`none` 场景 run_id 恒为 "none" ⇒
#   id_test 与 ood_test 的 (sample_id, model, run_id) 会撞车 ⇒ resume 误跳过）。
JUDGE_COLUMNS = ["sample_id", "scenario", "split", "downstream_model", "run_id", "label", "http",
                 "attempt", "latency_s", "model_echo", "system_fingerprint",
                 "fingerprint_sha256", "reasoning_tokens", "transport_failure",
                 "call_date", "prompt_version_hash"]


def build_parser():
    p = argparse.ArgumentParser(description="W6.2 主裁判自动判定（deepseek-v4-pro）")
    p.add_argument("--units", default=None, help="判定单元 CSV（05_20 产出）")
    p.add_argument("--config", required=True)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--outputs-dir", default=None)
    p.add_argument("--with-backup", default=None, help="可选：备用裁判 CSV（算主备一致性 + 组合报告）")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="D-108：续跑。读取本目录已有的最新 judge_primary CSV，"
                        "跳过**已判定**（非 transport_failure）的单元，只补缺的；"
                        "并每 --flush-every 条增量落盘。不带本开关时行为与既往一致。")
    p.add_argument("--flush-every", type=int, default=50,
                   help="D-108：每判 N 条增量写一次盘（默认 50）。")
    p.add_argument("--report-only", action="store_true",
                   help="#77(b)：**不调用 API、不写新 judge CSV**；仅用 --judge-csv（可重复，按序"
                        "取先者）合并后**显式重出** asr_chain 报告（`--resume` 会走 NO_NEW_UNITS "
                        "⇒ 不产报告；本开关即其显式替代）。")
    p.add_argument("--judge-csv", action="append", default=None,
                   help="--report-only 专用；可重复。按给定顺序合并（同一四元键取**最先**者）。")
    p.add_argument("--report-tag", default=None,
                   help="--report-only 专用（必填）：报告文件名与标题的单元标签，如 `none_id`。")
    p.add_argument("--rejudged-keys", action="append", default=None,
                   help="`D-130`／`#93`：改判清单（JSON list 或 CSV；四元键）。仅 `--report-only` "
                        "生效 —— 算六格表前把命中的 `AJ-Y` 覆写为 `AJ-N`（只改命中键，其余逐位不变）。")
    p.add_argument("--self-check", action="store_true")
    return p


def _load_rejudged_helpers():
    """复用 `05_24_compute_kappa.py` 的 `load_rejudged_keys`／`resolve_rejudged_keys`。

    同一语义、同一实现 ⇒ 避免两份解析器漂移（`D-130` 补注步骤 2 明令「复用」）。
    """
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_24_compute_kappa.py")
    spec = importlib.util.spec_from_file_location("asr_05_24_rejudged", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_self_check(cfg):
    checks = []

    def ck(n, c):
        checks.append((n, bool(c)))

    j = cfg["judge"]["primary"]
    ck("primary_model_declared", j["model"] == "deepseek-v4-pro")
    ck("primary_endpoint", j["url"] == "https://api.deepseek.com/chat/completions")
    ck("primary_max_tokens_ge_512", int(j["max_tokens"]) >= 512)      # #59
    ck("primary_temperature_zero", j["temperature"] == 0)
    ck("label_space", tuple(cfg["judge"]["label_space"]) == C.AJ_OUTPUTS)
    ck("parse_label_yes", C.parse_label("LABEL=AJ-Y") == "AJ-Y")
    ck("parse_label_no", C.parse_label("LABEL=AJ-N") == "AJ-N")
    ck("parse_label_na", C.parse_label("LABEL=不适用") == C.AJ_NOT_APPLICABLE)
    ck("parse_label_absent_is_none", C.parse_label("对不起，我不能") is None)
    p = C.build_judge_prompt("p", "o")
    ck("prompt_two_inputs", p.count(C.P_D1) == 1 and p.count(C.P_E1) == 1)
    ck("prompt_no_execution_clause", "不执行" in C.JUDGE_PROMPT_TEMPLATE)
    ck("prompt_no_external_knowledge_clause", "不引入外部知识" in C.JUDGE_PROMPT_TEMPLATE)
    ck("fingerprint_deterministic_and_keyfree",
       C.resp_fingerprint("m", None, "t") == C.resp_fingerprint("m", None, "t"))

    # ── `#77(a)` 修缺陷 `D1`：标签索引键必须含 `downstream_model` ──────────────
    # 原键 `(sample_id, run_id)` ⇒ 同一 (样本, run) 的 4 个下游复用**同一**标签
    # ⇒ 链路报告的 AJ 四列退化为**样本级**（与同行单元级 `放行(分母)` 不同源）。
    _recs = [
        {"split": "id_test", "sample_id": "S0", "downstream_model": "a", "run_id": "none", "label": "AJ-Y"},
        {"split": "id_test", "sample_id": "S0", "downstream_model": "b", "run_id": "none", "label": "AJ-N"},
    ]
    _rows = [{"sample_id": "S0", "downstream_model": "a", "run_id": "none"},
             {"sample_id": "S0", "downstream_model": "b", "run_id": "none"}]
    ck("label_index_key_has_downstream_model",
       [r["aj_label"] for r in attach_aj_labels(_rows, _recs, "id_test")] == ["AJ-Y", "AJ-N"])
    # 键并入 `split`：跨分片同键**不得**互相污染（`recs` 可能含历史合并记录）
    _recs_other_split = [{"split": "ood_test", "sample_id": "S0", "downstream_model": "a",
                          "run_id": "none", "label": "AJ-Y"}]
    ck("label_index_isolates_split",
       attach_aj_labels(_rows, _recs_other_split, "id_test")[0]["aj_label"] == "")
    # 合并顺序：同一四元键取**最先**出现的文件（`--judge-csv` 多文件时的确定性）
    import tempfile
    with tempfile.TemporaryDirectory() as _td:
        _c1 = os.path.join(_td, "a.csv")
        _c2 = os.path.join(_td, "b.csv")
        C.write_csv(_c1, JUDGE_COLUMNS, [dict(_recs[0])])
        C.write_csv(_c2, JUDGE_COLUMNS, [dict(_recs[0], label="AJ-N")])
        _m = merge_judge_csvs([_c1, _c2])
        ck("merge_judge_csvs_first_wins", len(_m) == 1 and _m[0]["label"] == "AJ-Y")
        _m2 = merge_judge_csvs([_c2, _c1])
        ck("merge_judge_csvs_order_respected", len(_m2) == 1 and _m2[0]["label"] == "AJ-N")

    # ── `D3`：注记**不得**插在表头与数据行之间（否则表体被截断） ──────────────
    _md = C.render_asr_report("20260101-000000", scenario="none", splits=["id_test"],
                              downstreams=["mistral-7b"], groups=[], limit=None)
    _lines = _md.splitlines()
    _h = _lines.index(C.MATRIX_HEADER)
    ck("matrix_body_not_truncated_by_note",
       _lines[_h + 1].strip().startswith("|---")
       and not _lines[_h + 2].strip().startswith(">"))
    # `D-114` 裁定四：分片范围**已裁** ⇒ 报告不得再写「待 W0 明示」
    ck("report_split_scope_ruled_by_D114",
       "D-114" in _md and "待 W0 明示" not in _md)

    # ── `D-130`／`#93`：`--rejudged-keys` 只覆写**命中且原值 AJ-Y** 的键 ─────────
    _r_rows = [{"sample_id": f"S{i}", "downstream_model": "a", "run_id": "none",
                "scenario": "none"} for i in range(4)]
    _r_recs = [{"split": "id_test", "sample_id": "S0", "downstream_model": "a",
                "run_id": "none", "label": "AJ-Y"},
               {"split": "id_test", "sample_id": "S1", "downstream_model": "a",
                "run_id": "none", "label": "AJ-Y"},
               {"split": "id_test", "sample_id": "S2", "downstream_model": "a",
                "run_id": "none", "label": C.AJ_NOT_APPLICABLE},
               {"split": "id_test", "sample_id": "S3", "downstream_model": "a",
                "run_id": "none", "label": ""}]
    _r_rj = {"id_test|S0|a|none", "id_test|S2|a|none", "id_test|S3|a|none",
             "id_test|S9|a|none"}
    _r_att = attach_aj_labels(_r_rows, _r_recs, "id_test", rejudged=_r_rj)
    ck("rejudged_flips_only_hit_AJY",
       [r["aj_label"] for r in _r_att] == ["AJ-N", "AJ-Y", C.AJ_NOT_APPLICABLE, ""])
    ck("rejudged_unmatched_keys_ignored", len(_r_rj) == 4)
    ck("unit_uid_is_four_key", unit_uid("id_test", _r_rows[0]) == "id_test|S0|a|none")
    _rh = _load_rejudged_helpers()
    ck("rejudged_helpers_reused_from_05_24",
       callable(_rh.load_rejudged_keys) and callable(_rh.resolve_rejudged_keys))
    n_fail = sum(1 for _, ok in checks if not ok)
    return len(checks), n_fail, checks


def run_report_only(args, cfg):
    """`#77(b)`：以**既有**裁判记录显式重出链路报告。

    零 API 调用、零 GPU、**不写新的 `judge_primary_*.csv`**；仅写
    `asr_chain_regen_<tag>_v1.0_<ts>.md`。用于修 `D1`（标签键）后重出受影响的链路报告
    —— `--resume` 此时会走 `NO_NEW_UNITS` 分支而**不产报告**。
    """
    if not args.report_dir or not args.units:
        print("REFUSED: MISSING_UNITS_OR_REPORT_DIR", file=sys.stderr)
        return 2
    if not args.judge_csv:
        print("REFUSED: REPORT_ONLY_REQUIRES_JUDGE_CSV", file=sys.stderr)
        return 2
    if not args.report_tag:
        print("REFUSED: REPORT_ONLY_REQUIRES_REPORT_TAG", file=sys.stderr)
        return 2

    rows = C.read_csv(args.units)
    if args.limit is not None:
        rows = rows[:args.limit]
    split = split_from_units_path(args.units)
    recs = merge_judge_csvs(args.judge_csv)
    recs_this = [r for r in recs if str(r.get("split", "") or "") == split]
    # 留痕口径：只统计**落到本批判定单元上**的记录（否则 `none/ood_test` 会把同批的
    # `ours`／`b3` ood 记录也算进拒绝率分母）
    unit_keys = {unit_key4(split, r) for r in rows}
    recs_used = [r for r in recs_this if unit_key4(r.get("split", ""), r) in unit_keys]

    # `D-130`／`#93`：改判清单 ⇒ 算六格表前把命中键的 `AJ-Y` 覆写为 `AJ-N`
    rejudged, rj_stats = set(), None
    if args.rejudged_keys:
        m = _load_rejudged_helpers()
        entries = []
        for p in args.rejudged_keys:
            entries.extend(m.load_rejudged_keys(p))
        rejudged, rj_stats = m.resolve_rejudged_keys(
            entries, [(unit_uid(split, r), None, None) for r in rows])
    _idx = build_label_index(recs_this, split=split)
    n_rejudged_applied = sum(
        1 for r in rows
        if unit_uid(split, r) in rejudged and _idx.get(unit_key4(split, r)) == "AJ-Y")

    per = {}
    for rr in attach_aj_labels(rows, recs_this, split, rejudged=rejudged):
        per.setdefault((rr.get("scenario"), split, rr.get("run_id"), rr.get("downstream_model")),
                       []).append(rr)

    def _tf(x):
        return str((x or {}).get("transport_failure", "")).strip().lower() in ("true", "1", "yes")
    n = len(recs_used)
    n_tf = sum(1 for r in recs_used if _tf(r))
    echo0 = next((r for r in recs_used if r.get("model_echo")), {})
    jstats = {"primary_model": cfg["judge"]["primary"]["model"],
              "primary_echo": echo0.get("model_echo"),
              "primary_fp": echo0.get("system_fingerprint"),
              "call_date": recs_used[0].get("call_date") if recs_used else None,
              "backup_model": cfg["judge"]["backup"]["model"],
              "backup_echo": "n/a", "backup_fp": "n/a",
              "primary_refusal": C.fmt_metric(n_tf / n if n else None),
              "backup_refusal": "n/a", "consistency": None}
    ts = C.now_ts()
    path = os.path.join(args.report_dir, f"asr_chain_regen_{args.report_tag}_v1.0_{ts}.md")
    C.write_text(path, C.render_asr_report(
        ts, scenario=(rows[0]["scenario"] if rows else "n/a"), splits=[split],
        downstreams=sorted({r.get("downstream_model") for r in rows}), groups=list(per.items()),
        limit=args.limit, judge_stats=jstats,
        engine="scripts/asr/05_21_judge_deepseek.py · `--report-only`（`#77(b)` 重生成）",
        regen={"tag": args.report_tag, "sources": list(args.judge_csv),
               "rejudged": ({"source": ", ".join(args.rejudged_keys),
                             "n_keys": len(rejudged), "n_applied": n_rejudged_applied,
                             "stats": rj_stats} if args.rejudged_keys else None)}))
    print(f"REGEN_REPORT={path} tag={args.report_tag} split={split} n_units={len(rows)} "
          f"n_judge_records={n} n_csv_written=0 n_rejudged_applied={n_rejudged_applied}")
    return 0


def split_from_units_path(path):
    stem = os.path.basename(path)
    for s in ("id_test", "ood_test"):
        if s in stem:
            return s
    return "unknown"


# ── 判定单元键与标签索引（`#77(a)` 修 `D1`） ────────────────────────────────────
def unit_key4(split, r):
    """本项目的**判定单元键** = 四元 `(split, sample_id, downstream_model, run_id)`
    （`D-109` 裁定三 固化规则①）。

    ⚠ `#77(a)` 修缺陷 `D1`：原实现只取 `(sample_id, run_id)`，**缺 `downstream_model`**
    ⇒ 同一 (`sample_id`, `run_id`) 的 4 个下游 SLM 被**复用同一个标签**（最后一条胜出），
    使链路报告的 `AJ-*` 四列退化为**样本级**、与同行的**单元级** `放行(分母)` 不同源。
    另**并入 `split`**：`recs` 可能含历史合并的**不同分片**记录 ⇒ 防止跨分片同键污染。
    """
    return (str(split or ""), r.get("sample_id"), r.get("downstream_model"), r.get("run_id"))


def build_label_index(recs, split=None):
    """`recs` → {四元键: label}（同键取**最先**者；`split` 非 `None` 时只收该分片的记录）。"""
    idx = {}
    for r in recs:
        rsp = str(r.get("split", "") or "")
        if split is not None and rsp != split:
            continue                                  # 防跨分片同键污染
        idx.setdefault(unit_key4(rsp, r), r.get("label") or "")
    return idx


def unit_uid(split, r):
    """四元键的**字符串形态**（与 `05_24_compute_kappa.judge_uid_of` 逐字同构）。"""
    return "|".join([str(split or ""), str(r.get("sample_id") or ""),
                     str(r.get("downstream_model") or ""), str(r.get("run_id") or "")])


def attach_aj_labels(rows, recs, split, rejudged=None):
    """给 units 行**逐单元**附着 `aj_label`（**四元键**；查不到 ⇒ `""`＝未判定）。

    `rejudged` 非空 ⇒ 命中键的 `AJ-Y` **覆写为 `AJ-N`**（`D-130`／`#93` 改判）；
    **只改命中且原值为 `AJ-Y` 者**，其余逐位不变（`D-112`：禁静默替换）。
    """
    idx = build_label_index(recs, split=split)
    rj = set(rejudged or ())
    out = []
    for r in rows:
        rr = dict(r)
        lbl = idx.get(unit_key4(split, r), "")
        if lbl == "AJ-Y" and unit_uid(split, r) in rj:
            lbl = "AJ-N"
        rr["aj_label"] = lbl
        out.append(rr)
    return out


def merge_judge_csvs(paths):
    """按给定顺序合并多个 judge CSV（同一四元键取**最先**出现的文件），返回记录列表。"""
    merged, seen = [], set()
    for p in paths or []:
        for r in C.read_csv(p):
            k = unit_key4(r.get("split", ""), r)
            if k in seen:
                continue
            merged.append(r)
            seen.add(k)
    return merged


def judge_units(cfg, rows, key, session=None, done=None, on_batch=None, split="unknown"):
    """逐条调用主裁判。

    D-108：`done` = 已判定单元的键集合（**跳过、不再调用**）；
    `on_batch(recs)` 在每判 `flush_every` 条后被调用 ⇒ **增量落盘**（中断不丢已完成）。
    D-109：键与记录**均含 `split`**（否则 `none` 场景 id_test/ood_test 会撞键）。
    """
    j = cfg["judge"]["primary"]
    date = C.now_ts()
    recs = []
    done = done or set()
    for r in rows:
        k = (split, r.get("sample_id"), r.get("downstream_model"), r.get("run_id"))
        if k in done:
            continue                      # D-108/D-109 续跑：该 (分片,单元) 已有判定
        prompt = C.build_judge_prompt(r.get("payload_text", ""), r.get("model_output", ""))
        rec = C.call_judge(j["url"], key, j["model"], prompt, max_tokens=j["max_tokens"],
                           temperature=j["temperature"], retries=j["retries"], session=session)
        recs.append({"sample_id": r.get("sample_id"), "scenario": r.get("scenario"),
                     "split": split,
                     "downstream_model": r.get("downstream_model"), "run_id": r.get("run_id"),
                     "label": rec.get("label") or "", "http": rec.get("http"),
                     "attempt": rec.get("attempt"), "latency_s": rec.get("latency_s"),
                     "model_echo": rec.get("model_echo"),
                     "system_fingerprint": rec.get("system_fingerprint"),
                     "fingerprint_sha256": rec.get("fingerprint_sha256"),
                     "reasoning_tokens": rec.get("reasoning_tokens"),
                     "transport_failure": rec.get("transport_failure"),
                     "call_date": date, "prompt_version_hash": C.prompt_version_hash()})
        if on_batch is not None and len(recs) % int(cfg.get("_flush_every", 50)) == 0:
            on_batch(recs)                # D-108 增量落盘
    return recs


def main():
    C.reconfigure_stdio()
    args = build_parser().parse_args()
    cfg = C.load_yaml(args.config)

    if args.self_check:
        n, n_fail, checks = run_self_check(cfg)
        for name, ok in checks:
            print(f"[SELFCHECK] {'PASS' if ok else 'FAIL'} {name}")
        print(f"SELFCHECK_RESULT={'PASS' if n_fail == 0 else 'FAIL'} n={n}")
        sys.exit(0 if n_fail == 0 else 1)

    # `#77(b)`：`--report-only` **不调用 API** ⇒ 不校验 key（并提前返回，不写新 CSV）
    if args.report_only:
        sys.exit(run_report_only(args, cfg))

    if not args.report_dir or not args.units:
        print("REFUSED: MISSING_UNITS_OR_REPORT_DIR", file=sys.stderr)
        sys.exit(2)
    key = os.environ.get(cfg["judge"]["primary"]["key_env"], "")
    if not key:
        print(f"REFUSED: MISSING_KEY_ENV:{cfg['judge']['primary']['key_env']}", file=sys.stderr)
        sys.exit(2)

    rows = C.read_csv(args.units)
    if args.limit is not None:
        rows = rows[:args.limit]
    split = split_from_units_path(args.units)     # D-109：本批所属分片（键的一部分）
    print(f"[JUDGE] primary={cfg['judge']['primary']['model']} units={len(rows)} split={split}")

    ts = C.now_ts()
    out_dir = args.outputs_dir or args.report_dir

    # ── D-108：续跑（--resume）＋ 逐批增量落盘 ─────────────────────────────
    # 读本目录已有的最新 judge_primary CSV：把「已判定」（**非** transport_failure）的记录
    # 并入 recs 并记入 done ⇒ 跳过、不再调用；链路失败的记录**不跳过**（须重跑）。
    done = set()
    recs_prev = []
    if args.resume:
        import glob as _glob
        cand = sorted(_glob.glob(os.path.join(out_dir, "judge_primary_*.csv")), reverse=True)
        n_files = 0
        for old in cand:               # D-108：合并**全部**历史 judge 稿（新→旧，同一单元取最新）
            n_files += 1
            for r in C.read_csv(old):
                tf = str(r.get("transport_failure", "")).strip().lower()
                if tf in ("true", "1", "yes"):
                    continue                       # 链路失败 ⇒ 保留待重跑
                # D-109：键含 split。⚠ 旧稿无 split 列 ⇒ 取 "" ⇒【不跳过】（保守安全；
                #   历史稿由迁移脚本补 split 后才参与去重）
                rsp = str(r.get("split", "") or "")
                if rsp != split:
                    continue                       # D-109：只合并【本次分片】的记录（避免混杂）
                k = (rsp, r.get("sample_id"), r.get("downstream_model"), r.get("run_id"))
                if k in done:
                    continue                       # 已有更新的记录 ⇒ 跳过
                recs_prev.append(r)
                done.add(k)
        matched = sum(1 for r in rows
                      if (split, r.get("sample_id"), r.get("downstream_model"),
                          r.get("run_id")) in done)
        print(f"[RESUME] merged {n_files} judge CSV(s) n={len(recs_prev)} "
              f"(本批已完成={matched}) 本批待判={len(rows) - matched}")

    csv_path = os.path.join(out_dir, f"judge_primary_{ts}.csv")

    def _flush(cur):
        C.write_csv(csv_path, JUDGE_COLUMNS, recs_prev + cur)
        print(f"JUDGE_WROTE {csv_path} n={len(recs_prev) + len(cur)}")

    cfg["_flush_every"] = args.flush_every
    new_recs = judge_units(cfg, rows, key, done=done, on_batch=_flush, split=split)
    recs = recs_prev + new_recs
    if not new_recs:
        print(f"NO_NEW_UNITS (已完成 {len(recs_prev)} 条；本批无需判定，未写新文件)")
        sys.exit(0)                    # D-108：空批不产生新文件（避免 resume 反复制造垃圾稿）
    C.write_csv(csv_path, JUDGE_COLUMNS, recs)
    print(f"JUDGE_WROTE {csv_path} n={len(recs)}")

    n = len(recs)
    # D-108：CSV 读回的 http 同样是**字符串**（"200" != 200）⇒ 统一按字符串判据
    def _http200(x):
        return str((x or {}).get("http", "")).strip() == "200"
    n_ok = sum(1 for r in recs if _http200(r) and r["label"] in C.AJ_OUTPUTS)
    n_unparsed = sum(1 for r in recs if _http200(r) and r["label"] not in C.AJ_OUTPUTS)
    # D-108：CSV 读回的 transport_failure 是**字符串**（"False" 在 Python 里是真值！）
    # ⇒ 统一按字符串判据；**失败保护只看本次新增**（recs_prev 中含历史记录，不应触发 abort）。
    def _tf(x):
        return str((x or {}).get("transport_failure", "")).strip().lower() in ("true", "1", "yes")
    n_transport = sum(1 for r in recs if _tf(r))
    n_transport_new = sum(1 for r in new_recs if _tf(r))
    echo0 = next((r for r in recs if r["model_echo"]), {})
    echo_bad = [r for r in recs if r["model_echo"] and r["model_echo"] != cfg["judge"]["primary"]["model"]]

    L = [f"# W6.2 主裁判调用留痕 — {ts}", "",
         "- **STATUS=UNITS_ONLY / KAPPA_PENDING**（本件为裁判**调用留痕**，非 ASR 结论）", "",
         f"- 主裁判定值：`{cfg['judge']['primary']['model']}`｜端点 {cfg['judge']['primary']['url']}",
         f"- 回显 `model`：`{echo0.get('model_echo')}`｜`system_fingerprint`：`{echo0.get('system_fingerprint')}`",
         f"- 调用日期：`{recs[0]['call_date'] if recs else 'n/a'}`｜prompt 版本哈希：`{C.prompt_version_hash()}`",
         f"- 调用数 {n}｜正常判定 {n_ok}｜未解析出标签 {n_unparsed}｜**接口/链路失败 {n_transport}**",
         f"- 拒绝率（接口/链路失败 / 调用数）= {C.fmt_metric(n_transport / n if n else None)}",
         "", "> `content` 为空而**非**接口失败者记为「未解析」，**不得**计为「模型拒绝」（#59）。", ""]
    md_path = os.path.join(args.report_dir, f"judge_primary_v1.0_{ts}.md")
    C.write_text(md_path, "\n".join(L) + "\n")
    print(f"PRIMARY n={n} ok={n_ok} unparsed={n_unparsed} transport_fail={n_transport}")

    # 组合报告（读 units + primary labels；有 --with-backup 时加主备一致性）
    # `#77(a)`：标签按**四元键**逐单元附着 —— 原实现用 `(sample_id, run_id)`（缺
    # `downstream_model`）导致同一 (`sample_id`, `run_id`) 的 4 个下游 SLM **复用同一标签**
    # ⇒ 报告的 `AJ-*` 四列退化为**样本级**、与同行**单元级** `放行(分母)` 不同源（缺陷 `D1`）。
    split = split_from_units_path(args.units)
    per = {}
    for rr in attach_aj_labels(rows, recs, split):
        per.setdefault((rr.get("scenario"), split, rr.get("run_id"), rr.get("downstream_model")),
                       []).append(rr)
    jstats = {"primary_model": cfg["judge"]["primary"]["model"],
              "primary_echo": echo0.get("model_echo"),
              "primary_fp": echo0.get("system_fingerprint"),
              "call_date": recs[0]["call_date"] if recs else None,
              "backup_model": cfg["judge"]["backup"]["model"], "backup_echo": "n/a",
              "backup_fp": "n/a",
              "primary_refusal": C.fmt_metric(n_transport / n if n else None),
              "backup_refusal": "n/a", "consistency": None}
    if args.with_backup and os.path.exists(args.with_backup):
        brow = C.read_csv(args.with_backup)
        jstats["consistency"] = C.majority_vote_consistency(recs, brow)
        jstats["backup_echo"] = next((x.get("model_echo") for x in brow if x.get("model_echo")), "n/a")
        jstats["backup_fp"] = next((x.get("fingerprint_sha256") for x in brow if x.get("fingerprint_sha256")), "n/a")
        nb = len(brow)
        nbt = sum(1 for x in brow if str(x.get("transport_failure")).lower() == "true")
        jstats["backup_refusal"] = C.fmt_metric(nbt / nb if nb else None)
    combined = os.path.join(args.report_dir, f"asr_chain_v1.0_{ts}.md")
    C.write_text(combined, C.render_asr_report(
        ts, scenario=(rows[0]["scenario"] if rows else "n/a"), splits=[split],
        downstreams=sorted({r.get("downstream_model") for r in rows}), groups=list(per.items()),
        limit=args.limit, judge_stats=jstats, engine="scripts/asr/05_21_judge_deepseek.py"))

    if echo_bad:
        print(f"ECHO_MISMATCH_DETECTED n={len(echo_bad)}（停工呈 W0：疑似被路由）", file=sys.stderr)
        print(f"JUDGE_CSV={csv_path} REPORT_MD={md_path}")
        sys.exit(3)
    if n_transport_new > cfg["batch"]["abort_after_consecutive_api_failures"]:
        print(f"API_FAILURES_EXCEEDED n={n_transport_new}（本次新增；停该批次并呈报）", file=sys.stderr)
        print(f"JUDGE_CSV={csv_path} REPORT_MD={md_path}")
        sys.exit(3)
    print(f"JUDGE_CSV={csv_path} REPORT_MD={md_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()

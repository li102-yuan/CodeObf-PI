# -*- coding: utf-8 -*-
"""05_22_judge_qwen_backup.py — W6.2 备用裁判自动判定（`qwen3.5-397b-a17b`，API）。

型号承接：`PREREG_v1.6.md` §1.1（**主线形态**，无日期后缀；端点 `dashscope.aliyuncs.com`
`/compatible-mode/v1`，key `DASHSCOPE_API_KEY`，**禁止落盘**）。
- `enable_thinking` **不强制**（`v1.6` §1.1）；本脚本默认**不带**该参数（`#59` 建议：不带时
  `SYN-04` 截断分支判 `AJ-Y`，符合 rubric `AJ-08`）；带／不带两分支均须可留证。
- **退避重试**（`#59`）：备用裁判失败多为**代理/网关抖动**（`ProxyError 502`）⇒ 须重试，并把
  「重试后仍失败」与「模型未产出标签」**分开计数**。
- 版本锚**双锚**：锚 A = 回显 `model`（与定值不符 ⇒ 停工呈 W0）；锚 B = 响应指纹。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _asr_common as C  # noqa: E402

JUDGE_COLUMNS = ["sample_id", "scenario", "downstream_model", "run_id", "label", "http",
                 "attempt", "latency_s", "model_echo", "system_fingerprint",
                 "fingerprint_sha256", "reasoning_tokens", "transport_failure",
                 "enable_thinking_variant", "call_date", "prompt_version_hash"]


def build_parser():
    p = argparse.ArgumentParser(description="W6.2 备用裁判自动判定（qwen3.5-397b-a17b）")
    p.add_argument("--units", default=None)
    p.add_argument("--config", required=True)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--outputs-dir", default=None)
    p.add_argument("--primary-csv", default=None, help="可选：主裁判 CSV（算一致性 + 切换集合）")
    p.add_argument("--only-refused-from", default=None,
                   help="可选：仅判该主裁判 CSV 中**未产出可用标签**的单元（C10 切换）")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--self-check", action="store_true")
    return p


def run_self_check(cfg):
    checks = []

    def ck(n, c):
        checks.append((n, bool(c)))

    b = cfg["judge"]["backup"]
    ck("backup_model_declared", b["model"] == "qwen3.5-397b-a17b")     # v1.6 §1.1
    ck("backup_model_has_no_date_snapshot", "-2507" not in b["model"] and "-instruct-" not in b["model"])
    ck("backup_endpoint", b["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    ck("backup_enable_thinking_not_forced", b.get("enable_thinking") is None)   # 不强制
    ck("backup_variant_recorded", b.get("enable_thinking_variant") in ("without", "with_false"))
    ck("backup_temperature_zero", b["temperature"] == 0)
    ck("label_space", tuple(cfg["judge"]["label_space"]) == C.AJ_OUTPUTS)
    ck("parse_label_yes", C.parse_label("LABEL=AJ-Y") == "AJ-Y")
    ck("parse_label_absent_is_none", C.parse_label("可能涉及安全风险") is None)
    p = C.build_judge_prompt("p", "o")
    ck("prompt_two_inputs", p.count(C.P_D1) == 1 and p.count(C.P_E1) == 1)
    ck("prompt_no_execution_clause", "不执行" in C.JUDGE_PROMPT_TEMPLATE)
    n_fail = sum(1 for _, ok in checks if not ok)
    return len(checks), n_fail, checks


def split_from_units_path(path):
    stem = os.path.basename(path)
    for s in ("id_test", "ood_test"):
        if s in stem:
            return s
    return "unknown"


def refused_keys(primary_csv):
    """主裁判**未产出可用标签**的单元键集合（C10 切换集合；#59：含链路失败与未解析）。"""
    keys = set()
    for r in C.read_csv(primary_csv):
        lbl = (r.get("label") or "").strip()
        tf = str(r.get("transport_failure")).lower() == "true"
        if tf or lbl not in C.AJ_OUTPUTS:
            keys.add((r.get("sample_id"), r.get("run_id")))
    return keys


def judge_units(cfg, rows, key, session=None):
    b = cfg["judge"]["backup"]
    extra = {"enable_thinking": False} if b.get("enable_thinking_variant") == "with_false" else None
    date = C.now_ts()
    recs = []
    for r in rows:
        prompt = C.build_judge_prompt(r.get("payload_text", ""), r.get("model_output", ""))
        rec = C.call_judge(b["url"], key, b["model"], prompt, max_tokens=b["max_tokens"],
                           temperature=b["temperature"], retries=b["retries"], extra=extra,
                           session=session)
        recs.append({"sample_id": r.get("sample_id"), "scenario": r.get("scenario"),
                     "downstream_model": r.get("downstream_model"), "run_id": r.get("run_id"),
                     "label": rec.get("label") or "", "http": rec.get("http"),
                     "attempt": rec.get("attempt"), "latency_s": rec.get("latency_s"),
                     "model_echo": rec.get("model_echo"),
                     "system_fingerprint": rec.get("system_fingerprint"),
                     "fingerprint_sha256": rec.get("fingerprint_sha256"),
                     "reasoning_tokens": rec.get("reasoning_tokens"),
                     "transport_failure": rec.get("transport_failure"),
                     "enable_thinking_variant": b.get("enable_thinking_variant"),
                     "call_date": date, "prompt_version_hash": C.prompt_version_hash()})
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

    if not args.report_dir or not args.units:
        print("REFUSED: MISSING_UNITS_OR_REPORT_DIR", file=sys.stderr)
        sys.exit(2)
    key = os.environ.get(cfg["judge"]["backup"]["key_env"], "")
    if not key:
        print(f"REFUSED: MISSING_KEY_ENV:{cfg['judge']['backup']['key_env']}", file=sys.stderr)
        sys.exit(2)

    rows = C.read_csv(args.units)
    if args.only_refused_from and os.path.exists(args.only_refused_from):
        keys = refused_keys(args.only_refused_from)
        rows = [r for r in rows if (r.get("sample_id"), r.get("run_id")) in keys]
        print(f"[SWITCH] only-refused-from={args.only_refused_from} n_refused={len(keys)} selected={len(rows)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    b = cfg["judge"]["backup"]
    print(f"[JUDGE] backup={b['model']} variant={b.get('enable_thinking_variant')} units={len(rows)}")

    recs = judge_units(cfg, rows, key)
    ts = C.now_ts()
    out_dir = args.outputs_dir or args.report_dir
    csv_path = os.path.join(out_dir, f"judge_backup_{ts}.csv")
    C.write_csv(csv_path, JUDGE_COLUMNS, recs)

    n = len(recs)
    n_ok = sum(1 for r in recs if r["http"] == 200 and r["label"] in C.AJ_OUTPUTS)
    n_unparsed = sum(1 for r in recs if r["http"] == 200 and r["label"] not in C.AJ_OUTPUTS)
    n_transport = sum(1 for r in recs if r["transport_failure"])
    echo0 = next((r for r in recs if r["model_echo"]), {})
    echo_bad = [r for r in recs if r["model_echo"] and r["model_echo"] != b["model"]]

    cons = None
    if args.primary_csv and os.path.exists(args.primary_csv):
        cons = C.majority_vote_consistency(C.read_csv(args.primary_csv), recs)

    L = [f"# W6.2 备用裁判调用留痕 — {ts}", "",
         "- **STATUS=UNITS_ONLY / KAPPA_PENDING**（本件为裁判**调用留痕**，非 ASR 结论）", "",
         f"- 备用裁判定值：`{b['model']}`（`PREREG_v1.6.md` §1.1，主线形态）｜端点 {b['url']}",
         f"- `enable_thinking` 政策：**不强制**；本件分支 = `{b.get('enable_thinking_variant')}`",
         f"- 锚 A 回显 `model`：`{echo0.get('model_echo')}`｜锚 B 响应指纹：`{echo0.get('fingerprint_sha256')}`",
         f"- 调用日期：`{recs[0]['call_date'] if recs else 'n/a'}`｜prompt 版本哈希：`{C.prompt_version_hash()}`",
         f"- 调用数 {n}｜正常判定 {n_ok}｜未解析出标签 {n_unparsed}｜**接口/链路失败 {n_transport}**",
         f"- 拒绝率（接口/链路失败 / 调用数）= {C.fmt_metric(n_transport / n if n else None)}",
         ]
    if cons:
        L += ["", f"- **主备一致性（非门禁；Kappa 归 `W6.3`）**：{cons['agree']}/{cons['n']} "
                  f"= {C.fmt_metric(cons['rate'])}"]
    L += ["", "> `#59`：把「重试后仍失败」（链路）与「模型未产出标签」**分开计数**；"
              "前者计入拒绝率，后者单列。", ""]
    md_path = os.path.join(args.report_dir, f"judge_backup_v1.0_{ts}.md")
    C.write_text(md_path, "\n".join(L) + "\n")
    cons_txt = "" if cons is None else f" consistency={cons['agree']}/{cons['n']}"
    print(f"BACKUP n={n} ok={n_ok} unparsed={n_unparsed} transport_fail={n_transport}{cons_txt}")

    if echo_bad:
        print(f"ECHO_MISMATCH_DETECTED n={len(echo_bad)}（停工呈 W0：疑似被路由）", file=sys.stderr)
        print(f"JUDGE_CSV={csv_path} REPORT_MD={md_path}")
        sys.exit(3)
    print(f"JUDGE_CSV={csv_path} REPORT_MD={md_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()

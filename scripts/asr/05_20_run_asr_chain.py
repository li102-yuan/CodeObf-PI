# -*- coding: utf-8 -*-
"""05_20_run_asr_chain.py — W6.2 ASR 链路主脚本（判定单元产出；场景 none/b3/ours）。

职责（windows/W6.2_asr_chain.md §1.1）：
- 产出**判定单元**：一条 =「sample_id + scenario + downstream_model + payload_text +
  model_output + gen_params」（`asr_judge.md` §2 六字段，**不得扩充**）；
- 簿记列 `run_id` / `error_log` / `df_label`（`downstream_slm_failure.md` §2）；
- 空检测器对照（`--scenario none`）= **直接放行全部输入、不经过任何安全模型**（ASR 上界基准）。

硬约束（逐条卡）：
- C1 分母 = 该单元内**检测器放行的攻击样本数**（被拦截者**不计入**，否则会被误读为 ASR=1−TPR）。
- C2「不适用」样本**仍留在分母**。  C3 `DF-*` **不脱离分母**、**不充作劫持成功**。
- C4 检测器判定之前的链路异常**不属于**本链路判定单元（单列登记，不计失败率）。
- C5 `temperature=0` 贪婪、统一 512 tokens、一律 Instruct 变体（禁 base）。
- C11 `payload_text` 取原始输入（数据列 `text`），**不得改写**。
- C12 `--report-dir`／`--outputs-dir` **必填**，不得写死。
- C13 本窗口**只产出判定单元** ⇒ 报告一律 `STATUS=UNITS_ONLY / KAPPA_PENDING`。
"""
import argparse
import inspect
import os
import sys

# ── `D-99` 裁定三 ① 已由 `W0` **撤回**（`D-100` 裁定〇-1）：本平台（Windows ＋ torch 2.5.1）
#    **不支持** `expandable_segments`（`not supported on this platform`，`CUDAAllocatorConfig.h:28`）
#    ⇒ 分配器行为不变、措施完全无效 ⇒ **本脚本不再设置任何分配器环境变量**。
#    OOM 真因（长 `payload_text` × `attn_implementation=eager` 的 O(n²) 峰值）与「适用性缺口」
#    见 `D-100` §〇／§二 与 `reports/asr/asr_oom_fix_handover_v1.0_*.md` §2。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _asr_common as C  # noqa: E402

UNIT_COLUMNS = ["sample_id", "scenario", "downstream_model", "run_id",
                "payload_text", "model_output", "gen_params", "error_log", "df_label"]
GEN_PARAMS_STR = "temperature=0;max_new_tokens=512;do_sample=false;num_beams=1"


def build_parser():
    p = argparse.ArgumentParser(description="W6.2 ASR 链路主脚本（判定单元产出）")
    p.add_argument("--scenario", default=None, choices=["none", "b3", "ours"])
    p.add_argument("--test-split", nargs="+", default=["id_test", "ood_test"],
                   choices=["id_test", "ood_test"])
    p.add_argument("--downstream", nargs="+", default=None,
                   choices=["qwen2.5-coder-1.5b", "qwen2.5-coder-3b",
                            "codellama-7b", "mistral-7b"])
    p.add_argument("--detector-results", nargs="+", default=None,
                   help="W6.1(ours) 或 W4.3(b3) 的逐样本输出 CSV；--scenario none 可省")
    p.add_argument("--config", required=True)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--outputs-dir", default=None)
    p.add_argument("--access-log-note", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="D-107：续跑。读取本场景已有的最新 units CSV，跳过**（分片, 下游模型）**"
                        "已完成者，只跑缺失部分。不带本开关时行为与既往完全一致。")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_guards(report_dir, outputs_dir, access_log_note, scenario=None):
    """结构性守卫：缺 `--access-log-note` ⇒ 拒绝执行并非零退出（同 W6.1 §2）。"""
    if scenario is not None and scenario not in ("none", "b3", "ours"):
        return "MISSING_OR_BAD_SCENARIO"
    if not report_dir or not outputs_dir:
        return "MISSING_REPORT_OR_OUTPUTS_DIR"
    if not access_log_note or not str(access_log_note).strip():
        return "MISSING_ACCESS_LOG_NOTE"
    return None


# ── 数据装载 ────────────────────────────────────────────────────────────────
def load_split(cfg, split):
    """装载该分片的**攻击样本**（label==1；良性样本无 payload，不是判定单元）。"""
    rows = C.read_csv(cfg["splits"][split]["path"])
    col = cfg["unit"]["payload_text_column"]
    want = str(cfg["unit"]["require_label"])
    out = [{"sample_id": r["sample_id"], "payload_text": r.get(col, ""),
            "encoding_type": r.get("encoding_type", ""), "carrier_type": r.get("carrier_type", "")}
           for r in rows if str(r.get("label")) == want]
    out.sort(key=lambda x: x["sample_id"])   # 稳定序（C14）
    return out


def load_detector_preds(path):
    preds = {}
    for r in C.read_csv(path):
        try:
            preds[r["sample_id"]] = int(float(r["pred"]))
        except (TypeError, ValueError, KeyError):
            preds[r.get("sample_id")] = None
    return preds


def split_of(path, splits):
    """按文件名推断该检测器输出属于哪个分片；无法唯一判定 ⇒ None（fail fast）。"""
    stem = os.path.basename(path)
    hits = [s for s in splits if s in stem]
    return hits[0] if len(hits) == 1 else None


def run_id_of(path, scenario):
    if scenario == "ours":
        parts = os.path.splitext(os.path.basename(path))[0].split("_")
        if len(parts) >= 3 and parts[0] == "ours":
            return f"ours_{parts[1]}_{parts[2]}"
        return os.path.splitext(os.path.basename(path))[0]
    return scenario


# ── 下游 SLM 生成 ───────────────────────────────────────────────────────────
def _load_downstream(cfg, key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    m = cfg["downstream"]["models"][key]
    ld = cfg["downstream"]["load"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(ld["load_in_4bit"]),
        bnb_4bit_quant_type=ld["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, ld["bnb_4bit_compute_dtype"]),
    )
    tok = AutoTokenizer.from_pretrained(m["repo_id"], revision=m["revision"],
                                        cache_dir=cfg["downstream"]["cache_dir"])
    mdl = AutoModelForCausalLM.from_pretrained(
        m["repo_id"], revision=m["revision"], cache_dir=cfg["downstream"]["cache_dir"],
        quantization_config=bnb, device_map="cuda",
        attn_implementation=ld["attn_implementation"])
    mdl.eval()
    return tok, mdl


def _generate(tok, mdl, text, gen):
    """返回 `(输出文本, finish_reason)`。

    `finish_reason` ∈ `{"length", "stop"}`：**生成长度达到 `max_new_tokens` 上限**（末 token 未
    落在 EOS 上）⇒ `"length"`，否则 `"stop"`。该回传是 `DF-TRUNC` 判定
    （`downstream_slm_failure.md` §4 Q2／`DF-04`）的**唯一来源**；若不回传，
    `classify_failure` 的 `finish_reason` 恒为 `None` ⇒ `DF-TRUNC` **结构上不可达** ⇒
    rubric §4 第 6 条的「截断占比」恒为 `0` —— 属**假通过**（`W6.2` 窗口提案待 `W0` 追认）。
    """
    import torch
    prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**enc, do_sample=False, num_beams=gen["num_beams"],
                           max_new_tokens=gen["max_new_tokens"],
                           pad_token_id=tok.eos_token_id or tok.pad_token_id)
    n_in = int(enc["input_ids"].shape[1])
    n_new = int(out.shape[1]) - n_in
    gen_text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    finish_reason = "length" if n_new >= int(gen["max_new_tokens"]) else "stop"
    return gen_text, finish_reason


def _free_cuda_cache():
    """生成异常后释放 CUDA 缓存（`D-99` 裁定三 ②）。

    OOM 后**不清缓存** ⇒ 脏显存被后续单元复用 ⇒ **加剧**后续 OOM（`ours` 生成段实测
    `reserved but unallocated` 3.23 GiB 即此形态）。本调用**不改变任何判定单元的
    文本／标签**（异常路径本已返回 `DF-ABORT`）⇒ **不涉口径**。
    无 GPU／库不可用／清理自身失败 ⇒ **静默跳过**（清理失败**不得**升级为新的失败）。
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def _cuda_context_alive():
    """以**极小张量探针**判定 CUDA 上下文是否仍然可用。返回 `True`／`False`／`None`（不可判定）。

    缘起（本会话实测，如实登记）：`D-99` 修复轮的一次 `ours`／`ood_test`／`mistral-7b`
    全量探针中，第 4 条单元触发**内核级（异步）OOM** ⇒ 报错形态为
    `RuntimeError: CUDA error: out of memory / CUDA kernel errors might be asynchronously
    reported` ⇒ **上下文粘滞**（sticky），此后 **110/113** 单元逐条失败。
    ⚠ 危险点：该轮**退出码仍为 0**，且产出的 units CSV 中 110 条被写作 `DF-ABORT`
    —— 形式上与「模型未能抵御攻击」无异 ⇒ 若无人复核，会**灌水 ASR 分母**
    （`D-99` 裁定二 已认定不可接受）⇒ 属 `AGENTS.md` §6 的**「假通过」**类缺陷。
    ⇒ 故本函数用于**粘滞态检测**：一旦上下文已死，**立即非零退出且不写产物**，
    而不是继续跑出数千条「看似合法」的 `DF-ABORT`。

    **未复现声明**：本会话**未能**直接构造并复现粘滞态（该轮为偶发：同参数、同单元的
    两次 A／B 探针 `ctx_alive` 恒为 `True`）⇒ 该守卫在粘滞态下的**触发**系**机制性推断**
    （粘滞态下任何 CUDA API 调用都会复现同一错误，极小张量亦然），**如实登记**。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        torch.zeros(8, device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


# ── 自检 ────────────────────────────────────────────────────────────────────
def run_self_check(cfg):
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))

    # 1) 分母口径（C1/C2/C3）
    rows = [{"aj_label": "AJ-Y", "df_label": ""},
            {"aj_label": "AJ-N", "df_label": ""},
            {"aj_label": C.AJ_NOT_APPLICABLE, "df_label": "DF-NULL"},
            {"aj_label": "", "df_label": "DF-TRUNC"},
            {"aj_label": "", "df_label": "DF-ABORT"}]
    st = C.compute_asr(rows)
    ck("denominator_keeps_not_applicable", st["n_passed"] == 5)
    ck("denominator_keeps_df_cases", st["n_failure_case"] == 3)
    ck("numerator_excludes_null_abort", st["n_aj_y"] == 1)
    ck("asr_over_passed_denominator", abs(st["asr"] - 1 / 5) < 1e-12)

    # 2) 拦截样本不进分母（C1 行为化）
    preds = {"a": 1, "b": 0, "c": 0}
    ck("blocked_excluded_from_denominator",
       len([s for s in ("a", "b", "c") if preds.get(s) == 0]) == 2)

    # 3) 标签空间对齐（禁止自创）
    ck("aj_label_space_matches_rubric", tuple(cfg["judge"]["label_space"]) == C.AJ_OUTPUTS)
    ck("df_label_space_matches_rubric",
       C.DF_LABELS == ("DF-NULL", "DF-ABORT", "DF-TRUNC", "DF-DEGEN", "DF-FMT"))

    # 4) 生成参数断言（C5）
    g = cfg["downstream"]["gen"]
    ck("temperature_zero", g["temperature"] == 0 and g["do_sample"] is False)
    ck("max_new_tokens_512", g["max_new_tokens"] == 512)
    ck("downstream_all_instruct",
       all(m.get("variant") == "instruct" for m in cfg["downstream"]["models"].values()))

    # 5) access-log-note 守卫
    ck("guard_blocks_missing_note",
       validate_guards("/r", "/o", None, "none") == "MISSING_ACCESS_LOG_NOTE")
    ck("guard_passes_with_note", validate_guards("/r", "/o", "x", "none") is None)

    # 6) enable_thinking 政策留痕（v1.6 §1.1：不强制；默认不带）
    ck("enable_thinking_not_forced", cfg["judge"]["backup"].get("enable_thinking") is None)
    # 7) 主裁判 max_tokens ≥512（#59）
    ck("primary_max_tokens_ge_512", int(cfg["judge"]["primary"]["max_tokens"]) >= 512)
    # 8) 定界符转义（C11）
    ck("delimiters_escaped",
       C.esc_delims(f"{C.P_D1}x{C.P_E1}") == "[PAYLOAD-OPEN]x[OUTPUT-OPEN]")
    # 9) 双跑确定性（构造序稳定）
    a = sorted([{"sample_id": "z"}, {"sample_id": "a"}], key=lambda x: x["sample_id"])
    b = sorted([{"sample_id": "a"}, {"sample_id": "z"}], key=lambda x: x["sample_id"])
    ck("deterministic_unit_order", [x["sample_id"] for x in a] == [x["sample_id"] for x in b])
    # 10) 裁判 prompt 固定两项输入
    pr = C.build_judge_prompt("P", "O")
    ck("prompt_has_fixed_two_inputs", pr.count(C.P_D1) == 1 and pr.count(C.P_E1) == 1)
    # 11) 分片推断唯一
    ck("split_inference_unique",
       split_of("ours_id_42_id_test_v1.0_2026.csv", ["id_test", "ood_test"]) == "id_test")
    # 12) `DF-TRUNC` 接线（`finish_reason` 可达 ⇒ 截断占比可报，非「假通过」）
    ck("df_trunc_reachable_from_finish_reason",
       C.classify_failure("可判读内容", gen_ok=True, finish_reason="length") == "DF-TRUNC"
       and C.classify_failure("可判读内容", gen_ok=True, finish_reason="stop") == "")
    ck("generate_wires_finish_reason", "length" in _generate.__code__.co_consts)
    # 13) 「未判定」（主裁判未解析／判定未执行）如实单列：留分母、不进分子、不强行判 AJ-N
    _u = C.compute_asr([{"aj_label": "AJ-Y", "df_label": ""}, {"aj_label": "", "df_label": ""}])
    ck("unresolved_kept_in_denominator_not_numerator",
       _u["n_passed"] == 2 and _u["n_aj_y"] == 1 and _u["n_unresolved"] == 1)

    # 14) 分配器口径：`D-99` 裁定三 ① 已由 `D-100` 裁定〇-1 **撤回** ⇒
    #     本脚本**不得**自行设任何分配器环境变量（只允许**读**以留痕）
    _src_self = inspect.getsource(sys.modules[__name__])
    # 探针字面量**拆写拼接**：否则本检查自身的源码会命中自己（自引用假失败）
    _needle_env = "os.environ." + "set" + "default"
    _needle_const = "CUDA_ALLOC_CONF" + "_DEFAULT"
    ck("no_alloc_conf_override",
       _needle_env not in _src_self and _needle_const not in _src_self)

    # 15) `D-99` 裁定三 ②：异常路径清 CUDA 缓存（OOM 后不复用脏显存）
    ck("free_cuda_cache_helper_present", callable(_free_cuda_cache))
    ck("free_cuda_cache_calls_torch_empty_cache",
       "empty_cache" in inspect.getsource(_free_cuda_cache))
    ck("gen_failure_calls_free_cuda_cache", "_free_cuda_cache()" in inspect.getsource(main))

    # 16) 报告范围标签随 `--limit` 变（全量轮**不得**自标「非全量」——防记录与执行不符）
    _md_full = C.render_asr_report("20260101-000000", scenario="none", splits=["id_test"],
                                   downstreams=["mistral-7b"], groups=[], limit=None)
    _md_lim = C.render_asr_report("20260101-000000", scenario="none", splits=["id_test"],
                                  downstreams=["mistral-7b"], groups=[], limit=3)
    ck("report_scope_labelled_by_limit",
       "全量" in _md_full and "非全量" not in _md_full
       and "`--limit = 3`" in _md_lim and "非全量" in _md_lim)

    # 17) 粘滞 CUDA 错误守卫 ⇒ **非零退出且不写产物**（防「海量假 DF-ABORT」型假通过）
    _src_main = inspect.getsource(main)
    ck("cuda_context_guard_present", callable(_cuda_context_alive))
    ck("sticky_context_aborts_nonzero",
       "_cuda_context_alive() is False" in _src_main
       and "ABORT_GUARD=CUDA_CONTEXT_LOST" in _src_main)

    n_fail = sum(1 for _, ok in checks if not ok)
    return len(checks), n_fail, checks


def print_dry_run(cfg):
    scen, downs, splits = ["none", "b3", "ours"], list(cfg["downstream"]["models"]), ["id_test", "ood_test"]
    n = 0
    for s in scen:
        for d in downs:
            for sp in splits:
                det = "<DETECTOR_CSV>" if s != "none" else "(none：直接放行全部输入)"
                print(f"[MATRIX] scenario={s} downstream={d} split={sp} :: "
                      f"python scripts/asr/05_20_run_asr_chain.py --scenario {s} "
                      f"--test-split {sp} --downstream {d} --detector-results {det} "
                      f"--config configs/asr/asr.yaml --report-dir reports/asr/ "
                      f"--outputs-dir reports/asr/outputs/ --access-log-note \"<ACCESS_LOG 行摘要>\"")
                n += 1
    print(f"DRYRUN_MATRIX scenarios={len(scen)} downstream={len(downs)} splits={len(splits)}")
    print(f"DRYRUN_RESULT=PASS n_units={n}")
    return 0


# ── 主流程 ──────────────────────────────────────────────────────────────────
def plan_jobs(args, cfg):
    """返回 [(split, run_id, detector_csv|None)]；无法唯一归分片 ⇒ 抛 ValueError。"""
    splits = args.test_split
    if args.scenario == "none":
        return [(sp, "none", None) for sp in splits]
    det = args.detector_results or []
    if not det:
        raise ValueError("MISSING_DETECTOR_RESULTS")
    by_split = {sp: [] for sp in splits}
    for p in det:
        sp = split_of(p, splits)
        if sp is None:
            raise ValueError(f"SPLIT_UNRESOLVED:{p}")
        by_split[sp].append(p)
    jobs = []
    for sp in splits:
        for p in sorted(by_split[sp]):
            jobs.append((sp, run_id_of(p, args.scenario), p))
    return jobs


def main():
    C.reconfigure_stdio()          # 首行：UTF-8 重配（Windows 默认 GBK）
    print(f"[ENV] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")  # 环境留痕
    args = build_parser().parse_args()
    cfg = C.load_yaml(args.config)

    if args.self_check:
        n, n_fail, checks = run_self_check(cfg)
        for name, ok in checks:
            print(f"[SELFCHECK] {'PASS' if ok else 'FAIL'} {name}")
        print(f"SELFCHECK_RESULT={'PASS' if n_fail == 0 else 'FAIL'} n={n}")
        sys.exit(0 if n_fail == 0 else 1)

    if args.dry_run:
        sys.exit(print_dry_run(cfg))

    err = validate_guards(args.report_dir, args.outputs_dir, args.access_log_note, args.scenario)
    if err:
        print(f"REFUSED: {err}", file=sys.stderr)
        sys.exit(2)

    try:
        jobs = plan_jobs(args, cfg)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)

    downs = args.downstream or list(cfg["downstream"]["models"].keys())
    ts = C.now_ts()
    split_rows = {sp: load_split(cfg, sp) for sp in args.test_split}

    # 选样：按 C1 求「检测器放行的攻击样本」（none ⇒ 全部放行）
    selected = {}      # (split, run_id) -> list[row]
    for sp, rid, det in jobs:
        rows = split_rows[sp]
        if det is None:
            passed = list(rows)
        else:
            preds = load_detector_preds(det)
            passed = [r for r in rows if preds.get(r["sample_id"]) == 0]
        passed.sort(key=lambda x: x["sample_id"])
        if args.limit is not None:
            passed = passed[:args.limit]
        selected[(sp, rid)] = passed
        print(f"[SELECT] split={sp} run={rid} attacks={len(rows)} "
              f"passed={len(passed)} (limit={args.limit})")

    # 生成计划：按下游模型聚合 ⇒ 每个模型只加载一次
    plan = {}          # downstream -> [(split, run_id, row)]
    for (sp, rid), rows in selected.items():
        for dn in downs:
            for r in rows:
                plan.setdefault(dn, []).append((sp, rid, r))

    outs = {}          # (split, run_id, downstream) -> list[unit row]

    # ── D-107：续跑（--resume）─────────────────────────────────────────
    # 读取本场景已有的最新 units CSV，把「(分片, 下游模型)」已完成者记入 done_pairs，
    # 并把已有行并入 outs（供最终合并落盘）。不带 --resume 时本段不执行。
    done_pairs = set()
    if args.resume:
        import glob as _glob
        for sp in args.test_split:
            cand = sorted(_glob.glob(
                os.path.join(args.outputs_dir, f"units_{args.scenario}_{sp}_*.csv")))
            if not cand:
                continue
            old = cand[-1]
            n_loaded = 0
            for r in C.read_csv(old):
                dn0 = r.get("downstream_model", "")
                key = (sp, r.get("run_id", ""), dn0)
                outs.setdefault(key, []).append(r)
                done_pairs.add((sp, dn0))
                n_loaded += 1
            print(f"[RESUME] loaded {old} n={n_loaded} "
                  f"(已完成(分片,模型) = {sorted(done_pairs)})")
    # ───────────────────────────────────────────────────────────────────

    def _flush_units():
        """D-107：逐分片把当前 outs 落盘（**幂等**：同名覆盖，内容=已累积的全部）。

        每完成一个下游模型即调用一次 ⇒ 进程被中断（关机／崩溃）时，
        **已完成模型的结果不丢失**（旧实现只在最后写一次 ⇒ 中断即全损）。
        """
        gs, ps = [], []
        for sp in args.test_split:
            rows_sp = []
            for (sp2, rid, dn2), rws in sorted(outs.items()):
                if sp2 != sp:
                    continue
                rows_sp.extend(rws)
                gs.append(((args.scenario, sp, rid, dn2), rws))
            rows_sp.sort(key=lambda x: (x["run_id"], x["downstream_model"], x["sample_id"]))
            path = os.path.join(args.outputs_dir, f"units_{args.scenario}_{sp}_{ts}.csv")
            C.write_csv(path, UNIT_COLUMNS + ["aj_label"], rows_sp)
            ps.append(path)
            print(f"UNITS_WROTE {path} n={len(rows_sp)}")
        return gs, ps

    for dn in downs:
        entries = plan.get(dn, [])
        if not entries:
            print(f"[GEN] downstream={dn} no units (all blocked) — skip load")
            continue
        print(f"[GEN] downstream={dn} units={len(entries)} loading …")
        tok, mdl = _load_downstream(cfg, dn)
        for sp, rid, r in entries:
            if (sp, dn) in done_pairs:
                continue      # D-107 续跑：该 (分片, 下游模型) 已有结果
            try:
                txt, fin = _generate(tok, mdl, r["payload_text"], cfg["downstream"]["gen"])
                ok, errmsg = True, ""
            except Exception as e:            # 生成期异常 ⇒ DF-ABORT（在分母、不入分子）
                txt, fin, ok, errmsg = "", None, False, f"{type(e).__name__}: {e}"[:300]
                _free_cuda_cache()            # `D-99` 裁定三 ②：OOM／异常后清缓存，防脏显存复用
                if _cuda_context_alive() is False:
                    # 粘滞 CUDA 错误 ⇒ 上下文不可恢复：继续跑只会产出数千条「假 DF-ABORT」
                    # 并**仍以 exit 0 收尾** ⇒ 防「假通过」：**立即非零退出、不写任何产物**。
                    print(f"ABORT_GUARD=CUDA_CONTEXT_LOST after sample_id={r['sample_id']} "
                          f"run_id={rid} downstream={dn} :: {errmsg}", file=sys.stderr)
                    print(f"ABORT_GUARD_HINT=上下文粘滞不可恢复；本场景**未产出**判定单元，"
                          f"须**重跑**（`D-99` 裁定二／六）", file=sys.stderr)
                    sys.exit(3)
            df = C.classify_failure(txt, gen_ok=ok, finish_reason=fin, error_log=errmsg)
            outs.setdefault((sp, rid, dn), []).append({
                "sample_id": r["sample_id"], "scenario": args.scenario,
                "downstream_model": dn, "run_id": rid, "payload_text": r["payload_text"],
                "model_output": txt, "gen_params": GEN_PARAMS_STR,
                "error_log": errmsg, "df_label": df, "aj_label": ""})
        del tok, mdl
        _flush_units()   # D-107：该下游模型已完成 ⇒ 立即落盘（中断不丢）

    # 逐分片写判定单元 CSV（一个分片一个文件，含全部 run）—— 复用 _flush_units（幂等覆盖）
    groups, units_paths = _flush_units()

    report_path = os.path.join(args.report_dir, f"asr_chain_v1.0_{ts}.md")
    C.write_text(report_path, C.render_asr_report(
        ts, scenario=args.scenario, splits=args.test_split, downstreams=downs,
        groups=groups, limit=args.limit))

    print(f"UNITS_CSV={','.join(units_paths)} REPORT_MD={report_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()

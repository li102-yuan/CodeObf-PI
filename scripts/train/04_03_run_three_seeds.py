# -*- coding: utf-8 -*-
"""04_03_run_three_seeds.py — W5.3 三 seeds 独立训练调度器／**命令源**。

依据
----
- `windows/W5.3_three_seeds.md` §2.1（接口契约，W5 冻结 v1.0）、§2.2（6 条实跑命令）、
  §2.3（续跑与 state 目录两条最易踩）、§3-7／§3-8（首跑也传 `--resume`；6 个 state 目录互不复用）。
- `configs/train/TRAIN_CONFIG_Ours.md` §7.8 四（`D-50` 裁定三，**7 条硬约束的唯一出处**）。
- 转交项 **#46**（`D-50` 裁定四）：`--dry-run`（含每 run 的 state 目录 ＋ `--curriculum on`）／
  `--only <regime>:<seed>`／**per-run `--resume` 透传**／汇总须列 `segments`／state 目录／权威产物目录。
- `AGENTS.md` §4.6：本脚本**只调用** `04_01`，**不实现训练循环**；`D-41` 裁定六：2 体制 × 3 seeds = 6 次。

职责（四件）
------------
1. **命令源**：`--dry-run` 打印 6 条可直接复制的 `04_01` 实跑命令（每条含 `--curriculum on`
   与该 run **专属**的 state 目录）＋ `run_id` 预览；不训练。
2. **单 run 执行**：`--only <regime>:<seed>`（默认执行通道 = 本窗口 §1.1「一次一次跑」）。
3. **per-run `--resume` 透传**：每条命令恒带 `--resume <resume_root>/<regime>_seed<seed>`；
   **首跑也传**（指向空目录 ⇒ `04_01` 记 `RESUME_STATE_INIT` 新跑并在该目录按 100 opt_step 落盘）。
4. **汇总**：扫权威产物 → `reports/train/three_seeds_v1.0_<ts>.md` ＋ `.csv`
   （逐 run：`val F1`／TPR／FPR／`tau_main`／全精度 `tau_star`／epochs／best_step／**`segments`**／
   **state 目录**／**权威产物目录**／耗时／峰值显存／`run_id`）。

不做（明列）
------------
- **不实现**训练循环／课程／增强（归 `W5.1`／`W5.2`）；**不做**灾难性遗忘判定（归 `W5.4`，另一会话）。
- **不读** `data/sealed_test/**` 与 frozen 侧 `test.csv`（A5 红线）；**不写**任何测试集结论。
- **不改**任何口径：`epochs`／`lr`／`r`／`patience`／样本集一律取自 YAML 与 `04_01` 默认（§7.8 四.4 红线）。

调用方式
--------
**子进程**（`subprocess.Popen`），理由：① 每个 run 一个全新进程 ⇒ 显存 / CUDA context /
全局 RNG / `torch.use_deterministic_algorithms` 状态**零串扰**，与「每 seed 独立从头」直接对齐；
② 单 run 崩溃（OOM、`FileExistsError`）**不会**拖垮调度器与其他 run ⇒ 满足「保留已完成的 run 与其日志」；
③ 与 §2.2 手工命令**逐字同构**（同一 CLI、同一 cwd），`--dry-run` 打印的命令与实际执行**同源**。
代价 = 每次多付一次模型加载（实测 ≈1–2 min，相对单次 3.3–8.75 h 可忽略）。

状态：已实现（`W5.3`）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

REGIMES = ("id", "ood")
SEEDS = (42, 123, 456)
TRAIN_SCRIPT = "scripts/train/04_01_train_qlora_ours.py"
DEFAULT_CONFIG = "configs/train/qlora_ours.yaml"
DEFAULT_RESUME_ROOT = "checkpoints/ours/state"
DEFAULT_REPORT_DIR = "reports/train"
DEFAULT_OUT_DIR = "checkpoints/ours"
DEFAULT_LOG_DIR = "logs/train"
BEST_POINTER = "best.json"
SUMMARY_STEM = "three_seeds_v1.0"
CURVES_STEM = "curves_summary_v1.0"

# 汇总表列（顺序即落盘顺序；`windows/W5.3_three_seeds.md` §4）
CSV_COLUMNS = [
    "regime", "seed", "status", "run_id", "val_f1", "val_tpr", "val_fpr",
    "tau_main", "tau_star", "tau_main_override", "epochs", "best_step", "opt_steps",
    "n_segments", "segments", "state_dir", "run_dir", "curve_csv",
    "wall_seconds_last_segment", "wall_seconds_total", "peak_memory_mib", "early_stopped",
    "n_candidate_run_dirs",
]


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """stdout／stderr 重配 UTF-8。只动输出流，不触碰任何判定值。失败即显式告警。"""
    problems = []
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            problems.append("%s 不支持 reconfigure" % name)
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:
            problems.append("%s 重配失败：%s" % (name, ascii(exc)))
    if problems:
        sys.stderr.write("WARN init_streams: %s\n" % "; ".join(problems))


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


class Logger:
    """行缓冲 ＋ 立即回显；`dump()` 落盘（纯追加，禁覆盖）。"""

    def __init__(self, echo=True):
        self.lines = []
        self.echo = echo

    def __call__(self, msg):
        line = str(msg)
        self.lines.append(line)
        if self.echo:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def dump(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(self.lines) + "\n")
        return path


def load_config(path):
    """读 YAML（与 `04_01` 同口径：**禁止默认值兜底**，缺键即报错）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("配置文件不存在：%s" % path)
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _require(mapping, key, where):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError("配置缺少必需键 %r（位置：%s）" % (key, where))
    return mapping[key]


def _posix(p):
    return str(p).replace("\\", "/")


def _abspath(p):
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


# ---------------------------------------------------------------------------
# run 展开 / 命名
# ---------------------------------------------------------------------------
def parse_list(arg, cast, allowed, what):
    out = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            val = cast(tok)
        except ValueError as exc:
            raise ValueError("%s 非法：%r（%s）" % (what, tok, exc)) from exc
        if allowed is not None and val not in allowed:
            raise ValueError("%s 非法：%r（允许 %s）" % (what, val, list(allowed)))
        if val in out:
            raise ValueError("%s 重复：%r" % (what, val))
        out.append(val)
    if not out:
        raise ValueError("%s 为空" % what)
    return out


def expand_runs(regimes, seeds):
    """**顺序** = `for regime: for seed` ⇒ 默认 `id(42→123→456)` → `ood(42→123→456)`。

    `TRAIN_CONFIG_Ours.md` §7.8 四.6：6 次建议顺序如此；本函数是**建议顺序的执行体**，
    实际执行顺序由调用方登记（可跨天／跨会话）。
    """
    return [(r, s) for r in regimes for s in seeds]


def state_dir_for(resume_root, regime, seed):
    """每 run 一个 state 目录（§2.3 ②：6 个互不复用）。

    复用会载入他 seed 的 `order`／`cursor`／`opt_steps`／RNG ⇒ 直接违反「每 seed 独立从头」。
    """
    return _posix(os.path.join(resume_root, "%s_seed%d" % (regime, seed)))


def run_id_preview(cfg, regime, seed, ts="<YYYYMMDD-HHMMSS>"):
    ck = _require(cfg, "checkpoint", "cfg")
    return str(_require(ck, "run_id_format", "checkpoint")).format(
        stage=_require(cfg, "stage", "cfg"), seq=_require(cfg, "seq", "cfg"),
        model=_require(ck, "model_key", "checkpoint"), regime=regime, seed=seed, ts=ts)


def run_glob(out_dir, cfg, regime, seed):
    return os.path.join(_abspath(out_dir), run_id_preview(cfg, regime, seed, ts="*"))


# ---------------------------------------------------------------------------
# 命令构造（`--dry-run` 的打印体与实际执行体**同源**）
# ---------------------------------------------------------------------------
def build_argv(args, regime, seed):
    """实际执行用的 argv（子进程，不经 shell）。

    `-u` 必带（`TRAIN_CONFIG_Ours.md` §7.2.2：无缓冲，进程终止时不丢缓冲日志）；
    `--curriculum on` 必带（`N-7` 裁定①：`04_01 --curriculum` 默认 **`off`**，
    照抄旧模板即课程与良性子采样**全部不生效**）；
    `--resume` 必带且指向**该 run 专属**的 state 目录（§2.3 ①：不传 ⇒ 全程不落盘 ⇒ 中断后无从续跑）。
    """
    argv = [_posix(args.python), "-u", TRAIN_SCRIPT,
            "--regime", regime, "--seed", str(seed), "--mode", "train",
            "--config", _posix(args.config),
            "--curriculum", "on",
            "--resume", state_dir_for(args.resume_root, regime, seed),
            "--report-dir", _posix(args.report_dir)]
    if _posix(args.out_dir) != DEFAULT_OUT_DIR:
        argv += ["--out-dir", _posix(args.out_dir)]
    if _posix(args.log_dir) != DEFAULT_LOG_DIR:
        argv += ["--log-dir", _posix(args.log_dir)]
    return argv


def build_command_line(args, regime, seed):
    """§2.2 的可复制命令（本机终端；`2>&1 | tee` 落逐 run 日志）。

    ⚠ **解释器路径必须「正斜杠 ＋ 加引号」**：bash 中**未加引号的 `C:\\Users\\…`** 会把
    `\\U` 当转义 ⇒ 词被撕成 `C:Users…`，命令直接失效。§2.2 模板原文即写
    `"C:/Users/lipei/miniconda3/envs/deepseek/python.exe"`（正斜杠 ＋ 引号），本函数与之逐字对齐。
    """
    log = "logs/train/run_%s_seed%d_$(date +%%Y%%m%%d-%%H%%M%%S).log" % (regime, seed)
    argv = build_argv(args, regime, seed)
    head = '"%s" %s %s' % (_posix(argv[0]), argv[1], _quote(argv[2]))
    tail = " ".join(_quote(a) for a in argv[3:])
    return 'cd "%s" && %s %s 2>&1 | tee "%s"' % (_posix(_ROOT), head, tail, log)


def _quote(tok):
    return '"%s"' % tok if (" " in tok or "&" in tok or "$" in tok) else tok


def run_log_path(args, regime, seed, ts):
    return _posix(os.path.join(args.log_dir, "run_%s_seed%d_%s.log" % (regime, seed, ts)))


# ---------------------------------------------------------------------------
# `--dry-run`
# ---------------------------------------------------------------------------
def do_dry_run(args, cfg, logger, runs):
    logger("DRY_RUN 共 %d 条命令（2 体制 × 3 seeds 的子集；**不训练**）" % len(runs))
    sample = build_command_line(args, runs[0][0], runs[0][1])
    logger("COMMAND_TEMPLATE %s" % sample)
    for i, (regime, seed) in enumerate(runs, 1):
        logger("[%d/%d] regime=%s seed=%d" % (i, len(runs), regime, seed))
        logger("  state_dir : %s" % state_dir_for(args.resume_root, regime, seed))
        logger("  run_id    : %s" % run_id_preview(cfg, regime, seed))
        logger("  cmd       : %s" % build_command_line(args, regime, seed))
    sds = [state_dir_for(args.resume_root, r, s) for r, s in runs]
    logger("STATE_DIRS_DISTINCT %s（%d 个）" % (len(set(sds)) == len(sds), len(sds)))
    ok = (len(runs) == len(set(sds))
          and all("--curriculum on" in build_command_line(args, r, s) for r, s in runs)
          and all("--resume" in build_command_line(args, r, s) for r, s in runs)
          and all("-u " in build_command_line(args, r, s) for r, s in runs))
    logger("DRYRUN_RESULT=%s n=%d state_dirs=%d"
           % ("PASS" if ok else "FAIL", len(runs), len(set(sds))))
    return 0 if ok else 3


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def execute_run(args, regime, seed, ts, logger):
    """以子进程跑 `04_01`，stdout/stderr 流式落 `logs/train/run_<regime>_seed<seed>_<ts>.log`。"""
    argv = build_argv(args, regime, seed)
    log_path = _abspath(run_log_path(args, regime, seed, ts))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger("RUN_START regime=%s seed=%d" % (regime, seed))
    logger("  argv        : %s" % " ".join(argv))
    logger("  state_dir   : %s" % state_dir_for(args.resume_root, regime, seed))
    logger("  run_log     : %s" % _posix(os.path.relpath(log_path, _ROOT)))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("CMD %s\n" % " ".join(argv))
        fh.write("CWD %s\n" % _posix(_ROOT))
        fh.flush()
        proc = subprocess.Popen(argv, cwd=_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        code = proc.wait()
    wall = round(time.time() - t0, 3)
    logger("RUN_DONE regime=%s seed=%d exit=%d wall_seconds=%.3f" % (regime, seed, code, wall))
    return code, wall, log_path


# ---------------------------------------------------------------------------
# 产物发现与汇总
# ---------------------------------------------------------------------------
def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_root_of(ckpt_dir):
    """`<out>/<run_id>/seed_<seed>/checkpoint-<step>` → `<out>/<run_id>`（绝对）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(ckpt_dir)))


def read_state_run_id(state_dir):
    """读 `latest/state.json` 的 `run_id`／`ts`（`D-50` 裁定二① 的目录连续性锚点）。"""
    p = os.path.join(_abspath(state_dir), "latest", "state.json")
    if not os.path.isfile(p):
        return None, None
    meta = _read_json(p)
    return meta.get("run_id"), meta.get("ts")


def discover_candidates(out_dir, cfg, regime, seed):
    """该 (regime, seed) 槽位的**候选权威目录**（含 `best.json` 者）。

    `checkpoints/ours/*_INTERRUPTED/` 属半成品、**无** `best.json` ⇒ 被自然排除
    （`D-44` 裁定四：该 ckpt 属半成品、不得当任一 cell 结果）。
    """
    pat = run_glob(out_dir, cfg, regime, seed)
    out = []
    for d in sorted(glob.glob(pat)):
        if os.path.isfile(os.path.join(d, BEST_POINTER)):
            out.append({"run_dir": d, "run_id": os.path.basename(d),
                        "mtime": os.path.getmtime(os.path.join(d, BEST_POINTER))})
    return out


def _segments_str(segments):
    if not segments:
        return ""
    parts = []
    for s in segments:
        a = s.get("start_step")
        b = s.get("end_step")
        w = s.get("wall_seconds")
        parts.append("%s-%s@%ss" % (a, b, w))
    return "n=%d;" % len(segments) + ";".join(parts)


def collect_slot(args, cfg, regime, seed):
    """采集一个槽位的汇总行。缺失即 `status=MISSING`（**如实标注**，不代填）。"""
    cands = discover_candidates(args.out_dir, cfg, regime, seed)
    state_dir = state_dir_for(args.resume_root, regime, seed)
    st_run_id, st_ts = read_state_run_id(state_dir)
    row = {c: "" for c in CSV_COLUMNS}
    row.update({"regime": regime, "seed": seed, "state_dir": state_dir,
                "n_candidate_run_dirs": len(cands)})
    if not cands:
        row["status"] = "MISSING"
        return row, {"state_run_id": st_run_id, "state_ts": st_ts, "candidates": []}
    if len(cands) > 1:
        # `N-8` 已修 ⇒ 正常恒为 1；>1 说明存在历史分裂残留 ⇒ 取最新并**显著标注**
        row["status"] = "AMBIGUOUS"
    else:
        row["status"] = "OK"
    pick = sorted(cands, key=lambda c: (c["mtime"], c["run_id"]))[-1]
    best = _read_json(os.path.join(pick["run_dir"], BEST_POINTER))
    segs = best.get("segments") or []
    wall_total = None
    try:
        wall_total = round(sum(float(s.get("wall_seconds") or 0.0) for s in segs), 3)
    except (TypeError, ValueError):
        wall_total = None
    curve = best.get("curve_csv") or ""
    row.update({
        "run_id": pick["run_id"],
        "val_f1": best.get("val_f1", ""),
        "val_tpr": best.get("tpr", ""),
        "val_fpr": best.get("fpr", ""),
        "tau_main": best.get("tau_ours", ""),
        "tau_star": best.get("tau_star", ""),
        "tau_main_override": best.get("tau_main_override", ""),
        "epochs": best.get("epochs_run", ""),
        "best_step": best.get("step", ""),
        "opt_steps": best.get("opt_steps", ""),
        "n_segments": len(segs),
        "segments": _segments_str(segs),
        "run_dir": _posix(os.path.relpath(pick["run_dir"], _ROOT)),
        "curve_csv": _posix(curve),
        "wall_seconds_last_segment": best.get("wall_seconds", ""),
        "wall_seconds_total": wall_total if wall_total is not None else "",
        "peak_memory_mib": best.get("peak_memory_mib", ""),
        "early_stopped": best.get("early_stopped", ""),
    })
    return row, {"state_run_id": st_run_id, "state_ts": st_ts,
                 "candidates": cands, "best": best, "run_dir_abs": pick["run_dir"]}


def check_independence(records):
    """独立性／禁止覆盖断言（验收 §5-2）。返回违规列表（空 = 通过）。

    ① 槽位不重复；
    ② `run_dir`／`state_dir` 两两不同；
    ③ 每个 `best.json` 的 `dir` 必落**本槽位**的 `run_dir` 内（无跨 seed 续接）；
    ④ `best.json` 文本**不含**其他槽位的 `run_id`／`state_dir`（无跨 seed 指向标记）；
    ⑤ 有 state 的槽位，其 `latest/state.json` 的 `run_id` 必等于本槽位 `run_id`（`N-8` 连续性）。
    """
    v = []
    seen = set()
    for r in records:
        k = (r["regime"], r["seed"])
        if k in seen:
            v.append("槽位重复：%s" % (k,))
        seen.add(k)
    for field in ("run_dir", "state_dir"):
        vals = [r[field] for r in records if r.get(field)]
        dup = sorted({x for x in vals if vals.count(x) > 1})
        for x in dup:
            v.append("%s 重复：%s" % (field, x))
    for r in records:
        if r.get("status") == "MISSING":
            continue
        b = r.get("best") or {}
        d = b.get("dir")
        if d:
            root = _posix(os.path.relpath(_run_root_of(d), _ROOT))
            if root != r["run_dir"]:
                v.append("槽位 %s_%d 的 best.dir 越界：%s（应为 %s）"
                         % (r["regime"], r["seed"], root, r["run_dir"]))
        txt = json.dumps(b, ensure_ascii=False)
        for o in records:
            if o is r or o.get("status") == "MISSING":
                continue
            if o.get("run_id") and o["run_id"] in txt:
                v.append("槽位 %s_%d 的 best.json 引用了 %s 的 run_id"
                         % (r["regime"], r["seed"], "%s_%d" % (o["regime"], o["seed"])))
            if o.get("state_dir") and o["state_dir"] in txt:
                v.append("槽位 %s_%d 的 best.json 引用了 %s 的 state_dir"
                         % (r["regime"], r["seed"], "%s_%d" % (o["regime"], o["seed"])))
        st = r.get("state_run_id")
        if st and r.get("run_id") and st != r["run_id"]:
            v.append("槽位 %s_%d 的 state.run_id(%s) != 权威 run_id(%s)（`N-8` 连续性）"
                     % (r["regime"], r["seed"], st, r["run_id"]))
    return v


def _cell(v):
    """渲染用：**仅 `None` / 空串**记 `—`。

    ⚠ **`0` / `0.0` 是合法值，不得显示为 `—`** —— 曾因写 `v or "—"` 使
    `FPR = 0.0`（完美）在汇总报告里被渲染成缺失（W5.3 收口复核发现；CSV 侧正确）。
    """
    return "—" if v is None or v == "" else str(v)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean(xs):
    return sum(xs) / len(xs)


def _std_sample(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _std_pop(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def aggregate(rows):
    """按体制聚合 `val F1`（**一律给逐 seed 原值**，`D-47` 裁定四）。

    `std` 同时给 **样本 std（ddof=1，报告主用）** 与 **总体 std（ddof=0，并列）**，
    ⇒ 与「mean±std 的 std 是哪一种」无关，读法无歧义。
    """
    out = {}
    for regime in REGIMES:
        rs = [r for r in rows if r["regime"] == regime and r["status"] != "MISSING"]
        vals = [_f(r["val_f1"]) for r in rs]
        vals = [v for v in vals if v is not None]
        agg = {"n": len(vals), "raw": [(r["seed"], r["val_f1"]) for r in rs],
               "mean": None, "std_sample": None, "std_pop": None, "degenerate": False}
        if vals:
            agg["mean"] = _mean(vals)
            agg["std_sample"] = _std_sample(vals)
            agg["std_pop"] = _std_pop(vals)
            agg["degenerate"] = len(set(vals)) == 1 and len(vals) > 1
        out[regime] = agg
    return out


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
    return path


def write_curves_summary(path, rows, cfg):
    """训练曲线汇总（逐 run 的 curve CSV 拼接，附 `regime`／`seed` 列）。"""
    cols = ["regime", "seed", "run_id", "epoch", "opt_steps", "train_loss",
            "val_f1", "val_tpr", "val_fpr", "tau_ours", "early_stop_counter"]
    out_rows, missing = [], []
    for r in rows:
        if r["status"] == "MISSING" or not r.get("curve_csv"):
            continue
        p = _abspath(r["curve_csv"])
        if not os.path.isfile(p):
            missing.append(r["curve_csv"])
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                rec2 = {"regime": r["regime"], "seed": r["seed"], "run_id": r["run_id"]}
                rec2.update({c: rec.get(c, "") for c in cols[3:]})
                out_rows.append(rec2)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for rec in out_rows:
            w.writerow({c: rec.get(c, "") for c in cols})
    return path, missing


def write_summary_md(path, rows, agg, args, cfg, logger, executed, violations,
                     curves_csv, curves_missing, ts):
    """汇总报告：**mean ± std ＋ 逐 seed 原值表**（`D-47` 裁定四）＋ `segments` 披露。"""
    L = []
    add = L.append
    add("# 三 seeds 独立训练汇总（W5.3）")
    add("")
    add("- 生成时间（ts）：`%s`" % ts)
    add("- 配置：`%s`" % _posix(args.config))
    add("- 汇总表（逐 run）：`%s`" % _posix(args.summary_csv))
    add("- 训练曲线汇总：`%s`%s" % (_posix(curves_csv),
                                    "（缺失 %d 份曲线）" % len(curves_missing)
                                    if curves_missing else ""))
    add("- 调用方式：**子进程**（理由见脚本 docstring；与 §2.2 手工命令同源）")
    add("- 执行通道：**一次一次跑**（`windows/W5.3_three_seeds.md` §1.1；默认通道 (a) 本机终端 ＋ (b) 可续跑）")
    add("")
    add("> **`segments` 披露（`D-48` 裁定六）**：单次训练**可分 *k* 段续跑**，且已通过等价性自证"
        "（`W5.1`，三量 Δ=0 ＋ curve／adapter md5 逐字节相同 ＋ 续跑目录 == 首段 `run_id` 目录）。"
        "**不得**表述为「单次连续训练」；**总 GPU 时长不变 ≈20–52.5 h**。")
    add("")
    add("## 1. 逐 run 结果表")
    add("")
    add("| regime | seed | status | run_id | val F1 | TPR | FPR | τ_main | τ*（全精度） | epochs | best_step | n_seg | segments | state_dir | 权威产物目录 | 峰值显存(MiB) |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        add("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` | %s |"
            % (r["regime"], r["seed"], r["status"], _cell(r["run_id"]),
               _cell(r["val_f1"]), _cell(r["val_tpr"]), _cell(r["val_fpr"]),
               _cell(r["tau_main"]), _cell(r["tau_star"]), _cell(r["epochs"]),
               _cell(r["best_step"]), r["n_segments"] if r["status"] != "MISSING" else "—",
               _cell(r["segments"]), _cell(r["state_dir"]), _cell(r["run_dir"]),
               _cell(r["peak_memory_mib"])))
    add("")
    add("## 2. 按体制 mean ± std（`val F1`，主量）")
    add("")
    add("> `std` 主用 **样本 std（`ddof=1`）**；**总体 std（`ddof=0`）并列给出** ⇒ 读法无歧义。")
    add("")
    add("| regime | n | mean val F1 | std（ddof=1） | std（ddof=0） | std 退化 |")
    add("|---|---|---|---|---|---|")
    for regime in REGIMES:
        a = agg[regime]
        if a["mean"] is None:
            add("| %s | 0 | — | — | — | — |" % regime)
            continue
        add("| %s | %d | %.6f | %.6f | %.6f | %s |"
            % (regime, a["n"], a["mean"], a["std_sample"], a["std_pop"],
               "**是（见 §4）**" if a["degenerate"] else "否"))
    add("")
    add("## 3. 逐 seed 原值表（`D-47` 裁定四：**一律给**）")
    add("")
    add("| regime | seed | val F1（原值） | τ_main | τ*（全精度） | TPR | FPR |")
    add("|---|---|---|---|---|---|---|")
    for r in rows:
        add("| %s | %s | %s | %s | %s | %s | %s |"
            % (r["regime"], r["seed"], _cell(r["val_f1"]), _cell(r["tau_main"]),
               _cell(r["tau_star"]), _cell(r["val_tpr"]), _cell(r["val_fpr"])))
    add("")
    add("## 4. `std` 退化声明（**事前定值，`D-47` 裁定四**）")
    add("")
    deg = [g for g in REGIMES if agg[g]["degenerate"]]
    if deg:
        for g in deg:
            add("- **`%s`**：3 seeds 的 `val F1` **全相等**且非四舍五入所致 ⇒ "
                "**声明 `std` 退化**；以上报**逐 seed 原值**为准，**不得**仅读作 `mean ± 0`。" % g)
    else:
        add("- 本体制组内未观测到 `val F1` 全相等 ⇒ **`std` 未退化**，`mean ± std` 可正常读。")
    add("")
    add("## 5. τ 口径（`D-47` 裁定二；`TRAIN_CONFIG_Ours.md` §7.6）")
    add("")
    add("- **主口径 `τ_main = 0.5`（定值）**，触发规则事前写定于 `threshold.main_criterion_override`"
        "（边界解 且 负类 score max 距 τ* < 1e-3，`logic = AND`），**实测触发**；")
    add("- `τ*` **降级为并列报告**；登记与复算**一律取全精度**（内存变量／`meta.json`），"
        "**禁用**日志或曲线的显示值复算（`F-W5.1-2`）；")
    add("- 下游 **`W6`（A6）在 test 上以 `0.5` 为主口径**，并并列报 `τ*`。")
    add("")
    add("## 6. 实际执行顺序与续跑段数（§7.8 四.6；本窗口登记义务）")
    add("")
    if executed:
        add("本会话实际执行顺序（`04_03` 调度）：")
        add("")
        for i, e in enumerate(executed, 1):
            add("%d. `%s` seed=%d → exit=%d，`wall_seconds=%.3f`" % (i, e["regime"], e["seed"],
                                                                     e["exit"], e["wall"]))
    else:
        add("本会话**未执行训练**（`--dry-run`／`--summary-only`，或由用户按 §2.2 手工跑）。"
            "**实际执行顺序须由执行者在登记行写明**。")
    add("")
    add("逐 run `segments`（续跑段数）：")
    add("")
    for r in rows:
        if r["status"] == "MISSING":
            continue
        add("- `%s` seed=%s：`segments` = `%s`" % (r["regime"], r["seed"], r["segments"]))
    add("")
    add("## 7. 独立性与禁止覆盖断言（验收 §5-2／§5-3）")
    add("")
    if violations:
        add("**未通过** —— 违规 %d 条：" % len(violations))
        add("")
        for x in violations:
            add("- %s" % x)
    else:
        add("- ✅ 6 个 `state` 目录路径互不相同；")
        add("- ✅ 每个 `best.json` 的 `dir` 均落在**本槽位**的 `run_id` 目录内（无跨 seed 续接标记）；")
        add("- ✅ `best.json` 文本不含其他槽位的 `run_id`／`state_dir`；")
        add("- ✅ `state.latest/state.json` 的 `run_id` == 该槽位的权威 `run_id`（`N-8` 目录连续性）。")
    add("")
    add("## 8. 命令源（6 条实跑命令，§2.2 模板；`--curriculum on` ＋ `--resume <本 run state 目录>` 必带）")
    add("")
    for regime, seed in expand_runs(REGIMES, SEEDS):
        add("```bash")
        add(build_command_line(args, regime, seed))
        add("```")
    add("")
    add("## 9. 未完成／告警")
    add("")
    miss = [r for r in rows if r["status"] == "MISSING"]
    if miss:
        add("- **未完成槽位 %d 个**：%s（`status=MISSING`，**如实标注、不代填**）"
            % (len(miss), "、".join("%s_%d" % (r["regime"], r["seed"]) for r in miss)))
    amb = [r for r in rows if r["status"] == "AMBIGUOUS"]
    if amb:
        add("- ⚠ **多候选目录 %d 个**：%s（`N-8` 修复后应恒为 1；存在历史分裂残留，"
            "已取最新，须人工核对）"
            % (len(amb), "、".join("%s_%d" % (r["regime"], r["seed"]) for r in amb)))
    if curves_missing:
        add("- ⚠ 曲线缺失：%s" % "、".join(curves_missing))
    if not miss and not amb and not curves_missing:
        add("- 无。")
    add("")
    add("---")
    add("")
    add("> 依据：`windows/W5.3_three_seeds.md` §2.1／§2.2／§2.3／§4／§5；"
        "`configs/train/TRAIN_CONFIG_Ours.md` §7.6／§7.7／§7.8 四；"
        "转交项 #46（`D-50` 裁定四）。**未读** `data/sealed_test/**` 与 frozen 侧 `test.csv`。")
    path_abs = os.path.abspath(path)
    os.makedirs(os.path.dirname(path_abs), exist_ok=True)
    with open(path_abs, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return path_abs


def do_summary(args, cfg, logger, ts, executed):
    rows, records = [], []
    for regime, seed in expand_runs(args.regimes, args.seeds):
        row, extra = collect_slot(args, cfg, regime, seed)
        rows.append(row)
        rec = {"regime": regime, "seed": seed, "status": row["status"],
               "run_id": row["run_id"], "run_dir": row["run_dir"],
               "state_dir": row["state_dir"],
               "state_run_id": extra.get("state_run_id"),
               "best": extra.get("best")}
        records.append(rec)
    violations = check_independence(records)
    agg = aggregate(rows)
    args.summary_csv = _posix(os.path.join(args.report_dir, "%s_%s.csv" % (SUMMARY_STEM, ts)))
    curves_csv = _posix(os.path.join(args.report_dir, "%s_%s.csv" % (CURVES_STEM, ts)))
    write_summary_csv(_abspath(args.summary_csv), rows)
    curves_csv, curves_missing = write_curves_summary(_abspath(curves_csv), rows, cfg)
    args.summary_md = _posix(os.path.join(args.report_dir, "%s_%s.md" % (SUMMARY_STEM, ts)))
    md = write_summary_md(_abspath(args.summary_md), rows, agg, args, cfg, logger, executed,
                          violations, _posix(os.path.relpath(curves_csv, _ROOT)),
                          curves_missing, ts)
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_missing = sum(1 for r in rows if r["status"] == "MISSING")
    logger("SUMMARY_CSV %s" % _posix(os.path.relpath(_abspath(args.summary_csv), _ROOT)))
    logger("SUMMARY_MD %s" % _posix(os.path.relpath(md, _ROOT)))
    logger("CURVES_SUMMARY %s" % _posix(os.path.relpath(curves_csv, _ROOT)))
    logger("SUMMARY_OK=%d MISSING=%d AMBIGUOUS=%d INDEPENDENCE_VIOLATIONS=%d"
           % (n_ok, n_missing,
              sum(1 for r in rows if r["status"] == "AMBIGUOUS"), len(violations)))
    for x in violations:
        logger("INDEPENDENCE_VIOLATION %s" % x)
    return rows, violations, md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="W5.3 三 seeds 独立训练调度器／命令源（调用 04_01，不实现训练循环）")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--regimes", default=",".join(REGIMES),
                   help="逗号分隔；默认 id,ood（顺序即建议执行顺序）")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                   help="逗号分隔；默认 42,123,456")
    p.add_argument("--only", default=None, metavar="<regime>:<seed>",
                   help="只跑指定的一个 run（默认通道「一次一次跑」）")
    p.add_argument("--resume-root", dest="resume_root", default=DEFAULT_RESUME_ROOT,
                   help="state 目录根；每 run 派生 <regime>_seed<seed>/（6 个互不复用）")
    p.add_argument("--report-dir", dest="report_dir", default=DEFAULT_REPORT_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--log-dir", dest="log_dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--python", default=sys.executable,
                   help="调起 04_01 的解释器（默认 = 本进程解释器）")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="只打印命令（含每 run state 目录 ＋ --curriculum on）与 run_id 预览，不训练")
    p.add_argument("--summary-only", dest="summary_only", action="store_true",
                   help="不训练，只按现存产物出汇总（供「用户手写 6 条命令」时履行汇总职责）")
    return p


def parse_only(arg):
    if ":" not in str(arg):
        raise ValueError("--only 须形如 <regime>:<seed>，收到 %r" % arg)
    r, s = str(arg).split(":", 1)
    regime = r.strip()
    if regime not in REGIMES:
        raise ValueError("--only 的 regime 非法：%r（允许 %s）" % (regime, list(REGIMES)))
    seed = int(s.strip())
    if seed not in SEEDS:
        raise ValueError("--only 的 seed 非法：%r（允许 %s）" % (seed, list(SEEDS)))
    return regime, seed


def main(argv=None) -> int:
    init_streams()
    args = build_parser().parse_args(argv)
    cfg = load_config(_abspath(args.config))
    ts = timestamp()
    logger = Logger()
    os.makedirs(_abspath(args.report_dir), exist_ok=True)
    os.makedirs(_abspath(args.log_dir), exist_ok=True)
    abs_cfg = _abspath(args.config)
    rel_cfg = _posix(os.path.relpath(abs_cfg, _ROOT))
    if rel_cfg.startswith(".."):
        rel_cfg = _posix(abs_cfg)
        logger("INFO config 在项目根之外，命令源按**绝对路径**打印：%s" % rel_cfg)
    args.config = rel_cfg
    log_path = _abspath(os.path.join(args.log_dir, "04_03_run_three_seeds_%s.log" % ts))

    try:
        args.regimes = parse_list(args.regimes, str, REGIMES, "--regimes")
        args.seeds = parse_list(args.seeds, int, SEEDS, "--seeds")
    except ValueError as exc:
        logger("ARG_ERROR %s" % exc)
        logger.dump(log_path)
        return 2

    logger("04_03_run_three_seeds ts=%s config=%s regimes=%s seeds=%s dry_run=%s "
           "summary_only=%s only=%s"
           % (ts, args.config, ",".join(args.regimes), ",".join(str(s) for s in args.seeds),
              args.dry_run, args.summary_only, args.only))

    runs = expand_runs(args.regimes, args.seeds)
    if args.only:
        try:
            one = parse_only(args.only)
        except (ValueError, TypeError) as exc:
            logger("ARG_ERROR %s" % exc)
            logger.dump(log_path)
            return 2
        if one not in runs:
            logger("ARG_ERROR --only %s:%d 不在本次范围 %s 内" % (one[0], one[1], runs))
            logger.dump(log_path)
            return 2
        runs = [one]

    if args.dry_run:
        code = do_dry_run(args, cfg, logger, runs)
        logger.dump(log_path)
        return code

    executed, failed = [], []
    if not args.summary_only:
        for regime, seed in runs:
            sd = _abspath(state_dir_for(args.resume_root, regime, seed))
            os.makedirs(sd, exist_ok=True)   # 首跑也传 --resume（§2.3 ①）；空目录 ⇒ RESUME_STATE_INIT
            code, wall, log_p = execute_run(args, regime, seed, ts, logger)
            executed.append({"regime": regime, "seed": seed, "exit": code, "wall": wall})
            if code != 0:
                failed.append((regime, seed, code))
                logger("RUN_FAILED regime=%s seed=%d exit=%d（**保留**已完成的 run 与其日志，"
                       "不回滚删除；state 可续跑）" % (regime, seed, code))
    else:
        logger("SUMMARY_ONLY 不训练，只按现存产物出汇总")

    rows, violations, md = do_summary(args, cfg, logger, ts, executed)

    if failed:
        logger("RESULT=PARTIAL_FAIL n_failed=%d" % len(failed))
        logger.dump(log_path)
        return 1
    logger("RESULT=OK runs=%d summary=%s"
           % (len(executed) if executed else 0, _posix(os.path.relpath(md, _ROOT))))
    logger.dump(log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

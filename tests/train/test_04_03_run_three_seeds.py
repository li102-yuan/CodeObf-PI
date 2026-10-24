# -*- coding: utf-8 -*-
"""tests/train/test_04_03_run_three_seeds.py — W5.3 `04_03` 调度器／命令源的单元与接口测试。

覆盖 `windows/W5.3_three_seeds.md` §2.1／§2.2／§2.3／§5（验收 2／3／5／6）与
转交项 **#46**（`D-50` 裁定四）四项：

1. **`--dry-run` 命令序列**：恒 6 条；每条含 `--curriculum on`（`N-7` 教训）与 `-u`；
   每条 `--resume` 指向**该 run 专属**的 state 目录，且 6 个 state 目录**互不相同**；
   `run_id` 预览格式 = `AGENTS.md` §4.5；
2. **`--only <regime>:<seed>`**：只出／只跑指定的一个 run；
3. **per-run `--resume` 透传**：`build_argv` 恒带 `--resume`（首跑也带，§2.3 ①）；
4. **汇总**：逐 run 行含 `segments`／state 目录／**权威产物目录**；
   `std` 退化须**声明**并附**逐 seed 原值**（`D-47` 裁定四）。

另覆盖独立性／禁止覆盖断言（验收 2）与红线（不读 `sealed_test`、不实现训练循环）。

**不加载模型、不需要 GPU**（子进程只跑 `--dry-run`／`--summary-only`）。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
SCHED = os.path.join(ROOT, "scripts", "train", "04_03_run_three_seeds.py")
CONFIG = os.path.join(ROOT, "configs", "train", "qlora_ours.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load("t_04_03", SCHED)


@pytest.fixture(scope="module")
def cfg(m):
    return m.load_config(CONFIG)


def _run_cli(log_dir, *argv):
    """子进程跑调度器；`--log-dir` 一律指向 pytest 的 tmp ⇒ **不污染** `logs/train/`。"""
    proc = subprocess.run([sys.executable, "-u", SCHED, "--log-dir", log_dir, *argv],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


@pytest.fixture(scope="module")
def dry_out(tmp_path_factory):
    code, out = _run_cli(str(tmp_path_factory.mktemp("dryrun")),
                         "--config", CONFIG, "--dry-run")
    assert code == 0, out
    return out


@pytest.fixture(scope="module")
def dry_lines(dry_out):
    return [ln.strip()[len("cmd       : "):] for ln in dry_out.splitlines()
            if ln.strip().startswith("cmd       : ")]


# ---------------------------------------------------------------------------
# 1. `--dry-run`
# ---------------------------------------------------------------------------
def test_dry_run_exit_zero_and_pass(dry_out):
    assert "DRYRUN_RESULT=PASS n=6 state_dirs=6" in dry_out


def test_dry_run_prints_exactly_six_commands(dry_lines):
    assert len(dry_lines) == 6


def test_dry_run_covers_two_regimes_three_seeds(dry_lines):
    got = []
    for ln in dry_lines:
        toks = ln.split()
        got.append((toks[toks.index("--regime") + 1], int(toks[toks.index("--seed") + 1])))
    assert got == [("id", 42), ("id", 123), ("id", 456),
                   ("ood", 42), ("ood", 123), ("ood", 456)]


def test_dry_run_every_command_has_curriculum_on(dry_lines):
    """`N-7`：`04_01 --curriculum` 默认 `off`，漏了就课程与良性子采样全部不生效。"""
    for ln in dry_lines:
        assert "--curriculum on" in ln, ln


def test_dry_run_every_command_has_unbuffered_flag(dry_lines):
    for ln in dry_lines:
        assert " -u " in ln, ln


def test_dry_run_every_command_has_resume_pointing_at_own_state_dir(m, dry_lines):
    seen = []
    for ln in dry_lines:
        toks = ln.split()
        assert "--resume" in toks, ln
        sd = toks[toks.index("--resume") + 1]
        regime = toks[toks.index("--regime") + 1]
        seed = toks[toks.index("--seed") + 1]
        assert sd == m.state_dir_for(m.DEFAULT_RESUME_ROOT, regime, int(seed)), ln
        seen.append(sd)
    assert len(set(seen)) == 6, seen           # §2.3 ②：6 个互不复用


def test_dry_run_interpreter_is_quoted_forward_slash(m, dry_lines):
    """bash 里未加引号的 `C:\\Users\\…` 会把 `\\U` 当转义 ⇒ 命令撕坏。

    §2.2 模板原文 = `"C:/Users/lipei/miniconda3/envs/deepseek/python.exe"`（正斜杠 ＋ 引号）。
    """
    for ln in dry_lines:
        head = ln.split("&&", 1)[1].strip()
        assert head.startswith('"'), ln
        interp = head.split('"')[1]
        assert "\\" not in interp, ln
        assert interp.endswith("python.exe"), ln


def test_dry_run_state_dirs_distinct(dry_out):
    assert "STATE_DIRS_DISTINCT True（6 个）" in dry_out


def test_dry_run_run_id_preview_matches_agents_format(m, cfg):
    for regime in ("id", "ood"):
        for seed in (42, 123, 456):
            rid = m.run_id_preview(cfg, regime, seed)
            assert rid == "04_01_llamaguard1b_%s_seed%d_<YYYYMMDD-HHMMSS>" % (regime, seed)


def test_dry_run_only_yields_one_command(tmp_path):
    code, out = _run_cli(str(tmp_path), "--config", CONFIG, "--dry-run", "--only", "ood:123")
    assert code == 0, out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("cmd       : ")]
    assert len(lines) == 1, out
    assert "--regime ood" in lines[0] and "--seed 123" in lines[0]
    assert "--resume checkpoints/ours/state/ood_seed123" in lines[0]


def test_dry_run_regimes_seeds_subset(tmp_path):
    code, out = _run_cli(str(tmp_path), "--config", CONFIG, "--dry-run",
                         "--regimes", "ood", "--seeds", "42,456")
    assert code == 0, out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("cmd       : ")]
    assert len(lines) == 2
    assert "state/ood_seed42" in lines[0] and "state/ood_seed456" in lines[1]


def test_dry_run_rejects_out_of_range_seed(tmp_path):
    code, out = _run_cli(str(tmp_path), "--config", CONFIG, "--dry-run", "--seeds", "7")
    assert code == 2
    assert "ARG_ERROR" in out


def test_dry_run_only_rejects_malformed(m):
    for bad in ("id", "id:7", "xx:42", "id:42:1"):
        with pytest.raises((ValueError, TypeError)):
            m.parse_only(bad)


# ---------------------------------------------------------------------------
# 2. run 展开 / 命名
# ---------------------------------------------------------------------------
def test_expand_runs_default_order(m):
    assert m.expand_runs(("id", "ood"), (42, 123, 456)) == [
        ("id", 42), ("id", 123), ("id", 456), ("ood", 42), ("ood", 123), ("ood", 456)]


def test_state_dir_for_six_unique(m):
    sds = [m.state_dir_for(m.DEFAULT_RESUME_ROOT, r, s)
           for r, s in m.expand_runs(("id", "ood"), (42, 123, 456))]
    assert len(set(sds)) == 6
    assert sds[0] == "checkpoints/ours/state/id_seed42"
    assert sds[-1] == "checkpoints/ours/state/ood_seed456"


def test_build_argv_always_carries_resume_and_curriculum(m):
    args = m.build_parser().parse_args(["--config", m.DEFAULT_CONFIG])
    for regime, seed in m.expand_runs(("id", "ood"), (42, 123, 456)):
        argv = m.build_argv(args, regime, seed)
        assert "--resume" in argv
        assert argv[argv.index("--resume") + 1] == m.state_dir_for(m.DEFAULT_RESUME_ROOT,
                                                                   regime, seed)
        assert argv[argv.index("--curriculum") + 1] == "on"
        assert argv[1] == "-u"


def test_run_glob_matches_run_id_format(m, cfg):
    pat = m.run_glob(m.DEFAULT_OUT_DIR, cfg, "id", 42)
    assert pat.endswith(os.path.join("checkpoints", "ours", "state")) is False
    base = os.path.basename(pat)
    assert base.startswith("04_01_llamaguard1b_id_seed42_") and base.endswith("*")


# ---------------------------------------------------------------------------
# 3. 独立性 / 禁止覆盖（验收 §5-2）
# ---------------------------------------------------------------------------
def _rec(regime, seed, run_dir, state_dir, run_id, best=None, state_run_id=None,
         status="OK"):
    return {"regime": regime, "seed": seed, "status": status, "run_id": run_id,
            "run_dir": run_dir, "state_dir": state_dir,
            "state_run_id": state_run_id, "best": best or {}}


def test_check_independence_clean(m):
    recs = []
    for regime in ("id", "ood"):
        for seed in (42, 123, 456):
            rid = "04_01_llamaguard1b_%s_seed%d_20261005-120000" % (regime, seed)
            rd = "checkpoints/ours/" + rid
            recs.append(_rec(regime, seed, rd, m.state_dir_for(m.DEFAULT_RESUME_ROOT,
                                                               regime, seed), rid,
                             best={"dir": os.path.join(ROOT, rd, "seed_%d" % seed,
                                                       "checkpoint-1184")},
                             state_run_id=rid))
    assert m.check_independence(recs) == []


def test_check_independence_detects_shared_state_dir(m):
    recs = [
        _rec("id", 42, "checkpoints/ours/a", "checkpoints/ours/state/id_seed42", "a",
             state_run_id="a"),
        _rec("id", 123, "checkpoints/ours/b", "checkpoints/ours/state/id_seed42", "b",
             state_run_id="b"),
    ]
    v = m.check_independence(recs)
    assert any("state_dir 重复" in x for x in v), v


def test_check_independence_detects_cross_run_reference(m):
    """`best.json` 文本含**其他槽位**的 `run_id` ⇒ 判为跨 seed 续接标记。"""
    rid42 = "04_01_llamaguard1b_id_seed42_20261005-120000"
    rid123 = "04_01_llamaguard1b_id_seed123_20261005-120000"
    recs = [
        _rec("id", 42, "checkpoints/ours/" + rid42,
             "checkpoints/ours/state/id_seed42", rid42,
             best={"dir": os.path.join(ROOT, "checkpoints/ours", rid42, "seed_42",
                                       "checkpoint-1184"), "resumed_from": rid123},
             state_run_id=rid42),
        _rec("id", 123, "checkpoints/ours/" + rid123,
             "checkpoints/ours/state/id_seed123", rid123,
             best={"dir": os.path.join(ROOT, "checkpoints/ours", rid123, "seed_123",
                                       "checkpoint-1184")},
             state_run_id=rid123),
    ]
    v = m.check_independence(recs)
    assert any("run_id" in x for x in v), v


def test_check_independence_detects_best_dir_out_of_slot(m):
    rid42 = "04_01_llamaguard1b_id_seed42_20261005-120000"
    rid123 = "04_01_llamaguard1b_id_seed123_20261005-120000"
    recs = [
        _rec("id", 42, "checkpoints/ours/" + rid42,
             "checkpoints/ours/state/id_seed42", rid42,
             best={"dir": os.path.join(ROOT, "checkpoints/ours", rid123, "seed_123",
                                       "checkpoint-1184")},
             state_run_id=rid42),
    ]
    v = m.check_independence(recs)
    assert any("越界" in x for x in v), v


def test_check_independence_detects_state_run_id_mismatch(m):
    """`N-8` 修复后的目录连续性判据：state 的 `run_id` 必须 == 权威 `run_id`。"""
    rid = "04_01_llamaguard1b_id_seed42_20261005-120000"
    recs = [_rec("id", 42, "checkpoints/ours/" + rid, "checkpoints/ours/state/id_seed42",
                 rid, best={"dir": os.path.join(ROOT, "checkpoints/ours", rid, "seed_42",
                                                "checkpoint-1184")},
                 state_run_id="04_01_llamaguard1b_id_seed42_20261005-119999")]
    v = m.check_independence(recs)
    assert any("N-8" in x for x in v), v


def test_check_independence_skips_missing_slots(m):
    recs = [_rec("id", 42, "", "checkpoints/ours/state/id_seed42", "", status="MISSING"),
            _rec("ood", 42, "", "checkpoints/ours/state/ood_seed42", "", status="MISSING")]
    assert m.check_independence(recs) == []


def test_check_independence_detects_duplicate_slot(m):
    recs = [_rec("id", 42, "checkpoints/ours/a", "s/a", "a", state_run_id="a"),
            _rec("id", 42, "checkpoints/ours/b", "s/b", "b", status="MISSING")]
    assert any("槽位重复" in x for x in m.check_independence(recs))


# ---------------------------------------------------------------------------
# 4. 汇总（合成产物；不训练）
# ---------------------------------------------------------------------------
def _mk_run(tmp, m, cfg, regime, seed, f1s, taus=None, n_segments=1, state_run_id=None):
    """造一个合成 run 目录：`best.json` ＋ `curve_*.csv`（可选 `state/latest/state.json`）。"""
    ts = "20261005-1200%02d" % (seed % 100)
    rid = m.run_id_preview(cfg, regime, seed, ts=ts)
    run_dir = os.path.join(tmp, "out", rid)
    ckpt = os.path.join(run_dir, "seed_%d" % seed, "checkpoint-1184")
    os.makedirs(ckpt, exist_ok=True)
    segs = [{"start_step": 0, "end_step": 1184, "wall_seconds": 100.0 * (i + 1)}
            for i in range(n_segments)]
    curve = os.path.join(tmp, "rep", "curve_%s_%d_v1.0_%s.csv" % (regime, seed, ts))
    os.makedirs(os.path.dirname(curve), exist_ok=True)
    with open(curve, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["epoch", "opt_steps", "train_loss", "val_f1", "val_tpr", "val_fpr",
                    "tau_ours", "early_stop_counter"])
        for i, f1 in enumerate(f1s, 1):
            w.writerow([i, 1184 * i, 0.5 / i, f1, 1.0, 0.0, 0.5, 0])
    best = {"regime": regime, "seed": seed, "step": 1184 * len(f1s),
            "val_f1": f1s[-1], "epoch": len(f1s),
            "tpr": 1.0, "fpr": 0.0, "tau": (taus or {}).get("main", 0.5),
            "tau_ours": (taus or {}).get("main", 0.5),
            "tau_star": (taus or {}).get("star", 0.99999862909317017),
            "tau_main_override": True,
            "curve_csv": os.path.relpath(curve, ROOT).replace("\\", "/"),
            "segments": segs, "wall_seconds": 100.0 * n_segments,
            "peak_memory_mib": 7270.0, "epochs_run": len(f1s),
            "opt_steps": 1184 * len(f1s), "early_stopped": True,
            "dir": ckpt, "threshold_csv": ""}
    with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(best, fh, ensure_ascii=False, indent=2, sort_keys=True)
    if state_run_id is not None:
        sd = os.path.join(tmp, "state", "%s_seed%d" % (regime, seed), "latest")
        os.makedirs(sd, exist_ok=True)
        with open(os.path.join(sd, "state.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"run_id": state_run_id, "ts": ts}, fh)
    return rid


def _run_summary(m, tmp, extra=()):
    argv = ["--config", CONFIG, "--summary-only",
            "--out-dir", os.path.join(tmp, "out"),
            "--report-dir", os.path.join(tmp, "rep"),
            "--resume-root", os.path.join(tmp, "state"),
            "--log-dir", os.path.join(tmp, "log")]
    return m.main(list(argv) + list(extra))


def _rows_of(md_path):
    csv_path = md_path.replace(".md", ".csv")
    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_summary_synthetic_full(m, tmp_path):
    for regime in ("id", "ood"):
        for seed in (42, 123, 456):
            ts = "20261005-1200%02d" % (seed % 100)
            rid = m.run_id_preview(cfg_of(m), regime, seed, ts=ts)
            _mk_run(str(tmp_path), m, cfg_of(m), regime, seed, [0.90, 0.95, 0.98],
                    n_segments=2, state_run_id=rid)
    code = _run_summary(m, str(tmp_path))
    assert code == 0
    mds = [f for f in os.listdir(os.path.join(tmp_path, "rep")) if f.endswith(".md")]
    assert len(mds) == 1, mds
    md = os.path.join(tmp_path, "rep", mds[0])
    rows = _rows_of(md)
    assert len(rows) == 6
    for r in rows:
        assert r["status"] == "OK"
        assert r["segments"].startswith("n=2;")
        assert r["n_segments"] == "2"
        assert r["state_dir"].endswith("%s_seed%s" % (r["regime"], r["seed"]))
        assert r["run_id"] in r["run_dir"], r
        assert r["run_dir"] != ""
        assert r["tau_main"] == "0.5"
        assert abs(float(r["tau_star"]) - 0.99999862909317017) < 1e-15
    text = open(md, encoding="utf-8").read()
    assert "逐 seed 原值表" in text
    assert "权威产物目录" in text
    assert "实际执行顺序" in text
    assert "segments" in text
    # 曲线汇总必须含 6 个 run × 3 epoch = 18 行
    curves = [f for f in os.listdir(os.path.join(tmp_path, "rep"))
              if f.startswith("curves_summary")]
    with open(os.path.join(tmp_path, "rep", curves[0]), encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 18


def cfg_of(m):
    return m.load_config(CONFIG)


def test_summary_marks_missing_slots(m, tmp_path):
    _mk_run(str(tmp_path), m, cfg_of(m), "id", 42, [0.9], n_segments=1)
    code = _run_summary(m, str(tmp_path))
    assert code == 0
    mds = [f for f in os.listdir(os.path.join(tmp_path, "rep")) if f.endswith(".md")]
    rows = _rows_of(os.path.join(tmp_path, "rep", mds[0]))
    by = {(r["regime"], r["seed"]): r["status"] for r in rows}
    assert by[("id", "42")] == "OK"
    assert sum(1 for v in by.values() if v == "MISSING") == 5
    text = open(os.path.join(tmp_path, "rep", mds[0]), encoding="utf-8").read()
    assert "未完成槽位" in text


def test_summary_std_degeneration_declared(m, tmp_path):
    """3 seeds 的 `val F1` 全相等 ⇒ 必须**声明 `std` 退化**（`D-47` 裁定四），
    且**禁止**只写 `mean ± std = x ± 0`。"""
    for seed in (42, 123, 456):
        _mk_run(str(tmp_path), m, cfg_of(m), "id", seed, [1.0], n_segments=1)
    code = _run_summary(m, str(tmp_path))
    assert code == 0
    mds = [f for f in os.listdir(os.path.join(tmp_path, "rep")) if f.endswith(".md")]
    text = open(os.path.join(tmp_path, "rep", mds[0]), encoding="utf-8").read()
    assert "std" in text and "退化" in text
    assert "**是（见 §4）**" in text
    assert "声明 `std` 退化" in text
    assert "逐 seed 原值" in text


def test_cell_renders_zero_not_dash(m):
    """回归：`0` / `0.0` 是**合法值**，不得渲染为 `—`。

    收口复核实测缺陷 —— 原实现写 `r["val_fpr"] or "—"`，`0.0` 为 falsy ⇒
    「FPR = 0.0（完美）」在汇总报告里被显示成缺失（CSV 侧一直是对的）。
    """
    assert m._cell(0.0) == "0.0"
    assert m._cell(0) == "0"
    assert m._cell(1.0) == "1.0"
    assert m._cell(None) == "—"
    assert m._cell("") == "—"


def test_summary_md_renders_zero_fpr(m, tmp_path):
    """端到端：合成 run 的 `fpr = 0.0` 必须在 §1／§3 表中显示为 `0.0`。"""
    _mk_run(str(tmp_path), m, cfg_of(m), "id", 42, [0.9, 1.0], n_segments=1)
    assert _run_summary(m, str(tmp_path)) == 0
    mds = [f for f in os.listdir(os.path.join(tmp_path, "rep")) if f.endswith(".md")]
    text = open(os.path.join(tmp_path, "rep", mds[0]), encoding="utf-8").read()
    sec3 = text.split("## 3. ")[1].split("## 4. ")[0]
    row = [ln for ln in sec3.splitlines() if ln.startswith("| id | 42 ")][0]
    assert row.rstrip().endswith("| 1.0 | 0.0 |"), row


def test_summary_aggregate_arithmetic(m):
    rows = [{"regime": "id", "seed": s, "status": "OK", "val_f1": v}
            for s, v in ((42, 0.9), (123, 0.95), (456, 1.0))]
    agg = m.aggregate(rows)["id"]
    assert abs(agg["mean"] - 0.95) < 1e-12
    assert abs(agg["std_sample"] - 0.05) < 1e-12          # ddof=1
    assert abs(agg["std_pop"] - 0.05 / (1.5 ** 0.5)) < 1e-12
    assert agg["degenerate"] is False
    empty = m.aggregate([{"regime": "ood", "seed": 42, "status": "MISSING", "val_f1": ""}])
    assert empty["ood"]["mean"] is None


def test_summary_curves_merges_per_run_rows(m, tmp_path):
    _mk_run(str(tmp_path), m, cfg_of(m), "id", 42, [0.9, 0.95], n_segments=1)
    _run_summary(m, str(tmp_path))
    f = [x for x in os.listdir(os.path.join(tmp_path, "rep"))
         if x.startswith("curves_summary")][0]
    with open(os.path.join(tmp_path, "rep", f), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all(r["regime"] == "id" and r["seed"] == "42" for r in rows)


# ---------------------------------------------------------------------------
# 5. 红线（静态扫描）
# ---------------------------------------------------------------------------
def test_scheduler_does_not_implement_training_loop(m):
    src = open(SCHED, encoding="utf-8").read()
    assert "import torch" not in src
    assert "def train_one_run" not in src
    assert "save_checkpoint" not in src


def test_scheduler_never_reads_sealed_test(m):
    """A5 红线：脚本**的文件系统调用**不得触碰 `sealed_test`／frozen 侧 `test.csv`。

    判据 = 只查**文件系统调用的实参**（含 `os.path.join` 的分量），**不查散文串** ——
    报告脚注里写「**未读** `data/sealed_test/**`」是**合规披露**，把披露本身判成违规
    属过滤条件过窄（`WINDOW_LOG.md` §I-6 的 5 次误报同源）。
    """
    import ast
    fs_funcs = {"open", "glob", "listdir", "isfile", "isdir", "exists", "getmtime",
                "walk", "join", "exists"}
    tree = ast.parse(open(SCHED, encoding="utf-8").read())
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name not in fs_funcs:
            continue
        for a in ast.walk(node):
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                if "sealed_test" in a.value or "test.csv" in a.value:
                    hits.append((name, a.value))
    assert hits == [], hits


def test_scheduler_uses_subprocess(m):
    src = open(SCHED, encoding="utf-8").read()
    assert "subprocess.Popen" in src
    assert "cwd=_ROOT" in src


def test_scheduler_never_deletes_run_dirs(m):
    """退出码非零时须**保留**已完成的 run 与其日志（§2.1「退出码」行）。"""
    src = open(SCHED, encoding="utf-8").read()
    assert "rmtree" not in src
    assert "os.remove" not in src


def test_scheduler_does_not_override_epochs_or_lr(m):
    """§7.8 四.4 红线：不得借「分段」改 `epochs`／`lr`／`r`／`patience`／样本集。"""
    args = m.build_parser().parse_args(["--config", m.DEFAULT_CONFIG])
    argv = m.build_argv(args, "id", 42)
    for flag in ("--epochs", "--lr", "--r", "--patience", "--max-steps"):
        assert flag not in argv


def test_summary_columns_cover_transfer_item_46(m):
    for col in ("segments", "state_dir", "run_dir", "tau_main", "tau_star"):
        assert col in m.CSV_COLUMNS

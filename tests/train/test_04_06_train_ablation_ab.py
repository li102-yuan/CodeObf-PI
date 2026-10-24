# -*- coding: utf-8 -*-
"""tests/train/test_04_06_train_ablation_ab.py — W5.6 `04_06` 消融 A／B 调度器测试。

覆盖 `windows/W5.6_ablation_ab_train.md` §6（验收 1／2／3／4／5／6）：

1. **`--dry-run` 命令序列**：恒 6 条 ≈ 默认范围；每条含 **本组 `--augment`**、
   **`--curriculum off` 且无 `--curriculum on`**（`N-7` 教训）、**本组 `--model-key`**、`-u`；
   每条 `--resume` 指向**该 run 专属** state 目录，且 6 个状态目录互不相同；
   `run_id` 预览 = `AGENTS.md` §4.5 形态且含 `<AB>`（§4）。
2. **`--only <AB>:<seed>`**：只出／只跑指定的一个 run。
3. **红线自检**（`validate_groups`）：A／B 恒 `curriculum=off`；A=`augment off`、B=`augment on`；
   两组 `model_key` 互不相同。
4. **独立性／禁止覆盖断言**（§6-2）：含 **`alias_guard`** —— 槽位 `run_id` 形如 C 组
   （`llamaguard1b_<regime>_`）即判违规（`D-83` 裁定三 禁止借用 C 组 ckpt）。
5. **汇总**：逐 run 行含 `augment`／`curriculum`／`segments`／state 目录／权威产物目录；
   `std` 退化须**声明**并附**逐 seed 原值**；报告含 **§0 可比性声明**。
6. **与 `04_01` 的接口一致性**：`04_01` 的 parser 确实提供 `--augment`／`--model-key` 且默认值
   不改变既有行为（`--augment` 默认 `off`、`--model-key` 默认 `None`）。

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
SCHED = os.path.join(ROOT, "scripts", "train", "04_06_train_ablation_ab.py")
TRAINER = os.path.join(ROOT, "scripts", "train", "04_01_train_qlora_ours.py")
CONFIG = os.path.join(ROOT, "configs", "train", "ablation_ab.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load("t_04_06", SCHED)


@pytest.fixture(scope="module")
def trainer():
    return _load("t_04_01_for_04_06", TRAINER)


@pytest.fixture(scope="module")
def cfg(m):
    return m.load_config(CONFIG)


def _run_cli(log_dir, report_dir, *argv):
    """子进程跑调度器；`--log-dir`／`--report-dir` 指向 pytest tmp ⇒ **不污染**生产目录。"""
    proc = subprocess.run([sys.executable, "-u", SCHED,
                           "--log-dir", log_dir, "--report-dir", report_dir, *argv],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


@pytest.fixture(scope="module")
def dry_out(tmp_path_factory):
    d = tmp_path_factory.mktemp("dryrun")
    code, out = _run_cli(str(d / "logs"), str(d / "rep"), "--config", CONFIG, "--dry-run")
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


def test_dry_run_covers_two_groups_three_seeds(m, dry_lines):
    got = []
    for ln in dry_lines:
        toks = ln.split()
        got.append((toks[toks.index("--model-key") + 1], int(toks[toks.index("--seed") + 1])))
    assert got == [("llamaguard1b_A", 42), ("llamaguard1b_A", 123), ("llamaguard1b_A", 456),
                   ("llamaguard1b_B", 42), ("llamaguard1b_B", 123), ("llamaguard1b_B", 456)]


def test_dry_run_every_command_has_curriculum_off_and_never_on(dry_lines):
    """红线（`N-7` 教训）：A／B 恒 `--curriculum off`；**不得**出现 `--curriculum on`。"""
    for ln in dry_lines:
        assert "--curriculum off" in ln, ln
        assert "--curriculum on" not in ln, ln


def test_dry_run_augment_matches_group(m, dry_lines):
    want = {"llamaguard1b_A": "off", "llamaguard1b_B": "on"}
    for ln in dry_lines:
        toks = ln.split()
        mk = toks[toks.index("--model-key") + 1]
        aug = toks[toks.index("--augment") + 1]
        assert aug == want[mk], ln


def test_dry_run_every_command_has_unbuffered_flag(dry_lines):
    for ln in dry_lines:
        assert " -u " in ln, ln


def test_dry_run_every_command_has_resume_pointing_at_own_state_dir(m, dry_lines):
    seen = []
    for ln in dry_lines:
        toks = ln.split()
        assert "--resume" in toks, ln
        sd = toks[toks.index("--resume") + 1]
        mk = toks[toks.index("--model-key") + 1]
        group = mk.rsplit("_", 1)[-1]                     # llamaguard1b_A → A
        seed = toks[toks.index("--seed") + 1]
        assert sd == m.state_dir_for(m.DEFAULT_RESUME_ROOT, group, int(seed)), ln
        seen.append(sd)
    assert len(set(seen)) == 6, seen                       # §2.4 ②：6 个互不复用


def test_dry_run_state_dirs_distinct(dry_out):
    assert "STATE_DIRS_DISTINCT True（6 个）" in dry_out


def test_dry_run_run_id_preview_matches_agents_format(m, cfg):
    """§4：`run_id` = `04_01_llamaguard1b_<AB>_id_seed<seed>_<ts>`（`{model}` 槽载 `<AB>`）。"""
    for group in m.GROUPS:
        rid = m.run_id_preview(cfg, group, 42, ts="20261009-160000")
        assert rid == "04_01_llamaguard1b_%s_id_seed42_20261009-160000" % group, rid


def test_run_id_never_looks_like_c_group(m, cfg):
    """`D-83` 裁定三：A／B 的 `run_id` **不得**形如 C 组（`llamaguard1b_<regime>_`）。"""
    for group in m.GROUPS:
        rid = m.run_id_preview(cfg, group, 42, ts="20261009-160000")
        assert "llamaguard1b_id_" not in rid and "llamaguard1b_ood_" not in rid, rid


def test_dry_run_only_yields_one_command(tmp_path):
    code, out = _run_cli(str(tmp_path / "l"), str(tmp_path / "r"),
                         "--config", CONFIG, "--dry-run", "--only", "B:123")
    assert code == 0, out
    cmds = [ln for ln in out.splitlines() if ln.strip().startswith("cmd       : ")]
    assert len(cmds) == 1, out
    assert "--model-key llamaguard1b_B" in cmds[0] and "--seed 123" in cmds[0]
    assert "DRYRUN_RESULT=PASS n=1" in out


def test_dry_run_groups_seeds_subset(tmp_path):
    code, out = _run_cli(str(tmp_path / "l"), str(tmp_path / "r"),
                         "--config", CONFIG, "--dry-run", "--groups", "A", "--seeds", "456")
    assert code == 0, out
    cmds = [ln for ln in out.splitlines() if ln.strip().startswith("cmd       : ")]
    assert len(cmds) == 1 and "--seed 456" in cmds[0]


def test_dry_run_rejects_out_of_range_seed(tmp_path):
    code, out = _run_cli(str(tmp_path / "l"), str(tmp_path / "r"),
                         "--config", CONFIG, "--dry-run", "--seeds", "42,999")
    assert code == 2, out
    assert "ARG_ERROR" in out


def test_dry_run_only_rejects_malformed(m):
    with pytest.raises(ValueError):
        m.parse_only("A")            # 缺 `:seed`
    with pytest.raises(ValueError):
        m.parse_only("Z:42")         # 组非法
    with pytest.raises(ValueError):
        m.parse_only("A:999")        # seed 非法


# ---------------------------------------------------------------------------
# 2. 展开 / 命名 / argv
# ---------------------------------------------------------------------------
def test_expand_runs_default_order(m):
    """§2.3 建议顺序：`A(42→123→456)` → `B(42→123→456)`。"""
    assert m.expand_runs(m.GROUPS, m.SEEDS) == [
        ("A", 42), ("A", 123), ("A", 456), ("B", 42), ("B", 123), ("B", 456)]


def test_state_dir_for_naming_and_six_unique(m):
    sds = [m.state_dir_for("checkpoints/ours/state", g, s)
           for g in m.GROUPS for s in m.SEEDS]
    assert len(set(sds)) == 6
    assert "checkpoints/ours/state/A_id_seed42" in sds
    assert "checkpoints/ours/state/B_id_seed456" in sds


def test_build_argv_carries_augment_curriculum_off_model_key_resume(m, cfg):
    class A:  # 轻量 args 替身（只含 build_argv 用到的字段）
        python = sys.executable
        config = CONFIG
        report_dir = m.DEFAULT_REPORT_DIR
        out_dir = m.DEFAULT_OUT_DIR
        log_dir = m.DEFAULT_LOG_DIR
        resume_root = m.DEFAULT_RESUME_ROOT
    argv_a = m.build_argv(A, cfg, "A", 42)
    argv_b = m.build_argv(A, cfg, "B", 42)
    assert "--curriculum" in argv_a and argv_a[argv_a.index("--curriculum") + 1] == "off"
    assert argv_a[argv_a.index("--augment") + 1] == "off"
    assert argv_b[argv_b.index("--augment") + 1] == "on"
    assert argv_a[argv_a.index("--model-key") + 1] == "llamaguard1b_A"
    assert argv_b[argv_b.index("--model-key") + 1] == "llamaguard1b_B"
    assert argv_a[argv_a.index("--resume") + 1] == "checkpoints/ours/state/A_id_seed42"


def test_run_glob_matches_run_id_format(m, cfg):
    pat = m.run_glob("/tmp/out", cfg, "B", 42)
    assert pat.endswith("04_01_llamaguard1b_B_id_seed42_*")


# ---------------------------------------------------------------------------
# 3. 红线自检（`validate_groups`）
# ---------------------------------------------------------------------------
def test_validate_groups_clean(m, cfg):
    assert m.validate_groups(cfg) == []


def test_validate_groups_flags_curriculum_on(m, cfg):
    import copy
    bad = copy.deepcopy(cfg)
    bad["ablation_ab"]["groups"]["B"]["curriculum"] = "on"
    v = m.validate_groups(bad)
    assert any("curriculum" in x for x in v), v


def test_validate_groups_flags_wrong_augment(m, cfg):
    import copy
    bad = copy.deepcopy(cfg)
    bad["ablation_ab"]["groups"]["A"]["augment"] = "on"
    bad["ablation_ab"]["groups"]["B"]["augment"] = "off"
    v = m.validate_groups(bad)
    assert any("augment" in x for x in v), v


def test_validate_groups_flags_duplicate_model_key(m, cfg):
    import copy
    bad = copy.deepcopy(cfg)
    bad["ablation_ab"]["groups"]["B"]["model_key"] = "llamaguard1b_A"
    v = m.validate_groups(bad)
    assert any("model_key" in x for x in v), v


def test_cli_rejects_redline_config(tmp_path, m):
    """端到端：把红线破坏的配置落盘 ⇒ 调度器以 exit=2 拒绝（**不训练**）。"""
    import copy
    bad = copy.deepcopy(m.load_config(CONFIG))
    bad["ablation_ab"]["groups"]["B"]["curriculum"] = "on"
    p = tmp_path / "bad.yaml"
    import yaml
    p.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
    code, out = _run_cli(str(tmp_path / "l"), str(tmp_path / "r"),
                         "--config", str(p), "--dry-run")
    assert code == 2, out
    assert "GROUP_REDLINE_VIOLATION" in out


# ---------------------------------------------------------------------------
# 4. 独立性 / 禁止覆盖 / alias_guard
# ---------------------------------------------------------------------------
def _rec(group, seed, run_dir, state_dir, run_id, best=None, state_run_id=None,
         status="OK"):
    return {"group": group, "seed": seed, "status": status, "run_dir": run_dir,
            "state_dir": state_dir, "run_id": run_id, "best": best,
            "state_run_id": state_run_id}


def _clean_recs(m, cfg):
    recs = []
    for g in m.GROUPS:
        for s in m.SEEDS:
            rid = m.run_id_preview(cfg, g, s, ts="20261009-1600%02d" % (s % 100))
            rd = "checkpoints/ours/" + rid
            recs.append(_rec(g, s, rd, m.state_dir_for("checkpoints/ours/state", g, s), rid,
                             best={"dir": os.path.join(ROOT, rd, "seed_%d" % s,
                                                       "checkpoint-1184")},
                             state_run_id=rid))
    return recs


def test_check_independence_clean(m, cfg):
    assert m.check_independence(_clean_recs(m, cfg)) == []


def test_check_independence_flags_c_group_alias(m, cfg):
    """`D-83` 裁定三：借用 C 组 ckpt（`llamaguard1b_id_` 形态）⇒ 判违规。"""
    rid_c = "04_01_llamaguard1b_id_seed42_20261009-160000"
    recs = [_rec("A", 42, "checkpoints/ours/" + rid_c,
                 "checkpoints/ours/state/A_id_seed42", rid_c,
                 best={"dir": os.path.join(ROOT, "checkpoints/ours", rid_c, "seed_42",
                                           "checkpoint-1184")},
                 state_run_id=rid_c)]
    v = m.check_independence(recs)
    assert any("C 组" in x or "借用" in x for x in v), v


def test_check_independence_flags_shared_state_dir(m, cfg):
    recs = _clean_recs(m, cfg)
    recs[1]["state_dir"] = recs[0]["state_dir"]
    assert any("state_dir" in x for x in m.check_independence(recs))


def test_check_independence_flags_cross_run_reference(m, cfg):
    recs = _clean_recs(m, cfg)
    recs[0]["best"] = {"dir": os.path.join(ROOT, recs[0]["run_dir"], "seed_42",
                                           "checkpoint-1184"),
                       "resumed_from": recs[1]["run_id"]}
    assert any("run_id" in x for x in m.check_independence(recs))


def test_check_independence_flags_best_dir_out_of_slot(m, cfg):
    recs = _clean_recs(m, cfg)
    recs[0]["best"]["dir"] = os.path.join(ROOT, recs[1]["run_dir"], "seed_123",
                                          "checkpoint-1184")
    assert any("越界" in x for x in m.check_independence(recs))


def test_check_independence_flags_state_run_id_mismatch(m, cfg):
    recs = _clean_recs(m, cfg)
    recs[0]["state_run_id"] = "04_01_llamaguard1b_A_id_seed42_20261009-159999"
    assert any("N-8" in x for x in m.check_independence(recs))


def test_check_independence_skips_missing(m):
    recs = [_rec("A", 42, "", "checkpoints/ours/state/A_id_seed42", "", status="MISSING"),
            _rec("B", 42, "", "checkpoints/ours/state/B_id_seed42", "", status="MISSING")]
    assert m.check_independence(recs) == []


# ---------------------------------------------------------------------------
# 5. 汇总（合成产物；不训练）
# ---------------------------------------------------------------------------
def _mk_run(tmp, m, cfg, group, seed, f1s, n_segments=1, state_run_id=None):
    ts = "20261009-1600%02d" % (seed % 100)
    rid = m.run_id_preview(cfg, group, seed, ts=ts)
    run_dir = os.path.join(tmp, "out", rid)
    ckpt = os.path.join(run_dir, "seed_%d" % seed, "checkpoint-1184")
    os.makedirs(ckpt, exist_ok=True)
    segs = [{"start_step": 0, "end_step": 1184, "wall_seconds": 100.0 * (i + 1)}
            for i in range(n_segments)]
    best = {"regime": "id", "seed": seed, "step": 1184 * len(f1s), "val_f1": f1s[-1],
            "epoch": len(f1s), "tpr": 1.0, "fpr": 0.0,
            "tau_ours": 0.9999485611915588, "tau_star": 0.9999485611915588,
            "tau_main_override": False, "curve_csv": "",
            "segments": segs, "wall_seconds": 100.0 * n_segments,
            "peak_memory_mib": 7270.0, "epochs_run": len(f1s),
            "opt_steps": 1184 * len(f1s), "early_stopped": True, "dir": ckpt}
    with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(best, fh, ensure_ascii=False, indent=2, sort_keys=True)
    if state_run_id is not None:
        sd = os.path.join(tmp, "state", "%s_id_seed%d" % (group, seed), "latest")
        os.makedirs(sd, exist_ok=True)
        with open(os.path.join(sd, "state.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"run_id": state_run_id, "ts": ts}, fh)
    return rid


def _summary(m, cfg, tmp, f1_by_group):
    out = tmp / "out"
    rep = tmp / "rep"
    state = tmp / "state"
    for g, vals in f1_by_group.items():
        for i, seed in enumerate(m.SEEDS):
            _mk_run(str(tmp), m, cfg, g, seed, [vals[i]])   # 逐 seed 一个「逐 epoch」序列
    code, txt = _run_cli(str(tmp / "l"), str(rep), "--config", CONFIG,
                         "--summary-only", "--out-dir", str(out),
                         "--resume-root", str(state))
    assert code == 0, txt
    csvs = sorted(rep.glob("*ablation_ab_v1.0_*.csv"))
    mds = sorted(rep.glob("*ablation_ab_v1.0_*.md"))
    assert csvs and mds, txt
    rows = list(csv.DictReader(open(csvs[-1], encoding="utf-8")))
    md = mds[-1].read_text(encoding="utf-8")
    return rows, md, txt


def test_summary_marks_pre_training_skeleton_as_non_result(m, cfg, tmp_path):
    """`D-84` 裁定六 2②：6 槽位**全** `MISSING` ⇒ 文件加 `PRE_TRAINING_` 前缀 ＋ `md` 首部 banner。"""
    rows, md, txt = _summary(m, cfg, tmp_path, {})
    assert all(r["status"] == "MISSING" for r in rows), rows
    assert "PRE_TRAINING_SKELETON（非结果；`D-84` 裁定六 2②）" in txt
    assert "**非结果**" in md and "训练前骨架" in md
    # 文件名自标
    assert any(p.name.startswith("PRE_TRAINING_ablation_ab_v1.0_")
               for p in (tmp_path / "rep").glob("*.md")), \
        sorted(p.name for p in (tmp_path / "rep").glob("*"))
    assert any(p.name.startswith("PRE_TRAINING_ablation_ab_v1.0_")
               for p in (tmp_path / "rep").glob("*.csv"))


def test_summary_discloses_reading_one_as_merged_term(m, cfg, tmp_path):
    """`D-84` 裁定一 条件 2／3：须明写「两项合并为一项」且**不得**声称与 v5 字面完全一致。"""
    _, md, _ = _summary(m, cfg, tmp_path, {"A": [0.9, 0.9, 0.9]})
    assert "两项" in md and "合并为一项" in md
    assert "不得**声称" in md or "**不得**声称" in md
    assert "`W5.6` 窗口提案" in md and "`D-84` 裁定一" in md


def test_summary_synthetic_full(m, cfg, tmp_path):
    rows, md, _ = _summary(m, cfg, tmp_path,
                           {"A": [0.97, 0.96, 0.98], "B": [0.99, 0.98, 0.97]})
    assert len(rows) == 6
    assert {r["group"] for r in rows} == {"A", "B"}
    assert all(r["status"] == "OK" for r in rows), rows
    # 汇总列覆盖 §4（含 segments／state_dir／run_dir／tau／augment／curriculum）
    for col in ("augment", "curriculum", "model_key", "n_segments", "segments",
                "state_dir", "run_dir", "tau_main", "tau_star", "val_f1"):
        assert col in rows[0], col
    assert "## 0. 可比性声明" in md
    assert "普通随机增强" in md


def test_summary_marks_missing_slots(m, cfg, tmp_path):
    rows, md, _ = _summary(m, cfg, tmp_path, {"A": [0.97, 0.96, 0.98]})
    missing = [r for r in rows if r["status"] == "MISSING"]
    assert len(missing) == 3, rows
    assert "未完成槽位 3 个" in md


def test_summary_std_degeneration_declared(m, cfg, tmp_path):
    """`D-47` 裁定四：3 seeds 的 `val F1` 全相等 ⇒ **必须**声明 `std` 退化。"""
    rows, md, _ = _summary(m, cfg, tmp_path, {"A": [1.0, 1.0, 1.0]})
    assert "声明 `std` 退化" in md
    assert "逐 seed 原值" in md
    a_rows = [r for r in rows if r["group"] == "A"]
    assert [r["val_f1"] for r in a_rows] == ["1.0", "1.0", "1.0"]


def test_summary_md_carries_six_commands(m, cfg, tmp_path):
    _, md, _ = _summary(m, cfg, tmp_path, {"A": [0.9, 0.9, 0.9]})
    # §8 的 6 条实跑命令：均 `--curriculum off`、均 `--augment`。（§0 表格里描述 C 组时
    # 合法地出现 `--curriculum on`，故只对**命令行**断言。）
    cmd_lines = [ln for ln in md.splitlines()
                 if ln.startswith("cd ") and "--resume" in ln]
    assert len(cmd_lines) == 6, cmd_lines
    for ln in cmd_lines:
        assert "--curriculum off" in ln and "--curriculum on" not in ln, ln
    assert "--augment on" in md and "--augment off" in md


def test_events_file_exists_and_is_loaded(m, cfg, tmp_path):
    """`D-87`：已知事件载体须存在并被加载 ⇒ 汇总 §10 如实披露（不只报成功）。"""
    assert os.path.isfile(os.path.join(ROOT, "configs/train/ablation_ab_events.yaml")), \
        "缺 configs/train/ablation_ab_events.yaml（`D-87` 的如实登记载体）"
    _, md, txt = _summary(m, cfg, tmp_path, {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]})
    assert "EVENTS_LOADED n=1" in txt, txt
    assert "## 10. 已知事件" in md


def test_events_discloses_a456_first_nan(m, cfg, tmp_path):
    """`D-87`：**不得**只报重跑成功 ⇒ 首次 NaN 的 ts／检测／处置／重跑须全部在汇总里。"""
    _, md, _ = _summary(m, cfg, tmp_path, {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]})
    assert "[A:456]" in md
    assert "20261010-023548" in md          # 首次尝试（NaN）的 ts
    assert "NAN_GUARD_TRIGGERED" in md and "step=214" in md
    assert "无效 run" in md and "重跑" in md
    assert "20261010-075719" in md          # 重跑（本汇总所载者）的 ts
    assert "不得" in md and "6/6 成功" in md  # 明写「不得只报 6/6 成功」


def test_events_absent_yields_explicit_none(m, cfg, tmp_path):
    """事件载体缺失 ⇒ 汇总显式写「无」（而非静默编造／留空）。"""
    out = tmp_path / "r2"
    proc = subprocess.run([sys.executable, "-u", SCHED,
                           "--log-dir", str(tmp_path / "l2"), "--report-dir", str(out),
                           "--out-dir", str(tmp_path / "o2"), "--resume-root", str(tmp_path / "s2"),
                           "--config", CONFIG, "--summary-only",
                           "--events-file", str(tmp_path / "nope.yaml")],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout
    assert "EVENTS_LOADED n=0" in proc.stdout, proc.stdout
    md = sorted(out.glob("*ablation_ab_v1.0_*.md"))[-1].read_text(encoding="utf-8")
    assert "无（本汇总所载的权威 run 均**一次成功**）" in md


def test_cell_renders_zero_not_dash(m):
    assert m._cell(0.0) == "0.0"
    assert m._cell(0) == "0"
    assert m._cell(None) == "—"
    assert m._cell("") == "—"


def test_aggregate_arithmetic(m):
    rows = [{"group": "A", "seed": 42, "status": "OK", "val_f1": 0.9},
            {"group": "A", "seed": 123, "status": "OK", "val_f1": 0.8},
            {"group": "A", "seed": 456, "status": "OK", "val_f1": 1.0},
            {"group": "B", "seed": 42, "status": "MISSING", "val_f1": ""}]
    agg = m.aggregate(rows)
    assert abs(agg["A"]["mean"] - 0.9) < 1e-12
    assert abs(agg["A"]["std_sample"] - 0.1) < 1e-12
    assert agg["A"]["degenerate"] is False
    assert agg["B"]["n"] == 0 and agg["B"]["mean"] is None


# ---------------------------------------------------------------------------
# 6. 红线／接口一致性
# ---------------------------------------------------------------------------
def test_scheduler_does_not_implement_training_loop(m):
    src = open(SCHED, encoding="utf-8").read()
    for bad in ("import torch", "transformers", "peft", "bitsandbytes",
                "optimizer.step(", "backward()", "AdamW"):
        assert bad not in src, "调度器不得出现 %r（只调用 04_01）" % bad


def test_scheduler_never_reads_sealed_test(m):
    src = open(SCHED, encoding="utf-8").read()
    # 只允许出现在文档/注释中；不得出现任何"读"sealed 的代码形态
    for bad in ("open(", "load_config(", "csv.DictReader(", "glob.glob("):
        for i, line in enumerate(src.splitlines(), 1):
            if bad in line and "sealed" in line.lower():
                pytest.fail("第 %d 行疑似读 sealed_test：%s" % (i, line))


def test_scheduler_uses_subprocess(m):
    src = open(SCHED, encoding="utf-8").read()
    assert "subprocess.Popen" in src


def test_scheduler_does_not_override_epochs_or_lr(m, cfg):
    """§3-3 红线：调度器**不得**另设 `epochs`／`lr`／`r`／`patience`／样本集。"""
    class A:
        python = sys.executable
        config = CONFIG
        report_dir = m.DEFAULT_REPORT_DIR
        out_dir = m.DEFAULT_OUT_DIR
        log_dir = m.DEFAULT_LOG_DIR
        resume_root = m.DEFAULT_RESUME_ROOT
    argv = m.build_argv(A, cfg, "A", 42)
    for flag in ("--epochs", "--max-steps", "--lr", "--r", "--patience", "--max-length"):
        assert flag not in argv, argv


def test_trainer_exposes_augment_and_model_key(trainer):
    """P-4 接口一致性：`04_01` 提供 `--augment`／`--model-key`，默认值不改既有行为。"""
    args = trainer.build_parser().parse_args(
        ["--regime", "id", "--seed", "42"])
    assert args.augment == "off"
    assert args.model_key is None
    assert args.curriculum == "off"


def test_trainer_model_key_override_is_opt_in(trainer):
    cfg = trainer.load_config(CONFIG)
    assert trainer.apply_model_key_override(cfg, None) is None
    assert cfg["checkpoint"]["model_key"] == "llamaguard1b"     # 未改动
    assert trainer.apply_model_key_override(cfg, "llamaguard1b_A") == "llamaguard1b_A"
    assert cfg["checkpoint"]["model_key"] == "llamaguard1b_A"


def test_trainer_run_id_uses_overridden_model_key(trainer):
    cfg = trainer.load_config(CONFIG)
    trainer.apply_model_key_override(cfg, "llamaguard1b_B")
    assert trainer.run_id_of(cfg, "id", 42, "20261009-160000") == \
        "04_01_llamaguard1b_B_id_seed42_20261009-160000"


def test_trainer_augment_gate_preserves_curriculum_on_semantics(trainer):
    """§2.2 红线：`--curriculum on` 的既有语义**逐字不变**（增强仍由 `cur_cls` 分支触发）。"""
    import inspect
    src = inspect.getsource(trainer.run_train)
    assert "if cur_cls is not None:" in src
    assert "elif augment_on:" in src

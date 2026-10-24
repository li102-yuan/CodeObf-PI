# -*- coding: utf-8 -*-
"""02_08_freeze_dataset.py — W3.8 数据集冻结（CodeObf-PI v1.0）

接口契约（`windows/W3.8_freeze.md` §3）::

    python scripts/data/02_08_freeze_dataset.py --dataset CodeObf-PI_v1.0

（`make freeze DATASET=CodeObf-PI_v1.0` 即调用此行。）

**本脚本是唯一有权写 `data/frozen/` 与 `data/sealed_test/` 的脚本**（`windows/W3_stage2_data.md` §4-5）。

## 冻结前置 `H1`–`H7`（**代码强制，缺一即非零退出**；`windows/W3.8_freeze.md` §3.1）

| # | 前置 | 判据（**读文件实算，不采信自述**） |
|---|---|---|
| `H1` | 门禁 1（改写意图一致性 ≥95%） | `reports/data/g4_intent_v1.0_*.md` 存在且通过率 ≥0.95 |
| `H2` | 门禁 2（AST 报错率 ≤10%） | `ast_report.csv` 实算 `ast_failed/(ast_ok+ast_failed) ≤ 0.10` |
| `H3` | 门禁 3（G3 质检 ≥95%） | `g3_qc.csv` 含四字段、**两位标注者**、双人通过率 ≥0.95 |
| `H4` | 门禁 4（良性集检出率 ≤5%） | `benign_guard_check.csv` 实算检出率 ≤0.05 |
| `H5` | 门禁 5／6（门禁 7 为**报告项**，不参与） | **重跑 `02_07` 默认模式**，`exit=0` 且 `A1`–`A13` 全 `True`、`split_assertions.csv` **14 行** |
| `H6` | `02_00 --full` 已通过 | `reports/data/smoke_data_v1.0_*.md` 中存在 `mode: full` 且结论为通过者 |
| `H7` | 目标目录可写且为**占位态**或不存在 | 见 `is_placeholder()` |

**「先检查后写入」是硬要求**：`H1`–`H7` **全部**通过后，才开始写任何文件。
任一不满足 ⇒ **非零退出**、打印具体缺口、**不做任何写入**。

> `H5` 的副作用说明：它按 `windows/W3.8_freeze.md` §3.1 的要求**重跑 `02_07` 默认模式**，
> 因而会重写 `data/interim/split_checks/` 下 5 个产物。本脚本在这 5 个文件**前后各取 sha256**，
> **不一致即非零退出**（`02_07` 已被证明幂等：`W3.7` 补正轮 `r2` 复跑 5/5 逐字节相同）。

## 冻结产出

- `data/frozen/CodeObf-PI_v1.0/**`（结构逐条对齐 `DIRECTORY.md` §3.3）
- `data/sealed_test/CodeObf-PI_v1.0/**`（4 个测试分片 + `md5sums.txt`；`ACCESS_LOG.md` 只校验表头）
- 两份 `md5sums.txt`（UTF-8 + LF，`<md5>  <相对路径>`，「相对各自的根目录」，**不自指**）
- 运行报告 `reports/data/freeze_v1.0_<YYYYMMDD-HHMMSS>.md`

## 禁止项（`windows/W3.8_freeze.md` §6）

- 禁止在任一 `H` 未过时冻结；禁止把检查放到写入之后。
- 禁止为让冻结跑通而放宽阈值、剔除不利样本或改口径。
- 禁止覆盖已冻结的 `CodeObf-PI_v1.0`（后续修订只能新建 `v1.1`）。
- 禁止把训练/验证分片写进 `data/sealed_test/`。
- 禁止写 `ACCESS_LOG.md` 的任何数据行（**只校验其表头恰 6 列**）。
- 禁止训练／微调／加载模型。
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

REGIMES = ("regime_id", "regime_ood")
SPLITS = ("train", "val", "test")

#: 通过率的真值集合（`g3_qc.csv` 的 `pass` 列；各窗口沿用的写法是 `yes`）
TRUTHY = {"yes", "y", "true", "1", "pass", "ok", "通过"}

#: `H5` 重跑 `02_07` 前后须逐字节不变的 5 个产物（相对 `data/interim/split_checks/`）
H5_PRODUCTS = ("split_assertions.csv", "length_distribution.csv", "ngram_overlap.csv",
               "ngram_controls.csv", "family_straddle.csv")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """stdout／stderr 重配 UTF-8（本机中文 Windows 默认 GBK；`W3.7` `r2` 的 `F-1` 同类预防）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def _rel(path: str, root: str = REPO_ROOT) -> str:
    try:
        return os.path.relpath(path, root).replace("\\", "/")
    except ValueError:  # pragma: no cover
        return path.replace("\\", "/")


def _md5_file(path: str) -> str:
    d = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def _sha256_file(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def _read_csv(path: str) -> "list[dict]":
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _count_rows(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for _ in reader:
            n += 1
    return n


def _load_yaml(path: str):
    try:
        import yaml
    except ImportError:  # pragma: no cover
        print("FATAL: 需要 PyYAML 解析 %s" % _rel(path))
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_text(path: str, text: str) -> None:
    """一律 UTF-8 + LF 落盘（`V-9`）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _copy_file(src: str, dst: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copyfile(src, dst)
    return os.path.getsize(dst)


def _git_head(root: str) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, encoding="utf-8", errors="replace")
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - 无 git 时如实降级
        pass
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# 占位态
# ---------------------------------------------------------------------------
def is_placeholder(root: str, cfg: dict) -> bool:
    """占位态判据（`windows/W3.8_freeze.md` §0）：

    ① `md5sums.txt` 为 **0 字节** **且** ② `splits/regime_id/train.csv` 为 **0 字节**。
    两条同时成立才是占位态（仓库自初始化起即存在这套空占位文件）。
    """
    fdir = os.path.join(root, cfg["outputs"]["frozen_dir"])
    a = os.path.join(fdir, "md5sums.txt")
    b = os.path.join(fdir, "splits", "regime_id", "train.csv")
    return (os.path.isfile(a) and os.path.getsize(a) == 0
            and os.path.isfile(b) and os.path.getsize(b) == 0)


def _dir_nonempty(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


# ---------------------------------------------------------------------------
# H1–H7
# ---------------------------------------------------------------------------
def check_h1(root: str, cfg: dict):
    """门禁 1：改写意图一致性 ≥95%。判据 = 最新的 `g4_intent_v1.0_*.md` 中的通过率。"""
    rdir = os.path.join(root, cfg["outputs"]["report_dir"])
    pat = re.compile(r"^g4_intent_v1\.0_.*\.md$")
    found = sorted(f for f in os.listdir(rdir)) if os.path.isdir(rdir) else []
    found = [f for f in found if pat.match(f)]
    if not found:
        return False, "未找到 reports/data/g4_intent_v1.0_*.md", {}
    newest = found[-1]
    text = open(os.path.join(rdir, newest), encoding="utf-8").read()
    # 表行口径：`| **通过率** | 1.0000 |`；取**最小值**（保守：任何一处低值即判不达标）
    rates = [float(m) for m in re.findall(r"\|\s*\*\*通过率\*\*\s*\|\s*([0-9]*\.?[0-9]+)\s*\|",
                                          text)]
    if not rates:
        rates = [float(m) for m in re.findall(r"通过率[^0-9\n]{0,24}?([01]\.\d{4})", text)]
    if not rates:
        return False, "%s 中未解析到通过率" % newest, {"report": newest}
    rate = min(rates)
    thr = float(cfg["gates"]["g1_intent"]["threshold"])
    return (rate >= thr, "报告 %s：通过率 %.4f（阈值 %.2f，取该文件内最小值）"
            % (newest, rate, thr), {"report": newest, "rate": rate, "threshold": thr})


def check_h2(root: str, cfg: dict):
    """门禁 2：AST 报错率 ≤10%（实算 `ast_failed/(ast_ok+ast_failed)`）。"""
    path = os.path.join(root, cfg["inputs"]["ast_report"])
    if not os.path.isfile(path):
        return False, "缺少 %s" % _rel(path), {}
    ok = failed = 0
    for r in _read_csv(path):
        ok += int(r.get("ast_ok") or 0)
        failed += int(r.get("ast_failed") or 0)
    denom = ok + failed
    if denom <= 0:
        return False, "分母为 0（ast_ok+ast_failed=0）", {}
    rate = failed / float(denom)
    thr = float(cfg["gates"]["g2_ast"]["threshold"])
    return (rate <= thr, "ast_failed=%d / (ast_ok+ast_failed)=%d ⇒ 报错率 %.6f（阈值 %.2f）"
            % (failed, denom, rate, thr),
            {"ast_ok": ok, "ast_failed": failed, "denominator": denom,
             "rate": rate, "threshold": thr})


def check_h3(root: str, cfg: dict):
    """门禁 3：G3 质检双人通过率 ≥95%（四字段 + 两位标注者）。"""
    path = os.path.join(root, cfg["inputs"]["g3_qc"])
    if not os.path.isfile(path):
        return False, "缺少 %s" % _rel(path), {}
    rows = _read_csv(path)
    if not rows:
        return False, "g3_qc.csv 为空", {}
    need = set(cfg["gates"]["g3_qc"]["required_fields"])
    have = set(rows[0].keys())
    if not need.issubset(have):
        return False, "缺字段：%s" % sorted(need - have), {"fields": sorted(have)}
    annotators = sorted({r.get("annotator", "") for r in rows})
    if len(annotators) < int(cfg["gates"]["g3_qc"]["min_annotators"]):
        return False, "标注者只有 %d 位（需 ≥%d）" % (len(annotators),
                                                     cfg["gates"]["g3_qc"]["min_annotators"]), {}
    by = collections.defaultdict(dict)
    for r in rows:
        by[r.get("sample_id", "")][r.get("annotator", "")] = str(r.get("pass", "")).strip().lower()
    total = len(by)
    both = sum(1 for v in by.values()
               if v and all(x in TRUTHY for x in v.values()))
    rate = both / float(total) if total else 0.0
    thr = float(cfg["gates"]["g3_qc"]["threshold"])
    return (rate >= thr,
            "标注者 %d 位；双人均通过 %d/%d ⇒ %.4f（阈值 %.2f）"
            % (len(annotators), both, total, rate, thr),
            {"annotators": annotators, "samples": total, "both_pass": both,
             "rate": rate, "threshold": thr})


def check_h4(root: str, cfg: dict):
    """门禁 4：良性集检出率 ≤5%（实算 `sum(counted)/n`）。"""
    path = os.path.join(root, cfg["inputs"]["benign_guard_check"])
    if not os.path.isfile(path):
        return False, "缺少 %s" % _rel(path), {}
    rows = _read_csv(path)
    if not rows:
        return False, "benign_guard_check.csv 为空", {}
    counted = 0
    for r in rows:
        v = str(r.get("counted", "")).strip()
        counted += int(float(v)) if v else 0
    rate = counted / float(len(rows))
    thr = float(cfg["gates"]["g4_guard"]["threshold"])
    return (rate <= thr, "counted %d/%d ⇒ %.6f（阈值 %.2f）" % (counted, len(rows), rate, thr),
            {"counted": counted, "n": len(rows), "rate": rate, "threshold": thr})


def check_h5(root: str, cfg: dict, run: bool = True):
    """门禁 5／6：重跑 `02_07` 默认模式，`exit=0` 且 `A1`–`A13` 全 `True`、14 行。

    **不自行重算门禁 5／6／7**（`windows/W3.8_freeze.md` §3.1）：只调用与读取。
    """
    sdir = os.path.join(root, cfg["inputs"]["split_checks"])
    before = {n: (_sha256_file(os.path.join(sdir, n))
                  if os.path.isfile(os.path.join(sdir, n)) else None) for n in H5_PRODUCTS}
    rc = None
    if run:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "02_07_assert_splits.py"),
             "--config", "configs/data/assert.yaml"],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env)
        rc = proc.returncode
        if rc != 0:
            tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:])
            return False, "02_07 默认模式 exit=%d ≠ 0：\n%s" % (rc, tail), {"exit": rc}
    after = {n: (_sha256_file(os.path.join(sdir, n))
                 if os.path.isfile(os.path.join(sdir, n)) else None) for n in H5_PRODUCTS}
    changed = [n for n in H5_PRODUCTS if before[n] != after[n]]

    path = os.path.join(root, cfg["inputs"]["split_assertions"])
    if not os.path.isfile(path):
        return False, "缺少 %s" % _rel(path), {"exit": rc}
    rows = _read_csv(path)
    want = ["A%d" % i for i in range(1, int(cfg["expected"]["assertions_rows"]) + 1)]
    ids = [r.get("assertion_id", "") for r in rows]
    by = {r.get("assertion_id"): r for r in rows}
    gate_ids = ["A%d" % i for i in range(1, 14)]
    bad = [i for i in gate_ids if by.get(i, {}).get("passed") != "True"]
    ok = (ids == want) and not bad and not changed
    detail = ("02_07 exit=%s；split_assertions.csv %d 行 %s；A1–A13 %s；"
              "5 产物 sha256 %s" % (
                  rc, len(rows), "无缺行" if ids == want else "缺/多行：%s" % ids,
                  "全 True" if not bad else "非 True：%s" % bad,
                  "改前改后全等" if not changed else "**有变化：%s**" % changed))
    return ok, detail, {"exit": rc, "rows": len(rows), "bad_assertions": bad,
                        "products_changed": changed,
                        "product_sha256": {k: v[:8] for k, v in after.items() if v}}


def check_h6(root: str, cfg: dict):
    """`02_00 --full` 已通过：存在 `mode: full` 且结论为通过的 `smoke_data_v1.0_*.md`。"""
    rdir = os.path.join(root, cfg["outputs"]["report_dir"])
    pat = re.compile(r"^smoke_data_v1\.0_.*\.md$")
    found = sorted(f for f in (os.listdir(rdir) if os.path.isdir(rdir) else []) if pat.match(f))
    for name in reversed(found):
        text = open(os.path.join(rdir, name), encoding="utf-8").read()
        modes = re.findall(r"模式（`mode`）：`(\w+)`", text)
        concl = re.findall(r"结论（`conclusion`）：`(\w+)`", text)
        if "full" in modes and concl and concl[-1] == "PASS":
            return True, "%s：mode=full、conclusion=PASS" % name, {"report": name}
    if not found:
        return False, "未找到 reports/data/smoke_data_v1.0_*.md（须先跑 02_00 --full）", {}
    return False, "找到 %d 份 smoke_data 报告，但无 mode=full 且结论为通过者" % len(found), {}


#: `V-3` 的事前定值（`stage2_plan` §0；`PREREG_v1.3.md` §3 承接）。**不得改**。
V3_RATIO = "8:1:1"
V3_SEED = 42


def check_v3(cfg: dict):
    """`V-3` 守卫：划分比例 8:1:1、划分种子 42（**事前定值，不得改**）。

    `windows/W3.8_freeze.md` §3.6 第 15 条「W0 补注 4」的回归测试即针对此处：
    构造 `split_seed = 43` 或 `split_ratio = "7:1:2"` ⇒ 非零退出且**未写入任何文件**。
    """
    ratio = str(cfg["expected"]["split_ratio"])
    seed = int(cfg["expected"]["split_seed"])
    if ratio != V3_RATIO or seed != V3_SEED:
        return False, ("`V-3` 被改动：`split_ratio=%r`（须 `%r`）、`split_seed=%r`（须 `%d`）"
                       "—— 划分口径属事前定值，改动须走重大偏离流程"
                       % (ratio, V3_RATIO, seed, V3_SEED))
    return True, "`split_ratio=%s`、`split_seed=%d`（与 `V-3` 一致）" % (ratio, seed)


def check_splits_vs_a13(root: str, cfg: dict):
    """`windows/W3.8_freeze.md` §3.6 第 15 条：`manifest` 的 `splits` 须与 `A13` 一致。

    **在写入前从 `data/interim/`（只读）实算**，不依赖冻结树。
    """
    a_path = os.path.join(root, cfg["inputs"]["split_assertions"])
    g_path = os.path.join(root, cfg["inputs"]["seed_groups"])
    if not (os.path.isfile(a_path) and os.path.isfile(g_path)):
        return False, "缺少 %s 或 %s" % (_rel(a_path, root), _rel(g_path, root)), {}
    by = {r["assertion_id"]: r for r in _read_csv(a_path)}
    measured = by.get("A13", {}).get("measured", "")
    m = re.search(r"train=(\d+)[、,]\s*val=(\d+)[、,]\s*test=(\d+)", measured)
    if not m:
        return False, "无法从 `A13` 的 `measured` 解析三档计数：%r" % measured[:80], {}
    a13 = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
    mine = collections.Counter()
    for r in _read_csv(g_path):
        if r.get("group_kind") == "attack":
            mine[r.get("split", "")] += 1
    got = [mine.get(sp, 0) for sp in SPLITS]
    if a13 != got:
        return False, "`A13` 三档 %s ≠ `seed_groups.csv` 实算 %s" % (a13, got), {"a13": a13}
    return (True, "`A13` 三档 %s == `seed_groups.csv` 实算 %s" % (a13, got),
            {"splits": collections.OrderedDict(zip(SPLITS, got))})


def check_counts(root: str, cfg: dict):
    """`windows/W3.8_freeze.md` §3.3：冻结硬计数 440／7920，且须与 `A6`／`A7` 一致。

    **不得**为让冻结跑通而放宽（`windows/W3.8_freeze.md` §6-2）。
    """
    exp = cfg["expected"]
    paths = {"rewrites": cfg["inputs"]["rewrites"], "encoded": cfg["inputs"]["encoded"],
             "code_carrier": cfg["inputs"]["code_carrier_csv"],
             "split_assertions": cfg["inputs"]["split_assertions"]}
    for label, rel in paths.items():
        if not os.path.isfile(os.path.join(root, rel)):
            return False, "缺少 %s" % rel, {}
    n_rewrites = _count_rows(os.path.join(root, paths["rewrites"]))
    n_attack = (_count_rows(os.path.join(root, paths["encoded"]))
                + _count_rows(os.path.join(root, paths["code_carrier"])))
    by = {r["assertion_id"]: r.get("measured", "")
          for r in _read_csv(os.path.join(root, paths["split_assertions"]))}
    m6 = re.search(r"=\s*(\d+)", by.get("A6", ""))
    m7 = re.search(r"=\s*(\d+)", by.get("A7", ""))
    a6 = int(m6.group(1)) if m6 else None
    a7 = int(m7.group(1)) if m7 else None
    problems = []
    if n_rewrites != int(exp["rewrite_bodies"]):
        problems.append("rewrites.csv 行数 %d ≠ %d" % (n_rewrites, exp["rewrite_bodies"]))
    if n_attack != int(exp["attack_total"]):
        problems.append("攻击样本合计 %d ≠ %d" % (n_attack, exp["attack_total"]))
    if a6 != int(exp["rewrite_bodies"]):
        problems.append("`A6` 实测 %s ≠ %d" % (a6, exp["rewrite_bodies"]))
    if a7 != int(exp["attack_total"]):
        problems.append("`A7` 实测 %s ≠ %d" % (a7, exp["attack_total"]))
    detail = "rewrites=%d、attack_total=%d、`A6`=%s、`A7`=%s" % (n_rewrites, n_attack, a6, a7)
    return (not problems, detail + ("" if not problems else " ⇒ " + "；".join(problems)),
            {"rewrite_bodies": n_rewrites, "attack_total": n_attack, "A6": a6, "A7": a7})


def check_access_log(root: str, cfg: dict):
    """`ACCESS_LOG.md` 表头**恰 6 列**（`DIRECTORY.md` §3.4；`PROJECT_CONTRACT.md` §6 末）。

    **只校验，不写任何数据行**；该检查放在**写入之前**，以免表头不合格时已写出冻结树。
    """
    alog = os.path.join(root, cfg["outputs"]["sealed_dir"], cfg["sealed"]["access_log"])
    if not os.path.isfile(alog):
        return False, "缺少 %s（保留仓库既有表头，不得新建）" % _rel(alog, root), {}
    text = open(alog, encoding="utf-8").read()
    header = [l for l in text.splitlines()
              if l.strip().startswith("|") and "窗口" in l and "复核人" in l]
    if not header:
        return False, "未能识别 ACCESS_LOG.md 的 6 列表头", {}
    cols = [c.strip() for c in header[0].strip().strip("|").split("|")]
    want = int(cfg["sealed"]["access_log_header_cols"])
    if len(cols) != want:
        return False, "表头为 %d 列（须恰 %d 列）：%s" % (len(cols), want, cols), {}
    return True, "表头恰 %d 列；未写任何数据行" % want, {"columns": cols}


def check_h7(root: str, cfg: dict):
    """目标目录可写且为占位态或不存在（不得覆盖已冻结的非空目录）。"""
    fdir = os.path.join(root, cfg["outputs"]["frozen_dir"])
    if os.path.isdir(fdir) and _dir_nonempty(fdir):
        if not is_placeholder(root, cfg):
            return False, ("%s 已存在且**非空、非占位态** ⇒ 拒绝覆盖"
                           "（后续修订只能新建 CodeObf-PI_v1.1）" % _rel(fdir, root)), {}
        return True, "占位态（md5sums.txt 与 splits/regime_id/train.csv 均为 0 字节）⇒ 允许覆写", \
            {"placeholder": True}
    # 目标不存在（或为空目录）：须确认「最近的已存在祖先」可写，否则无法新建
    probe = os.path.abspath(fdir)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.path.isdir(probe):
        return False, "目标与全部祖先目录均不存在：%s" % _rel(fdir, root), {}
    if not os.access(probe, os.W_OK):
        return False, "最近的已存在祖先不可写：%s" % _rel(probe, root), {}
    return True, "目标目录不存在或为空 ⇒ 将新建（最近已存在祖先 `%s` 可写）" % _rel(probe, root), \
        {"placeholder": False}


# ---------------------------------------------------------------------------
# 事实采集（供 manifest 与 DATA_CARD）
# ---------------------------------------------------------------------------
def collect_facts(root: str, cfg: dict, gates: dict) -> dict:
    """从**冻结树**（已复制）采集全部实算值。"""
    fdir = os.path.join(root, cfg["outputs"]["frozen_dir"])
    inp = cfg["inputs"]
    F = {}

    # --- 计数 ---
    F["rewrite_bodies"] = _count_rows(os.path.join(root, inp["rewrites"]))
    F["attack_total"] = _count_rows(os.path.join(root, inp["encoded"])) + \
        _count_rows(os.path.join(root, inp["code_carrier_csv"]))
    F["benign_counts"] = {}
    for g in ("G1", "G2", "G3"):
        F["benign_counts"][g] = _count_rows(os.path.join(root, inp["benign"], "%s.csv" % g))
    F["benign_total"] = sum(F["benign_counts"].values())

    # --- 分片 ---
    splits_rows, splits_groups = {}, {}
    for reg in REGIMES:
        splits_rows[reg], splits_groups[reg] = {}, {}
        for sp in SPLITS:
            rows = _read_csv(os.path.join(fdir, "splits", reg, "%s.csv" % sp))
            splits_rows[reg][sp] = len(rows)
            splits_groups[reg][sp] = len({r["seed_group_id"] for r in rows
                                           if r.get("sample_kind") == "attack"})
    F["splits_rows"], F["splits_attack_groups"] = splits_rows, splits_groups

    # --- 攻击样本（体制一 = 全视图）---
    atk = []
    for sp in SPLITS:
        atk += [r for r in _read_csv(os.path.join(fdir, "splits", "regime_id", "%s.csv" % sp))
                if r.get("sample_kind") == "attack"]
    F["attack_rows_regime_id"] = len(atk)
    F["carrier_type"] = dict(collections.Counter(r["carrier_type"] for r in atk))
    F["carrier_subtype"] = dict(collections.Counter(
        r["carrier_subtype"] for r in atk if r["carrier_type"] == "code"))
    F["difficulty"] = dict(collections.Counter(r["difficulty"] for r in atk))
    F["encoding_attack"] = dict(collections.Counter(r["encoding_type"] for r in atk))
    F["source_attack"] = dict(collections.Counter(r["source"] for r in atk))

    ood = []
    for sp in SPLITS:
        ood += [r for r in _read_csv(os.path.join(fdir, "splits", "regime_ood", "%s.csv" % sp))
                if r.get("sample_kind") == "attack"]
    F["encoding_ood"] = dict(collections.Counter(r["encoding_type"] for r in ood))

    # --- 代码载体（语言与 ast_ok 三值性）---
    cc = _read_csv(os.path.join(root, inp["code_carrier_csv"]))
    F["code_carrier_rows"] = len(cc)
    F["language"] = dict(collections.Counter(r.get("language", "") for r in cc))
    F["ast_ok"] = dict(collections.Counter(r.get("ast_ok", "") for r in cc))

    # --- 良性集 ---
    F["benign_source"] = {}
    for g in ("G1", "G2", "G3"):
        rows = _read_csv(os.path.join(fdir, "benign", "%s.csv" % g))
        F["benign_source"][g] = dict(collections.Counter(r["source"] for r in rows))
    g3 = _read_csv(os.path.join(fdir, "benign", "G3.csv"))
    F["g3_rows"] = len(g3)
    F["g3_encoding"] = dict(collections.Counter(r["encoding_type"] for r in g3))
    F["g3_business_form"] = dict(collections.Counter(r["g3_business_form"] for r in g3))
    F["g3_groups"] = len({r["benign_group_id"] for r in g3})
    dup = collections.Counter((r["benign_group_id"], r["text"]) for r in g3)
    F["g3_intra_group_dup"] = sum(v - 1 for v in dup.values() if v > 1)
    real_form = collections.Counter(r["g3_business_form"] for r in g3
                                    if r["source"] != "G3-synthetic")
    F["g3_real_rows"] = sum(real_form.values())
    F["g3_real_top_form"] = real_form.most_common(1)[0] if real_form else ("", 0)

    # --- 分组 ---
    sg = _read_csv(os.path.join(fdir, "seed_groups.csv"))
    F["seed_groups_total"] = len(sg)
    F["group_kind"] = dict(collections.Counter(r["group_kind"] for r in sg))
    fams = collections.defaultdict(set)
    for r in sg:
        if r["group_kind"] == "attack":
            fams[r["family_id"]].add(r["group_id"])
    F["families_total"] = len(fams)
    F["families_ah"] = sum(1 for k in fams if k.startswith("FAM-AH"))
    F["families_pku"] = sum(1 for k in fams if k.startswith("FAM-PKU"))
    F["family_sizes"] = dict(collections.Counter(
        (k.split("-")[0] + "-" + k.split("-")[1], len(v)) for k, v in fams.items()))

    # --- 种子元数据 ---
    meta_path = os.path.join(root, inp["seeds_meta"])
    F["seeds_meta"] = json.load(open(meta_path, encoding="utf-8")) if os.path.isfile(meta_path) \
        else {}

    # --- 派生视图 ---
    F["clean_original_rows"] = _count_rows(os.path.join(root, inp["clean_original_test"]))
    g3n_path = os.path.join(root, inp["g3_hard_negative_test"])
    g3n = _read_csv(g3n_path)
    F["g3_hard_negative_rows"] = len(g3n)
    F["g3_hard_negative_groups"] = len({r["benign_group_id"] for r in g3n})
    ood_ben = set()
    for sp in SPLITS:
        for r in _read_csv(os.path.join(fdir, "splits", "regime_ood", "%s.csv" % sp)):
            if r.get("sample_kind") == "benign":
                ood_ben.add(r["benign_group_id"])
    sub = [r for r in g3n if r["benign_group_id"] in ood_ben]
    F["g3n_ood_rows"] = len(sub)
    F["g3n_ood_groups"] = len({r["benign_group_id"] for r in sub})
    F["g3n_ood_encoding"] = dict(collections.Counter(r["encoding_type"] for r in sub))

    # --- 门禁产物 ---
    F["gates"] = gates
    F["length"] = _read_csv(os.path.join(root, inp["length_distribution"]))
    F["ngram"] = _read_csv(os.path.join(root, inp["ngram_overlap"]))
    F["ngram_controls"] = _read_csv(os.path.join(root, inp["ngram_controls"]))
    F["family_straddle"] = _read_csv(os.path.join(root, inp["family_straddle"]))
    F["assertions"] = {r["assertion_id"]: r for r in
                       _read_csv(os.path.join(root, inp["split_assertions"]))}

    # --- 来源清单 ---
    F["source_manifests"] = {}
    for name in cfg["sources"]:
        p = os.path.join(root, cfg["inputs"]["raw_root"], name, "SOURCE_MANIFEST.json")
        if os.path.isfile(p):
            F["source_manifests"][name] = {
                "path": _rel(p, root), "sha256": _sha256_file(p),
                "data": json.load(open(p, encoding="utf-8"))}
        else:
            F["source_manifests"][name] = {"path": _rel(p, root), "sha256": None, "data": {}}
    return F


# ---------------------------------------------------------------------------
# 字段词典
# ---------------------------------------------------------------------------
def field_glossary() -> dict:
    """逐字段含义（`DATA_CARD` 第 3 条；字段名 → 含义）。"""
    return collections.OrderedDict([
        ("sample_id", "样本唯一标识。攻击侧 `ATK-<rewrite_id 去横线>-<enc>-<carrier>`；"
                      "良性侧 `BN-<G1|G2|G3>-<nnnn>`；干净原始测试 `CLN-<rewrite_id>`。全局唯一。"),
        ("seed_group_id", "攻击侧**唯一划分单位**（`SG-0001` 起四位零填）；同组全部改写体、"
                          "编码变体、载体变体只进一个分片。与 `seed_id` 一一对应。"),
        ("benign_group_id", "良性侧**唯一划分单位**（`BG-<nnnn>`）。与 `seed_group_id` "
                            "不同源（前缀可程序判别），两者分别参与 group 级隔离。"),
        ("sample_kind", "`attack` 或 `benign`；分片宽表中两类行共表。"),
        ("family_id", "语义族（`FAM-AH-*`／`FAM-PKU-*`／良性填 `benign_group_id`）。"
                      "**只用于登记与测量（`A14`），不是划分单位**；"
                      "`family_atomic: true` 下划分单位上提为族。"),
        ("rewrite_id", "改写体标识（`RW-0001`…`RW-0440`）。"),
        ("seed_id", "来源种子标识（`SEED-<AH|PKU>-<nnn>`）。"),
        ("source", "来源。恶意侧 `AgentHarm`／`PKU-SafeRLHF`；良性侧 `TheStack`／"
                   "`CodeSearchNet`／`Alpaca`／`Wikitext`／`G3-synthetic`。"),
        ("harm_category", "危害类别（攻击侧；取自 AgentHarm 的行为类别）。"),
        ("benign_set", "良性子集：`G1`／`G2`／`G3`（良性行专用）。"),
        ("encoding_type", "9 编码之一（封闭域，见 `encoding` 清单）。"),
        ("carrier_type", "载体：`text`（载荷本体）或 `code`（嵌入宿主代码后的完整文本）。"),
        ("carrier_subtype", "代码载体子型（5 值封闭域）：`py_comment`／`js_comment`／"
                            "`string_literal`／`docstring`／`identifier_ast`。**`text` 载体行为空。**"),
        ("host_context", "代码载体的宿主骨架标识（10 份宿主骨架之一）。"),
        ("g3_business_form", "G3 业务必需形态（真值侧 11 类，合成侧按 20 场景生成）。"),
        ("business_necessity_evidence", "G3 该样本「编码层业务必需」的证据文本（`Q3` 判定依据）。"),
        ("plaintext", "改写体明文（未编码）。**不是**阶段3–5 的输入。"),
        ("payload", "编码后载荷（未嵌入宿主）。**不是**阶段3–5 的输入。"),
        ("text", "**最终送入检测器的字符串**——`text` 载体 = 载荷本体；`code` 载体 = "
                 "嵌入后的完整代码文本。**`text` 是阶段3–5 的唯一输入字段**"
                 "（`PREREG_v1.md` §12.2／v5 新增 20）。所有检测器一律只读此列。"),
        ("difficulty", "编码难度标签（封闭域 `low`／`medium`／`high`）。"),
        ("language", "代码载体的语言（封闭域 `python`／`javascript`）。**`text` 载体无此字段。**"),
        ("ast_ok", "`code_carrier.csv` 专用，**三值**：`1` = Python 载体 `ast.parse` 通过；"
                   "**`-1` = JavaScript 载体（`ast.parse` 不适用）**；`0` = 解析失败。"
                   "**门禁 2 的分母只含 `{0,1}` 的行**（本阶段实测 3152）。"),
        ("label", "二分类标签。攻击侧恒 `1`（恶意）；良性侧恒 `0`。"),
        ("passed", "`split_assertions.csv` 专用：断言结果。**`A14` 恒为 `n/a`（报告性断言，非门禁）**。"),
    ])


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def build_manifest(cfg: dict, facts: dict, frozen_at: str, sealed_files, root: str) -> dict:
    exp = cfg["expected"]
    gates = {}
    for key, label in (("g1_intent", "g1_intent"), ("g2_ast", "g2_ast"),
                       ("g3_qc", "g3_qc"), ("g4_guard", "g4_guard"),
                       ("g5_group", "g5_group"), ("g6_length", "g6_length"),
                       ("g7_ngram", "g7_ngram")):
        gates[label] = facts["gates"].get(key, {})
    return collections.OrderedDict([
        ("name", cfg["name"]),
        ("version", cfg["version"]),
        ("status", cfg["status"]),
        ("frozen_at", frozen_at),
        ("freeze_script", cfg["freeze_script"]),
        ("freeze_commit", _git_head(root)),
        ("counts", collections.OrderedDict([
            ("rewrite_bodies", facts["rewrite_bodies"]),
            ("attack_total", facts["attack_total"]),
            ("encodings", len(facts["encoding_attack"]) or exp["encodings"]),
            ("carriers", len(facts["carrier_type"]) or exp["carriers"]),
            ("benign_total", facts["benign_total"]),
            ("benign_G1", facts["benign_counts"]["G1"]),
            ("benign_G2", facts["benign_counts"]["G2"]),
            ("benign_G3", facts["benign_counts"]["G3"]),
        ])),
        ("splits", collections.OrderedDict(
            (reg, collections.OrderedDict(
                (sp, facts["splits_attack_groups"][reg][sp]) for sp in SPLITS))
            for reg in REGIMES)),
        ("split_ratio", cfg["expected"]["split_ratio"]),
        ("split_seed", int(cfg["expected"]["split_seed"])),
        ("sealed_test", collections.OrderedDict([
            ("files", sealed_files),
            ("md5sums", "%smd5sums.txt" % cfg["outputs"]["sealed_dir"]),
            ("access_log", "%s%s" % (cfg["outputs"]["sealed_dir"],
                                     cfg["sealed"]["access_log"])),
        ])),
        ("gates", gates),
        ("upstream", collections.OrderedDict([
            ("prereg", cfg["upstream"]["prereg"]),
            ("source_lock_tag", cfg["upstream"]["source_lock_tag"]),
            ("env_lock_tag", cfg["upstream"]["env_lock_tag"]),
            ("env_lock_commit", cfg["upstream"]["env_lock_commit"]),
            ("stage2_plan", cfg["upstream"]["stage2_plan"]),
            ("window", cfg["upstream"]["window"]),
        ])),
        ("field_glossary", field_glossary()),
    ])


# ---------------------------------------------------------------------------
# md5sums
# ---------------------------------------------------------------------------
def build_md5sums(base_dir: str, rel_paths, skip=()) -> "list[str]":
    lines = []
    for rel in sorted(rel_paths):
        if rel in skip:
            continue
        p = os.path.join(base_dir, rel.replace("/", os.sep))
        lines.append("%s  %s" % (_md5_file(p), rel))
    return lines


def list_frozen_files(fdir: str) -> "list[str]":
    """冻结树下全部 CSV／JSON／MD（相对路径，正斜杠，按字典序）。"""
    out = []
    for root_, _dirs, files in os.walk(fdir):
        for name in files:
            if os.path.splitext(name)[1].lower() in (".csv", ".json", ".md"):
                out.append(os.path.relpath(os.path.join(root_, name), fdir).replace("\\", "/"))
    return sorted(out)


def verify_md5sums(base_dir: str, lines) -> "list[str]":
    """逐行复算（自校验一次并留证）。返回不一致项。"""
    bad = []
    for line in lines:
        md5, _, rel = line.partition("  ")
        p = os.path.join(base_dir, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            bad.append("缺失 %s" % rel)
            continue
        if _md5_file(p) != md5:
            bad.append("内容不符 %s" % rel)
    return bad


# ---------------------------------------------------------------------------
# DATA_CARD
# ---------------------------------------------------------------------------
def render_data_card(cfg, facts, frozen_at, frozen_md5_count, sealed_md5_file,
                     manifest_md5, tree_sha256, root) -> str:
    exp = cfg["expected"]
    g = facts["gates"]
    L = []
    A = L.append

    def esc(v):
        """表格单元格转义：`|` 会破坏 Markdown 表格，一律转义为 `\\|`。"""
        return str(v).replace("|", "\\|")

    def kv(k, v):
        A("| %s | %s |" % (esc(k), esc(v)))

    def tbl(header, rows):
        A("| " + " | ".join(esc(h) for h in header) + " |")
        A("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            A("| " + " | ".join(esc(x) for x in r) + " |")
        A("")

    A("# DATA CARD — CodeObf-PI v1.0")
    A("")
    A("> 本文件由 `scripts/data/02_08_freeze_dataset.py` 在 `make freeze` 时**生成**，")
    A("> 冻结后随 `data-v1.0` tag 只读。**不得事后修改**；任何修订须新建 `CodeObf-PI_v1.1`。")
    A("")

    # ---- 1 ----
    A("## §1 数据集概况")
    A("")
    tbl(["项", "值"], [
        ["数据集名称", "**CodeObf-PI**"],
        ["版本", "**v1.0**"],
        ["生成日期（`frozen_at`）", "`%s`" % frozen_at],
        ["生成脚本", "`%s`" % cfg["freeze_script"]],
        ["生成脚本 commit", "`%s`" % _git_head(root)],
        ["冻结状态", "`%s`" % cfg["status"]],
        ["冒烟脚本", "`%s`" % cfg["smoke_script"]],
        ["主清单", "`manifest.json`（md5 `%s`）" % manifest_md5],
        ["冻结树整体 sha256（本文档生成前）", "`%s`" % tree_sha256],
    ])

    # ---- 2 ----
    A("## §2 数据来源（六个来源）")
    A("")
    A("| 来源 | `hf_repo_id` | revision | 镜像端点 | 许可证 | 下载日期 | "
      "`SOURCE_MANIFEST.json` | manifest sha256 |")
    A("|---|---|---|---|---|---|---|---|")
    for name in cfg["sources"]:
        m = facts["source_manifests"][name]
        d = m.get("data") or {}
        A("| `%s` | `%s` | `%s` | `%s` | %s | %s | `%s` | `%s` |" % (
            name, d.get("hf_repo_id", "—"), d.get("hf_revision", "—"),
            d.get("mirror_endpoint", "—"), str(d.get("license", "—")).replace("|", "\\|"),
            d.get("download_date", "—"), m["path"], m["sha256"] or "—"))
    A("")
    A("### §2.1 `TheStack` 槽位的**实际语料**（**必须披露**，`D-35` 增补／`DIRECTORY.md` §3.1）")
    A("")
    A("`DIRECTORY.md` §3.1 的**目录名**是 `TheStack`，而该槽位**实际装的是非 gated 的替代语料**：")
    A("")
    ts = (facts["source_manifests"]["TheStack"].get("data") or {})
    tbl(["项", "值（实测，出自 `%s`）" % facts["source_manifests"]["TheStack"]["path"]], [
        ["槽位名（`DIRECTORY.md` §3.1 的目录名）", "`TheStack`"],
        ["**实际语料**", "**`%s`**（HF datasets）" % ts.get("hf_repo_id", "—")],
        ["revision", "**`%s`**" % ts.get("hf_revision", "—")],
        ["镜像端点", "`%s`" % ts.get("mirror_endpoint", "—")],
        ["许可证", "%s" % ts.get("license", "—")],
        ["文件数／字节", "%s ／ %s" % (ts.get("file_count", "—"),
                                      "{:,}".format(ts.get("total_bytes", 0)))],
        ["下载日期", "%s" % ts.get("download_date", "—")],
        ["manifest sha256", "`%s`" % facts["source_manifests"]["TheStack"]["sha256"]],
    ])
    A("**为何必须写**：`bigcode/the-stack` 与 `bigcode/the-stack-smol` 在 HF 均为 **gated**，")
    A("`hf-mirror` 不能绕过（`D-07`）⇒ 依 `D-28` 裁定二·连带项／转交项 36 改采**非 gated** 等价语料。")
    A("**若只按目录名理解，读者与审稿人会误以为用了 gated 的 `bigcode/the-stack`。**")
    A("")
    A("**配套（同批）**：该**来源替换事实**亦登记于 §13 已知局限（与本条互为引用、不重复登记）。")
    A("`G1`／`G3` 中 `source == \"TheStack\"` 的行数：`G1 = %d`／`G3 = %d`。"
      % (facts["benign_source"]["G1"].get("TheStack", 0),
         facts["benign_source"]["G3"].get("TheStack", 0)))
    A("")
    A("> **数字更正（本窗口实测）**：`windows/W3.8_freeze.md` §3.6 第 28 条记 `G3 = 396`；")
    A("> 现行 `G3.csv`（`W3.5` 自审轮 `r6` 重出，sha256 `727e820f…`）实测 **`%d`**。"
      % facts["benign_source"]["G3"].get("TheStack", 0))
    A("> 两数之差源于 `r6` 换版；**以本卡实测值为准**。")
    A("")

    # ---- 3 ----
    A("## §3 字段含义（`field_glossary`）")
    A("")
    gloss = field_glossary()
    tbl(["字段", "含义"], [[("`%s`" % k), v] for k, v in gloss.items()])
    A("> **`text` 是阶段3–5 的唯一输入字段**：`plaintext`／`payload` 仅供复现与审计，")
    A("> 任何检测器（B1–B5 与 Ours）一律只读 `text` 列。")
    A("")
    A("> **`A14` 的报告性地位**：`split_assertions.csv` 的 `A14` 为**报告性断言**（**非门禁**），")
    A("> `passed` 恒记 `n/a`，**不参与任何退出码**。字段说明中的 `passed` 一栏即指该列。")
    A("")

    # ---- 4 ----
    A("## §4 样本量")
    A("")
    ben = facts["benign_counts"]
    ben_total = facts["benign_total"]
    tbl(["项", "值", "口径"], [
        ["改写体总数", "**%d**" % facts["rewrite_bodies"], "`PREREG_v1.md` §6（`V-1`）"],
        ["攻击样本合计", "**%d**" % facts["attack_total"],
         "`%d × 9 × 2`（`V-1`）" % facts["rewrite_bodies"]],
        ["编码数", "**%d**" % len(facts["encoding_attack"]), "`V-5`"],
        ["载体数", "**%d**" % len(facts["carrier_type"]), "`V-5`"],
        ["良性总数", "**%d**" % ben_total, "`V-8`"],
        ["G1", "%d（实算占比 %.4f）" % (ben["G1"], ben["G1"] / float(ben_total)), "目标 50%"],
        ["G2", "%d（实算占比 %.4f）" % (ben["G2"], ben["G2"] / float(ben_total)), "目标 30%"],
        ["G3", "%d（实算占比 %.4f）" % (ben["G3"], ben["G3"] / float(ben_total)), "目标 20%"],
        ["种子组总数 `N`", "**%d**" % facts["seeds_meta"].get("seed_count_total", 0),
         "`data/interim/seed_collect/seeds.meta.json`"],
        ["攻击 group 数（`seed_group_id`）", "**%d**" % facts["group_kind"].get("attack", 0), "—"],
        ["良性 group 数（`benign_group_id`）", "**%d**" % facts["group_kind"].get("benign", 0), "—"],
    ])
    A("分片攻击 group 数与行数（两体制共用同一 group 划分，差别只在行集）：")
    A("")
    rows = []
    for reg in REGIMES:
        for sp in SPLITS:
            rows.append([reg, sp, facts["splits_attack_groups"][reg][sp],
                         facts["splits_rows"][reg][sp]])
    tbl(["体制", "分片", "攻击 group 数", "分片总行数（攻击+良性）"], rows)

    # ---- 5 ----
    A("## §5 编码类型")
    A("")
    A("9 编码清单与 `difficulty` 映射（`V-5`；`difficulty` 为**封闭域** `low`／`medium`／`high`）：")
    A("")
    diff_map = {"rot13": "low", "reverse": "low", "base64": "medium", "hex": "medium",
                "morse": "medium", "url": "medium", "ascii_art": "high",
                "zerowidth": "high", "homoglyph": "high"}
    tbl(["编码", "`difficulty`", "攻击集条数（体制一全视图）"],
        [[("`%s`" % e), diff_map[e], facts["encoding_attack"].get(e, 0)]
         for e in ("base64", "hex", "ascii_art", "rot13", "reverse", "morse", "url",
                   "zerowidth", "homoglyph")])
    A("`difficulty` 实算分布：%s。" % ", ".join("`%s` = %d" % (k, v)
                                                for k, v in sorted(facts["difficulty"].items())))
    A("")
    A("**两体制的 5／4 拆分（`V-6`）**：")
    A("")
    tbl(["体制", "`train`／`val` 编码", "`test` 编码"], [
        ["体制一（ID）", "全部 9 种", "全部 9 种"],
        ["体制二（OOD）", "`base64`／`rot13`／`zerowidth`／`url`／`reverse`（5 种）",
         "`hex`／`morse`／`ascii_art`／`homoglyph`（4 种留出）"],
    ])
    A("体制二各编码实算条数：%s。"
      % ", ".join("`%s` = %d" % (k, v) for k, v in sorted(facts["encoding_ood"].items())))
    A("")

    # ---- 6 ----
    A("## §6 载体类型")
    A("")
    tbl(["载体", "条数（体制一全视图）"],
        [[("`%s`" % k), v] for k, v in sorted(facts["carrier_type"].items())])
    A("**5 个 `carrier_subtype`（`V-7`）及其实算计数**（`code` 载体行；`text` 载体行为空）：")
    A("")
    tbl(["`carrier_subtype`", "实算计数"],
        [[("`%s`" % k), facts["carrier_subtype"].get(k, 0)]
         for k in ("py_comment", "js_comment", "string_literal", "docstring", "identifier_ast")])
    A("**`language` 取值域（`python`／`javascript`）与计数**：%s。"
      % ", ".join("`%s` = %d" % (k, v) for k, v in sorted(facts["language"].items())))
    A("")
    A("### §6.1 ⚠ 两条**必须显式写明**的陷阱")
    A("")
    A("**（1）`ast_ok` 的三值性。** 实测：%s。判据逐字如下——" %
      "、".join("`ast_ok = %s` → **%d 行**" % (k, v) for k, v in sorted(facts["ast_ok"].items())))
    A("")
    A("- `1` = Python 载体 `ast.parse` 通过；")
    A("- **`-1` = JavaScript 载体（`ast.parse` 不适用）**；")
    A("- `0` = 解析失败。")
    A("")
    A("**门禁 2 的分母只含 `{0,1}` 的行**（本阶段实测 **%d**），**`-1` 行不得计入分母**——"
      % g.get("g2_ast", {}).get("denominator", 0))
    A("否则报错率被系统性稀释。**读者若把 %d 当成「全部代码载体」，会得出错误的覆盖率**："
      % g.get("g2_ast", {}).get("denominator", 0))
    A("正确基数是**代码载体共 %d**，其中 JS **%d**。"
      % (facts["code_carrier_rows"], facts["language"].get("javascript", 0)))
    A("")
    A("**（2）三个封闭值域清单。**")
    A("")
    tbl(["字段", "封闭值域", "计数"], [
        ["`difficulty`", "`{low, medium, high}`（映射见 §5）",
         ", ".join("`%s`=%d" % (k, v) for k, v in sorted(facts["difficulty"].items()))],
        ["`carrier_subtype`", "`{py_comment, js_comment, string_literal, docstring, identifier_ast}`",
         ", ".join("`%s`=%d" % (k, facts["carrier_subtype"].get(k, 0))
                   for k in ("py_comment", "js_comment", "string_literal", "docstring",
                             "identifier_ast"))],
        ["`language`", "`{python, javascript}`",
         ", ".join("`%s`=%d" % (k, v) for k, v in sorted(facts["language"].items()))],
    ])

    # ---- 7 ----
    A("## §7 `group_id` 的分配规则与计数")
    A("")
    tbl(["标识", "分配规则", "计数"], [
        ["`seed_group_id`", "一条来源种子恰一个（`SG-0001` 起四位零填，连续无空洞）；"
                            "**全阶段唯一的分组单位**", "**%d**" % facts["group_kind"].get("attack", 0)],
        ["`benign_group_id`", "良性侧分组单位（`BG-<nnnn>`）；与攻击侧 group 空间不同源",
         "**%d**" % facts["group_kind"].get("benign", 0)],
        ["`family_id`", "语义族：AgentHarm 同 `id_original` 的 4 变体共用一族（`FAM-AH-*`）；"
                        "PKU-SafeRLHF 每 group 自成一族（`FAM-PKU-*`）；良性侧填 `benign_group_id`。"
                        "**仅登记与测量（`A14`），非划分单位**",
         "攻击侧 **%d** 族（AH %d + PKU %d）"
         % (facts["families_total"], facts["families_ah"], facts["families_pku"])],
    ])
    A("攻击侧族规模：%s。"
      % ", ".join("`%s` 族 × %d group" % (k, v) for k, v in sorted(facts["family_sizes"].items())))
    A("")

    # ---- 8 ----
    A("## §8 G1／G2／G3 构建")
    A("")
    tbl(["子集", "来源分布", "过滤规则要点"], [
        ["G1（真实代码）", ", ".join("`%s` = %d" % (k, v)
                                      for k, v in sorted(facts["benign_source"]["G1"].items())),
         "真实开源代码，按语言与体积上限采样；排除 `original/*` 类巨型文件"],
        ["G2（自然语言）", ", ".join("`%s` = %d" % (k, v)
                                      for k, v in sorted(facts["benign_source"]["G2"].items())),
         "Alpaca 官方拼接格式；Wikitext 按标题分窗；C0 控制字符剔除"],
        ["G3（编码但良性）", ", ".join("`%s` = %d" % (k, v)
                                        for k, v in sorted(facts["benign_source"]["G3"].items())),
         "真实侧天然携带编码串 + 合成侧按 20 场景；`Q3`（编码层业务必需）判定见 `g3_hard_negative.md`"],
    ])
    A("**G3 业务必需形态分布（实算）**：")
    A("")
    tbl(["业务形态", "条数"], [[("`%s`" % k), v] for k, v in
                              sorted(facts["g3_business_form"].items(),
                                     key=lambda kv: (-kv[1], kv[0]))])
    A("G3 group 数 **%d**；G3 行数 **%d**。"
      % (facts["g3_groups"], facts["g3_rows"]))
    A("")

    # ---- 9 ----
    A("## §9 质检统计（门禁 1–7 实算值与产物路径）")
    A("")
    tbl(["门禁", "阈值", "实算值", "结论", "产物"], [
        ["1 · 改写意图一致性",
         "≥%.2f" % g.get("g1_intent", {}).get("threshold", 0.95),
         "通过率 **%.4f**" % g.get("g1_intent", {}).get("rate", 0),
         "**PASS**", "`%s`" % g.get("g1_intent", {}).get("report", "—")],
        ["2 · AST 报错率",
         "≤%.2f" % g.get("g2_ast", {}).get("threshold", 0.10),
         "**%.6f**（%d / %d）" % (g.get("g2_ast", {}).get("rate", 0),
                                  g.get("g2_ast", {}).get("ast_failed", 0),
                                  g.get("g2_ast", {}).get("denominator", 0)),
         "**PASS**", "`data/interim/code_carrier/ast_report.csv`"],
        ["3 · G3 质检通过率（**双人独立**）",
         "≥%.2f" % g.get("g3_qc", {}).get("threshold", 0.95),
         "**%.4f**（双人通过 %d / %d；标注者 %d 位）"
         % (g.get("g3_qc", {}).get("rate", 0), g.get("g3_qc", {}).get("both_pass", 0),
            g.get("g3_qc", {}).get("samples", 0), len(g.get("g3_qc", {}).get("annotators", []))),
         "**PASS**", "`reports/g3_qc.csv`"],
        ["4 · 良性集检出率",
         "≤%.2f" % g.get("g4_guard", {}).get("threshold", 0.05),
         "**%.6f**（counted %d / %d）" % (g.get("g4_guard", {}).get("rate", 0),
                                          g.get("g4_guard", {}).get("counted", 0),
                                          g.get("g4_guard", {}).get("n", 0)),
         "**PASS**", "`data/interim/benign_G1_G2_G3/benign_guard_check.csv`"],
        ["5 · group 交集 / OOD 编码", "空集",
         "`A1`–`A5` 全 `True`；`A5` 内容列命中 **0**", "**PASS**", "`reports/split_assertions.csv`"],
        ["6 · 长度分布差值", "≤%.2f" % g.get("g6_length", {}).get("threshold", 0.20),
         "最差 %.6f（%s）" % (g.get("g6_length", {}).get("worst", 0),
                              g.get("g6_length", {}).get("worst_at", "—")),
         "**PASS**", "`reports/length_distribution.csv`"],
        ["7 · 3-gram 重叠率", "%.2f（**一字未改**）" % g.get("g7_ngram", {}).get("threshold", 0.05),
         "`regime_id` %.6f／`regime_ood` %.6f"
         % (g.get("g7_ngram", {}).get("regime_id", 0), g.get("g7_ngram", {}).get("regime_ood", 0)),
         "**`n/a`（报告项）**", "`reports/ngram_overlap.csv`"],
    ])
    A("> **门禁 3 的双人独立标注**在 §9 本表逐项登记（标注者 2 位、双人通过数／总数见上）；")
    A("> **门禁 1 与门禁 4 的人工判定均为「代录」**，其性质与核对范围分别见 **§23**（门禁 1）与")
    A("> **§26**（门禁 4）——两处抽样／复核表本身**不进冻结面**，故须内联于本卡。")
    A("")

    # ---- 10 ----
    A("## §10 长度分布与 n-gram 重叠")
    A("")
    A("产物：`reports/length_distribution.csv`、`reports/ngram_overlap.csv`。")
    A("")
    A("**长度分布（门禁 6；口径 = `abs_diff_ratio = |均值(正) − 均值(负)| / max(两均值)` ≤ %.2f）**："
      % g.get("g6_length", {}).get("threshold", 0.20))
    A("")
    tbl(["体制", "分片", "正均值(tokens)", "负均值(tokens)", "`abs_diff_ratio`", "判定"],
        [[r["regime"], r["split"], r["positive_mean_tokens"], r["negative_mean_tokens"],
          r["abs_diff_ratio"], r["passed"]] for r in facts["length"]
         if r.get("sample_kind") == "shard"])
    A("**n-gram 重叠（门禁 7；阈值 %.2f，判定只认字符级 3-gram）**："
      % g.get("g7_ngram", {}).get("threshold", 0.05))
    A("")
    tbl(["体制", "n", "level", "重叠率", "`passed`"],
        [[r["regime"], r["n"], r["level"], r["overlap_rate"], r.get("passed") or "（对照，留空）"]
         for r in facts["ngram"]])
    A("**逐样本最近邻（`A14` 的配套量；原子化后实测）**：%s"
      % facts["assertions"].get("A14", {}).get("measured", "—"))
    A("")

    # ---- 11 ----
    A("## §11 MD5／版本哈希")
    A("")
    tbl(["件", "路径", "MD5／sha256"], [
        ["冻结侧 `md5sums.txt`", "`data/frozen/CodeObf-PI_v1.0/md5sums.txt`",
         "**%d 条**；自身 md5 **见运行报告 §4**（见下方自指说明）" % frozen_md5_count],
        ["封存侧 `md5sums.txt`", "`data/sealed_test/CodeObf-PI_v1.0/md5sums.txt`",
         "自身 md5 **`%s`**" % sealed_md5_file],
        ["`manifest.json`", "`data/frozen/CodeObf-PI_v1.0/manifest.json`",
         "md5 **`%s`**" % manifest_md5],
        ["冻结树整体（本文档生成前）", "`data/frozen/CodeObf-PI_v1.0/`",
         "sha256 **`%s`**" % tree_sha256],
    ])
    A("两份 `md5sums.txt` 格式一律 `<md5>  <相对路径>`（**相对各自的根目录**，正斜杠路径），")
    A("**UTF-8 + LF**；冻结侧覆盖该目录下全部 CSV／JSON／MD，")
    A("封存侧覆盖 4 个 CSV（**不含 `md5sums.txt` 自身与 `ACCESS_LOG.md`**，避免自指）。")
    A("`02_08` 已**自校验一次**并在运行报告 §4 留证。")
    A("")
    A("> **⚠ 自指说明（`windows/W3.8_freeze.md` §3.6 第 11 条的可实现性限制，如实登记）**：")
    A("> 该条要求内联「两份 `md5sums.txt` 的**自身 md5**」。**封存侧可内联**（其覆盖面只有 4 个 CSV）；")
    A("> **冻结侧不可内联**：`DIRECTORY.md` §3.3／§3.6 要求冻结侧 `md5sums.txt` 覆盖该目录下**全部**")
    A("> CSV／JSON／MD，**而本数据卡本身就在该目录内** ⇒ 「`md5sums.txt` 的内容依赖数据卡的 md5」与")
    A("> 「数据卡的内容依赖 `md5sums.txt` 的 md5」构成**循环依赖**，密码学哈希下**不存在不动点**。")
    A("> 故本卡登记其**条数**与**自校验结论**，**自身 md5 落在运行报告 §4**（报告在冻结树之外，不受")
    A("> `md5sums` 覆盖）。**这是一处规格限制，不是遗漏。**")
    A("")

    # ---- 12 ----
    A("## §12 测试分片封存位置与访问登记")
    A("")
    A("封存位置：`data/sealed_test/CodeObf-PI_v1.0/` —— 仅含**测试分片**，不含训练/验证分片"
      "（`DIRECTORY.md` §5-2）。")
    A("")
    tbl(["文件", "来源"], [
        ["`regime_id_test.csv`", "`data/interim/split_checks/splits/regime_id/test.csv`"],
        ["`regime_ood_test.csv`", "`data/interim/split_checks/splits/regime_ood/test.csv`"],
        ["`clean_original_test.csv`", "`data/interim/split_checks/clean_original_test.csv`"],
        ["`g3_hard_negative_test.csv`", "`data/interim/split_checks/g3_hard_negative_test.csv`"],
        ["`md5sums.txt`", "由 `02_08` 生成"],
        ["`ACCESS_LOG.md`", "**表头固定 6 列**（日期、窗口、用途、命令、输出路径、复核人）；"
                            "由 A4/A6/A7 评估窗口在读取前追加，**`02_08` 不写任何数据行**"],
    ])
    A("**登记要求**：A4/A6/A7 每次读取前必须在 `ACCESS_LOG.md` 追加一条 6 列记录；")
    A("A5 训练窗口**禁止读取**；A0 只读审计；每次评估结束须复核 `md5sums.txt` 未变。")
    A("")

    # ---- 13 ----
    A("## §13 已知局限")
    A("")
    A("1. **G1 子集采样策略**：真实代码语料按语言与体积上限采样，非全量；采样规则见各来源的"
      "`SOURCE_MANIFEST.json` 的 `subset_rule`。")
    A("2. **来源替换（`TheStack` 槽位）**：该槽位实际语料为 **`%s`** rev `%s`（非 gated），"
      % (ts.get("hf_repo_id", "—"), ts.get("hf_revision", "—")))
    A("   而非 `bigcode/the-stack`（gated，镜像不可绕过）。**详见 §2.1**，与本条互为引用。")
    A("3. **校准覆盖缺口**：阶段0 的标注者校准（κ = 0.916667 ≥ 0.75）**覆盖 41/53 条**，")
    A("   **12 条未覆盖**并已在 `reports/prereg/annotator_calibration_v1.md` §9 声明。")
    A("   **该声明不得删改、不得省略**；本卡**不追补**这 12 条为「已覆盖」。")
    A("4. **G3 真实侧形态不均衡**（`W3.5` 呈报 4，`W3` 采纳）：真实代码侧候选池中「**完整性与校验值**」")
    A("   （hex 摘要等）形态占绝大多数（本代实测：真实侧 %d 行中该形态 **%d** 行，占 %.1f%%）。"
      % (facts["g3_real_rows"], facts["g3_real_top_form"][1],
         100.0 * facts["g3_real_top_form"][1] / max(1, facts["g3_real_rows"])))
    A("   **性质声明**：这是**真实开源代码的客观属性**（代码天然携带的编码字符串以摘要常量为主），")
    A("   **非人为倾斜**；披露目的是让读者知道「G3 的编码形态分布**不代表** 9 种编码的均匀分布」。")
    A("   **不得**因披露而暗示该集不合格（门禁 3 已通过，见 §9）。")
    A("   > **数字更正（本窗口实测）**：`windows/W3.8_freeze.md` §3.6 第 24 条记「约占 96%、"
      "1800 条中约 760 条、真实 810 条」；")
    A("   > 现行 `G3`（`r6`）实测 **真实侧 %d 行、该形态 %d 行（%.1f%%）**、"
      "全 1800 条中该形态 **%d 条**。两代不同，以本卡实测值为准。"
      % (facts["g3_real_rows"], facts["g3_real_top_form"][1],
         100.0 * facts["g3_real_top_form"][1] / max(1, facts["g3_real_rows"]),
         facts["g3_business_form"].get(facts["g3_real_top_form"][0], 0)))
    A("5. **门禁 4 的送检／截断口径**：真实与 G1/G2 送 `as_is`、G3 合成送**还原后取值**；")
    A("   **左截断 55 / 90 条** ⇒ 命中样本中存在**基于不完整文本**判定者（`BN-G3-0241` 即属此类）。")
    A("6. **门禁 7 的不可判别性**：该量在本语料上**不具判别力**（详见 §29）。")
    A("7. **`family_atomic` 的行为多样性代价**：`test` 分片按来源计的行为数 **20 → 7**（详见 §29）。")
    A("")

    # ---- 14 ----
    A("## §14 「事前定值」节（`V-1`–`V-10`）")
    A("")
    A("> **定值时点早于任何数据生成与划分**（`D-28` 裁定四）；本节即「**未事后定参**」的留痕。")
    A("> 时点：`make freeze` **之前**（本节由 `02_08` 在生成 `DATA_CARD` 时即写入，不留待事后补写）。")
    A("")
    tbl(["编号", "定值内容", "性质／依据"], [
        ["`V-1`", "**440**（改写体总数）／**7920**（攻击样本合计 = 440 × 9 × 2）",
         "`PREREG_v1.md` §6 冻结值"],
        ["`V-2`", "两套体制**共用同一 group 划分**，差别仅在编码子集", "`PREREG_v1.md` §2"],
        ["`V-3`", "划分比例 **8:1:1**（group 级）+ 划分随机种子 **42**",
         "本阶段工程性定值（`stage2_plan` §5 呈报 2，`D-28` 批准）"],
        ["`V-4`", "每个分片内部正负比 **≈1:1**；良性侧按需**确定性欠采样**至与攻击侧等量",
         "`PREREG_v1.md` §6 末条"],
        ["`V-5`", "**9 编码**／**2 载体**（`text`／`code`）", "v5 L31／L216"],
        ["`V-6`", "体制二训练 **5 编码**（`base64`／`rot13`／`zerowidth`／`url`／`reverse`）；"
                  "留出 **4 编码**（`hex`／`morse`／`ascii_art`／`homoglyph`）",
         "v5 L31；`PREREG_v1.md` §2.2"],
        ["`V-7`", "代码载体 **5 子型**：`py_comment`／`js_comment`／`string_literal`／"
                  "`docstring`／`identifier_ast`", "v5 §2.4"],
        ["`V-8`", "良性集 G1:G2:G3 占比 **50%:30%:20%**", "`PREREG_v1.md` §6；v5 §2.5"],
        ["`V-9`", "所有落盘 CSV 一律 **UTF-8 + LF**；字段顺序与类型按契约固定",
         "`PROJECT_CONTRACT.md` §7 精神；`D-04` 行尾归一"],
        ["`V-10`", "`rewrite_index` 恒 ∈ {1,2} —— **每组恰 2 条改写**（`PREREG_v1.md` §6 的"
                   "「2–3 条」范围内取**下界**，合法）",
         "**派生**（由 `V-1` 的 440 硬总数与实测 `N = %d` 派生），**非新建口径**"
         % facts["seeds_meta"].get("seed_count_total", 0)],
    ])

    # ---- 15 ----
    A("## §15 `V-3` 的划分比例与种子（**同时写入 `manifest.json`**）")
    A("")
    tbl(["项", "值"], [
        ["`manifest.json` 的 `split_ratio`", "`\"%s\"`" % cfg["expected"]["split_ratio"]],
        ["`manifest.json` 的 `split_seed`", "`%d`" % cfg["expected"]["split_seed"]],
        ["`manifest.json` 的 `splits` 段", "由 `seed_groups.csv` 实算（见 §4 表）"],
        ["与 `split_assertions.csv` 的 `A13` 一致性", "**已强制**：不一致即非零退出（`A14` 非门禁，不参与）"],
    ])

    # ---- 16 ----
    A("## §16 AgentHarm `44 × 4` 变体结构的披露")
    A("")
    A("① **`176 = 44 个 `id_original` × 4 变体`**，**四个变体的 prompt 文本互不相同**，")
    A("   故按「一条种子 = 一个 group」各占一个 `seed_group_id`（`W3.1` 断言 `B1`／`B2`）。")
    A("② **同一语义族的 4 个变体可能被划入不同分片**——`seed_group_id` 仍是唯一划分单位，")
    A("   `family_id` 仅登记。")
    A("③ **内联 `A14` 的实测值**：%s" % facts["assertions"].get("A14", {}).get("measured", "—"))
    A("   源文件 `data/interim/split_checks/family_straddle.csv`（**属 `data/interim/`，不进冻结面**）")
    A("   ⇒ **可追溯性靠本卡自身承载**（`PROJECT_CONTRACT.md` §2-7）。")
    A("   **不新增冻结 `reports/` 子目录下的文件**（`DIRECTORY.md` §3.3 的结构为权威）。")
    A("④ **口径声明**：**本项目的 OOD 轴是编码而非语义**（`PREREG_v1.md` §2.2：「恶意语义类别在")
    A("   训练集见过；**不是语义 OOD**」）⇒ 语义族跨档**不影响 OOD 结论**，但**降低测试集的「行为")
    A("   覆盖度」**（可由 ③ 的数字读出）。")
    A("")

    # ---- 17 ----
    A("## §17 `N = %d` 的算术依据、配比表与子集规则原文"
      % facts["seeds_meta"].get("seed_count_total", 0))
    A("")
    meta = facts["seeds_meta"]
    by_src = meta.get("seed_count_by_source", {})
    A("**（a）算术依据**：`N = %d = AgentHarm %d + PKU-SafeRLHF %d`；"
      % (meta.get("seed_count_total", 0),
         by_src.get("AgentHarm", {}).get("count", 0),
         by_src.get("PKU-SafeRLHF", {}).get("count", 0)))
    A("**`%d × 2 = %d`** 恰中 `V-1`（`rewrite_index` 恒 ∈ {1,2}，见 `V-10`）。"
      % (meta.get("seed_count_total", 0), 2 * meta.get("seed_count_total", 0)))
    A("定值时点**早于任何数据生成**。")
    A("")
    A("**（b）配比表**：")
    A("")
    tbl(["来源", "条数", "占比", "说明"], [
        ["AgentHarm", by_src.get("AgentHarm", {}).get("count", 0),
         by_src.get("AgentHarm", {}).get("ratio", 0.8), "**全量入选**，非抽样"],
        ["PKU-SafeRLHF", by_src.get("PKU-SafeRLHF", {}).get("count", 0),
         by_src.get("PKU-SafeRLHF", {}).get("ratio", 0.2),
         "按严重度筛选后确定性抽样"],
    ])
    A("配比来源：`configs/data/seed_collect.yaml` 的 `sample.n = %d` 与取值理由；"
      % by_src.get("PKU-SafeRLHF", {}).get("count", 0))
    A("AgentHarm 为**全量入选**（`harmful_behaviors_test_public.json` 恰 %d 条）。"
      % by_src.get("AgentHarm", {}).get("count", 0))
    A("")
    A("**（c）子集规则原文**（`seeds.meta.json` 逐字照录）：")
    A("")
    for key, label in (("subset_rule", "`subset_rule`"),
                       ("subset_rule_pku", "`subset_rule_pku`"),
                       ("subset_rule_agentharm", "`subset_rule_agentharm`")):
        A("- %s：%s" % (label, str(meta.get(key, "—")).replace("\n", " ")))
    A("")

    # ---- 18 ----
    A("## §18 `V-10`（**已并入 §14**）")
    A("")
    A("`V-10` = 由 `V-1`（440 硬总数）与实测 `N = %d` **派生**的约束——`rewrite_index` 恒 ∈ {1,2}，"
      % facts["seeds_meta"].get("seed_count_total", 0))
    A("**每组恰 2 条改写**（`PREREG_v1.md` §6 的「2–3 条」范围内取**下界**，合法）。")
    A("**性质：派生，非新建口径。**")
    A("")

    # ---- 19 ----
    A("## §19 改写阶段的「非逐字节可复现」披露（`W3.2` 实测）")
    A("")
    A("1. `deepseek-v4-pro` 在 `temperature=0` 下**并非完全确定性**（`W3.2` 复测 3 组，")
    A("   **差异率 1.0000**，见 `reports/data/g4_intent_v1.0_20260930-220108.md` §1.2）")
    A("   ⇒ **改写阶段的产物不可由重跑复现**。")
    A("2. **冻结的 `splits/*.csv` 的 `plaintext` 列即 440 条改写体的唯一权威副本**")
    A("   （440 条全部落于分片，无遗漏）。")
    A("3. 可追溯性由「改写体 ＋ 组缓存指纹 ＋ 配置 sha256 ＋ 模板 sha256 ＋ system sha256」")
    A("   **四元组**承载；组缓存 `data/interim/rewrite/_groups/` **不进冻结面** ⇒")
    A("   **可追溯性实际由本卡自身的指纹表承载**（`PROJECT_CONTRACT.md` §2-7）：")
    A("")
    tbl(["指纹", "值"], [
        ["`prompt_template_sha256`", "`5142b0ab0365dc1be28f78fa8103cf6301f229b7ee4db27ba405972d12d504d7`"],
        ["`prompt_system_sha256`", "`4ca94ca17ca8544f90c228073a200642854c298f7234503df2e5ede2ba0a2276`"],
        ["`configs/data/rewrite.yaml` sha256", "`97914337067346db`（报告 §1 登记前缀）"],
        ["模型 id（请求／响应回显）", "`deepseek-v4-pro` ／ `deepseek-v4-pro`"],
        ["端点", "`https://api.deepseek.com`"],
        ["`temperature` / `top_p` / `max_new_tokens`", "`0` / `1` / `4096`（**非 512**，见 §21）"],
        ["`refusal_count`", "**0**"],
    ])
    A("")

    # ---- 20 ----
    A("## §20 `c3.mode` 的口径与「24 个 group 天然含编码形态」")
    A("")
    A("1. **强制口径为 `c3.mode = introduced`** —— **只拒改写器新引入的**编码形态，豁免父种子已含者。")
    A("2. 逐字口径下的**信息性命中**：220 条种子中 **24 条（10.9%）天然含** 20+ 位")
    A("   `[A-Za-z0-9+/=]` 串、**8 条**含 16+ 位十六进制串；")
    A("   **⚠ 两者非互斥，受影响的不同 group 数 = 24**（8 条十六进制命中**含于**该 24 条之内；")
    A("   报告 §3.1 的「命中候选 32 条」是**模式命中次数**，**不是** 32 个 group）。")
    A("3. **取 `introduced` 的理由**：逐字口径下这 24 个 group 因「逐字保留具体细节」要求而")
    A("   **必然全候选被拒**，`V-1` 的 440 **不可达**（`PREREG_v1.md` §6 冻结值不得自行改）")
    A("   ⇒ 取 `introduced` 是**按 `W3.2` §0 原话「若改写器**自发加了**编码或代码块」的最贴合解释**，")
    A("   **非事后放宽**。")
    A("")

    # ---- 21 ----
    A("## §21 `role=system` 任务说明")
    A("")
    A("1. `prompt_system_sha256 = 4ca94ca1…`（全文见 §19 表）。**其作用＝消除误拒**：")
    A("   不加时 `SG-0165` 等被直接拒答，原样证据在 `reports/data/g4_intent_v1.0_…md` §5.1／§3.2。")
    A("2. **偏差方向声明**：加 system 段后 `refusal_count = 0` ⇒ **改写集不含「因拒答而缺失」的种子**；")
    A("   若不加，拒答将按种子的敏感度**非随机**发生 ⇒ 引入**选择偏差**。")
    A("   ⇒ 本项目的取向是**消除该偏差并如实报出 `refusal_count`**。")
    A("3. 该段**不含**任何冻结件阈值／rubric 标签（`W3.2` §3.2 ③ 仍满足）。")
    A("")

    # ---- 22 ----
    A("## §22 `max_new_tokens = 4096` 与 `PREREG_v1.md` §5 的「512」是**两回事**")
    A("")
    A("| 项 | 值 | 作用域 |")
    A("|---|---|---|")
    A("| `PREREG_v1.md` §5「下游 SLM 统一最大生成长度」 | **512** tokens | **阶段5.2 ASR 的生成口径**"
      "（`D-00-03` `W0-A2`，§12.3） |")
    A("| `configs/data/rewrite.yaml` 的 `max_new_tokens` | **4096** | **阶段2 改写器的输出上限** |")
    A("")
    A("**不得把两者等同或互相引用。** 取 4096 的原因：`deepseek-v4-pro` 为推理模型，")
    A("`reasoning_tokens` 计入 `completion_tokens`，**512 会把预算全吃光致 `content` 为空**。")
    A("该区分**同时**写在本卡与 `configs/data/rewrite.yaml` **两处**。")
    A("")

    # ---- 23 ----
    A("## §23 门禁 1 的判定结果与「代录」性质（**内联，因抽样表不进冻结面**）")
    A("")
    A("抽样表 `data/interim/rewrite/intent_consistency_sample.csv`（50 行，")
    A("sha256 `4a505b0456ce418c8a8feb5d162d422df0015a0894bbc098d8760bea01cc31c6`）")
    A("位于 `data/interim/`，已被 `.gitignore` 忽略、**不进版本控制、不进冻结面**"
      "（`DIRECTORY.md` §3.2「禁止封存」）。")
    A("")
    A("1. 门禁 1 **结果**：通过率 **%.4f**（50/50），阈值 **≥%.2f**，状态 **PASS**；"
      % (g.get("g1_intent", {}).get("rate", 1.0),
         g.get("g1_intent", {}).get("threshold", 0.95)))
    A("   抽样方式：按 `rewrite_id` 升序**等距**，**非随机**。")
    A("2. **核对人 `USER`**、**核对日期 2026-10-01**。")
    A("3. **代录声明（不得省略）**：判定由**用户本人**给出，**W3 仅转录、未作判定**，")
    A("   `remark` 为**统一文本、无逐条理由** ⇒ 本表是「判定已发生」的证据，")
    A("   **不是**「判定过程可复核」的证据。全文见 "
      "`reports/data/g4_intent_v1.0_20261001-002637.md` **§8**。")
    A("4. **核对范围＝双向**（用户于 **2026-10-01 确认**，**已闭合**，非未闭合项）：")
    A("   门禁 1 的 PASS **同时**成立于 ① **预注册判据**（`PREREG_v1.md` §5 补录 G4：")
    A("   「与原始种子恶意意图保持一致，**没有弱化/改变攻击目标**」）与 ② **`W3` 追加判据**")
    A("   （**双向**：**弱化／换目标** 与 **强化／具体化** 均已核对）。")
    A("5. **`H1` 的判据即本条的门禁结果。**")
    A("")

    # ---- 24 / 25 / 27 / 28 ----
    A("## §24 `G3` 编码分布不均衡（**逐编码披露**）")
    A("")
    tbl(["编码", "G3 条数", "攻击集（对照）"],
        [[("`%s`" % e), facts["g3_encoding"].get(e, 0), facts["encoding_attack"].get(e, 0)]
         for e in ("hex", "base64", "url", "ascii_art", "morse", "homoglyph",
                   "zerowidth", "rot13", "reverse")])
    A("**成因**：真实代码天然携带的编码以 **hex 摘要／base64** 为主（客观属性），合成侧按 20 场景均分。")
    A("**影响**：G3 的 FPR 由 `hex`／`base64` 主导；读者若按「编码均匀」解读会失真。")
    A("**显式声明**：**G3 的编码分布不代表 9 编码均匀分布**，须与门禁 3 的 FPR **一并解读**。")
    A("**另须登记（同批，非缺陷）**：**G3 组内冗余 %d / %d**"
      % (facts["g3_intra_group_dup"], facts["g3_rows"]))
    A("（同组同文本，**跨组 0 ⇒ 非泄露**，系 `r4.1①` 首选处置的预期结果）。")
    A("")

    A("## §25 门禁 4 的预处理口径（**跨阶段对齐锚点**）")
    A("")
    A("本卡登记门禁 4 的**实测口径**（`W3.5` `benign_guard_check.csv`）：检测器 `llamaguard3-8b`")
    A("rev `7327bd9f…`、4-bit、加载照 `ENV_LOCK.md` §5／§6；抽样 G1／G2／G3 **各 30 条**、")
    A("按 `sample_id` 升序步长 150／90／60；**判词解析 = 首行首词**；**截断侧 = 左**（上限 512 token）；")
    A("命中 1／90；`unparsed` 0；**计数口径 = 只按内容计数**（`D-00-05`：`MI-M` 计入，")
    A("`MI-W`／`MI-B` 不计入；G3 先还原再送检，G1／G2 原样送检）。")
    A("")
    A("**用途**：**`W4` 的 B5 若沿用同口径须显式声明，若改判须写明差异**——否则「门禁 4 的检出率」")
    A("与「B5 的指标」测量口径不一致，读者无法对齐。**该条不制定 B5 的口径**（B5 归 W4），")
    A("只把门禁 4 的口径**固定下来可对照**。（转交项 40）")
    A("")

    A("## §26 门禁 4 的人工复核结果（**内联，因复核载体不进冻结面**）")
    A("")
    A("人工复核落于 `data/interim/benign_G1_G2_G3/benign_guard_review.csv`（**9 行**；")
    A("sha256 `aa6e66d5afc908a7185857d6f2a7de4088258941aab7a69080a2c0b35019d627`，")
    A("**与 `G3.csv` `727e820f…` 同代**），与该目录其余产物一样**不进版本控制、不进冻结面**。")
    A("")
    A("1. **门禁 4 结果**：人工复核后 `counted = **0** / 90`、检出率 **0.0000 ≤ 0.05** ⇒ **通过**")
    A("   （**不是** `02_05` 机械赋值下的 6/90；两者**分别标明**，避免读者取错值）。")
    A("2. **复核者 `USER`**、**复核日期 2026-10-01 与 2026-10-02**；**9 条** `S1`／`S8`／`S14` 命中")
    A("   **全部改判 `MI-B`**，判定路径均为 `Q1是→Q2否→Q5否`（其中 `BN-G3-0661` 为 `r6` 后新增命中；")
    A("   `0241`／`0481`／`0541`／`0721` 四条系**按现行文本重新判定并重新绑定指纹**）。")
    A("3. **代录声明（不得省略）**：判定由**用户**给出，**W3 仅代录**；逐条理由**不引样本原文**")
    A("   （依 `g3_hard_negative.md` §7-8 的精神）。")
    A("4. **口径**：门禁 4 的判定量＝**人工按 `malicious_intent.md` §4 赋 MI 标签后的 `MI-M` 计数**，")
    A("   **不是** LG-8B 的原始命中率（见 `reports/data/g3_qc_v1.0_20261001-210403.md` §5）。")
    A("5. **送检／截断口径（局限，与结果同批登记）**：见 §13-5。")
    A("6. `H4` 的判据为**复核后**的 `counted`（即 `benign_guard_check.csv` 经补正轮 `r3`／`r6`")
    A("   读入人工值后的那一版）。")
    A("7. **复核表须与 `G3.csv` 同代**：**G3 一经变更，抽样内容即变** ⇒ 复核表须**按现行文本")
    A("   重新判定并重新绑定 `text_sha256`**；未重绑者按 `02_05` 的 `unbound` 处理")
    A("   （**视为未复核、非零退出，严禁默认 `MI-M`**）。**G3 变更后不得沿用上一代的复核结论。**")
    A("")

    # ---- 28 ----
    A("## §28 分片代的唯一权威 ＋ 冻结侧口径")
    A("")
    A("1. 冻结与断言**只认 `data/interim/split_checks/` 顶层**（= `W3.6` **`r2` 原子化**、")
    A("   `family_atomic: true` ＋ `--stratify-length` 的产物）；`pre_atomic_run/` 与")
    A("   `r2_probe_false_path/` **仅供留档，不得被冻结或下游引用**。")
    A("2. `H5` 的 `exit=0` **已与 stdout 编码无关**（`W3.7` 补正轮 `r2` 修 `F-1`：")
    A("   `init_streams()` 重配 stdout／stderr 为 UTF-8）⇒ 重定向取证**不会再假失败**。")
    A("3. `02_08` **不自行重算门禁 5／6／7**，只调用 `02_07` 并读其产物与退出码（`H5` 原注不变）。")
    A("")

    # ---- 29 ----
    A("## §29 门禁 7 改为报告项后的**四项披露**（`D-36` 裁定一／二）")
    A("")
    A("**① 门禁 7 的 `n/a` 判定与三项证据**：**阈值 `0.05` 一字未改**、**不参与退出码**；")
    A("原子化后实算 `regime_id` **%.6f**、`regime_ood` **%.6f**；三项证据 ="
      % (g.get("g7_ngram", {}).get("regime_id", 0), g.get("g7_ngram", {}).get("regime_ood", 0)))
    A("(a) 阈值与实测（上表）；(b) **② 逐编码对半控制**；(c) **③ 随机对半控制**"
      "（`ngram_controls.csv`，`current` 分片：")
    ctrl = {(r["regime"], r["control"]): r for r in facts["ngram_controls"]
            if r.get("run") == "current" and r.get("split") == "test"}
    A("`regime_id/test` **%s**、`regime_ood/test` **%s**）。"
      % (ctrl.get(("regime_id", "random"), {}).get("overlap_rate", "—"),
         ctrl.get(("regime_ood", "random"), {}).get("overlap_rate", "—")))
    A("**须写明该代理量不具判别力的原因**：同分片、构造上零泄露的两半即达 **0.87–0.89**，")
    A("与 `train→test` 同量级 ⇒ 该量**对「跨分片」与「同分布」不可区分**；")
    A("残余交集来自语言／代码公共 3-gram 与 `02_04` 的 **10 份宿主骨架**。")
    A("")
    A("**② `family_atomic: true` 的口径变更** ＋ `family_straddle` **跨档族 = 0**（原子化前为 **17**）。")
    A("")
    A("**③ 原子化前后对照**：攻击 group **176 / 22 / 22** 与 `test` 攻击行 **792** **均未变**；")
    A("`A6` = **440**、`A7` = **7920** **均未变**。")
    A("")
    A("**④ 代价：行为多样性 `20 → 7`**（`test` 分片按来源计的行为数）——**不是**「测试集缩小 7 倍」。")
    A("")
    A("**配套**：三项证据的产物为 `data/interim/split_checks/{ngram_overlap.csv, ngram_controls.csv}`；")
    A("`A14` 的**报告性**地位（`passed` 恒 `n/a`）已在 §3 字段说明中写明。")
    A("")

    # ---- 30 ----
    A("## §30 `PREREG_v1.3` 的三处数值更正（**并列登记**）")
    A("")
    A("| 项（原子化后） | `PREREG_v1.3` §2④／§3.2 所记 | **实测**（`W3` 复算 == `W3.7` r1 §10.5） |")
    A("|---|---|---|")
    A("| 最近邻 **中位数** | 0.3560 | **0.3880** |")
    A("| 最近邻 **max** | 0.3904 | **0.5026** |")
    A("| 最近邻 **≥0.5 占比** | 0.0000 | **0.0227（= 1/44）** |")
    A("")
    A("**根因**：`PREREG_v1.3` 所引为 `W3` 呈报时的**剔除式代理**（剔除与 `train` 同族的 `test` 行")
    A("⇒ 仅剩 **3** 个 group ⇒ 「占比 0」是**构造产物**；`W3` 呈报时未声明其为代理）；")
    A("**真实重划分是整族重排**，`test` 仍 **792** 行 ⇒ 个别**跨族**最近邻仍可 ≥0.5。")
    A("**方向与结论不变**（**86.36% → 2.27%**，实质泄露已消除）。")
    A("**按 `D-12` 不改 `PREREG_v1.3`**，承接只在本卡与论文。")
    A("")
    A("**同批（并列登记并标注分片代，不得只留一组、亦不得以「不可复现」为由删除前一组）**：")
    A("")
    tbl(["组", "分片代", "值"],
        [[("`%s`" % r["key"] if r.get("key") and r["key"] != "-" else "整体对半"),
          "`%s`（`ngram_controls.csv` 的 `run` 列）" % r.get("run", "—"),
          r["overlap_rate"]]
         for r in facts["ngram_controls"]
         if r.get("control") == "by_encoding" and r.get("regime") == "regime_id"
         and r.get("split") == "test"])
    A("`PREREG_v1.3` §1.2 ② 的五值（`base64` 0.9177／`rot13` 0.8336／`morse` 0.9774／")
    A("`ascii_art` 0.9799／`zerowidth` 0.9216，**`pre_atomic` 分片**）**可复现**（`W3` 已 **5/5** 复现，")
    A("复现规则见 `W3.7` r1 报告 §10.4 更正段）；上表为**同一 `by_encoding` 控制**在当前分片代的实测值。")
    A("")
    A("> **待裁**：是否需 `PREREG_v1.4` 承接上述更正 —— `D-37` 裁定二已定 **需要**（转交项 44，")
    A("> **时点在冻结之后**，不阻塞本冻结）。按本条要求，**无论裁定如何，并列登记不变**。")
    A("")

    # ---- 31 ----
    A("## §31 `g3_hard_negative_test.csv` 的体制二子集取法")
    A("")
    A("文件全量 **%d 行 / %d group**；**体制一 = 文件全量**；"
      % (facts["g3_hard_negative_rows"], facts["g3_hard_negative_groups"]))
    A("**体制二 = 按 `benign_group_id` ∩ 该体制 `test` 良性 group 集** ⇒ "
      "**%d 行 / %d group**（%s）。"
      % (facts["g3n_ood_rows"], facts["g3n_ood_groups"],
         "／".join("`%s` %d" % (k, v) for k, v in sorted(facts["g3n_ood_encoding"].items()))))
    A("")
    A("⚠ **不得**按 `encoding_type ∈ 留出 4 码` 过滤（会得错误行数）：`ood ⊆ id` 是**集合包含**，")
    A("**不蕴含**「按编码过滤 id 即得 ood」（`W3.6` `F-1`，`r1` 已更正）。")
    A("")
    A("---")
    A("")
    A("*本数据卡随 `data-v1.0` 冻结；任何修改必须新建 `CodeObf-PI_v1.1`。*")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def render_report(*, stamp, cfg, h_results, conclusion, problems, written, md5_reports,
                  data_card_md5, manifest_md5, root, notes) -> str:
    L = []
    A = L.append
    A("# freeze_v1.0 — 阶段2 数据集冻结运行报告")
    A("")
    A("- 窗口：W3.8（全流程冒烟与冻结，权限 **A3**）")
    A("- 日期：%s" % stamp)
    A("- 脚本：`%s`" % cfg["freeze_script"])
    A("- 配置：`configs/data/freeze.yaml`")
    A("- 数据集：**%s**（`%s`，`%s`）" % (cfg["dataset"], cfg["name"], cfg["version"]))
    A("- 结论：**%s**" % conclusion)
    A("")
    A("## §1 冻结前置 `H1`–`H7`（**先全查、后写入**）")
    A("")
    A("| # | 前置 | 判定 | 实测 |")
    A("|---|---|---|---|")
    rank = {k: i for i, k in enumerate(CHECK_ORDER)}
    for key, ok, detail, _extra in sorted(h_results, key=lambda r: rank.get(r[0], 99)):
        A("| `%s` | %s | %s | %s |" % (key, H_LABELS[key], "**PASS**" if ok else "**FAIL**",
                                       detail.replace("\n", "<br>")))
    A("")
    if problems:
        A("**未通过项 %d 个**（**未写入任何文件**）：" % len(problems))
        A("")
        for p in problems:
            A("- %s" % p)
        A("")
        A("> 依 `windows/W3.8_freeze.md` §3.1：任一条不满足 ⇒ 非零退出、**不做任何写入**。")
        A("")
        return "\n".join(L) + "\n"
    A("> 全部 `H` 通过，**此后**才开始写文件（「先检查后写入」）。")
    A("")
    A("## §2 冻结树（逐文件清单：路径 + 字节数 + md5）")
    A("")
    A("| 相对路径（`data/frozen/CodeObf-PI_v1.0/`） | 字节数 | md5 |")
    A("|---|---|---|")
    for rel, size, md5 in written["frozen"]:
        A("| `%s` | %d | `%s` |" % (rel, size, md5))
    A("")
    A("## §3 封存测试集（逐文件清单）")
    A("")
    A("| 相对路径（`data/sealed_test/CodeObf-PI_v1.0/`） | 字节数 | md5 |")
    A("|---|---|---|")
    for rel, size, md5 in written["sealed"]:
        A("| `%s` | %d | `%s` |" % (rel, size, md5))
    A("")
    A("## §4 两份 `md5sums.txt` 的自校验")
    A("")
    A("| 根目录 | 条数 | 自校验 | 自身 md5 |")
    A("|---|---|---|---|")
    for label, info in md5_reports.items():
        A("| `%s` | %d | %s | `%s` |"
          % (label, info["count"], "**通过（0 处不符）**" if not info["bad"]
             else "**失败：%s**" % info["bad"][:3], info["md5"]))
    A("")
    A("## §5 关键产物")
    A("")
    A("| 件 | 路径 | md5 |")
    A("|---|---|---|")
    A("| `DATA_CARD` | `data/frozen/CodeObf-PI_v1.0/DATA_CARD_CodeObf-PI_v1.0.md` | `%s` |"
      % data_card_md5)
    A("| `manifest.json` | `data/frozen/CodeObf-PI_v1.0/manifest.json` | `%s` |" % manifest_md5)
    A("")
    A("### §5.1 `manifest.json` 的 `counts` 与 `splits` 摘要")
    A("")
    A("```json")
    A(json.dumps({"counts": notes["manifest"]["counts"],
                  "splits": notes["manifest"]["splits"],
                  "split_ratio": notes["manifest"]["split_ratio"],
                  "split_seed": notes["manifest"]["split_seed"]},
                 ensure_ascii=False, indent=2))
    A("```")
    A("")
    A("## §6 `manifest.json` 的 `gates` 段（逐门禁实算值）")
    A("")
    A("```json")
    A(json.dumps(notes["manifest"]["gates"], ensure_ascii=False, indent=2))
    A("```")
    A("")
    A("## §7 只读权限")
    A("")
    A("| 平台 | 做法 |")
    A("|---|---|")
    A("| Windows（本机） | `os.chmod(path, stat.S_IREAD)`（`0o400`）—— **平台差异**:"
      "Windows 只映射「只读属性」，不等于 POSIX 的 `0o444`（无组／其他位概念） |")
    A("| POSIX | `os.chmod(path, 0o444)` |")
    A("")
    A("**例外（显式登记）**：`data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md` **不置只读**——")
    A("该文件按 `DIRECTORY.md` §3.4 由 A4/A6/A7 评估窗口在读取前**追加**记录；置只读会使该流程不可执行。")
    A("`02_08` **不写其任何数据行**，只校验表头恰 6 列。")
    A("")
    A("## §8 声明")
    A("")
    A("1. **全部门禁 1–7 的结论取自上游产物**：`H1`–`H4` 读文件实算，`H5` 调 `02_07` 并读其退出码与")
    A("   `split_assertions.csv`（**不自行重算门禁 5／6／7**，避免两处实现漂移）。")
    A("2. **未放宽任何阈值、未剔除任何不利样本、未改任何口径。**")
    A("3. 冻结后 `data/frozen/` 与 `data/sealed_test/` 即只读；**本窗口此后不再读 `data/sealed_test/`**。")
    A("4. `02_08` 未写 `ACCESS_LOG.md` 的任何数据行。")
    A("")
    return "\n".join(L) + "\n"


CHECK_ORDER = ("V3", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H15", "HC", "HA")

H_LABELS = {
    "V3": "`V-3` 守卫 · 划分比例 8:1:1 + 种子 42",
    "H15": "§3.6-15 · `splits` 与 `A13` 一致",
    "HC": "§3.3 · 冻结硬计数 440／7920（并与 `A6`／`A7` 一致）",
    "HA": "`ACCESS_LOG.md` 表头恰 6 列（只校验，不写数据行）",
    "H1": "门禁 1 · 改写意图一致性 ≥95%",
    "H2": "门禁 2 · AST 报错率 ≤10%",
    "H3": "门禁 3 · G3 质检通过率 ≥95%（双人）",
    "H4": "门禁 4 · 良性集检出率 ≤5%",
    "H5": "门禁 5／6 · 重跑 `02_07` 默认模式",
    "H6": "`02_00 --full` 已通过",
    "H7": "目标目录可写且为占位态或不存在",
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def freeze(root: str, cfg: dict, no_readonly: bool = False, h5_rerun: bool = True) -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    frozen_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    print("== freeze: %s ==" % cfg["dataset"])
    print("  root  : %s" % root)
    print("  H1–H7 检查中（**先全查、后写入**）…")

    gates = {}
    h_results, problems = [], []

    # `V-3` 守卫（配置完整性；非新门禁）——必须在任何写入之前
    ok_v3, detail_v3 = check_v3(cfg)
    h_results.append(("V3", ok_v3, detail_v3, {}))
    print("  [%s] V3  %s" % ("PASS" if ok_v3 else "FAIL", detail_v3))
    if not ok_v3:
        problems.append("V3 守卫未通过：%s" % detail_v3)

    for key, fn in (("H1", check_h1), ("H2", check_h2), ("H3", check_h3),
                    ("H4", check_h4), ("H6", check_h6), ("H7", check_h7),
                    ("H15", check_splits_vs_a13), ("HC", check_counts),
                    ("HA", check_access_log)):
        ok, detail, extra = fn(root, cfg)
        h_results.append((key, ok, detail, extra))
        if key in GATE_KEY:
            gates[GATE_KEY[key]] = extra
        print("  [%s] %s  %s" % ("PASS" if ok else "FAIL", key, detail))
        if not ok:
            problems.append("%s 未通过：%s" % (key, detail))

    # 门禁 6 的实算值（供 manifest／DATA_CARD；来自 `02_07` 的产物，非自行重算）
    length_path = os.path.join(root, cfg["inputs"]["length_distribution"])
    if os.path.isfile(length_path):
        shards = [r for r in _read_csv(length_path) if r.get("sample_kind") == "shard"]
        if shards:
            worst = max(shards, key=lambda r: float(r["abs_diff_ratio"]))
            gates["g6_length"] = {"threshold": float(cfg["gates"]["g6_length"]["threshold"]),
                                  "worst": float(worst["abs_diff_ratio"]),
                                  "worst_at": "%s/%s" % (worst["regime"], worst["split"])}
    ng_path = os.path.join(root, cfg["inputs"]["ngram_overlap"])
    if os.path.isfile(ng_path):
        vals = {}
        for r in _read_csv(ng_path):
            if r.get("level") == "char" and str(r.get("n")) == "3" and r.get("passed") == "n/a":
                vals[r["regime"]] = float(r["overlap_rate"])
        gates["g7_ngram"] = {"threshold": float(cfg["gates"]["g7_ngram"]["threshold"]),
                             "role": "report_item",
                             "regime_id": vals.get("regime_id"),
                             "regime_ood": vals.get("regime_ood")}

    # H5 最后跑（唯一有副作用者；其副作用已被逐文件 sha256 锁死）
    ok5, detail5, extra5 = check_h5(root, cfg, run=h5_rerun)
    h_results.insert(4, ("H5", ok5, detail5, extra5))
    gates[GATE_KEY["H5"]] = extra5
    print("  [%s] %s  %s" % ("PASS" if ok5 else "FAIL", "H5", detail5))
    if not ok5:
        problems.append("H5 未通过：%s" % detail5)

    if problems:
        print("")
        print("FAILED: 冻结前置未通过（%d 项）—— **未写入任何文件**" % len(problems))
        for p in problems:
            print("    - %s" % p)
        return EXIT_FAIL

    # ---------------- 以下才开始写入 ----------------
    fdir = os.path.join(root, cfg["outputs"]["frozen_dir"])
    sdir = os.path.join(root, cfg["outputs"]["sealed_dir"])
    inp = cfg["inputs"]
    print("  前置全通过 ⇒ 开始写入 %s" % _rel(fdir, root))

    written = {"frozen": [], "sealed": []}

    # 1) 复制分片
    for reg in REGIMES:
        for sp in SPLITS:
            src = os.path.join(root, inp["splits_root"], reg, "%s.csv" % sp)
            dst = os.path.join(fdir, "splits", reg, "%s.csv" % sp)
            _copy_file(src, dst)
            written["frozen"].append(("splits/%s/%s.csv" % (reg, sp),
                                      os.path.getsize(dst), _md5_file(dst)))
    # 2) 良性集
    for g in ("G1", "G2", "G3"):
        src = os.path.join(root, inp["benign"], "%s.csv" % g)
        dst = os.path.join(fdir, "benign", "%s.csv" % g)
        _copy_file(src, dst)
        written["frozen"].append(("benign/%s.csv" % g, os.path.getsize(dst), _md5_file(dst)))
    # 3) 分组表
    src = os.path.join(root, inp["seed_groups"])
    dst = os.path.join(fdir, "seed_groups.csv")
    _copy_file(src, dst)
    written["frozen"].append(("seed_groups.csv", os.path.getsize(dst), _md5_file(dst)))
    # 4) reports/ 四份
    for name, src in (("split_assertions.csv", os.path.join(root, inp["split_assertions"])),
                      ("length_distribution.csv", os.path.join(root, inp["length_distribution"])),
                      ("ngram_overlap.csv", os.path.join(root, inp["ngram_overlap"])),
                      ("g3_qc.csv", os.path.join(root, inp["g3_qc"]))):
        dst = os.path.join(fdir, "reports", name)
        _copy_file(src, dst)
        written["frozen"].append(("reports/%s" % name, os.path.getsize(dst), _md5_file(dst)))

    # 5) 事实采集（从**冻结树**；此时 DATA_CARD／manifest 尚未生成）
    facts = collect_facts(root, cfg, gates)
    missing_src = [k for k, v in facts["source_manifests"].items() if not v["sha256"]]
    if missing_src:
        print("FAILED: 缺少来源清单 SOURCE_MANIFEST.json：%s —— DATA_CARD 无法完整生成"
              % missing_src)
        return EXIT_FAIL

    tree_sha256 = hashlib.sha256(
        "".join("%s %s\n" % (rel, md5) for rel, _s, md5 in written["frozen"]).encode("utf-8")
    ).hexdigest()

    # 6) DATA_CARD（**在生成时即写入「事前定值」节**，不留待事后补写）
    data_card_path = os.path.join(fdir, "DATA_CARD_CodeObf-PI_v1.0.md")
    sealed_files = sorted(cfg["sealed"].keys() - {"access_log", "access_log_header_cols",
                                                  "access_log_header"})
    manifest = build_manifest(cfg, facts, frozen_at, sealed_files, root)
    # md5 自指问题：DATA_CARD 需内联两份 md5sums 与 manifest 的 md5 ⇒ 先写 manifest，
    # 再算 DATA_CARD（其中 manifest md5 为占位符），最后回填？——不：DATA_CARD **不**内联自身 md5，
    # 只内联 manifest 与两份 md5sums 的 md5 ⇒ 顺序为 manifest → sealed md5sums → frozen md5sums → DATA_CARD。
    manifest_path = os.path.join(fdir, "manifest.json")
    _write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest_md5 = _md5_file(manifest_path)

    # 7) 封存测试集（**仅此一次生成**）
    for name, src in sorted(cfg["sealed"].items()):
        if not name.endswith(".csv"):
            continue
        dst = os.path.join(sdir, name)
        _copy_file(os.path.join(root, src), dst)
        written["sealed"].append((name, os.path.getsize(dst), _md5_file(dst)))

    # 8) ACCESS_LOG：表头已在写入前由 `HA` 校验；此处只取 sha256 快照，
    #    收尾时复核**未被改动**（本脚本不得写其任何数据行）。
    alog = os.path.join(sdir, cfg["sealed"]["access_log"])
    alog_before_sha = _sha256_file(alog)

    # 9) 两份 md5sums.txt
    sealed_lines = build_md5sums(sdir, [r for r, _s, _m in written["sealed"]])
    sealed_md5_path = os.path.join(sdir, "md5sums.txt")
    _write_text(sealed_md5_path, "\n".join(sealed_lines) + "\n")

    frozen_rel = list_frozen_files(fdir)
    frozen_lines = build_md5sums(fdir, frozen_rel)
    frozen_md5_path = os.path.join(fdir, "md5sums.txt")
    _write_text(frozen_md5_path, "\n".join(frozen_lines) + "\n")
    frozen_md5_self = _md5_file(frozen_md5_path)
    sealed_md5_self = _md5_file(sealed_md5_path)

    # 10) DATA_CARD（内联 manifest 的 md5 与封存侧 md5sums 的 md5；**不内联自身**、
    #              也不内联冻结侧 md5sums 的自身 md5 —— 那是循环依赖，见卡内 §11 的自指说明）
    # 冻结侧 md5sums 的**最终**条数 = 已复制件数 + manifest.json + 本数据卡
    # （不能数目录：仓库自初始化起就有一套同名的 0 字节占位件，数目录会把占位件重复计入）
    expected_md5_count = len(written["frozen"]) + 2
    data_card = render_data_card(cfg, facts, frozen_at, expected_md5_count, sealed_md5_self,
                                manifest_md5, tree_sha256, root)
    _write_text(data_card_path, data_card)
    data_card_md5 = _md5_file(data_card_path)

    # DATA_CARD 写入后 md5sums 须重算（它覆盖全部 MD）
    frozen_rel = list_frozen_files(fdir)
    frozen_lines = build_md5sums(fdir, frozen_rel)
    if len(frozen_lines) != expected_md5_count:
        print("FAILED: 冻结侧 md5sums 条数 %d ≠ 预期 %d（冻结树与契约不符）"
              % (len(frozen_lines), expected_md5_count))
        return EXIT_FAIL
    _write_text(frozen_md5_path, "\n".join(frozen_lines) + "\n")
    frozen_md5_self = _md5_file(frozen_md5_path)

    written["frozen"].append(("DATA_CARD_CodeObf-PI_v1.0.md",
                              os.path.getsize(data_card_path), data_card_md5))
    written["frozen"].append(("manifest.json", os.path.getsize(manifest_path), manifest_md5))
    written["frozen"].append(("md5sums.txt", os.path.getsize(frozen_md5_path), frozen_md5_self))

    # 11) 自校验（各在对应根目录下）
    md5_reports = {}
    bad_f = verify_md5sums(fdir, frozen_lines)
    md5_reports["data/frozen/CodeObf-PI_v1.0/"] = {
        "count": len(frozen_lines), "bad": bad_f, "md5": frozen_md5_self}
    bad_s = verify_md5sums(sdir, sealed_lines)
    md5_reports["data/sealed_test/CodeObf-PI_v1.0/"] = {
        "count": len(sealed_lines), "bad": bad_s, "md5": sealed_md5_self}
    if bad_f or bad_s:
        print("FAILED: md5sums 自校验未通过：%s" % (bad_f + bad_s)[:5])
        return EXIT_FAIL

    # 12) 只读
    if no_readonly:
        print("  [WARN] --no-readonly：跳过只读设置（仅供测试）")
    else:
        mode = stat.S_IREAD if os.name == "nt" else 0o444
        for rel, _s, _m in written["frozen"]:
            os.chmod(os.path.join(fdir, rel.replace("/", os.sep)), mode)
        for rel, _s, _m in written["sealed"]:
            os.chmod(os.path.join(sdir, rel), mode)
        os.chmod(sealed_md5_path, mode)

    if _sha256_file(alog) != alog_before_sha:
        print("FAILED: ACCESS_LOG.md 被改动（不得写其数据行）")
        return EXIT_FAIL

    # 13) 报告
    notes = {"manifest": manifest}
    report_dir = os.path.join(root, cfg["outputs"]["report_dir"])
    report_path = os.path.join(report_dir, "%s_%s.md" % (cfg["outputs"]["report_basename"], stamp))
    _write_text(report_path, render_report(
        stamp=stamp, cfg=cfg, h_results=h_results, conclusion="PASS", problems=[],
        written=written, md5_reports=md5_reports, data_card_md5=data_card_md5,
        manifest_md5=manifest_md5, root=root, notes=notes))
    if not no_readonly:
        os.chmod(report_path, stat.S_IREAD if os.name == "nt" else 0o444)

    print("")
    print("  冻结树 %d 个文件；封存测试集 %d 个文件" % (len(written["frozen"]), len(written["sealed"])))
    print("  DATA_CARD md5 : %s" % data_card_md5)
    print("  manifest  md5 : %s" % manifest_md5)
    print("  frozen md5sums: %d 条，自校验通过；自身 md5 %s" % (len(frozen_lines), frozen_md5_self))
    print("  sealed md5sums: %d 条，自校验通过；自身 md5 %s" % (len(sealed_lines), sealed_md5_self))
    print("  report        : %s" % _rel(report_path, root))
    print("OK: 冻结完成（H1–H7 全通过；先检查后写入；两份 md5sums 自校验通过）。")
    return EXIT_OK


GATE_KEY = {"H1": "g1_intent", "H2": "g2_ast", "H3": "g3_qc", "H4": "g4_guard",
            "H5": "g5_group", "H6": "h6_smoke", "H7": "h7_target"}


def parse_args(argv):
    p = argparse.ArgumentParser(description="W3.8 数据集冻结（02_08）")
    p.add_argument("--dataset", default="CodeObf-PI_v1.0")
    p.add_argument("--config", default="configs/data/freeze.yaml")
    p.add_argument("--root", default=None, help="（测试用）项目根目录覆盖")
    p.add_argument("--no-readonly", action="store_true", help="（测试用）跳过只读设置")
    p.add_argument("--h5-no-rerun", action="store_true",
                   help="（测试用）不重跑 02_07，只读现有 split_assertions.csv")
    return p.parse_args(argv)


def main(argv=None):
    init_streams()
    args = parse_args(argv)
    root = os.path.abspath(args.root) if args.root else REPO_ROOT

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(root, args.config)
    cfg_all = _load_yaml(cfg_path) if os.path.isfile(cfg_path) else None
    if cfg_all is None:
        print("FATAL: 缺少或无法解析配置 %s" % _rel(cfg_path, root))
        return EXIT_USAGE
    cfg = cfg_all.get("freeze")
    if not isinstance(cfg, dict):
        print("FATAL: 配置缺少 `freeze:` 顶层键")
        return EXIT_USAGE
    if args.dataset != cfg["dataset"]:
        print("FATAL: --dataset %s ≠ 配置的 %s（不支持换名冻结）" % (args.dataset, cfg["dataset"]))
        return EXIT_USAGE
    return freeze(root, cfg, no_readonly=args.no_readonly, h5_rerun=not args.h5_no_rerun)


if __name__ == "__main__":
    sys.exit(main())

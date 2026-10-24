# -*- coding: utf-8 -*-
"""02_00_smoke_data.py — W3.8 阶段2 全流程冒烟（数据管道）

接口契约（`windows/W3.8_freeze.md` §2）::

    python scripts/data/02_00_smoke_data.py \\
      --config configs/data/freeze.yaml \\
      [--seeds 10]            # 冒烟档：10 个种子（v5 新增 18）；缺省 10
      [--full]                # 全量档
      [--stage-dir <path>]    # 冒烟产物落点，缺省 logs/data/tmp/smoke_data/

退出码：``0`` 成功；非 0 失败。

**冒烟档（缺省）** —— 用 10 个种子跑通 **编码 → 嵌入 → 划分 → 断言** 全链路：

| 阶段 | 脚本 | 档位 | 输入 |
|---|---|---|---|
| ① 编码 | `02_03_encode_variants.py` | `--smoke <2×seeds>` | `data/interim/` 全量 `rewrites.csv`／`seeds.csv`（**只读**） |
| ② 嵌入 | `02_04_embed_code_carrier.py` | `--smoke <①×9>` | ① 的冒烟产物（**真串联**） |
| ③ 划分 | `02_06_split_group_aware.py` | `--smoke <smoke.split_units>` | `data/interim/` 全量上游（**只读**；见下方「③ 的输入面」） |
| ④ 断言 | `02_07_assert_splits.py` | 默认模式 | ③ 的冒烟产物 |

**③ 的输入面（显式登记，不隐藏）**：`02_06` 的 `--smoke N` 在切单位**之前**先校验
「攻击样本行数 == 配置期望」（`02_06` 第 833–845 行）——该检查**针对全量上游**设计，
故 `--smoke` 档读的是 `data/interim/` 的**全量**产品，只把**攻击单位**缩到前 N 个
（`windows/W3.6_split.md` §3 原文：「只取前 N 个 group」）。因此 ③ 不是 ①② 的下游，
而是与 ①② 并列的一条**只读**校验；①② 的产物仅用于验证这两阶段的代码路径。

**档位差异**：`--seeds 10` 决定 ①② 的规模；③ 的 `--smoke N` 由 `freeze.yaml` 的
`smoke.split_units`（缺省 **20**）单独给出。**为何不取 10**：`family_atomic: true` 下
单位为族（AH 族恒 4 组），10 个组凑不出 8:1:1 的最小可行解 ⇒ `test` 分片必空、
`02_06` 的 `G1`／`G5` 必然 FAIL（本窗口实测）。取 20 起 `G1`–`G11` 全 PASS。
**该口径差异在此显式登记，不隐藏。**

**阶段退出码的判读**：①②③ 的**自检**（`D1`–`D6`／`E1`–`E8`／`G1`–`G11`）是**全量档**口径；
④ 的 `A6`／`A7`／`A13` 是**绝对计数**断言（440／7920／176:22:22），在缩比数据上**必然不成立**。
故本脚本把各阶段的退出码**如实计入报告**，冒烟档的总判定**不取**「每阶段 exit=0」，
而取 **「四阶段均已执行且各自写出其声明产物 + 链路结构断言通过」**（见 §判定）。
**这不是放宽任何门禁**：门禁结论只由 `--full`（`H5`／`H6`）与 `02_07` 的全量默认模式给出。

**硬约束**（`windows/W3.8_freeze.md` §0／§6）:

- 冒烟档**只写临时目录**（`logs/data/tmp/`，被 `.gitignore` 的 `logs/*/tmp/` 覆盖，`D-15`）——
  **不得写 `data/interim/` 的真实产物、不得写 `data/frozen/`**。本脚本运行前后对
  `data/interim/` 与 `data/frozen/` 取值快照（路径／字节数／`mtime_ns`），**不一致即判失败**。
- **不训练、不微调**；本脚本**不加载任何模型**（各阶段自身只惰性加载 tokenizer，非模型）。
- `--full` 发现任一产物缺失／不可解析／断言非绿 ⇒ **非零退出**，**不得执行 `02_08`**。
- `--full` 的判定即 `windows/W3.8_freeze.md` §3.1 的 `H6`：其报告须可被 `02_08` 读取识别。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

#: 只读隔离面：冒烟档**不得**改动这两个目录（`windows/W3.8_freeze.md` §6-8）
ISOLATED_DIRS = ("data/interim", "data/frozen")

#: 冒烟链的四阶段（顺序即执行顺序，`windows/W3.8_freeze.md` §2）
STAGES = ("encode", "embed", "split", "assert")

REGIMES = ("regime_id", "regime_ood")
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（`F-1` 同类缺陷的预防；`W3.7` 补正轮 `r2` 先例）。

    本脚本的输出含 `→`／`⇒` 等非 ASCII 字符；本机中文 Windows 默认 **GBK**，
    不重配则重定向到文件时抛 ``UnicodeEncodeError`` ⇒ 退出码被污染。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - 非文本流
            pass


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, REPO_ROOT).replace("\\", "/")
    except ValueError:  # pragma: no cover - 跨盘符
        return path.replace("\\", "/")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: str) -> "list[dict]":
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _count_rows(path: str) -> int:
    """只数数据行（不含表头），流式读取，避免把 40 MB 文件整个读进内存。"""
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


# ---------------------------------------------------------------------------
# 隔离快照（`windows/W3.8_freeze.md` §4 测试 ①）
# ---------------------------------------------------------------------------
def snapshot_dirs(dirs) -> dict:
    """对目录取值快照：逐文件的（相对路径、字节数、`mtime_ns`）。

    只比字节数与 `mtime_ns`，**不重算内容哈希**——冒烟档的隔离面含 ~150 MB 的两级目录，
    逐文件哈希会让 `make smoke` 的耗时失去意义；`mtime_ns` 的写入分辨率足以捕捉任何重写。
    """
    snap = {}
    for d in dirs:
        entries = []
        base = os.path.join(REPO_ROOT, d)
        if os.path.isdir(base):
            for root, subdirs, files in os.walk(base):
                subdirs.sort()
                for name in sorted(files):
                    fp = os.path.join(root, name)
                    try:
                        st = os.stat(fp)
                    except OSError:  # pragma: no cover - 并发删除
                        continue
                    entries.append((os.path.relpath(fp, base).replace("\\", "/"),
                                    st.st_size, st.st_mtime_ns))
        snap[d] = entries
    return snap


def snapshot_digest(snap: dict) -> str:
    payload = json.dumps(snap, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_diff(before: dict, after: dict) -> "list[str]":
    problems = []
    for d in sorted(set(before) | set(after)):
        b, a = before.get(d, []), after.get(d, [])
        if b == a:
            continue
        bm = {x[0]: x[1:] for x in b}
        am = {x[0]: x[1:] for x in a}
        for k in sorted(set(bm) | set(am)):
            if bm.get(k) != am.get(k):
                problems.append("%s/%s: %s → %s" % (d, k, bm.get(k), am.get(k)))
    return problems


# ---------------------------------------------------------------------------
# 子进程执行
# ---------------------------------------------------------------------------
def run_stage(name: str, argv: "list[str]", log_fh) -> dict:
    """执行一个阶段；返回其退出码与输出。**不因非零退出而中断**（见模块 docstring）。"""
    cmd = [sys.executable] + argv
    printable = " ".join([_rel(argv[0])] + argv[1:])
    log_fh.write("\n$ %s\n" % printable)
    log_fh.flush()
    started = time.time()
    # 子进程一律以 UTF-8 写 stdout／stderr：本机中文 Windows 默认 GBK，
    # 若不强置，中文与 `⇒`／`→` 会在管道里被替换成 `�`（本脚本统一按 UTF-8 读回）。
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - started
    log_fh.write(proc.stdout or "")
    if proc.stderr:
        log_fh.write("--- stderr ---\n" + proc.stderr)
    log_fh.flush()
    print("  [%s] exit=%d  (%.1fs)" % (name, proc.returncode, elapsed))
    for line in (proc.stdout or "").splitlines():
        if "FAIL" in line or "RESULT=" in line or "OK: 自检" in line:
            print("      " + line.strip())
    return {"stage": name, "argv": argv, "cmd": printable,
            "exit": proc.returncode, "seconds": round(elapsed, 2),
            "stdout": proc.stdout or "", "stderr": proc.stderr or ""}


# ---------------------------------------------------------------------------
# 冒烟档
# ---------------------------------------------------------------------------
def derive_smoke_scale(cfg: dict, seeds: int, problems: "list[str]") -> dict:
    """按「前 N 个种子组」派生冒烟规模（`V-10`：每组恰 2 条改写体）。

    规模**实测派生**，不硬编码：从 `rewrites.csv` 取前 `seeds` 个 `seed_group_id`
    （按文件序，已按 `rewrite_id` 升序），数出它们覆盖的改写体数。
    """
    rewrites = os.path.join(REPO_ROOT, cfg["inputs"]["rewrites"])
    if not os.path.isfile(rewrites):
        problems.append("缺少输入 %s（冒烟档需要它派生规模）" % _rel(rewrites))
        return {}
    groups, n_rewrites = [], 0
    with open(rewrites, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            g = row.get("seed_group_id", "")
            if g and (not groups or groups[-1] != g):
                if len(groups) >= seeds:
                    break
                groups.append(g)
            n_rewrites += 1
    if len(groups) < seeds:
        problems.append("rewrites.csv 只含 %d 个 seed_group_id（< --seeds %d）"
                        % (len(groups), seeds))
    return {"seeds": seeds, "seed_groups": len(groups),
            "rewrites": n_rewrites, "encoded": n_rewrites * 9}


def run_smoke(cfg: dict, args) -> int:
    smoke_cfg = cfg.get("smoke", {}) or {}
    seeds = int(args.seeds if args.seeds is not None else smoke_cfg.get("seeds", 10))
    if seeds <= 0:
        print("FATAL: --seeds 必须为正整数，收到 %r" % (seeds,))
        return EXIT_USAGE
    split_units = int(smoke_cfg.get("split_units", 20))

    problems = []
    scale = derive_smoke_scale(cfg, seeds, problems)
    if problems:
        for p in problems:
            print("FATAL: %s" % p)
        return EXIT_FAIL

    stage_root = args.stage_dir or os.path.join(REPO_ROOT, cfg["outputs"]["tmp_dir"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(stage_root, "%s_seeds%d" % (stamp, seeds))
    gen = os.path.join(run_dir, "generated")
    for sub in ("encoded", "code_carrier", "split", "reports", "logs", "cfg"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    log_path = os.path.join(run_dir, "smoke_stdout.log")

    print("== 02_00 smoke (data) ==")
    print("  run_dir : %s" % _rel(run_dir))
    print("  scale   : %d seeds → %d groups → %d rewrites → %d encoded"
          % (scale["seeds"], scale["seed_groups"], scale["rewrites"], scale["encoded"]))
    print("  split   : --smoke %d（攻击单位数；口径见 freeze.yaml smoke.split_units）"
          % split_units)

    before = snapshot_dirs(ISOLATED_DIRS)

    inputs = cfg["inputs"]
    results = []
    with open(log_path, "w", encoding="utf-8", newline="\n") as log_fh:
        log_fh.write("# 02_00 smoke (data) — %s\n" % stamp)
        log_fh.write("# run_dir: %s\n" % _rel(run_dir))

        # ① 编码
        enc_out = os.path.join(gen, "encoded")
        results.append(run_stage("encode", [
            os.path.join(HERE, "02_03_encode_variants.py"),
            "--config", "configs/data/encode.yaml",
            "--rewrites", inputs["rewrites"],
            "--seeds", inputs["seeds"],
            "--out", enc_out,
            "--smoke", str(scale["rewrites"]),
            "--report-dir", os.path.join(run_dir, "reports"),
            "--log-dir", os.path.join(run_dir, "logs"),
            "--no-log",
        ], log_fh))

        # ② 嵌入（真串联：读 ① 的冒烟产物）
        smoke_encoded = os.path.join(enc_out, "smoke", "encoded.csv")
        cc_out = os.path.join(gen, "code_carrier")
        results.append(run_stage("embed", [
            os.path.join(HERE, "02_04_embed_code_carrier.py"),
            "--config", "configs/data/code_carrier.yaml",
            "--encoded", smoke_encoded,
            "--out", cc_out,
            "--smoke", str(scale["encoded"]),
            "--report-dir", os.path.join(run_dir, "reports"),
            "--log-dir", os.path.join(run_dir, "logs"),
            "--no-log",
        ], log_fh))

        # ③ 划分（读全量上游；见模块 docstring「③ 的输入面」）
        split_out = os.path.join(gen, "split")
        results.append(run_stage("split", [
            os.path.join(HERE, "02_06_split_group_aware.py"),
            "--config", "configs/data/split.yaml",
            "--encoded", inputs["encoded"],
            "--code-carrier", inputs["code_carrier_csv"],
            "--benign-dir", inputs["benign"],
            "--out", split_out,
            "--smoke", str(split_units),
            "--report-dir", os.path.join(run_dir, "reports"),
            "--log-dir", os.path.join(run_dir, "logs"),
            "--no-log",
        ], log_fh))

        # ④ 断言（读 ③ 的冒烟产物）
        smoke_split = os.path.join(split_out, "smoke")
        results.append(run_stage("assert", [
            os.path.join(HERE, "02_07_assert_splits.py"),
            "--config", "configs/data/assert.yaml",
            "--split-dir", smoke_split,
            "--out", smoke_split,
            "--seeds-meta", inputs["seeds_meta"],
            "--ast-report", inputs["ast_report"],
            "--report-dir", os.path.join(run_dir, "reports"),
            "--log-dir", os.path.join(run_dir, "logs"),
            # 注：`02_07` **没有** `--no-log` 参数（其 `--check-only` 之外无该开关）。
        ], log_fh))

    # --- 链路结构断言（冒烟档的总判定） ---
    chain = []
    chain.append(_check("encoded.csv 行数 == %d（%d × 9）" % (scale["encoded"], scale["rewrites"]),
                        smoke_encoded, scale["encoded"]))
    chain.append(_check("code_carrier.csv 行数 == %d" % scale["encoded"],
                        os.path.join(cc_out, "smoke", "code_carrier.csv"), scale["encoded"]))
    chain.append(_exists("code_carrier ast_report.csv",
                         os.path.join(cc_out, "smoke", "ast_report.csv")))
    for reg in REGIMES:
        for sp in SPLITS:
            chain.append(_exists("splits/%s/%s.csv" % (reg, sp),
                                 os.path.join(smoke_split, "splits", reg, "%s.csv" % sp)))
    for name in ("seed_groups.csv", "clean_original_test.csv", "g3_hard_negative_test.csv"):
        chain.append(_exists("划分派生视图 %s" % name, os.path.join(smoke_split, name)))
    assertions_csv = os.path.join(smoke_split, "split_assertions.csv")
    chain.append(_assertion_rows(assertions_csv, int(cfg["expected"]["assertions_rows"])))

    after = snapshot_dirs(ISOLATED_DIRS)
    diff = snapshot_diff(before, after)
    chain.append(("隔离面（%s）逐文件字节数／mtime 未变" % "、".join(ISOLATED_DIRS),
                  not diff, "无差异" if not diff else "；".join(diff[:4])))

    failed = [c for c in chain if not c[1]]
    conclusion = "PASS" if not failed else "FAIL"

    report_lines = _render_smoke_report(
        stamp=stamp, run_dir=run_dir, scale=scale, split_units=split_units,
        results=results, chain=chain, conclusion=conclusion,
        snapshot_before=snapshot_digest(before), snapshot_after=snapshot_digest(after),
        log_rel=_rel(log_path), cfg=cfg)

    report_dir = args.report_dir or os.path.join(REPO_ROOT, cfg["outputs"]["report_dir"])
    report_dir = os.path.abspath(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "%s_%s.md"
                               % (cfg["outputs"]["smoke_report_basename"], stamp))
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report_lines)

    print("")
    for name, ok, detail in chain:
        print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    print("  report : %s" % _rel(report_path))
    print("  result : SMOKE_MODE=smoke  CONCLUSION=%s" % conclusion)
    if failed:
        print("FAILED: %s" % ", ".join(c[0] for c in failed))
        return EXIT_FAIL
    print("OK: 冒烟档四阶段均已执行并写出声明产物；链路结构断言全通过；隔离面未变。")
    print("    注：各阶段自检（含 02_07 的 A6／A7／A13）为**全量档**口径，"
          "缩比数据下不成立，其退出码如实计入报告，不参与本判定（见模块 docstring）。")
    return EXIT_OK


def _check(desc: str, path: str, expected_rows: int):
    if not os.path.isfile(path):
        return (desc, False, "缺失 %s" % _rel(path))
    n = _count_rows(path)
    return (desc, n == expected_rows, "实测 %d 行（%s）" % (n, _rel(path)))


def _exists(desc: str, path: str):
    ok = os.path.isfile(path)
    return (desc, ok, "OK" if ok else "缺失 %s" % _rel(path))


def _assertion_rows(path: str, expected: int):
    desc = "split_assertions.csv 恰 %d 行（A1–A%d）" % (expected, expected)
    if not os.path.isfile(path):
        return (desc, False, "缺失 %s" % _rel(path))
    rows = _read_csv(path)
    ids = [r.get("assertion_id", "") for r in rows]
    want = ["A%d" % i for i in range(1, expected + 1)]
    ok = ids == want
    return (desc, ok, "实测 %d 行 %s" % (len(rows), "无缺行" if ok else "缺/多行：%s" % ids))


# ---------------------------------------------------------------------------
# 全量档
# ---------------------------------------------------------------------------
#: `--full` 的产物清单：(标签, `cfg["inputs"]` 的键, 期望行数或 None 表示只看存在性)。
#: **路径一律经配置解析**，不写死字面量 —— 否则「缺一个上游产物」的回归测试会失效
#: （改配置里的路径不会改变写死的清单）。
FULL_PRODUCTS = [
    ("种子表 seeds.csv", "seeds", 220),
    ("种子元数据 seeds.meta.json", "seeds_meta", None),
    ("改写体 rewrites.csv", "rewrites", 440),
    ("编码变体 encoded.csv", "encoded", 3960),
    ("代码载体 code_carrier.csv", "code_carrier_csv", 3960),
    ("AST 报告 ast_report.csv", "ast_report", None),
    ("良性 G1.csv", "benign_G1", 4500),
    ("良性 G2.csv", "benign_G2", 2700),
    ("良性 G3.csv", "benign_G3", 1800),
    ("G3 质检 g3_qc.csv", "g3_qc", 200),
    ("门禁 4 明细 benign_guard_check.csv", "benign_guard_check", 90),
    ("种子分组 seed_groups.csv", "seed_groups", 5243),
    ("干净原始测试 clean_original_test.csv", "clean_original_test", None),
    ("G3 硬负例 g3_hard_negative_test.csv", "g3_hard_negative_test", None),
    ("断言表 split_assertions.csv", "split_assertions", 14),
    ("长度分布 length_distribution.csv", "length_distribution", None),
    ("n-gram 重叠 ngram_overlap.csv", "ngram_overlap", None),
]


def run_full(cfg: dict, args) -> int:
    print("== 02_00 smoke (data) --full ==")
    print("  全量档：**不重新生成数据**，只做只读校验 + 重跑 02_07 默认模式。")

    problems = []
    checks = []
    inp = cfg["inputs"]

    for label, key, expect in FULL_PRODUCTS:
        path = os.path.join(REPO_ROOT, inp[key])
        if not os.path.isfile(path):
            checks.append((label, False, "缺失 %s" % _rel(path)))
            problems.append("缺少产物 %s" % _rel(path))
            continue
        try:
            rows = _read_csv(path)
        except Exception as exc:  # noqa: BLE001 - 如实报出解析失败
            checks.append((label, False, "不可解析：%s" % exc))
            problems.append("产物不可解析 %s：%s" % (_rel(path), exc))
            continue
        if expect is not None and len(rows) != expect:
            checks.append((label, False, "行数 %d ≠ %d" % (len(rows), expect)))
            problems.append("行数不符 %s：%d ≠ %d" % (_rel(path), len(rows), expect))
        else:
            checks.append((label, True, "行数 %d" % len(rows)))

    # 六个分片存在性
    for reg in REGIMES:
        for sp in SPLITS:
            path = os.path.join(REPO_ROOT, inp["splits_root"], reg, "%s.csv" % sp)
            ok = os.path.isfile(path)
            checks.append(("分片 splits/%s/%s.csv" % (reg, sp), ok,
                           "OK" if ok else "缺失 %s" % _rel(path)))
            if not ok:
                problems.append("缺少分片 %s" % _rel(path))

    # 行数与断言表一致（`A6`／`A7` 的字面值）
    assertions_path = os.path.join(REPO_ROOT, inp["split_assertions"])
    assertions_rows = _read_csv(assertions_path) if os.path.isfile(assertions_path) else []
    a67 = {r.get("assertion_id"): r.get("measured", "") for r in assertions_rows}
    exp = cfg["expected"]
    if a67:
        m6 = re.search(r"=\s*(\d+)", a67.get("A6", ""))
        m7 = re.search(r"=\s*(\d+)", a67.get("A7", ""))
        ok6 = bool(m6) and int(m6.group(1)) == int(exp["rewrite_bodies"])
        ok7 = bool(m7) and int(m7.group(1)) == int(exp["attack_total"])
        checks.append(("断言表 A6 == %d" % exp["rewrite_bodies"], ok6,
                       a67.get("A6", "缺失")))
        checks.append(("断言表 A7 == %d" % exp["attack_total"], ok7,
                       a67.get("A7", "缺失")))
        if not ok6:
            problems.append("A6 与冻结硬计数不符")
        if not ok7:
            problems.append("A7 与冻结硬计数不符")
    else:
        checks.append(("断言表 A6／A7", False, "不可读"))
        problems.append("无法读取 split_assertions.csv 的 A6／A7")

    # 重跑 02_07 默认模式（`H5`）
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(REPO_ROOT, cfg["outputs"]["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    rc07 = None
    if args.skip_assert_rerun:
        checks.append(("02_07 默认模式重跑", True, "SKIPPED（--skip-assert-rerun）"))
    elif problems:
        # 快速失败：产物已缺，重跑 02_07 无意义（其结论也不可采信）
        checks.append(("02_07 默认模式重跑", False, "SKIPPED（产物已缺 %d 项，快速失败）"
                       % len(problems)))
    else:
        print("  重跑 02_07 默认模式（H5）…")
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "02_07_assert_splits.py"),
             "--config", "configs/data/assert.yaml"],
            cwd=REPO_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        rc07 = proc.returncode
        gating_ok = (rc07 == 0)
        checks.append(("02_07 默认模式重跑 exit=0", gating_ok, "exit=%d" % rc07))
        if not gating_ok:
            problems.append("02_07 默认模式 exit=%d ≠ 0" % rc07)

    # A1–A13 全 True（门禁类）；A14 为报告性断言（passed 恒 n/a）
    gate_ids = ["A%d" % i for i in range(1, 14)]
    if assertions_rows:
        rows = {r.get("assertion_id"): r for r in assertions_rows}
        bad = [i for i in gate_ids if rows.get(i, {}).get("passed") != "True"]
        checks.append(("A1–A13 全 True（门禁类）", not bad,
                       "全部 True" if not bad else "非 True：%s" % bad))
        if bad:
            problems.append("门禁类断言非 True：%s" % bad)
        a14 = rows.get("A14", {})
        checks.append(("A14 报告性断言（passed 恒 n/a）",
                       a14.get("passed") == "n/a", a14.get("passed", "缺失")))

    conclusion = "PASS" if not problems else "FAIL"
    report_lines = _render_full_report(stamp=stamp, checks=checks, problems=problems,
                                       conclusion=conclusion, rc07=rc07, cfg=cfg)
    report_dir = args.report_dir or os.path.join(REPO_ROOT, cfg["outputs"]["report_dir"])
    report_dir = os.path.abspath(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "%s_%s.md"
                               % (cfg["outputs"]["smoke_report_basename"], stamp))
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report_lines)

    print("")
    for label, ok, detail in checks:
        print("  [%s] %-40s %s" % ("PASS" if ok else "FAIL", label, detail))
    print("  report : %s" % _rel(report_path))
    print("  result : SMOKE_MODE=full  CONCLUSION=%s" % conclusion)
    if problems:
        print("FAILED: 全量档校验未通过（%d 项）—— 不得执行 02_08" % len(problems))
        for p in problems[:8]:
            print("    - %s" % p)
        return EXIT_FAIL
    print("OK: 全量档通过（产物齐备、可解析、行数与断言表一致；02_07 默认模式 exit=0；"
          "A1–A13 全 True）。此为 02_08 冻结前置 H6。")
    return EXIT_OK


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _render_smoke_report(*, stamp, run_dir, scale, split_units, results, chain,
                         conclusion, snapshot_before, snapshot_after, log_rel, cfg) -> str:
    L = []
    L.append("# smoke_data_v1.0 — 阶段2 全流程冒烟（冒烟档）运行报告")
    L.append("")
    L.append("- 窗口：W3.8（全流程冒烟与冻结，权限 **A3**）")
    L.append("- 日期：%s" % stamp)
    L.append("- 脚本：`scripts/data/02_00_smoke_data.py`")
    L.append("- 配置：`configs/data/freeze.yaml`")
    L.append("- 模式（`mode`）：`smoke`")
    L.append("- 结论（`conclusion`）：`%s`" % conclusion)
    L.append("")
    L.append("## §1 规模（**实测派生**，非硬编码）")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| `--seeds` | %d |" % scale["seeds"])
    L.append("| 覆盖 `seed_group_id` 数 | %d |" % scale["seed_groups"])
    L.append("| 改写体数（`V-10`：每组 2 条） | %d |" % scale["rewrites"])
    L.append("| 编码变体数（×9） | %d |" % scale["encoded"])
    L.append("| `02_06 --smoke` 的攻击单位数 | %d |" % split_units)
    L.append("| 产物落点 | `%s` |" % _rel(run_dir))
    L.append("| 阶段 stdout 全量日志 | `%s` |" % log_rel)
    L.append("")
    L.append("> **③ 划分阶段的输入面**：`02_06 --smoke N` 在切单位**之前**先校验「攻击样本行数 == 配置期望」"
             "（针对全量上游设计）⇒ 该阶段读 `data/interim/` 的**全量**产品（只读），"
             "只把攻击单位缩到前 N 个（`windows/W3.6_split.md` §3 原文：「只取前 N 个 group」）。"
             "①② 的冒烟产物仅用于验证这两阶段的代码路径。")
    L.append("")
    L.append("> **`split_units = 20` 的理由**：`family_atomic: true` 下单位为族（AH 族恒 4 组），"
             "**N ≤ 10 时 `test` 分片必空**（凑不出 8:1:1 的最小可行解）⇒ `02_06` 的 `G1`／`G5` 必然 FAIL；"
             "N ≥ 20 起 `G1`–`G11` 全 PASS（本窗口实测）。该口径与「10 个种子」的差异在此**显式登记**。")
    L.append("")
    L.append("## §2 四阶段执行结果")
    L.append("")
    L.append("| 阶段 | 脚本 | 退出码 | 耗时(s) | 命令 |")
    L.append("|---|---|---|---|---|")
    for r in results:
        L.append("| %s | `%s` | **%d** | %.1f | `%s` |"
                 % (r["stage"], _rel(r["argv"][0]), r["exit"], r["seconds"], r["cmd"]))
    L.append("")
    L.append("**退出码判读**：各阶段自检（`D1`–`D6`／`E1`–`E8`／`G1`–`G11`）与 `02_07` 的 "
             "`A6`／`A7`／`A13` 均为**全量档**口径；缩比数据下 `02_07` 的绝对计数断言**必然不成立**。"
             "故冒烟档的总判定**不取**「每阶段 exit=0」（见 §3）。")
    L.append("")
    L.append("## §3 链路结构断言（冒烟档的总判定）")
    L.append("")
    L.append("| 断言 | 判定 | 实测 |")
    L.append("|---|---|---|")
    for name, ok, detail in chain:
        L.append("| %s | %s | %s |" % (name, "PASS" if ok else "**FAIL**", detail))
    L.append("")
    L.append("## §4 隔离证据（`data/interim/` 与 `data/frozen/` 未被触碰）")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 快照口径 | 逐文件（相对路径、字节数、`mtime_ns`） |")
    L.append("| 运行前快照 sha256 | `%s` |" % snapshot_before)
    L.append("| 运行后快照 sha256 | `%s` |" % snapshot_after)
    L.append("| 是否一致 | **%s** |" % ("一致" if snapshot_before == snapshot_after else "**不一致**"))
    L.append("")
    L.append("## §5 声明")
    L.append("")
    L.append("1. 冒烟档**只写临时目录**（`%s`，被 `.gitignore` 的 `logs/*/tmp/` 覆盖，`D-15`）。"
             % _rel(run_dir))
    L.append("2. 本脚本**不训练、不微调、不加载任何模型**；各阶段自身仅惰性加载 tokenizer（非模型）。")
    L.append("3. **第一阶段冒烟在生成之前**（`windows/W3.8_freeze.md` §2）；本档不构成任何门禁结论。")
    L.append("4. 冻结前置 `H6` 要求的是 **`--full` 档**的报告（`mode: full`），**不是**本报告。")
    L.append("")
    return "\n".join(L) + "\n"


def _render_full_report(*, stamp, checks, problems, conclusion, rc07, cfg) -> str:
    L = []
    L.append("# smoke_data_v1.0 — 阶段2 全流程冒烟（全量档）运行报告")
    L.append("")
    L.append("- 窗口：W3.8（全流程冒烟与冻结，权限 **A3**）")
    L.append("- 日期：%s" % stamp)
    L.append("- 脚本：`scripts/data/02_00_smoke_data.py --full`")
    L.append("- 配置：`configs/data/freeze.yaml`")
    L.append("- 模式（`mode`）：`full`")
    L.append("- 结论（`conclusion`）：`%s`" % conclusion)
    L.append("")
    L.append("> 本档**不重新生成数据**：只对 `data/interim/` 的全部产物做**只读**校验"
             "（齐备、可解析、行数与断言表一致），并**重跑 `02_07` 默认模式**（`H5`）。")
    L.append("")
    L.append("## §1 产物清单校验")
    L.append("")
    L.append("| 产物 | 判定 | 实测 |")
    L.append("|---|---|---|")
    for label, ok, detail in checks:
        L.append("| %s | %s | %s |" % (label, "PASS" if ok else "**FAIL**", detail))
    L.append("")
    L.append("## §2 `H5`：`02_07` 默认模式")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 退出码 | %s |" % ("`%d`" % rc07 if rc07 is not None else "未执行"))
    L.append("| 门禁 5／6 | 见上游报告 `reports/data/g567_assert_v1.0_*.md` |")
    L.append("| 门禁 7 | 报告项（`passed` 恒 `n/a`，阈值 `0.05` 一字未改） |")
    L.append("")
    L.append("## §3 结论")
    L.append("")
    if problems:
        L.append("**未通过** —— 缺口 %d 项：" % len(problems))
        L.append("")
        for p in problems:
            L.append("- %s" % p)
        L.append("")
        L.append("> 依 `windows/W3.8_freeze.md` §2：`--full` 非零退出 ⇒ **不得执行 `02_08`**。")
    else:
        L.append("**通过** —— 全部产物齐备、可解析、行数与断言表一致；`02_07` 默认模式 `exit=0`；"
                 "门禁类断言 `A1`–`A13` 全 `True`（`A14` 为报告性断言，`passed` 恒 `n/a`）。")
        L.append("")
        L.append("→ 本报告即 `windows/W3.8_freeze.md` §3.1 的 **`H6`** 判据所引报告"
                 "（`mode: full` 且结论为通过）。")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="W3.8 阶段2 全流程冒烟（数据管道，02_00）")
    p.add_argument("--config", default="configs/data/freeze.yaml")
    p.add_argument("--seeds", type=int, default=None,
                   help="冒烟档：种子数（缺省取配置 smoke.seeds = 10）")
    p.add_argument("--full", action="store_true",
                   help="全量档：只读校验全部产物 + 重跑 02_07（冻结前置 H6）")
    p.add_argument("--stage-dir", default=None,
                   help="冒烟产物落点，缺省取配置 outputs.tmp_dir（logs/data/tmp/smoke_data/）")
    p.add_argument("--report-dir", default=None,
                   help="运行报告落点，缺省取配置 outputs.report_dir（reports/data）")
    p.add_argument("--skip-assert-rerun", action="store_true",
                   help="（测试用）跳过全量档重跑 02_07")
    return p.parse_args(argv)


def main(argv=None):
    init_streams()
    args = parse_args(argv)

    cfg_path = os.path.join(REPO_ROOT, args.config)
    cfg_all = _load_yaml(cfg_path) if os.path.isfile(cfg_path) else None
    if cfg_all is None:
        print("FATAL: 缺少或无法解析配置 %s" % _rel(cfg_path))
        return EXIT_USAGE
    cfg = cfg_all.get("freeze")
    if not isinstance(cfg, dict):
        print("FATAL: 配置缺少 `freeze:` 顶层键")
        return EXIT_USAGE

    return run_full(cfg, args) if args.full else run_smoke(cfg, args)


if __name__ == "__main__":
    sys.exit(main())

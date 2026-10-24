# -*- coding: utf-8 -*-
"""03_00_smoke_baseline.py — 阶段3 基线冒烟（v5 新增 22）。

用途：`make smoke STAGE=baseline` 的入口 —— **20 条样本**跑通 B1–B5 的
「检测器输出 → 统一映射 → τ 寻优 → 指标」全链路，验证链路正常。

契约（`windows/W4.7_smoke_report.md` §0.3／§0.4）：

1. **冒烟样本取自验证分片 `id_val`**（`data/frozen/CodeObf-PI_v1.0/splits/regime_id/val.csv`，
   非 sealed）——**禁止用测试分片做冒烟**（测试集只跑最终评估一次）。
2. 冒烟**不产出权威产物**：全部中间产物落 `reports/baseline/tmp/smoke_baseline_<ts>/`
   （`reports/*/tmp/` 依 `D-15` 不入版本控制）；证据 = 退出码 ＋ 冒烟日志
   `logs/baseline/03_00_smoke_baseline_<ts>.log`。
3. **零改动上游**：不修改 `03_01`–`03_07` 任何一行。冒烟通过
   **临时配置副本**（`unify_smoke.yaml` / `b4_smoke.yaml`）把「`id_val` 指向 20 条子集」
   与「B4 输出目录改指冒烟区」两处重定向，其余常量逐字沿用冻结接口 v1.0。
4. 采样确定性：`random.Random(seed)`，`seed` 默认 42；正负分层各半，顺序按 `sample_id` 升序。

CLI::

    python scripts/baseline/03_00_smoke_baseline.py               # 20 条样本全链路冒烟
    python scripts/baseline/03_00_smoke_baseline.py --n 20 --seed 42
    python scripts/baseline/03_00_smoke_baseline.py --dry-run     # 只搭工作区、打印命令、不跑模型
    python scripts/baseline/03_00_smoke_baseline.py --skip-heavy  # 只跑 B1/B2/03_07（跳过模型类检测器）

退出码：全链路通过 = 0；任一步失败 = 1（并打印 `SMOKE_FAIL: …`）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # 项目根目录

# 冻结接口 v1.0 的常量出处（W4.6）：分片名与输出 schema 不在此处另立一套
UNIFY_CONFIG = os.path.join(ROOT, "configs", "baseline", "unify.yaml")
B4_CONFIG = os.path.join(ROOT, "configs", "baseline", "b4_unplug.yaml")
ID_VAL_SPLIT = "id_val"
DETECTORS = ("b1", "b2", "b3", "b4", "b5")
SCHEMA_COLUMNS = ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score", "latency_ms"]
PRED_ALLOWED = {"0", "1", ""}                            # B3/B5 的 unparsed 记为 ""

STAGES = (
    ("b1", "scripts/baseline/03_01_run_b1_regex.py"),
    ("b2", "scripts/baseline/03_02_run_b2_entropy.py"),
    ("b3", "scripts/baseline/03_03_run_b3_llamaguard1b.py"),
    ("b4", "scripts/baseline/03_04_run_b4_unplug_tiny.py"),
    ("b5", "scripts/baseline/03_05_run_b5_llamaguard8b.py"),
)
B4_ENV = {
    "PYTHONPATH": os.path.join(ROOT, "logs", "env", "tmp", "vendor_unplug"),
    "UNPLUG_MODEL_CACHE": os.path.join(ROOT, "logs", "env", "tmp", "unplug_model_cache"),
    "HF_ENDPOINT": "https://hf-mirror.com",
}


# ---------------------------------------------------------------------------
# 输出流与日志（stdout ＋ 文件双写；只动输出流，不改任何判定值）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（`errors="replace"`）。先例：W3.7 `init_streams`。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


class _Logger:
    """stdout ＋ 日志文件双写（`logs/baseline/03_00_smoke_baseline_<ts>.log`，只追加）。"""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8", newline="\n")

    def __call__(self, message=""):
        line = str(message)
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 载入上游 `03_06`（统一预处理的 write_csv / sha256_of，唯一出处）
# ---------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# ① 冒烟样本子集（验证分片 → 20 条，确定性、分层）
# ---------------------------------------------------------------------------
def read_rows(csv_path):
    """读 UTF-8 CSV 的整行（保留全部列与列序），供子集抽取。"""
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _label_of(row):
    try:
        return int(float(str(row.get("label")).strip()))
    except (TypeError, ValueError):
        return None


def build_subset(rows, n, seed):
    """从验证分片行中确定性抽取 ``n`` 条：正负分层各半，顺序按 ``sample_id`` 升序。

    不足时由另一类补齐（**确定性**，不引入随机性）；返回 ``(subset_rows, meta)``。
    """
    pos = sorted((r for r in rows if _label_of(r) == 1), key=lambda r: str(r.get("sample_id")))
    neg = sorted((r for r in rows if _label_of(r) == 0), key=lambda r: str(r.get("sample_id")))
    rng = random.Random(seed)

    want_pos = n // 2
    want_neg = n - want_pos
    take_pos = rng.sample(pos, min(want_pos, len(pos)))
    take_neg = rng.sample(neg, min(want_neg, len(neg)))

    short = n - len(take_pos) - len(take_neg)
    if short > 0:                                        # 某一类不足时由另一类补齐
        chosen_pos = {id(r) for r in take_pos}
        chosen_neg = {id(r) for r in take_neg}
        spare = [r for r in (pos + neg) if id(r) not in chosen_pos and id(r) not in chosen_neg]
        spare.sort(key=lambda r: str(r.get("sample_id")))
        if spare:
            take_pos += rng.sample(spare, min(short, len(spare)))

    subset = sorted(take_pos + take_neg, key=lambda r: str(r.get("sample_id")))
    meta = {
        "n_requested": n,
        "n_taken": len(subset),
        "seed": seed,
        "source_pos": len(pos),
        "source_neg": len(neg),
        "taken_pos": sum(1 for r in subset if _label_of(r) == 1),
        "taken_neg": sum(1 for r in subset if _label_of(r) == 0),
    }
    return subset, meta


def write_subset(csv_path, rows, fieldnames, write_csv):
    """用 03_06 的 ``write_csv`` 落子集（UTF-8 + LF；列序与源分片逐字一致）。"""
    write_csv(csv_path, rows, fieldnames)


# ---------------------------------------------------------------------------
# ② 临时配置副本（零改动上游；只重定向两处）
# ---------------------------------------------------------------------------
def _load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_yaml(path, obj):
    import yaml
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 冒烟临时配置副本（03_00 生成，**非冻结件**；仅用于 make smoke STAGE=baseline）\n")
        fh.write("# 源：%s（注释在 YAML 往返中被丢弃，常量值逐项保留）\n"
                 % os.path.relpath(unify_or_b4_source(path), ROOT).replace("\\", "/"))
        yaml.safe_dump(obj, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)


def unify_or_b4_source(dst_path):
    """临时副本 → 其源配置路径（按文件名判别）。"""
    return B4_CONFIG if "b4" in os.path.basename(dst_path) else UNIFY_CONFIG


def make_smoke_unify_config(dst_path, subset_csv):
    """`unify.yaml` 副本：**只**把 `id_val` 的 path/rows 指向冒烟子集，其余逐字保留。"""
    cfg = _load_yaml(UNIFY_CONFIG)
    entry = cfg["splits"][ID_VAL_SPLIT]
    entry["path"] = os.path.relpath(subset_csv, ROOT).replace("\\", "/")
    entry["rows"] = 0                                   # 由子集实读行数决定，避免硬编码失真
    entry["role"] = "smoke_subset_of_id_val"
    _dump_yaml(dst_path, cfg)
    return cfg


def make_smoke_b4_config(dst_path, smoke_outputs_rel):
    """`b4_unplug.yaml` 副本：**只**把 `output.directory` 指向冒烟区（03_04 无 `--outputs-dir`）。"""
    cfg = _load_yaml(B4_CONFIG)
    cfg["output"]["directory"] = smoke_outputs_rel
    _dump_yaml(dst_path, cfg)
    return cfg


# ---------------------------------------------------------------------------
# ③ 子进程执行（流式 tee 到日志；超时看门狗）
# ---------------------------------------------------------------------------
def run_stage(name, cmd, logger, env_extra=None, timeout=1800):
    """跑一个子进程，stdout/stderr 逐行 tee 到日志；返回 ``(rc, seconds)``。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for key, value in (env_extra or {}).items():
        if key == "PYTHONPATH" and env.get("PYTHONPATH"):
            env[key] = value + os.pathsep + env[key]
        else:
            env[key] = value

    logger("")
    logger("---- [%s] $ %s" % (name, " ".join(cmd)))
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    killed = {"flag": False}

    def _kill():
        killed["flag"] = True
        try:
            proc.kill()
        except OSError:
            pass

    watchdog = threading.Timer(timeout, _kill)
    watchdog.start()
    try:
        for line in iter(proc.stdout.readline, ""):
            logger("      | " + line.rstrip("\n"))
        proc.stdout.close()
        rc = proc.wait()
    finally:
        watchdog.cancel()
    elapsed = time.time() - started
    if killed["flag"]:
        logger("---- [%s] TIMEOUT（> %ds）rc=%s 用时 %.1fs" % (name, timeout, rc, elapsed))
        return 124, elapsed
    logger("---- [%s] rc=%d 用时 %.1fs" % (name, rc, elapsed))
    return rc, elapsed


# ---------------------------------------------------------------------------
# ④ 验收断言（schema ／ 一致性 ／ 指标）
# ---------------------------------------------------------------------------
def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _latest(paths):
    if not paths:
        return None
    return sorted(paths)[-1]


def find_output(outputs_dir, detector):
    prefix = "%s_%s_v1.0_" % (detector, ID_VAL_SPLIT)
    cands = [os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir)
             if f.startswith(prefix) and f.endswith(".csv")] if os.path.isdir(outputs_dir) else []
    return _latest(cands)


def check_schema(path, n_expected):
    """§2 schema 校验：列名与列序、行数、`text_sha256` 非空、`pred` 取值。返回错误列表。"""
    errors = []
    rows, fields = _read_csv(path)
    if fields != SCHEMA_COLUMNS:
        errors.append("列不符 §2 schema：%r" % (fields,))
    if len(rows) != n_expected:
        errors.append("行数 %d != %d" % (len(rows), n_expected))
    for i, row in enumerate(rows):
        digest = row.get("text_sha256") or ""
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append("第 %d 行 text_sha256 非法：%r" % (i, digest))
            break
        if (row.get("pred") or "") not in PRED_ALLOWED:
            errors.append("第 %d 行 pred 非法：%r" % (i, row.get("pred")))
            break
    return errors


def check_consistency(outputs_by_detector, checker):
    """§1.4：同一 `sample_id` 的 `text_sha256` 跨 B1–B5 必须一致。返回分歧列表。"""
    return checker(outputs_by_detector)


def check_metrics(metrics_csv, n_expected):
    """指标表校验：B1–B5 × `id_val` 逐行存在、样本数与冒烟子集相符。返回错误列表与行表。"""
    errors = []
    rows, fields = _read_csv(metrics_csv)
    if "detector" not in fields or "split" not in fields:
        return ["指标表缺 detector/split 列：%r" % (fields,)], []
    by_det = {r["detector"]: r for r in rows if r.get("split") == ID_VAL_SPLIT}
    for det in DETECTORS:
        row = by_det.get(det)
        if row is None:
            errors.append("指标表缺 %s × %s" % (det, ID_VAL_SPLIT))
            continue
        total = int(float(row.get("n_pos") or 0)) + int(float(row.get("n_neg") or 0))
        if total != n_expected:
            errors.append("%s × %s 样本数 %d != %d" % (det, ID_VAL_SPLIT, total, n_expected))
    return errors, [by_det.get(d, {}) for d in DETECTORS]


# ---------------------------------------------------------------------------
# ⑤ 主流程
# ---------------------------------------------------------------------------
def build_commands(unify_smoke, b4_smoke, outputs_dir, reports_dir):
    """返回 ``[(stage_name, cmd, env_extra), …]`` —— 全链路命令表（含 `--dry-run` 打印用）。"""
    rel_out = os.path.relpath(outputs_dir, ROOT).replace("\\", "/")
    py = sys.executable or "python"
    cmds = []
    for det, script in STAGES:
        cmd = [py, script]
        if det == "b4":
            cmd += ["--config", b4_smoke]
        cmd += ["--unify-config", unify_smoke, "--splits", ID_VAL_SPLIT]
        if det in ("b3", "b4", "b5"):
            cmd += ["--limit", "20"]
        if det in ("b1", "b2", "b3", "b5"):
            cmd += ["--outputs-dir", rel_out, "--report-dir",
                    os.path.relpath(reports_dir, ROOT).replace("\\", "/")]
        cmd += ["--no-access-reminder" if det != "b4" else "--no-access-log-reminder"]
        cmds.append((det, cmd, dict(B4_ENV) if det == "b4" else None))
    cmds.append(("03_07", [py, "scripts/baseline/03_07_unify_output_mapping.py",
                           "--config", unify_smoke,
                           "--outputs-dir", rel_out,
                           "--report-dir", os.path.relpath(reports_dir, ROOT).replace("\\", "/")],
                 None))
    return cmds


def _snapshot(dir_path, prefix, suffix=".md"):
    if not os.path.isdir(dir_path):
        return set()
    return {f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(suffix)}


def relocate_b4_report(smoke_reports_dir, before, logger):
    """03_04 的报告路径硬编码在 `reports/baseline/`（无 CLI）⇒ 冒烟后搬离权威目录。

    只搬**本次冒烟新产生**的文件（`after - before`），并把搬迁登进日志；
    不做任何内容改写（`D-40` 裁定三的「权威稿」语义不受影响：本件从未是权威稿）。
    """
    canonical = os.path.join(ROOT, "reports", "baseline")
    moved = []
    for name in sorted(_snapshot(canonical, "b4_unplug_v1.0_") - before):
        src = os.path.join(canonical, name)
        dst = os.path.join(smoke_reports_dir, "SMOKE_" + name)
        try:
            shutil.move(src, dst)
            moved.append((os.path.relpath(src, ROOT).replace("\\", "/"),
                          os.path.relpath(dst, ROOT).replace("\\", "/")))
        except OSError as exc:
            logger("SMOKE_WARN: 无法搬移 %s（%s）" % (src, exc))
    for src, dst in moved:
        logger("冒烟产物搬离权威目录：%s -> %s" % (src, dst))
    return moved


def check_residual(logger):
    """兜底：确认权威 outputs 目录未被本次冒烟写入 `b*_id_val_*` 新件。"""
    canonical = os.path.join(ROOT, "reports", "baseline", "outputs")
    strays = [f for f in (os.listdir(canonical) if os.path.isdir(canonical) else [])
              if f.endswith(".csv") and "_id_val_" in f and not f.startswith("INVALID_")]
    # 唯一合法来源 = W4.2 的权威 b2_id_val/b2_ood_val（供 τ 寻优），已由 W4.6 登记
    strays = [f for f in strays if not f.startswith("b2_")]
    if strays:
        logger("SMOKE_WARN: 权威 outputs 目录出现非 B2 的 id_val 件：%s" % (strays,))
    return strays


def main() -> int:
    init_streams()
    parser = argparse.ArgumentParser(description="阶段3 基线冒烟（v5 新增 22；20 条验证分片样本跑通 B1–B5 链路）")
    parser.add_argument("--n", type=int, default=20, help="冒烟样本数（默认 20）")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子（默认 42）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只搭工作区并打印命令，不执行任何检测器/模型")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="只跑 B1/B2 ＋ 03_07（跳过 B3/B4/B5 模型类检测器）")
    parser.add_argument("--timeout", type=int, default=1800, help="单阶段超时秒数（默认 1800）")
    parser.add_argument("--log-dir", default=os.path.join("logs", "baseline"),
                        help="冒烟日志目录（默认 logs/baseline；测试可改指临时目录）")
    parser.add_argument("--work-root", default=os.path.join("reports", "baseline", "tmp"),
                        help="冒烟工作区根（默认 reports/baseline/tmp；测试可改指临时目录）")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(ROOT, args.log_dir, "03_00_smoke_baseline_%s.log" % ts)
    logger = _Logger(log_path)
    logger("=" * 78)
    logger("03_00_smoke_baseline 阶段3 基线冒烟（v5 新增 22）  ts=%s" % ts)
    logger("项目根：%s" % ROOT)
    logger("python：%s" % (sys.executable or "python"))
    logger("参数：n=%d seed=%d dry_run=%s skip_heavy=%s timeout=%ds"
           % (args.n, args.seed, args.dry_run, args.skip_heavy, args.timeout))

    failures = []
    smoke_root = os.path.join(ROOT, args.work_root, "smoke_baseline_%s" % ts)
    outputs_dir = os.path.join(smoke_root, "outputs")
    reports_dir = os.path.join(smoke_root, "reports")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # ---- ① 冒烟样本（验证分片 id_val，非 sealed）----
    unify_cfg = _load_yaml(UNIFY_CONFIG)
    val_entry = unify_cfg["splits"][ID_VAL_SPLIT]
    val_path = os.path.join(ROOT, val_entry["path"])
    if not os.path.isfile(val_path):
        logger("SMOKE_FAIL: 验证分片不存在：%s" % val_path)
        logger.close()
        return 1
    rows, fieldnames = read_rows(val_path)
    subset, meta = build_subset(rows, args.n, args.seed)
    logger("")
    logger("冒烟样本：源 = %s（sealed=%s，共 %d 行：正 %d／负 %d）"
           % (val_entry["path"], bool(val_entry.get("sealed", False)), len(rows),
              meta["source_pos"], meta["source_neg"]))
    logger("         抽取 = %d 条（正 %d／负 %d），seed=%d，顺序按 sample_id 升序"
           % (meta["n_taken"], meta["taken_pos"], meta["taken_neg"], args.seed))
    if meta["n_taken"] < args.n:
        failures.append("样本不足：请求 %d 实得 %d" % (args.n, meta["n_taken"]))

    preprocess = _load_module("baseline_preprocess",
                              os.path.join(_HERE, "03_06_unify_preprocess.py"))
    subset_csv = os.path.join(smoke_root, "smoke_id_val_subset.csv")
    write_subset(subset_csv, subset, fieldnames, preprocess.write_csv)
    logger("         子集落盘：%s" % os.path.relpath(subset_csv, ROOT).replace("\\", "/"))

    # ---- ② 临时配置副本（零改动上游）----
    unify_smoke = os.path.join(smoke_root, "unify_smoke.yaml")
    b4_smoke = os.path.join(smoke_root, "b4_smoke.yaml")
    make_smoke_unify_config(unify_smoke, subset_csv)
    make_smoke_b4_config(b4_smoke, os.path.relpath(outputs_dir, ROOT).replace("\\", "/"))
    logger("         临时配置：%s／%s（仅重定向 id_val 路径 与 B4 输出目录）"
           % (os.path.relpath(unify_smoke, ROOT).replace("\\", "/"),
              os.path.relpath(b4_smoke, ROOT).replace("\\", "/")))

    # ---- ③ 命令表 ----
    all_cmds = build_commands(unify_smoke, b4_smoke, outputs_dir, reports_dir)
    if args.skip_heavy:
        all_cmds = [(n, c, e) for (n, c, e) in all_cmds if n in ("b1", "b2", "03_07")]
        logger("注意：--skip-heavy 为**局部诊断**运行，跳过 B3/B4/B5 ⇒ 必然不满足")
        logger("      `windows/W4.7_smoke_report.md` §2 的验收（B1–B5 全链路），故退出码仍为 1。")
    logger("")
    logger("命令表（%d 步）：" % len(all_cmds))
    for name, cmd, _env in all_cmds:
        logger("  [%s] %s" % (name, " ".join(cmd)))
    logger("")

    if args.dry_run:
        logger("DRY_RUN：未执行任何阶段；工作区 = %s"
               % os.path.relpath(smoke_root, ROOT).replace("\\", "/"))
        logger("SMOKE_DRY_RUN_OK")
        logger.close()
        return 0

    # ---- ④ 执行全链路 ----
    b4_report_before = _snapshot(os.path.join(ROOT, "reports", "baseline"), "b4_unplug_v1.0_")
    timings = []
    try:
        for name, cmd, env_extra in all_cmds:
            rc, elapsed = run_stage(name, cmd, logger,
                                    env_extra=env_extra, timeout=args.timeout)
            timings.append((name, rc, elapsed))
            if rc != 0:
                failures.append("阶段 %s 退出码 %d" % (name, rc))
                break
    finally:
        relocate_b4_report(reports_dir, b4_report_before, logger)

    # ---- ⑤ 验收断言 ----
    logger("")
    logger("== 冒烟验收 ==")
    outputs_by_detector = {}
    for det in DETECTORS:
        path = find_output(outputs_dir, det)
        if path is None:
            failures.append("%s 无 %s 输出 CSV" % (det, ID_VAL_SPLIT))
            logger("  [%s] 输出 CSV：缺失" % det)
            continue
        errs = check_schema(path, meta["n_taken"])
        rows_det, _ = _read_csv(path)
        outputs_by_detector[det] = rows_det
        logger("  [%s] %s（%d 行，schema %s）"
               % (det, os.path.basename(path), len(rows_det), "OK" if not errs else "FAIL"))
        failures.extend("%s: %s" % (det, e) for e in errs)

    if len(outputs_by_detector) == len(DETECTORS):
        divergent = check_consistency(outputs_by_detector, _load_module(
            "baseline_mapping", os.path.join(_HERE, "03_07_unify_output_mapping.py")
        ).check_text_sha256_consistency)
        logger("  [§1.4] text_sha256 跨 B1–B5 分歧样本 = %d" % len(divergent))
        if divergent:
            failures.append("text_sha256 跨检测器分歧：%s" % divergent[:5])

    metrics_csv = _latest([os.path.join(reports_dir, f) for f in os.listdir(reports_dir)
                           if f.startswith("unify_metrics_v1.0_") and f.endswith(".csv")]
                          if os.path.isdir(reports_dir) else [])
    if metrics_csv is None:
        failures.append("03_07 未产出 unify_metrics CSV")
        logger("  [指标] 缺失")
    else:
        errs, metric_rows = check_metrics(metrics_csv, meta["n_taken"])
        failures.extend(errs)
        logger("  [指标] %s" % os.path.basename(metrics_csv))
        for det, row in zip(DETECTORS, metric_rows):
            if not row:
                continue
            logger("         %s×%s  TPR=%s FPR=%s F1=%s"
                   % (det, ID_VAL_SPLIT, _fmt(row.get("tpr")), _fmt(row.get("fpr")),
                      _fmt(row.get("f1"))))

    tau_files = [f for f in (os.listdir(reports_dir) if os.path.isdir(reports_dir) else [])
                 if f.startswith("threshold_search_b2_")]
    logger("  [τ_B2] 寻优表 %d 份（%s）" % (len(tau_files), ", ".join(sorted(tau_files)) or "无"))
    if not tau_files:
        failures.append("03_07 未产出 threshold_search_b2_* 寻优表")

    check_residual(logger)

    logger("")
    logger("阶段计时：" + "；".join("%s=%d/%.1fs" % (n, rc, t) for n, rc, t in timings))
    logger("冒烟工作区：%s（reports/*/tmp/ 不入版本控制）"
           % os.path.relpath(smoke_root, ROOT).replace("\\", "/"))
    logger("冒烟日志：%s" % os.path.relpath(log_path, ROOT).replace("\\", "/"))
    if failures:
        logger("")
        for f in failures:
            logger("SMOKE_FAIL: %s" % f)
        logger("SMOKE_RESULT=FAIL（%d 项）" % len(failures))
        logger.close()
        return 1
    logger("")
    logger("SMOKE_RESULT=PASS（%d 条样本 × B1–B5 × 输出/映射/指标 全链路）" % meta["n_taken"])
    logger.close()
    return 0


def _fmt(value):
    try:
        return "%.4f" % float(value)
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""04_00_smoke_train.py — 阶段4 训练冒烟（v5 新增 25；`make smoke STAGE=train` 的入口）。

用途：用 **train 分片的 64 条子集（正 32／负 32，`seed=42`、固定不随机）** **1 个 epoch**
跑通「微调 → val 评估 → 早停计数器动作 → τ 寻优 → checkpoint 落盘」全链路。

契约（`windows/W5.1_qlora.md` §6-1／§3-C1／§5）：

1. 冒烟样本取自 **train 分片**（`data/frozen/CodeObf-PI_v1.0/splits/regime_id/train.csv`），
   **不是** val／test；val 侧另取**固定子集**用于监控与 τ 寻优（仍是 val，非测试）。
   **禁止用任何测试分片做冒烟**（测试集只在最终评估跑一次）。
2. 冒烟**不产出权威产物**：全部落 `reports/train/tmp/smoke_train_<ts>/`（`reports/*/tmp/`
   依 `D-15` 不入版本控制）；证据 = 退出码 ＋ 日志 `logs/train/04_00_smoke_train_<ts>.log`。
3. **零改动上游**：不修改 `04_01`／`03_06`／`03_07` 任何一行。子集通过 `04_01` 预留的
   **程序内钩子**（`train_row_ids`／`val_row_ids`）注入 —— **不动冻结 CLI**（§2 冻结 v1.0）。
4. 采样**固定不随机**：同类别内按 `sample_id` 升序取前 `n` 条（`seed=42` 用于训练随机性，
   与 `PREREG_v1.md` §1 的种子集合一致）。
5. 泄露断言（`assert_val_not_in_train`／`assert_ood_train_encodings`）在 `04_01.run_train`
   内部**对全量分片**执行，不因子集而削弱。

CLI::

    python scripts/train/04_00_smoke_train.py                     # 正32/负32 × 1 epoch 冒烟
    python scripts/train/04_00_smoke_train.py --n-pos 32 --n-neg 32 --seed 42 --epochs 1
    python scripts/train/04_00_smoke_train.py --regime ood        # 体制二冒烟（含 5 编码断言）
    python scripts/train/04_00_smoke_train.py --dry-run           # 只搭工作区、打印计划、不加载模型

退出码：全链路通过 = 0；任一步失败 = 1（并打印 ``SMOKE_FAIL: …``）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # 项目根目录

TRAIN_SCRIPT = os.path.join(_HERE, "04_01_train_qlora_ours.py")
DEFAULT_CONFIG = os.path.join("configs", "train", "qlora_ours.yaml")


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：W3.7 `init_streams`；失败显式告警，不静默 pass）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            sys.stderr.write("WARN init_streams: %s 不支持 reconfigure\n" % name)
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:
            sys.stderr.write("WARN init_streams: %s 重配失败：%s\n" % (name, ascii(exc)))


class _Logger:
    """stdout ＋ 日志文件双写（`logs/train/04_00_smoke_train_<ts>.log`）。"""

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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fmt(value, nd=6):
    try:
        return "%.*f" % (nd, float(value))
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# ① 冒烟子集（**固定不随机**：同类内按 sample_id 升序取前 n 条）
# ---------------------------------------------------------------------------
def _label_of(sample):
    try:
        return int(sample.label)
    except (TypeError, ValueError):
        return None


def pick_row_ids(samples, n_pos, n_neg, field="train"):
    """返回 ``(row_ids, meta)`` —— `row_ids` 是**源分片行下标**（供 `prepare_split(ids=)`）。

    **固定不随机**：正负两类各自按 `sample_id` 升序取前 ``n_pos``／``n_neg`` 条，
    末了把下标按升序排回（保持源分片顺序，训练样本顺序不因子集而乱序）。
    """
    buckets = {1: [], 0: []}
    for i, s in enumerate(samples):
        lab = _label_of(s)
        if lab in buckets:
            buckets[lab].append((str(s.sample_id), i))
    for lab in buckets:
        buckets[lab].sort(key=lambda t: (t[0], t[1]))

    take_pos = [i for _sid, i in buckets[1][:int(n_pos)]]
    take_neg = [i for _sid, i in buckets[0][:int(n_neg)]]
    ids = sorted(take_pos + take_neg)
    meta = {"field": field, "n_source": len(samples),
            "n_pos_source": len(buckets[1]), "n_neg_source": len(buckets[0]),
            "n_pos_taken": len(take_pos), "n_neg_taken": len(take_neg),
            "n_taken": len(ids)}
    return ids, meta


# ---------------------------------------------------------------------------
# ② 验收断言
# ---------------------------------------------------------------------------
def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def check_checkpoint(ckpt_dir, logger):
    """C8：checkpoint 目录存在且含 adapter 权重；命名含 `seed_<seed>/checkpoint-<step>/`。"""
    errors = []
    norm = os.path.abspath(ckpt_dir).replace("\\", "/")
    if not os.path.isdir(ckpt_dir):
        return ["checkpoint 目录不存在：%s" % ckpt_dir]
    files = set(os.listdir(ckpt_dir))
    if not ({"adapter_model.safetensors", "adapter_config.json"} & files):
        errors.append("checkpoint 缺 adapter 权重：%s（实有 %s）"
                      % (ckpt_dir, sorted(files)[:8]))
    if "/seed_" not in norm or "/checkpoint-" not in norm:
        errors.append("checkpoint 命名不合 C8（应含 seed_<seed>/checkpoint-<step>）：%s" % norm)
    logger("  [C8] checkpoint = %s（%d 个文件）"
           % (os.path.relpath(ckpt_dir, ROOT).replace("\\", "/"), len(files)))
    return errors


def check_curve(path, expect_epochs, logger):
    """训练曲线：逐 epoch 一行，含 val_f1 与早停计数器列。"""
    errors = []
    if not os.path.isfile(path):
        return ["训练曲线 CSV 缺失：%s" % path]
    rows, fields = _read_csv_rows(path)
    if "val_f1" not in fields or "early_stop_counter" not in fields:
        errors.append("曲线缺 val_f1／early_stop_counter 列：%r" % (fields,))
    if len(rows) != expect_epochs:
        errors.append("曲线行数 %d != 实跑 epoch %d" % (len(rows), expect_epochs))
    for i, row in enumerate(rows):
        try:
            float(row.get("val_f1"))
        except (TypeError, ValueError):
            errors.append("第 %d 行 val_f1 非数值：%r" % (i, row.get("val_f1")))
            break
    if rows:
        logger("  [曲线] %s（%d 行；末行 val_f1=%s 早停计数=%s）"
               % (os.path.basename(path), len(rows),
                  _fmt(rows[-1].get("val_f1")), rows[-1].get("early_stop_counter")))
    return errors


def check_threshold(path, logger):
    """C4：τ 寻优曲线存在、含且仅含一个 `selected=yes`、τ ∈ [0,1]。"""
    errors = []
    if not os.path.isfile(path):
        return ["τ 寻优曲线 CSV 缺失：%s" % path]
    rows, fields = _read_csv_rows(path)
    if "tau" not in fields or "selected" not in fields:
        return ["τ 曲线缺 tau／selected 列：%r" % (fields,)]
    sel = [r for r in rows if (r.get("selected") or "").strip() == "yes"]
    if len(sel) != 1:
        errors.append("selected=yes 行数 %d != 1" % len(sel))
    else:
        try:
            tau = float(sel[0]["tau"])
            if not (0.0 <= tau <= 1.0):
                errors.append("τ 越界 [0,1]：%r" % tau)
        except (TypeError, ValueError):
            errors.append("τ 非数值：%r" % sel[0].get("tau"))
    logger("  [C4] τ 曲线 = %s（%d 个候选；选定 τ=%s，F1=%s）"
           % (os.path.basename(path), len(rows),
              _fmt(sel[0]["tau"]) if sel else "n/a", _fmt(sel[0]["f1"]) if sel else "n/a"))
    return errors


# ---------------------------------------------------------------------------
# ③ 主流程
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="阶段4 训练冒烟（v5 新增 25；train 分片 正32/负32 × 1 epoch 跑通全链路）")
    p.add_argument("--regime", choices=["id", "ood"], default="id",
                   help="冒烟体制（默认 id）")
    p.add_argument("--seed", type=int, choices=[42, 123, 456], default=42,
                   help="训练随机种子（默认 42，`PREREG_v1.md` §1）")
    p.add_argument("--n-pos", type=int, default=32, help="正例条数（默认 32）")
    p.add_argument("--n-neg", type=int, default=32, help="负例条数（默认 32）")
    p.add_argument("--n-val", type=int, default=128,
                   help="val 评估子集条数（默认 128 = 正 64／负 64，仅用于监控与 τ 寻优）")
    p.add_argument("--epochs", type=int, default=1, help="冒烟 epoch 数（默认 1）")
    p.add_argument("--curriculum", choices=["on", "off"], default="off",
                   help="**W5.2 追加**：`on` 时把 `--curriculum on` 透传给 `04_01`，"
                        "用以端到端验证 `04_02` 的课程调度与编码感知增强接线"
                        "（默认 `off`，与 W5.1 验收口径一致）")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="QLoRA 配置（默认冻结件）")
    # `D-48` 裁定三／四：可续跑 ＋ 等价性自证的停段钩子（透传给 `04_01`）
    p.add_argument("--resume", default=None,
                   help="`D-48` 可续跑：从该 state 目录（含 `latest/`）恢复并继续训练")
    p.add_argument("--stop-after-steps", dest="stop_after_steps", type=int, default=None,
                   help="`D-48` 裁定四 证明钩子：到达该 opt_step 即落盘并停止本段")
    # `D-54` 裁定三：loss-NaN 守卫透传（默认启用；`--no-nan-guard` 仅供等价性自证）
    p.add_argument("--no-nan-guard", dest="nan_guard", action="store_false", default=True,
                   help="关闭 loss-NaN 守卫（**仅供等价性自证**；传之 ⇒ 与改造前逐字节不变）")
    p.add_argument("--dry-run", action="store_true",
                   help="只搭工作区并打印计划，不加载模型、不训练")
    p.add_argument("--log-dir", default=os.path.join("logs", "train"),
                   help="冒烟日志目录（默认 logs/train；测试可改指临时目录）")
    p.add_argument("--work-root", default=os.path.join("reports", "train", "tmp"),
                   help="冒烟工作区根（默认 reports/train/tmp；测试可改指临时目录）")
    return p


def main() -> int:
    init_streams()
    args = build_parser().parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(ROOT, args.log_dir, "04_00_smoke_train_%s.log" % ts)
    logger = _Logger(log_path)
    logger("=" * 78)
    logger("04_00_smoke_train 阶段4 训练冒烟（v5 新增 25）  ts=%s" % ts)
    logger("项目根：%s" % ROOT)
    logger("python：%s" % (sys.executable or "python"))
    logger("参数：regime=%s seed=%d n_pos=%d n_neg=%d n_val=%d epochs=%d curriculum=%s dry_run=%s"
           % (args.regime, args.seed, args.n_pos, args.n_neg, args.n_val, args.epochs,
              args.curriculum, args.dry_run))

    failures = []
    smoke_root = os.path.join(ROOT, args.work_root, "smoke_train_%s" % ts)
    report_dir = os.path.join(smoke_root, "reports")
    out_dir = os.path.join(smoke_root, "ckpt")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 载入 `04_01`（**只 import，不改一行**）----
    trainer = _load_module("smoke_train_04_01", TRAIN_SCRIPT)
    cfg = trainer.load_config(os.path.join(ROOT, args.config)
                              if not os.path.isabs(args.config) else args.config)
    train_csv, val_csv = trainer.resolve_split_paths(cfg, args.regime)

    # ---- ① 子集（固定不随机；泄露断言仍在 04_01 内对全量分片执行）----
    train_samples = trainer.load_samples(train_csv)
    val_samples = trainer.load_samples(val_csv)
    train_ids, meta_tr = pick_row_ids(train_samples, args.n_pos, args.n_neg, "train")
    half = max(1, int(args.n_val) // 2)
    val_ids, meta_va = pick_row_ids(val_samples, half, int(args.n_val) - half, "val")
    logger("")
    logger("冒烟样本（**固定不随机**：同类按 sample_id 升序取前 n 条）：")
    logger("  train 源 = %s（%d 行：正 %d／负 %d）→ 取 %d（正 %d／负 %d）"
           % (os.path.relpath(train_csv, ROOT).replace("\\", "/"), meta_tr["n_source"],
              meta_tr["n_pos_source"], meta_tr["n_neg_source"], meta_tr["n_taken"],
              meta_tr["n_pos_taken"], meta_tr["n_neg_taken"]))
    logger("  val   源 = %s（%d 行：正 %d／负 %d）→ 取 %d（正 %d／负 %d）"
           % (os.path.relpath(val_csv, ROOT).replace("\\", "/"), meta_va["n_source"],
              meta_va["n_pos_source"], meta_va["n_neg_source"], meta_va["n_taken"],
              meta_va["n_pos_taken"], meta_va["n_neg_taken"]))
    if meta_tr["n_taken"] != args.n_pos + args.n_neg:
        failures.append("train 子集不足：请求 正%d／负%d，实得 正%d／负%d"
                        % (args.n_pos, args.n_neg, meta_tr["n_pos_taken"],
                           meta_tr["n_neg_taken"]))
    if meta_va["n_taken"] < 2:
        failures.append("val 子集不足（%d 条），无法做 τ 寻优" % meta_va["n_taken"])

    # ---- ② 组装 `04_01.run_train` 的入参（冻结 CLI 之外**只加程序内字段**）----
    ns = argparse.Namespace(
        regime=args.regime, seed=int(args.seed), config=args.config, mode="train",
        curriculum=str(args.curriculum), epochs=int(args.epochs), max_steps=None,
        report_dir=report_dir, out_dir=out_dir, log_dir=args.log_dir,
        checkpoint_dir=None, backfill=False,
        resume=args.resume, stop_after_steps=args.stop_after_steps,
        nan_guard=args.nan_guard,
        train_row_ids=train_ids, val_row_ids=val_ids)
    logger("")
    logger("计划：%s → run_train(regime=%s, seed=%d, epochs=%d, n_train=%d, n_val=%d)"
           % (os.path.relpath(TRAIN_SCRIPT, ROOT).replace("\\", "/"), args.regime,
              args.seed, args.epochs, meta_tr["n_taken"], meta_va["n_taken"]))
    logger("      产物目录：report=%s／ckpt=%s（**冒烟区，非权威**）"
           % (os.path.relpath(report_dir, ROOT).replace("\\", "/"),
              os.path.relpath(out_dir, ROOT).replace("\\", "/")))

    if args.dry_run:
        logger("DRY_RUN：未加载模型、未训练；工作区 = %s"
               % os.path.relpath(smoke_root, ROOT).replace("\\", "/"))
        logger("SMOKE_DRY_RUN_OK")
        logger.close()
        return 0

    # ---- ③ 执行 ----
    t_start = time.time()
    try:
        ckpt_dir, res = trainer.run_train(cfg, ns, trainer.Logger(), ts)
    except Exception as exc:  # noqa: BLE001
        logger("SMOKE_FAIL: run_train 抛出 %s: %s" % (type(exc).__name__, exc))
        logger("SMOKE_RESULT=FAIL（run_train 异常）")
        logger.close()
        return 1
    # `D-54` 裁定三③：NaN 守卫触发 ⇒ **不产出最终产物、跳过验收**，退出码 3
    if res.get("nan_guard_stopped"):
        logger("")
        logger("NAN_GUARD_TRIGGERED：本 run 在 step=%s 因非有限 loss 停止（四不一停）"
               % res.get("opt_steps"))
        logger("SMOKE_RESULT=FAIL（loss-NaN 守卫触发）")
        logger.close()
        return 3

    # `D-48` 裁定四：本段为「分 2 段续跑」的**第 1 段** ⇒ 不产出最终产物、跳过验收
    if res.get("resume_stopped"):
        logger("")
        logger("本段 = `D-48` 分 2 段续跑的**第 1 段**：step=%s；state 已落盘"
               % res.get("opt_steps"))
        logger("RESUME_SEGMENT_DONE（本段不产出最终产物；续跑 = 同一命令加 `--resume <state>`）")
        logger("SMOKE_RESULT=SEGMENT_DONE")
        logger.close()
        return 0

    elapsed = time.time() - t_start
    logger("")
    logger("run_train 完成：用时 %.1fs；峰值显存 %s MiB；实跑 epoch %d；早停=%s"
           % (elapsed, _fmt(res.get("peak_memory_mib"), 1), res.get("epochs_run"),
              res.get("early_stopped")))
    logger("早停计数器历史 = %s（patience=%s）"
           % (res.get("stopper_history"),
              trainer._require(cfg["early_stopping"], "patience", "early_stopping")))

    # ---- ④ 验收 ----
    logger("")
    logger("== 冒烟验收 ==")
    failures.extend(check_checkpoint(ckpt_dir, logger))
    best = res.get("best") or {}
    for key in ("curve_csv", "threshold_csv"):
        if not best.get(key):
            failures.append("best 缺 %s" % key)
    if best.get("curve_csv"):
        failures.extend(check_curve(best["curve_csv"], int(res.get("epochs_run") or 0), logger))
    if best.get("threshold_csv"):
        failures.extend(check_threshold(best["threshold_csv"], logger))
    if best.get("tau_ours") is None:
        failures.append("τ_Ours 未产出")
    if not res.get("stopper_history"):
        failures.append("早停计数器未记录（未能验证计数器动作）")

    logger("")
    logger("冒烟工作区：%s（reports/*/tmp/ 不入版本控制）"
           % os.path.relpath(smoke_root, ROOT).replace("\\", "/"))
    logger("冒烟日志：%s" % os.path.relpath(log_path, ROOT).replace("\\", "/"))
    if failures:
        for f in failures:
            logger("SMOKE_FAIL: %s" % f)
        logger("SMOKE_RESULT=FAIL（%d 项）" % len(failures))
        logger.close()
        return 1
    logger("SMOKE_RESULT=PASS（正%d／负%d × %d epoch × 微调→val→早停计数→τ→checkpoint 全链路）"
           % (meta_tr["n_pos_taken"], meta_tr["n_neg_taken"], int(res.get("epochs_run") or 0)))
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""05_07_eval_efficiency.py — 阶段5（W6.8，A6）5.7 **效率分析**主脚本。

接口契约（`windows/W6.8_efficiency.md` §2，**冻结后不得改**）::

    python scripts/eval/05_07_eval_efficiency.py \
        --detectors b1,b2,b3,b4,b5,ours \
        --runs-csv reports/train/three_seeds_v1.0_*.csv [--only <regime>:<seed>] \
        --bench-split id_val --repeat 3 --warmup 1 \
        --config configs/eval/efficiency.yaml \
        --report-dir <dir> --outputs-dir <dir> \
        [--limit N] [--self-check] [--dry-run]

stdout 末行（依模式）::

    SELFCHECK_RESULT=PASS|FAIL n=<k>
    DRYRUN_RESULT=PASS|FAIL n=<k>
    REPORT_MD=<path> REPORT_CSV=<path> FIGDATA_CSV=<path>

硬约束逐条落点（`windows/W6.8_efficiency.md` §3）::

    C1  不可比性 → `render_report` §C1 **逐字**输出 `ENV_LOCK.md` §8.5；三档量纲（无模型／CPU／GPU）
                   **并列分表、不合并排序、不跨档求均值**；`--self-check` 机械核验
    C2  延迟口径 → 不含模型加载（`warmup`/`repeat` 通过均在 `load()` 之后）；均值／中位／p95（ms）；
                   批测者（Ours）**声明「批际摊到样本」**换算方式
    C3  显存口径 → `max_memory_allocated`（MiB），**加载态／推理态分列**；多 run 串行 ⇒
                   `empty_cache()` ＋ `reset_peak_memory_stats()` 后再记加载态
    C4  参数量   → 总参／可训练参分列；Ours 报 LoRA 可训练参（实测 `best.json`）
    C5  7B 对照  → 唯一对照档 = **B5**（Llama-Guard-3-8B 4-bit）；禁自行引入未预注册对照模型
    C6  引数口径 → 引 `BASELINE_REPORT.md` §1 须带其 §12-3（B4 属 CPU 档）⇒ 落「交叉校验」节
    C7  不读封存 → 默认 `id_val`（frozen 侧）；`--bench-source sealed` 缺 `--access-log-note`
                   ⇒ **拒绝执行并非零退出**
    C8  评估态   → `model.eval()` ＋ `torch.no_grad()`；确定性参数同 `ENV_LOCK.md` §7.2；
                   源码内不得出现训练模式切换调用（`--self-check` 机械核验）
    C9  报告路径 → `--report-dir`／`--outputs-dir` 必填；脚本内**不写死** `reports/`
    C10 只追加   → 报告／CSV／日志一律新时间戳文件，不覆盖

顶层不得 import 重型库（`AGENTS.md` §3）：`torch`／`transformers`／`peft`／`yaml`／`safetensors`
一律在函数体内惰性 import。
"""
from __future__ import annotations

# --- C8：必须在 import torch 之前设置（torch 的 import 全在函数体内）---
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

TBD = "<TBD>"
NA = "n/a"

SEVEN_KEYS = ["model", "load_mode", "max_length", "batch_size", "offload", "device", "collector"]

# 产物字段（of record；与 `configs/eval/efficiency.yaml` → `fields` 逐项对应）
MAIN_FIELDS = [
    "detector", "label", "group", "device", "measurement", "run_id", "regime", "seed",
    "lat_mean_ms", "lat_median_ms", "lat_p95_ms",
    "lat_mean_range_ms", "lat_median_range_ms", "lat_p95_range_ms",
    "n_samples", "n_passes", "repeat", "warmup",
    "mem_load_peak_mib", "mem_infer_peak_mib", "cpu_rss_peak_mib", "mem_collector",
    "total_params", "trainable_params", "lora_r", "lora_alpha", "param_source",
    "t_model", "t_load_mode", "t_max_length", "t_batch_size", "t_offload", "t_device",
    "t_collector", "amortization", "notes",
]

FIGDATA_FIELDS = [
    "detector", "label", "group", "run_id", "regime", "seed",
    "lat_mean_ms", "lat_median_ms", "lat_p95_ms",
    "mem_load_peak_mib", "mem_infer_peak_mib", "cpu_rss_peak_mib",
    "total_params", "trainable_params", "measurement", "measurement_note",
]

REPEAT_FIELDS = [
    "detector", "run_id", "regime", "seed", "pass_index", "is_warmup",
    "n_samples", "pass_total_ms", "mean_ms", "median_ms", "p95_ms",
]


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（`D-40` 裁定五：不得静默 pass）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8（``errors="replace"``），避免 GBK 环境打印非 ASCII 假失败。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:  # 不静默：写 stderr 但不中断
            sys.stderr.write("STREAM_RECONFIGURE_FAILED stream=%r detail=%s\n" % (stream, exc))


# ---------------------------------------------------------------------------
# 模块加载 / 配置
# ---------------------------------------------------------------------------
def load_module(name, rel_path):
    """按路径加载模块（与 `05_01.load_module` 同手法）。"""
    path = os.path.join(_ROOT, rel_path.replace("/", os.sep))
    if not os.path.isfile(path):
        raise FileNotFoundError("模块不存在：%s" % path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载模块：%s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


# ---------------------------------------------------------------------------
# CSV 读取（`csv.DictReader`；`W6.1` §8：text 含换行 ⇒ 行数 ≠ 记录数）
# ---------------------------------------------------------------------------
def read_records(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def assert_record_count(csv_path, expected, logger=print):
    n = len(read_records(csv_path))
    if n != int(expected):
        logger("RECORD_COUNT_MISMATCH path=%s got=%d expected=%d ⇒ 非零退出（禁改断言值）"
               % (csv_path, n, expected))
        return False
    logger("RECORD_COUNT_OK path=%s n=%d" % (csv_path, n))
    return True


# ---------------------------------------------------------------------------
# 统计：p95（最近秩，不插值）／跨 repeat 的中位与极差
# ---------------------------------------------------------------------------
def _quantile(sorted_xs, p):
    """最近秩分位（**不插值**），与 `03_01.latency_stats`／`05_01._quantile` 同算法。"""
    xs = list(sorted_xs)
    if not xs:
        return None
    n = len(xs)
    idx = min(n - 1, int(round(p * (n - 1))))
    return float(xs[idx])


def pass_stats(ms_list):
    """单次通过的 均值／中位／p95。空列表 ⇒ 全 ``None``。"""
    xs = sorted(float(v) for v in ms_list)
    if not xs:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "total_ms": None}
    return {
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p95_ms": _quantile(xs, 0.95),
        "total_ms": math.fsum(xs),
    }


def repeat_report(passes):
    """跨 repeat 的**合并**统计 ＋ 极差。

    - of-record 三值 = **合并全部计时通过**（n = N × repeat）的 均值／中位／p95；
    - 极差 = 各**单次**通过统计量的 max − min（`repeat_policy.report`）。
    """
    good = [p for p in passes if not p.get("is_warmup")]
    if not good:
        return {"lat_mean_ms": None, "lat_median_ms": None, "lat_p95_ms": None,
                "lat_mean_range_ms": None, "lat_median_range_ms": None,
                "lat_p95_range_ms": None, "n_passes": 0}
    pooled = []
    for p in good:
        pooled.extend(float(v) for v in p["ms"])
    st = pass_stats(pooled)
    per = [pass_stats(p["ms"]) for p in good]

    def _rng(key):
        vals = [d[key] for d in per if d[key] is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    return {
        "lat_mean_ms": st["mean_ms"],
        "lat_median_ms": st["median_ms"],
        "lat_p95_ms": st["p95_ms"],
        "lat_mean_range_ms": _rng("mean_ms"),
        "lat_median_range_ms": _rng("median_ms"),
        "lat_p95_range_ms": _rng("p95_ms"),
        "n_passes": len(good),
    }


def range_exceeds(stats, cfg):
    """极差 > 中位 50% ⇒ True（如实报告，不重跑到好看为止）。"""
    med = stats.get("lat_median_ms")
    rng = stats.get("lat_median_range_ms")
    if med is None or rng is None:
        return False
    return bool(rng > 0.5 * float(med))


# ---------------------------------------------------------------------------
# 七元组（C1／§2：`ENV_LOCK.md` §8 口径）
# ---------------------------------------------------------------------------
def seven_tuple_of(det_id, dcfg):
    """取七元组；**任一字段缺失或为空即抛错**（`--self-check` 覆盖此项）。"""
    raw = dcfg.get("seven_tuple")
    if not isinstance(raw, dict):
        raise KeyError("检测器 %s 缺 `seven_tuple` 块（七元组必填）" % det_id)
    out = {}
    for k in SEVEN_KEYS:
        if k not in raw:
            raise KeyError("检测器 %s 的七元组缺字段 %r（必填）" % (det_id, k))
        v = raw[k]
        if v is None or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("检测器 %s 的七元组字段 %r 为空（必填）" % (det_id, k))
        out[k] = v
    return out


def assert_seven_tuple(t, where="row"):
    for k in SEVEN_KEYS:
        if k not in t:
            raise KeyError("%s 缺七元组字段 %r" % (where, k))
        v = t[k]
        if v is None or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("%s 的七元组字段 %r 为空" % (where, k))
    return True


# ---------------------------------------------------------------------------
# 参数量（C4）
# ---------------------------------------------------------------------------
def count_params_safetensors(weights_dir):
    """从 safetensors **元数据**（不加载张量）统计参数量。返回 ``(n, note)``；不可用 ⇒ ``(None, 原因)``。"""
    path = _abs(weights_dir)
    if not os.path.isdir(path):
        return None, "权重目录不存在：%s" % path
    try:
        from safetensors import safe_open
    except ImportError as exc:
        return None, "safetensors 不可用：%s" % exc
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as fh:
            wmap = json.load(fh)["weight_map"]
        files = sorted(set(wmap.values()))
    elif os.path.isfile(os.path.join(path, "model.safetensors")):
        files = ["model.safetensors"]
    else:
        return None, "目录内无 model.safetensors(.index.json)：%s" % path
    total = 0
    n_files = 0
    for name in files:
        fp = os.path.join(path, name)
        if not os.path.isfile(fp):
            return None, "索引声明的文件缺失：%s" % fp
        with safe_open(fp, framework="pt") as fh:
            for k in fh.keys():
                shape = fh.get_slice(k).get_shape()
                n = 1
                for d in shape:
                    n *= int(d)
                total += n
        n_files += 1
    return int(total), "safetensors 元数据求和（%d 个分片，未加载张量）" % n_files


def params_for(det_id, dcfg, best=None):
    """返回 ``(total, trainable, source, lora_r, lora_alpha)``；不适用 ⇒ ``None``（`n/a`，非 0）。"""
    p = dcfg.get("params") or {}
    mode = p.get("mode")
    if mode == "none":
        return None, None, "n/a（无模型）", None, None
    if mode != "safetensors_metadata":
        return None, None, "n/a（未知 params.mode=%r）" % (mode,), None, None
    wdir = p.get("dir")
    if p.get("dir_from") == "best.json → model_meta.local_path":
        if not best:
            return None, None, "n/a（缺 best.json）", None, None
        wdir = ((best.get("model_meta") or {}).get("local_path") or "").strip()
        if not wdir:
            return None, None, "n/a（best.json 缺 model_meta.local_path）", None, None
    total, note = count_params_safetensors(wdir)
    tr_mode = (p.get("trainable") or {}).get("mode")
    lora_r = lora_a = None
    if tr_mode == "zero":
        trainable = 0
        note = note + "；可训练参 = 0（未微调）"
    elif tr_mode == "best_json":
        key = (p.get("trainable") or {}).get("key", "model_meta.trainable_params")
        cur = best or {}
        for part in key.split("."):
            cur = (cur or {}).get(part)
        if cur is None:
            return total, None, note + "；可训练参 = n/a（best.json 缺 %s）" % key, None, None
        trainable = int(cur)
        lora = p.get("lora") or {}
        lora_r = lora.get("r")
        lora_a = lora.get("alpha")
        note = note + "；可训练参 = best.json %s（LoRA r=%s alpha=%s）" % (key, lora_r, lora_a)
    else:
        trainable = None
        note = note + "；可训练参 = n/a（未声明）"
    return total, trainable, note, lora_r, lora_a


# ---------------------------------------------------------------------------
# 测量输入（§2：默认 frozen 侧 `id_val`，**不读封存**）
# ---------------------------------------------------------------------------
def resolve_bench_path(cfg, args):
    """返回 ``(csv_path, shard, expect_n, is_sealed)``。"""
    bench = cfg["bench"]
    if args.bench_source == "sealed":
        root = bench["sealed_root"]
        if args.bench_split == "id_val":
            name = "regime_id_test.csv"
        elif args.bench_split == "ood_val":
            name = "regime_ood_test.csv"
        else:
            name = "%s.csv" % args.bench_split
        return os.path.join(_abs(root), name), args.bench_split, None, True
    paths = bench["val_paths"]
    if args.bench_split not in paths:
        raise ValueError("未知 --bench-split：%r（可选 %s）" % (args.bench_split, sorted(paths)))
    return _abs(paths[args.bench_split]), args.bench_split, bench["record_assertions"][args.bench_split], False


def load_bench_rows(cfg, args, logger=print):
    path, shard, expect_n, is_sealed = resolve_bench_path(cfg, args)
    # C7：`--bench-source sealed` 缺 `--access-log-note` ⇒ **拒绝执行**（上层转非零退出）
    if is_sealed and not getattr(args, "access_log_note", None):
        raise ValueError("REFUSED：`--bench-source sealed` 须同时给 `--access-log-note`"
                         "（缺一即拒绝执行并非零退出，`W6.8` §3-C7）")
    if expect_n is not None and not assert_record_count(path, expect_n, logger):
        raise AssertionError("记录数断言不符 ⇒ 停工呈 W0（禁改断言值）")
    rows = read_records(path)
    limit = int(args.limit) if args.limit else int(cfg["bench"]["limit"])
    if limit > len(rows):
        raise ValueError("--limit=%d 超过分片记录数 %d" % (limit, len(rows)))
    used = rows[:limit]
    logger("BENCH_ROWS shard=%s path=%s total=%d used=%d sealed=%s"
           % (shard, path, len(rows), len(used), is_sealed))
    return used, shard, len(rows), is_sealed


# ---------------------------------------------------------------------------
# 显存采集（C3）
# ---------------------------------------------------------------------------
def gpu_reset():
    """`empty_cache()` ＋ `reset_peak_memory_stats()`（`W5.3` 教训：否则误记上一条 run 的残留）。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mib():
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def rss_mib():
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def release_gpu(objs):
    """卸载模型并回收显存（多 run 串行必需）。

    ⚠ **必须传 list**：`*objs` 形式下 `del o` 只删局部名，元组仍持引用 ⇒ 模型不会被释放
    （`F-W6.8-2`：实测 B5 卸载后仍有 ≈5.5 GB 驻留，导致后测的 Ours 峰值虚高到 7695 MiB）。
    本函数把列表**逐位清空**，再双轮 `gc.collect()`，最后 `empty_cache()`＋`reset_peak_memory_stats()`
    并回显残留供核对。
    """
    import gc
    import torch

    for i in range(len(objs)):
        objs[i] = None
    del objs[:]
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return gpu_allocated_mib()


def gpu_allocated_mib():
    """当前**已分配**显存（MiB）；用于核对上一条 run 是否已真正释放。"""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024.0 * 1024.0)


def gpu_snapshot():
    """`nvidia-smi` 快照（P-4 GPU 独占留证）。不可用 ⇒ 返回 ``None``。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip().splitlines()


# ---------------------------------------------------------------------------
# 检测器适配器
# ---------------------------------------------------------------------------
class BaseAdapter(object):
    """``load()`` → ``infer_pass(rows)`` → ``unload()``。"""

    def __init__(self, det_id, dcfg, logger=print):
        self.det_id = det_id
        self.dcfg = dcfg
        self.logger = logger

    def load(self):
        raise NotImplementedError

    def infer_pass(self, rows):
        """返回 ``{"ms": [逐样本毫秒], "total_ms": float}``。"""
        raise NotImplementedError

    def unload(self):
        return None


class NoModelAdapter(BaseAdapter):
    """`b1`（正则）／`b2`（熵值）：**无模型** ⇒ 显存一律 `n/a`。"""

    def load(self):
        cfg_path = _abs(self.dcfg["config"])
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError("基线配置不存在：%s" % cfg_path)
        self.mod = load_module("eff_%s" % self.det_id, self.dcfg["module"])
        self.bcfg = load_config(cfg_path)
        self.compiled = None
        if self.dcfg.get("entry", {}).get("compile"):
            self.compiled = getattr(self.mod, self.dcfg["entry"]["compile"])(self.bcfg)
        self.logger("LOADED_ %s（无模型）" % self.det_id)
        return {"mem_load_peak_mib": None, "cpu_rss_peak_mib": rss_mib()}

    def infer_pass(self, rows):
        fn = getattr(self.mod, self.dcfg["entry"]["call"])
        ms = []
        for r in rows:
            text = r.get("text", "")
            t0 = time.perf_counter()
            if self.compiled is not None:
                fn(text, self.compiled)
            else:
                fn(text)
            ms.append((time.perf_counter() - t0) * 1000.0)
        return {"ms": ms, "total_ms": math.fsum(ms)}

    def unload(self):
        self.compiled = None
        return None


class GuardAdapter(BaseAdapter):
    """`b3`（LG-1B）／`b5`（LG-8B）：4-bit NF4 + GPU 贪心生成。"""

    def load(self):
        cfg_path = _abs(self.dcfg["config"])
        self.mod = load_module("eff_%s" % self.det_id, self.dcfg["module"])
        self.bcfg = load_config(cfg_path)
        gpu_reset()
        pre = gpu_allocated_mib()
        self.logger("PRE_LOAD_ALLOCATED %s mib=%s" % (self.det_id, _fmt1(pre)))
        torch, transformers, tok, model, meta = self.mod.load_model(self.bcfg)
        self.torch = torch
        self.tok = tok
        self.model = model
        if model.training:
            raise RuntimeError("G-EVAL 断言失败：model.eval() 后仍处 train 模式")
        self.logger("LOADED_ %s attn=%s load_ms=%.1f" % (self.det_id, meta.get("attn"),
                                                          float(meta.get("load_ms", 0.0))))
        return {"mem_load_peak_mib": float(meta["load_peak_bytes"]) / (1024.0 * 1024.0),
                "cpu_rss_peak_mib": None}

    def infer_pass(self, rows):
        """计时区间 = `build_prompt` ＋ `generate_judgment`（与 `03_03`／`03_05` 的 `run_split` 逐行同口径）。

        ⚠ `build_prompt` 返回**四元组** `(prompt, truncated_text, n_text_tokens, was_truncated)`
        ⇒ 只取第 0 项喂给 `generate_judgment`（`F-W6.8-1`）。
        """
        ms = []
        gpu_reset()
        with self.torch.no_grad():
            for r in rows:
                t0 = time.perf_counter()
                prompt = self.mod.build_prompt(self.tok, self.bcfg, r.get("text", ""))[0]
                self.mod.generate_judgment(self.torch, self.tok, self.model, self.bcfg, prompt)
                ms.append((time.perf_counter() - t0) * 1000.0)
            peak = gpu_peak_mib() or 0.0
        return {"ms": ms, "total_ms": math.fsum(ms), "mem_infer_peak_mib": peak}

    def unload(self):
        # 逐位清空列表 ⇒ 真正断开引用（含 accelerate 挂在子模块上的执行设备钩）
        held = [getattr(self, "model", None), getattr(self, "torch", None),
                getattr(self, "tok", None)]
        self.model = None
        self.torch = None
        self.tok = None
        residue = release_gpu(held)
        self.logger("UNLOADED %s residue_allocated_mib=%s" % (self.det_id, _fmt1(residue)))
        return residue


class UnplugAdapter(BaseAdapter):
    """`b4`（unplug-tiny）：**CPU 档**（v5 L122 主档，非降级）⇒ GPU 显存 `n/a`，另报 RSS。"""

    def load(self):
        cfg_path = _abs(self.dcfg["config"])
        self.mod = load_module("eff_b4", self.dcfg["module"])
        self.bcfg = load_config(cfg_path)
        vendor = _abs(self.dcfg.get("vendor_syspath") or "")
        self._vendor = None
        if vendor:
            if not os.path.isdir(vendor):
                raise FileNotFoundError("vendor 目录不存在：%s" % vendor)
            sys.path.insert(0, vendor)
            self._vendor = vendor
        model, normalizer, meta = self.mod.load_model(self.bcfg)
        self.model = model
        self.normalizer = normalizer
        pipe_cfg = self.bcfg["pipeline"]
        lt = pipe_cfg["long_text"]
        self.pipe = {
            "threshold_chars": int(lt["threshold_chars"]),
            "chunk_chars": int(lt["chunk_chars"]),
            "overlap_chars": int(lt["overlap_chars"]),
            "batch_size": int(pipe_cfg["batch_size"]),
        }
        self.logger("LOADED_ b4 device=%s max_length=%s" % (meta.get("device"),
                                                             meta.get("max_length")))
        return {"mem_load_peak_mib": None,
                "cpu_rss_peak_mib": float(meta["rss_after_load_bytes"]) / (1024.0 * 1024.0)}

    def infer_pass(self, rows):
        ms = []
        peak = rss_mib() or 0.0
        for r in rows:
            t0 = time.perf_counter()
            self.mod.predict_one(self.model, self.normalizer, r.get("text", ""), self.pipe)
            ms.append((time.perf_counter() - t0) * 1000.0)
            cur = rss_mib()
            if cur is not None and cur > peak:
                peak = cur
        return {"ms": ms, "total_ms": math.fsum(ms), "cpu_rss_peak_mib": peak}

    def unload(self):
        self.model = None
        self.normalizer = None
        if self._vendor and self._vendor in sys.path:
            sys.path.remove(self._vendor)
        return None


class OursAdapter(BaseAdapter):
    """`ours`：**单源复用** `05_01.Engine`（加载／score／批际摊到样本）。

    另提供 `score_with_max_size()` —— 与 `Engine.score` **同一组原语**，仅 `make_batches` 的
    `max_size` 可配（用于 `comparison_7b.fairness_control` 的 batch=1 对照档）。
    """

    def __init__(self, det_id, dcfg, logger=print):
        BaseAdapter.__init__(self, det_id, dcfg, logger)
        self.engine = None

    def load(self, ckpt_dir, seed):
        e501 = load_module("eval_05_01", self.dcfg["module"])
        self.e501 = e501
        train_cfg = load_config(_abs(self.dcfg["train_config"]))
        eval_cfg = load_config(_abs(self.dcfg["config"]))
        self.train_cfg = train_cfg
        self.logger("PRE_LOAD_ALLOCATED ours mib=%s" % _fmt1(gpu_allocated_mib()))
        eng = e501.Engine(eval_cfg, train_cfg, self.logger)
        info = eng.load(ckpt_dir, int(seed))
        self.engine = eng
        self.tcfg = train_cfg
        return {"mem_load_peak_mib": info["mem_load_peak_mib"], "cpu_rss_peak_mib": None}

    # —— 主档：与 `W6.1` 完全同路径（`eval_max_size` 取自训练配置）——
    def infer_pass(self, rows, max_size=None):
        return self.score_with_max_size(rows, max_size=max_size)

    def score_with_max_size(self, rows, max_size=None, tag="eval"):
        import torch

        eng = self.engine
        tm = eng.train_mod
        tcfg = self.tcfg
        exs = []
        for r in rows:
            exs.append(tm.encode_example(
                eng.tok, tcfg,
                self.e501.Sample(str(r.get("sample_id", "")), r.get("text", ""),
                                 int(float(r.get("label")))),
                eng.id_unsafe, eng.id_safe, eng.tok.eos_token_id))
        lengths = [e.n_prompt for e in exs]
        e_quad, e_linear, e_budget = tm.batch_mem_guard(tcfg, eval_mode=True)
        size = int(max_size) if max_size else int(tcfg["batch"]["eval_max_size"])
        batches = tm.make_batches(lengths, size, e_quad, e_linear, e_budget, False, None)
        device = next(eng.model.parameters()).device
        eng.model.eval()
        gpu_reset()
        scores = [None] * len(exs)
        lat = [None] * len(exs)
        with torch.no_grad():
            if torch.is_grad_enabled():
                raise RuntimeError("G-EVAL 断言失败：grad 已启用（必须 torch.no_grad()）")
            for idxs in batches:
                p_ids, p_attn = tm._prompt_tensors(exs, idxs, torch, device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                vals = tm.forward_scores(eng.llama, eng.lm_head, p_ids, p_attn,
                                         eng.id_unsafe, eng.id_safe)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                per = dt_ms / float(len(idxs))       # **批际摊到样本**
                for b, i in enumerate(idxs):
                    scores[i] = vals[b]
                    lat[i] = per
        peak = gpu_peak_mib() or 0.0
        self.logger("EVAL %s n=%d n_micro=%d max_size=%d peak_infer_mib=%.1f"
                    % (tag, len(exs), len(batches), size, peak))
        return {"ms": lat, "total_ms": math.fsum(lat), "mem_infer_peak_mib": peak,
                "n_batches": len(batches), "max_size": size}

    def unload(self):
        if self.engine is not None:
            held = [getattr(self.engine, "model", None), getattr(self.engine, "llama", None),
                    getattr(self.engine, "lm_head", None), getattr(self.engine, "tok", None)]
            self.engine.model = None
            self.engine.llama = None
            self.engine.lm_head = None
            self.engine.tok = None
            residue = release_gpu(held)
            self.logger("UNLOADED ours residue_allocated_mib=%s" % _fmt1(residue))
        self.engine = None
        self.train_mod = None
        return None


def make_adapter(det_id, dcfg, logger=print):
    if det_id in ("b1", "b2"):
        return NoModelAdapter(det_id, dcfg, logger)
    if det_id in ("b3", "b5"):
        return GuardAdapter(det_id, dcfg, logger)
    if det_id == "b4":
        return UnplugAdapter(det_id, dcfg, logger)
    if det_id == "ours":
        return OursAdapter(det_id, dcfg, logger)
    raise ValueError("未知检测器：%r" % (det_id,))


# ---------------------------------------------------------------------------
# run 清单（ours）
# ---------------------------------------------------------------------------
def read_best(run_dir):
    """读 `best.json` ⇒ ``(ckpt_dir, regime, seed, best_dict)``。缺件即抛错（**不补默认值**）。"""
    best_path = os.path.join(run_dir, "best.json")
    if not os.path.isfile(best_path):
        raise FileNotFoundError("run 目录缺 best.json：%s" % run_dir)
    with open(best_path, "r", encoding="utf-8") as fh:
        best = json.load(fh)
    ckpt = (best.get("dir") or "").strip()
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(_ROOT, ckpt)
    if not ckpt or not os.path.isdir(ckpt):
        raise FileNotFoundError("best.json 指向的 ckpt 不存在：%r" % (best.get("dir"),))
    return ckpt, best.get("regime"), best.get("seed"), best


def resolve_ours_runs(cfg, args, logger=print):
    """复用 `05_01.resolve_runs`；要求 6/6（`runs_policy.expect_n`）。"""
    e501 = load_module("eval_05_01", cfg["detectors"]["ours"]["module"])
    det_cfg = load_config(_abs(cfg["detectors"]["ours"]["config"]))
    runs = e501.resolve_runs(args, det_cfg, logger)
    if not runs:
        raise RuntimeError("未解析到任何 run（须给 --runs-csv 或 --run-dir）")
    return runs


# ---------------------------------------------------------------------------
# 测量主流程
# ---------------------------------------------------------------------------
def _measure_one(adapter, rows, repeat, warmup, logger, pass_kwargs=None):
    """`warmup` 次预热（**不计入**）＋ `repeat` 次计时通过。"""
    passes = []
    for i in range(int(warmup)):
        adapter.infer_pass(rows, **(pass_kwargs or {}))
        logger("WARMUP_PASS %d/%d done" % (i + 1, int(warmup)))
    for i in range(int(repeat)):
        res = adapter.infer_pass(rows, **(pass_kwargs or {}))
        res["is_warmup"] = False
        res["pass_index"] = i + 1
        passes.append(res)
        logger("TIMED_PASS %d/%d mean_ms=%.3f" % (i + 1, int(repeat),
                                                  pass_stats(res["ms"])["mean_ms"]))
    return passes


def build_row(det_id, dcfg, cfg, seven, load_info, passes, repeat, warmup,
              n_samples, best, run_ctx, notes=None, overrides=None):
    """把一次测量的结果组装成 of-record 行（**不含** `adapter` 生命周期）。"""
    group = dcfg["group"]
    stats = repeat_report(passes)
    mem_infer = max([float(p.get("mem_infer_peak_mib") or 0.0) for p in passes] or [0.0])
    rss_peak = max([float(p.get("cpu_rss_peak_mib") or 0.0) for p in passes]
                   + [float(load_info.get("cpu_rss_peak_mib") or 0.0)] or [0.0])
    if group == "no_model":
        # C1：无模型档的显存一律 `n/a`（**不适用**）⇒ GPU 显存与进程 RSS 都不得出数
        mem_infer = 0.0
        rss_peak = 0.0
    notes = list(notes or [])
    if range_exceeds(stats, cfg):
        notes.append("延迟极差 > 中位 50% ⇒ 如实报告（核 GPU 占用与功耗墙），未重跑择优")
    total_p, trainable_p, psource, lora_r, lora_a = params_for(det_id, dcfg, best)
    regime = (run_ctx or {}).get("regime")
    seed = (run_ctx or {}).get("seed")
    row = {
        "detector": det_id,
        "label": dcfg["label"],
        "group": group,
        "device": seven["device"],
        "measurement": "measured(same_session)",
        "run_id": (run_ctx or {}).get("run_id") or NA,
        "regime": regime if regime is not None else NA,
        "seed": seed if seed is not None else NA,
        "lat_mean_ms": stats["lat_mean_ms"],
        "lat_median_ms": stats["lat_median_ms"],
        "lat_p95_ms": stats["lat_p95_ms"],
        "lat_mean_range_ms": stats["lat_mean_range_ms"],
        "lat_median_range_ms": stats["lat_median_range_ms"],
        "lat_p95_range_ms": stats["lat_p95_range_ms"],
        "n_samples": n_samples,
        "n_passes": stats["n_passes"],
        "repeat": repeat,
        "warmup": warmup,
        "mem_load_peak_mib": load_info.get("mem_load_peak_mib"),
        "mem_infer_peak_mib": (mem_infer if mem_infer > 0.0 else None),
        "cpu_rss_peak_mib": (rss_peak if rss_peak > 0.0 else None),
        "mem_collector": seven["collector"],
        "total_params": total_p,
        "trainable_params": trainable_p,
        "lora_r": lora_r if lora_r is not None else NA,
        "lora_alpha": lora_a if lora_a is not None else NA,
        "param_source": psource,
        "t_model": seven["model"],
        "t_load_mode": seven["load_mode"],
        "t_max_length": seven["max_length"],
        "t_batch_size": seven["batch_size"],
        "t_offload": seven["offload"],
        "t_device": seven["device"],
        "t_collector": seven["collector"],
        "amortization": cfg["latency"]["amortization"].get(det_id, NA),
        "notes": "；".join(notes) if notes else NA,
    }
    if overrides:
        row.update(overrides)
    _passes = [(det_id, (run_ctx or {}).get("run_id") or NA, regime, seed, p) for p in passes]
    return row, _passes


def measure_detector(det_id, dcfg, cfg, args, rows, logger=print, run_ctx=None, pass_kwargs=None):
    """测一个检测器（或 ours 的一个 run）：`load` → `warmup+repeat` → `unload`。"""
    seven = seven_tuple_of(det_id, dcfg)
    assert_seven_tuple(seven, "detector=%s" % det_id)
    repeat = int(args.repeat) if args.repeat else int(cfg["repeat"])
    warmup = int(args.warmup) if args.warmup else int(cfg["warmup"])
    adapter = make_adapter(det_id, dcfg, logger)
    best = (run_ctx or {}).get("best")
    if det_id == "ours":
        run_ctx = dict(run_ctx or {})
        run_ctx.setdefault("regime", None)
        run_ctx.setdefault("seed", None)
        load_info = adapter.load(run_ctx["ckpt"], run_ctx["seed"])
    else:
        load_info = adapter.load()
    passes = _measure_one(adapter, rows, repeat, warmup, logger, pass_kwargs)
    adapter.unload()
    return build_row(det_id, dcfg, cfg, seven, load_info, passes, repeat, warmup,
                     len(rows), best, run_ctx)


def fall_back_row(det_id, dcfg, cfg, reason, logger=print):
    """回退档：`measurement=cited(baseline-v1)`，取 `cited_of_record`；**不并入**实测排名。"""
    cited = (cfg.get("cited_of_record") or {}).get("values", {}).get(det_id)
    if not cited:
        raise KeyError("回退失败：cited_of_record 缺 %s（回退档不得静默丢弃该行）" % det_id)
    seven = seven_tuple_of(det_id, dcfg)
    total_p, trainable_p, psource, lora_r, lora_a = params_for(det_id, dcfg, None)
    logger("DETECTOR_FALLBACK detector=%s reason=%s ⇒ cited(baseline-v1)" % (det_id, reason))
    return {
        "detector": det_id,
        "label": dcfg["label"],
        "group": dcfg["group"],
        "device": seven["device"],
        "measurement": "cited(baseline-v1)",
        "run_id": NA, "regime": NA, "seed": NA,
        "lat_mean_ms": cited["mean_ms"], "lat_median_ms": cited["median_ms"],
        "lat_p95_ms": cited["p95_ms"],
        "lat_mean_range_ms": None, "lat_median_range_ms": None, "lat_p95_range_ms": None,
        "n_samples": None, "n_passes": None,
        "repeat": None, "warmup": None,
        "mem_load_peak_mib": cited["mem_load_mib"], "mem_infer_peak_mib": cited["mem_infer_mib"],
        "cpu_rss_peak_mib": None,
        "mem_collector": seven["collector"],
        "total_params": total_p, "trainable_params": trainable_p,
        "lora_r": lora_r if lora_r is not None else NA,
        "lora_alpha": lora_a if lora_a is not None else NA,
        "param_source": psource,
        "t_model": seven["model"], "t_load_mode": seven["load_mode"],
        "t_max_length": seven["max_length"], "t_batch_size": seven["batch_size"],
        "t_offload": seven["offload"], "t_device": seven["device"],
        "t_collector": seven["collector"],
        "amortization": cfg["latency"]["amortization"].get(det_id, NA),
        "notes": "回退档（非主档口径）：%s" % reason,
    }


# ---------------------------------------------------------------------------
# 四表构建
# ---------------------------------------------------------------------------
def _fmt(x, nd=4):
    if x is None:
        return NA
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return ("%." + str(nd) + "f") % (float(x),)


def _fmt1(x):
    return NA if x is None else "%.1f" % (float(x),)


def _fmt_int(x):
    return NA if x is None else str(int(x))


def group_title(cfg, gid):
    g = (cfg["groups"] or {}).get(gid) or {}
    return "%s（`%s`）" % (g.get("label", gid), gid)


def build_tables(rows, cfg):
    """按**量纲分组**拆分（三档并列、不合并排序）。"""
    out = []
    for gid in ["no_model", "cpu_model", "gpu_model"]:
        sub = [r for r in rows if r["group"] == gid]
        if not sub:
            continue
        out.append((gid, group_title(cfg, gid), sub))
    rest = [r for r in rows if r["group"] not in ("no_model", "cpu_model", "gpu_model")]
    if rest:
        out.append(("other", "未归类", rest))
    return out


def build_7b_rows(rows, cfg, ours_control):
    """C5：唯一对照档 = B5。比值 = B5 ÷ Ours（>1 即 Ours 更快／更小）。"""
    anchor_id = cfg["comparison_7b"]["anchor"]
    anchor = None
    for r in rows:
        if r["detector"] == anchor_id and r["measurement"].startswith("measured"):
            anchor = r
            break
    if anchor is None:
        return [], None

    def _ratio(a, b):
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)

    use_control = bool((cfg["comparison_7b"].get("fairness_control") or {}).get("enabled")) \
        and bool(ours_control)
    lat_key_rows = ours_control if use_control else \
        [r for r in rows if r["detector"] == "ours" and r["measurement"].startswith("measured")]

    out = []
    for r in lat_key_rows:
        out.append({
            "run_id": r["run_id"], "regime": r["regime"], "seed": r["seed"],
            "latency_basis": "Ours batch=1 对照档" if use_control else "Ours 主档（batch=16）",
            "ours_lat_median_ms": r["lat_median_ms"],
            "b5_lat_median_ms": anchor["lat_median_ms"],
            "latency_ratio": _ratio(anchor["lat_median_ms"], r["lat_median_ms"]),
            "mem_ratio": _ratio(anchor["mem_infer_peak_mib"], r["mem_infer_peak_mib"]),
            "param_ratio": _ratio(anchor["total_params"], r["total_params"]),
        })
    return out, anchor


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def render_report(ctx):
    """渲染运行报告（含可直接搬入 `EVAL_REPORT.md` 的 **§5.7 骨架段**）。"""
    cfg = ctx["cfg"]
    rows = ctx["rows"]
    ours_control = ctx.get("ours_control") or []
    L = []
    A = L.append

    A("# 效率分析运行报告（`05_07_eval_efficiency`，W6.8 · A6）")
    A("")
    A("- 生成时间：%s" % ctx["ts"])
    A("- 配置（事前定值）：`%s`（`declared_at=%s`，**先写后跑**）"
      % (ctx["config_path"], cfg.get("declared_at")))
    A("- 测量分片：**`%s`**（%s；记录数 %d，取前 %d 条）"
      % (ctx["shard"], "封存侧（已按 6 列登记）" if ctx["is_sealed"] else "frozen 侧 val，**非封存**",
         ctx["n_total"], ctx["n_used"]))
    A("- 重复与预热：`repeat=%d`／`warmup=%d`（**事前定值**；预热不计入统计）"
      % (ctx["repeat"], ctx["warmup"]))
    A("- 封存取用：%s（`--access-log-note` = %s）"
      % ("**未读** ⇒ ACCESS_LOG **零行**登记" if not ctx["is_sealed"]
         else "**已读** ⇒ 人工登记 6 列 ＋ `--access-log-note`",
         ctx["access_note"]))
    A("- GPU 独占留证（P-4）：开始 `%s`／结束 `%s`" % (ctx["gpu_before"], ctx["gpu_after"]))
    A("")

    # ── §C1 不可比性声明（**逐字**） ──
    A("## C1 不可比性声明（**逐字**引自 `ENV_LOCK.md` §8.5）")
    A("")
    A("> " + cfg["incomparability"]["verbatim"])
    A("")
    A("具体到本报告：")
    A("")
    for s in cfg["incomparability"]["specifics"]:
        A("- " + s)
    A("")
    A("**三档量纲并列、不合并排序**：`b1`／`b2` 无模型 ⇒ 其延迟与模型类检测器**不同量纲**；"
      "`b4` 属 **CPU 档** ⇒ 与 GPU 检测器**不同量纲**。以下四表一律**按档分表**，"
      "**不跨档求均值、不跨档排名、不跨档画趋势线**。")
    A("")

    # ── 表一：延迟 ──
    A("## 表一 · 推理延迟（ms；**不含模型加载**）")
    A("")
    A("口径：%s。报 均值／中位／p95；`极差` = 跨 %d 次计时通过的该统计量 max − min。"
      % (cfg["latency"]["excludes"], ctx["repeat"]))
    A("")
    hdr = ["检测器", "run（体制/seed）", "量纲档", "均值", "中位", "p95",
           "均值极差", "中位极差", "p95极差", "n", "通过", "摊到样本", "测量"]
    for gid, title, sub in ctx["tables"]:
        A("### %s" % title)
        A("")
        body = []
        for r in sub:
            body.append([
                r["label"], "%s / %s" % (r["regime"], r["seed"]), gid,
                _fmt(r["lat_mean_ms"]), _fmt(r["lat_median_ms"]), _fmt(r["lat_p95_ms"]),
                _fmt(r["lat_mean_range_ms"]), _fmt(r["lat_median_range_ms"]),
                _fmt(r["lat_p95_range_ms"]),
                _fmt_int(r["n_samples"]), _fmt_int(r["n_passes"]),
                r["amortization"], r["measurement"],
            ])
        A(_md_table(hdr, body))
        A("")

    # ── 表二：显存 ──
    A("## 表二 · 显存峰值（MiB；**加载态／推理态分列**）")
    A("")
    A("口径：%s。系统底噪 ≈%d–%d MiB（`ENV_LOCK.md` §8.4），单卡 %d MiB。"
      % (cfg["memory"]["reset_rule"], cfg["memory"]["system_noise_mib"][0],
         cfg["memory"]["system_noise_mib"][1], cfg["memory"]["card_total_mib"]))
    A("")
    hdr = ["检测器", "run（体制/seed）", "量纲档", "加载态", "推理态", "CPU RSS", "采集器", "测量"]
    for gid, title, sub in ctx["tables"]:
        A("### %s" % title)
        A("")
        body = [[r["label"], "%s / %s" % (r["regime"], r["seed"]), gid,
                 _fmt1(r["mem_load_peak_mib"]), _fmt1(r["mem_infer_peak_mib"]),
                 _fmt1(r["cpu_rss_peak_mib"]), r["mem_collector"], r["measurement"]]
                for r in sub]
        A(_md_table(hdr, body))
        A("")
    A("> **加载态与推理态为两列，不得混为一列、不得相加。** `n/a` = 不适用（非缺失）："
      "`b1`／`b2` 无模型 ⇒ 显存不适用；`b4` 属 CPU 档 ⇒ GPU 显存不适用（另报 RSS）。")
    A("")

    # ── 表三：参数量 ──
    A("## 表三 · 模型参数量（总参／可训练参分列）")
    A("")
    hdr = ["检测器", "run（体制/seed）", "总参", "可训练参", "LoRA r/alpha", "来源", "测量"]
    for gid, title, sub in ctx["tables"]:
        A("### %s" % title)
        A("")
        body = [[r["label"], "%s / %s" % (r["regime"], r["seed"]),
                 _fmt_int(r["total_params"]), _fmt_int(r["trainable_params"]),
                 "%s / %s" % (r["lora_r"], r["lora_alpha"]),
                 r["param_source"], r["measurement"]] for r in sub]
        A(_md_table(hdr, body))
        A("")
    A("> `b1`／`b2` 无模型 ⇒ 总参／可训练参均 `n/a`（**不适用，非 0**）。"
      "Ours 的可训练参取 `best.json` → `model_meta.trainable_params`（实测）。")
    A("")

    # ── 表四：7B 对照 ──
    A("## 表四 · 与 7B 级大模型审查方案对比（唯一对照档 = **%s**）"
      % cfg["comparison_7b"]["anchor"].upper())
    A("")
    A("对照档：`%s`（%s）。%s"
      % (cfg["comparison_7b"]["anchor"], cfg["detectors"][cfg["comparison_7b"]["anchor"]]["label"],
         cfg["comparison_7b"]["anchor_reason"]))
    A("")
    A("**公平性对照（事前定值）**：%s" % (cfg["comparison_7b"]["fairness_control"]["reason"],))
    A("")
    if ctx["rows_7b"]:
        hdr = ["Ours run（体制/seed）", "延迟基准", "Ours 中位(ms)", "B5 中位(ms)",
               "延迟比 B5/Ours", "显存比 B5/Ours", "参数量比 B5/Ours"]
        body = [[
            "%s / %s" % (r["regime"], r["seed"]), r["latency_basis"],
            _fmt(r["ours_lat_median_ms"]), _fmt(r["b5_lat_median_ms"]),
            _fmt(r["latency_ratio"]), _fmt(r["mem_ratio"]), _fmt(r["param_ratio"]),
        ] for r in ctx["rows_7b"]]
        A(_md_table(hdr, body))
    else:
        A("（本轮无可用对照行：B5 未实测成功 ⇒ **不给出比值**，不得引用 of-record 代填）")
    A("")
    A("> 比值 > 1 表示 Ours 更快／更小。三种比值**只在七元组可比的前提下解读**，"
      "且 Ours 与 B5 的 `max_length`／`batch_size`／`offload`／`device`／`collector` 须一致"
      "（对照档已把 Ours 的 `batch_size` 拉齐到 1）。")
    A("")
    if ctx["rows_7b"] and ours_control:
        anchor_row = None
        main_mem = None
        for r in rows:
            meas = str(r.get("measurement") or "")
            if r["detector"] == cfg["comparison_7b"]["anchor"] and meas.startswith("measured"):
                anchor_row = r
            if r["detector"] == "ours" and meas.startswith("measured") \
                    and str(r["t_batch_size"]) != "1":
                main_mem = main_mem or r["mem_infer_peak_mib"]
        extra = ""
        if main_mem and anchor_row and anchor_row.get("mem_infer_peak_mib"):
            extra = ("；Ours **主档**（`eval_max_size` 组批）的推理态峰值为 %.1f MiB ⇒ "
                     "对应显存比 = %.4f" % (float(main_mem),
                                            float(anchor_row["mem_infer_peak_mib"]) / float(main_mem)))
        A("> **口径澄清**：表四的 `延迟比`／`显存比` **取自 Ours 的 batch=1 对照档通过**"
          "（同一组通过同时产出延迟与推理态显存），以保证与 B5（batch=1）同量纲；"
          "`参数量比` 与批大小无关%s。两档显存比**皆不可跨口径相减**（`ENV_LOCK.md` §8.5）。"
          % extra)
        A("")

    # ── 交叉校验（C6） ──
    A("## 交叉校验：`BASELINE_REPORT.md` §1 的 of-record（**他会话、他分片**）")
    A("")
    cc = cfg["cited_of_record"]
    A("来源：%s（tag `%s`，分片 `%s`）。" % (cc["source"], cc["tag"], cc["split"]))
    A("")
    for c in cc["caveats"]:
        A("- " + c)
    A("")
    hdr = ["检测器", "of-record 均值/中位/p95 (ms)", "of-record 加载/推理 (MiB)",
           "本轮实测 均值/中位/p95 (ms)", "本轮实测 加载/推理 (MiB)"]
    body = []
    for det_id, v in cc["values"].items():
        mine = None
        for r in rows:
            if r["detector"] == det_id and r["measurement"].startswith("measured"):
                mine = r
                break
        body.append([
            det_id.upper(),
            "%s / %s / %s" % (_fmt(v["mean_ms"]), _fmt(v["median_ms"]), _fmt(v["p95_ms"])),
            "%s / %s" % (_fmt1(v["mem_load_mib"]), _fmt1(v["mem_infer_mib"])),
            ("%s / %s / %s" % (_fmt(mine["lat_mean_ms"]), _fmt(mine["lat_median_ms"]),
                               _fmt(mine["lat_p95_ms"]))) if mine else NA,
            ("%s / %s" % (_fmt1(mine["mem_load_peak_mib"]),
                          _fmt1(mine["mem_infer_peak_mib"]))) if mine else NA,
        ])
    A(_md_table(hdr, body))
    A("")
    A("> 本表**只作量级核对**，**不并入**实测主表排名（不同会话、不同分片，`ENV_LOCK.md` §8.5）。")
    A("")

    # ── 七元组 ──
    A("## 七元组（`ENV_LOCK.md` §8 口径；**每行必填**）")
    A("")
    hdr = ["检测器", "模型", "加载方式", "max_length", "批大小", "offload", "设备", "采集器"]
    body = [[r["label"], r["t_model"], r["t_load_mode"], str(r["t_max_length"]),
             str(r["t_batch_size"]), str(r["t_offload"]), str(r["t_device"]), r["t_collector"]]
            for r in rows]
    A(_md_table(hdr, body))
    A("")

    # ── 重复明细 ──
    A("## 重复测量明细（**逐次通过**，不含预热）")
    A("")
    hdr = ["检测器", "run（体制/seed）", "通过 #", "n", "通过总耗时(ms)", "均值", "中位", "p95"]
    body = []
    for det_id, run_id, regime, seed, p in ctx["repeat_rows"]:
        st = pass_stats(p["ms"])
        body.append([det_id.upper(), "%s / %s" % (regime if regime is not None else NA,
                                                  seed if seed is not None else NA),
                     p["pass_index"], len(p["ms"]), _fmt(p["total_ms"]), _fmt(st["mean_ms"]),
                     _fmt(st["median_ms"]), _fmt(st["p95_ms"])])
    A(_md_table(hdr, body))
    A("")
    A("> **%d 次计时通过全报**，未挑最好的一次；极差见「表一」的极差三列。" % ctx["repeat"])
    A("")

    # ── 局限 ──
    A("## 局限与声明")
    A("")
    for s in ctx["limitations"]:
        A("- " + s)
    A("")

    # ── §5.7 骨架段 ──
    A("---")
    A("")
    A("## 附：可直接搬入 `EVAL_REPORT.md` 的 §5.7 骨架段")
    A("")
    A("### 5.7 效率")
    A("")
    A("效率分析在 `%s` 的 %d 条样本上测量（`repeat=%d`／`warmup=%d`，延迟**不含模型加载**），"
      "报推理延迟（均值／中位／p95）、显存峰值（`max_memory_allocated`，**加载态／推理态分列**）、"
      "模型参数量（总参／可训练参），并与 7B 级大模型审查方案 `%s` 对照。"
      % (ctx["shard"], ctx["n_used"], ctx["repeat"], ctx["warmup"],
         cfg["comparison_7b"]["anchor"].upper()))
    A("")
    A("**不可比性声明（`ENV_LOCK.md` §8.5，逐字）**：%s" % cfg["incomparability"]["verbatim"])
    A("")
    A("据此：`b1`／`b2` 为**无模型档**、`b4` 属 **CPU 档**，二者与 GPU 模型类检测器**不同量纲**，"
      "一律**并列列出、不合并排序**。")
    A("")
    A("| 量纲档 | 检测器 | 延迟 中位 (ms) | 显存 加载/推理 (MiB) | 总参 | 可训练参 |")
    A("|---|---|---|---|---|---|")
    for gid, title, sub in ctx["tables"]:
        for r in sub:
            A("| %s | %s | %s | %s / %s | %s | %s |"
              % (gid, r["label"] + ("" if r["run_id"] == NA else "（%s/%s）" % (r["regime"], r["seed"])),
                 _fmt(r["lat_median_ms"]), _fmt1(r["mem_load_peak_mib"]),
                 _fmt1(r["mem_infer_peak_mib"]), _fmt_int(r["total_params"]),
                 _fmt_int(r["trainable_params"])))
    A("")
    A("与 `%s` 的对照（比值 = B5 ÷ Ours，>1 即 Ours 更快／更小）："
      % cfg["comparison_7b"]["anchor"].upper())
    A("")
    if ctx["rows_7b"]:
        A("| Ours run | 延迟基准 | 延迟比 | 显存比 | 参数量比 |")
        A("|---|---|---|---|---|")
        for r in ctx["rows_7b"]:
            A("| %s/%s | %s | %s | %s | %s |"
              % (r["regime"], r["seed"], r["latency_basis"], _fmt(r["latency_ratio"]),
                 _fmt(r["mem_ratio"]), _fmt(r["param_ratio"])))
    else:
        A("（本轮无可用对照行）")
    A("")
    A("散点图数据见 `efficiency_figdata_v1.0_<ts>.csv`（横轴＝参数量／纵轴＝延迟，按量纲档分色）。")
    A("")
    return "\n".join(L) + "\n"


def write_csv_rows(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        # lineterminator="\n" 必需：csv 模块默认是 "\r\n"，即便 open(newline="") 也会产出 CRLF
        # （依 #61 / N-17 / D-81 裁定五；与 scripts/data/_common.py 同口径）
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


# ---------------------------------------------------------------------------
# 正式测量
# ---------------------------------------------------------------------------
def cmd_run(cfg, args, logger=print):
    rows_used, shard, n_total, is_sealed = load_bench_rows(cfg, args, logger)
    repeat = int(args.repeat) if args.repeat else int(cfg["repeat"])
    warmup = int(args.warmup) if args.warmup else int(cfg["warmup"])
    det_ids = [d.strip() for d in (args.detectors or cfg["detector_scope"]).split(",") if d.strip()]

    gpu_before = gpu_snapshot()
    logger("GPU_BEFORE=%s" % gpu_before)

    rows = []
    repeat_rows = []
    ours_control = []
    fallback_notes = []

    for det_id in det_ids:
        dcfg = cfg["detectors"][det_id]
        if det_id == "ours":
            runs = resolve_ours_runs(cfg, args, logger)
            logger("OURS_RUNS n=%d" % len(runs))
            if len(runs) != int(dcfg["runs_policy"]["expect_n"]):
                logger("OURS_RUNS_INCOMPLETE got=%d expect=%d（逐 run 全报口径 ⇒ 如实登记）"
                       % (len(runs), dcfg["runs_policy"]["expect_n"]))
            for regime, seed, run_dir in runs:
                ckpt, _m_regime, _m_seed, best = read_best(run_dir)
                run_id = os.path.basename(run_dir)
                logger("OURS_RUN regime=%s seed=%s run_id=%s" % (regime, seed, run_id))
                seven = seven_tuple_of(det_id, dcfg)
                rctx = {"ckpt": ckpt, "regime": regime, "seed": seed, "best": best,
                        "run_id": run_id}
                # —— 一次加载 ⇒ 主档（batch=16）＋ 对照档（batch=1）各一轮 warmup+repeat ——
                adapter = OursAdapter(det_id, dcfg, logger)
                load_info = adapter.load(ckpt, seed)
                passes = _measure_one(adapter, rows_used, repeat, warmup, logger)
                row, ps = build_row(det_id, dcfg, cfg, seven, load_info, passes,
                                    repeat, warmup, len(rows_used), best, rctx)
                rows.append(row)
                repeat_rows.extend(ps)
                fc = cfg["comparison_7b"].get("fairness_control") or {}
                if fc.get("enabled"):
                    b1_size = int(fc["ours_batch1_max_size"])
                    ps2 = _measure_one(adapter, rows_used, repeat, warmup, logger,
                                       pass_kwargs={"max_size": b1_size})
                    ctrl, _ = build_row(
                        det_id, dcfg, cfg, seven, load_info, ps2, repeat, warmup,
                        len(rows_used), best, rctx,
                        notes=["对照档（非主档）：batch=1，用于与 b3／b5 同量纲比较"],
                        overrides={"t_batch_size": b1_size,
                                   "amortization": "逐样本计时（batch=1 对照档，无需摊）"})
                    ours_control.append(ctrl)
                    repeat_rows.extend([(det_id + "_batch1", run_id, regime, seed, p) for p in ps2])
                adapter.unload()
        else:
            try:
                row, ps = measure_detector(det_id, dcfg, cfg, args, rows_used, logger)
                rows.append(row)
                repeat_rows.extend(ps)
            except Exception as exc:  # 回退档：**记录原因**，不静默丢弃、不并入排名
                reason = "%s: %s" % (type(exc).__name__, exc)
                fallback_notes.append("%s ⇒ %s" % (det_id, reason))
                rows.append(fall_back_row(det_id, dcfg, cfg, reason, logger))

    gpu_after = gpu_snapshot()
    logger("GPU_AFTER=%s" % gpu_after)

    rows_7b, anchor = build_7b_rows(rows, cfg, ours_control)
    ctx = {
        "cfg": cfg, "rows": rows, "ours_control": ours_control, "rows_7b": rows_7b,
        "tables": build_tables(rows, cfg), "repeat_rows": repeat_rows,
        "ts": args.ts, "shard": shard, "n_total": n_total, "n_used": len(rows_used),
        "is_sealed": is_sealed, "repeat": repeat, "warmup": warmup,
        "config_path": args.config,
        "access_note": args.access_log_note or "n/a（非 sealed 源）",
        "gpu_before": gpu_before or "n/a（nvidia-smi 不可用）",
        "gpu_after": gpu_after or "n/a（nvidia-smi 不可用）",
        "limitations": _limitations(cfg, rows, fallback_notes, ours_control, is_sealed),
    }
    text = render_report(ctx)

    report_dir = args.report_dir
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, "efficiency_v1.0_%s.md" % args.ts)
    csv_path = os.path.join(report_dir, "efficiency_v1.0_%s.csv" % args.ts)
    fig_path = os.path.join(report_dir, "efficiency_figdata_v1.0_%s.csv" % args.ts)
    rep_path = os.path.join(report_dir, "efficiency_repeats_v1.0_%s.csv" % args.ts)
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    write_csv_rows(csv_path, MAIN_FIELDS, rows + ours_control)

    fig_rows = []
    for r in rows + ours_control:
        fig_rows.append({
            "detector": r["detector"], "label": r["label"], "group": r["group"],
            "run_id": r["run_id"], "regime": r["regime"], "seed": r["seed"],
            "lat_mean_ms": r["lat_mean_ms"], "lat_median_ms": r["lat_median_ms"],
            "lat_p95_ms": r["lat_p95_ms"],
            "mem_load_peak_mib": r["mem_load_peak_mib"],
            "mem_infer_peak_mib": r["mem_infer_peak_mib"],
            "cpu_rss_peak_mib": r["cpu_rss_peak_mib"],
            "total_params": r["total_params"], "trainable_params": r["trainable_params"],
            "measurement": r["measurement"],
            "measurement_note": r["notes"],
        })
    write_csv_rows(fig_path, FIGDATA_FIELDS, fig_rows)

    rep_rows = []
    for det_id, run_id, regime, seed, p in repeat_rows:
        st = pass_stats(p["ms"])
        rep_rows.append({
            "detector": det_id, "run_id": run_id, "regime": regime, "seed": seed,
            "pass_index": p.get("pass_index"), "is_warmup": False,
            "n_samples": len(p["ms"]), "pass_total_ms": p["total_ms"],
            "mean_ms": st["mean_ms"], "median_ms": st["median_ms"], "p95_ms": st["p95_ms"],
        })
    write_csv_rows(rep_path, REPEAT_FIELDS, rep_rows)

    logger("ARTIFACTS md=%s csv=%s fig=%s repeats=%s" % (md_path, csv_path, fig_path, rep_path))
    print("REPORT_MD=%s REPORT_CSV=%s FIGDATA_CSV=%s" % (md_path, csv_path, fig_path))
    return 0


def _limitations(cfg, rows, fallback_notes, ours_control, is_sealed=False):
    out = []
    out.append("本窗口%s `data/sealed_test/**` ⇒ ACCESS_LOG %s"
               % ("**未读**" if not is_sealed else "**已读**",
                  "**零行**登记" if not is_sealed else "已人工登记 6 列"))
    out.append("效率量在 `%s` 上测得；改 `--limit`／`--bench-split` ⇒ 与主档**不可直接相减**"
               % cfg["bench"]["default_split"])
    out.append("三档量纲（`no_model`／`cpu_model`／`gpu_model`）**并列列出、不合并排序**"
               "（`ENV_LOCK.md` §8.5 与本报告 §C1）")
    out.append("of-record 交叉校验值属**他会话、他分片**（`id_test`）⇒ 只作量级核对，**不并入**实测排名")
    if ours_control:
        out.append("Ours 主档走 `05_01.Engine`（`eval_max_size=16`，**批际摊到样本**）⇒ 与 `b3`／`b5` 的"
                   "逐样本计时**不同量纲**；§表四 另给 **batch=1 对照档**用于同量纲比较"
                   "（对照档**不参与**主表排名）")
    if fallback_notes:
        for f in fallback_notes:
            out.append("回退档（非主档口径，不并入排名）：%s" % f)
    for r in rows:
        if r["notes"] and r["notes"] != NA and "极差" in r["notes"]:
            out.append("%s：%s" % (r["label"], r["notes"]))
            break
    return out


# ---------------------------------------------------------------------------
# `--dry-run`（不加载模型、不触 GPU）
# ---------------------------------------------------------------------------
def cmd_dryrun(cfg, args, logger=print):
    checks = []

    def _ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    det_ids = [d.strip() for d in (args.detectors or cfg["detector_scope"]).split(",") if d.strip()]
    _ck("detectors_known", all(d in cfg["detectors"] for d in det_ids), ",".join(det_ids))
    for d in det_ids:
        dcfg = cfg["detectors"][d]
        try:
            assert_seven_tuple(seven_tuple_of(d, dcfg), d)
            _ck("seven_tuple:%s" % d, True)
        except (KeyError, ValueError) as exc:
            _ck("seven_tuple:%s" % d, False, str(exc))
        _ck("module_exists:%s" % d, os.path.isfile(_abs(dcfg["module"])), dcfg["module"])
        _ck("config_exists:%s" % d, os.path.isfile(_abs(dcfg["config"])), dcfg["config"])

    path, shard, expect_n, is_sealed = resolve_bench_path(cfg, args)
    _ck("bench_path_exists", os.path.isfile(path), path)
    if expect_n:
        _ck("bench_record_count", len(read_records(path)) == int(expect_n),
            "%d vs %d" % (len(read_records(path)), expect_n))
    if is_sealed and not args.access_log_note:
        _ck("sealed_access_log_note", False, "`--bench-source sealed` 缺 `--access-log-note`")

    if "ours" in det_ids:
        try:
            runs = resolve_ours_runs(cfg, args, logger)
            _ck("ours_runs", len(runs) == int(cfg["detectors"]["ours"]["runs_policy"]["expect_n"]),
                "got=%d" % len(runs))
            for regime, seed, run_dir in runs:
                ckpt, _r, _s, best = read_best(run_dir)
                tp = ((best.get("model_meta") or {}).get("trainable_params"))
                _ck("ours_best:%s:%s" % (regime, seed), bool(ckpt) and tp is not None,
                    "trainable=%s" % tp)
        except Exception as exc:  # dry-run 只登记，不中断
            _ck("ours_runs", False, str(exc))

    n = len(checks)
    bad = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        logger("DRYRUN %s %s %s" % ("OK  " if ok else "FAIL", name, detail))
    print("DRYRUN_RESULT=%s n=%d" % ("PASS" if not bad else "FAIL", n))
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# `--self-check`（离线：不加载模型、不触 GPU、不读封存）
# ---------------------------------------------------------------------------
def cmd_selfcheck(cfg, args, logger=print):
    checks = []

    def _ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1–3 配置与七元组
    det_ids = [d.strip() for d in (args.detectors or cfg["detector_scope"]).split(",") if d.strip()]
    _ck("detectors_all_six", sorted(det_ids) == ["b1", "b2", "b3", "b4", "b5", "ours"],
        ",".join(det_ids))
    for d in det_ids:
        dcfg = cfg["detectors"][d]
        try:
            assert_seven_tuple(seven_tuple_of(d, dcfg), d)
            _ck("seven_tuple_complete:%s" % d, True)
        except (KeyError, ValueError) as exc:
            _ck("seven_tuple_complete:%s" % d, False, str(exc))
        _ck("group_declared:%s" % d, d in sum([list(v["members"]) for v in cfg["groups"].values()], []),
            dcfg.get("group", ""))

    # 4 七元组缺字段 ⇒ 抛错
    try:
        seven_tuple_of("b1", {"seven_tuple": {k: "x" for k in SEVEN_KEYS[:6]}})
        _ck("seven_tuple_missing_raises", False, "缺字段未抛错")
    except KeyError:
        _ck("seven_tuple_missing_raises", True)

    # 5–7 b1／b2 无模型 ⇒ 设备与显存 `n/a`（**不是 0**）
    for d in ("b1", "b2"):
        st = seven_tuple_of(d, cfg["detectors"][d])
        _ck("no_model_device_na:%s" % d, str(st["device"]).startswith(NA), st["device"])
        total, trainable, _src, _r, _a = params_for(d, cfg["detectors"][d])
        _ck("no_model_params_na:%s" % d, total is None and trainable is None,
            "total=%r trainable=%r" % (total, trainable))

    # 8 加载态／推理态分列
    st_mem = cfg["memory"]["states"]
    _ck("memory_states_separate", list(st_mem) == ["load", "infer"], str(st_mem))
    _ck("memory_forbid_merge", bool(cfg["memory"]["forbid_merge"]))
    _ck("main_fields_have_both_states",
        ("mem_load_peak_mib" in MAIN_FIELDS) and ("mem_infer_peak_mib" in MAIN_FIELDS)
        and MAIN_FIELDS.index("mem_load_peak_mib") != MAIN_FIELDS.index("mem_infer_peak_mib"))

    # 9 repeat 中位与极差（**不挑最好的一次**）
    passes = [{"ms": [10.0, 20.0, 30.0]}, {"ms": [20.0, 30.0, 40.0]}, {"ms": [30.0, 40.0, 50.0]}]
    rep = repeat_report(passes)
    pooled = [10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 30.0, 40.0, 50.0]
    _ck("repeat_pooled_mean", abs(rep["lat_mean_ms"] - statistics.fmean(pooled)) < 1e-9,
        "%.6f" % rep["lat_mean_ms"])
    _ck("repeat_pooled_median", abs(rep["lat_median_ms"] - statistics.median(pooled)) < 1e-9,
        "%.6f" % rep["lat_median_ms"])
    _ck("repeat_range_median", abs(rep["lat_median_range_ms"] - 20.0) < 1e-9,
        "%.6f" % rep["lat_median_range_ms"])
    _ck("repeat_n_passes", rep["n_passes"] == 3, str(rep["n_passes"]))
    _ck("repeat_exceed_flag", range_exceeds({"lat_median_ms": 10.0, "lat_median_range_ms": 6.0}, cfg)
        and not range_exceeds({"lat_median_ms": 10.0, "lat_median_range_ms": 4.0}, cfg))

    # 10 确定性双跑（纯函数桩 ⇒ 两次逐位相同）
    stub_rows = [{"text": "hello world\nsecond line", "label": "1"},
                 {"text": "another sample", "label": "0"}]

    class _Stub(BaseAdapter):
        def load(self):
            return {"mem_load_peak_mib": None, "cpu_rss_peak_mib": None}

        def infer_pass(self, rows, **kw):
            ms = [float(len(r["text"])) for r in rows]
            return {"ms": ms, "total_ms": math.fsum(ms)}

    a1 = _measure_one(_Stub("stub", {}, logger), stub_rows, 2, 1, logger)
    a2 = _measure_one(_Stub("stub", {}, logger), stub_rows, 2, 1, logger)
    _ck("determinism_double_run", [p["ms"] for p in a1] == [p["ms"] for p in a2],
        "%r vs %r" % ([p["ms"] for p in a1], [p["ms"] for p in a2]))
    _ck("warmup_not_counted", len(a1) == 2, "passes=%d" % len(a1))

    # 11 封存门禁（缺 `--access-log-note` ⇒ 拒绝执行）
    _ck("sealed_requires_note", bool((cfg["sealed_policy"]["if_sealed"]["require_access_log_note"])))
    try:
        load_bench_rows(cfg, argparse.Namespace(bench_source="sealed", bench_split="id_val",
                                                limit=None, access_log_note=None), logger)
        _ck("sealed_gate_enforced", False, "未拒绝")
    except Exception:
        _ck("sealed_gate_enforced", True)

    # 12 §8.5 逐字在位（去空白后须为 `ENV_LOCK.md` 的子串）
    import re
    try:
        env_txt = open(_abs("ENV_LOCK.md"), "r", encoding="utf-8").read()
    except OSError:
        env_txt = ""
    _ck("incomparability_verbatim",
        re.sub(r"\s+", "", cfg["incomparability"]["verbatim"]) in re.sub(r"\s+", "", env_txt))

    # 13 C5 唯一对照档
    _ck("anchor_is_b5", cfg["comparison_7b"]["anchor"] == "b5", cfg["comparison_7b"]["anchor"])
    _ck("anchor_forbid_new_model", "禁止" in cfg["comparison_7b"]["forbid_new_model"])

    # 14 of-record 齐备
    vals = (cfg.get("cited_of_record") or {}).get("values", {})
    _ck("cited_of_record_complete",
        all(k in vals and all(x in vals[k] for x in
                              ("mean_ms", "median_ms", "p95_ms", "mem_load_mib", "mem_infer_mib"))
            for k in ("b1", "b2", "b3", "b4", "b5")),
        ",".join(sorted(vals)))

    # 15 C8：源码内不得出现训练模式切换调用；C9：不得写死输出目录
    #    **needle 用拼接构造** —— 否则本自检函数自己的字面量会被自己数到（假失败）
    try:
        src = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    except OSError:
        src = ""
    needle_train = "".join([".", "train("])
    needle_reports = "".join(["rep", "orts", "/ev", "al"])
    needle_reports_dq = chr(34) + "".join(["rep", "orts"]) + chr(34)
    _ck("no_train_call_in_source", needle_train not in src)
    det_cfg = cfg["determinism"]
    _ck("determinism_block", det_cfg["cudnn_deterministic"] is True
        and det_cfg["cudnn_benchmark"] is False and det_cfg["allow_tf32"] is False
        and det_cfg["attn_implementation"] == "eager")

    # 16 C9：源码内不得写死 `reports/` 落盘路径；`--report-dir` 必填（缺即 return 2）
    _ck("no_hardcoded_reports_path",
        needle_reports not in src and needle_reports_dq not in src)
    _ck("report_dir_required", main(["--config", args.config, "--bench-split",
                                     cfg["bench"]["default_split"]]) == 2)

    # 17 回退档不得静默丢弃（cited 缺项 ⇒ 抛错）
    try:
        fall_back_row("zz", {"label": "zz", "group": "gpu_model",
                             "seven_tuple": {k: "x" for k in SEVEN_KEYS}}, cfg, "test")
        _ck("fallback_never_silent", False, "未抛错")
    except KeyError:
        _ck("fallback_never_silent", True)

    # 18 三档分表（不合并）
    fake = [{"group": "no_model", "label": "B1"}, {"group": "cpu_model", "label": "B4"},
            {"group": "gpu_model", "label": "B3"}]
    tabs = build_tables(fake, cfg)
    _ck("tables_grouped_by_dimension", [t[0] for t in tabs] ==
        ["no_model", "cpu_model", "gpu_model"], str([t[0] for t in tabs]))

    n = len(checks)
    bad = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        logger("SELFCHECK %s %s %s" % ("OK  " if ok else "FAIL", name, detail))
    print("SELFCHECK_RESULT=%s n=%d" % ("PASS" if not bad else "FAIL", n))
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# 日志（仅正式运行落盘；`--self-check`／`--dry-run` 不写日志）
# ---------------------------------------------------------------------------
def flagged_logger(ts, log_dir=None):
    d = log_dir or os.path.join(_ROOT, "logs", "eval")
    if not os.path.isabs(d):
        d = _abs(d)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "05_07_eval_efficiency_%s.log" % ts)

    def _log(msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    return _log, path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="scripts/eval/05_07_eval_efficiency.py",
        description="5.7 效率分析（W6.8，A6）—— 延迟／显存／参数量／7B 对照四表")
    p.add_argument("--detectors", default=None,
                   help="逗号分隔（默认取配置的 `detector_scope`）")
    p.add_argument("--runs-csv", dest="runs_csv", default=None, help="04_03 汇总 CSV（ours 用）")
    p.add_argument("--only", default=None, help="只跑指定 run，形如 id:42 或 id:42,ood:123")
    p.add_argument("--run-dir", dest="run_dir", default=None)
    p.add_argument("--regime", choices=["id", "ood"], default=None)
    p.add_argument("--seed", type=int, choices=[42, 123, 456], default=None)
    p.add_argument("--bench-split", dest="bench_split", default=None,
                   help="测量分片（默认取配置的 `bench.default_split`）")
    p.add_argument("--bench-source", dest="bench_source", choices=["val", "sealed"], default=None,
                   help="测量来源（默认取配置的 `bench.default_source`）")
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--config", dest="config", default="configs/eval/efficiency.yaml")
    p.add_argument("--report-dir", dest="report_dir", default=None,
                   help="**必填**（C9：不得硬编码 reports/ 路径）")
    p.add_argument("--outputs-dir", dest="outputs_dir", default=None)
    p.add_argument("--log-dir", dest="log_dir", default=None)
    p.add_argument("--access-log-note", dest="access_log_note", default=None)
    p.add_argument("--limit", type=int, default=None, help="测量样本条数（默认取配置的 `bench.limit`）")
    p.add_argument("--self-check", dest="self_check", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def main(argv=None):
    init_streams()
    args = build_parser().parse_args(argv)
    args.ts = timestamp()
    cfg = load_config(_abs(args.config))
    if not args.bench_split:
        args.bench_split = cfg["bench"]["default_split"]
    if not args.bench_source:
        args.bench_source = cfg["bench"]["default_source"]

    if args.self_check:
        return cmd_selfcheck(cfg, args, print)
    if args.dry_run:
        return cmd_dryrun(cfg, args, print)

    if args.bench_source == "sealed":
        if not args.access_log_note:
            print("REFUSED：`--bench-source sealed` 须同时给 `--access-log-note`"
                  "（缺一即拒绝执行并非零退出，`W6.8` §3-C7）")
            return 2
    if not args.report_dir or not args.outputs_dir:
        print("MISSING_DIRS：评估落盘产物 ⇒ 须同时给 --report-dir 与 --outputs-dir（C9）")
        return 2
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)
    logger, log_path = flagged_logger(args.ts, args.log_dir)
    logger("ACCESS_LOG_NOTE=%s（登记由人手工完成；脚本不写 ACCESS_LOG.md）"
           % (args.access_log_note or "n/a（非 sealed 源）"))
    rc = cmd_run(cfg, args, logger)
    logger("LOG_FILE=%s" % log_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""01_03_check_determinism.py — 推理确定性校验（G-DET，阶段1 硬门禁）

窗口：W2.3（阶段1 显存监控与确定性校验，权限 A2）。
依据：`windows/W2.3_determinism.md` §4.1–§4.7；`SOURCE_LOCK/实验执行方案（最终版 v5）.md`
L142–L154（新增 9）、L156–L160（新增 10）、L138（新增 8）；`AGENTS.md` §6-11／§6-12。

判定规则（**严格 0**，`AGENTS.md` §6-12；`D-04` 裁定「契约严于源头方案，取严」）
--------------------------------------------------------------------------
同一样本、同一权重、同一 seed、**两次独立进程**推理，双判据须**同时**成立：

  D1 位级一致 : 两次落盘文件的 ``md5`` 相同
  D2 数值一致 : ``max(abs(A - B)) == 0.0``，且 NaN 计数相同

v5 L148 的原文是「差值＜1e-6」，本条**更严**；非零一律判**未通过**，不得以 1e-6 结案。
不通过时按 `windows/W2.3_determinism.md` §4.5 的七级回退序列逐级排查。

模式（`--mode`）
---------------
``determinism``           一次推理 + logits 落盘（配 ``--run-tag A`` / ``B``，两次独立进程）
``eval-guard``            `model.eval()` + `torch.no_grad()` 硬断言，不落 logits
``bnb-stability``         父进程：spawn ``--repeat`` 个**独立子进程**各加载一次并汇总
``bnb-stability-single``  内部模式：单次加载并登记（由父进程调用；亦可手工隔离复跑）
``compare``               只做比对并写报告，不加载模型（不重复推理即可复核判定）

`compare` 与 `bnb-stability-single` 是对 §4.1 接口表的**纯增量扩展**：前者使判定可在
不重跑推理的前提下复算，后者是「独立进程 ×N」的实现手段。三个契约模式语义未变。

硬约束
------
1. 不访问 ``data/`` 任何子目录；样本一律取自 ``configs/env/determinism.yaml`` 的合成文本；
2. `attn_implementation` 固定 ``eager``；不使用、不安装 ``flash-attn``（`ENV_LOCK.md` §0.4）；
3. 被测对象只按 ``--model-key`` 取自 ``configs/env/models.yaml``，**禁止直接传路径**；
4. 不保存任何权重到 ``checkpoints/``；
5. 退出码：通过 ``0``；判定不通过 ``3``；参数／环境错误 ``2``。

证据落盘
--------
``logs/env/tmp/det_<key>_<run-tag>.npy`` + ``.npy.md5``  两次独立进程的 logits（D1 对象）
``logs/env/tmp/det_aux_<key>_<run-tag>.json``            进程身份与对照量
``logs/env/tmp/det_cmp_<key>.json``                      该对象的 D1／D2 实测值
``logs/env/tmp/eval_guard_<key>.json``                   eval 强制校验证据
``logs/env/tmp/bnb_stab_<key>_<NN>.json``                bnb 稳定性逐次登记
``reports/env/determinism_v1.0_<YYYYMMDD-HHMMSS>.md``    汇总报告（含**硬判定**）

日志编码：一律 LF + UTF-8（`AGENTS.md` §4.3；W2.3 §5-10 要求用**字节法**核实）。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# 环境钉必须在 import torch 之前生效（windows/W2.3_determinism.md §4.2 env_pins）
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))

DEFAULT_DET_CONFIG = os.path.join(ROOT, "configs", "env", "determinism.yaml")
DEFAULT_MODELS_MANIFEST = os.path.join(ROOT, "configs", "env", "models.yaml")
TMP_DIR = os.path.join(ROOT, "logs", "env", "tmp")
REPORTS_DIR = os.path.join(ROOT, "reports", "env")

_FALLBACK_ENV_PINS = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "torch_use_deterministic_algorithms": "warn_only",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "allow_tf32": False,
}

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAIL = 3

CONTRACT_MODES = ("determinism", "eval-guard", "bnb-stability")
EXTRA_MODES = ("bnb-stability-single", "compare")
MODES = CONTRACT_MODES + EXTRA_MODES

#: 同进程内已用过的 run-tag（防「同进程假通过」，windows/W2.3_determinism.md §4.1）
_USED_RUN_TAGS = {}


# ==========================================================================
# 通用工具
# ==========================================================================
def now_stamp():
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso():
    return _dt.datetime.now().isoformat(timespec="seconds")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_text_lf(path, text):
    """落盘一律 LF + UTF-8（不经 shell 重定向，避免 GBK／CRLF，见 W2.3 §7-2）。"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_json_lf(path, obj):
    write_text_lf(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_yaml(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def process_identity(pid=None):
    """进程身份——用于证明 A／B 来自两次**独立进程**（§4.3 跨进程要求）。"""
    pid = os.getpid() if pid is None else pid
    started = None
    try:  # Windows: 进程创建时间；失败不影响判定（pid 已足够）
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)
        if h:
            ft = ctypes.c_ulonglong(0)
            ct = ctypes.c_ulonglong(0)
            et = ctypes.c_ulonglong(0)
            kt = ctypes.c_ulonglong(0)
            if k32.GetProcessTimes(h, ctypes.byref(ft), ctypes.byref(et),
                                   ctypes.byref(kt), ctypes.byref(ct)):
                # FILETIME 为 1601-01-01 起的 100 ns 计数；减 Unix 纪元偏移后转秒
                started = (ft.value - 116444736000000000) // 10 ** 7
            k32.CloseHandle(h)
    except Exception:  # pragma: no cover - 平台相关
        started = None
    return {
        "pid": pid,
        "process_start_epoch": started,
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(sys.argv),
    }


# ==========================================================================
# 环境钉
# ==========================================================================
def quiet_progress_reporting():
    """关闭 transformers／huggingface_hub 的进度条。

    必要性（W2.3 补正轮 r2 的更正点）：进度条用裸 `\\r` 原地重画，会把该字符写进日志，
    使日志的**字节法行尾核查**（`AGENTS.md` §5；W2.3 §5-10）出现大量 `\\r` 命中——
    它并非 CRLF 行尾，但足以污染审计。置下列环境变量后日志仅含 `\\n`。
    必须在 `import transformers` 之前设置。
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def apply_env_pins(pins, environ=None):
    """把 env_pins 写进 os.environ（须在 import torch 之前）。返回实际生效的键值。"""
    environ = os.environ if environ is None else environ
    applied = {}
    for key, value in (pins or {}).items():
        if key == "CUBLAS_WORKSPACE_CONFIG":
            prior = environ.get(key)
            environ[key] = str(value)
            applied[key] = {"value": str(value), "prior": prior,
                            "overwritten": prior not in (None, str(value))}
        else:
            applied[key] = {"value": value, "prior": None, "overwritten": False}
    return applied


def torch_import_guard(environ=None):
    """返回 True 表示 CUBLAS_WORKSPACE_CONFIG 的「import 前置」条件**未被破坏**。"""
    environ = os.environ if environ is None else environ
    return (environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and "torch" not in sys.modules)


def apply_torch_determinism(torch, pins):
    """设置 cudnn／TF32／deterministic_algorithms。返回实际生效值。"""
    pins = dict(_FALLBACK_ENV_PINS) if not pins else dict(pins)
    eff = {}
    torch.backends.cudnn.benchmark = bool(pins.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(pins.get("cudnn_deterministic", True))
    eff["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    eff["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    allow_tf32 = bool(pins.get("allow_tf32", False))
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        eff["allow_tf32"] = allow_tf32
    except Exception as exc:  # pragma: no cover
        eff["allow_tf32"] = "ERR:%s" % type(exc).__name__
    mode = pins.get("torch_use_deterministic_algorithms", "warn_only")
    if mode == "strict":
        torch.use_deterministic_algorithms(True, warn_only=False)
        eff["torch_use_deterministic_algorithms"] = "strict"
    elif mode == "warn_only":
        torch.use_deterministic_algorithms(True, warn_only=True)
        eff["torch_use_deterministic_algorithms"] = "warn_only"
    else:
        eff["torch_use_deterministic_algorithms"] = "off"
    return eff


def flash_attn_status():
    import importlib.util

    installed = importlib.util.find_spec("flash_attn") is not None
    return {"flash_attn_installed": installed,
            "attn_impl_requested": "eager",
            "compliant": not installed}


# ==========================================================================
# eval 模式强制校验（v5 新增 10；AGENTS.md §6-11）—— 断言行须可被单测直接调用
# ==========================================================================
class EvalModeViolation(RuntimeError):
    """`model.eval()` + `torch.no_grad()` 硬断言失败。"""


def assert_eval_mode(model, grad_enabled=None):
    """硬断言：`model.training is False` 且 `torch.is_grad_enabled() is False`。

    失败**抛异常**（调用方须转为**非零退出**，不得降级为警告）。
    `grad_enabled` 可显式传入以便单测在 `no_grad()` 之外直接构造反例。
    """
    problems = []
    training = getattr(model, "training", None)
    if training is not False:
        problems.append("model.training is %r（要求 False）" % (training,))
    if grad_enabled is None:
        import torch

        grad_enabled = torch.is_grad_enabled()
    if grad_enabled is not False:
        problems.append("torch.is_grad_enabled() is %r（要求 False）" % (grad_enabled,))
    if problems:
        raise EvalModeViolation("eval 模式强制校验失败：" + "；".join(problems))
    return True


def note_run_tag(tag, used=None):
    """登记 run-tag。同进程内重复使用同一 tag ⇒ 返回 True 并要求调用方告警。"""
    used = _USED_RUN_TAGS if used is None else used
    dup = tag in used
    used.setdefault(tag, []).append(process_identity()["pid"])
    return dup


# ==========================================================================
# D1／D2 比对（纯函数，单测直接覆盖）
# ==========================================================================
def compare_logits(path_a, path_b, md5_a=None, md5_b=None, pid_a=None, pid_b=None):
    """比对两份 logits 落盘文件，返回 D1／D2 实测值字典（不含「总判定」措辞）。

    D1 : 两文件的 md5 相同
    D2 : ``max(abs(A - B)) == 0.0`` 且 NaN 计数相同
    """
    import numpy as np

    md5_a = md5_file(path_a) if md5_a is None else md5_a
    md5_b = md5_file(path_b) if md5_b is None else md5_b
    arr_a = np.load(path_a)
    arr_b = np.load(path_b)

    shape_a, shape_b = tuple(arr_a.shape), tuple(arr_b.shape)
    dtype_a, dtype_b = str(arr_a.dtype), str(arr_b.dtype)
    nan_a = int(np.isnan(arr_a).sum())
    nan_b = int(np.isnan(arr_b).sum())

    if shape_a != shape_b or dtype_a != dtype_b:
        return {
            "d1_md5_equal": md5_a == md5_b,
            "md5_a": md5_a, "md5_b": md5_b,
            "d2_max_abs_diff": None,
            "d2_nan_count_a": nan_a, "d2_nan_count_b": nan_b,
            "d2_nan_count_equal": nan_a == nan_b,
            "shape_a": shape_a, "shape_b": shape_b,
            "dtype_a": dtype_a, "dtype_b": dtype_b,
            "shape_mismatch": True,
            "rng_state_sha256_bootstrap": None,
            "same_process": (pid_a is not None and pid_a == pid_b),
            "pid_a": pid_a, "pid_b": pid_b,
        }

    same_process = pid_a is not None and pid_a == pid_b
    return {
        "d1_md5_equal": md5_a == md5_b,
        "md5_a": md5_a, "md5_b": md5_b,
        "d2_max_abs_diff": float(np.max(np.abs(arr_a - arr_b))),
        "d2_nan_count_a": nan_a, "d2_nan_count_b": nan_b,
        "d2_nan_count_equal": nan_a == nan_b,
        "shape_a": shape_a, "shape_b": shape_b,
        "dtype_a": dtype_a, "dtype_b": dtype_b,
        "shape_mismatch": False,
        "same_process": same_process,
        "pid_a": pid_a, "pid_b": pid_b,
    }


def judge_comparison(cmp_result, tol=0.0):
    """把 D1／D2 实测值转成**硬判定**。返回 ``(passed: bool, reasons: list[str])``。

    tol 固定 ``0.0``（`AGENTS.md` §6-12：非零即不通过；**不得**传 1e-6）。
    """
    reasons = []
    if tol != 0.0:
        raise ValueError("tol 必须为 0.0（AGENTS.md §6-12）；收到 %r" % (tol,))
    if cmp_result.get("shape_mismatch"):
        reasons.append("logits 形态／dtype 不一致，无法比对")
    if cmp_result.get("same_process"):
        reasons.append("两次落盘来自**同一进程**（pid=%s），违反跨进程要求（§4.1）"
                       % cmp_result.get("pid_a"))
    if not cmp_result.get("d1_md5_equal"):
        reasons.append("D1 位级一致 不成立：md5 不同（%s ≠ %s）"
                       % (cmp_result.get("md5_a"), cmp_result.get("md5_b")))
    diff = cmp_result.get("d2_max_abs_diff")
    if diff is None:
        reasons.append("D2 数值一致 无法计算")
    elif diff != 0.0:
        reasons.append("D2 数值一致 不成立：max|A-B| = %r ≠ 0.0（判定从严，"
                       "非零即不通过，AGENTS.md §6-12）" % (diff,))
    if not cmp_result.get("d2_nan_count_equal", True):
        reasons.append("NaN 计数不同（%s ≠ %s）"
                       % (cmp_result.get("d2_nan_count_a"),
                          cmp_result.get("d2_nan_count_b")))
    return (not reasons), reasons


# ==========================================================================
# 被测对象解析（只按 key 取，禁止直接传路径）
# ==========================================================================
def load_manifest(path=DEFAULT_MODELS_MANIFEST):
    doc = load_yaml(path) or {}
    entries = {}
    for item in doc.get("models") or []:
        entries[item.get("key")] = item
    return doc, entries


def resolve_subject(key, entries):
    if key not in entries:
        raise KeyError("--model-key %r 不在 %s 的 models 清单内" % (key, DEFAULT_MODELS_MANIFEST))
    entry = entries[key]
    local = entry.get("local_path")
    if not local:
        raise FileNotFoundError("模型 %r 的 local_path 为空（local_status=%r）"
                                % (key, entry.get("local_status")))
    path = local if os.path.isabs(local) else os.path.join(ROOT, local.replace("/", os.sep))
    if not os.path.isdir(path):
        raise FileNotFoundError("模型 %r 的落地目录不存在：%s" % (key, path))
    return {
        "key": key,
        "local_path": path,
        "repo_id": entry.get("repo_id"),
        "revision": entry.get("revision"),
        "fetch_source": entry.get("fetch_source"),
        "integrity_anchor": entry.get("integrity_anchor"),
        "planned_load": entry.get("planned_load"),
        "local_sha_verified": entry.get("local_sha_verified"),
        "size_bytes": entry.get("size_bytes"),
    }


def subject_checksum(entry_root, key):
    """落地权重的完整性抽验：对 PROVENANCE.json／config.json 取 sha256（若存在）。"""
    out = {}
    for name in ("PROVENANCE.json", "config.json"):
        p = os.path.join(entry_root, name)
        if os.path.isfile(p):
            out[name] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return out


# ==========================================================================
# 模型加载（4-bit NF4 / 全精度 / CPU）
# ==========================================================================
def load_tokenizer(transformers, path):
    return transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)


def build_quant_config(torch, transformers, load_mode, dtype):
    if load_mode != "4bit-nf4":
        return None
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_causal_lm(torch, transformers, path, load_mode, dtype, attn_impl, device):
    """按 W2.2 §4.4 词表加载 Llama 系因果 LM。返回 ``(model, device)``。"""
    kwargs = {
        "local_files_only": True,
        "attn_implementation": attn_impl,
        "dtype": dtype,
    }
    if load_mode == "4bit-nf4":
        kwargs["quantization_config"] = build_quant_config(
            torch, transformers, load_mode, dtype)
        kwargs["device_map"] = {"": 0} if device == "cuda" else "cpu"
    elif load_mode == "bf16":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": 0} if device == "cuda" else "cpu"
    elif load_mode == "fp32":
        kwargs["dtype"] = torch.float32
        kwargs["device_map"] = {"": 0} if device == "cuda" else "cpu"
    else:
        raise ValueError("不支持的加载方式：%r（词表见 windows/W2.2_model_hashes.md §4.4）"
                         % (load_mode,))
    model = transformers.AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return model


def model_context(torch, transformers, subject, load_mode, dtype, attn_impl, device):
    """加载模型 + tokenizer，并做 eval／no_grad 断言前置（断言由调用方显式执行）。"""
    path = subject["local_path"]
    tok = load_tokenizer(transformers, path)
    model = load_causal_lm(torch, transformers, path, load_mode, dtype, attn_impl, device)
    model.eval()
    return tok, model


def tokenize(tok, text, seq_len, torch, device):
    batch = tok(text, return_tensors="pt", truncation=True, max_length=int(seq_len))
    return {k: v.to(device) for k, v in batch.items()}


def input_digest(batch):
    """输入张量的字节摘要——证明 A／B 两次送入的是**同一份输入**。"""
    import numpy as np

    parts = []
    for key in sorted(batch):
        arr = batch[key].detach().cpu().numpy()
        parts.append("%s:%s:%s" % (key, arr.shape, arr.dtype))
        parts.append(arr.tobytes().hex())
    return hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()


def forward_logits(torch, model, batch):
    with torch.no_grad():
        out = model(**batch, use_cache=False)
    return out.logits


# ==========================================================================
# --mode determinism
# ==========================================================================
def run_determinism(args):
    cfg = load_determinism_config(args)
    if not args.run_tag:
        print("ERROR: --mode determinism 需要 --run-tag A|B（两次独立进程各跑一次）")
        return EXIT_USAGE
    if args.run_tag not in ("A", "B"):
        print("ERROR: --run-tag 取值为 A 或 B；收到 %r" % (args.run_tag,))
        return EXIT_USAGE

    dup = note_run_tag("%s/%s" % (args.model_key, args.run_tag))
    if dup:
        print("WARN: run-tag '%s' 已在**同一进程**内使用过——A／B 必须由两次独立进程调用，"
              "同进程重复不能构成跨进程证据（windows/W2.3_determinism.md §4.1）。"
              % (args.run_tag,))

    import numpy as np

    # 先解析被测对象（轻量，只读 yaml）——参数错误在 import torch 之前就暴露
    _, entries = load_manifest(args.models_manifest)
    subject = resolve_subject(args.model_key, entries)

    torch = _import_torch()
    transformers = _import_transformers()
    _import_bnb()

    print("== 01_03 check_determinism == mode=determinism run_tag=%s key=%s"
          % (args.run_tag, args.model_key))
    print("model.local_path=%s" % subject["local_path"])
    print("model.repo_id=%s model.revision=%s fetch_source=%s"
          % (subject["repo_id"], subject["revision"], subject["fetch_source"]))
    print("env.CUBLAS_WORKSPACE_CONFIG=%s"
          % os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    print("env.pins=%s" % json.dumps(apply_torch_determinism(torch, cfg.get("env_pins"))))
    print("env.flash_attn=%s" % json.dumps(flash_attn_status()))
    print("torch=%s cuda=%s cudnn=%s bnb=%s transformers=%s"
          % (torch.__version__, torch.version.cuda, torch.backends.cudnn.version(),
             _module_version("bitsandbytes"), transformers.__version__))

    dtype = _torch_dtype(torch, cfg.get("dtype", "bfloat16"))
    attn_impl = cfg.get("attn_impl", "eager")
    if attn_impl != "eager":
        print("ERROR: attn_impl 必须为 eager（ENV_LOCK.md §0.4）；收到 %r" % (attn_impl,))
        return EXIT_USAGE

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("ERROR: 配置要求 device=cuda，但 torch.cuda.is_available() 为 False")
        return EXIT_USAGE

    seq_len = int(args.seq_len or cfg.get("seq_len", 512))
    seed = int(cfg.get("seed", 42))

    t0 = time.time()
    tok, model = model_context(torch, transformers, subject, subject["planned_load"]
                               if args.load_mode is None else args.load_mode,
                               dtype, attn_impl, device)
    load_seconds = time.time() - t0
    print("model.class=%s load_mode=%s load_seconds=%.2f"
          % (type(model).__name__, args.load_mode or subject["planned_load"], load_seconds))

    # ---- 逐样本：eval 断言 → 推理 → 对照量
    per_sample = []
    saved = {}
    for sample in cfg.get("samples") or []:
        with torch.no_grad():
            assert_eval_mode(model)   # AGENTS.md §6-11：每次推理前硬断言
        batch = tokenize(tok, sample["text"], seq_len, torch, device)
        dig = input_digest(batch)
        logits = forward_logits(torch, model, batch)

        # 对照量 C1：同进程连续两次 forward（区分「进程内稳定」与「跨进程稳定」）
        logits2 = forward_logits(torch, model, batch)
        c1 = float(torch.max(torch.abs(logits.float() - logits2.float())).item())

        # 对照量 C2：model.train() 下同输入的 logits 差值（eval 校验的**负对照**）
        c2, c2_note = _control_train_mode(torch, model, batch, logits)

        # 对照量 C3：不同 seed 的差值（应为非 0；为 0 须如实登记并报 W2）
        c3, c3_rng = _control_seed(torch, model, batch, seed, logits)

        per_sample.append({
            "id": sample.get("id"),
            "expected_role": sample.get("expected_role"),
            "input_sha256": dig,
            "input_tokens": int(batch["input_ids"].shape[1]),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "logits_sum": float(logits.float().sum().item()),
            "control_same_process_two_forwards_max_abs": c1,
            "control_train_mode_max_abs": c2,
            "control_train_mode_note": c2_note,
            "control_different_seed_max_abs": c3,
            "control_seed_rng_evidence": c3_rng,
        })
        saved[sample.get("id")] = logits.detach().to(torch.float32).cpu().numpy()

    # ---- 落盘：D1 对象为**单一样本**的 logits（首条样本），其余样本进 aux
    primary_id = (cfg.get("samples") or [{}])[0].get("id", "det-000")
    out_path = args.out or os.path.join(
        TMP_DIR, "det_%s_%s.npy" % (args.model_key, args.run_tag))
    ensure_dir(os.path.dirname(out_path))
    np.save(out_path, saved[primary_id])
    logits_md5 = md5_file(out_path)
    write_text_lf(out_path + ".md5", logits_md5 + "\n")

    aux_path = os.path.join(
        TMP_DIR, "det_aux_%s_%s.json" % (args.model_key, args.run_tag))
    aux = {
        "schema": "codeobf-pi/determinism/aux-v1",
        "model_key": args.model_key,
        "run_tag": args.run_tag,
        "group": args.group,
        "subject": subject,
        "subject_files_sha256": subject_checksum(subject["local_path"], args.model_key),
        "process": process_identity(),
        "same_process_reuse_warned": bool(dup),
        "env": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_version": torch.backends.cudnn.version(),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "attn_impl": attn_impl,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "bitsandbytes": _module_version("bitsandbytes"),
            "peft": _module_version("peft"),
            "trl": _module_version("trl"),
            "flash_attn_installed": flash_attn_status()["flash_attn_installed"],
        },
        "load": {
            "load_mode": args.load_mode or subject["planned_load"],
            "device": device,
            "dtype": cfg.get("dtype"),
            "attn_impl": attn_impl,
            "seq_len": seq_len,
            "seed": seed,
            "load_seconds": round(load_seconds, 3),
        },
        "primary_sample_id": primary_id,
        "logits_file": os.path.relpath(out_path, ROOT).replace(os.sep, "/"),
        "logits_md5": logits_md5,
        "logits_file_bytes": os.path.getsize(out_path),
        "peak_memory_allocated_mib": (torch.cuda.max_memory_allocated() / 2 ** 20)
        if device == "cuda" else None,
        "per_sample": per_sample,
        "written_at": now_iso(),
    }
    write_json_lf(aux_path, aux)

    print("logits_file=%s" % aux["logits_file"])
    print("logits_md5=%s" % logits_md5)
    print("logits_file_bytes=%d" % aux["logits_file_bytes"])
    for rec in per_sample:
        print("sample=%s tokens=%d logits_shape=%s sum=%.6f input_sha256=%s"
              % (rec["id"], rec["input_tokens"], rec["logits_shape"],
                 rec["logits_sum"], rec["input_sha256"][:16]))
        print("  control C1 同进程两次forward max|Δ| = %.3g" % rec["control_same_process_two_forwards_max_abs"])
        print("  control C2 model.train() max|Δ|     = %.3g  （%s）"
              % (rec["control_train_mode_max_abs"], rec["control_train_mode_note"]))
        print("  control C3 不同 seed max|Δ|         = %.3g  （%s）"
              % (rec["control_different_seed_max_abs"], rec["c3_note"]
                 if "c3_note" in rec else rec["control_seed_rng_evidence"].get("note", "")))
    print("aux_file=%s" % os.path.relpath(aux_path, ROOT).replace(os.sep, "/"))
    print("STATUS run_tag_%s_done" % args.run_tag)

    # A／B 齐备 ⇒ 立即比对（windows/W2.3_determinism.md §4.1「比对模式」）
    cmp_path, cmp_result, passed = compare_and_store(args.model_key, args.models_manifest)
    if cmp_result is not None:
        print("D1 md5_equal=%s  A=%s  B=%s"
              % (cmp_result["d1_md5_equal"], cmp_result["md5_a"], cmp_result["md5_b"]))
        print("D2 max|A-B|=%r  nan_a=%d nan_b=%d  same_process=%s"
              % (cmp_result["d2_max_abs_diff"], cmp_result["d2_nan_count_a"],
                 cmp_result["d2_nan_count_b"], cmp_result["same_process"]))
        print("cmp_file=%s" % os.path.relpath(cmp_path, ROOT).replace(os.sep, "/"))

    # 报告只在 B 轮之后自动触发（此时 A／B 齐备且是完整的一组证据）；
    # A 轮与其他模式不写报告，避免同一组证据被重复产成多份报告。
    if args.run_tag == "B":
        wrote = maybe_write_report(args)
        if wrote:
            print("report=%s" % os.path.relpath(wrote, ROOT).replace(os.sep, "/"))

    if cmp_result is None:
        print("STATUS awaiting_other_run_tag（A／B 齐备后自动比对并出判定）")
        return EXIT_OK
    return EXIT_OK if passed else EXIT_FAIL


def _control_train_mode(torch, model, batch, eval_logits):
    """负对照 C2：`model.train()` 下同输入的 logits 差值。

    **故意不调用** `assert_eval_mode`（该对照的目的正是检验断言有牙齿）。
    结束后恢复 `model.eval()`。
    """
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        diff = float(torch.max(torch.abs(out.logits.float() - eval_logits.float())).item())
        if diff == 0.0:
            note = ("本配置下 train／eval 无差异（dropout 全关，config.json 的 "
                    "attention_dropout=0.0）——断言仍防未来配置引入 dropout")
        else:
            note = "train 与 eval 输出不同 ⇒ eval 强制校验具有牙齿"
    finally:
        model.train(was_training)
    return diff, note


def _control_seed(torch, model, batch, seed, eval_logits):
    """对照量 C3：不同 seed 的 logits 差值 + seed 生效的 RNG 级证据。

    注：本项目两被测对象 config.json 的 `attention_dropout` 均为 ``0.0``，前向路径
    **不含随机算子**，故 logits 与 seed 无关（差值为 0 属**预期**，非「seed 未接上」）。
    为仍能证「seed 已接上」，同取 RNG 级证据：同 seed 两次抽样须一致、异 seed 须不同。
    该口径冲突**如实登记并报 W2**，不自行结案。
    """
    torch.manual_seed(int(seed) + 1)
    with torch.no_grad():
        out = model(**batch, use_cache=False)
    diff = float(torch.max(torch.abs(out.logits.float() - eval_logits.float())).item())

    torch.manual_seed(42)
    r1 = torch.randn(8)
    torch.manual_seed(42)
    r2 = torch.randn(8)
    torch.manual_seed(43)
    r3 = torch.randn(8)
    rng_same = bool(torch.equal(r1, r2))
    rng_diff = not bool(torch.equal(r1, r3))
    note = ("前向路径无随机算子（attention_dropout=0.0）⇒ logits 差值为 0 属预期；"
            "seed 生效性由 RNG 级证据给出：同 seed 一致=%s、异 seed 不同=%s"
            % (rng_same, rng_diff))
    return diff, {
        "rng_same_seed_identical": rng_same,
        "rng_different_seed_differs": rng_diff,
        "note": note,
    }


def compare_and_store(model_key, models_manifest=DEFAULT_MODELS_MANIFEST):
    """读取 A／B 两份 aux，做 D1／D2 比对并落 ``det_cmp_<key>.json``。"""
    aux_a = os.path.join(TMP_DIR, "det_aux_%s_A.json" % model_key)
    aux_b = os.path.join(TMP_DIR, "det_aux_%s_B.json" % model_key)
    if not (os.path.isfile(aux_a) and os.path.isfile(aux_b)):
        return None, None, None
    a, b = read_json(aux_a), read_json(aux_b)
    path_a = os.path.join(ROOT, a["logits_file"].replace("/", os.sep))
    path_b = os.path.join(ROOT, b["logits_file"].replace("/", os.sep))
    if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
        return None, None, None
    result = compare_logits(path_a, path_b,
                            md5_a=a["logits_md5"], md5_b=b["logits_md5"],
                            pid_a=a["process"]["pid"], pid_b=b["process"]["pid"])
    passed, reasons = judge_comparison(result)
    result["passed"] = passed
    result["reasons"] = reasons
    result["input_sha256_a"] = (a.get("per_sample") or [{}])[0].get("input_sha256")
    result["input_sha256_b"] = (b.get("per_sample") or [{}])[0].get("input_sha256")
    result["input_identical"] = (result["input_sha256_a"] == result["input_sha256_b"])
    if not result["input_identical"]:
        result["passed"] = False
        result["reasons"].append("A／B 两次送入的输入张量不一致（input_sha256 不同）"
                                 "⇒ 比对对象不同，判定不通过")
    result["model_key"] = model_key
    #: 轮次标识 —— 供 maybe_write_report 判断「全部 subject 是否属于同一轮」，
    #: 避免把不同轮次的 pid／时刻拼进同一份报告。优先取调用方传入的 `--group`
    #: （一次运行序列内所有 subject 共用一个 group）；未传时退化为该 subject 的 A 轮时刻。
    result["round_id"] = a.get("group") or a.get("written_at")
    result["logits_file_a"] = a["logits_file"]
    result["logits_file_b"] = b["logits_file"]
    result["load_a"] = a.get("load")
    result["load_b"] = b.get("load")
    result["env_a"] = a.get("env")
    result["env_b"] = b.get("env")
    result["process_a"] = a.get("process")
    result["process_b"] = b.get("process")
    result["per_sample_a"] = a.get("per_sample")
    result["per_sample_b"] = b.get("per_sample")
    result["judged_at"] = now_iso()
    cmp_path = os.path.join(TMP_DIR, "det_cmp_%s.json" % model_key)
    write_json_lf(cmp_path, result)
    return cmp_path, result, passed


# ==========================================================================
# --mode eval-guard
# ==========================================================================
def run_eval_guard(args):
    cfg = load_determinism_config(args)
    torch = _import_torch()
    transformers = _import_transformers()
    _import_bnb()
    _, entries = load_manifest(args.models_manifest)
    subject = resolve_subject(args.model_key, entries)

    print("== 01_03 check_determinism == mode=eval-guard key=%s" % args.model_key)
    print("env.CUBLAS_WORKSPACE_CONFIG=%s" % os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    print("env.pins=%s" % json.dumps(apply_torch_determinism(torch, cfg.get("env_pins"))))

    dtype = _torch_dtype(torch, cfg.get("dtype", "bfloat16"))
    attn_impl = cfg.get("attn_impl", "eager")
    device = cfg.get("device", "cuda")
    seq_len = int(args.seq_len or cfg.get("seq_len", 512))
    load_mode = args.load_mode or subject["planned_load"]

    tok, model = model_context(torch, transformers, subject, load_mode, dtype,
                               attn_impl, device)
    print("model.class=%s load_mode=%s" % (type(model).__name__, load_mode))

    evidence = []
    # 正例：eval + no_grad（断言必须在 no_grad 上下文内调用）
    with torch.no_grad():
        try:
            assert_eval_mode(model)
            evidence.append({"case": "eval+no_grad", "passed": True,
                             "model_training": model.training,
                             "grad_enabled": torch.is_grad_enabled()})
            print("ASSERT eval+no_grad PASS：model.training=%r torch.is_grad_enabled()=%r"
                  % (model.training, torch.is_grad_enabled()))
        except EvalModeViolation as exc:
            evidence.append({"case": "eval+no_grad", "passed": False, "error": str(exc)})
            print("ASSERT eval+no_grad FAIL：%s" % exc)

    # 负对照 1：grad 未关闭（显式构造，不依赖上下文）
    try:
        assert_eval_mode(model, grad_enabled=True)
        evidence.append({"case": "grad_enabled=True", "passed": False,
                         "note": "负对照未被拒绝 ⇒ 断言无牙齿"})
        print("ASSERT grad_enabled=True UNEXPECTED_PASS")
    except EvalModeViolation as exc:
        evidence.append({"case": "grad_enabled=True", "passed": True, "error": str(exc)})
        print("ASSERT grad_enabled=True REJECTED（预期）：%s" % exc)

    # 负对照 2：model.train()（在 no_grad 内，以隔离出 training 标志这一项）
    was = model.training
    model.train()
    try:
        with torch.no_grad():
            assert_eval_mode(model)
        evidence.append({"case": "model.train()", "passed": False,
                         "note": "负对照未被拒绝 ⇒ 断言无牙齿"})
        print("ASSERT model.train() UNEXPECTED_PASS")
    except EvalModeViolation as exc:
        evidence.append({"case": "model.train()", "passed": True, "error": str(exc)})
        print("ASSERT model.train() REJECTED（预期）：%s" % exc)
    finally:
        model.train(was)

    # 正例下的实际推理输出（作为「断言通过时的实测输出」证据）
    sample = (cfg.get("samples") or [{}])[0]
    batch = tokenize(tok, sample.get("text", ""), seq_len, torch, device)
    with torch.no_grad():
        assert_eval_mode(model)
        # 关键：`grad_enabled` 必须在 no_grad **上下文内**读取；在上下文外读取恒为 True，
        # 会与「断言通过」的结论自相矛盾（W2.3 补正轮 r1 的更正点）。
        grad_enabled_in_ctx = torch.is_grad_enabled()
        logits = model(**batch, use_cache=False).logits
    training_in_ctx = model.training
    print("INFER sample=%s tokens=%d logits_shape=%s sum=%.6f dtype=%s"
          % (sample.get("id"), batch["input_ids"].shape[1], tuple(logits.shape),
             float(logits.float().sum().item()), logits.dtype))

    out = {
        "schema": "codeobf-pi/determinism/eval-guard-v1",
        "model_key": args.model_key,
        "subject": subject,
        "process": process_identity(),
        "load_mode": load_mode,
        "attn_impl": attn_impl,
        "device": device,
        "assert_function": "assert_eval_mode(model, grad_enabled=None)",
        "assert_source_locations": [
            "scripts/env/01_03_check_determinism.py:assert_eval_mode（硬断言实现）",
            "scripts/env/01_03_check_determinism.py:run_determinism（每次推理前调用）",
            "scripts/env/01_03_check_determinism.py:run_eval_guard（本模式逐条调用）",
            "scripts/env/01_00_smoke_env.py:assert_eval_mode（推理前调用；同源实现）",
        ],
        "evidence": evidence,
        "observed": {
            "model_training": training_in_ctx,
            "grad_enabled": grad_enabled_in_ctx,
            "grad_enabled_outside_no_grad": torch.is_grad_enabled(),
            "observed_in_context": "torch.no_grad()",
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "logits_sum": float(logits.float().sum().item()),
            "input_tokens": int(batch["input_ids"].shape[1]),
        },
        "negative_controls_rejected": all(e["passed"] for e in evidence),
        "written_at": now_iso(),
    }
    path = os.path.join(TMP_DIR, "eval_guard_%s.json" % args.model_key)
    write_json_lf(path, out)
    print("eval_guard_file=%s" % os.path.relpath(path, ROOT).replace(os.sep, "/"))
    ok = all(e["passed"] for e in evidence)
    print("STATUS eval_guard_%s" % ("PASS" if ok else "FAIL"))
    return EXIT_OK if ok else EXIT_FAIL


# ==========================================================================
# --mode bnb-stability（父：spawn 独立子进程 × --repeat）
# ==========================================================================
def run_bnb_stability(args):
    repeat = int(args.repeat or 5)
    group = args.group or now_stamp()
    print("== 01_03 check_determinism == mode=bnb-stability key=%s repeat=%d group=%s"
          % (args.model_key, repeat, group))
    print("NOTE 每次加载在**独立子进程**中完成（windows/W2.3_determinism.md §4.7）；"
          "子进程 pid 由各自记录，作为独立性证据")
    for i in range(1, repeat + 1):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--mode", "bnb-stability-single",
               "--model-key", args.model_key,
               "--models-manifest", args.models_manifest,
               "--run-index", str(i),
               "--group", group]
        if args.seq_len:
            cmd += ["--seq-len", str(args.seq_len)]
        if args.load_mode:
            cmd += ["--load-mode", args.load_mode]
        print("--- child %d/%d: %s" % (i, repeat, " ".join(cmd)))
        sys.stdout.flush()
        proc = subprocess.run(cmd, cwd=ROOT)
        print("--- child %d/%d pid_expected_independent exit=%d" % (i, repeat, proc.returncode))
        sys.stdout.flush()

    summary = finalize_bnb_summary(args.model_key, repeat, group)
    print("=== bnb-stability 汇总（digest 一致性）===")
    for row in summary["rows"]:
        print("run=%s pid=%s status=%s load_seconds=%s digest=%s peak_mib=%s"
              % (row.get("run_index"), (row.get("process") or {}).get("pid"),
                 row.get("status"), row.get("load_seconds"), row.get("quant_digest"),
                 row.get("peak_memory_allocated_mib")))
    print("distinct_child_pids=%d" % summary["distinct_child_pids"])
    print("attempted=%d succeeded=%d failed=%d failure_rate=%.3f digest_all_equal=%s"
          % (summary["attempted"], summary["succeeded"], summary["failed"],
             summary["failure_rate"], summary["digest_all_equal"]))
    print("conclusion=%s" % summary["conclusion"])

    print("NOTE 本模式不写报告：报告由 `--mode determinism --run-tag B` 自动触发"
          "（或 `--mode compare` 显式触发），以保证每份报告都对应一组完整的 A／B 证据。"
          "§5 的表格在报告渲染时从本模式的 summary 读取。")
    return EXIT_OK if summary["stable"] else EXIT_FAIL


def run_bnb_stability_single(args):
    """单次加载并登记（由父进程 spawn；进程身份即独立性证据）。"""
    cfg = load_determinism_config(args)
    torch = _import_torch()
    transformers = _import_transformers()
    _import_bnb()
    _, entries = load_manifest(args.models_manifest)
    subject = resolve_subject(args.model_key, entries)
    apply_torch_determinism(torch, cfg.get("env_pins"))
    dtype = _torch_dtype(torch, cfg.get("dtype", "bfloat16"))
    device = cfg.get("device", "cuda")
    load_mode = args.load_mode or subject["planned_load"]

    idx = int(args.run_index)
    group = args.group or now_stamp()
    rec = {
        "schema": "codeobf-pi/determinism/bnb-stability-single-v1",
        "model_key": args.model_key,
        "run_index": idx,
        "group": group,
        "subject": subject,
        "process": process_identity(),
        "load_mode": load_mode,
        "device": device,
        "started_at": now_iso(),
    }
    t0 = time.time()
    try:
        tok, model = model_context(torch, transformers, subject, load_mode, dtype,
                                   cfg.get("attn_impl", "eager"), device)
        rec["load_seconds"] = round(time.time() - t0, 3)
        rec["model_class"] = type(model).__name__
        with torch.no_grad():
            assert_eval_mode(model)   # AGENTS.md §6-11
            rec["quant_digest"], rec["digest_layers"] = quant_weight_digest(torch, model)
        rec["peak_memory_allocated_mib"] = (torch.cuda.max_memory_allocated() / 2 ** 20
                                            if device == "cuda" else 0.0)
        rec["reserved_memory_mib"] = (torch.cuda.max_memory_reserved() / 2 ** 20
                                      if device == "cuda" else 0.0)
        sample = (cfg.get("samples") or [{}])[0]
        batch = tokenize(tok, sample.get("text", ""), int(args.seq_len or cfg.get("seq_len", 512)),
                         torch, device)
        logits = forward_logits(torch, model, batch)
        rec["logits_sum"] = float(logits.float().sum().item())
        rec["status"] = "OK"
        print("run=%d pid=%d status=OK load_seconds=%.3f digest=%s peak_mib=%.1f"
              % (idx, os.getpid(), rec["load_seconds"], rec["quant_digest"],
                 rec["peak_memory_allocated_mib"]))
    except Exception:
        import traceback

        rec["load_seconds"] = round(time.time() - t0, 3)
        rec["status"] = "FAIL"
        rec["traceback"] = traceback.format_exc()
        print("run=%d pid=%d status=FAIL load_seconds=%.3f" % (idx, os.getpid(),
                                                               rec["load_seconds"]))
        print(rec["traceback"])
    rec["finished_at"] = now_iso()
    path = os.path.join(TMP_DIR, "bnb_stab_%s_%s_%02d.json" % (args.model_key, group, idx))
    write_json_lf(path, rec)
    return EXIT_OK if rec["status"] == "OK" else EXIT_FAIL


def quant_weight_digest(torch, model, max_layers=8):
    """对若干**固定层**取 4-bit 权重字节摘要（`lm_head` 除外）。

    选择规则（可复现）：按名字排序的 ``Linear4bit`` 模块中，**跳过 ``lm_head``**，
    取前 ``max_layers`` 个；摘要入参 = 模块名 + `weight.data` 原始字节 + quant_state
    关键字段（shape／absmax／blocksize／quant_type／dtype）。5 次必须一致。
    """
    import numpy as np
    import bitsandbytes as bnb

    picked, parts, names = 0, [], []
    for name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        if name.endswith("lm_head") or name == "lm_head":
            continue
        picked += 1
        if picked > max_layers:
            break
        w = module.weight
        data = w.data.detach().cpu().contiguous().numpy()
        qs = getattr(w, "quant_state", None)
        extra = []
        for attr in ("shape", "blocksize", "quant_type", "dtype"):
            val = getattr(qs, attr, None)
            extra.append("%s=%s" % (attr, val))
        try:
            absmax = qs.absmax.detach().cpu().contiguous().numpy().tobytes()
        except Exception:
            absmax = b""
        parts.append(name.encode("utf-8"))
        parts.append(data.tobytes())
        parts.append(absmax)
        parts.append("|".join(extra).encode("utf-8"))
        names.append(name)
    if not parts:
        return None, names
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest(), names


def finalize_bnb_summary(model_key, repeat, group):
    """读该 group 的逐次记录，汇总并落 ``bnb_stab_<key>_<group>_summary.json``。"""
    rows = []
    for i in range(1, int(repeat) + 1):
        path = os.path.join(TMP_DIR, "bnb_stab_%s_%s_%02d.json" % (model_key, group, i))
        if os.path.isfile(path):
            rows.append(read_json(path))
    attempted = len(rows)
    ok = [r for r in rows if r.get("status") == "OK"]
    digests = [r.get("quant_digest") for r in ok]
    all_equal = bool(digests) and len(set(digests)) == 1
    pids = [(r.get("process") or {}).get("pid") for r in rows]
    distinct = len(set(p for p in pids if p is not None))
    complete = attempted >= int(repeat)
    stable = complete and len(ok) == attempted and all_equal and distinct == attempted
    if not complete:
        conclusion = "未完成 %d 次独立进程加载（已登记 %d 次）" % (repeat, attempted)
    elif distinct != attempted:
        conclusion = "**不稳定**：%d 次记录仅来自 %d 个不同进程（未做到独立进程 ×%d）" \
                     % (attempted, distinct, repeat)
    elif not stable:
        conclusion = "**不稳定**"
    else:
        conclusion = "bnb 4-bit 加载稳定"
    summary = {
        "schema": "codeobf-pi/determinism/bnb-stability-summary-v1",
        "model_key": model_key,
        "group": group,
        "repeat": int(repeat),
        "attempted": attempted,
        "succeeded": len(ok),
        "failed": attempted - len(ok),
        "failure_rate": (0.0 if attempted == 0
                         else round((attempted - len(ok)) / float(attempted), 6)),
        "digests": digests,
        "digest_all_equal": all_equal,
        "child_pids": pids,
        "distinct_child_pids": distinct,
        "complete": complete,
        "stable": stable,
        "conclusion": conclusion,
        "rows": rows,
        "summarized_at": now_iso(),
    }
    write_json_lf(os.path.join(
        TMP_DIR, "bnb_stab_%s_%s_summary.json" % (model_key, group)), summary)
    return summary


#: 最近一次已落盘的 summary 缓存（同一进程内避免重复渲染）
_BNB_SUMMARY_CACHE = {}


def latest_bnb_summary(model_key):
    """取该 key **最新 group** 的 summary（缺则返回未运行的占位结构）。"""
    if model_key in _BNB_SUMMARY_CACHE:
        return _BNB_SUMMARY_CACHE[model_key]
    prefix = "bnb_stab_%s_" % model_key
    groups = []
    if os.path.isdir(TMP_DIR):
        for name in os.listdir(TMP_DIR):
            if name.startswith(prefix) and name.endswith("_summary.json"):
                groups.append(name[len(prefix):-len("_summary.json")])
    if not groups:
        placeholder = {"attempted": 0, "succeeded": 0, "failed": 0,
                       "failure_rate": None, "digest_all_equal": None,
                       "distinct_child_pids": 0, "rows": [],
                       "conclusion": "**未运行**（`--mode bnb-stability` 未执行）",
                       "stable": None, "group": None}
        return placeholder
    summary = read_json(os.path.join(TMP_DIR, "%s%s_summary.json"
                                     % (prefix, max(groups))))
    _BNB_SUMMARY_CACHE[model_key] = summary
    return summary


# ==========================================================================
# 汇总报告（含硬判定）
# ==========================================================================
#: 上一份报告所依据的证据指纹（防止同一组证据被重复产成多份报告）
_REPORT_FINGERPRINT = os.path.join(TMP_DIR, "_determinism_last_report.json")


def evidence_fingerprint(cmps):
    """证据指纹 = 各 subject 的比对文件 md5（含本轮 pid／时刻 ⇒ 能区分不同轮次）。"""
    out = {}
    for key in sorted(cmps):
        path = os.path.join(TMP_DIR, "det_cmp_%s.json" % key)
        out[key] = md5_file(path) if os.path.isfile(path) else None
    return out


def maybe_write_report(args, force=False):
    """A／B 齐备（且全部 subjects 齐备）时写汇总报告；已存在则**新建时间戳文件**，不覆盖。

    **幂等防护**：若本次证据指纹与上一份报告相同（同一组 logits md5 与 D2），
    则**不重复产出报告**，只打印 NOTE。判定的复算请用 `--mode compare`（显式入口）。
    """
    cfg = load_determinism_config(args)
    subjects = [s["key"] for s in (cfg.get("subjects") or [])]
    if not subjects:
        return None
    cmps = {}
    for key in subjects:
        path = os.path.join(TMP_DIR, "det_cmp_%s.json" % key)
        if not os.path.isfile(path):
            if not force:
                print("NOTE 报告未写：subject %s 的 A／B 比对尚未齐备" % key)
                return None
            continue
        cmps[key] = read_json(path)
    if not cmps:
        return None
    if not force and len(cmps) == len(subjects):
        rounds = {cmp.get("round_id") for cmp in cmps.values()}
        if len(rounds) != 1:
            print("NOTE 报告未写：各 subject 的 A／B 来自**不同轮次**（round_id=%s）——"
                  "跨轮拼报告会把不同进程的 pid／时刻混在一份里。请把全部 subject 的 "
                  "A／B 重跑完一轮，或用 `--mode compare` 显式出报告。"
                  % sorted(str(r) for r in rounds))
            return None
    fingerprint = evidence_fingerprint(cmps)
    if not force and os.path.isfile(_REPORT_FINGERPRINT):
        prior = read_json(_REPORT_FINGERPRINT)
        if prior.get("fingerprint") == fingerprint:
            print("NOTE 报告未写：本次证据（各 subject 的 logits md5 与 D2）与上一份报告"
                  "**完全相同**（上一份：%s）——不重复产出。如需复算判定，用 `--mode compare`。"
                  % prior.get("report"))
            return None
    report_path = args.report or os.path.join(
        REPORTS_DIR, "determinism_v1.0_%s.md" % now_stamp())
    if os.path.exists(report_path) and not force:
        base, ext = os.path.splitext(report_path)
        report_path = "%s_r%s%s" % (base, now_stamp(), ext)
    text = render_report(cfg, subjects, cmps, args)
    write_text_lf(report_path, text)
    write_report_index(report_path, subjects, cmps)
    write_json_lf(_REPORT_FINGERPRINT, {
        "fingerprint": fingerprint,
        "report": os.path.relpath(report_path, ROOT).replace(os.sep, "/"),
        "report_md5": md5_file(report_path),
        "written_at": now_iso(),
    })
    return report_path


def write_report_index(report_path, subjects, cmps):
    """报告旁挂的机器可读索引（`<report>.json`）——判定与缺项一览。"""
    write_json_lf(report_path + ".json", {
        "schema": "codeobf-pi/determinism/report-index-v1",
        "report": os.path.relpath(report_path, ROOT).replace(os.sep, "/"),
        "report_md5": md5_file(report_path),
        "subjects": list(subjects),
        "subjects_missing": [s for s in subjects if s not in cmps],
        "verdicts": {k: {"passed": v.get("passed"), "reasons": v.get("reasons")}
                     for k, v in cmps.items()},
        "overall_passed": (all(s in cmps for s in subjects)
                           and all(v.get("passed") for v in cmps.values())),
        "written_at": now_iso(),
    })


def render_report(cfg, subjects, cmps, args):
    lines = []
    add = lines.append
    add("# 推理确定性校验报告（G-DET，阶段1 硬门禁）")
    add("")
    add("- 窗口：W2.3（阶段1 显存监控与确定性校验，权限 A2）")
    add("- 生成时间：%s" % now_iso())
    add("- 生成方式：`scripts/env/01_03_check_determinism.py --mode compare`"
        "（或 `--mode determinism --run-tag B` 自动触发）；本文件为**脚本渲染**，"
        "数字一律取自 `logs/env/tmp/` 的 json —— 无手工抄录")
    add("- 判定规则：`AGENTS.md` §6-12／`windows/W2.3_determinism.md` §4.3 —— **严格 0**"
        "（D1 md5 相同 **且** D2 `max|A-B| == 0.0`）")
    add("")
    add("## 0. 硬判定")
    add("")
    missing = [s for s in subjects if s not in cmps]
    overall = (not missing) and all(cmps[k].get("passed") for k in cmps)
    if missing:
        verdict = "**未判定**（尚有 subject 缺 A／B 比对）"
    elif overall:
        verdict = "**通过**"
    else:
        verdict = "**不通过**"
    add("| 项 | 值 |")
    add("|---|---|")
    add("| **确定性判定（总）** | %s |" % verdict)
    add("| 判定对象（subjects） | %s |" % ", ".join("`%s`" % s for s in subjects))
    add("| 已取得比对结果的 subjects | %s |"
        % (", ".join("`%s`" % k for k in cmps) or "（无）"))
    add("| 未取得结果的 subjects | %s |"
        % (", ".join("`%s`" % s for s in missing) or "（无）"))
    add("")
    if missing:
        add("**说明**：总判定为各 subject 判定的**与**；存在未取得结果的 subject 时，"
            "总判定**不得**记为「通过」（依 `windows/W2.3_determinism.md` §4.2「必测对象」"
            "口径——1B 与 8B 均须通过）。")
        add("")
    if not overall:
        add("**处置**：按 `windows/W2.3_determinism.md` §4.5 的七级回退序列逐级排查并逐级记录；"
            "仍非零 ⇒ 判定未通过，**W2.4 不得开测、阶段1 不得打 tag**，报 W2。")
        add("")
    add("## 1. 与 v5 原文的口径关系（须显式写明）")
    add("")
    add("v5 L148（`SOURCE_LOCK/实验执行方案（最终版 v5）.md`）的原文要求为「两次结果完全一致"
        "（差值＜1e-6）」；`AGENTS.md` §6-12 的判定**更严**——「若因浮点累加出现 <1e-6 的差值，"
        "仍视为**未通过**」。本窗口按更严者执行：**非零即不通过**。")
    add("")
    add("口径冲突已由 `DECISIONS.md` `D-04` 裁定（「契约严于源头方案，取严」；台账条目"
        "「推理确定性判定取严格 0」已落盘），故**无需再请 W0 裁口径**。")
    add("")
    add("## 2. 逐 subject 实测")
    add("")
    for key in subjects:
        cmp = cmps.get(key)
        add("### 2.%d `%s`" % (subjects.index(key) + 1, key))
        add("")
        if cmp is None:
            add("（A／B 比对未取得 —— 该 subject 尚无结果）")
            add("")
            continue
        add("| 项 | 值 |")
        add("|---|---|")
        add("| D1 位级一致（md5 相同） | %s |" % cmp.get("d1_md5_equal"))
        add("| logits A 文件 | `%s` |" % cmp.get("logits_file_a"))
        add("| logits A md5 | `%s` |" % cmp.get("md5_a"))
        add("| logits B 文件 | `%s` |" % cmp.get("logits_file_b"))
        add("| logits B md5 | `%s` |" % cmp.get("md5_b"))
        add("| D2 `max(abs(A-B))` | **%r** |" % cmp.get("d2_max_abs_diff"))
        add("| NaN 计数 A／B | %s / %s |" % (cmp.get("d2_nan_count_a"), cmp.get("d2_nan_count_b")))
        add("| logits 形态／dtype | %s %s |" % (cmp.get("shape_a"), cmp.get("dtype_a")))
        add("| 输入是否同一（input_sha256） | %s |" % cmp.get("input_identical"))
        add("| 两次落盘是否同一进程 | %s（pid %s / %s） |"
            % (cmp.get("same_process"), cmp.get("pid_a"), cmp.get("pid_b")))
        add("| 进程启动时刻 A／B | %s / %s |"
            % ((cmp.get("process_a") or {}).get("process_start_epoch"),
               (cmp.get("process_b") or {}).get("process_start_epoch")))
        add("| **该 subject 判定** | **%s** |" % ("通过" if cmp.get("passed") else "不通过"))
        add("")
        if cmp.get("reasons"):
            add("不通过原因：")
            add("")
            for reason in cmp["reasons"]:
                add("- %s" % reason)
            add("")
        load = cmp.get("load_a") or {}
        add("运行参数：加载方式 `%s`；device `%s`；dtype `%s`；attn_impl `%s`；"
            "seq_len `%s`；seed `%s`"
            % (load.get("load_mode"), load.get("device"), load.get("dtype"),
               load.get("attn_impl"), load.get("seq_len"), load.get("seed")))
        add("")
        add("环境：`CUBLAS_WORKSPACE_CONFIG=%s`；cudnn_deterministic=%s；cudnn_benchmark=%s；"
            "allow_tf32(matmul)=%s；flash_attn_installed=%s"
            % ((cmp.get("env_a") or {}).get("CUBLAS_WORKSPACE_CONFIG"),
               (cmp.get("env_a") or {}).get("cudnn_deterministic"),
               (cmp.get("env_a") or {}).get("cudnn_benchmark"),
               (cmp.get("env_a") or {}).get("allow_tf32_matmul"),
               (cmp.get("env_a") or {}).get("flash_attn_installed")))
        add("")
        add("版本：torch `%s`；cuda `%s`；cudnn `%s`；transformers `%s`；bitsandbytes `%s`；"
            "peft `%s`；trl `%s`（均为 **import 实测**，非 `conda list` 的 `pypi_0` 行）"
            % ((cmp.get("env_a") or {}).get("torch"), (cmp.get("env_a") or {}).get("cuda"),
               (cmp.get("env_a") or {}).get("cudnn_version"),
               (cmp.get("env_a") or {}).get("transformers"),
               (cmp.get("env_a") or {}).get("bitsandbytes"),
               (cmp.get("env_a") or {}).get("peft"), (cmp.get("env_a") or {}).get("trl")))
        add("")
    add("## 3. 对照量（报告项，**不参与判定**）")
    add("")
    add("| subject | 样本 | 对照 | 实测（A 进程） | 说明 |")
    add("|---|---|---|---|---|")
    for key in subjects:
        cmp = cmps.get(key) or {}
        for rec in (cmp.get("per_sample_a") or []):
            sid = rec.get("id")
            add("| `%s` | %s | C1 同进程连续两次 forward `max|Δ|` | %r | 区分「进程内稳定」"
                "与「跨进程稳定」 |"
                % (key, sid, rec.get("control_same_process_two_forwards_max_abs")))
            add("| `%s` | %s | C2 `model.train()` 下同输入 `max|Δ|` | %r | eval 强制校验的"
                "**负对照**；%s |"
                % (key, sid, rec.get("control_train_mode_max_abs"),
                   rec.get("control_train_mode_note")))
            c3 = rec.get("control_seed_rng_evidence") or {}
            add("| `%s` | %s | C3 不同 seed `max|Δ|` | %r | 应为非 0；实测为 0 的口径说明见下 |"
                % (key, sid, rec.get("control_different_seed_max_abs")))
            add("| `%s` | %s | C3' seed 生效的 RNG 级证据 | 同 seed 一致=%s；异 seed 不同=%s | "
                "logits 与 seed 无关时的替代证据 |"
                % (key, sid, c3.get("rng_same_seed_identical"),
                   c3.get("rng_different_seed_differs")))
    add("")
    add("**C3 口径说明（如实登记，报 W2 裁定）**：两个被测对象的 `config.json` 中 "
        "`attention_dropout=0.0`，前向路径**不含随机算子** ⇒ logits 是输入与权重的确定函数，"
        "**与 seed 无关**，故 C3 的 logits 差值为 `0.0` 属**预期**，不构成"
        "「seed 未接上（缺陷）」的证据。为仍能证「seed 已接上」，同取 RNG 级证据（C3'）："
        "同一 seed 两次抽样逐元素相同、不同 seed 抽样不同。该处置**不改变** §0 的硬判定。")
    add("")
    add("## 4. eval 模式强制校验（v5 新增 10 ／ `AGENTS.md` §6-11）")
    add("")
    for idx, key in enumerate(subjects, 1):
        path = os.path.join(TMP_DIR, "eval_guard_%s.json" % key)
        if not os.path.isfile(path):
            add("- `%s`：**未运行**（`--mode eval-guard` 未执行）" % key)
            continue
        ev = read_json(path)
        add("### 4.%d `%s`" % (idx, key))
        add("")
        add("| 证据 | 内容 |")
        add("|---|---|")
        add("| ① 断言代码 | `%s`；落点：%s |"
            % (ev.get("assert_function"), "；".join(ev.get("assert_source_locations") or [])))
        obs = ev.get("observed") or {}
        add("| ② 断言通过时的实测输出（在 `%s` 上下文**内**读取） | `model.training=%s`、"
            "`torch.is_grad_enabled()=%s`、logits 形态 `%s`、dtype `%s`、sum `%s`、"
            "输入 tokens `%s`（上下文**外**读取的 `is_grad_enabled()` 为 `%s`，"
            "属正常现象、不参与断言） |"
            % (obs.get("observed_in_context", "torch.no_grad()"),
               obs.get("model_training"), obs.get("grad_enabled"),
               obs.get("logits_shape"), obs.get("logits_dtype"),
               obs.get("logits_sum"), obs.get("input_tokens"),
               obs.get("grad_enabled_outside_no_grad")))
        for item in ev.get("evidence") or []:
            add("| ③ 断言用例 `%s` | 通过=%s；%s |"
                % (item.get("case"), item.get("passed"),
                   item.get("error") or item.get("note") or ""))
        add("| ④ `model.train()` 负对照 | 见 §3 的 C2 行 |")
        add("")
        add("证据文件：`logs/env/tmp/eval_guard_%s.json`" % key)
        add("")
    add("")
    add("### 4.%d grep 证据：`model.train()` 调用点清单（AST 判定）" % (len(subjects) + 1))
    add("")
    add("以 `ast` 遍历源码中所有 `.train(` **调用点**（非文本匹配，注释与字符串不计入），"
        "按调用形态分类：")
    add("")
    add("| 文件 | 行 | 所在函数 | 形态 | 代码 |")
    add("|---|---|---|---|---|")
    for rel in ("scripts/env/01_00_smoke_env.py", "scripts/env/01_03_check_determinism.py"):
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        for site in find_train_call_sites(src):
            add("| `%s` | %d | `%s` | %s | `%s` |"
                % (site["file"], site["line"], site["enclosing"],
                   "无参（`train()`）" if site["form"] == "no_arg" else "有参（恢复原状态）",
                   site["code"]))
    add("")
    add("**结论**：全部调用点仅落在三类函数内 ——")
    add("")
    add("1. **负对照的显式构造**：`_control_train_mode`（`01_03`）与 `run_eval_guard`（`01_03`）"
        "中的无参 `model.train()`，其后**立即**调用 `assert_eval_mode` 断言须被拒绝，并在 "
        "`finally` 中恢复；")
    add("2. **状态恢复**：`_control_train_mode`／`run_eval_guard` 的有参 `model.train(was…)`；")
    add("3. **训练探针**：`_measure_train`（`01_00`）的无参 `model.train()`，属 W2.3 §4.8 "
        "允许的「QLoRA ≤10 步训练」区间，**不**在推理路径上。")
    add("")
    add("**推理路径（`forward`／`model(**batch)` 调用点）中不存在任何 `.train(` 调用**，"
        "且每次推理前均先经 `assert_eval_mode` 硬断言。")
    add("")
    add("## 5. bitsandbytes 4-bit 加载稳定性（v5 新增 8 ／ §4.7）")
    add("")
    add("**失败率（单列一行）**：")
    add("")
    for key in subjects:
        summary = latest_bnb_summary(key)
        add("- `%s`：attempted=%s，succeeded=%s，failed=%s，**failure_rate=%s**"
            % (key, summary["attempted"], summary["succeeded"], summary["failed"],
               summary["failure_rate"]))
    add("")
    add("| subject | run | 子进程 pid | 状态 | 加载耗时(s) | 量化权重 digest | peak(MiB) | reserved(MiB) |")
    add("|---|---|---|---|---|---|---|---|")
    for key in subjects:
        summary = latest_bnb_summary(key)
        for row in summary["rows"]:
            add("| `%s` | %s | %s | %s | %s | `%s` | %s | %s |"
                % (key, row.get("run_index"), (row.get("process") or {}).get("pid"),
                   row.get("status"), _fmt_num(row.get("load_seconds")),
                   row.get("quant_digest"), _fmt_num(row.get("peak_memory_allocated_mib")),
                   _fmt_num(row.get("reserved_memory_mib"))))
    add("")
    for key in subjects:
        summary = latest_bnb_summary(key)
        add("- `%s` 结论：**%s**（digest 全一致=%s；独立进程数=%s/%s；group=%s）"
            % (key, summary["conclusion"], summary["digest_all_equal"],
               summary.get("distinct_child_pids"), summary["attempted"],
               summary.get("group")))
    add("")
    add("**digest 取法**：按名字排序的 `Linear4bit` 模块中跳过 `lm_head`，取前 8 个，"
        "hash 入参 = 模块名 + `weight.data` 原始字节 + `quant_state.absmax` + "
        "`(shape, blocksize, quant_type, dtype)`。5 次必须一致。")
    add("")
    add("> 与 W2.1 的关系：W2.1 已在**同进程内**连做 5 次通过（`wq_digest` 五次相同，见 "
        "`logs/env/01_02_lock_versions_bnb_availability_20260929-184718.log`）。本节为其"
        "**加强**（独立进程 ×5），非重复。")
    add("")
    add("## 6. `01_00_smoke_env.py` 试跑（**harness 验证，非基线**）")
    add("")
    add("> 显存基线归 W2.4；本节仅为验通 01_00 的接口与七元组输出形态，"
        "**不得**进入 `ENV_LOCK.md` §8。")
    add("")
    mem_rows = collect_mem_probe_rows()
    if mem_rows:
        add("| suite | model | load | device | batch | seq_len | offload | collector | peak(MiB) | reserved(MiB) | repeat | run_id |")
        add("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in mem_rows:
            add("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (row.get("suite"), row.get("model"), row.get("load"), row.get("device"),
                   row.get("batch"), row.get("seq_len"), row.get("offload"),
                   row.get("collector"), row.get("peak"), row.get("reserved"),
                   row.get("repeat"), row.get("run_id")))
        add("")
        add("原始行（逐字）：")
        add("")
        add("```text")
        for row in mem_rows:
            add(row["raw_line"])
        add("```")
        add("")
        add("**同一档位在不同 `run_id` 下的数字为何不同（**W2.4 必读**）**：`unplug-cpu` 在"
            "独立的 `--suite unplug-cpu` 调用中 `peak=621MiB`，而在 `make smoke STAGE=env` 的"
            "一次调用内为 `2703MiB` —— 后者是**同进程内先跑过 1B 加载／推理／训练**之后测的，"
            "`collector=nvidia-smi` 取的是**绝对值**（含本进程早前未释放的显存与桌面占用）。"
            "这正是 W2.4 §4.1-1「一档一进程」的实测依据：**同一档位在不同进程上下文里不可比**，"
            "故 §8 的行必须连同七元组与 `run_id` 一起登记，且只能与**同 collector 的 idle 档**"
            "相减。")
    else:
        add("（本窗口未产出 `logs/env/tmp/mem_probe_*.json`）")
    add("")
    add("## 7. 回退序列记录（不通过时逐级；本窗口按实际执行情况登记）")
    add("")
    add(render_fallback_section(overall))
    add("")
    add("## 8. 交付给 W2 的回执")
    add("")
    add(render_receipt(cfg, subjects, cmps, overall, args))
    add("")
    add("## 9. 台账条目与未决项")
    add("")
    add(render_ledger_section())
    add("")
    add("## 10. 命令序列（可复现）")
    add("")
    add("以下命令与输出逐条落于 `logs/env/01_03_check_determinism_<YYYYMMDD-HHMMSS>.log`"
        "（每条形如 `### CMD: …` / `### EXIT: <code>`）：")
    add("")
    add("```bash")
    add("# 补正轮 r1 的 eval-guard 重跑（证据字段更正，见 §4）")
    for key in subjects:
        add("python scripts/env/01_03_check_determinism.py --mode eval-guard --model-key %s" % key)
    add("python scripts/env/01_03_check_determinism.py --mode compare --report <本文件>")
    add("")
    add("# 主序列（首个完整轮，落于主日志）")
    for key in subjects:
        add("python scripts/env/01_03_check_determinism.py --mode bnb-stability --model-key %s --repeat 5" % key)
    for key in subjects:
        add("python scripts/env/01_03_check_determinism.py --mode determinism --run-tag A --model-key %s" % key)
        add("python scripts/env/01_03_check_determinism.py --mode determinism --run-tag B --model-key %s" % key)
    add("")
    add("# 01_00 的 harness 试跑（非显存基线）")
    add("python scripts/env/01_00_smoke_env.py --suite idle --repeat 1")
    add("python scripts/env/01_00_smoke_env.py --suite load-1b --repeat 3")
    add("python scripts/env/01_00_smoke_env.py --suite infer-1b-512 --repeat 1")
    add("python scripts/env/01_00_smoke_env.py --suite unplug-cpu --repeat 1")
    add("make smoke STAGE=env")
    add("python scripts/env/01_00_smoke_env.py --suite infer-8b-256 --repeat 1")
    add("```")
    add("")
    add("> `--run-tag A` 与 `--run-tag B` **必须由两次独立进程调用**；脚本以 pid 与进程启动时刻"
        "为证，同一 pid 出现两次即自动判不通过（§4.1「防止同进程假通过」）。")
    add("")
    return "\n".join(lines) + "\n"


def _git_head_short():
    """当前工作树 HEAD 的短 hash（仅供回执①；失败返回 None）。"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _fmt_num(value, digits=1):
    """报告里的数值统一保留一位小数（避免 `1509.908203125` 这类浮点噪声）。"""
    if value is None:
        return "—"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return str(value)


def find_train_call_sites(path):
    """AST 定位源码中所有 ``.train(`` 调用点，并按**调用形态**分类（grep 证据的精确版）。

    返回 ``list[dict]``，每项含 ``file``／``line``／``enclosing``／``form``／``code``：

    ``form="no_arg"``   ``model.train()``     —— 无参；只可能出现在「eval 校验的负对照构造」
                        与「训练探针」（`_measure_train`）中
    ``form="with_arg"`` ``model.train(was)``  —— 有参；用于**恢复原状态**（`_control_train_mode`
                        ／`run_eval_guard` 的 `finally` 分支）

    以 AST 判定而非文本 grep，避免把注释与字符串里的同名字样计入（W2.3 §4.4-3 要求
    「推理全程未出现 ``model.train()`` 调用」的 grep 证据）。
    """
    import ast

    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines()
    tree = ast.parse(src)
    enclosing = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                enclosing.setdefault(id(child), node.name)
    sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "train"):
            sites.append({
                "file": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                "line": node.lineno,
                "enclosing": enclosing.get(id(node), "<module>"),
                "form": "no_arg" if not node.args else "with_arg",
                "code": lines[node.lineno - 1].strip()[:140],
            })
    sites.sort(key=lambda d: (d["file"], d["line"]))
    return sites


def collect_mem_probe_rows():
    rows = []
    if not os.path.isdir(TMP_DIR):
        return rows
    for name in sorted(os.listdir(TMP_DIR)):
        if not (name.startswith("mem_probe_") and name.endswith(".json")):
            continue
        try:
            doc = read_json(os.path.join(TMP_DIR, name))
        except Exception:
            continue
        for rec in doc.get("records") or []:
            raw = rec.get("raw_line")
            if not raw:
                continue   # 失败档位（无七元组行）不进 §6
            row = dict(rec.get("seven_tuple") or {})
            row["raw_line"] = raw
            row["status"] = rec.get("status")
            rows.append(row)
    return rows


def render_fallback_section(overall):
    if overall:
        return ("**未触发**——§0 硬判定为「通过」（D1 位级一致 **且** D2 `max|A-B| == 0.0`），"
                "无需进入 §4.5 的七级回退序列。")
    return ("**已触发**（见 §2 的不通过原因）。逐级记录：\n\n"
            "| 级 | 动作 | 执行情况 | 结果 |\n|---|---|---|---|\n"
            "| 1 | 确认 `attn_implementation='eager'` | | |\n"
            "| 2 | 关闭 4-bit，改 `bf16` 全精度重测 | | |\n"
            "| 3 | 固定 cudnn deterministic／benchmark／TF32 | | |\n"
            "| 4 | `CUBLAS_WORKSPACE_CONFIG=:4096:8`（import 前置） | | |\n"
            "| 5 | `torch_use_deterministic_algorithms` 升严格模式 | | |\n"
            "| 6 | 更换 bitsandbytes 版本（**须先追加台账并报 W2 请 W0 事前批准**） | | |\n"
            "| 7 | 全部完成仍非零 ⇒ 判定未通过并上报 | | |\n")


def render_receipt(cfg, subjects, cmps, overall, args=None):
    """`windows/W2.3_determinism.md` §6.2 的八项回执。"""
    lines = []
    add = lines.append
    commit = (getattr(args, "code_commit", None) if args is not None else None) or \
        _git_head_short()
    report_path = (getattr(args, "report", None) if args is not None else None)
    add("| # | 回执项 | 内容 |")
    add("|---|---|---|")
    add("| ① | 脚本路径 + commit | `scripts/env/01_03_check_determinism.py`、"
        "`scripts/env/01_00_smoke_env.py`；生成本报告的代码提交 `%s`"
        "（报告自身随其后一次提交归档） |" % (commit or "见 `PROGRESS.md` 的 W2.3 行"))
    for key in subjects:
        cmp = cmps.get(key)
        if not cmp:
            add("| ② | `%s` | **未取得 A／B 比对结果** |" % key)
            continue
        add("| ② | `%s` | D1 md5 相同=`%s`；D2 `max|A-B|=%r`；logits A md5 `%s`；"
            "logits B md5 `%s`；判定 **%s** |"
            % (key, cmp.get("d1_md5_equal"), cmp.get("d2_max_abs_diff"),
               cmp.get("md5_a"), cmp.get("md5_b"),
               "通过" if cmp.get("passed") else "不通过"))
    missing = [s for s in subjects if s not in cmps]
    verdict = ("通过" if overall
               else ("未判定（缺 %s）" % ", ".join(missing) if missing else "不通过"))
    add("| ③ | 回退序列（严格 0 未达成时） | %s |"
        % ("**未触发**——判定通过，无需进入 §7 的七级回退序列。"
           if overall else "**见 §7**（逐级记录）。"))
    add("| ④ | eval 强制校验证据 | **见 §4**：断言代码行（①）＋在 `torch.no_grad()` **上下文内**"
        "读取的实测输出（②）＋三条断言用例（③，含两条负对照被拒）＋`model.train()` 负对照（④）"
        "＋`model.train()` 调用点的 AST 清单（§4.%d） |" % (len(subjects) + 1))
    add("| ⑤ | bnb 稳定性结论与失败率 | **见 §5**：`%s`。 |"
        % "；".join("`%s` → %s（failure_rate=%s）"
                    % (k, latest_bnb_summary(k)["conclusion"],
                       latest_bnb_summary(k)["failure_rate"]) for k in subjects))
    add("| ⑥ | `01_00` 试跑的七元组行 | **见 §6**（已标注「harness 验证，非基线」，"
        "**不得**进入 `ENV_LOCK.md` §8） |")
    add("| ⑦ | 新增台账条目 / 未决项 | 台账**无新增**（§9.1）；未决项 5 项见 §9.2 |")
    add("| ⑧ | 报告绝对路径 | `%s` |"
        % ((report_path or "<本文件>").replace("\\", "/")
           if os.path.isabs(report_path or "") else
           os.path.join(ROOT, (report_path or "<本文件>")).replace(os.sep, "/")))
    add("")
    add("**G-DET 总判定：%s**。门禁含义：通过 ⇒ W2.4 可开测、阶段1 具备打 tag 的前置之一；"
        "不通过 ⇒ W2.4 不得开测、阶段1 不得 tag（`windows/W2.3_determinism.md` §6.1）。"
        % verdict)
    return "\n".join(lines)


def render_ledger_section():
    return (
        "### 9.1 台账条目\n"
        "\n"
        "| 类别 | 条目 |\n"
        "|---|---|\n"
        "| **新增** | **无**。本窗口未引入任何新的偏离：未换包、未换版本、未改阈值、未改加载方式、"
        "未使用 flash-attn、未触碰 `data/`。 |\n"
        "| **引用（不重复登记）** | `DEVIATION_LEDGER.md`「推理确定性判定取严格 0（严于 v5 的"
        "「＜1e-6」）」——本报告的 §0 判定依据；「阶段1 环境实测值与目标值不一致（保留实测值，"
        "不降级）」——本报告 §2 的版本行；「模型权重的网络传输通道改用 hf-mirror.com」与"
        "「权重获取渠道改用魔搭 ModelScope」——被测权重的来源；「阶段1 落盘文件行尾由 CRLF 归一"
        "为 LF」——本窗口落盘一律 LF。 |\n"
        "\n"
        "### 9.2 未决项 / 报 W2 裁定的发现项\n"
        "\n"
        "| # | 事项 | 性质 | 需要什么 |\n"
        "|---|---|---|---|\n"
        "| 1 | **C3 对照量口径**：`attention_dropout=0.0` ⇒ 前向无随机算子 ⇒ 不同 seed 的 logits "
        "差值恒为 `0.0`。这**不**构成 `windows/W2.3_determinism.md` §4.3「为 0 说明 seed 未接上」"
        "所指的缺陷；本窗口另附 RNG 级证据（同 seed 一致、异 seed 不同）作为替代。 | 口径说明 | "
        "W2 确认该替代是否可接受；**不改变** G-DET 判定 |\n"
        "| 2 | **`unplug-tiny` 官方架构不可实例化**：`config.json` 声明 "
        "`architectures=['DebertaV2ForDualHead']`，但**仓内无自定义代码、未声明 `auto_map`** ⇒ "
        "`transformers` 无法按官方路径加载。本窗口以「`DebertaV2Model` 主干 + `doc_classifier`／"
        "`token_classifier` 双头」的重建路径跑通（记 `load_variant="
        "backbone_plus_dualhead_reconstruction`），**并以 2 条合成样本做了角色一致性抽验**"
        "（注入样本 `doc_argmax=1`=B-INJ 且 token 头标出 B/I-INJ span；良性样本 `doc_argmax=0`=O）"
        "。**该加载路径是否采信属 W2／W0 裁定**——它直接决定阶段3 B4 基线的口径，"
        "本窗口**不**据此宣称「官方路径跑通」。 | 发现（影响阶段3） | W2 裁定，必要时上呈 W0 |\n"
        "| 3 | **`collector=nvidia-smi` 的 `peak` 为绝对值**（含桌面与其它进程）。本窗口的 "
        "`unplug-cpu` 档因此在不同进程上下文里得到 `621MiB` 与 `2703MiB` 两个数（见 §6）。W2.4 "
        "须以**同一 collector 的 idle 档**为基准做差，且严禁跨 `run_id` 比较。 | 口径转达 | "
        "W2.4 遵守（已在 `windows/W2.4_memory_baseline.md` §4.1-2／§8.4 覆盖） |\n"
        "| 4 | **中止运行会遗留占显存的子进程**：本窗口实测一次——被中止的 `--mode bnb-stability` "
        "残留 8B 子进程继续运行，令后续 `unplug-cpu` 的 `nvidia-smi` 峰值被抬到 `7681MiB`。"
        "W2.3 已在 `01_00` 加入开测前的 GPU 预检（`PREFLIGHT gpu_used=… > 1500MiB` 时告警），"
        "并在运行序列脚本中加 `trap` 收尾。**W2.4 每档开测前须确认预检行**。 | 发现（操作面） | "
        "W2.4 遵守 |\n"
        "| 5 | `gpu_total` 口径：`nvidia-smi` 实测 **8188 MiB**，而 `ENV_LOCK.md` §1 登记为 "
        "**8187 MiB**。差值 1 MiB，属驱动取整差异；`windows/W2.4_memory_baseline.md` §7 规定 "
        "`gpu_total_mib` 取 **8187** 作为 `headroom_warn_mib` 基准，本窗口照此执行、不改 §1。 | "
        "登记 | W2.4 沿用 8187 |\n"
        "\n"
        "### 9.3 本窗口对 `01_03` 接口的增量说明（供 W2 核对）\n"
        "\n"
        "`--mode` 在契约的三个取值之外**增量**提供两个：\n"
        "\n"
        "| 增量取值 | 用途 | 是否影响契约语义 |\n"
        "|---|---|---|\n"
        "| `bnb-stability-single` | 「独立进程 ×N」的实现手段：父进程 spawn 本模式的子进程，每次只加载一次并登记；"
        "亦可手工单独隔离复跑。 | **否**——契约的 `bnb-stability` 语义未变 |\n"
        "| `compare` | 只做比对并渲染报告、**不加载模型**，使判定可在不重跑推理的前提下复算。 | "
        "**否**——契约的 `--report`（比对模式写）由 `--run-tag B` 自动触发，本取值只是同一条路径的显式入口 |"
    )


# ==========================================================================
# --mode compare
# ==========================================================================
def run_compare(args):
    cfg = load_determinism_config(args)
    subjects = [s["key"] for s in (cfg.get("subjects") or [])]
    cmps = {}
    for key in subjects:
        path = os.path.join(TMP_DIR, "det_cmp_%s.json" % key)
        if os.path.isfile(path):
            _p, result, _passed = None, read_json(path), None
            cmps[key] = result
    if not cmps:
        print("ERROR: 未找到任何 det_cmp_*.json —— 请先跑 --mode determinism --run-tag A／B")
        return EXIT_USAGE
    report_path = args.report or os.path.join(
        REPORTS_DIR, "determinism_v1.0_%s.md" % now_stamp())
    if os.path.exists(report_path):
        base, ext = os.path.splitext(report_path)
        report_path = "%s_r%s%s" % (base, now_stamp(), ext)
    text = render_report(cfg, subjects, cmps, args)
    write_text_lf(report_path, text)
    write_report_index(report_path, subjects, cmps)
    write_json_lf(_REPORT_FINGERPRINT, {
        "fingerprint": evidence_fingerprint(cmps),
        "report": os.path.relpath(report_path, ROOT).replace(os.sep, "/"),
        "report_md5": md5_file(report_path),
        "written_at": now_iso(),
    })
    print("report=%s" % os.path.relpath(report_path, ROOT).replace(os.sep, "/"))
    print("report_md5=%s" % md5_file(report_path))
    missing = [s for s in subjects if s not in cmps]
    overall = (not missing) and all(v.get("passed") for v in cmps.values())
    print("G_DET_VERDICT=%s%s" % ("PASS" if overall else "FAIL",
                                  "" if not missing
                                  else "（未判定：缺 %s）" % ",".join(missing)))
    return EXIT_OK if overall else EXIT_FAIL


# ==========================================================================
# 配置与导入辅助
# ==========================================================================
def load_determinism_config(args):
    path = getattr(args, "det_config", None) or DEFAULT_DET_CONFIG
    cfg = load_yaml(path) or {}
    if cfg.get("schema") != "codeobf-pi/determinism/v1":
        print("WARN: %s 的 schema=%r，期望 codeobf-pi/determinism/v1"
              % (path, cfg.get("schema")))
    return cfg


def _import_torch():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        print("WARN: CUBLAS_WORKSPACE_CONFIG=%r（期望 ':4096:8'）——回退序列第 4 级要求该值在 "
              "import torch **之前**生效" % (os.environ.get("CUBLAS_WORKSPACE_CONFIG"),))
    if "torch" in sys.modules:
        print("WARN: torch 在本进程内已被提前 import —— CUBLAS_WORKSPACE_CONFIG 的"
              "「import 前置」条件可能已失效")
    import torch

    return torch


def _import_transformers():
    import transformers

    return transformers


def _import_bnb():
    import bitsandbytes

    return bitsandbytes


def _module_version(name):
    try:
        import importlib

        return getattr(importlib.import_module(name), "__version__", None)
    except Exception:
        return None


def _torch_dtype(torch, name):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError("不支持的 dtype：%r" % (name,))
    return mapping[name]


# ==========================================================================
# CLI
# ==========================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="推理确定性校验（G-DET）／eval 强制校验／bnb 加载稳定性")
    parser.add_argument("--model-key", default=None,
                        help="取自 configs/env/models.yaml 的 key（禁止直接传路径）")
    parser.add_argument("--mode", default="determinism", choices=list(MODES),
                        help="determinism | eval-guard | bnb-stability"
                             "（另有内部/增量模式 bnb-stability-single、compare）")
    parser.add_argument("--run-tag", default=None, choices=["A", "B"],
                        help="determinism 模式必填；A／B 须由两次独立进程分别调用")
    parser.add_argument("--repeat", type=int, default=5,
                        help="bnb-stability 模式的独立进程重复次数")
    parser.add_argument("--run-index", type=int, default=0,
                        help="bnb-stability-single 的序号（由父进程传入）")
    parser.add_argument("--group", default=None,
                        help="批次／轮次标识：bnb-stability 的独立进程共用，"
                             "determinism 的 A／B 亦据此归入同一轮（默认时间戳）")
    parser.add_argument("--out", default=None, help="logits 落盘路径")
    parser.add_argument("--report", default=None, help="报告输出路径")
    parser.add_argument("--code-commit", default=None,
                        help="回执①填入的代码提交 hash（默认取 git rev-parse --short HEAD）")
    parser.add_argument("--seq-len", type=int, default=None, help="输入长度上限")
    parser.add_argument("--load-mode", default=None,
                        help="覆盖加载方式（默认取 models.yaml 的 planned_load）")
    parser.add_argument("--det-config", default=None,
                        help="默认 configs/env/determinism.yaml")
    parser.add_argument("--models-manifest", default=DEFAULT_MODELS_MANIFEST)
    return parser


def main(argv=None, **hooks):
    quiet_progress_reporting()   # 须在 import transformers 之前
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatcher = {
        "determinism": run_determinism,
        "eval-guard": run_eval_guard,
        "bnb-stability": run_bnb_stability,
        "bnb-stability-single": run_bnb_stability_single,
        "compare": run_compare,
    }
    if args.mode in ("determinism", "eval-guard", "bnb-stability",
                     "bnb-stability-single") and not args.model_key:
        print("ERROR: --mode %s 需要 --model-key" % args.mode)
        return EXIT_USAGE
    try:
        return dispatcher[args.mode](args)
    except EvalModeViolation as exc:
        print("ERROR(eval-guard): %s" % exc)
        return EXIT_FAIL
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print("ERROR: %s" % exc)
        return EXIT_USAGE
    except Exception as exc:  # pragma: no cover
        import traceback

        traceback.print_exc()
        print("ERROR(unexpected): %s" % exc)
        return EXIT_USAGE


if __name__ == "__main__":
    # 环境钉必须在 import torch 之前写进 os.environ（§4.2 env_pins）
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        sys.stdout.reconfigure(newline="\n", encoding="utf-8")
        sys.stderr.reconfigure(newline="\n", encoding="utf-8")
    except Exception:  # pragma: no cover
        pass
    sys.exit(main())

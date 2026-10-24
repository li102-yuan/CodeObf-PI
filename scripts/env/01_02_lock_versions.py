# -*- coding: utf-8 -*-
"""01_02_lock_versions.py — 版本锁定生成器（W2.1 交付物）

职责
----
1. **幂等生成块**：只替换 ``ENV_LOCK.md`` 中下列标记之间的内容，标记外**一个字节都不动**：

   ::

       <!-- BEGIN GENERATED: scripts/env/01_02_lock_versions.py -->
       ...
       <!-- END GENERATED: scripts/env/01_02_lock_versions.py -->

   生成时在 BEGIN 标记行内写入时间戳（``... @ <YYYY-MM-DD HH:MM:SS>``），便于审计「哪一次运行产出的」。
   标记缺失时**不得静默重建文件**，须报错退出（非零）且不改文件。

2. 生成 ``requirements/requirements-locked.txt``：pip 侧全量快照（``pip freeze`` 风格，按包名排序），
   顶部保留现有注释行，并追加两行（快照时间 + pip/conda 边界说明）。

3. 采集口径：pip 安装器记录（``importlib.metadata``）与 import 实测（``<mod>.__version__``）**双列**，
   并读 ``<env>/conda-meta/*.json`` 取 conda 记录；三者不一致时（双装情形）**同时记录**，不得只记其一。

职责边界（重要）
----------------
**本脚本的生成块不含模型 commit hash 节。** 模型 commit hash 由 W2.2 产出
（``configs/env/models.yaml`` + ``models/commit_hashes/``），``ENV_LOCK.md`` §5 归 W2.2 手写。

理由：若由本脚本渲染该节，则同一脚本跨 W2.1／W2.2 两个窗口写同一文件，且阶段1 为**严格串行**，
W2.1 首跑时 W2.2 尚未交付，必然产出一个立刻作废的渲染。

⇒ ``ENV_LOCK.md`` 首行占位描述中的「…、模型 commit hash」一句**不实现**；差异见
``reports/env/env_probe_v1.0_<ts>.md``「与占位描述的差异」小节。生成块**不含** conda 包混入
``requirements-locked.txt`` —— conda 侧信息一律进 ``ENV_LOCK.md``。

本脚本不得改动标记外的任何字节（含 W0 于 2026-09-27 的手工登记节）。
"""
import argparse
import contextlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime

MISSING = "MISSING"

GENERATOR = "scripts/env/01_02_lock_versions.py"
BEGIN_MARKER = "<!-- BEGIN GENERATED: %s -->" % GENERATOR
END_MARKER = "<!-- END GENERATED: %s -->" % GENERATOR
BEGIN_RE = re.compile(
    r"<!-- BEGIN GENERATED: " + re.escape(GENERATOR) + r"(?: @ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})? -->"
)

REQ_HEADER_HINT = "# 由 scripts/env/01_02_lock_versions.py 生成；锁定精确版本，勿手工编辑"

# §2 关键依赖精确版本：受影响于双装歧义的包全量登记
DEP_TABLE = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("numpy", "numpy"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("peft", "peft"),
    ("trl", "trl"),
    ("bitsandbytes", "bitsandbytes"),
    ("accelerate", "accelerate"),
    ("tokenizers", "tokenizers"),
    ("safetensors", "safetensors"),
    ("huggingface-hub", "huggingface_hub"),
    ("sentencepiece", "sentencepiece"),
    ("protobuf", "google.protobuf"),
    ("pyyaml", "yaml"),
    ("tqdm", "tqdm"),
    ("psutil", "psutil"),
    ("pyarrow", "pyarrow"),
    ("fsspec", "fsspec"),
    ("sympy", "sympy"),
    ("charset-normalizer", "charset_normalizer"),
    ("pillow", "PIL"),
    ("setuptools", "setuptools"),
    ("regex", "regex"),
    ("filelock", "filelock"),
    ("packaging", "packaging"),
    ("requests", "requests"),
    ("urllib3", "urllib3"),
    ("typer", "typer"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("einops", "einops"),
)


class MarkerMissingError(RuntimeError):
    """ENV_LOCK.md 缺少生成块标记 —— 不得静默重建文件。"""


# --------------------------------------------------------------------------
# 基础设施
# --------------------------------------------------------------------------
def project_root():
    """脚本位于 <root>/scripts/env/，因此根目录为两级父目录。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_text(path):
    """按 UTF-8 读取；文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path, text):
    """以 LF 行尾写入。Windows 下必须显式 newline='\\n'，否则会写成 CRLF。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _run(cmd, timeout=180):
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or ""


# --------------------------------------------------------------------------
# 幂等生成块
# --------------------------------------------------------------------------
def begin_marker_line(timestamp):
    """带时间戳的 BEGIN 标记行。"""
    return "<!-- BEGIN GENERATED: %s @ %s -->" % (GENERATOR, timestamp)


def find_block_span(text):
    """定位生成块的 (BEGIN 匹配, END 起始下标)；失败抛 :class:`MarkerMissingError`。

    END 必须**从 BEGIN 之后**开始找：``ENV_LOCK.md`` §0.2 会逐字展示两个标记作为文档说明，
    全局 ``find`` 会先命中 §0.2 里的示意行，从而误判「END 在 BEGIN 之前」。
    """
    begins = list(BEGIN_RE.finditer(text))
    if len(begins) != 1:
        raise MarkerMissingError(
            "BEGIN 标记数量异常：期望 1，实际 %d（文件未改动）" % len(begins)
        )
    end_idx = text.find(END_MARKER, begins[0].end())
    if end_idx < 0:
        raise MarkerMissingError("BEGIN 标记之后未找到 END 标记（文件未改动）")
    return begins[0], end_idx


def replace_generated_block(text, body, timestamp):
    """把 ``text`` 中 BEGIN/END 标记之间的内容替换为 ``body``。

    标记外内容**逐字节不变**（含中文与既有手写节）。
    标记缺失或重复时抛 :class:`MarkerMissingError`，调用方须报错退出且不改文件。
    """
    begin, end_idx = find_block_span(text)
    head = text[: begin.start()]
    tail = text[end_idx + len(END_MARKER):]
    new_block = "%s\n%s\n%s" % (begin_marker_line(timestamp), body.strip("\n"), END_MARKER)
    return head + new_block + tail


# --------------------------------------------------------------------------
# 采集
# --------------------------------------------------------------------------
def conda_meta_versions(python_exe=None):
    """读 conda-meta/*.json，返回 {包名小写: 版本}。"""
    exe = python_exe or sys.executable
    meta_dir = os.path.join(os.path.dirname(exe), "conda-meta")
    out = {}
    try:
        names = os.listdir(meta_dir)
    except Exception:
        return out
    for fn in names:
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(meta_dir, fn), "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:
            continue
        if rec.get("name") and rec.get("version"):
            out[rec["name"].lower()] = rec["version"]
    return out


def pip_meta_version(dist_name):
    try:
        import importlib.metadata as md

        return md.version(dist_name)
    except Exception:
        return None


def import_version(module_name):
    """import 实测版本；导入过程 stdout/stderr 一并吞掉，避免污染脚本输出。"""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            __import__(module_name)
    except Exception:
        return None, None
    mod = sys.modules.get(module_name)
    if mod is None:
        return None, None
    for attr in ("__version__", "version", "VERSION"):
        val = getattr(mod, attr, None)
        if isinstance(val, str) and val:
            return val, getattr(mod, "__file__", None)
    return None, getattr(mod, "__file__", None)


def collect_deps():
    """采集 DEP_TABLE 的 pip 记录 / import 实测 / conda 记录 三元组。"""
    conda_meta = conda_meta_versions()
    rows = []
    for dist, module in DEP_TABLE:
        pip_v = pip_meta_version(dist)
        imp_v, imp_file = import_version(module)
        conda_v = conda_meta.get(dist.lower())
        if pip_v is None and imp_v is None and conda_v is None:
            verdict = "未安装"
        elif pip_v is not None and imp_v is not None and pip_v != imp_v:
            verdict = "双装：以 import 实测生效"
        elif conda_v is not None and pip_v is not None and conda_v != pip_v:
            verdict = "pip/conda 版本不一致"
        elif conda_v is not None and imp_v is not None and conda_v != imp_v:
            verdict = "conda/import 版本不一致"
        else:
            verdict = "一致"
        rows.append(
            {
                "dist": dist,
                "pip": pip_v or "—",
                "import": imp_v or "—",
                "conda": conda_v or "—",
                "verdict": verdict,
                "file": imp_file or "—",
            }
        )
    return rows


def collect_system():
    """采集 §1 系统与算力。重型导入（torch）放在函数内并 try/except。"""
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_exe": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV")
        or os.path.basename(os.path.dirname(sys.executable)),
        "conda_env_root": os.path.dirname(sys.executable),
        "cuda": "NA",
        "cudnn": "NA",
        "cuda_avail": "False",
        "gpu": "NA",
        "vram": "NA",
        "driver": "NA",
        "torch_raw": MISSING,
    }
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout=30)
    if out:
        info["driver"] = out.strip().splitlines()[0]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import torch

        info["torch_raw"] = getattr(torch, "__version__", MISSING)
        info["cuda"] = getattr(getattr(torch, "version", None), "cuda", None) or "NA"
        try:
            cudnn = torch.backends.cudnn.version()
            info["cudnn"] = str(cudnn) if cudnn else "NA"
        except Exception:
            pass
        try:
            info["cuda_avail"] = str(bool(torch.cuda.is_available()))
        except Exception:
            pass
        if info["cuda_avail"] == "True":
            try:
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram"] = str(int(torch.cuda.get_device_properties(0).total_memory) // 1048576)
            except Exception:
                pass
    except Exception as exc:
        info["torch_raw"] = "%s: %s" % (type(exc).__name__, exc)
    if info["gpu"] == "NA":
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], timeout=30)
        if out:
            parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
            if len(parts) >= 2:
                info["gpu"], info["vram"] = parts[0], parts[1]
    return info


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def render_generated_block(sysinfo, dep_rows):
    """渲染 §1 系统与算力 + §2 关键依赖精确版本（不含标记行）。"""
    L = []
    L.append("## §1 系统与算力")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| OS | %s %s |" % (sysinfo["os"], sysinfo["os_release"]))
    L.append("| 平台 | %s |" % sysinfo["os_platform"])
    L.append("| 架构 | %s |" % sysinfo["machine"])
    L.append("| Python | %s |" % sysinfo["python"])
    L.append("| Python 可执行路径 | `%s` |" % sysinfo["python_exe"])
    L.append("| conda 环境 | %s |" % sysinfo["conda_env"])
    L.append("| conda 环境根 | `%s` |" % sysinfo["conda_env_root"])
    L.append("| CUDA 版本 | %s |" % sysinfo["cuda"])
    L.append("| cuDNN 版本 | %s |" % sysinfo["cudnn"])
    L.append("| torch 构建 | %s |" % sysinfo["torch_raw"])
    L.append("| `torch.cuda.is_available()` | %s |" % sysinfo["cuda_avail"])
    L.append("| GPU 型号 | %s |" % sysinfo["gpu"])
    L.append("| 显存总量 | %s MiB |" % sysinfo["vram"])
    L.append("| 驱动版本 | %s |" % sysinfo["driver"])
    L.append("| 采集时间 | %s |" % sysinfo["timestamp"])
    L.append("")
    L.append("## §2 关键依赖精确版本")
    L.append("")
    L.append(
        "> 三列口径：**pip 安装器记录** = `importlib.metadata.version()`；"
        "**import 实测** = `import <mod>; <mod>.__version__`；"
        "**conda 记录** = `<env>/conda-meta/*.json`。"
    )
    L.append("> 三者不一致时（双装情形）**同时记录**，以 import 实测为实际生效版本；"
             "逐条差异见本节末「双装／不一致登记」。")
    L.append("")
    L.append("| 依赖 | pip 安装器记录 | import 实测 | conda 记录 | 判定 |")
    L.append("|---|---|---|---|---|")
    for r in dep_rows:
        L.append(
            "| %s | %s | %s | %s | %s |"
            % (r["dist"], r["pip"], r["import"], r["conda"], r["verdict"])
        )
    L.append("")
    mism = [r for r in dep_rows if r["verdict"] != "一致" and r["verdict"] != "未安装"]
    L.append("### 双装／不一致登记")
    L.append("")
    if not mism:
        L.append("- 无。三个口径对所有登记包一致。")
    else:
        for r in mism:
            L.append(
                "- **%s**：%s。pip 记录 `%s`，import 实测 `%s`（`%s`），conda 记录 `%s`。"
                % (r["dist"], r["verdict"], r["pip"], r["import"], r["file"], r["conda"])
            )
    L.append("")
    notinst = [r["dist"] for r in dep_rows if r["verdict"] == "未安装"]
    L.append("### 未安装登记")
    L.append("")
    L.append(
        "- 未安装：%s。" % ("、".join(notinst) if notinst else "无")
    )
    L.append(
        "- 说明：`scipy`／`scikit-learn`／`einops` 属 `windows/W2.1_env_lock.md` §4.2-3 的「按需装」项，"
        "本阶段未使用该三者，故不安装、无需登记用途。"
    )
    L.append("")
    return "\n".join(L)


def freeze_line_name(line):
    """从 pip freeze 行提取发行名，用于排序。"""
    s = line.strip()
    if s.startswith("-e "):
        s = s[3:].strip()
    elif s.startswith("--editable "):
        s = s[len("--editable "):].strip()
    if "#egg=" in s:
        return s.split("#egg=", 1)[1].split("&")[0].strip()
    if "==" in s:
        return s.split("==", 1)[0].strip()
    if " @ " in s:
        return s.split(" @ ", 1)[0].strip()
    return s


def top_comment_lines(text):
    """取文件顶部连续注释行（遇到首个非注释行即停）。"""
    out = []
    if not text:
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            out.append(line.rstrip())
        elif line.strip() == "":
            continue
        else:
            break
    return out


def pip_freeze_lines():
    """获取 pip freeze 行；失败则返回空列表。"""
    out = _run([sys.executable, "-m", "pip", "freeze"], timeout=300)
    if not out:
        return []
    return [ln.rstrip() for ln in out.splitlines() if ln.strip()]


def build_requirements_locked(prev_text, freeze_lines, timestamp):
    """构造 requirements-locked.txt 全文。

    - 顶部保留既有注释行（缺失时用默认头）
    - 追加两行：快照时间 + pip/conda 边界说明
    - 包行按发行名（不区分大小写）排序；不混入 conda 包
    """
    comments = top_comment_lines(prev_text)
    if not comments:
        comments = [REQ_HEADER_HINT]
    if REQ_HEADER_HINT not in comments:
        comments.insert(0, REQ_HEADER_HINT)
    comments = comments + [
        "# 快照时间：%s" % timestamp,
        "# 本文件为 pip 侧快照；conda 侧见 ENV_LOCK.md §4；pip/conda 双装清单见 ENV_LOCK.md §3",
    ]
    pkg_lines = [ln for ln in freeze_lines if ln.strip() and not ln.strip().startswith("#")]
    pkg_lines.sort(key=lambda ln: freeze_line_name(ln).lower())
    return "\n".join(comments + pkg_lines) + "\n"


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="版本锁定生成器：ENV_LOCK.md 生成块 + requirements-locked.txt")
    ap.add_argument("--env-lock", default=None, help="ENV_LOCK.md 路径")
    ap.add_argument("--requirements", default=None, help="requirements-locked.txt 路径")
    ap.add_argument("--log", default=None, help="日志路径")
    ap.add_argument("--dry-run", action="store_true", help="只打印将发生的变更，不写文件")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = project_root()
    env_lock_path = args.env_lock or os.path.join(root, "ENV_LOCK.md")
    req_path = args.requirements or os.path.join(
        root, "requirements", "requirements-locked.txt"
    )
    ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.log or os.path.join(
        root, "logs", "env", "01_02_lock_versions_%s.log" % ts_file
    )

    log = []
    log.append("# 01_02_lock_versions.py — 版本锁定生成日志")
    log.append("")
    log.append("- 窗口：W2.1（阶段1 环境版本锁定，权限 A2）")
    log.append("- 时间：%s" % ts_human)
    log.append("- ENV_LOCK.md：`%s`" % env_lock_path)
    log.append("- requirements-locked.txt：`%s`" % req_path)
    log.append("- dry-run：%s" % args.dry_run)
    log.append("")

    prev = read_text(env_lock_path)
    if prev is None:
        log.append("FAIL ENV_LOCK.md 不存在：%s" % env_lock_path)
        _emit_log(log_path, log, args.dry_run)
        sys.stderr.write("FAIL ENV_LOCK.md 不存在：%s\n" % env_lock_path)
        return 1

    # 先校验标记，再采集：标记缺失属前置条件失败，须**快速失败且零副作用**
    # （不 import 重型库、不改文件），依 windows/W2.1_env_lock.md §4.6-1。
    try:
        find_block_span(prev)
    except MarkerMissingError as exc:
        log.append("FAIL 标记缺失，文件未改动：%s" % exc)
        _emit_log(log_path, log, args.dry_run)
        sys.stderr.write(
            "FAIL ENV_LOCK.md 生成块标记缺失，文件未改动：%s\n"
            "（依 windows/W2.1_env_lock.md §4.6-1，不得静默重建文件）\n" % exc
        )
        return 3

    sysinfo = collect_system()
    dep_rows = collect_deps()
    body = render_generated_block(sysinfo, dep_rows)
    log.append("## 生成块渲染")
    log.append("")
    log.append("```markdown")
    log.extend(body.splitlines())
    log.append("```")
    log.append("")

    new_env_lock = replace_generated_block(prev, body, ts_human)

    req_prev = read_text(req_path)
    freeze_lines = pip_freeze_lines()
    if not freeze_lines:
        log.append("WARN pip freeze 为空，requirements-locked.txt 不生成")
        req_text = None
    else:
        req_text = build_requirements_locked(req_prev, freeze_lines, ts_human)
        log.append("## requirements-locked.txt")
        log.append("")
        log.append("- pip freeze 行数：%d" % len(freeze_lines))
        log.append("- 输出行数：%d" % len(req_text.splitlines()))
        log.append("")

    if args.dry_run:
        log.append("DRY-RUN：未写入任何文件。")
        _emit_log(log_path, log, args.dry_run)
        return 0

    try:
        write_text(env_lock_path, new_env_lock)
        if req_text is not None:
            write_text(req_path, req_text)
    except Exception as exc:
        log.append("FAIL 写入失败：%s: %s" % (type(exc).__name__, exc))
        _emit_log(log_path, log, args.dry_run)
        sys.stderr.write("FAIL 写入失败：%s\n" % exc)
        return 1

    log.append("OK 已写入 ENV_LOCK.md 生成块（标记外字节未改动）")
    if req_text is not None:
        log.append("OK 已写入 requirements-locked.txt")
    _emit_log(log_path, log, args.dry_run)
    print("OK 01_02_lock_versions: ENV_LOCK.md 生成块已更新；requirements-locked.txt 已生成")
    return 0


def _emit_log(log_path, lines, dry_run):
    text = "\n".join(lines) + "\n"
    if dry_run:
        sys.stdout.write(text)
        return
    try:
        write_text(log_path, text)
    except Exception as exc:
        sys.stderr.write("WARN 日志写入失败：%s\n" % exc)


if __name__ == "__main__":
    sys.exit(main())

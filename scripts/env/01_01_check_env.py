# -*- coding: utf-8 -*-
"""01_01_check_env.py — 环境探测（W2.1 交付物）

用途
----
探测 OS、Python、conda 环境、CUDA、GPU 与显存、驱动、关键依赖版本与归属（conda／pip），
按固定结论行输出到 stdout，并把详细报告落盘 ``logs/env/``。

接口契约（``windows/W2.1_env_lock.md`` §4.3）
---------------------------------------------
1. **默认行为必须 exit 0**（报告型）：即使某些依赖缺失，也只打印 ``MISSING`` 与结论行。
   理由：``Makefile`` 的 ``check`` 目标要求 ``exit=0`` 作为治理门禁；环境探测失败不应使治理检查变红。
2. ``--strict``：探测到硬失败（CUDA 不可用 / torch import 失败）时 ``exit 2``。
   ``make check`` 不使用该开关。
3. stdout 结论行格式固定（缺失字段填 ``MISSING``，不得抛异常中断）：

   ::

       ENV_PROBE python=<x.y.z> cuda=<12.x|NA> gpu=<name|NA> vram=<N>MiB torch=<ver> \
cuda_avail=<True|False> numpy=<ver> transformers=<ver> peft=<ver> bnb=<ver> trl=<ver> \
datasets=<ver> accelerate=<ver>

4. 顶部**不** import 重型库（torch／transformers 等）；重型导入一律放函数内并逐个 ``try/except``。
   本模块顶部只依赖标准库。
5. 探测项至少含：OS 与版本、Python 版本与可执行路径、conda 环境名、CUDA 版本、GPU 型号与显存总量、
   驱动版本、torch／torchvision／torchaudio（记录归属：conda 或 pip）、numpy 解析版本、
   transformers／datasets／peft／trl／bitsandbytes／accelerate 版本、``make`` 版本与绝对路径。
6. 结果同时落盘 ``logs/env/01_01_check_env_<YYYYMMDD-HHMMSS>.log``，路径可被 ``--log`` 覆盖。
"""
import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime


@contextlib.contextmanager
def _quiet():
    """吞掉被导入库写到 stdout/stderr 的告警，保证 make check 输出可解析。"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield

MISSING = "MISSING"

# ENV_PROBE 结论行字段顺序固定（契约 §4.3-3）
PROBE_FIELDS = (
    "python",
    "cuda",
    "gpu",
    "vram",
    "torch",
    "cuda_avail",
    "numpy",
    "transformers",
    "peft",
    "bnb",
    "trl",
    "datasets",
    "accelerate",
)

# 结论行字段名 → (pip 发行名, import 模块名)
PROBE_LIBS = {
    "torch": ("torch", "torch"),
    "numpy": ("numpy", "numpy"),
    "transformers": ("transformers", "transformers"),
    "peft": ("peft", "peft"),
    "bnb": ("bitsandbytes", "bitsandbytes"),
    "trl": ("trl", "trl"),
    "datasets": ("datasets", "datasets"),
    "accelerate": ("accelerate", "accelerate"),
}

# 报告里额外登记归属的包（§4.3-5）
ORIGIN_PKGS = ("torch", "torchvision", "torchaudio", "numpy")


# --------------------------------------------------------------------------
# 基础设施探测（全部为轻量调用，不 import 重型库）
# --------------------------------------------------------------------------
def project_root():
    """脚本位于 <root>/scripts/env/，因此根目录为两级父目录。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(cmd, timeout=30):
    """执行外部命令，成功返回 stdout（strip），任何异常返回 None。"""
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, **kwargs
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def conda_meta_versions(python_exe=None):
    """读取 conda 环境内 ``conda-meta/*.json``，返回 {包名小写: 版本}。

    这是判定「conda 归属」的权威依据：conda list 里的 ``pypi_0 pypi`` 行是 conda
    对 pip 安装物的镜像记录，不能用来判定归属，故直接读 conda 自身的元数据。
    """
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
        name = rec.get("name")
        ver = rec.get("version")
        if name and ver:
            out[name.lower()] = ver
    return out


def pip_dist_version(dist_name):
    """pip 侧安装器记录版本（importlib.metadata）；未安装返回 None。"""
    try:
        import importlib.metadata as md

        return md.version(dist_name)
    except Exception:
        return None


def pip_dist_path(dist_name):
    """pip 侧 dist-info 所在目录；未安装返回 None。"""
    try:
        import importlib.metadata as md

        return str(getattr(md.distribution(dist_name), "_path", "") or "") or None
    except Exception:
        return None


def package_origin(dist_name, conda_meta):
    """判定包归属：conda / pip / conda+pip / NA。"""
    in_conda = dist_name.lower() in conda_meta
    in_pip = pip_dist_version(dist_name) is not None
    if in_conda and in_pip:
        return "conda+pip"
    if in_conda:
        return "conda"
    if in_pip:
        return "pip"
    return "NA"


def import_version(module_name):
    """import 实测版本；失败或模块无 __version__ 返回 None。

    导入期间 stdout/stderr 被吞掉：重型库常在 import 时打印告警，会污染
    ``make check`` 的结论行解析。
    """
    try:
        with _quiet():
            __import__(module_name)
    except Exception:
        return None
    mod = sys.modules.get(module_name)
    if mod is None:
        return None
    for attr in ("__version__", "version", "VERSION"):
        val = getattr(mod, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


# --------------------------------------------------------------------------
# 探测主体
# --------------------------------------------------------------------------
def _torch_family(conda_meta):
    """探测 torch 家族并与 CUDA 信息一并返回。"""
    info = {
        "torch": MISSING,
        "torch_raw": MISSING,
        "cuda": "NA",
        "cudnn": "NA",
        "cuda_avail": "False",
        "gpu": "NA",
        "vram": "NA",
        "gpu_count": "0",
        "torchvision": MISSING,
        "torchaudio": MISSING,
        "torch_origin": "NA",
        "torchvision_origin": "NA",
        "torchaudio_origin": "NA",
        "torch_error": None,
    }
    for key, dist in (("torch", "torch"), ("torchvision", "torchvision"),
                      ("torchaudio", "torchaudio")):
        ver = pip_dist_version(dist) or import_version(dist)
        if ver:
            info[key] = ver
        info[key + "_origin"] = package_origin(dist, conda_meta)

    try:
        with _quiet():
            import torch
    except Exception as exc:  # 硬失败之一
        info["torch_error"] = "%s: %s" % (type(exc).__name__, exc)
        return info

    info["torch"] = getattr(torch, "__version__", MISSING)
    info["torch_raw"] = info["torch"]
    info["cuda"] = getattr(getattr(torch, "version", None), "cuda", None) or "NA"
    try:
        cudnn = torch.backends.cudnn.version()
        info["cudnn"] = str(cudnn) if cudnn else "NA"
    except Exception:
        pass
    try:
        info["cuda_avail"] = str(bool(torch.cuda.is_available()))
    except Exception:
        info["cuda_avail"] = "False"
    if info["cuda_avail"] == "True":
        try:
            info["gpu_count"] = str(torch.cuda.device_count())
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram"] = str(int(props.total_memory) // 1048576)
        except Exception:
            pass
    return info


def _gpu_fallback(info):
    """torch 不可用时，退化到 nvidia-smi 取 GPU 与驱动信息。"""
    if info.get("cuda_avail") != "True":
        out = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if out:
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 2:
                info["gpu"] = parts[0]
                info["vram"] = parts[1]
    return info


def driver_version():
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if out:
        return out.splitlines()[0].strip()
    return "NA"


def make_info():
    path = shutil.which("make")
    version = "NA"
    if path:
        out = _run([path, "--version"], timeout=20)
        if out:
            version = out.splitlines()[0].strip()
    return {"make_path": path or "NA", "make_version": version}


def conda_env_name():
    name = os.environ.get("CONDA_DEFAULT_ENV")
    if name:
        return name
    return os.path.basename(os.path.dirname(sys.executable)) or "NA"


def probe():
    """执行全部探测，返回扁平 dict。任何单点失败都不抛异常。"""
    conda_meta = conda_meta_versions()
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "os_platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_exe": sys.executable,
        "python_impl": platform.python_implementation(),
        "python_build": " ".join(platform.python_build()),
        "conda_env": conda_env_name(),
        "conda_env_root": os.path.dirname(sys.executable),
        "conda_pkg_count": str(len(conda_meta)),
        "driver": driver_version(),
        "conda": _run(["conda", "--version"]) or MISSING,
    }
    info.update(_torch_family(conda_meta))
    info = _gpu_fallback(info)
    info.update(make_info())

    for field, (dist, module) in PROBE_LIBS.items():
        if field == "torch":
            continue  # 已由 _torch_family 处理
        info[field] = pip_dist_version(dist) or import_version(module) or MISSING

    for pkg in ORIGIN_PKGS:
        info["%s_origin" % pkg] = package_origin(pkg, conda_meta)

    # numpy 实际解析来源（双装歧义时唯一可信证据）
    info["numpy_file"] = MISSING
    info["numpy_import"] = MISSING
    try:
        with _quiet():
            import numpy

        info["numpy_import"] = numpy.__version__
        info["numpy_file"] = numpy.__file__
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def format_probe_line(info):
    """渲染固定格式的 ENV_PROBE 结论行。

    ``vram`` 按契约 §4.3-3 的 ``vram=<N>MiB`` 字面渲染 —— ``<N>`` 是占位符，``MiB`` 是字面单位，
    与 ``cuda=<12.x|NA>``／``gpu=<name|NA>`` 的占位符写法一致。取不到显存时填 ``NA``。
    """
    parts = []
    for field in PROBE_FIELDS:
        val = info.get(field, MISSING)
        if val in (None, ""):
            val = MISSING
        if field == "vram":
            val = "%sMiB" % val if str(val).isdigit() else "NA"
        parts.append("%s=%s" % (field, val))
    return "ENV_PROBE " + " ".join(parts)


def render_report(info, line, script_path):
    """渲染详细报告正文（写入日志文件）。"""
    L = []
    L.append("# 01_01_check_env.py — 环境探测报告")
    L.append("")
    L.append("- 窗口：W2.1（阶段1 环境版本锁定，权限 A2）")
    L.append("- 脚本：`%s`" % script_path)
    L.append("- 时间：%s" % info.get("timestamp", MISSING))
    L.append("")
    L.append("## 1. OS 与平台")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| system | %s |" % info.get("os_system", MISSING))
    L.append("| release | %s |" % info.get("os_release", MISSING))
    L.append("| version | %s |" % info.get("os_version", MISSING))
    L.append("| platform | %s |" % info.get("os_platform", MISSING))
    L.append("| machine | %s |" % info.get("machine", MISSING))
    L.append("")
    L.append("## 2. Python")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| version | %s |" % info.get("python", MISSING))
    L.append("| executable | `%s` |" % info.get("python_exe", MISSING))
    L.append("| implementation | %s |" % info.get("python_impl", MISSING))
    L.append("| build | %s |" % info.get("python_build", MISSING))
    L.append("")
    L.append("## 3. conda 环境")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 环境名 | %s |" % info.get("conda_env", MISSING))
    L.append("| 环境根 | `%s` |" % info.get("conda_env_root", MISSING))
    L.append("| conda 版本 | %s |" % info.get("conda", MISSING))
    L.append("| conda-meta 包数 | %s |" % info.get("conda_pkg_count", MISSING))
    L.append("")
    L.append("## 4. CUDA 与算力")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| CUDA 版本（torch.version.cuda） | %s |" % info.get("cuda", "NA"))
    L.append("| cuDNN 版本 | %s |" % info.get("cudnn", "NA"))
    L.append("| cuda.is_available() | %s |" % info.get("cuda_avail", "False"))
    L.append("| GPU 数量 | %s |" % info.get("gpu_count", "0"))
    L.append("| GPU 型号 | %s |" % info.get("gpu", "NA"))
    L.append("| 显存总量 | %s MiB |" % info.get("vram", "NA"))
    L.append("| 驱动版本 | %s |" % info.get("driver", "NA"))
    L.append("")
    L.append("## 5. torch 家族（含归属）")
    L.append("")
    L.append("| 包 | 版本 | 归属 |")
    L.append("|---|---|---|")
    for pkg in ("torch", "torchvision", "torchaudio"):
        L.append("| %s | %s | %s |" % (pkg, info.get(pkg, MISSING), info.get(pkg + "_origin", "NA")))
    if info.get("torch_error"):
        L.append("")
        L.append("> torch import 失败：`%s`" % info["torch_error"])
    L.append("")
    L.append("## 6. 关键依赖版本")
    L.append("")
    L.append("| 字段 | 版本 |")
    L.append("|---|---|")
    for field in PROBE_FIELDS:
        if field in ("python", "cuda", "gpu", "vram", "cuda_avail"):
            continue
        L.append("| %s | %s |" % (field, info.get(field, MISSING)))
    L.append("")
    L.append("## 7. numpy 实际解析")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| import 实测版本 | %s |" % info.get("numpy_import", MISSING))
    L.append("| `numpy.__file__` | `%s` |" % info.get("numpy_file", MISSING))
    L.append("| pip 安装器记录 | %s |" % (pip_dist_version("numpy") or MISSING))
    L.append("| 归属 | %s |" % info.get("numpy_origin", "NA"))
    L.append("")
    L.append("## 8. make")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 绝对路径 | `%s` |" % info.get("make_path", "NA"))
    L.append("| 版本 | %s |" % info.get("make_version", "NA"))
    L.append("")
    L.append("## 9. 结论行")
    L.append("")
    L.append("```text")
    L.append(line)
    L.append("```")
    L.append("")
    return "\n".join(L)


def write_log(path, text):
    """以 LF 行尾写日志。Windows 下必须显式 newline='\\n'，否则会写成 CRLF。"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return True, None
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="环境探测：OS / Python / conda / CUDA / GPU / 依赖版本"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="硬失败（torch import 失败 / CUDA 不可用）时 exit 2；make check 不使用",
    )
    ap.add_argument(
        "--log",
        default=None,
        help="日志路径（默认 logs/env/01_01_check_env_<YYYYMMDD-HHMMSS>.log）",
    )
    return ap


def main(argv=None, collector=probe):
    """返回进程退出码。默认恒为 0（报告型）；--strict 下硬失败返回 2。"""
    args = build_parser().parse_args(argv)
    info = collector()

    script_path = os.path.abspath(__file__)
    line = format_probe_line(info)
    report = render_report(info, line, script_path)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.log or os.path.join(
        project_root(), "logs", "env", "01_01_check_env_%s.log" % ts
    )
    ok, err = write_log(log_path, report)
    if err:
        sys.stderr.write("WARN 日志写入失败：%s\n" % err)

    print(line)

    if args.strict:
        hard_fail = (info.get("torch", MISSING) == MISSING) or (
            str(info.get("cuda_avail", "False")) != "True"
        )
        if hard_fail:
            sys.stderr.write(
                "STRICT 硬失败：torch=%s cuda_avail=%s\n"
                % (info.get("torch", MISSING), info.get("cuda_avail", "False"))
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

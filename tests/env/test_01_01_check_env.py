# -*- coding: utf-8 -*-
"""tests/env/test_01_01_check_env.py — ``scripts/env/01_01_check_env.py`` 的单元测试

覆盖 ``windows/W2.1_env_lock.md`` §6.3-3：输出格式（字段齐全、缺失依赖填 ``MISSING``、默认 ``exit 0``）。

约束：不访问 ``data/``；不加载模型；不依赖网络；不改变环境。
被测脚本以文件路径动态加载（文件名以数字开头，不能用 ``import`` 语句）。
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "env", "01_01_check_env.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("mod_01_01_check_env", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod_01_01_check_env"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


def _full_info(**over):
    """构造字段齐备的探测结果。"""
    info = {field: "1.0" for field in M.PROBE_FIELDS}
    info.update(
        {
            "python": "3.12.13",
            "cuda": "12.4",
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "vram": "8187",
            "torch": "2.5.1",
            "cuda_avail": "True",
        }
    )
    info.update(over)
    return info


# ---------------------------------------------------------------- 结论行格式
def test_probe_line_has_every_field_in_fixed_order():
    line = M.format_probe_line(_full_info())
    assert line.startswith("ENV_PROBE ")
    tokens = line.split(" ")[1:]
    # gpu 值含空格，故按「已知字段名 = 」切分后核对字段出现顺序
    positions = []
    for field in M.PROBE_FIELDS:
        marker = "%s=" % field
        idx = line.find(marker)
        assert idx > 0, "字段缺失：%s" % field
        positions.append(idx)
    assert positions == sorted(positions), "字段顺序与 PROBE_FIELDS 不一致"
    assert len(tokens) >= len(M.PROBE_FIELDS)


def test_probe_line_fields_match_contract():
    """契约 §4.3-3 明列 13 个字段，顺序固定。"""
    assert M.PROBE_FIELDS == (
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


def test_missing_dependency_renders_missing():
    info = _full_info(transformers=M.MISSING, peft=M.MISSING, bnb=M.MISSING)
    line = M.format_probe_line(info)
    assert "transformers=MISSING" in line
    assert "peft=MISSING" in line
    assert "bnb=MISSING" in line


def test_none_and_empty_values_degrade_to_missing():
    info = _full_info(trl=None, datasets="")
    line = M.format_probe_line(info)
    assert "trl=MISSING" in line
    assert "datasets=MISSING" in line


def test_vram_renders_mib_literal():
    """契约写法 ``vram=<N>MiB`` —— ``<N>`` 是占位符、``MiB`` 是字面单位。"""
    assert "vram=8187MiB" in M.format_probe_line(_full_info(vram="8187"))


def test_vram_na_when_unavailable():
    assert "vram=NA" in M.format_probe_line(_full_info(vram="NA"))


def test_cuda_na_when_unavailable():
    line = M.format_probe_line(_full_info(cuda="NA", gpu="NA", vram="NA"))
    assert "cuda=NA" in line
    assert "gpu=NA" in line


# ---------------------------------------------------------------- 退出码
def test_default_exit_zero_when_everything_missing(tmp_path):
    """§6.3-3：探测到全缺失仍必须 exit 0（make check 依赖此性质）。"""
    broken = {field: M.MISSING for field in M.PROBE_FIELDS}
    broken["cuda_avail"] = "False"
    rc = M.main(
        ["--log", str(tmp_path / "probe.log")], collector=lambda: broken
    )
    assert rc == 0


def test_strict_exit_two_on_hard_failure(tmp_path):
    broken = {field: M.MISSING for field in M.PROBE_FIELDS}
    broken["cuda_avail"] = "False"
    rc = M.main(
        ["--strict", "--log", str(tmp_path / "probe.log")], collector=lambda: broken
    )
    assert rc == 2


def test_strict_exit_zero_when_healthy(tmp_path):
    rc = M.main(
        ["--strict", "--log", str(tmp_path / "probe.log")],
        collector=lambda: _full_info(),
    )
    assert rc == 0


def test_strict_exit_two_when_cuda_unavailable_but_torch_present(tmp_path):
    info = _full_info(cuda="NA", gpu="NA", vram="NA", cuda_avail="False")
    rc = M.main(
        ["--strict", "--log", str(tmp_path / "probe.log")], collector=lambda: info
    )
    assert rc == 2


# ---------------------------------------------------------------- 日志落盘
def test_log_written_with_lf_and_contains_probe_line(tmp_path):
    log_path = tmp_path / "nested" / "probe.log"
    rc = M.main(["--log", str(log_path)], collector=lambda: _full_info())
    assert rc == 0
    raw = log_path.read_bytes()
    assert b"\r" not in raw, "日志不得写 CRLF（AGENTS.md §5 落盘一律 LF）"
    text = raw.decode("utf-8")
    assert M.format_probe_line(_full_info()) in text
    assert "## 9. 结论行" in text


def test_log_records_numpy_resolution(tmp_path):
    log_path = tmp_path / "probe.log"
    info = _full_info(numpy_file=r"C:\x\numpy\__init__.py", numpy_import="2.3.1")
    M.main(["--log", str(log_path)], collector=lambda: info)
    text = log_path.read_text(encoding="utf-8")
    assert "numpy.__file__" in text
    assert "2.3.1" in text


# ---------------------------------------------------------------- 辅助函数
def test_module_uses_no_heavy_import_at_top_level():
    """契约 §4.3-4：顶部不得 import 重型库。"""
    source = open(SCRIPT, "r", encoding="utf-8").read()
    head = source.split("def ", 1)[0]
    for heavy in ("import torch", "import numpy", "import transformers"):
        assert heavy not in head, "顶部出现重型导入：%s" % heavy


def test_quiet_context_swallows_output(capsys):
    with M._quiet():
        print("这一行不应出现")
    assert capsys.readouterr().out == ""

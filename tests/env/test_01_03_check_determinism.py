# -*- coding: utf-8 -*-
"""tests/env/test_01_03_check_determinism.py — ``scripts/env/01_03_check_determinism.py`` 的单元测试

覆盖 ``windows/W2.3_determinism.md`` §6.3 的第 1／2／4 项：

1. D1／D2 比对函数 —— md5 相同但数值不同（构造反例）必须判**不通过**；
   ``max|Δ| = 1e-7`` 必须判**不通过**（回归防护，对应 `AGENTS.md` §6-12「非零即不通过」）；
2. ``assert_eval_mode`` —— ``model.training=True`` 时抛错；``torch.no_grad()`` 外调用时抛错；
4. 同进程重复调用 ``--run-tag A`` 的防护告警触发。

（§6.3 第 3 项「七元组行解析／序列化」的实现在 ``01_00_smoke_env.py``，
其用例挂在 ``tests/env/test_01_00_smoke_env.py``；本文件另加一条**跨文件契约**断言，
确保两处对七元组字段集合的理解一致。）

约束：不加载模型、不访问网络、不访问 ``data/``、不改变环境。
被测脚本以文件路径动态加载（文件名以数字开头，不能用 ``import`` 语句）。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "env", "01_03_check_determinism.py")
SCRIPT_00 = os.path.join(ROOT, "scripts", "env", "01_00_smoke_env.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("mod_01_03_check_determinism", SCRIPT)


class _FakeModel(object):
    """只带 ``training`` 标志的最小替身——不加载任何真实模型。"""

    def __init__(self, training):
        self.training = training


# ---------------------------------------------------------------- D1／D2 比对
def _write_npy(path, array):
    np.save(str(path), array)
    return str(path)


def test_identical_arrays_pass(tmp_path):
    a = _write_npy(tmp_path / "a.npy", np.zeros((2, 3), dtype=np.float32))
    b = _write_npy(tmp_path / "b.npy", np.zeros((2, 3), dtype=np.float32))
    result = M.compare_logits(a, b)
    passed, reasons = M.judge_comparison(result)
    assert passed, reasons
    assert result["d1_md5_equal"] is True
    assert result["d2_max_abs_diff"] == 0.0


def test_md5_equal_but_values_differ_must_fail(tmp_path):
    """构造反例：md5 判据被人为置为相同，但数值不同 ⇒ 仍必须判不通过。"""
    a = _write_npy(tmp_path / "a.npy", np.zeros((2, 2), dtype=np.float32))
    b = _write_npy(tmp_path / "b.npy", np.ones((2, 2), dtype=np.float32))
    forced = "0" * 32
    result = M.compare_logits(a, b, md5_a=forced, md5_b=forced)
    assert result["d1_md5_equal"] is True
    assert result["d2_max_abs_diff"] != 0.0
    passed, reasons = M.judge_comparison(result)
    assert passed is False
    assert any("D2" in r for r in reasons)


def test_tiny_nonzero_delta_must_fail_regression(tmp_path):
    """回归防护：max|Δ| = 1e-7 必须判**不通过**（AGENTS.md §6-12 取严，非 1e-6）。"""
    base = np.zeros((4, 4), dtype=np.float32)
    other = base.copy()
    other[0, 0] = 1e-7
    a = _write_npy(tmp_path / "a.npy", base)
    b = _write_npy(tmp_path / "b.npy", other)
    result = M.compare_logits(a, b)
    assert result["d2_max_abs_diff"] == pytest.approx(1e-7)
    passed, reasons = M.judge_comparison(result)
    assert passed is False
    assert any("非零即不通过" in r for r in reasons)


def test_judge_refuses_nonzero_tolerance():
    """`tol` 固定 0.0——传 1e-6 属违规用法，必须直接报错而非静默放宽。"""
    with pytest.raises(ValueError):
        M.judge_comparison(
            {"d1_md5_equal": True, "d2_max_abs_diff": 0.0,
             "d2_nan_count_equal": True, "shape_mismatch": False}, tol=1e-6)
    with pytest.raises(ValueError):
        M.judge_comparison(
            {"d1_md5_equal": True, "d2_max_abs_diff": 0.0,
             "d2_nan_count_equal": True, "shape_mismatch": False}, tol=1e-7)


def test_shape_mismatch_must_fail(tmp_path):
    a = _write_npy(tmp_path / "a.npy", np.zeros((2, 3), dtype=np.float32))
    b = _write_npy(tmp_path / "b.npy", np.zeros((3, 2), dtype=np.float32))
    result = M.compare_logits(a, b)
    assert result["shape_mismatch"] is True
    passed, reasons = M.judge_comparison(result)
    assert passed is False
    assert any("形态" in r for r in reasons)


def test_nan_count_difference_must_fail(tmp_path):
    a = _write_npy(tmp_path / "a.npy", np.zeros((2, 2), dtype=np.float32))
    arr = np.zeros((2, 2), dtype=np.float32)
    arr[0, 0] = np.nan
    b = _write_npy(tmp_path / "b.npy", arr)
    result = M.compare_logits(a, b)
    assert result["d2_nan_count_equal"] is False
    passed, reasons = M.judge_comparison(result)
    assert passed is False
    assert any("NaN" in r for r in reasons)


def test_same_process_must_fail_even_when_bytes_match(tmp_path):
    """同进程「假通过」防护：即使 md5 与数值都相同，同 pid 也必须判不通过（§4.1）。"""
    a = _write_npy(tmp_path / "a.npy", np.zeros((2, 2), dtype=np.float32))
    b = _write_npy(tmp_path / "b.npy", np.zeros((2, 2), dtype=np.float32))
    result = M.compare_logits(a, b, pid_a=1234, pid_b=1234)
    assert result["same_process"] is True
    passed, reasons = M.judge_comparison(result)
    assert passed is False
    assert any("同一进程" in r for r in reasons)


def test_distinct_processes_pass(tmp_path):
    a = _write_npy(tmp_path / "a.npy", np.zeros((2, 2), dtype=np.float32))
    b = _write_npy(tmp_path / "b.npy", np.zeros((2, 2), dtype=np.float32))
    result = M.compare_logits(a, b, pid_a=111, pid_b=222)
    passed, reasons = M.judge_comparison(result)
    assert passed, reasons


def test_md5_file_matches_hashlib(tmp_path):
    import hashlib

    path = _write_npy(tmp_path / "a.npy", np.arange(6, dtype=np.float32))
    with open(path, "rb") as fh:
        expect = hashlib.md5(fh.read()).hexdigest()
    assert M.md5_file(path) == expect


# ---------------------------------------------------------------- eval 强制校验
def test_assert_eval_mode_accepts_eval_without_grad():
    assert M.assert_eval_mode(_FakeModel(False), grad_enabled=False) is True


def test_assert_eval_mode_rejects_training_true():
    with pytest.raises(M.EvalModeViolation):
        M.assert_eval_mode(_FakeModel(True), grad_enabled=False)


def test_assert_eval_mode_rejects_grad_enabled_true():
    with pytest.raises(M.EvalModeViolation):
        M.assert_eval_mode(_FakeModel(False), grad_enabled=True)


def test_assert_eval_mode_raises_outside_no_grad_context():
    """§6.3-2：在 ``torch.no_grad()`` 之外调用必须抛错（不依赖显式传参）。"""
    torch = pytest.importorskip("torch")
    assert torch.is_grad_enabled() is True  # 前提：本测试运行在 no_grad 之外
    with pytest.raises(M.EvalModeViolation):
        M.assert_eval_mode(_FakeModel(False))


def test_assert_eval_mode_ok_inside_no_grad_context():
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        assert M.assert_eval_mode(_FakeModel(False)) is True


def test_eval_mode_violation_is_runtime_error():
    assert issubclass(M.EvalModeViolation, RuntimeError)


# ---------------------------------------------------------------- run-tag 防护
def test_note_run_tag_warns_on_second_use_in_same_process():
    used = {}
    assert M.note_run_tag("llamaguard3-1b/A", used) is False
    assert M.note_run_tag("llamaguard3-1b/A", used) is True
    assert M.note_run_tag("llamaguard3-1b/B", used) is False


def test_run_tag_guard_is_per_tag_and_per_process_state():
    assert M.note_run_tag("A", {}) is False
    assert M.note_run_tag("A", {}) is False


# ---------------------------------------------------------------- 接口契约
def test_exit_codes_follow_contract():
    """windows/W2.3_determinism.md §4.1：通过 0；不通过 3；参数／环境错误 2。"""
    assert (M.EXIT_OK, M.EXIT_USAGE, M.EXIT_FAIL) == (0, 2, 3)


def test_contract_modes_present_and_extra_modes_are_additive():
    for mode in ("determinism", "eval-guard", "bnb-stability"):
        assert mode in M.MODES
    assert set(M.CONTRACT_MODES).issubset(set(M.MODES))


def test_parser_accepts_contract_arguments():
    parser = M.build_parser()
    args = parser.parse_args(["--model-key", "llamaguard3-1b",
                              "--mode", "determinism", "--run-tag", "A"])
    assert args.model_key == "llamaguard3-1b"
    assert args.run_tag == "A"
    assert args.repeat == 5
    assert args.report is None


def test_parser_rejects_unknown_run_tag():
    parser = M.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "determinism", "--run-tag", "C"])


def test_determinism_mode_requires_run_tag_and_model_key():
    assert M.main(["--mode", "determinism", "--model-key", "llamaguard3-1b"]) == M.EXIT_USAGE
    assert M.main(["--mode", "determinism", "--run-tag", "A"]) == M.EXIT_USAGE


def test_unknown_model_key_is_usage_error():
    assert M.main(["--mode", "determinism", "--run-tag", "A",
                   "--model-key", "no-such-model"]) == M.EXIT_USAGE


def test_torch_dtype_mapping_rejects_unknown():
    pytest.importorskip("torch")
    torch = sys.modules["torch"]
    assert M._torch_dtype(torch, "bfloat16") is torch.bfloat16
    with pytest.raises(ValueError):
        M._torch_dtype(torch, "int8")


# ---------------------------------------------------------------- 跨文件契约
def test_seven_tuple_field_set_matches_01_00_definition():
    """`01_03` 的报告从 `01_00` 的 `seven_tuple` 取值——两处字段集合必须一致。"""
    m00 = _load("mod_01_00_smoke_env", SCRIPT_00)
    assert m00.SEVEN_TUPLE_FIELDS == (
        "suite", "model", "load", "device", "batch", "seq_len",
        "offload", "collector", "peak", "reserved", "repeat", "run_id",
    )
    # 01_03 的报告渲染按这些键取列，键名必须来自同一集合
    for field in ("suite", "model", "load", "device", "batch", "seq_len",
                  "offload", "collector", "peak", "reserved", "repeat", "run_id"):
        assert field in m00.SEVEN_TUPLE_FIELDS


def _module_level_imports(path):
    """AST 判定模块级（顶层）import 的**顶层包名**集合——不含函数内 import。"""
    import ast

    tree = ast.parse(open(path, "r", encoding="utf-8").read())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return found


def test_no_module_level_torch_import_in_01_03():
    """模块级不得 import torch——否则 CUBLAS_WORKSPACE_CONFIG 的「import 前置」失效。"""
    heavy = _module_level_imports(SCRIPT) & {"torch", "transformers", "bitsandbytes",
                                             "peft", "trl", "numpy"}
    assert heavy == set(), "模块级出现重型导入：%s" % sorted(heavy)

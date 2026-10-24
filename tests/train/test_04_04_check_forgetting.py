# -*- coding: utf-8 -*-
"""tests/train/test_04_04_check_forgetting.py — W5.4 `04_04` 灾难性遗忘验证的单元与接口测试。

覆盖 `windows/W5.4_forgetting.md` §2（口径）／§3（接口契约）／§4（硬约束）／§6（验收）：

1. **`--self-check`** OK（纯合成，不读分片、不加载模型）；
2. **路径拒绝（验收 3）**：`data/sealed_test/...` 与 frozen 侧 `.../test.csv` **两类**均
   **非零退出（=2）**并打印 `REJECTED` 行；frozen 侧 `val.csv` 必须被接受；
3. **Δ 口径（§2.1）**：主口径**绝对百分点**；并列**相对降幅**；阈值 `≤0.03` 的边界
   （0.03 ⇒ PASS／0.0300001 ⇒ FAIL）；`None`（FPR 无定义）与 `base=0` 的处置；
4. **三项指标齐备（验收 2）**：明细 CSV 含 `f1`／`fpr`／`tpr` 及其三组 Δ；
5. **`STATUS`／`GATE` 自洽（§4-4）**：A5（不传 `--clean-test`）恒
   `STATUS=SELFCHECK_ONLY`＋`GATE=UNKNOWN` 且 **exit=0**；传 `--clean-test` 才 `OFFICIAL`；
6. **clean 冻结基准（§2.2①）**：F1 **0.9885**／TPR **0.9773**／FPR **None** 不得被动过；
7. **指标单源**：退化 F1 = `2TP/(2TP+FN)`、无负样本时 FPR = `None`；`pred` 空按 `03_07` 同款计 0；
8. **`--dry-run`**：只校验不跑模型 ⇒ 单测无需 GPU。

**不加载模型、不需要 GPU**（子进程只跑 `--self-check`／`--dry-run`／路径拒绝用例）。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(ROOT, "scripts", "train", "04_04_check_forgetting.py")
FORGETTING_CFG = os.path.join(ROOT, "configs", "train", "forgetting.yaml")

VAL_ID = "data/frozen/CodeObf-PI_v1.0/splits/regime_id/val.csv"
VAL_OOD = "data/frozen/CodeObf-PI_v1.0/splits/regime_ood/val.csv"
ID_RUNS = [
    "checkpoints/ours/04_01_llamaguard1b_id_seed42_20261005-150806",
    "checkpoints/ours/04_01_llamaguard1b_id_seed123_20261006-010210",
    "checkpoints/ours/04_01_llamaguard1b_id_seed456_20261006-120248",
]
OOD_RUNS = [
    "checkpoints/ours/04_01_llamaguard1b_ood_seed42_20261007-144100",
    "checkpoints/ours/04_01_llamaguard1b_ood_seed123_20261007-180342",
    "checkpoints/ours/04_01_llamaguard1b_ood_seed456_20261007-202634",
]

HAS_CKPTS = all(os.path.isdir(os.path.join(ROOT, d)) for d in ID_RUNS)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load("t_04_04", SCRIPT)


def _run_cli(*argv):
    proc = subprocess.run([sys.executable, "-u", SCRIPT, *argv],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


# ---------------------------------------------------------------------------
# 1. --self-check
# ---------------------------------------------------------------------------
def test_self_check_ok():
    code, out = _run_cli("--self-check")
    assert code == 0, out
    assert "04_04_check_forgetting SELF_CHECK OK" in out


def test_self_check_module_level(m):
    assert m._self_check() == 0


# ---------------------------------------------------------------------------
# 2. 路径拒绝（验收 3：两类均非零退出）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "data/sealed_test/CodeObf-PI_v1.0/clean_original_test.csv",
    "data/sealed_test/CodeObf-PI_v1.0/regime_id_test.csv",
    "data/sealed_test/CodeObf-PI_v1.0/regime_ood_test.csv",
])
def test_cli_rejects_sealed_val(bad):
    code, out = _run_cli("--regime", "id", "--checkpoints", ID_RUNS[0], "--val", bad)
    assert code == 2, out
    assert "REJECTED" in out
    assert "data/sealed_test" in out


@pytest.mark.parametrize("bad", [
    "data/frozen/CodeObf-PI_v1.0/splits/regime_id/test.csv",
    "data/frozen/CodeObf-PI_v1.0/splits/regime_ood/test.csv",
])
def test_cli_rejects_frozen_test_csv(bad):
    code, out = _run_cli("--regime", "id", "--checkpoints", ID_RUNS[0], "--val", bad)
    assert code == 2, out
    assert "REJECTED" in out


def test_path_guard_accepts_frozen_val(m):
    for p, regime in ((VAL_ID, "id"), (VAL_OOD, "ood")):
        assert m.assert_no_sealed(p, "--val").endswith(m.ALLOWED_VAL_SUFFIXES)


def test_path_guard_rejects(m):
    for bad in ("data/sealed_test/CodeObf-PI_v1.0/clean_original_test.csv",
                "data/frozen/CodeObf-PI_v1.0/splits/regime_id/test.csv",
                "data/frozen/CodeObf-PI_v1.0/splits/regime_ood/test.csv"):
        with pytest.raises(m.PathRejected):
            m.assert_no_sealed(bad, "--val")


def test_regime_must_match_val(m):
    """`--regime id` 配 `ood` 的 val ⇒ 拒绝（防「跑错体制」静默通过）。"""
    with pytest.raises((m.PathRejected, m.ForgettingAssert)):
        m.assert_val_allowed(VAL_OOD, "id")


# ---------------------------------------------------------------------------
# 3. Δ 口径（§2.1）
# ---------------------------------------------------------------------------
def test_delta_absolute_and_relative(m):
    a, r = m.delta(0.9885, 0.9700)
    assert abs(a - 0.0185) < 1e-12
    assert abs(r - 0.0185 / 0.9885) < 1e-12          # 并列口径：相对降幅必须同时给出


def test_delta_none_and_zero_base(m):
    assert m.delta(None, 0.9) == (None, None)
    assert m.delta(0.9, None) == (None, None)
    assert m.delta(0.0, 0.0) == (0.0, None)          # base=0 ⇒ 相对项 n/a（不除零）


def test_verdict_threshold_boundary(m):
    assert m.verdict_of(0.03) == m.GATE_PASS          # ≤ 阈值 ⇒ PASS
    assert m.verdict_of(0.0299999) == m.GATE_PASS
    assert m.verdict_of(0.0300001) == m.GATE_FAIL
    assert m.verdict_of(-0.2) == m.GATE_PASS          # 提升 ⇒ PASS
    assert m.verdict_of(None) == m.GATE_UNKNOWN


def test_threshold_is_frozen(m):
    assert m.DF1_THRESHOLD == 0.03                    # `PREREG_v1.md` §5 冻结值


# ---------------------------------------------------------------------------
# 4. 指标单源与 clean 退化口径（§2.2①）
# ---------------------------------------------------------------------------
def test_clean_frozen_baseline_values(m):
    assert m.FROZEN_BASE_CLEAN_F1 == 0.9885
    assert m.FROZEN_BASE_CLEAN_TPR == 0.9773
    assert m.FROZEN_BASE_CLEAN_FPR is None            # 44 行全 label=1 ⇒ FPR 无定义


def test_degenerate_f1_and_undefined_fpr(m):
    met = m.compute_binary_metrics([1] * 44, [1] * 43 + [0])
    assert met["fpr"] is None
    assert abs(met["f1"] - 2.0 * 43 / (2.0 * 43 + 1)) < 1e-12


def test_metrics_from_03_03_outputs_counts_unparsed(tmp_path, m):
    """`pred` 空（`unparsed`）按 `03_07` 同款计 0，并单独披露计数（不静默）。"""
    val_csv = tmp_path / "val.csv"
    val_csv.write_text("sample_id,text,label\n"
                       "s1,aaa,1\ns2,bbb,1\ns3,ccc,0\ns4,ddd,0\n",
                       encoding="utf-8", newline="\n")
    # s1 TP（label 1／pred 1）；s2 空 pred ⇒ 计 0 ⇒ FN；s3 TN；s4 FP
    out_csv = tmp_path / "b3_x_v1.0_t.csv"
    out_csv.write_text("sample_id,text_sha256,pred,raw_type,raw,score,latency_ms\n"
                       "s1,x,1,judgment,unsafe,1.0\n"
                       "s2,x,,judgment,maybe,1.0\n"
                       "s3,x,0,judgment,safe,1.0\n"
                       "s4,x,1,judgment,unsafe,1.0\n",
                       encoding="utf-8", newline="\n")
    met = m.metrics_from_03_03_outputs(str(out_csv), str(val_csv))
    assert met["n"] == 4
    assert met["n_unparsed"] == 1                     # s2 空 pred
    assert (met["tp"], met["fp"], met["fn"], met["tn"]) == (1, 1, 1, 1)
    assert abs(met["tpr"] - 0.5) < 1e-12 and abs(met["fpr"] - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# 5. STATUS／GATE 自洽（§4-4）＋ dry-run
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_CKPTS, reason="需要 W5.3 的 3 个 id run 目录")
def test_dry_run_selfcheck_only_and_unknown_gate():
    code, out = _run_cli("--regime", "id", "--checkpoints", ",".join(ID_RUNS),
                         "--val", VAL_ID, "--dry-run")
    assert code == 0, out
    assert "DRYRUN_RESULT=PASS n_ckpt=3" in out
    assert out.strip().endswith("STATUS=SELFCHECK_ONLY GATE=UNKNOWN")


@pytest.mark.skipif(not HAS_CKPTS, reason="需要 W5.3 的 3 个 id run 目录")
def test_dry_run_flips_status_when_clean_given():
    """传 `--clean-test` ⇒ `STATUS=OFFICIAL`（`--dry-run` 不真的读文件）。"""
    code, out = _run_cli("--regime", "id", "--checkpoints", ID_RUNS[0],
                         "--val", VAL_ID, "--dry-run",
                         "--clean-test", "data/sealed_test/CodeObf-PI_v1.0/clean_original_test.csv")
    assert code == 0, out
    assert "STATUS=OFFICIAL" in out


def test_status_gate_constants(m):
    assert m.STATUS_SELFCHECK_ONLY == "SELFCHECK_ONLY"
    assert m.STATUS_OFFICIAL == "OFFICIAL"
    assert set((m.GATE_PASS, m.GATE_FAIL, m.GATE_UNKNOWN)) == {"PASS", "FAIL", "UNKNOWN"}


def test_exit_codes(m):
    assert m.EXIT_OK == 0
    assert m.EXIT_PATH_REJECTED == 2                  # 路径拒绝必须非零
    assert m.EXIT_ASSERT == 3


# ---------------------------------------------------------------------------
# 6. 明细 CSV schema（验收 2：三项指标与其变化量齐备）
# ---------------------------------------------------------------------------
def test_detail_columns_complete(m):
    for c in ("regime", "seed", "run_id", "step", "f1", "fpr", "tpr",
              "d_f1_abs", "d_f1_rel", "d_fpr_abs", "d_fpr_rel",
              "d_tpr_abs", "d_tpr_rel", "verdict_val_selfcheck"):
        assert c in m.DETAIL_COLUMNS


def test_write_csv_roundtrip(tmp_path, m):
    row = {c: "" for c in m.DETAIL_COLUMNS}
    row.update({"regime": "id", "seed": 42, "f1": 1.0, "fpr": 0.0, "tpr": 1.0,
                "d_f1_abs": -0.2, "verdict_val_selfcheck": "PASS"})
    p = str(tmp_path / "d.csv")
    m.write_csv(p, [row], m.DETAIL_COLUMNS)
    with open(p, "r", encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 1 and got[0]["seed"] == "42" and got[0]["d_f1_abs"] == "-0.2"


# ---------------------------------------------------------------------------
# 7. checkpoint 解析：τ 取 best.json（§4-3：不在本窗口重新寻 τ）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_CKPTS, reason="需要 W5.3 的 run 目录")
@pytest.mark.parametrize("d", ID_RUNS + OOD_RUNS)
def test_resolve_checkpoint_matches_best_json(m, d):
    info = m.resolve_checkpoint(d)
    with open(os.path.join(ROOT, d, "best.json"), "r", encoding="utf-8") as fh:
        best = json.load(fh)
    assert info["tau"] == best["tau_ours"]
    assert info["seed"] == best["seed"]
    assert info["regime"] == best["regime"]
    assert str(info["step"]) == str(best["step"])
    assert os.path.isdir(info["ckpt_dir"])


def test_resolve_checkpoint_missing_dir(m):
    with pytest.raises(m.ForgettingAssert):
        m.resolve_checkpoint("checkpoints/ours/__does_not_exist__")


# ---------------------------------------------------------------------------
# 8. 报告：STATUS／GATE 两字段同时出现；不传 clean 时不得出现正式结论
# ---------------------------------------------------------------------------
def test_report_has_status_and_gate_and_no_official_verdict(tmp_path, m):
    ctx = {
        "ts": "20990101-000000", "regime": "id",
        "val_rel": VAL_ID, "val_md5": "0" * 32,
        "config": "configs/train/forgetting.yaml",
        "ours_config": "configs/train/qlora_ours.yaml",
        "status": m.STATUS_SELFCHECK_ONLY, "gate": m.GATE_UNKNOWN,
        "status_reason": "A5", "gate_reason": "未传 --clean-test",
        "exit_code": 0, "exit_reason": "SELFCHECK_ONLY 不失败",
        "base_rows": [{"split": "id_val", "regime": "id", "n": 10, "tp": 5, "fp": 1,
                       "fn": 0, "tn": 4, "tpr": 1.0, "fpr": 0.2, "f1": 0.909,
                       "n_unparsed": 0}],
        "base_metrics_csv": "reports/train/base_val_metrics_v1.0_x.csv",
        "detail_rows": [{"track": "val", "seed": 42, "run_id": "r", "step": 1, "tau": 0.5,
                         "f1": 1.0, "fpr": 0.0, "tpr": 1.0, "d_f1_abs": -0.091,
                         "d_f1_rel": -0.1, "d_fpr_abs": 0.2, "d_fpr_rel": 1.0,
                         "d_tpr_abs": 0.0, "d_tpr_rel": 0.0,
                         "verdict_val_selfcheck": "PASS"}],
        "summary": {"d_f1": "-0.091 ± 0.0", "d_fpr": "0.2 ± 0.0",
                    "d_tpr": "0.0 ± 0.0", "per_seed_raw": "seed 42: …",
                    "std_degenerate": False},
        "base_clean_f1": m.FROZEN_BASE_CLEAN_F1, "base_clean_tpr": m.FROZEN_BASE_CLEAN_TPR,
        "has_clean": False, "ckpt_dirs": ["checkpoints/ours/x"],
        "base_score_track": None,
        "artifacts": [("明细", "reports/train/x.csv")],
        "to_w0": ["事项一"], "receipt": ["回执一"],
    }
    p = str(tmp_path / "r.md")
    m.write_report(p, ctx)
    text = open(p, "r", encoding="utf-8").read()
    assert "STATUS=SELFCHECK_ONLY" in text
    assert "GATE=UNKNOWN" in text
    assert "0.9885" in text and "0.9773" in text        # ① clean 冻结基准被显式引用
    assert "clean 轨道（正式结论轨）—— **本轮未跑**" in text
    assert "GATE=PASS" not in text and "GATE=FAIL" not in text


# ---------------------------------------------------------------------------
# 9. 配置载体：不得含 sealed 分片（`D-41` 裁定五③）
# ---------------------------------------------------------------------------
def test_forgetting_config_has_no_sealed_splits():
    import yaml
    cfg = yaml.safe_load(open(FORGETTING_CFG, "r", encoding="utf-8"))
    splits = (cfg.get("splits") or {})
    assert set(splits.get("all") or []) == {"id_val", "ood_val"}
    assert splits.get("sealed") is False
    assert cfg["integrity"]["access_log_required"] is False
    assert cfg["weights"]["finetuned"] is False          # 基座 = 原始权重
    assert cfg["weights"]["hf_revision"] == "acf7aafa60f0410f8f42b1fa35e077d705892029"


def test_base_val_cmd_passes_splits_separately(m):
    """`03_03 --splits` 是 `nargs="*"` ⇒ 必须逐个传入；拼成逗号串会被当单个分片名而静默 SKIP。"""
    cmd = m.build_base_val_cmd(["id_val", "ood_val"], "cfg.yaml", "unify.yaml",
                               "out", "rep", "20261008-000000")
    i = cmd.index("--splits")
    assert cmd[i + 1] == "id_val" and cmd[i + 2] == "ood_val"
    assert ",".join(["id_val", "ood_val"]) not in cmd
    assert "--report-dir" in cmd and "rep" in cmd          # `D-41` 裁定五①
    assert "--outputs-dir" in cmd and "out" in cmd         # `D-41` 裁定五②
    assert "configs/baseline/b3_llamaguard1b.yaml" not in cmd   # `D-41` 裁定五③


def test_baseline_b3_config_is_not_reused(m):
    """默认 `--config` 必须是 train 侧那份，不得指向 `configs/baseline/b3_llamaguard1b.yaml`。"""
    p = m.build_parser().parse_args(["--regime", "id", "--checkpoints", "x", "--val", VAL_ID])
    assert p.config == "configs/train/forgetting.yaml"
    assert "sealed" not in p.config


# ---------------------------------------------------------------------------
# 10. 冻结接口契约（§3）必填项
# ---------------------------------------------------------------------------
def test_cli_contract_arguments(m):
    p = m.build_parser().parse_args(["--regime", "ood", "--checkpoints", "a,b",
                                     "--val", VAL_OOD, "--clean-test", "c.csv",
                                     "--base-val-metrics", "m.csv",
                                     "--base-clean-f1", "0.9885", "--base-clean-tpr", "0.9773",
                                     "--config", "cfg.yaml", "--report-dir", "r", "--out-dir", "o"])
    assert p.regime == "ood"
    assert p.checkpoints == "a,b"
    assert p.clean_test == "c.csv"
    assert p.base_clean_f1 == 0.9885 and p.base_clean_tpr == 0.9773
    assert p.config == "cfg.yaml" and p.report_dir == "r" and p.out_dir == "o"

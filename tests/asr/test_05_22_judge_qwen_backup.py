# -*- coding: utf-8 -*-
"""tests/asr/test_05_22_judge_qwen_backup.py — W6.2 备用裁判脚本单元与接口测试。

不联网（`call_judge` 被 monkeypatch）、不读封存件、不触 GPU。
覆盖：自检 exit=0；缺 key ⇒ 非零退出；型号为 `PREREG_v1.6.md` §1.1 定值；
`enable_thinking` **不强制**且分支可留证；退避重试口径；主备一致性；
`--only-refused-from` 切换集合。
"""
from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SCRIPT = os.path.join(_ROOT, "scripts", "asr", "05_22_judge_qwen_backup.py")
CONFIG = os.path.join(_ROOT, "configs", "asr", "asr.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("asr_05_22", SCRIPT)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_self_check_exit_zero():
    r = subprocess.run([sys.executable, SCRIPT, "--self-check", "--config", CONFIG],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "SELFCHECK_RESULT=PASS" in r.stdout


def test_missing_key_nonzero(tmp_path):
    units = tmp_path / "u.csv"
    units.write_text("sample_id,scenario,downstream_model,run_id,payload_text,model_output,"
                     "gen_params,error_log,df_label\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "DASHSCOPE_API_KEY"}
    r = subprocess.run([sys.executable, SCRIPT, "--units", str(units), "--config", CONFIG,
                        "--report-dir", str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "MISSING_KEY_ENV" in r.stderr


def test_backup_model_is_v16_value(cfg):
    assert cfg["judge"]["backup"]["model"] == "qwen3.5-397b-a17b"
    assert cfg["judge"]["backup"]["enable_thinking"] is None      # 不强制
    assert cfg["judge"]["backup"]["url"].endswith("/compatible-mode/v1/chat/completions")


def test_enable_thinking_variant_passed(mod, cfg, monkeypatch):
    seen = {}

    def fake_call(url, key, model, prompt, **kw):
        seen["extra"] = kw.get("extra")
        return {"http": 200, "attempt": 1, "latency_s": 0.1, "model_echo": "qwen3.5-397b-a17b",
                "system_fingerprint": None, "fingerprint_sha256": "fp", "reasoning_tokens": None,
                "raw": "LABEL=AJ-N", "label": "AJ-N", "transport_failure": False}

    monkeypatch.setattr(mod.C, "call_judge", fake_call)
    rows = [{"sample_id": "S0", "scenario": "none", "downstream_model": "mistral-7b",
             "run_id": "none", "payload_text": "P", "model_output": "O"}]
    mod.judge_units(cfg, rows, "k")                 # 默认 without ⇒ 不带该参数
    assert seen["extra"] is None

    cfg["judge"]["backup"]["enable_thinking_variant"] = "with_false"
    mod.judge_units(cfg, rows, "k")
    assert seen["extra"] == {"enable_thinking": False}   # 两分支均可留证
    cfg["judge"]["backup"]["enable_thinking_variant"] = "without"


def test_refused_keys(mod, tmp_path):
    p = tmp_path / "jp.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "run_id", "label", "transport_failure"])
        w.writerow(["S0", "none", "AJ-Y", "False"])    # 正常
        w.writerow(["S1", "none", "", "True"])         # 链路失败 ⇒ 切
        w.writerow(["S2", "none", "", "False"])        # 未解析 ⇒ 切
    keys = mod.refused_keys(str(p))
    assert keys == {("S1", "none"), ("S2", "none")}


def test_consistency(mod):
    prim = [{"sample_id": "S0", "run_id": "none", "label": "AJ-Y"},
            {"sample_id": "S1", "run_id": "none", "label": "AJ-N"}]
    back = [{"sample_id": "S0", "run_id": "none", "label": "AJ-Y"},
            {"sample_id": "S1", "run_id": "none", "label": "AJ-Y"}]
    c = mod.C.majority_vote_consistency(prim, back)
    assert c["n"] == 2 and c["agree"] == 1 and abs(c["rate"] - 0.5) < 1e-12


def test_top_imports_light():
    src = open(SCRIPT, encoding="utf-8").read().splitlines()
    tops = [ln.split()[1].split(".")[0] for ln in src if ln.startswith(("import ", "from "))]
    assert "torch" not in tops and "transformers" not in tops

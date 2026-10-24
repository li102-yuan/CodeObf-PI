# -*- coding: utf-8 -*-
"""tests/train/test_04_01_deconfound.py — `W5.7` `--deconfound`（`P-K'`）单元测试。

覆盖 `windows/W5.7_encoded_benign_retrain.md` §3-8 列出的八项：

1. **off 不变**（`enabled=False` 逐字返回 ＋ 默认 `off`）；
2. **攻击对角**（最低位 `1` ⇒ `text := plaintext`；`0` ⇒ 保持）；
3. **良性对角**（同位 `1` ⇒ `text := enc(text[:L])`；`0` ⇒ 保持明文）；
4. **ood 码集**（仅 5 训练码；4 留出码零暴露；`id` ⊆ 9 码）；
5. **行数比例**（替换而非新增 ⇒ n／n_pos／n_neg 不变）；
6. **L 规则**（按扩展比反推使输出 ≈950；`ascii_art` 下限钳 64）；
7. **只改 text**（其余列零改动，逐列断言）；
8. **val 同规**（同一 `sha256` 判定 ＋ `run_train`/`run_threshold` 接线在位）。

另附：确定性、契约日志行格式、空 `plaintext` 报错、CLI 开关默认值、`--self-check` 通过。

**设计纪律**：不加载模型权重、不需要 GPU；判定一律与独立复算的
`sha256(sample_id)` 规则对照（不自证）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # tests/train -> 项目根
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train", "04_01_train_qlora_ours.py")
CONFIG = os.path.join(ROOT, "configs", "train", "qlora_ours.yaml")
COMMON = os.path.join(ROOT, "scripts", "data", "_common.py")
ENCODERS_MOD = os.path.join(ROOT, "scripts", "data", "02_03_encode_variants.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load("t_04_01_deconfound", TRAIN_SCRIPT)


@pytest.fixture(scope="module")
def cfg(m):
    return m.load_config(CONFIG)


@pytest.fixture(scope="module")
def common():
    return _load("t_data_common_deconfound", COMMON)


@pytest.fixture(scope="module")
def enc_mod():
    return _load("t_02_03_deconfound", ENCODERS_MOD)


# ---------------------------------------------------------------------------
# 工具：独立复算的对角规则（不复用被测实现 ⇒ 不自证）
# ---------------------------------------------------------------------------
def _rule(sample_id, allowed):
    h = int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest(), 16)
    return (h & 1) == 1, allowed[(h >> 1) % len(allowed)]


def _synth_frame(n=96):
    """合成帧：攻击／良性各半；`sample_id` 连续可复算；攻击行带 `plaintext`。"""
    rows = []
    for i in range(n):
        sid = "SYN-%04d" % i
        if i % 2 == 0:
            rows.append({"sample_id": sid, "label": "1", "text": "ENC_PAYLOAD_%d" % i,
                         "plaintext": "PLAIN_ATTACK_%d" % i, "group_id": "SG-%d" % i,
                         "encoding_type": "base64", "sample_kind": "attack"})
        else:
            rows.append({"sample_id": sid, "label": "0",
                         "text": ("benign prompt body %d " % i) * 80,
                         "plaintext": "", "group_id": "BG-%d" % i,
                         "encoding_type": "", "sample_kind": "benign"})
    return rows


# ---------------------------------------------------------------------------
# ① off 不变
# ---------------------------------------------------------------------------
def test_off_returns_input_unchanged_and_default_off(m):
    rows = _synth_frame()
    out, st = m.deconfound_apply(rows, "id", enabled=False)
    assert out is rows and st is None, "`off` 必须逐字返回输入（同一对象、无 stats）"
    args = m.build_parser().parse_args(["--self-check"])
    assert args.deconfound == "off", "`--deconfound` 默认必须 off"
    args2 = m.build_parser().parse_args(["--regime", "id", "--seed", "42"])
    assert args2.deconfound == "off"


# ---------------------------------------------------------------------------
# ② 攻击对角
# ---------------------------------------------------------------------------
def test_attack_diagonal(m):
    rows = _synth_frame()
    out, stats = m.deconfound_apply(rows, "id", enabled=True)
    allowed = list(m.deconfound_allowed_codes("id"))
    n_plain = 0
    for a, b in zip(rows, out):
        if a["label"] != "1":
            continue
        replace, _code = _rule(a["sample_id"], allowed)
        if replace:
            assert b["text"] == a["plaintext"], "攻击行被选中须换 plaintext：%s" % a["sample_id"]
            n_plain += 1
        else:
            assert b["text"] == a["text"], "攻击行未选中须保持 text：%s" % a["sample_id"]
    assert stats["attacks_plaintext"] == n_plain
    assert 0 < n_plain < sum(1 for r in rows if r["label"] == "1"), \
        "攻击臂应有替换且非全替换（p=0.5 对角）"


# ---------------------------------------------------------------------------
# ③ 良性对角
# ---------------------------------------------------------------------------
def test_benign_diagonal(m, enc_mod):
    rows = _synth_frame()
    out, stats = m.deconfound_apply(rows, "id", enabled=True)
    allowed = list(m.deconfound_allowed_codes("id"))
    n_enc = 0
    for a, b in zip(rows, out):
        if a["label"] != "0":
            continue
        replace, code = _rule(a["sample_id"], allowed)
        if replace:
            L = m.deconfound_L(code, a["text"])
            assert b["text"] == enc_mod.ENCODERS[code][0](a["text"][:L]), \
                "良性行被选中须换 enc(text[:L])：%s" % a["sample_id"]
            n_enc += 1
        else:
            assert b["text"] == a["text"], "良性行未选中须保持明文：%s" % a["sample_id"]
    assert stats["benign_encoded"] == n_enc
    assert 0 < n_enc < sum(1 for r in rows if r["label"] == "0")


# ---------------------------------------------------------------------------
# ④ ood 码集（仅 5 训练码；留出码零暴露）＋ id ⊆ 9 码
# ---------------------------------------------------------------------------
def test_code_sets_and_ood_zero_heldout(m, common):
    assert tuple(m.deconfound_allowed_codes("id")) == tuple(common.ENCODING_CODES) == (
        "base64", "hex", "ascii_art", "rot13", "reverse", "morse", "url", "zerowidth",
        "homoglyph")
    assert tuple(m.deconfound_allowed_codes("ood")) == tuple(common.OOD_TRAIN_ENCODINGS) == (
        "base64", "rot13", "zerowidth", "url", "reverse")
    held = set(common.OOD_HELDOUT_ENCODINGS)
    assert held == {"hex", "morse", "ascii_art", "homoglyph"}

    rows = _synth_frame(256)
    out, st_ood = m.deconfound_apply(rows, "ood", enabled=True)
    used = set(st_ood["codes_used"])
    assert used and not (used - set(common.OOD_TRAIN_ENCODINGS)), \
        "ood 增广码越出 5 训练码：%s" % used
    assert not (used & held), "ood 增广命中留出码：%s" % sorted(used & held)
    _o, st_id = m.deconfound_apply(rows, "id", enabled=True)
    assert not (set(st_id["codes_used"]) - set(common.ENCODING_CODES))


def test_ood_real_split_zero_heldout(m, common):
    """真实 `ood` train/val 切片上同样零暴露（全量扫描归 `--self-check` S5）。"""
    held = set(common.OOD_HELDOUT_ENCODINGS)
    for split in ("train", "val"):
        tr, va = m.resolve_split_paths(m.load_config(CONFIG), "ood")
        rows = m.deconfound_load_frame(m.load_config(CONFIG),
                                       tr if split == "train" else va)[:600]
        _o, st = m.deconfound_apply(rows, "ood", enabled=True)
        assert not (set(st["codes_used"]) & held), "真实 ood %s 命中留出码" % split
        assert not (set(st["codes_used"]) - set(common.OOD_TRAIN_ENCODINGS))


# ---------------------------------------------------------------------------
# ⑤ 行数／比例不变
# ---------------------------------------------------------------------------
def test_rowcount_and_ratio_invariant(m):
    rows = _synth_frame(128)
    for regime in ("id", "ood"):
        out, st = m.deconfound_apply(rows, regime, enabled=True)
        n_pos = sum(1 for r in rows if r["label"] == "1")
        assert len(out) == len(rows)
        assert st["n_rows"] == len(rows) and st["n_pos"] == n_pos \
            and st["n_neg"] == len(rows) - n_pos
        out_pos = sum(1 for r in out if r["label"] == "1")
        assert out_pos == n_pos, "替换不得改动 label 分布"


# ---------------------------------------------------------------------------
# ⑥ L 规则
# ---------------------------------------------------------------------------
def test_l_rule_targets_950_with_ascii_art_floor(m, enc_mod):
    text = ("The quick brown fox jumps over the lazy dog. 0123456789. " * 50)  # 2900 字符
    for code, (enc, _dec) in enc_mod.ENCODERS.items():
        L = m.deconfound_L(code, text)
        ratio = len(enc(text)) / float(len(text))
        expect = max(1, min(len(text), int(round(950.0 / ratio))))
        if code == "ascii_art":
            expect = max(expect, 64)
        assert L == expect, "L 规则不符：%s got=%d expect=%d" % (code, L, expect)
        out = enc(text[:L])
        if code == "ascii_art":
            assert L == 64, "ascii_art 下限钳 64（反推值 %d 被钳）" % int(round(950.0 / ratio))
        else:
            assert abs(len(out) - 950) <= 250, \
                "%s 输出应 ≈950，实得 %d" % (code, len(out))


def test_l_rule_short_text_and_floor(m, enc_mod):
    short = "hello world" * 3          # 33 字符
    L = m.deconfound_L("hex", short)
    assert L == min(len(short), int(round(950.0 / 2.0))) == len(short)
    L_art = m.deconfound_L("ascii_art", short)
    assert L_art == 64, "ascii_art 下限钳 64 即使短文本（切片自然截断）"
    with pytest.raises(AssertionError):
        m.deconfound_L("hex", "")


# ---------------------------------------------------------------------------
# ⑦ 只改 text
# ---------------------------------------------------------------------------
def test_only_text_column_changed(m):
    rows = _synth_frame(128)
    for regime in ("id", "ood"):
        out, _st = m.deconfound_apply(rows, regime, enabled=True)
        for i, (a, b) in enumerate(zip(rows, out)):
            assert set(a.keys()) == set(b.keys()), "第 %d 行列集合被改动" % i
            for k in a:
                if k == "text":
                    continue
                assert a[k] == b[k], "第 %d 行列 %r 被改动（只允许改 text）" % (i, k)


# ---------------------------------------------------------------------------
# ⑧ val 同规
# ---------------------------------------------------------------------------
def test_val_same_rule_and_wiring(m):
    rows = _synth_frame(64)
    out, st = m.deconfound_apply(rows, "id", enabled=True)
    allowed = list(m.deconfound_allowed_codes("id"))
    for a, b in zip(rows, out):
        replace, _code = _rule(a["sample_id"], allowed)
        changed = a["text"] != b["text"]
        if a["label"] == "1":
            assert changed == replace
        elif replace:
            assert changed
    # 接线静态检查：`run_train`／`run_threshold` 均把 val 送进 `deconfound_apply`
    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as fh:
        src = fh.read()
    fn_train = src.split("def run_train(", 1)[1].split("\ndef ", 1)[0]
    assert "deconfound_apply(val_rows" in fn_train, "run_train 的 val 未同规"
    assert "deconfound_apply(train_rows" in fn_train, "run_train 的 train 未接入"
    fn_thr = src.split("def run_threshold(", 1)[1].split("\ndef ", 1)[0]
    assert "deconfound_apply(val_rows" in fn_thr, "run_threshold 的 val 未同规"


# ---------------------------------------------------------------------------
# 附加：确定性／契约日志行／空 plaintext 报错／CLI／self-check
# ---------------------------------------------------------------------------
def test_deterministic_two_runs_identical(m):
    rows = _synth_frame(128)
    a, sa = m.deconfound_apply(rows, "ood", enabled=True)
    b, sb = m.deconfound_apply(rows, "ood", enabled=True)
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert sa == sb


def test_log_line_matches_contract(m):
    rows = _synth_frame(64)
    _o, st = m.deconfound_apply(rows, "id", enabled=True)
    seen = []
    line = m.deconfound_log(st, seen.append, "train")
    assert line.startswith("DECONFOUND=on attacks_plaintext=%d/%d benign_encoded=%d/%d codes="
                           % (st["attacks_plaintext"], st["n_pos"],
                              st["benign_encoded"], st["n_neg"]))
    assert line.endswith("split=train")
    assert seen == [line]
    off_line = m.deconfound_log(None, seen.append, "train")
    assert off_line == "DECONFOUND=off"


def test_attack_empty_plaintext_raises(m):
    rows = _synth_frame(256)
    allowed = list(m.deconfound_allowed_codes("id"))
    for r in rows:
        if r["label"] == "1" and _rule(r["sample_id"], allowed)[0]:
            bad = [dict(r)]
            bad[0]["plaintext"] = ""
            with pytest.raises(AssertionError):
                m.deconfound_apply(bad, "id", enabled=True)
            break
    else:
        pytest.fail("未找到被选中的攻击行（合成帧应覆盖两个分支）")


def test_cli_deconfound_and_self_check_flags(m):
    a = m.build_parser().parse_args(["--regime", "id", "--seed", "42"])
    assert a.deconfound == "off" and a.self_check is False
    b = m.build_parser().parse_args(["--regime", "ood", "--seed", "456",
                                     "--deconfound", "on"])
    assert b.deconfound == "on"
    c = m.build_parser().parse_args(["--self-check"])
    assert c.self_check is True and c.regime is None and c.seed is None
    with pytest.raises(SystemExit):
        m.build_parser().parse_args(["--regime", "id", "--seed", "7"])   # 种子须 ∈ {42,123,456}


def test_self_check_pass(m, cfg):
    """`--self-check` 的函数级等价：§3-1～§3-7 全过 ⇒ 返回 8。"""
    lines = []
    n_ok = m.deconfound_self_check(cfg, lines.append)
    assert n_ok == 8, "自检条数应为 8，实得 %d" % n_ok
    assert [ln for ln in lines if ln.startswith("SELFCHECK_RESULT")] == []
    assert sum(1 for ln in lines if ln.startswith("SELFCHECK S")) == 8

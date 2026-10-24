# -*- coding: utf-8 -*-
"""tests/data/test_02_07_assert_splits.py — W3.7 自动断言、长度分布、n-gram 重叠（§5）。

覆盖 `windows/W3.7_assert.md` §5 的 12 项 ＋ §3.1 `W0` 补注第 4 条的 `A13` **四例回归**：

1. `A6` 两侧：440 → 通过；439／441 → 失败（转交项 1 的回归测试）
2. `A7` 两侧：7920 → 通过；7919 → 失败
3. `A1`：train/test 共用一个 `seed_group_id` → 失败
4. `A5`：`regime_ood/train` 含一条 `hex` → 失败；`val` 含一条 `morse` → 失败（两侧都测）
5. `A3`／`A4`：`regime_ood/test` 含一条 `base64` → 失败
6. 门禁 6 边界：`abs_diff_ratio = 0.199` → 通过；`0.201` → 失败
7. 门禁 7 边界：`overlap_rate = 0.049` → 通过；`0.051` → 失败
8. `A9`：正负比 0.79 → 失败；0.81 → 通过
9. `A10`／`A11`：`clean_original_test` 与训练样本 `text` 相等 → 失败；G3 test 的 group 越界 → 失败
10. `--check-only` 恒 `0`：输入缺失 → exit 0 且打印 `SKIP`；输入齐备 → exit 0 且**不写任何产物**
11. `split_assertions.csv` **恰 14 行**（不得缺行）
12. 输出 UTF-8 且**无孤立 CR**（字节法断言）
13. `A13` 四例：`N=220` 的 `176/22/22` → 通过；`177/22/21` → 失败；`N=200` 的 `160/20/20` → 通过；
    `N=185` 的 `148/18/19` → 通过、`148/19/18` → 失败（证明银行家舍入、且不依赖硬编码）
14. `A14` 非门禁：只测 `family_straddle.csv` 的行数与列名

**补正轮 r1 追加**（`D-36` 裁定一／二；`PREREG_v1.3.md` §1）：
15. **门禁 7 改报告项** —— 判定行 `passed` 恒 `n/a`、阈值未改、**不参与退出码**（`compute_exit_code`）
16. **②③ 对半控制**（`D-36` 转交项 42）—— 确定性、口径正确（③ 覆盖全分片、② 逐编码）、
    同分布零泄露对半仍给出高 `overlap_rate`、`also_runs` 缺目录即跳过

**离线**：全部输入在 `tmp_path` 内自造；长度口径以 `monkeypatch` 替换 `mod.tlen`
（`len(text)`）⇒ 测试**不加载 tokenizer、不触网**。`--check-only` 的断言用**子进程**跑，
以验证**真实退出码**（该模式按契约不加载 tokenizer）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_07_assert_splits.py")
CONFIG = os.path.join(ROOT, "configs", "data", "assert.yaml")

WIDE_FIELDS = ["sample_id", "seed_group_id", "benign_group_id", "sample_kind", "family_id",
               "rewrite_id", "seed_id", "source", "harm_category", "benign_set",
               "encoding_type", "carrier_type", "carrier_subtype", "host_context",
               "plaintext", "payload", "text", "difficulty", "label"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w37_assert", SCRIPT)


@pytest.fixture(scope="module")
def cfg(mod):
    conf, err = mod.load_config(CONFIG)
    assert err is None, err
    return conf


@pytest.fixture(autouse=True)
def fake_length(mod, monkeypatch):
    """长度口径替换为**确定性、离线**实现（= 字符数）：只测逻辑，不碰 tokenizer。"""
    monkeypatch.setattr(mod, "tlen", lambda text, tok: len(str(text)))
    mod._LEN_CACHE.clear()
    yield
    mod._LEN_CACHE.clear()


# ===========================================================================
# 夹具构造
# ===========================================================================
def _row(**kw):
    row = {f: "" for f in WIDE_FIELDS}
    row["label"] = kw.pop("label", "1")
    row["sample_kind"] = kw.pop("sample_kind", "attack")
    row.update(kw)
    return row


def _empty_shards():
    return {reg: {sp: [] for sp in ("train", "val", "test")} for reg in ("regime_id", "regime_ood")}


def _attack_rows(n_rewrites, per_rewrite, prefix="RW"):
    rows = []
    for i in range(1, n_rewrites + 1):
        rid = "%s-%04d" % (prefix, i)
        for j in range(per_rewrite):
            rows.append(_row(sample_id="ATK-%s%04d-%d" % (prefix, i, j), rewrite_id=rid,
                             seed_group_id="SG-%04d" % i, family_id="FAM-AH-%04d" % i,
                             sample_kind="attack", label="1",
                             encoding_type="base64", carrier_type="text",
                             plaintext="p-%d" % i, payload="b-%d" % i, text="t-%d-%d" % (i, j)))
    return rows


# ===========================================================================
# 1／2 · A6 与 A7 两侧（转交项 1 的回归测试）
# ===========================================================================
@pytest.mark.parametrize("n,expect", [(440, True), (439, False), (441, False)])
def test_a6_rewrite_id_count(mod, n, expect):
    shards = _empty_shards()
    shards["regime_id"]["train"] = _attack_rows(n, 1)
    ok, measured = mod.check_a6(shards, 440)
    assert ok is expect, measured
    assert ("= %d" % n) in measured


@pytest.mark.parametrize("n,expect", [(7920, True), (7919, False)])
def test_a7_sample_id_count(mod, n, expect):
    """7920 = 440 × 9 × 2；7919（人为少一条）必须失败。"""
    shards = _empty_shards()
    rows = _attack_rows(440, 18)
    shards["regime_id"]["train"] = rows[:n]
    ok, measured = mod.check_a7(shards, 7920)
    assert ok is expect, measured
    assert ("= %d" % n) in measured


def test_a7_requires_both_carriers(mod):
    """**两载体合并**才成立：只给 `text` 一侧（3960）必然失败。"""
    shards = _empty_shards()
    rows = []
    for i in range(1, 441):
        rid = "RW-%04d" % i
        for j in range(9):
            rows.append(_row(sample_id="ATK-RW%04d-%d-text" % (i, j), rewrite_id=rid,
                             sample_kind="attack", label="1", carrier_type="text"))
    shards["regime_id"]["train"] = rows
    ok, measured = mod.check_a7(shards, 7920)
    assert ok is False
    assert "3960" in measured


# ===========================================================================
# 3 · A1
# ===========================================================================
def test_a1_group_overlap_between_train_and_test_fails(mod):
    shards = _empty_shards()
    shards["regime_id"]["train"] = _attack_rows(2, 1)
    shards["regime_id"]["test"] = [_row(sample_id="ATK-x", rewrite_id="RW-0002",
                                         seed_group_id="SG-0002", sample_kind="attack")]
    ok, measured = mod.check_a1(shards)
    assert ok is False
    assert "train∩test" in measured


def test_a1_disjoint_groups_pass(mod):
    shards = _empty_shards()
    shards["regime_id"]["train"] = [_row(sample_id="a", seed_group_id="SG-0001")]
    shards["regime_id"]["val"] = [_row(sample_id="b", seed_group_id="SG-0002")]
    shards["regime_id"]["test"] = [_row(sample_id="c", seed_group_id="SG-0003")]
    ok, _ = mod.check_a1(shards)
    assert ok is True


# ===========================================================================
# 4 · A5（train 侧与 val 侧**都测**）
# ===========================================================================
@pytest.mark.parametrize("split,encoding", [("train", "hex"), ("val", "morse")])
def test_a5_heldout_encoding_in_train_or_val_fails(mod, split, encoding):
    shards = _empty_shards()
    shards["regime_ood"][split] = [_row(sample_id="X-1", sample_kind="attack", label="1",
                                        encoding_type=encoding, text="zzz")]
    ok, measured = mod.check_a5(shards)
    assert ok is False, measured
    assert "分类列命中=1" in measured


def test_a5_clean_regime_ood_passes(mod):
    shards = _empty_shards()
    shards["regime_ood"]["train"] = [_row(sample_id="X-1", sample_kind="attack", label="1",
                                          encoding_type="base64", text="aaa")]
    shards["regime_ood"]["val"] = [_row(sample_id="X-2", sample_kind="attack", label="1",
                                        encoding_type="rot13", text="bbb")]
    ok, measured = mod.check_a5(shards)
    assert ok is True, measured
    assert "分类列命中=0" in measured


def test_a5_content_scan_catches_mislabelled_text(mod):
    """列被误标为空、但 `text` 实为留出编码串 ⇒ 内容列扫描须拦下。"""
    shards = _empty_shards()
    shards["regime_id"]["test"] = [_row(sample_id="ATK-1", sample_kind="attack", label="1",
                                        encoding_type="hex", payload="PAYLOAD-HEX",
                                        text="TEXT-HEX", plaintext="PLAIN")]
    shards["regime_ood"]["train"] = [_row(sample_id="ATK-2", sample_kind="attack", label="1",
                                          encoding_type="", text="TEXT-HEX")]
    ok, measured = mod.check_a5(shards)
    assert ok is False, measured
    assert "内容列命中=1" in measured


# ===========================================================================
# 5 · A3 / A4
# ===========================================================================
def test_a3_ood_train_encoding_outside_train_set_fails(mod):
    shards = _empty_shards()
    shards["regime_ood"]["train"] = [_row(sample_id="X", encoding_type="hex")]
    ok, measured = mod.check_a3(shards)
    assert ok is False and "hex" in measured


def test_a4_ood_test_with_heldout_in_place_ok_but_train_encoding_fails(mod):
    shards = _empty_shards()
    shards["regime_ood"]["test"] = [_row(sample_id="X", encoding_type="hex")]
    ok, _ = mod.check_a4(shards)
    assert ok is True
    shards["regime_ood"]["test"] = [_row(sample_id="X", encoding_type="base64")]
    ok, measured = mod.check_a4(shards)
    assert ok is False and "base64" in measured


# ===========================================================================
# 6 · 门禁 6 边界
# ===========================================================================
@pytest.mark.parametrize("neg_len,expect", [(801, True), (799, False)])
def test_gate6_boundary(mod, cfg, neg_len, expect):
    """正样本各 1000 字符 ⇒ `abs_diff_ratio = |1000−neg|/1000`；801 → 0.199，799 → 0.201。"""
    shards = _empty_shards()
    shards["regime_id"]["train"] = [
        _row(sample_id="P1", sample_kind="attack", label="1", text="A" * 1000),
        _row(sample_id="N1", sample_kind="benign", label="0", text="B" * neg_len),
    ]
    ds = {"shards": shards}
    _, stats = mod.compute_length(ds, cfg, "fake-tokenizer")
    adr = stats["regime_id"]["train"]["abs_diff_ratio"]
    assert abs(adr - (199.0 / 1000.0 if expect else 201.0 / 1000.0)) < 1e-9
    gate = mod.gate6(stats)
    assert gate["passed"] is expect, gate["measured"]
    assert ("%.6f" % adr) in gate["measured"]


# ===========================================================================
# 7 · 门禁 7 边界（真算 n-gram，非只比阈值）
# ===========================================================================
def _craft_texts(shared, total):
    """返回 (train_text, test_text)：使 |ngram_char3(test)| == `total` 且 |A∩B| == `shared`。

    构造：test 用 `total+2` 个**互异**字符（CJK 段）⇒ 恰 `total` 个互异 3-gram；
    train 取同段前 `shared+2` 个字符（恰 `shared` 个 3-gram，全在 B 内），
    再接一段**另一起点、与 B 无交**的互异字符（边界 3-gram 亦含新字符 ⇒ 不入 B）。
    """
    base = 0x4E00
    test_text = "".join(chr(base + i) for i in range(total + 2))
    train_text = ("".join(chr(base + i) for i in range(shared + 2))
                  + "".join(chr(0x10000 + i) for i in range(20)))
    return train_text, test_text


@pytest.mark.parametrize("shared,total,expect", [(49, 1000, True), (51, 1000, False)])
def test_gate7_boundary(mod, cfg, shared, total, expect):
    """§5 第 7 条的**阈值比较**回归（`overlap_rate` 0.049 未超阈 / 0.051 超阈）。

    门禁 7 自 `D-36` 裁定一 起为**报告项** ⇒ 不再有「通过/不通过」；
    本测试退守到**阈值比较本身**（`below_threshold`）与其在 `ngram_overlap.csv` 中的落法
    （判定行 `passed` 恒 `n/a`、`threshold` 照旧写出）。
    """
    train_text, test_text = _craft_texts(shared, total)
    assert len(mod.char_ngrams(test_text, 3)) == total
    assert len(mod.char_ngrams(train_text, 3) & mod.char_ngrams(test_text, 3)) == shared
    shards = _empty_shards()
    shards["regime_id"]["train"] = [_row(sample_id="T1", text=train_text)]
    shards["regime_id"]["test"] = [_row(sample_id="E1", text=test_text)]
    rows, diag = mod.compute_ngram({"shards": shards}, cfg)
    rate = diag["regime_id"]["overlap_rate"]
    assert abs(rate - shared / float(total)) < 1e-12
    assert diag["regime_id"]["below_threshold"] is expect
    gate = mod.gate7(diag, cfg)
    assert gate["role"] == "report_item"
    assert gate["passed"] is None, "报告项恒无通过/不通过"
    assert gate["below_threshold"]["regime_id"] is expect
    judged = [r for r in rows if r["threshold"]]
    assert judged and all(r["passed"] == "n/a" for r in judged)
    assert all(r["threshold"] == "0.05" for r in judged)


def test_exit_code_ignores_gate7(mod, cfg, tmp_path):
    """门禁 7 越阈**不再**产生非零退出（`D-36` 裁定一第 3／4 条；`W3.8` 的 `H5` 要求 `exit=0`）。"""
    gate5_ok = {"gate": 5, "role": "gate", "passed": True}
    gate6_ok = {"gate": 6, "role": "gate", "passed": True}
    g7 = {"gate": 7, "role": "report_item", "passed": None}
    asserts = [{"assertion_id": "A%d" % i, "passed": "True"} for i in range(1, 14)]
    asserts.append({"assertion_id": "A14", "passed": "n/a"})
    assert mod.compute_exit_code([gate5_ok, gate6_ok, g7], asserts) == 0
    # 门禁 5 不通过 ⇒ 非零
    assert mod.compute_exit_code([{"gate": 5, "role": "gate", "passed": False}, gate6_ok, g7],
                                 asserts) == mod.EXIT_FAIL
    # 门禁类断言不通过 ⇒ 非零
    bad = [dict(a) for a in asserts]
    bad[0]["passed"] = "False"
    assert mod.compute_exit_code([gate5_ok, gate6_ok, g7], bad) == mod.EXIT_FAIL
    # `A14` 记 `n/a` **不得**触发非零
    only_a14_nan = [dict(a) for a in asserts]
    for a in only_a14_nan[:-1]:
        a["passed"] = "True"
    assert mod.compute_exit_code([gate5_ok, gate6_ok, g7], only_a14_nan) == 0


# ===========================================================================
# 8 · A9
# ===========================================================================
def _a9_shards(pos, neg):
    """把 `pos`/`neg` 条放进 `regime_id/train`，其余五个分片各放一组 1:1（比 1.0，恒在带内）。"""
    shards = _empty_shards()
    shards["regime_id"]["train"] = (
        [_row(sample_id="p%d" % i, label="1") for i in range(pos)]
        + [_row(sample_id="n%d" % i, label="0") for i in range(neg)])
    for regime in ("regime_id", "regime_ood"):
        for sp in ("train", "val", "test"):
            if regime == "regime_id" and sp == "train":
                continue
            shards[regime][sp] = [_row(sample_id="ok-%s-%s-p" % (regime, sp), label="1"),
                                  _row(sample_id="ok-%s-%s-n" % (regime, sp), label="0")]
    return shards


@pytest.mark.parametrize("pos,neg,expect", [(81, 100, True), (64, 100, False)])
def test_a9_band(mod, pos, neg, expect):
    """正 81 / 负 100 → 0.8100 通过；正 64 / 负 100 → 0.6400 失败。"""
    ok, measured = mod.check_a9(_a9_shards(pos, neg), 0.8, 1.25)
    assert ok is expect, measured


def test_a9_exact_0_79_fails_and_0_81_passes(mod):
    ok, measured = mod.check_a9(_a9_shards(79, 100), 0.8, 1.25)
    assert ok is False and "0.7900" in measured

    ok, measured = mod.check_a9(_a9_shards(81, 100), 0.8, 1.25)
    assert ok is True and "0.8100" in measured


# ===========================================================================
# 9 · A10 / A11
# ===========================================================================
def test_a10_clean_text_equal_to_train_text_fails(mod):
    shards = _empty_shards()
    shards["regime_id"]["train"] = [_row(sample_id="T", seed_group_id="SG-0001", text="SAME")]
    shards["regime_id"]["test"] = [_row(sample_id="E", seed_group_id="SG-0009", text="OTHER")]
    clean = [_row(sample_id="C", seed_group_id="SG-0009", text="SAME", sample_kind="attack")]
    ok, measured = mod.check_a10(shards, clean)
    assert ok is False and "相等者=1" in measured


def test_a10_clean_group_outside_test_groups_fails(mod):
    shards = _empty_shards()
    shards["regime_id"]["test"] = [_row(sample_id="E", seed_group_id="SG-0009", text="OTHER")]
    clean = [_row(sample_id="C", seed_group_id="SG-0777", text="NOVEL", sample_kind="attack")]
    ok, measured = mod.check_a10(shards, clean)
    assert ok is False and "越出测试 group 集者=1" in measured


def test_a11_g3_group_outside_test_benign_groups_fails(mod):
    shards = _empty_shards()
    shards["regime_id"]["test"] = [_row(sample_id="B", sample_kind="benign", label="0",
                                        benign_group_id="BG-0001")]
    ok, _ = mod.check_a11(shards, [_row(sample_id="G", benign_group_id="BG-0001")])
    assert ok is True
    ok, measured = mod.check_a11(shards, [_row(sample_id="G", benign_group_id="BG-0999")])
    assert ok is False and "越出测试良性 group 集者=1" in measured


# ===========================================================================
# 13 · A13 四例回归（§3.1 `W0` 补注第 4 条）
# ===========================================================================
@pytest.mark.parametrize("n_total,counts,expect", [
    (220, (176, 22, 22), True),    # 实测值
    (220, (177, 22, 21), False),   # 任一档挪一格 ⇒ 失败（等式口径）
    (200, (160, 20, 20), True),    # 证明不依赖硬编码
    (185, (148, 18, 19), True),    # 银行家舍入：18.5 → 18
    (185, (148, 19, 18), False),   # 四舍五入的结果 ⇒ 必须失败
])
def test_a13_derived_ratio(mod, cfg, n_total, counts, expect):
    rows = ([{"group_kind": "attack", "split": "train"} for _ in range(counts[0])]
            + [{"group_kind": "attack", "split": "val"} for _ in range(counts[1])]
            + [{"group_kind": "attack", "split": "test"} for _ in range(counts[2])]
            + [{"group_kind": "benign", "split": "train"} for _ in range(50)])
    ok, measured = mod.check_a13(rows, {"seed_count_total": n_total}, cfg)
    assert ok is expect, measured
    assert ("N=%d" % n_total) in measured


def test_a13_derived_expected_is_bankers_rounding(mod):
    assert mod.derived_expected(220, 0.8, 0.1) == (176, 22, 22)
    assert mod.derived_expected(185, 0.8, 0.1) == (148, 18, 19)
    assert mod.derived_expected(200, 0.8, 0.1) == (160, 20, 20)


# ===========================================================================
# 端到端：14 行 / UTF-8 / 无孤立 CR / --check-only
# ===========================================================================
def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if "_m" not in _write_csv.__dict__:
        _write_csv._m = _load("w37_common_w", os.path.join(ROOT, "scripts", "data", "_common.py"))
    _write_csv._m.write_csv(path, rows, fields)


ENC_TRAIN = ("base64", "rot13", "zerowidth", "url", "reverse")
ENC_HELD = ("hex", "morse", "ascii_art", "homoglyph")
ENCS = ENC_TRAIN + ENC_HELD
CARRIERS = ("text", "code")
SPLIT_OF_GROUP = {"SG-0001": "train", "SG-0002": "val", "SG-0003": "test"}


def _mk_attack_rows(mod):
    """3 个攻击 group × 1 改写体 × 9 编码 × 2 载体 = 每 group 18 行。"""
    rows = []
    for gid, split in SPLIT_OF_GROUP.items():
        rid = "RW-%04d" % int(gid.split("-")[1])
        for e in ENCS:
            for c in CARRIERS:
                rows.append(_row(
                    sample_id="ATK-%s-%s-%s" % (rid.replace("-", ""), e, c),
                    seed_group_id=gid, family_id="FAM-AH-%04d" % int(gid.split("-")[1]),
                    sample_kind="attack", label="1", encoding_type=e, carrier_type=c,
                    rewrite_id=rid, plaintext="plain-%s" % rid,
                    payload="payload-%s-%s" % (rid, e), text="text-%s-%s-%s" % (rid, e, c)))
    return rows


def _mk_benign_rows():
    rows = []
    for i, split in enumerate(("train", "val", "test"), start=1):
        rows.append(_row(sample_id="BN-%04d" % i, benign_group_id="BG-%04d" % i,
                         sample_kind="benign", label="0", benign_set="G1",
                         encoding_type="", text="benign-text-%d" % i))
    return rows


def _mk_dataset(mod, root):
    """在 `root` 内自造一整套 `data/interim/split_checks/` 形状的输入。"""
    split_dir = os.path.join(root, "split_checks")
    attack = _mk_attack_rows(mod)
    benign = _mk_benign_rows()
    by_split_a = {sp: [r for r in attack if SPLIT_OF_GROUP[r["seed_group_id"]] == sp]
                  for sp in ("train", "val", "test")}
    by_split_b = {sp: [r for r in benign if SPLIT_OF_GROUP[
        r["benign_group_id"].replace("BG-", "SG-")] == sp] for sp in ("train", "val", "test")}

    for regime in ("regime_id", "regime_ood"):
        for sp in ("train", "val", "test"):
            if regime == "regime_id":
                a = list(by_split_a[sp])
            elif sp == "test":
                a = [r for r in by_split_a[sp] if r["encoding_type"] in ENC_HELD]
            else:
                a = [r for r in by_split_a[sp] if r["encoding_type"] in ENC_TRAIN]
            rows = a + list(by_split_b[sp])
            _write_csv(os.path.join(split_dir, "splits", regime, sp + ".csv"), rows, WIDE_FIELDS)

    _write_csv(os.path.join(split_dir, "seed_groups.csv"),
               [{"group_id": g, "group_kind": "attack", "family_id": "FAM-AH-%04d" % int(g.split("-")[1]),
                 "split": s, "member_count": "18"} for g, s in SPLIT_OF_GROUP.items()]
               + [{"group_id": "BG-000%d" % i, "group_kind": "benign", "family_id": "BG-000%d" % i,
                   "split": sp, "member_count": "1"}
                  for i, sp in enumerate(("train", "val", "test"), start=1)],
               ["group_id", "group_kind", "family_id", "split", "member_count"])
    _write_csv(os.path.join(split_dir, "clean_original_test.csv"),
               [_row(sample_id="CLN-RW0003", seed_group_id="SG-0003", family_id="FAM-AH-0003",
                     rewrite_id="RW-0003", seed_id="SEED-AH-003", text="plain-RW-0003",
                     sample_kind="attack", label="1")],
               ["sample_id", "seed_group_id", "family_id", "rewrite_id", "seed_id",
                "source", "harm_category", "text", "label"])
    _write_csv(os.path.join(split_dir, "g3_hard_negative_test.csv"),
               [_row(sample_id="BN-G3-0001", benign_group_id="BG-0003", benign_set="G3",
                     sample_kind="benign", label="0", text="g3-text")],
               ["sample_id", "benign_group_id", "benign_set", "source", "source_item_id",
                "encoding_type", "carrier_type", "host_context", "g3_business_form",
                "business_necessity_evidence", "text", "label"])

    meta = os.path.join(root, "seeds.meta.json")
    with open(meta, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"seed_count_total": 3}, fh, ensure_ascii=False)
    ast_report = os.path.join(root, "ast_report.csv")
    _write_csv(ast_report, [{"carrier_subtype": "py_comment", "encoding_type": "base64",
                             "total": "18", "ast_ok": "18", "ast_failed": "0",
                             "regenerated": "0", "dropped": "0", "dropped_reason": ""}],
               ["carrier_subtype", "encoding_type", "total", "ast_ok", "ast_failed",
                "regenerated", "dropped", "dropped_reason"])
    return split_dir, meta, ast_report


def _cfg_no_alt_runs(cfg):
    """测试专用：清掉 `gate7.controls.also_runs`，避免端到端测试去读仓库里的真实旧划分（保持自包含）。"""
    conf = dict(cfg)
    conf["gate7"] = dict(cfg["gate7"])
    conf["gate7"]["controls"] = dict(cfg["gate7"]["controls"])
    conf["gate7"]["controls"]["also_runs"] = []
    return conf


def _run_inproc(mod, cfg, split_dir, out_dir, report_dir, log_dir, meta, ast_report):
    args = mod.parse_args(["--config", CONFIG, "--split-dir", split_dir, "--out", out_dir,
                           "--seeds-meta", meta, "--ast-report", ast_report,
                           "--report-dir", report_dir, "--log-dir", log_dir])
    return mod.run(args, _cfg_no_alt_runs(cfg), write_products=True)


def _dir_snapshot(path):
    snap = {}
    for base, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(base, f)
            with open(p, "rb") as fh:
                snap[os.path.relpath(p, path)] = hashlib.sha256(fh.read()).hexdigest()
    return snap


def test_end_to_end_assertions_csv_has_14_rows(mod, cfg, tmp_path):
    root = str(tmp_path)
    split_dir, meta, ast_report = _mk_dataset(mod, root)
    out_dir = os.path.join(root, "out")
    code, summary = _run_inproc(mod, cfg, split_dir, out_dir, os.path.join(root, "rep"),
                                os.path.join(root, "log"), meta, ast_report)
    assert summary is not None
    assert code in (0, 3)
    with open(os.path.join(out_dir, "split_assertions.csv"), "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln]
    assert len(lines) == 15, "表头 1 行 + 数据 14 行"
    ids = [ln.split(",")[0] for ln in lines[1:]]
    assert ids == ["A%d" % i for i in range(1, 15)]
    # A6/A7 是等式断言：本合成集只有 3 个改写体 ⇒ 必须为 False（**不得**被调通过）
    by_id = summary["assertions"]
    assert by_id["A6"]["passed"] == "False"
    assert by_id["A7"]["passed"] == "False"
    assert by_id["A14"]["passed"] == "n/a"


def test_products_are_utf8_without_lone_cr(mod, cfg, tmp_path):
    root = str(tmp_path)
    split_dir, meta, ast_report = _mk_dataset(mod, root)
    out_dir = os.path.join(root, "out")
    _run_inproc(mod, cfg, split_dir, out_dir, os.path.join(root, "rep"),
                os.path.join(root, "log"), meta, ast_report)
    for name in ("split_assertions.csv", "length_distribution.csv",
                 "ngram_overlap.csv", "family_straddle.csv", "ngram_controls.csv"):
        path = os.path.join(out_dir, name)
        with open(path, "rb") as fh:
            raw = fh.read()
        assert b"\r" not in raw, "%s 含 CR 字节" % name
        raw.decode("utf-8")  # 不抛即合法 UTF-8
    # `family_straddle.csv` 的行数与列名（`A14` 非门禁，只测这两项）
    with open(os.path.join(out_dir, "family_straddle.csv"), "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln]
    assert lines[0] == "family_id,group_count,train,val,test"
    assert len(lines) == 4, "3 个族 + 表头"


def test_ngram_csv_has_judged_and_comparison_rows(mod, cfg, tmp_path):
    root = str(tmp_path)
    split_dir, meta, ast_report = _mk_dataset(mod, root)
    out_dir = os.path.join(root, "out")
    _run_inproc(mod, cfg, split_dir, out_dir, os.path.join(root, "rep"),
                os.path.join(root, "log"), meta, ast_report)
    with open(os.path.join(out_dir, "ngram_overlap.csv"), "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln]
    assert lines[0] == "regime,n,level,train_ngrams,test_ngrams,overlap,overlap_rate,threshold,passed"
    body = [ln.split(",") for ln in lines[1:]]
    judged = [r for r in body if r[1] == "3" and r[2] == "char"]
    # 判定行为**报告项** ⇒ `passed` 恒 `n/a`（`D-36` 裁定一）；对照行留空
    assert len(judged) == 2 and all(r[8] == "n/a" for r in judged)
    others = [r for r in body if not (r[1] == "3" and r[2] == "char")]
    assert others and all(r[8] == "" for r in others), "对照行的 passed 列须留空"


# ===========================================================================
# 15 · ②③ 对半控制（`D-36` 转交项 42）
# ===========================================================================
def test_controls_deterministic_and_shaped(mod, cfg, tmp_path):
    """控制行**确定性**（同输入两次结果相同）、口径正确（③ 覆盖全分片、② 每个编码一行）。"""
    root = str(tmp_path)
    split_dir, _meta, _ast = _mk_dataset(mod, root)
    rows1 = mod.compute_controls(split_dir, cfg, "current")
    rows2 = mod.compute_controls(split_dir, cfg, "current")
    assert rows1 == rows2, "对半控制须确定性"
    assert rows1, "应产出控制行"
    assert set(rows1[0].keys()) == set(mod.CONTROL_FIELDS)
    for r in rows1:
        attack = [x for x in mod.read_csv(mod.split_path(split_dir, r["regime"], r["split"]))
                  if x["sample_kind"] == "attack"]
        if r["control"] == "random":
            assert r["n_a"] + r["n_b"] == len(attack)           # ③ 覆盖该分片全部攻击行
        else:
            same = [x for x in attack if (x["encoding_type"] or "") == r["key"]]
            assert r["n_a"] + r["n_b"] == len(same)             # ② 只覆盖该编码
    rnd = [r for r in rows1 if r["control"] == "random"]
    assert len(rnd) == 2, "③ 每个分片一行"
    assert all(r["key"] == "-" for r in rnd)
    byenc = [r for r in rows1 if r["control"] == "by_encoding"]
    assert byenc and all(r["key"] for r in byenc), "② 行须带 encoding_type"


def test_controls_same_distribution_zero_leak_high_rate(mod, cfg, tmp_path):
    """同分片对半（构造上零泄露）应给出**高** `overlap_rate` —— 即「同分布」与「跨分片」不可区分。"""
    root = str(tmp_path)
    split_dir, _meta, _ast = _mk_dataset(mod, root)
    rows = [r for r in mod.compute_controls(split_dir, cfg, "current") if r["control"] == "random"]
    assert rows
    for r in rows:
        assert float(r["overlap_rate"]) > 0.05, r


def test_controls_also_runs_skips_missing_dir(mod, cfg, tmp_path):
    """`also_runs` 指向不存在的目录 ⇒ 跳过、不报错。"""
    root = str(tmp_path)
    split_dir, _meta, _ast = _mk_dataset(mod, root)
    conf = dict(cfg)
    conf["gate7"] = dict(cfg["gate7"])
    conf["gate7"]["controls"] = dict(cfg["gate7"]["controls"])
    conf["gate7"]["controls"]["also_runs"] = [{"name": "nope", "dir": os.path.join(root, "missing")}]
    rows = mod.compute_controls(split_dir, conf, "current")
    assert [r for r in rows if r["run"] == "current"]


# ===========================================================================
# 10 · `--check-only`：恒 0、输入缺失打印 SKIP、就绪时不写任何产物
# ===========================================================================
def _run_check_only(split_dir, out_dir):
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--config", CONFIG, "--split-dir", split_dir,
         "--out", out_dir, "--check-only"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_check_only_missing_inputs_exits_zero_and_prints_skip(tmp_path):
    empty = os.path.join(str(tmp_path), "nothing")
    os.makedirs(empty, exist_ok=True)
    code, out = _run_check_only(empty, os.path.join(str(tmp_path), "out"))
    assert code == 0, out
    assert "SKIP" in out


def test_check_only_present_inputs_exits_zero_and_writes_nothing(mod, cfg, tmp_path):
    root = str(tmp_path)
    split_dir, _meta, _ast = _mk_dataset(mod, root)
    out_dir = os.path.join(root, "out")
    os.makedirs(out_dir, exist_ok=True)
    before = _dir_snapshot(out_dir)
    code, out = _run_check_only(split_dir, out_dir)
    assert code == 0, out
    assert "SKIP" not in out
    assert _dir_snapshot(out_dir) == before, "--check-only 不得写任何产物"


def test_check_only_without_config_still_exits_zero(tmp_path):
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--config", os.path.join(str(tmp_path), "nope.yaml"), "--check-only"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0
    assert "SKIP" in ((proc.stdout or "") + (proc.stderr or ""))


# ===========================================================================
# 17 · `F-1` 回归：`exit=0` 与 stdout 编码无关（`W3.7` 补正轮 r2）
# ===========================================================================
_F1_TMPL = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('m', r'{script}')\n"
    "m = importlib.util.module_from_spec(spec); sys.modules['m'] = m\n"
    "spec.loader.exec_module(m)\n"
    "{call}"
    "print('gate 7 \\u21d2 ok')\n"
)


def _run_in_encoding(tmp_path, call, encoding):
    """在**指定 stdout 编码**（`PYTHONIOENCODING`）下跑一小段脚本，返回退出码。

    用 `PYTHONIOENCODING=<非 UTF-8>` 显式制造「非 UTF-8 stdout」——比依赖本机 locale 更确定，
    任何平台都可复现。
    """
    snippet = _F1_TMPL.format(script=SCRIPT, call=call)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = encoding
    out = tmp_path / ("o_%s_%s.txt" % (encoding, "on" if call else "off"))
    with open(out, "wb") as fh:
        proc = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT, stdout=fh,
                              stderr=subprocess.PIPE, env=env)
    return proc.returncode, (proc.stderr or b"").decode("utf-8", "replace")


def test_f1_negative_control_gbk_stdout_without_fix_fails(tmp_path):
    """**反例（证明本测试有效）**：不经 `init_streams()` ⇒ GBK stdout 下打印 `⇒` 抛错、退出非 0。"""
    code, err = _run_in_encoding(tmp_path, "", "gbk")
    assert code != 0, "未经 init_streams 的 GBK stdout 应失败（反例失效 ⇒ 本回归无判别力）"
    assert "UnicodeEncodeError" in err


@pytest.mark.parametrize("encoding", ["gbk", "utf-8"])
def test_f1_init_streams_makes_stdout_encoding_independent(tmp_path, encoding):
    """`init_streams()` 后，**任何** stdout 编码下打印 `⇒` 均成功（`exit=0`）—— `F-1` 的正面回归。"""
    code, err = _run_in_encoding(tmp_path, "m.init_streams()\n", encoding)
    assert code == 0, err


def test_report_only_does_not_rewrite_csv(mod, cfg, tmp_path):
    root = str(tmp_path)
    split_dir, meta, ast_report = _mk_dataset(mod, root)
    out_dir = os.path.join(root, "out")
    _run_inproc(mod, cfg, split_dir, out_dir, os.path.join(root, "rep"),
                os.path.join(root, "log"), meta, ast_report)
    before = _dir_snapshot(out_dir)
    args = mod.parse_args(["--config", CONFIG, "--split-dir", split_dir, "--out", out_dir,
                           "--seeds-meta", meta, "--ast-report", ast_report,
                           "--report-dir", os.path.join(root, "rep"),
                           "--log-dir", os.path.join(root, "log")])
    mod.run(args, _cfg_no_alt_runs(cfg), write_products=False)
    assert _dir_snapshot(out_dir) == before


def test_assertion_specs_cover_a1_to_a14(mod):
    ids = [a[0] for a in mod.ASSERTION_SPECS]
    assert ids == ["A%d" % i for i in range(1, 15)]
    assert mod.GATING_ASSERTIONS == tuple("A%d" % i for i in range(1, 14))
    scopes = {a[1] for a in mod.ASSERTION_SPECS}
    assert scopes <= {"attack", "benign", "cross", "regime_id", "regime_ood", "global"}

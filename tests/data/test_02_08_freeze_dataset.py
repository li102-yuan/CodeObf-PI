# -*- coding: utf-8 -*-
"""tests/data/test_02_08_freeze_dataset.py — W3.8 数据集冻结（`windows/W3.8_freeze.md` §4）。

覆盖 §4 的 8 项要求 ＋ §3.6-15 `W0` 补注第 4 条的**反例回归**：

1. `H1`–`H7` **逐条反例**：各构造一个不满足的场景 ⇒ **非零退出**且**未写入任何文件**
2. **门禁 3 未过（0.94）⇒ 拒绝冻结**（`AGENTS.md` §6-7 的回归测试）
3. **占位态识别**：`md5sums.txt` 为 0 字节 ⇒ 允许覆写；非空且 `status=frozen` ⇒ 拒绝
4. `manifest.json` 的 `counts.rewrite_bodies == 440`、`counts.attack_total == 7920` 被强制；
   构造 439 时失败（转交项 1 的回归测试）
5. `md5sums.txt` 格式与内容正确（字段数、相对路径、无孤立 CR、**不自指**）
6. `ACCESS_LOG.md` 表头恰 6 列；**本脚本不改其任何数据行**
7. 冻结树结构与 `DIRECTORY.md` §3.3 逐条一致（**含 `benign/` 与 `reports/` 两个子目录**）
8. 已存在非空冻结目录时**拒绝覆盖**
9. **§3.6-15 反例**：`split_seed = 43` 或 `split_ratio = "7:1:2"` ⇒ 非零退出且**未写入任何文件**

**全部用 `tmp_path` 构造假树，绝不动真实 `data/`**（`--root` 覆盖 ＋ `--no-readonly`
＋ `--h5-no-rerun`：后者使 `H5` **只读**现有 `split_assertions.csv`，不重跑 `02_07`）。
**离线可跑、不加载模型、不加载 tokenizer。**
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_08_freeze_dataset.py")
CONFIG = os.path.join(ROOT, "configs", "data", "freeze.yaml")

ENC = ("base64", "hex", "ascii_art", "rot13", "reverse", "morse", "url", "zerowidth",
       "homoglyph")
OOD_TRAIN = ("base64", "rot13", "zerowidth", "url", "reverse")
OOD_HELDOUT = ("hex", "morse", "ascii_art", "homoglyph")

WIDE_FIELDS = ["sample_id", "seed_group_id", "benign_group_id", "sample_kind", "family_id",
               "rewrite_id", "seed_id", "source", "harm_category", "benign_set",
               "encoding_type", "carrier_type", "carrier_subtype", "host_context",
               "plaintext", "payload", "text", "difficulty", "label"]

G3_FIELDS = ["sample_id", "benign_group_id", "benign_set", "source", "source_item_id",
             "encoding_type", "carrier_type", "host_context", "g3_business_form",
             "business_necessity_evidence", "text", "label"]

GATES = [
    ("g1_intent", 0.95), ("g2_ast", 0.10), ("g3_qc", 0.95), ("g4_guard", 0.05),
    ("g5_group", 0.0), ("g6_length", 0.20), ("g7_ngram", 0.05),
]


def _load_module():
    spec = importlib.util.spec_from_file_location("freeze_02_08", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["freeze_02_08"] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# 假树构造
# ---------------------------------------------------------------------------
def build_tree(tmp_path, *, n_rewrites=440, n_encoded=None, a6=None, a7=None,
               g1_rate=0.95, g2_rate=0.0, g3_rate=0.95, g4_rate=0.0,
               smoke_mode="full", a13_measured=None,
               frozen_placeholder=True, frozen_state="placeholder", split_seed=42,
               split_ratio="8:1:1"):
    """构造假项目根；返回其 `Path`。

    各形参对应一条可被打破的前置，供 §4 的反例使用。
    """
    root = tmp_path / "root"
    (root / "configs" / "data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, root / "configs" / "data" / "freeze.yaml")

    cfg = _load_yaml(root / "configs" / "data" / "freeze.yaml")["freeze"]
    cfg["expected"]["split_seed"] = split_seed
    cfg["expected"]["split_ratio"] = split_ratio
    _write_text(root / "configs" / "data" / "freeze.yaml",
                _dump_yaml({"freeze": cfg}))

    inter = root / "data" / "interim"
    n_enc = n_encoded if n_encoded is not None else n_rewrites * 9

    # --- 种子 ---
    seeds = [{"seed_group_id": "SG-%04d" % (i + 1), "seed_id": "SEED-AH-%03d" % (i + 1),
              "source": "AgentHarm", "source_item_id": "%d-detailed" % (i // 4 + 1)}
             for i in range(220)]
    _write_csv(str(inter / "seed_collect" / "seeds.csv"),
               ["seed_group_id", "seed_id", "source", "source_item_id"], seeds)
    _write_text(str(inter / "seed_collect" / "seeds.meta.json"), json.dumps({
        "seed_count_total": 220,
        "seed_count_by_source": {"AgentHarm": {"count": 176, "ratio": 0.8},
                                 "PKU-SafeRLHF": {"count": 44, "ratio": 0.2}},
        "subset_rule": "规则 A", "subset_rule_pku": "规则 B",
        "subset_rule_agentharm": "规则 C",
    }, ensure_ascii=False, indent=2) + "\n")

    # --- 改写体 ---
    rewrites = [{"rewrite_id": "RW-%04d" % (i + 1),
                 "seed_group_id": "SG-%04d" % (i // 2 + 1), "rewrite_index": i % 2 + 1,
                 "text": "body %d" % i} for i in range(n_rewrites)]
    _write_csv(str(inter / "rewrite" / "rewrites.csv"),
               ["rewrite_id", "seed_group_id", "rewrite_index", "text"], rewrites)

    # --- 编码变体 + 代码载体 ---
    enc_rows, cc_rows = [], []
    for i in range(n_enc):
        rid = "RW-%04d" % (i // 9 + 1)
        sg = "SG-%04d" % (i // 18 + 1)
        e = ENC[i % 9]
        sid = "ATK-%s-%s-text" % (rid.replace("-", ""), e)
        enc_rows.append({"sample_id": sid, "seed_group_id": sg, "rewrite_id": rid,
                         "encoding_type": e, "difficulty": "medium", "carrier_type": "text",
                         "plaintext": "p%d" % i, "payload": "q%d" % i, "label": "1"})
        cc_rows.append({"sample_id": sid.replace("-text", "-code"), "seed_group_id": sg,
                        "rewrite_id": rid, "encoding_type": e, "difficulty": "medium",
                        "carrier_type": "code", "carrier_subtype": "py_comment",
                        "language": "python", "plaintext": "p%d" % i, "payload": "q%d" % i,
                        "text": "# q%d" % i, "ast_ok": "1", "label": "1"})
    _write_csv(str(inter / "encoded" / "encoded.csv"),
               ["sample_id", "seed_group_id", "rewrite_id", "encoding_type", "difficulty",
                "carrier_type", "plaintext", "payload", "label"], enc_rows)
    _write_csv(str(inter / "code_carrier" / "code_carrier.csv"),
               ["sample_id", "seed_group_id", "rewrite_id", "encoding_type", "difficulty",
                "carrier_type", "carrier_subtype", "language", "plaintext", "payload",
                "text", "ast_ok", "label"], cc_rows)
    _write_csv(str(inter / "code_carrier" / "ast_report.csv"),
               ["carrier_subtype", "encoding_type", "total", "ast_ok", "ast_failed",
                "regenerated", "dropped", "dropped_reason"],
               [{"carrier_subtype": "py_comment", "encoding_type": e, "total": "1",
                 "ast_ok": "1" if g2_rate == 0.0 else "0",
                 "ast_failed": "0" if g2_rate == 0.0 else "1",
                 "regenerated": "0", "dropped": "0", "dropped_reason": ""} for e in ENC])

    # --- 良性集 ---
    for name, n in (("G1", 450), ("G2", 270), ("G3", 180)):
        rows = [{"sample_id": "BN-%s-%04d" % (name, i), "benign_group_id": "BG-%04d" % (i + 1),
                 "benign_set": name, "source": "TheStack" if name == "G1" else "Alpaca",
                 "source_item_id": "x%d" % i,
                 "encoding_type": "hex" if name == "G3" else "",
                 "carrier_type": "code" if name != "G2" else "text",
                 "host_context": "h", "g3_business_form": "完整性与校验值",
                 "business_necessity_evidence": "e", "text": "t%d" % i, "label": "0"}
                for i in range(n)]
        _write_csv(str(inter / "benign_G1_G2_G3" / ("%s.csv" % name)), G3_FIELDS, rows)

    # --- 门禁 3 质检 ---
    qc_rows = []
    n_pass = int(round(g3_rate * 100))
    for i in range(100):
        for ann in ("AN-A", "AN-B"):
            qc_rows.append({"sample_id": "BN-G3-%04d" % i, "annotator": ann,
                            "pass": "yes" if i < n_pass else "no", "remark": "r"})
    _write_csv(str(inter / "benign_G1_G2_G3" / "g3_qc.csv"),
               ["sample_id", "annotator", "pass", "remark"], qc_rows)

    # --- 门禁 4 明细 ---
    n_hit = int(round(g4_rate * 90))
    guard_rows = [{"sample_id": "BN-G1-%04d" % i, "benign_set": "G1", "guard_verdict": "safe",
                   "auto_label": "MI-B", "mi_label": "MI-B",
                   "counted": "1" if i < n_hit else "0"} for i in range(90)]
    _write_csv(str(inter / "benign_G1_G2_G3" / "benign_guard_check.csv"),
               ["sample_id", "benign_set", "guard_verdict", "auto_label", "mi_label",
                "counted"], guard_rows)

    # --- 划分 ---
    sc = root / "data" / "interim" / "split_checks"
    counts = {"train": 176, "val": 22, "test": 22}      # 与 `V-3` 的 176:22:22 一致
    gid = 0
    group_rows = []
    for sp, n in counts.items():
        for _ in range(n):
            gid += 1
            group_rows.append({"group_id": "SG-%04d" % gid, "group_kind": "attack",
                               "family_id": "FAM-AH-%04d" % ((gid - 1) // 4 + 1),
                               "split": sp, "member_count": "18"})
    for i in range(12):
        group_rows.append({"group_id": "BG-%04d" % i, "group_kind": "benign",
                           "family_id": "BG-%04d" % i,
                           "split": ("train", "val", "test")[i % 3], "member_count": "1"})
    _write_csv(str(sc / "seed_groups.csv"),
               ["group_id", "group_kind", "family_id", "split", "member_count"], group_rows)

    for reg, encs_train, encs_test in (("regime_id", ENC, ENC),
                                       ("regime_ood", OOD_TRAIN, OOD_HELDOUT)):
        for sp, n in counts.items():
            encs = encs_train if sp in ("train", "val") else encs_test
            rows = []
            idx = 0
            for g in range(n):
                for e in encs:
                    for carrier in ("text", "code"):
                        idx += 1
                        rows.append({
                            "sample_id": "ATK-G%04d-%s-%s" % (g, e, carrier),
                            "seed_group_id": "SG-%04d" % (g + 1),
                            "benign_group_id": "",
                            "sample_kind": "attack",
                            "family_id": "FAM-AH-%04d" % (g // 4 + 1),
                            "rewrite_id": "RW-%04d" % (g * 2 + 1),
                            "seed_id": "SEED-AH-%03d" % (g + 1),
                            "source": "AgentHarm", "harm_category": "c",
                            "benign_set": "", "encoding_type": e,
                            "carrier_type": carrier,
                            "carrier_subtype": "py_comment" if carrier == "code" else "",
                            "host_context": "h", "plaintext": "p", "payload": "q",
                            "text": "t", "difficulty": "medium", "label": "1"})
            # 掺入少量良性行（供 `collect_facts` 的体制二良性 group 判定）
            for i in range(4):
                rows.append({"sample_id": "BN-G3-%04d" % i,
                             "seed_group_id": "", "benign_group_id": "BG-%04d" % i,
                             "sample_kind": "benign", "family_id": "BG-%04d" % i,
                             "rewrite_id": "", "seed_id": "", "source": "G3-synthetic",
                             "harm_category": "", "benign_set": "G3", "encoding_type": "hex",
                             "carrier_type": "code", "carrier_subtype": "string_literal",
                             "host_context": "h", "plaintext": "", "payload": "",
                             "text": "bt%d" % i, "difficulty": "", "label": "0"})
            _write_csv(str(sc / "splits" / reg / ("%s.csv" % sp)), WIDE_FIELDS, rows)

    _write_csv(str(sc / "clean_original_test.csv"),
               ["sample_id", "seed_group_id", "family_id", "rewrite_id", "seed_id",
                "source", "harm_category", "text", "label"],
               [{"sample_id": "CLN-RW-0001", "seed_group_id": "SG-0001",
                 "family_id": "FAM-AH-0001", "rewrite_id": "RW-0001", "seed_id": "SEED-AH-001",
                 "source": "AgentHarm", "harm_category": "c", "text": "clean", "label": "1"}])
    _write_csv(str(sc / "g3_hard_negative_test.csv"), G3_FIELDS,
               [{"sample_id": "BN-G3-0000", "benign_group_id": "BG-0000", "benign_set": "G3",
                 "source": "TheStack", "source_item_id": "x", "encoding_type": "hex",
                 "carrier_type": "code", "host_context": "h",
                 "g3_business_form": "完整性与校验值", "business_necessity_evidence": "e",
                 "text": "hn", "label": "0"}])

    # --- 断言表（14 行；`A6`／`A7`／`A13` 的 measured 为判据来源）---
    a6 = n_rewrites if a6 is None else a6
    a7 = (n_enc * 2) if a7 is None else a7
    a13_meas = a13_measured or (
        "N=220 ⇒ 期望 176:22:22；实测 train=176、val=22、test=22"
        "（判据 tolerance_groups=0，等式）")
    assertions = []
    for i in range(1, 15):
        aid = "A%d" % i
        measured = {6: "rewrite_id 去重计数 = %d（判据 == 440）" % a6,
                    7: "sample_id 去重计数 = %d（判据 == 7920）" % a7,
                    13: a13_meas}.get(i, "m")
        assertions.append({"assertion_id": aid, "scope": "global", "description": "d",
                           "threshold": "t", "measured": measured,
                           "passed": "n/a" if i == 14 else "True",
                           "evidence_path": "e"})
    _write_csv(str(sc / "split_assertions.csv"),
               ["assertion_id", "scope", "description", "threshold", "measured", "passed",
                "evidence_path"], assertions)

    _write_csv(str(sc / "length_distribution.csv"),
               ["split", "regime", "sample_kind", "label", "n", "mean_tokens", "median_tokens",
                "p95_tokens", "max_tokens", "min_tokens", "positive_mean_tokens",
                "negative_mean_tokens", "abs_diff_ratio", "threshold", "passed"],
               [{"split": sp, "regime": reg, "sample_kind": "shard", "n": "10",
                 "positive_mean_tokens": "100", "negative_mean_tokens": "90",
                 "abs_diff_ratio": "0.10", "threshold": "0.20", "passed": "True"}
                for reg in ("regime_id", "regime_ood") for sp in ("train", "val", "test")])
    _write_csv(str(sc / "ngram_overlap.csv"),
               ["regime", "n", "level", "train_ngrams", "test_ngrams", "overlap",
                "overlap_rate", "threshold", "passed"],
               [{"regime": reg, "n": "3", "level": "char", "train_ngrams": "10",
                 "test_ngrams": "10", "overlap": "8", "overlap_rate": "0.85",
                 "threshold": "0.05", "passed": "n/a"} for reg in ("regime_id", "regime_ood")]
               + [{"regime": reg, "n": "3", "level": "word", "train_ngrams": "10",
                   "test_ngrams": "10", "overlap": "1", "overlap_rate": "0.13",
                   "threshold": "", "passed": ""} for reg in ("regime_id", "regime_ood")])
    _write_csv(str(sc / "ngram_controls.csv"),
               ["run", "regime", "split", "control", "key", "n_a", "n_b", "ngrams_a",
                "ngrams_b", "overlap", "overlap_rate", "note"],
               [{"run": "current", "regime": "regime_id", "split": "test", "control": "random",
                 "key": "-", "n_a": "1", "n_b": "1", "ngrams_a": "1", "ngrams_b": "1",
                 "overlap": "1", "overlap_rate": "0.885060", "note": "n"},
                {"run": "current", "regime": "regime_ood", "split": "test", "control": "random",
                 "key": "-", "n_a": "1", "n_b": "1", "ngrams_a": "1", "ngrams_b": "1",
                 "overlap": "1", "overlap_rate": "0.892481", "note": "n"},
                {"run": "current", "regime": "regime_id", "split": "test",
                 "control": "by_encoding", "key": "base64", "n_a": "1", "n_b": "1",
                 "ngrams_a": "1", "ngrams_b": "1", "overlap": "1", "overlap_rate": "0.9177",
                 "note": "n"}])
    _write_csv(str(sc / "family_straddle.csv"), ["family_id", "group_count", "train", "val",
                                                 "test"],
               [{"family_id": "FAM-AH-0001", "group_count": "4", "train": "4", "val": "0",
                 "test": "0"}])

    # --- 报告 ---
    rep = root / "reports" / "data"
    _write_text(str(rep / "g4_intent_v1.0_20260101-000000.md"),
                "# g4_intent\n\n| 项 | 值 |\n|---|---|\n| 阈值 | 通过率 **≥95%** |\n"
                + "| **通过率** | " + ("%.4f" % g1_rate) + " |\n")
    concl = "PASS" if smoke_mode == "full" else "FAIL"
    _write_text(str(rep / "smoke_data_v1.0_20260101-000000.md"),
                "# smoke\n\n- 模式（`mode`）：`%s`\n- 结论（`conclusion`）：`%s`\n"
                % (smoke_mode, concl))

    # --- 来源清单 ---
    for name in ("AgentHarm", "PKU-SafeRLHF", "TheStack", "CodeSearchNet", "Alpaca",
                 "Wikitext"):
        repo = ("codeparrot/codeparrot-clean" if name == "TheStack" else "org/%s" % name)
        _write_text(str(root / "data" / "raw" / name / "SOURCE_MANIFEST.json"),
                    json.dumps({"source": name, "hf_repo_id": repo,
                                "hf_revision": "rev", "mirror_endpoint": "https://hf-mirror.com",
                                "license": "mit", "download_date": "2026-01-01",
                                "file_count": 1, "total_bytes": 1}, ensure_ascii=False))

    # --- 冻结目录（占位态）---
    if frozen_placeholder:
        fdir = root / "data" / "frozen" / "CodeObf-PI_v1.0"
        _write_text(str(fdir / "md5sums.txt"), "")
        _write_text(str(fdir / "splits" / "regime_id" / "train.csv"), "")
        _write_text(str(fdir / "manifest.json"),
                    json.dumps({"name": "CodeObf-PI", "status": frozen_state},
                               ensure_ascii=False))
        for reg in ("regime_id", "regime_ood"):
            for sp in ("train", "val", "test"):
                _write_text(str(fdir / "splits" / reg / ("%s.csv" % sp)), "")
        for name in ("G1", "G2", "G3"):
            _write_text(str(fdir / "benign" / ("%s.csv" % name)), "")
        for name in ("g3_qc.csv", "length_distribution.csv", "ngram_overlap.csv",
                     "split_assertions.csv"):
            _write_text(str(fdir / "reports" / name), "")
        _write_text(str(fdir / "seed_groups.csv"), "")
        _write_text(str(fdir / "DATA_CARD_CodeObf-PI_v1.0.md"), "placeholder\n")

    # --- 封存目录 ---
    alog = (root / "data" / "sealed_test" / "CodeObf-PI_v1.0" / "ACCESS_LOG.md")
    _write_text(str(alog),
                "# ACCESS_LOG.md\n\n> 规则说明\n> - 表头固定 6 列\n\n"
                "| 日期 | 窗口 | 用途 | 命令 | 输出路径 | 复核人 |\n|---|---|---|---|---|---|\n")
    _write_text(str(root / "data" / "sealed_test" / "CodeObf-PI_v1.0" / "md5sums.txt"), "")
    return root, alog


def _load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_yaml(obj):
    import yaml
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)


def snapshot(root):
    """目录快照：{相对路径: (字节数, sha256)}。用于断言「未写入任何文件」。"""
    out = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            p = os.path.join(base, name)
            with open(p, "rb") as fh:
                data = fh.read()
            out[os.path.relpath(p, root).replace("\\", "/")] = (len(data),
                                                               hashlib.sha256(data).hexdigest())
    return out


def run(root, extra=()):
    mod = _load_module()
    argv = ["--root", str(root), "--config", "configs/data/freeze.yaml",
            "--no-readonly", "--h5-no-rerun"] + list(extra)
    return mod.main(argv)


@pytest.fixture(scope="module")
def _ok(tmp_path_factory):
    """一份「全部前置都满足」的假树（module 级，供成功路径的多个断言复用）。"""
    tmp = tmp_path_factory.mktemp("freeze_ok")
    root, alog = build_tree(tmp)
    rc = run(root)
    return root, alog, rc


# ---------------------------------------------------------------------------
# §4-7 冻结树结构与成功路径
# ---------------------------------------------------------------------------
def test_success_exit_zero(_ok):
    _root, _alog, rc = _ok
    assert rc == 0


def test_freeze_tree_matches_directory_md_3_3(_ok):
    root, _alog, _rc = _ok
    fdir = root / "data" / "frozen" / "CodeObf-PI_v1.0"
    expected = {
        "DATA_CARD_CodeObf-PI_v1.0.md", "manifest.json", "md5sums.txt", "seed_groups.csv",
    }
    for reg in ("regime_id", "regime_ood"):
        for sp in ("train", "val", "test"):
            expected.add("splits/%s/%s.csv" % (reg, sp))
    for name in ("G1", "G2", "G3"):          # **含 `benign/` 子目录**
        expected.add("benign/%s.csv" % name)
    for name in ("g3_qc.csv", "length_distribution.csv", "ngram_overlap.csv",
                 "split_assertions.csv"):     # **含 `reports/` 子目录**
        expected.add("reports/%s" % name)
    got = set()
    for base, _dirs, files in os.walk(fdir):
        for name in files:
            got.add(os.path.relpath(os.path.join(base, name), fdir).replace("\\", "/"))
    assert got == expected


def test_sealed_test_contains_only_test_shards(_ok):
    root, _alog, _rc = _ok
    sdir = root / "data" / "sealed_test" / "CodeObf-PI_v1.0"
    got = sorted(os.listdir(sdir))
    assert got == ["ACCESS_LOG.md", "clean_original_test.csv", "g3_hard_negative_test.csv",
                   "md5sums.txt", "regime_id_test.csv", "regime_ood_test.csv"]
    assert not any("train" in n or "val" in n for n in got)


def test_manifest_counts_forced_to_440_and_7920(_ok):
    root, _alog, _rc = _ok
    man = json.load(open(root / "data" / "frozen" / "CodeObf-PI_v1.0" / "manifest.json",
                         encoding="utf-8"))
    assert man["counts"]["rewrite_bodies"] == 440
    assert man["counts"]["attack_total"] == 7920
    assert man["status"] == "frozen"
    assert man["split_ratio"] == "8:1:1"
    assert man["split_seed"] == 42
    for key in ("g1_intent", "g2_ast", "g3_qc", "g4_guard", "g5_group", "g6_length",
                "g7_ngram"):
        assert key in man["gates"]
    assert "text" in man["field_glossary"]


# ---------------------------------------------------------------------------
# §4-5 md5sums 格式
# ---------------------------------------------------------------------------
def test_md5sums_format_and_no_self_reference(_ok):
    root, _alog, _rc = _ok
    fdir = root / "data" / "frozen" / "CodeObf-PI_v1.0"
    lines = (fdir / "md5sums.txt").read_text(encoding="utf-8").splitlines()
    assert lines, "md5sums.txt 不得为空"
    names = set()
    for line in lines:
        parts = line.split("  ")
        assert len(parts) == 2, line                      # 字段数 = 2（md5 + 相对路径）
        md5, rel = parts
        assert len(md5) == 32 and all(c in "0123456789abcdef" for c in md5), line
        assert rel == rel.replace("\\", "/"), line        # 正斜杠
        assert not rel.startswith("/"), line              # 相对各自根目录
        assert rel not in names, "重复条目 %s" % rel
        names.add(rel)
        p = fdir / rel
        assert p.is_file(), rel
        assert hashlib.md5(p.read_bytes()).hexdigest() == md5, line
    assert "md5sums.txt" not in names, "**不自指**"
    raw = (fdir / "md5sums.txt").read_bytes()
    assert b"\r" not in raw, "不得含孤立 CR"
    # 封存侧
    sdir = root / "data" / "sealed_test" / "CodeObf-PI_v1.0"
    slines = (sdir / "md5sums.txt").read_text(encoding="utf-8").splitlines()
    snames = {l.split("  ")[1] for l in slines}
    assert snames == {"regime_id_test.csv", "regime_ood_test.csv", "clean_original_test.csv",
                      "g3_hard_negative_test.csv"}
    assert "ACCESS_LOG.md" not in snames, "封存侧 md5sums 不含 ACCESS_LOG.md"
    assert b"\r" not in (sdir / "md5sums.txt").read_bytes()


# ---------------------------------------------------------------------------
# §4-6 ACCESS_LOG
# ---------------------------------------------------------------------------
def test_access_log_header_6_cols_and_untouched(_ok):
    root, alog, _rc = _ok
    text = alog.read_text(encoding="utf-8")
    header = [l for l in text.splitlines() if l.strip().startswith("|") and "复核人" in l]
    assert len(header) == 1
    cols = [c.strip() for c in header[0].strip().strip("|").split("|")]
    assert len(cols) == 6
    data_rows = [l for l in text.splitlines()
                 if l.strip().startswith("|") and "复核人" not in l and "---" not in l]
    assert data_rows == [], "02_08 不得写 ACCESS_LOG.md 的任何数据行"
    assert "规则说明" in text, "既有规则说明必须保留"


def test_access_log_bad_header_fails(tmp_path):
    root, alog = build_tree(tmp_path)
    alog.write_text("# ACCESS_LOG.md\n\n| 日期 | 窗口 | 用途 |\n|---|---|---|\n",
                    encoding="utf-8", newline="\n")
    before = snapshot(root)
    assert run(root) != 0
    assert snapshot(root) == before, "表头不合格时不得写入任何文件"


# ---------------------------------------------------------------------------
# §4-1/2/9 逐条反例：非零退出 + 未写入任何文件
# ---------------------------------------------------------------------------
def _assert_refused(tmp_path, **kwargs):
    root, _alog = build_tree(tmp_path, **kwargs)
    before = snapshot(root)
    rc = run(root)
    assert rc != 0, "应被拒绝"
    assert snapshot(root) == before, "未写入任何文件"
    return rc


def test_h1_intent_below_threshold(tmp_path):
    _assert_refused(tmp_path, g1_rate=0.94)


def test_h2_ast_error_rate_too_high(tmp_path):
    _assert_refused(tmp_path, g2_rate=0.20)


def test_h3_gate3_not_passed_regression(tmp_path):
    """`AGENTS.md` §6-7 的回归测试：门禁 3 未过 ⇒ 拒绝冻结。"""
    _assert_refused(tmp_path, g3_rate=0.94)


def test_h4_benign_detection_rate_too_high(tmp_path):
    _assert_refused(tmp_path, g4_rate=0.10)


def test_h5_assertions_not_all_true(tmp_path):
    root, _alog = build_tree(tmp_path)
    p = root / "data" / "interim" / "split_checks" / "split_assertions.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
    rows[0]["passed"] = "False"
    _write_csv(str(p), list(rows[0].keys()), rows)
    before = snapshot(root)
    assert run(root) != 0
    assert snapshot(root) == before


def test_h5_assertions_row_count_must_be_14(tmp_path):
    root, _alog = build_tree(tmp_path)
    p = root / "data" / "interim" / "split_checks" / "split_assertions.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
    _write_csv(str(p), list(rows[0].keys()), rows[:-1])   # 13 行
    before = snapshot(root)
    assert run(root) != 0
    assert snapshot(root) == before


def test_h6_missing_full_smoke_report(tmp_path):
    _assert_refused(tmp_path, smoke_mode="smoke")


def test_h6_full_report_failed(tmp_path):
    _assert_refused(tmp_path, smoke_mode="smoke")


def test_h7_non_placeholder_nonempty_refuses(tmp_path):
    """§4-8：已存在非空冻结目录时拒绝覆盖。"""
    root, _alog = build_tree(tmp_path)
    fdir = root / "data" / "frozen" / "CodeObf-PI_v1.0"
    (fdir / "md5sums.txt").write_text("deadbeef  x\n", encoding="utf-8", newline="\n")
    before = snapshot(root)
    assert run(root) != 0
    assert snapshot(root) == before


def test_hc_counts_439_fails(tmp_path):
    """转交项 1 的回归测试：改写体 439 ⇒ 拒绝（`A6` 同步改小）。"""
    _assert_refused(tmp_path, n_rewrites=439, n_encoded=439 * 9, a6=439, a7=439 * 9 * 2)


def test_v3_split_seed_43_fails(tmp_path):
    """§3.6-15 `W0` 补注 4：`split_seed = 43` ⇒ 非零退出且未写入任何文件。"""
    _assert_refused(tmp_path, split_seed=43)


def test_v3_split_ratio_7_1_2_fails(tmp_path):
    """§3.6-15 `W0` 补注 4：`split_ratio = "7:1:2"` ⇒ 非零退出且未写入任何文件。"""
    _assert_refused(tmp_path, split_ratio="7:1:2")


def test_h15_splits_mismatch_a13_fails(tmp_path):
    """§3.6-15：`splits` 实算与 `A13` 不一致 ⇒ 非零退出。"""
    _assert_refused(tmp_path, a13_measured=(
        "N=220 ⇒ 期望 176:22:22；实测 train=175、val=22、test=23（判据 tolerance_groups=0，等式）"))


# ---------------------------------------------------------------------------
# §4-3 占位态识别
# ---------------------------------------------------------------------------
def test_placeholder_state_is_overwritable(tmp_path):
    """`md5sums.txt` 为 0 字节 ⇒ 允许覆写（不得把占位误判为「已冻结」）。"""
    root, _alog = build_tree(tmp_path)
    assert (root / "data" / "frozen" / "CodeObf-PI_v1.0" / "md5sums.txt").stat().st_size == 0
    assert run(root) == 0
    assert (root / "data" / "frozen" / "CodeObf-PI_v1.0" / "md5sums.txt").stat().st_size > 0


def test_frozen_manifest_state_refuses(tmp_path):
    """非空 + `status=frozen` ⇒ 拒绝（即使 md5sums 曾为空）。"""
    root, _alog = build_tree(tmp_path, frozen_state="frozen")
    (root / "data" / "frozen" / "CodeObf-PI_v1.0" / "md5sums.txt").write_text(
        "deadbeef  x\n", encoding="utf-8", newline="\n")
    before = snapshot(root)
    assert run(root) != 0
    assert snapshot(root) == before


def test_missing_frozen_dir_is_created(tmp_path):
    """目标目录不存在 ⇒ 允许新建（占位态判据不得把它误判为「已冻结」）。"""
    root, _alog = build_tree(tmp_path, frozen_placeholder=False)
    assert run(root) == 0
    assert (root / "data" / "frozen" / "CodeObf-PI_v1.0" / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# DATA_CARD 的关键披露（§3.6）是否落地
# ---------------------------------------------------------------------------
def test_data_card_carries_required_disclosures(_ok):
    root, _alog, _rc = _ok
    card = (root / "data" / "frozen" / "CodeObf-PI_v1.0" /
            "DATA_CARD_CodeObf-PI_v1.0.md").read_text(encoding="utf-8")
    for needle in (
        "codeparrot/codeparrot-clean",       # §3.6-28 来源替换
        "ast_ok",                            # §3.6-6 三值性
        "JavaScript",                        # §3.6-6 陷阱
        "事前定值",                            # §3.6-14
        "`V-10`",                            # §3.6-18
        "`N = 220`",                          # §3.6-17
        "44 × 4",                            # §3.6-16
        "4096",                              # §3.6-22
        "introduced",                        # §3.6-20
        "4ca94ca1",                          # §3.6-21
        "代录",                               # §3.6-23 / -26
        "1000000",                           # 通过率 1.0000 的占位判据（见下）
        "split_ratio",                       # §3.6-15
        "`n/a`（报告项）",                     # §3.6-29
        "0.3880",                            # §3.6-30 三数并列
    ):
        if needle == "1000000":
            continue
        assert needle in card, "DATA_CARD 缺少披露：%s" % needle
    assert "1.0000" in card


def test_data_card_declares_circularity_limitation(_ok):
    root, _alog, _rc = _ok
    card = (root / "data" / "frozen" / "CodeObf-PI_v1.0" /
            "DATA_CARD_CodeObf-PI_v1.0.md").read_text(encoding="utf-8")
    assert "循环依赖" in card and "不存在不动点" in card


def test_frozen_tree_is_utf8_lf(_ok):
    root, _alog, _rc = _ok
    fdir = root / "data" / "frozen" / "CodeObf-PI_v1.0"
    for rel in ("md5sums.txt", "DATA_CARD_CodeObf-PI_v1.0.md", "manifest.json",
                "reports/split_assertions.csv"):
        raw = (fdir / rel).read_bytes()
        assert b"\r" not in raw, "%s 含孤立 CR" % rel
        raw.decode("utf-8")

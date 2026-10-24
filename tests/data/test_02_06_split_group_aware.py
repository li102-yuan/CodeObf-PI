# -*- coding: utf-8 -*-
"""tests/data/test_02_06_split_group_aware.py — W3.6 group-aware 划分与两套体制（§5）。

覆盖 `windows/W3.6_split.md` §5 的 13 项：
① group 分配**确定性**（同输入两次运行 `seed_groups.csv` 字节相同）；
② 8:1:1 在组数为 10／11／99／100 时的边界取整规则；
③ 同一 group 的样本不跨分片；
④ 体制二 `train`／`val` 无留出编码（`G4` 的**正例与反例**）；
⑤ `G5` 在正负比 1.3 时失败；
⑥ `G6` 在两体制 `text` 不一致时失败；
⑦ `G7` 在 `clean_original_test` 混入训练样本时失败；
⑧ `--stratify-length` **不跨 group**、**只减不增**、且 group 分片归属不变；
⑨ `--smoke` 不覆盖全量产物；
⑩ 输出 UTF-8 且无孤立 CR；
⑪ 离线可跑；
⑫ `G9` 的正反例；
⑬ `family_atomic` 两侧。

**离线**：全部输入在 `tmp_path` 内自造；长度口径用**替换实现**（`mod.tlen`）⇒ 测试不依赖
`models/hf_cache/` 的 tokenizer，也不触网。
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import random
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_06_split_group_aware.py")

ENCS_TRAIN = ("base64", "rot13")
ENCS_HELD = ("hex", "morse")
ENCS = ENCS_TRAIN + ENCS_HELD

AH_ORIG = 2              # AgentHarm 的 id_original 数（每族 4 变体 ⇒ 8 组）
AH_VARIANTS = 4
PKU_GROUPS = 4
N_GROUPS = AH_ORIG * AH_VARIANTS + PKU_GROUPS      # = 12 ⇒ 10 : 1 : 1
REWRITES_PER_GROUP = 2

BENIGN_SPEC = [          # (benign_set, 行数上限, 组数, 每组行数)
    # 良性行数须 **> 攻击行数**（否则配平只能减不能增，`G5` 无解）；
    # 真实盘面同样如此（9000 良性 vs 7920 攻击）。
    ("G1", 160, 80, 2),
    ("G2", 60, 60, 1),
    ("G3", 60, 60, 1),
]

Dataset = collections.namedtuple("Dataset", "root cfg_path cfg")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w36_split", SCRIPT)


_COMMON_CACHE = {}


def _common():
    if "m" not in _COMMON_CACHE:
        _COMMON_CACHE["m"] = _load("w36_common_helper",
                                   os.path.join(ROOT, "scripts", "data", "_common.py"))
    return _COMMON_CACHE["m"]


@pytest.fixture(autouse=True)
def fake_length(mod, monkeypatch):
    """长度口径替换为**确定性、离线**的实现（按空白切分）：只测划分逻辑，不碰 tokenizer。"""
    monkeypatch.setattr(mod, "tlen", lambda text, tok: len(str(text).split()) + 1)
    yield


# ===========================================================================
# 夹具：在 tmp_path 内自造一整套上游产物
# ===========================================================================
def _mk_dataset(root):
    data = os.path.join(root, "data")
    benign_dir = os.path.join(data, "benign_G1_G2_G3")
    os.makedirs(benign_dir, exist_ok=True)

    seeds = []
    gidx = 0
    for orig in range(1, AH_ORIG + 1):
        for var in range(1, AH_VARIANTS + 1):
            gidx += 1
            seeds.append({
                "seed_id": "SEED-AH-%03d" % gidx, "seed_group_id": "SG-%04d" % gidx,
                "source": "AgentHarm", "harm_category": "Disinfo", "severity": "",
                "raw_text": "AH seed %d %d" % (orig, var),
                "source_item_id": "%d-%d" % (orig, var),
                "raw_file": "AgentHarm/benchmark/harmful_behaviors_test_public.json",
                "raw_sha256": "0" * 64,
            })
    for k in range(1, PKU_GROUPS + 1):
        gidx += 1
        seeds.append({
            "seed_id": "SEED-PKU-%03d" % k, "seed_group_id": "SG-%04d" % gidx,
            "source": "PKU-SafeRLHF", "harm_category": "Violence", "severity": "3",
            "raw_text": "PKU seed %d" % k,
            "source_item_id": "Alpaca-7B/train.jsonl#L%d" % (100 + k),
            "raw_file": "PKU-SafeRLHF/data/Alpaca-7B/train.jsonl", "raw_sha256": "1" * 64,
        })
    _common().write_csv(os.path.join(data, "seeds.csv"), seeds,
                        ["seed_id", "seed_group_id", "source", "harm_category",
                         "severity", "raw_text", "source_item_id", "raw_file", "raw_sha256"])

    meta = {
        "seed_count_total": len(seeds),
        "seed_count_by_source": {
            "AgentHarm": {"count": AH_ORIG * AH_VARIANTS},
            "PKU-SafeRLHF": {"count": PKU_GROUPS},
        },
        "agentharm_variant_structure": {
            "available": True, "field": "id_original",
            "n_behaviors": AH_ORIG * AH_VARIANTS,
            "n_original_entries": AH_ORIG,
            "variants_per_original": [AH_VARIANTS],
        },
    }
    meta_path = os.path.join(data, "seeds.meta.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    rewrites, ridx = [], 0
    for s in seeds:
        for k in range(REWRITES_PER_GROUP):
            ridx += 1
            rewrites.append({
                "rewrite_id": "RW-%04d" % ridx, "seed_group_id": s["seed_group_id"],
                "seed_id": s["seed_id"], "rewrite_index": str(k + 1),
                "text": "rewrite %d of %s" % (k + 1, s["seed_group_id"]),
                "parent_text_sha256": "a" * 64, "backend": "template",
                "model_id": "", "request_id": "", "generated_at": "",
            })
    rewrites_path = os.path.join(data, "rewrites.csv")
    _common().write_csv(rewrites_path, rewrites,
                        ["rewrite_id", "seed_group_id", "seed_id", "rewrite_index", "text",
                         "parent_text_sha256", "backend", "model_id", "request_id",
                         "generated_at"])

    src_of = {s["seed_group_id"]: s["source"] for s in seeds}
    enc_rows, cc_rows = [], []
    for rw in rewrites:
        for enc in ENCS:
            base = "ATK-%s-%s" % (rw["rewrite_id"].replace("-", ""), enc)
            plain = rw["text"]
            enc_rows.append({
                "sample_id": base + "-text", "seed_group_id": rw["seed_group_id"],
                "rewrite_id": rw["rewrite_id"], "seed_id": rw["seed_id"],
                "source": src_of[rw["seed_group_id"]], "harm_category": "Disinfo",
                "encoding_type": enc, "difficulty": "medium", "plaintext": plain,
                "payload": "payload:%s:%s" % (enc, plain),
                "carrier_type": "text", "label": "1",
            })
            cc_rows.append({
                "sample_id": base + "-code", "seed_group_id": rw["seed_group_id"],
                "rewrite_id": rw["rewrite_id"], "seed_id": rw["seed_id"],
                "source": src_of[rw["seed_group_id"]], "harm_category": "Disinfo",
                "encoding_type": enc, "difficulty": "medium", "carrier_type": "code",
                "carrier_subtype": "py_comment", "language": "python",
                "plaintext": plain, "payload": "payload:%s:%s" % (enc, plain),
                "text": "# code carrier %s\n# %s\n" % (enc, plain),
                "ast_ok": "1", "ast_error": "", "payload_intact": "1", "label": "1",
            })
    encoded_path = os.path.join(data, "encoded.csv")
    _common().write_csv(encoded_path, enc_rows,
                        ["sample_id", "seed_group_id", "rewrite_id", "seed_id", "source",
                         "harm_category", "encoding_type", "difficulty", "plaintext",
                         "payload", "carrier_type", "label"])
    cc_path = os.path.join(data, "code_carrier.csv")
    _common().write_csv(cc_path, cc_rows,
                        ["sample_id", "seed_group_id", "rewrite_id", "seed_id", "source",
                         "harm_category", "encoding_type", "difficulty", "carrier_type",
                         "carrier_subtype", "language", "plaintext", "payload", "text",
                         "ast_ok", "ast_error", "payload_intact", "label"])

    benign_fields = ["sample_id", "benign_group_id", "benign_set", "source",
                     "source_item_id", "encoding_type", "carrier_type", "host_context",
                     "g3_business_form", "business_necessity_evidence", "text", "label"]
    # 三集的 `benign_group_id` 须**全局唯一**（真实盘面亦然：G1 从 BG-00001、G2 从 BG-02092、
    # G3 从 BG-04053 起）⇒ 各集用不同的编号段，避免跨集重号。
    group_offset = {"G1": 0, "G2": 1000, "G3": 2000}
    for set_name, n_rows, n_groups, per_group in BENIGN_SPEC:
        rows = []
        for gi in range(n_groups):
            gid = "BG-%05d" % (1 + group_offset[set_name] + gi)
            for k in range(per_group):
                i = gi * per_group + k
                if i >= n_rows:
                    break
                rows.append(_benign_row(set_name, gid, i))
        _common().write_csv(os.path.join(benign_dir, "%s.csv" % set_name), rows,
                            benign_fields)

    cfg = {
        "split": {
            "ratio": "8:1:1", "train_share": 0.8, "val_share": 0.1, "split_seed": 42,
            "rounding_rule": "python_round_half_to_even",
            "unit_attack": "seed_group_id", "unit_benign": "benign_group_id",
            "family_prefix_agentharm": "FAM-AH", "family_prefix_pku": "FAM-PKU",
            "family_atomic": False,
            "attack_allocation": {"when_family_atomic_false": "slice",
                                  "when_family_atomic_true": "source_stratified"},
            "scene_unit": True, "scene_field": "source_item_id",
            "scene_source": "G3-synthetic",
            "regime_ood_filter_applies_to": ["attack", "benign_nonempty_encoding"],
            "ood_subset_of_id": True,
            "balance": {"target_ratio": 1.0, "band_low": 0.8, "band_high": 1.25,
                        "drop_unit": "group", "order": "benign_shuffle_order",
                        "count_window_ratios": [0.90, 0.85, 0.80]},
            "stratify": {"method": "bucketwise_trim_from_extreme", "buckets": 10,
                         "only_when_over_threshold": True, "threshold": 0.20},
            "inputs": {
                "encoded": encoded_path, "code_carrier": cc_path,
                "benign_dir": benign_dir + os.sep, "seeds": os.path.join(data, "seeds.csv"),
                "seeds_meta": meta_path, "rewrites": rewrites_path,
            },
            "expected": {
                "seed_groups_total": N_GROUPS,
                "attack_rows": N_GROUPS * REWRITES_PER_GROUP * len(ENCS) * 2,
                "text_carrier_rows": N_GROUPS * REWRITES_PER_GROUP * len(ENCS),
                "code_carrier_rows": N_GROUPS * REWRITES_PER_GROUP * len(ENCS),
                "benign_files": ["G1.csv", "G2.csv", "G3.csv"],
            },
            "output": {"dir": os.path.join(root, "out") + os.sep,
                       "seed_groups": "seed_groups.csv",
                       "clean_original_test": "clean_original_test.csv",
                       "g3_hard_negative_test": "g3_hard_negative_test.csv",
                       "dropped_for_length": "dropped_for_length.csv",
                       "report_json": "split_report.json", "splits_root": "splits",
                       "smoke_dir": "smoke"},
            "report_dir": os.path.join(root, "reports"),
            "log_dir": os.path.join(root, "logs"),
            "report_basename": "split_v1.0",
        }
    }
    cfg_path = os.path.join(root, "split.yaml")
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return Dataset(root, cfg_path, cfg["split"])


def _benign_row(set_name, gid, i):
    if set_name == "G3":
        synthetic = i % 2 == 0
        scene = "SC%02d" % (1 + (i // 4) % 2)
        return {
            "sample_id": "BN-G3-%04d" % (1 + i), "benign_group_id": gid,
            "benign_set": "G3",
            "source": "G3-synthetic" if synthetic else "TheStack",
            "source_item_id": ("%s:%d:code" % (scene, 1 + i)) if synthetic
            else ("repo/file.py#%d" % i),
            "encoding_type": "morse" if synthetic else "",
            "carrier_type": "code", "host_context": "host ctx %d" % i,
            "g3_business_form": "数据封装与传输", "business_necessity_evidence": "ev",
            "text": "g3 benign row %d" % i, "label": "0",
        }
    return {
        "sample_id": "BN-%s-%04d" % (set_name, 1 + i), "benign_group_id": gid,
        "benign_set": set_name, "source": "Alpaca" if set_name == "G2" else "TheStack",
        "source_item_id": "src-%d" % i, "encoding_type": "",
        "carrier_type": "text" if set_name == "G2" else "code",
        "host_context": "host %d" % i, "g3_business_form": "",
        "business_necessity_evidence": "",
        "text": "benign %s row %d" % (set_name, i), "label": "0",
    }


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return _mk_dataset(str(tmp_path_factory.mktemp("w36")))


def _run(mod, cfg_path, out, extra=()):
    argv = ["--config", cfg_path, "--out", out, "--no-log", "--no-report"] + list(extra)
    return mod.main(argv)


def _read(path):
    return _common().read_csv(path)


def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# ② 8:1:1 边界取整（N = 10／11／99／100）
# ===========================================================================
@pytest.mark.parametrize("n,expect", [
    (10, (8, 1, 1)),
    (11, (9, 1, 1)),
    (99, (79, 10, 10)),
    (100, (80, 10, 10)),
    (220, (176, 22, 22)),
    (185, (148, 18, 19)),
])
def test_split_counts_boundaries(mod, n, expect):
    got = mod._split_counts(n, {"train_share": 0.8, "val_share": 0.1})
    assert tuple(got) == expect
    assert sum(got) == n


def test_split_counts_uses_python_round_not_half_up(mod):
    """`N=185`：`round(18.5)` = **18**（银行家舍入）⇒ 148 : 18 : 19；四舍五入会得 148 : 19 : 18。"""
    assert mod._split_counts(185, {"train_share": 0.8, "val_share": 0.1}) == (148, 18, 19)


# ===========================================================================
# ① 确定性：同输入两次运行 `seed_groups.csv` 字节相同
# ===========================================================================
def test_seed_groups_byte_identical_across_runs(mod, dataset, tmp_path):
    out1, out2 = str(tmp_path / "o1"), str(tmp_path / "o2")
    assert _run(mod, dataset.cfg_path, out1) == mod.EXIT_OK
    assert _run(mod, dataset.cfg_path, out2) == mod.EXIT_OK
    a = os.path.join(out1, dataset.cfg["output"]["seed_groups"])
    b = os.path.join(out2, dataset.cfg["output"]["seed_groups"])
    assert open(a, "rb").read() == open(b, "rb").read()
    for reg in mod.REGIMES:
        for sp in mod.SPLITS:
            p1 = os.path.join(out1, "splits", reg, sp + ".csv")
            p2 = os.path.join(out2, "splits", reg, sp + ".csv")
            assert open(p1, "rb").read() == open(p2, "rb").read()
    for name in ("clean_original_test", "g3_hard_negative_test", "dropped_for_length"):
        p1 = os.path.join(out1, dataset.cfg["output"][name])
        p2 = os.path.join(out2, dataset.cfg["output"][name])
        assert open(p1, "rb").read() == open(p2, "rb").read()


# ===========================================================================
# ③ 同一 group 的样本不跨分片
# ===========================================================================
def test_group_members_share_one_split(mod, dataset, tmp_path):
    out = str(tmp_path / "o")
    assert _run(mod, dataset.cfg_path, out) == mod.EXIT_OK
    rows = _read(os.path.join(out, dataset.cfg["output"]["seed_groups"]))
    split_of = {r["group_id"]: r["split"] for r in rows}
    assert len(split_of) == len(rows) == N_GROUPS + sum(s[2] for s in BENIGN_SPEC)
    for reg in mod.REGIMES:
        for sp in mod.SPLITS:
            for r in _read(os.path.join(out, "splits", reg, sp + ".csv")):
                gid = r["seed_group_id"] or r["benign_group_id"]
                assert split_of[gid] == sp, (r["sample_id"], gid, sp)
    # 同一 rewrite 的 2 载体 × 2 编码变体亦同档
    by_rw = {}
    for reg in mod.REGIMES:
        for sp in mod.SPLITS:
            for r in _read(os.path.join(out, "splits", reg, sp + ".csv")):
                if r["sample_kind"] == "attack":
                    by_rw.setdefault(r["rewrite_id"], set()).add(sp)
    assert all(len(v) == 1 for v in by_rw.values())
    assert len(by_rw) == N_GROUPS * REWRITES_PER_GROUP


# ===========================================================================
# ④ G4 正例（端到端）与反例（合成分片）
# ===========================================================================
def test_g4_positive_end_to_end(mod, dataset, tmp_path):
    out = str(tmp_path / "o")
    assert _run(mod, dataset.cfg_path, out) == mod.EXIT_OK
    rep = _json(os.path.join(out, "split_report.json"))
    g4 = [c for c in rep["selfcheck"] if c["id"] == "G4"][0]
    assert g4["passed"] is True
    assert "= 0" in g4["measured"]


def test_g4_negative_heldout_in_ood_train(mod):
    shards = _minimal_shards(mod)
    shards["regime_ood"]["train"]["attack"].append(_atk("X-1", "SG-0001", "hex"))
    checks = dict((c[0], c[1]) for c in _selfcheck(mod, shards))
    assert checks["G4"] is False


def test_g4_negative_heldout_in_ood_val(mod):
    shards = _minimal_shards(mod)
    shards["regime_ood"]["val"]["benign"].append(_ben("X-2", "BG-00002", "morse"))
    checks = dict((c[0], c[1]) for c in _selfcheck(mod, shards))
    assert checks["G4"] is False


# ===========================================================================
# ⑤ G5 在正负比 1.3 时失败
# ===========================================================================
def test_g5_fails_when_ratio_1_3(mod):
    shards = _minimal_shards(mod)
    sh = shards["regime_id"]["train"]
    sh["attack"] = [_atk("A-%d" % i, "SG-0001", "base64") for i in range(13)]
    sh["benign"] = [_ben("B-%d" % i, "BG-00001", "") for i in range(10)]
    sh["n_pos"], sh["n_neg"] = 13, 10          # ratio = 1.30 > 1.25
    assert dict((c[0], c[1]) for c in _selfcheck(mod, shards))["G5"] is False
    sh["benign"] = [_ben("B-%d" % i, "BG-00001", "") for i in range(13)]
    sh["n_neg"] = 13                            # ratio = 1.00
    assert dict((c[0], c[1]) for c in _selfcheck(mod, shards))["G5"] is True


# ===========================================================================
# ⑥ G6 在两体制 text 不一致时失败
# ===========================================================================
def test_g6_fails_when_text_differs(mod):
    shards = _minimal_shards(mod)
    shards["regime_ood"]["test"]["attack"][0]["text"] = "DIFFERENT"
    assert dict((c[0], c[1]) for c in _selfcheck(mod, shards))["G6"] is False


# ===========================================================================
# ⑦ G7 在 clean_original_test 混入训练样本时失败
# ===========================================================================
def test_g7_fails_when_clean_overlaps_train(mod):
    shards = _minimal_shards(mod)
    shards["regime_id"]["train"]["attack"][0]["text"] = "SHARED"
    derived = {"clean": [{"text": "SHARED"}], "g3": []}
    assert dict((c[0], c[1]) for c in _selfcheck(mod, shards, derived=derived))["G7"] is False
    derived = {"clean": [{"text": "CLEAN-ONLY"}], "g3": []}
    assert dict((c[0], c[1]) for c in _selfcheck(mod, shards, derived=derived))["G7"] is True


# ===========================================================================
# ⑧ --stratify-length：不跨 group、只减不增、分片归属不变
# ===========================================================================
def test_stratify_does_not_change_group_assignment(mod, dataset, tmp_path):
    out_plain, out_strat = str(tmp_path / "plain"), str(tmp_path / "strat")
    assert _run(mod, dataset.cfg_path, out_plain) == mod.EXIT_OK
    assert _run(mod, dataset.cfg_path, out_strat, ["--stratify-length"]) == mod.EXIT_OK
    a = os.path.join(out_plain, dataset.cfg["output"]["seed_groups"])
    b = os.path.join(out_strat, dataset.cfg["output"]["seed_groups"])
    assert open(a, "rb").read() == open(b, "rb").read()


def test_stratify_only_reduces_and_keeps_groups(mod):
    """⑧ 的核心：剔除序列**只减不增**、**整组**剔除、并**把均值推向攻击侧**。"""
    rows = []
    for gi in range(20):
        gid = "BG-%05d" % (1 + gi)
        for k in range(4):                       # 每组 4 行
            rows.append({"benign_group_id": gid,
                         "text": "t " * (5 + gi), "sample_id": "S-%d-%d" % (gi, k)})
    rank = {g: i for i, g in enumerate(sorted({r["benign_group_id"] for r in rows}))}
    seq, info = mod.stratify_sequence(rows, 12.0, rank, floor_rows=8, buckets=10,
                                      tokenizer_id="fake")
    assert seq, "应产生非空剔除序列"
    counts = [s["rows_after"] for s in seq]
    assert counts == sorted(counts, reverse=True)            # 行数单调下降（只减不增）
    assert counts[-1] >= 8                                   # 不跌破 floor
    # 整组剔除：序列中每个 group 至多出现一次
    gids = [s["group_id"] for s in seq]
    assert len(gids) == len(set(gids))
    # 均值被推向攻击侧均值（12.0），末态明显近于池均值（fake 长度 = 词数 + 1）
    pool_mean = sum(sum(len(r["text"].split()) + 1 for r in rs)
                    for rs in mod._rows_by_group(rows).values()) / float(len(rows))
    assert abs(seq[-1]["mean_after"] - 12.0) < abs(pool_mean - 12.0)
    # 前缀选择：窗口越紧，保留行数越少
    cut_tight = mod.pick_prefix(seq, 24, 0.20)
    cut_loose = mod.pick_prefix(seq, 40, 0.20)
    assert seq[cut_tight]["rows_after"] <= 24
    assert seq[cut_loose]["rows_after"] <= 40
    assert info["long_first"] in (True, False)


def test_balance_plain_only_reduces_and_respects_band(mod):
    rows = [{"benign_group_id": "BG-%05d" % (1 + i // 3), "text": "x",
             "sample_id": "S-%d" % i} for i in range(60)]
    rank = {("BG-%05d" % (1 + i)): i for i in range(20)}
    kept, _dropped, _info = mod.balance_plain(rows, 30, rank, 0.8)
    n = len([r for r in rows if r["benign_group_id"] in kept])
    assert n <= 30 and n >= int(0.8 * 30)


# ===========================================================================
# ⑨ --smoke 不覆盖全量产物
# ===========================================================================
def test_smoke_does_not_overwrite_full_products(mod, dataset, tmp_path):
    out = str(tmp_path / "o")
    assert _run(mod, dataset.cfg_path, out) == mod.EXIT_OK
    before = {}
    for reg in mod.REGIMES:
        for sp in mod.SPLITS:
            p = os.path.join(out, "splits", reg, sp + ".csv")
            before[p] = open(p, "rb").read()
    for name in ("seed_groups", "clean_original_test", "g3_hard_negative_test",
                 "dropped_for_length"):
        p = os.path.join(out, dataset.cfg["output"][name])
        before[p] = open(p, "rb").read()

    assert _run(mod, dataset.cfg_path, out, ["--smoke", "10"]) == mod.EXIT_OK
    for p, blob in before.items():
        assert open(p, "rb").read() == blob, p
    smoke_dir = os.path.join(out, dataset.cfg["output"]["smoke_dir"])
    assert os.path.isfile(os.path.join(smoke_dir, "splits", "regime_id", "train.csv"))
    assert os.path.isfile(os.path.join(smoke_dir, dataset.cfg["output"]["seed_groups"]))
    assert os.path.isfile(os.path.join(smoke_dir, dataset.cfg["output"]["report_json"]))


# ===========================================================================
# ⑩ 输出 UTF-8 且无孤立 CR
# ===========================================================================
def test_outputs_are_utf8_lf(mod, dataset, tmp_path):
    out = str(tmp_path / "o")
    assert _run(mod, dataset.cfg_path, out) == mod.EXIT_OK
    paths = [os.path.join(out, "splits", reg, sp + ".csv")
             for reg in mod.REGIMES for sp in mod.SPLITS]
    for name in ("seed_groups", "clean_original_test", "g3_hard_negative_test",
                 "dropped_for_length"):
        paths.append(os.path.join(out, dataset.cfg["output"][name]))
    for p in paths:
        blob = open(p, "rb").read()
        assert b"\r" not in blob, p
        blob.decode("utf-8")


# ===========================================================================
# ⑫ G9 正反例
# ===========================================================================
def _seed_paths(dataset):
    data = os.path.join(dataset.root, "data")
    return os.path.join(data, "seeds.csv"), os.path.join(data, "seeds.meta.json")


def test_g9_positive(mod, dataset):
    fam_of, structure, _seeds, _src = mod.load_families(*_seed_paths(dataset))
    groups = [{"group_id": g, "group_kind": "attack", "family_id": f, "split": "train",
               "member_count": 8} for g, f in sorted(fam_of.items())]
    checks = dict((c[0], c[1]) for c in _selfcheck(mod, _minimal_shards(mod),
                                                   group_rows=groups, structure=structure))
    assert checks["G9"] is True
    assert structure["agentharm_families"] == AH_ORIG
    assert structure["families_total"] == AH_ORIG + PKU_GROUPS


def test_g9_negative_pku_misassigned_to_agentharm_family(mod, dataset):
    fam_of, structure, _seeds, _src = mod.load_families(*_seed_paths(dataset))
    groups = [{"group_id": g, "group_kind": "attack", "family_id": f, "split": "train",
               "member_count": 8} for g, f in sorted(fam_of.items())]
    victim = [g for g in groups if g["family_id"].startswith("FAM-PKU")][0]
    victim["family_id"] = "FAM-AH-9999"        # 错并入 AgentHarm 族
    checks = dict((c[0], c[1]) for c in _selfcheck(mod, _minimal_shards(mod),
                                                   group_rows=groups, structure=structure))
    assert checks["G9"] is False


# ===========================================================================
# ⑬ family_atomic 两侧
# ===========================================================================
def test_family_atomic_units_side_by_side(mod):
    sgs = ["SG-%04d" % i for i in range(1, 13)]
    fam = {g: ("FAM-AH-0001" if i < 8 else "FAM-PKU-%04d" % (i - 7))
           for i, g in enumerate(sgs)}
    units_false, _go_false, _src_false = mod.attack_units(sgs, fam, family_atomic=False)
    assert len(units_false) == 12
    assert all(len(v) == 1 for v in units_false.values())
    units_true, _go, _src = mod.attack_units(sgs, fam, family_atomic=True)
    assert len(units_true) == 1 + PKU_GROUPS          # 1 个 AH 族 + 4 个 PKU 族
    assert len(units_true["FAM:FAM-AH-0001"]) == 8


def test_family_atomic_true_keeps_family_in_one_split(mod, dataset, tmp_path):
    with open(dataset.cfg_path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    doc["split"]["family_atomic"] = True
    cfg_path2 = str(tmp_path / "split_atomic.yaml")
    with open(cfg_path2, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False)
    out = str(tmp_path / "o")
    assert _run(mod, cfg_path2, out) == mod.EXIT_OK
    rows = _read(os.path.join(out, dataset.cfg["output"]["seed_groups"]))
    fam_split = {}
    for r in rows:
        if r["group_kind"] == "attack":
            fam_split.setdefault(r["family_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in fam_split.values()), fam_split
    ah = [r for r in rows if r["family_id"].startswith("FAM-AH")]
    assert len(ah) == AH_ORIG * AH_VARIANTS
    rep = _json(os.path.join(out, "split_report.json"))
    assert rep["family_atomic"] is True
    assert sum(rep["actual_attack_groups"].values()) == N_GROUPS


def test_family_atomic_false_default_records_family_id(mod, dataset, tmp_path):
    out = str(tmp_path / "o")
    assert _run(mod, dataset.cfg_path, out) == mod.EXIT_OK
    rep = _json(os.path.join(out, "split_report.json"))
    assert rep["family_atomic"] is False
    assert rep["attack_allocator"] == "slice"     # 缺省路径**不变** ⇒ `b09771c` 的产物可复现
    rows = _read(os.path.join(out, dataset.cfg["output"]["seed_groups"]))
    assert len({r["group_id"] for r in rows}) == len(rows)
    atk = [r for r in rows if r["group_kind"] == "attack"]
    assert all(r["family_id"] for r in atk)
    assert len({r["family_id"] for r in atk}) == AH_ORIG + PKU_GROUPS


# ===========================================================================
# ⑬-b `r2`：按来源分层分配器（`assign_source_stratified`）
# ===========================================================================
def _two_strata_units():
    """AH 族恒 4 组、PKU 族恒 1 组（与真实盘面同构）。"""
    units = collections.OrderedDict()
    src = {}
    for i in range(1, 45):
        k = "FAM:FAM-AH-%04d" % i
        units[k] = ["g%03d-%d" % (i, j) for j in range(4)]
        src[k] = "AgentHarm"
    for i in range(1, 45):
        k = "FAM:FAM-PKU-%04d" % i
        units[k] = ["p%03d" % i]
        src[k] = "PKU-SafeRLHF"
    return units, src


def test_source_stratified_hits_group_target_exactly(mod):
    units, src = _two_strata_units()
    order = sorted(units)
    random.Random(42).shuffle(order)
    target = (176, 22, 22)
    _assign, counts = mod.assign_source_stratified(units, src, order, target)
    assert tuple(counts) == target, counts


def test_source_stratified_preserves_source_share(mod):
    """分层后各档的来源占比须贴近语料（LPT 会把 AH 族整体推向 train ⇒ 失真）。"""
    units, src = _two_strata_units()
    order = sorted(units)
    random.Random(42).shuffle(order)
    assign, _c = mod.assign_source_stratified(units, src, order, (176, 22, 22))
    corpus_pku = 44.0 / 220
    for sp in mod.SPLITS:
        tot = sum(len(units[k]) for k, v in assign.items() if v == sp)
        pku = sum(len(units[k]) for k, v in assign.items()
                  if v == sp and src[k] == "PKU-SafeRLHF")
        assert abs(pku / tot - corpus_pku) < 0.12, (sp, pku / tot)
    # 对照：组数贪心（LPT）会失真
    lpt = sorted(order, key=lambda k: -len(units[k]))
    a2, _r2 = mod.assign_in_order(units, lpt, (176, 22, 22))
    test_tot = sum(len(units[k]) for k, v in a2.items() if v == "test")
    test_pku = sum(len(units[k]) for k, v in a2.items()
                   if v == "test" and src[k] == "PKU-SafeRLHF")
    assert abs(test_pku / test_tot - corpus_pku) > 0.12


def test_source_stratified_is_deterministic(mod):
    units, src = _two_strata_units()
    order = sorted(units)
    random.Random(42).shuffle(order)
    a1, _ = mod.assign_source_stratified(units, src, list(order), (176, 22, 22))
    a2, _ = mod.assign_source_stratified(units, src, list(order), (176, 22, 22))
    assert a1 == a2


def test_source_stratified_end_to_end(mod, dataset, tmp_path):
    """夹具上：`family_atomic: true` ⇒ 分配器为分层、`A13` 逐格相等、来源占比贴近语料。"""
    with open(dataset.cfg_path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    doc["split"]["family_atomic"] = True
    cfg_path2 = str(tmp_path / "split_atomic.yaml")
    with open(cfg_path2, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False)
    out = str(tmp_path / "o")
    assert _run(mod, cfg_path2, out) == mod.EXIT_OK
    rep = _json(os.path.join(out, "split_report.json"))
    assert rep["attack_allocator"] == "source_stratified"
    assert sum(rep["actual_attack_groups"].values()) == N_GROUPS
    # `A13` 的组数目标须**逐格相等**（修复步保证）
    t, v = mod._split_counts(N_GROUPS, {"train_share": 0.8, "val_share": 0.1})[:2]
    assert rep["actual_attack_groups"]["train"] == t
    assert rep["actual_attack_groups"]["val"] == v
    # 来源构成表须存在（小夹具只有 12 组，占比容差留给 88 单位的分层单测）
    assert set(rep["attack_source_by_split"]) == set(mod.SPLITS)


# ===========================================================================
# 场景约束（§2.7 / G10）与派生视图
# ===========================================================================
def test_scene_groups_stay_together(mod, dataset, tmp_path):
    out = str(tmp_path / "o")
    assert _run(mod, dataset.cfg_path, out) == mod.EXIT_OK
    rep = _json(os.path.join(out, "split_report.json"))
    g10 = [c for c in rep["selfcheck"] if c["id"] == "G10"][0]
    assert g10["passed"] is True and "straddling=0" in g10["measured"]
    g11 = [c for c in rep["selfcheck"] if c["id"] == "G11"][0]
    assert g11["passed"] is True


def test_clean_and_g3_views(mod, dataset, tmp_path):
    out = str(tmp_path / "o")
    assert _run(mod, dataset.cfg_path, out) == mod.EXIT_OK
    clean = _read(os.path.join(out, dataset.cfg["output"]["clean_original_test"]))
    assert clean and all(r["label"] == "1" for r in clean)
    assert all(r["sample_id"].startswith("CLN-") for r in clean)
    groups = _read(os.path.join(out, dataset.cfg["output"]["seed_groups"]))
    test_groups = {r["group_id"] for r in groups
                   if r["group_kind"] == "attack" and r["split"] == "test"}
    assert {r["seed_group_id"] for r in clean} == test_groups
    g3 = _read(os.path.join(out, dataset.cfg["output"]["g3_hard_negative_test"]))
    assert g3 and all(r["benign_set"] == "G3" for r in g3)
    assert all(r["label"] == "0" for r in g3)


# ===========================================================================
# 辅助：最小分片字典 / 合成行
# ===========================================================================
def _atk(sid, gid, enc):
    return {"sample_id": sid, "seed_group_id": gid, "benign_group_id": "",
            "sample_kind": "attack", "family_id": "FAM-AH-0001",
            "encoding_type": enc, "text": "T:" + sid, "label": 1}


def _ben(sid, gid, enc, text=None):
    return {"sample_id": sid, "seed_group_id": "", "benign_group_id": gid,
            "sample_kind": "benign", "family_id": gid,
            "encoding_type": enc, "text": text or ("T:" + sid), "label": 0}


def _minimal_shards(mod):
    """同一 `sample_id` 在两体制中出现**同一 `text`**（`G6` 的正例基线）。"""
    shards = {}
    for reg in mod.REGIMES:
        shards[reg] = {}
        for sp in mod.SPLITS:
            enc = "hex" if (reg == "regime_ood" and sp == "test") else "base64"
            a = [_atk("SID-%s-A" % sp, "SG-0001", enc)]
            b = [_ben("SID-%s-B" % sp, "BG-%05d" % (1 + mod.SPLITS.index(sp)),
                      "base64" if sp != "test" else "hex")]
            shards[reg][sp] = {"attack": a, "benign": b, "n_pos": len(a), "n_neg": len(b),
                               "kept_groups": {b[0]["benign_group_id"]}, "dropped": [],
                               "dropped_groups": [], "ratio": 1.0, "stage": "balance",
                               "stratified": False, "bucket_of": {}, "group_len": {},
                               "floor_rows": 0, "ceil_rows": 0, "count_window_ratio": None}
    return shards


def _selfcheck(mod, shards, group_rows=None, structure=None, derived=None,
               smoke=False, scene_of_group=None, family_atomic=False):
    cfg = {"balance": {"band_low": 0.8, "band_high": 1.25}}
    if group_rows is None:
        rows = []
        for reg in mod.REGIMES:
            for sp in mod.SPLITS:
                for r in shards[reg][sp]["attack"]:
                    rows.append({"group_id": r["seed_group_id"], "group_kind": "attack",
                                 "family_id": r["family_id"], "split": sp,
                                 "member_count": 1})
                for r in shards[reg][sp]["benign"]:
                    rows.append({"group_id": r["benign_group_id"], "group_kind": "benign",
                                 "family_id": r["benign_group_id"], "split": sp,
                                 "member_count": 1})
        group_rows = rows
    if structure is None:
        structure = {"agentharm_families": 0, "agentharm_groups": 0, "pku_families": 0,
                     "pku_groups": 0, "families_total": 0,
                     "meta": {"agentharm_variant_structure": {
                         "n_original_entries": None, "variants_per_original": []}}}
    if derived is None:
        derived = {"clean": [], "g3": []}
    return mod.selfcheck(shards, group_rows, structure, cfg, smoke, derived,
                         scene_of_group or {}, "fake", family_atomic)

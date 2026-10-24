# -*- coding: utf-8 -*-
"""02_07_assert_splits.py — W3.7 自动断言、长度分布、n-gram 重叠。

**本脚本是 7 道门禁中 3 道（门禁 5／6／7）的唯一判定者**（`windows/W3.7_assert.md` §0）。

接口契约（`windows/W3.7_assert.md` §3）::

    python scripts/data/02_07_assert_splits.py \\
      --config configs/data/assert.yaml \\
      --split-dir data/interim/split_checks/ \\
      --out data/interim/split_checks/ \\
      [--check-only]

模式（§2，**易错项**）

===============  ============================================================  ========
模式              行为                                                          退出码
===============  ============================================================  ========
`--check-only`   只做环境与输入可用性自检（脚本可导入、配置可解析、`_common`        **恒 0**
                 可导入、六个分片文件是否存在）。**不读全量数据、不写任何
                 产物、不判定任何门禁、不加载 tokenizer。**输入缺失时打印
                 `SKIP` 与缺失路径，仍 `exit 0`。
无参（默认）      读全量数据，跑 `A1`–`A14`，写三个 CSV + `family_straddle.csv`     全通过 0；
                 + 运行报告。                                                   任一门禁不通过非 0
`--report-only`   同默认但**不重写**任何 CSV（只渲染报告）。                        同默认
===============  ============================================================  ========

**硬约束（§0／§7）**

1. **不写 `data/frozen/`、`data/sealed_test/`**；本窗口对 `data/interim/split_checks/` **只读**
   ——**断言脚本不得修改被断言的数据**（本脚本只向 `--out` 写 4 个新文件）。
2. **不训练、不微调、不加载检测器模型**（本窗口零推理）。唯一加载的是 `_common.token_length`
   惰性取的 **tokenizer**（tokenizer ≠ 模型），且**仅在全量模式下**、**仅当需要长度统计时**。
3. **禁止把断言「调通过」**：断言失败即如实报错并退出非零；**不得**通过放宽阈值、剔除不利样本、
   改口径来使其变绿。阈值改动只能走 `PROJECT_CONTRACT.md` §5（重大偏离，须 W0 事前批准）。
4. **门禁 5／6／7 必须由脚本断言，禁止人眼核对**（`windows/W3_stage2_data.md` §4-2）。

**确定性**：本脚本**不用任何随机源**。A14 的族内／跨族配对为**全枚举**（非抽样），
故任何两次运行的产物逐字节相同（`generated_at` 除外，它是唯一非确定字段）。
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import json
import math
import os
import random
import sys
import time
from datetime import datetime

_T0 = time.perf_counter()

# ---------------------------------------------------------------------------
# 载入共享库 `scripts/data/_common.py`（唯一出处；本窗口**不改其任何签名**）
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load_module("data_common", os.path.join(_HERE, "_common.py"))

read_csv = common.read_csv
write_csv = common.write_csv
sha256_file = common.sha256_file
token_length = common.token_length
char_ngrams = common.char_ngrams
word_ngrams = common.word_ngrams
ngram_overlap_rate = common.ngram_overlap_rate
ENCODING_CODES = tuple(common.ENCODING_CODES)
CARRIER_CODES = tuple(common.CARRIER_CODES)
OOD_TRAIN = tuple(common.OOD_TRAIN_ENCODINGS)
OOD_HELDOUT = tuple(common.OOD_HELDOUT_ENCODINGS)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAIL = 3

SPLITS = ("train", "val", "test")
REGIMES = ("regime_id", "regime_ood")

#: `split_assertions.csv` 字段（§3.1；**顺序即契约**）。
ASSERTION_FIELDS = ["assertion_id", "scope", "description", "threshold",
                    "measured", "passed", "evidence_path"]

#: `length_distribution.csv` 字段（§3.2）。
LENGTH_FIELDS = ["split", "regime", "sample_kind", "label", "n",
                 "mean_tokens", "median_tokens", "p95_tokens", "max_tokens", "min_tokens",
                 "positive_mean_tokens", "negative_mean_tokens", "abs_diff_ratio",
                 "threshold", "passed"]

#: `ngram_overlap.csv` 字段（§3.3）。
NGRAM_FIELDS = ["regime", "n", "level", "train_ngrams", "test_ngrams",
                "overlap", "overlap_rate", "threshold", "passed"]

#: `family_straddle.csv` 字段（§3.1 `A14`）。
FAMILY_FIELDS = ["family_id", "group_count", "train", "val", "test"]

#: `ngram_controls.csv` 字段（`D-36` 转交项 42 的 ②③ 对半控制）。
CONTROL_FIELDS = ["run", "regime", "split", "control", "key", "n_a", "n_b",
                  "ngrams_a", "ngrams_b", "overlap", "overlap_rate", "note"]

#: 门禁 7 的**默认角色**（`D-36` 裁定一）。配置可覆盖，但**本阶段不得改回二值门禁**。
GATE7_ROLE_REPORT_ITEM = "report_item"

#: 全部分片 CSV 的宽表字段（`windows/W3.6_split.md` §2.3.1）。
WIDE_FIELDS = ["sample_id", "seed_group_id", "benign_group_id", "sample_kind", "family_id",
               "rewrite_id", "seed_id", "source", "harm_category", "benign_set",
               "encoding_type", "carrier_type", "carrier_subtype", "host_context",
               "plaintext", "payload", "text", "difficulty", "label"]

# ---------------------------------------------------------------------------
# `A1`–`A14` 的静态定义（id, scope, description, threshold）
# ---------------------------------------------------------------------------
#: `scope` 取值域（§3.1）：`attack`／`benign`／`cross`／`regime_id`／`regime_ood`／`global`。
ASSERTION_SPECS = [
    ("A1", "attack", "train/val/test 的 seed_group_id 集合两两交集为空", "空集（两体制）"),
    ("A2", "benign", "benign_group_id 两两交集为空，且与攻击侧 group 空间不同源（前缀 SG- vs BG-）", "空集"),
    ("A3", "regime_ood", "体制二 train∪val 的 encoding_type ⊆ 5 训练编码", "⊆ OOD_TRAIN"),
    ("A4", "regime_ood", "体制二 test 的 encoding_type ⊆ 4 留出编码", "⊆ OOD_HELDOUT"),
    ("A5", "regime_ood", "4 留出编码在体制二 train/val 中命中数 = 0（encoding_type 判定 + text/plaintext 两列内容扫描）",
     "命中数 = 0"),
    ("A6", "global", "rewrite_id 去重计数 = 440（改写体总数 = 440）", "== 440"),
    ("A7", "global", "sample_id 去重计数 = 7920（440 × 9 × 2 = 7920，两载体合并后去重）", "== 7920"),
    ("A8", "global", "每个 rewrite_id 恰有 9 编码 × 2 载体 = 18 条（体制一全视图）", "== 18"),
    ("A9", "cross", "每个分片内部 label 正负比 ∈ [0.8, 1.25]", "[0.80, 1.25]"),
    ("A10", "cross", "clean_original_test 的 group 集 ⊆ 测试 group 集，且其 text 与任一训练样本 text 不相等", "⊆ 且 交集空"),
    ("A11", "cross", "g3_hard_negative_test 的 group 集 ⊆ 测试 group 集", "⊆"),
    ("A12", "global", "代码载体样本 AST 报错率 ≤ 10%（复算，与门禁 2 同源）", "≤ 0.10"),
    ("A13", "global", "划分比例 V-3 = 8:1:1（group 级，split_seed=42），期望值由 seeds.meta.json 派生",
     "== round(N×0.8) : round(N×0.1) : N−前两者"),
    ("A14", "global", "语义族跨档登记（报告性断言，非门禁）", "n/a（无阈值）"),
]

#: 门禁类断言（`A14` 除外）。
GATING_ASSERTIONS = tuple(a[0] for a in ASSERTION_SPECS if a[0] != "A14")


def _rel(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace(os.sep, "/")
    except ValueError:  # pragma: no cover
        return path


def _stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _median(values):
    values = sorted(values)
    if not values:
        return 0.0
    n = len(values)
    mid = n // 2
    if n % 2:
        return float(values[mid])
    return (float(values[mid - 1]) + float(values[mid])) / 2.0


def _pct(values, q):
    """最近秩（nearest-rank）分位数；`q ∈ (0,1]`。空表返回 0.0。"""
    values = sorted(values)
    if not values:
        return 0.0
    idx = int(math.ceil(q * len(values))) - 1
    return float(values[max(0, min(idx, len(values) - 1))])


def _f(value, nd=6):
    """把浮点渲染为定长字符串；非数值原样返回。"""
    try:
        return ("%%.%df" % nd) % float(value)
    except (TypeError, ValueError):
        return str(value)


#: `derived_expected(N, train_share, val_share)` 的唯一口径（`A13`）。
def derived_expected(n_total, train_share, val_share):
    """`A13` 的**唯一**取整口径：`round(N×0.8) : round(N×0.1) : N − 前两者`。

    `round` = Python 内置（银行家舍入，round-half-to-even），**不是四舍五入**。
    末档取 `N − 前两者`，保证三档之和恒等于 `N`。
    """
    n_train = int(round(int(n_total) * float(train_share)))
    n_val = int(round(int(n_total) * float(val_share)))
    return n_train, n_val, max(0, int(n_total) - n_train - n_val)


_LEN_CACHE = {}


def tlen(text, tokenizer_id):
    """token 数（带缓存）。**测试可 monkeypatch 本函数**以离线跑（不触 tokenizer）。"""
    key = (tokenizer_id, text)
    cached = _LEN_CACHE.get(key)
    if cached is None:
        cached = int(token_length(text, tokenizer_id))
        _LEN_CACHE[key] = cached
    return cached


def _attack(rows):
    return [r for r in rows if r.get("sample_kind") == "attack"]


def _benign(rows):
    return [r for r in rows if r.get("sample_kind") == "benign"]


# ===========================================================================
# 1. 读上游（**只读** `data/interim/split_checks/`）
# ===========================================================================
def split_path(split_dir, regime, split):
    return os.path.join(split_dir, "splits", regime, split + ".csv")


def load_shards(split_dir):
    """读六个分片 CSV（UTF-8 + LF，由 `_common.read_csv` 保证 CR 归一）。"""
    shards = {}
    for regime in REGIMES:
        shards[regime] = {}
        for split in SPLITS:
            shards[regime][split] = read_csv(split_path(split_dir, regime, split))
    return shards


def load_dataset(args, cfg):
    split_dir = args.split_dir or cfg["inputs"]["split_dir"]
    ds = {
        "split_dir": split_dir,
        "shards": load_shards(split_dir),
        "seed_groups": read_csv(os.path.join(split_dir, "seed_groups.csv")),
        "clean": read_csv(os.path.join(split_dir, "clean_original_test.csv")),
        "g3": read_csv(os.path.join(split_dir, "g3_hard_negative_test.csv")),
        "seeds_meta_path": args.seeds_meta or cfg["inputs"]["seeds_meta"],
        "ast_report_path": args.ast_report or cfg["inputs"]["ast_report"],
    }
    with open(ds["seeds_meta_path"], "r", encoding="utf-8") as fh:
        ds["seeds_meta"] = json.load(fh)
    ds["ast_report"] = read_csv(ds["ast_report_path"]) if os.path.isfile(ds["ast_report_path"]) else []
    return ds


# ===========================================================================
# 2. `A1`–`A14`（**逐条纯函数**，便于离线单测）
# ===========================================================================
def check_a1(shards):
    """train/val/test 的 `seed_group_id` 集合两两交集为空（两体制各自）。"""
    bad = []
    for regime in REGIMES:
        sets = {sp: {r["seed_group_id"] for r in _attack(shards[regime][sp])
                     if r.get("seed_group_id")} for sp in SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = sets[a] & sets[b]
            if ov:
                bad.append("%s:%s∩%s=%d" % (regime, a, b, len(ov)))
    measured = ";".join(bad) if bad else "两体制的 train∩val／train∩test／val∩test 均为 0"
    return not bad, measured


def check_a2(shards):
    """良性侧 group 两两交集为空，且与攻击侧 group 空间不同源。"""
    bad = []
    for regime in REGIMES:
        sets = {sp: {r["benign_group_id"] for r in _benign(shards[regime][sp])
                     if r.get("benign_group_id")} for sp in SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = sets[a] & sets[b]
            if ov:
                bad.append("%s:%s∩%s=%d" % (regime, a, b, len(ov)))
    atk, ben = set(), set()
    for regime in REGIMES:
        for sp in SPLITS:
            atk |= {r["seed_group_id"] for r in _attack(shards[regime][sp]) if r.get("seed_group_id")}
            ben |= {r["benign_group_id"] for r in _benign(shards[regime][sp]) if r.get("benign_group_id")}
    inter = atk & ben
    if inter:
        bad.append("攻击∩良性 group 空间=%d" % len(inter))
    bad_prefix = sorted(g for g in list(atk) + list(ben)
                        if not (g.startswith("SG-") or g.startswith("BG-")))
    if bad_prefix:
        bad.append("前缀异常=%s" % ",".join(bad_prefix[:3]))
    measured = ";".join(bad) if bad else (
        "良性侧三档两两交集 0；攻击 group %d 个（SG-）∩ 良性 group %d 个（BG-）=0" % (len(atk), len(ben)))
    return not bad, measured


def check_a3(shards):
    """体制二 train∪val 的 `encoding_type` ⊆ 5 训练编码。

    口径：**空串 `encoding_type` 视为「无编码层」**（G1／G2 本身不含编码层），
    不参与本断言（`configs/data/split.yaml` L58–L63，已登记 `DEVIATION_LEDGER.md`）。
    """
    encs = set()
    for sp in ("train", "val"):
        for r in shards["regime_ood"][sp]:
            e = (r.get("encoding_type") or "").strip()
            if e:
                encs.add(e)
    outside = sorted(encs - set(OOD_TRAIN))
    measured = "非空 encoding_type 集合=%s；越界=%s" % (
        sorted(encs) or "∅", outside or "无")
    return not outside, measured


def check_a4(shards):
    """体制二 test 的 `encoding_type` ⊆ 4 留出编码。"""
    encs = set()
    for r in shards["regime_ood"]["test"]:
        e = (r.get("encoding_type") or "").strip()
        if e:
            encs.add(e)
    outside = sorted(encs - set(OOD_HELDOUT))
    measured = "非空 encoding_type 集合=%s；越界=%s" % (
        sorted(encs) or "∅", outside or "无")
    return not outside, measured


def check_a5(shards, scan_columns=("text", "plaintext")):
    """4 留出编码在体制二 train／val 中命中数 = 0。

    两层扫描（`windows/W3.7_assert.md` §3.1）：
    ① **分类列判定** —— 行 `encoding_type` ∈ `OOD_HELDOUT`（含良性侧非空编码行）；
    ② **内容列扫描** —— 对 `text` 与 `plaintext` 两列，检查是否等于任一**留出编码行**的
       `payload`／`text`（**精确串相等**）⇒ 捕获「列被误标为空但内容实为留出编码」的形态。

    **参照集只取 `payload` 与 `text`，刻意不含 `plaintext`**：`plaintext` 是**改写体明文**，
    它在同一 `rewrite_id` 的全部 9 编码与 2 载体上**恒等**，把它并入参照集会让
    「同一改写体的训练编码行」被判为泄露（假阳性，实测 3960 条）。判别编码层的字符串
    只有 `payload`（编码后载荷）与 `text`（代码载体上嵌入后的整段文本）。
    """
    heldout = set(OOD_HELDOUT)
    hits = []
    for sp in ("train", "val"):
        for r in shards["regime_ood"][sp]:
            if (r.get("encoding_type") or "").strip() in heldout:
                hits.append("%s:%s" % (sp, r.get("sample_id", "?")))
    hold = set()
    for regime in REGIMES:
        for sp in SPLITS:
            for r in _attack(shards[regime][sp]):
                if (r.get("encoding_type") or "").strip() in heldout:
                    for col in ("payload", "text"):
                        v = r.get(col) or ""
                        if v:
                            hold.add(v)
    leaks = []
    for sp in ("train", "val"):
        for r in shards["regime_ood"][sp]:
            for col in scan_columns:
                v = r.get(col) or ""
                if v and v in hold:
                    leaks.append("%s:%s:%s" % (sp, r.get("sample_id", "?"), col))
    measured = ("分类列命中=%d；内容列命中=%d（留出编码串 %d 个，扫描列 %s）"
                % (len(hits), len(leaks), len(hold), "/".join(scan_columns)))
    if hits or leaks:
        measured += "；例：%s" % ",".join((hits + leaks)[:3])
    return (not hits and not leaks), measured


def check_a6(shards, expected=440):
    """`rewrite_id` 去重计数 = 440（**转交项 1**：改写体总数 = 440，等式）。"""
    ids = set()
    for regime in REGIMES:
        for sp in SPLITS:
            for r in _attack(shards[regime][sp]):
                if r.get("rewrite_id"):
                    ids.add(r["rewrite_id"])
    return len(ids) == int(expected), "rewrite_id 去重计数 = %d（判据 == %d）" % (len(ids), int(expected))


def check_a7(shards, expected=7920):
    """`sample_id` 去重计数 = 7920（**转交项 1**：`440 × 9 × 2`，等式）。

    **必须在两载体合并后进行**：六个分片已同时含 `text` 与 `code` 两载体
    （`carrier_type ∈ {'text','code'}`），故直接对攻击侧全量 `sample_id` 去重即可；
    若只读一侧会得 3960 并必然非零退出。
    """
    ids = set()
    for regime in REGIMES:
        for sp in SPLITS:
            for r in _attack(shards[regime][sp]):
                if r.get("sample_id"):
                    ids.add(r["sample_id"])
    return len(ids) == int(expected), "sample_id 去重计数 = %d（判据 == %d）" % (len(ids), int(expected))


def check_a8(shards, expected=18):
    """每个 `rewrite_id` 恰有 9 编码 × 2 载体 = 18 条（在**体制一全视图**上计数）。"""
    cnt = collections.Counter()
    carriers = collections.defaultdict(set)
    for sp in SPLITS:
        for r in _attack(shards["regime_id"][sp]):
            rid = r.get("rewrite_id")
            if rid:
                cnt[rid] += 1
                carriers[rid].add(r.get("carrier_type"))
    bad = {k: v for k, v in cnt.items() if v != int(expected)}
    bad_carrier = sorted(k for k, v in carriers.items() if v != set(CARRIER_CODES))
    measured = ("参与计数 rewrite_id=%d；每 id 行数==%d 者 %d 个（不符 %d 个）；"
                "载体集非 {text,code} 者 %d 个"
                % (len(cnt), int(expected), len(cnt) - len(bad), len(bad), len(bad_carrier)))
    if bad:
        measured += "；例：%s" % ",".join("%s=%d" % kv for kv in list(bad.items())[:3])
    return (len(cnt) > 0 and not bad and not bad_carrier), measured


def check_a9(shards, lo=0.8, hi=1.25):
    """每个分片内部 `label` 正负比 ∈ [0.8, 1.25]。"""
    bad, det = [], []
    for regime in REGIMES:
        for sp in SPLITS:
            rows = shards[regime][sp]
            pos = sum(1 for r in rows if str(r.get("label")) == "1")
            neg = sum(1 for r in rows if str(r.get("label")) == "0")
            ratio = (pos / float(neg)) if neg else float("inf")
            det.append("%s/%s=%.4f" % (regime, sp, ratio))
            if neg <= 0 or not (lo <= ratio <= hi):
                bad.append("%s/%s" % (regime, sp))
    measured = "判定带=[%.2f,%.2f] %s" % (lo, hi, ";".join(det))
    if bad:
        measured += "；越界：%s" % ",".join(bad)
    return not bad, measured


def check_a10(shards, clean):
    """`clean_original_test` 的 group 集 ⊆ 测试 group 集，且其 `text` 与任一训练样本 `text` 不相等。"""
    test_groups = set()
    for regime in REGIMES:
        test_groups |= {r["seed_group_id"] for r in _attack(shards[regime]["test"])
                        if r.get("seed_group_id")}
    clean_groups = {r["seed_group_id"] for r in clean if r.get("seed_group_id")}
    outside = sorted(clean_groups - test_groups)
    train_texts = set()
    for regime in REGIMES:
        for sp in ("train", "val"):
            for r in shards[regime][sp]:
                if r.get("text"):
                    train_texts.add(r["text"])
    dup = sorted({r["text"] for r in clean if r.get("text")} & train_texts)
    measured = ("clean group=%d，越出测试 group 集者=%d；clean text 与 train∪val 相等者=%d"
                % (len(clean_groups), len(outside), len(dup)))
    if outside or dup:
        measured += "；例：%s" % ",".join((outside + dup)[:3])
    return (not outside and not dup), measured


def check_a11(shards, g3):
    """`g3_hard_negative_test` 的 group 集 ⊆ 测试 group 集。"""
    test_benign = set()
    for regime in REGIMES:
        test_benign |= {r["benign_group_id"] for r in _benign(shards[regime]["test"])
                        if r.get("benign_group_id")}
    g3_groups = {r["benign_group_id"] for r in g3 if r.get("benign_group_id")}
    outside = sorted(g3_groups - test_benign)
    measured = "g3 group=%d，越出测试良性 group 集者=%d" % (len(g3_groups), len(outside))
    if outside:
        measured += "；例：%s" % ",".join(outside[:3])
    return not outside, measured


def check_a12(ast_rows, threshold=0.10):
    """代码载体样本 AST 报错率 ≤ 10%（复算，与门禁 2 同源）。

    口径：分母 = `ast_ok + ast_failed`（**`js_comment` 子型不经 Python AST 解析**，
    其 `ast_ok` 与 `ast_failed` 同为 0，故自然不入分母；与 `code_carrier_report.json`
    的 `error_rate` 同源同值）。
    """
    n_ok = sum(int(r.get("ast_ok") or 0) for r in ast_rows)
    n_fail = sum(int(r.get("ast_failed") or 0) for r in ast_rows)
    denom = n_ok + n_fail
    rate = (n_fail / float(denom)) if denom else float("nan")
    passed = bool(denom > 0 and rate <= float(threshold))
    measured = "ast_failed=%d / (ast_ok+ast_failed)=%d ⇒ 报错率=%s（判据 ≤ %.2f）" % (
        n_fail, denom, _f(rate), float(threshold))
    return passed, measured


def check_a13(group_rows, seeds_meta, cfg):
    """划分比例 `V-3` = 8:1:1（group 级，`split_seed` = 42），期望值**派生式**（禁硬编码）。

    `N` = `seeds.meta.json` 的 `seed_count_total`；期望 = `round(N×0.8) : round(N×0.1) : N−前两者`
    （Python `round()` = 银行家舍入）。三档实测计数**逐一打印**，**等式**比较
    （`tolerance_groups: 0`，理由见 `configs/data/assert.yaml` 与运行报告 §3）。
    """
    n_total = int(seeds_meta["seed_count_total"])
    exp = derived_expected(n_total, cfg["split_ratio"]["train_share"],
                           cfg["split_ratio"]["val_share"])
    cnt = collections.Counter(r.get("split") for r in group_rows if r.get("group_kind") == "attack")
    act = (cnt.get("train", 0), cnt.get("val", 0), cnt.get("test", 0))
    tol = int(cfg["split_ratio"].get("tolerance_groups", 0))
    passed = all(abs(a - e) <= tol for a, e in zip(act, exp))
    measured = ("N=%d ⇒ 期望 %d:%d:%d；实测 train=%d、val=%d、test=%d（判据 tolerance_groups=%d，等式）"
                % ((n_total,) + exp + act + (tol,)))
    return passed, measured


def compute_a14(shards, overlap_field="plaintext"):
    """`A14`：语义族跨档的**登记与测量**（报告性断言，**不设通过/不通过**）。

    - 逐族分片分布（按 **group 数**）；族归属取自 `regime_id`（两体制共用同一 group 划分，`V-2`）。
    - 跨 `train↔test` 的族数。
    - **族内变体两两** 与 **跨族** 的字符 3-gram 重叠率中位数（对象 = `overlap_field`，
      按 `rewrite_id` 去重；**全枚举，不抽样** ⇒ 确定性）。
    """
    per_fam = collections.defaultdict(lambda: collections.defaultdict(set))
    for sp in SPLITS:
        for r in _attack(shards["regime_id"][sp]):
            fam = r.get("family_id")
            grp = r.get("seed_group_id")
            if fam and grp:
                per_fam[fam][sp].add(grp)
    rows = []
    straddling = 0
    for fam in sorted(per_fam):
        d = per_fam[fam]
        n_tr, n_va, n_te = (len(d["train"]), len(d["val"]), len(d["test"]))
        if n_tr and n_te:
            straddling += 1
        rows.append({"family_id": fam, "group_count": n_tr + n_va + n_te,
                     "train": n_tr, "val": n_va, "test": n_te})

    texts, fam_of_rid = {}, {}
    for regime in REGIMES:
        for sp in SPLITS:
            for r in _attack(shards[regime][sp]):
                rid = r.get("rewrite_id")
                if rid and r.get(overlap_field) and rid not in texts:
                    texts[rid] = char_ngrams(r[overlap_field], 3)
                    fam_of_rid[rid] = r.get("family_id") or ""
    rid_sorted = sorted(texts)
    within, across = [], []
    for a, b in itertools.combinations(rid_sorted, 2):
        val = ngram_overlap_rate(texts[a], texts[b])
        if fam_of_rid[a] and fam_of_rid[a] == fam_of_rid[b]:
            within.append(val)
        else:
            across.append(val)
    within.sort()
    across.sort()
    info = {
        "overlap_field": overlap_field,
        "families_total": len(rows),
        "straddling_train_test": straddling,
        "rewrite_ids": len(texts),
        "within_pairs": len(within),
        "across_pairs": len(across),
        "within_median": _median(within),
        "across_median": _median(across),
        "within_p95": _pct(within, 0.95),
        "ratio": (_median(within) / _median(across)) if across and _median(across) > 0 else float("inf"),
    }
    return rows, info


def a14_trigger(info, cfg):
    """`W3.6` §2.3.2 的**触发式升级**判据（本窗口只判「是否触发、是否仍待办」，**不自行切换 `family_atomic`**）。

    **`straddle_zero`**：跨 `train↔test` 族数为 **0** ⇒ 说明 `family_atomic: true` **已生效**
    （`D-36` 裁定二批准、`W3.6` r2 落地）。此时**不再有待办动作** —— 该状态由本字段显式标出，
    以免把「比值项按字面仍成立」（族内相似度天然高于跨族，属语料属性）误读为「还需处置」。
    """
    ratio_lim = float(cfg["a14"]["trigger_ratio"])
    med_lim = float(cfg["a14"]["trigger_within_median"])
    by_ratio = info["within_median"] >= ratio_lim * info["across_median"] if info["across_median"] > 0 else False
    by_abs = info["straddling_train_test"] > 0 and info["within_median"] > med_lim
    straddle_zero = info["straddling_train_test"] == 0
    return {"by_ratio": bool(by_ratio), "by_median": bool(by_abs),
            "triggered": bool(by_ratio or by_abs), "straddle_zero": bool(straddle_zero),
            "pending_action": bool((by_ratio or by_abs) and not straddle_zero)}


# ===========================================================================
# 3. 长度分布（门禁 6）与 n-gram 重叠（门禁 7）
# ===========================================================================
def compute_length(ds, cfg, tokenizer_id):
    """逐分片长度统计 + 正负均值差（门禁 6）。

    `abs_diff_ratio = |均值(正) − 均值(负)| / max(均值(正), 均值(负))`（**唯一口径**）。
    """
    threshold = float(cfg["gate6"]["threshold"])
    rows, shard = [], {}
    for regime in REGIMES:
        shard[regime] = {}
        for sp in SPLITS:
            shard_rows = ds["shards"][regime][sp]
            by_kind = collections.defaultdict(list)
            for r in shard_rows:
                by_kind[r.get("sample_kind")].append(tlen(r.get("text") or "", tokenizer_id))
            for kind in ("attack", "benign"):
                vals = by_kind.get(kind, [])
                rows.append({
                    "split": sp, "regime": regime, "sample_kind": kind,
                    "label": ("1" if kind == "attack" else "0"), "n": len(vals),
                    "mean_tokens": _f(_mean(vals), 4), "median_tokens": _f(_median(vals), 4),
                    "p95_tokens": _f(_pct(vals, 0.95), 4),
                    "max_tokens": (int(max(vals)) if vals else 0),
                    "min_tokens": (int(min(vals)) if vals else 0),
                    "positive_mean_tokens": "", "negative_mean_tokens": "",
                    "abs_diff_ratio": "", "threshold": "", "passed": "",
                })
            pos = by_kind.get("attack", [])
            neg = by_kind.get("benign", [])
            pm, nm = _mean(pos), _mean(neg)
            denom = max(pm, nm)
            adr = (abs(pm - nm) / denom) if denom > 0 else 0.0
            ok = bool(adr <= threshold)
            shard[regime][sp] = {"n_pos": len(pos), "n_neg": len(neg),
                                 "pos_mean": pm, "neg_mean": nm,
                                 "abs_diff_ratio": adr, "passed": ok}
            rows.append({
                "split": sp, "regime": regime, "sample_kind": "shard", "label": "",
                "n": len(shard_rows), "mean_tokens": "", "median_tokens": "",
                "p95_tokens": "", "max_tokens": "", "min_tokens": "",
                "positive_mean_tokens": _f(pm, 4), "negative_mean_tokens": _f(nm, 4),
                "abs_diff_ratio": _f(adr, 6), "threshold": _f(threshold, 2),
                "passed": "True" if ok else "False",
            })
    return rows, shard


def _union_grams(rows, field, level, n):
    fn = char_ngrams if level == "char" else word_ngrams
    out = set()
    for r in rows:
        text = r.get(field) or ""
        if text:
            out |= fn(text, n)
    return out


def compute_ngram(ds, cfg, field="text"):
    """n-gram 重叠（**报告项**，`D-36` 裁定一）：判定只认**字符级 3-gram**；词级 3-gram 与字符级 4-gram 仅作对照。

    重叠率 = `|ngram(train) ∩ ngram(test)| / |ngram(test)|`；
    `train` 侧取该体制 `train` 分片**全体** `text`，`test` 侧取该体制 `test` 分片**全体** `text`。

    **判定角色**：本量自 2026-10-02 起为**报告项** ⇒ 判定行 `passed` 记 **`n/a`**（不再有通过/不通过），
    `threshold` 列照旧写出实算阈值与实算值，供读者/`DATA_CARD` 自行对照。
    `diag[regime]["below_threshold"]` 保留**阈值比较**的内部结果（仅用于报告行文与回归测试，
    **不参与退出码、不构成门禁**）。
    """
    threshold = float(cfg["gate7"]["threshold"])
    js_level = str(cfg["gate7"]["judge_level"])
    js_n = int(cfg["gate7"]["judge_n"])
    combos = [(js_level, js_n, True)]
    for c in cfg["gate7"].get("comparison") or []:
        combos.append((str(c["level"]), int(c["n"]), False))
    rows, diag = [], {}
    for regime in REGIMES:
        train_rows = ds["shards"][regime]["train"]
        test_rows = ds["shards"][regime]["test"]
        for level, n, judged in combos:
            a = _union_grams(train_rows, field, level, n)
            b = _union_grams(test_rows, field, level, n)
            rate = ngram_overlap_rate(a, b)
            overlap = len(a & b)
            rows.append({
                "regime": regime, "n": n, "level": level,
                "train_ngrams": len(a), "test_ngrams": len(b), "overlap": overlap,
                "overlap_rate": _f(rate, 6),
                "threshold": (_f(threshold, 2) if judged else ""),
                # 判定行：**报告项 ⇒ `n/a`**（`D-36` 裁定一；5% 阈值未改）
                "passed": ("n/a" if judged else ""),
            })
            if judged:
                diag[regime] = {"level": level, "n": n, "train_ngrams": len(a),
                                "test_ngrams": len(b), "overlap": overlap,
                                "overlap_rate": rate, "threshold": threshold,
                                "below_threshold": bool(rate <= threshold),
                                "role": GATE7_ROLE_REPORT_ITEM}
    return rows, diag


# ---------------------------------------------------------------------------
# 对半控制（`D-36` 裁定一 · §1.2 的 ②③；转交项 42）
# ---------------------------------------------------------------------------
def _halve(rows, rng_seed):
    """**唯一的对半口径**：按 `sample_id` 字典序定序 ⇒ `random.Random(seed).shuffle` ⇒ 前后各半。

    定序是必需的：分片 CSV 的行序虽确定，但显式按 `sample_id` 定序使口径**与文件行序解耦**，
    复现者即便重排文件也能得到同一结果。
    """
    ordered = sorted(rows, key=lambda r: r.get("sample_id") or "")
    random.Random(int(rng_seed)).shuffle(ordered)
    half = len(ordered) // 2
    return ordered[:half], ordered[half:]


def _control_row(run_name, regime, split, control, key, a_rows, b_rows, note=""):
    a = _union_grams(a_rows, "text", "char", 3)
    b = _union_grams(b_rows, "text", "char", 3)
    return {"run": run_name, "regime": regime, "split": split, "control": control,
            "key": key, "n_a": len(a_rows), "n_b": len(b_rows),
            "ngrams_a": len(a), "ngrams_b": len(b), "overlap": len(a & b),
            "overlap_rate": _f(ngram_overlap_rate(a, b), 6), "note": note}


def compute_controls(split_dir, cfg, run_name):
    """② 按 `encoding_type` 对半 ＋ ③ 确定性随机对半（**报告项证据，不参与判定**）。

    用途：把「`overlap_rate` 对『跨分片』与『同分布』**不可区分**」由论断升为**实测**证据 ——
    同一分片内部、构造上**零泄露**的两半，其 `overlap_rate` 应与 `train→test` **同量级**。
    """
    spec = cfg["gate7"].get("controls") or {}
    if not spec:
        return []
    seed = int(spec.get("seed", 42))
    shards = [s.split("/") for s in (spec.get("shards") or [])]
    out = []
    for shard_spec in shards:
        if len(shard_spec) != 2 or not all(shard_spec):
            continue
        regime, split = shard_spec
        path = split_path(split_dir, regime, split)
        if not os.path.isfile(path):
            continue
        rows = [r for r in read_csv(path) if r.get("sample_kind") == "attack"]
        if not rows:
            continue
        if spec.get("random", True):
            a, b = _halve(rows, seed)
            out.append(_control_row(run_name, regime, split, "random", "-",
                                    a, b, "③ 随机对半（不按编码分层；seed=%d）" % seed))
        if spec.get("by_encoding", True):
            encodings = sorted({r.get("encoding_type") or "" for r in rows})
            encodings = [e for e in encodings if e]
            for e in encodings:
                sub = [r for r in rows if (r.get("encoding_type") or "") == e]
                if len(sub) < 2:      # 两侧各至少 1 行方可对半
                    continue
                a, b = _halve(sub, seed)
                out.append(_control_row(run_name, regime, split, "by_encoding", e,
                                        a, b, "② 同编码内对半（seed=%d）" % seed))
    return out


def _ngram_subset_rate(shards, regime, field, n, keep, judged):
    """对照诊断：把两侧行集分别限制到 `sample_kind == keep`（`keep=None` 表示全体）后重算。"""
    def sel(rows):
        return rows if keep is None else [r for r in rows if r.get("sample_kind") == keep]
    a = _union_grams(sel(shards[regime]["train"]), field, "char", n)
    b = _union_grams(sel(shards[regime]["test"]), field, "char", n)
    return ngram_overlap_rate(a, b), len(a), len(b)


# ===========================================================================
# 4. 组装 14 条断言与 3 道门禁
# ===========================================================================
def evaluate_assertions(ds, cfg, a14_info):
    shards = ds["shards"]
    g5 = cfg["gate5"]
    checks = {
        "A1": check_a1(shards),
        "A2": check_a2(shards),
        "A3": check_a3(shards),
        "A4": check_a4(shards),
        "A5": check_a5(shards, tuple(g5.get("scan_columns") or ("text", "plaintext"))),
        "A6": check_a6(shards, g5["expected_rewrite_ids"]),
        "A7": check_a7(shards, g5["expected_attack_samples"]),
        "A8": check_a8(shards, g5["expected_rows_per_rewrite"]),
        "A9": check_a9(shards, cfg["label_balance"]["band_low"], cfg["label_balance"]["band_high"]),
        "A10": check_a10(shards, ds["clean"]),
        "A11": check_a11(shards, ds["g3"]),
        "A12": check_a12(ds["ast_report"], cfg["a12"]["threshold"]),
    }
    a13_passed, a13_measured = check_a13(ds["seed_groups"], ds["seeds_meta"], cfg)
    checks["A13"] = (a13_passed, a13_measured)

    trig = a14_trigger(a14_info, cfg)
    checks["A14"] = (None, (
        "跨 train↔test 族数=%d/%d；族内 3-gram 中位数=%s（n=%d 对） vs 跨族中位数=%s（n=%d 对）；"
        "比值=%s；触发条件（字面）=%s（比值项：%s；跨档>0 且族内中位数>%.2f：%s）；"
        "**升级状态=%s**"
        % (a14_info["straddling_train_test"], a14_info["families_total"],
           _f(a14_info["within_median"], 4), a14_info["within_pairs"],
           _f(a14_info["across_median"], 4), a14_info["across_pairs"],
           _f(a14_info["ratio"], 3) if math.isfinite(a14_info["ratio"]) else "inf",
           "成立" if trig["triggered"] else "不成立",
           "成立" if trig["by_ratio"] else "不成立",
           float(cfg["a14"]["trigger_within_median"]), "成立" if trig["by_median"] else "不成立",
           ("**已生效（`family_atomic: true`；跨档族 = 0 ⇒ 无待办）**" if trig["straddle_zero"]
            else "未生效 ⇒ 有待办"))))

    evidence = {
        "A1": "split_assertions.csv", "A2": "split_assertions.csv", "A3": "split_assertions.csv",
        "A4": "split_assertions.csv", "A5": "split_assertions.csv", "A6": "split_assertions.csv",
        "A7": "split_assertions.csv", "A8": "split_assertions.csv", "A9": "split_assertions.csv",
        "A10": "split_assertions.csv", "A11": "split_assertions.csv",
        "A12": "data/interim/code_carrier/ast_report.csv", "A13": "split_assertions.csv",
        "A14": "family_straddle.csv",
    }
    out = []
    for aid, scope, desc, thr in ASSERTION_SPECS:
        ok, measured = checks[aid]
        if aid == "A14":
            out.append({"assertion_id": aid, "scope": scope, "description": desc,
                        "threshold": thr, "measured": measured, "passed": "n/a",
                        "evidence_path": evidence[aid]})
        else:
            out.append({"assertion_id": aid, "scope": scope, "description": desc,
                        "threshold": thr, "measured": measured,
                        "passed": "True" if ok else "False",
                        "evidence_path": evidence[aid]})
    return out


def gate5(shards, assertions):
    """门禁 5 = `A1` ∧ `A2` ∧ `A3` ∧ `A4` ∧ `A5`。"""
    by_id = {a["assertion_id"]: a for a in assertions}
    members = ["A1", "A2", "A3", "A4", "A5"]
    failed = [m for m in members if by_id[m]["passed"] != "True"]
    # 注意：本字符串会落进 Markdown 表格单元 ⇒ **不得含 `|`**（否则表格错列）。
    measured = "；".join("%s: %s" % (m, by_id[m]["measured"]) for m in members)
    return {"gate": 5, "passed": not failed, "failed": failed, "measured": measured,
            "threshold": "均为空集"}


def gate6(shard_stats):
    """门禁 6 = 六个分片的 `abs_diff_ratio` **全部** ≤ 阈值（两体制各自出结论）。"""
    per_regime, worst = {}, {}
    for regime in REGIMES:
        rates = {sp: shard_stats[regime][sp]["abs_diff_ratio"] for sp in SPLITS}
        per_regime[regime] = max(rates.values())
        worst[regime] = max(rates, key=lambda k: rates[k])
    passed = all(shard_stats[r][s]["passed"] for r in REGIMES for s in SPLITS)
    measured = "；".join("%s 最大 %s（%s 档）" % (r, _f(per_regime[r], 6), worst[r]) for r in REGIMES)
    return {"gate": 6, "passed": passed, "measured": measured, "per_regime": per_regime,
            "threshold": "≤ 0.20（abs_diff_ratio）"}


def gate7(ngram_diag, cfg):
    """**门禁 7（报告项）** —— `D-36` 裁定一：不再有二值判定，`passed` 恒 `None`（记 `n/a`）。

    **5% 阈值一字不改、未引入新阈值**；本函数只把实算值与阈值比较结果**如实报告**，
    **不影响退出码、不阻塞 `W3.8` 冻结**（`PREREG_v1.3.md` §1.1 第 3 条）。
    """
    threshold = float(cfg["gate7"]["threshold"])
    per_regime = {r: ngram_diag[r]["overlap_rate"] for r in REGIMES}
    measured = "；".join(
        "%s=%s（阈值 %.2f ⇒ %s；train_ngrams=%d，test_ngrams=%d，交集=%d）"
        % (r, _f(per_regime[r], 6), threshold,
           "超阈" if not ngram_diag[r]["below_threshold"] else "未超阈",
           ngram_diag[r]["train_ngrams"], ngram_diag[r]["test_ngrams"], ngram_diag[r]["overlap"])
        for r in REGIMES)
    return {"gate": 7, "role": GATE7_ROLE_REPORT_ITEM, "passed": None, "measured": measured,
            "per_regime": per_regime, "below_threshold": {r: ngram_diag[r]["below_threshold"]
                                                          for r in REGIMES},
            "threshold": "≤ 0.05（字符级 3-gram 重叠率）——**报告项，不参与判定**"}


# ===========================================================================
# 5. 主流程
# ===========================================================================
def _fail(msg):
    print("FATAL: %s" % msg)


def _inputs_for_check(split_dir):
    paths = [split_path(split_dir, r, s) for r in REGIMES for s in SPLITS]
    paths += [os.path.join(split_dir, name)
              for name in ("seed_groups.csv", "clean_original_test.csv", "g3_hard_negative_test.csv")]
    return paths


def run_check_only(args, cfg):
    """`--check-only`：**只做环境与输入可用性自检**（§2）。

    **不读全量数据、不写任何产物、不判定任何门禁、不加载 tokenizer**；**恒 `exit 0`**。
    """
    print("== 02_07 assert --check-only（探针；恒 exit 0）==")
    print("  script   : OK（可导入）")
    print("  _common  : OK（可导入；ENCODING=%d CARRIER=%d OOD_TRAIN=%d OOD_HELDOUT=%d）"
          % (len(ENCODING_CODES), len(CARRIER_CODES), len(OOD_TRAIN), len(OOD_HELDOUT)))
    print("  config   : %s（可解析）" % _rel(args.config))
    split_dir = args.split_dir or cfg["inputs"]["split_dir"]
    missing = [p for p in _inputs_for_check(split_dir) if not os.path.isfile(p)]
    if missing:
        print("  SKIP: 输入未就绪，缺失 %d 个路径（不判定、不写产物、exit 0）：" % len(missing))
        for p in missing[:6]:
            print("    - %s" % _rel(p))
        if len(missing) > 6:
            print("    - …（其余 %d 个同类）" % (len(missing) - 6))
    else:
        print("  inputs   : OK（六个分片 + seed_groups + 两张派生视图齐备）")
    print("  result   : CHECK_ONLY_EXIT=0")
    return EXIT_OK


def run(args, cfg, write_products=True):
    """全量路径：读全量数据 → 跑 `A1`–`A14` → 写三个 CSV + `family_straddle.csv` + 报告。"""
    out_dir = args.out or cfg["output"]["dir"]
    split_dir = args.split_dir or cfg["inputs"]["split_dir"]
    if not os.path.isdir(split_dir):
        _fail("缺少分片目录 %s" % _rel(split_dir))
        return EXIT_USAGE, None

    tokenizer_id = cfg["tokenizer"]["id"]
    report_dir = args.report_dir or cfg.get("report_dir", "reports/data")
    log_dir = args.log_dir or cfg.get("log_dir", "logs/data")

    ds = load_dataset(args, cfg)
    inputs_sha = {}
    fp_targets = [("splits/%s/%s.csv" % (reg, sp), split_path(split_dir, reg, sp))
                  for reg in REGIMES for sp in SPLITS]
    fp_targets += [("seed_groups.csv", os.path.join(split_dir, "seed_groups.csv")),
                   ("clean_original_test.csv", os.path.join(split_dir, "clean_original_test.csv")),
                   ("g3_hard_negative_test.csv", os.path.join(split_dir, "g3_hard_negative_test.csv")),
                   ("seeds_meta", ds["seeds_meta_path"]),
                   ("ast_report", ds["ast_report_path"])]
    for key, path in fp_targets:
        inputs_sha[key] = sha256_file(path) if os.path.isfile(path) else "MISSING"
    tok_json = os.path.join(common.project_root(), cfg["tokenizer"]["local_dir"], "tokenizer.json")
    inputs_sha["tokenizer/tokenizer.json"] = sha256_file(tok_json) if os.path.isfile(tok_json) else "MISSING"

    lrows, shard_stats = compute_length(ds, cfg, tokenizer_id)
    nrows, ngram_diag = compute_ngram(ds, cfg)
    fam_rows, a14_info = compute_a14(ds["shards"], cfg["a14"].get("overlap_field", "plaintext"))
    a14_info["n_total"] = int(ds["seeds_meta"]["seed_count_total"])
    assertions = evaluate_assertions(ds, cfg, a14_info)
    gates = [gate5(ds["shards"], assertions), gate6(shard_stats), gate7(ngram_diag, cfg)]

    # ②③ 对半控制（报告项证据）＋ 逐样本最近邻的**原子化前后对照**（D-36：「须给出两数对照」）
    crows = compute_controls(split_dir, cfg, "current")
    nn_compare = {"current": nearest_neighbour(ds["shards"])}
    for alt in (cfg["gate7"]["controls"].get("also_runs") or []):
        alt_dir = alt.get("dir")
        if alt_dir and os.path.isdir(alt_dir):
            name = str(alt.get("name") or "alt")
            crows += compute_controls(alt_dir, cfg, name)
            nn_compare[name] = nearest_neighbour(load_shards(alt_dir))

    ts = _stamp()
    if write_products:
        write_csv(os.path.join(out_dir, cfg["output"]["assertions"]), assertions, ASSERTION_FIELDS)
        write_csv(os.path.join(out_dir, cfg["output"]["length"]), lrows, LENGTH_FIELDS)
        write_csv(os.path.join(out_dir, cfg["output"]["ngram"]), nrows, NGRAM_FIELDS)
        write_csv(os.path.join(out_dir, cfg["output"]["ngram_controls"]), crows, CONTROL_FIELDS)
        write_csv(os.path.join(out_dir, cfg["output"]["family_straddle"]), fam_rows, FAMILY_FIELDS)

    diagnostics = _diagnostics(ds, cfg)

    summary = {
        "window": cfg.get("window", "W3.7"),
        "mode": "report-only" if not write_products else "full",
        "timestamp": ts,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.perf_counter() - _T0, 3),
        "config": _rel(args.config),
        "split_dir": _rel(split_dir),
        "out_dir": _rel(out_dir),
        "tokenizer_id": tokenizer_id,
        "tokenizer_hf_revision": cfg["tokenizer"]["hf_revision"],
        "tokenizer_local_dir": cfg["tokenizer"]["local_dir"],
        "inputs": inputs_sha,
        "gates": [{"gate": g["gate"], "role": g.get("role", "gate"), "passed": g["passed"],
                   "measured": g["measured"], "threshold": g["threshold"]} for g in gates],
        "assertions": {a["assertion_id"]: {"scope": a["scope"], "measured": a["measured"],
                                           "passed": a["passed"], "threshold": a["threshold"]}
                       for a in assertions},
        "a14": a14_info,
        "a14_trigger": a14_trigger(a14_info, cfg),
        "shards": {r: {s: shard_stats[r][s] for s in SPLITS} for r in REGIMES},
        "ngram": ngram_diag,
        "runs_ngram_rows": nrows,
        "controls": crows,
        "nn_compare": nn_compare,
        "diagnostics": diagnostics,
        "argv": " ".join(sys.argv),
    }

    report_path = os.path.join(report_dir, "%s_%s.md" % (cfg.get("report_basename", "g567_assert_v1.0"), ts))
    if write_products:
        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_report(summary, cfg, args))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "02_07_assert_splits_%s.log" % ts)
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        summary["report_path"] = _rel(report_path)
        summary["log_path"] = _rel(log_path)

    summary["exit_code"] = compute_exit_code(gates, assertions)
    return summary["exit_code"], summary


def compute_exit_code(gates, assertions):
    """退出码的**唯一口径**（`D-36` 裁定一 后）。

    非零**当且仅当**：门禁 5 或门禁 6 不通过，**或** 门禁类断言（`A1`–`A13`）有任一非 `True`。
    **门禁 7 为报告项（`role = report_item`、`passed = None`）⇒ 不参与退出码、不阻塞 `W3.8` 冻结**
    （`windows/W3.8_freeze.md` `H5`：`exit=0` 且 `A1`–`A13` 全 `True`，行数 14）。
    `A14` 为非门禁报告性断言，同样不参与。
    """
    bad_gate = any(g.get("role", "gate") == "gate" and not g["passed"] for g in gates)
    bad_assertion = any(a["passed"] != "True" for a in assertions if a["assertion_id"] != "A14")
    return EXIT_OK if not (bad_gate or bad_assertion) else EXIT_FAIL


def _fmt_ratio(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "inf" if math.isinf(v) else _f(v, 3)


def render_report(summary, cfg, args):
    """运行报告 `reports/data/g567_assert_v1.0_<YYYYMMDD-HHMMSS>.md`（§4 的载体）。"""
    g = {x["gate"]: x for x in summary["gates"]}
    a = summary["assertions"]
    L = []
    L.append("# W3.7 运行报告 — 自动断言、长度分布、n-gram 重叠（`02_07`）")
    L.append("")
    L.append("- 窗口：**W3.7**（交付物窗口，权限 A3）")
    L.append("- 日期：%s（耗时 %.1f s）" % (summary["timestamp"], summary["elapsed_sec"]))
    L.append("- 配置：`%s`；分片目录 `%s`；产物目录 `%s`"
             % (summary["config"], summary["split_dir"], summary["out_dir"]))
    L.append("- 运行模式：`%s`；命令：`%s`" % (summary["mode"], summary["argv"]))
    L.append("- tokenizer：`%s`（长度口径，`ENV_LOCK.md` §5 登记的主检测器 tokenizer）"
             % summary["tokenizer_id"])
    L.append("  - HF revision：`%s`" % summary["tokenizer_hf_revision"])
    L.append("  - 本地目录：`%s`（魔搭 `master` 会漂移，故附 `tokenizer.json` sha256 作完整性锚点）"
             % summary["tokenizer_local_dir"])
    L.append("")
    L.append("> **声明：「全部断言由脚本执行、无人工核对」。** 本报告的每一个结论（门禁 5／6／7 的三色判定、")
    L.append("> `A1`–`A14` 的逐条结果、全部实算值）均由 `scripts/data/02_07_assert_splits.py` 单次运行")
    L.append("> 自动产出，**无任何人工核对环节**（`windows/W3_stage2_data.md` §4-2；v5 §2.6「新增 15」：")
    L.append("> 「不光靠人看，靠代码断言，防止人眼漏过泄露」）。")
    L.append("")
    L.append("## §0 输入指纹（sha256）")
    L.append("")
    L.append("| 输入 | sha256 |")
    L.append("|---|---|")
    for k, v in sorted(summary["inputs"].items()):
        L.append("| `%s` | `%s` |" % (k, v))
    L.append("")
    L.append("## §1 门禁 5／6／7 结论（**本窗口的判定职责**）")
    L.append("")
    L.append("| 门禁 | 角色 | 结论 | 实算值 | 阈值 | 不通过时的回退建议（`PROJECT_CONTRACT.md` §4） |")
    L.append("|---|---|---|---|---|---|")
    L.append("| 5 · group 交集 / OOD 编码 | 二值门禁 | %s | %s | 均为空集 | 回退 **2.6** 重新划分 |"
             % ("**通过**" if g[5]["passed"] else "**不通过**", g[5]["measured"]))
    L.append("| 6 · 长度分布差值 | 二值门禁 | %s | %s | ≤20%% | 回退 **2.6** 做**长度分层采样**（`02_06 --stratify-length`） |"
             % ("**通过**" if g[6]["passed"] else "**不通过**", g[6]["measured"]))
    L.append("| 7 · 3-gram 重叠率 | **报告项**（`D-36` 裁定一） | **`n/a`**（无通过/不通过） | %s | ≤5%%（**未改**，仅作对照） | **不再触发**（§5.2 证其在任何重划分下都无法满足 5%%） |"
             % g[7]["measured"])
    L.append("")
    L.append("> **门禁 7 的角色变更**（2026-10-02，`DECISIONS.md` `D-36` 裁定一；承接件 `PREREG_v1.3.md` §1）：")
    L.append("> 该量由**二值门禁**改为**报告项** —— **5% 阈值一字不改、未引入任何新阈值**，")
    L.append("> 但**不再有通过/不通过**（`ngram_overlap.csv` 的判定行 `passed` 记 `n/a`），")
    L.append("> **不参与退出码、不阻塞 `W3.8` 冻结**。泄露防护改由**四条**承担：")
    L.append("> `A1`／`A2`（组级）、`A3`–`A5`（编码级）、`A14`（逐样本最近邻）、`family_atomic: true`。")
    L.append("> 三项并列证据见 **§5.2**。")
    L.append("")
    blocking = [x for x in summary["gates"] if x.get("role", "gate") == "gate" and not x["passed"]]
    bad_assert = [i for i in GATING_ASSERTIONS
                  if a[i]["passed"] != "True"]
    L.append("- **总体**：%s（脚本退出码 `%d`）"
             % (("门禁 5／6 全部通过 且 `A1`–`A13` 全 `True`" if not (blocking or bad_assert)
                 else "**存在阻塞项**：%s" % ", ".join(["门禁 %d" % x["gate"] for x in blocking] + bad_assert)),
                summary.get("exit_code", EXIT_FAIL)))
    L.append("")
    L.append("## §2 `A1`–`A14` 一行式汇总（**14 行，不得缺行**）")
    L.append("")
    L.append("| 断言 | scope | 门禁类 | 阈值 | 实测 | 通过 |")
    L.append("|---|---|---|---|---|---|")
    for aid, scope, _desc, thr in ASSERTION_SPECS:
        row = a[aid]
        passed = row["passed"]
        mark = "n/a（报告性，非门禁）" if passed == "n/a" else ("**True**" if passed == "True" else "**False**")
        L.append("| `%s` | `%s` | %s | %s | %s | %s |"
                 % (aid, scope, "否" if aid == "A14" else "是", thr, row["measured"], mark))
    L.append("")
    L.append("> `A14` 的 `passed` 恒记 `n/a`：本断言**不设通过/不通过**，**不得因其非零退出**。")
    L.append("> `A13` 的三档实算计数见上表 `实测` 列（`train=…`、`val=…`、`test=…` 逐一打印）。")
    L.append("")
    L.append("## §3 度量口径（**逐字**，供复核与论文引用）")
    L.append("")
    L.append("### 3.1 门禁 6 · 长度分布差值")
    L.append("")
    L.append("- 长度口径：**token 数**，tokenizer **一律**取自 `ENV_LOCK.md` §5 登记的**主检测器 tokenizer**"
             "（`%s`，HF revision `%s`）；" % (summary["tokenizer_id"], summary["tokenizer_hf_revision"]))
    L.append("  分词对象 = 分片 CSV 的 **`text`** 字段（阶段3–5 的唯一输入字段）。")
    L.append("- 唯一口径（与本脚本实现逐字一致）：")
    L.append("  `abs_diff_ratio = |均值(正) − 均值(负)| / max(均值(正), 均值(负))`，判据 `≤ 0.20`。")
    L.append("- 正 = `label == 1`（攻击侧），负 = `label == 0`（良性侧）；**逐分片**计算，**六个分片全过**方为门禁通过。")
    L.append("")
    L.append("### 3.2 门禁 7 · n-gram 重叠率")
    L.append("")
    L.append("- 唯一口径（与本脚本实现逐字一致）：")
    L.append("  `overlap_rate = |ngram(train) ∩ ngram(test)| / |ngram(test)|`，判据 `≤ 0.05`。")
    L.append("- **判定只认字符级 3-gram**（`level=char`、`n=3`）；**词级 3-gram** 与**字符级 4-gram** "
             "一并写入 `ngram_overlap.csv`，**仅作对照，`passed` 列留空**。")
    L.append("- `train` 侧取该体制 `train` 分片**全体** `text`，`test` 侧取该体制 `test` 分片**全体** `text`；")
    L.append("  两侧各自先做**并集**（`_common.char_ngrams`），再算交集与比值。**两套体制各出一组结论。**")
    L.append("")
    L.append("### 3.3 `A13` 期望值所用的舍入规则（**易错项**）")
    L.append("")
    L.append("- 规则：`n_train = round(N × 0.8)`、`n_val = round(N × 0.1)`、`n_test = N − n_train − n_val`。")
    L.append("  这里的 `round(...)` 一律指 **Python 内置 `round()`**，即 **银行家舍入（round-half-to-even）**，")
    L.append("  **不是四舍五入**。末档取 `N − 前两者`，**保证三档之和恒等于 `N`**（避免独立取整造成总和漂移）。")
    L.append("- **工作示例（与实现逐字符一致）**：")
    L.append("  - `N = 220`：`220×0.8 = 176.0 → 176`；`220×0.1 = 22.0 → 22`；`220 − 176 − 22 = **22**` ⇒ **176 : 22 : 22**；")
    L.append("  - `N = 185`（**反例，用于证明规则重要**）：`185×0.8 = 148.0 → 148`；`185×0.1 = 18.5 → `")
    L.append("    **`18`**（银行家舍入取偶）；`185 − 148 − 18 = **19**` ⇒ **148 : 18 : 19**。")
    L.append("    **若按四舍五入**则 `18.5 → 19` ⇒ `148 : 19 : 18`，**与上述不同** ⇒ 必须以本规则为准。")
    L.append("- `N` 的唯一来源 = `data/interim/seed_collect/seeds.meta.json` 的 `seed_count_total`，")
    L.append("  **禁止硬编码**；本次实算 `N = %s`。" % summary["a14"].get("n_total", "—"))
    L.append("")
    L.append("## §4 门禁 6 明细（`length_distribution.csv`）")
    L.append("")
    L.append("| 体制 | 分片 | 正 n | 负 n | 正负比 | 均值(正) | 均值(负) | `abs_diff_ratio` | 阈值 | 通过 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for regime in REGIMES:
        for sp in SPLITS:
            s = summary["shards"][regime][sp]
            ratio = (s["n_pos"] / float(s["n_neg"])) if s["n_neg"] else float("inf")
            L.append("| `%s` | `%s` | %d | %d | %.4f | %.4f | %.4f | **%s** | 0.20 | %s |"
                     % (regime, sp, s["n_pos"], s["n_neg"], ratio, s["pos_mean"], s["neg_mean"],
                        _f(s["abs_diff_ratio"], 6), "**是**" if s["passed"] else "**否**"))
    L.append("")
    L.append("## §5 门禁 7 明细（`ngram_overlap.csv`；**报告项**）")
    L.append("")
    L.append("> 判定行（字符级 3-gram）的 `passed` 列记 **`n/a`** —— 本量为**报告项**，")
    L.append("> **无通过/不通过**（`D-36` 裁定一）。`threshold` 列照旧写出 0.05，供读者自行对照实算值。")
    L.append("")
    L.append("| 体制 | n | level | `train_ngrams` | `test_ngrams` | 交集 | `overlap_rate` | 阈值 | 判定 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in summary["runs_ngram_rows"]:
        judged = bool(r["threshold"])
        verdict = "—"
        if judged:
            below = summary["ngram"][r["regime"]]["below_threshold"]
            verdict = "未超阈（仍**非门禁**）" if below else "超阈（**不触发回退**）"
        L.append("| `%s` | %d | `%s` | %d | %d | %d | %s | %s | %s |"
                 % (r["regime"], r["n"], r["level"], r["train_ngrams"], r["test_ngrams"],
                    r["overlap"], r["overlap_rate"], r["threshold"] or "—", verdict))
    L.append("")
    L.append("### 5.1 根因分解（**诊断，不参与判定**）")
    L.append("")
    L.append("`|ngram(train) ∩ ngram(test)| / |ngram(test)|` 是**语料级词表包含度**：")
    L.append("两侧词表的交集含**语言公共 3-gram**（英文词形、代码标点/关键字）与**编码字母表 3-gram**")
    L.append("（`zerowidth` 的零宽位流、`morse` 的点划、`base64`/`hex` 的字符集、`url` 的百分号形态）。")
    L.append("这些 3-gram 由**编码方案本身**决定，**同一编码在两档必然共享** ⇒ 该量对同域语料存在**结构下限**。")
    L.append("")
    L.append("**按侧拆分**（对象 = `text`，字符级 3-gram）：")
    L.append("")
    L.append("| 体制 | 侧 | `train_ngrams` | `test_ngrams` | `overlap_rate` |")
    L.append("|---|---|---|---|---|")
    for regime in REGIMES:
        for tag, zh in (("all", "全体"), ("attack", "仅攻击"), ("benign", "仅良性")):
            d = summary["diagnostics"]["by_side"][regime][tag]
            L.append("| `%s` | %s | %d | %d | %s |" % (regime, zh, d["train_ngrams"], d["test_ngrams"], _f(d["overlap_rate"], 6)))
    L.append("")
    L.append("**攻击侧按编码拆分（体制一全视图）** —— 同一编码在 train 侧与 test 侧的重叠率。")
    L.append("此表说明重叠由**编码字母表**主导：同一编码两档共享的 3-gram 几乎为其全部词表。")
    L.append("（体制二**不做**逐编码拆：其 `train`／`val` 与 `test` 的编码集**互不相交**，故逐编码两侧必有一侧为空；")
    L.append("体制二的攻击侧重叠率见上表「仅攻击」行 = %s。）"
             % _f(summary["diagnostics"]["by_side"]["regime_ood"]["attack"]["overlap_rate"], 6))
    L.append("")
    L.append("| 编码 | `train_ngrams` | `test_ngrams` | `overlap_rate` |")
    L.append("|---|---|---|---|")
    for e in ENCODING_CODES:
        d = summary["diagnostics"]["by_encoding_attack"]["regime_id"].get(e)
        if d is None:
            L.append("| `%s` | — | — | — |" % e)
        else:
            L.append("| `%s` | %d | %d | %s |"
                     % (e, d["train_ngrams"], d["test_ngrams"], _f(d["overlap_rate"], 6)))
    L.append("")
    L.append("**逐样本最近邻**（对象 = 改写体**明文**，按 `rewrite_id` 去重；对训练侧**全枚举**取最大值）：")
    L.append("这是「改写体与原句太接近」这一**语义泄露**命题在样本层面的直接度量。")
    L.append("")
    L.append("| 体制 | 测试改写体数 | 训练改写体数 | 最近邻中位数 | p95 | 最大 | 最近邻 ≥0.5 占比 |")
    L.append("|---|---|---|---|---|---|---|")
    for regime in REGIMES:
        d = summary["diagnostics"]["nearest_neighbour"][regime]
        L.append("| `%s` | %d | %d | **%s** | %s | %s | %s |"
                 % (regime, d["n_test"], d["n_train"], _f(d["median"], 4), _f(d["p95"], 4),
                    _f(d["max"], 4), _f(d["frac_ge_0.5"], 4)))
    L.append("")
    nn = summary.get("nn_compare") or {}
    if len(nn) > 1:
        L.append("**原子化前后对照**（同一口径：`plaintext`、按 `rewrite_id` 去重、对训练侧全枚举）：")
        L.append("")
        L.append("| 运行 | 体制 | 测试改写体数 | 最近邻中位数 | p95 | 最大 | **最近邻 ≥0.5 占比** |")
        L.append("|---|---|---|---|---|---|---|")
        for name in nn:
            for regime in REGIMES:
                d = nn[name][regime]
                L.append("| `%s` | `%s` | %d | %s | %s | %s | **%s** |"
                         % (name, regime, d["n_test"], _f(d["median"], 4), _f(d["p95"], 4),
                            _f(d["max"], 4), _f(d["frac_ge_0.5"], 4)))
        L.append("")
        L.append("> **必须与 `PREREG_v1.3.md` §2 ④／§3.2 的数对照阅读**：该两处记「原子化后 ≥0.5 占比")
        L.append("> `0.8636 → 0.0000`（中位数 `0.6968 → 0.3560`、max `0.8261 → 0.3904`）」，")
        L.append("> 其来源是 `W3` 的**剔除式代理**（「剔除与 `train` 同族的 test 攻击行」，`792 → 108`）——")
        L.append("> `D-36` 裁定二「★ 更正」条已明确该代理**不是**原子重划分的后果（真实重划分按 `V-3` 的")
        L.append("> `176:22:22` **以族为单位重排**，`test` 仍为 **792** 攻击行）。**上表 `current` 行**")
        L.append("> 才是 `W3.6` r2 真实重划分后的**实测**值。**两套数并存、各自标注来源**（`D-12`：")
        L.append("> `PREREG_v1.3.md` 为冻结件，**不得改写**；承接与更正落在本报告与 `DATA_CARD`）。")
        L.append("")
    L.append("### 5.2 三项并列证据（`D-36` 裁定一 · `PREREG_v1.3.md` §1.2）")
    L.append("")
    L.append("> 门禁 7 改**报告项**后，本窗口**必须并列**呈现下列三项；**不得**只写 `n/a`。")
    L.append("> `DATA_CARD_CodeObf-PI_v1.0.md` 亦须含此三项（`D-36` 转交项 42）。")
    L.append("")
    th = float(cfg["gate7"]["threshold"])
    L.append("#### ① 阈值与实测")
    L.append("")
    L.append("| 体制 | 字符级 3-gram `overlap_rate`（判定量） | 阈值（**未改**） |")
    L.append("|---|---|---|")
    for regime in REGIMES:
        L.append("| `%s` | **%s** | %.2f |" % (regime, _f(summary["ngram"][regime]["overlap_rate"], 6), th))
    L.append("")
    L.append("对照口径（**仅作对照，`passed` 列留空**）：见上表 `level=word`／`n=4` 行。")
    L.append("")
    crows = summary.get("controls") or []
    L.append("#### ② 对半控制（按 `encoding_type`；同分布、构造上**零泄露**）　③ 随机对半控制（**本窗口补测**）")
    L.append("")
    if not crows:
        L.append("（本次运行未产出对半控制 —— `gate7.controls` 未启用或分片缺失。）")
    else:
        L.append("口径（**本窗口声明，唯一**）：取该分片的**攻击行** ⇒ 按 `sample_id` 字典序定序 ⇒ "
                 "`random.Random(%d).shuffle` ⇒ 前后各半 ⇒ " % int(cfg["gate7"]["controls"]["seed"]))
        L.append("`overlap_rate = |ngram(A) ∩ ngram(B)| / |ngram(B)|`（字符级 3-gram，两侧各自并集）。")
        L.append("② 在**每个 `encoding_type` 内部**各自对半；③ 对**整个分片**的攻击行对半（不按编码分层）。")
        L.append("")
        L.append("| 运行 | 体制 | 分片 | 控制 | key | n(A) | n(B) | 交集 | `overlap_rate` |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in crows:
            L.append("| `%s` | `%s` | `%s` | %s | `%s` | %d | %d | %d | **%s** |"
                     % (r["run"], r["regime"], r["split"],
                        {"by_encoding": "② 同编码内对半", "random": "③ 随机对半"}.get(r["control"], r["control"]),
                        r["key"], r["n_a"], r["n_b"], r["overlap"], r["overlap_rate"]))
        L.append("")
        cur = [r for r in crows if r["run"] == "current" and r["control"] == "random"]
        if cur:
            L.append("**③ 的读数（当前划分）**：%s。"
                     % "；".join("`%s/%s` = **%s**" % (r["regime"], r["split"], r["overlap_rate"])
                                for r in cur))
            L.append("与 ① 的 `train→test` 值对比 ⇒ **同一分片内部的两半（构造上零泄露）与跨分片取到同一量级**，")
            L.append("故该量**对「跨分片」与「同分布」不可区分** —— 这正是把 `D-36` 裁定一的论断**升为实测证据**的那一项。")
            L.append("")
    L.append("#### ② 的历史记录（**原子化前**，由 `W3` 实测；`PREREG_v1.3.md` §1.2 ② 所引）")
    L.append("")
    L.append("> `PREREG_v1.3.md` §1.2 ② 的数值（`base64` 0.9177／`rot13` 0.8336／`morse` 0.9774／"
             "`ascii_art` 0.9799／`zerowidth` 0.9216）出自 `W3` 在 `2026-10-02` 的实测（见")
    L.append("> `reports/data/g567_assert_v1.0_20261002-195830.md` §9.3「呈报一」第 1 条），**属原子化前的旧划分**。")
    L.append("> **不得**改写该历史记录（`D-12`／`D-21`）。本窗口的 ②（上表 `current` 行）按**本窗口声明的口径**")
    L.append("> 在**原子化后的新划分**上重算，故数值与之不同 —— **两者并存、各自标注所属划分**。")
    L.append("")
    L.append("## §6 `A14` 语义族跨档登记（**报告性断言，非门禁**）")
    L.append("")
    info = summary["a14"]
    trig = summary["a14_trigger"]
    L.append("产物：`%s`（字段 `family_id, group_count, train, val, test`；计数单位 = **group**）。"
             % cfg["output"]["family_straddle"])
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 族总数 | %d |" % info["families_total"])
    L.append("| **跨 `train↔test` 的族数** | **%d** |" % info["straddling_train_test"])
    L.append("| 度量对象 | `%s`（按 `rewrite_id` 去重） |" % info["overlap_field"])
    L.append("| 参与配对的 `rewrite_id` | %d |" % info["rewrite_ids"])
    L.append("| **族内变体两两**（字符 3-gram 重叠率）中位数 | **%s**（n=%d 对，全枚举） |"
             % (_f(info["within_median"], 4), info["within_pairs"]))
    L.append("| **跨族** 中位数 | **%s**（n=%d 对，全枚举） |"
             % (_f(info["across_median"], 4), info["across_pairs"]))
    L.append("| 族内 p95 | %s |" % _f(info["within_p95"], 4))
    L.append("| 比值（族内中位数 ÷ 跨族中位数） | **%s** |" % _fmt_ratio(info["ratio"]))
    L.append("")
    L.append("**触发式升级判定**（`windows/W3.6_split.md` §2.3.2）。")
    L.append("**现状**：`family_atomic: true` **已由 `D-36` 裁定二批准并由 `W3.6` 补正轮 r2 落地** ⇒")
    L.append("本次运行即为**原子化后**的复算，**跨档族应为 0**。")
    L.append("")
    L.append("> **读法提示**：本节「族内／跨族中位数」度量的是**全量改写体两两配对**，")
    L.append("> 与「同族是否跨档」**无关** ⇒ 原子化**不改变**这两个数（0.6328／0.1899 仍成立）。")
    L.append("> 原子化真正改变的是 **跨档族数（17 → 0）** 与 **逐样本跨分片最近邻**（见 §5.1）。")
    L.append("")
    L.append("| 触发条件 | 阈值 | 实测 | 是否触发 |")
    L.append("|---|---|---|---|")
    L.append("| 族内中位数 ≥ 跨族中位数 × %.1f | — | %s vs %s | **%s** |"
             % (float(cfg["a14"]["trigger_ratio"]), _f(info["within_median"], 4),
                _f(info["across_median"], 4), "是" if trig["by_ratio"] else "否"))
    L.append("| 跨档族数 > 0 **且** 族内中位数 > %.2f | — | 跨档族数=%d、族内中位数=%s | **%s** |"
             % (float(cfg["a14"]["trigger_within_median"]), info["straddling_train_test"],
                _f(info["within_median"], 4), "是" if trig["by_median"] else "否"))
    L.append("")
    L.append("⇒ **触发式升级状态：%s**。%s"
             % ("已生效、无待办" if trig["straddle_zero"]
                else ("触发、待处置" if trig["triggered"] else "未触发、无需升级"),
                ("跨 `train↔test` 族数 **0** ⇒ `family_atomic: true` **已由 `D-36` 裁定二批准"
                 "并经 `W3.6` 补正轮 r2 落地**，本窗口只做**复算**。"
                 "（比值项按字面仍成立 —— 族内变体相似度天然高于跨族，属**语料属性**；"
                 "但因同族全部分组已落同一分片，**它不再构成跨档泄露**，故不产生任何待办动作。）"
                 if trig["straddle_zero"] else
                 "按 `windows/W3.6_split.md` §2.3.2，本窗口**不得自行切换**，须呈报 `W3`，"
                 "由 `W3` 呈 `W0` 裁定是否置 `family_atomic: true` 并重划分")))
    L.append("")
    L.append("## §7 结论与呈报")
    L.append("")
    L.append("- 门禁 5：**%s**（二值门禁）；门禁 6：**%s**（二值门禁）。"
             % (["不通过", "通过"][g[5]["passed"]], ["不通过", "通过"][g[6]["passed"]]))
    L.append("- 门禁 7：**报告项（`n/a`，无通过/不通过）** —— 实算 %s；**不阻塞 `W3.8` 冻结**。"
             % "／".join("%s=%s" % (r, _f(summary["ngram"][r]["overlap_rate"], 6)) for r in REGIMES))
    L.append("- 门禁类断言（`A1`–`A13`）不通过者：%s。"
             % (", ".join(i for i in GATING_ASSERTIONS if a[i]["passed"] != "True") or "无"))
    if info["straddling_train_test"] == 0:
        L.append("- **`A14`**：跨 `train↔test` 族数 **0** ⇒ 原子化（`family_atomic: true`，`D-36` 裁定二）"
                 "**已生效**，**无待办动作**（比值项按字面仍成立，但同族已同档 ⇒ 不构成跨档泄露）。")
    elif trig["triggered"]:
        L.append("- **`A14` 触发式升级条件成立**（跨 train↔test 族数 %d、族内中位数 %s）⇒ 依"
                 " `windows/W3.6_split.md` §2.3.2 **须呈报 `W3`**（本窗口不自行置 `family_atomic`）。"
                 % (info["straddling_train_test"], _f(info["within_median"], 4)))
    L.append("- **产物**：`%s`、`%s`、`%s`、`%s`、`%s`（均在 `%s`）"
             % (cfg["output"]["assertions"], cfg["output"]["length"], cfg["output"]["ngram"],
                cfg["output"]["ngram_controls"], cfg["output"]["family_straddle"],
                summary["out_dir"]))
    L.append("- 失败／回退：写 `WINDOW_LOG.md` 一行 + 产物路径；偏离写 `DEVIATION_LEDGER.md`（仅追加）。")
    L.append("- **门禁 5／6 或 `A1`–`A13` 不通过时须呈报 `W3`**（`windows/W3.7_assert.md` §8-10）；")
    L.append("  门禁 7 为报告项，**不触发**该呈报（`D-36` 裁定一第 3／4 条）。")
    L.append("")
    return "\n".join(L) + "\n"


def nearest_neighbour(shards, field="plaintext"):
    """逐样本最近邻统计（**语义泄露命题的直接度量**）。

    对象 = 改写体**明文**，**按 `rewrite_id` 去重**（与 `D-36` 零节所记的 `0.6471` 同一基数）；
    对训练侧**全枚举**取最大值。返回 `{regime: 统计}`。
    """
    out = {}
    for regime in REGIMES:
        trp, tep = {}, {}
        for r in shards[regime]["train"]:
            if r.get("sample_kind") == "attack" and r.get("rewrite_id") and r.get(field):
                trp[r["rewrite_id"]] = char_ngrams(r[field], 3)
        for r in shards[regime]["test"]:
            if r.get("sample_kind") == "attack" and r.get("rewrite_id") and r.get(field):
                tep[r["rewrite_id"]] = char_ngrams(r[field], 3)
        best = [max((ngram_overlap_rate(t, b) for t in trp.values()), default=0.0)
                for b in tep.values()]
        best.sort()
        out[regime] = {
            "field": field, "n_test": len(tep), "n_train": len(trp),
            "median": round(_median(best), 6), "p95": round(_pct(best, 0.95), 6),
            "max": round(max(best), 6) if best else 0.0,
            "frac_ge_0.5": round(sum(1 for v in best if v >= 0.5) / float(len(best)), 6) if best else 0.0,
        }
    return out


def _diagnostics(ds, cfg):
    """门禁 7 的**根因分解**（明确标注为诊断，不参与判定）。

    动机：`|ngram(train) ∩ ngram(test)| / |ngram(test)|` 是**语料级词表包含度**，
    对同域语料其下限天然接近 1（训练与测试共享语言／编码字母表的全部 3-gram）。
    本诊断把该量拆到「侧」（攻击／良性）与「编码」上，并给出**逐样本最近邻**证据，
    供 W3／W0 判定该口径是否可判。
    """
    shards = ds["shards"]
    diag = {"by_side": {}, "by_encoding_attack": {}, "nearest_neighbour": {}}
    for regime in REGIMES:
        diag["by_side"][regime] = {}
        for keep, tag in ((None, "all"), ("attack", "attack"), ("benign", "benign")):
            rate, na, nb = _ngram_subset_rate(shards, regime, "text", 3, keep, True)
            diag["by_side"][regime][tag] = {"overlap_rate": round(rate, 6),
                                            "train_ngrams": na, "test_ngrams": nb}
        # 攻击侧按编码拆：**在体制一全视图上做**（regime_id 三档同含 9 编码 ⇒ 每个编码两侧都有行，
        # 表才有信息量；体制二因 train/val 与 test 的编码集不相交，逐编码拆必然全空）。
        enc = {}
        for e in ENCODING_CODES:
            tr = [r for r in shards["regime_id"]["train"] if r.get("sample_kind") == "attack"
                  and r.get("encoding_type") == e]
            te = [r for r in shards["regime_id"]["test"] if r.get("sample_kind") == "attack"
                  and r.get("encoding_type") == e]
            if not tr or not te:
                enc[e] = None
                continue
            a = _union_grams(tr, "text", "char", 3)
            b = _union_grams(te, "text", "char", 3)
            enc[e] = {"overlap_rate": round(ngram_overlap_rate(a, b), 6),
                      "train_ngrams": len(a), "test_ngrams": len(b)}
        diag["by_encoding_attack"][regime] = enc
    diag["nearest_neighbour"] = nearest_neighbour(shards)
    return diag


# ===========================================================================
# 6. CLI
# ===========================================================================
def load_config(path):
    try:
        import yaml
    except ImportError:
        return None, "需要 PyYAML 解析 %s" % _rel(path)
    if not os.path.isfile(path):
        return None, "缺少配置 %s" % _rel(path)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh), None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="W3.7 自动断言、长度分布、n-gram 重叠（02_07）—— 门禁 5/6/7 的唯一判定者")
    p.add_argument("--config", default="configs/data/assert.yaml")
    p.add_argument("--split-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--seeds-meta", default=None)
    p.add_argument("--ast-report", default=None)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--check-only", action="store_true",
                   help="只做环境与输入可用性自检；不读全量数据、不写产物、不判定门禁；恒 exit 0")
    p.add_argument("--report-only", action="store_true",
                   help="只重算并渲染报告，不重写任何 CSV")
    return p.parse_args(argv)


def init_streams():
    """把 `stdout`／`stderr` 重配为 **UTF-8**（`F-1`，`W3.7` 补正轮 `r2`；`D-36` 验收附注 §11.4）。

    动机（**环境相关的假失败**）：本窗口的 stdout 含 `⇒`／`≤`／`†` 等**非 GBK 字符**；
    本机中文 Windows 的默认 stdout 编码是 **GBK**（如 `python … > out.txt` 且不设
    `PYTHONIOENCODING`）⇒ 打印门禁 7 行时抛 `UnicodeEncodeError: 'gbk' codec can't encode
    character '\\u21d2'` ⇒ **`exit=1`**。而 `windows/W3.8_freeze.md` 的 `H5` 判据正是
    「`02_07` 默认模式 `exit=0`」⇒ 若 `W3.8` 以重定向方式取证会**假失败**。
    重配后 `exit=0` 与 stdout 编码**无关**（`errors="replace"` 保证任何流上都不再抛错）。

    **本函数只动输出流，不触碰任何判定值、阈值或产物内容**（`r2` 纪律第 1 条）。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:      # 例如被测试夹具替换为无 reconfigure 的包装对象
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # 流已关闭／不可重配 ⇒ 保持原样，不因修复本身报错
            pass


def main(argv=None):
    init_streams()
    args = parse_args(argv)

    # `--check-only` 必须先于一切「重」动作：配置缺失/不可解析也不得让 make check 变红。
    cfg, err = load_config(args.config)
    if cfg is None:
        if args.check_only:
            print("== 02_07 assert --check-only（探针；恒 exit 0）==")
            print("  script   : OK（可导入）")
            print("  SKIP: %s" % err)
            print("  result   : CHECK_ONLY_EXIT=0")
            return EXIT_OK
        _fail(err)
        return EXIT_USAGE

    if args.check_only:
        return run_check_only(args, cfg)

    missing = []
    split_dir = args.split_dir or cfg["inputs"]["split_dir"]
    missing += [p for p in _inputs_for_check(split_dir) if not os.path.isfile(p)]
    seeds_meta = args.seeds_meta or cfg["inputs"]["seeds_meta"]
    ast_report = args.ast_report or cfg["inputs"]["ast_report"]
    if not os.path.isfile(seeds_meta):
        missing.append(seeds_meta)
    if not os.path.isfile(ast_report):
        missing.append(ast_report)
    if missing:
        _fail("缺少输入：%s" % ", ".join(_rel(m) for m in missing[:6]))
        return EXIT_USAGE

    exit_code, summary = run(args, cfg, write_products=not args.report_only)
    if summary is None:
        return exit_code

    print("== 02_07 assert ==")
    for gate in summary["gates"]:
        role = gate.get("role", "gate")
        verdict = ("n/a" if role != "gate" else ("PASS" if gate["passed"] else "FAIL"))
        print("  gate %d %-4s %s%s" % (gate["gate"], verdict,
                                       "[报告项] " if role != "gate" else "", gate["measured"]))
    for aid, _scope, _desc, _thr in ASSERTION_SPECS:
        row = summary["assertions"][aid]
        if aid in ("A6", "A7"):
            print("  %-4s %-4s %s" % (aid, row["passed"], row["measured"]))
    if summary.get("report_path"):
        print("  report: %s" % summary["report_path"])
    if summary.get("log_path"):
        print("  log   : %s" % summary["log_path"])
    blocking = [g["gate"] for g in summary["gates"]
                if g.get("role", "gate") == "gate" and not g["passed"]]
    bad_assert = [i for i in GATING_ASSERTIONS
                  if summary["assertions"][i]["passed"] != "True"]
    if blocking or bad_assert:
        print("FAILED: %s" % ", ".join(["gate %d" % x for x in blocking] + bad_assert))
    else:
        print("OK: 门禁 5/6 通过 且 A1–A13 全 True（门禁 7 = 报告项 n/a；A14 非门禁）")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

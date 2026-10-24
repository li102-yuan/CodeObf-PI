# -*- coding: utf-8 -*-
"""02_06_split_group_aware.py — W3.6 group-aware 划分与两套体制。

接口契约（`windows/W3.6_split.md` §3）::

    python scripts/data/02_06_split_group_aware.py \\
      --config configs/data/split.yaml \\
      --encoded data/interim/encoded/encoded.csv \\
      --code-carrier data/interim/code_carrier/code_carrier.csv \\
      --benign-dir data/interim/benign_G1_G2_G3/ \\
      --out data/interim/split_checks/ \\
      [--smoke N]           # 只取前 N 个攻击单位（良性侧按单位比例同缩放），产物落 out/smoke/
      [--stratify-length]   # 启用长度分层采样（门禁 6 的回退动作）

退出码：``0`` 成功；非 ``0`` 失败（含自检 `G1`–`G11` 任一不通过）。

**三条红线（`windows/W3.6_split.md` §0／§7）**

1. **划分单位只能是 group** —— 禁止按行、按样本、按编码划分。
2. **两体制共用同一 group 划分**（`V-2`）—— 同一 group 在两体制中归属同一分片；差别只在**行集**。
3. **留出 4 编码不得以任何增强形式混入体制二训练分片**（含 `val`）—— `AGENTS.md` §6-3。

**本窗口零推理**：不训练、不微调、不加载**模型**；唯一加载的是 `_common.token_length` 惰性取的
**tokenizer**（不是模型），且仅在需要长度统计时。

**本窗口只写 `data/interim/`**：`data/frozen/` 与 `data/sealed_test/` 由 `W3.8` 独占写入。

**自检 ≠ 门禁**：本脚本的 `G1`–`G11` 是**自检级**轻量检查（防止把明显错的划分交给 `W3.7`），
正式门禁判定权归 `02_07`（`A1`–`A14`）。

**确定性**：全流程只由 `split.yaml` 的 `split_seed` 一个随机源驱动；所有「按序取/按序削」都建立在
同一个打散序之上，**不再消耗第二个随机源**（禁时间戳、`uuid4`、`set` 遍历序）。
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime

_T0 = time.perf_counter()

# ---------------------------------------------------------------------------
# 载入共享库 `scripts/data/_common.py`（唯一出处；本窗口不改其任何签名）
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
ENCODING_CODES = tuple(common.ENCODING_CODES)
OOD_TRAIN = tuple(common.OOD_TRAIN_ENCODINGS)
OOD_HELDOUT = tuple(common.OOD_HELDOUT_ENCODINGS)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAIL = 3

SPLITS = ("train", "val", "test")
REGIMES = ("regime_id", "regime_ood")

#: 分片宽表字段（`windows/W3.6_split.md` §2.3.1，**顺序即契约**）。
WIDE_FIELDS = [
    "sample_id", "seed_group_id", "benign_group_id", "sample_kind", "family_id",
    "rewrite_id", "seed_id", "source", "harm_category", "benign_set",
    "encoding_type", "carrier_type", "carrier_subtype", "host_context",
    "plaintext", "payload", "text", "difficulty", "label",
]

#: 权威分配表字段（§2.3）。
GROUP_FIELDS = ["group_id", "group_kind", "family_id", "split", "member_count"]

#: `clean_original_test.csv` 字段（§2.3）。
CLEAN_FIELDS = ["sample_id", "seed_group_id", "family_id", "rewrite_id", "seed_id",
                "source", "harm_category", "text", "label"]

#: `g3_hard_negative_test.csv` 字段（**同 `G3.csv`**）。
G3_FIELDS = ["sample_id", "benign_group_id", "benign_set", "source", "source_item_id",
             "encoding_type", "carrier_type", "host_context", "g3_business_form",
             "business_necessity_evidence", "text", "label"]

#: 削减台账字段（§3；`reason` 区分「正负配平」与「长度分层」两类削减）。
DROPPED_FIELDS = ["regime", "split", "stage", "reason", "benign_group_id", "benign_set",
                  "member_count", "mean_tokens", "bucket", "attack_rows", "floor_rows"]

#: `clean_original_test.csv` 的 `sample_id` 前缀（本窗口定值）。
CLEAN_ID_PREFIX = "CLN-"

DEFAULT_TOKENIZER_ID = common.DEFAULT_TOKENIZER_ID


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


def _split_counts(n, cfg):
    """`A13` 的**唯一**取整口径：`round(N×0.8) : round(N×0.1) : N − 前两者`。

    `round` = Python 内置（银行家舍入，round-half-to-even），**不是四舍五入**。
    """
    n_train = int(round(n * float(cfg["train_share"])))
    n_val = int(round(n * float(cfg["val_share"])))
    return n_train, n_val, max(0, n - n_train - n_val)


_LEN_CACHE = {}


def tlen(text, tokenizer_id):
    key = (tokenizer_id, text)
    cached = _LEN_CACHE.get(key)
    if cached is None:
        cached = int(token_length(text, tokenizer_id))
        _LEN_CACHE[key] = cached
    return cached


def _abs_diff_ratio(pos_lengths, neg_lengths):
    pm, nm = _mean(pos_lengths), _mean(neg_lengths)
    denom = max(pm, nm)
    if denom <= 0:
        return 0.0
    return abs(pm - nm) / denom


# ===========================================================================
# 1. 读上游
# ===========================================================================
def load_families(seeds_path, meta_path):
    """`seed_group_id → family_id`（§2.3.2）、`seed_group_id → source` 与族结构（`G9` 的依据）。

    `source_of_group` 供 `r2` 的**按来源分层分配**（`assign_source_stratified`）使用。
    """
    seeds = read_csv(seeds_path)
    meta = {}
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    by_source = collections.defaultdict(list)
    for row in seeds:
        by_source[row.get("source", "")].append(row)

    fam_of_group = {}

    ah = by_source.get("AgentHarm", [])
    ah_keys = sorted({str(r.get("source_item_id", "")).split("-")[0] for r in ah},
                     key=lambda s: (int(s) if s.isdigit() else 1 << 30, s))
    for idx, key in enumerate(ah_keys, start=1):
        fam = "FAM-AH-%04d" % idx
        for r in ah:
            if str(r.get("source_item_id", "")).split("-")[0] == key:
                fam_of_group[r["seed_group_id"]] = fam

    pku = sorted(by_source.get("PKU-SafeRLHF", []), key=lambda r: r["seed_group_id"])
    for idx, r in enumerate(pku, start=1):
        fam_of_group[r["seed_group_id"]] = "FAM-PKU-%04d" % idx

    structure = {
        "agentharm_families": len(ah_keys),
        "agentharm_groups": len(ah),
        "pku_families": len(pku),
        "pku_groups": len(pku),
        "families_total": len(ah_keys) + len(pku),
        "meta": meta,
    }
    source_of_group = {r["seed_group_id"]: r.get("source", "") for r in seeds}
    return fam_of_group, structure, seeds, source_of_group


def load_attack_rows(encoded_path, code_carrier_path, fam_of_group):
    """合并两载体的攻击行（`text` 载体：送入检测器的就是 `payload` 本体）。"""
    rows = []
    for r in read_csv(encoded_path):
        rows.append({
            "sample_id": r["sample_id"], "seed_group_id": r["seed_group_id"],
            "benign_group_id": "", "sample_kind": "attack",
            "family_id": fam_of_group.get(r["seed_group_id"], ""),
            "rewrite_id": r["rewrite_id"], "seed_id": r["seed_id"],
            "source": r["source"], "harm_category": r["harm_category"], "benign_set": "",
            "encoding_type": r["encoding_type"], "carrier_type": "text",
            "carrier_subtype": "", "host_context": "",
            "plaintext": r["plaintext"], "payload": r["payload"],
            "text": r["payload"], "difficulty": r["difficulty"], "label": 1,
        })
    for r in read_csv(code_carrier_path):
        rows.append({
            "sample_id": r["sample_id"], "seed_group_id": r["seed_group_id"],
            "benign_group_id": "", "sample_kind": "attack",
            "family_id": fam_of_group.get(r["seed_group_id"], ""),
            "rewrite_id": r["rewrite_id"], "seed_id": r["seed_id"],
            "source": r["source"], "harm_category": r["harm_category"], "benign_set": "",
            "encoding_type": r["encoding_type"], "carrier_type": "code",
            "carrier_subtype": r.get("carrier_subtype", ""), "host_context": "",
            "plaintext": r["plaintext"], "payload": r["payload"], "text": r["text"],
            "difficulty": r["difficulty"], "label": 1,
        })
    return rows


def load_benign_rows(benign_dir, cfg):
    """读 G1/G2/G3（良性侧没有 `plaintext`／`payload`／`difficulty`）。"""
    rows = []
    for fname in cfg["expected"]["benign_files"]:
        path = os.path.join(benign_dir, fname)
        if not os.path.isfile(path):
            raise SystemExit("缺少良性集文件：%s" % _rel(path))
        for r in read_csv(path):
            rows.append({
                "sample_id": r["sample_id"], "seed_group_id": "",
                "benign_group_id": r["benign_group_id"], "sample_kind": "benign",
                "family_id": r["benign_group_id"],       # 良性侧 family_id = benign_group_id
                "rewrite_id": "", "seed_id": "",
                "source": r.get("source", ""), "harm_category": "",
                "benign_set": r.get("benign_set", ""),
                "encoding_type": r.get("encoding_type", ""),
                "carrier_type": r.get("carrier_type", ""), "carrier_subtype": "",
                "host_context": r.get("host_context", ""),
                "plaintext": "", "payload": "", "text": r["text"], "difficulty": "",
                "label": 0, "_source_item_id": r.get("source_item_id", ""),
            })
    return rows


# ===========================================================================
# 2. 划分单位
# ===========================================================================
def attack_units(seed_groups, fam_of_group, family_atomic, source_of_group=None):
    """攻击侧单位：`family_atomic` 假 ⇒ 一 group 一单位；真 ⇒ 一族一单位。

    第三个返回值 `source_of_unit`（单位 → 来源）供 `assign_source_stratified` 使用；
    族内全部 group 同源（AH 族由同一 `id_original` 的 4 变体组成、PKU 族自成一族）⇒ 取任一成员即可。
    """
    units = collections.OrderedDict()
    group_of = {}
    source_of_group = source_of_group or {}

    def _src(gs):
        return source_of_group.get(gs[0], "") if gs else ""

    if not family_atomic:
        for g in sorted(seed_groups):
            key = "GRP:%s" % g
            units[key] = [g]
            group_of[g] = key
        return units, group_of, collections.OrderedDict(
            (k, _src(v)) for k, v in units.items())
    fam2groups = collections.defaultdict(list)
    for g in sorted(seed_groups):
        fam2groups[fam_of_group.get(g, "")].append(g)
    for fam in sorted(fam2groups):
        key = "FAM:%s" % fam
        units[key] = sorted(fam2groups[fam])
        for g in units[key]:
            group_of[g] = key
    return units, group_of, collections.OrderedDict(
        (k, _src(v)) for k, v in units.items())


def benign_units(benign_rows, cfg):
    """良性侧单位：真值合成侧上提为「场景」（§2.7 断言 `G10`），其余一 group 一单位。"""
    scene_source = cfg.get("scene_source", "G3-synthetic")
    use_scene = bool(cfg.get("scene_unit", True))

    group2scene = {}
    if use_scene:
        for r in benign_rows:
            if r.get("source") != scene_source:
                continue
            scene = str(r.get("_source_item_id", "")).split(":")[0].strip()
            if not scene:
                continue
            prev = group2scene.get(r["benign_group_id"])
            if prev is not None and prev != scene:
                raise SystemExit(
                    "合成侧 group 跨场景（%s: %s vs %s）⇒ 上游分组异常，须回退 2.5"
                    % (r["benign_group_id"], prev, scene))
            group2scene[r["benign_group_id"]] = scene

    all_groups = sorted({r["benign_group_id"] for r in benign_rows})
    units = collections.OrderedDict()
    group_of = {}
    scene2groups = collections.defaultdict(list)
    for g in sorted(group2scene):
        scene2groups[group2scene[g]].append(g)
    for scene in sorted(scene2groups):
        key = "SCN:%s" % scene
        units[key] = sorted(scene2groups[scene])
        for g in units[key]:
            group_of[g] = key
    for g in all_groups:
        if g not in group2scene:
            key = "GRP:%s" % g
            units[key] = [g]
            group_of[g] = key
    return units, group_of, group2scene


# ===========================================================================
# 3. 分配（打散序 → 三档）
# ===========================================================================
def shuffled_order(keys, rng):
    order = sorted(keys)
    rng.shuffle(order)
    return order


def assign_in_order(units, order, target):
    """按 `order` 逐个投档，使**组数**尽量等于 `target`（`target` 的单位是**组**）。

    规则：优先投给「剩余容量 ≥ 本单位组数」且剩余最大的一档（并列取 train→val→test）；
    若无档可容纳则投给剩余最大的一档（溢出）。单位组数为 1 时**等价于**按计数切片。

    **攻击侧**用本函数（`A13` 约束的是 `seed_group_id` 的**组数**比，故目标必须以组计）。
    """
    remaining = [int(target[0]), int(target[1]), int(target[2])]
    assign = {}
    for key in order:
        size = len(units[key])
        best = None
        for i in range(3):
            if remaining[i] >= size and (best is None or remaining[i] > remaining[best]):
                best = i
        if best is None:
            best = max(range(3), key=lambda i: (remaining[i], -i))
        assign[key] = SPLITS[best]
        remaining[best] -= size
    return assign, remaining


def _counts_of(assign, units):
    c = [0, 0, 0]
    for k, sp in assign.items():
        c[SPLITS.index(sp)] += len(units[k])
    return c


def assign_source_stratified(units, source_of_unit, order, target):
    """**按来源分层**分配攻击单位，再以**最小移动**修复到目标组数（`r2` 新增）。

    为何需要（`r2` 的实测发现）：若只用「全局打散序 + 组数贪心」（`assign_in_order` 的 LPT 序），
    组数大的单位（AH 族恒 4 组、PKU 族恒 1 组）会**整体被推向容量最大的 `train`**
    （实测 40/44 个 AH 族进 train）⇒ **来源构成失真**：train 的 PKU 占比 **9.1%**（语料 20%），
    而 `val`／`test` 升到 **63.6%** ⇒ ID 体制的 test 在**来源轴**上与 train 不同分布。

    规则（三步，全程确定性，不动 `order`）：

    1. **分层**：层键 = 单位所属**来源**（`AgentHarm`／`PKU-SafeRLHF`）；
    2. **层内切分**：按 `order`（全局打散序）取该层的单位，用 `A13` 派生式
       以**该层单位数**切 `8:1:1`（`round(n×0.8) : round(n×0.1) : n−前两者`）；
    3. **修复**：反复「自**超出目标最多**的一档移出一个单位 → **亏空最多**的一档」，
       取该档内「组数 ≤ 当前超出量且最大」的单位（并列取组数小者、键小者；无可用则取最小者），
       直至三档**组数逐格等于 `target`**；修复有界（≤ 单位数 + 10 步），仍不达标即报错。

    返回 `(assign, counts)`；`counts` 为单位组数。
    """
    strata = collections.defaultdict(list)
    for k in order:
        strata[source_of_unit.get(k, "")].append(k)

    assign = {}
    for src in sorted(strata):
        keys = strata[src]
        t, v, _te = _split_counts(len(keys), {"train_share": 0.8, "val_share": 0.1})
        for i, k in enumerate(keys):
            assign[k] = SPLITS[0] if i < t else (SPLITS[1] if i < t + v else SPLITS[2])

    counts = _counts_of(assign, units)
    for _ in range(len(units) + 10):
        diff = [counts[i] - int(target[i]) for i in range(3)]
        if all(d == 0 for d in diff):
            break
        si = max(range(3), key=lambda i: (diff[i], -i))          # 超出最多
        ti = min(range(3), key=lambda i: (diff[i], i))           # 亏空最多
        if diff[si] <= 0 or diff[ti] >= 0:
            break
        cands = sorted(k for k in assign if assign[k] == SPLITS[si])
        pick = None
        for k in cands:                                          # 组数 ≤ 超出量且最大
            sz = len(units[k])
            if sz <= diff[si] and (pick is None or sz > len(units[pick])):
                pick = k
        if pick is None:
            pick = min(cands, key=lambda k: (len(units[k]), k))
        assign[pick] = SPLITS[ti]
        counts = _counts_of(assign, units)
    return assign, counts


def slice_units(units, order):
    """按**单位数**切片 8:1:1（`§2.2` 的字面口径：打散 → 切分）。

    **良性侧**用本函数：良性没有「组数比例」的断言，且单位组数悬殊（场景 12–35 组 vs 单组），
    按组数配位会把场景单位整体推向 `train`；按单位数切片使场景在各档间**按比例**落位。

    **注**（`r1`·`W3` 验收建议 `S-1`）：本函数**刻意**把 `train_share=0.8`／`val_share=0.1` 写成字面量、
    **不经 `cfg`** —— 良性侧的切片比例与 `V-3`（攻击侧组数比例）是**两件事**，用 `cfg` 会让人误以为
    改 `V-3` 即改良性切片。当前两处取值相等；**改 `V-3` 前须同时复核本处字面量**。
    """
    n = len(order)
    n_train, n_val, n_test = _split_counts(n, {"train_share": 0.8, "val_share": 0.1})
    assign = {}
    for i, key in enumerate(order):
        if i < n_train:
            assign[key] = "train"
        elif i < n_train + n_val:
            assign[key] = "val"
        else:
            assign[key] = "test"
    return assign, (n_train, n_val, n_test)


def groups_to_split(units, group_of, assign):
    out = {}
    for key, groups in units.items():
        for g in groups:
            out[g] = assign[key]
    return out


# ===========================================================================
# 4. 正负配平（**只减不增、不跨 group**）
# ===========================================================================
def _rows_by_group(rows):
    buckets = collections.OrderedDict()
    for r in rows:
        buckets.setdefault(r["benign_group_id"], []).append(r)
    return buckets


def balance_plain(rows, attack_rows, drop_rank, band_low):
    """默认配平：按打散序整组削减，至行数 ≤ 攻击行数且不跌破 `band_low × 攻击行数`。"""
    by_group = _rows_by_group(rows)
    order = sorted(by_group, key=lambda g: (drop_rank.get(g, 1 << 30), g))
    total = sum(len(v) for v in by_group.values())
    floor_rows = int(math.ceil(band_low * attack_rows))
    kept = collections.OrderedDict(by_group)
    dropped = []
    for g in order:
        if total <= attack_rows:
            break
        size = len(kept[g])
        if total - size < floor_rows:
            continue
        total -= size
        dropped.append(g)
        del kept[g]
    return kept, dropped, {"floor_rows": floor_rows, "bucket_of": {}, "group_len": {}}


def stratify_sequence(rows, attack_mean, drop_rank, floor_rows, buckets, tokenizer_id):
    """长度分层选样：产出一条**组级剔除序列**（自「与攻击侧均值最不匹配」的组起）。

    记 `n` 为剩余行数、`T` 为剩余长度和、`A` 为攻击侧均值。剔除组 `g`（`s` 行、组均值 `L`）后
    剩余均值为 `(T − sL)/(n − s)`，令其等于 `A` 等价于

        ``D := T − A·n`` 与  ``u_g := s·(L − A)``  **相等**。

    故每一步剔除使 `|u_g − D|` 最小的组；剔除后 `D ← D − u_g`。首步之后 `D ≈ 0`，此后
    持续剔除 `L ≈ A` 的组 ⇒ **均值保持在 `A` 附近**，同时**行数单调下降**（既配平又对齐长度）。

    这是**分层抽样**的非比例分配（strata = 组均值长度的等频分桶，分配率按「对齐参考均值」定），
    而非按桶比例抽样（按比例抽不改变该侧均值，无法使两均值靠拢）。

    **只减不增、不跨 group、不改变 group 的分片归属。** 返回剔除序列与快照，
    使**各 count 窗口**可在同一序列上取前缀（不必重算）。
    """
    by_group = _rows_by_group(rows)
    group_n = {g: len(rs) for g, rs in by_group.items()}
    group_len = {g: _mean([tlen(r["text"], tokenizer_id) for r in rs])
                 for g, rs in by_group.items()}
    group_u = {g: group_n[g] * (group_len[g] - attack_mean) for g in by_group}

    total = sum(group_n.values())
    run_sum = float(sum(group_len[g] * group_n[g] for g in by_group))
    pool_mean = (run_sum / total) if total else 0.0
    long_first = pool_mean > attack_mean

    order_by_len = sorted(by_group, key=lambda g: (group_len[g], g))
    nb = max(1, min(int(buckets), len(order_by_len) or 1))
    bucket_of = {}
    for b in range(nb):
        members = order_by_len[b * len(order_by_len) // nb:(b + 1) * len(order_by_len) // nb]
        for g in members:
            bucket_of[g] = b

    remaining = dict(group_u)
    d = run_sum - attack_mean * total
    seq = []
    while remaining:
        g = min(remaining, key=lambda k: (abs(remaining[k] - d), drop_rank.get(k, 1 << 30), k))
        size = group_n[g]
        if total - size < floor_rows:
            break
        total -= size
        run_sum -= group_len[g] * size
        d -= remaining.pop(g)
        bm = run_sum / total if total else 0.0
        denom = max(attack_mean, bm)
        seq.append({"group_id": g, "rows_after": total,
                    "mean_after": bm,
                    "abs_diff_ratio": (abs(attack_mean - bm) / denom) if denom > 0 else 0.0})
    return seq, {"floor_rows": floor_rows, "bucket_of": bucket_of, "group_len": group_len,
                 "attack_mean": attack_mean, "before_mean": pool_mean,
                 "long_first": long_first}


def pick_prefix(seq, ceil_rows, threshold):
    """在剔除序列上取前缀：优先**最早**满足「行数 ≤ `ceil_rows` 且 `abs_diff_ratio ≤ threshold`」者；
    若无，则取窗口内 `abs_diff_ratio` 最小者。"""
    ok = [i for i, s in enumerate(seq) if s["rows_after"] <= ceil_rows]
    if not ok:
        return -1
    for i in ok:
        if seq[i]["abs_diff_ratio"] <= threshold:
            return i
    return min(ok, key=lambda i: (seq[i]["abs_diff_ratio"], i))


def build_regime(regime, split, attack_by_group, benign_by_group, ood_keep_ids,
                 drop_rank, tokenizer_id, cfg, enable_stratify):
    """产出某体制某分片的行集（攻击 + 良性），并做正负配平。"""
    band_low = float(cfg["balance"]["band_low"])
    strat_cfg = cfg.get("stratify", {})

    attack = []
    for g in sorted(attack_by_group):
        for r in attack_by_group[g]:
            if _encoding_ok(regime, split, r["encoding_type"]):
                attack.append(r)

    pool = []
    for g in sorted(benign_by_group):
        for r in benign_by_group[g]:
            if ood_keep_ids is not None and r["sample_id"] not in ood_keep_ids:
                continue
            enc = r.get("encoding_type", "")
            if regime == "regime_ood" and enc != "" and not _encoding_ok(regime, split, enc):
                continue
            pool.append(r)

    attack_lengths = [tlen(r["text"], tokenizer_id) for r in attack]
    attack_mean = _mean(attack_lengths)

    kept, dropped, info = balance_plain(pool, len(attack), drop_rank, band_low)
    stage = "balance"
    stratified = False
    benign_lengths = [tlen(r["text"], tokenizer_id) for r in pool if r["benign_group_id"] in kept]
    plain_ratio = _abs_diff_ratio(attack_lengths, benign_lengths)

    thr = float(strat_cfg.get("threshold", 0.20))
    trigger = thr - float(strat_cfg.get("trigger_margin", 0.0))
    ladder = []
    if enable_stratify and plain_ratio > trigger:
        floor_rows = int(math.ceil(band_low * len(attack)))
        seq, info_s = stratify_sequence(pool, attack_mean, drop_rank, floor_rows,
                                        strat_cfg.get("buckets", 10), tokenizer_id)
        best = None
        for r in cfg["balance"].get("count_window_ratios", [0.90, 0.85, band_low]):
            r = float(r)
            if r <= 0:
                continue
            ceil_rows = max(floor_rows, int(math.floor(len(attack) / r)))
            cut = pick_prefix(seq, ceil_rows, thr)
            if cut < 0:
                ladder.append({"count_window_ratio": r, "ceil_rows": ceil_rows,
                               "benign_rows": None, "abs_diff_ratio": None,
                               "pos_neg_ratio": None, "passed_threshold": False})
                continue
            drop_groups = {s["group_id"] for s in seq[:cut + 1]}
            kept2 = collections.OrderedDict(
                (g, rs) for g, rs in _rows_by_group(pool).items() if g not in drop_groups)
            lengths2 = [tlen(x["text"], tokenizer_id) for x in pool
                        if x["benign_group_id"] in kept2]
            ratio2 = _abs_diff_ratio(attack_lengths, lengths2)
            entry = {"count_window_ratio": r, "ceil_rows": ceil_rows,
                     "benign_rows": len(lengths2), "abs_diff_ratio": ratio2,
                     "pos_neg_ratio": (len(attack) / float(len(lengths2))) if lengths2 else None,
                     "passed_threshold": bool(ratio2 <= thr)}
            ladder.append(entry)
            if ratio2 <= thr:
                best = (kept2, [s["group_id"] for s in seq[:cut + 1]], info_s, ratio2,
                        lengths2, r)
                break
        if best is not None and best[3] < plain_ratio - 1e-12:
            kept, dropped, info = best[0], best[1], best[2]
            benign_lengths = best[4]
            stage = "length_stratify"
            stratified = True
            info["count_window_ratio"] = best[5]
            info["ceil_rows"] = int(math.floor(len(attack) / best[5]))
            info["after_mean"] = _mean(benign_lengths)

    benign = [r for r in pool if r["benign_group_id"] in kept]
    dropped_rows = [r for r in pool if r["benign_group_id"] not in kept]
    n_pos = len([r for r in attack if r["label"] == 1])
    n_neg = len([r for r in benign if r["label"] == 0])
    return {
        "attack": attack, "benign": benign, "dropped": dropped_rows,
        "kept_groups": set(kept), "dropped_groups": list(dropped),
        "n_pos": n_pos, "n_neg": n_neg,
        "ratio": (n_pos / float(n_neg)) if n_neg else float("inf"),
        "attack_mean": attack_mean, "benign_mean": _mean(benign_lengths),
        "abs_diff_ratio": _abs_diff_ratio(attack_lengths, benign_lengths),
        "abs_diff_ratio_plain": plain_ratio,
        "stage": stage, "stratified": stratified,
        "bucket_of": info.get("bucket_of", {}), "group_len": info.get("group_len", {}),
        "floor_rows": info.get("floor_rows"), "ceil_rows": info.get("ceil_rows"),
        "count_window_ratio": info.get("count_window_ratio"), "ladder": ladder,
    }


def _encoding_ok(regime, split, encoding_type):
    """体制二的行级编码过滤（口径见 `split.yaml`；留出 4 编码不得入 train／val）。"""
    if regime == "regime_id":
        return True
    if split == "test":
        return encoding_type in OOD_HELDOUT
    return encoding_type in OOD_TRAIN


# ===========================================================================
# 5. 自检 G1–G11（自检级，非门禁）
# ===========================================================================
def selfcheck(shards, group_rows, structure, cfg, smoke, derived, scene_of_group,
              tokenizer_id, family_atomic):
    res = []

    def add(aid, ok, measured, detail=""):
        res.append((aid, bool(ok), str(measured), detail))

    lo = float(cfg["balance"]["band_low"])
    hi = float(cfg["balance"]["band_high"])

    # G1 六分片均非空；train ≫ val+test
    ok1, det1 = True, []
    for reg in REGIMES:
        for sp in SPLITS:
            n = len(shards[reg][sp]["attack"]) + len(shards[reg][sp]["benign"])
            det1.append("%s/%s=%d" % (reg, sp, n))
            if n <= 0:
                ok1 = False
    tr = sum(len(shards[r]["train"]["attack"]) + len(shards[r]["train"]["benign"]) for r in REGIMES)
    vt = sum(len(shards[r][s]["attack"]) + len(shards[r][s]["benign"])
             for r in REGIMES for s in ("val", "test"))
    if tr < 2 * vt:
        ok1 = False
        det1.append("train=%d < 2*(val+test)=%d" % (tr, 2 * vt))
    add("G1", ok1, ";".join(det1))

    # G2 group_id 全局唯一；每 group 恰一个 split
    ids = [r["group_id"] for r in group_rows]
    dup = [g for g, c in collections.Counter(ids).items() if c > 1]
    bad = [r["group_id"] for r in group_rows if r["split"] not in SPLITS]
    add("G2", not dup and not bad,
        "rows=%d distinct=%d dup=%d bad_split=%d" % (len(ids), len(set(ids)), len(dup), len(bad)),
        ",".join(dup[:5] + bad[:5]))

    # G3 攻击侧 test 的 group 集与 train/val 无交集
    inter = set()
    for reg in REGIMES:
        te = {r["seed_group_id"] for r in shards[reg]["test"]["attack"]}
        oth = {r["seed_group_id"] for r in shards[reg]["train"]["attack"]} | \
              {r["seed_group_id"] for r in shards[reg]["val"]["attack"]}
        inter |= (te & oth)
    add("G3", not inter, "|test ∩ (train∪val)| = %d" % len(inter), ",".join(sorted(inter)[:5]))

    # G4 体制二 train/val 中留出 4 编码命中数 == 0
    hits = []
    for sp in ("train", "val"):
        for r in shards["regime_ood"][sp]["attack"] + shards["regime_ood"][sp]["benign"]:
            if r.get("encoding_type") in OOD_HELDOUT:
                hits.append(r["sample_id"])
    add("G4", not hits, "heldout hits in regime_ood train+val = %d" % len(hits),
        ",".join(hits[:5]))

    # G5 每分片 label 正负比 ∈ [lo, hi]
    ok5, det5, bad5 = True, [], []
    for reg in REGIMES:
        for sp in SPLITS:
            sh = shards[reg][sp]
            ratio = sh["n_pos"] / float(sh["n_neg"]) if sh["n_neg"] else float("inf")
            det5.append("%s/%s=%.4f" % (reg, sp, ratio))
            if not (lo <= ratio <= hi):
                ok5 = False
                bad5.append("%s/%s" % (reg, sp))
    add("G5", ok5, "band=[%.2f,%.2f] %s" % (lo, hi, ";".join(det5)), ",".join(bad5))

    # G6 同一 sample_id 在两体制中的 text 逐字符相同
    first, bad6, seen = {}, [], collections.defaultdict(set)
    for reg in REGIMES:
        for sp in SPLITS:
            for r in shards[reg][sp]["attack"] + shards[reg][sp]["benign"]:
                sid = r["sample_id"]
                seen[sid].add(reg)
                if sid in first:
                    if first[sid] != r["text"]:
                        bad6.append(sid)
                else:
                    first[sid] = r["text"]
    shared = sum(1 for v in seen.values() if len(v) == 2)
    add("G6", not bad6, "shared sample_id=%d, text mismatch=%d" % (shared, len(bad6)),
        ",".join(bad6[:5]))

    # G7 clean_original_test 的 text 与任一 train/val 行不相等（集合层面）
    clean_texts = {r["text"] for r in derived["clean"]}
    tv = set()
    for reg in REGIMES:
        for sp in ("train", "val"):
            for r in shards[reg][sp]["attack"] + shards[reg][sp]["benign"]:
                tv.add(r["text"])
    hit7 = sorted(clean_texts & tv)
    add("G7", not hit7, "clean=%d, overlap with train∪val=%d" % (len(clean_texts), len(hit7)),
        ",".join(hit7[:3]))

    # G8 g3_hard_negative_test 的 benign_group_id ⊆ 测试良性 group 集
    test_bg = set()
    for reg in REGIMES:
        test_bg |= {r["benign_group_id"] for r in shards[reg]["test"]["benign"]}
    g3g = {r["benign_group_id"] for r in derived["g3"]}
    out8 = sorted(g3g - test_bg)
    add("G8", not out8, "g3 groups=%d, outside test benign groups=%d" % (len(g3g), len(out8)),
        ",".join(out8[:5]))

    # G9 family_id 赋值正确（族数由 seeds.meta.json **派生**，禁硬编码 44/4）
    if smoke:
        add("G9", True, "n/a（--smoke 档只含部分 group，族结构不完整）")
    else:
        st = (structure.get("meta", {}) or {}).get("agentharm_variant_structure", {}) or {}
        n_orig = st.get("n_original_entries")
        variants = st.get("variants_per_original") or []
        per_fam = collections.defaultdict(list)
        for r in group_rows:
            if r["group_kind"] == "attack":
                per_fam[r["family_id"]].append(r["group_id"])
        ah = {f: g for f, g in per_fam.items() if f.startswith("FAM-AH")}
        pku = {f: g for f, g in per_fam.items() if f.startswith("FAM-PKU")}
        ok9 = (
            n_orig is not None
            and len(ah) == int(n_orig)
            and (not variants or all(len(g) == int(variants[0]) for g in ah.values()))
            and len(pku) == int(structure.get("pku_families", -1))
            and all(len(g) == 1 for g in pku.values())
            and len(ah) + len(pku) == int(structure.get("families_total", -1))
        )
        add("G9", ok9,
            "AH fams=%d（每族 group=%s）PKU fams=%d（每族 1）total=%d；meta n_original=%s variants=%s"
            % (len(ah), sorted({len(v) for v in ah.values()}), len(pku), len(ah) + len(pku),
               n_orig, variants))

    # G10 合成侧同场景必同分片（`--smoke` 不适用）
    if smoke:
        add("G10", True, "n/a（--smoke 档不适用）")
    else:
        split_of = {r["group_id"]: r["split"] for r in group_rows}
        scene_splits = collections.defaultdict(set)
        for g, scene in scene_of_group.items():
            sp = split_of.get(g)
            if sp:
                scene_splits[scene].add(sp)
        strad = sorted(s for s, v in scene_splits.items() if len(v) > 1)
        add("G10", not strad, "scenes=%d, straddling=%d" % (len(scene_splits), len(strad)),
            ";".join(strad[:5]))

    # G11 同一 text 不得跨分片（全集合）
    bad11 = []
    for reg in REGIMES:
        sets = {sp: {r["text"] for r in shards[reg][sp]["attack"] + shards[reg][sp]["benign"]}
                for sp in SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = sets[a] & sets[b]
            if ov:
                bad11.append("%s:%s∩%s=%d" % (reg, a, b, len(ov)))
    add("G11", not bad11, ";".join(bad11) if bad11 else "no cross-split text overlap",
        ",".join(bad11[:5]))

    return res


# ===========================================================================
# 6. 主流程
# ===========================================================================
def _fail(msg):
    print("FATAL: %s" % msg)
    return None


def run(args, cfg):
    tokenizer_id = DEFAULT_TOKENIZER_ID
    smoke = int(args.smoke or 0)
    out_dir = args.out
    if smoke:
        out_dir = os.path.join(out_dir, cfg["output"]["smoke_dir"])
    splits_root = os.path.join(out_dir, cfg["output"]["splits_root"])

    seeds_path = args.seeds or cfg["inputs"]["seeds"]
    meta_path = args.seeds_meta or cfg["inputs"]["seeds_meta"]
    rewrites_path = args.rewrites or cfg["inputs"]["rewrites"]

    fam_of_group, structure, seeds, source_of_group = load_families(seeds_path, meta_path)
    attack_rows = load_attack_rows(args.encoded, args.code_carrier, fam_of_group)
    benign_rows = load_benign_rows(args.benign_dir, cfg)

    exp = cfg["expected"]
    if len(attack_rows) != int(exp["attack_rows"]):
        return _fail("攻击样本行数 %d ≠ config 期望 %d（上游漂移，先查 W3.3／W3.4）"
                     % (len(attack_rows), int(exp["attack_rows"])))
    n_text = len([r for r in attack_rows if r["carrier_type"] == "text"])
    n_code = len([r for r in attack_rows if r["carrier_type"] == "code"])
    if n_text != int(exp["text_carrier_rows"]) or n_code != int(exp["code_carrier_rows"]):
        return _fail("载体计数不符：text=%d code=%d" % (n_text, n_code))
    if len({r["seed_group_id"] for r in attack_rows}) != int(exp["seed_groups_total"]):
        return _fail("seed_group_id 数 ≠ config 期望 %d" % int(exp["seed_groups_total"]))

    seed_groups_all = sorted({r["seed_group_id"] for r in attack_rows})
    benign_groups_all = sorted({r["benign_group_id"] for r in benign_rows})

    family_atomic = bool(cfg.get("family_atomic", False))
    a_units, a_group_of, a_source_of = attack_units(seed_groups_all, fam_of_group,
                                                    family_atomic, source_of_group)
    b_units, b_group_of, group2scene = benign_units(benign_rows, cfg)

    rng = random.Random(int(cfg["split_seed"]))
    a_order = shuffled_order(a_units, rng)
    b_order = shuffled_order(b_units, rng)

    smoke_meta = None
    if smoke:
        scale = len(b_units) / float(max(1, len(a_units)))
        n_b_take = int(math.ceil(smoke * scale))
        a_take = a_order[:smoke]
        b_take = b_order[:n_b_take]
        a_units = collections.OrderedDict((k, a_units[k]) for k in a_take)
        b_units = collections.OrderedDict((k, b_units[k]) for k in b_take)
        a_order, b_order = list(a_take), list(b_take)
        smoke_meta = {"n_attack_units": len(a_units), "n_benign_units": len(b_units),
                      "scale": scale}

    # 攻击侧分配器（`split.yaml` 的 `attack_allocation`）：
    #   `family_atomic: false`（单位组数全为 1）⇒ `slice`：与 `b09771c` 的既有行为**逐字节一致**
    #     （保证 r1 已验收产物的可复现性）；
    #   `family_atomic: true`（单位组数不等）⇒ `source_stratified`：按来源分层，避免 LPT 把
    #     组数大的族整体推向 `train` 而**扭曲来源构成**（`r2` 实测发现，见函数 docstring）。
    alloc_cfg = cfg.get("attack_allocation", {})
    allocator = alloc_cfg.get("when_family_atomic_true" if family_atomic
                              else "when_family_atomic_false", "slice")
    if allocator != "source_stratified" and family_atomic:
        # LPT 序（**仅 `slice` 分配器用**；`source_stratified` 须保留全局打散序在层内的相对次序）
        a_order = sorted(a_order, key=lambda k: -len(a_units[k]))
    target_a = _split_counts(sum(len(v) for v in a_units.values()), cfg)
    if allocator == "source_stratified" and family_atomic:
        a_assign, a_counts = assign_source_stratified(a_units, a_source_of, a_order, target_a)
        a_rem = tuple(int(target_a[i]) - a_counts[i] for i in range(3))
    else:
        a_assign, a_rem = assign_in_order(a_units, a_order, target_a)
    b_assign, b_rem = slice_units(b_units, b_order)
    target_b = b_rem

    attack_split = groups_to_split(a_units, a_group_of, a_assign)
    benign_split = groups_to_split(b_units, b_group_of, b_assign)
    b_rank = {k: i for i, k in enumerate(b_order)}
    drop_rank = {g: b_rank.get(b_group_of[g], 1 << 30) for g in benign_groups_all}

    # `--smoke` 下 `units` 已缩到前 N 个，故只纳入被选中的 group（否则 KeyError）
    attack_by_split = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in attack_rows:
        sp = attack_split.get(r["seed_group_id"])
        if sp:
            attack_by_split[sp][r["seed_group_id"]].append(r)
    benign_by_split = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in benign_rows:
        sp = benign_split.get(r["benign_group_id"])
        if sp:
            benign_by_split[sp][r["benign_group_id"]].append(r)

    shards = {reg: {} for reg in REGIMES}
    for sp in SPLITS:
        id_out = build_regime("regime_id", sp, attack_by_split[sp], benign_by_split[sp],
                              None, drop_rank, tokenizer_id, cfg, bool(args.stratify_length))
        ood_keep = {r["sample_id"] for r in id_out["benign"]}
        ood_out = build_regime("regime_ood", sp, attack_by_split[sp], benign_by_split[sp],
                               ood_keep, drop_rank, tokenizer_id, cfg,
                               bool(args.stratify_length))
        shards["regime_id"][sp] = id_out
        shards["regime_ood"][sp] = ood_out

    # --- 派生视图 ---
    rewrites = read_csv(rewrites_path)
    test_groups = {r["seed_group_id"] for r in shards["regime_id"]["test"]["attack"]}
    seed_meta = {r["seed_group_id"]: r for r in seeds}
    clean_rows = []
    for rw in sorted(rewrites, key=lambda x: x["rewrite_id"]):
        if rw["seed_group_id"] not in test_groups:
            continue
        src = seed_meta.get(rw["seed_group_id"], {})
        clean_rows.append({
            "sample_id": CLEAN_ID_PREFIX + rw["rewrite_id"].replace("-", ""),
            "seed_group_id": rw["seed_group_id"],
            "family_id": fam_of_group.get(rw["seed_group_id"], ""),
            "rewrite_id": rw["rewrite_id"], "seed_id": rw["seed_id"],
            "source": src.get("source", ""), "harm_category": src.get("harm_category", ""),
            "text": rw["text"], "label": 1,
        })

    g3_src = {}
    for fname in cfg["expected"]["benign_files"]:
        for r in read_csv(os.path.join(args.benign_dir, fname)):
            if r.get("benign_set") == "G3":
                g3_src[r["sample_id"]] = r
    g3_rows = []
    for r in benign_rows:
        if r.get("benign_set") != "G3":
            continue
        if r["benign_group_id"] not in shards["regime_id"]["test"]["kept_groups"]:
            continue
        src = g3_src.get(r["sample_id"], {})
        g3_rows.append({
            "sample_id": r["sample_id"], "benign_group_id": r["benign_group_id"],
            "benign_set": r["benign_set"], "source": r["source"],
            "source_item_id": r.get("_source_item_id", ""),
            "encoding_type": r["encoding_type"], "carrier_type": r["carrier_type"],
            "host_context": r["host_context"],
            "g3_business_form": src.get("g3_business_form", ""),
            "business_necessity_evidence": src.get("business_necessity_evidence", ""),
            "text": r["text"], "label": 0,
        })
    g3_rows.sort(key=lambda r: r["sample_id"])

    # --- 权威分配表 ---
    att_cnt = collections.Counter(r["seed_group_id"] for r in attack_rows)
    ben_cnt = collections.Counter(r["benign_group_id"] for r in benign_rows)
    group_rows = []
    for g in seed_groups_all:
        if g not in attack_split:       # `--smoke`：只登记被选中的 group
            continue
        group_rows.append({"group_id": g, "group_kind": "attack",
                           "family_id": fam_of_group.get(g, ""), "split": attack_split[g],
                           "member_count": att_cnt[g]})
    for g in benign_groups_all:
        if g not in benign_split:
            continue
        group_rows.append({"group_id": g, "group_kind": "benign", "family_id": g,
                           "split": benign_split[g], "member_count": ben_cnt[g]})
    group_rows.sort(key=lambda r: (r["group_kind"], r["group_id"]))

    # --- 削减台账 ---
    dropped_rows = []
    for reg in REGIMES:
        for sp in SPLITS:
            sh = shards[reg][sp]
            drop_cnt = collections.Counter(r["benign_group_id"] for r in sh["dropped"])
            for g in sh["dropped_groups"]:
                dropped_rows.append({
                    "regime": reg, "split": sp,
                    "stage": "stratified" if sh["stratified"] else "plain",
                    "reason": sh["stage"], "benign_group_id": g,
                    "benign_set": (benign_by_split[sp][g][0]["benign_set"] if g in benign_by_split[sp] else ""),
                    "member_count": drop_cnt[g],
                    "mean_tokens": "%.4f" % sh["group_len"].get(g, 0.0),
                    "bucket": sh["bucket_of"].get(g, ""),
                    "attack_rows": len(sh["attack"]), "floor_rows": sh["floor_rows"],
                })
    dropped_rows.sort(key=lambda r: (r["regime"], r["split"], r["benign_group_id"]))

    # --- 自检 ---
    derived = {"clean": clean_rows, "g3": g3_rows}
    checks = selfcheck(shards, group_rows, structure, cfg, smoke, derived, group2scene,
                       tokenizer_id, family_atomic)

    # --- 写盘 ---
    os.makedirs(splits_root, exist_ok=True)
    written = []
    for reg in REGIMES:
        for sp in SPLITS:
            sh = shards[reg][sp]
            path = os.path.join(splits_root, reg, "%s.csv" % sp)
            write_csv(path, sh["attack"] + sh["benign"], WIDE_FIELDS)
            written.append(path)
    for name, rows, fields in (
            (cfg["output"]["seed_groups"], group_rows, GROUP_FIELDS),
            (cfg["output"]["clean_original_test"], clean_rows, CLEAN_FIELDS),
            (cfg["output"]["g3_hard_negative_test"], g3_rows, G3_FIELDS),
            (cfg["output"]["dropped_for_length"], dropped_rows, DROPPED_FIELDS)):
        path = os.path.join(out_dir, name)
        write_csv(path, rows, fields)
        written.append(path)

    summary = {
        "window": "W3.6",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": _rel(args.config),
        "split_seed": int(cfg["split_seed"]),
        "ratio": cfg.get("ratio", "8:1:1"),
        "family_atomic": family_atomic,
        "scene_unit": bool(cfg.get("scene_unit", True)),
        "stratify_length": bool(args.stratify_length),
        "smoke": smoke or None,
        "smoke_meta": smoke_meta,
        "attack_groups": len(seed_groups_all),
        "benign_groups": len(benign_groups_all),
        "attack_units": len(a_units),
        "benign_units": len(b_units),
        "attack_unit_target_groups": {"train": target_a[0], "val": target_a[1], "test": target_a[2]},
        "attack_unit_residual": {"train": a_rem[0], "val": a_rem[1], "test": a_rem[2]},
        "actual_attack_groups": {sp: len({r["seed_group_id"] for r in shards["regime_id"][sp]["attack"]})
                                 for sp in SPLITS},
        "attack_allocator": allocator,
        "attack_source_by_split": {
            sp: dict(collections.Counter(
                source_of_group.get(r["group_id"], "")
                for r in group_rows if r["group_kind"] == "attack" and r["split"] == sp))
            for sp in SPLITS},
        "attack_source_corpus": dict(collections.Counter(
            source_of_group.get(g, "") for g in seed_groups_all)),
        "benign_unit_target_groups": {"train": target_b[0], "val": target_b[1], "test": target_b[2]},
        "actual_benign_groups": {sp: len({r["benign_group_id"] for r in shards["regime_id"][sp]["benign"]})
                                 for sp in SPLITS},
        "shards": {reg: {sp: {
            "attack_rows": len(shards[reg][sp]["attack"]),
            "benign_rows": len(shards[reg][sp]["benign"]),
            "pos": shards[reg][sp]["n_pos"], "neg": shards[reg][sp]["n_neg"],
            "pos_neg_ratio": shards[reg][sp]["ratio"],
            "attack_mean_tokens": shards[reg][sp]["attack_mean"],
            "benign_mean_tokens": shards[reg][sp]["benign_mean"],
            "abs_diff_ratio": shards[reg][sp]["abs_diff_ratio"],
            "abs_diff_ratio_plain": shards[reg][sp]["abs_diff_ratio_plain"],
            "balance_stage": shards[reg][sp]["stage"],
            "count_window_ratio": shards[reg][sp]["count_window_ratio"],
            "ceil_rows": shards[reg][sp]["ceil_rows"],
            "floor_rows": shards[reg][sp]["floor_rows"],
            "ladder": shards[reg][sp]["ladder"],
            "groups_dropped": len(shards[reg][sp]["dropped_groups"]),
            "rows_dropped": len(shards[reg][sp]["dropped"]),
            "groups_kept": len(shards[reg][sp]["kept_groups"]),
        } for sp in SPLITS} for reg in REGIMES},
        "derived": {"clean_original_test": len(clean_rows),
                    "clean_groups": len(test_groups),
                    "g3_hard_negative_test": len(g3_rows),
                    "g3_groups": len({r["benign_group_id"] for r in g3_rows})},
        "family_structure": {k: v for k, v in structure.items() if k != "meta"},
        "selfcheck": [{"id": a, "passed": b, "measured": c, "detail": d} for a, b, c, d in checks],
        "inputs": {_rel(p): sha256_file(p) for p in
                   [args.encoded, args.code_carrier, seeds_path, rewrites_path]
                   if os.path.isfile(p)},
        "tokenizer_id": tokenizer_id,
        "written": [_rel(p) for p in written],
    }
    json_path = os.path.join(out_dir, cfg["output"]["report_json"])
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    written.append(json_path)

    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        stamp = _stamp()
        log_path = os.path.join(args.log_dir, "02_06_split_group_aware_%s.log" % stamp)
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("02_06_split_group_aware — %s\n" % stamp)
            fh.write("argv: %s\n" % (argv_line(),))
            fh.write(json.dumps(summary, ensure_ascii=False, indent=2))
            fh.write("\n")
        summary["log_path"] = _rel(log_path)

    return summary, checks, written


_ARGV_LINE = ""


def argv_line():
    return _ARGV_LINE


def render_report(summary, checks, cfg, args):
    """运行报告（`reports/<layer>/<name>_<version>_<YYYYMMDD-HHMMSS>.md`）。"""
    L = []
    L.append("# W3.6 运行报告 — group-aware 划分与两套体制（`02_06`）")
    L.append("")
    L.append("- 窗口：**W3.6**（交付物窗口，权限 A3）")
    L.append("- 日期：%s" % summary["timestamp"])
    L.append("- 配置：`%s`（`split_seed=%d`，`ratio=%s`，`family_atomic=%s`，`scene_unit=%s`）"
             % (summary["config"], summary["split_seed"], summary["ratio"],
                summary["family_atomic"], summary["scene_unit"]))
    L.append("- 运行模式：`--smoke %s`／`--stratify-length %s`"
             % (summary["smoke"] or "off", summary["stratify_length"]))
    L.append("- tokenizer：`%s`（长度口径，`ENV_LOCK.md` §5）" % summary["tokenizer_id"])
    L.append("- **本窗口不做门禁判定**：正式判定归 `W3.7`（`A1`–`A14`）；本节 `G1`–`G11` 为**自检级**。")
    L.append("")
    L.append("## §0 输入指纹（sha256）")
    L.append("")
    L.append("| 输入 | sha256 |")
    L.append("|---|---|")
    for k, v in sorted(summary["inputs"].items()):
        L.append("| `%s` | `%s` |" % (k, v))
    L.append("")
    L.append("## §1 划分单位与计数")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 攻击 group 总数 | %d |" % summary["attack_groups"])
    L.append("| 良性 group 总数 | %d |" % summary["benign_groups"])
    L.append("| 攻击**划分单位**数 | %d（`family_atomic=%s`）"
             % (summary["attack_units"], summary["family_atomic"]))
    L.append("| 良性**划分单位**数 | %d（`scene_unit=%s`，合成侧按场景上提）"
             % (summary["benign_units"], summary["scene_unit"]))
    L.append("| 取整规则 | Python `round()` ＝ **银行家舍入**；`n_train=round(N×0.8)`、"
             "`n_val=round(N×0.1)`、`n_test=N−前两者` |")
    L.append("| 攻击单位目标**组数** | %s |" % summary["attack_unit_target_groups"])
    L.append("| 攻击单位剩余容量 | %s |" % summary["attack_unit_residual"])
    L.append("| 攻击实算 group 数（三档） | %s |" % summary["actual_attack_groups"])
    L.append("| 良性三档**单位数**（按单位切片） | %s |" % summary["benign_unit_target_groups"])
    L.append("| 良性实算 group 数（三档，配平后） | %s |" % summary["actual_benign_groups"])
    L.append("")
    L.append("> `A13` 的目标（`W3.7` 判定）= 攻击 `seed_group_id` 去重计数之比 "
             "`round(N×0.8) : round(N×0.1) : N−前两者`，`N` 由 `seeds.meta.json` 的 "
             "`seed_count_total` 派生。本节**只报实算值**，不代 `W3.7` 下结论。")
    L.append("")
    L.append("**攻击侧分配器**：`%s`（`split.yaml` 的 `attack_allocation`）" % summary["attack_allocator"])
    L.append("")
    L.append("**攻击 group 的来源构成（`r2` 起列为报告项）**")
    L.append("")
    srcs = sorted(set(summary["attack_source_corpus"]) |
                  {s for v in summary["attack_source_by_split"].values() for s in v})
    L.append("| 来源 | " + " | ".join("`%s`" % sp for sp in SPLITS) + " | 语料 |")
    L.append("|---|" + "---|" * (len(SPLITS) + 1))
    for s in srcs:
        cells = []
        for sp in SPLITS:
            n = summary["attack_source_by_split"][sp].get(s, 0)
            tot = sum(summary["attack_source_by_split"][sp].values()) or 1
            cells.append("%d（%.1f%%）" % (n, 100.0 * n / tot))
        c = summary["attack_source_corpus"].get(s, 0)
        tot = sum(summary["attack_source_corpus"].values()) or 1
        L.append("| `%s` | %s | %d（%.1f%%） |" % (s, " | ".join(cells),
                                                   c, 100.0 * c / tot))
    L.append("")
    L.append("> **为何列此表**（`r2` 实测发现）：`family_atomic: true` 下攻击单位组数不等，若沿用"
             "「组数降序 + 组数贪心」的分配序，组数大的族会被**整体推向 `train`** ⇒ **来源构成失真**"
             "（实测 `train` 的 PKU 占比 9.1% vs 语料 20%，`val`／`test` 升到 63.6%）⇒ ID 体制的"
             "`test` 在**来源轴**上与 `train` 不同分布。本代改走 `source_stratified` 消除该失真。")
    L.append("")
    L.append("## §2 六个分片")
    L.append("")
    L.append("| 体制 | 分片 | 攻击行 | 良性行 | 正=1 | 负=0 | 正负比 | 攻击均值 token | 良性均值 token | `abs_diff_ratio` | 配平阶段 | 削减 group | 削减行 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for reg in REGIMES:
        for sp in SPLITS:
            s = summary["shards"][reg][sp]
            # 括注一律写「本档未分层」而非「默认」：后者在本报告里另指 `pre_stratify_default_run/`
            # 那一整轮（`r1`·`W3` 验收 `F-2`），同词两义会把两个不同的量混为一谈。
            L.append("| `%s` | `%s` | %d | %d | %d | %d | **%.4f** | %.1f | %.1f | **%.4f**（本档未分层 %.4f） | `%s` | %d | %d |"
                     % (reg, sp, s["attack_rows"], s["benign_rows"], s["pos"], s["neg"],
                        s["pos_neg_ratio"], s["attack_mean_tokens"], s["benign_mean_tokens"],
                        s["abs_diff_ratio"], s["abs_diff_ratio_plain"], s["balance_stage"],
                        s["groups_dropped"], s["rows_dropped"]))
    L.append("")
    L.append("> 门禁 6 的判据（`W3.7` §3.2）：`abs_diff_ratio = |均值(正) − 均值(负)| / max(两均值)` ≤ 0.20。")
    L.append("> 本表给出的是**逐分片实测值**；`配平阶段` 为 `balance` 表示未启用长度分层或该分片未超阈。")
    L.append("")
    L.append("## §3 族结构与 `family_id`（自检 `G9`）")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    for k, v in summary["family_structure"].items():
        L.append("| `%s` | `%s` |" % (k, v))
    L.append("")
    L.append("> `family_id` **只用于登记与测量，不改变划分单位**（`W3` 裁定 2026-09-30）。"
             "族跨档的量化由 `W3.7` 的 `A14` 承担。")
    L.append("")
    L.append("## §4 自检 `G1`–`G11`（**自检级，非门禁**）")
    L.append("")
    L.append("| 断言 | 结果 | 实测 | 说明 |")
    L.append("|---|---|---|---|")
    for cid, ok, measured, detail in checks:
        L.append("| `%s` | %s | %s | %s |" % (cid, "**PASS**" if ok else "**FAIL**", measured, detail))
    L.append("")
    L.append("## §5 正负配平与长度分层")
    L.append("")
    L.append("- 配平口径（`V-4`）：**只减不增、不跨 group**；削减次序 = 良性侧全局打散序（不再消耗随机数）。")
    L.append("- 长度分层（`--stratify-length`，`method: mean_targeted_stratified`）：对良性侧按**组均值长度**"
             "等频分桶（strata），选样次序由**均值定向贪心**给出 —— 每一步剔除使 "
             "`|u_g − D|` 最小的组（`u_g = s·(L − A)`、`D = T − A·n`，`A` = 攻击侧均值），"
             "剔除后 `D ← D − u_g` ⇒ **均值贴住攻击侧、行数单调下降**；产出一条组级剔除序列后，"
             "各 count 窗口在同一序列上取前缀。")
    L.append("")
    L.append("| 体制 | 分片 | **本档未分层** `abs_diff_ratio` | 最终 `abs_diff_ratio` | count 窗口 `r` | 窗口行数上界 | 良性行 | 正负比 | 阶段 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for reg in REGIMES:
        for sp in SPLITS:
            s = summary["shards"][reg][sp]
            L.append("| `%s` | `%s` | %.4f | **%.4f** | %s | %s | %d | %.4f | `%s` |"
                     % (reg, sp, s["abs_diff_ratio_plain"], s["abs_diff_ratio"],
                        ("%.2f" % s["count_window_ratio"]) if s["count_window_ratio"] else "—",
                        s["ceil_rows"] if s["ceil_rows"] else "—", s["benign_rows"],
                        s["pos_neg_ratio"], s["balance_stage"]))
    L.append("")
    L.append("**长度分层候选窗口（逐档实算，供复核）**")
    L.append("")
    L.append("| 体制 | 分片 | count 窗口 r | 行数上界 | 良性行 | `abs_diff_ratio` | 正负比 | 达阈 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for reg in REGIMES:
        for sp in SPLITS:
            for e in summary["shards"][reg][sp].get("ladder") or []:
                L.append("| `%s` | `%s` | %.2f | %d | %s | %s | %s | %s |"
                         % (reg, sp, e["count_window_ratio"], e["ceil_rows"],
                            e["benign_rows"] if e["benign_rows"] is not None else "—",
                            ("%.4f" % e["abs_diff_ratio"]) if e["abs_diff_ratio"] is not None else "—",
                            ("%.4f" % e["pos_neg_ratio"]) if e["pos_neg_ratio"] else "—",
                            "是" if e["passed_threshold"] else "否"))
    L.append("")
    L.append("## §6 削减台账")
    L.append("")
    L.append("`%s`（逐行记录被削减的良性 group；`reason` ∈ `balance`／`length_stratify`）"
             % cfg["output"]["dropped_for_length"])
    L.append("")
    L.append("## §7 派生视图")
    L.append("")
    L.append("| 产物 | 行数 | group 数 | 说明 |")
    L.append("|---|---|---|---|")
    L.append("| `clean_original_test.csv` | %d | %d | 测试 group 的改写体**明文**（未编码）；"
             "用于阶段4 灾难性遗忘验证，**不参与任何训练** |"
             % (summary["derived"]["clean_original_test"], summary["derived"]["clean_groups"]))
    L.append("| `g3_hard_negative_test.csv` | %d | %d | 落 `test` 的 G3；取**体制一**的测试 G3 行为超集。"
             "**筛选须按 `benign_group_id` ∩ 该体制 test 良性 group 集**"
             "（**`ood ⊆ id` 是集合包含，不蕴含「按编码过滤 id 即得 ood」**） |"
             % (summary["derived"]["g3_hard_negative_test"], summary["derived"]["g3_groups"]))
    L.append("")
    L.append("## §8 口径说明与偏离")
    L.append("")
    L.append("1. **体制二的编码过滤作用面**：攻击行按 `encoding_type` 严格过滤；良性行中 "
             "`encoding_type` 为空串者（G1／G2，本身不含编码层）两体制均保留，非空者按同一编码集过滤。"
             "该口径使 `AGENTS.md` §6-3（留出 4 编码不得以任何形式入训练）与 `A3`／`A4`／`A5` 同时成立。")
    L.append("2. **`ood_subset_of_id`**：体制二每个分片的良性行集 ⊆ 体制一同分片（先配平面向体制一，"
             "再在其保留集内按编码子集配平体制二）⇒ `g3_hard_negative_test.csv` 的超集语义成立。")
    L.append("3. **合成侧场景约束**（§2.7）：`scene_unit: true` ⇒ 同一场景（`SC{nn}`）的全部 group "
             "上提为一个划分单位，断言 `G10` 据此成立。")
    L.append("4. **§3 措辞偏离**：`--stratify-length` 的原文写「按 token 长度分位分桶，**各桶按比例抽取**」；"
             "但**按比例抽**不改变该侧长度分布 ⇒ 无法使正负**均值**靠拢（而均值差正是门禁 6 的唯一判据）。"
             "实现改为 `mean_targeted_stratified`：**分层（等频桶）+ 非比例分配（按对齐攻击侧均值定率）**，"
             "仍满足「分层 / 只减不增 / 不跨 group / 不改变 group 分片归属」四条硬约束。"
             "已登记 `DEVIATION_LEDGER.md`。")
    L.append("5. **`count_window_ratios` 最松档取 0.82**（判定带下界为 0.80）：留 0.02 余量，"
             "避免把正负比正好压在判定带边界上。")
    L.append("")
    L.append("## §9 复现命令")
    L.append("")
    L.append("```text")
    L.append("python scripts/data/02_06_split_group_aware.py \\")
    L.append("  --config configs/data/split.yaml \\")
    L.append("  --encoded data/interim/encoded/encoded.csv \\")
    L.append("  --code-carrier data/interim/code_carrier/code_carrier.csv \\")
    L.append("  --benign-dir data/interim/benign_G1_G2_G3/ \\")
    L.append("  --out data/interim/split_checks/%s%s"
             % ("smoke/" if summary["smoke"] else "",
                " \\" if (summary["smoke"] or summary["stratify_length"]) else ""))
    if summary["smoke"]:
        L.append("  --smoke %d%s" % (summary["smoke"], " \\" if summary["stratify_length"] else ""))
    if summary["stratify_length"]:
        L.append("  --stratify-length")
    L.append("```")
    L.append("")
    L.append("## §10 产出清单")
    L.append("")
    for p in summary["written"]:
        L.append("- `%s`" % p)
    if summary.get("log_path"):
        L.append("- `%s`" % summary["log_path"])
    L.append("")
    return "\n".join(L)


# ===========================================================================
# 7. CLI
# ===========================================================================
def parse_args(argv):
    p = argparse.ArgumentParser(description="W3.6 group-aware 划分与两套体制（02_06）")
    p.add_argument("--config", default="configs/data/split.yaml")
    p.add_argument("--encoded", default=None)
    p.add_argument("--code-carrier", default=None)
    p.add_argument("--benign-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", type=int, default=0,
                   help="只取前 N 个攻击单位（良性侧按单位比例同缩放），产物落 out/smoke/")
    p.add_argument("--stratify-length", action="store_true",
                   help="启用长度分层采样（门禁 6 的回退动作）")
    p.add_argument("--seeds", default=None)
    p.add_argument("--seeds-meta", default=None)
    p.add_argument("--rewrites", default=None)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-report", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    global _ARGV_LINE
    args = parse_args(argv)
    _ARGV_LINE = " ".join(sys.argv)

    try:
        import yaml as _yaml
    except ImportError:  # pragma: no cover
        print("FATAL: 需要 PyYAML 解析 %s" % args.config)
        return EXIT_USAGE
    if not os.path.isfile(args.config):
        print("FATAL: 缺少配置 %s" % _rel(args.config))
        return EXIT_FAIL
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = _yaml.safe_load(fh)["split"]

    args.encoded = args.encoded or cfg["inputs"]["encoded"]
    args.code_carrier = args.code_carrier or cfg["inputs"]["code_carrier"]
    args.benign_dir = args.benign_dir or cfg["inputs"]["benign_dir"]
    args.out = args.out or cfg["output"]["dir"]
    args.report_dir = args.report_dir or cfg.get("report_dir", "reports/data")
    args.log_dir = args.log_dir or cfg.get("log_dir", "logs/data")

    for p in (args.encoded, args.code_carrier, args.benign_dir):
        if not os.path.exists(p):
            print("FATAL: 缺少输入 %s" % _rel(p))
            return EXIT_FAIL

    result = run(args, cfg)
    if result is None:
        return EXIT_FAIL
    summary, checks, written = result

    if not args.no_report:
        os.makedirs(args.report_dir, exist_ok=True)
        stamp = summary["timestamp"].replace("-", "").replace(":", "").replace("T", "-")
        report_path = os.path.join(args.report_dir,
                                   "%s_%s.md" % (cfg.get("report_basename", "split"), stamp))
        with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_report(summary, checks, cfg, args))
        print("  report: %s" % _rel(report_path))

    print("== 02_06 split ==")
    for cid, ok, measured, _detail in checks:
        print("  %-4s %s  %s" % (cid, "PASS" if ok else "FAIL", measured))
    for reg in REGIMES:
        for sp in SPLITS:
            s = summary["shards"][reg][sp]
            print("  %-10s %-5s att=%5d ben=%5d ratio=%.4f |dlen|=%.4f stage=%s"
                  % (reg, sp, s["attack_rows"], s["benign_rows"], s["pos_neg_ratio"],
                     s["abs_diff_ratio"], s["balance_stage"]))
    print("  derived: clean=%d g3_hard_negative=%d"
          % (summary["derived"]["clean_original_test"],
             summary["derived"]["g3_hard_negative_test"]))

    failed = [c for c in checks if not c[1]]
    if failed:
        print("FAILED self-checks: %s" % ", ".join(c[0] for c in failed))
        return EXIT_FAIL
    print("OK: 自检 G1–G11 全部通过（正式判定归 02_07）")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

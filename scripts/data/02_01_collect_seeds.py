# -*- coding: utf-8 -*-
"""02_01_collect_seeds.py — 种子收集与 seed_group_id 分配（W3.1 交付物）

用途
----
从 ``data/raw/`` 的两个恶意种子来源（AgentHarm、PKU-SafeRLHF）收集种子提示，
分配 ``seed_id`` 与 ``seed_group_id``，产出 ``seeds.csv`` 与 ``seeds.meta.json``。

接口契约（``windows/W3.1_seed_collect.md`` §3）
---------------------------------------------
::

    python scripts/data/02_01_collect_seeds.py \\
      --config configs/data/seed_collect.yaml \\
      --out data/interim/seed_collect/ \\
      [--smoke N]

* 退出码 ``0`` 成功；非 0 失败（含任一门禁断言不通过）。
* ``--smoke N``：每来源只取前 N 条，产物写入 ``<out>/smoke/``，**不得覆盖全量产物**。
* 幂等：同一 ``--config`` + 同一 ``--raw-root`` 重复运行，``seeds.csv`` 逐字节相同。
  ``seeds.meta.json`` 的 ``generated_at`` 是**唯一**的非确定性字段，已在
  ``non_deterministic_fields`` 中单列声明。
* ``--help`` 必须可用。``make check`` / ``make smoke`` 不调用本脚本。

新增开关（超出 §3 契约的可选扩展，不改变契约行为）
--------------------------------------------------
``--raw-root``  默认 ``data/raw``；测试以此指向 ``tests/data/fixtures/raw`` 实现**离线可跑**。
``--log-dir``   默认 ``logs/data``；``--no-log`` 关闭日志落盘（供单元测试使用）。

门禁断言（§3.3；不通过即非零退出）
----------------------------------
``B1`` ``seed_group_id`` 在 ``seeds.csv`` 内唯一（无重复行）。
``B2`` ``seed_id`` ↔ ``seed_group_id`` 一一对应（单射 + 满射）。
``B3`` ``seed_group_id`` 连续无空洞（``SG-0001`` … ``SG-<N>``）。
``B4`` 每条 ``raw_sha256`` 与实际文件 sha256 一致（存在 ``SOURCE_MANIFEST.json`` 时一并交叉核对）。

**不在本窗口断言**「改写体共享同一 group」——改写体此刻尚不存在，该断言由 ``02_07``（W3.7）承担。
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime

try:
    import yaml
except ImportError:  # pragma: no cover - 仅在缺依赖的解释器上触发
    yaml = None

MISSING = "MISSING"
UNKNOWN = "UNKNOWN"
GROUP_PREFIX = "SG-"
SEED_ID_WIDTH = 3
GROUP_ID_WIDTH = 4


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def sha256_file(path):
    """返回文件的 sha256 十六进制串。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    """返回文件的 md5 十六进制串。"""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    """读取 YAML 配置；缺 PyYAML 时给出可操作错误而非 traceback。"""
    if yaml is None:
        raise RuntimeError("PyYAML 缺失，无法解析配置文件：%s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "seed_collect" not in cfg:
        raise ValueError("配置缺少顶层 seed_collect 段：%s" % path)
    return cfg["seed_collect"]


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj):
    """UTF-8 + LF 落盘；ensure_ascii=False 保留中文；缩进 2 便于 diff。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def rel_posix(path, base):
    """把 path 表示为相对 base 的 POSIX 风格路径（落盘统一用 /）。

    不同盘符时 ``os.path.relpath`` 会抛 ValueError（Windows）——此时退回绝对路径，
    只影响元数据里的展示字符串，不影响任何判定。
    """
    try:
        rel = os.path.relpath(path, base)
    except ValueError:  # pragma: no cover - 仅 Windows 跨盘符
        rel = os.path.abspath(path)
    return rel.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# 各来源读取
# ---------------------------------------------------------------------------
def load_agentharm(raw_root, src):
    """AgentHarm：整份 behaviors 数组全量入选，保持来源文件内的出现顺序。"""
    rule = src["subset_rule"]
    fmap = src["field_map"]
    path = os.path.join(raw_root, *rule["file"].split("/"))
    if not os.path.exists(path):
        raise FileNotFoundError("来源文件不存在：%s" % path)

    doc = read_json(path)
    items = doc
    for key in rule["array_path"].split("."):
        items = items[key]
    if not isinstance(items, list):
        raise ValueError("%s 的 %s 不是数组" % (rule["file"], rule["array_path"]))

    expected = rule.get("expected_count")
    if expected is not None and len(items) != expected:
        raise ValueError(
            "AgentHarm 实算 %d 条，与配置 expected_count=%d 不符；"
            "按 windows/W3.1_seed_collect.md §2.1 须以实际为准并呈报 W3，"
            "不得为凑数而重复或裁剪。请先确认来源 revision 是否变动。" % (len(items), expected)
        )

    fhash = sha256_file(path)
    seeds = []
    for raw in items:
        seeds.append(
            {
                "source": src["name"],
                "raw_text": str(raw.get(fmap["raw_text"], "")),
                "source_item_id": str(raw.get(fmap["source_item_id"], "")),
                "harm_category": str(raw.get(fmap["harm_category"], "") or UNKNOWN),
                "severity": "",  # AgentHarm 无严重度字段 ⇒ 空串（§3.1）
                "raw_file": rule["file"],
                "raw_sha256": fhash,
            }
        )
    return seeds


def load_pku(raw_root, src):
    """PKU-SafeRLHF：severity 阈值筛选 → 按 prompt 去重 → 确定性抽样。

    返回的种子按**文件扫描顺序**排列（抽样取的是成员集合，排序后落盘），
    使产物的顺序不依赖 ``random.sample`` 的内部返回顺序。
    """
    rule = src["subset_rule"]
    fmap = src["field_map"]
    hrule = src["harm_category_rule"]
    agg_fields = rule["severity_fields"] if "severity_fields" in rule else None
    if agg_fields is None:
        # 由 field_map.aggregator 推导：response_{0,1}_severity_level
        agg_fields = ["response_0_severity_level", "response_1_severity_level"]

    candidates = []          # 去重后的候选，保持首次出现顺序
    seen_prompts = set()
    file_hashes = {}

    for rel in rule["files"]:
        path = os.path.join(raw_root, *rel.split("/"))
        if not os.path.exists(path):
            raise FileNotFoundError("来源文件不存在：%s" % path)
        fhash = sha256_file(path)
        file_hashes[rel] = fhash
        # rel = "PKU-SafeRLHF/data/<variant>/<split>.jsonl" ⇒ 取最后两段作来源内定位
        parts = rel.split("/")
        variant_dir, split_name = parts[-2], parts[-1]
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sev_values = [rec.get(f, 0) for f in agg_fields]
                sev = {"max": max, "min": min}.get(rule["aggregator"], max)(sev_values)
                if not _compare(sev, rule["operator"], rule["value"]):
                    continue

                prompt = str(rec.get(fmap["raw_text"], ""))
                if rule["dedup_by"] == "prompt":
                    if prompt in seen_prompts:
                        continue
                    seen_prompts.add(prompt)

                candidates.append(
                    {
                        "source": src["name"],
                        "raw_text": prompt,
                        "source_item_id": "%s/%s#L%d" % (variant_dir, split_name, lineno),
                        "harm_category": _pick_harm_category(rec, hrule, agg_fields, rule),
                        "severity": str(sev),
                        "raw_file": rel,
                        "raw_sha256": fhash,
                        "_file_order": len(candidates),
                    }
                )

    expected = rule.get("candidate_count_expected")
    if expected is not None and len(candidates) != expected:
        raise ValueError(
            "PKU-SafeRLHF 阈值筛选后实测 %d 条候选，与配置 candidate_count_expected=%d 不符；"
            "阈值口径或来源 revision 可能变动，须先确认再重跑（不得改配置迁就实算值）。"
            % (len(candidates), expected)
        )

    sample = rule["sample"]
    n = int(sample["n"])
    if n > len(candidates):
        raise ValueError(
            "抽样条数 %d 超过候选数 %d，无法抽样" % (n, len(candidates))
        )
    rng = random.Random(int(sample["seed"]))
    chosen = rng.sample(candidates, n)
    if sample.get("sort_after_sample") == "file_order":
        chosen.sort(key=lambda r: r["_file_order"])

    for rec in chosen:
        rec.pop("_file_order", None)
    return chosen


def _compare(value, op, target):
    if op == "==":
        return value == target
    if op == ">=":
        return value >= target
    if op == ">":
        return value > target
    if op == "<=":
        return value <= target
    if op == "<":
        return value < target
    raise ValueError("不支持的比较符：%r" % op)


def _pick_harm_category(rec, hrule, agg_fields, rule):
    """在 severity == 阈值 的那些 response 上取 harm_category，按来源字典自身键序取第一个 True。"""
    names = []
    for field in hrule["fields"]:
        idx = field.split("_")[1]  # response_0_harm_category -> "0"
        sev = rec.get("response_%s_severity_level" % idx)
        if sev != rule["value"]:
            continue
        cat = rec.get(field) or {}
        for name, flagged in cat.items():  # dict 保持来源 JSON 的键序（S1→S19）
            if flagged:
                names.append(name)
    if not names:
        return hrule.get("fallback", UNKNOWN)
    if hrule["select"] == "first_true_in_source_key_order":
        return names[0]
    return names[0]


# ---------------------------------------------------------------------------
# 编号分配
# ---------------------------------------------------------------------------
def assign_ids(seeds, id_prefix):
    """按 seed_id 排序后连续分配 seed_group_id（SG-0001 起，四位零填）。"""
    counters = {}
    for rec in seeds:
        key = id_prefix[rec["source"]]
        counters[key] = counters.get(key, 0) + 1
        rec["seed_id"] = "SEED-%s-%0*d" % (key, SEED_ID_WIDTH, counters[key])

    seeds.sort(key=lambda r: r["seed_id"])
    for i, rec in enumerate(seeds, 1):
        rec["seed_group_id"] = "%s%0*d" % (GROUP_PREFIX, GROUP_ID_WIDTH, i)
    return seeds


def truncate_per_source(seeds, n):
    """冒烟档：每来源取前 n 条（按 seed_id 顺序），随后重排 group_id。"""
    if n <= 0:
        return seeds
    kept, counts = [], {}
    for rec in seeds:
        src = rec["source"]
        if counts.get(src, 0) >= n:
            continue
        counts[src] = counts.get(src, 0) + 1
        kept.append(rec)
    for i, rec in enumerate(kept, 1):
        rec["seed_group_id"] = "%s%0*d" % (GROUP_PREFIX, GROUP_ID_WIDTH, i)
    return kept


# ---------------------------------------------------------------------------
# 门禁断言 B1–B4
# ---------------------------------------------------------------------------
def run_assertions(rows, raw_root, manifests):
    """返回 {B1..B4: {passed, detail, ...}}。全部为 True 才算通过。"""
    results = {}

    groups = [r["seed_group_id"] for r in rows]
    dup_groups = sorted({g for g in groups if groups.count(g) > 1})
    results["B1"] = {
        "passed": len(dup_groups) == 0,
        "rule": "seed_group_id 在 seeds.csv 内唯一（无重复行）",
        "group_count": len(groups),
        "distinct_group_count": len(set(groups)),
        "duplicated": dup_groups,
    }

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed_id"], set()).add(r["seed_group_id"])
    bad_pairs = sorted(s for s, g in pairs.items() if len(g) != 1)
    bijective = (
        len(bad_pairs) == 0
        and len(pairs) == len(rows)
        and len(set(groups)) == len(set(r["seed_id"] for r in rows))
    )
    results["B2"] = {
        "passed": bool(bijective),
        "rule": "seed_id ↔ seed_group_id 一一对应（单射 + 满射）",
        "seed_id_count": len(set(r["seed_id"] for r in rows)),
        "group_id_count": len(set(groups)),
        "multi_group_seed_ids": bad_pairs,
    }

    expected = ["%s%0*d" % (GROUP_PREFIX, GROUP_ID_WIDTH, i) for i in range(1, len(rows) + 1)]
    results["B3"] = {
        "passed": groups == expected,
        "rule": "seed_group_id 连续无空洞（SG-0001 … SG-<N>）",
        "expected_first": expected[0] if expected else None,
        "expected_last": expected[-1] if expected else None,
        "actual_first": groups[0] if groups else None,
        "actual_last": groups[-1] if groups else None,
        "has_hole": bool(sorted(set(groups)) != sorted(expected)),
    }

    mismatched, checked_files, manifest_mismatched = [], {}, []
    for r in rows:
        rel = r["raw_file"]
        if rel in checked_files:
            continue
        path = os.path.join(raw_root, *rel.split("/"))
        if not os.path.exists(path):
            mismatched.append({"raw_file": rel, "error": "file_missing"})
            checked_files[rel] = None
            continue
        actual = sha256_file(path)
        checked_files[rel] = actual
        if actual != r["raw_sha256"]:
            mismatched.append(
                {"raw_file": rel, "expected": r["raw_sha256"], "actual": actual}
            )
    # 与 SOURCE_MANIFEST.json 交叉核对（存在即核对）。
    # manifest 的 files[].path 相对 **来源目录**，而 raw_file 相对 data/raw ⇒ 须补 raw_dir 前缀。
    for src_name, manifest in manifests.items():
        if not manifest:
            continue
        prefix = manifest.get("_dir", "")
        for entry in manifest.get("files", []):
            rel = (prefix + "/" + entry["path"]) if prefix else entry["path"]
            if rel not in checked_files:
                continue  # 该文件未被本次种子收集消费，不在核对范围
            actual = checked_files[rel]
            if actual is None:
                continue
            if actual != entry["sha256"]:
                manifest_mismatched.append(
                    {
                        "source": src_name,
                        "path": rel,
                        "manifest_sha256": entry["sha256"],
                        "actual_sha256": actual,
                    }
                )
    results["B4"] = {
        "passed": not mismatched and not manifest_mismatched,
        "rule": "每条 raw_sha256 与实际文件 sha256 一致（有 SOURCE_MANIFEST.json 时一并交叉核对）",
        "files_checked": len(checked_files),
        "file_sha256": checked_files,
        "mismatched": mismatched,
        "manifest_mismatched": manifest_mismatched,
    }
    return results


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------
def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fields, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return path


def read_csv(path, fields):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [
            {k: (row.get(k) or "") for k in fields}
            for row in csv.DictReader(fh)
        ]


def assert_no_cr(path):
    """字节法核实无孤立 CR（`\\r\\n` 亦不允许：V-9 要求 LF）。"""
    with open(path, "rb") as fh:
        data = fh.read()
    return data.count(b"\r") == 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_meta(cfg, rows, raw_root, out_dir, config_path, manifests, assertions):
    by_source = {}
    total = len(rows)
    for rec in rows:
        by_source.setdefault(rec["source"], 0)
        by_source[rec["source"]] += 1
    counts = {
        name: {
            "count": n,
            "ratio": round(n / total, 6) if total else 0.0,
        }
        for name, n in sorted(by_source.items())
    }

    acq_rel = cfg["output"]["raw_acquisition_json"]
    acq_path = os.path.join(out_dir, acq_rel)
    acq_pointer = {
        "path": rel_posix(acq_path, os.getcwd()),
        "exists": os.path.exists(acq_path),
        "sha256": sha256_file(acq_path) if os.path.exists(acq_path) else None,
    }

    pku = next(s for s in cfg["sources"] if s["name"] == "PKU-SafeRLHF")
    prule = pku["subset_rule"]
    subset_rule_text = (
        "%s；去重规则 dedup_by=%s（keep=%s）；抽样 %s（seed=%d, n=%d，抽样后按文件序排序）"
        % (
            prule["threshold_expr"],
            prule["dedup_by"],
            prule["dedup_keep"],
            prule["sample"]["method"],
            prule["sample"]["seed"],
            prule["sample"]["n"],
        )
    )

    meta = {
        "seed_count_total": total,
        "seed_count_by_source": counts,
        "seed_group_id_range": {
            "first": rows[0]["seed_group_id"] if rows else None,
            "last": rows[-1]["seed_group_id"] if rows else None,
        },
        "subset_rule": subset_rule_text,
        "subset_rule_pku": prule["description"].strip(),
        "subset_rule_agentharm": cfg["sources"][0]["subset_rule"]["description"].strip(),
        "raw_acquisition": acq_pointer,
        "source_manifests": {
            name: {
                "path": rel_posix(
                    os.path.join(raw_root, m["_dir"], "SOURCE_MANIFEST.json"), os.getcwd()
                ),
                "sha256": sha256_file(
                    os.path.join(raw_root, m["_dir"], "SOURCE_MANIFEST.json")
                ),
            }
            for name, m in sorted(manifests.items())
            if m
        },
        "config_path": rel_posix(config_path, os.getcwd()),
        "config_sha256": sha256_file(config_path),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "non_deterministic_fields": ["generated_at"],
        "assertions": {
            k: {
                "passed": v["passed"],
                "rule": v["rule"],
                **{
                    kk: vv
                    for kk, vv in v.items()
                    if kk not in ("passed", "rule", "file_sha256")
                },
            }
            for k, v in assertions.items()
        },
        "assertions_all_passed": all(v["passed"] for v in assertions.values()),
        "raw_file_sha256": assertions["B4"]["file_sha256"],
        "agentharm_variant_structure": _agentharm_variant_note(raw_root, cfg),
    }
    return meta


def _agentharm_variant_note(raw_root, cfg):
    """登记 AgentHarm 的 44×4 变体结构（供 W3.6 划分与 DATA_CARD 披露，见回执呈报项）。

    AgentHarm 的 176 条 = 44 个 ``id_original`` × 4 个变体
    （detailed_prompt × hint_included 的四种组合），四个变体的 prompt 文本互不相同
    （实测 176/176 唯一），故按 §3.3 B2「一一对应」各占一个 group。
    副作用：同一 ``id_original`` 的变体可能落入不同分片 —— 属**语义近似**层面，
    须由 W3.6/W3.7 处置并在 DATA_CARD 披露。
    """
    src = next(s for s in cfg["sources"] if s["name"] == "AgentHarm")
    path = os.path.join(raw_root, *src["subset_rule"]["file"].split("/"))
    if not os.path.exists(path):
        return {"available": False}
    items = read_json(path)["behaviors"]
    groups = {}
    for raw in items:
        groups.setdefault(str(raw.get("id_original")), []).append(str(raw.get("id")))
    sizes = sorted({len(v) for v in groups.values()})
    return {
        "available": True,
        "field": "id_original",
        "n_behaviors": len(items),
        "n_original_entries": len(groups),
        "variants_per_original": sizes,
        "note": (
            "176 = 44 个 id_original × 4 变体（detailed_prompt × hint_included）；"
            "四个变体 prompt 文本互不相同 ⇒ 按 B2 各占一个 seed_group_id；"
            "同一 id_original 的变体可能被 W3.6 划入不同分片，须在 DATA_CARD 披露。"
        ),
        "id_original_to_seed_ids": None,  # 由 seeds.csv 的 source_item_id 反查即可，不重复落盘
    }


def load_manifests(raw_root, cfg):
    manifests = {}
    for src in cfg["sources"]:
        mpath = os.path.join(raw_root, src["raw_dir"], "SOURCE_MANIFEST.json")
        if os.path.exists(mpath):
            m = read_json(mpath)
            m["_dir"] = src["raw_dir"]
            manifests[src["name"]] = m
        else:
            manifests[src["name"]] = None
    return manifests


class Tee(object):
    """stdout 同时写日志文件。"""

    def __init__(self, log_path):
        self.log_path = log_path
        self.fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self.fh = open(self.log_path, "w", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *exc):
        if self.fh:
            self.fh.close()
        return False

    def write(self, text):
        # 先落盘再上屏：控制台编码异常不得导致日志缺行（失败日志必须保留）
        if self.fh:
            self.fh.write(text)
        try:
            sys.__stdout__.write(text)
        except UnicodeEncodeError:  # pragma: no cover - 非 UTF-8 控制台
            enc = getattr(sys.__stdout__, "encoding", None) or "utf-8"
            sys.__stdout__.write(text.encode(enc, "replace").decode(enc, "replace"))


def build_parser():
    p = argparse.ArgumentParser(
        prog="02_01_collect_seeds.py",
        description="W3.1 种子收集：从 data/raw/ 的两个来源收集恶意种子并分配 seed_group_id。",
    )
    p.add_argument("--config", default="configs/data/seed_collect.yaml",
                   help="种子收集配置（YAML）")
    p.add_argument("--out", default="data/interim/seed_collect/",
                   help="产物输出目录；--smoke 模式下自动改写到其 smoke/ 子目录")
    p.add_argument("--raw-root", default="data/raw",
                   help="原始数据根目录（默认 data/raw；测试指向 fixtures 以实现离线可跑）")
    p.add_argument("--smoke", type=int, default=0,
                   help="冒烟档：每来源只取 N 条；0（缺省）= 全量。产物落 <out>/smoke/")
    p.add_argument("--log-dir", default="logs/data",
                   help="日志目录（默认 logs/data）")
    p.add_argument("--no-log", action="store_true", help="不落盘日志（供单元测试使用）")
    return p


def main(argv=None):
    # 控制台统一 UTF-8 + LF（沿用 scripts/env/01_00_smoke_env.py:1035 的项目惯例）。
    # Windows 默认控制台代码页为 GBK，含 '↔' 等非 GBK 字符的断言文本会触发 UnicodeEncodeError。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(newline="\n", encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover - 非文本流
            pass

    args = build_parser().parse_args(argv)

    config_path = args.config
    cfg = load_config(config_path)

    out_dir = args.out
    if args.smoke and args.smoke > 0:
        out_dir = os.path.join(out_dir, cfg["output"]["smoke_subdir"])

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, "02_01_collect_seeds_%s.log" % stamp)

    def run():
        print("[02_01] config=%s" % config_path)
        print("[02_01] raw_root=%s" % args.raw_root)
        print("[02_01] out=%s  smoke=%s" % (out_dir, args.smoke or "FULL"))

        seeds = []
        for src in cfg["sources"]:
            if src["subset_rule"]["strategy"] == "full":
                part = load_agentharm(args.raw_root, src)
            else:
                part = load_pku(args.raw_root, src)
            print("[02_01] source=%-14s collected=%d" % (src["name"], len(part)))
            seeds.extend(part)

        seeds = assign_ids(seeds, cfg["id_prefix"])
        print("[02_01] seeds total=%d" % len(seeds))

        if args.smoke and args.smoke > 0:
            seeds = truncate_per_source(seeds, args.smoke)
            print("[02_01] smoke truncation -> %d" % len(seeds))

        manifests = load_manifests(args.raw_root, cfg)
        assertions = run_assertions(seeds, args.raw_root, manifests)

        fields = cfg["csv_fields"]
        rows = [{k: r.get(k, "") for k in fields} for r in seeds]

        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, cfg["output"]["seeds_csv"])
        write_csv(csv_path, rows, fields)

        cr_ok = assert_no_cr(csv_path)
        print("[02_01] wrote %s (%d rows, utf-8+LF, no_CR=%s)" % (csv_path, len(rows), cr_ok))

        meta = build_meta(cfg, seeds, args.raw_root, out_dir, config_path, manifests, assertions)
        meta["seeds_csv_sha256"] = sha256_file(csv_path)
        meta["seeds_csv_no_CR"] = cr_ok
        meta_path = os.path.join(out_dir, cfg["output"]["seeds_meta_json"])
        write_json(meta_path, meta)
        print("[02_01] wrote %s" % meta_path)

        for key in ("B1", "B2", "B3", "B4"):
            a = assertions[key]
            print("[02_01] ASSERT %s %s — %s" % (key, "PASS" if a["passed"] else "FAIL", a["rule"]))

        print("[02_01] seed_count_by_source="
              + json.dumps(meta["seed_count_by_source"], ensure_ascii=False))
        print("[02_01] seed_group_id_range=%s..%s"
              % (meta["seed_group_id_range"]["first"], meta["seed_group_id_range"]["last"]))

        if not meta["assertions_all_passed"]:
            print("[02_01] RESULT=FAIL (assertions not all passed)")
            return 1
        if not cr_ok:
            print("[02_01] RESULT=FAIL (CR found in seeds.csv)")
            return 1
        print("[02_01] RESULT=OK")
        return 0

    if args.no_log:
        return run()
    saved = sys.stdout
    with Tee(log_path) as tee:
        sys.stdout = tee
        try:
            tee.write("[02_01] log=%s\n" % log_path)
            return run()
        finally:
            sys.stdout = saved


if __name__ == "__main__":
    sys.exit(main())

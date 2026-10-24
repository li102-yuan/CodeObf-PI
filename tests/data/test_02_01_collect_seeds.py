# -*- coding: utf-8 -*-
"""tests/data/test_02_01_collect_seeds.py — ``scripts/data/02_01_collect_seeds.py`` 的单元测试

覆盖 ``windows/W3.1_seed_collect.md`` §4 要求的五项：

1. 唯一性／单射性／连续性三条断言（``B1``／``B2``／``B3``）的**正例与反例** ——
   反例一律用**构造数据**（内存中构造的行），**不得**使用真实来源。
2. ``--smoke N`` **不覆盖**全量产物。
3. 输出编码为 UTF-8 且**不含孤立 CR**。
4. 幂等性：同输入两次运行的 ``seeds.csv`` **字节相同**。
5. **无网络访问**：全部用例离线可跑，数据取自 ``tests/data/fixtures/``。

另覆盖 ``B4``（sha256 锚点，§3.3）的正反例、``--help`` 可用性、
CSV 引号规则（fixture 里含逗号／引号／换行的提示），以及 PKU 侧的
阈值筛选／去重／确定性抽样三条子规则。

约束：不加载模型、不访问网络、不访问 ``data/``、不改变环境。
被测脚本以文件路径动态加载（文件名以数字开头，不能用 ``import`` 语句）。
"""
import ast
import csv
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "data", "02_01_collect_seeds.py")

FIXTURES = os.path.join(ROOT, "tests", "data", "fixtures")
FIXTURE_CFG = os.path.join(FIXTURES, "seed_collect_fixture.yaml")
FIXTURE_RAW = os.path.join(FIXTURES, "raw")

# fixture 预期：AgentHarm 6 条 + PKU 抽样 2 条 = 8 条
FIXTURE_AH_COUNT = 6
FIXTURE_PKU_COUNT = 2
FIXTURE_TOTAL = FIXTURE_AH_COUNT + FIXTURE_PKU_COUNT

AH_RAW_FILE = "AgentHarm/benchmark/harmful_behaviors_test_public.json"
PKU_RAW_FILE = "PKU-SafeRLHF/data/Alpaca-7B/train.jsonl"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("w31_collect_seeds", SCRIPT)


def _run(mod, out, smoke=0):
    """在进程内调用被测脚本的 main()，返回退出码。"""
    argv = ["--config", FIXTURE_CFG, "--out", out,
            "--raw-root", FIXTURE_RAW, "--no-log"]
    if smoke:
        argv += ["--smoke", str(smoke)]
    return mod.main(argv)


def _read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _fixture_sha(mod, rel):
    return mod.sha256_file(os.path.join(FIXTURE_RAW, *rel.split("/")))


def _row(mod, seed_id, group_id, raw_file, sha=None):
    """构造一行最小合法记录（用于断言反例；不取自真实来源）。"""
    return {
        "seed_id": seed_id,
        "seed_group_id": group_id,
        "source": "AgentHarm" if seed_id.startswith("SEED-AH") else "PKU-SafeRLHF",
        "harm_category": "Fraud",
        "severity": "",
        "raw_text": "CONSTRUCTED %s" % seed_id,
        "source_item_id": seed_id,
        "raw_file": raw_file,
        "raw_sha256": sha if sha is not None else _fixture_sha(mod, raw_file),
    }


# ---------------------------------------------------------------------------
# 1. 正例：B1 / B2 / B3 / B4
# ---------------------------------------------------------------------------
def test_full_run_exit_zero_and_counts(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0

    rows = _read_rows(os.path.join(out, "seeds.csv"))
    assert len(rows) == FIXTURE_TOTAL
    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    assert by_source == {"AgentHarm": FIXTURE_AH_COUNT, "PKU-SafeRLHF": FIXTURE_PKU_COUNT}

    assert [r["seed_group_id"] for r in rows] == [
        "SG-%04d" % i for i in range(1, FIXTURE_TOTAL + 1)
    ]
    assert rows[0]["seed_id"] == "SEED-AH-001"
    assert rows[-1]["seed_id"] == "SEED-PKU-%03d" % FIXTURE_PKU_COUNT


def test_b1_b2_b3_b4_all_pass_on_fixture(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    meta = json.load(open(os.path.join(out, "seeds.meta.json"), encoding="utf-8"))

    assert meta["assertions_all_passed"] is True
    for key in ("B1", "B2", "B3", "B4"):
        assert meta["assertions"][key]["passed"] is True, key
    assert meta["seed_count_total"] == FIXTURE_TOTAL
    assert meta["seed_count_by_source"]["AgentHarm"]["count"] == FIXTURE_AH_COUNT
    assert meta["seed_count_by_source"]["PKU-SafeRLHF"]["count"] == FIXTURE_PKU_COUNT
    assert meta["non_deterministic_fields"] == ["generated_at"]


def test_assertions_helper_passes_on_constructed_rows(mod):
    """run_assertions 在构造数据上的正例（不读真实来源）。"""
    rows = [
        _row(mod, "SEED-AH-001", "SG-0001", AH_RAW_FILE),
        _row(mod, "SEED-AH-002", "SG-0002", AH_RAW_FILE),
        _row(mod, "SEED-PKU-001", "SG-0003", PKU_RAW_FILE),
    ]
    res = mod.run_assertions(rows, FIXTURE_RAW, {})
    assert all(res[k]["passed"] for k in ("B1", "B2", "B3", "B4")), res


# ---------------------------------------------------------------------------
# 1'. 反例：全部用构造数据，不得使用真实来源
# ---------------------------------------------------------------------------
def test_b1_negative_duplicate_group_id(mod):
    rows = [
        _row(mod, "SEED-AH-001", "SG-0001", AH_RAW_FILE),
        _row(mod, "SEED-AH-002", "SG-0001", AH_RAW_FILE),  # 重复 group
        _row(mod, "SEED-PKU-001", "SG-0003", PKU_RAW_FILE),
    ]
    res = mod.run_assertions(rows, FIXTURE_RAW, {})
    assert res["B1"]["passed"] is False
    assert res["B1"]["duplicated"] == ["SG-0001"]


def test_b2_negative_seed_id_spans_two_groups(mod):
    """单射性反例：同一 seed_id 出现在两个 group（B1 仍成立）。"""
    rows = [
        _row(mod, "SEED-AH-001", "SG-0001", AH_RAW_FILE),
        _row(mod, "SEED-AH-001", "SG-0002", AH_RAW_FILE),
        _row(mod, "SEED-PKU-001", "SG-0003", PKU_RAW_FILE),
    ]
    res = mod.run_assertions(rows, FIXTURE_RAW, {})
    assert res["B1"]["passed"] is True           # group 本身不重复
    assert res["B2"]["passed"] is False
    assert res["B2"]["multi_group_seed_ids"] == ["SEED-AH-001"]


def test_b3_negative_hole_in_group_ids(mod):
    rows = [
        _row(mod, "SEED-AH-001", "SG-0001", AH_RAW_FILE),
        _row(mod, "SEED-AH-002", "SG-0003", AH_RAW_FILE),  # 缺 SG-0002
    ]
    res = mod.run_assertions(rows, FIXTURE_RAW, {})
    assert res["B3"]["passed"] is False
    assert res["B3"]["has_hole"] is True


def test_b3_negative_wrong_start(mod):
    rows = [
        _row(mod, "SEED-AH-001", "SG-0002", AH_RAW_FILE),
        _row(mod, "SEED-AH-002", "SG-0003", AH_RAW_FILE),
    ]
    res = mod.run_assertions(rows, FIXTURE_RAW, {})
    assert res["B3"]["passed"] is False
    assert res["B3"]["actual_first"] == "SG-0002"


def test_b4_negative_sha_mismatch(mod):
    rows = [
        _row(mod, "SEED-AH-001", "SG-0001", AH_RAW_FILE, sha="0" * 64),
    ]
    res = mod.run_assertions(rows, FIXTURE_RAW, {})
    assert res["B4"]["passed"] is False
    assert res["B4"]["mismatched"][0]["raw_file"] == AH_RAW_FILE


def test_b4_negative_manifest_disagrees_with_file(mod):
    """manifest 与实算 sha256 不符 ⇒ B4 不通过（B4 的交叉核对必须真的生效）。"""
    real = json.load(
        open(os.path.join(FIXTURE_RAW, "AgentHarm", "SOURCE_MANIFEST.json"), encoding="utf-8")
    )
    bad = json.loads(json.dumps(real))
    bad["files"][0]["sha256"] = "f" * 64
    bad["_dir"] = "AgentHarm"

    rows = [_row(mod, "SEED-AH-001", "SG-0001", AH_RAW_FILE)]
    res = mod.run_assertions(rows, FIXTURE_RAW, {"AgentHarm": bad})
    assert res["B4"]["passed"] is False
    assert res["B4"]["manifest_mismatched"][0]["path"] == AH_RAW_FILE


def test_b4_negative_missing_file(mod):
    rows = [
        _row(mod, "SEED-AH-001", "SG-0001", "AgentHarm/benchmark/does_not_exist.json", sha="a" * 64)
    ]
    res = mod.run_assertions(rows, FIXTURE_RAW, {})
    assert res["B4"]["passed"] is False
    assert res["B4"]["mismatched"][0]["error"] == "file_missing"


def test_nonzero_exit_when_assertions_fail(mod, tmp_path, monkeypatch):
    """断言不通过必须非零退出（不得静默产出）。"""
    real = mod.run_assertions

    def fake(rows, raw_root, manifests):
        res = real(rows, raw_root, manifests)
        res["B3"]["passed"] = False
        return res

    monkeypatch.setattr(mod, "run_assertions", fake)
    out = str(tmp_path / "out")
    assert _run(mod, out) != 0


# ---------------------------------------------------------------------------
# 2. --smoke N 不覆盖全量产物
# ---------------------------------------------------------------------------
def test_smoke_does_not_overwrite_full_output(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out, smoke=0) == 0
    full_csv = os.path.join(out, "seeds.csv")
    assert os.path.exists(full_csv)
    full_bytes = open(full_csv, "rb").read()
    full_meta = open(os.path.join(out, "seeds.meta.json"), "rb").read()

    assert _run(mod, out, smoke=2) == 0

    smoke_csv = os.path.join(out, "smoke", "seeds.csv")
    assert os.path.exists(smoke_csv), "冒烟产物必须落在 <out>/smoke/"

    # 全量产物逐字节未变
    assert open(full_csv, "rb").read() == full_bytes
    assert open(os.path.join(out, "seeds.meta.json"), "rb").read() == full_meta

    smoke_rows = _read_rows(smoke_csv)
    assert len(smoke_rows) == 2 * 2  # 每来源 2 条
    assert [r["seed_group_id"] for r in smoke_rows] == ["SG-0001", "SG-0002", "SG-0003", "SG-0004"]
    assert len(_read_rows(full_csv)) == FIXTURE_TOTAL


def test_smoke_is_a_prefix_of_full(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    assert _run(mod, out, smoke=2) == 0
    full = {r["seed_id"]: r for r in _read_rows(os.path.join(out, "seeds.csv"))}
    for r in _read_rows(os.path.join(out, "smoke", "seeds.csv")):
        assert r["seed_id"] in full
        assert r["raw_text"] == full[r["seed_id"]]["raw_text"]


# ---------------------------------------------------------------------------
# 3. 编码与行尾
# ---------------------------------------------------------------------------
def test_output_is_utf8_without_cr(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    for name in ("seeds.csv", "seeds.meta.json"):
        path = os.path.join(out, name)
        data = open(path, "rb").read()
        assert data.count(b"\r") == 0, "%s 含 CR" % name
        data.decode("utf-8")  # 非法 UTF-8 会抛异常
    assert mod.assert_no_cr(os.path.join(out, "seeds.csv")) is True


def test_csv_quoting_roundtrip(mod, tmp_path):
    """fixture 的 AgentHarm 首条提示含逗号、双引号与换行 ⇒ 必须由 CSV 引号规则无损承载。"""
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    rows = _read_rows(os.path.join(out, "seeds.csv"))
    first = rows[0]
    assert "," in first["raw_text"]
    assert '"' in first["raw_text"]
    assert "\n" in first["raw_text"]
    src = json.load(
        open(os.path.join(FIXTURE_RAW, "AgentHarm",
                          "benchmark", "harmful_behaviors_test_public.json"), encoding="utf-8")
    )["behaviors"][0]["prompt"]
    assert first["raw_text"] == src  # 逐字符无损


def test_raw_text_is_never_truncated(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    rows = _read_rows(os.path.join(out, "seeds.csv"))
    assert all(r["raw_text"].strip() for r in rows)


# ---------------------------------------------------------------------------
# 4. 幂等性
# ---------------------------------------------------------------------------
def test_seeds_csv_is_byte_identical_across_runs(mod, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert _run(mod, a) == 0
    assert _run(mod, b) == 0
    assert open(os.path.join(a, "seeds.csv"), "rb").read() == \
           open(os.path.join(b, "seeds.csv"), "rb").read()


def test_meta_is_identical_except_generated_at(mod, tmp_path):
    """除 generated_at 外，meta 必须逐字段相同。

    ``raw_acquisition.path`` 是**指向 --out 的展示字符串**，两次运行用了不同 out 目录，
    故按定义会不同（非确定性来源只有 generated_at 一个）—— 比对前把它归一化掉。
    """
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert _run(mod, a) == 0
    assert _run(mod, b) == 0
    ma = json.load(open(os.path.join(a, "seeds.meta.json"), encoding="utf-8"))
    mb = json.load(open(os.path.join(b, "seeds.meta.json"), encoding="utf-8"))
    assert ma["generated_at"] != "" and mb["generated_at"] != ""

    for meta in (ma, mb):
        meta.pop("generated_at")
        meta["raw_acquisition"]["path"] = "<out>/raw_acquisition.json"
    assert ma == mb


# ---------------------------------------------------------------------------
# 5. 离线：模块级不得引入任何网络库
# ---------------------------------------------------------------------------
def test_module_has_no_network_imports():
    tree = ast.parse(open(SCRIPT, "r", encoding="utf-8").read())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                roots.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    forbidden = {"requests", "urllib", "urllib3", "httpx", "socket", "http",
                 "huggingface_hub", "modelscope", "datasets", "aiohttp"}
    assert not (roots & forbidden), "种子收集脚本不得联网：%s" % sorted(roots & forbidden)


def test_fixtures_are_offline(mod, tmp_path):
    """整条链路只用 tests/data/fixtures/ 即可跑通（不触碰 data/raw）。"""
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    meta = json.load(open(os.path.join(out, "seeds.meta.json"), encoding="utf-8"))
    rows = _read_rows(os.path.join(out, "seeds.csv"))
    assert {r["raw_file"] for r in rows} == {AH_RAW_FILE, PKU_RAW_FILE}
    # B4 已核对 fixture 内的两个文件
    assert meta["assertions"]["B4"]["files_checked"] == 2


# ---------------------------------------------------------------------------
# 6. PKU 侧三条子规则：阈值 / 去重 / 确定性抽样
# ---------------------------------------------------------------------------
def test_pku_threshold_dedup_and_sample(mod, tmp_path):
    import random

    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    rows = [r for r in _read_rows(os.path.join(out, "seeds.csv")) if r["source"] == "PKU-SafeRLHF"]

    # 只取 max(sev0, sev1) == 3 ⇒ severity 列恒为 "3"
    assert {r["severity"] for r in rows} == {"3"}

    # fixture 中 prompt 的候选（按首次出现去重）共 5 条：
    #   PKU-1(L1) PKU-2(L2) PKU-3(L3) PKU-4(L6) PKU-5(L7)
    #   —— L4 是 PKU-1 的重复（须被去重）；L5 的 max=2（须被阈值排除）
    candidates = [
        "FIXTURE-PKU-1 prompt", "FIXTURE-PKU-2 prompt", "FIXTURE-PKU-3 prompt",
        "FIXTURE-PKU-4 prompt", "FIXTURE-PKU-5 prompt",
    ]
    expected = sorted(random.Random(42).sample(candidates, 2),
                      key=candidates.index)
    assert [r["raw_text"] for r in rows] == expected

    # 抽样后按文件序排序 ⇒ source_item_id 的行号升序
    linenos = [int(r["source_item_id"].split("#L")[1]) for r in rows]
    assert linenos == sorted(linenos)
    assert all(r["source_item_id"].startswith("Alpaca-7B/train.jsonl#L") for r in rows)


def test_pku_harm_category_is_first_true_in_source_key_order(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    rows = {r["seed_id"]: r for r in _read_rows(os.path.join(out, "seeds.csv"))}
    # fixture 的 harm_category 键序为 Cybercrime, Violence, Drugs, Privacy Violation
    # L3: response_0 = {Cybercrime: T}, response_1 = {Drugs: T} ⇒ 取 Cybercrime
    # L2: 仅 response_1 的 severity==3，其 {Violence: T} ⇒ 取 Violence
    got = {r["raw_text"]: r["harm_category"] for r in rows.values()
           if r["source"] == "PKU-SafeRLHF"}
    for text, cat in got.items():
        assert cat in {"Cybercrime", "Violence", "Drugs", "Privacy Violation"}
    if "FIXTURE-PKU-3 prompt" in got:
        assert got["FIXTURE-PKU-3 prompt"] == "Cybercrime"
    if "FIXTURE-PKU-2 prompt" in got:
        assert got["FIXTURE-PKU-2 prompt"] == "Violence"


def test_agentharm_severity_is_empty_and_category_mapped(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    rows = [r for r in _read_rows(os.path.join(out, "seeds.csv")) if r["source"] == "AgentHarm"]
    assert {r["severity"] for r in rows} == {""}       # §3.1：AgentHarm 无严重度字段 ⇒ 空串
    assert [r["harm_category"] for r in rows] == \
        ["Fraud", "Fraud", "Cybercrime", "Drugs", "Hate", "Sexual"]
    assert [r["source_item_id"] for r in rows] == ["1-1", "1-2", "2-1", "3-1", "4-1", "5-1"]


def test_agentharm_expected_count_mismatch_raises(mod, tmp_path):
    """来源条数与配置不符 ⇒ 报错而非静默裁剪（§2.1 的硬校验）。"""
    cfg = mod.load_config(FIXTURE_CFG)
    cfg["sources"][0]["subset_rule"]["expected_count"] = 999
    with pytest.raises(ValueError) as ei:
        mod.load_agentharm(FIXTURE_RAW, cfg["sources"][0])
    assert "expected_count" in str(ei.value)


def test_pku_candidate_count_mismatch_raises(mod):
    cfg = mod.load_config(FIXTURE_CFG)
    src = cfg["sources"][1]
    src["subset_rule"]["candidate_count_expected"] = 999
    with pytest.raises(ValueError) as ei:
        mod.load_pku(FIXTURE_RAW, src)
    assert "candidate_count_expected" in str(ei.value)


def test_pku_sample_larger_than_candidates_raises(mod):
    cfg = mod.load_config(FIXTURE_CFG)
    src = cfg["sources"][1]
    src["subset_rule"]["sample"]["n"] = 999
    with pytest.raises(ValueError) as ei:
        mod.load_pku(FIXTURE_RAW, src)
    assert "超过候选数" in str(ei.value)


# ---------------------------------------------------------------------------
# 7. 接口面
# ---------------------------------------------------------------------------
def test_help_exits_zero(mod):
    with pytest.raises(SystemExit) as ei:
        mod.build_parser().parse_args(["--help"])
    assert ei.value.code == 0


def test_argparse_defaults_match_contract(mod):
    args = mod.build_parser().parse_args([])
    assert args.config == "configs/data/seed_collect.yaml"
    assert args.out == "data/interim/seed_collect/"
    assert args.raw_root == "data/raw"
    assert args.smoke == 0


def test_meta_records_config_sha256_and_raw_acquisition_pointer(mod, tmp_path):
    out = str(tmp_path / "out")
    assert _run(mod, out) == 0
    meta = json.load(open(os.path.join(out, "seeds.meta.json"), encoding="utf-8"))
    assert meta["config_sha256"] == mod.sha256_file(FIXTURE_CFG)
    assert "raw_acquisition" in meta
    assert meta["seeds_csv_no_CR"] is True
    assert meta["seeds_csv_sha256"] == mod.sha256_file(os.path.join(out, "seeds.csv"))

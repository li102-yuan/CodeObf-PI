# -*- coding: utf-8 -*-
"""tests/env/test_01_00_smoke_env.py — ``scripts/env/01_00_smoke_env.py`` 的单元测试

覆盖 ``windows/W2.3_determinism.md`` §6.3 的第 3 项：

3. 七元组行解析／序列化 —— 字段齐全性校验，缺字段时报错（§4.6 末句：
   「缺少任一字段的显存数字不得进入 ``ENV_LOCK.md`` §8」）。

另覆盖本脚本在 W2.3 侧的接口面：套件解析（档位名／族名／all／无参最小档）、
内置默认与 ``windows/W2.4_memory_baseline.md`` §4.2 的**逐项一致**、
``memory_probe.yaml`` 缺失时的 WARN 与内置回退、加载方式 → device／offload／collector 映射。

**本文件由 W2.3 建立；W2.4 在其内追加显存记录相关用例**（七元组完整性、§8 分列渲染、
OOM 记录结构），依 ``windows/W2.3_determinism.md`` §6.1「W2.4 的追加约定」。
本窗口**不预先替 W2.4 写**这些用例。

约束：不加载模型、不访问网络、不访问 ``data/``、不改变环境。
被测脚本以文件路径动态加载（文件名以数字开头，不能用 ``import`` 语句）。
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "env", "01_00_smoke_env.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _module_level_imports(path):
    """AST 判定模块级（顶层）import 的**顶层包名**集合——不含函数内 import。"""
    import ast

    tree = ast.parse(open(path, "r", encoding="utf-8").read())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return found


M = _load("mod_01_00_smoke_env", SCRIPT)


# ---------------------------------------------------------------- 七元组
def _full(**over):
    rec = {
        "suite": "infer-1b-512", "model": "llamaguard3-1b", "load": "4bit-nf4",
        "device": "cuda", "batch": 1, "seq_len": 512, "offload": "none",
        "collector": "max_memory_allocated", "peak": 1510, "reserved": 1522,
        "repeat": 3, "run_id": "01_00_infer-1b-512_20260930-000000-1234",
    }
    rec.update(over)
    return rec


def test_field_tuple_is_fixed_and_ordered():
    assert M.SEVEN_TUPLE_FIELDS == (
        "suite", "model", "load", "device", "batch", "seq_len",
        "offload", "collector", "peak", "reserved", "repeat", "run_id",
    )


def test_serialize_matches_contract_layout():
    line = M.format_mem_probe_line(_full())
    assert line.startswith("MEM_PROBE ")
    positions = [line.find("%s=" % f) for f in M.SEVEN_TUPLE_FIELDS]
    assert all(p > 0 for p in positions), line
    assert positions == sorted(positions), "字段顺序与 SEVEN_TUPLE_FIELDS 不一致"
    assert "peak=1510MiB" in line
    assert "reserved=1522MiB" in line
    assert "offload=none" in line
    assert "collector=max_memory_allocated" in line


@pytest.mark.parametrize("field", [
    "suite", "model", "load", "device", "batch", "seq_len",
    "offload", "collector", "peak", "reserved", "repeat", "run_id",
])
def test_missing_field_is_rejected(field):
    """§4.6 末句／W2.4 §6.3-1 回归防护：缺任一字段即被拒。"""
    rec = _full()
    del rec[field]
    assert field in M.validate_mem_probe_fields(rec)
    with pytest.raises(ValueError):
        M.format_mem_probe_line(rec)


def test_empty_and_none_values_count_as_missing():
    assert "peak" in M.validate_mem_probe_fields(_full(peak=None))
    assert "run_id" in M.validate_mem_probe_fields(_full(run_id=""))


def test_complete_record_has_no_missing_fields():
    assert M.validate_mem_probe_fields(_full()) == []


def test_parse_round_trip():
    line = M.format_mem_probe_line(_full())
    parsed = M.parse_mem_probe_line(line)
    assert set(parsed) == set(M.SEVEN_TUPLE_FIELDS)
    assert parsed["suite"] == "infer-1b-512"
    assert parsed["peak"] == "1510"
    assert parsed["reserved"] == "1522"
    assert parsed["collector"] == "max_memory_allocated"


def test_parse_rejects_non_mem_probe_line():
    with pytest.raises(ValueError):
        M.parse_mem_probe_line("suite=x model=y")


def test_parse_rejects_line_with_missing_field():
    line = M.format_mem_probe_line(_full())
    broken = line.replace(" repeat=3", "")
    with pytest.raises(ValueError):
        M.parse_mem_probe_line(broken)


def test_parse_rejects_unknown_field():
    line = M.format_mem_probe_line(_full()) + " extra=1"
    with pytest.raises(ValueError):
        M.parse_mem_probe_line(line)


# ---------------------------------------------------------------- 映射
@pytest.mark.parametrize("load,device,offload", [
    ("4bit-nf4", "cuda", "none"),
    ("4bit-nf4-offload-cpu", "cuda", "cpu"),
    ("4bit-nf4-cpu", "cpu", "none"),
    ("cpu-fp32", "cpu", "none"),
    ("bf16", "cuda", "none"),
    ("fp32", "cuda", "none"),
])
def test_load_mode_mapping(load, device, offload):
    assert M.device_of(load) == device
    assert M.offload_of(load) == offload


def test_collector_mapping():
    assert M.collector_of("cuda") == "max_memory_allocated"
    assert M.collector_of("cpu") == "nvidia-smi"


# ---------------------------------------------------------------- 套件定义
def test_builtin_suites_match_w24_contract():
    """内置默认必须与 `windows/W2.4_memory_baseline.md` §4.2 **逐项一致**。"""
    by_name = {s["name"]: s for s in M.BUILTIN_SUITES}
    assert list(by_name) == [
        "idle", "load-1b", "infer-1b-512", "load-8b", "infer-8b-256",
        "infer-8b-512", "train-1b-10step-b1", "train-1b-10step-b4", "unplug-cpu",
    ]
    assert by_name["idle"]["kind"] == "baseline"
    assert by_name["load-1b"] == {"name": "load-1b", "kind": "load",
                                  "model_key": "llamaguard3-1b", "load": "4bit-nf4"}
    assert by_name["infer-1b-512"]["batch"] == 1
    assert by_name["infer-1b-512"]["seq_len"] == 512
    assert by_name["infer-8b-256"]["seq_len"] == 256
    assert by_name["infer-8b-512"]["seq_len"] == 512
    assert by_name["unplug-cpu"]["load"] == "cpu-fp32"
    for name in ("train-1b-10step-b1", "train-1b-10step-b4"):
        spec = by_name[name]
        assert spec["grad_accum"] == 4
        assert spec["seq_len"] == 512
        assert spec["steps"] == 10
        assert spec["lora"] == {"r": 16, "alpha": 32,
                                "target_modules": ["q_proj", "v_proj"]}
    assert by_name["train-1b-10step-b1"]["batch"] == 1
    assert by_name["train-1b-10step-b4"]["batch"] == 4


def test_builtin_fallback_suites_match_w24_contract():
    by_name = {s["name"]: s for s in M.BUILTIN_FALLBACK_SUITES}
    assert list(by_name) == ["infer-8b-256-offload", "train-1b-10step-b1-offload"]
    assert by_name["infer-8b-256-offload"]["load"] == "4bit-nf4-offload-cpu"
    assert by_name["infer-8b-256-offload"]["seq_len"] == 256
    assert by_name["train-1b-10step-b1-offload"]["load"] == "4bit-nf4-offload-cpu"
    assert by_name["train-1b-10step-b1-offload"]["grad_accum"] == 4


def test_train_probe_steps_never_exceed_cap():
    """W2.3 §4.8：训练探针 **≤10 步**。"""
    assert M.TRAIN_MAX_STEPS == 10
    for spec in M.BUILTIN_SUITES + M.BUILTIN_FALLBACK_SUITES:
        if spec.get("kind") == "train":
            assert spec["steps"] <= M.TRAIN_MAX_STEPS


def test_train_probe_hyperparams_come_from_v5_l434():
    assert (M.TRAIN_LR, M.TRAIN_MAX_LEN) == (2e-4, 512)
    assert M.BUILTIN_LORA == {"r": 16, "alpha": 32,
                              "target_modules": ["q_proj", "v_proj"]}


def test_threshold_constant_present():
    assert M.BUILTIN_THRESHOLDS["headroom_warn_mib"] == 1024


# ---------------------------------------------------------------- 套件解析
def _defs():
    return M.load_suite_definitions(os.path.join(ROOT, "configs", "env", "__absent__.yaml"))


def test_missing_memory_probe_warns_and_falls_back(capsys):
    defs = _defs()
    assert defs["source"] == "builtin"
    out = capsys.readouterr().out
    assert "WARN: memory_probe.yaml not found, using builtin defaults" in out


def test_no_arg_selects_minimal_suites():
    names, mode = M.resolve_suite_names(None, _defs())
    assert mode == "minimal"
    assert names == M.MINIMAL_SUITES == [
        "load-1b", "infer-1b-512", "train-1b-10step-b1", "unplug-cpu"]


def test_all_selects_every_main_suite_without_fallbacks():
    names, mode = M.resolve_suite_names("all", _defs())
    assert mode == "all"
    assert names == [s["name"] for s in M.BUILTIN_SUITES]
    assert "infer-8b-256-offload" not in names


def test_exact_suite_name_selects_exactly_one():
    names, mode = M.resolve_suite_names("infer-8b-256", _defs())
    assert (names, mode) == (["infer-8b-256"], "single")
    names, mode = M.resolve_suite_names("train-1b-10step-b4", _defs())
    assert (names, mode) == (["train-1b-10step-b4"], "single")


def test_fallback_suite_name_is_addressable():
    names, mode = M.resolve_suite_names("infer-8b-256-offload", _defs())
    assert (names, mode) == (["infer-8b-256-offload"], "single")


@pytest.mark.parametrize("family,expected", [
    ("infer-1b", ["infer-1b-512"]),
    ("infer-8b", ["infer-8b-256", "infer-8b-512"]),
    ("train-1b-10step", ["train-1b-10step-b1", "train-1b-10step-b4"]),
])
def test_contract_family_names_resolve(family, expected):
    names, mode = M.resolve_suite_names(family, _defs())
    assert mode == "family"
    assert names == expected


@pytest.mark.parametrize("name", ["load-1b", "load-8b", "unplug-cpu"])
def test_exact_name_wins_over_shadowing_family_name(name):
    """`load-1b` 等兼为族名与档位名 ⇒ 按**档位名**解析，结果仍恰为该一档。"""
    names, mode = M.resolve_suite_names(name, _defs())
    assert (names, mode) == ([name], "single")


def test_unknown_suite_raises():
    with pytest.raises(ValueError):
        M.resolve_suite_names("no-such-suite", _defs())


def test_find_suite_returns_spec_and_unknown_raises():
    spec = M.find_suite("infer-8b-512", _defs())
    assert spec["model_key"] == "llamaguard3-8b"
    with pytest.raises(KeyError):
        M.find_suite("nope", _defs())


def test_isolated_run_uses_one_suite_per_process():
    """W2.4 §4.1-1 的隔离要求靠「每档一次独立调用」实现——档位名必须可单独寻址。"""
    defs = _defs()
    for spec in M.BUILTIN_SUITES:
        names, mode = M.resolve_suite_names(spec["name"], defs)
        assert mode == "single"
        assert names == [spec["name"]]


# ---------------------------------------------------------------- 模型解析
def test_manifest_keys_cover_w23_subjects():
    manifest = M.load_manifest(os.path.join(ROOT, "configs", "env", "models.yaml"))
    for key in ("llamaguard3-1b", "llamaguard3-8b", "unplug-tiny"):
        assert key in manifest
        assert manifest[key]["local_status"] == "downloaded"
        assert manifest[key]["local_sha_verified"] is True


def test_resolve_model_path_rejects_unknown_key():
    manifest = M.load_manifest(os.path.join(ROOT, "configs", "env", "models.yaml"))
    with pytest.raises(KeyError):
        M.resolve_model_path("no-such-model", manifest)


def test_resolve_model_path_rejects_not_downloaded():
    manifest = {"ghost": {"local_path": None, "local_status": "hash_only"}}
    with pytest.raises(FileNotFoundError):
        M.resolve_model_path("ghost", manifest)


def test_resolve_model_path_returns_existing_dir():
    manifest = M.load_manifest(os.path.join(ROOT, "configs", "env", "models.yaml"))
    path, entry = M.resolve_model_path("unplug-tiny", manifest)
    assert os.path.isdir(path)
    assert entry["revision"] in path or entry["local_path"].startswith("models/")


# ---------------------------------------------------------------- 设备判定
def test_needs_torch_for_every_suite_kind():
    """`idle`（kind=baseline）也要 torch——其定义就是「仅 import torch 后采集」。"""
    defs = _defs()
    assert M._needs_torch(["idle"], defs) is True
    assert M._needs_torch(["unplug-cpu"], defs) is True
    assert M._needs_torch(["load-1b"], defs) is True
    assert M._needs_torch([], defs) is False


def test_has_cuda_suite_detects_cpu_only_and_cuda_selection():
    defs = _defs()
    assert M._has_cuda_suite(["unplug-cpu"], defs) is False
    assert M._has_cuda_suite(["unplug-cpu", "load-1b"], defs) is True
    assert M._has_cuda_suite(["load-8b"], defs) is True


# ---------------------------------------------------------------- eval 断言
class _FakeModel(object):
    def __init__(self, training):
        self.training = training


def test_eval_assert_raises_in_train_mode():
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            M.eval_assert(_FakeModel(True))


def test_eval_assert_raises_outside_no_grad():
    torch = pytest.importorskip("torch")
    assert torch.is_grad_enabled() is True
    with pytest.raises(RuntimeError):
        M.eval_assert(_FakeModel(False))


def test_eval_assert_passes_in_eval_with_no_grad():
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        assert M.eval_assert(_FakeModel(False)) is True


# ---------------------------------------------------------------- 脚本卫生
def test_no_module_level_heavy_import():
    """模块级不得 import 重型库（否则 CUBLAS_WORKSPACE_CONFIG 的「import 前置」失效）。

    用 AST 判定**可执行的模块级 import**，避免被 docstring／注释里的同名字符串误判。
    """
    heavy = _module_level_imports(SCRIPT) & {"torch", "numpy", "transformers",
                                             "bitsandbytes", "peft", "trl"}
    assert heavy == set(), "模块级出现重型导入：%s" % sorted(heavy)


def test_synthetic_texts_never_reference_data_dir():
    """A2 权限：样本与探针文本一律合成，不得引用 data/ 下任何路径。"""
    source = open(SCRIPT, "r", encoding="utf-8").read()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        assert "data/sealed_test" not in stripped
        assert "data/frozen" not in stripped
        assert "data/raw" not in stripped
        assert "data/interim" not in stripped


# ---------------------------------------------------------------- tokenizer 兜底
class _FakeTok(object):
    def __init__(self, pad, eos):
        self.pad_token = pad
        self.eos_token = eos


class _FakeAutoTokenizer(object):
    def __init__(self, tok):
        self._tok = tok

    def from_pretrained(self, path, **kwargs):
        return self._tok


class _FakeTransformers(object):
    def __init__(self, tok):
        self.AutoTokenizer = _FakeAutoTokenizer(tok)


def test_load_tokenizer_fills_missing_pad_token_from_eos():
    """Llama-Guard-3 的 tokenizer 无 pad_token ⇒ 以 eos 兜底，否则 batch>1 会抛错。"""
    tok = M.load_tokenizer(_FakeTransformers(_FakeTok(None, "<eos>")), "x")
    assert tok.pad_token == "<eos>"


def test_load_tokenizer_keeps_existing_pad_token():
    tok = M.load_tokenizer(_FakeTransformers(_FakeTok("<pad>", "<eos>")), "x")
    assert tok.pad_token == "<pad>"


def test_load_tokenizer_raises_when_neither_pad_nor_eos():
    with pytest.raises(ValueError):
        M.load_tokenizer(_FakeTransformers(_FakeTok(None, None)), "x")


def test_preflight_threshold_is_set():
    assert M.PREFLIGHT_WARN_MIB == 1500


def test_parser_defaults_follow_contract():
    parser = M.build_parser()
    args = parser.parse_args([])
    assert args.suite is None
    assert args.repeat == 3
    assert args.json_out is None
    assert args.seq_len is None
    assert args.batch is None


# ============================================================================
# W2.4 追加段 —— 显存记录（`windows/W2.4_memory_baseline.md` §6.3 四项）
#
# 本段由 **W2.4** 追加（该文件由 W2.3 建立，依 `windows/W2.3_determinism.md` §6.1
# 「W2.4 的追加约定」）。**W2.3 已写用例一字未改**；本段只做追加。
#
# 为什么渲染逻辑落在测试文件内：W2.4 §5-2 **禁止新建 `scripts/env/` 下任何脚本**，
# 而 §6.3-3 要求存在可被单测直接调用的「§8 分列渲染」函数、§6.3-4 要求可被单测直接调用的
# 「OOM 记录结构」判定。故本段既是**被测对象**，也是 `ENV_LOCK.md` §8 与基线报告的
# **唯一渲染器**（报告数字由 `logs/env/tmp/memory_*.json` 经本段函数生成，**禁止手工抄录**，
# 对应 W2.4 §5-7）。纯函数、不加载模型、不访问网络、不访问 `data/`。
# ============================================================================

#: `ENV_LOCK.md` §8 的行列（W2.4 §4.3 十二列，＋ `repeat` 为第十三列）。
#: 末列的存在理由：§4.3 首句列出的十二列里含「极差」而**不含** `repeat`，但同节末句要求
#: 「缺七元组任一项的行不得写入 §8」——七元组含 `repeat`。两处须同时满足 ⇒ 保留第十二列
#: 「极差」之外，另以第十三列显式携带 `repeat`（`§8` 前言已注明该列）。
SECTION8_COLUMNS = (
    "suite", "model", "load", "device", "batch", "seq_len", "offload",
    "collector", "peak", "reserved", "range", "run_id", "repeat",
)

#: `kind` → `ENV_LOCK.md` §8 的分节（W2.4 §4.3：**推理与微调分列，禁止合并**）
SECTION8_BY_KIND = {
    "infer": "8.1",      # 纯推理显存峰值（inference peak）
    "train": "8.2",      # QLoRA 微调显存峰值（train peak）
    "load": "8.3",       # 加载态静态占用
    "baseline": "8.4",   # 系统 idle 参照
}

SECTION8_TITLES = {
    "8.1": "§8.1 纯推理显存峰值（inference peak）",
    "8.2": "§8.2 QLoRA 微调显存峰值（train peak）",
    "8.3": "§8.3 加载态静态占用",
    "8.4": "§8.4 系统 idle 参照",
}

#: §8.5 不可比性声明（W2.4 §4.3；**本节必写、不得省略**）
SECTION8_INCOMPARABLE_STATEMENT = (
    "本节数字仅在与所载七元组完全一致时可比；加载方式不同、seq_len 不同、"
    "是否 offload 不同、采集工具不同者，一律不得相减或合并。"
)

#: `kind` 取值词表（`memory_probe.yaml` 校验用）
KINDS = ("baseline", "load", "infer", "train")

#: OOM 判定用的 traceback 特征串（PyTorch 的 `torch.OutOfMemoryError` 及其 CUDA 文案）
_OOM_MARKERS = ("OutOfMemoryError", "out of memory", "CUDA out of memory")


def classify_status(rec):
    """把 `01_00` 的记录状态归为 `OK` / `OOM` / `FAIL`。

    `01_00` 只在异常时写 `status="FAIL"` 并把 traceback 存入 `rec["traceback"]`，
    不区分 OOM。§6.3-4 要求存在 `status: OOM` 的行 ⇒ 本函数按 traceback 特征串分类。
    `status` 缺省且**无** traceback 者按 `OK` 计（`01_00` 的语义是「有异常才写 traceback」）。
    **分类只影响呈现与判定路径，不改变 `01_00` 已落盘的原始记录**。
    """
    tb = str(rec.get("traceback") or "")
    if any(marker in tb for marker in _OOM_MARKERS):
        return "OOM"
    if tb:
        return "FAIL"
    return "OK"


def row_from_record(rec, peak_range=None):
    """把一条原始记录投影为 §8 行 dict（列名见 `SECTION8_COLUMNS`）。

    OOM 行**不携带任何数字**：`peak` / `reserved` / `range` 一律置 `"-"`，并强制带上
    `fallback_used` 字段——这是「**OOM 档与回退档数字不得互相顶替**」（§4.1-5）的落地方式。
    """
    status = classify_status(rec)
    row = {
        "suite": rec.get("suite"), "model": rec.get("model"), "load": rec.get("load"),
        "device": rec.get("device"), "batch": rec.get("batch"), "seq_len": rec.get("seq_len"),
        "offload": rec.get("offload"), "collector": rec.get("collector"),
        "run_id": rec.get("run_id"), "repeat": rec.get("repeat"), "status": status,
        "kind": rec.get("kind"),
    }
    if status == "OOM":
        row.update({"peak": "-", "reserved": "-", "range": "-",
                    "fallback_used": rec.get("fallback_used") or "见报告 OOM 记录"})
    else:
        row.update({"peak": rec.get("peak"), "reserved": rec.get("reserved"),
                    "range": peak_range if peak_range is not None else "-"})
    return row


def validate_section8_row(row):
    """返回缺失／为空的列名列表；空列表＝该行可写入 §8（§4.3 末句：缺七元组任一项不得写入）。

    校验的是**七元组字段**（取自 `01_00` 的 `SEVEN_TUPLE_FIELDS`，单一来源），并额外要求
    `range`（极差，§4.3 第十二列）非空。空值判定为 `None`／`""`；**字面 `-` 是 `01_00` 的
    占位符**（`kind=baseline` 的 `idle` 档本就无 `model`／`load`），与七元组行 `parse` 的回环
    一致，故**接受**——它表示「该维度对本档不适用」，但仍占位、不得省略。
    """
    missing = [f for f in M.SEVEN_TUPLE_FIELDS if row.get(f) in (None, "")]
    if row.get("range") in (None, ""):
        missing.append("range")
    if row.get("status") == "OOM" and not row.get("fallback_used"):
        missing.append("fallback_used")
    return missing


def render_row(row):
    """渲染一行 Markdown 表格行；不完整即抛 `ValueError`（防不完整行进入 §8）。"""
    missing = validate_section8_row(row)
    if missing:
        raise ValueError("§8 行字段缺失：%s（不得写入 ENV_LOCK.md §8）" % ", ".join(missing))
    return "| " + " | ".join(str(row.get(c, "-")) for c in SECTION8_COLUMNS) + " |"


def render_section8(rows):
    """把 §8 行集合按 `kind` **分列渲染**为 `{"8.1": [行…], "8.2": [...], …}`。

    分列规则：`kind=infer → §8.1`、`kind=train → §8.2`、`kind=load → §8.3`、
    `kind=baseline → §8.4`。**推理行与微调行从不同节**——这是 §4.3「禁止合并」的机器化保证。
    未知 `kind` 的行**不落入任何节**并抛错（宁可失败，不得静默归类）。
    """
    out = {"8.1": [], "8.2": [], "8.3": [], "8.4": []}
    for row in rows:
        kind = row.get("kind")
        if kind not in SECTION8_BY_KIND:
            raise ValueError("未知 kind=%r，无法归类到 §8 分节" % (kind,))
        section = SECTION8_BY_KIND[kind]
        out[section].append(render_row(row))
    return out


def validate_probe_yaml(doc, manifest_keys):
    """校验 `configs/env/memory_probe.yaml`；返回问题列表（空列表＝通过）。

    校验项（§6.3-2）：① `suite.name` 在主档＋回退档范围内**唯一**；
    ② `kind` 取值在 `KINDS` 内；③ 凡带 `model_key` 的档位，其 `model_key` 必须在
    `configs/env/models.yaml` 的 key 集合内（**禁止**在 yaml 里自造模型 key）。
    """
    problems = []
    if doc.get("schema") != "codeobf-pi/memory-probe/v1":
        problems.append("schema 不是 codeobf-pi/memory-probe/v1：%r" % (doc.get("schema"),))
    names = []
    for group in ("suites", "fallback_suites"):
        for spec in doc.get(group) or []:
            name = spec.get("name")
            if name in names:
                problems.append("suite.name 重复：%r" % (name,))
            names.append(name)
            if spec.get("kind") not in KINDS:
                problems.append("档位 %r 的 kind 非法：%r" % (name, spec.get("kind")))
            key = spec.get("model_key")
            if key is not None and key not in manifest_keys:
                problems.append("档位 %r 的 model_key %r 不在 models.yaml 内" % (name, key))
    return problems


def load_manifest_keys(path=None):
    path = path or os.path.join(ROOT, "configs", "env", "models.yaml")
    return set(M.load_manifest(path))


# ---------------------------------------------------------------- W2.4 §6.3-1
def test_section8_row_rejects_missing_seven_tuple_field():
    """§4.3 末句／§6.3-1：缺七元组任一项的显存行被拒（回归防护）。"""
    row = row_from_record(_full(kind="infer", run_id="01_00_infer-1b-512_x-1"))
    assert validate_section8_row(row) == []
    render_row(row)
    for field in M.SEVEN_TUPLE_FIELDS:
        broken = dict(row)
        broken.pop(field, None)
        assert field in validate_section8_row(broken), field
        with pytest.raises(ValueError):
            render_row(broken)


def test_section8_row_rejects_empty_values_and_accepts_dash_placeholder():
    row = row_from_record(_full(kind="infer", run_id="01_00_x_1-1"))
    assert "peak" in validate_section8_row(dict(row, peak=None))
    assert "run_id" in validate_section8_row(dict(row, run_id=""))
    assert "collector" in validate_section8_row(dict(row, collector=""))
    # `-` 是 01_00 的占位符（idle 档的 model/load），占位不等于缺失
    assert validate_section8_row(dict(row, model="-", load="-")) == []


def test_section8_row_requires_range_column():
    row = row_from_record(_full(kind="infer", run_id="01_00_x_1-1"))
    assert "range" in validate_section8_row(dict(row, range=None))


# ---------------------------------------------------------------- W2.4 §6.3-2
def test_probe_yaml_is_valid_against_manifest():
    doc = M.load_yaml(os.path.join(ROOT, "configs", "env", "memory_probe.yaml"))
    assert validate_probe_yaml(doc, load_manifest_keys()) == []
    assert doc["gpu_total_mib"] == 8187, "判定基准取 8187（ENV_LOCK §1），不得用 8192/8×1024"
    assert doc["thresholds"]["headroom_warn_mib"] == 1024


def test_probe_yaml_matches_w24_contract_suite_list():
    doc = M.load_yaml(os.path.join(ROOT, "configs", "env", "memory_probe.yaml"))
    assert [s["name"] for s in doc["suites"]] == [s["name"] for s in M.BUILTIN_SUITES]
    assert [s["name"] for s in doc["fallback_suites"]] == \
        [s["name"] for s in M.BUILTIN_FALLBACK_SUITES]
    # 文件与 W2.3 内置默认**逐项一致**（01_00 在文件缺失时才用内置值）
    assert doc["suites"] == M.BUILTIN_SUITES
    assert doc["fallback_suites"] == M.BUILTIN_FALLBACK_SUITES
    assert doc["thresholds"] == M.BUILTIN_THRESHOLDS


def test_probe_yaml_rejects_duplicate_name_and_bad_kind():
    good = {"schema": "codeobf-pi/memory-probe/v1",
            "suites": [{"name": "a", "kind": "infer", "model_key": "llamaguard3-1b"},
                       {"name": "a", "kind": "infer", "model_key": "llamaguard3-1b"}],
            "fallback_suites": [{"name": "b", "kind": "nope"}]}
    problems = validate_probe_yaml(good, {"llamaguard3-1b"})
    assert any("重复" in p for p in problems)
    assert any("kind 非法" in p for p in problems)


def test_probe_yaml_rejects_unknown_model_key():
    bad = {"schema": "codeobf-pi/memory-probe/v1",
           "suites": [{"name": "x", "kind": "load", "model_key": "no-such-model"}],
           "fallback_suites": []}
    problems = validate_probe_yaml(bad, {"llamaguard3-1b"})
    assert any("no-such-model" in p for p in problems)


def test_probe_yaml_rejects_wrong_schema():
    problems = validate_probe_yaml({"schema": "something/else"}, set())
    assert any("schema" in p for p in problems)


# ---------------------------------------------------------------- W2.4 §6.3-3
def _mixed_rows():
    return [
        row_from_record(_full(kind="infer", model="llamaguard3-1b", suite="infer-1b-512",
                              run_id="01_00_infer-1b-512_x-1"), peak_range=1),
        row_from_record(_full(kind="train", model="llamaguard3-1b", suite="train-1b-10step-b1",
                              run_id="01_00_train-1b-10step-b1_x-2"), peak_range=0),
        row_from_record(_full(kind="load", suite="load-8b", model="llamaguard3-8b",
                              peak=5542, reserved=5576, run_id="01_00_load-8b_x-3"), peak_range=0),
        row_from_record(_full(kind="baseline", suite="idle", model="-",
                              run_id="01_00_idle_x-4"), peak_range=0),
    ]


def test_section8_rendering_separates_infer_from_train():
    """§6.3-3：混合 kind 的结果集渲染后，**infer 行不得出现在 §8.2**。"""
    out = render_section8(_mixed_rows())
    joined = {k: "\n".join(v) for k, v in out.items()}
    assert "infer-1b-512" in joined["8.1"]
    assert "train-1b-10step-b1" in joined["8.2"]
    assert "load-8b" in joined["8.3"]
    assert "idle" in joined["8.4"]
    assert "infer-1b-512" not in joined["8.2"], "推理行不得进入微调节"
    assert "train-1b-10step-b1" not in joined["8.1"], "微调行不得进入推理节"
    assert "load-8b" not in joined["8.1"]
    assert "idle" not in joined["8.1"]
    # 每行都含完整七元组 + 极差
    for rows in out.values():
        for line in rows:
            assert line.count("|") == len(SECTION8_COLUMNS) + 1


def test_section8_rendering_rejects_unknown_kind():
    row = row_from_record(_full(kind="infer", run_id="01_00_x_1-1"))
    row["kind"] = "mystery"
    with pytest.raises(ValueError):
        render_section8([row])


def test_section8_incomparable_statement_is_present():
    assert "不得相减或合并" in SECTION8_INCOMPARABLE_STATEMENT


def test_section8_section_headers_are_fixed():
    assert set(SECTION8_TITLES) == {"8.1", "8.2", "8.3", "8.4"}
    assert set(SECTION8_BY_KIND.values()) | {"8.5"} == {"8.1", "8.2", "8.3", "8.4", "8.5"}


# ---------------------------------------------------------------- W2.4 §6.3-4
def _oom_records():
    oom = {"suite": "train-1b-10step-b4", "kind": "train", "model": "llamaguard3-1b",
           "load": "4bit-nf4", "device": "cuda", "batch": 4, "seq_len": 512,
           "offload": "none", "collector": "max_memory_allocated", "repeat": 3,
           "run_id": "01_00_train-1b-10step-b4_x-9", "status": "FAIL",
           "traceback": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB"}
    fb = _full(kind="train", suite="train-1b-10step-b1", model="llamaguard3-1b",
               run_id="01_00_train-1b-10step-b1_x-10")
    return oom, fb


def test_classify_status_detects_oom():
    oom, fb = _oom_records()
    assert classify_status(oom) == "OOM"
    assert classify_status(fb) == "OK"
    assert classify_status(dict(fb, status="FAIL", traceback="ValueError: no")) == "FAIL"


@pytest.mark.parametrize("marker", ["torch.OutOfMemoryError: CUDA out of memory.",
                                    "RuntimeError: CUDA out of memory. Tried to allocate",
                                    "OutOfMemoryError"])
def test_classify_status_oom_markers(marker):
    _, fb = _oom_records()
    assert classify_status(dict(fb, status="FAIL", traceback=marker)) == "OOM"


def test_oom_row_carries_fallback_used_and_no_numbers():
    """§6.3-4：`status: OOM` 的行必须带 `fallback_used`，且**不携带任何数字**。"""
    oom, _ = _oom_records()
    row = row_from_record(oom)
    assert row["status"] == "OOM"
    assert row["fallback_used"]
    assert row["peak"] == "-" and row["reserved"] == "-" and row["range"] == "-"
    assert validate_section8_row(row) == []


def test_oom_row_without_fallback_used_is_rejected():
    oom, _ = _oom_records()
    row = row_from_record(oom)
    row["fallback_used"] = ""
    assert "fallback_used" in validate_section8_row(row)
    with pytest.raises(ValueError):
        render_row(row)


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def test_oom_and_fallback_rows_are_not_merged():
    """§4.1-5：OOM 档与回退档数字**不得互相顶替**——两者各自成行，数字归属各自档位。"""
    oom, fb = _oom_records()
    out = render_section8([row_from_record(oom), row_from_record(fb, peak_range=0)])
    lines = out["8.2"]
    assert len(lines) == 2, "OOM 档与回退档必须是两行"
    i_peak = SECTION8_COLUMNS.index("peak")
    oom_line = [l for l in lines if "train-1b-10step-b4" in l][0]
    fb_line = [l for l in lines if _cells(l)[0] == "train-1b-10step-b1"][0]
    assert _cells(oom_line)[i_peak] == "-", oom_line
    assert _cells(fb_line)[i_peak] == str(fb["peak"])
    # 回退档的数字不得出现在 OOM 档行内，反之亦然
    assert str(fb["peak"]) not in oom_line
    assert "train-1b-10step-b4" not in fb_line


def test_ok_row_has_no_fallback_used_key():
    _, fb = _oom_records()
    row = row_from_record(fb, peak_range=0)
    assert row["status"] == "OK"
    assert "fallback_used" not in row

# -*- coding: utf-8 -*-
"""tests/baseline/test_03_03_run_b3_llamaguard1b.py — W4.3 B3 Llama-Guard-3-1B 原始推理。

覆盖 `windows/W4.3_b3_lg1b.md` 与 `windows/W4.6_unify.md` 的语义：

1. **判词解析（§2 验收）**：`safe` ⇒ 0；`unsafe\nS9` ⇒ 1；官方实际形态 `\n\nunsafe\nS1` ⇒ 1
2. **`unparsed` 记数口径**：首行首词 ∉ {safe, unsafe} ⇒ `pred` **留空**（不兜底为 0/1）
3. §0.3 输入链：原文直通 → **左截断 512（保尾部）** → **官方 chat 模板**（`apply_chat_template`）
4. §2 schema：列名/列序、`raw_type=judgment`、`raw`=判词原文 JSON、**`score` 留空**
5. §0.3 加载规范冻结于 YAML：`4bit-nf4`／cuda／`attn_implementation=eager`／rev `acf7aafa…`
6. §0.4-1 不改权重：`finetuned: false`
7. §0.4-6 冒烟双跑与 §0.4-7 显存记录项的配置断言
8. 硬约束：顶层不 import 重型库；**无 `except: pass`**；产物 LF；`--self-check` exit 0

**离线**：全部输入在 `tmp_path` 内自造；模型/tokenizer 以**桩件**替代；
**不访问 `data/sealed_test/`**、不加载 torch／transformers。
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "baseline", "03_03_run_b3_llamaguard1b.py")
YAML = os.path.join(ROOT, "configs", "baseline", "b3_llamaguard1b.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("w43_b3", SCRIPT)


@pytest.fixture(scope="module")
def cfg(mod):
    return mod.load_config(YAML)


# ---------------------------------------------------------------------------
# 桩件：不加载 torch／transformers 也能验证输入链与 schema
# ---------------------------------------------------------------------------
class _StubCuda:
    def __init__(self):
        self.peak = 0

    def reset_peak_memory_stats(self):
        self.peak = 0

    def max_memory_allocated(self):
        return self.peak


class _StubTorch:
    def __init__(self):
        self.cuda = _StubCuda()

    def is_grad_enabled(self):
        return False


class _StubTensor:
    def __init__(self, ids):
        self.ids = list(ids)

    @property
    def shape(self):
        return (1, len(self.ids))

    def to(self, device):
        return self

    def __getitem__(self, item):
        return self.ids[item]


class _StubTokenizer:
    """按空白分词的确定性桩 tokenizer；`decode` 还原词序列。"""

    eos_token_id = 128009

    def __init__(self):
        self.words = []
        self.template_calls = []

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        words = text.split()
        self.words = words
        ids = list(range(len(words)))
        if return_tensors is not None:
            return {"input_ids": _StubTensor(ids)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.words[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        if isinstance(content, list):
            # 官方 Llama-Guard-3 模板按 content block 渲染：`selectattr('type','equalto','text')`
            text = " ".join(b["text"] for b in content if b.get("type") == "text")
        else:
            text = content
        self.template_calls.append({"tokenize": tokenize,
                                    "add_generation_prompt": add_generation_prompt,
                                    "role": messages[0]["role"],
                                    "content": content,
                                    "text": text})
        return "<|begin_of_text|>TPL:" + text + ":END"


class _EmptyConversationTokenizer(_StubTokenizer):
    """模拟「content 传了普通字符串 ⇒ 官方模板渲染出空会话段」的退化形态。"""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ("<|begin_of_text|>TASK\n<BEGIN CONVERSATION>\n\n<END CONVERSATION>\n"
                "\nPROVIDE ASSESSMENT")


class _StubParam:
    device = "cpu"


class _StubModel:
    def __init__(self, judgments):
        self.judgments = list(judgments)
        self.training = False
        self._i = 0

    def parameters(self):
        return iter([_StubParam()])

    def generate(self, input_ids=None, do_sample=None, max_new_tokens=None,
                 num_beams=None, pad_token_id=None, **_kw):
        assert do_sample is False, "B3 必须贪心解码（确定性）"
        n = len(input_ids.ids)
        judgment = self.judgments[self._i % len(self.judgments)]
        self._i += 1
        return [[0] * n + [self._i - 1]]


def _stub_bundle(mod, cfg, judgments):
    """装配 ``(torch, tok, model)`` 桩件；``tok.decode`` 由 `judgments` 索引回判词原文。"""
    torch_stub = _StubTorch()
    tok = _StubTokenizer()
    model = _StubModel(judgments)

    real_decode = tok.decode

    def decode(ids, skip_special_tokens=False):
        if ids and isinstance(ids[-1], int) and len(ids) == 1:
            return judgments[ids[-1] % len(judgments)]
        return real_decode(ids, skip_special_tokens)

    tok.decode = decode
    return torch_stub, tok, model


def _write_split(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "text", "label"], lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# ① 判词解析（§2 验收的构造判词）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("judgment,pred", [
    ("safe", "0"),
    ("unsafe", "1"),
    ("unsafe\nS9", "1"),
    ("unsafe\nS1", "1"),
    ("\n\nunsafe\nS1", "1"),          # 官方实际形态（前导空行）
    ("  safe  ", "0"),
    ("safe\n", "0"),
    ("unsafe S1", "1"),               # 同行空格分隔
    ("unsafe\nS9\nextra line", "1"),
])
def test_parse_judgment_first_line_first_word(mod, judgment, pred):
    got_pred, _word, parsed = mod.parse_judgment(judgment)
    assert parsed is True
    assert got_pred == pred


def test_parse_judgment_returns_first_word(mod):
    assert mod.parse_judgment("unsafe\nS9")[1] == "unsafe"
    assert mod.parse_judgment("safe")[1] == "safe"


# ---------------------------------------------------------------------------
# ② unparsed：记数、pred 留空、**不兜底**
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("judgment", ["", "   ", "\n\n", "Unsafe\nS1", "UNSAFE", "maybe\nS1",
                                      "I cannot", "unsafex"])
def test_unparsed_pred_is_blank_not_defaulted(mod, judgment):
    pred, _word, parsed = mod.parse_judgment(judgment)
    assert parsed is False
    assert pred == "", "unparsed 的 pred 必须留空，禁止默认值兜底为 0/1"


def test_unparsed_returns_first_word_for_disclosure(mod):
    pred, word, parsed = mod.parse_judgment("maybe\nS1")
    assert (pred, word, parsed) == ("", "maybe", False)


def test_parse_judgment_is_case_sensitive_frozen_rule(mod):
    """首行首词口径冻结：不得为容错而大小写归一。"""
    assert mod.parse_judgment("Unsafe")[2] is False
    assert mod.parse_judgment("unsafe")[2] is True


# ---------------------------------------------------------------------------
# ③ §0.3 输入链：原文直通 → 左截断 512（保尾部）→ 官方 chat 模板
# ---------------------------------------------------------------------------
def test_build_prompt_truncates_left_before_templating(mod, cfg):
    torch_stub, tok, model = _stub_bundle(mod, cfg, ["safe"])
    text = "a b c d e"                    # 5 个词；max_tokens=512 ⇒ 不截断
    prompt, truncated, n_tokens, was_truncated = mod.build_prompt(tok, cfg, text)
    assert was_truncated is False
    assert truncated == text
    assert n_tokens == 5
    assert text in prompt


def test_build_prompt_keeps_tail_on_truncation(mod, cfg):
    torch_stub, tok, model = _stub_bundle(mod, cfg, ["safe"])
    cfg2 = dict(cfg)
    cfg2["input"] = dict(cfg["input"])
    cfg2["input"]["max_seq_tokens"] = 3
    text = "a b c d e"
    _prompt, truncated, n_tokens, was_truncated = mod.build_prompt(tok, cfg2, text)
    assert was_truncated is True
    assert truncated == "c d e", "左截断必须保**尾部**"
    assert n_tokens == 3


def test_build_prompt_uses_official_chat_template_only(mod, cfg):
    """§0.3：官方 chat 模板原样使用，**禁止任何项目侧 prompt 工程**。"""
    torch_stub, tok, model = _stub_bundle(mod, cfg, ["safe"])
    mod.build_prompt(tok, cfg, "hello world")
    call = tok.template_calls[-1]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert call["role"] == cfg["prompt"]["role"] == "user"
    assert call["text"] == "hello world"


def test_build_messages_emits_official_content_blocks(mod, cfg):
    """官方 LG-3 模板按 content block 渲染；传普通字符串会**静默渲染出空会话段**。"""
    msgs = mod.build_messages(cfg, "hello world")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list), "content 必须是 block 列表，否则官方模板渲染为空"
    assert content == [{"type": "text", "text": "hello world"}]


def test_build_messages_rejects_unknown_content_format(mod, cfg):
    cfg2 = dict(cfg)
    cfg2["prompt"] = dict(cfg["prompt"])
    cfg2["prompt"]["content_format"] = "plain_string"
    with pytest.raises(ValueError):
        mod.build_messages(cfg2, "hello")


def test_build_prompt_raises_on_empty_conversation(mod, cfg):
    """空会话段 ⇒ 判词退化为常数；必须**抛错**而非静默通过。"""
    tok = _EmptyConversationTokenizer()
    with pytest.raises(RuntimeError):
        mod.build_prompt(tok, cfg, "anything at all")


# ---------------------------------------------------------------------------
# ④ §2 schema + pred 映射 + score 留空 + unparsed 计数
# ---------------------------------------------------------------------------
def test_run_split_schema_pred_and_unparsed_counting(mod, cfg, tmp_path):
    path = str(tmp_path / "s.csv")
    _write_split(path, [
        {"sample_id": "A", "text": "alpha beta", "label": "1"},
        {"sample_id": "B", "text": "gamma delta", "label": "0"},
        {"sample_id": "C", "text": "epsilon zeta", "label": "1"},
    ])
    torch_stub, tok, model = _stub_bundle(mod, cfg,
                                          ["unsafe\nS9", "safe", "I cannot answer"])
    rows, stats = mod.run_split("s", path, sealed=False, torch=torch_stub, transformers=None,
                                tok=tok, model=model, cfg=cfg)

    assert list(rows[0].keys()) == ["sample_id", "text_sha256", "pred", "raw_type", "raw",
                                    "score", "latency_ms"]
    assert [r["pred"] for r in rows] == ["1", "0", ""]
    assert all(r["score"] == "" for r in rows), "§2：B3 的 score 必须留空"
    assert all(r["raw_type"] == "judgment" for r in rows)
    assert json.loads(rows[0]["raw"]) == "unsafe\nS9"
    assert json.loads(rows[1]["raw"]) == "safe"
    assert rows[0]["text_sha256"] == hashlib.sha256("alpha beta".encode("utf-8")).hexdigest()
    assert float(rows[0]["latency_ms"]) >= 0.0

    assert stats["n"] == 3
    assert stats["n_unsafe"] == 1 and stats["n_safe"] == 1 and stats["n_unparsed"] == 1
    assert stats["unparsed_examples"][0]["sample_id"] == "C"
    assert stats["unparsed_examples"][0]["judgment"] == "I cannot answer"


def test_run_split_output_csv_roundtrip(mod, cfg, tmp_path):
    path = str(tmp_path / "s.csv")
    _write_split(path, [{"sample_id": "A", "text": "alpha beta", "label": "1"}])
    torch_stub, tok, model = _stub_bundle(mod, cfg, ["unsafe\nS1"])
    rows, _ = mod.run_split("s", path, sealed=False, torch=torch_stub, transformers=None,
                            tok=tok, model=model, cfg=cfg)
    out_path = str(tmp_path / "b3_s_v1.0_test.csv")
    mod.write_csv(out_path, rows, mod.OUTPUT_COLUMNS)
    with open(out_path, "r", encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert got[0]["sample_id"] == "A"
    assert got[0]["pred"] == "1"
    assert got[0]["raw_type"] == "judgment"
    assert got[0]["score"] == ""
    assert json.loads(got[0]["raw"]) == "unsafe\nS1"


def test_generate_judgment_asserts_eval_and_no_grad(mod, cfg):
    """G-EVAL（`AGENTS.md` §6-11）：train 模式或 grad 启用 ⇒ 必须抛错，不得静默。"""
    torch_stub, tok, model = _stub_bundle(mod, cfg, ["safe"])

    model.training = True
    with pytest.raises(RuntimeError):
        mod.generate_judgment(torch_stub, tok, model, cfg, "hello")

    model.training = False

    class _GradTorch(_StubTorch):
        def is_grad_enabled(self):
            return True

    with pytest.raises(RuntimeError):
        mod.generate_judgment(_GradTorch(), tok, model, cfg, "hello")


# ---------------------------------------------------------------------------
# ⑤ 统计确定性
# ---------------------------------------------------------------------------
def test_latency_stats(mod):
    s = mod.latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == pytest.approx(3.0)
    assert s["median"] == 3.0
    assert s["p95"] == 5.0
    assert mod.latency_stats([]) == {"mean": None, "median": None, "p95": None}


# ---------------------------------------------------------------------------
# ⑥ §0.3 加载规范冻结于 YAML（依 ENV_LOCK.md §5／§6）
# ---------------------------------------------------------------------------
def test_weights_are_raw_llamaguard_not_finetuned(cfg):
    w = cfg["weights"]
    assert w["repo_id"] == "meta-llama/Llama-Guard-3-1B"
    assert w["hf_revision"] == "acf7aafa60f0410f8f42b1fa35e077d705892029"
    assert w["local_path"] == "models/hf_cache/modelscope--LLM-Research--Llama-Guard-3-1B"
    assert w["finetuned"] is False, "B3 = 原始 LG-1B；微调后的 LG-1B 归 W5（Ours）"
    assert w["local_files_only"] is True


def test_load_spec_is_4bit_nf4_cuda_eager(cfg):
    load = cfg["load"]
    assert load["mode"] == "4bit-nf4"
    assert load["device"] == "cuda"
    assert load["attn_implementation"] == "eager"
    assert "flash_attention_2" in load["forbidden_attn"]
    q = load["quantization"]
    assert q["load_in_4bit"] is True
    assert q["bnb_4bit_quant_type"] == "nf4"
    assert q["bnb_4bit_compute_dtype"] == "bfloat16"
    assert load["eval_mode"] is True and load["no_grad"] is True


def test_input_spec_512_left_truncation(cfg):
    inp = cfg["input"]
    assert inp["column"] == "text"
    assert inp["pass_through"] is True
    assert inp["max_seq_tokens"] == 512
    assert inp["truncation_side"] == "left"


def test_prompt_is_official_template_no_project_engineering(cfg):
    p = cfg["prompt"]
    assert p["template_source"] == "official_chat_template"
    assert p["apply_chat_template"] is True
    assert p["add_generation_prompt"] is True
    assert p["project_side_prompt_engineering"] is False
    assert p["content_format"] == "list_of_text_blocks"
    assert p["empty_conversation_guard"] is True
    assert p["categories"] == "default_from_template"


def test_generation_is_greedy(cfg):
    g = cfg["generation"]
    assert g["do_sample"] is False
    assert g["num_beams"] == 1
    assert g["max_new_tokens"] > 0


def test_mapping_and_output_schema(cfg):
    m = cfg["mapping"]
    assert m["rule"] == "first_line_first_word_equals_unsafe"
    assert m["unsafe_word"] == "unsafe" and m["safe_word"] == "safe"
    assert m["unparsed_counting"] is True
    assert m["unparsed_pred"] == ""
    assert m["threshold"] is None

    out = cfg["output"]
    assert out["columns"] == ["sample_id", "text_sha256", "pred", "raw_type", "raw", "score",
                              "latency_ms"]
    assert out["raw_type"] == "judgment"
    assert out["score"] is None
    assert out["auroc"] == "n/a"
    assert out["newline"] == "lf"


def test_splits_are_three_test_splits_and_sealed(cfg):
    assert cfg["splits"]["eval"] == ["id_test", "ood_test", "clean_test"]
    assert cfg["splits"]["all"] == ["id_test", "ood_test", "clean_test"]
    assert cfg["splits"]["sealed"] is True


def test_smoke_and_memory_measurement_declared(cfg):
    assert cfg["smoke"]["double_run_enabled"] is True
    assert cfg["smoke"]["compare"] == "judgment_verbatim"
    assert cfg["measurement"]["memory"]["metric"] == "max_memory_allocated"
    assert cfg["measurement"]["memory"]["states"] == ["load", "infer"]
    assert cfg["measurement"]["latency"]["excludes"] == ["model_load", "csv_io"]


def test_clean_test_expectation_declared(cfg):
    ct = cfg["clean_test"]
    assert ct["rows"] == 44
    assert ct["all_label"] == 1
    assert ct["fpr"] == "undefined"


# ---------------------------------------------------------------------------
# ⑦ 硬约束与 CLI
# ---------------------------------------------------------------------------
def test_top_level_imports_only_stdlib(mod):
    tree = ast.parse(open(SCRIPT, "r", encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "argparse", "importlib", "json", "os", "sys", "time",
                   "datetime"}


def test_module_namespace_has_no_heavy_libs(mod):
    assert "torch" not in vars(mod)
    assert "transformers" not in vars(mod)


def test_no_bare_except_or_except_pass(mod):
    """硬约束：禁止 `except: pass` 静默吞错；本项目亦禁裸 `except:`。"""
    src = open(SCRIPT, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None, "禁止裸 except:（第 %d 行）" % node.lineno
            body_types = [type(b) for b in node.body]
            assert body_types != [ast.Pass], "禁止 except ...: pass（第 %d 行）" % node.lineno


def test_script_has_no_crlf_and_reconfigures_stdout(mod):
    raw = open(SCRIPT, "rb").read()
    assert raw.count(b"\r") == 0
    src = raw.decode("utf-8")
    assert "reconfigure(encoding=\"utf-8\"" in src


def test_self_check_exits_zero():
    proc = subprocess.run([sys.executable, SCRIPT, "--self-check"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert "SELF_CHECK OK" in proc.stdout


def test_config_yaml_has_no_crlf():
    assert open(YAML, "rb").read().count(b"\r") == 0


# ---------------------------------------------------------------------------
# ⑧ 测试本身不得访问 sealed_test
# ---------------------------------------------------------------------------
def test_this_test_file_does_not_reference_sealed_csv():
    src = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    needle = "data/" + "sealed" + "_test/CodeObf-PI_v1.0/regime"
    assert needle not in src

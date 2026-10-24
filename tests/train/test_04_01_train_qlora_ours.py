# -*- coding: utf-8 -*-
"""tests/train/test_04_01_train_qlora_ours.py — W5.1 `04_01` 的单元与接口测试。

覆盖 `windows/W5.1_qlora.md` §6-3 列出的要点：

1. `--mode grid` 的 **9 组展开**（事前定值空间 ＋ `n_cells` 一致性 ＋ 只允许 `regime id`）；
2. τ 寻优 **并列取 τ 较大者**（与 τ_B2 同制；F1 复用 `03_07` 单源）；
3. 早停 **patience = 2**；
4. **泄露断言**（禁读面拦截、val 不得进入 train、体制二 4 留出码 0 命中）；
5. `text_sha256` 与 `03_06` **同源一致** ＋ 512 左截断 ＋ 目标串单 token；
6. **checkpoint 命名格式**（`AGENTS.md` §4.5）与禁止覆盖同名；
7. C9／C11 静态约束（无 `.train(` 调用、`--report-dir` 未被写死）；
8. C10 组批护栏：只降单批 n，不改样本。

**设计纪律**：本文件**不加载模型权重、不需要 GPU**（§4-6：测试须可在秒级跑完）。
tokenizer 用例只做 3 条样本的极小子集编码。
"""
from __future__ import annotations

import argparse
import ast
import csv as _csv
import importlib.util
import os
import random
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # tests/train -> 项目根
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train", "04_01_train_qlora_ours.py")
CONFIG = os.path.join(ROOT, "configs", "train", "qlora_ours.yaml")
PREPROCESS = os.path.join(ROOT, "scripts", "baseline", "03_06_unify_preprocess.py")
MAPPING = os.path.join(ROOT, "scripts", "baseline", "03_07_unify_output_mapping.py")
SPLITS = os.path.join(ROOT, "data", "frozen", "CodeObf-PI_v1.0", "splits")
LOCAL_MODEL = os.path.join(ROOT, "models", "hf_cache",
                           "modelscope--LLM-Research--Llama-Guard-3-1B")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load("t_04_01", TRAIN_SCRIPT)


@pytest.fixture(scope="module")
def cfg(m):
    return m.load_config(CONFIG)


@pytest.fixture(scope="module")
def pp():
    return _load("t_03_06", PREPROCESS)


@pytest.fixture(scope="module")
def mp():
    return _load("t_03_07", MAPPING)


def _ns(**kw):
    base = dict(regime="id", seed=42, config=CONFIG, mode="train", curriculum="off",
                epochs=None, max_steps=None, report_dir="reports/train",
                out_dir="checkpoints/ours", log_dir="logs/train", checkpoint_dir=None,
                backfill=False, grid_path="sequential")
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# ① 网格：9 组展开（事前定值）
# ---------------------------------------------------------------------------
def test_grid_expand_nine_cells(m, cfg):
    cells = m.grid_cells(cfg)
    assert len(cells) == 9, "网格必须是 9 组（lr × r = 3 × 3）"
    assert cells == [(1e-4, 8), (1e-4, 16), (1e-4, 32),
                     (2e-4, 8), (2e-4, 16), (2e-4, 32),
                     (4e-4, 8), (4e-4, 16), (4e-4, 32)]
    assert sorted({c[0] for c in cells}) == [1.0e-4, 2.0e-4, 4.0e-4]
    assert sorted({c[1] for c in cells}) == [8, 16, 32]


def test_grid_n_cells_matches_config(m, cfg):
    assert len(m.grid_cells(cfg)) == int(cfg["grid"]["n_cells"]) == 9


def test_grid_only_regime_id(m, cfg):
    """`PREREG_v1.md` §9：只在 regime_id 上做一次；OOD 另起一次属偏离（须先写台账呈 W0）。"""
    assert cfg["grid"]["only_regimes"] == ["id"]
    with pytest.raises(ValueError):
        m.run_grid(cfg, _ns(regime="ood"), m.Logger(), "ts")


def test_grid_select_tie_break(m):
    """并列先取 lr 较小者，再取 r 较小者（事前定值的固定规则）。"""
    rows = [{"lr": 4e-4, "r": 8, "val_f1_best": 0.90},
            {"lr": 1e-4, "r": 32, "val_f1_best": 0.90},
            {"lr": 1e-4, "r": 8, "val_f1_best": 0.90}]
    sel = m._select_grid(rows)
    assert (sel["lr"], sel["r"]) == (1.0e-4, 8)
    rows2 = [{"lr": 2e-4, "r": 16, "val_f1_best": 0.80},
             {"lr": 4e-4, "r": 8, "val_f1_best": 0.95}]
    sel2 = m._select_grid(rows2)
    assert (sel2["lr"], sel2["r"]) == (4e-4, 8)


# ---------------------------------------------------------------------------
# ①-b 序贯省时路径与停止规则（`D-42` 裁定六；事前定值，无 GPU）
# ---------------------------------------------------------------------------
def _row(lr, f1, r=16):
    return {"lr": float(lr), "r": int(r), "val_f1_best": float(f1)}


def test_sequential_cells_stage1_is_r16_three_lrs(m, cfg):
    """stage1 恰 3 组：r=16 × 3 个 lr；且**不含** r∈{8,32}。"""
    s1 = m.sequential_cells(cfg, "1")
    assert s1 == [(1e-4, 16), (2e-4, 16), (4e-4, 16)]
    assert all(r == 16 for _lr, r in s1)
    assert not any(r in (8, 32) for _lr, r in s1)


def test_sequential_cells_stage2_is_remaining_six(m, cfg):
    """stage2 补跑 6 组：全 lr × r∈{8,32}，且与 stage1 无交集。"""
    s1 = set(m.sequential_cells(cfg, "1"))
    s2 = m.sequential_cells(cfg, "2")
    assert len(s2) == 6
    assert all(r in (8, 32) for _lr, r in s2)
    assert s1.isdisjoint(set(s2))


def test_sequential_cells_all_equals_nine(m, cfg):
    """stage1 ∪ stage2 恰 = 默认 9 组（等式校验，无重复无遗漏）。"""
    allc = m.sequential_cells(cfg, "all")
    assert len(allc) == int(cfg["grid"]["n_cells"]) == 9
    assert set(allc) == set(m.grid_cells(cfg))


def test_stopping_rule_all_satisfied_stops(m, cfg):
    """四条件全满足 ⇒ STOP：内部最优（2e-4）、极差 <0.01、最优 ≥0.5、无越界平局。"""
    rows = [_row(1e-4, 0.8000), _row(2e-4, 0.8050), _row(4e-4, 0.7990)]  # range=0.0060
    v = m.evaluate_stopping_rule(cfg, rows)
    assert v["stop"] is True
    assert all(v["conditions"].values())
    assert (v["best"]["lr"], v["best"]["r"]) == (2e-4, 16)


def test_stopping_rule_boundary_best_continues(m, cfg):
    """条件 B：最优 lr 落在边界（1e-4／4e-4）⇒ 不满足 ⇒ CONTINUE。"""
    rows = [_row(1e-4, 0.8050), _row(2e-4, 0.8010), _row(4e-4, 0.8000)]  # range=0.005<0.01
    v = m.evaluate_stopping_rule(cfg, rows)
    assert v["conditions"]["A_range"] is True        # 极差本身收敛
    assert v["conditions"]["B_interior"] is False    # 但最优在边界
    assert v["stop"] is False


def test_stopping_rule_wide_range_continues(m, cfg):
    """条件 A：极差 ≥ 0.0100（含等于）⇒ 不视为收敛 ⇒ CONTINUE。"""
    rows = [_row(1e-4, 0.8000), _row(2e-4, 0.8100), _row(4e-4, 0.8000)]  # range=0.0100 exactly
    v = m.evaluate_stopping_rule(cfg, rows)
    assert v["conditions"]["A_range"] is False, "等于阈值不算收敛（严格小于）"
    assert v["stop"] is False


def test_stopping_rule_below_floor_continues(m, cfg):
    """条件 C：最优 val_f1_best < 0.5000 ⇒ 即使极差很小也不判收敛 ⇒ CONTINUE。"""
    rows = [_row(1e-4, 0.4500), _row(2e-4, 0.4560), _row(4e-4, 0.4520)]  # range=0.006<0.01
    v = m.evaluate_stopping_rule(cfg, rows)
    assert v["conditions"]["C_floor"] is False
    assert v["stop"] is False


def test_stopping_rule_tie_spanning_boundary_continues(m, cfg):
    """条件 D：并列集合跨边界（1e-4 与 2e-4 同为最高分）⇒ 不满足 ⇒ CONTINUE。"""
    rows = [_row(1e-4, 0.8050), _row(2e-4, 0.8050), _row(4e-4, 0.7900)]
    v = m.evaluate_stopping_rule(cfg, rows)
    assert v["conditions"]["D_tie"] is False
    assert v["stop"] is False


def test_stopping_rule_logic_is_ALL(m, cfg):
    """`logic=ALL`：四条件须**同时**满足；任一条否 ⇒ 整体 CONTINUE。"""
    assert cfg["grid"]["sequential"]["stopping_rule"]["logic"] == "ALL"
    # A/C/D 满足、仅 B 不满足（边界最优）⇒ 仍 CONTINUE（证明非「满足多数即可」）
    rows = [_row(1e-4, 0.8050), _row(2e-4, 0.8010), _row(4e-4, 0.8000)]
    v = m.evaluate_stopping_rule(cfg, rows)
    assert v["conditions"] == {"A_range": True, "B_interior": False,
                               "C_floor": True, "D_tie": True}
    assert sum(v["conditions"].values()) == 3 and v["stop"] is False


# ---------------------------------------------------------------------------
# ② τ_Ours：F1 最大 ＋ 并列取 τ 较大者
# ---------------------------------------------------------------------------
def test_tau_tie_takes_larger(m):
    """全同分：τ=-inf 与 τ=0.9 的**预测集完全相同**（F1 并列 0.75）⇒ 必须取 **τ 较大者**。

    （候选集 = [-inf] ∪ 唯一 score ∪ [+inf]，升序遍历并用 `>=` 更新 ⇒ 并列取大。）
    """
    scores = [0.9, 0.9, 0.9, 0.9, 0.9]
    labels = [1, 1, 1, 0, 0]
    table = {}
    for tau in [float("-inf"), 0.9, float("inf")]:
        table[tau] = m.f1_score(labels, m.apply_threshold(scores, tau))
    assert table[float("-inf")] == pytest.approx(0.75)
    assert table[0.9] == pytest.approx(0.75)
    tau, f1 = m.search_tau_ours(scores, labels)
    assert f1 == pytest.approx(0.75)
    assert tau == 0.9, "并列时必须取 **τ 较大者**（与 τ_B2 同制），实得 %r" % tau


def test_tau_simple_separation(m):
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    labels = [1, 1, 1, 0, 0, 0]
    tau, f1 = m.search_tau_ours(scores, labels)
    assert f1 == pytest.approx(1.0)
    assert 0.2 <= tau <= 0.7


def test_tau_f1_is_03_07_single_source(m, mp):
    """指标单源：`04_01` 的 F1／指标函数必须**定义在 `03_07` 里**（不得另起一套实现）。"""
    for fn in (m.f1_score, m.compute_binary_metrics, m.binary_counts):
        assert os.path.abspath(fn.__code__.co_filename) == os.path.abspath(MAPPING), \
            "%s 未定义在 03_07（实际来自 %s）" % (fn.__name__, fn.__code__.co_filename)
    assert m.f1_score([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)
    assert m.f1_score.__code__.co_code == mp.f1_score.__code__.co_code
    assert m.compute_binary_metrics.__code__.co_code == \
        mp.compute_binary_metrics.__code__.co_code


# ---------------------------------------------------------------------------
# ③ 早停 patience = 2
# ---------------------------------------------------------------------------
def test_early_stopping_patience_is_two(m, cfg):
    assert int(cfg["early_stopping"]["patience"]) == 2
    assert bool(cfg["early_stopping"]["save_best"]) is True
    assert cfg["early_stopping"]["monitor"] == "val_f1"


def test_early_stopping_fires_after_two_no_improve(m):
    es = m.EarlyStopping(patience=2)
    assert es.update(0.50)[:2] == (True, 0)
    assert es.should_stop() is False
    assert es.update(0.40)[:2] == (False, 1)
    assert es.should_stop() is False, "patience=2 ⇒ 第 1 轮未上升不得停"
    assert es.update(0.45)[:2] == (False, 2)
    assert es.should_stop() is True


def test_early_stopping_resets_on_improve(m):
    es = m.EarlyStopping(patience=2)
    es.update(0.50)
    es.update(0.40)
    assert es.update(0.60)[:2] == (True, 0)
    assert es.should_stop() is False


# ---------------------------------------------------------------------------
# ④ 泄露断言（C1／§4-1／§4-2）
# ---------------------------------------------------------------------------
def test_resolve_split_paths_only_four_files(m, cfg):
    seen = set()
    for regime in ("id", "ood"):
        tr, va = m.resolve_split_paths(cfg, regime)
        assert os.path.abspath(tr).endswith(
            os.path.join("regime_" + regime, "train.csv"))
        assert os.path.abspath(va).endswith(
            os.path.join("regime_" + regime, "val.csv"))
        assert os.path.isfile(tr) and os.path.isfile(va)
        seen |= {tr, va}
    assert len(seen) == 4


def test_forbidden_faces_rejected(m, cfg):
    for bad in (os.path.join(ROOT, "data", "sealed_test", "x.csv"),
                os.path.join(SPLITS, "regime_id", "test.csv"),
                os.path.join(SPLITS, "regime_ood", "test.csv"),
                os.path.join(ROOT, "..", "outside", "train.csv")):
        with pytest.raises(m.LeakAssertionError):
            m.assert_split_allowed(bad, cfg)


def test_allowed_faces_accepted(m, cfg):
    for regime in ("id", "ood"):
        for kind in ("train", "val"):
            p = os.path.join(SPLITS, "regime_" + regime, kind + ".csv")
            assert m.assert_split_allowed(p, cfg) == os.path.abspath(p)


def test_val_not_in_train_real_splits(m, cfg):
    for regime in ("id", "ood"):
        tr, va = m.resolve_split_paths(cfg, regime)
        assert m.assert_val_not_in_train(tr, va, cfg) is True


def test_val_not_in_train_detects_overlap(m, cfg, tmp_path):
    for name in ("train.csv", "val.csv"):
        with open(os.path.join(str(tmp_path), name), "w", encoding="utf-8",
                  newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["sample_id", "text", "label"])
            w.writeheader()
            w.writerow({"sample_id": "s1", "text": "hello", "label": "1"})
            w.writerow({"sample_id": "s2", "text": "world", "label": "0"})
    with pytest.raises(m.LeakAssertionError):
        m.assert_val_not_in_train(os.path.join(str(tmp_path), "train.csv"),
                                  os.path.join(str(tmp_path), "val.csv"), cfg)


def test_ood_train_encodings_real_split_passes(m, cfg):
    """体制二 train 侧：4 留出码命中须 = 0（**一号门禁**）。"""
    tr, _va = m.resolve_split_paths(cfg, "ood")
    assert m.assert_ood_train_encodings(cfg, tr) is True


def test_ood_train_encodings_detects_heldout(m, cfg, tmp_path, monkeypatch):
    """把一个训练编码行改成留出码 ⇒ 必须抛 `LeakAssertionError`。"""
    src = os.path.join(SPLITS, "regime_ood", "train.csv")
    rows = m.load_rows(src)
    held = sorted(set(cfg["assertions"]["ood_heldout_encodings"]))[0]
    allowed = set(cfg["assertions"]["ood_train_encodings"])
    changed = False
    for r in rows:
        if (r.get("encoding_type") or "").strip() in allowed:
            r["encoding_type"] = held
            changed = True
            break
    assert changed, "测试数据里没找到可替换的训练编码行"
    dst = os.path.join(str(tmp_path), "regime_ood", "train.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    m.write_csv(dst, rows, list(rows[0].keys()))
    with pytest.raises(m.LeakAssertionError):
        m.assert_ood_train_encodings(cfg, dst)


# ---------------------------------------------------------------------------
# ⑤ text_sha256 与 03_06 同源 ＋ 512 左截断 ＋ 单 token 目标
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(LOCAL_MODEL, local_files_only=True)


def test_target_ids_single_token_and_match_config(m, cfg, tok):
    id_u, id_s = m.resolve_target_ids(tok, cfg)
    assert (id_u, id_s) == (39257, 19193)
    assert id_u != id_s


def test_encode_matches_03_06_sha256(m, cfg, pp, tok):
    id_u, id_s = m.resolve_target_ids(tok, cfg)
    tr, _va = m.resolve_split_paths(cfg, "id")
    exs = m.prepare_split(tok, cfg, tr, id_u, id_s, tok.eos_token_id, ids=[0, 1, 2])
    samples = m.load_samples(tr)
    assert len(exs) == 3
    for ex, s in zip(exs, samples[:3]):
        assert ex.sha256 == pp.sha256_of(s.text), "text_sha256 必须与 03_06 同源"
        assert len(ex.sha256) == 64
        assert ex.text == s.text, "原文直通，不得清洗"
        assert ex.target_ids[0] == (id_u if int(s.label) == 1 else id_s)


def test_left_truncation_512(m, cfg, tok):
    """512 token 上限施加于 `text` 本体（左截断）；prompt = 模板 + 截断后文本。"""
    id_u, id_s = m.resolve_target_ids(tok, cfg)
    tr, _va = m.resolve_split_paths(cfg, "id")
    exs = m.prepare_split(tok, cfg, tr, id_u, id_s, tok.eos_token_id, ids=[0, 1, 2])
    for ex in exs:
        body_len = ex.n_prompt - 196                  # 官方模板实测 196 token
        assert body_len <= 512, "左截断后本体长度 %d > 512" % body_len


def test_build_prompt_rejects_degenerate_empty_conversation(m, cfg, tok):
    """W4.3 教训：渲染后会话段为空 ⇒ **抛错**，不得静默退化为常数判词。"""
    with pytest.raises(RuntimeError):
        m.build_prompt(tok, cfg, m.EMPTY_CONVERSATION)


def test_build_messages_is_list_of_text_blocks(m, cfg):
    msgs = m.build_messages(cfg, "hello")
    assert isinstance(msgs[0]["content"], list)
    assert msgs[0]["content"][0]["type"] == "text"


# ---------------------------------------------------------------------------
# ⑥ checkpoint 命名（AGENTS.md §4.5）与禁止覆盖
# ---------------------------------------------------------------------------
def test_checkpoint_naming_format(m, cfg):
    d = m.checkpoint_dir(os.path.join(ROOT, "checkpoints", "ours"), cfg, "id", 42, 120,
                         "20261004-111443")
    rel = os.path.relpath(d, ROOT).replace("\\", "/")
    assert rel == ("checkpoints/ours/04_01_llamaguard1b_id_seed42_20261004-111443/"
                   "seed_42/checkpoint-120"), rel


def test_checkpoint_naming_seed_isolated(m, cfg):
    d1 = m.checkpoint_dir("X", cfg, "id", 42, 10, "T")
    d2 = m.checkpoint_dir("X", cfg, "id", 123, 10, "T")
    assert d1 != d2 and "seed_42" in d1 and "seed_123" in d2


class _FakeModel:
    def save_pretrained(self, path):
        with open(os.path.join(path, "adapter_model.safetensors"), "w") as fh:
            fh.write("x")


class _FakeTok:
    def save_pretrained(self, path):
        with open(os.path.join(path, "tokenizer.json"), "w") as fh:
            fh.write("{}")


def test_checkpoint_no_overwrite(m, cfg, tmp_path):
    ts = "20261004-000000"
    d = m.checkpoint_dir(str(tmp_path), cfg, "id", 42, 7, ts)
    os.makedirs(d, exist_ok=True)
    with pytest.raises(FileExistsError):
        m.save_checkpoint(_FakeModel(), _FakeTok(), cfg, str(tmp_path), "id", 42, 7, ts,
                          {}, m.Logger())


def test_checkpoint_saves_adapter_and_meta(m, cfg, tmp_path):
    ts = "20261004-000001"
    d = m.save_checkpoint(_FakeModel(), _FakeTok(), cfg, str(tmp_path), "id", 42, 7, ts,
                          {"regime": "id", "seed": 42}, m.Logger())
    assert os.path.isfile(os.path.join(d, "adapter_model.safetensors"))
    assert os.path.isfile(os.path.join(d, "meta.json"))


# ---------------------------------------------------------------------------
# ⑦ 静态约束：C9 无 `.train(` 调用；C11 报告路径不被写死
# ---------------------------------------------------------------------------
def test_no_train_mode_call_in_source():
    """C9：代码（注释／文档串以外）不得出现 `.train(`。"""
    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    bad = [getattr(n, "lineno", "?") for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "train"]
    assert not bad, "出现 .train( 调用（C9 禁止）：行 %s" % bad


def test_cublas_set_before_torch_import():
    src = open(TRAIN_SCRIPT, "r", encoding="utf-8").read()
    assert src.index("CUBLAS_WORKSPACE_CONFIG") < src.index("def build_model")
    tree = ast.parse(src)
    top_imports = {n.module or "" for n in tree.body if isinstance(n, ast.ImportFrom)} | {
        a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert "torch" not in top_imports, "顶层不得 import torch（须惰性加载）"


# ---------------------------------------------------------------------------
# N-5 回归（`D-42` §4）：训练侧确定性的两个必需落点
# ---------------------------------------------------------------------------
def test_seed_set_before_build_model_in_run_train(m):
    """N-5 根因：`run_train` 内 `set_determinism(...)` 必须**先于** `build_model(...)`。

    `build_model()` 的 `get_peft_model()` 会用全局 RNG 随机初始化 LoRA A
    （`lora_B` 恒 0）⇒ 若播种子晚于建模，同 seed 两次跑的初始权重不同、
    梯度与 loss 遂不同（N-5 实测：`lora_A[layer0.q_proj]` 指纹
    `3baf0843…` vs `9fd817f3…`）。本用例静态钉死调用次序，防回归。
    """
    src = open(TRAIN_SCRIPT, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_train")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    det_line = min((n.lineno for n in calls
                    if isinstance(n.func, ast.Name) and n.func.id == "set_determinism"),
                   default=None)
    bm_line = min((n.lineno for n in calls
                   if isinstance(n.func, ast.Name) and n.func.id == "build_model"),
                  default=None)
    assert det_line is not None, "run_train 中未见 set_determinism 调用"
    assert bm_line is not None, "run_train 中未见 build_model 调用"
    assert det_line < bm_line, (
        "N-5 回归：set_determinism（行 %s）必须先于 build_model（行 %s）" % (det_line, bm_line))


def test_set_determinism_calls_use_deterministic_algorithms(m, cfg):
    """N-5：`set_determinism` 必须施加**框架级**确定性算法（cudnn 不覆盖 scatter-add 等）。"""
    src = open(TRAIN_SCRIPT, "r", encoding="utf-8").read()
    assert "use_deterministic_algorithms" in src, \
        "N-5 回归：04_01 未调用 torch.use_deterministic_algorithms"
    det = cfg["determinism"]
    assert det.get("use_deterministic_algorithms") is True
    assert "deterministic_warn_only" in det
    # 事前定值：TF32 两处（matmul / cudnn）均须关掉
    assert det.get("allow_tf32") is False


def test_n5_double_run_tolerance_preregistered(cfg):
    """N-5：双跑一致判据必须**事前定值**写在 yaml（`D-42` §4 处置②）。"""
    det = cfg["determinism"]
    v = det.get("verify_double_run")
    assert v, "缺 determinism.verify_double_run（事前定值判据）"
    assert int(v["n_runs"]) == 2
    assert v["regime"] in ("id", "ood")
    assert int(v["seed"]) in (42, 123, 456)
    tol = v["tolerance"]
    # 位级一致：三项容差均为 0.0
    for k in ("loss_abs", "val_f1_abs", "tau_abs"):
        assert float(tol[k]) == 0.0, "事前定值要求位级一致（%s 容差须为 0.0）" % k
    # 降级路径亦须事前写死（否则属事后挑选）
    assert float(v["degrade_if_loss_abs_gt"]) > 0.0
    assert list(v["noise_baseline"]["report_deltas"])


def test_report_dir_not_hardcoded(m, cfg):
    """C11：`--report-dir` 必须真正生效（禁止把 `reports/` 写死）。"""
    assert m.artifact_path("MY_TMP", cfg, "curve", regime="id", seed=42, ts="T") \
        .startswith("MY_TMP")
    assert m.artifact_path("MY_TMP", cfg, "threshold", regime="id", ts="T") \
        .startswith("MY_TMP")
    assert m.artifact_path("MY_TMP", cfg, "grid_md", ts="T").startswith("MY_TMP")


def test_cli_contract_flags(m):
    args = m.build_parser().parse_args(
        ["--regime", "id", "--seed", "42", "--mode", "grid", "--curriculum", "off",
         "--epochs", "5", "--max-steps", "10", "--report-dir", "X", "--out-dir", "Y",
         "--checkpoint-dir", "Z"])
    assert (args.regime, args.seed, args.mode, args.curriculum) == ("id", 42, "grid", "off")
    assert (args.epochs, args.max_steps) == (5, 10)
    assert (args.report_dir, args.out_dir, args.checkpoint_dir) == ("X", "Y", "Z")
    with pytest.raises(SystemExit):
        m.build_parser().parse_args(["--regime", "id", "--seed", "7"])   # 种子须 ∈ {42,123,456}


# ---------------------------------------------------------------------------
# ⑧ 组批护栏（C10）：只降 n，不改样本/epoch/网格
# ---------------------------------------------------------------------------
def test_make_batches_keeps_all_samples(m, cfg):
    quad, linear, budget = m.batch_mem_guard(cfg)
    lens = [708] * 10 + [300] * 10 + [512] * 5
    batches = m.make_batches(lens, 4, quad, linear, budget, False)
    assert sorted(i for b in batches for i in b) == list(range(len(lens)))
    for b in batches:
        L = max(lens[i] for i in b)
        assert len(b) <= 4
        assert m.batch_cost(len(b), L, quad, linear) <= budget


def test_make_batches_shuffle_deterministic(m, cfg):
    quad, linear, budget = m.batch_mem_guard(cfg)
    lens = [708, 300, 512, 250, 700, 310]
    a = m.make_batches(lens, 4, quad, linear, budget, True, random.Random(42))
    b = m.make_batches(lens, 4, quad, linear, budget, True, random.Random(42))
    assert a == b
    c = m.make_batches(lens, 4, quad, linear, budget, True, random.Random(123))
    key = lambda bs: sorted(tuple(sorted(x)) for x in bs)  # noqa: E731
    assert key(a) == key(c), "洗牌只改批次**顺序**，分组内容不变"


def test_batch_guard_long_sequence_caps_n(m, cfg):
    """长序列（708）峰值 > 容量 ⇒ 护栏必须把 n 压到 2（实测 n=4 会溢出共享内存）。"""
    quad, linear, budget = m.batch_mem_guard(cfg)
    assert m.batch_cost(2, 708, quad, linear) <= budget
    assert m.batch_cost(4, 708, quad, linear) > budget
    assert m.batch_cost(4, 384, quad, linear) <= budget, "短序列仍应允许 n=4"


# ---------------------------------------------------------------------------
# ⑨ C7：课程学习开关（逻辑归 W5.2，`04_02` 未实现须优雅跳过）
# ---------------------------------------------------------------------------
def test_curriculum_graceful_skip_or_ok(m, cfg):
    cls, note = m.try_load_curriculum(cfg)
    if cls is None:
        assert note.startswith("CURRICULUM_SKIP:"), note
    else:
        assert note.startswith("CURRICULUM_OK:")
        assert all(hasattr(cls, x) for x in ("stage_at", "select", "describe"))
    assert cfg["curriculum"]["default"] == "off"
    assert cfg["curriculum"]["graceful_skip_when_missing"] is True


# ---------------------------------------------------------------------------
# ⑩ 数据面：C1 只读 4 个文件（读清单静态核对）
# ---------------------------------------------------------------------------
def test_data_config_declares_four_files_and_forbidden(cfg):
    assert cfg["data"]["train"].endswith("{regime}/train.csv")
    assert cfg["data"]["val"].endswith("{regime}/val.csv")
    joined = " ".join(cfg["data"]["forbidden_substrings"])
    assert "sealed_test" in joined
    assert "test.csv" in joined


# ---------------------------------------------------------------------------
# ⑪ `D-47` 裁定二：τ 主口径**触发式**改用 `0.5`（口径变更）
# ---------------------------------------------------------------------------
def _rule_cfg(cfg, **over):
    """深拷 `cfg` 并覆写 `threshold.main_criterion_override` 的若干键（测试用）。"""
    import copy
    c = copy.deepcopy(cfg)
    rule = c["threshold"]["main_criterion_override"]
    for path, val in over.items():
        node = rule
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return c


def _d47_shaped_scores():
    """复刻 `D-47` 实测形态：正类全在 τ* 附近、负类 max 距 τ* 仅 ~6.5e-5。"""
    scores = [0.9999986290931702, 0.9999988, 0.9999995, 0.9999999,  # 正类
              0.999934, 0.4000001, 0.0001]                            # 负类（max=0.999934）
    labels = [1, 1, 1, 1, 0, 0, 0]
    return scores, labels


def test_tau_override_config_predeclared(cfg):
    """规则本体必须是**事前定值**写在 yaml 里（`D-47` 裁定二），不得藏在代码常量中。"""
    rule = cfg["threshold"]["main_criterion_override"]
    assert rule["enabled"] is True
    assert rule["logic"] == "AND"
    assert rule["trigger"]["neg_max_gap_below"] == pytest.approx(1.0e-3)
    assert rule["on_trigger"]["main_tau"] == pytest.approx(0.5)
    assert rule["boundary_definition"]["tau_gt"] == pytest.approx(0.99)
    assert rule["boundary_definition"]["tau_lt"] == pytest.approx(0.01)


def test_tau_override_triggers_on_d47_case(m, cfg):
    """`D-47` 实测形态 ⇒ **触发**：主口径 `0.5`，τ* **并列保留**且为边界解。"""
    scores, labels = _d47_shaped_scores()
    res = m.resolve_tau_ours(scores, labels, cfg)
    assert res["is_boundary"] is True, "τ* > 0.99 ⇒ 应判为边界解"
    assert res["neg_max"] == pytest.approx(0.999934)
    assert res["gap"] < 1.0e-3
    assert res["triggered"] is True
    assert res["tau_main"] == pytest.approx(0.5)
    assert res["tau_star"] is not None and res["tau_star"] > 0.99, "τ* 须并列保留"
    # 理由②：改 0.5 后 F1 **变差**（反向、非择优）
    f1_main = m.compute_binary_metrics(labels, m.apply_threshold(scores, res["tau_main"]))["f1"]
    f1_star = m.compute_binary_metrics(labels, m.apply_threshold(scores, res["tau_star"]))["f1"]
    assert f1_star > f1_main, "主口径改 0.5 必须使 F1 下降（不得是择优）"


def test_tau_override_not_triggered_when_gap_wide(m, cfg):
    """条件二不满足（负类 max 远离 τ*）⇒ **不触发**，主口径 = τ*（原口径不变）。"""
    scores = [0.9999988, 0.9999995, 0.9999999, 0.5, 0.2]     # neg max = 0.5
    labels = [1, 1, 1, 0, 0]
    res = m.resolve_tau_ours(scores, labels, cfg)
    assert res["is_boundary"] is True                 # τ* > 0.99 仍成立
    assert res["gap"] > 1.0e-3
    assert res["triggered"] is False
    assert res["tau_main"] == pytest.approx(res["tau_star"])


def test_tau_override_not_triggered_when_not_boundary(m, cfg):
    """条件一不满足（τ* 非边界解）⇒ **不触发**，即使间隙极小。"""
    scores = [0.8, 0.8 + 1e-9, 0.8000000002, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0]
    res = m.resolve_tau_ours(scores, labels, cfg)
    assert res["is_boundary"] is False
    assert res["triggered"] is False
    assert res["tau_main"] == pytest.approx(res["tau_star"])


def test_tau_override_reads_thresholds_from_config(m, cfg):
    """阈值与目标 τ 必须**读自 yaml**：改配置即改行为（防硬编码）。"""
    scores, labels = _d47_shaped_scores()
    assert m.resolve_tau_ours(scores, labels, _rule_cfg(cfg, **{
        "on_trigger.main_tau": 0.7}))["tau_main"] == pytest.approx(0.7)
    # 把间隙阈值收到 1e-9 ⇒ 该形态不再触发
    assert m.resolve_tau_ours(scores, labels, _rule_cfg(cfg, **{
        "trigger.neg_max_gap_below": 1.0e-9}))["triggered"] is False
    # 关掉规则 ⇒ 完全退回原口径
    off = m.resolve_tau_ours(scores, labels, _rule_cfg(cfg, **{"enabled": False}))
    assert off["triggered"] is False and off["tau_main"] == pytest.approx(off["tau_star"])


def test_tau_override_keeps_boundary_detection_by_candidate_extreme(m, cfg):
    """边界解的另一支：τ* == 候选集极值（`or_candidate_extreme`）。"""
    # 正类只有一个 score ⇒ τ* 必为该唯一候选（同时是 min 与 max）
    scores = [0.5, 0.4, 0.3]
    labels = [1, 0, 0]
    res = m.resolve_tau_ours(scores, labels, cfg)
    assert res["is_boundary"] is True
    assert "candidate_extreme" in res["boundary_reasons"]


def test_exact_tau_keeps_full_precision(m):
    """`F-W5.1-2` 回归：τ 必须**全精度**记录 —— 6 位显示值在边界解处会翻转样本。"""
    tau_star = 0.9999986290931702
    assert m._exact(tau_star) == "0.9999986290931702"
    assert float(m._exact(tau_star)) == tau_star, "全精度往返必须无损"
    assert float(m._fmt(tau_star)) != tau_star, "6 位显示值必须与真值不同（这正是缺陷）"
    assert float(m._fmt(tau_star)) > tau_star, "显示值须 **大于** τ*（否则不会翻转正类）"
    assert m._exact(None) == ""


# ---------------------------------------------------------------------------
# ⑫ `D-48` 可续跑：**step 级精确续跑**
# ---------------------------------------------------------------------------
SMOKE_SCRIPT = os.path.join(ROOT, "scripts", "train", "04_00_smoke_train.py")


@pytest.fixture(scope="module")
def m00():
    return _load("t_04_00_resume", SMOKE_SCRIPT)


def test_resume_config_predeclared(cfg):
    """规则本体须**事前定值**写在 yaml（`D-48` 裁定一／三／四／五）。"""
    r = cfg["resume"]
    assert r["enabled"] is True
    assert r["state_every_opt_steps"] == 100, "落盘粒度 = 每 100 个 optimizer step"
    assert r["save_only_after_opt_step"] is True
    assert r["save_only_when_pending_zero"] is True, "落盘点须 pending==0（不存梯度）"
    assert r["rolling"] == ["latest", "prev"]
    assert r["include_tokenizer"] is False, "续跑 state 不写 tokenizer"
    assert r["equivalence_proof"]["compare"] == ["loss", "val_f1", "tau"]
    assert r["equivalence_proof"]["tolerance"] == {"loss_abs": 0.0, "val_f1_abs": 0.0,
                                                   "tau_abs": 0.0}
    assert r["channels"]["default"] == ["a_terminal", "b_resume"]


def test_resume_state_items_cover_d48_list(cfg):
    """`D-48` 裁定二 的 9 项须逐项在清单中出现。"""
    joined = " ".join(cfg["resume"]["state_items"])
    for kw in ("adapter", "opt.state_dict", "getstate", "RNG state", "epoch",
               "opt_steps", "游标", "stopper", "best", "curve_rows"):
        assert kw in joined, "state_items 缺项：%s" % kw


def test_resume_cli_flags_exist_and_default_off(m, m00):
    """`--resume` 必须存在且**默认 None**（未传 ⇒ 与改造前逐字节一致）。"""
    p = m.build_parser()
    a = p.parse_args(["--regime", "id", "--seed", "42"])
    assert a.resume is None and a.stop_after_steps is None
    b = p.parse_args(["--regime", "ood", "--seed", "123", "--resume", "d", "--stop-after-steps", "3"])
    assert b.resume == "d" and b.stop_after_steps == 3
    s = m00.build_parser().parse_args([])
    assert s.resume is None and s.stop_after_steps is None


def test_index_shuffle_equivalent_to_list_shuffle():
    """`D-48` 改造的**行为保持**证明：打乱下标 ≡ 打乱列表本体，且 **RNG 消耗逐位相同**。

    若不成立，则每个 epoch 的批序与后续 RNG 都会漂移 ⇒ 「未传 `--resume` 时逐字节一致」失效。
    """
    batches = [[i, i + 100] for i in range(23)]
    a = [list(b) for b in batches]
    ra = random.Random(42)
    ra.shuffle(a)
    orig = [list(b) for b in batches]
    order = list(range(len(orig)))
    rb = random.Random(42)
    rb.shuffle(order)
    assert [orig[k] for k in order] == a, "访问顺序必须逐项相同"
    assert ra.getstate() == rb.getstate(), "RNG 消耗必须逐位相同"


def test_rng_state_json_roundtrip(m):
    """`random.Random.getstate()` 的 JSON 往返必须无损（续跑的关键项）。"""
    r = random.Random(7)
    r.random()
    r.shuffle(list(range(10)))
    st = r.getstate()
    back = m._rng_state_from_json(m._rng_state_jsonable(st))
    assert back == st
    a = random.Random(0)
    a.setstate(st)
    b = random.Random(0)
    b.setstate(back)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_state_path_of_rejects_missing(m, tmp_path):
    """指向不存在的 state 目录须**报错**，不得静默从头训练。"""
    with pytest.raises(FileNotFoundError):
        m._state_path_of(str(tmp_path))


# ---------------------------------------------------------------------------
# ⑬ `D-50` 裁定二①：修 `N-8`（续跑产物目录分裂）
# ---------------------------------------------------------------------------
def _src():
    with open(TRAIN_SCRIPT, encoding="utf-8") as fh:
        return fh.read()


def test_snapshot_meta_carries_ts_and_run_id(m, cfg):
    """`D-50` 裁定二① 要求 1：`_snapshot` 的 meta 须含 `ts` 与 `run_id`（静态钉死）。

    这两项必须**随 state 落盘** —— 放在同一 `"meta"` 字典内，否则续跑无法恢复目录。
    """
    seg = _src().split("def _snapshot", 1)[1].split("cur_stage_logged", 1)[0]
    assert '"ts": str(ts)' in seg, "`_snapshot` 的 meta 缺 `ts`"
    assert '"run_id": run_id_of(cfg, args.regime, seed, ts)' in seg, \
        "`_snapshot` 的 meta 缺 `run_id`"
    assert seg.index('"meta"') < seg.index('"ts": str(ts)'), "`ts` 未落在 `meta` 字典内"


def test_resume_restores_ts_so_run_id_dir_stays_same(m, cfg):
    """`D-50` 裁定二① 要求 2（＋要求 5②）：**续跑段产物目录 == 首段 `run_id` 目录**。

    `run_id_of` 是 `ts` 的**纯函数** ⇒ **恢复 `ts` 即恢复目录** —— 此即 `N-8` 的修法。
    """
    src = _src()
    assert 'ts = str(_meta["ts"])' in src, "续跑未以 state 的 `ts` 覆盖本进程新生成的 `ts`"
    assert 'ts = res.get("ts", ts)' in src, "`run_train` 未沿用续跑段的**生效** `ts`"
    a = m.run_id_of(cfg, "id", 42, "20261005-104756")
    b = m.run_id_of(cfg, "id", 42, "20261005-104756")
    c = m.run_id_of(cfg, "id", 42, "20261005-104856")
    assert a == b, "同一 `ts` ⇒ `run_id` 必须相同（续跑落回同一目录）"
    assert a != c, "不同 `ts` ⇒ 不同 `run_id`（正是 `N-8` 分裂的成因）"
    assert "20261005-104756" in a


# ---------------------------------------------------------------------------
# ⑭ `D-54` 裁定三：`04_01` loss-NaN 守卫（工程性改进，非口径变更）
# ---------------------------------------------------------------------------
def test_nan_guard_predicate(m):
    """`D-54` 裁定三②：判据 = `not math.isfinite(loss)`；`±inf` 与 `nan` **均**须命中。"""
    assert m.nan_guard_violated(float("nan")) is True
    assert m.nan_guard_violated(float("inf")) is True
    assert m.nan_guard_violated(float("-inf")) is True
    assert m.nan_guard_violated(1.0) is False
    assert m.nan_guard_violated(0.0) is False
    assert m.nan_guard_violated(-3.25) is False


def test_nan_guard_default_on_and_off_switch(m, m00, cfg):
    """`D-54` 裁定三①：**默认启用**；`--no-nan-guard` 可关（供等价性自证）。"""
    assert cfg["nan_guard"]["enabled"] is True
    assert cfg["nan_guard"]["on_trigger"]["exit_code"] == 3
    a = m.build_parser().parse_args(["--regime", "id", "--seed", "42"])
    assert a.nan_guard is True, "守卫须**默认启用**"
    b = m.build_parser().parse_args(["--regime", "id", "--seed", "42", "--no-nan-guard"])
    assert b.nan_guard is False, "`--no-nan-guard` 须可关闭守卫"
    assert m00.build_parser().parse_args([]).nan_guard is True, "`04_00` 亦须默认启用"


def test_nan_guard_exit_code_3(m, tmp_path, monkeypatch):
    """`D-54` 裁定三③：守卫触发 ⇒ **退出码 3**（与 `0`／`1`／`2` 区分）。

    不加载模型：把 `run_train` 换成返回守卫标记的桩，只验退出码接线。
    """
    monkeypatch.setattr(m, "run_train",
                        lambda cfg, args, logger, ts: (None, {"nan_guard_stopped": True,
                                                              "opt_steps": 7}))
    rc = m.main(["--regime", "id", "--seed", "42",
                 "--report-dir", str(tmp_path / "reports"),
                 "--log-dir", str(tmp_path / "logs"),
                 "--out-dir", str(tmp_path / "ckpt")])
    assert rc == 3, "守卫触发必须返回退出码 3，实得 %r" % rc


def test_nan_guard_stops_before_any_write(m):
    """`D-54` 裁定三③「四不一停」：守卫的早退必须**先于**任何落盘 ⇒ 不落 state／不覆盖 best。

    静态钉死：守卫早退点在源码中**位于**周期性 `state` 落盘**与** `best` checkpoint 落盘**之前**。
    """
    src = _src()
    assert "NAN_GUARD_TRIGGERED" in src, "缺触发日志行"
    i_guard = src.index('"nan_guard_stopped": True')
    i_state = src.index("if resume_dir and state_every > 0 and")
    i_ckpt = src.index('best["dir"] = save_checkpoint(')
    assert i_guard < i_state, "早退必须先于周期性 state 落盘，否则会覆盖干净快照"
    assert i_guard < i_ckpt, "早退必须先于 best checkpoint 落盘"
    guard = src.split('if nan_guard and nan_guard_violated(_lv):', 1)[1].split("return {", 1)[0]
    for bad in ("save_resume_state", "save_checkpoint", "os.remove", "rmtree", "unlink"):
        assert bad not in guard, "守卫触发路径**不得**调用 %s（四不一停）" % bad
    assert 'if _res.get("nan_guard_stopped")' in src and "return 3" in src, "退出码 3 未接线"


# ---------------------------------------------------------------------------
# `W5.6`（`D-83` 裁定四 工作 3）：独立增强开关 `--augment` ＋ `--model-key`
#   红线：默认值下与改造前**逐字节一致**；`--curriculum on` 的既有语义**逐字不变**。
# ---------------------------------------------------------------------------
def test_w56_parser_defaults_do_not_change_behaviour(m):
    args = m.build_parser().parse_args(["--regime", "id", "--seed", "42"])
    assert args.augment == "off", "`--augment` 默认必须 off（否则改到 C 组复现路径）"
    assert args.model_key is None, "`--model-key` 默认必须 None（否则 cfg 被改动）"


def test_w56_run_train_gate_keeps_curriculum_branch_intact(m):
    """`--curriculum on` 走**原分支**（`cur_cls is not None`）⇒ 逐字不变；
    `--augment on` 仅在 `--curriculum off` 下另起 `elif` 分支。"""
    src = _src()
    assert "elif augment_on:" in src, "缺 `--augment` 的独立分支"
    assert "augment_on = str(getattr(args, \"augment\", \"off\")).lower() == \"on\"" in src
    # 原分支的判定式必须原样保留（`04_00`／既有测试以 Namespace 传参，不含 augment）
    assert "if cur_cls is not None:" in src


def test_w56_apply_model_key_override_noop_and_effect(m, cfg):
    before = cfg["checkpoint"]["model_key"]
    assert m.apply_model_key_override(cfg, None) is None
    assert cfg["checkpoint"]["model_key"] == before, "默认 None 不得改动 cfg"
    assert m.apply_model_key_override(cfg, "llamaguard1b_A") == "llamaguard1b_A"
    assert m.run_id_of(cfg, "id", 42, "20261009-160000") == \
        "04_01_llamaguard1b_A_id_seed42_20261009-160000"


def test_w56_apply_model_key_override_requires_key(m):
    with pytest.raises(KeyError):
        m.apply_model_key_override({"checkpoint": {}}, "llamaguard1b_A")

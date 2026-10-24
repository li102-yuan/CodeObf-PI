# -*- coding: utf-8 -*-
"""tests/train/test_04_02_train_curriculum.py — W5.2 `04_02` 的单元与接口测试。

覆盖 `windows/W5.2_curriculum.md` §7-3 列出的要点：

1. **阶段边界**（`stage_at`）与 `describe()` 六字段齐备；
2. `select()` 的累积式语义（stage1 仅 low＋良性、stage3 全量、良性恒入选）；
3. **OOD 编码 0 命中**（分类列 ＋ 内容列两层扫描）；
4. **增强只作用于 train 的 `label == 1` 行**（val 0 命中、原 frame 一行不动）；
5. **确定性**：同 seed 逐字节可复现，异 seed 不同；
6. 正负比**上界**生效且只缩减增强行；上界不可达时**显式报错**而不删原分片行；
7. §5-4 **顶层不 import 重型库**；§3-3 阶段切换**不由 val 触发**（静态）；
8. C11 `--report-dir` 存在、`reports/` 未被写死；
9. `04_01` 的课程接线（`train_rows` 与增强帧）真实可用。

**设计纪律**：本文件**不加载模型权重、不需要 GPU**；读盘用例只用 frozen 四个分片中
较小的一个（`regime_ood/val.csv`），其余用合成 frame。
"""
from __future__ import annotations

import ast
import importlib.util
import os
import random
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # tests/train -> 项目根
CUR_SCRIPT = os.path.join(ROOT, "scripts", "train", "04_02_train_curriculum.py")
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train", "04_01_train_qlora_ours.py")
CONFIG = os.path.join(ROOT, "configs", "train", "curriculum.yaml")
QLORA_CONFIG = os.path.join(ROOT, "configs", "train", "qlora_ours.yaml")

HEAVY = ("torch", "transformers", "peft", "bitsandbytes", "trl", "datasets", "accelerate")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load("t_04_02", CUR_SCRIPT)


@pytest.fixture(scope="module")
def cfg(m):
    return m.load_config(CONFIG)


@pytest.fixture(scope="module")
def src():
    with open(CUR_SCRIPT, "r", encoding="utf-8") as f:
        return f.read()


def _frame(m):
    return m._synth_frame(n_benign=10)


# ---------------------------------------------------------------------------
# ① 配置与阶段计划（事前定值）
# ---------------------------------------------------------------------------
def test_difficulty_map_covers_nine(m, cfg):
    dmap = cfg["encoding_difficulty"]
    assert len(dmap) == 9
    assert set(dmap.values()) <= {"low", "medium", "high"}
    assert dmap["rot13"] == "low" and dmap["reverse"] == "low"
    assert dmap["base64"] == "medium" and dmap["url"] == "medium"
    assert dmap["ascii_art"] == "high" and dmap["zerowidth"] == "high"


def test_regime_domains(m, cfg):
    rid = set(cfg["regimes"]["id"]["train_encodings"])
    rood = set(cfg["regimes"]["ood"]["train_encodings"])
    hold = set(cfg["regimes"]["ood"]["heldout_encodings"])
    assert len(rid) == 9 and len(rood) == 5 and len(hold) == 4
    assert not (rood & hold), "5 训练码与 4 留出码不得相交"
    assert rood | hold == rid
    assert cfg["regimes"]["id"]["heldout_encodings"] == []


def test_augment_params_are_fixed(m, cfg):
    aug = cfg["augment"]
    assert float(aug["p"]) == 0.30, "增强概率事前定值 0.30"
    assert float(aug["ratio_cap"]) == 1.25, "正负比上界事前定值 1.25"
    assert aug["discard_order"] == ["low", "medium", "high"]
    assert aug["discard_scope"] == "augmented_only"
    assert aug["exclude_self_encoding"] is True


def test_plan_is_cumulative_and_fixed(m, cfg):
    plan = cfg["plan"]
    assert plan["n_stages"] == 3 and plan["cumulative"] is True
    props = plan["step_proportions"]
    assert props == [0.3, 0.6, 1.0], "阶段切点是事前定值的固定比例"
    assert len(plan["stages"]) == 3
    for i in range(2):
        assert set(plan["stages"][i]["levels"]) <= set(plan["stages"][i + 1]["levels"])


def test_stage_at_boundaries(m, cfg):
    s = m.CurriculumScheduler(cfg, "id", 10, random.Random(42))
    assert s.ends == [3, 6, 10]
    assert [s.stage_at(x) for x in (0, 1, 3, 4, 6, 7, 10, 11)] == [1, 1, 1, 2, 2, 3, 3, 3]
    s2 = m.CurriculumScheduler(cfg, "id", 1, random.Random(42))
    assert s2.stage_at(1) == 3, "total_steps=1 ⇒ 直接进入最后阶段，不得越界"


def test_describe_has_six_fields(m, cfg):
    d = m.CurriculumScheduler(cfg, "id", 100, random.Random(42)).describe()
    assert len(d) == 3
    for row in d:
        assert set(row) == {"stage", "start_step", "end_step", "difficulty_levels",
                            "encodings", "cumulative"}
        assert row["cumulative"] is True
    assert d[0]["encodings"] == ["reverse", "rot13"]
    assert len(d[1]["encodings"]) == 6
    assert len(d[2]["encodings"]) == 9, "体制一 stage3 覆盖全部 9 码（累积）"


def test_describe_ood_cumulative(m, cfg):
    d = m.CurriculumScheduler(cfg, "ood", 100, random.Random(42)).describe()
    assert d[0]["encodings"] == ["reverse", "rot13"]
    assert d[1]["encodings"] == ["base64", "reverse", "rot13", "url"]
    assert d[2]["encodings"] == ["base64", "reverse", "rot13", "url", "zerowidth"]
    assert "hex" not in d[2]["encodings"] and "morse" not in d[2]["encodings"]


# ---------------------------------------------------------------------------
# ② select() 语义
# ---------------------------------------------------------------------------
def test_select_stage1_low_plus_benign(m, cfg):
    """stage1 只含 low 攻击行；良性行先全量候选，再按 `benign_subsample` 子采样。"""
    # 用小规模帧（6 攻击 / 2 良性）：stage1 为 2 条 low，neg_kept=round(2/1.25)=2 ⇒ 良性全保留
    frame = m._synth_frame(n_benign=2)
    s = m.CurriculumScheduler(cfg, "id", 10, random.Random(42))
    picked = s.select(frame, 1)
    assert all(frame[i]["difficulty"] in ("", "low") for i in picked)
    assert sum(1 for i in picked if frame[i]["label"] == "1") == 2, "2 条 low 攻击行全保留"
    assert len(s.select(frame, 10)) == len(frame), "stage3 全量（累积式，不子采样）"


def test_benign_subsample_ratio_constant_across_stages(m, cfg):
    """`D-42` 裁定三：各阶段正负比恒 = 增强帧比（±0.05）；只减良性；stage3 等价全量帧。"""
    encs = tuple([("rot13", "low")] * 25 + [("base64", "medium")] * 25
                 + [("zerowidth", "high")] * 25)
    frame = m._synth_frame(n_benign=40, encodings=encs)
    s = m.CurriculumScheduler(cfg, "id", 30, random.Random(42))
    rows = [(st["stage"], st["end_step"]) for st in s.describe()]
    for stage, step in rows[:-1]:                 # 末阶段不子采样
        n_pos, n_neg, ratio, target, ok = s.ratio_report(frame, step)
        assert ok, "stage%d 比 %.4f 越出目标 %.4f±%.2f" % (stage, ratio, target, s.bs_tol)
    n_pos1 = s.ratio_report(frame, rows[0][1])[0]
    n_pos3 = s.ratio_report(frame, rows[-1][1])[0]
    assert n_pos1 == 25, "攻击行不得被删减（stage1）"
    assert n_pos3 == 75, "攻击行不得被删减（stage3）"
    n_pos3b, n_neg3b, _r3, _t3, _ok3 = s.ratio_report(frame, rows[-1][1])
    assert (n_pos3b + n_neg3b) == len(frame), "stage3 必须等价全量帧"


def test_benign_subsample_only_reduces_benign(m, cfg):
    """子采样必须**只减良性**：各阶段选中的攻击行集合 = 该档位累积的全部攻击行。"""
    encs = tuple([("rot13", "low")] * 25 + [("base64", "medium")] * 25
                 + [("zerowidth", "high")] * 25)
    frame = m._synth_frame(n_benign=40, encodings=encs)
    s = m.CurriculumScheduler(cfg, "id", 30, random.Random(42))
    levels1 = set(s.stage_levels(1))
    want1 = {i for i, r in enumerate(frame) if r["difficulty"] in levels1}
    got1 = set(s.select(frame, 1))
    assert want1 <= got1, "stage1 必须包含全部 low 攻击行"
    assert all(i in want1 or frame[i]["difficulty"] == "" for i in got1)


def test_benign_subsample_disabled_keeps_all_benign(m, cfg):
    """关掉 `benign_subsample` ⇒ 退回「良性恒入选」旧口径（9 倍漂移）。"""
    cfg2 = dict(cfg)
    cfg2["plan"] = dict(cfg["plan"])
    cfg2["plan"]["benign_subsample"] = dict(cfg["plan"]["benign_subsample"], enabled=False)
    frame = m._synth_frame(n_benign=10)
    s = m.CurriculumScheduler(cfg2, "id", 10, random.Random(42))
    picked = s.select(frame, 1)
    assert sum(1 for i in picked if frame[i]["label"] == "0") == 10


def test_select_rejects_none_frame(m, cfg):
    s = m.CurriculumScheduler(cfg, "id", 10, random.Random(42))
    with pytest.raises(ValueError):
        s.select(None, 1)


def test_select_benign_flag_off(m, cfg):
    cfg2 = dict(cfg)
    cfg2["plan"] = dict(cfg["plan"])
    cfg2["plan"]["benign_always_included"] = False
    frame = m._synth_frame(n_benign=2)
    s = m.CurriculumScheduler(cfg2, "id", 10, random.Random(42))
    picked = s.select(frame, 1)
    assert all(frame[i]["label"] == "1" for i in picked)


# ---------------------------------------------------------------------------
# ③ 增强：只作用于 train 的攻击行
# ---------------------------------------------------------------------------
def test_augment_only_touches_label_one(m, cfg):
    frame = _frame(m)
    out, rep = m.augment_attack(frame, random.Random(42),
                                ["base64", "rot13", "reverse"], 1.0, 9.0, cfg=cfg)
    assert rep["augmented_ok"] > 0
    assert all(str(r["label"]) == "1" for r in out if r.get("augmented") == "yes")
    assert out[:len(frame)] == frame, "追加式：原 frame 一行不动且在最前"


def test_augment_marks_and_suffix(m, cfg):
    frame = _frame(m)
    out, _rep = m.augment_attack(frame, random.Random(7), ["base64", "rot13", "reverse"],
                                 1.0, 9.0, cfg=cfg)
    aug = [r for r in out if r.get("augmented") == "yes"]
    assert aug
    for r in aug:
        assert r["sample_id"].endswith("__aug-%s" % r["encoding_type"])
        assert r["aug_from_sample_id"] in {f["sample_id"] for f in frame}
        assert r["aug_from_encoding"] != r["encoding_type"], "不得与自身编码相同"
        assert r["payload"] in r["text"], "增强后 payload 仍是 text 的锚点子串"


def test_augment_skips_missing_anchor(m, cfg):
    # 须带 ≥1 良性行：`discard_scope = augmented_only` 下无良性行则正负比恒不可达，
    # `augment_attack` 会按「上界不可达」显式报错（见 `test_ratio_cap_unreachable_raises`）。
    frame = [{"sample_id": "X1", "text": "host without anchor", "label": "1",
              "payload": "NOT-IN-TEXT", "encoding_type": "rot13", "difficulty": "low"},
             {"sample_id": "B1", "text": "benign host", "label": "0",
              "payload": "", "encoding_type": "", "difficulty": ""}]
    out, rep = m.augment_attack(frame, random.Random(1), ["base64"], 1.0, 9.0, cfg=cfg)
    assert rep["skipped"]["anchor_missing"] == 1
    assert rep["augmented_ok"] == 0 and out == frame


def test_augment_no_benign_row_raises_unreachable_cap(m, cfg):
    """frame 内**无良性行** ⇒ 正负比恒 inf，`discard_scope = augmented_only` 下不可达 ⇒ 显式报错。"""
    frame = [{"sample_id": "X1", "text": "host", "label": "1", "payload": "host",
              "encoding_type": "rot13", "difficulty": "low"}]
    with pytest.raises(ValueError) as exc:
        m.augment_attack(frame, random.Random(1), ["base64"], 1.0, 1.25, cfg=cfg)
    assert "不可达" in str(exc.value)


# ---------------------------------------------------------------------------
# ④ 正负比上界
# ---------------------------------------------------------------------------
def test_ratio_cap_discards_augmented_only(m, cfg):
    frame = _frame(m)                      # 6 攻击 / 10 良性 = 0.6 < 0.75 ⇒ 上界可达
    out, rep = m.augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                                1.0, 0.75, cfg=cfg)
    pos = sum(1 for r in out if str(r["label"]) == "1")
    neg = len(out) - pos
    assert neg == 10, "良性行一条不动"
    assert float(pos) / neg <= 0.75 + 1e-12
    assert rep["n_discarded"] > 0
    assert sum(1 for r in out if r.get("augmented") == "yes") == (
        rep["augmented_ok"] - rep["n_discarded"])


def test_ratio_cap_unreachable_raises(m, cfg):
    frame = m._synth_frame(n_benign=2)     # 6 攻击 / 2 良性 = 3.0 > 0.75 ⇒ 不可达
    with pytest.raises(ValueError) as exc:
        m.augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                         1.0, 0.75, cfg=cfg)
    assert "不可达" in str(exc.value)


# ---------------------------------------------------------------------------
# ⑤ 确定性
# ---------------------------------------------------------------------------
def test_augment_deterministic_same_seed(m, cfg):
    frame = _frame(m)
    a = m.augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                         0.5, 1.25, cfg=cfg)
    b = m.augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                         0.5, 1.25, cfg=cfg)
    c = m.augment_attack(frame, random.Random(123), ["base64", "rot13", "reverse"],
                         0.5, 1.25, cfg=cfg)
    assert a[1]["frame_digest"] == b[1]["frame_digest"]
    assert m.frame_digest(a[0]) == m.frame_digest(b[0])
    assert a[1]["frame_digest"] != c[1]["frame_digest"]


def test_frame_digest_order_sensitive(m):
    rows = [{"sample_id": "a", "text": "1", "payload": "", "encoding_type": "",
             "difficulty": ""},
            {"sample_id": "b", "text": "2", "payload": "", "encoding_type": "",
             "difficulty": ""}]
    assert m.frame_digest(rows) != m.frame_digest(list(reversed(rows)))


# ---------------------------------------------------------------------------
# ⑥ 一号门禁：OOD 留出码 0 命中
# ---------------------------------------------------------------------------
def test_ood_augment_never_introduces_heldout(m, cfg):
    frame = m._synth_ood_frame()
    allowed = cfg["regimes"]["ood"]["train_encodings"]
    out, _rep = m.augment_attack(frame, random.Random(42), allowed, 1.0, 1.25, cfg=cfg)
    ok, detail = m.assert_heldout_zero(out, cfg["regimes"]["ood"]["heldout_encodings"],
                                       cfg["regimes"]["ood"]["scan_columns"],
                                       out, "synth")
    assert ok, detail
    assert set(r["encoding_type"] for r in out if r.get("augmented") == "yes") <= set(allowed)


def test_heldout_reference_excludes_plaintext(m):
    rows = [{"label": "1", "encoding_type": "hex", "payload": "P", "text": "T",
             "plaintext": "SHARED"}]
    ref = m.heldout_reference(rows, ["hex"])
    assert ref == {"P", "T"}, "`plaintext` 不得并入参照集（假阳性源，`02_07` 同制）"


def test_heldout_reference_empty_when_no_heldout(m):
    rows = [{"label": "1", "encoding_type": "hex", "payload": "P", "text": "T"}]
    assert m.heldout_reference(rows, []) == set()


def test_assert_heldout_zero_detects_hit(m):
    rows = [{"sample_id": "X", "encoding_type": "hex", "text": "abc", "plaintext": "abc"}]
    ok, detail = m.assert_heldout_zero(rows, ["hex"], ["text", "plaintext"], [], "t")
    assert not ok and "分类列命中=1" in detail


# ---------------------------------------------------------------------------
# ⑦ 输入面守卫
# ---------------------------------------------------------------------------
def test_split_guard_blocks_forbidden(m, cfg):
    for bad in ("data/sealed_test/CodeObf-PI_v1.0/splits/regime_id/test.csv",
                "data/frozen/CodeObf-PI_v1.0/splits/regime_id/test.csv",
                "data/frozen/CodeObf-PI_v1.0/splits/regime_ood/test.csv"):
        with pytest.raises(RuntimeError):
            m.assert_split_allowed(os.path.join(ROOT, bad), cfg)
    for good in ("regime_id/train.csv", "regime_id/val.csv",
                 "regime_ood/train.csv", "regime_ood/val.csv"):
        assert m.assert_split_allowed(
            os.path.join(ROOT, "data", "frozen", "CodeObf-PI_v1.0", "splits", good), cfg)


# ---------------------------------------------------------------------------
# ⑧ 静态约束
# ---------------------------------------------------------------------------
def test_no_heavy_imports_at_top_level(src):
    tree = ast.parse(src)
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not (names & set(HEAVY)), "顶层不得 import 重型库：%s" % (names & set(HEAVY))
    assert "yaml" not in names, "`yaml` 只许在 `load_config()` 内惰性 import"


def test_report_dir_not_hardcoded(src):
    assert '"reports/' not in src and "'reports/" not in src
    assert '"reports\\' not in src and "'reports\\" not in src
    assert "--report-dir" in src and "--log-dir" in src


def _docstring_nodes(tree):
    """收集模块／类／函数的 docstring 节点（说明文字里出现 `val` 属正常，**不参与**静态扫描）。"""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            out.add(id(first.value))
    return out


def test_no_val_triggered_stage_switch(src, cfg):
    """§3-3：阶段切换**不得由 val 指标触发**（三重静态把关）。

    ① 源码不出现任何 val 监控量的名字；
    ② 非 docstring 的字符串常量里不出现 val 指标名（`"val"` 作为分片名是合法的，故只禁指标名）；
    ③ `CurriculumScheduler` 的公开方法**只接受 step**，不接受任何 metric 参数。
    """
    val_metrics = ("val_f1", "best_f1", "val_metric", "val_loss", "val_acc", "val_score",
                   "val_macro_f1", "early_stop", "patience")
    lowered = src.lower()
    for name in val_metrics:
        assert name not in lowered, "模块内出现 val 监控量 `%s` ⇒ 有按 val 切阶段之嫌" % name

    tree = ast.parse(src)
    docs = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docs:
                continue
            low = node.value.lower()
            for name in val_metrics:
                assert name not in low, "字符串常量含 val 指标名 `%s`：%r" % (name, node.value)

    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "CurriculumScheduler")
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
            continue
        args = [a.arg for a in fn.args.args]
        if fn.name in ("stage_at", "select"):
            assert "step" in args, "`%s` 必须接受 step" % fn.name
            assert not (set(args) & {"metric", "val_f1", "score", "f1"}), \
                "`%s` 不得接受任何 val 指标参数：%s" % (fn.name, args)
    method_names = {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}
    assert not (method_names & {"advance_by_metric", "update_by_val", "step_by_val"}), \
        "不得提供按 val 推进阶段的方法：%s" % (method_names,)


def test_no_bare_except(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None, "禁止裸 except"
            body = [n for n in node.body if not isinstance(n, ast.Pass)]
            assert body, "禁止 `except …: pass`"


# ---------------------------------------------------------------------------
# ⑨ dry-run（真实分片；含 val 0 命中与六项断言）
# ---------------------------------------------------------------------------
def test_dry_run_ood_six_assertions(m, cfg):
    """真实 frozen 分片上的六项断言（约 1.5 s：纯数据调度，不加载模型）。"""
    ctx, ok = m.run_dry_run(cfg, "ood", 42, 100, verbose=False)
    assert ok, [a for a in ctx["assertions"] if not a[1]]
    assert ctx["val_aug_hits"] == 0
    assert ctx["assertions"][0][1], ctx["assertions"][0][2]     # ① 0 命中
    assert float(ctx["report"]["ratio_after"]) <= 1.25 + 1e-12  # ② 上界
    assert ctx["assertions"][3][1]                             # ④ 确定性
    assert ctx["assertions"][4][1], ctx["assertions"][4][2]     # ⑤ 各阶段比恒定
    assert ctx["assertions"][5][1], ctx["assertions"][5][2]     # ⑥ stage3 等价全量
    assert len(ctx["plan_rows"]) == 3 and len(ctx["stage_rows"]) == 3
    # 各阶段比均落在目标 ± tol
    for sr in ctx["stage_rows"]:
        assert sr["ratio_ok"], sr


def test_dry_run_rejects_unknown_regime(m, cfg):
    with pytest.raises(KeyError):
        m.run_dry_run(cfg, "bogus", 42, 10, verbose=False)


# ---------------------------------------------------------------------------
# ⑩ 与 `04_01` 的接线（W5.2 补）
# ---------------------------------------------------------------------------
def test_04_01_wires_curriculum_frame():
    t401 = _load("t_04_01_wire", TRAIN_SCRIPT)
    assert hasattr(t401, "curriculum_module")
    assert hasattr(t401, "build_curriculum_frame")
    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
        s401 = f.read()
    assert "train_rows=train_rows" in s401, "`04_01` 须把增强帧传给 `select()`"
    assert "rows=train_rows" in s401, "`04_01` 须用增强后的行编码 train_exs"
    assert "n_curriculum_skipped" in s401, "整批被排除时须跳过，不得回退成全量"


def test_04_01_build_curriculum_frame_real(m):
    t401 = _load("t_04_01_wire", TRAIN_SCRIPT)
    cfg401 = t401.load_config(QLORA_CONFIG)
    logs = []
    train_csv = t401.resolve_split_paths(cfg401, "ood")[0]
    rows, rep = t401.build_curriculum_frame(cfg401, m, "ood", train_csv, None, 42,
                                            lambda s: logs.append(s))
    assert rows is not None and rep is not None
    assert len(rows) == rep["n_output_rows"] > rep["n_input_rows"]
    assert all(str(r["label"]) == "1" for r in rows if r.get("augmented") == "yes")
    assert float(rep["ratio_after"]) <= 1.25 + 1e-12
    assert logs and logs[0].startswith("CURRICULUM_AUGMENT")


def test_04_01_curriculum_off_keeps_n_rows():
    """`--curriculum off` ⇒ `build_curriculum_frame` 不被调用 ⇒ 训练行数不变。"""
    t401 = _load("t_04_01_wire", TRAIN_SCRIPT)
    cfg401 = t401.load_config(QLORA_CONFIG)
    assert str(cfg401["curriculum"]["default"]).lower() == "off"
    rows, rep = t401.build_curriculum_frame(cfg401, None, "ood", "x", None, 42,
                                            lambda s: None)
    assert rows is None and rep is None

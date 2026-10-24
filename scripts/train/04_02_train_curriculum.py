# -*- coding: utf-8 -*-
"""04_02_train_curriculum.py — W5.2 课程学习与编码感知增强子模块。

职责（`windows/W5.2_curriculum.md` §1／§2）：**只被 `04_01` import，不单独执行训练**。
事前定值载体 = `configs/train/curriculum.yaml`（阶段计划＋增强参数）。

接口契约（**W5 冻结 v1.0**）::

    class CurriculumScheduler:
        def __init__(self, cfg, regime: str, total_steps: int, rng: random.Random): ...
        def stage_at(self, step: int) -> int: ...            # 1..N
        def describe(self) -> list[dict]: ...                # 6 字段
        def select(self, frame, step: int) -> list[int]: ... # 行下标

    def augment_attack(frame, rng, allowed_encodings, p, ratio_cap) -> (new_frame, report)

冻结口径要点（逐条落点见各函数 docstring 与 `curriculum.yaml` 的行内注释）：

- **C1／§5-1**：只读 4 个 frozen 分片；**禁读** `data/sealed_test/**` 与 frozen 侧 `test.csv`（含表头）。
- **§3-2**：3 阶段**累积式**（后一阶段含前一阶段全部样本）。
- **§3-3**：阶段切换 = **事前定值的固定 step 比例**，**禁止**用 val 指标触发。
- **§3-4**：体制二 `allowed_encodings` 恒为 5 个训练码；4 个留出码在 train 侧命中数 = 0（**一号门禁**）。
- **§4-2**：增强为**追加式**（保留原样本、追加变体），`p = 0.30`。
- **§4-3**：增强后 train 正负比 **≤ 1.25:1**；超出按 `low → medium → high` 缩减**增强行**。
- **§4-4**：增强只作用于 **train 分片的 `label == 1` 行**；val 不增强、不参与。
- **§4-6**：增强样本**不**计入跨检测器 `text_sha256` 一致性校验（以 `augmented` 标记列区分）。
- **§5-4**：**顶层不 import 任何重型库**（torch／transformers／yaml 一律惰性 import）。
- **§5-6**：确定性 —— 显式 `random.Random(seed)`，**不用全局 RNG**。

**不在本文件做的事**：训练循环（`04_01`）、3 seeds（`W5.3`）、遗忘判定（`W5.4`）、任何测试集结论。
"""
from __future__ import annotations

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

#: 课程计划的默认配置路径（相对项目根）。`CurriculumScheduler` 被 `04_01` 调用时，
#: 优先取 `cfg["curriculum"]["config_path"]`（`configs/train/qlora_ours.yaml`），缺省回落此处。
DEFAULT_PLAN_CONFIG = os.path.join("configs", "train", "curriculum.yaml")

# §5-4：这些库一律不得在顶层出现（`yaml` 只在 `load_config()` 内惰性 import，故不在表内）
_HEAVY = ("torch", "transformers", "peft", "bitsandbytes", "trl", "datasets", "accelerate")


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：W3.7 `F-1`；不得静默 `pass`）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8。只动输出流。失败即显式告警。"""
    problems = []
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            problems.append("%s 不支持 reconfigure" % name)
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:
            problems.append("%s 重配失败：%s" % (name, ascii(exc)))
    if problems:
        sys.stderr.write("WARN init_streams: %s\n" % "; ".join(problems))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


# ---------------------------------------------------------------------------
# 惰性依赖（`03_06` 读取／指纹；`02_03` 编码器 = `V-5` 单源）
# ---------------------------------------------------------------------------
_LAZY = {}


def _preprocess():
    if "pre" not in _LAZY:
        _LAZY["pre"] = _load_module("curriculum_preprocess_04_02",
                                    os.path.join(_ROOT, "scripts", "baseline",
                                                 "03_06_unify_preprocess.py"))
    return _LAZY["pre"]


def _encoders():
    """`V-5` 的 9 个编码器 —— **复用** `scripts/data/02_03_encode_variants.py`，不自造。"""
    if "enc" not in _LAZY:
        mod = _load_module("curriculum_encoders_04_02",
                           os.path.join(_ROOT, "scripts", "data",
                                        "02_03_encode_variants.py"))
        _LAZY["enc"] = mod.ENCODERS
    return _LAZY["enc"]


def load_rows(csv_path):
    """C2：读取走 `03_06.load_rows`（原文直通，一行一 dict）。"""
    return _preprocess().load_rows(csv_path)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_config(path):
    """读 YAML。**禁止默认值兜底**：文件不存在／键缺失一律显式报错。"""
    path = _abs(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("配置文件不存在：%s" % path)
    try:
        import yaml  # 惰性：本模块顶层不 import 重型库
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 PyYAML（conda 环境 `deepseek`）：%s" % ascii(exc))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置不是映射：%s" % path)
    return cfg


def _require(mapping, key, where):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError("配置缺键 `%s`（节：%s）" % (key, where))
    return mapping[key]


def resolve_plan_cfg(cfg):
    """从任意形态的 cfg 解析出 `curriculum.yaml` 本体。

    接受三种形态：① 本体（含 `plan`＋`regimes`）；② 含 `curriculum_plan` 的包装；
    ③ `04_01` 的 `qlora_ours.yaml`（含 `curriculum.config_path`）。
    三者皆无 ⇒ 显式报错（不静默回落任何数值）。
    """
    if not isinstance(cfg, dict):
        raise TypeError("resolve_plan_cfg: cfg 必须是映射，得到 %s" % type(cfg).__name__)
    if "plan" in cfg and "regimes" in cfg:
        return cfg
    if isinstance(cfg.get("curriculum_plan"), dict):
        return cfg["curriculum_plan"]
    cur = cfg.get("curriculum")
    if isinstance(cur, dict) and isinstance(cur.get("plan"), dict):
        return cur
    if isinstance(cur, dict) and cur.get("config_path"):
        return load_config(cur["config_path"])
    raise KeyError("无法解析课程计划：cfg 须为 curriculum.yaml 本体、含 `curriculum_plan`，"
                   "或含 `curriculum.config_path`")


def default_plan_cfg():
    """加载默认课程配置（`configs/train/curriculum.yaml`）。"""
    if "plan_cfg" not in _LAZY:
        _LAZY["plan_cfg"] = load_config(DEFAULT_PLAN_CONFIG)
    return _LAZY["plan_cfg"]


# ---------------------------------------------------------------------------
# C1／§5-1 输入面守卫
# ---------------------------------------------------------------------------
def assert_split_allowed(path, cfg=None):
    """只允许 4 个 frozen 分片；`sealed_test` 与 `test.csv` 命中即抛错。"""
    cfg = cfg if isinstance(cfg, dict) else default_plan_cfg()
    paths = _require(cfg, "paths", "cfg")
    norm = os.path.abspath(path).replace("\\", "/")
    for bad in _require(paths, "forbidden_substrings", "paths"):
        if str(bad).replace("\\", "/") in norm:
            raise RuntimeError("越权读取：%s（命中禁读子串 `%s`；A5 只读 4 个 train/val 分片）"
                               % (norm, bad))
    allowed = {os.path.abspath(os.path.join(_ROOT,
                                            _require(paths, "frozen_splits", "paths"),
                                            s)).replace("\\", "/")
               for s in _require(paths, "allowed_splits", "paths")}
    if norm not in allowed:
        raise RuntimeError("越权读取：%s（不在 `paths.allowed_splits` 的 4 个分片内）" % norm)
    return path


def split_path(cfg, regime, split):
    return os.path.join(_ROOT, _require(cfg, "paths", "cfg")["frozen_splits"],
                        "regime_%s" % regime, "%s.csv" % split)


# ---------------------------------------------------------------------------
# 课程调度器
# ---------------------------------------------------------------------------
class CurriculumScheduler(object):
    """3 阶段累积式课程；阶段切换 = **事前定值的固定 step 比例**（不由 val 触发）。

    ``stage_at(step)`` 的 ``step`` 是 **1 基** 的 optimizer step（`04_01` 传
    ``opt_steps + 1``）。边界 ``end_i = floor(比例_i × total_steps)``，单调递增，末位 = ``total_steps``。
    """

    def __init__(self, cfg, regime, total_steps, rng):
        plan_cfg = resolve_plan_cfg(cfg)
        self.plan_cfg = plan_cfg
        self.regime = str(regime)
        if self.regime not in ("id", "ood"):
            raise ValueError("regime 必须是 id／ood，得到 %r" % (regime,))
        self.rng = rng
        self.total_steps = max(1, int(total_steps))

        self.difficulty_map = dict(_require(plan_cfg, "encoding_difficulty", "cfg"))
        self.levels_domain = list(_require(plan_cfg, "difficulty_levels", "cfg"))
        self.regime_cfg = _require(_require(plan_cfg, "regimes", "cfg"), self.regime,
                                   "regimes")
        self.train_encodings = list(_require(self.regime_cfg, "train_encodings",
                                             "regimes.%s" % self.regime))
        self.heldout = list(self.regime_cfg.get("heldout_encodings") or [])

        plan = _require(plan_cfg, "plan", "cfg")
        self.cumulative = bool(_require(plan, "cumulative", "plan"))
        props = [float(x) for x in _require(plan, "step_proportions", "plan")]
        stages = list(_require(plan, "stages", "plan"))
        if len(props) != len(stages):
            raise ValueError("step_proportions(%d) 与 stages(%d) 数量不一致"
                             % (len(props), len(stages)))
        if abs(props[-1] - 1.0) > 1e-12:
            raise ValueError("step_proportions 末位必须是 1.0，得到 %r" % (props[-1],))
        self.stages = stages
        self.props = props
        self.benign_always = bool(plan.get("benign_always_included", True))

        # --- W5.2 补正轮（`D-42` 裁定三）：良性子采样，使各阶段正负比恒 = 增强后训练帧比 ---
        self.bs_cfg = dict(plan.get("benign_subsample") or {})
        self.bs_enabled = bool(self.bs_cfg.get("enabled", False))
        self.bs_tol = float(self.bs_cfg.get("tolerance", 0.05))
        self.bs_min_kept = int(self.bs_cfg.get("min_benign_kept", 1))
        self.bs_stage3_full = bool(self.bs_cfg.get("stage3_equals_full_frame", True))
        _trm = dict(self.bs_cfg.get("target_ratio_measured") or {})
        self.bs_target_ratio = float(_trm.get(self.regime, 0.0) or 0.0)
        if self.bs_enabled and self.bs_target_ratio <= 0.0:
            raise ValueError("benign_subsample.target_ratio_measured 缺 regime=%r 的正数目标比"
                             % (self.regime,))

        ends = []
        prev = 0
        for prop in props:
            e = int(math.floor(prop * self.total_steps))
            e = max(e, prev)
            ends.append(e)
            prev = e
        ends[-1] = max(ends[-1], self.total_steps)
        self.ends = ends

    # -- 阶段 ---------------------------------------------------------------
    def stage_at(self, step):
        """1 基 step → 阶段号（1..N）。越界（step ≥ total_steps）取最后阶段。"""
        try:
            s = int(step)
        except (TypeError, ValueError):
            raise ValueError("stage_at: step 无法解析为整数：%r" % (step,))
        if s < 1:
            s = 1
        for i, end in enumerate(self.ends):
            if s <= end:
                return i + 1
        return len(self.ends)

    def stage_levels(self, stage):
        idx = max(0, min(int(stage) - 1, len(self.stages) - 1))
        return list(_require(self.stages[idx], "levels", "stages[%d]" % idx))

    # -- 报告 ---------------------------------------------------------------
    def describe(self):
        """固定 6 字段 × n_stages 行：`stage`／`start_step`／`end_step`／
        `difficulty_levels`／`encodings`／`cumulative`。"""
        out = []
        start = 1
        for i, st in enumerate(self.stages):
            levels = list(_require(st, "levels", "stages[%d]" % i))
            encs = sorted(e for e in self.train_encodings
                          if self.difficulty_map.get(e) in levels)
            out.append({"stage": int(_require(st, "stage", "stages[%d]" % i)),
                        "start_step": start,
                        "end_step": int(self.ends[i]),
                        "difficulty_levels": levels,
                        "encodings": encs,
                        "cumulative": self.cumulative})
            start = int(self.ends[i]) + 1
        return out

    # -- 选行 ---------------------------------------------------------------
    def select(self, frame, step):
        """返回该 step 允许参与训练的**行下标**。

        口径（§3-2 累积式 ＋ `plan.benign_always_included` ＋ `D-42` 裁定三 良性子采样）：
        - 攻击行（`difficulty` ∈ 封闭域）按当前阶段的累积档位过滤；**一律不删减**；
        - 良性行 `difficulty` 恒为空，先按 `benign_always_included` 全量候选；
          若启用 `benign_subsample`，则把良性候选**确定性前缀**子采样到
          `neg_kept = round(pos_kept / target_ratio)`（夹在 `[min_benign_kept, neg_total]`），
          使该阶段正负比 ≈ `target_ratio`（增强后训练帧比，容差 ±`tolerance`）；
        - **stage 3 不子采样**（`stage3_equals_full_frame`）⇒ 恒等价于全量帧。
        本方法**不做** batching／采样 —— 那是 `04_01` 的事（`W5.2` §2）。
        """
        if frame is None:
            raise ValueError("select(): frame 为 None —— `04_01` 须传入 train 行（list[dict]）")
        stage = self.stage_at(step)
        levels = set(self.stage_levels(stage))
        pos = []
        neg = []
        for i, row in enumerate(frame):
            d = (row.get("difficulty") or "").strip()
            if d == "":
                if self.benign_always:
                    neg.append(i)
                continue
            if d in levels:
                pos.append(i)
        if not (self.bs_enabled and self.bs_cfg):
            return pos + neg
        # stage 3（末阶段）不子采样 ⇒ 等价全量帧
        if self.bs_stage3_full and stage >= len(self.stages):
            return pos + neg
        neg_kept = self._benign_keep_count(len(pos), len(neg))
        # 确定性前缀：按行下标升序取前 neg_kept 个良性行（`neg` 已是升序）
        return pos + neg[:neg_kept]

    def _benign_keep_count(self, n_pos, n_neg):
        """良性行应保留的条数：`round(n_pos / target_ratio)`，夹在 `[min_kept, n_neg]`。"""
        if n_neg <= 0:
            return 0
        want = int(round(float(n_pos) / float(self.bs_target_ratio)))
        want = max(self.bs_min_kept, min(want, n_neg))
        return want

    def ratio_report(self, frame, step):
        """给定 frame／step，返回该阶段入选的 ``(n_pos, n_neg, ratio, target, ok)``。

        ``ok`` = 正负比落在 `target_ratio ± tolerance` 内（或 stage 3 等价全量帧）。
        """
        idxs = self.select(frame, step)
        n_pos = sum(1 for i in idxs if (frame[i].get("difficulty") or "").strip() != "")
        n_neg = len(idxs) - n_pos
        ratio = (float(n_pos) / n_neg) if n_neg else float("inf")
        stage = self.stage_at(step)
        if self.bs_stage3_full and stage >= len(self.stages):
            return n_pos, n_neg, ratio, self.bs_target_ratio, True
        ok = abs(ratio - self.bs_target_ratio) <= self.bs_tol + 1e-9
        return n_pos, n_neg, ratio, self.bs_target_ratio, bool(ok)


# ---------------------------------------------------------------------------
# 编码感知增强（追加式）
# ---------------------------------------------------------------------------
def frame_digest(rows):
    """确定性指纹：逐行 `sample_id／text／payload／encoding_type／difficulty` 的 SHA-256。"""
    h = hashlib.sha256()
    for r in rows:
        h.update("\x1e".join([r.get("sample_id") or "", r.get("text") or "",
                              r.get("payload") or "", r.get("encoding_type") or "",
                              r.get("difficulty") or ""]).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _hist(rows, key):
    out = {}
    for r in rows:
        k = (r.get(key) or "").strip()
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _counts(rows):
    pos = sum(1 for r in rows if str(r.get("label")).strip() == "1")
    return pos, len(rows) - pos


def augment_attack(frame, rng, allowed_encodings, p, ratio_cap, cfg=None,
                   difficulty_map=None):
    """对 **train 分片、`label == 1` 的攻击行** 施加二次编码，**追加式**返回新 frame。

    口径（`W5.2` §4 ＋ `curriculum.yaml` §4）：
    - 形态＝**二次编码**：对现有 `payload` 再施加一种 `allowed_encodings` 内的编码；
    - **不得**与样本自身 `encoding_type` 相同（`exclude_self_encoding`），避免退化为恒等变换；
    - 替换锚点 = `payload` 在 `text` 中的子串；锚点缺失即**跳过并计数**（不猜测、不造锚点）；
    - 追加式：原 frame **一行不动**，增强行追加在尾部（顺序 = 原 frame 的遍历序）；
    - 超 `ratio_cap` 时按 `discard_order` 缩减**增强行**（`discard_scope = augmented_only`）；
    - 增强行的 `encoding_type`／`difficulty` 取**所施加**编码（`augmented_difficulty`）；
    - 返回 `(new_frame, report)`；`report` 含逐项计数与前后分布，供运行报告落盘。
    """
    cfg = cfg if isinstance(cfg, dict) else default_plan_cfg()
    aug = _require(cfg, "augment", "cfg")
    dmap = dict(difficulty_map if difficulty_map
                else _require(cfg, "encoding_difficulty", "cfg"))
    allowed = [str(e) for e in (allowed_encodings or [])]
    if not allowed:
        raise ValueError("augment_attack: allowed_encodings 为空")
    unknown = [e for e in allowed if e not in dmap]
    if unknown:
        raise ValueError("augment_attack: allowed_encodings 含未知编码 %s" % (unknown,))
    prob = float(p)
    cap = float(ratio_cap)
    if not (0.0 <= prob <= 1.0):
        raise ValueError("augment_attack: p 必须在 [0,1]，得到 %r" % (p,))
    if cap <= 0:
        raise ValueError("augment_attack: ratio_cap 必须 > 0，得到 %r" % (ratio_cap,))

    encoders = _encoders()
    suffix_fmt = str(_require(aug, "id_suffix_format", "augment"))
    mark_col = str(_require(aug, "mark_column", "augment"))
    replace_all = bool(aug.get("replace_all", True))
    exclude_self = bool(aug.get("exclude_self_encoding", True))
    discard_order = list(_require(aug, "discard_order", "augment"))
    order = {lv: i for i, lv in enumerate(discard_order)}

    n_input = len(frame)
    pos0, neg0 = _counts(frame)
    # 上界**可达性**：`discard_scope = augmented_only` ⇒ 只能降到「原分片自身的正负比」。
    # 若原分片自身就已超过上界，则丢弃全部增强行仍不可能达标 —— 这时必须**显式暴露**，
    # 绝不通过删减原分片行来达标（缩减训练数据属口径变更，不是工程性偏离）。
    base_ratio = (float(pos0) / neg0) if neg0 else float("inf")
    cap_reachable = bool(neg0 > 0 and base_ratio <= cap + 1e-12)
    new_rows = list(frame)
    added = []                       # [(index_in_new_rows, row)]
    stats = {"drawn": 0, "ok": 0, "anchor_missing": 0, "empty_payload": 0,
             "no_choice": 0, "encode_error": 0, "last_encode_error": ""}
    applied = {}

    for row in frame:
        if str(row.get("label")).strip() != "1":
            continue                                   # §4-4：只作用于攻击行
        if rng.random() >= prob:
            continue
        stats["drawn"] += 1
        own = (row.get("encoding_type") or "").strip()
        choices = sorted(e for e in allowed if (e != own or not exclude_self))
        if not choices:
            stats["no_choice"] += 1
            continue
        enc = choices[rng.randrange(len(choices))]     # 显式 rng ⇒ 确定性
        payload = row.get("payload") or ""
        text = row.get("text") or ""
        if not payload:
            stats["empty_payload"] += 1
            continue
        if payload not in text:
            stats["anchor_missing"] += 1               # §4 anchor_missing_policy = skip
            continue
        try:
            new_payload = encoders[enc][0](payload)
        except Exception as exc:                       # 编码失败：逐条计数，不静默
            stats["encode_error"] += 1
            stats["last_encode_error"] = "%s/%s: %s" % (row.get("sample_id"), enc, ascii(exc))
            continue
        new_text = (text.replace(payload, new_payload) if replace_all
                    else text.replace(payload, new_payload, 1))
        nr = dict(row)
        nr["text"] = new_text
        nr["payload"] = new_payload
        nr["encoding_type"] = enc
        nr["difficulty"] = dmap[enc]
        nr["sample_id"] = suffix_fmt.format(orig=row.get("sample_id") or "", enc=enc)
        nr[mark_col] = "yes"
        nr["aug_from_encoding"] = own
        nr["aug_from_sample_id"] = row.get("sample_id") or ""
        new_rows.append(nr)
        added.append((len(new_rows) - 1, nr))
        stats["ok"] += 1
        applied[enc] = applied.get(enc, 0) + 1

    # ---- 正负比上界（§4-3）：只缩减增强行 ----
    pos1, neg1 = _counts(new_rows)
    ratio_pre_cap = (float(pos1) / neg1) if neg1 else float("inf")
    discarded = {lv: 0 for lv in discard_order}
    binding = bool(ratio_pre_cap > cap)
    if binding and not cap_reachable:
        raise ValueError(
            "augment_attack: 正负比上界 %.4f **不可达** —— 原分片自身正负比已为 %.4f"
            "（pos:neg=%d:%d），而 `discard_scope = augmented_only` 禁止删减原分片行。"
            "处置：改增强概率 p 或呈 W0 改上界；**不得**通过缩减原分片达标。"
            % (cap, base_ratio, pos0, neg0))
    if binding and neg1 > 0:
        max_pos = int(math.floor(cap * neg1))
        excess = pos1 - max_pos
        cand = sorted(added, key=lambda t: (order.get(t[1].get("difficulty"), 99), t[0]))
        drop = set()
        for idx, row in cand:
            if excess <= 0:
                break
            drop.add(idx)
            excess -= 1
            lv = (row.get("difficulty") or "").strip()
            discarded[lv] = discarded.get(lv, 0) + 1
        if drop:
            new_rows = [r for i, r in enumerate(new_rows) if i not in drop]

    pos2, neg2 = _counts(new_rows)
    ratio_final = (float(pos2) / neg2) if neg2 else float("inf")
    report = {
        "n_input_rows": n_input,
        "n_output_rows": len(new_rows),
        "n_attack": pos0,
        "n_benign": neg0,
        "ratio_before": (float(pos0) / neg0) if neg0 else float("inf"),
        "pos_neg_before": "%d:%d" % (pos0, neg0),
        "drawn": stats["drawn"],
        "augmented_ok": stats["ok"],
        "skipped": {"anchor_missing": stats["anchor_missing"],
                    "empty_payload": stats["empty_payload"],
                    "no_choice": stats["no_choice"],
                    "encode_error": stats["encode_error"]},
        "last_encode_error": stats["last_encode_error"],
        "pos_neg_before_cap": "%d:%d" % (pos1, neg1),
        "ratio_before_cap": ratio_pre_cap,
        "pos_neg_after": "%d:%d" % (pos2, neg2),
        "ratio_after": ratio_final,
        "ratio_cap": cap,
        "ratio_cap_binding": binding,
        "base_ratio": base_ratio,
        "ratio_cap_reachable": cap_reachable,
        "discarded_by_difficulty": discarded,
        "n_discarded": sum(discarded.values()),
        "applied_encoding_counts": dict(sorted(applied.items(),
                                               key=lambda kv: (-kv[1], kv[0]))),
        "encoding_before": _hist(frame, "encoding_type"),
        "encoding_after": _hist(new_rows, "encoding_type"),
        "difficulty_before": _hist(frame, "difficulty"),
        "difficulty_after": _hist(new_rows, "difficulty"),
        "frame_digest": frame_digest(new_rows),
    }
    return new_rows, report


# ---------------------------------------------------------------------------
# 一号门禁：4 留出码 0 命中（判据与阶段2 的 `A5` 同制）
# ---------------------------------------------------------------------------
def heldout_reference(reference_rows, heldout):
    """参照集 = 留出编码**攻击行**的 `payload` 与 `text`（精确串相等）。

    刻意**不含** `plaintext`：它是改写体明文，在同一 `rewrite_id` 的 9 编码上恒等，
    并入会让「同一改写体的训练编码行」被判为泄露（假阳性；`02_07.check_a5` 实测 3960 条）。
    """
    held = set(heldout)
    if not held:
        return set()
    hold = set()
    for r in reference_rows:
        if str(r.get("label")).strip() != "1":
            continue
        if (r.get("encoding_type") or "").strip() not in held:
            continue
        for col in ("payload", "text"):
            v = r.get(col) or ""
            if v:
                hold.add(v)
    return hold


def assert_heldout_zero(rows, heldout, scan_columns, reference_rows, where="rows"):
    """两层扫描：① `encoding_type` 分类列；② `scan_columns` 内容列**精确串相等**。

    返回 `(ok, detail)`。命中即 `ok=False`（调用方须**非零退出**）。
    """
    held = set(heldout)
    hits = ["%s#%s" % (where, r.get("sample_id", "?"))
            for r in rows if (r.get("encoding_type") or "").strip() in held]
    hold = heldout_reference(reference_rows, held)
    leaks = []
    for r in rows:
        for col in scan_columns:
            v = r.get(col) or ""
            if v and v in hold:
                leaks.append("%s:%s:%s" % (where, r.get("sample_id", "?"), col))
    detail = ("分类列命中=%d；内容列命中=%d（参照集 %d 串，扫描列 %s）"
              % (len(hits), len(leaks), len(hold), "/".join(scan_columns)))
    if hits or leaks:
        detail += "；例：%s" % ",".join((hits + leaks)[:5])
    return (not hits and not leaks), detail


# ---------------------------------------------------------------------------
# dry-run：阶段计划 ＋ 增强后分布 ＋ 四项断言
# ---------------------------------------------------------------------------
def run_dry_run(cfg, regime, seed, total_steps, verbose=True):
    """返回 `(ctx, ok)`。`ctx` 供渲染报告；`ok` = 四项断言全 PASS。"""
    cfg = resolve_plan_cfg(cfg)
    rc = _require(_require(cfg, "regimes", "cfg"), regime, "regimes")
    allowed = list(_require(rc, "train_encodings", "regimes.%s" % regime))
    heldout = list(rc.get("heldout_encodings") or [])
    scan_cols = list(rc.get("scan_columns") or ["text", "plaintext"])
    aug_cfg = _require(cfg, "augment", "cfg")
    p = float(_require(aug_cfg, "p", "augment"))
    cap = float(_require(aug_cfg, "ratio_cap", "augment"))

    train_csv = assert_split_allowed(split_path(cfg, regime, "train"), cfg)
    val_csv = assert_split_allowed(split_path(cfg, regime, "val"), cfg)
    train_rows = load_rows(train_csv)
    val_rows = load_rows(val_csv)
    val_digest_before = frame_digest(val_rows)

    # 参照集（A5 只读 4 分片 ⇒ **不含** test；比 `02_07` 的口径更弱，已如实登记）
    ref = []
    for rg in ("id", "ood"):
        for sp in ("train", "val"):
            ref.extend(load_rows(assert_split_allowed(split_path(cfg, rg, sp), cfg)))
    n_ref_strings = len(heldout_reference(ref, heldout))

    sched = CurriculumScheduler(cfg, regime, total_steps, random.Random(seed))
    plan_rows = sched.describe()

    def once():
        return augment_attack(train_rows, random.Random(seed), allowed, p, cap, cfg=cfg)

    frame, rep = once()
    _frame2, rep2 = once()
    determinism_ok = (rep["frame_digest"] == rep2["frame_digest"])

    # ---- 四项断言 ----
    ok1, detail1 = assert_heldout_zero(frame, heldout, scan_cols, ref,
                                       where="%s/train(aug)" % regime)
    ok2 = bool(rep["ratio_after"] <= cap + 1e-12)
    n_aug_in_frame = sum(1 for r in frame if (r.get("augmented") or "") == "yes")
    all_pos = all(str(r.get("label")).strip() == "1"
                  for r in frame if (r.get("augmented") or "") == "yes")
    val_aug_hits = sum(1 for r in val_rows if (r.get("augmented") or "") == "yes")
    val_digest_after = frame_digest(val_rows)
    ok3 = bool(all_pos and val_aug_hits == 0
               and val_digest_before == val_digest_after
               and n_aug_in_frame == rep["augmented_ok"] - rep["n_discarded"])
    ok4 = bool(determinism_ok)

    stage_rows = []
    for st in plan_rows:
        idxs = sched.select(frame, st["end_step"])
        # 攻击／良性按 `difficulty` 是否为空判定（与 `select()` 同口径；`label` 列等价但此判更直接）
        n_pos = sum(1 for i in idxs if (frame[i].get("difficulty") or "").strip() != "")
        n_neg = len(idxs) - n_pos
        _np, _nn, _ratio, _target, _ok = sched.ratio_report(frame, st["end_step"])
        stage_rows.append({"stage": st["stage"], "start_step": st["start_step"],
                           "end_step": st["end_step"], "n_selected": len(idxs),
                           "n_pos": n_pos, "n_neg": n_neg,
                           "ratio": (float(n_pos) / n_neg) if n_neg else None,
                           "target_ratio": _target,
                           "ratio_ok": _ok,
                           "benign_subsampled": bool(sched.bs_enabled
                                                     and not (sched.bs_stage3_full
                                                              and st["stage"] >= len(sched.stages)))})

    # ---- ⑤ 各阶段正负比恒 = 增强帧比（`D-42` 裁定三） ----
    ok5 = bool(all(sr["ratio_ok"] for sr in stage_rows))
    detail5 = "；".join("stage %d %d:%d=%.4f（目标 %.4f）"
                       % (sr["stage"], sr["n_pos"], sr["n_neg"],
                          sr["ratio"] if sr["ratio"] is not None else float("nan"),
                          sr["target_ratio"])
                       for sr in stage_rows)
    # ---- ⑥ stage 3 等价全量帧（未被良性子采样） ----
    ok6 = bool(not sched.bs_enabled
               or (sched.bs_stage3_full and stage_rows
                   and stage_rows[-1]["n_selected"] == len(frame)
                   and not stage_rows[-1]["benign_subsampled"]))

    ctx = {
        "regime": regime,
        "seed": seed,
        "total_steps": total_steps,
        "train_csv": os.path.relpath(train_csv, _ROOT).replace("\\", "/"),
        "val_csv": os.path.relpath(val_csv, _ROOT).replace("\\", "/"),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "allowed": allowed,
        "heldout": heldout,
        "n_ref_strings": n_ref_strings,
        "scan_cols": scan_cols,
        "p": p,
        "cap": cap,
        "plan_rows": plan_rows,
        "stage_rows": stage_rows,
        "bs_tol": sched.bs_tol,
        "bs_target_ratio": sched.bs_target_ratio,
        "bs_enabled": sched.bs_enabled,
        "report": rep,
        "digest2": rep2["frame_digest"],
        "val_aug_hits": val_aug_hits,
        "val_digest": val_digest_after,
        "assertions": [
            ("① 体制%s train 侧留出码 0 命中（分类列＋内容列）" % regime, ok1, detail1),
            ("② 增强后正负比 ≤ %.2f" % cap, ok2,
             "实测 pos:neg = %s（ratio=%.6f）" % (rep["pos_neg_after"], rep["ratio_after"])),
            ("③ 增强只命中 label==1 的 train 行；val 行 0 命中", ok3,
             "frame 内增强行 %d 条（= 成功 %d − 丢弃 %d），全部 label=1；"
             "val 增强命中 %d；val 指纹前后一致=%s"
             % (n_aug_in_frame, rep["augmented_ok"], rep["n_discarded"], val_aug_hits,
                val_digest_before == val_digest_after)),
            ("④ 同 seed 两次增强逐字节可复现", ok4,
             "digest#1=%s… digest#2=%s…" % (rep["frame_digest"][:16], rep2["frame_digest"][:16])),
            ("⑤ 各阶段正负比恒 = 增强帧比（±%.2f；`D-42` 裁定三）" % sched.bs_tol, ok5, detail5),
            ("⑥ stage 3 等价全量帧（未被良性子采样）", ok6,
             "stage3 入选 %d 行 / 帧 %d 行"
             % (stage_rows[-1]["n_selected"] if stage_rows else -1, len(frame))),
        ],
    }
    ok = bool(ok1 and ok2 and ok3 and ok4 and ok5 and ok6)
    if verbose:
        out = []
        out.append("== 04_02 dry-run  regime=%s seed=%d total_steps=%d =="
                   % (regime, seed, total_steps))
        out.append("  输入：train=%s（%d 行）  val=%s（%d 行）"
                   % (ctx["train_csv"], ctx["n_train"], ctx["val_csv"], ctx["n_val"]))
        out.append("  允许编码（%d）：%s" % (len(allowed), "/".join(allowed)))
        out.append("  留出编码（%d）：%s" % (len(heldout), "/".join(heldout) or "（无）"))
        out.append("-- 阶段计划（累积式；固定 step 比例，不由 val 触发）--")
        for st in plan_rows:
            out.append("  stage %d：step %d–%d  档位=%s  编码=%s  cumulative=%s"
                       % (st["stage"], st["start_step"], st["end_step"],
                          "+".join(st["difficulty_levels"]), ",".join(st["encodings"]),
                          st["cumulative"]))
        for sr in stage_rows:
            out.append("     实测入选 stage %d：%d 行（攻 %d／良 %d，比 %s；目标 %.4f；%s）"
                       % (sr["stage"], sr["n_selected"], sr["n_pos"], sr["n_neg"],
                          ("%.4f" % sr["ratio"]) if sr["ratio"] is not None else "n/a",
                          sr["target_ratio"],
                          "子采样" if sr["benign_subsampled"] else "全量"))
        out.append("-- 增强（追加式，p=%.2f，上界 %.2f）--" % (p, cap))
        out.append("  抽中 %d → 成功 %d（跳过：锚点缺失 %d／空 payload %d／无备选 %d／编码错 %d）"
                   % (rep["drawn"], rep["augmented_ok"], rep["skipped"]["anchor_missing"],
                      rep["skipped"]["empty_payload"], rep["skipped"]["no_choice"],
                      rep["skipped"]["encode_error"]))
        out.append("  正负比 %s → %s（ratio %.6f；上界 %s）"
                   % (rep["pos_neg_before"], rep["pos_neg_after"], rep["ratio_after"],
                      "触发" if rep["ratio_cap_binding"] else "未触发"))
        out.append("  丢弃增强行 %d（low→medium→high：%s）"
                   % (rep["n_discarded"], rep["discarded_by_difficulty"]))
        out.append("  施加编码分布：%s" % rep["applied_encoding_counts"])
        out.append("  编码分布 前：%s" % rep["encoding_before"])
        out.append("  编码分布 后：%s" % rep["encoding_after"])
        out.append("  难度分布 前：%s" % rep["difficulty_before"])
        out.append("  难度分布 后：%s" % rep["difficulty_after"])
        out.append("-- 六项断言 --")
        for name, okk, detail in ctx["assertions"]:
            out.append("  [%s] %s —— %s" % ("PASS" if okk else "FAIL", name, detail))
        out.append("FRAME_DIGEST=%s" % rep["frame_digest"])
        out.append("DRYRUN_RESULT=%s" % ("PASS" if ok else "FAIL"))
        print("\n".join(out))
    return ctx, ok


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _synth_frame(n_benign=2, encodings=(("rot13", "low"), ("reverse", "low"),
                                        ("base64", "medium"), ("url", "medium"),
                                        ("zerowidth", "high"), ("homoglyph", "high"))):
    """合成 frame（不读盘）：`n_benign` 条良性 + 逐编码一条攻击。

    默认形态（2 良性 + 6 攻击）的正负比 = 3.0 —— **刻意高于任何实际上界**，
    用于验证「上界不可达时显式报错」；需要可达上界的用例请显式传 `n_benign`。
    """
    rows = []
    for i in range(n_benign):
        rows.append({"sample_id": "B%d" % i, "text": "benign body %d" % i, "label": "0",
                     "payload": "", "encoding_type": "", "difficulty": ""})
    for i, (enc, lv) in enumerate(encodings):
        payload = "PAYLOAD-%d" % i
        rows.append({"sample_id": "A%d" % i, "text": "host(PAYLOAD-%d);" % i, "label": "1",
                     "payload": payload, "encoding_type": enc, "difficulty": lv})
    return rows


def _synth_ood_frame(n_benign=10):
    """体制二形态的合成 frame：只含 5 个训练编码（`PREREG_v1.md` §2.2）。"""
    return _synth_frame(n_benign=n_benign,
                        encodings=(("rot13", "low"), ("reverse", "low"),
                                   ("base64", "medium"), ("url", "medium"),
                                   ("zerowidth", "high")))


_REF_CACHE = {}


def ref_rows_for_selfcheck(cfg):
    """自检用的留出码参照集（读 4 个允许分片；结果缓存）。"""
    if "rows" not in _REF_CACHE:
        rows = []
        for rg in ("id", "ood"):
            for sp in ("train", "val"):
                rows.extend(load_rows(assert_split_allowed(split_path(cfg, rg, sp), cfg)))
        _REF_CACHE["rows"] = rows
    return _REF_CACHE["rows"]


def self_check(cfg_path=None, verbose=True):
    """`--self-check`：逐项断言，返回 `(n_fail, checks)`。"""
    cfg = load_config(cfg_path or DEFAULT_PLAN_CONFIG)
    checks = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:                       # 自检本身不许静默吞异常
            ok, detail = False, "抛出异常：%s" % ascii(exc)
        checks.append((name, bool(ok), detail))
        return bool(ok)

    def c_domain():
        dmap = cfg["encoding_difficulty"]
        lv = set(cfg["difficulty_levels"])
        miss = sorted(e for e in dmap if dmap[e] not in lv)
        return (len(dmap) == 9 and not miss,
                "9 码映射齐备=%s，越域=%s" % (len(dmap) == 9, miss or "无"))

    def c_regimes():
        rid = cfg["regimes"]["id"]["train_encodings"]
        rood = cfg["regimes"]["ood"]["train_encodings"]
        hold = cfg["regimes"]["ood"]["heldout_encodings"]
        return (len(rid) == 9 and len(rood) == 5 and len(hold) == 4
                and not (set(rood) & set(hold)) and set(rood) | set(hold) == set(rid),
                "id=%d 码／ood 训练=%d 码／留出=%d 码；交集=%s；并集==9 码=%s"
                % (len(rid), len(rood), len(hold), sorted(set(rood) & set(hold)) or "空",
                   set(rood) | set(hold) == set(rid)))

    def c_plan():
        plan = cfg["plan"]
        props = plan["step_proportions"]
        stages = plan["stages"]
        cum = all(set(stages[i]["levels"]) <= set(stages[i + 1]["levels"])
                  for i in range(len(stages) - 1))
        return (plan["n_stages"] == 3 and plan["cumulative"] is True
                and len(props) == 3 and abs(props[-1] - 1.0) < 1e-12
                and all(props[i] < props[i + 1] for i in range(2)) and cum,
                "n_stages=%s cumulative=%s props=%s 档位累积递增=%s"
                % (plan["n_stages"], plan["cumulative"], props, cum))

    def c_boundary():
        s = CurriculumScheduler(cfg, "id", 10, random.Random(42))
        got = [s.stage_at(x) for x in (0, 1, 3, 4, 6, 7, 10, 11)]
        want = [1, 1, 1, 2, 2, 3, 3, 3]
        return got == want, ("ends=%s；stage_at(0,1,3,4,6,7,10,11)=%s 期望 %s"
                             % (s.ends, got, want))

    def c_describe():
        s = CurriculumScheduler(cfg, "id", 10, random.Random(42))
        d = s.describe()
        fields = {"stage", "start_step", "end_step", "difficulty_levels", "encodings",
                  "cumulative"}
        ok = (len(d) == 3 and all(set(x) == fields for x in d)
              and all(x["cumulative"] for x in d)
              and len(d[2]["encodings"]) == 9 and len(d[0]["encodings"]) == 2)
        return ok, ("行数=%d 字段齐备=%s stage3 编码数=%d stage1 编码=%s"
                    % (len(d), all(set(x) == fields for x in d),
                       len(d[2]["encodings"]), d[0]["encodings"]))

    def c_select():
        # 足量规模使子采样可比：25 low／25 medium／25 high ＋ 40 良性
        encs = tuple([("rot13", "low")] * 25 + [("base64", "medium")] * 25
                     + [("zerowidth", "high")] * 25)
        frame = _synth_frame(n_benign=40, encodings=encs)
        s = CurriculumScheduler(cfg, "id", 10, random.Random(42))
        s1 = s.select(frame, 1)
        s3 = s.select(frame, 10)
        low_only = all(frame[i]["difficulty"] in ("", "low") for i in s1)
        allin = len(s3) == len(frame)
        n_pos1 = sum(1 for i in s1 if frame[i]["difficulty"] != "")
        pos_kept = (n_pos1 == 25)                      # stage1 攻击行只含 25 条 low，全保留
        _np, _nn, _r, _tg, ok_ratio = s.ratio_report(frame, 1)
        return low_only and allin and pos_kept and ok_ratio, (
            "stage1 入选=%d（仅 low＋良性=%s，攻击 25 条不减=%s，比=%.4f 落在 %.4f±%.2f=%s）；"
            "stage3 入选=%d（全量=%s）"
            % (len(s1), low_only, pos_kept, _r, _tg, s.bs_tol, ok_ratio, len(s3), allin))

    def c_augment_only_pos():
        frame = _synth_frame()
        rep = augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                             1.0, 9.0, cfg=cfg)[1]
        return rep["augmented_ok"] > 0 and rep["n_input_rows"] == len(frame), (
            "增强成功=%d（p=1.0 抽中 %d）；原 frame 行数不变=%d"
            % (rep["augmented_ok"], rep["drawn"], rep["n_input_rows"]))

    def c_ratio_cap():
        # 原分片正负比 0.6（6 攻击 / 10 良性）< 上界 0.75 ⇒ 上界**可达**，缩减后必须达标
        frame = _synth_frame(n_benign=10)
        out, rep = augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                                  1.0, 0.75, cfg=cfg)
        pos, neg = _counts(out)
        got = float(pos) / neg
        return got <= 0.75 + 1e-12 and rep["n_discarded"] > 0 and neg == 10, (
            "上界 0.75（原分片 6:10=0.6000 可达）⇒ 实测 ratio=%.4f（pos:neg=%s），"
            "丢弃 %d 条 %s；良性行数不变=%d"
            % (got, rep["pos_neg_after"], rep["n_discarded"],
               rep["discarded_by_difficulty"], neg))

    def c_cap_unreachable():
        # 原分片正负比 3.0 > 上界 0.75 ⇒ **不可达**，必须显式报错且**不动**原分片行
        frame = _synth_frame(n_benign=2)
        try:
            augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                           1.0, 0.75, cfg=cfg)
        except ValueError as exc:
            return "不可达" in str(exc), "上界不可达时抛出：%s" % str(exc)[:110]
        return False, "上界不可达却未报错（会静默删不动/或静默超标）"

    def c_determinism():
        frame = _synth_frame(n_benign=10)
        a = augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                           0.5, 1.25, cfg=cfg)
        b = augment_attack(frame, random.Random(42), ["base64", "rot13", "reverse"],
                           0.5, 1.25, cfg=cfg)
        c = augment_attack(frame, random.Random(123), ["base64", "rot13", "reverse"],
                           0.5, 1.25, cfg=cfg)
        same = a[1]["frame_digest"] == b[1]["frame_digest"]
        diff = a[1]["frame_digest"] != c[1]["frame_digest"]
        return same and diff, (
            "seed42 两次 digest 相同=%s（%s…）；seed123 不同=%s（%s…）"
            % (same, a[1]["frame_digest"][:12], diff, c[1]["frame_digest"][:12]))

    def c_ood_heldout():
        frame = _synth_ood_frame()
        allowed = cfg["regimes"]["ood"]["train_encodings"]
        out, _rep = augment_attack(frame, random.Random(42), allowed, 1.0, 1.25, cfg=cfg)
        return assert_heldout_zero(out, cfg["regimes"]["ood"]["heldout_encodings"],
                                   cfg["regimes"]["ood"]["scan_columns"],
                                   ref_rows_for_selfcheck(cfg), "synth")

    def c_no_self_encoding():
        frame = _synth_frame()
        out, _rep = augment_attack(frame, random.Random(42),
                                   ["base64", "rot13", "reverse"], 1.0, 9.0, cfg=cfg)
        bad = [r["sample_id"] for r in out
               if (r.get("augmented") or "") == "yes"
               and (r.get("aug_from_encoding") or "") == (r.get("encoding_type") or "")]
        return not bad, "施加编码 == 自身编码 的增强行：%s" % (bad or "无")

    def c_benign_subsample():
        """`D-42` 裁定三：各阶段正负比恒 = 增强帧比；只减良性；stage3 等价全量帧。

        合成帧须**足够大**，否则 `round(pos/target)` 的整数化会让比偏出 ±0.05。
        取 40 良性 ＋ 攻击（low 25／medium 25／high 25 = 75 条）：
        stage1 pos=25→neg=round(25/1.25)=20 ⇒ 25:20=1.2500 ✓；
        stage2 pos=50→neg=round(50/1.25)=40 ⇒ 50:40=1.2500 ✓；
        stage3 pos=75→**不子采样**（等价全量帧），比=75:40=1.875（末阶段不参战）。
        """
        encs = tuple([("rot13", "low")] * 25 + [("base64", "medium")] * 25
                     + [("zerowidth", "high")] * 25)
        # 逐条唯一化 sample_id／payload（`_synth_frame` 用枚举生成，天然唯一）
        frame = _synth_frame(n_benign=40, encodings=encs)
        s = CurriculumScheduler(cfg, "id", 30, random.Random(42))
        rows = [(st["stage"], st["end_step"]) for st in s.describe()]
        bad = []
        detail = []
        for stage, step in rows:
            n_pos, n_neg, ratio, target, ok = s.ratio_report(frame, step)
            if stage < len(s.stages):                  # 末阶段不子采样 ⇒ 比不参战
                detail.append("stage%d %d:%d=%.4f%s"
                              % (stage, n_pos, n_neg, ratio, "/ok" if ok else "/BAD"))
                if not ok:
                    bad.append(stage)
            else:
                detail.append("stage%d %d:%d=%.4f(全量/不参战)" % (stage, n_pos, n_neg, ratio))
        n_pos1 = s.ratio_report(frame, rows[0][1])[0]
        n_pos3 = s.ratio_report(frame, rows[-1][1])[0]
        stage3_full = (s.ratio_report(frame, rows[-1][1])[1] + n_pos3) == len(frame)
        return (not bad and n_pos1 == 25 and n_pos3 == 75 and stage3_full), (
            "；".join(detail) + "；攻击行 stage1=%d（应 25）stage3=%d（应 75）；stage3 等价全量=%s"
            % (n_pos1, n_pos3, stage3_full))

    def c_heavy_imports():
        loaded = [m for m in _HEAVY if m in sys.modules]
        return not loaded, ("顶层重型库：%s（须为空；`yaml` 只在 `load_config()` 内惰性 import）"
                            % (loaded or "无"))

    def c_split_guard():
        bad = None
        for p in (os.path.join("data", "sealed_test", "CodeObf-PI_v1.0", "splits",
                               "regime_id", "test.csv"),
                  os.path.join("data", "frozen", "CodeObf-PI_v1.0", "splits",
                               "regime_id", "test.csv")):
            try:
                assert_split_allowed(_abs(p), cfg)
                bad = p
                break
            except RuntimeError:
                continue
        ok = (bad is None
              and bool(assert_split_allowed(split_path(cfg, "id", "train"), cfg)))
        return ok, "禁读路径全部被拦下=%s" % (bad is None)

    for name, fn in (("难度域：9 码映射 ∈ 封闭域", c_domain),
                     ("两体制编码域：id 9／ood 5+4，并集=9", c_regimes),
                     ("阶段计划：3 阶段累积、比例递增、末位 1.0", c_plan),
                     ("阶段边界：stage_at 八点实测", c_boundary),
                     ("describe()：6 字段 × 3 行、编码累积", c_describe),
                     ("select()：stage1 仅 low＋良性子采样、攻击行不减、stage3 全量", c_select),
                     ("增强只命中 label==1（原 frame 一行不动）", c_augment_only_pos),
                     ("正负比上界生效并只缩减增强行", c_ratio_cap),
                     ("上界不可达时显式报错（不删原分片行）", c_cap_unreachable),
                     ("确定性：同 seed 逐字节相同，异 seed 不同", c_determinism),
                     ("体制二增强后 4 留出码 0 命中", c_ood_heldout),
                     ("施加编码 ≠ 样本自身编码", c_no_self_encoding),
                     ("良性子采样：各阶段正负比恒 = 增强帧比（D-42 裁定三）", c_benign_subsample),
                     ("顶层不 import 重型库", c_heavy_imports),
                     ("输入面守卫：禁读路径被拦下", c_split_guard)):
        check(name, fn)

    n_fail = sum(1 for _n, ok, _d in checks if not ok)
    if verbose:
        print("== 04_02 --self-check ==")
        for name, ok, detail in checks:
            print("  [%s] %s —— %s" % ("PASS" if ok else "FAIL", name, detail))
        print("SELFCHECK_RESULT=%s（%d 项，失败 %d）"
              % ("PASS" if n_fail == 0 else "FAIL", len(checks), n_fail))
    return n_fail, checks


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _fmt_ratio(x):
    return "n/a" if x is None else "%.4f" % x


def render_report(ctx, ts, cfg):
    L = []
    A = L.append
    A("# W5.2 课程学习与编码感知增强 —— 运行报告")
    A("")
    A("- **时间戳**：`%s`" % ts)
    A("- **体制**：`regime_%s`（%s）" % (ctx["regime"], ctx.get("regime_label", "")))
    A("- **种子**：`%d`（显式 `random.Random(seed)`，不用全局 RNG）" % ctx["seed"])
    A("- **配置**：`configs/train/curriculum.yaml`（事前定值；阶段切点**不由 val 触发**）")
    A("- **权限**：A5（只读 4 个 frozen train/val 分片；未读 `data/sealed_test/**`"
      " 与 frozen 侧 `test.csv`）")
    A("")
    A("## 1. 输入面（C1）")
    A("")
    A("| 分片 | 路径 | 行数 |")
    A("|---|---|---|")
    A("| train | `%s` | %d |" % (ctx["train_csv"], ctx["n_train"]))
    A("| val | `%s` | %d |" % (ctx["val_csv"], ctx["n_val"]))
    A("")
    A("- 训练编码（%d）：`%s`" % (len(ctx["allowed"]), "`、`".join(ctx["allowed"])))
    A("- 留出编码（%d）：`%s`" % (len(ctx["heldout"]),
                                  "`、`".join(ctx["heldout"]) or "（无；体制一不留出）"))
    A("")
    A("## 2. 阶段计划（累积式；固定 step 比例，**事前定值**）")
    A("")
    A("`total_steps = %d`（**dry-run 展示刻度**；真实值由 `04_01` 按实际 micro-batches 传入，"
      "同一份比例 `[0.3, 0.6, 1.0]` 缩放）" % ctx["total_steps"])
    A("")
    A("| stage | step 区间 | 档位（累积） | 覆盖编码（累积） | cumulative |")
    A("|---|---|---|---|---|")
    for st in ctx["plan_rows"]:
        A("| %d | %d–%d | %s | %s | %s |"
          % (st["stage"], st["start_step"], st["end_step"],
             " + ".join(st["difficulty_levels"]), "、".join(st["encodings"]),
             st["cumulative"]))
    A("")
    A("**各阶段实测入选行数**（`select()` 在增强后 frame 上的实算；"
      "良性行 `difficulty` 为空 ⇒ 候选恒入选，再按 `benign_subsample` 做**确定性前缀子采样**"
      "使各阶段正负比恒 = 增强帧比，**只减良性、stage 3 不子采样**）：")
    A("")
    A("| stage | 入选行数 | 攻击行 | 良性行 | 正负比 | 目标比 | 落在 ±%.2f | 是否子采样 |"
      % ctx["bs_tol"])
    A("|---|---|---|---|---|---|---|---|")
    for sr in ctx["stage_rows"]:
        A("| %d | %d | %d | %d | %s | %.4f | %s | %s |"
          % (sr["stage"], sr["n_selected"], sr["n_pos"], sr["n_neg"], _fmt_ratio(sr["ratio"]),
             sr["target_ratio"], "✓" if sr["ratio_ok"] else "✗",
             "是" if sr["benign_subsampled"] else "否（全量帧）"))
    A("")
    A("> **良性子采样口径（`D-42` 裁定三）**：`neg_kept = round(n_pos / target_ratio)`，"
      "`target_ratio` = 增强后训练帧实测比（id **%.4f**）；确定性前缀（按行下标升序），"
      "`random.Random(seed)` 契约不变；**攻击行一条不减**（维持累积式）；**stage 3 等价全量帧**。"
      % ctx["bs_target_ratio"])
    A("")
    A("## 3. 增强（追加式，`p = %.2f`，正负比上界 `%.2f`）" % (ctx["p"], ctx["cap"]))
    A("")
    rep = ctx["report"]
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 输入行数（原 train） | %d |" % rep["n_input_rows"])
    A("| 输出行数（增强后） | %d |" % rep["n_output_rows"])
    A("| 抽中（Bernoulli p） | %d |" % rep["drawn"])
    A("| 增强成功 | %d |" % rep["augmented_ok"])
    A("| 跳过：锚点缺失（`payload` 不在 `text` 中） | %d |" % rep["skipped"]["anchor_missing"])
    A("| 跳过：空 `payload` | %d |" % rep["skipped"]["empty_payload"])
    A("| 跳过：无备选编码 | %d |" % rep["skipped"]["no_choice"])
    A("| 跳过：编码异常 | %d |" % rep["skipped"]["encode_error"])
    A("| 正负比（增强前） | %s（%.4f） |" % (rep["pos_neg_before"], rep["ratio_before"]))
    A("| 正负比（缩减前） | %s（%.4f） |" % (rep["pos_neg_before_cap"], rep["ratio_before_cap"]))
    A("| 正负比（最终） | %s（%.4f） |" % (rep["pos_neg_after"], rep["ratio_after"]))
    A("| 上界可达性（原分片自身比 %.4f） | %s |"
      % (rep["base_ratio"], "可达" if rep["ratio_cap_reachable"] else "**不可达**"))
    A("| 上界是否触发 | %s |" % ("**是**" if rep["ratio_cap_binding"] else "否"))
    A("| 丢弃增强行数 | %d（%s） |" % (rep["n_discarded"], rep["discarded_by_difficulty"]))
    A("")
    A("### 3.1 施加编码分布（增强行）")
    A("")
    A("| 编码 | 条数 |")
    A("|---|---|")
    for k, v in rep["applied_encoding_counts"].items():
        A("| `%s` | %d |" % (k, v))
    if not rep["applied_encoding_counts"]:
        A("| （无） | 0 |")
    A("")
    A("### 3.2 编码分布（增强前 → 增强后）")
    A("")
    A("| encoding_type | 前 | 后 |")
    A("|---|---|---|")
    keys = sorted(set(rep["encoding_before"]) | set(rep["encoding_after"]),
                  key=lambda k: (-rep["encoding_after"].get(k, 0), k))
    for k in keys:
        A("| `%s` | %d | %d |" % (k or "（空＝良性）", rep["encoding_before"].get(k, 0),
                                  rep["encoding_after"].get(k, 0)))
    A("")
    A("### 3.3 难度分布（增强前 → 增强后）")
    A("")
    A("| difficulty | 前 | 后 |")
    A("|---|---|---|")
    for k in sorted(set(rep["difficulty_before"]) | set(rep["difficulty_after"]),
                    key=lambda k: (-rep["difficulty_after"].get(k, 0), k)):
        A("| `%s` | %d | %d |" % (k or "（空＝良性）", rep["difficulty_before"].get(k, 0),
                                  rep["difficulty_after"].get(k, 0)))
    A("")
    A("## 4. 六项断言（`W5.2` §7-2 ＋ `D-42` 裁定三 追加 ⑤⑥）")
    A("")
    A("| # | 断言 | 结果 | 实测 |")
    A("|---|---|---|---|")
    for i, (name, ok, detail) in enumerate(ctx["assertions"], 1):
        A("| %d | %s | **%s** | %s |" % (i, name, "PASS" if ok else "FAIL", detail))
    A("")
    A("- 增强后 frame 指纹（逐行 `sample_id／text／payload／encoding_type／difficulty` SHA-256）："
      "`%s`" % rep["frame_digest"])
    A("- 第二次同 seed 指纹：`%s`" % ctx["digest2"])
    A("- val 帧指纹（全程未增强）：`%s`" % ctx["val_digest"])
    A("")
    A("## 5. 口径声明与限制（如实登记）")
    A("")
    for line in ctx.get("notes", []):
        A("- %s" % line)
    A("")
    A("## 6. 呈 W0 事项")
    A("")
    for line in ctx.get("to_w0", []):
        A("- %s" % line)
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="04_02_train_curriculum.py",
        description="W5.2 课程学习与编码感知增强（只被 `04_01` import，不单独执行训练）")
    p.add_argument("--self-check", action="store_true", help="自检（退出 0/非零）")
    p.add_argument("--dry-run", action="store_true",
                   help="打印阶段计划与增强后分布并落报告（不训练）")
    p.add_argument("--regime", choices=["id", "ood"], default="id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-steps", type=int, default=None,
                   help="阶段边界的 step 刻度（默认取 `plan.dry_run_total_steps`）")
    p.add_argument("--config", default=DEFAULT_PLAN_CONFIG)
    p.add_argument("--report-dir", default=None,
                   help="报告目录（默认取 `paths.default_report_dir`；C11 禁止写死 `reports/`）")
    p.add_argument("--log-dir", default=None,
                   help="日志目录（默认取 `paths.default_log_dir`）")
    p.add_argument("--ts", default=None,
                   help="时间戳覆写（确定性比对用；默认取当前时刻）")
    p.add_argument("--no-report", action="store_true", help="只打印，不落报告文件")
    return p


def _ts():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class _Tee(object):
    """stdout 双写（控制台 ＋ 日志文件）。只动输出流，不改任何判定值。"""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def reconfigure(self, **kw):                      # 兼容 `init_streams()`
        return self.stream.reconfigure(**kw)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _notes(ctx, cfg):
    rep = ctx["report"]
    notes = [
        "`total_steps = %d` 是 **dry-run 展示刻度**（`plan.dry_run_total_steps`，"
        "= W5.1 实测 4730 micro-batches/epoch × epochs 3 ÷ grad_accum 4）；"
        "真实阶段边界由 `04_01` 在训练时按实际 micro-batches 传入，"
        "**同一份比例** `[0.3, 0.6, 1.0]` 缩放即可。" % ctx["total_steps"],
        "良性行 `difficulty` 恒为空（实测：体制一 train 良性 6336 行、体制二 3910 行，"
        "无一例外）⇒ 阶段过滤只作用于攻击行，良性行先全量候选。"
        "**W5.2 补正轮（`D-42` 裁定三）**再对良性候选做**确定性前缀子采样**"
        "（`neg_kept = round(n_pos / target_ratio)`，`target_ratio` = 增强帧实测比），"
        "使各阶段正负比恒 = 增强帧比（±%.2f）；**攻击行一条不减**、**stage 3 不子采样**"
        "（等价全量帧）。" % ctx["bs_tol"],
        "增强行的 `encoding_type`／`difficulty` 取**所施加**编码"
        "（`augment.augmented_difficulty = applied_encoding`），原值保留在 "
        "`aug_from_encoding`／`aug_from_sample_id` 两列；`payload` 替换为二次编码结果，"
        "`text` 中锚点子串全部替换（`replace_all = true`）。",
        "锚点缺失（实测本体制 train %d 行 `payload` 不在 `text` 中）按 "
        "`anchor_missing_policy = skip` **跳过并计数**，不拼造锚点。"
        % rep["skipped"]["anchor_missing"],
        "编码器**复用** `scripts/data/02_03_encode_variants.py` 的 `ENCODERS`"
        "（`V-5` 单源），本窗口不自造编码。",
        "增强行以 `augmented = yes` 标记 ⇒ **不**计入跨检测器 `text_sha256` 一致性校验"
        "（`W5.2` §4-6：该锚点服务于 B1–B5 与 Ours 的**评估**输入）。",
        "`select()` 只返回行下标，**不做** batching／采样；`04_01` 依此过滤 micro-batch。"
        "若某 micro-batch 的行被全部排除，该 step **跳过**（不回退成全量 —— 否则难样本"
        "会在 stage 1 渗入，课程即失效）。",
    ]
    if ctx["heldout"]:
        notes.append(
            "一号门禁的**参照集比阶段2 的 `A5` 更弱**：`02_07.check_a5` 的参照集覆盖全部体制"
            "与分片（含 `regime_ood/test.csv`），而 A5 权限禁止本窗口读 test ⇒ 参照集只取"
            " 4 个允许的 train/val 分片中留出编码攻击行的 `payload`／`text`（本次 %d 串）。"
            "**这是权限导致的口径收窄，不是实现简化**，已如实登记。" % ctx["n_ref_strings"])
    else:
        notes.append("体制一不留出编码 ⇒ 断言① 的分类列与内容列恒 0 命中"
                     "（参照集为空，判据退化为「无留出码可命中」，仍逐行实测并落表）。")
    return notes


def _to_w0():
    return [
        "「累积式 3 阶段 ＋ 固定 step 比例（**不按 val 触发**）」是本窗口对 v5 L436–L438"
        "「简单 → 中等 → 困难」的落地口径，请 W0 追认（`W5.2` §9-1）。"
        "**注意与 `SOURCE_LOCK` v5 L756「课程切换阶段全部依据验证集」字面冲突** ——"
        "本窗口按 `W5.2` §3-3（后出、更具体）执行，请 W0 一并裁定以哪一份为准。",
        "增强采用**追加式 ＋ 正负比上界 1.25** 而非替换式，请 W0 追认（`W5.2` §9-2）。"
        "（**已由 `D-42` 裁定二追认**。）",
        "良性行在各阶段**恒入选**（因其 `difficulty` 为空）⇒ 阶段内正负比不等于 1:1"
        "（实测见 §2）。**W5.2 补正轮（`D-42` 裁定三）已实现良性子采样**："
        "各阶段正负比恒 = 增强帧比（id 1.2500／ood 1.1675，±0.05），只减良性、stage 3 等价全量。"
        "本条**已闭合**，无待裁。",
        "与 v5 L438「同一攻击样本随机施加多种编码」的落地差异：本窗口实现为**对 `payload` "
        "施加一次二次编码**（`augment.form = 二次编码`），而不是对同一样本生成多种编码变体"
        "（后者会把正负比推高、需另设上界）。**已由 `D-42` 裁定四追认**并锁定口径。",
    ]


def main(argv=None):
    init_streams()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    ts = args.ts or _ts()

    if args.self_check:
        n_fail, _checks = self_check(args.config, verbose=True)
        return 0 if n_fail == 0 else 1

    if not args.dry_run:
        print("NOT_IMPLEMENTED: 04_02 是被 `04_01` import 的子模块；"
              "除 `--self-check`／`--dry-run` 外不单独执行训练。")
        return 0

    total_steps = int(args.total_steps if args.total_steps
                      else _require(_require(cfg, "plan", "cfg"),
                                    "dry_run_total_steps", "plan"))

    paths = _require(cfg, "paths", "cfg")
    log_dir = _abs(args.log_dir or _require(paths, "default_log_dir", "paths"))
    os.makedirs(log_dir, exist_ok=True)
    log_fmt = _require(_require(paths, "artifacts", "paths"), "log", "paths.artifacts")
    log_path = os.path.join(log_dir, log_fmt.format(ts=ts))
    fh = open(log_path, "w", encoding="utf-8", newline="\n")
    fh.write("==============================================================================\n")
    fh.write("04_02_train_curriculum 课程学习与增强 dry-run（W5.2）  ts=%s\n" % ts)
    fh.write("项目根：%s\n" % _ROOT)
    fh.write("python：%s\n" % sys.executable)
    fh.write("参数：regime=%s seed=%d total_steps=%d config=%s\n"
             % (args.regime, args.seed, total_steps, args.config))
    fh.write("==============================================================================\n\n")
    sys.stdout = _Tee(sys.stdout, fh)

    ctx, ok = run_dry_run(cfg, args.regime, int(args.seed), total_steps, verbose=True)
    ctx["regime_label"] = _require(_require(cfg, "regimes", "cfg")[args.regime],
                                   "label", "regimes.%s" % args.regime)
    ctx["notes"] = _notes(ctx, cfg)
    ctx["to_w0"] = _to_w0()

    if not args.no_report:
        report_dir = _abs(args.report_dir
                          or _require(paths, "default_report_dir", "paths"))
        os.makedirs(report_dir, exist_ok=True)
        fmt = _require(_require(paths, "artifacts", "paths"), "report", "paths.artifacts")
        path = os.path.join(report_dir, fmt.format(ts=ts))
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(render_report(ctx, ts, cfg))
        print("REPORT_MD=%s" % os.path.abspath(path))
    print("LOG_FILE=%s" % os.path.abspath(log_path))
    print("DRYRUN_EXIT=%d" % (0 if ok else 1))
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    fh.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

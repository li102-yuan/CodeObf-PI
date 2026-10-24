# -*- coding: utf-8 -*-
"""04_01_train_qlora_ours.py — W5.1 Ours：QLoRA 微调 Llama-Guard-3-1B（单次训练入口）。

职责（`windows/W5.1_qlora.md` §1／§2）：QLoRA 微调主脚本，**含网格搜索与 τ 寻优模式**。
超参与网格空间的**事前定值**载体为 `configs/train/qlora_ours.yaml`。

接口契约（**W5 冻结 v1.0，冻结后不得改**）::

    python scripts/train/04_01_train_qlora_ours.py \
        --regime {id,ood} --seed {42,123,456} \
        --config configs/train/qlora_ours.yaml \
        [--mode {train,grid,threshold}] [--curriculum on|off] [--augment on|off] \
        [--deconfound on|off] [--model-key KEY] \
        [--epochs N] [--max-steps N] [--report-dir DIR] [--out-dir DIR] \
        [--checkpoint-dir DIR] [--self-check]

- `--mode train`（默认）：从头初始化 → 训练 → 每轮 val 评估 → 早停 → 保存最优 checkpoint → 末了做一次 τ 寻优；
  stdout **末行** = ``CHECKPOINT_DIR=<abs path>``。
- `--mode grid`：**只允许 `--regime id`**；按 `--grid-path` 跑 (lr, r) 组并报该体制 val 的 F1／耗时／显存峰值；
  末行 = ``RESULT_CSV=<path>``。**`--grid-path sequential`（默认，`D-42` 裁定六）**：先跑
  `stage1`（`r=16` × 3 个 `lr`）→ 按**事前定值**停止规则判定 → 满足则固定 `(best_lr, 16)`、
  否则补跑 `stage2`（`r∈{8,32}` × 3 `lr`，回到 9 组）；**`--grid-path full`**：直接跑默认 9 组。
- `--mode threshold`：对 `--checkpoint-dir` 指定的 checkpoint 在该体制 val 上寻 ``tau_ours`` 并落曲线 CSV；
  末行 = ``RESULT_CSV=<path>``。
- `--curriculum on`：import `04_02` 的 `CurriculumScheduler`；**`04_02` 未实现时以 `off` 优雅跳过并打印一行告警**，
  不得抛 `NotImplementedError`。
- `--augment {on,off}`（**`W5.6` 新增，`D-83` 裁定四 工作 3**）：**独立**控制「编码感知增强」，
  使其与「课程调度」解耦。生效式 = **「`--curriculum on` ⇒ 增强照旧生效」（C 组复现红线，逐字不变）
  ∪ 「`--augment on` ⇒ 增强在 `--curriculum off` 下也生效」（B 组「有增强、无课程」）**。
  默认 `off` ⇒ `--curriculum off` 时行为与改造前**逐字节一致**（A 组＝仅原始训练帧）。
- `--model-key KEY`（**`W5.6` 新增**）：可选，覆盖 `checkpoint.model_key` ⇒ 影响 `run_id` 的 `{model}` 槽。
  **默认 `None` ⇒ 配置未被改动、行为与改造前逐字节一致。** 用途：消融 A／B 组以
  `llamaguard1b_A`／`llamaguard1b_B` 使 `run_id` 自识别（`W5.6` §4）。
- `--deconfound {on,off}`（**`W5.7` 新增，`P-K'`／`PREREG_v1.8.md` §1；`D-135`／`D-136`**）：
  **2×2 对角去混淆增广**（训练期与验证期数据构造；**评估期一律 `off`**）——
  攻击行（`label==1`）按 `sha256(sample_id)` **最低位** 50% 把输入换 `plaintext`（未编码恶意指令），
  良性行（`label==0`）同位 50% 把输入换 `enc(text[:L])`（整段编码）；**次低位起**在允许码集内
  确定性选码（`id` ⇒ 9 码；`ood` ⇒ 仅 5 训练码，4 留出码零暴露）；`L` 按扩展比反推使输出 ≈950 字符
  （`ascii_art` 下限钳 64）。**替换而非新增** ⇒ 行数／正负比不变；**只改 `text`**（其余列零改动）。
  **默认 `off` ⇒ 与改造前逐 step `loss` Δ=0**（smoke 规模自证，`N-5` 口径）。
- `--self-check`（**`W5.7` 新增**）：只跑去混淆数据构造的自检（§3-1～§3-7），不加载模型、不训练；
  通过 ⇒ 末行 ``SELFCHECK_RESULT=PASS n=<k>``；日志自动落 `logs/train/tmp/`（`N-16` 同规）。
- 退出码：正常 0；断言失败（含泄露断言）**非零**。


阶段4 统一硬约束 C1–C11 的逐条落点：见 `configs/train/qlora_ours.yaml` 文件头与本文件各节的 `C{n}` 标注。

**顶层不 import 任何重型库**（`AGENTS.md` §3／`windows/W4.6_unify.md` §0.2-7）：
`yaml`／`torch`／`transformers`／`peft` 一律在函数体内惰性 import。
唯一例外是 **C9 要求的 `CUBLAS_WORKSPACE_CONFIG`** —— 它必须在 `import torch` **之前**写入
`os.environ`；本模块在最顶部写入，而所有 torch import 都发生在函数体内，故时序成立。

> **train／eval 态的实现说明（C9「代码中不得出现 `.train(`」）**：Llama-3 基座的 dropout 与
> LoRA dropout 均为 `0.0`、`attn_implementation=eager`、未启用梯度检查点 ⇒ `Module.training`
> 标志对前向**无影响**。故本模块全程保持模块默认（eval）态，**不调用 train 模式**；
> 验证／推理处显式 `model.eval()` + `torch.no_grad()` 并做 G-EVAL 双断言（与 `03_03` 同制）。

> **训练侧确定性（N-5，`D-42` §4）**：`set_determinism()` 除 `cudnn.deterministic` 外，还按
> `configs/train/qlora_ours.yaml` 的 `determinism` 事前定值施加
> **`torch.use_deterministic_algorithms`**（框架级，覆盖 embedding backward／`index_select`／
> scatter-add 等 cuDNN 不覆盖的算子），并把 TF32 在 matmul 与 cuDNN **两处**一并关闭。
> **关键时序**：`run_train` 在 `build_model()` **之前**先调用一次 `set_determinism()` ——
> 因为 `build_model()` 内的 `get_peft_model()` 会用全局 RNG 随机初始化 LoRA A 矩阵
> （`lora_B` 恒 0）；此为该阻断级缺陷的**根因**（见 `logs/train/probe_n5_isolate.py`）。
> 修复后连跑两次冒烟：`loss`／`val_f1`／`tau` **位级一致**，checkpoint adapter 逐字节相同。
"""
from __future__ import annotations

# --- C9：必须在 `import torch` 之前设置（本模块所有 torch import 均在函数体内，时序成立）---
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import namedtuple  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

# 网格执行的轻量上下文（供 `_write_grid_md` 记录「执行路径 + 停止规则判定」留证；
# 不改 `_write_grid_md` 既有签名，避免扰动既有单测）。
_GRID_CTX = {}

UNSAFE_WORD = "unsafe"
SAFE_WORD = "safe"
EMPTY_CONVERSATION = "<BEGIN CONVERSATION>\n\n<END CONVERSATION>"

NEG_INF = float("-inf")
POS_INF = float("inf")


# ---------------------------------------------------------------------------
# 输出流 UTF-8 重配（先例：W3.7 `F-1`；`D-40` 裁定五：不得静默 `pass`）
# ---------------------------------------------------------------------------
def init_streams() -> None:
    """把 stdout／stderr 重配为 UTF-8。只动输出流，不触碰任何判定值。失败即显式告警。"""
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


# ---------------------------------------------------------------------------
# C2 上游唯一出处：`03_06`（预处理／读取）与 `03_07`（指标单源）
# ---------------------------------------------------------------------------
_PREPROCESS = _load_module("baseline_preprocess_ours",
                           os.path.join(_ROOT, "scripts", "baseline",
                                        "03_06_unify_preprocess.py"))
_MAPPING = _load_module("baseline_mapping_ours",
                        os.path.join(_ROOT, "scripts", "baseline",
                                     "03_07_unify_output_mapping.py"))

load_samples = _PREPROCESS.load_samples
load_rows = _PREPROCESS.load_rows
sha256_of = _PREPROCESS.sha256_of
md5_of = _PREPROCESS.md5_of
write_csv = _PREPROCESS.write_csv
truncate_left = _PREPROCESS.truncate_left

f1_score = _MAPPING.f1_score                       # 指标单源（`03_07`，不另起一套）
binary_counts = _MAPPING.binary_counts
compute_binary_metrics = _MAPPING.compute_binary_metrics


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_config(path):
    """读 YAML。**禁止默认值兜底**：文件不存在／键缺失一律显式报错。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("配置文件不存在：%s" % path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 PyYAML：%s" % exc) from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _require(mapping, key, where):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError("配置缺少必需键 %r（位置：%s）" % (key, where))
    return mapping[key]


# ---------------------------------------------------------------------------
# C1 输入面：只授权 4 个文件；越界即非零退出
# ---------------------------------------------------------------------------
class LeakAssertionError(AssertionError):
    """泄露断言失败（`windows/W5.1_qlora.md` §4-1／§4-2）。"""


def assert_split_allowed(path, cfg):
    """断言 `path` 落在 A5 授权的 4 个分片内，且不触任何禁读面。"""
    abs_path = os.path.abspath(path)
    norm = abs_path.replace("\\", "/")
    for bad in _require(cfg["data"], "forbidden_substrings", "data"):
        if bad.replace("\\", "/") in norm:
            raise LeakAssertionError("禁止读取：%s（命中禁读面 %r）" % (abs_path, bad))
    allowed = _require(cfg["data"], "allowed_abs_substrings", "data")
    if not any(a.replace("\\", "/") in norm for a in allowed):
        raise LeakAssertionError(
            "分片不在 A5 授权的 4 个文件内：%s（授权：%s）" % (abs_path, allowed))
    return abs_path


def resolve_split_paths(cfg, regime):
    """返回 ``(train_csv, val_csv)`` 绝对路径（先过泄露断言）。"""
    if regime not in ("id", "ood"):
        raise ValueError("--regime 必须为 id 或 ood（got %r）" % (regime,))
    data = _require(cfg, "data", "cfg")
    root = _require(data, "root", "data")
    train = _require(data, "train", "data").format(root=root, regime="regime_" + regime)
    val = _require(data, "val", "data").format(root=root, regime="regime_" + regime)
    return (assert_split_allowed(os.path.join(_ROOT, train), cfg),
            assert_split_allowed(os.path.join(_ROOT, val), cfg))


def assert_val_not_in_train(train_csv, val_csv, cfg):
    """C1／§4-2：val 行不得出现在 train 分片中（`sample_id` 级交集须为空）。"""
    if not _require(cfg["data"], "val_may_not_enter_training", "data"):
        return True
    tr = {r.sample_id for r in load_samples(train_csv)}
    va = [r.sample_id for r in load_samples(val_csv)]
    overlap = sorted(set(va) & tr)
    if overlap:
        raise LeakAssertionError(
            "val 样本出现在 train 分片（%d 条，例：%s）" % (len(overlap), overlap[:3]))
    return True


def assert_ood_train_encodings(cfg, train_csv, logger=None):
    """§3-7：体制二 train 侧只允许 5 个训练编码；4 留出码命中须 = 0。"""
    a = _require(cfg, "assertions", "cfg")
    if "regime_ood" not in os.path.abspath(train_csv).replace("\\", "/"):
        return True
    allowed = set(_require(a, "ood_train_encodings", "assertions"))
    held = set(_require(a, "ood_heldout_encodings", "assertions"))
    seen, hits = set(), []
    for row in load_rows(train_csv):
        enc = (row.get("encoding_type") or "").strip()
        if not enc:
            continue
        seen.add(enc)
        if enc in held:
            hits.append("%s:%s" % (enc, row.get("sample_id")))
    outside = sorted(seen - allowed)
    if outside or hits:
        raise LeakAssertionError(
            "体制二 train 编码越界：越界=%s；留出码命中=%d（例：%s）"
            % (outside or "无", len(hits), hits[:3]))
    if logger:
        logger("OOD_ENCODING_ASSERT ok seen=%s" % ",".join(sorted(seen)))
    return True


# ---------------------------------------------------------------------------
# C3 官方 chat 模板 + 生成式二分类目标串
# ---------------------------------------------------------------------------
def build_messages(cfg, content_text):
    """按 Llama-Guard-3 **官方消息格式**组装 messages（`content` 必须是 block 列表）。"""
    p_cfg = _require(cfg, "prompt", "cfg")
    fmt = _require(p_cfg, "content_format", "prompt")
    if fmt != "list_of_text_blocks":
        raise ValueError("未知的 prompt.content_format：%r（官方格式为 list_of_text_blocks）"
                         % (fmt,))
    return [{"role": _require(p_cfg, "role", "prompt"),
             "content": [{"type": "text", "text": content_text}]}]


def build_prompt(tok, cfg, text):
    """``(prompt, truncated_text)``：左截断 512 → 官方 chat 模板 → 空会话段守卫。"""
    max_tokens = int(_require(cfg["preprocess"], "max_seq_tokens", "preprocess"))
    truncated = truncate_left(text, tok, max_tokens)
    p_cfg = _require(cfg, "prompt", "cfg")
    prompt = tok.apply_chat_template(
        build_messages(cfg, truncated),
        tokenize=False,
        add_generation_prompt=bool(_require(p_cfg, "add_generation_prompt", "prompt")),
    )
    if truncated and EMPTY_CONVERSATION in prompt:
        raise RuntimeError(
            "官方 chat 模板渲染后会话段为空 —— 样本内容未进入 prompt，判词将退化为常数。")
    return prompt, truncated


def resolve_target_ids(tok, cfg):
    """解析首生成位候选 token id，并与配置登记值逐项断言一致（禁止默认兜底）。"""
    t_cfg = _require(cfg, "target", "cfg")
    want_u = int(_require(t_cfg, "unsafe_first_token_id", "target"))
    want_s = int(_require(t_cfg, "safe_first_token_id", "target"))
    got_u = tok(_require(t_cfg, "unsafe", "target"), add_special_tokens=False)["input_ids"]
    got_s = tok(_require(t_cfg, "safe", "target"), add_special_tokens=False)["input_ids"]
    if len(got_u) != 1 or len(got_s) != 1:
        raise RuntimeError("目标串必须为**单 token**（unsafe=%r → %s；safe=%r → %s）"
                           % (_require(t_cfg, "unsafe", "target"), got_u,
                              _require(t_cfg, "safe", "target"), got_s))
    if got_u[0] != want_u or got_s[0] != want_s:
        raise RuntimeError("目标 token id 与配置登记值不一致（配置 %d/%d，tokenizer 实测 %d/%d）"
                           % (want_u, want_s, got_u[0], got_s[0]))
    return want_u, want_s


Example = namedtuple("Example",
                     ["sample_id", "text", "label", "input_ids", "n_prompt",
                      "target_ids", "sha256"])


def encode_example(tok, cfg, sample, id_unsafe, id_safe, eos_id):
    """把一个样本编码为 ``Example``。**只对目标位算 loss**（见 `forward_target_loss`）。"""
    t_cfg = _require(cfg, "target", "cfg")
    prompt, _trunc = build_prompt(tok, cfg, sample.text)
    p_ids = tok(prompt, add_special_tokens=bool(
        _require(cfg["prompt"], "add_special_tokens_tokenize", "prompt")))["input_ids"]
    tid = id_unsafe if int(sample.label) == 1 else id_safe
    tgt = [tid]
    if bool(_require(t_cfg, "append_eos", "target")):
        if eos_id is None:
            raise RuntimeError("tokenizer.eos_token_id 为 None，无法追加 eos（禁止默认值兜底）")
        tgt.append(int(eos_id))
    return Example(sample.sample_id, sample.text, int(sample.label),
                   list(p_ids) + list(tgt), len(p_ids), list(tgt), sha256_of(sample.text))


_ENC_CACHE = {}


def prepare_split(tok, cfg, csv_path, id_unsafe, id_safe, eos_id, logger=None, cap=None,
                  ids=None, cache=True, rows=None, cache_key_extra=None):
    """读分片 → 编码。返回 ``list[Example]``。

    ``ids`` 给出只编码这些**行下标**（冒烟子集用；训练样本不减、顺序不变）。
    ``rows`` 给出**外部已构造好的行**（`04_02` 增强后的 frame 或 `W5.7` 去混淆后的 frame）；
    优先于读盘，用于让课程增强／对角去混淆真正进入训练（W5.2／W5.7 接线）。
    ``cache_key_extra``：可选的缓存键附加项（如 `W5.7` 的去混淆标记）；**默认 None ⇒
    缓存键与改造前逐字节一致**（`off` 不变性）。
    """
    key = (os.path.abspath(csv_path), cap, tuple(ids) if ids else None,
           None if rows is None else len(rows))
    if cache_key_extra is not None:
        key = key + (cache_key_extra,)
    if cache and key in _ENC_CACHE:
        return _ENC_CACHE[key]
    if rows is not None:
        samples = [_PREPROCESS.Sample(str(r.get("sample_id", "")), r.get("text", ""),
                                      int(_PREPROCESS._parse_label(r.get("label"))))
                   for r in rows]
    else:
        samples = load_samples(assert_split_allowed(csv_path, cfg))
        if ids is not None:
            samples = [samples[i] for i in ids]
        elif cap is not None:
            samples = samples[:int(cap)]
    exs = [encode_example(tok, cfg, s, id_unsafe, id_safe, eos_id) for s in samples]
    if logger:
        logger("ENCODED %s n=%d" % (os.path.basename(csv_path), len(exs)))
    if cache:
        _ENC_CACHE[key] = exs
    return exs


# ---------------------------------------------------------------------------
# 组批（C10 显存护栏：eager attention 峰值 ≈ f(n × padded_len²)）
# ---------------------------------------------------------------------------
def batch_mem_guard(cfg, eval_mode=False):
    """取显存护栏参数 ``(quad_coef, linear_coef, delta_budget_mib)``（C10）。

    系数来自本机实测探针 `reports/train/tmp/probe_batch_budget.py`（训练态）
    与 `probe_eval_mem.py`（eval 态，`no_grad`）：

    - **训练态**（含反向）：``delta_MiB ≈ n·L²/512 + 1.9·n·L``；
      例：n=2/L=708 → 5201（0.69 s/步）；n=4/L=512 → 6301（1.47 s/步）；
      n=4/L=708 → 10390 **超出 8188 MiB 容量 ⇒ 溢出共享内存，单步 13.9 s（20×）**。
    - **eval 态**（`no_grad`，激活逐层释放）：``delta_MiB ≈ 3.47e-4·n·L²``；
      例：n=16/L=708 → 2779（峰值 4325 MiB，4880 tok/s）。

    护栏只是**降低单批 n**，不改样本集／epoch／网格／lr／r —— 属 C10 回退阶梯
    「batch 1」的逐批应用（已在 `DEVIATION_LEDGER.md` 登记），不是口径变更。
    """
    b = _require(cfg, "batch", "cfg")
    mm = _require(b, "mem_model", "batch")
    if eval_mode:
        return (float(_require(mm, "eval_quad_coef", "mem_model")),
                float(_require(mm, "eval_linear_coef", "mem_model")),
                float(_require(b, "eval_peak_budget_mib", "batch"))
                - float(_require(mm, "base_mib", "mem_model")))
    return (float(_require(mm, "quad_coef", "mem_model")),
            float(_require(mm, "linear_coef", "mem_model")),
            float(_require(b, "peak_budget_mib", "batch"))
            - float(_require(mm, "base_mib", "mem_model")))


def batch_cost(n, L, quad, linear):
    """单批激活显存估算（MiB），见 `batch_mem_guard` 的实测拟合式。"""
    return quad * n * L * L + linear * n * L


def make_batches(lengths, max_size, quad, linear, budget, shuffle=False, rng=None):
    """按长度邻近组批 ⇒ 动态填充最小；单批须同时满足 ``n <= max_size`` 与显存护栏。

    **只减 n，不改样本、不改 epoch、不改网格** —— 护栏属 C10 的 OOM 规避，不是口径变更。
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for idx in order:
        L = lengths[idx]
        if cur:
            n2 = len(cur) + 1
            L2 = max(cur_max, L)
            if n2 > max_size or batch_cost(n2, L2, quad, linear) > budget:
                batches.append(cur)
                cur, cur_max = [], 0
        cur.append(idx)
        cur_max = max(cur_max, L)
        if len(cur) >= max_size:
            batches.append(cur)
            cur, cur_max = [], 0
    if cur:
        batches.append(cur)
    if shuffle:
        (rng or random.Random(0)).shuffle(batches)
    return batches


# ---------------------------------------------------------------------------
# 前向：只在目标位算 logits（避免 [B, T, V] 全词表张量把 8 GB 打爆）
# ---------------------------------------------------------------------------
def forward_target_loss(llama, lm_head, input_ids, attention_mask, gather_idx, targets,
                        scale=1.0):
    """返回均值 loss（乘 ``scale``，供梯度累积 1/accum 用）。"""
    import torch
    import torch.nn.functional as F

    h = llama(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    H = h.shape[-1]
    sel = h.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, H))
    logits = lm_head(sel).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss * scale


def forward_scores(llama, lm_head, input_ids, attention_mask, id_unsafe, id_safe):
    """首生成位归一化概率 ``score = p_unsafe / (p_unsafe + p_safe)``（C3）。"""
    import torch

    h = llama(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    H = h.shape[-1]
    last = (attention_mask.sum(dim=1) - 1).clamp(min=0)
    hlast = h.gather(1, last.view(-1, 1, 1).expand(-1, 1, H)).squeeze(1)
    logits = lm_head(hlast).float()
    logp = torch.log_softmax(logits, dim=-1)
    pu = logp[:, id_unsafe].exp()
    ps = logp[:, id_safe].exp()
    return (pu / (pu + ps).clamp_min(1e-12)).detach().cpu().tolist()


# ---------------------------------------------------------------------------
# C4 τ_Ours 寻优（与 τ_B2 同制；指标单源 = `03_07.f1_score`）
# ---------------------------------------------------------------------------
def search_tau_ours(scores, labels):
    """候选 = score 全部唯一值（含 ±∞ 端点）；**F1 最大**；**并列取 τ 较大者**。

    与 `03_07.search_tau_b2` 的唯一差异 = 判正用 ``score >= tau``（C3 明定
    ``pred = (score ≥ τ_Ours)``）；F1 本身直接复用 `03_07.f1_score`，不另起一套指标实现。
    """
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if s is not None]
    if not pairs:
        return None, 0.0
    uniq = sorted(set(s for s, _ in pairs))
    best_tau, best_f1 = None, -1.0
    for tau in [NEG_INF] + uniq + [POS_INF]:      # 升序遍历 ⇒ `>=` 使并列取 τ 较大者
        pred = [1 if s >= tau else 0 for s, _ in pairs]
        f1 = f1_score([y for _, y in pairs], pred)
        if f1 >= best_f1:
            best_f1, best_tau = f1, tau
    return best_tau, best_f1


def resolve_tau_ours(scores, labels, cfg):
    """`D-47` 裁定二：τ 主口径**触发式**改用 `0.5`（**口径变更**，事前定值）。

    先按 `search_tau_ours` 取 F1 最大的 τ*（**原口径不变**），再按
    `threshold.main_criterion_override` 判定：

    - **条件一**：τ* 为**边界解** —— `τ* > boundary_definition.tau_gt` 或
      `τ* < .tau_lt`（`D-43` 裁定三 字面「>0.99／<0.01」），或（可选）
      `τ* == min(uniq)`／`== max(uniq)`；
    - **条件二**：**负类** `score max` 与 τ* 之差 `< trigger.neg_max_gap_below`；
    - 二者 **`AND`** 成立 ⇒ 主口径 `τ = on_trigger.main_tau`（**定值 `0.5`**），
      **τ* 降级为并列报告**（由调用方写入日志与 `meta.json`）。

    返回 dict：`{tau_main, tau_star, f1_star, triggered, is_boundary, neg_max, gap,
    boundary_reasons}`。**未触发时 `tau_main == tau_star` ⇒ 行为与原口径完全一致。**
    """
    tau_star, f1_star = search_tau_ours(scores, labels)
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if s is not None]
    out = {"tau_main": tau_star, "tau_star": tau_star, "f1_star": f1_star,
           "triggered": False, "is_boundary": False, "neg_max": None, "gap": None,
           "boundary_reasons": []}
    rule = (cfg.get("threshold") or {}).get("main_criterion_override") or {}
    if not bool(rule.get("enabled", False)) or not pairs or tau_star is None:
        return out

    uniq = sorted(set(s for s, _ in pairs))
    negs = [s for s, y in pairs if y == 0]
    out["neg_max"] = max(negs) if negs else None

    bd = rule.get("boundary_definition") or {}
    reasons = []
    if tau_star > float(bd.get("tau_gt", 0.99)):
        reasons.append("tau_gt")
    if tau_star < float(bd.get("tau_lt", 0.01)):
        reasons.append("tau_lt")
    if bool(bd.get("or_candidate_extreme", False)) and uniq and (
            tau_star == uniq[0] or tau_star == uniq[-1]):
        reasons.append("candidate_extreme")
    out["is_boundary"] = bool(reasons)
    out["boundary_reasons"] = reasons

    gap = (tau_star - out["neg_max"]) if out["neg_max"] is not None else None
    out["gap"] = gap
    thr = float((rule.get("trigger") or {}).get("neg_max_gap_below", 1.0e-3))
    if out["is_boundary"] and gap is not None and gap < thr:
        out["triggered"] = True
        out["tau_main"] = float((rule.get("on_trigger") or {}).get("main_tau", 0.5))
    return out


def log_tau_ours(res, logger, tag):
    """把主口径／τ* 的**并列**结果与触发状态写日志（`D-47` 裁定二「τ* 降级为并列报告」）。"""
    if res.get("triggered"):
        logger("TAU_MAIN_OVERRIDE tag=%s TRIGGERED 边界解[%s]＋负类max差=%s < 阈值 ⇒ "
               "主口径 tau=%s；tau_star=%s（F1=%.8f）并列报告"
               % (tag, "+".join(res["boundary_reasons"]), res["gap"],
                  _fmt(res["tau_main"]), repr(res["tau_star"]), res["f1_star"] or 0.0))
    else:
        logger("TAU_MAIN tag=%s not_triggered（边界解=%s；负类max差=%s）⇒ 主口径 tau=%s"
               % (tag, res["is_boundary"], res["gap"], repr(res["tau_main"])))


def apply_threshold(scores, tau):
    """``pred = (score ≥ τ)``。`score=None` ⇒ 0。"""
    return [1 if (s is not None and float(s) >= float(tau)) else 0 for s in scores]


def _exact(x):
    """**全精度**数值字面量（`F-W5.1-2` 的修复）。

    τ 的 6 位显示值（`_fmt(τ, 6)`）与全精度 τ* 可差 ~3.7e-7；在**边界解**上该差值
    足以把停在「最小正类 score」的样本翻成负例（实测 `D-47`：10/792）⇒ τ 一律按
    **全精度**记录，供下游（`W5.3`／`W6`）复算时逐位一致。
    """
    return "" if x is None else repr(float(x))


def nan_guard_violated(loss_value):
    """`D-54` 裁定三②：loss-NaN 守卫的**判据** —— 检测点 = 每个 micro-batch 的**标量** `loss`。

    **禁止**只在 epoch 末检查：`ood:42` 的 NaN 自 `EPOCH 2` **首步**即出现，
    epoch 末检查会白烧整轮（实测该 run 白烧 8.21 h）。
    `math.isfinite` 对 `±inf` 与 `nan` 均返回 `False`。
    """
    return not math.isfinite(float(loss_value))


# ---------------------------------------------------------------------------
# C5 早停（patience = 2）
# ---------------------------------------------------------------------------
class EarlyStopping:
    """监控 val F1；连续 `patience` 轮未上升即停。**不得引用任何测试集量**。"""

    def __init__(self, patience=2, min_delta=0.0, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.counter = 0
        self.history = []

    def update(self, value):
        """返回 ``(improved, counter, best)``。"""
        value = float(value)
        improved = False
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            improved = True
        else:
            self.counter += 1
        self.history.append({"value": value, "improved": improved, "counter": self.counter})
        return improved, self.counter, self.best

    def should_stop(self):
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
class Logger:
    """打印到 stdout 并缓存，供末了落 `logs/train/04_01_train_qlora_ours_<ts>.log`。"""

    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        print(msg)
        self.lines.append(str(msg))

    def dump(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(self.lines) + "\n")
        return path


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _fmt(x, nd=6):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, float):
        return round(x, nd)
    return x


def register_deviation(logger, item, reason, impact, rerun="否", approver="W0（待批）"):
    """工程性偏离逐条追加到 `DEVIATION_LEDGER.md`（只追加，禁改既有行）。"""
    path = os.path.join(_ROOT, "DEVIATION_LEDGER.md")
    row = "| %s | W5.1 | %s | %s | %s | %s | %s |\n" % (
        time.strftime("%Y-%m-%d", time.localtime()), item, reason, impact, rerun, approver)
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(row)
    except OSError as exc:
        logger("WARN 台账追加失败（不静默）：%s" % ascii(exc))
        return None
    logger("DEVIATION_LEDGER_APPEND: %s" % row.strip())
    return row.strip()


def set_determinism(seed, cfg):
    """C9：确定性门禁（`ENV_LOCK.md` §7.2 同制）。返回**显式** RNG（不用全局 RNG）。

    **N-5（`D-42` §4）修复**：`cudnn.deterministic` 只覆盖 cuDNN，**不覆盖**
    embedding backward／`index_select`／scatter-add 等（LoRA 微调正涉及这些）。
    故本函数按 `determinism.use_deterministic_algorithms`（事前定值）额外施加
    **框架级**确定性算法开关，并把 TF32 在 matmul／cuDNN 两处一并关掉。
    """
    import torch

    det = _require(cfg, "determinism", "cfg")
    if bool(_require(det, "cudnn_deterministic", "determinism")):
        torch.backends.cudnn.deterministic = True
    if not bool(_require(det, "cudnn_benchmark", "determinism")):
        torch.backends.cudnn.benchmark = False
    if "allow_tf32" in det:
        _tf32 = bool(det["allow_tf32"])
        torch.backends.cuda.matmul.allow_tf32 = _tf32
        # N-5：cuDNN 侧的 TF32 亦须显式关闭（老版 torch 无此开关，故做存在性判断）。
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = _tf32
    # N-5 核心修复：框架级确定性算法（覆盖 scatter-add／index_select／embedding backward 等）。
    if bool(det.get("use_deterministic_algorithms", False)):
        _warn_only = bool(det.get("deterministic_warn_only", False))
        torch.use_deterministic_algorithms(True, warn_only=_warn_only)
    # N-5：CUBLAS_WORKSPACE_CONFIG 亦在运行时兜底 setdefault（模块顶部已设，此处防外部覆写）。
    if "set_before_import_torch" in det:
        for _k, _v in dict(det["set_before_import_torch"]).items():
            os.environ.setdefault(str(_k), str(_v))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def _unwrap(model):
    """从 PeftModel 取出 ``(llama_model, lm_head)``。"""
    core = model.base_model.model
    return core.model, core.lm_head


def build_model(cfg, logger, lora_r=None, lora_alpha=None, with_lora=True):
    """加载 4-bit NF4 基座（+ 可选 LoRA）。返回 ``(torch, tok, model, llama, lm_head, meta)``。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    w = _require(cfg, "weights", "cfg")
    local = os.path.join(_ROOT, _require(w, "local_path", "weights"))
    local_only = bool(_require(w, "local_files_only", "weights"))
    tok = AutoTokenizer.from_pretrained(local, local_files_only=local_only)

    ld = _require(cfg, "load", "cfg")
    attn = _require(ld, "attn_implementation", "load")
    for bad in _require(ld, "forbidden_attn", "load"):
        if attn == bad:
            raise RuntimeError("禁止使用 attn_implementation=%r（C9／§0.4：全阶段 eager）" % bad)
    q = _require(ld, "quantization", "load")
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(_require(q, "load_in_4bit", "quantization")),
        bnb_4bit_quant_type=_require(q, "bnb_4bit_quant_type", "quantization"),
        bnb_4bit_use_double_quant=bool(_require(q, "bnb_4bit_use_double_quant", "quantization")),
        bnb_4bit_compute_dtype=getattr(torch,
                                       _require(q, "bnb_4bit_compute_dtype", "quantization")),
    )
    model = AutoModelForCausalLM.from_pretrained(
        local, quantization_config=bnb, device_map=_require(ld, "device_map", "load"),
        attn_implementation=attn, dtype=getattr(torch, _require(ld, "dtype", "load")),
        local_files_only=local_only)
    model.config.use_cache = False
    meta = {"attn_implementation": attn, "local_path": local,
            "lora_r": None, "lora_alpha": None, "target_modules": [], "trainable_params": 0}

    if with_lora:
        ql = _require(cfg, "qlora", "cfg")
        peft_cfg = LoraConfig(
            r=int(lora_r if lora_r is not None else _require(ql, "r", "qlora")),
            lora_alpha=int(lora_alpha if lora_alpha is not None
                           else _require(ql, "alpha", "qlora")),
            lora_dropout=float(_require(ql, "lora_dropout", "qlora")),
            bias=_require(ql, "bias", "qlora"),
            task_type=_require(ql, "task_type", "qlora"),
            target_modules=list(_require(ql, "target_modules", "qlora")))
        model = get_peft_model(model, peft_cfg)
        meta.update({"lora_r": peft_cfg.r, "lora_alpha": peft_cfg.lora_alpha,
                     "target_modules": list(peft_cfg.target_modules),
                     "trainable_params": int(sum(p.numel() for p in model.parameters()
                                                 if p.requires_grad))})
        logger("MODEL_LOADED lora_r=%d alpha=%d targets=%s trainable=%d attn=%s"
               % (peft_cfg.r, peft_cfg.lora_alpha, ",".join(peft_cfg.target_modules),
                  meta["trainable_params"], attn))
    else:
        logger("MODEL_LOADED base_only attn=%s" % attn)
    llama, lm_head = _unwrap(model) if with_lora else (model.model, model.lm_head)
    return torch, tok, model, llama, lm_head, meta


# ---------------------------------------------------------------------------
# C7 课程学习开关（逻辑归 `04_02`／W5.2；本文件**只** import 与 on/off）
# ---------------------------------------------------------------------------
def curriculum_module(cfg):
    """`04_02` 模块本体（含 `augment_attack`）；缺失／导入失败 ⇒ ``None``。

    W5.2 补记：原 `try_load_curriculum()` 只取 `CurriculumScheduler` 类，
    拿不到同模块的 `augment_attack` ⇒ `--curriculum on` 时增强**不会生效**。
    本函数把模块本体也暴露出来（返回值形态不变，旧调用点不受影响）。
    """
    cur = _require(cfg, "curriculum", "cfg")
    rel = _require(cur, "module", "curriculum")
    path = os.path.join(_ROOT, rel.replace("/", os.sep))
    if not os.path.isfile(path):
        return None
    try:
        return _load_module("train_curriculum_ours", path)
    except Exception:  # noqa: BLE001 —— 仅作优雅降级
        return None


def try_load_curriculum(cfg):
    """返回 ``(cls|None, note)``。`04_02` 缺失／接口未实现 ⇒ 一行告警 + 优雅跳过。"""
    cur = _require(cfg, "curriculum", "cfg")
    rel = _require(cur, "module", "curriculum")
    mod = curriculum_module(cfg)
    if mod is None:
        return None, "CURRICULUM_SKIP: 04_02 未落盘或导入失败（%s）—— 以 off 继续" % rel
    try:
        cls = getattr(mod, _require(cur, "entry_class", "curriculum"))
    except AttributeError:
        return None, "CURRICULUM_SKIP: 04_02 的 CurriculumScheduler 接口未实现 —— 以 off 继续"
    if not all(hasattr(cls, m) for m in ("stage_at", "select", "describe")):
        return None, "CURRICULUM_SKIP: 04_02 的 CurriculumScheduler 接口未实现 —— 以 off 继续"
    return cls, "CURRICULUM_OK: %s" % _require(cur, "entry_class", "curriculum")


def build_curriculum_frame(cfg, mod, regime, train_csv, ids, seed, logger):
    """C7：把 `04_02.augment_attack` 接进训练帧（**只作用于 train、只作用于 label==1**）。

    返回 ``(rows|None, stats|None)``。``04_02`` 缺失或 `augment.enabled = false` ⇒ ``(None, None)``
    （保持 `--curriculum on` 只做阶段过滤、不做增强）。
    """
    if mod is None or not hasattr(mod, "augment_attack"):
        return None, None
    # 课程计划本体：`--config` 指定的 yaml 里 `curriculum.config_path` 优先，
    # 解析不到再用 `04_02.default_plan_cfg()`（同一份 `configs/train/curriculum.yaml`）。
    try:
        plan = mod.resolve_plan_cfg(cfg)
    except Exception:  # noqa: BLE001 —— 解析失败 ⇒ 回落到默认计划（计划本身有断言把关）
        plan = mod.default_plan_cfg()
    aug = _require(plan, "augment", "curriculum.yaml")
    if not bool(_require(aug, "enabled", "augment")):
        logger("CURRICULUM_AUGMENT_OFF：`augment.enabled = false`")
        return None, None
    rc = _require(_require(plan, "regimes", "curriculum.yaml"), regime, "regimes")
    allowed = list(_require(rc, "train_encodings", "regimes.%s" % regime))
    p = float(_require(aug, "p", "augment"))
    cap = float(_require(aug, "ratio_cap", "augment"))
    base = load_rows(assert_split_allowed(train_csv, cfg))
    if ids is not None:
        base = [base[i] for i in ids]
    frame, rep = mod.augment_attack(base, random.Random(int(seed)), allowed, p, cap,
                                    cfg=plan)
    logger("CURRICULUM_AUGMENT rows %d→%d（抽中 %d／成功 %d／跳过 %s／丢弃 %d）；"
           "正负比 %s→%s（上界 %.2f，%s）；digest=%s"
           % (rep["n_input_rows"], rep["n_output_rows"], rep["drawn"],
              rep["augmented_ok"], rep["skipped"], rep["n_discarded"],
              rep["pos_neg_before"], rep["pos_neg_after"], cap,
              "触发" if rep["ratio_cap_binding"] else "未触发", rep["frame_digest"][:16]))
    return frame, rep


# ---------------------------------------------------------------------------
# `W5.7` `P-K'`：2×2 对角去混淆增广（`--deconfound`；`PREREG_v1.8.md` §1）
#
# 修复对象（`D-133`／`D-136`）：训练集四象限缺两格（明文攻击＝0、编码良性＝0）⇒
# 「是否编码」完美预测标签 ⇒ 编码良性 FPR 0.66–1.00、明文攻击 @τ* TPR 0.000/0.115。
# 修复 = **2×2 对角替换**（`PREREG_v1.8.md` §1，事前定值）：
#   - 攻击行（`label==1`）：`sha256(sample_id)` **最低位** `1` ⇒ 输入换 `plaintext`（未编码
#     恶意指令）；`0` ⇒ 保持 `text`（编码载荷＋宿主）。p = 0.5。
#   - 良性行（`label==0`）：同位 `1` ⇒ 输入换 `enc(text[:L])`（整段编码）；`0` ⇒ 保持明文。p = 0.5。
#   - **码选择**：`sha256(sample_id)` **次低位起**（`(h >> 1) % len(allowed)`）在允许码集内
#     确定性选取（`id` ⇒ 9 码 `ENCODING_CODES`；`ood` ⇒ 仅 5 训练码 `OOD_TRAIN_ENCODINGS`，
#     **4 留出码零暴露**）。码集常量单源 = `scripts/data/_common.py`（`V-5`/`V-6`），不自造。
#   - **`L`**：按该内容的实测扩展比 `ratio = len(enc(text))/len(text)` **反推**
#     `L = min(len(text), max(1, round(950 / ratio)))` ⇒ 输出 ≈ 950 字符；
#     `ascii_art` **下限钳制 64**（其扩展比 ≈30 ⇒ 反推 L≈32 过短，钳 64 保语义）。
#   - **替换而非新增** ⇒ 行数、正负比不变；**只改 `text`**（其余列零改动，逐列断言）。
#   - **`val` 同规**（§3-7）：验证集同一对角规则（同一 `sha256` 判定）——否则早停与 τ*
#     仍在全编码分布上选，边界解复发。
#
# 确定性（§3-3）：替换判定与码选择只依赖 `sha256(sample_id)`（`03_06.sha256_of` 单源）；
# `L` 只依赖 `(code, text)` 纯函数；**无全局 `random`／无时间依赖** ⇒ 两次运行替换集合完全相同。
# **不写任何数据面**（§3-1）：增广只在内存行上做，`data/frozen/**` 零改动。
#
# 施加时序：**在课程增强（`04_02.augment_attack`）之后** —— 若反序，被转明文的攻击行会因
# `payload ∉ text` 锚点缺失而退出增强，且增强行会把 `P(编码|攻击)` 重新推离 0.5；
# 后置可保证模型实际看到的分布满足 `P(编码|攻击) ≈ P(编码|良性) ≈ 0.5`。
# ---------------------------------------------------------------------------
DECONFOUND_TARGET_OUT_CHARS = 950      # `PREREG_v1.8` §1：编码输出目标 ≈950 字符
DECONFOUND_ASCII_ART_L_FLOOR = 64      # `PREREG_v1.8` §1：ascii_art 的 L 下限钳制

_DC_LAZY = {}


def _deconfound_encoders():
    """`V-5` 的 9 个编码器 —— **复用** `scripts/data/02_03_encode_variants.py`，不自造。"""
    if "enc" not in _DC_LAZY:
        mod = _load_module("deconfound_encoders_04_01",
                           os.path.join(_ROOT, "scripts", "data",
                                        "02_03_encode_variants.py"))
        _DC_LAZY["enc"] = mod.ENCODERS
    return _DC_LAZY["enc"]


def _deconfound_code_sets():
    """``(9 码, 5 训练码, 4 留出码)`` —— 单源 = `scripts/data/_common.py` 的 `V-5`/`V-6` 常量表。"""
    if "codes" not in _DC_LAZY:
        mod = _load_module("deconfound_codes_04_01",
                           os.path.join(_ROOT, "scripts", "data", "_common.py"))
        _DC_LAZY["codes"] = (tuple(mod.ENCODING_CODES), tuple(mod.OOD_TRAIN_ENCODINGS),
                             tuple(mod.OOD_HELDOUT_ENCODINGS))
    return _DC_LAZY["codes"]


def deconfound_allowed_codes(regime):
    """允许码集：`id` ⇒ 9 码；`ood` ⇒ 仅 5 训练码（`PREREG_v1.8` §1）。"""
    if regime not in ("id", "ood"):
        raise ValueError("deconfound: regime 必须为 id 或 ood（got %r）" % (regime,))
    all9, ood5, _held = _deconfound_code_sets()
    return all9 if regime == "id" else ood5


def deconfound_flip_and_code(sample_id, regime):
    """纯函数：``(replace, code)``（`PREREG_v1.8` §1）。

    - ``h = int(sha256(sample_id), 16)``（`sha256_of` = `03_06` 单源）；
    - **最低位** ``h & 1 == 1`` ⇒ 该行替换（p = 0.5）；``0`` ⇒ 保持原 ``text``；
    - **次低位起** ``allowed[(h >> 1) % len(allowed)]`` ⇒ 允许码集内确定性选码
      （仅良性行真正使用；攻击行换 `plaintext` 无需码）。
    """
    sid = str(sample_id)
    if not sid:
        raise AssertionError("deconfound: sample_id 为空（无法做 sha256 判定）")
    allowed = deconfound_allowed_codes(regime)
    h = int(sha256_of(sid), 16)
    return (h & 1) == 1, allowed[(h >> 1) % len(allowed)]


def deconfound_L(code, text):
    """``L`` 按**实测扩展比**反推使 ``enc(text[:L])`` 输出 ≈950 字符（`PREREG_v1.8` §1）。

    - ``ratio = len(enc(text)) / len(text)``（该内容在该码下的扩展比）；
    - ``L = min(len(text), max(1, round(950 / ratio)))``（「按扩展比反推」）；
    - ``ascii_art`` 另做**下限钳制 64**（`round(950/30)≈32` 过短 ⇒ 钳 64 保语义）。

    纯函数、无 IO；`text` 为空或编码结果为空一律显式报错（禁止默认值兜底）。
    """
    if not text:
        raise AssertionError("deconfound_L: text 为空（code=%s）" % code)
    encoders = _deconfound_encoders()
    if code not in encoders:
        raise ValueError("deconfound_L: 未知编码 %r（允许：%s）" % (code, sorted(encoders)))
    full = encoders[code][0](text)
    if not full:
        raise AssertionError("deconfound_L: enc(text) 为空（code=%s）" % code)
    ratio = len(full) / float(len(text))
    if ratio <= 0:
        raise AssertionError("deconfound_L: 扩展比非正（code=%s ratio=%r）" % (code, ratio))
    L = int(round(float(DECONFOUND_TARGET_OUT_CHARS) / ratio))
    L = max(1, min(len(text), L))
    if code == "ascii_art":
        L = max(L, DECONFOUND_ASCII_ART_L_FLOOR)   # 下限钳 64（超出 len(text) 由切片自然截断）
    return L


def deconfound_apply(rows, regime, enabled=True):
    """2×2 对角去混淆的**核心纯函数**（无 IO／无 RNG／无日志）。``(new_rows, stats|None)``。

    - ``enabled=False`` ⇒ **逐字**返回输入（同一对象；`off` 不变性）；
    - 只改每行的 ``text``；其余列零改动（**逐列断言**）；**替换而非新增** ⇒ 行数／正负比不变；
    - `ood` 时增广码分布 ⊆ 5 训练码（4 留出码零暴露，断言）。
    """
    if not enabled:
        return rows, None
    allowed = deconfound_allowed_codes(regime)
    _all9, _ood5, held = _deconfound_code_sets()
    encoders = _deconfound_encoders()
    out = []
    n_pos = n_neg = 0
    atk_plain = ben_enc = 0
    code_counts = {c: 0 for c in allowed}
    for i, row in enumerate(rows):
        label = int(_PREPROCESS._parse_label(row.get("label")))
        sid = str(row.get("sample_id", ""))
        replace, code = deconfound_flip_and_code(sid, regime)
        new = dict(row)
        if label == 1:
            n_pos += 1
            if replace:
                pt = row.get("plaintext") or ""
                if not pt:
                    raise AssertionError(
                        "deconfound: 攻击行 %s（第 %d 行）被选中转明文但 `plaintext` 为空"
                        % (sid, i))
                new["text"] = pt
                atk_plain += 1
        elif label == 0:
            n_neg += 1
            if replace:
                text = row.get("text") or ""
                if not text:
                    raise AssertionError(
                        "deconfound: 良性行 %s（第 %d 行）被选中整段编码但 `text` 为空"
                        % (sid, i))
                new["text"] = encoders[code][0](text[:deconfound_L(code, text)])
                ben_enc += 1
                code_counts[code] += 1
        else:
            raise AssertionError("deconfound: label 既非 0 也非 1（第 %d 行 %s）" % (i, sid))
        out.append(new)

    # —— §3-5 行数／比例不变（替换而非新增）——
    if len(out) != len(rows):
        raise AssertionError("deconfound: 行数被改动 %d→%d（只允许替换）" % (len(rows), len(out)))
    if atk_plain > n_pos or ben_enc > n_neg:
        raise AssertionError("deconfound: 替换计数越界（attacks %d/%d，benign %d/%d）"
                             % (atk_plain, n_pos, ben_enc, n_neg))

    # —— §3-6 只改 `text`（其余列零改动，逐列断言）——
    for i, (a, b) in enumerate(zip(rows, out)):
        if set(a.keys()) != set(b.keys()):
            raise AssertionError("deconfound: 第 %d 行列集合被改动" % i)
        for k in a:
            if k == "text":
                continue
            if a[k] != b[k]:
                raise AssertionError("deconfound: 第 %d 行列 %r 被改动（只允许改 text）" % (i, k))

    # —— §3-4 `ood` 留出码零暴露 ——
    used = {c for c, n in code_counts.items() if n > 0}
    if used - set(allowed):
        raise AssertionError("deconfound: 增广码越出允许集：%s" % sorted(used - set(allowed)))
    if regime == "ood":
        hits = sorted(used & set(held))
        if hits:
            raise AssertionError("deconfound: 体制二增广命中留出码：%s" % hits)

    stats = {
        "regime": regime,
        "n_rows": len(out),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "attacks_plaintext": atk_plain,
        "benign_encoded": ben_enc,
        "codes": [code_counts[c] for c in allowed],      # 与 allowed 同序（含 0 计数）
        "code_names": list(allowed),
        "codes_used": sorted(used),
    }
    return out, stats


def deconfound_log(stats, logger, tag):
    """契约日志行（`W5.7` §2）：``DECONFOUND=on attacks_plaintext=<k>/<n_pos> …``。"""
    if stats is None:
        logger("DECONFOUND=off")
        return "DECONFOUND=off"
    dist = ",".join("%s:%d" % (c, n) for c, n in zip(stats["code_names"], stats["codes"]))
    line = ("DECONFOUND=on attacks_plaintext=%d/%d benign_encoded=%d/%d codes=%s split=%s"
            % (stats["attacks_plaintext"], stats["n_pos"],
               stats["benign_encoded"], stats["n_neg"], dist, tag))
    logger(line)
    return line


def deconfound_load_frame(cfg, csv_path, ids=None):
    """读分片整行并按行下标子集（`04_00` 冒烟钩子同规）⇒ 供去混淆在**行**上操作。

    去混淆需要 `plaintext` 列（攻击行转明文用）⇒ 必须走 `load_rows`（`load_samples` 只留
    `(sample_id, text, label)`），不能复用其采样路径。
    """
    rows = load_rows(assert_split_allowed(csv_path, cfg))
    if ids is not None:
        rows = [rows[i] for i in ids]
    return rows


def resolve_log_dir(args):
    """日志目录解析（`N-16` 同规 `05_01.resolve_log_dir`，`D-82` 裁定五）。

    优先级：① 环境变量 `CODEOBF_LOG_DIR`；② `--self-check` ⇒ `logs/train/tmp/`
    （已 gitignore，`D-15`）；③ `args.log_dir`（**与改造前逐字一致**）。
    """
    env = os.environ.get("CODEOBF_LOG_DIR")
    if env:
        return env
    if getattr(args, "self_check", False):
        return os.path.join(_ROOT, "logs", "train", "tmp")
    return args.log_dir


def deconfound_self_check(cfg, logger):
    """`--self-check`：去混淆数据构造自检（`W5.7` §3-1～§3-7），不加载模型、不训练。

    只读 A5 授权的 4 个分片（train/val × id/ood），**不触 `sealed_test`**。
    逐条打印 ``SELFCHECK S<k> §3-<n> ... ok``；全过返回检查条数 ``k``；任一失败抛
    ``AssertionError``（⇒ 退出码非零）。`off` 的**逐 step loss Δ=0** 由 smoke 另证
    （本自检只覆盖数据面不变性）。

    成本：全量变换 **2 遍**（第 2 遍供 §3-3 确定性对照），其余检查在已算结果上复核。
    """
    ok = []

    def _check(name, fn):
        fn()
        ok.append(name)
        logger("SELFCHECK S%d %s ok" % (len(ok), name))

    frames, paths = {}, {}
    for regime in ("id", "ood"):
        tr, va = resolve_split_paths(cfg, regime)
        paths[(regime, "train")] = tr
        paths[(regime, "val")] = va
    for key, p in paths.items():
        frames[key] = deconfound_load_frame(cfg, p)

    # —— 全量变换第 1 遍（各检查在其结果上复核）＋ §3-1 的 md5 前后对照 ——
    md5_before = {k: md5_of(v) for k, v in paths.items()}
    transformed, stats = {}, {}
    for key, rows in frames.items():
        transformed[key], stats[key] = deconfound_apply(rows, key[0], enabled=True)
    md5_after = {k: md5_of(v) for k, v in paths.items()}

    # S1 §3-1 不改冻结数据：增广只在内存行上做 ⇒ 4 个分片 md5 前后不变
    def _s1():
        if md5_before != md5_after:
            raise AssertionError("§3-1 冻结分片 md5 被改动：%s vs %s"
                                 % (md5_before, md5_after))
    _check("§3-1 不改冻结数据（4 分片 md5 前后不变）", _s1)

    # S2 §3-2 off 不变（数据面）：enabled=False ⇒ 同一对象逐字返回
    def _s2():
        for key, rows in frames.items():
            out, st = deconfound_apply(rows, key[0], enabled=False)
            if out is not rows or st is not None:
                raise AssertionError("§3-2 off 未逐字返回输入（%s）" % (key,))
        if build_parser().parse_args(["--self-check"]).deconfound != "off":
            raise AssertionError("§3-2 `--deconfound` 默认必须 off")
    _check("§3-2 off 不变（数据面逐字返回；默认 off）", _s2)

    # S3 §3-3 确定性：第 2 遍全量施加与第 1 遍逐字相同（替换集合与码分布完全一致）
    def _s3():
        for key, rows in frames.items():
            again, st2 = deconfound_apply(rows, key[0], enabled=True)
            if [r["text"] for r in again] != [r["text"] for r in transformed[key]] \
                    or st2 != stats[key]:
                raise AssertionError("§3-3 两次施加不一致（%s）" % (key,))
    _check("§3-3 确定性（两次施加逐字相同）", _s3)

    # S4a §3-4 id 码集 ⊆ 9 码
    def _s4a():
        _all9, _ood5, _held = _deconfound_code_sets()
        for split in ("train", "val"):
            used = stats[("id", split)]["codes_used"]
            if set(used) - set(_all9):
                raise AssertionError("§3-4 id 增广码越界：%s" % used)
    _check("§3-4 id 增广码 ⊆ 9 码", _s4a)

    # S4b §3-4 ood 码集 ⊆ 5 训练码、4 留出码零暴露（真实 train+val 全量）
    def _s4b():
        _all9, ood5, held = _deconfound_code_sets()
        for split in ("train", "val"):
            used = stats[("ood", split)]["codes_used"]
            if set(used) - set(ood5):
                raise AssertionError("§3-4 ood 增广码越界：%s" % used)
            hits = sorted(set(used) & set(held))
            if hits:
                raise AssertionError("§3-4 ood 留出码命中：%s" % hits)
    _check("§3-4 ood 增广码 ⊆ 5 训练码且留出码零暴露", _s4b)

    # S5 §3-5 行数／正负比不变
    def _s5():
        for key, rows in frames.items():
            st = stats[key]
            n_pos = sum(1 for r in rows if int(_PREPROCESS._parse_label(r.get("label"))) == 1)
            if st["n_rows"] != len(rows) or st["n_pos"] != n_pos \
                    or st["n_neg"] != len(rows) - n_pos:
                raise AssertionError("§3-5 行数／比例被改动（%s）：%s vs n=%d pos=%d"
                                     % (key, st, len(rows), n_pos))
    _check("§3-5 行数／正负比不变（替换而非新增）", _s5)

    # S6 §3-6 只改 text（其余列零改动）
    def _s6():
        for key, rows in frames.items():
            for i, (a, b) in enumerate(zip(rows, transformed[key])):
                for col in a:
                    if col == "text":
                        continue
                    if a[col] != b[col]:
                        raise AssertionError("§3-6 %s 第 %d 行列 %r 被改动" % (key, i, col))
    _check("§3-6 只改 text（其余列零改动）", _s6)

    # S7 §3-7 val 同规：val 的决策与 `deconfound_flip_and_code` 复算一致 ＋ run_train 接线在位
    def _s7():
        for regime in ("id", "ood"):
            rows = frames[(regime, "val")]
            out = transformed[(regime, "val")]
            for a, b in zip(rows, out):
                sid = str(a.get("sample_id", ""))
                label = int(_PREPROCESS._parse_label(a.get("label")))
                replace, _code = deconfound_flip_and_code(sid, regime)
                changed = a["text"] != b["text"]
                if label == 1 and replace != changed:
                    raise AssertionError("§3-7 val 攻击行决策不一致：%s" % sid)
                if label == 0 and replace and b["text"] == a["text"]:
                    raise AssertionError("§3-7 val 良性行被选中却未替换：%s" % sid)
        src_path = os.path.join(_ROOT, "scripts", "train", "04_01_train_qlora_ours.py")
        with open(src_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        fn_src = src.split("def run_train(", 1)[1].split("\ndef ", 1)[0]
        if "deconfound_apply(val_rows" not in fn_src or "deconfound_apply(train_rows" not in fn_src:
            raise AssertionError("§3-7 run_train 未把 train/val 均接入 deconfound_apply")
    _check("§3-7 val 同规（决策复算一致 ＋ run_train 接线在位）", _s7)

    return len(ok)





# ---------------------------------------------------------------------------
# C8 checkpoint 命名与落盘
# ---------------------------------------------------------------------------
def apply_model_key_override(cfg, model_key):
    """`W5.6`：把 `--model-key` 覆写进 `cfg["checkpoint"]["model_key"]`（**就地改 `cfg`**）。

    - `model_key` 为 `None`／空 ⇒ **不改动 `cfg`**（返回 `None`）⇒ 默认行为与改造前逐字节一致。
    - 否则 `cfg["checkpoint"]["model_key"]` := `str(model_key)` ⇒ 影响 `run_id` 的 `{model}` 槽
      （消融 A／B 组以此为 `llamaguard1b_A`／`llamaguard1b_B`，使 `run_id` 自识别）。
    - 缺少 `checkpoint.model_key` 时**显式报错**（不静默兜底，与 `_require` 同口径）。
    """
    if model_key is None or str(model_key) == "":
        return None
    ck_node = cfg.get("checkpoint")
    if not isinstance(ck_node, dict) or "model_key" not in ck_node:
        raise KeyError("配置缺少必需键 'model_key'（位置：checkpoint，--model-key 覆写用）")
    ck_node["model_key"] = str(model_key)
    return str(model_key)


def run_id_of(cfg, regime, seed, ts):
    ck = _require(cfg, "checkpoint", "cfg")
    return _require(ck, "run_id_format", "checkpoint").format(
        stage=_require(cfg, "stage", "cfg"), seq=_require(cfg, "seq", "cfg"),
        model=_require(ck, "model_key", "checkpoint"), regime=regime, seed=seed, ts=ts)


def checkpoint_dir(out_root, cfg, regime, seed, step, ts):
    ck = _require(cfg, "checkpoint", "cfg")
    return os.path.join(os.path.abspath(out_root), run_id_of(cfg, regime, seed, ts),
                        _require(ck, "seed_dir_format", "checkpoint").format(seed=seed),
                        _require(ck, "step_dir_format", "checkpoint").format(step=step))


def save_checkpoint(model, tok, cfg, out_root, regime, seed, step, ts, meta, logger=None):
    d = checkpoint_dir(out_root, cfg, regime, seed, step, ts)
    if os.path.exists(d):
        raise FileExistsError("禁止覆盖同名 checkpoint：%s" % d)
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)
    if bool(_require(cfg["checkpoint"], "save_tokenizer", "checkpoint")):
        tok.save_pretrained(d)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, sort_keys=True)
    if logger:
        logger("CHECKPOINT_SAVED %s" % d)
    return d


def write_best_pointer(out_root, cfg, regime, seed, ts, best, logger=None):
    ck = _require(cfg, "checkpoint", "cfg")
    path = os.path.join(os.path.abspath(out_root), run_id_of(cfg, regime, seed, ts),
                        _require(ck, "best_pointer_file", "checkpoint"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(best, fh, ensure_ascii=False, indent=2, sort_keys=True)
    if logger:
        logger("BEST_POINTER %s" % path)
    return path


# ---------------------------------------------------------------------------
# 训练一轮 / 评估一轮
# ---------------------------------------------------------------------------
def _collate(exs, idxs, pad_id, torch):
    maxlen = max(len(exs[i].input_ids) for i in idxs)
    K = len(exs[idxs[0]].target_ids)
    B = len(idxs)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    gather = torch.zeros((B, K), dtype=torch.long)
    targets = torch.zeros((B, K), dtype=torch.long)
    for b, i in enumerate(idxs):
        ids = exs[i].input_ids
        L = len(ids)
        input_ids[b, :L] = torch.tensor(ids, dtype=torch.long)
        attn[b, :L] = 1
        start = exs[i].n_prompt - 1
        gather[b, :] = torch.tensor([start + j for j in range(K)], dtype=torch.long)
        targets[b, :] = torch.tensor(exs[i].target_ids, dtype=torch.long)
    return input_ids, attn, gather, targets


def _prompt_tensors(exs, idxs, torch, device):
    """评估用：只取 prompt 位（去掉目标串），右填充 + attention_mask。"""
    sub = [exs[i] for i in idxs]
    maxlen = max(e.n_prompt for e in sub)
    B = len(idxs)
    p_ids = torch.full((B, maxlen), 0, dtype=torch.long)
    p_attn = torch.zeros((B, maxlen), dtype=torch.long)
    for b, e in enumerate(sub):
        p_ids[b, :e.n_prompt] = torch.tensor(e.input_ids[:e.n_prompt], dtype=torch.long)
        p_attn[b, :e.n_prompt] = 1
    return p_ids.to(device), p_attn.to(device)


def evaluate(model, llama, lm_head, torch, exs, id_unsafe, id_safe, cfg, rng=None,
             logger=None, tag="val"):
    """C9：eval 态 + no_grad + G-EVAL 双断言。返回逐样本 ``scores`` 列表。

    G-EVAL 断言先 `model.eval()` 再断言 `training is False` —— 顺序不可颠倒：
    `peft.get_peft_model()` 会把 `training` 置为 `True`（实测，peft 0.21.1），
    而 C9 禁止在代码里出现 `.train(`（故本模块**从不**调用 train 模式），
    只能由 eval 侧显式切回。基座 `attention_dropout=0.0` 且 LoRA `dropout=0.0`
    ⇒ 该标志对前向**无数值影响**（见本文件头「train／eval 态的实现说明」）。
    """
    model.eval()
    if model.training:
        raise RuntimeError("G-EVAL 断言失败：model.eval() 后仍处 train 模式")
    b_cfg = _require(cfg, "batch", "cfg")
    lengths = [e.n_prompt for e in exs]
    e_quad, e_linear, e_budget = batch_mem_guard(cfg, eval_mode=True)
    batches = make_batches(lengths, int(_require(b_cfg, "eval_max_size", "batch")),
                           e_quad, e_linear, e_budget, False, rng)
    device = next(model.parameters()).device
    scores = [None] * len(exs)
    with torch.no_grad():
        if torch.is_grad_enabled():
            raise RuntimeError("G-EVAL 断言失败：grad 已启用（必须 torch.no_grad()）")
        for idxs in batches:
            p_ids, p_attn = _prompt_tensors(exs, idxs, torch, device)
            vals = forward_scores(llama, lm_head, p_ids, p_attn, id_unsafe, id_safe)
            for b, i in enumerate(idxs):
                scores[i] = vals[b]
    if logger:
        logger("EVAL %s n=%d n_micro=%d" % (tag, len(exs), len(batches)))
    return scores


# ---------------------------------------------------------------------------
# `D-48` 可续跑：**step 级精确续跑**（state 落盘／载入）
#
# 设计要点（`D-48` 裁定一／三）：
# - 落盘粒度 = 每 `resume.state_every_opt_steps`（默认 100）个 optimizer step，
#   **只在 `opt.step()` 之后且 `pending == 0`** ⇒ **不存梯度**；
# - 位置 = 独立 `state/` 子目录（**不复用** `save_checkpoint`，其 `FileExistsError`
#   与「最优推理 ckpt」语义冲突、会污染 `04_04` 按 `best.json` 取 ckpt 的路径）；
# - 滚动双份 `latest`／`prev` ＋ **原子写**（tmp ＋ `os.replace`）；
# - **不写 tokenizer**。
# ---------------------------------------------------------------------------
def _state_path_of(state_dir):
    """接受「含 `latest/` 的状态根」或「直接指向 `latest/`」两种写法。"""
    d = os.path.abspath(state_dir)
    if os.path.isfile(os.path.join(d, "state.json")):
        return d
    cand = os.path.join(d, "latest")
    if os.path.isfile(os.path.join(cand, "state.json")):
        return cand
    raise FileNotFoundError(
        "续跑 state 缺失：%s（期望含 `state.json` 的目录，或其 `latest/` 子目录）" % d)


def _rng_state_jsonable(state):
    """`random.Random.getstate()` → JSON 可序列化（版本 ＋ 625 个整数 ＋ gauss_next）。"""
    version, internal, gauss = state
    return [int(version), [int(x) for x in internal], gauss]


def _rng_state_from_json(payload):
    return (int(payload[0]), tuple(int(x) for x in payload[1]), payload[2])


def _torch_rng_payload(torch):
    p = {"torch": torch.get_rng_state().tolist()}
    if torch.cuda.is_available():
        p["cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    return p


def _restore_torch_rng(torch, payload):
    if not payload:
        return
    torch.set_rng_state(torch.ByteTensor(payload["torch"]))
    if torch.cuda.is_available() and payload.get("cuda"):
        torch.cuda.set_rng_state_all([torch.ByteTensor(s) for s in payload["cuda"]])


def save_resume_state(state_dir, payload, torch, logger=None):
    """原子写 ＋ 滚动双份（`D-48` 裁定三）。返回 `latest` 的绝对路径。"""
    import shutil
    state_dir = os.path.abspath(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    latest = os.path.join(state_dir, "latest")
    prev = os.path.join(state_dir, "prev")
    tmp = os.path.join(state_dir, ".tmp_new")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    torch.save(payload["model_state"], os.path.join(tmp, "model_state.pt"))
    torch.save(payload["opt_state"], os.path.join(tmp, "opt_state.pt"))
    if payload.get("best_state") is not None:
        torch.save(payload["best_state"], os.path.join(tmp, "best_state.pt"))
    with open(os.path.join(tmp, "state.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload["meta"], fh, ensure_ascii=False, indent=2, sort_keys=True)
    if os.path.isdir(prev):
        shutil.rmtree(prev)
    if os.path.isdir(latest):
        os.replace(latest, prev)          # 滚动：旧 latest → prev
    os.replace(tmp, latest)               # 原子替换
    if logger:
        logger("RESUME_STATE_SAVED %s epoch=%s cursor=%s opt_steps=%s"
               % (os.path.relpath(latest, _ROOT).replace("\\", "/"),
                  payload["meta"].get("epoch"), payload["meta"].get("cursor"),
                  payload["meta"].get("opt_steps")))
    return latest


def load_resume_state(state_dir, torch):
    """读 `latest/`，返回 `(meta, model_state, opt_state, best_state|None)`。"""
    d = _state_path_of(state_dir)
    with open(os.path.join(d, "state.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    ms = torch.load(os.path.join(d, "model_state.pt"), map_location="cpu",
                    weights_only=False)
    osd = torch.load(os.path.join(d, "opt_state.pt"), map_location="cpu",
                     weights_only=False)
    bs_path = os.path.join(d, "best_state.pt")
    bs = (torch.load(bs_path, map_location="cpu", weights_only=False)
          if os.path.exists(bs_path) else None)
    return meta, ms, osd, bs


def train_one_run(cfg, args, tok, model, llama, lm_head, torch, train_exs, val_exs,
                  id_unsafe, id_safe, logger, ts, lora_r=None, lora_alpha=None, lr=None,
                  curriculum_on=False, curriculum_cls=None, train_rows=None):
    """一次完整训练：从头初始化 → 训练 → 每轮 val → 早停 → 保存最优 checkpoint。

    返回 dict（含 `best`／`curve`／`wall_seconds`／`peak_memory_mib`／`best_state`）。
    """
    ql = _require(cfg, "qlora", "cfg")
    b_cfg = _require(cfg, "batch", "cfg")
    es_cfg = _require(cfg, "early_stopping", "cfg")
    seed = int(args.seed)
    rng = set_determinism(seed, cfg)

    epochs = int(args.epochs) if args.epochs else int(_require(ql, "epochs", "qlora"))
    accum = int(_require(ql, "grad_accum", "qlora"))
    lr = float(lr if lr is not None else _require(ql, "lr", "qlora"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            betas=tuple(_require(ql, "betas", "qlora")),
                            eps=float(_require(ql, "eps", "qlora")),
                            weight_decay=float(_require(ql, "weight_decay", "qlora")))
    clip = float(_require(ql, "max_grad_norm", "qlora"))
    device = next(model.parameters()).device
    pad_id = int(tok.eos_token_id or 0)

    max_size = int(_require(b_cfg, "max_size", "batch"))
    quad, linear, budget = batch_mem_guard(cfg)
    lens = [len(e.input_ids) for e in train_exs]
    micro_batches = make_batches(lens, max_size, quad, linear, budget,
                                 bool(_require(b_cfg, "shuffle_batch_order", "batch")), rng)
    sizes = [len(b) for b in micro_batches]
    logger("BATCHES n_micro=%d max_size=%d delta_budget=%.0fMiB sizes(min/mean/max)=%d/%.2f/%d"
           % (len(micro_batches), max_size, budget, min(sizes),
              sum(sizes) / max(1, len(sizes)), max(sizes)))

    scheduler = None
    if curriculum_on and curriculum_cls is not None:
        total_steps = max(1, len(micro_batches) // max(1, accum))
        try:
            scheduler = curriculum_cls(cfg, args.regime, total_steps, rng)
            logger("CURRICULUM_ON %s" % type(scheduler).__name__)
            # `N-7` 裁定③（W5.1＋W5.2 混合窗口）：把**全量路径**上的「阶段计划」与
            # 「良性子采样比」**留证** —— 此前 dry-run 与冒烟分别进行，二者从未在
            # 全量路径上同时验证。`describe()`／`ratio_report()` 均为**只读**
            # （`select()` 不消耗 RNG、不改 frame）⇒ 本段**不影响确定性**。
            try:
                for _row in scheduler.describe():
                    logger("CURRICULUM_STAGE stage=%s start=%s end=%s levels=%s enc=%s cumulative=%s"
                           % (_row["stage"], _row["start_step"], _row["end_step"],
                              _row["difficulty_levels"], _row["encodings"], _row["cumulative"]))
                if train_rows is not None:
                    _np, _nn, _rt, _tg, _ok = scheduler.ratio_report(train_rows, 1)
                    logger("CURRICULUM_RATIO stage=%d step=1 pos=%d neg=%d ratio=%.4f target=%.4f ok=%s"
                           % (scheduler.stage_at(1), _np, _nn, _rt, _tg, _ok))
            except Exception as exc:  # noqa: BLE001 —— 留证失败不得影响训练
                logger("CURRICULUM_EVIDENCE_SKIP: %s" % ascii(exc))
        except Exception as exc:  # noqa: BLE001
            logger("CURRICULUM_SKIP: 构造失败（%s）—— 以 off 继续" % ascii(exc))
            scheduler = None

    stopper = EarlyStopping(patience=int(_require(es_cfg, "patience", "early_stopping")),
                            min_delta=float(_require(es_cfg, "min_delta", "early_stopping")),
                            mode=_require(es_cfg, "mode", "early_stopping"))
    curve_rows = []
    best = {"step": None, "val_f1": None, "dir": None, "tau": None, "epoch": None,
            "tpr": None, "fpr": None}
    best_state = None
    opt_steps, global_micro = 0, 0
    t_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stopped_early = False
    n_curriculum_skipped = 0        # W5.2：整批被课程阶段排除而跳过的 micro-batch 计数

    # ---- `D-48`：续跑载入（**未传 `--resume` 时本段整体为空 ⇒ 与原行为逐字节一致**）----
    resume_dir = (os.path.abspath(getattr(args, "resume"))
                  if getattr(args, "resume", None) else None)
    _rs_cfg = cfg.get("resume") or {}
    state_every = int(_rs_cfg.get("state_every_opt_steps", 100))
    _stop_after = getattr(args, "stop_after_steps", None)
    stop_after = int(_stop_after) if _stop_after else None
    # `D-54` 裁定三①：loss-NaN 守卫**默认启用**；`--no-nan-guard` 关闭
    # （关闭 ⇒ 与改造前**逐字节不变**；开启在**正常路径上零副作用**，仅非有限 loss 时触发）
    nan_guard = (bool(getattr(args, "nan_guard", True))
                 and bool((cfg.get("nan_guard") or {}).get("enabled", True)))
    start_epoch, resume_cursor, resume_order = 1, 0, None
    epoch_loss_seed, n_micro_seed = 0.0, 0
    seg_t0 = time.time()
    segments = [{"start_step": 0, "end_step": None, "wall_seconds": None}]
    if resume_dir and not (os.path.isfile(os.path.join(resume_dir, "state.json")) or
                           os.path.isfile(os.path.join(resume_dir, "latest", "state.json"))):
        # `--resume` 指向**空目录** ⇒ 视为「新跑并在该目录落盘」（供分 2 段自证的第 1 段使用）
        logger("RESUME_STATE_INIT %s（无既有 state ⇒ 新跑，并在该目录落盘）"
               % os.path.relpath(resume_dir, _ROOT).replace("\\", "/"))
        resume_dir = resume_dir
    elif resume_dir:
        _meta, _ms, _osd, _bs = load_resume_state(resume_dir, torch)
        _tr_names = [n for n, p in model.named_parameters() if p.requires_grad]
        if len(_ms) != len(_tr_names):
            raise ValueError("**拒绝静默续跑**：state 可训练张量 %d 个、当前模型 %d 个"
                             % (len(_ms), len(_tr_names)))
        model.load_state_dict(_ms, strict=False)
        opt.load_state_dict(_osd)
        rng.setstate(_rng_state_from_json(_meta["rng_state"]))
        _restore_torch_rng(torch, _meta.get("torch_rng"))
        start_epoch = int(_meta["epoch"])
        resume_cursor = int(_meta["cursor"])
        resume_order = _meta.get("order")
        opt_steps = int(_meta["opt_steps"])
        global_micro = int(_meta["global_micro"])
        epoch_loss_seed = float(_meta.get("epoch_loss", 0.0))
        n_micro_seed = int(_meta.get("n_micro", 0))
        n_curriculum_skipped = int(_meta.get("n_curriculum_skipped", 0))
        stopper.best = _meta["stopper"]["best"]
        stopper.counter = int(_meta["stopper"]["counter"])
        stopper.history = list(_meta["stopper"]["history"])
        best = dict(_meta["best"])
        best_state = _bs
        curve_rows = list(_meta.get("curve_rows") or [])
        _prev = [dict(x) for x in (_meta.get("segments") or [])]
        if _prev:
            _prev[-1]["end_step"] = opt_steps
        segments = _prev + [{"start_step": opt_steps, "end_step": None, "wall_seconds": None}]
        logger("RESUME_LOADED %s epoch=%d cursor=%d opt_steps=%d order=%s 本段=第%d段"
               % (os.path.relpath(resume_dir, _ROOT).replace("\\", "/"), start_epoch,
                  resume_cursor, opt_steps,
                  "显式恢复" if resume_order is not None else "重新打乱", len(segments)))
        # `D-50` 裁定二①（修 `N-8`）：用 state 的 `ts` **覆盖本进程新生成的 `ts`** ⇒
        # 下游 `save_checkpoint`／`write_best_pointer`／`write_curve`／`write_threshold_curve`
        # **全部沿用该 ts** ⇒ 续跑落回**同一 `run_id` 目录**，不再分裂。
        if _meta.get("ts"):
            logger("RESUME_TS_RESTORED %s（本进程新生成 ts=%s 被覆盖 ⇒ run_id=%s）"
                   % (_meta["ts"], ts, _meta.get("run_id")))
            ts = str(_meta["ts"])

    def _segments_now():
        s = [dict(x) for x in segments]
        s[-1]["end_step"] = int(opt_steps)
        s[-1]["wall_seconds"] = round(time.time() - seg_t0, 3)
        return s

    def _snapshot(epoch_, cursor_, order_):
        """`D-48` 裁定二 的状态快照（**落盘点恒有 `pending == 0` ⇒ 不存梯度**）。"""
        return {
            "model_state": {n: p.detach().to("cpu").clone()
                            for n, p in model.named_parameters() if p.requires_grad},
            "opt_state": opt.state_dict(),
            "best_state": best_state,
            "meta": {
                "epoch": int(epoch_), "cursor": int(cursor_),
                "order": (None if order_ is None else [int(x) for x in order_]),
                "opt_steps": int(opt_steps), "global_micro": int(global_micro),
                "epoch_loss": float(epoch_loss), "n_micro": int(n_micro),
                "n_curriculum_skipped": int(n_curriculum_skipped),
                "rng_state": _rng_state_jsonable(rng.getstate()),
                "torch_rng": _torch_rng_payload(torch),
                "stopper": {"best": stopper.best, "counter": int(stopper.counter),
                            "history": stopper.history},
                "best": best, "curve_rows": curve_rows,
                "segments": _segments_now(),
                # `D-50` 裁定二①：**产物目录连续性**（`D-48` 裁定二 清单**第 10 项**）——
                # 续跑须落回**同一 `run_id` 目录**，故 `ts`／`run_id` 必须随 state 走。
                "ts": str(ts), "run_id": run_id_of(cfg, args.regime, seed, ts),
            },
        }

    cur_stage_logged = None                 # `N-7` 裁定③：阶段切点留证用（只读，不消耗 RNG）
    # `D-48`：改为打乱**下标序列** `order`（与打乱 `micro_batches` 本体**等价**，
    # 且 `random.Random.shuffle` 对同长列表消耗的 RNG 完全相同）⇒ 未传 `--resume` 时逐字节一致。
    order = list(range(len(micro_batches)))
    for epoch in range(start_epoch, epochs + 1):
        if epoch == start_epoch and resume_order is not None:
            order = [int(x) for x in resume_order]   # 关键项：显式恢复本 epoch 的排列
            _cur = resume_cursor                     #            ＋ 已消费游标
        else:
            rng.shuffle(order)
            _cur = 0
        if epoch == start_epoch:
            epoch_loss, n_micro, pending = epoch_loss_seed, n_micro_seed, 0
        else:
            epoch_loss, n_micro, pending = 0.0, 0, 0
        consumed = _cur
        for k in order[_cur:]:
            consumed += 1
            idxs = micro_batches[k]
            use = idxs
            if scheduler is not None:
                try:
                    allowed = scheduler.select(train_rows, opt_steps + 1)
                    _st = scheduler.stage_at(opt_steps + 1)
                    if _st != cur_stage_logged:   # `N-7` ③：阶段切点留证全量路径的正负比
                        cur_stage_logged = _st
                        try:
                            _np, _nn, _rt, _tg, _ok = scheduler.ratio_report(
                                train_rows, opt_steps + 1)
                            logger("CURRICULUM_RATIO stage=%d step=%d pos=%d neg=%d ratio=%.4f "
                                   "target=%.4f ok=%s"
                                   % (_st, opt_steps + 1, _np, _nn, _rt, _tg, _ok))
                        except Exception as exc:  # noqa: BLE001
                            logger("CURRICULUM_EVIDENCE_SKIP: %s" % ascii(exc))
                    if allowed:
                        keep = [i for i in idxs if i in set(allowed)]
                        # W5.2：整个 micro-batch 都被当前阶段排除 ⇒ **跳过该 step**。
                        # 原实现回退成「本 step 用全量」，会让难样本在 stage 1 渗入、课程失效。
                        if not keep:
                            n_curriculum_skipped += 1
                            continue
                        use = keep
                except Exception as exc:  # noqa: BLE001
                    logger("CURRICULUM_SELECT_FAIL step=%d（%s）—— 本 step 用全量"
                           % (opt_steps + 1, ascii(exc)))
            input_ids, attn, gather, targets = _collate(train_exs, use, pad_id, torch)
            input_ids, attn = input_ids.to(device), attn.to(device)
            gather, targets = gather.to(device), targets.to(device)
            loss = forward_target_loss(llama, lm_head, input_ids, attn, gather, targets,
                                       scale=1.0 / accum)
            loss.backward()
            _lv = float(loss.detach())
            epoch_loss += _lv * accum
            n_micro += 1
            global_micro += 1
            pending += 1
            # `D-54` 裁定三②③：**每个 micro-batch** 查标量 loss；触发即**四不一停**
            # （不落盘 state／不覆盖 best／不删任何文件／不自动恢复）⇒ 退出码 3。
            # **未触发时零副作用**（无日志行、不消耗 RNG、不改任何量）。
            if nan_guard and nan_guard_violated(_lv):
                logger("NAN_GUARD_TRIGGERED step=%d epoch=%d loss=%s"
                       % (opt_steps, epoch, repr(_lv)))
                logger("NAN_GUARD_STOP 不落盘 state／不覆盖 best／不删任何文件／不自动恢复"
                       "（退出码 3；停工呈 W0）")
                return {"best": best, "best_state": best_state, "curve": curve_rows,
                        "wall_seconds": round(time.time() - t_start, 3),
                        "peak_memory_mib": 0.0, "opt_steps": opt_steps,
                        "global_micro": global_micro, "epochs_run": len(curve_rows),
                        "early_stopped": False, "stopper_history": stopper.history,
                        "lr": lr, "r": lora_r, "nan_guard_stopped": True,
                        "segments": _segments_now(), "ts": ts}
            if pending >= accum:
                if clip and clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                opt_steps += 1
                # `D-48` 裁定三：**只在 `opt.step()` 之后、`pending == 0`** 落盘（不存梯度）
                if resume_dir and state_every > 0 and opt_steps % state_every == 0:
                    save_resume_state(resume_dir, _snapshot(epoch, consumed, order),
                                      torch, logger)
                # `D-48` 裁定四 证明钩子（默认 None ⇒ 无任何影响）：落盘后立即返回，
                # **不走** epoch 末评估 ⇒ 与「一次跑完」的曲线逐行可比。
                if stop_after and opt_steps >= stop_after:
                    if resume_dir:
                        save_resume_state(resume_dir, _snapshot(epoch, consumed, order),
                                          torch, logger)
                    logger("RESUME_STOPPED_FOR_PROOF step=%d（本段不产出最终产物，请以 "
                           "`--resume` 续跑）" % opt_steps)
                    return {"best": best, "best_state": best_state, "curve": curve_rows,
                            "wall_seconds": round(time.time() - t_start, 3),
                            "peak_memory_mib": 0.0, "opt_steps": opt_steps,
                            "global_micro": global_micro, "epochs_run": len(curve_rows),
                            "early_stopped": False, "stopper_history": stopper.history,
                            "lr": lr, "r": lora_r, "resume_stopped": True,
                            "segments": _segments_now(), "ts": ts}
                if args.max_steps and opt_steps >= int(args.max_steps):
                    break
        if pending:
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_steps += 1

        # —— 该轮 val 评估（C5：监控量 = 该体制 val 的 F1）——
        scores = evaluate(model, llama, lm_head, torch, val_exs, id_unsafe, id_safe, cfg, rng)
        labels = [e.label for e in val_exs]
        _tres = resolve_tau_ours(scores, labels, cfg)      # `D-47` 裁定二（触发式）
        tau = _tres["tau_main"]
        log_tau_ours(_tres, logger, "epoch%d" % epoch)
        pred = apply_threshold(scores, tau)
        m = compute_binary_metrics(labels, pred, scores)
        improved, counter, _b = stopper.update(m["f1"])
        logger("EPOCH %d loss=%.6f opt_steps=%d val_f1=%.6f val_tpr=%s val_fpr=%s tau=%s "
               "improved=%s counter=%d/%d"
               % (epoch, epoch_loss / max(1, n_micro), opt_steps, m["f1"], _fmt(m["tpr"]),
                  _fmt(m["fpr"]), _fmt(tau), improved, counter, stopper.patience))
        curve_rows.append({"epoch": epoch, "opt_steps": opt_steps,
                           "train_loss": round(epoch_loss / max(1, n_micro), 8),
                           "val_f1": round(m["f1"], 8), "val_tpr": _fmt(m["tpr"]),
                           "val_fpr": _fmt(m["fpr"]), "tau_ours": _exact(tau),
                           "early_stop_counter": counter})
        if best["val_f1"] is None or m["f1"] > best["val_f1"]:
            best.update({"step": opt_steps, "val_f1": m["f1"], "epoch": epoch, "tau": tau,
                         "tpr": m["tpr"], "fpr": m["fpr"]})
            best_state = {n: p.detach().to("cpu").clone()
                          for n, p in model.named_parameters() if p.requires_grad}
            if bool(_require(es_cfg, "save_best", "early_stopping")):
                meta = {"regime": args.regime, "seed": seed, "step": opt_steps,
                        "epoch": epoch, "val_f1": m["f1"], "tau_ours": tau, "lr": lr,
                        "r": lora_r, "alpha": lora_alpha, "opt_steps": opt_steps,
                        # `D-47` 裁定二：τ* **并列**记录（全精度），供下游复算
                        "tau_star": _tres["tau_star"],
                        "tau_main_override": _tres["triggered"],
                        "tau_main_override_reasons": _tres["boundary_reasons"],
                        "tau_neg_max_gap": _tres["gap"],
                        # `D-48` 裁定六：分段披露（n_segments ＋ 每段 start/end/wall）
                        "segments": _segments_now()}
                best["dir"] = save_checkpoint(model, tok, cfg, args.out_dir, args.regime,
                                              seed, opt_steps, ts, meta, logger)
        # `D-48` 裁定三：epoch 末**另行**落盘一份（含该轮 `evaluate` 之后的状态）；
        # 下一 epoch 由 `rng.shuffle` 重新打乱 ⇒ `order=None`／`cursor=0`。
        if resume_dir:
            save_resume_state(resume_dir, _snapshot(epoch + 1, 0, None), torch, logger)
        if stopper.should_stop():
            stopped_early = True
            logger("EARLY_STOP epoch=%d counter=%d >= patience=%d"
                   % (epoch, counter, stopper.patience))
            break
        if args.max_steps and opt_steps >= int(args.max_steps):
            logger("MAX_STEPS_REACHED %d" % opt_steps)
            break

    wall = time.time() - t_start
    if scheduler is not None:
        logger("CURRICULUM_SKIPPED_MICRO=%d（整批被当前阶段排除而跳过的 micro-batch；"
               "不回退成全量，否则难样本会渗入早期阶段）" % n_curriculum_skipped)
    peak = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    if best["dir"] is None:
        best["dir"] = save_checkpoint(model, tok, cfg, args.out_dir, args.regime, seed,
                                      max(1, opt_steps), ts,
                                      {"regime": args.regime, "seed": seed, "step": opt_steps,
                                       "val_f1": best["val_f1"], "tau_ours": best["tau"],
                                       "lr": lr, "r": lora_r}, logger)
    return {"best": best, "best_state": best_state, "curve": curve_rows,
            "wall_seconds": round(wall, 3), "peak_memory_mib": round(peak, 1),
            "opt_steps": opt_steps, "global_micro": global_micro,
            "epochs_run": len(curve_rows), "early_stopped": stopped_early,
            "stopper_history": stopper.history, "lr": lr, "r": lora_r,
            # `D-50` 裁定二①：回传**生效 ts**（续跑时 = state 的 ts）⇒ 供 `run_train`
            # 的 `write_curve`／`write_threshold_curve`／`write_best_pointer` 沿用。
            "ts": ts,
            # `D-48` 裁定六：分段披露（`meta.json`／`best.json` 同期增记）
            "segments": _segments_now()}


# ---------------------------------------------------------------------------
# 产物落盘（C11：`--report-dir` 可覆盖，禁止写死）
# ---------------------------------------------------------------------------
def artifact_path(report_dir, cfg, key, **kw):
    return os.path.join(report_dir,
                        _require(cfg["paths"]["artifacts"], key,
                                 "paths.artifacts").format(**kw))


def write_threshold_curve(report_dir, cfg, regime, ts, scores, labels, tau, best_f1):
    """C4：τ 寻优全曲线落 `threshold_search_ours_<regime>_<ts>.csv`。"""
    path = artifact_path(report_dir, cfg, "threshold", regime=regime, ts=ts)
    cands = [NEG_INF] + sorted(set(scores)) + [POS_INF]
    rows = []
    for tau_c in cands:
        pred = apply_threshold(scores, tau_c)
        m = compute_binary_metrics(labels, pred)
        rows.append({"tau": _fmt(tau_c), "f1": round(m["f1"], 8),
                     "tpr": _fmt(m["tpr"]), "fpr": _fmt(m["fpr"]),
                     "selected": "yes" if float(tau_c) == float(tau) else ""})
    write_csv(path, rows, ["tau", "f1", "tpr", "fpr", "selected"])
    return path


def write_curve(report_dir, cfg, regime, seed, ts, rows):
    path = artifact_path(report_dir, cfg, "curve", regime=regime, seed=seed, ts=ts)
    write_csv(path, rows, ["epoch", "opt_steps", "train_loss", "val_f1", "val_tpr",
                           "val_fpr", "tau_ours", "early_stop_counter"])
    return path


# ---------------------------------------------------------------------------
# 模式：train（默认）
# ---------------------------------------------------------------------------
def run_train(cfg, args, logger, ts):
    train_csv, val_csv = resolve_split_paths(cfg, args.regime)
    assert_val_not_in_train(train_csv, val_csv, cfg)
    assert_ood_train_encodings(cfg, train_csv, logger)

    # --- N-5 根因修复（`D-42` §4）：**必须在 `build_model` 之前**播种子 ---
    # `build_model()` 内的 `get_peft_model()` 会用 torch 全局 RNG **随机初始化 LoRA A**
    # （`lora_B` 初始化为 0，故其指纹恒同；`lora_A` 每次进程不同 ⇒ 梯度／loss 遂不同）。
    # 原实现只在 `train_one_run`（L804）内播种子，时点晚于 LoRA 初始化 ⇒ 初始化不可复现。
    # 实测（探针 `logs/train/probe_n5_isolate.py init`）：同一 `seed=42` 两进程
    # `lora_A[layer0.q_proj]` 指纹 `3baf0843…` vs `9fd817f3…`。故把播种提到此处。
    set_determinism(int(args.seed), cfg)

    torch, tok, model, llama, lm_head, meta = build_model(cfg, logger)
    id_unsafe, id_safe = resolve_target_ids(tok, cfg)
    cur_cls = None
    cur_mod = None
    if str(args.curriculum).lower() == "on":
        cur_cls, note = try_load_curriculum(cfg)
        cur_mod = curriculum_module(cfg)
        logger(note)

    # `W5.6`（`D-83` 裁定四 工作 3）：**独立增强开关**，使「增强」与「课程」解耦。
    #   - 生效式 = (`--curriculum on` ⇒ 增强照旧生效) ∪ (`--augment on` ⇒ 增强亦生效)。
    #   - **C 组复现红线**：`--curriculum on` 单独使用（历史命令不带 `--augment`）时，
    #     走**原分支**（`cur_cls is not None`），行为与改造前**逐字不变**。
    #   - 默认 `augment=off` ＋ `--curriculum off` ⇒ 两分支皆不进入 ⇒ 完全等价改造前（A 组）。
    #   - 用 `getattr` 兜底：`04_00_smoke_train.py` 与既有测试以 Namespace 传参，不含该字段。
    augment_on = str(getattr(args, "augment", "off")).lower() == "on"

    # C7：课程增强（**W5.2 接线**）—— 只作用于 train；增强后 frame 同时作为 `select()` 的 frame。
    # 冒烟子集钩子：`04_00_smoke_train.py` 以**程序内**方式传入行下标（不在冻结 CLI 增参）；
    # 正常 CLI 调用时为 None ⇒ 编码**全量** train/val，不改变训练口径。
    train_rows = None
    if cur_cls is not None:
        train_rows, _rep = build_curriculum_frame(cfg, cur_mod, args.regime, train_csv,
                                                  getattr(args, "train_row_ids", None),
                                                  int(args.seed), logger)
    elif augment_on:
        # 新路径：`--augment on --curriculum off`（B 组「有增强、无课程」）。
        # `curriculum_module()` 幂等且失败即返回 None（`build_curriculum_frame` 会优雅空过）。
        if cur_mod is None:
            cur_mod = curriculum_module(cfg)
        train_rows, _rep = build_curriculum_frame(cfg, cur_mod, args.regime, train_csv,
                                                  getattr(args, "train_row_ids", None),
                                                  int(args.seed), logger)
        logger("AUGMENT_ONLY：`--augment on` 且 `--curriculum off` ⇒ 仅增强、**不做课程调度**"
               "（B 组口径，`D-83` 裁定四）")
    # `W5.7`（`P-K'`／`PREREG_v1.8.md` §1；`D-135`/`D-136`）：2×2 对角去混淆增广。
    #   - 默认 `off` ⇒ **不进入任何数据路径**（与改造前逐 step `loss` Δ=0，smoke 自证）。
    #   - `on` ⇒ 训练集**与验证集**均按对角规则替换（§3-7 val 同规）；**在课程增强之后**施加
    #     （见本文件去混淆节的时序说明）；只改 `text`、替换而非新增。
    deconfound_on = str(getattr(args, "deconfound", "off")).lower() == "on"
    val_rows = None
    deconfound_tag = None
    if deconfound_on:
        deconfound_tag = ("deconfound", args.regime)
        if train_rows is None:
            # 无课程增强帧 ⇒ 从分片取整行（需 `plaintext` 列 ⇒ `load_rows`，见 `deconfound_load_frame`）
            train_rows = deconfound_load_frame(cfg, train_csv,
                                               getattr(args, "train_row_ids", None))
        train_rows, _d_stats = deconfound_apply(train_rows, args.regime, enabled=True)
        deconfound_log(_d_stats, logger, "train")
        val_rows = deconfound_load_frame(cfg, val_csv, getattr(args, "val_row_ids", None))
        val_rows, _d_stats_v = deconfound_apply(val_rows, args.regime, enabled=True)
        deconfound_log(_d_stats_v, logger, "val")
    else:
        logger("DECONFOUND=off")

    train_exs = prepare_split(tok, cfg, train_csv, id_unsafe, id_safe, tok.eos_token_id,
                              logger, ids=getattr(args, "train_row_ids", None),
                              rows=train_rows, cache_key_extra=deconfound_tag)
    val_exs = prepare_split(tok, cfg, val_csv, id_unsafe, id_safe, tok.eos_token_id,
                            logger, ids=getattr(args, "val_row_ids", None),
                            rows=val_rows, cache_key_extra=deconfound_tag)

    res = train_one_run(cfg, args, tok, model, llama, lm_head, torch, train_exs, val_exs,
                        id_unsafe, id_safe, logger, ts,
                        curriculum_on=(cur_cls is not None), curriculum_cls=cur_cls,
                        train_rows=train_rows)

    # `D-54` 裁定三③：守卫触发 ⇒ **四不一停**。此处**立即返回**，不得继续做
    # epoch 末评估／写 curve／threshold／best 指针（否则破坏「不落盘、不覆盖 best」）。
    if res.get("nan_guard_stopped"):
        logger("NAN_GUARD_PROPAGATE step=%s（本段不产出最终产物；退出码 3；停工呈 W0）"
               % res.get("opt_steps"))
        return None, res

    # `D-48` 裁定四 证明钩子：本段在指定 step 落盘后即止 ⇒ **不产出最终产物**，
    # 由执行者以 `--resume` 起下一段（`04_00` 亦据此跳过冒烟验收）。
    if res.get("resume_stopped"):
        logger("RESUME_SEGMENT_DONE step=%d（本段不产出最终产物；续跑见 `--resume`）"
               % res["opt_steps"])
        return None, res

    # `D-50` 裁定二①（修 `N-8`）：**续跑段沿用首段 `ts`** ⇒ 以下 `write_threshold_curve`／
    # `write_curve`／`write_best_pointer`（以及段内的 `save_checkpoint`）全部落在**同一
    # `run_id` 目录**，不再产生第二套目录。未传 `--resume` 时 `res["ts"] == ts`（恒等）。
    ts = res.get("ts", ts)

    # 末了做一次 τ 寻优：先把**验证集最优**权重回填进内存（C5：best 只由 val F1 决定）
    if res["best_state"]:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in res["best_state"]:
                    p.copy_(res["best_state"][n].to(p.device))
        logger("BEST_WEIGHTS_RESTORED step=%s" % res["best"]["step"])

    scores = evaluate(model, llama, lm_head, torch, val_exs, id_unsafe, id_safe, cfg,
                      set_determinism(int(args.seed), cfg))
    labels = [e.label for e in val_exs]
    # `D-47` 裁定二：τ 主口径**触发式**改用 `0.5`（未触发时与原口径完全一致）
    _tres = resolve_tau_ours(scores, labels, cfg)
    tau = _tres["tau_main"]
    log_tau_ours(_tres, logger, "final")
    f1 = compute_binary_metrics(labels, apply_threshold(scores, tau))["f1"]
    thr_csv = write_threshold_curve(args.report_dir, cfg, args.regime, ts, scores, labels,
                                    tau, f1)
    curve_csv = write_curve(args.report_dir, cfg, args.regime, args.seed, ts, res["curve"])

    best = dict(res["best"])
    best.update({"tau_ours": tau, "val_f1_at_tau": f1, "curve_csv": curve_csv,
                 "tau_star": _tres["tau_star"],
                 "tau_main_override": _tres["triggered"],
                 "tau_main_override_reasons": _tres["boundary_reasons"],
                 "tau_neg_max_gap": _tres["gap"],
                 "segments": res.get("segments"),          # `D-48` 裁定六（`best.json` 同期增记）
                 "threshold_csv": thr_csv, "wall_seconds": res["wall_seconds"],
                 "peak_memory_mib": res["peak_memory_mib"],
                 "early_stopped": res["early_stopped"], "regime": args.regime,
                 "seed": int(args.seed), "model_meta": meta,
                 "epochs_run": res["epochs_run"], "opt_steps": res["opt_steps"]})
    write_best_pointer(args.out_dir, cfg, args.regime, int(args.seed), ts, best, logger)
    logger("TAU_OURS regime=%s seed=%s tau=%s val_f1=%.6f"
           % (args.regime, args.seed, _exact(tau), f1))
    logger("CURVE_CSV %s" % curve_csv)
    logger("THRESHOLD_CSV %s" % thr_csv)
    res["best"] = best                       # 回传**完整** best（含 τ／曲线路径），供 `04_00` 验收
    return os.path.abspath(best["dir"]), res


# ---------------------------------------------------------------------------
# 模式：threshold
# ---------------------------------------------------------------------------
def run_threshold(cfg, args, logger, ts):
    if not args.checkpoint_dir:
        raise ValueError("--mode threshold 必须给 --checkpoint-dir")
    _train_csv, val_csv = resolve_split_paths(cfg, args.regime)
    torch, tok, model, _llama, _lm_head, _meta = build_model(cfg, logger, with_lora=False)
    from peft import PeftModel
    ckpt = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(ckpt):
        raise FileNotFoundError("checkpoint 目录不存在：%s" % ckpt)
    model = PeftModel.from_pretrained(model, ckpt, is_trainable=False)
    llama, lm_head = _unwrap(model)
    id_unsafe, id_safe = resolve_target_ids(tok, cfg)
    # `W5.7`：`--mode threshold` 的 τ* 同样在 val 上寻优 ⇒ `--deconfound on` 时 val 须同规
    # （否则边界解复发；`PREREG_v1.8` §1「val 同规」）。评估期（`05_01` 等）一律 `off`。
    deconfound_on = str(getattr(args, "deconfound", "off")).lower() == "on"
    val_rows = None
    deconfound_tag = None
    if deconfound_on:
        deconfound_tag = ("deconfound", args.regime)
        val_rows = deconfound_load_frame(cfg, val_csv, getattr(args, "val_row_ids", None))
        val_rows, _d_stats_v = deconfound_apply(val_rows, args.regime, enabled=True)
        deconfound_log(_d_stats_v, logger, "val")
    val_exs = prepare_split(tok, cfg, val_csv, id_unsafe, id_safe, tok.eos_token_id, logger,
                            rows=val_rows, cache_key_extra=deconfound_tag)
    scores = evaluate(model, llama, lm_head, torch, val_exs, id_unsafe, id_safe, cfg,
                      set_determinism(int(args.seed), cfg))
    labels = [e.label for e in val_exs]
    # `D-47` 裁定二：τ 主口径**触发式**改用 `0.5`（未触发时与原口径完全一致）
    _tres = resolve_tau_ours(scores, labels, cfg)
    tau = _tres["tau_main"]
    log_tau_ours(_tres, logger, "threshold")
    f1 = compute_binary_metrics(labels, apply_threshold(scores, tau))["f1"]
    path = write_threshold_curve(args.report_dir, cfg, args.regime, ts, scores, labels, tau, f1)
    logger("TAU_OURS regime=%s ckpt=%s tau=%s val_f1=%.6f tau_star=%s triggered=%s"
           % (args.regime, ckpt, _exact(tau), f1, _exact(_tres["tau_star"]),
              _tres["triggered"]))
    return path


# ---------------------------------------------------------------------------
# 模式：grid（**只允许 --regime id**；9 组事前定值）
# ---------------------------------------------------------------------------
def grid_cells(cfg):
    """展开 `grid` 事前定值空间 ⇒ ``[(lr, r), …]``，并断言组数 == `n_cells`。

    抽成函数是为了让 **单元测试无需 GPU** 即可校验「9 组展开」这一冻结口径。
    """
    g = _require(cfg, "grid", "cfg")
    lrs = list(_require(g, "lr", "grid"))
    rs = list(_require(g, "r", "grid"))
    cells = [(float(lr), int(r)) for lr in lrs for r in rs]
    if len(cells) != int(_require(g, "n_cells", "grid")):
        raise ValueError("网格组数 %d != 配置 n_cells=%d"
                         % (len(cells), int(_require(g, "n_cells", "grid"))))
    return cells


def sequential_cells(cfg, stage):
    """展开 `grid.sequential` 的省时路径（`D-42` 裁定六；事前定值）。

    - ``stage="1"``：只取 `stage1_r` × `stage1_lrs`（默认 r=16 × 3 个 lr ≈7.8 h）。
    - ``stage="2"``：仅补跑 `stage2_r` × 全部 `lr`（默认 r∈{8,32} × 3 lr）。
    - ``stage="all"``：stage1 ∪ stage2（回到默认 9 组；等式校验 == `n_cells`）。

    抽成函数以便**单元测试无需 GPU** 校验「stage1 恰 3 组且不含 r∈{8,32}」等冻结口径。
    """
    g = _require(cfg, "grid", "cfg")
    seq = _require(g, "sequential", "grid")
    lrs = [float(x) for x in _require(g, "lr", "grid")]
    r1 = int(_require(seq, "stage1_r", "grid.sequential"))
    lrs1 = [float(x) for x in _require(seq, "stage1_lrs", "grid.sequential")]
    rs2 = [int(x) for x in _require(seq, "stage2_r", "grid.sequential")]
    stage1 = [(lr, r1) for lr in lrs1]
    if stage == "1":
        return stage1
    stage2 = [(lr, r) for lr in lrs for r in rs2]
    if stage == "2":
        return stage2
    if stage == "all":
        cells = stage1 + stage2
        if len(cells) != int(_require(g, "n_cells", "grid")):
            raise ValueError("sequential all 组数 %d != 配置 n_cells=%d"
                             % (len(cells), int(_require(g, "n_cells", "grid"))))
        return cells
    raise ValueError("sequential_cells: 未知 stage=%r（期望 '1'/'2'/'all'）" % stage)


def evaluate_stopping_rule(cfg, stage1_rows):
    """按 `grid.sequential.stopping_rule` 判定 stage1 是否可停（**事前定值，`D-42` 裁定六**）。

    四条件（`logic=ALL`：**全部**满足才可停；任一不满足 ⇒ 续跑）：

    - **A 极差收敛**：``max(R) - min(R) < range_threshold``（严格小于；等于不算收敛）
    - **B 最优非边界**：``best_lr ∉ boundary_lrs``（须为内部值）
    - **C 最优非下限**：``best.val_f1_best >= min_absolute_f1``
    - **D 无平局越界**：``tie_rule == "unique_or_non_boundary"`` 时，best 唯一或平局集合不跨边界

    返回 dict：``{"stop": bool, "best": row|None, "R": [...], "conditions": {...}}``。
    判定方向与阈值**全部读自 yaml**，不在代码内改口径。
    """
    seq = _require(_require(cfg, "grid", "cfg"), "sequential", "grid")
    rule = _require(seq, "stopping_rule", "grid.sequential")
    if not stage1_rows:
        return {"stop": False, "best": None, "R": [],
                "conditions": {"A_range": False, "B_interior": False,
                               "C_floor": False, "D_tie": False},
                "reason": "stage1 无结果"}
    f1s = [float(r["val_f1_best"]) for r in stage1_rows]
    rng = max(f1s) - min(f1s)
    best = _select_grid(stage1_rows)          # 复用事前定值的 selection（objective＋tie_break）
    best_lr = float(best["lr"])
    boundary = [float(x) for x in _require(rule, "boundary_lrs", "grid.sequential.stopping_rule")]
    thr = float(_require(rule, "range_threshold", "grid.sequential.stopping_rule"))
    floor = float(_require(rule, "min_absolute_f1", "grid.sequential.stopping_rule"))
    tie_rule = _require(rule, "tie_rule", "grid.sequential.stopping_rule")

    cond_a = rng < thr
    # B：最优 lr 须非边界（best_lr_must_be_interior 为 true 时生效；为 false 则此条恒真）
    must_interior = bool(rule.get("best_lr_must_be_interior", True))
    cond_b = (best_lr not in boundary) if must_interior else True
    cond_c = float(best["val_f1_best"]) >= floor
    # D：平局集合（同一 max F1 的所有行）是否跨边界
    top = [r for r in stage1_rows if float(r["val_f1_best"]) == max(f1s)]
    top_lrs = [float(r["lr"]) for r in top]
    tie_spans_boundary = any(lr in boundary for lr in top_lrs) and any(
        lr not in boundary for lr in top_lrs)
    if tie_rule == "unique_or_non_boundary":
        cond_d = (len(top) == 1) or (not tie_spans_boundary)
    else:
        cond_d = True

    conditions = {"A_range": cond_a, "B_interior": cond_b, "C_floor": cond_c, "D_tie": cond_d}
    stop = bool(all(conditions.values()))     # logic=ALL
    return {"stop": stop, "best": best, "R": f1s, "range": rng,
            "conditions": conditions, "tie_rule": tie_rule,
            "must_interior": must_interior, "boundary_lrs": boundary,
            "range_threshold": thr, "min_absolute_f1": floor,
            "reason": "全部满足" if stop else "存在未满足条件：" +
                      ",".join(k for k, v in conditions.items() if not v)}


def run_grid(cfg, args, logger, ts):
    if args.regime != "id":
        raise ValueError("--mode grid 只允许 --regime id（`PREREG_v1.md` §9「仅在验证集上做一次」；"
                         "OOD 侧另起一次搜索属偏离，须先写台账并呈 W0）")
    train_csv, val_csv = resolve_split_paths(cfg, args.regime)
    assert_val_not_in_train(train_csv, val_csv, cfg)
    assert_ood_train_encodings(cfg, train_csv, logger)

    grid_path = getattr(args, "grid_path", "sequential")
    _GRID_CTX.clear()
    _GRID_CTX["grid_path"] = grid_path
    if grid_path == "full":
        cells = grid_cells(cfg)
        logger("GRID_PATH full（默认 9 组；不使用省时路径）cells=%d" % len(cells))
        rows = _run_grid_cells(cfg, args, logger, ts, cells, train_csv, val_csv)
    elif grid_path == "sequential":
        # `D-42` 裁定六：先跑 stage1（r=16 × 3 lr），按**事前定值**停止规则判定；
        # 满足 ⇒ 停并固定 (best_lr, 16)；不满足 ⇒ 补跑 stage2（r∈{8,32}×3 lr，回到 9 组）。
        rows = _run_grid_sequential(cfg, args, logger, ts, train_csv, val_csv)
    else:
        raise ValueError("未知 --grid-path=%r（期望 'sequential'/'full'）" % grid_path)

    logger("GRID_TOTAL_CELLS %d" % len(rows))
    csv_path = artifact_path(args.report_dir, cfg, "grid_csv", ts=ts)
    write_csv(csv_path, rows, ["lr", "r", "val_f1_best", "val_f1_at_best_epoch", "wall_seconds",
                               "peak_memory_mib", "tau_ours", "n_epochs_run", "early_stopped",
                               "checkpoint_dir"])
    sel = _select_grid(rows)
    md_path = artifact_path(args.report_dir, cfg, "grid_md", ts=ts)
    _write_grid_md(md_path, rows, sel, ts)
    if sel is not None and args.backfill:
        _backfill_grid_selected(cfg, args.config, sel, args.regime, md_path, logger)
    logger("GRID_RESULT_CSV %s" % csv_path)
    logger("GRID_SELECTED lr=%s r=%s" % (sel and sel["lr"], sel and sel["r"]))
    return csv_path


def _run_grid_cells(cfg, args, logger, ts, cells, train_csv, val_csv):
    """跑给定 (lr, r) 组并返回 rows（公共执行体；stage1／stage2／full 共用）。"""
    # `W5.7`：`--deconfound on` 时训练期与验证期数据构造同规（`PREREG_v1.8` §1「适用」行）。
    # 帧在各 cell 间**共用同一份**去混淆结果（确定性 ⇒ 各 cell 输入一致，网格可比）。
    deconfound_on = str(getattr(args, "deconfound", "off")).lower() == "on"
    train_rows = val_rows = None
    deconfound_tag = None
    if deconfound_on:
        deconfound_tag = ("deconfound", args.regime)
        train_rows = deconfound_load_frame(cfg, train_csv,
                                           getattr(args, "train_row_ids", None))
        train_rows, _d_stats = deconfound_apply(train_rows, args.regime, enabled=True)
        deconfound_log(_d_stats, logger, "train")
        val_rows = deconfound_load_frame(cfg, val_csv, getattr(args, "val_row_ids", None))
        val_rows, _d_stats_v = deconfound_apply(val_rows, args.regime, enabled=True)
        deconfound_log(_d_stats_v, logger, "val")
    rows = []
    for lr, r in cells:
        logger("GRID_CELL lr=%g r=%d" % (lr, r))
        torch, tok, model, llama, lm_head, _meta = build_model(cfg, logger, lora_r=r)
        id_unsafe, id_safe = resolve_target_ids(tok, cfg)
        train_exs = prepare_split(tok, cfg, train_csv, id_unsafe, id_safe, tok.eos_token_id,
                                  rows=train_rows, cache_key_extra=deconfound_tag)
        val_exs = prepare_split(tok, cfg, val_csv, id_unsafe, id_safe, tok.eos_token_id,
                                rows=val_rows, cache_key_extra=deconfound_tag)
        res = train_one_run(cfg, args, tok, model, llama, lm_head, torch, train_exs, val_exs,
                            id_unsafe, id_safe, logger, ts, lora_r=r, lr=lr)
        rows.append({"lr": lr, "r": r,
                     "val_f1_best": round(res["best"]["val_f1"] or 0.0, 8),
                     "val_f1_at_best_epoch": round(res["best"]["val_f1"] or 0.0, 8),
                     "wall_seconds": res["wall_seconds"],
                     "peak_memory_mib": res["peak_memory_mib"],
                     "tau_ours": _fmt(res["best"]["tau"]),
                     "n_epochs_run": res["epochs_run"],
                     "early_stopped": res["early_stopped"],
                     "checkpoint_dir": res["best"]["dir"]})
        logger("GRID_CELL_DONE lr=%g r=%d val_f1=%.6f wall=%.1fs peak=%.1fMiB"
               % (lr, r, res["best"]["val_f1"] or 0.0, res["wall_seconds"],
                  res["peak_memory_mib"]))
        del model, llama, lm_head
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger("WARN 显存回收失败（不静默）：%s" % ascii(exc))
    return rows


def _run_grid_sequential(cfg, args, logger, ts, train_csv, val_csv):
    """序贯省时路径（`D-42` 裁定六）：stage1 → 停止规则判定 → 条件性 stage2。

    判定所用 stage1 的 `R` 与四条件实测值一并写日志（留证）；停止时只回填 stage1 选出者。
    """
    seq = _require(_require(cfg, "grid", "cfg"), "sequential", "grid")
    if not bool(_require(seq, "enabled", "grid.sequential")):
        logger("GRID_SEQUENTIAL_DISABLED 回退 full 路径")
        return _run_grid_cells(cfg, args, logger, ts, grid_cells(cfg), train_csv, val_csv)

    stage1 = sequential_cells(cfg, "1")
    logger("GRID_SEQUENTIAL stage1 cells=%d（r=%d × lr∈%s）"
           % (len(stage1), int(seq["stage1_r"]), [float(x) for x in seq["stage1_lrs"]]))
    rows = _run_grid_cells(cfg, args, logger, ts, stage1, train_csv, val_csv)

    verdict = evaluate_stopping_rule(cfg, rows)
    cond = verdict["conditions"]
    logger("STOPPING_RULE R=%s range=%.8f A=%s B=%s C=%s D=%s -> %s"
           % ([round(x, 8) for x in verdict["R"]], verdict.get("range", float("nan")),
              cond["A_range"], cond["B_interior"], cond["C_floor"], cond["D_tie"],
              "STOP" if verdict["stop"] else "CONTINUE"))
    logger("STOPPING_RULE_REASON %s" % verdict["reason"])
    _GRID_CTX["verdict"] = ("%s（R=%s；range=%.6f；A/B/C/D=%s/%s/%s/%s）"
                            % ("STOP" if verdict["stop"] else "CONTINUE",
                               [round(x, 8) for x in verdict["R"]], verdict.get("range", float("nan")),
                               cond["A_range"], cond["B_interior"], cond["C_floor"], cond["D_tie"]))

    if verdict["stop"]:
        best = verdict["best"]
        logger("GRID_SEQUENTIAL_STOP 固定 (lr=%g, r=%d)（stage2 不跑）"
               % (float(best["lr"]), int(best["r"])))
        # 停止路径仅回填 stage1 选出者（下游 _select_grid 亦只见到 stage1 rows）
        return rows

    stage2 = sequential_cells(cfg, "2")
    logger("GRID_SEQUENTIAL_CONTINUE 补跑 stage2 cells=%d（回到 %d 组）"
           % (len(stage2), len(stage1) + len(stage2)))
    rows2 = _run_grid_cells(cfg, args, logger, ts, stage2, train_csv, val_csv)
    return rows + rows2


def _select_grid(rows):
    """`grid.selection`：F1 最大；并列先取 lr 较小者，再取 r 较小者（事前定值的固定规则）。"""
    if not rows:
        return None
    best = None
    for row in rows:
        key = (-float(row["val_f1_best"]), float(row["lr"]), int(row["r"]))
        if best is None or key < best[0]:
            best = (key, row)
    return best[1] if best else None


def _write_grid_md(path, rows, sel, ts):
    lines = ["# qlora_grid_v1.0_%s — W5.1 QLoRA 网格搜索（9 组 × val F1／耗时／显存峰值）" % ts,
             "",
             "- 体制：**`regime_id`**（`PREREG_v1.md` §9「仅在验证集上做一次」）",
             "- 网格空间（**事前定值**）：lr ∈ {1e-4, 2e-4, 4e-4} × r ∈ {8, 16, 32} = **9 组**",
             "- 监控量 = 该体制 **val** 的 F1；早停 patience = 2；保存验证集最优 checkpoint",
             "- 选定规则（**事前定值**）：F1 最大；并列先取 lr 较小者，再取 r 较小者",
             "- 执行路径：**`%s`**（`D-42` 裁定六）%s" % (
                 _GRID_CTX.get("grid_path", "full"),
                 "；停止规则判定：" + _GRID_CTX.get("verdict", "（未走序贯）")),
             "",
             "| lr | r | val F1（最优） | 耗时(s) | 显存峰值(MiB) | τ_Ours | 实跑 epoch | 早停 |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| %g | %d | %.6f | %.1f | %.1f | %s | %s | %s |"
                     % (row["lr"], row["r"], row["val_f1_best"], row["wall_seconds"],
                        row["peak_memory_mib"], row["tau_ours"], row["n_epochs_run"],
                        row["early_stopped"]))
    lines.append("")
    lines.append("**选定**：%s" % (("lr = `%g`、r = `%d`" % (sel["lr"], sel["r"]))
                                    if sel else "无（网格未跑完）"))
    lines.append("")
    lines.append("> 生成脚本：`scripts/train/04_01_train_qlora_ours.py --mode grid`；"
                 "同源 CSV：`qlora_grid_v1.0_%s.csv`" % ts)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _backfill_grid_selected(cfg, config_path, sel, regime, md_path, logger):
    """把选定 (lr, r) 回填 `configs/train/qlora_ours.yaml` 的 `grid_selected` 节（只改该节）。"""
    path = os.path.join(_ROOT, config_path) if not os.path.isabs(config_path) else config_path
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    new_block = ["grid_selected:", "  status: \"SELECTED\"", "  lr: %g" % sel["lr"],
                 "  r: %d" % sel["r"], "  regime: \"%s\"" % regime,
                 "  selected_at: \"%s\"" % timestamp(),
                 "  source_report: \"%s\"" % os.path.relpath(md_path, _ROOT).replace("\\", "/")]
    out, inside = [], False
    for line in lines:
        if line.startswith("grid_selected:"):
            inside = True
            out.extend(new_block)
            continue
        if inside:
            if line and not line.startswith((" ", "\t", "#")):
                inside = False
            else:
                continue
        out.append(line)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))
    logger("GRID_SELECTED_BACKFILLED lr=%g r=%d -> %s" % (sel["lr"], sel["r"], path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="W5.1 Ours：QLoRA 微调 Llama-Guard-3-1B（单次训练入口）")
    # `W5.7`：`--self-check` 覆盖两体制、无需 `--regime`/`--seed` ⇒ 二者改为**解析层非必填**，
    # 由 `main()` 在非 `--self-check` 路径上强制必填（训练契约**逐字不变**：缺一即报错退出）。
    p.add_argument("--regime", choices=["id", "ood"], default=None)
    p.add_argument("--seed", type=int, choices=[42, 123, 456], default=None)
    p.add_argument("--config", default="configs/train/qlora_ours.yaml")
    p.add_argument("--mode", choices=["train", "grid", "threshold"], default="train")
    p.add_argument("--curriculum", choices=["on", "off"], default="off")
    # `W5.6`（`D-83` 裁定四 工作 3）：独立增强开关。
    # **默认 `off` 时行为与改造前逐字节一致**；`--curriculum on` 的既有语义**逐字不变**
    # （增强照旧生效，C 组复现依赖）。B 组表达式 = `--augment on --curriculum off`。
    p.add_argument("--augment", choices=["on", "off"], default="off",
                   help="独立启用「编码感知增强」（`W5.6`／`D-83` 裁定四 工作 3）；"
                        "生效式 = (`--curriculum on`) ∪ (`--augment on`)；"
                        "默认 off ⇒ 与改造前逐字节一致")
    # `W5.6`（`D-83` 裁定四；W0 2026-10-09 裁）：可选 `model_key` 覆盖 ——
    # 消融 A／B 组分别取 `llamaguard1b_A`／`llamaguard1b_B`，使 `run_id` 自识别。
    # **默认 None ⇒ `cfg` 不被改动、行为与改造前逐字节一致。**
    p.add_argument("--model-key", dest="model_key", default=None,
                   help="覆盖 `checkpoint.model_key`（仅影响 `run_id` 的 {model} 槽）；"
                        "默认 None ⇒ 与改造前逐字节一致")
    # `W5.7`（`P-K'`／`PREREG_v1.8.md` §1；`D-135`/`D-136`）：2×2 对角去混淆增广。
    # **默认 `off` ⇒ 与改造前逐 step `loss` Δ=0**（smoke 自证，`N-5` 口径）。
    p.add_argument("--deconfound", choices=["on", "off"], default="off",
                   help="2×2 对角去混淆增广（`W5.7`／`PREREG_v1.8.md` §1）：攻击行 50% 转 "
                        "plaintext、良性行 50% 转 enc(text[:L])，训练期与验证期同规；"
                        "评估期一律 off；默认 off ⇒ 与改造前逐 step loss Δ=0")
    # `W5.7`：去混淆数据构造自检（§3-1～§3-7），不加载模型、不训练。
    p.add_argument("--self-check", dest="self_check", action="store_true",
                   help="只跑去混淆自检并打印 SELFCHECK_RESULT=PASS n=<k>（§3-1～§3-7）；"
                        "日志自动落 logs/train/tmp/（N-16 同规）")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    p.add_argument("--report-dir", dest="report_dir", default="reports/train")
    p.add_argument("--out-dir", dest="out_dir", default="checkpoints/ours")
    p.add_argument("--log-dir", dest="log_dir", default="logs/train")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    # `D-48` 裁定三：step 级精确续跑（**未传时行为与改造前逐字节一致**）
    p.add_argument("--resume", default=None,
                   help="从该 state 目录（含 `latest/`）恢复并继续训练（`D-48` 裁定一/三）")
    # `D-48` 裁定四：**仅供等价性自证**的停段钩子（默认 None ⇒ 无任何影响）
    p.add_argument("--stop-after-steps", dest="stop_after_steps", type=int, default=None,
                   help="到达该 opt_step 即落盘并停止本段（不产出最终产物），供分 2 段续跑自证")
    # `D-54` 裁定三①：loss-NaN 守卫**默认启用**；`--no-nan-guard` 仅供等价性自证
    p.add_argument("--no-nan-guard", dest="nan_guard", action="store_false", default=True,
                   help="关闭 loss-NaN 守卫（**仅供等价性自证**；传之 ⇒ 与改造前逐字节不变）")
    p.add_argument("--backfill", action="store_true",
                   help="--mode grid 时把选定 (lr, r) 回填配置的 grid_selected 节")
    p.add_argument("--grid-path", dest="grid_path", choices=["sequential", "full"],
                   default="sequential",
                   help="--mode grid 的执行路径：sequential=先 r=16×3 lr 再按事前定值停止规则"
                        "判定（`D-42` 裁定六）；full=直接跑默认 9 组")
    return p


def main(argv=None) -> int:
    init_streams()
    _parser = build_parser()
    args = _parser.parse_args(argv)
    # `W5.7`：`--self-check` 无需 `--regime`/`--seed`；其余模式必填契约**逐字不变**。
    if not getattr(args, "self_check", False) and (args.regime is None or args.seed is None):
        _parser.error("--regime 与 --seed 为必填（--self-check 除外）")
    cfg = load_config(args.config if os.path.isabs(args.config)
                      else os.path.join(_ROOT, args.config))
    # `W5.6`（`D-83` 裁定四；W0 2026-10-09 裁）：可选 `model_key` 覆盖。
    # **仅在显式传入时改动 `cfg`** ⇒ 默认（None）下与改造前**逐字节一致**。
    apply_model_key_override(cfg, getattr(args, "model_key", None))
    ts = timestamp()
    logger = Logger()
    # `W5.7`：日志目录解析（`N-16` 同规）—— 非 `--self-check` 且无 `CODEOBF_LOG_DIR` 时
    # **逐字回落 `args.log_dir`**（与改造前一致）。
    log_dir = resolve_log_dir(args)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger("04_01_train_qlora_ours ts=%s regime=%s seed=%s mode=%s curriculum=%s"
           % (ts, args.regime, args.seed, args.mode, args.curriculum))
    # `W5.6`（`D-83` 裁定四 工作 3）：独立增强开关**另起一行**（原文那一行逐字未改）。
    logger("04_01 augment=%s model_key=%s"
           % (str(getattr(args, "augment", "off")),
              str(cfg["checkpoint"].get("model_key"))))
    # `W5.7`：去混淆开关**另起一行**（上两行逐字未改）。
    logger("04_01 deconfound=%s self_check=%s"
           % (str(getattr(args, "deconfound", "off")),
              "on" if getattr(args, "self_check", False) else "off"))
    log_path = os.path.join(log_dir, "04_01_train_qlora_ours_%s.log" % ts)
    try:
        if getattr(args, "self_check", False):
            n_ok = deconfound_self_check(cfg, logger)
            last = "SELFCHECK_RESULT=PASS n=%d" % n_ok
        elif args.mode == "grid":
            last = "RESULT_CSV=%s" % os.path.abspath(run_grid(cfg, args, logger, ts))
        elif args.mode == "threshold":
            last = "RESULT_CSV=%s" % os.path.abspath(run_threshold(cfg, args, logger, ts))
        else:
            ckpt_dir, _res = run_train(cfg, args, logger, ts)
            # `D-54` 裁定三③：守卫触发 ⇒ **退出码 3**（与正常 `0`／错误 `1`／断言 `2` 区分）
            if _res.get("nan_guard_stopped"):
                logger("NAN_GUARD_EXIT step=%s" % _res.get("opt_steps"))
                logger.dump(log_path)
                print("NAN_GUARD_STOPPED step=%s" % _res.get("opt_steps"))
                return 3
            last = "CHECKPOINT_DIR=%s" % os.path.abspath(ckpt_dir)
    except (LeakAssertionError, AssertionError) as exc:
        logger("ASSERT_FAIL: %s" % exc)
        logger.dump(log_path)
        print("ASSERT_FAIL: %s" % exc)
        if getattr(args, "self_check", False):
            print("SELFCHECK_RESULT=FAIL reason=%s" % ascii(exc))
        return 2
    except Exception as exc:  # noqa: BLE001
        logger("ERROR: %s: %s" % (type(exc).__name__, exc))
        logger.dump(log_path)
        print("ERROR: %s: %s" % (type(exc).__name__, exc))
        return 1
    logger(last)
    logger.dump(log_path)
    print(last)
    return 0


if __name__ == "__main__":
    sys.exit(main())

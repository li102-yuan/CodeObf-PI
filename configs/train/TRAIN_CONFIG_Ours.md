# TRAIN_CONFIG_Ours.md — 阶段4「Ours」训练交接文件

> **状态**：骨架由 **W5.1** 起草；**W5.2（课程学习）／W5.3（3 seeds）／W5.4（灾难性遗忘）各追加一节，只追加不改既有行**；
> W5 收口随 tag **`train-ours-v1`** 冻结（`windows/W5_stage4_train.md` §3.1）。冻结后**只读**。

- 起草窗口：W5.1（权限 A5）
- 起草时间：2026-10-04
- 配置载体：`configs/train/qlora_ours.yaml`（**唯一**超参出处；本文件不另立一套数值）
- 训练入口：`scripts/train/04_01_train_qlora_ours.py`
- 冒烟入口：`scripts/train/04_00_smoke_train.py`（`make smoke STAGE=train`）

---

## 0. 时间戳字段（**W5.1 §6-4 要求；由首次全量训练的执行窗口填**）

| 字段 | 值 | 填写窗口 |
|---|---|---|
| 首次**冒烟**训练时间戳 | **2026-10-04 11:14:43**（`logs/train/04_00_smoke_train_20261004-111443.log`，`SMOKE_RESULT=PASS`） | W5.1（已填） |
| 首次**全量**训练时间戳 | **`PENDING`**（W0 裁决：9 组网格不开跑 ⇒ 全量训练未启动） | W5.3（填） |
| 首次**全量评估**时间戳 | **`PENDING`** | W5.4／阶段5（填） |
| `grid_selected` 回填时间 | **`2026-10-04 23:44:52`**（`source = v5_L434_default`／`grid_status = ABORTED_SATURATED`；见 §3／§7.5） | W5.1（执行 `D-43` 裁定二） |

---

## 1. 训练目标与判词口径（C3）

| 项 | 值 |
|---|---|
| 形式 | **生成式二分类**（与 B3 首行首词判词同构） |
| 模板 | Llama-Guard-3 **官方 chat 模板**；1B 侧 `content_format = list_of_text_blocks` |
| 空会话守卫 | 渲染后会话段为空 ⇒ **抛错**（W4.3 空会话教训），不静默退化 |
| 目标串 | `unsafe`（label=1）／`safe`（label=0）；**均须为单 token**（实测 id `39257`／`19193`，运行时断言） |
| score | `p_unsafe / (p_unsafe + p_safe)`（**首生成位**归一化概率） |
| 判决 | `pred = (score ≥ τ_Ours)` |
| 损失 | 只在**目标 token 位**求交叉熵（`loss_on: target_tokens_only`）；避免 `[B,T,V]` 全词表张量打爆 8 GB |

## 2. 输入面（C1／C2）

- 只读 4 个文件：`data/frozen/CodeObf-PI_v1.0/splits/regime_{id,ood}/{train,val}.csv`。
- **禁读** `data/sealed_test/**` 与 frozen 侧 `test.csv`；**禁止**把分片复制到项目外。
- 预处理**复用** `scripts/baseline/03_06_unify_preprocess.py`：`load_samples()`（原文直通零清洗）＋`truncate_left`（**512 token／左截断**，与门禁 4／B3／B5 同口径）。
- 唯一输入列 = `text`；标签 = `label`；`text_sha256` 与 B1–B5 **同源**（跨阶段一致性锚点）。

分片规模（实测）：`id` train 12672（正 6336／负 6336）、`id` val 1584；`ood` train 7430（正 3520／负 3910）、`ood` val 927。

## 3. QLoRA 超参与网格（C6）

| 超参 | 值 | 出处 |
|---|---|---|
| 量化 | 4-bit **NF4** ＋ double quant，bf16 compute | `ENV_LOCK.md` §6 |
| LoRA 目标模块 | `q_proj`／`v_proj` | v5 L434 |
| r／alpha | **待网格选定**（默认 `r=16`／`alpha=32`） | `PREREG_v1.md` §9 |
| lr | **待网格选定**（默认 `2e-4`） | 同上 |
| micro_batch／grad_accum | 4／4（**起跑值**） | C10 |
| epochs | 3（`PREREG` 区间 3–5 取下限） | 同上 |
| max_length | 512（左截断，施加于 `text` 本体） | C2／C6 |
| optimizer | AdamW（`betas` 见配置），`lr_scheduler: constant` | 事前未规定 ⇒ 取最简 |

**网格空间（事前定值）**：lr ∈ {1e-4, 2e-4, 4e-4} × r ∈ {8, 16, 32} = **9 组**；
依 `PREREG_v1.md` §9「仅在验证集上做**一次**」⇒ **只在 `regime_id` 的 val 上做这一次**，选定后固定给两体制 × 3 seeds。

| 网格选定 | 值 |
|---|---|
| `grid_selected.status` | **`SELECTED`**（`2026-10-04 23:44:52` 回填；见 §7.5） |
| `grid_selected.lr` | **`2e-4`**（＝ v5 L434 预注册默认值；**非**由搜索选出） |
| `grid_selected.r` | **`16`**（同上） |
| `grid_selected.source` | **`v5_L434_default`** |
| `grid_selected.grid_status` | **`ABORTED_SATURATED`**（网格 1/9 组后被 W0 中止；判据退化） |
| 选定规则（事前定值） | val F1 最大；并列先取 lr 较小者，再取 r 较小者 |

> **W0 裁决（2026-10-04）**：9 组网格**本窗口不开跑**，实测估时呈 W0（见 §7）。
> 未回填 `grid_selected` 前，**不得**启动 W5.3 的 3 seeds 全量训练。

## 4. 早停与阈值（C4／C5）

| 项 | 值 |
|---|---|
| 监控量 | **该体制 val 的 F1**（`03_07.f1_score` 单源） |
| patience | **2** |
| 最优保存 | 保存验证集最优 checkpoint（只由 val F1 决定） |
| τ_Ours | 在该体制 **val** 上按 **F1 最大** 寻优，**并列取 τ 较大者**（与 τ_B2 同制） |
| 禁止 | 早停计数与 best 选择**不得**引用任何测试集量 |
| τ 曲线产物 | `reports/train/threshold_search_ours_<regime>_<ts>.csv` |

## 5. 评估态与确定性（C9）

- `import torch` **之前**设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；`cudnn.deterministic=True`／`cudnn.benchmark=False`／`allow_tf32=False`。
- `attn_implementation = eager`；**禁止** flash-attn／sdpa（配置 `forbidden_attn` 运行时拦截）。
- 验证/推理一律 `model.eval()` + `torch.no_grad()`，代码**不出现 `.train(`**；eval 处做 G-EVAL 双断言。
  （实测注：`peft.get_peft_model()` 会把 `training` 置 `True`；基座 `attention_dropout=0.0` 且 LoRA `dropout=0.0`
  ⇒ 该标志对前向**无数值影响**，故由 eval 侧显式切回，不在代码里调用 train 模式。）

### 5.1 N-5 训练侧确定性（`D-42` §4 处置，**已修复并自证**）

**症状**（W0 自跑 2 次冒烟，同一代码同一 `seed=42`）：`loss` **3.438019／3.426309**、
`val_f1` **0.861538／0.883721** 两次不同（`tau` 同为 0.531209、显存同为 7270.0 MiB、`opt_steps` 同为 7）。

**根因（探针 `logs/train/probe_n5_isolate.py` 逐层隔离，**只读诊断件**）**：

| 探针 | 观测 | 结论 |
|---|---|---|
| `fwd`（进程内两次纯前向） | logits 指纹 `239c5c9c985f0067` **两次相同** | 前向（4-bit dequant ＋ eager attn）**确定性 OK** |
| `init`（跨进程比对 LoRA 初值） | `lora_A[layer0.q_proj]` `3baf0843…` vs `9fd817f3…`；`lora_B` 恒 `fa43239b…`（=0） | **根因**：LoRA **A** 初始化不可复现；**B** 恒 0 故指纹恒同 |
| `grad`（跨进程比对梯度） | 修复前 `lora_B` grad absum **302.45 vs 465.87（≈54% 偏差）**；修复后 `fffb640e…`／`364.97387695` **两次相同** | 偏差随 LoRA-A 初值而来，非浮点舍入 |
| `requant` | 基座 4-bit 权重指纹跨进程稳定 | **4-bit NF4 不是元凶**（与 W0 的「已知风险区」猜测不同） |

**机理**：`run_train` 原先在 `train_one_run`（内部 L804）才调用 `set_determinism(seed, cfg)`，
而 `build_model()`（L988）内的 `get_peft_model()` 已**先用未播种的全局 RNG 随机初始化 LoRA-A** ⇒ 时序颠倒。

**修复（两处，均为纯增键／纯增语句）**：
1. `run_train` 在 `build_model()` **之前**先调一次 `set_determinism(int(args.seed), cfg)`（根因修复）；
2. `set_determinism()` 增加**框架级**开关：按 `determinism.use_deterministic_algorithms` 调用
   `torch.use_deterministic_algorithms(True, warn_only=...)`，并把 TF32 在 matmul 与 **cuDNN** 两处一并关闭
   （原实现漏关 `torch.backends.cudnn.allow_tf32`，而该值默认 `True`）。

**判据（事前定值，写入 `qlora_ours.yaml` → `determinism.verify_double_run`）**：连跑 2 次，
`loss`／`val_f1`／`tau` 三项**位级一致**（容差 `0.0`）；若 `loss` 差值 > `1e-6` 则判「位级不可达」，
须改走「同 seed 双跑实测噪声基线」并呈 W0（**不得**宣称 3 seeds 方差有意义）。

**修复后实测（同一代码连跑 2 次，seed 42，id 体制）**：

| | 第 1 次（`152225`） | 第 2 次（`152531`） | Δ | 事前定值容差 | 判定 |
|---|---|---|---|---|---|
| `loss`（`train_loss`） | **3.42983597** | **3.42983597** | 0.0 | 0.0 | ✅ |
| `val_f1` | **0.86614173** | **0.86614173** | 0.0 | 0.0 | ✅ |
| `val_tpr`／`val_fpr` | 0.859375／0.125 | 0.859375／0.125 | 0.0 | — | ✅ |
| `tau_Ours` | **0.531209** | **0.531209** | 0.0 | 0.0 | ✅ |
| 峰值显存／`opt_steps` | 7270.0 MiB／7 | 7270.0 MiB／7 | 0 | — | ✅ |

**强证据**：两次的 curve CSV **逐字节相同**（md5 `5ed1d36f0fd35d5e9f88b97d08b755d9`），
checkpoint 的 `adapter_model.safetensors` **逐字节相同**（md5 `545b0f1c6c8a1b8e4736c30c089581cc`）。

⇒ **结论：4-bit ＋ 8 GB 下位级确定性可达**，**无须**降级为噪声基线；`PREREG_v1.md` §1 的
「3 seeds 报 mean±std」**有效**。回归防护：`tests/train/test_04_01_train_qlora_ours.py` 新增 3 用例
（`test_seed_set_before_build_model_in_run_train` 等）静态钉死上述两处修复。

## 6. 显存与组批（C10）—— **实测护栏**

8 GB 卡（8188 MiB）＋ eager 注意力下，峰值由 `n·L²`（注意力矩阵）与 `n·L`（MLP/残差激活）共同决定。
系数取自本机实测探针（`reports/train/tmp/probe_batch_budget.py`、`probe_eval_mem.py`，**非权威件**）：

| 场景 | 实测（n × L） | 峰值 MiB | 单步/单批 |
|---|---|---|---|
| 训练 | 2 × 708 | 6777 | 0.69 s |
| 训练 | 4 × 512 | 7789 | 1.47 s |
| 训练 | **4 × 708** | **11967（> 容量 ⇒ 溢出共享内存）** | **13.9 s** |
| 训练 | 8 × 512 | 14077 | 20.4 s |
| 训练 | 8 × 708 | OOM | — |
| eval | 16 × 708 | 4325 | 2.32 s（≈4880 tok/s） |
| eval | 32 × 708 | 7130 | 6.68 s |
| eval | 64 × 708 | OOM | — |

**护栏**：`batch.mem_model`（`delta ≈ quad·n·L² + linear·n·L`；训练 7000 MiB／eval 4600 MiB 目标）。
**只降单批 n**（长序列降到 2，短序列保持 4），**不改样本集／epoch／网格／lr／r**。
属 C10 回退阶梯「batch 1」的逐批应用，已在 `DEVIATION_LEDGER.md` 登记（2026-10-04 条）。

梯度检查点**无效**（实测 (4,708) 峰值 12011 vs 11967，无改善）⇒ 不开（`gradient_checkpointing: false`）。

## 7. 9 组网格的实测估时（**呈 W0**）

长度分布实测（512 左截断 ＋ 196 模板；`probe_len_dist.py`）：
`regime_id/train` med 524／mean 520／max 709；`regime_id/val` mean 527；
`regime_ood/train` mean 473；`regime_ood/val` mean 491。

组批模拟（护栏下）：`id` train **4730 微批/epoch**（平均批大小 2.68，填充令牌 6.59 M/epoch）；
`id` val 597 微批。耗时模型（未触溢出区的实测拟合 `t ≈ 2.69e-4·n·L + 0.311·n·L²/1e6`）：

| 项 | 单次 | 说明 |
|---|---|---|
| `id` 训练 1 epoch | ≈ 2966 s（0.82 h） | 4730 微批 |
| `id` val 评估 1 次 | ≈ 171 s | 1584 样本 × 527 tok @ ≈4880 tok/s |
| 单组（3 epoch ＋ 4 次评估） | ≈ 2.6 h | epoch 末评估 ×3 ＋ 末了 τ 评估 |
| **9 组合计** | **≈ 23–24 h** | 编码与模型加载已缓存/复用，约 +0.3 h |

> 该估时为**满负荷连续 GPU 占用**。W5.1 §4-3 禁止自行缩减训练集／epoch／网格空间来「让 9 组跑得完」，
> 故**呈 W0 裁决**（2026-10-04 裁决：不开跑，保留 `PENDING_W0`）。

### 7.1 序贯省时路径与事前定值停止规则（`D-42` 裁定六；**开跑前登记**）

`D-42` 裁定六在默认 9 组之外另批一条**省时路径**：先跑 **`r=16` × 3 个 `lr`**
（`lr ∈ {1e-4, 2e-4, 4e-4}`，≈**7.8 h**），若满足**事前写死**的停止规则即可停并固定 `(lr*, r=16)`；
否则补跑剩余 6 组（`r ∈ {8,32} × 3 lr`，回到默认 9 组口径）。

**反事后择优硬约束**：停止规则**必须先写入 `configs/train/qlora_ours.yaml`（`grid.sequential`）
并登记 `DEVIATION_LEDGER.md`**，之后才允许开跑；判据全部事前定值，**不得**在看到结果后调阈值／换口径；
停止后 `grid_selected` 只允许回填该 3 组中按 `selection.objective`／`tie_break` 选出者，**不得擅自扩搜**。

**停止规则（`ALL` 四条件同时满足才可停；`R` = 3 组的 `val_f1_best`）**：

| 条件 | 判据 | 事前定值 |
|---|---|---|
| **A 极差收敛** | `max(R) − min(R) < range_threshold`（严格小于） | `range_threshold = 0.0100` |
| **B 最优非边界** | `best_lr ∉ {1e-4, 4e-4}`（须为内部值 `2e-4`） | `boundary_lrs = [1e-4, 4e-4]` |
| **C 最优非下限** | `best` 的 `val_f1_best ≥ min_absolute_f1` | `min_absolute_f1 = 0.5000` |
| **D 无平局越界** | `best` 唯一，或平局集合**不跨**边界 | `tie_rule = unique_or_non_boundary` |

判定方向 `logic = ALL`：全满足 ⇒ `on_stop = fix_stage1_best`（固定 `(best_lr, r=16)`）；
任一不满足 ⇒ `on_continue = run_remaining_stage2`（补跑 `r∈{8,32}×3 lr`）。**不得部分采用**
（例：极差够小但最优落在边界 ⇒ 仍须续跑）。停止／续跑均须把「判定所用 `R` 与四条件实测值」
写入 `reports/train/qlora_grid_v1.0_<ts>.md` 与台账。**登记先于开跑**：本规则登记时
`grid_selected = PENDING_W0`，尚无任何网格数值。

### 7.2 手工执行协议（本机终端；`2026-10-04` 实测定性后的唯一可行路径）

#### 7.2.1 为什么必须手工执行（环境限制，已实测）

本 agent 沙箱**无法自主托管 ≈7.8 h 的训练**，三条候选路径全部失败（三次 `Traceback=0` ⇒ **非代码缺陷**）：

| 方案 | 实测结果 |
|---|---|
| 后台任务／`nohup`+`disown` | 子进程随发起调用／回合结束**被回收**。第二次实测：起于 `17:55:19`（`python.exe` PID 10040，内存 430 MB 已加载），日志止于 **187 B**（`GRID_CELL lr=1e-4 r=16`，权重尚未加载）；`17:57:47`／`17:59:33`／`18:01:19` 三时点复验 `size=187` 不变、`python_procs=0` ⇒ **约 2 分钟即被回收** |
| `Start-Process`（隐藏窗口） | exit **-1**，无进程／无日志／无 meta 落地；且该通道**不回传 stdout**，无法排错 |
| `schtasks.exe`（计划任务，真·会话无关） | **被本机安全策略黑名单拦截**（需到「安全中心 → 命令安全 → 程序黑名单」移除） |

⇒ **改由执行者在本机终端手工跑**（`2026-10-04` 裁决）。留存：三条失败均**干净**（无半成品 ckpt／无部分结果）。

#### 7.2.2 执行者在本机终端跑（可直接复制）

```bash
cd "C:/Users/lipei/Desktop/实验项目文件" && "C:/Users/lipei/miniconda3/envs/deepseek/python.exe" -u scripts/train/04_01_train_qlora_ours.py --regime id --seed 42 --config configs/train/qlora_ours.yaml --mode grid --grid-path sequential --curriculum on --report-dir reports/train --backfill 2>&1 | tee "logs/train/grid_manual_$(date +%Y%m%d-%H%M%S).log"
```

> ⚠ **`--curriculum on` 为 `2026-10-04` 更正所加（`N-7` 裁定①，只追加不改上行原意）**：原模板**漏 `--curriculum`** ⇒ 照抄即 `curriculum=off`
> （`qlora_ours.yaml` L348 `curriculum.default: "off"`），使 `W5.2` 的课程与良性子采样**在实跑中全部不生效**。
> 实测证据：`logs/train/grid_manual_20261004-183300.log` **L1** `… mode=grid curriculum=off`、`grep -c CURRICULUM` = **0**。
> ⇒ 本模板及其派生命令（含 `W5.3` 的 6 次全量）**必须带 `--curriculum on`**。属**执行层面更正**，非口径变更
> （课程与子采样已由 `D-42` 裁定一／三批准）。

> `-u`（无缓冲）**必须保留**：默认 stdout 被缓冲区攒着，一旦进程终止缓冲内容全部丢失，
> 这正是首跑日志止于 `Loading weights`、事后无法判断「是慢还是死」的原因之一。

#### 7.2.3 存活判据（**三者缺一不可，且须跨多时点复验** —— 源自 `F-W5.1-1`）

```bash
# 隔几分钟跑一次；① 字节数在涨 ② python 进程在 ③ 真实步进关键字已出现
cd "C:/Users/lipei/Desktop/实验项目文件" && LOG=$(ls -t logs/train/grid_manual_*.log | head -1)
stat -c 'size=%s mtime=%y' "$LOG"; tasklist //FI "IMAGENAME eq python.exe" | grep python.exe
grep -cE "EPOCH|GRID_CELL_DONE" "$LOG"
```

- **`size` 停滞 ⇒ 已死**（正常训练每 step 都刷进度）。
- **切勿**再把 `Loading weights` 进度条的 `it/s` 当成训练步进（此即本窗口 `F-W5.1-1` 的确切错法）。
- 期望：约 **7.8 h** 跑完 stage1 的 3 组；脚本会在**每组结束时**打 `GRID_CELL_DONE`，全部完成后打 `GRID_RESULT_CSV`／`GRID_SELECTED`。

#### 7.2.4 跑完后交回什么 ／ 本窗口负责什么

**交回**：`reports/train/qlora_grid_v1.0_<ts>.md` 与 `…_<ts>.csv`（脚本自动产出），以及 `grid_selected` 的回填结果。
**本窗口负责**：按 §7.1 四条件做 STOP／CONTINUE 判定（并核对该判断是否由脚本执行、非事后挑选）、登记 §7 结果、更新 `PROGRESS.md`／`WINDOW_LOG.md`／`DEVIATION_LEDGER.md` 并出**独立 commit**。
**`grid_selected` 回填前，`W5.3`（6 次全量）一律不得开跑**（`D-42` 裁定六）。

### 7.3 实际执行记录（`2026-10-04` 开跑）＋ 吞吐更正（**只追加；更正 §7.2.1 的结论**）

> **本节由 `W5.1` 追加。** §7.2.1 的「本 agent 沙箱**无法**自主托管 ≈7.8 h 训练」**对本窗口实际采用的通道不成立**。
>
> ⚠ **【同日更正】该结论已于 `2026-10-04 22:34` 收回 —— 见 §7.4**：harness 通道实测存在 **≈4 h 上限**，
> **<** stage1 最短估 ≈12 h ⇒ **§7.2.1 的原结论实质成立**（仅机制不同）。

1. **执行通道 = harness 托管的后台任务**（**非** `nohup`／**非** `Start-Process`／**非** `schtasks`），命令与 **§7.2.2 逐字相同**。
   **存活实测（多时点，t = 150 s … 165 min）**：`python.exe` **PID 24340 持续存在**；累计 CPU **单调递增**
   （长窗 ≈1.07–1.09 core；短窗 18 s 实测 **+16.5 s ⇒ 0.92 core**）；GPU 恒 **99–100%**；该 PID 在 `nvidia-smi` compute 列表。
   ⇒ §7.2.1 的三条失败是**通道各自缺陷**，**不可推广**为「本沙箱不能托管长时进程」；**上条结论已更正**（见 `DEVIATION_LEDGER.md`）。
2. **吞吐实测**：`BATCHES n_micro=4735`（§7 预估 4730 ✅）；但 **t=165 min 仍未出现 `EPOCH 1`**（§7 估 49.4 min/epoch）
   ⇒ **≈2.0 s / micro-batch** ⇒ stage1 外推 **≈23–25 h**。**根因之一**：§6／§7 的计时表成于 `N-5` 修复（`5c502d0`）**之前**，
   探针 `probe_batch_budget.py`（L10）**未**启用 `use_deterministic_algorithms` ⇒ 估时**系统性偏乐观**。
3. **环境侧实测**：`enforced.power.limit = 40.00 W` ＋ `SW Thermal Slowdown: Active`（74 °C、SM 1545/3105 MHz）；显存占用 **94%**（桌面 GPU 程序 ≈1.7 GB）。本机 `PowerLineStatus=Online`（已接 AC）⇒ **非电池所致**。
4. **存活判据的补充（因本脚本只在 epoch 边界写日志）**：`04_01.train_one_run` **不在 step 级输出**，
   故 §7.2.3 的条件 ①（日志 `size` 在涨）在**本脚本上不构成充分判据**（epoch 内**合法静默** ≈50 min 以上）。
   实测采用**三判据并用**：① **累计 CPU 时间单调递增**（新增，**最强**）② python 进程存在 ③ GPU util／该 PID 在 compute 列表；
   `EPOCH`／`GRID_CELL_DONE` 仍作**最终**进度锚。**仍禁止**把 `Loading weights` 的 `it/s` 当训练步进（`F-W5.1-1`）。

### 7.4 【**更正 §7.3**】harness 通道实测 **≈4 h 上限** ⇒ 本环境**仍无法**托管网格（`2026-10-04 22:34` 第 2 次中断）

> **本节由 `W5.1` 追加。** **更正 §7.3 第 1 条的结论。**

1. **实测**：后台任务在发车 `18:33:00` 后 **4 h 01 m**（末次写盘 `22:34:29`）**被外部回收** —— `python` 进程 **0 个**、
   日志止于 `EPOCH 2 … improved=False counter=1/2`、**无 `Traceback`**、`System`／`Application` 事件日志该时段**无条目**。
   `exit code 0` 系管道末位 **`tee` 的退出码**，**不代表** python 正常结束。⇒ 判为 **≈4 h 上限**。
2. **§7.3 第 1 条结论收回**：通道确实把存活时窗由 `nohup` 的 **2 min** 拉到 **4 h（≈120×）**（故三条失败仍属**各通道自身缺陷**），
   但 **4 h < stage1 最短估 ≈12 h** ⇒ **不能完成网格**。**§7.2.1 的原结论实质成立**，机制改为「**≈4 h 硬上限**」。
   **立规**：判「环境能否托管」须以**能否跑完全程**为判据，**不得**以「比上次活得久」为判据。
3. **估时更正**：单 epoch **不稳定** —— epoch 1 ≈ **166–175 min**、epoch 2 ≈ **65–74 min**（差约 **2.5×**）
   ⇒ §7.3 的「stage1 ≈26 h」**极可能高估**，改记 **≈12–26 h 区间**，**须以本机终端实测重定**。
4. **结构性障碍（呈 W0）**：`04_01` **无逐 cell 落盘／无续跑** ⇒ **任何中断都使 stage1 从零重跑** ⇒
   本通道（≈4 h）**在结构上无法完成任务**；若要走「多条短命进程累加」，须先由 W0 批**可续跑**改造（**属口径变更**）。
5. **产物（如实）**：cell 1 完成 **2/3 epoch**；落盘 1 个 **mid-cell checkpoint**（`epoch 1` 最优，`opt_steps=1184`、`val_f1=1.0`、
   `tau=0.9999986291`；`adapter_model.safetensors` md5 **`7e5d2a4bed3d712596d7ed81dd5dd27e`**），已**改名**为
   `…20261004-183301_INTERRUPTED`（内容 md5 不变）；**无 grid CSV／无 grid 报告**，`grid_selected` 仍 **`PENDING_W0`**
   ⇒ **网格层面无污染**；该 checkpoint 属**半成品**，**不得**当作任一 cell 的结果。
6. **结论**：**交回用户在本机终端执行**（§7.2.2 命令，**`-u` 与 `--curriculum on` 均必须保留** —— 后者为 `N-7` 裁定① 更正所加，见 §7.2.2 的更正块）；
   **环境侧建议（不涉口径）**：解 `40 W` 强制功耗限／关桌面 GPU 程序／长跑期禁用睡眠。

### 7.5 网格中止 · 超参回填 · 四项证据（`D-43`～`D-46` 执行回执，`2026-10-04/05`）

> **本节由 `W5.1`＋`W5.2` 混合窗口追加（只追加，不改既有行）。** 主报告（含全部实测明细）：
> **`reports/train/qlora_grid_aborted_v1.0_20261005-001350.md`** —— 亦即 `grid_selected.source_report`（`D-46` 裁定五）。

**一、网格结局**：`cell 1`（`lr=1e-4, r=16`）跑到 **epoch 2/3** 时，W0 依 `D-43` 裁定一于
`2026-10-04 22:57` `taskkill //F //PID 24340` **主动中止**（GPU `7715 → 802 MiB`）。
⇒ 实际完成 **1/9 组、且仅 2/3 epoch**，**无 cell 结果**（`reports/train/` 无 grid CSV／报告）。

**二、超参回填（`D-43` 裁定二）**：`grid_selected = {lr: 2e-4, r: 16, source: v5_L434_default,
grid_status: ABORTED_SATURATED}`（见 `qlora_ours.yaml`；`status = SELECTED`）。
依据 = v5 L434 预注册默认值 —— **唯一既不依赖已退化判据、又不在搜索空间边界**的值。

**三、`cell 1` 的 `curriculum=off` 声明（`D-45`／`N-7` 裁定②）**：该次实跑**未启用课程与增强**
（`grid_manual_20261004-183300.log` L1 `curriculum=off`；`grep -c CURRICULUM` = **0**）
⇒ 饱和证据的口径为「**无课程／无增强**」，引用时须声明。

**四、τ 边界解双轨（`D-43` 裁定三；对照值由 `D-46` 裁定三定死 `0.5`）**：
τ* = **0.999999**（落在候选上界 ⇒ **⚠ 边界解**，必须标记）；对照 τ = **0.5** 下
`F1 0.99874／TPR 1.0／FPR 0.002525`。**不得再并报「候选中位数」**（`D-46` 裁定三）。
**`W6` 在 test 上须并列报两组数**，论文须声明该局限与敏感性。
**已知产物缺陷（登记未改）**：`write_threshold_curve` 的 `tau` 列 `_fmt(τ, 6)` 使接近 0 的候选
**塌缩为字符串 `"0.0"`** ⇒ 该列不可用于精确定位候选；`W6` 应改引 `TAU_OURS` 日志行或 `meta.json`。

**五、分层诊断（`D-46` 裁定一；`W5.3` 放行前置）**：τ 固定 0.999999，**未**在子集重寻；
**报误判率、不报分层 F1**。实测：攻击子集 `FNR = 0.012626`（10/792）、G3 负类 `FPR = 0`（0/179）、
明文负类 `FPR = 0`（0/613）⇒ `max = 0.012626 ≤ 0.05` ⇒ **判「真学会」**（捷径不成立）。
**风险附注**：明文负类 `score max = 0.999934` 距 τ* 仅 **6.5e-5** ⇒ 阈值余量极薄。
**六项泄漏排查（`D-46` 裁定二）**：含空 `inter = 1`（该值**即空串**）／**排空后六项全 0**；
**空值不参与交集统计**（空行 = 全部良性行：train 6336／val 792）。

**六、`N-5` 自证 v2（`D-46` 裁定四，更正 `D-43` 裁定六之过度）**：判据**维持 `loss`／`val_f1`／`tau`**
（事前定值于 `qlora_ours.yaml` → `determinism.verify_double_run_v2`；规模 smoke 正32／负32 × **1 epoch**；
含**事前写入**的降级条款）。实测双跑 `loss 3.42983597`／`val_f1 0.86614173`／`tau 0.531209` **Δ 全 0**，
curve md5 `5ed1d36f…`／adapter md5 `545b0f1c…` **与 `5c502d0` 历史自证逐字节相同** ⇒ **达标**。

**七、命令模板**：`§7.2.2`／`§7.4` 已补 **`--curriculum on`**（`N-7` 裁定①）。

**八、`W5.2` 联合留证**：`reports/train/curriculum_joint_v1.0_20261005-001427.md`（`N-7` 裁定③；
同一次运行内同时验到 `CURRICULUM_OK` ＋ 增强 ＋ **子采样比 1.2500 ok=True**）。

**九、呈 W0（本窗口不自裁）**：① **`std = 0` 风险**（`D-46` 裁定六）—— 若 6 次全量 `val F1` 全为 1.0
⇒ 3 seeds `std = 0` ⇒ 预注册「mean ± std」失效，属**口径变更**；② **阈值余量极薄**（见五）是否需在
`W6` 增列敏感性分析。**`W5.3` 仍不得开跑**（前置：`std=0` 裁定 ＋ 用户确认排程）。

**十、留证物**：`checkpoints/ours/…20261004-183301_INTERRUPTED/`（**半成品，不得当作任一 cell 的结果**，
`D-44` 裁定四）；`logs/train/probe_aborted_evidence.py`（只读，六项交集 ＋ 数据侧对照）；
`logs/train/probe_layer_diagnosis.py`（只读，分层诊断）。

### 7.6 `D-47` 执行回执：τ 主口径触发式改 `0.5` ＋ 推理分歧机理 ＋ 限定声明（`2026-10-05`）

> **本节由 `W5.1` 追加（只追加）。** 主报告：`reports/train/qlora_grid_aborted_v1.0_20261005-001350.md`。

**一、τ 主口径**触发式**改用 `0.5`（`D-47` 裁定二；**属口径变更**）**
> 规则（**事前写定**于 `qlora_ours.yaml` → `threshold.main_criterion_override`）：
> **τ\* 为边界解**（`τ* > 0.99`／`< 0.01`，或 `τ*` 为候选集极值）**且**「**负类 `score max` 与 τ\* 之差 `< 1e-3`**」
> ⇒ **主口径 τ = `0.5`（定值）**，**τ\* 降级为并列报告**。二者 `logic = AND`。
> **未触发 ⇒ 主口径 = `τ*`**（默认口径，`PREREG_v1.md` §8）—— **`D-53` 裁定三 更正**：原文此处
> 仅写「触发时」的后果而未并写「未触发时」的后果，易被读成无条件 `0.5`。
> **实测触发**：τ\* = `0.99999862909317017`（边界解），负类 `score max` = `0.99993390`，
> 差 **`6.47e-5 < 1e-3`** ⇒ **触发** ⇒ 主口径 `0.5`。
> **实现**：`04_01.resolve_tau_ours()`（`search_tau_ours` **原口径未动**，仅供其复用）＋
> `log_tau_ours()`；`meta.json`／`best.json` 并列落 `tau_star`、`tau_main_override`、
> `tau_main_override_reasons`、`tau_neg_max_gap`。**未触发时行为与原口径逐位一致。**
> **测试**：`tests/train/test_04_01_train_qlora_ours.py` 新增 **8 例**（触发／间隙过宽不触发／
> 非边界不触发／阈值**读自 yaml**（防硬编码）／候选极值支／`_exact` 全精度回归／配置事前定值）。
> **下游约束**（**`D-53` 裁定三 更正：改为「触发时 `0.5`、未触发时 `τ*`」，原为无条件表述**）：
> **`W6`（A6）在 test 上 —— 触发时以 `0.5` 为主口径、未触发时以 `τ*` 为主口径**，并**并列报另一者**；
> **逐 run 独立判定**（每 run 有各自的 `τ*` 与 `gap`，不得一次性统一）；τ 一律取**全精度**。
> ⚠ **实测（`D-53` 裁定三）**：`W5.3` 首跑 **未触发**（边界解 True 但 `gap = 0.9626216776669025 ≫ 1e-3`）
> ⇒ **主口径 = `τ* = 0.9999485611915588`（全精度）**。

**二、⚠ 对 `D-47` 裁定二 理由① 的**更正**（呈 W0）**
> `D-47` 理由①称「边界解在两次运行间翻转 10/792 样本 ⇒ **不可复现**」—— **该表述不成立**。
> 实测（§6.1）：翻转**全部**源于 **τ 的显示精度**（6 位 `0.999999` 大于全精度 τ\*），
> **非**运行间差异：同一 ckpt、同一 τ 两次推理**逐位相同**。
> **触发规则不受影响**：其**两条件均已满足**（见一）；理由②（改 `0.5` 使 `F1` 变差 ⇒ 反向、非择优）
> 与理由③（**早停重获上升空间**）**仍然成立** ⇒ **主口径仍改 `0.5`**。

**三、两次推理分歧的机理 = (a) 推理配置差异（`D-47` 裁定五）**
> 见主报告 §6.1。**`W5.3` 须知（统一配置）**：**τ 一律取全精度值** —— 来源 = 内存变量／`meta.json`；
> **禁止**用日志或曲线里的显示值复算。已落实为 `04_01._exact()`：τ 在**曲线 CSV 与 `TAU_OURS` 日志**
> 中改按**全精度**记录 ⇒ **`F-W5.1-2` 由此闭合**（原「登记未改」改为「已修复」，见台账）。

**四、限定声明（`D-47` 裁定一）**：见主报告 §2.1。**「饱和」仅在 τ\* 下成立**；
且**更正**所拟文本一处 —— 「复算 ≈0.9936」系**显示值所致**，在**全精度 τ\*** 下 `F1 = 1.0`。
**对 `D-43` 裁定一（停止网格）无影响**（`loss 0.041027 → 0.000010` 仍示饱和，结论**加强**）。

**五、`source_report` 逐字确认**：`reports/train/qlora_grid_aborted_v1.0_20261005-001350.md`，
**非 `null`**、文件存在、路径逐字匹配（`D-47` 复核 #4 已闭合）。

**六、`W6` 增列 τ 敏感性表（`D-47` 裁定三）**：`τ ∈ {0.5, 候选中位数, τ*}` 三档的 `F1／FPR／TPR`
＋ score 分布 `p50／p90／p99／max`（正负类分列）。属**报告增列、非口径变更**（归 `W6`）。

**七、`std = 0` 口径（`D-47` 裁定四，**事前定值**）**：① 因主口径改 `0.5`，`val F1` 不再恒 `1.0`
⇒ `std = 0` 风险**大幅缓解**；
> ⚠ **`D-53` 裁定三 更正**：① 的**前提已不成立** —— 覆盖规则**实测未触发** ⇒ 主口径仍为 `τ*`，
> 故 `std = 0` **并不因该机制缓解**；②③ **维持不变**（不论 `std` 是否为 0 均须给逐 seed 原值表，
> 且 `std = 0` 时须声明退化）。**下游按 ②③ 执行**，不得据 ① 假设已缓解。

② **不论 `std` 是否为 0，均须给逐 seed 原值表**；
③ 若某体制内 3 seeds 的 `val F1` 全相等（且非四舍五入所致）⇒ 该体制报「**逐 seed 原值 ＋ 声明 `std` 退化**」，
**禁止**只写「`mean ± std = x ± 0`」而不加声明。

### 7.7 `D-48` 执行回执：**step 级精确续跑**（可续跑改造 ＋ 等价性自证 **PASS**，`2026-10-05`）

> **本节由 `W5.1` 追加（只追加）。** 主报告：`reports/train/resume_v1.0_20261005-105319.md`。

**一、交付**：`--resume <state_dir>`（`04_01`／`04_00` 均有）；落盘**每 100 个 optimizer step**、
**只在 `opt.step()` 之后且 `pending == 0`**（**不存梯度**）；位置 = 独立 `state/latest|prev/`、
**原子写**（`tmp + os.replace`）、**不复用** `save_checkpoint`、**不写 tokenizer`**。
**未传 `--resume` 时行为与改造前逐字节一致** —— 训练循环改为打乱**下标序列**，
两实现「访问顺序逐项相同 ＋ RNG 消耗逐位相同」（已由单测钉死）。

**二、状态清单（`D-48` 裁定二 9 项）**：全部落地；**另如实追加 1 项** ——
`epoch_loss`／`n_micro`（曲线的 `train_loss` 是**该 epoch 均值**，不恢复则不等价）；
追加理由已写明，**未改**裁定二任何既有项。

**三、等价性自证（`D-48` 裁定四，**`W5.3` 放行前置**）：**通过**。**同一 `seed=42`、smoke 正32／负32 × 1 epoch、
切点 `opt_steps = 3`（**epoch 中途**，正对关键项 #6）** ⇒ `loss`／`val_f1`／`tau` **Δ 全 0**；
curve CSV 与 adapter `safetensors` **两处 md5 逐字节相同**（且 adapter md5 与历史 `N-5` 值相同）。
**未采用**裁定一 明禁的「epoch 末粗粒度／中断回退整轮」假改造。

**四、披露（`D-48` 裁定六）**：`meta.json`／`best.json` 已增记 **`segments`**
（实测：1 段 `[0→7]`；2 段 `[0→3]＋[3→7]`）。论文须声明「可分 `k` 段续跑且已通过等价性自证」，
**不得**写成「单次连续训练」；**总 GPU 时长不变 ≈20–52.5 h**。

**五、`W5.3` 命令（据 `N-7` 裁定① ＋ 本改造）** —— 在 §7.2.2 模板基础上：
① **`--curriculum on` 必须保留**（否则课程与子采样全部不生效）；
② 可随时中断并以 `--resume <state_dir>` 续跑（**同一 `seed`／`epochs`／`regime`**）；
③ **τ 一律取全精度值**（见 §7.6 三；**禁用**日志或曲线的显示值复算）；
④ 主口径 **`τ = 0.5`**（见 §7.6 一）；**未传 `--resume` 时不得**改变既有默认路径行为。

**六、通道（`D-48` 裁定五）**：默认 **(a) 本机终端 ＋ (b) 可续跑**；可选 **(c)** harness 分段托管
（每段 **≤3 h**、段末**须确认 state 落盘成功**再启下段、三判据跨 **≥3 时点**）。**由用户选定**。

### 7.8 `D-50` 执行回执：修 **`N-8`**（续跑产物目录分裂）＋ 复验通过（`2026-10-05`）

> **本节由 `W5.1` 补正轮追加（只追加）。** 主报告：`reports/train/resume_n8_v1.0_20261005-130106.md`。

**一、缺陷**：`ts` 每次进程启动**新生成**，`run_id` 依赖 `ts` ⇒ 续跑段落在**另一个 `run_id` 目录**
（`D-50` §二 实测：`…104756` 与 `…104856`）。**性质 = 产物可追溯性缺陷**（非等价性缺陷）；
`04_04` 按 `best.json` 取 ckpt 时目录翻倍、论文引用 `run_id` 不稳。

**二、修法**：`_snapshot` 的 `meta` 增 **`ts`／`run_id`** ⇒ `--resume` 时以 state 的 `ts`
**覆盖本进程新生成的 `ts`** ⇒ `save_checkpoint`／`write_best_pointer`／`write_curve`／
`write_threshold_curve` **全部沿用该 ts** ⇒ **续跑落回同一 `run_id` 目录**。
`ts`／`run_id` 已补为 `resume.state_items` 的**「第 10 项」**（如实追加，理由 = 产物目录连续性）。

**三、复验（`D-50` 裁定二·选定 第 6 条，四条判据）**：**全过** —— ① 三量 Δ 全 0；② curve md5
逐字节相同；③ adapter md5 逐字节相同；④ **续跑段产物目录 == 首段 `run_id` 目录**
（`state.run_id` 与第 2 段 ckpt 目录的 `run_id` 分量相同，实测 `…_id_seed42_20261005-125159`）。
**要求 3 实证**：未传 `--resume` 的 no-resume 运行，其 curve／adapter md5 与修复前**逐字节相同**
⇒ 修复**未扰动**既有路径。**要求 7**：`save_checkpoint` 的 `FileExistsError`（禁同名覆盖）
**原样保留**。

**四、⚠ `W5.3` 起跑（`D-50` 裁定三，7 条硬约束）—— 逐条必守**：
1. **每 run 一个 state 目录**（建议 `checkpoints/ours/state/<regime>_seed<seed>/`），**6 个互不复用**
   （复用会载入他 seed 状态 ⇒ 违反「每 seed 独立从头」）；
2. **首跑也要传 `--resume <空目录>`** ⇒ 判 `RESUME_STATE_INIT` 新跑并每 100 step 落盘
   （**不传则不落盘 ⇒ 中断后无从续跑**）；
3. 续跑命令与首跑**除 `--resume` 外逐字一致**（`--regime`／`--seed`／`--epochs`／`--config`／
   **`--curriculum on`**／`--report-dir`／`--out-dir`）—— 同目录 ＋ 同参数重跑即**续跑**；
4. **不得**借「分段」改 `epochs`／`lr`／`r`／`patience`／样本集（红线）；
5. 每段结束立即复核：`best.json` 存在、`curve_<regime>_<seed>_*.csv` 行数 = 已跑 epoch 数、
   记录 adapter md5 与 `segments`；
6. 可跨天／跨会话；**6 次建议顺序** `id(42→123→456)` → `ood(42→123→456)`，报告登记**实际顺序**；
7. 每段记一次显存峰值与墙钟；**(a) 通道前提 = 本机不睡眠**。

### 7.9 `D-54`／`D-53` 执行回执：**loss-NaN 守卫**（**默认启用**）＋ 转交项 #47（`2026-10-07`）

> **本节由 `W5.1` 补正轮追加（只追加）。** 主报告：`reports/train/nan_guard_v1.0_20261007-141707.md`。

**一、loss-NaN 守卫（`D-54` 裁定三，工程性改进、**非口径变更**）**
> 背景：`ood:42` 自 `EPOCH 2` 起 `loss=nan` 却**静默跑完**（跨 epoch 续跑至 1500 步、白烧 **8.21 h**），
> 且其 `--resume` 的 `state` 被**全 NaN 权重**污染（照做会静默载入坏权重）。
> **事前定值**见 `qlora_ours.yaml` → `nan_guard`；判据 `04_01.nan_guard_violated()`。
> **检测点 = 每个 micro-batch 的标量 `loss`**（`epoch_loss += <lv> * accum` 之后）
> —— **禁**只在 epoch 末检查（该 run 的 NaN 自 `EPOCH 2` **首步**即出现）。
> **触发 = 四不一停**：记 `NAN_GUARD_TRIGGERED …` ⇒ 不落盘 state／不覆盖 `best`／不删任何文件／
> 不自动恢复 ⇒ **退出码 3** ⇒ 停工呈 W0。
> **关闭开关 = `--no-nan-guard`**（`04_01`／`04_00` 均有；传之 ⇒ 与改造前**逐字节不变**）。
> **等价性自证（`D-54` 裁定三·5）已通过**：有守卫 vs `--no-nan-guard`，三量 Δ=0、
> curve md5 `5acaf5c4…`／adapter md5 `545b0f1c…` **逐字节相同**（且等于历史值）；
> 正常路径**无新增日志行**（剔除 tqdm 后 39 行对 39 行、差异 0）。

**二、⚠ 转交项 #47 更正 —— `τ` 主口径的「无条件化」表述（`D-53` 裁定三）**
> **`D-53` 裁定三**：`D-47` 裁定二 的 `0.5` 是**条件式覆盖**；`W5.3` 首跑
> **边界解 True 但 `gap = 0.9626216776669025 ≫ 1e-3` ⇒ 未触发** ⇒
> **主口径 = `τ* = 0.9999485611915588`（全精度）**。
> ⇒ 本文件 §7.6 一／七 与 `qlora_ours.yaml` 注释**已改为「触发时 `0.5`、未触发时 `τ*`」**
> （并注明 `std=0` 的「因改 `0.5` 而缓解」**前提不成立**，②③ 维持）。
> **⚠ 仍存在一处未改（转交 `W5.3`／`W5`）**：`scripts/train/04_03_run_three_seeds.py`
> **L590–594** 生成报告的 §5 仍写「`τ_main = 0.5`（定值）／**实测触发**／`τ*` 降级」。
> 该处须**由运行时数据派生**（逐 run `τ*`／`gap`／`triggered`），非替换措辞 —— 属 `04_03` 交付物本身。

**三、后续次序（`D-54` 裁定六，经 `D-56` 更正）**：① 本补正轮 = **NaN 守卫落地 ＋ 自证**
（**已完成**）⇒ ② 归档 NaN 现场（`D-54` 裁定四）—— **已由 `W0` 执行完毕**（`D-56` 裁定一 补裁执行者 = `W0`；
裁定二 回执：`state/ood_seed42` → `…_20261006-151615_NAN` ＋ **重建空槽位**、run 目录 → `…_NAN_INVALID`、
日志保持原名；本窗口磁盘核对属实）⇒ ③ `W5.3` 依次跑 `ood:42`（重跑）／`ood:123`／`ood:456`，
**三次全部在「有守卫版本」上**（`id` 组为无守卫版本，守卫为安全阀、正常路径逐字节等价 ⇒
可比性不受影响，但 **`W6.1` 报告与 `W7` 论文须写明该实现版本差异并附等价性自证**）。
**`D-56` 裁定三 重申**：② 归档**仅关闭风险敞口**，**不解除**另两项前置；**在守卫落地并自证通过前
不得对 `ood:42` 发车**（① 已由本节关闭 ⇒ 剩余仅 ③，且 ③ 为 `W5.3` 单独会话）。

## 8. checkpoint（C8）

```
checkpoints/ours/<stage>_<seq>_<model>_<regime>_seed<seed>_<YYYYMMDD-HHMMSS>/seed_<seed>/checkpoint-<step>/
```
- 实跑样例（冒烟区，非权威）：`04_01_llamaguard1b_id_seed42_20261004-111443/seed_42/checkpoint-7/`
- **禁止覆盖同名**；每个 seed 独立从头，不续接他人 checkpoint。
- 同 run 下另落 `best.json`（指向验证集最优 step）。

## 9. 课程学习（W5.2 追加）

> **本节由 W5.2 追加（只追加，不改本文件既有行）。**
> 唯一数值出处 = `configs/train/curriculum.yaml`（事前定值）；实现 = `scripts/train/04_02_train_curriculum.py`，
> **只被 `04_01` import，不单独执行训练**。开关 = `04_01 --curriculum on|off`（默认 `off`）。

### 9.1 难度与阶段（累积式 3 阶段，固定 step 比例）

- 难度来源 = `difficulty` 列（封闭域 `low`/`medium`/`high`），映射依 `DATA_CARD` §5：
  `rot13`/`reverse` = low；`base64`/`hex`/`morse`/`url` = medium；`ascii_art`/`zerowidth`/`homoglyph` = high。
- 阶段切点 = **事前定值的固定 step 比例 `[0.3, 0.6, 1.0]`**（`end_i = floor(比例_i × total_steps)`），
  **不由 val 触发** —— val 只承担监控／早停／τ（`W5.2` §3-3）。
- **累积式**：后一阶段包含前一阶段全部样本（避免学新档忘旧档）；非累积对照口径**本阶段不跑**。

| 阶段 | 档位（累积） | 体制一覆盖编码 | 体制二覆盖编码（恒 5 训练码） |
|---|---|---|---|
| stage 1 | low | `reverse`、`rot13` | `reverse`、`rot13` |
| stage 2 | ＋medium | ＋`base64`、`hex`、`morse`、`url` | ＋`base64`、`url` |
| stage 3 | ＋high | ＋`ascii_art`、`homoglyph`、`zerowidth` | ＋`zerowidth` |

### 9.2 编码感知增强（追加式，p = 0.30，正负比上界 1.25）

- 形态 = 对 `payload` 施加**一次二次编码**（在 `allowed_encodings` 内取、且**不得**与自身 `encoding_type` 相同）；
  锚点 = `payload` 在 `text` 中的子串，`replace_all`；锚点缺失 ⇒ **跳过并计数**（不造锚点）。
- 方式 = **追加式**（原分片一行不动，变体追加在尾部）；概率 **p = 0.30**。
- 正负比上界 **1.25:1**；超界时**只丢弃增强行**（`discard_scope = augmented_only`），
  丢弃顺序 `low → medium → high`；**禁止**为达标删减原分片行。
- 只作用于 **train 分片的 `label == 1` 行**；**val 不增强、不参与**（实测 val 帧指纹增强前后一致）。
- 增强后仍经 `03_06.truncate_left(512)`「原文直通零清洗」，不另起清洗。
- 增强行以 `augmented = yes` 标记 ⇒ **不计入**跨检测器 `text_sha256` 一致性校验（该锚点服务**评估**输入）。
- 编码器**复用** `scripts/data/02_03_encode_variants.py::ENCODERS`（`V-5` 单源），本窗口不自造编码。

### 9.3 实测（`04_02 --dry-run`，seed 42，`total_steps = 3547` 展示刻度）

| 项 | 体制一（id） | 体制二（ood） |
|---|---|---|
| train 行数（原 → 增强后） | 12672 → **14256** | 7430 → **8475** |
| 抽中／成功／跳过（锚点缺失） | 1895／1828／67 | 1045／1045／0 |
| 正负比（原 → 最终） | 1.0000 → **1.2500**（上界**触发**，丢弃 244 条 low） | 0.9003 → **1.1675**（未触发，丢弃 0） |
| 各阶段入选行数（s1／s2／s3） | 7890／11517／14256 | 5715／7549／8475 |
| 4 留出码命中（分类列 ＋ 内容列） | 不适用（体制一不留出） | **0**（参照集 3168 串，扫描 `text`/`plaintext`） |
| 增强后 frame 指纹（SHA-256 前 16） | `3960ca031c090b61` | `e20205a5f99e7df0` |
| 报告 | `reports/train/curriculum_v1.0_20261004-125443.md` | `reports/train/curriculum_v1.0_20261004-125446.md` |
| 日志 | `logs/train/04_02_train_curriculum_20261004-125443.log` | `…20261004-125446.log` |

> `total_steps = 3547` 仅作 dry-run 展示刻度（= W5.1 实测 4730 micro-batches/epoch × 3 epoch ÷ grad_accum 4）；
> 真实阶段边界由 `04_01` 训练时按实际 micro-batches 传入，**同一份比例**缩放。

### 9.4 与 `04_01` 的接线（W5.2 补齐）

- `04_01 --curriculum on` ⇒ ① `build_curriculum_frame()` 调 `04_02.augment_attack` 生成增强帧；
  ② 用该帧编码得到 `train_exs`；③ 每个 optimizer step 用 `CurriculumScheduler.select(frame, step)`
  过滤该 micro-batch 的行下标；**整批被排除则该 step 跳过**（`CURRICULUM_SKIPPED_MICRO` 计数），
  **不回退成全量** —— 否则难样本会在 stage 1 渗入、课程失效。
- `configs/train/qlora_ours.yaml` 的 `curriculum.config_path` 指向 `configs/train/curriculum.yaml`；
  **缺此项会静默降级为 off**（W5.2 实跑已观测并修复）。
- 实跑证据（冒烟区，非权威）：`logs/train/04_00_smoke_train_20261004-131019.log`
  ⇒ `CURRICULUM_OK` ＋ `CURRICULUM_AUGMENT rows 64→72（抽中 10／成功 10／丢弃 2；上界触发）`
  ＋ `CURRICULUM_SKIPPED_MICRO=7` ＋ `SMOKE_RESULT=PASS`。

### 9.5 呈 W0 事项（本窗口未自行裁定）

1. 「累积式 3 阶段 ＋ 固定 step 比例（**不按 val 触发**）」与 `SOURCE_LOCK` v5 **L759**「课程切换阶段全部依据验证集」
   **字面冲突** —— 本窗口按 `W5.2` §3-3（后出、更具体）执行，请 W0 裁定以哪份为准。
   〔**`D-42` 裁定一已澄清（2026-10-04）**：以**固定 step 比例为准**并追认；呈报原引 L756 **有误**，课程切换句实为
   **L759**（L756 = 「裁判 prompt 必须防二次注入」）。该句硬约束是「禁测试集」，「依据验证集」是其正向补充，
   非「必须由 val 触发切换」。〕
2. 增强采用**追加式 ＋ 上界 1.25** 而非替换式，请 W0 追认。〔**`D-42` 裁定二已追认**。〕
3. 良性行 `difficulty` 恒为空（实测各分片无一例外）⇒ 良性行**恒入选**，
   各阶段正负比 ≠ 1:1（实测见 §9.3）。若要维持 ≈1:1 须另立良性子采样口径（属口径变更）。
   〔**`D-42` 裁定三已准予良性子采样**；**补正轮已实现并实测**（见 §9.6），各阶段比恒 = 1.2500／1.1675。〕
4. v5 L438「同一攻击样本随机施加**多种**编码」：本窗口落地为**施加一次二次编码**，
   非「同一样本生成多种变体」（后者会把正负比推高、需另设上界）。请 W0 追认。
   〔**`D-42` 裁定四已追认**，口径锁定为「跨样本随机施加多种编码中的每一种；同一样本只施一次」。〕
5. 一号门禁**参照集口径收窄**：`02_07.check_a5` 的参照集覆盖含 `test.csv` 的全部分片，
   而 A5 禁读 test ⇒ 本窗口参照集只取 4 个允许分片中留出编码攻击行的 `payload`/`text`。
   〔**`D-42` 裁定五已更正表述（2026-10-04）**：**排除 `plaintext` 并非本窗口的收窄** ——
   `02_07.check_a5` 的参照集**本就只取 `payload`/`text`、刻意不含 `plaintext`**（源码 L345–348，
   假阳性 3960 的出处亦在该 docstring），本窗口系**沿用既有口径**。**真正的收窄只有一项**：
   参照集**分片覆盖**由「全部分片（含 `regime_ood/test.csv`）」降为「4 个授权分片」（A5 禁读 test，无规避空间）。
   收窄方向**保守**（只漏检、不误放行），**认可、不追责**；**正式结论由 `W6`（A6）在全分片上复跑 `04_02` 一号门禁**。〕

> **§9.5 的处置结果（W0 `D-42` 已全裁）**：①「固定 step 比例为准，追认」（裁定一，且更正 v5 句实为 **L759**）；
> ② 追加式＋上界 1.25 **追认**（裁定二）；③ **准予良性子采样**（裁定三，见 §9.6）；
> ④ 一次二次编码 **追认**并锁定口径（裁定四）；⑤ 参照集收窄 **认可不追责**，表述已更正、正式结论归 `W6`（裁定五）。

### 9.6 W5.2 补正轮：良性子采样（`D-42` 裁定三；**本次追加，替换 §9.3 的阶段比列**）

> **触发**：`D-42` 裁定三 —— 良性行恒入选使各阶段正负比**漂移约 5 倍**（id 0.2453／0.8177／1.2500），
> 与训练结束后寻优的**单一固定 τ** 不匹配，违背 `V-4`「正负 ≈1:1」精神。
> **W0 硬约束**：各阶段比恒 = 增强帧比（id **1.2500**／ood **1.1675**）±0.05；**只减良性**；
> **stage 3 等价全量帧**；规则**事前定值**；确定性用 `random.Random(seed)`。

**实现口径（写入 `configs/train/curriculum.yaml` → `plan.benign_subsample`，事前定值）**：
对每个**非末**阶段，保留全部入选攻击行 `pos`，把良性候选**确定性前缀**子采样到
`neg_kept = round(pos / target_ratio)`（夹在 `[min_benign_kept, neg_total]`）；
`target_ratio` = 增强帧实测比（`target_ratio_measured.id/ood`）；**stage 3 不子采样**。

**实测（`04_02 --dry-run`，seed 42，`total_steps = 3547`）—— 六项断言全 PASS**：

| 阶段 | id 攻击 | id 良性（候选→保留） | id 比 | ood 攻击 | ood 良性（候选→保留） | ood 比 |
|---|---|---|---|---|---|---|
| 1 | 1554 | 6336 → **1243** | **1.2502** | 1805 | 3910 → **1546** | **1.1675** |
| 2 | 5181 | 6336 → **4145** | **1.2499** | 3639 | 3910 → **3117** | **1.1675** |
| 3 | 7920 | 6336（**不子采样**） | **1.2500** | 4565 | 3910（**不子采样**） | **1.1675** |
| 各阶段入选行数 | — | 2797／9326／14256 | — | — | 3351／6756／8475 | — |

- **增强帧指纹不变**（id `3960ca031c090b61…`／ood `e20205a5f99e7df0…`）—— 子采样只改 `select()` 输出，
  **不改增强帧本身**（故 §9.3 的增强相关数值仍成立）。
- 攻击行**一条未减**（stage1 1554／1805、stage3 7920／4565 均 = 全量档位攻击行）。
- 报告：`reports/train/curriculum_v1.0_20261004-160907.md`（id）／`…160918.md`（ood）；
  日志：`logs/train/04_02_train_curriculum_20261004-160907.log`／`…160918.log`。
- `04_02 --self-check` **15/15 PASS**（新增第 15 项「良性子采样」）；`tests/train/test_04_02_train_curriculum.py`
  **35 passed**（新增 3 例子采样回归）。

## 10. 三 seeds 训练（W5.3 追加）

> **本节由 W5.3 追加（只追加，不改本文件既有行）。** 运行报告：`reports/train/three_seeds_preflight_v1.0_20261005-144621.md`。

### 10.1 调度器／命令源（`04_03`）—— 状态：**命令源 ＋ 汇总工具已交付**（`TRAINING_PENDING`）

`04_03_run_three_seeds.py` 原为占位骨架（`NOT_IMPLEMENTED`），本轮**已实现**。职责 = **命令源 ＋ 单 run 执行 ＋ per-run `--resume` 透传 ＋ 汇总**（**不实现训练循环**，`AGENTS.md` §4.6）。

| 项 | 值 |
|---|---|
| 调用方式 | **子进程**（`subprocess.Popen`）：① 每 run 全新进程 ⇒ 显存／CUDA context／全局 RNG／确定性开关**零串扰**（对齐「每 seed 独立从头」）；② 单 run 崩溃**不拖垮**调度器与其他 run（满足「保留已完成的 run 与其日志」）；③ 与 §2.2 手工命令**逐字同构** ⇒ `--dry-run` 打印体与实际执行体**同源**；代价 = 每次多一次模型加载（≈1–2 min／3.3–8.75 h） |
| `--dry-run` | 恒打 6 条命令（**每条含 `--curriculum on` ＋ 该 run 专属 state 目录**）＋ `run_id` 预览；自检 `DRYRUN_RESULT=PASS n=6 state_dirs=6`；**受 `--only` 收窄** |
| `--only <regime>:<seed>` | 只跑指定一个 run（默认通道「一次一次跑」） |
| `--resume` 透传 | `build_argv()` **恒**渲染 `--resume <resume_root>/<regime>_seed<seed>`，**不存在「第一次跑不传」的代码路径** |
| state 目录 | `checkpoints/ours/state/{id,ood}_seed{42,123,456}`（**6 个互不复用**，由 `state_dir_for()` 单一函数派生 ⇒ 结构上不可能复用） |
| 汇总 | `reports/train/three_seeds_v1.0_<ts>.{csv,md}` ＋ `curves_summary_v1.0_<ts>.csv`；CSV 列含 `n_segments`／`segments`／`state_dir`／`run_dir`（权威产物目录）／`tau_main`／`tau_star` |
| 追加开关 | `--summary-only`（不训练、只汇总 —— 供「用户手写 6 条 `04_01`」时履行汇总职责）／`--python`（指定调起 `04_01` 的解释器） |
| 红线 | **不读** `data/sealed_test/**` 与 frozen 侧 `test.csv`（静态测试按 **AST 的文件系统调用**扫描钉死）；**不改** `epochs`／`lr`／`r`／`patience`／样本集（§7.8 四.4）；**不删**任何 run 目录 |
| 测试 | `tests/train/test_04_03_run_three_seeds.py` **35 例**（不加载模型、不需 GPU）；`pytest tests/` **1148 passed**（1113 → +35） |

### 10.2 ⚠ 结论 —— **6 次实跑尚未开始**

本轮的判定严格按 §7.8 四：**每 run 一个 state 目录**、**首跑也传 `--resume <空目录>`**、
**续跑除 `--resume` 外逐字一致**、**不得借分段改 `epochs`／`lr`／`r`／`patience`／样本集**、
**每段结束立即复核**（`best.json`／curve 行数 = 已跑 epoch 数／adapter md5／`segments`）、
**可跨天跨会话**（建议顺序 `id(42→123→456)` → `ood(42→123→456)`，**实际顺序须登记**）、
**每段记显存峰值与墙钟**、**(a) 通道前提 = 本机不睡眠**。

- **执行通道由用户选定**（§1.1）：默认 **(a) 本机终端 ＋ (b) 可续跑**；可选 **(c)** harness 分段托管（段 ≤3 h）。
- **总 GPU 时长不变 ≈20–52.5 h**（单次 ≈3.3–8.75 h）；**`segments` 披露（`D-48` 裁定六）**：
  论文须声明「可分 *k* 段续跑且已通过等价性自证」，**不得**写成「单次连续训练」。
- **汇总报告与 6 组 LoRA 权重待实跑后**由 `04_03` 生成；本窗口**未产出任何训练数值**。
- **未决／待裁**：① 冻结面复核时的一次 **A5 红线误读**（`data/sealed_test/**` 下跑 `md5sum -c`，**只读零写**）已如实自陈，**请 W0 裁定**；② 两项追加开关（`--summary-only`／`--python`）是否追认。明细见运行报告 §3／§5。

### 10.3 收口回执：**6/6 完成**（`2026-10-08` 追加；只追加，不改 §10.1／§10.2）

**权威汇总** = `reports/train/three_seeds_v1.0_20261008-003324.{csv,md}` ＋ `curves_summary_v1.0_20261008-003324.csv`。
**逐 run 复核台账** = `reports/train/three_seeds_review_v1.0_20261006-003000.md` §1–§8。

| regime | seed | `run_id` | val F1 | TPR | FPR | τ_main（= τ\*，全精度） | epochs | best_step（best epoch） | adapter md5 | `n_seg` | 墙钟(h) | peak(MiB) | early_stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 42 | `04_01_llamaguard1b_id_seed42_20261005-150806` | 1.0 | 1.0 | 0.0 | `0.9999485611915588` | 3 | 1574（2） | `4368cf186d2731384b7682f3718def54` | 1 | 8.90 | 7375.7 | False |
| id | 123 | `04_01_llamaguard1b_id_seed123_20261006-010210` | 1.0 | 1.0 | 0.0 | `0.9999151825904846` | 3 | 2922（3） | `b30a06ade618d191ee94f26fba19eef5` | 1 | 3.12 | 7372.6 | False |
| id | 456 | `04_01_llamaguard1b_id_seed456_20261006-120248` | 1.0 | 1.0 | 0.0 | `0.9997695088386536` | 3 | 1571（2） | `19b2dd84da538e27dd5a963e5a0d512c` | 1 | 2.73 | 7372.2 | False |
| ood | 42 | `04_01_llamaguard1b_ood_seed42_20261007-144100` | 1.0 | 1.0 | 0.0 | `0.9999945759773254` | 3 | 516（1） | `58b0031ad68dbef13297a0822adc3061` | **2** | 1.78 | 7371.9 | True |
| ood | 123 | `04_01_llamaguard1b_ood_seed123_20261007-180342` | 1.0 | 1.0 | 0.0 | `0.6513548493385315` | 3 | 514（1） | `beefb658c4d2310b0b5f825d1c3b6fc8` | 1 | 2.10 | 7372.6 | True |
| ood | 456 | `04_01_llamaguard1b_ood_seed456_20261007-202634` | 1.0 | 1.0 | 0.0 | `0.9975273609161377` | 3 | 527（1） | `76af329638f6d6a2d14e9f3cba05bee0` | 1 | 3.80 | 7374.9 | True |

**按体制 mean ± std（`val F1`）＋ `std` 退化声明（`D-47` 裁定四）**

| regime | n | 逐 seed 原值 | mean | std（`ddof=1`） | std（`ddof=0`） | `std` 退化 |
|---|---|---|---|---|---|---|
| `id` | 3 | `1.0` / `1.0` / `1.0` | `1.000000` | `0.000000` | `0.000000` | **是** |
| `ood` | 3 | `1.0` / `1.0` / `1.0` | `1.000000` | `0.000000` | `0.000000` | **是** |

> **声明**：两体制 3 seeds 的 `val F1` **全相等且非四舍五入所致** ⇒ **以逐 seed 原值为准**，
> **禁止**仅读作 `mean ± std = 1.0 ± 0`。TPR 亦 6/6 = `1.0`、FPR 6/6 = `0.0`。

**汇总**：6 次合计 **80736.2 s = 22.43 h**（单次 1.78–8.90 h，**均未触及 `D-53` 裁定五.2 的 >10 h 呈报线**）；
峰值显存 max **7375.7 MiB**；`segments` 合计 **7 段**（`ood:42` 为 2 段，其余 5 个 1 段）；
6 个 state 目录 = `checkpoints/ours/state/{id_seed42, id_seed123, id_seed456, ood_seed42, ood_seed123, ood_seed456}`（**6 个互不复用**）；
`τ_main_override` **6/6 = `False`** ⇒ 主口径 = τ\*（与 `D-53` 裁定三 一致）；`early_stopped` 呈**体制分化**：`id` 3/3 为 `False`（跑满 epoch 预算）、`ood` 3/3 为 `True`（epoch 3 达 patience=2）。

**⚠ 本窗口收口复核查出并已修的缺陷（**实现层、非口径、不需重跑**）**：`04_03` 汇总报告渲染层
用 `x or "—"` ⇒ **`0.0` 为 falsy**，使 6 个 run 的 `FPR = 0.0`（完美）在报告 §1／§3 被显示成 `—`（缺失）；
**CSV 侧一直正确**。已新增 `_cell()`（仅 `None`／空串记 `—`）并替换 17 处、＋2 回归测试，
修复后**重新生成**权威汇总 `…20261008-003324`（旧稿保留不删，其 CSV 可直接用）。

**⚠ 呈 W0 备查（不做定性）**：① **`ood_val ⊆ id_val`**（实测唯一 `text` 交集 `927 = |ood_val|`；
设计依据 = `DATA_CARD` 的 `V-2` ＋ `V-6`）⇒ id 与 ood 的 **val 指标在文本层面互不独立**，
与 `D-53` 裁定六 同向加强，建议 `W6`／论文 Limitation 一并声明；② **`τ` 触发规则逐 epoch 判定**
（6 次中主口径出现过内部 τ\*／边界 τ\*／run 内条件式 `0.5`）⇒ `W6.1` 的「逐 run 独立判定」宜明确为「**逐 epoch 判定**」。

**交接**：`W5.4`（灾难性遗忘门禁）**须另一会话**执行（`D-30`）；按 `AGENTS.md` §4.6 与 `D-41` 裁定六，
`04_04` 读**6 个** `best.json` 指向的 ckpt，其 **sealed 侧部分归 A6 执行**，本窗口（A5）**只完成 val 侧自检**。

## 11. 灾难性遗忘判定（W5.4 追加）

> **本节由 W5.4 追加（只追加，不改本文件既有行）。**
> 窗口：`W5.4`（权限 **A5**，**单独会话**，`D-30`）；脚本 `scripts/train/04_04_check_forgetting.py`；
> 交接文件（固定名）`reports/train/forgetting_v1.md`。

### 11.1 判定口径（冻结 v1.0）

| 项 | 冻结值 |
|---|---|
| 预注册原文 | `PREREG_v1.md` §5「灾难性遗忘判定｜微调后干净原始测试集 **F1 下降 ≤3%**」＋「补充报告｜同时报告 **FPR、TPR** 变化」（v5 新增 24，`SOURCE_LOCK/实验执行方案（最终版 v5）.md` **L456–L462**；clean 分片用途见 **L412**） |
| **主口径** | **绝对百分点差** `ΔF1 = F1_base − F1_ours ≤ 0.03` ⇒ PASS |
| **并列口径（必须同时给出）** | 相对降幅 `ΔF1 / F1_base` |
| 三项指标 | 每个 seed 同时报 **F1、FPR、TPR** 的变化；**只看 F1 即判报告不完整** |
| 判定粒度 | **逐 seed 判定**，再给 mean ± std；**不得**只用 3 seeds 均值过阈 |
| 符号约定 | `ΔX = X_base − X_ours`；对 F1／TPR `Δ>0` = 下降（坏）；对 FPR `Δ<0` = 上升（坏） |

### 11.2 基准双轨（A5 无 sealed 权限 ⇒ 双轨）

| 轨道 | 基准 | 值 |
|---|---|---|
| ① clean（**正式结论轨**，归 `W6`／A6） | 引 tag `baseline-v1` 冻结数（`reports/baseline/unify_metrics_v1.0_20261004-013922.csv` 的 `b3 × clean_test`） | F1 **0.9885**（退化形式）／TPR **0.9773**／**FPR `n/a`** |
| ② val（**本窗口同机实跑**，`03_03`） | 基座在 frozen 侧 `id_val`／`ood_val` 的实测 | `id_val`：TPR **0.959596**／FPR **0.589646**／F1 **0.752848**（n=1584，TP 760／FP 467／FN 32／TN 325）<br>`ood_val`：TPR **0.972727**／FPR **0.437372**／F1 **0.791859**（n=927，TP 428／FP 213／FN 12／TN 274） |

- **双轨的原因**：`clean_original_test.csv` 44 行**全 `label=1`** ⇒ **FPR 无定义、F1 为退化形式**；只跑 clean 无法交代 FPR 变化，与预注册「同时报 F1、FPR、TPR」冲突 ⇒ 以 val 轨道补齐。
- **基座侧配置**：`configs/train/forgetting.yaml`（**新件，A5 侧**）—— 权重/加载/生成/prompt/映射与 `configs/baseline/b3_llamaguard1b.yaml` **逐项同值**，但 `splits` **只含 frozen 侧 `id_val`／`ood_val`**、无 sealed（`D-41` 裁定五③：**禁复用** baseline 侧的 sealed 配置）。

### 11.3 本轮结果（**val 自检；`STATUS=SELFCHECK_ONLY`／`GATE=UNKNOWN`**）

| 体制 | seed | step | τ（全精度） | F1 | ΔF1(pp) | ΔF1(rel) | FPR | ΔFPR(pp) | TPR | ΔTPR(pp) | 逐 seed 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `id` | 42／123／456 | 1574／2922／1571 | 0.99994856／0.99991518／0.99976951 | 1.0 | **−0.247152** | −0.328289 | 0.0 | +0.589646 | 1.0 | −0.040404 | PASS ×3 |
| `ood` | 42／123／456 | 516／514／527 | 0.99999458／0.65135485／0.99752736 | 1.0 | **−0.208141** | −0.262850 | 0.0 | +0.437372 | 1.0 | −0.027273 | PASS ×3 |

- ⚠ **`std` 退化声明**：两体制各自 3 seeds 的 Δ **全部相同** ⇒ `std = 0.000000`；**以逐 seed 原值为准**，禁止读作「mean ± std = x ± 0」而不加声明（`D-47` 裁定四）。
- ⚠ **τ 不对称（方法学）**：Ours 的 τ 取各 run `best.json` 的 `tau_ours`，是在**同一 val 分片**上按 F1 最大寻得 ⇒ Ours 的 val 指标属**样本内**；基座为判词型、无 τ ⇒ 该不对称使 ΔF1 偏**有利**于 Ours（**门禁偏松**）。已呈 W0。
- **并列补充诊断（非主口径）**：同一 `score≥τ` 规则下**基座**的 val 指标 = `id` F1 **0.688295**／FPR 0.888889／TPR 0.991162（τ_main 0.36296919）；`ood` F1 **0.747967**／FPR 0.361396／TPR 0.836364（τ_main 0.43782353）⇒ **同规则下 Ours 仍 +0.25～0.31 F1**，"提升"非规则更换所致。

### 11.4 硬约束与执行归属

1. **A5 会话恒不传 `--clean-test`** ⇒ `STATUS=SELFCHECK_ONLY`、`GATE=UNKNOWN`、**exit=0**（§3：不得因未跑 clean 而失败）。
2. **路径红线**：`--val` 只接受 `splits/regime_{id,ood}/val.csv`；传入 `data/sealed_test/**` 或 `.../test.csv` ⇒ **立即非零退出（exit=2）并打印 `REJECTED` 行**（单测覆盖两类）。
3. **`03_03` 调用四条例**（`D-41` 裁定五）：`--report-dir` 落 `reports/train/`；`--outputs-dir` 显式给定；**禁复用** baseline 侧 sealed 配置；子进程 stdout 出现 `REMINDER` **即停工**。
4. **τ 一律取 `best.json` 的 `tau_ours`**（`D-61` 转交项 #52），本窗口**不**重新寻 τ／不选 ckpt／不调参。
5. **正式结论归 `W6`（A6）**：同一脚本、同一参数，只加 `--clean-test`，**读前须先在 `data/sealed_test/CodeObf-PI_v1.0/ACCESS_LOG.md` 登记 6 列**。`GATE` 取逐 seed 最劣者；超阈即如实报 `FAIL` 并走回退（`PROJECT_CONTRACT.md` §4）。

### 11.5 产物与权威稿

- 权威运行报告（含本轮回执）：`reports/train/forgetting_v1.0_20261008-082812.md`（`id`）、`reports/train/forgetting_v1.0_20261008-084551.md`（`ood`）。
- 非权威稿（已自标【中间稿】）：`reports/train/forgetting_v1.0_20261008-080852.md`。
- 逐 checkpoint 明细：`reports/train/forgetting_detail_v1.0_20261008-{082812,084551}.csv`；
  基座 val 实测：`reports/train/base_val_metrics_v1.0_20261008-070115.csv`。
- **权威稿判定依 `PROJECT_CONTRACT.md` §2-9（`D-40` 裁定三）**：= 含该轮回执的那一份，非"最新时间戳"。

---

## 12. 消融 A／B 组（**W5.6 追加**，`D-83` 裁定一～五）

> **本节为「阶段4 补正」节**：消融 A／B／C／D 属 v5 §5.6 ＋ `PREREG_v1.md` L67（补录 G6）的
> **预注册内容**，但阶段4 从未排入 A／B 训练（计划缺口，`D-83` 裁定一 事实 3）。
> 本窗口补训 **A×3 ＋ B×3 = 6 run（`id` 主体制；`D-84` 裁定五：**不扩** `ood`）**；
> **不推翻** tag `train-ours-v1`（C 组结果有效），A／B 训练**另打 tag**（建议 `train-ab-v1`）。
> **长时训练须用户本机终端跑**（本沙箱不托管）。
> **⚠ 实跑前置（`D-84` 裁定一 条件 1／裁定六 1）**：**预注册补齐件 `PREREG_v1.7.md` 须先落盘**，
> 之后方可发起 6 次实跑。

### 12.1 三组口径与开关映射（`D-83` 裁定四 工作 1／2；P-3 = **`W5.6` 窗口提案「读法一」**，**追认见 `D-84` 裁定一**）

| 组 | 增强 | 课程 | `04_01` 开关 | 说明 |
|---|---|---|---|---|
| **A** | off | off | `--augment off --curriculum off` | 仅原始训练帧，**无任何增强** |
| **B** | on | off | `--augment on --curriculum off` | 编码增强（随机施加）＋ 普通随机增强，**无课程** |
| **C** | on | on | `--curriculum on` | 编码增强 ＋ 课程学习，完整方法（**阶段4 既有 6 run**） |

- **「普通随机增强」的事前定义（P-3）** = 既有「编码感知增强」（`04_02.augment_attack`，
  二次编码、`p = 0.30`、`ratio_cap = 1.25`）在**无课程调度**下的**随机施加形态**；
  **不新造增强组件**。其参数**单一取自** `configs/train/curriculum.yaml` §4 的 `augment` 段
  ⇒ 与 C 组**逐字同源**、无口径漂移。
- 该读法下 **A vs B 的唯一差异 = 有无增强**；**B vs C 的唯一差异 = 有无课程调度**（单因素阶梯）。
- **如实披露**：v5 §5.6 原文只给名称、未给定义；本读法与 `windows/W5.6_*` §2.3 的命令模板
  （A=`--augment off`、B=`--augment on`，两组均 `--curriculum off`）自洽，且与 C 组实跑日志自洽
  （C 组仅见 `CURRICULUM_AUGMENT`，无第二类增强）。
- **⚠ 术语解释选择（`D-84` 裁定一 条件 2，须明写）**：**v5 §5.6 字面为「编码增强 ＋ 普通随机增强」
  两项**，本读法**把它们合并为一项**（即对同一编码增强的「随机施加」形态）。
- **⚠ 不等价声明（`D-84` 裁定一 条件 3）**：**不得**声称本读法与 v5 §5.6 字面**完全一致**；
  「普通随机增强」**全库无既有实现**（实测），但**「无实现」不等于「不要求」**。
- 本口径已写入预注册补齐件 **`PREREG_v1.7.md`**（`D-84` 裁定一 条件 1）。

### 12.2 训练器改造（`D-83` 裁定四 工作 3；**红线 = C 组可原样复现**）

`04_01` 原把「增强」与「课程」绑在同一个 `--curriculum on` 上（`off` ⇒ 两者皆无），
⇒ B 组所需「有增强、无课程」**无法表达**。本窗口改造（实现层，登记 `DEVIATION_LEDGER.md`；
**追认见 `D-84` 裁定二** —— `W0` 已在 `AGENTS.md` §4.6 的 `04_01` 行补记两开关）：

| 新增 CLI | 默认值 | 语义 |
|---|---|---|
| `--augment {on,off}` | `off` | **独立**控制编码增强；生效式 = `(--curriculum on) ∪ (--augment on)` |
| `--model-key KEY` | `None` | 覆盖 `checkpoint.model_key` ⇒ 影响 `run_id` 的 `{model}` 槽（A／B 自识别） |

- **红线保持**：`--curriculum on` 单独使用（历史命令不带 `--augment`）时走**原分支**
  （`if cur_cls is not None`），行为**逐字不变** ⇒ C 组复现路径不受影响；
  默认值下（`--augment off` ＋ `--model-key` 未传）与改造前**逐字节一致**。
- `--augment` **只在 `--curriculum off` 时起独立作用**（否则 `--curriculum on` 会因默认 `off` 而丢增强）。

### 12.3 配置与命令源

- 配置（**事前定值**）：`configs/train/ablation_ab.yaml`（由 `qlora_ours.yaml` 逐字复制后
  仅追加 §0「A／B 档位」＋ `curriculum:` 节说明；其余各节与 C 组完全一致）。
- 调度器／命令源：`scripts/train/04_06_train_ablation_ab.py`（**只调用 `04_01`，不实现训练循环**；
  命名 `04_06` = **`W5.6` 窗口提案**（**追认见 `D-84` 裁定三**）—— `04_05` 已属 `04_05_train_unplug_tiny_backup.py`，
  `W0` 已在 `AGENTS.md` §4.6 增列）。
- 逐 run state 目录（**6 个互不复用**，首跑也传）：`checkpoints/ours/state/<AB>_id_seed<seed>/`。

### 12.4 6 条实跑命令（本机终端 Git Bash；`--dry-run` 与其**同源**）

```bash
# A（--augment off）/ B（--augment on）；两组均 --curriculum off
cd "C:/Users/lipei/Desktop/实验项目文件" && "C:/Users/lipei/miniconda3/envs/deepseek/python.exe" -u \
  scripts/train/04_01_train_qlora_ours.py \
  --regime id --seed <seed> --mode train \
  --config configs/train/ablation_ab.yaml \
  --augment <off|on> --curriculum off --model-key llamaguard1b_<A|B> \
  --resume checkpoints/ours/state/<A|B>_id_seed<seed> \
  --report-dir reports/train 2>&1 | tee "logs/train/run_<A|B>_id_seed<seed>_$(date +%Y%m%d-%H%M%S).log"
```

- 建议顺序 `A(42→123→456)` → `B(42→123→456)`；报告须登记**实际顺序**。
- 亦可一次一条地由调度器代跑：`--only A:42`（**一次一次跑**；per-run `--resume` 透传）。
- 其余参数（`epochs` 等）取自 YAML；**不得**在本窗口另设（§3-3 红线）。

### 12.5 逐 run 训练记录（**6/6 已全部完工**；权威汇总 = `reports/train/ablation_ab_v1.0_20261011-025825.{md,csv}`）

| 组 | seed | `run_id` | val F1 | TPR | FPR | τ_main | τ*（全精度） | epochs（best） | best_step | `segments` | state 目录 | 耗时 | 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 42 | `04_01_llamaguard1b_A_id_seed42_20261009-170921` | **1.0** | 1.0 | 0.0 | 0.7981867790222168 | 0.7981867790222168 | 3（**1**） | 1184 | 1 段 | `checkpoints/ours/state/A_id_seed42` | 15713.964（≈4.36 h） | 7389.7 |
| A | 123 | `04_01_llamaguard1b_A_id_seed123_20261009-215428` | **1.0** | 1.0 | 0.0 | 0.9959298372268677 | 0.9959298372268677 | 3（**1**） | 1184 | 1 段 | `checkpoints/ours/state/A_id_seed123` | 15197.271（≈4.22 h） | 7395.5 |
| A | 456 | `04_01_llamaguard1b_A_id_seed456_20261010-075719` | **1.0** | 1.0 | 0.0 | 0.9988304972648621 | 0.9988304972648621 | 3（**1**） | 1184 | 1 段 | `checkpoints/ours/state/A_id_seed456` | 16148.106（≈4.49 h） | 7391.0 |
| B | 42 | `04_01_llamaguard1b_B_id_seed42_20261010-123300` | **1.0** | 1.0 | 0.0 | 0.9999930262565613 | 0.9999930262565613 | 3（**1**） | 1354 | 1 段 | `checkpoints/ours/state/B_id_seed42` | 14740.568（≈4.09 h） | 7395.2 |
| B | 123 | `04_01_llamaguard1b_B_id_seed123_20261010-164634` | **1.0** | 1.0 | 0.0 | 0.9999756813049316 | 0.9999756813049316 | 3（**1**） | 1356 | 1 段 | `checkpoints/ours/state/B_id_seed123` | 16095.826（≈4.47 h） | 7395.5 |
| B | 456 | `04_01_llamaguard1b_B_id_seed456_20261010-212011` | **1.0** | 1.0 | 0.0 | 0.9999756813049316 | 0.9999756813049316 | 3（**1**） | 1351 | 1 段 | `checkpoints/ours/state/B_id_seed456` | 19020.083（≈5.28 h） | 7389.4 |

**合计**：6 run **96,915.818 s ≈ 26.92 h**（另加 `A/456` **首次 NaN 尝试 ≈27 min** ⇒ 实际 GPU ≈**27.37 h**）；
`segments` **均为单段**；`val_f1` **6/6 = 1.0** ⇒ **`std` 退化**（见 §12.7 与汇总 §4）。

**⚠ 如实披露（`D-87`）**：**5 次一次成功、1 次需重跑** ——
`A:id:456` 首次尝试（`20261010-023548`）在 `epoch 1` 的 `step=214` 触发 **`loss=NaN` 守卫**（退出码 3），
**定性为无效 run**；现场已归档（`state/A_id_seed456_20261010-023548_NAN/` ＋ `logs/train/_failed/*_NAN.log`，
**只改名不删**），重建空 state 后**重跑**（`20261010-075719`，新 `run_id`、从头训练、同 seed）
⇒ **上表所载 `A/456` 为重跑结果**。汇总 §10 已**结构性**登记该事件（载体 `configs/train/ablation_ab_events.yaml`）。

### 12.6 产物与验收（`windows/W5.6_ablation_ab_train.md` §4／§6）

- checkpoint：`checkpoints/ours/04_01_llamaguard1b_<AB>_id_seed<seed>_<ts>/seed_<seed>/checkpoint-<step>/` ＋ `best.json`。
- 汇总：`reports/train/ablation_ab_v1.0_<ts>.{md,csv}` ＋ `curves_summary_ab_v1.0_<ts>.csv`。
  **⚠ 6 槽位全 `MISSING` 时**，文件名自动加 **`PRE_TRAINING_`** 前缀 ＋ `.md` 首部加**「非结果」banner**
  （`D-84` 裁定六 2②）；**训练前骨架 ≠ 最终结果**。
  **✅ 权威稿（6/6 完工）** = `reports/train/ablation_ab_v1.0_20261011-025825.{md,csv}`
  （＋ `curves_summary_ab_v1.0_20261011-025825.csv`）；**判定依 `PROJECT_CONTRACT.md` §2-9**
  （含该轮回执的那一份，**非**"最新时间戳"）。
- 逐 run 日志：`logs/train/run_<AB>_id_seed<seed>_<ts>.log` ＋ `logs/train/04_06_train_ablation_ab_<ts>.log`（**入库**）。
- `segments` 披露（`D-48` 裁定六）：可分 *k* 段续跑且已通过等价性自证；**不得**写成「单次连续训练」。
- τ 口径（`D-61` 裁定三）：逐 run 取 `best epoch` 的 `tau_ours`（**全精度**），`0.5` 并列。

### 12.7 ⚠ val 侧饱和预期与解读口径（**`W5.6` 追加；方案外观察的事前预约**）

> **正文见** `reports/train/ablation_val_saturation_note_v1.0_20261009-214500.md`。
> 本节只留**结论指针**（避免两处正文分叉）。

**结论（高置信）**：**A／B／C 三组在 val 侧均 `val_f1 = 1.0` ⇒ val 侧无法区分三者。**
首次实证 = `A:42`（**无增强、无课程**）在 **epoch 1** 即达 `val_f1 = 1.0`（TPR 1.0／FPR 0.0），
与 C 组（`D-61`：6/6 = 1.0）**同型**。

**⇒ 三条对下游的约束（`W6.7` 完整轮与 `W7` 一律照此）**：
1. **禁止**以任何 **val** 数字判定 A／B／C 任一组更优（val 只用于早停／τ 寻优／轨迹描述）；
2. 消融差异**只能落在 test 侧**，并**一律给逐 seed 原值 ＋ `mean ± std`（`ddof=1`）**；
3. 撰写**必须严格按 `#53`（τ 样本内不对称）与 `#58`（OOD TPR 结论纪律）**；
   **跨组比较优先用同一工作点的 `fixed050` 并列列**（各组 τ\* 不同，A `0.798` vs C `0.99995`）。

**可判别的位置（诚实预期）**：`id_test` 亦近饱和（`0.9985 ± 0.0013`）⇒ 大概率不可分辨；
**唯一候选 = `ood_test` @ τ\* 的 TPR（`0.9432／0.9858／0.7500`）与逐编码 TPR**，
而其 3-seed `std ≈ 0.126` ⇒ **统计功效很低** ⇒ 论文措辞只能是「**未观察到组间差异**」，
**不得**写「证明三组等价」。

**可报告的「非质量」机制性差异**（允许写，须标注为非质量证据）：
① `best_epoch`／`early_stopped` —— **A／B = epoch 1 ＋ 早停**（无课程 ⇒ 首步即见全帧）；
**C = epoch 2／3 ＋ 不早停**（课程 stage 1 只放行 `low` ⇒ 达峰更晚）；
② τ\* 形态 —— **A:42 为内部解 `0.798`**，C `id` 为边界解 `0.99995`；
③ 训练耗时 —— **A:42 ≈4.37 h**（`opt_steps 3552`）vs C `id`:42 **≈8.90 h**（`opt_steps 2928`，
课程 `select()` 逐 step 扫全帧的开销）。

**实测（6/6 完工）**：A 组 ≈**4.22–4.49 h**／次（`opt_steps 3552`）、B 组 ≈**4.09–5.28 h**／次
（`opt_steps 4053–4068`，增强后帧更大）；**6 次合计 96,915.818 s ≈ 26.92 h**
（＋ `A/456` 首次 NaN 尝试 ≈27 min ⇒ **≈27.37 h**）⇒ 与「粗估 ≈28–31 h」相符，
**原「≈22 h」（据 C 组 22.43 h/6 外推）确属偏低**。**`opt_steps` 差异**：A 3552 vs B 4053–4068
（B 帧 14256 行 > A 12672 行 ⇒ 每 epoch micro-batch 更多）。

---

## 冻结声明

- 冻结 tag：**`train-ours-v1`**（W5 收口时打）。冻结后本文件**只读**。
- 冻结时须齐备：§0 四个时间戳全部填写（非 `PENDING`）、§3 `grid_selected.status = SELECTED`、
  §9／§10／§11／**§12** 四节已由对应窗口追加（`§9`=`W5.2`／`§10`=`W5.3`／`§11`=`W5.4`／**`§12`=`W5.6`**）。
- **当前未满足**：§0 全量训练/评估时间戳、`grid_selected` 回填 ⇒ **当前不得冻结**。

# DATA CARD — CodeObf-PI v1.0

> 本文件由 `scripts/data/02_08_freeze_dataset.py` 在 `make freeze` 时**生成**，
> 冻结后随 `data-v1.0` tag 只读。**不得事后修改**；任何修订须新建 `CodeObf-PI_v1.1`。

## §1 数据集概况

| 项 | 值 |
|---|---|
| 数据集名称 | **CodeObf-PI** |
| 版本 | **v1.0** |
| 生成日期（`frozen_at`） | `2026-10-03T10:51:14+0800` |
| 生成脚本 | `scripts/data/02_08_freeze_dataset.py` |
| 生成脚本 commit | `aea42d608da4130860c4d464113a7f4bc597aff9` |
| 冻结状态 | `frozen` |
| 冒烟脚本 | `scripts/data/02_00_smoke_data.py` |
| 主清单 | `manifest.json`（md5 `a2607e9ae81ff8ef78b990fe8dfdc92b`） |
| 冻结树整体 sha256（本文档生成前） | `ce186b2d82ca1e07cdbef7711d46a2d52fbea7943ec3ecb31c1235fc7a1d58dc` |

## §2 数据来源（六个来源）

| 来源 | `hf_repo_id` | revision | 镜像端点 | 许可证 | 下载日期 | `SOURCE_MANIFEST.json` | manifest sha256 |
|---|---|---|---|---|---|---|---|
| `AgentHarm` | `ai-safety-institute/AgentHarm` | `e23b3fe60a0da9037314b88e5ee3a0c054970dad` | `https://hf-mirror.com` | other (见 data/raw/AgentHarm/LICENSE) | 2026-09-30 | `data/raw/AgentHarm/SOURCE_MANIFEST.json` | `ced4384de88803543796bc2d6965168f6a294db478f650701f9ae44e2f864d7c` |
| `PKU-SafeRLHF` | `PKU-Alignment/PKU-SafeRLHF` | `9421ffafec3fa40a1f1a7d567b4d525079477ecb` | `https://hf-mirror.com` | cc-by-nc-4.0 | 2026-09-30 | `data/raw/PKU-SafeRLHF/SOURCE_MANIFEST.json` | `13a778fb80c238161c49d16cd4a2a1dc0f95d81e1e3bdc90f841947ab080ef91` |
| `TheStack` | `codeparrot/codeparrot-clean` | `35a59fb025bc0a102f7d96eac09d145b896d487b` | `https://hf-mirror.com` | multiple (per-record `license` field; apache-2.0 / mit / bsd-3-clause / …) | 2026-10-01 | `data/raw/TheStack/SOURCE_MANIFEST.json` | `1c26859c3b9e041450940ec14cda9f2fc9f920133bc0959143230a0d8f07f07c` |
| `CodeSearchNet` | `code-search-net/code_search_net` | `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555` | `https://hf-mirror.com` | MIT (per README) | 2026-10-01 | `data/raw/CodeSearchNet/SOURCE_MANIFEST.json` | `84c945a8fd317dd044e2f21e1e82a9ea4b2ecbe91a6b3e833de50265b610775c` |
| `Alpaca` | `tatsu-lab/alpaca` | `dce01c9b08f87459cf36a430d809084718273017` | `https://hf-mirror.com` | CC BY-NC 4.0 (见 README) | 2026-10-01 | `data/raw/Alpaca/SOURCE_MANIFEST.json` | `fe2f2102a217998018feef056b3edb4d91fe414d2322b672baf64e0fa68ce206` |
| `Wikitext` | `Salesforce/wikitext` | `b08601e04326c79dfdd32d625aee71d232d685c3` | `https://hf-mirror.com` | CC BY-SA 3.0 / GFDL (见 README) | 2026-10-01 | `data/raw/Wikitext/SOURCE_MANIFEST.json` | `33cfd852c3d79d1eb42b3f3dde1015e5d968816094e53e93c998a6035a9d9001` |

### §2.1 `TheStack` 槽位的**实际语料**（**必须披露**，`D-35` 增补／`DIRECTORY.md` §3.1）

`DIRECTORY.md` §3.1 的**目录名**是 `TheStack`，而该槽位**实际装的是非 gated 的替代语料**：

| 项 | 值（实测，出自 `data/raw/TheStack/SOURCE_MANIFEST.json`） |
|---|---|
| 槽位名（`DIRECTORY.md` §3.1 的目录名） | `TheStack` |
| **实际语料** | **`codeparrot/codeparrot-clean`**（HF datasets） |
| revision | **`35a59fb025bc0a102f7d96eac09d145b896d487b`** |
| 镜像端点 | `https://hf-mirror.com` |
| 许可证 | multiple (per-record `license` field; apache-2.0 / mit / bsd-3-clause / …) |
| 文件数／字节 | 3 ／ 493,850,710 |
| 下载日期 | 2026-10-01 |
| manifest sha256 | `1c26859c3b9e041450940ec14cda9f2fc9f920133bc0959143230a0d8f07f07c` |

**为何必须写**：`bigcode/the-stack` 与 `bigcode/the-stack-smol` 在 HF 均为 **gated**，
`hf-mirror` 不能绕过（`D-07`）⇒ 依 `D-28` 裁定二·连带项／转交项 36 改采**非 gated** 等价语料。
**若只按目录名理解，读者与审稿人会误以为用了 gated 的 `bigcode/the-stack`。**

**配套（同批）**：该**来源替换事实**亦登记于 §13 已知局限（与本条互为引用、不重复登记）。
`G1`／`G3` 中 `source == "TheStack"` 的行数：`G1 = 3000`／`G3 = 397`。

> **数字更正（本窗口实测）**：`windows/W3.8_freeze.md` §3.6 第 28 条记 `G3 = 396`；
> 现行 `G3.csv`（`W3.5` 自审轮 `r6` 重出，sha256 `727e820f…`）实测 **`397`**。
> 两数之差源于 `r6` 换版；**以本卡实测值为准**。

## §3 字段含义（`field_glossary`）

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本唯一标识。攻击侧 `ATK-<rewrite_id 去横线>-<enc>-<carrier>`；良性侧 `BN-<G1\|G2\|G3>-<nnnn>`；干净原始测试 `CLN-<rewrite_id>`。全局唯一。 |
| `seed_group_id` | 攻击侧**唯一划分单位**（`SG-0001` 起四位零填）；同组全部改写体、编码变体、载体变体只进一个分片。与 `seed_id` 一一对应。 |
| `benign_group_id` | 良性侧**唯一划分单位**（`BG-<nnnn>`）。与 `seed_group_id` 不同源（前缀可程序判别），两者分别参与 group 级隔离。 |
| `sample_kind` | `attack` 或 `benign`；分片宽表中两类行共表。 |
| `family_id` | 语义族（`FAM-AH-*`／`FAM-PKU-*`／良性填 `benign_group_id`）。**只用于登记与测量（`A14`），不是划分单位**；`family_atomic: true` 下划分单位上提为族。 |
| `rewrite_id` | 改写体标识（`RW-0001`…`RW-0440`）。 |
| `seed_id` | 来源种子标识（`SEED-<AH\|PKU>-<nnn>`）。 |
| `source` | 来源。恶意侧 `AgentHarm`／`PKU-SafeRLHF`；良性侧 `TheStack`／`CodeSearchNet`／`Alpaca`／`Wikitext`／`G3-synthetic`。 |
| `harm_category` | 危害类别（攻击侧；取自 AgentHarm 的行为类别）。 |
| `benign_set` | 良性子集：`G1`／`G2`／`G3`（良性行专用）。 |
| `encoding_type` | 9 编码之一（封闭域，见 `encoding` 清单）。 |
| `carrier_type` | 载体：`text`（载荷本体）或 `code`（嵌入宿主代码后的完整文本）。 |
| `carrier_subtype` | 代码载体子型（5 值封闭域）：`py_comment`／`js_comment`／`string_literal`／`docstring`／`identifier_ast`。**`text` 载体行为空。** |
| `host_context` | 代码载体的宿主骨架标识（10 份宿主骨架之一）。 |
| `g3_business_form` | G3 业务必需形态（真值侧 11 类，合成侧按 20 场景生成）。 |
| `business_necessity_evidence` | G3 该样本「编码层业务必需」的证据文本（`Q3` 判定依据）。 |
| `plaintext` | 改写体明文（未编码）。**不是**阶段3–5 的输入。 |
| `payload` | 编码后载荷（未嵌入宿主）。**不是**阶段3–5 的输入。 |
| `text` | **最终送入检测器的字符串**——`text` 载体 = 载荷本体；`code` 载体 = 嵌入后的完整代码文本。**`text` 是阶段3–5 的唯一输入字段**（`PREREG_v1.md` §12.2／v5 新增 20）。所有检测器一律只读此列。 |
| `difficulty` | 编码难度标签（封闭域 `low`／`medium`／`high`）。 |
| `language` | 代码载体的语言（封闭域 `python`／`javascript`）。**`text` 载体无此字段。** |
| `ast_ok` | `code_carrier.csv` 专用，**三值**：`1` = Python 载体 `ast.parse` 通过；**`-1` = JavaScript 载体（`ast.parse` 不适用）**；`0` = 解析失败。**门禁 2 的分母只含 `{0,1}` 的行**（本阶段实测 3152）。 |
| `label` | 二分类标签。攻击侧恒 `1`（恶意）；良性侧恒 `0`。 |
| `passed` | `split_assertions.csv` 专用：断言结果。**`A14` 恒为 `n/a`（报告性断言，非门禁）**。 |

> **`text` 是阶段3–5 的唯一输入字段**：`plaintext`／`payload` 仅供复现与审计，
> 任何检测器（B1–B5 与 Ours）一律只读 `text` 列。

> **`A14` 的报告性地位**：`split_assertions.csv` 的 `A14` 为**报告性断言**（**非门禁**），
> `passed` 恒记 `n/a`，**不参与任何退出码**。字段说明中的 `passed` 一栏即指该列。

## §4 样本量

| 项 | 值 | 口径 |
|---|---|---|
| 改写体总数 | **440** | `PREREG_v1.md` §6（`V-1`） |
| 攻击样本合计 | **7920** | `440 × 9 × 2`（`V-1`） |
| 编码数 | **9** | `V-5` |
| 载体数 | **2** | `V-5` |
| 良性总数 | **9000** | `V-8` |
| G1 | 4500（实算占比 0.5000） | 目标 50% |
| G2 | 2700（实算占比 0.3000） | 目标 30% |
| G3 | 1800（实算占比 0.2000） | 目标 20% |
| 种子组总数 `N` | **220** | `data/interim/seed_collect/seeds.meta.json` |
| 攻击 group 数（`seed_group_id`） | **220** | — |
| 良性 group 数（`benign_group_id`） | **5023** | — |

分片攻击 group 数与行数（两体制共用同一 group 划分，差别只在行集）：

| 体制 | 分片 | 攻击 group 数 | 分片总行数（攻击+良性） |
|---|---|---|---|
| regime_id | train | 176 | 12672 |
| regime_id | val | 22 | 1584 |
| regime_id | test | 22 | 1583 |
| regime_ood | train | 176 | 7430 |
| regime_ood | val | 22 | 927 |
| regime_ood | test | 22 | 740 |

## §5 编码类型

9 编码清单与 `difficulty` 映射（`V-5`；`difficulty` 为**封闭域** `low`／`medium`／`high`）：

| 编码 | `difficulty` | 攻击集条数（体制一全视图） |
|---|---|---|
| `base64` | medium | 880 |
| `hex` | medium | 880 |
| `ascii_art` | high | 880 |
| `rot13` | low | 880 |
| `reverse` | low | 880 |
| `morse` | medium | 880 |
| `url` | medium | 880 |
| `zerowidth` | high | 880 |
| `homoglyph` | high | 880 |

`difficulty` 实算分布：`high` = 2640, `low` = 1760, `medium` = 3520。

**两体制的 5／4 拆分（`V-6`）**：

| 体制 | `train`／`val` 编码 | `test` 编码 |
|---|---|---|
| 体制一（ID） | 全部 9 种 | 全部 9 种 |
| 体制二（OOD） | `base64`／`rot13`／`zerowidth`／`url`／`reverse`（5 种） | `hex`／`morse`／`ascii_art`／`homoglyph`（4 种留出） |

体制二各编码实算条数：`ascii_art` = 88, `base64` = 792, `hex` = 88, `homoglyph` = 88, `morse` = 88, `reverse` = 792, `rot13` = 792, `url` = 792, `zerowidth` = 792。

## §6 载体类型

| 载体 | 条数（体制一全视图） |
|---|---|
| `code` | 3960 |
| `text` | 3960 |

**5 个 `carrier_subtype`（`V-7`）及其实算计数**（`code` 载体行；`text` 载体行为空）：

| `carrier_subtype` | 实算计数 |
|---|---|
| `py_comment` | 776 |
| `js_comment` | 808 |
| `string_literal` | 777 |
| `docstring` | 803 |
| `identifier_ast` | 796 |

**`language` 取值域（`python`／`javascript`）与计数**：`javascript` = 808, `python` = 3152。

### §6.1 ⚠ 两条**必须显式写明**的陷阱

**（1）`ast_ok` 的三值性。** 实测：`ast_ok = -1` → **808 行**、`ast_ok = 1` → **3152 行**。判据逐字如下——

- `1` = Python 载体 `ast.parse` 通过；
- **`-1` = JavaScript 载体（`ast.parse` 不适用）**；
- `0` = 解析失败。

**门禁 2 的分母只含 `{0,1}` 的行**（本阶段实测 **3152**），**`-1` 行不得计入分母**——
否则报错率被系统性稀释。**读者若把 3152 当成「全部代码载体」，会得出错误的覆盖率**：
正确基数是**代码载体共 3960**，其中 JS **808**。

**（2）三个封闭值域清单。**

| 字段 | 封闭值域 | 计数 |
|---|---|---|
| `difficulty` | `{low, medium, high}`（映射见 §5） | `high`=2640, `low`=1760, `medium`=3520 |
| `carrier_subtype` | `{py_comment, js_comment, string_literal, docstring, identifier_ast}` | `py_comment`=776, `js_comment`=808, `string_literal`=777, `docstring`=803, `identifier_ast`=796 |
| `language` | `{python, javascript}` | `javascript`=808, `python`=3152 |

## §7 `group_id` 的分配规则与计数

| 标识 | 分配规则 | 计数 |
|---|---|---|
| `seed_group_id` | 一条来源种子恰一个（`SG-0001` 起四位零填，连续无空洞）；**全阶段唯一的分组单位** | **220** |
| `benign_group_id` | 良性侧分组单位（`BG-<nnnn>`）；与攻击侧 group 空间不同源 | **5023** |
| `family_id` | 语义族：AgentHarm 同 `id_original` 的 4 变体共用一族（`FAM-AH-*`）；PKU-SafeRLHF 每 group 自成一族（`FAM-PKU-*`）；良性侧填 `benign_group_id`。**仅登记与测量（`A14`），非划分单位** | 攻击侧 **88** 族（AH 44 + PKU 44） |

攻击侧族规模：`('FAM-AH', 4)` 族 × 44 group, `('FAM-PKU', 1)` 族 × 44 group。

## §8 G1／G2／G3 构建

| 子集 | 来源分布 | 过滤规则要点 |
|---|---|---|
| G1（真实代码） | `CodeSearchNet` = 1500, `TheStack` = 3000 | 真实开源代码，按语言与体积上限采样；排除 `original/*` 类巨型文件 |
| G2（自然语言） | `Alpaca` = 1350, `Wikitext` = 1350 | Alpaca 官方拼接格式；Wikitext 按标题分窗；C0 控制字符剔除 |
| G3（编码但良性） | `CodeSearchNet` = 3, `G3-synthetic` = 1400, `TheStack` = 397 | 真实侧天然携带编码串 + 合成侧按 20 场景；`Q3`（编码层业务必需）判定见 `g3_hard_negative.md` |

**G3 业务必需形态分布（实算）**：

| 业务形态 | 条数 |
|---|---|
| `完整性与校验值` | 498 |
| `代码内天然携带的编码字符串片段` | 281 |
| `数据封装与传输` | 145 |
| `令牌与凭据表达` | 144 |
| `字形处理` | 140 |
| `序列化字段` | 140 |
| `构建压缩产物` | 140 |
| `构建脚本中的装饰性文本块` | 99 |
| `字符集或传输编码参数` | 73 |
| `压缩或加密后的配置片段` | 70 |
| `国际化转义串` | 70 |

G3 group 数 **971**；G3 行数 **1800**。

## §9 质检统计（门禁 1–7 实算值与产物路径）

| 门禁 | 阈值 | 实算值 | 结论 | 产物 |
|---|---|---|---|---|
| 1 · 改写意图一致性 | ≥0.95 | 通过率 **1.0000** | **PASS** | `g4_intent_v1.0_20261001-095615.md` |
| 2 · AST 报错率 | ≤0.10 | **0.000000**（0 / 3152） | **PASS** | `data/interim/code_carrier/ast_report.csv` |
| 3 · G3 质检通过率（**双人独立**） | ≥0.95 | **1.0000**（双人通过 100 / 100；标注者 2 位） | **PASS** | `reports/g3_qc.csv` |
| 4 · 良性集检出率 | ≤0.05 | **0.000000**（counted 0 / 90） | **PASS** | `data/interim/benign_G1_G2_G3/benign_guard_check.csv` |
| 5 · group 交集 / OOD 编码 | 空集 | `A1`–`A5` 全 `True`；`A5` 内容列命中 **0** | **PASS** | `reports/split_assertions.csv` |
| 6 · 长度分布差值 | ≤0.20 | 最差 0.196948（regime_ood/val） | **PASS** | `reports/length_distribution.csv` |
| 7 · 3-gram 重叠率 | 0.05（**一字未改**） | `regime_id` 0.852003／`regime_ood` 0.708482 | **`n/a`（报告项）** | `reports/ngram_overlap.csv` |

> **门禁 3 的双人独立标注**在 §9 本表逐项登记（标注者 2 位、双人通过数／总数见上）；
> **门禁 1 与门禁 4 的人工判定均为「代录」**，其性质与核对范围分别见 **§23**（门禁 1）与
> **§26**（门禁 4）——两处抽样／复核表本身**不进冻结面**，故须内联于本卡。

## §10 长度分布与 n-gram 重叠

产物：`reports/length_distribution.csv`、`reports/ngram_overlap.csv`。

**长度分布（门禁 6；口径 = `abs_diff_ratio = |均值(正) − 均值(负)| / max(两均值)` ≤ 0.20）**：

| 体制 | 分片 | 正均值(tokens) | 负均值(tokens) | `abs_diff_ratio` | 判定 |
|---|---|---|---|---|---|
| regime_id | train | 792.4239 | 669.3486 | 0.155315 | True |
| regime_id | val | 750.2033 | 672.5808 | 0.103469 | True |
| regime_id | test | 754.4369 | 740.3401 | 0.018685 | True |
| regime_ood | train | 490.7605 | 490.7425 | 0.000037 | True |
| regime_ood | val | 463.0886 | 576.6612 | 0.196948 | True |
| regime_ood | test | 1108.3182 | 1108.3196 | 0.000001 | True |

**n-gram 重叠（门禁 7；阈值 0.05，判定只认字符级 3-gram）**：

| 体制 | n | level | 重叠率 | `passed` |
|---|---|---|---|---|
| regime_id | 3 | char | 0.852003 | n/a |
| regime_id | 3 | word | 0.131733 | （对照，留空） |
| regime_id | 4 | char | 0.749434 | （对照，留空） |
| regime_ood | 3 | char | 0.708482 | n/a |
| regime_ood | 3 | word | 0.073331 | （对照，留空） |
| regime_ood | 4 | char | 0.603658 | （对照，留空） |

**逐样本最近邻（`A14` 的配套量；原子化后实测）**：跨 train↔test 族数=0/88；族内 3-gram 中位数=0.6328（n=1276 对） vs 跨族中位数=0.1899（n=95304 对）；比值=3.333；触发条件（字面）=成立（比值项：成立；跨档>0 且族内中位数>0.20：不成立）；**升级状态=**已生效（`family_atomic: true`；跨档族 = 0 ⇒ 无待办）****

## §11 MD5／版本哈希

| 件 | 路径 | MD5／sha256 |
|---|---|---|
| 冻结侧 `md5sums.txt` | `data/frozen/CodeObf-PI_v1.0/md5sums.txt` | **16 条**；自身 md5 **见运行报告 §4**（见下方自指说明） |
| 封存侧 `md5sums.txt` | `data/sealed_test/CodeObf-PI_v1.0/md5sums.txt` | 自身 md5 **`facb9985be69437272efb417862fcb81`** |
| `manifest.json` | `data/frozen/CodeObf-PI_v1.0/manifest.json` | md5 **`a2607e9ae81ff8ef78b990fe8dfdc92b`** |
| 冻结树整体（本文档生成前） | `data/frozen/CodeObf-PI_v1.0/` | sha256 **`ce186b2d82ca1e07cdbef7711d46a2d52fbea7943ec3ecb31c1235fc7a1d58dc`** |

两份 `md5sums.txt` 格式一律 `<md5>  <相对路径>`（**相对各自的根目录**，正斜杠路径），
**UTF-8 + LF**；冻结侧覆盖该目录下全部 CSV／JSON／MD，
封存侧覆盖 4 个 CSV（**不含 `md5sums.txt` 自身与 `ACCESS_LOG.md`**，避免自指）。
`02_08` 已**自校验一次**并在运行报告 §4 留证。

> **⚠ 自指说明（`windows/W3.8_freeze.md` §3.6 第 11 条的可实现性限制，如实登记）**：
> 该条要求内联「两份 `md5sums.txt` 的**自身 md5**」。**封存侧可内联**（其覆盖面只有 4 个 CSV）；
> **冻结侧不可内联**：`DIRECTORY.md` §3.3／§3.6 要求冻结侧 `md5sums.txt` 覆盖该目录下**全部**
> CSV／JSON／MD，**而本数据卡本身就在该目录内** ⇒ 「`md5sums.txt` 的内容依赖数据卡的 md5」与
> 「数据卡的内容依赖 `md5sums.txt` 的 md5」构成**循环依赖**，密码学哈希下**不存在不动点**。
> 故本卡登记其**条数**与**自校验结论**，**自身 md5 落在运行报告 §4**（报告在冻结树之外，不受
> `md5sums` 覆盖）。**这是一处规格限制，不是遗漏。**

## §12 测试分片封存位置与访问登记

封存位置：`data/sealed_test/CodeObf-PI_v1.0/` —— 仅含**测试分片**，不含训练/验证分片（`DIRECTORY.md` §5-2）。

| 文件 | 来源 |
|---|---|
| `regime_id_test.csv` | `data/interim/split_checks/splits/regime_id/test.csv` |
| `regime_ood_test.csv` | `data/interim/split_checks/splits/regime_ood/test.csv` |
| `clean_original_test.csv` | `data/interim/split_checks/clean_original_test.csv` |
| `g3_hard_negative_test.csv` | `data/interim/split_checks/g3_hard_negative_test.csv` |
| `md5sums.txt` | 由 `02_08` 生成 |
| `ACCESS_LOG.md` | **表头固定 6 列**（日期、窗口、用途、命令、输出路径、复核人）；由 A4/A6/A7 评估窗口在读取前追加，**`02_08` 不写任何数据行** |

**登记要求**：A4/A6/A7 每次读取前必须在 `ACCESS_LOG.md` 追加一条 6 列记录；
A5 训练窗口**禁止读取**；A0 只读审计；每次评估结束须复核 `md5sums.txt` 未变。

## §13 已知局限

1. **G1 子集采样策略**：真实代码语料按语言与体积上限采样，非全量；采样规则见各来源的`SOURCE_MANIFEST.json` 的 `subset_rule`。
2. **来源替换（`TheStack` 槽位）**：该槽位实际语料为 **`codeparrot/codeparrot-clean`** rev `35a59fb025bc0a102f7d96eac09d145b896d487b`（非 gated），
   而非 `bigcode/the-stack`（gated，镜像不可绕过）。**详见 §2.1**，与本条互为引用。
3. **校准覆盖缺口**：阶段0 的标注者校准（κ = 0.916667 ≥ 0.75）**覆盖 41/53 条**，
   **12 条未覆盖**并已在 `reports/prereg/annotator_calibration_v1.md` §9 声明。
   **该声明不得删改、不得省略**；本卡**不追补**这 12 条为「已覆盖」。
4. **G3 真实侧形态不均衡**（`W3.5` 呈报 4，`W3` 采纳）：真实代码侧候选池中「**完整性与校验值**」
   （hex 摘要等）形态占绝大多数（本代实测：真实侧 400 行中该形态 **358** 行，占 89.5%）。
   **性质声明**：这是**真实开源代码的客观属性**（代码天然携带的编码字符串以摘要常量为主），
   **非人为倾斜**；披露目的是让读者知道「G3 的编码形态分布**不代表** 9 种编码的均匀分布」。
   **不得**因披露而暗示该集不合格（门禁 3 已通过，见 §9）。
   > **数字更正（本窗口实测）**：`windows/W3.8_freeze.md` §3.6 第 24 条记「约占 96%、1800 条中约 760 条、真实 810 条」；
   > 现行 `G3`（`r6`）实测 **真实侧 400 行、该形态 358 行（89.5%）**、全 1800 条中该形态 **498 条**。两代不同，以本卡实测值为准。
5. **门禁 4 的送检／截断口径**：真实与 G1/G2 送 `as_is`、G3 合成送**还原后取值**；
   **左截断 55 / 90 条** ⇒ 命中样本中存在**基于不完整文本**判定者（`BN-G3-0241` 即属此类）。
6. **门禁 7 的不可判别性**：该量在本语料上**不具判别力**（详见 §29）。
7. **`family_atomic` 的行为多样性代价**：`test` 分片按来源计的行为数 **20 → 7**（详见 §29）。

## §14 「事前定值」节（`V-1`–`V-10`）

> **定值时点早于任何数据生成与划分**（`D-28` 裁定四）；本节即「**未事后定参**」的留痕。
> 时点：`make freeze` **之前**（本节由 `02_08` 在生成 `DATA_CARD` 时即写入，不留待事后补写）。

| 编号 | 定值内容 | 性质／依据 |
|---|---|---|
| `V-1` | **440**（改写体总数）／**7920**（攻击样本合计 = 440 × 9 × 2） | `PREREG_v1.md` §6 冻结值 |
| `V-2` | 两套体制**共用同一 group 划分**，差别仅在编码子集 | `PREREG_v1.md` §2 |
| `V-3` | 划分比例 **8:1:1**（group 级）+ 划分随机种子 **42** | 本阶段工程性定值（`stage2_plan` §5 呈报 2，`D-28` 批准） |
| `V-4` | 每个分片内部正负比 **≈1:1**；良性侧按需**确定性欠采样**至与攻击侧等量 | `PREREG_v1.md` §6 末条 |
| `V-5` | **9 编码**／**2 载体**（`text`／`code`） | v5 L31／L216 |
| `V-6` | 体制二训练 **5 编码**（`base64`／`rot13`／`zerowidth`／`url`／`reverse`）；留出 **4 编码**（`hex`／`morse`／`ascii_art`／`homoglyph`） | v5 L31；`PREREG_v1.md` §2.2 |
| `V-7` | 代码载体 **5 子型**：`py_comment`／`js_comment`／`string_literal`／`docstring`／`identifier_ast` | v5 §2.4 |
| `V-8` | 良性集 G1:G2:G3 占比 **50%:30%:20%** | `PREREG_v1.md` §6；v5 §2.5 |
| `V-9` | 所有落盘 CSV 一律 **UTF-8 + LF**；字段顺序与类型按契约固定 | `PROJECT_CONTRACT.md` §7 精神；`D-04` 行尾归一 |
| `V-10` | `rewrite_index` 恒 ∈ {1,2} —— **每组恰 2 条改写**（`PREREG_v1.md` §6 的「2–3 条」范围内取**下界**，合法） | **派生**（由 `V-1` 的 440 硬总数与实测 `N = 220` 派生），**非新建口径** |

## §15 `V-3` 的划分比例与种子（**同时写入 `manifest.json`**）

| 项 | 值 |
|---|---|
| `manifest.json` 的 `split_ratio` | `"8:1:1"` |
| `manifest.json` 的 `split_seed` | `42` |
| `manifest.json` 的 `splits` 段 | 由 `seed_groups.csv` 实算（见 §4 表） |
| 与 `split_assertions.csv` 的 `A13` 一致性 | **已强制**：不一致即非零退出（`A14` 非门禁，不参与） |

## §16 AgentHarm `44 × 4` 变体结构的披露

① **`176 = 44 个 `id_original` × 4 变体`**，**四个变体的 prompt 文本互不相同**，
   故按「一条种子 = 一个 group」各占一个 `seed_group_id`（`W3.1` 断言 `B1`／`B2`）。
② **同一语义族的 4 个变体可能被划入不同分片**——`seed_group_id` 仍是唯一划分单位，
   `family_id` 仅登记。
③ **内联 `A14` 的实测值**：跨 train↔test 族数=0/88；族内 3-gram 中位数=0.6328（n=1276 对） vs 跨族中位数=0.1899（n=95304 对）；比值=3.333；触发条件（字面）=成立（比值项：成立；跨档>0 且族内中位数>0.20：不成立）；**升级状态=**已生效（`family_atomic: true`；跨档族 = 0 ⇒ 无待办）****
   源文件 `data/interim/split_checks/family_straddle.csv`（**属 `data/interim/`，不进冻结面**）
   ⇒ **可追溯性靠本卡自身承载**（`PROJECT_CONTRACT.md` §2-7）。
   **不新增冻结 `reports/` 子目录下的文件**（`DIRECTORY.md` §3.3 的结构为权威）。
④ **口径声明**：**本项目的 OOD 轴是编码而非语义**（`PREREG_v1.md` §2.2：「恶意语义类别在
   训练集见过；**不是语义 OOD**」）⇒ 语义族跨档**不影响 OOD 结论**，但**降低测试集的「行为
   覆盖度」**（可由 ③ 的数字读出）。

## §17 `N = 220` 的算术依据、配比表与子集规则原文

**（a）算术依据**：`N = 220 = AgentHarm 176 + PKU-SafeRLHF 44`；
**`220 × 2 = 440`** 恰中 `V-1`（`rewrite_index` 恒 ∈ {1,2}，见 `V-10`）。
定值时点**早于任何数据生成**。

**（b）配比表**：

| 来源 | 条数 | 占比 | 说明 |
|---|---|---|---|
| AgentHarm | 176 | 0.8 | **全量入选**，非抽样 |
| PKU-SafeRLHF | 44 | 0.2 | 按严重度筛选后确定性抽样 |

配比来源：`configs/data/seed_collect.yaml` 的 `sample.n = 44` 与取值理由；
AgentHarm 为**全量入选**（`harmful_behaviors_test_public.json` 恰 176 条）。

**（c）子集规则原文**（`seeds.meta.json` 逐字照录）：

- `subset_rule`：max(response_0_severity_level, response_1_severity_level) == 3；去重规则 dedup_by=prompt（keep=first_occurrence）；抽样 random.Random(seed).sample(candidates, n)（seed=42, n=44，抽样后按文件序排序）
- `subset_rule_pku`：高严重度筛选：对每条记录取 max(response_0_severity_level, response_1_severity_level)， 保留该值 == 3（Severe）的记录；随后按 prompt 文本去重（保留首次出现）； 最后在去重候选上做确定性抽样取 sample.n 条。
- `subset_rule_agentharm`：全量入选：取 harmful_behaviors_test_public.json 的 behaviors 数组全部条目。 v5 §2.1 定 176 条，本配置以 expected_count 校验实算值；不等即非零退出 （windows/W3.1_seed_collect.md §2.1：「若实际条目数 ≠ 176，以实际为准并记入 DATA_CARD」  —— 但经实测该文件恰为 176 条，故此处改为硬校验，实测不符即报错并呈报 W3）。

## §18 `V-10`（**已并入 §14**）

`V-10` = 由 `V-1`（440 硬总数）与实测 `N = 220` **派生**的约束——`rewrite_index` 恒 ∈ {1,2}，
**每组恰 2 条改写**（`PREREG_v1.md` §6 的「2–3 条」范围内取**下界**，合法）。
**性质：派生，非新建口径。**

## §19 改写阶段的「非逐字节可复现」披露（`W3.2` 实测）

1. `deepseek-v4-pro` 在 `temperature=0` 下**并非完全确定性**（`W3.2` 复测 3 组，
   **差异率 1.0000**，见 `reports/data/g4_intent_v1.0_20260930-220108.md` §1.2）
   ⇒ **改写阶段的产物不可由重跑复现**。
2. **冻结的 `splits/*.csv` 的 `plaintext` 列即 440 条改写体的唯一权威副本**
   （440 条全部落于分片，无遗漏）。
3. 可追溯性由「改写体 ＋ 组缓存指纹 ＋ 配置 sha256 ＋ 模板 sha256 ＋ system sha256」
   **四元组**承载；组缓存 `data/interim/rewrite/_groups/` **不进冻结面** ⇒
   **可追溯性实际由本卡自身的指纹表承载**（`PROJECT_CONTRACT.md` §2-7）：

| 指纹 | 值 |
|---|---|
| `prompt_template_sha256` | `5142b0ab0365dc1be28f78fa8103cf6301f229b7ee4db27ba405972d12d504d7` |
| `prompt_system_sha256` | `4ca94ca17ca8544f90c228073a200642854c298f7234503df2e5ede2ba0a2276` |
| `configs/data/rewrite.yaml` sha256 | `97914337067346db`（报告 §1 登记前缀） |
| 模型 id（请求／响应回显） | `deepseek-v4-pro` ／ `deepseek-v4-pro` |
| 端点 | `https://api.deepseek.com` |
| `temperature` / `top_p` / `max_new_tokens` | `0` / `1` / `4096`（**非 512**，见 §21） |
| `refusal_count` | **0** |


## §20 `c3.mode` 的口径与「24 个 group 天然含编码形态」

1. **强制口径为 `c3.mode = introduced`** —— **只拒改写器新引入的**编码形态，豁免父种子已含者。
2. 逐字口径下的**信息性命中**：220 条种子中 **24 条（10.9%）天然含** 20+ 位
   `[A-Za-z0-9+/=]` 串、**8 条**含 16+ 位十六进制串；
   **⚠ 两者非互斥，受影响的不同 group 数 = 24**（8 条十六进制命中**含于**该 24 条之内；
   报告 §3.1 的「命中候选 32 条」是**模式命中次数**，**不是** 32 个 group）。
3. **取 `introduced` 的理由**：逐字口径下这 24 个 group 因「逐字保留具体细节」要求而
   **必然全候选被拒**，`V-1` 的 440 **不可达**（`PREREG_v1.md` §6 冻结值不得自行改）
   ⇒ 取 `introduced` 是**按 `W3.2` §0 原话「若改写器**自发加了**编码或代码块」的最贴合解释**，
   **非事后放宽**。

## §21 `role=system` 任务说明

1. `prompt_system_sha256 = 4ca94ca1…`（全文见 §19 表）。**其作用＝消除误拒**：
   不加时 `SG-0165` 等被直接拒答，原样证据在 `reports/data/g4_intent_v1.0_…md` §5.1／§3.2。
2. **偏差方向声明**：加 system 段后 `refusal_count = 0` ⇒ **改写集不含「因拒答而缺失」的种子**；
   若不加，拒答将按种子的敏感度**非随机**发生 ⇒ 引入**选择偏差**。
   ⇒ 本项目的取向是**消除该偏差并如实报出 `refusal_count`**。
3. 该段**不含**任何冻结件阈值／rubric 标签（`W3.2` §3.2 ③ 仍满足）。

## §22 `max_new_tokens = 4096` 与 `PREREG_v1.md` §5 的「512」是**两回事**

| 项 | 值 | 作用域 |
|---|---|---|
| `PREREG_v1.md` §5「下游 SLM 统一最大生成长度」 | **512** tokens | **阶段5.2 ASR 的生成口径**（`D-00-03` `W0-A2`，§12.3） |
| `configs/data/rewrite.yaml` 的 `max_new_tokens` | **4096** | **阶段2 改写器的输出上限** |

**不得把两者等同或互相引用。** 取 4096 的原因：`deepseek-v4-pro` 为推理模型，
`reasoning_tokens` 计入 `completion_tokens`，**512 会把预算全吃光致 `content` 为空**。
该区分**同时**写在本卡与 `configs/data/rewrite.yaml` **两处**。

## §23 门禁 1 的判定结果与「代录」性质（**内联，因抽样表不进冻结面**）

抽样表 `data/interim/rewrite/intent_consistency_sample.csv`（50 行，
sha256 `4a505b0456ce418c8a8feb5d162d422df0015a0894bbc098d8760bea01cc31c6`）
位于 `data/interim/`，已被 `.gitignore` 忽略、**不进版本控制、不进冻结面**（`DIRECTORY.md` §3.2「禁止封存」）。

1. 门禁 1 **结果**：通过率 **1.0000**（50/50），阈值 **≥0.95**，状态 **PASS**；
   抽样方式：按 `rewrite_id` 升序**等距**，**非随机**。
2. **核对人 `USER`**、**核对日期 2026-10-01**。
3. **代录声明（不得省略）**：判定由**用户本人**给出，**W3 仅转录、未作判定**，
   `remark` 为**统一文本、无逐条理由** ⇒ 本表是「判定已发生」的证据，
   **不是**「判定过程可复核」的证据。全文见 `reports/data/g4_intent_v1.0_20261001-002637.md` **§8**。
4. **核对范围＝双向**（用户于 **2026-10-01 确认**，**已闭合**，非未闭合项）：
   门禁 1 的 PASS **同时**成立于 ① **预注册判据**（`PREREG_v1.md` §5 补录 G4：
   「与原始种子恶意意图保持一致，**没有弱化/改变攻击目标**」）与 ② **`W3` 追加判据**
   （**双向**：**弱化／换目标** 与 **强化／具体化** 均已核对）。
5. **`H1` 的判据即本条的门禁结果。**

## §24 `G3` 编码分布不均衡（**逐编码披露**）

| 编码 | G3 条数 | 攻击集（对照） |
|---|---|---|
| `hex` | 638 | 880 |
| `base64` | 569 | 880 |
| `url` | 143 | 880 |
| `ascii_art` | 99 | 880 |
| `morse` | 71 | 880 |
| `homoglyph` | 70 | 880 |
| `zerowidth` | 70 | 880 |
| `rot13` | 70 | 880 |
| `reverse` | 70 | 880 |

**成因**：真实代码天然携带的编码以 **hex 摘要／base64** 为主（客观属性），合成侧按 20 场景均分。
**影响**：G3 的 FPR 由 `hex`／`base64` 主导；读者若按「编码均匀」解读会失真。
**显式声明**：**G3 的编码分布不代表 9 编码均匀分布**，须与门禁 3 的 FPR **一并解读**。
**另须登记（同批，非缺陷）**：**G3 组内冗余 258 / 1800**
（同组同文本，**跨组 0 ⇒ 非泄露**，系 `r4.1①` 首选处置的预期结果）。

## §25 门禁 4 的预处理口径（**跨阶段对齐锚点**）

本卡登记门禁 4 的**实测口径**（`W3.5` `benign_guard_check.csv`）：检测器 `llamaguard3-8b`
rev `7327bd9f…`、4-bit、加载照 `ENV_LOCK.md` §5／§6；抽样 G1／G2／G3 **各 30 条**、
按 `sample_id` 升序步长 150／90／60；**判词解析 = 首行首词**；**截断侧 = 左**（上限 512 token）；
命中 1／90；`unparsed` 0；**计数口径 = 只按内容计数**（`D-00-05`：`MI-M` 计入，
`MI-W`／`MI-B` 不计入；G3 先还原再送检，G1／G2 原样送检）。

**用途**：**`W4` 的 B5 若沿用同口径须显式声明，若改判须写明差异**——否则「门禁 4 的检出率」
与「B5 的指标」测量口径不一致，读者无法对齐。**该条不制定 B5 的口径**（B5 归 W4），
只把门禁 4 的口径**固定下来可对照**。（转交项 40）

## §26 门禁 4 的人工复核结果（**内联，因复核载体不进冻结面**）

人工复核落于 `data/interim/benign_G1_G2_G3/benign_guard_review.csv`（**9 行**；
sha256 `aa6e66d5afc908a7185857d6f2a7de4088258941aab7a69080a2c0b35019d627`，
**与 `G3.csv` `727e820f…` 同代**），与该目录其余产物一样**不进版本控制、不进冻结面**。

1. **门禁 4 结果**：人工复核后 `counted = **0** / 90`、检出率 **0.0000 ≤ 0.05** ⇒ **通过**
   （**不是** `02_05` 机械赋值下的 6/90；两者**分别标明**，避免读者取错值）。
2. **复核者 `USER`**、**复核日期 2026-10-01 与 2026-10-02**；**9 条** `S1`／`S8`／`S14` 命中
   **全部改判 `MI-B`**，判定路径均为 `Q1是→Q2否→Q5否`（其中 `BN-G3-0661` 为 `r6` 后新增命中；
   `0241`／`0481`／`0541`／`0721` 四条系**按现行文本重新判定并重新绑定指纹**）。
3. **代录声明（不得省略）**：判定由**用户**给出，**W3 仅代录**；逐条理由**不引样本原文**
   （依 `g3_hard_negative.md` §7-8 的精神）。
4. **口径**：门禁 4 的判定量＝**人工按 `malicious_intent.md` §4 赋 MI 标签后的 `MI-M` 计数**，
   **不是** LG-8B 的原始命中率（见 `reports/data/g3_qc_v1.0_20261001-210403.md` §5）。
5. **送检／截断口径（局限，与结果同批登记）**：见 §13-5。
6. `H4` 的判据为**复核后**的 `counted`（即 `benign_guard_check.csv` 经补正轮 `r3`／`r6`
   读入人工值后的那一版）。
7. **复核表须与 `G3.csv` 同代**：**G3 一经变更，抽样内容即变** ⇒ 复核表须**按现行文本
   重新判定并重新绑定 `text_sha256`**；未重绑者按 `02_05` 的 `unbound` 处理
   （**视为未复核、非零退出，严禁默认 `MI-M`**）。**G3 变更后不得沿用上一代的复核结论。**

## §28 分片代的唯一权威 ＋ 冻结侧口径

1. 冻结与断言**只认 `data/interim/split_checks/` 顶层**（= `W3.6` **`r2` 原子化**、
   `family_atomic: true` ＋ `--stratify-length` 的产物）；`pre_atomic_run/` 与
   `r2_probe_false_path/` **仅供留档，不得被冻结或下游引用**。
2. `H5` 的 `exit=0` **已与 stdout 编码无关**（`W3.7` 补正轮 `r2` 修 `F-1`：
   `init_streams()` 重配 stdout／stderr 为 UTF-8）⇒ 重定向取证**不会再假失败**。
3. `02_08` **不自行重算门禁 5／6／7**，只调用 `02_07` 并读其产物与退出码（`H5` 原注不变）。

## §29 门禁 7 改为报告项后的**四项披露**（`D-36` 裁定一／二）

**① 门禁 7 的 `n/a` 判定与三项证据**：**阈值 `0.05` 一字未改**、**不参与退出码**；
原子化后实算 `regime_id` **0.852003**、`regime_ood` **0.708482**；三项证据 =
(a) 阈值与实测（上表）；(b) **② 逐编码对半控制**；(c) **③ 随机对半控制**（`ngram_controls.csv`，`current` 分片：
`regime_id/test` **0.885060**、`regime_ood/test` **0.892481**）。
**须写明该代理量不具判别力的原因**：同分片、构造上零泄露的两半即达 **0.87–0.89**，
与 `train→test` 同量级 ⇒ 该量**对「跨分片」与「同分布」不可区分**；
残余交集来自语言／代码公共 3-gram 与 `02_04` 的 **10 份宿主骨架**。

**② `family_atomic: true` 的口径变更** ＋ `family_straddle` **跨档族 = 0**（原子化前为 **17**）。

**③ 原子化前后对照**：攻击 group **176 / 22 / 22** 与 `test` 攻击行 **792** **均未变**；
`A6` = **440**、`A7` = **7920** **均未变**。

**④ 代价：行为多样性 `20 → 7`**（`test` 分片按来源计的行为数）——**不是**「测试集缩小 7 倍」。

**配套**：三项证据的产物为 `data/interim/split_checks/{ngram_overlap.csv, ngram_controls.csv}`；
`A14` 的**报告性**地位（`passed` 恒 `n/a`）已在 §3 字段说明中写明。

## §30 `PREREG_v1.3` 的三处数值更正（**并列登记**）

| 项（原子化后） | `PREREG_v1.3` §2④／§3.2 所记 | **实测**（`W3` 复算 == `W3.7` r1 §10.5） |
|---|---|---|
| 最近邻 **中位数** | 0.3560 | **0.3880** |
| 最近邻 **max** | 0.3904 | **0.5026** |
| 最近邻 **≥0.5 占比** | 0.0000 | **0.0227（= 1/44）** |

**根因**：`PREREG_v1.3` 所引为 `W3` 呈报时的**剔除式代理**（剔除与 `train` 同族的 `test` 行
⇒ 仅剩 **3** 个 group ⇒ 「占比 0」是**构造产物**；`W3` 呈报时未声明其为代理）；
**真实重划分是整族重排**，`test` 仍 **792** 行 ⇒ 个别**跨族**最近邻仍可 ≥0.5。
**方向与结论不变**（**86.36% → 2.27%**，实质泄露已消除）。
**按 `D-12` 不改 `PREREG_v1.3`**，承接只在本卡与论文。

**同批（并列登记并标注分片代，不得只留一组、亦不得以「不可复现」为由删除前一组）**：

| 组 | 分片代 | 值 |
|---|---|---|
| `ascii_art` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.984772 |
| `base64` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.925081 |
| `hex` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.931085 |
| `homoglyph` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.883350 |
| `morse` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.991071 |
| `reverse` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.825025 |
| `rot13` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.885428 |
| `url` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.885806 |
| `zerowidth` | `current`（`ngram_controls.csv` 的 `run` 列） | 0.839477 |
| `ascii_art` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.965000 |
| `base64` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.861301 |
| `hex` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.971165 |
| `homoglyph` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.874260 |
| `morse` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.952586 |
| `reverse` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.886076 |
| `rot13` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.864507 |
| `url` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.917414 |
| `zerowidth` | `pre_atomic`（`ngram_controls.csv` 的 `run` 列） | 0.897398 |

`PREREG_v1.3` §1.2 ② 的五值（`base64` 0.9177／`rot13` 0.8336／`morse` 0.9774／
`ascii_art` 0.9799／`zerowidth` 0.9216，**`pre_atomic` 分片**）**可复现**（`W3` 已 **5/5** 复现，
复现规则见 `W3.7` r1 报告 §10.4 更正段）；上表为**同一 `by_encoding` 控制**在当前分片代的实测值。

> **待裁**：是否需 `PREREG_v1.4` 承接上述更正 —— `D-37` 裁定二已定 **需要**（转交项 44，
> **时点在冻结之后**，不阻塞本冻结）。按本条要求，**无论裁定如何，并列登记不变**。

## §31 `g3_hard_negative_test.csv` 的体制二子集取法

文件全量 **64 行 / 47 group**；**体制一 = 文件全量**；
**体制二 = 按 `benign_group_id` ∩ 该体制 `test` 良性 group 集** ⇒ **31 行 / 31 group**（`ascii_art` 3／`hex` 28）。

⚠ **不得**按 `encoding_type ∈ 留出 4 码` 过滤（会得错误行数）：`ood ⊆ id` 是**集合包含**，
**不蕴含**「按编码过滤 id 即得 ood」（`W3.6` `F-1`，`r1` 已更正）。

---

*本数据卡随 `data-v1.0` 冻结；任何修改必须新建 `CodeObf-PI_v1.1`。*

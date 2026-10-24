# -*- coding: utf-8 -*-
"""02_09_audit_g3.py — W3.5 G3 质检审计：门禁 3（双人质检）+ 门禁 4 复核 + 落点分布比对表。

三段式（`windows/W3.5_benign.md` §4.2）::

    # ① 开窗（产出待标注表 + 落点分布比对表；**必须早于人工标注**，转交项 2）
    python scripts/data/02_09_audit_g3.py --make-worksheet

    # ② 两位标注者各自独立填写 AN-A / AN-B 两份表（脚本不代填、不代判）

    # ③ 汇总与门禁判定（`make audit GATE=G3` 即无参运行；非零退出＝门禁未过）
    python scripts/data/02_09_audit_g3.py

**硬约束（`windows/W3.5_benign.md` §8-3／§8-4）**

* **禁止脚本伪造门禁 3 的双人标注结论**：`pass`／`verdict`／`annotator`／`remark`
  四列在 ① 阶段**一律为空**；③ 阶段只做**汇总与判定**，不产生任何标注值。
* **禁止在标注列全空时判「通过」**；**禁止按「已填的算」凑通过率**（`F8`：两张表的
  `sample_id` 集合必须完全一致且等于抽样的 100 条，缺失即报错）。
* **禁止为凑通过率修改判定规则、业务必需形态清单或更换标注者**。
* 判定不合格（`G3-NONSENSE`／`G3-LEAK`／`G3-EXEC`）的样本**须移出 G3 集**，
  不得以任何方式（含改标）留在 G3 集或良性集中（`g3_hard_negative.md` §7-4）。

**产物位置**：`data/interim/benign_G1_G2_G3/`（W3.5 的 A3 权限只到 `data/interim/`）。
`PROJECT_CONTRACT.md` §3.1 规定的归档路径 `data/frozen/CodeObf-PI_v1.0/reports/g3_qc.csv`
由 **W3.8** 在冻结时自 `data/interim/` 拷入（`data/frozen/` 由 W3.8 独占写入）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load_module("data_common", os.path.join(_HERE, "_common.py"))
# `02_05` 的 `equidistant_sample` / `dec_payload` / `MaliciousFilter` 为**唯一实现**，
# 本脚本复用，保证「同一抽样集合、同一还原口径」，避免两份实现漂移。
build_mod = _load_module("data_build_benign", os.path.join(_HERE, "02_05_build_benign.py"))

read_csv = common.read_csv
write_csv = common.write_csv
sha256_text = common.sha256_text

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_GATE = 3

#: 待标注表字段。`verdict`／`pass`／`remark`／`annotator` 为**标注列**（① 阶段一律为空）。
WORKSHEET_FIELDS = [
    "sample_id", "benign_set", "benign_group_id", "source", "carrier_type",
    "encoding_type", "g3_business_form", "case_id_mechanical", "host_context",
    # **回退轮 r2 §r2.2.1 ⓒ 新增**：每条合成样本的「该宿主为何需要该编码」，
    # 随工作表一并给标注者（否则 `Q3` 业务必需问无从判断）。
    "business_necessity_evidence",
    "text_sha256", "text", "verdict", "pass", "remark", "annotator",
]
#: 标注列（`--make-worksheet` 必须留空）
ANNOTATION_COLUMNS = ("verdict", "pass", "remark", "annotator")
#: 归档字段（`PROJECT_CONTRACT.md` §3.1；`g3_hard_negative.md` §7-8：**不含样本原文**）
ARCHIVE_FIELDS = ["sample_id", "annotator", "pass", "remark"]

#: `PREREG_v1.md` §10 的 `W0-A1` 原文（逐字引用；出处 `reports/prereg/annotator_calibration_v1.md` §10）
W0_A1_TEXT = (
    "W0-A1：阶段0 校准的 20 条样本须覆盖四份 rubric 各自的判定对象，且必须包含各 rubric "
    "边界 case 清单中列出的类型实例；若后续某次全量标注（G3 质检、ASR 人工标注）的样本分布"
    "与阶段0 校准样本显著不同，须在该任务全量标注前追加一次校准，新增校准使用新的 20 条，"
    "不得重用。依据：v5 新增6「正式开展 G3 质检、ASR 人工标注前，必须先做标注者校准」的意图。"
)

#: `reports/prereg/annotator_calibration_v1.md` §9 第 4 条 —— **逐字保留，不得删改、不得省略、不得转述**
#: （出处：该文件 L341，`windows/W3.5_benign.md` §4.4-2）
CALIBRATION_S9_4_TEXT = (
    "4. **同时声明的限定**：本结论的覆盖范围受 §3.5 的覆盖率（41/53）与 §4 的分布局限约束——"
    "`g3_hard_negative` 的 7 条「业务必需形态」类边界 case 与 `asr_judge` 的 4 条边界 case "
    "**未被本批校准**，其标注一致性**在本报告中无证据**。该限定不改变门禁判定（门禁只认 κ 值），"
    "但须随本报告一并引用（见 §10、§11）。"
)
CALIBRATION_S9_4_SOURCE = ("reports/prereg/annotator_calibration_v1.md §9 第 4 条"
                           "（该文件 L341，逐字引用）")


def _rel(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace(os.sep, "/")
    except ValueError:  # pragma: no cover
        return path


# ===========================================================================
# 落点（`case_id`）的**机械推导** —— 只供覆盖比对表使用，**不是判定**
# ===========================================================================
#: `Q2` 执行能力问的判据子集（一旦执行会造成危害的可执行内容）
Q2_EXECUTION_REGEXES = ("pipe_to_shell", "shell_download_exec", "base64_exec_chain",
                        "windows_download_exec")


def derive_case_id(row, cfg, mal, restored):
    """按 rubric §4 的 `Q1`→`Q2`→`Q3` 顺序机械推导落点 `case_id`。

    **口径声明**：本函数服务于转交项 2 的**覆盖比对**（`windows/W3.5_benign.md` §4.3），
    依据是「**可观测文本特征**」而非生成者自述，且**只取机械可判的部分**
    （明文恶意串、还原后的恶意串、危害性执行串、编码层可否还原、业务必需形态）。
    它**不替代**标注者的判定：门禁 3 的通过与否**只**由两人的 `verdict` 决定。

    返回 ``(case_id, trace)``；``case_id`` 为 ``None`` 表示该形态在 rubric §5 的
    `case_id` 清单中**没有专属条目**（如实登记，不为凑覆盖而指到别处）。
    """
    trace = {}
    plaintext_hits = mal.hits(row.get("text") or "")
    trace["Q1a_plaintext"] = plaintext_hits
    if plaintext_hits:
        return "G3-15", trace

    restored_hits = mal.hits(restored) if restored else []
    trace["Q1b_restored"] = restored_hits
    if restored_hits:
        return "G3-11", trace

    q2 = []
    for _phrase, regex_name in [(None, name) for name in Q2_EXECUTION_REGEXES]:
        for name, pattern in mal.regexes:
            if name == regex_name and pattern.search(restored or ""):
                q2.append("regex:%s" % name)
    trace["Q2_execution"] = q2
    if q2:
        return "G3-12", trace

    form = row.get("g3_business_form") or ""
    trace["Q3_business_form"] = form
    if not row.get("recoverable", True):
        return "G3-14", trace
    case_id = (cfg["audit"]["business_form_case_ids"] or {}).get(form) or None
    trace["Q3_case_id"] = case_id
    if case_id is None:
        return None, trace          # 形态无专属 `case_id`（如实登记）
    return case_id, trace


def build_coverage(sampled, cfg, mal):
    """转交项 2：构建 T1（`case_id` 级）与 T2（分支级）两张表的数据结构。"""
    audit = cfg["audit"]
    all_ids = list(audit["g3_case_ids"])
    branch_ids = list(audit["business_necessary_branch"])
    uncovered = set((audit["calibration_uncovered"] or {}).get("G3") or [])

    per_sample = []
    for row in sampled:
        restored = row.get("_restored")
        case_id, trace = derive_case_id(row, cfg, mal, restored)
        per_sample.append({"sample_id": row["sample_id"],
                           "case_id": case_id or "（无对应 case_id）",
                           "case_id_raw": case_id,
                           "business_form": row.get("g3_business_form") or "",
                           "source": row["source"],
                           "carrier_type": row["carrier_type"],
                           "encoding_type": row["encoding_type"],
                           "in_business_branch": bool(case_id in branch_ids),
                           "trace": trace})

    t1 = []
    for cid in all_ids:
        instances = [p for p in per_sample if p["case_id_raw"] == cid]
        t1.append({
            "case_id": cid,
            "calibration_covered": cid not in uncovered,
            "calibration_note": "" if cid not in uncovered else "校准报告 §3.5 登记为**未覆盖**",
            "instances": len(instances),
            "covered_this_time": bool(instances),
            "samples": [p["sample_id"] for p in instances][:8],
        })

    branch_instances = [p for p in per_sample if p["in_business_branch"]]
    unassigned = [p for p in per_sample if p["case_id_raw"] is None]
    branch_covered_ids = sorted({p["case_id_raw"] for p in branch_instances})
    branch_uncovered_ids = sorted(set(branch_ids) - uncovered)
    t2 = {
        "dimension": "「业务必需形态」分支（`g3_hard_negative.md` §4 `Q3` 的形态清单）",
        "instances": len(branch_instances),
        "total": len(per_sample),
        "ratio": (len(branch_instances) / float(len(per_sample))) if per_sample else 0.0,
        "covered_case_ids": branch_covered_ids,
        "still_uncovered_case_ids": [c for c in sorted(uncovered) if c in branch_ids],
        "calibration_evidence": (
            "校准报告 §3.5 明写「`G3` 的 7 条未覆盖项全部是「编码层为业务必需形态」的良性类型……"
            "该维度**未被本批校准**」；该维度在本批校准中仅有 `G3-01`（N8）一个实例。"),
        "unassigned_instances": len(unassigned),
    }
    return {"per_sample": per_sample, "t1": t1, "t2": t2,
            "uncovered": sorted(uncovered)}


def render_coverage_md(ctx):
    """转交项 2 的产物正文（`reports/data/g3_coverage_v1.0_<ts>.md`）。"""
    cov = ctx["coverage"]
    cfg = ctx["cfg"]
    audit = cfg["audit"]
    out = []
    out.append("# G3 落点分布比对表 v1.0（转交项 2）\n")
    out.append("- 窗口：**W3.5**（阶段2 G1/G2/G3 良性样本构建）")
    out.append("- 日期：%s" % ctx["stamp"])
    out.append("- 脚本：`scripts/data/02_09_audit_g3.py --make-worksheet`")
    out.append("- 对照物：`reports/prereg/annotator_calibration_v1.md` §3（覆盖矩阵）／§3.5（汇总）")
    out.append("- 依据：`PREREG_v1.md` §10 `W0-A1`；`DECISIONS.md` `D-00-07`；"
               "`windows/W3.5_benign.md` §4.3")
    out.append("- **触发条件**：本表须在 G3 质检（门禁 3）**开窗之前**产出；"
               "未产出本表不得启动人工标注。\n")
    out.append("## 1. 抽样\n")
    out.append("| 项 | 值 |")
    out.append("|---|---|")
    out.append("| 抽样规模 | **%d** 条（`audit.qc_sample_size`）|" % len(cov["per_sample"]))
    out.append("| 抽样方式 | **确定性等距**（按 `sample_id` 升序，取下标 `0, k, 2k, …`，"
               "`k = floor(n/%d)` = %s）|"
               % (audit["qc_sample_size"], ctx["step"]))
    out.append("| G3 全量 | %d 条 |" % ctx["g3_total"])
    out.append("| 落点推导 | 依 rubric §4 的 `Q1`→`Q2`→`Q3` 顺序在**可观测特征**上机械推导，"
               "**不依生成者自述**；推导**不是判定**，判定归标注者 |")
    out.append("")
    out.append("## 2. T1 · `case_id` 级\n")
    out.append("| `case_id` | 校准报告 §3 是否覆盖 | 本次抽样实例数 | 本次是否覆盖 | 实例（前 8）|")
    out.append("|---|---|---|---|---|")
    for item in cov["t1"]:
        out.append("| `%s` | %s | %d | %s | %s |"
                   % (item["case_id"], "是" if item["calibration_covered"] else "**否**",
                      item["instances"], "**是**" if item["covered_this_time"] else "否",
                      "、".join(item["samples"]) or "—"))
    out.append("")
    n_cal_uncovered = len([i for i in cov["t1"] if not i["calibration_covered"]])
    n_cal_uncovered_hit = len([i for i in cov["t1"]
                               if not i["calibration_covered"] and i["covered_this_time"]])
    out.append("**校准报告 §3.5 登记未覆盖的 `G3` 项共 %d 条**，本次覆盖其中的 **%d** 条。"
               % (n_cal_uncovered, n_cal_uncovered_hit))
    out.append("逐条登记（**不得追补为「已覆盖」**，`windows/W3.5_benign.md` §4.4-3）：\n")
    out.append("| 未覆盖 `case_id` | 本次是否出现实例 |")
    out.append("|---|---|")
    for item in cov["t1"]:
        if item["calibration_covered"]:
            continue
        out.append("| `%s` | %s |" % (item["case_id"],
                                      "是（%d 条）" % item["instances"] if item["instances"]
                                      else "否"))
    out.append("")
    if cov["t2"]["unassigned_instances"]:
        out.append("另有 **%d** 条样本的 `g3_business_form` 为「代码内天然携带的编码字符串片段」，"
                   "该形态在 rubric §5 的 16 条 `case_id` 清单中**没有专属条目** —— 如实登记为"
                   "「无对应 `case_id`」，**不为凑覆盖而指到别的 `case_id`**。\n"
                   % cov["t2"]["unassigned_instances"])
    out.append("### 2.1 逐条落点（%d 条）\n" % len(cov["per_sample"]))
    out.append("| sample_id | `case_id` | 业务必需形态 | 是否落在「业务必需形态」分支 | 载体 | 编码 |")
    out.append("|---|---|---|---|---|---|")
    for item in cov["per_sample"][:ctx["max_per_sample_rows"]]:
        out.append("| %s | %s | %s | %s | %s | %s |"
                   % (item["sample_id"], item["case_id"], item["business_form"],
                      "是" if item["in_business_branch"] else "否",
                      item["carrier_type"], item["encoding_type"] or "—"))
    if len(cov["per_sample"]) > ctx["max_per_sample_rows"]:
        out.append("| … | （其余 %d 条见 `g3_coverage_per_sample.csv`）| | | | |"
                   % (len(cov["per_sample"]) - ctx["max_per_sample_rows"]))
    out.append("")
    out.append("## 3. T2 · 分支级\n")
    out.append("| 项 | 值 |")
    out.append("|---|---|")
    out.append("| 分支 | %s |" % cov["t2"]["dimension"])
    out.append("| 本次抽样实例数 | **%d / %d**（占比 %.4f）|"
               % (cov["t2"]["instances"], cov["t2"]["total"], cov["t2"]["ratio"]))
    out.append("| **分支级是否出现** | **是**（出现 ≠ 覆盖）|")
    out.append("| 校准证据 | %s |" % cov["t2"]["calibration_evidence"])
    out.append("| 本次覆盖到的该分支 `case_id` | %s |"
               % "、".join("`%s`" % c for c in cov["t2"]["covered_case_ids"]))
    out.append("| 该分支内**仍未被校准**的 `case_id` | %s |"
               % ("、".join("`%s`" % c for c in cov["t2"]["still_uncovered_case_ids"]) or "—"))
    out.append("")
    out.append("## 4. 判定与后续\n")
    out.append("### 4.1 两种口径的并列\n")
    out.append("| 口径 | 分支的定义 | 本批校准是否覆盖该分支 | 按此口径的触发结论 |")
    out.append("|---|---|---|---|")
    out.append("| A（`windows/W3.5_benign.md` §4.3 的 T2 字面读法） | 「业务必需形态」维度 | "
               "**否**（校准报告 §3.5 明写该维度未被校准） | 会触发追加校准 |")
    out.append("| B（`D-00-07` 事项二的「分支级」定义 = rubric §4 的**判定分支**，"
               "即 `Q1`→`Q2`→`Q3` 的四个终态） | 四个终态标签 `G3-PASS`／`G3-NONSENSE`／"
               "`G3-LEAK`／`G3-EXEC` | **是**（`G3-01`／`05`／`09`→`G3-PASS`；"
               "`G3-10`／`14`→`G3-NONSENSE`；`G3-11`／`15`／`16`→`G3-LEAK`；"
               "`G3-12`→`G3-EXEC`） | **不触发** |")
    out.append("")
    out.append("### 4.2 W3.5 的判定\n")
    out.append("> **判定：不触发追加校准。**（依 **口径 B**，即 `DECISIONS.md` `D-00-07` 事项二）\n")
    out.append("**理由（三条）**：\n")
    out.append("1. `D-00-07` 事项二**逐字**裁定「阶段2 G3 质检 | **不触发追加校准**」，并给出理由："
               "G3 的四个终态分支均已覆盖；7 条未覆盖 `case_id`「与已覆盖者同走 "
               "`Q1`否→`Q2`否→`Q3`是→`G3-PASS`」。这是 W0 对 `W0-A1` 的**解释性裁定**，"
               "且**专门针对本阶段**。")
    out.append("2. `windows/W3.5_benign.md` §4.3 自身即以「`D-00-07` 的『显著不同』取**分支级**口径」"
               "为依据；其 T2 行把「分支」写为「业务必需形态」**维度**，与 `D-00-07` 对「分支」的"
               "定义（判定分支 = 四个终态标签）不一致。**冲突时以 `D-00-07` 为准**"
               "（`W0_STATE.md` §2 的生效裁定索引）。")
    out.append("3. 按口径 A 触发会使 `D-00-07` 对**本阶段**的明示结论失效，"
               "构成对 W0 裁定的推翻——该权限不在 W3.5（`PROJECT_CONTRACT.md` §5："
               "重大偏离须 W0 事前批准）。")
    out.append("")
    out.append("**该判定由 W3.5 自行作出并登记，登记后不回呈 W0**"
               "（`windows/W3.5_benign.md` §4.3；`D-00-07` 对阶段5.2 的同型口径）。\n")
    out.append("### 4.3 局限声明（口径 A 所反映的真实缺口，**不得因判定为「不触发」而省略**）\n")
    out.append("「**业务必需形态**」这一维度在本批校准中**确无一致性证据**"
               "（校准报告 §3.5 明写未被校准）；本次抽样的 **%.1f%%**（%d/%d）落于该维度。"
               % (cov["t2"]["ratio"] * 100, cov["t2"]["instances"], cov["t2"]["total"]))
    out.append("本表按 `D-00-07` 的要求**同时登记分支级与 `case_id` 级**两项覆盖状态，"
               "使该缺口在表上可见；G3 质检报告正文须一并声明该维度未校准"
               "（见 `reports/data/g3_qc_v1.0_*.md`）。\n")
    out.append("## 5. 与门禁 3 的关系\n")
    out.append("按本判定，**不触发追加校准** ⇒ 门禁 3 的 %d 条人工标注可**直接启动**。"
               "抽样表见 `%s`。\n" % (audit["qc_sample_size"], ctx["worksheet_path"]))
    return "\n".join(out) + "\n"


# ===========================================================================
# 待标注表
# ===========================================================================
def build_worksheet_rows(sampled, cfg, mal):
    rows = []
    for row in sampled:
        restored = row.get("_restored")
        case_id, _trace = derive_case_id(row, cfg, mal, restored)
        item = {k: "" for k in WORKSHEET_FIELDS}
        item.update({
            "sample_id": row["sample_id"], "benign_set": row["benign_set"],
            "benign_group_id": row["benign_group_id"], "source": row["source"],
            "carrier_type": row["carrier_type"],
            "encoding_type": row["encoding_type"],
            "g3_business_form": row["g3_business_form"],
            "business_necessity_evidence": row.get("business_necessity_evidence") or "",
            "case_id_mechanical": case_id or "（无对应 case_id）",
            "host_context": row["host_context"],
            "text": row["text"],
            "text_sha256": sha256_text(row["text"]),
        })
        rows.append(item)
    return rows


def assert_worksheet_blank(rows):
    """① 阶段自检：标注列**必须全空**（禁止脚本预填，`F8` 的同源纪律）。"""
    offenders = []
    for row in rows:
        for col in ANNOTATION_COLUMNS:
            if str(row.get(col) or "").strip():
                offenders.append((row["sample_id"], col))
    return offenders


# ===========================================================================
# 自审轮 `r6.7`：`--make-worksheet` **覆盖销毁人工标注**的硬前置
# ===========================================================================
# 事实（W3 实测 2026-10-02）：全库扫描 `data/interim/`／`reports/data/`／`logs/data/`
# （含 `tmp/`）的 CSV，`pass`／`verdict` 非空的表 **0 份** ⇒ **第一轮的 200 条人工标注
# （100 条 × 2 位标注者）已不存在**；`archive_round3/*.csv` 三份全为**空白**（覆盖之后才归档）。
# 根因：本脚本向**固定文件名**直接 `write_csv`，**无「已存在且已填 ⇒ 拒绝覆盖」的守卫**
# （`assert_worksheet_blank` 校验的是**新生成的行**，不保护**已存在的文件**）。
# ⇒ 每当 G3 重建并重跑 `--make-worksheet`，已填的工作表被**静默覆盖**（违反 `AGENTS.md` §6-4）。
# 硬前置两条：① **拒绝覆盖**（已填即非零退出）；② **先归档、后覆盖**（`archive_round<N>/`，
# 带时间戳与「是否已填」标记）。
def annotation_filled(rows):
    """该表是否**已填**（`pass`／`verdict`／`annotator` 任一非空）。返回 `(filled, 例)`。"""
    for row in rows:
        for col in ANNOTATION_COLUMNS:
            if str(row.get(col) or "").strip():
                return True, "%s:%s" % (row.get("sample_id") or "?", col)
    return False, None


def next_archive_dir(benign_dir):
    """下一个 `archive_round<N>/` 目录路径（扫描现有目录取 N+1）。"""
    n = 0
    if os.path.isdir(benign_dir):
        for name in os.listdir(benign_dir):
            m = re.fullmatch(r"archive_round(\d+)", name)
            if m:
                n = max(n, int(m.group(1)))
    return os.path.join(benign_dir, "archive_round%d" % (n + 1))


def archive_worksheets(paths, benign_dir, stamp, log):
    """**覆盖前归档**（`r6.7` 第 2 条）：把现有三份文件复制到 `archive_round<N>/`，
    文件名带**时间戳**与**是否已填**标记；**复制而非移动**（原文件随后被新表覆盖）。"""
    target = next_archive_dir(benign_dir)
    os.makedirs(target, exist_ok=True)
    archived = []
    for path in paths:
        rows = read_csv(path)
        filled, example = annotation_filled(rows)
        tag = "filled" if filled else "blank"
        name = "%s_%s_%s%s" % (os.path.splitext(os.path.basename(path))[0], tag, stamp,
                               os.path.splitext(path)[1])
        dst = os.path.join(target, name)
        shutil.copyfile(path, dst)
        archived.append({"src": path, "dst": dst, "filled": filled, "example": example})
    log("[02_09] r6.7 覆盖前归档：%d 份 → %s（%s）\n"
        % (len(archived), target,
           "、".join("%s=%s" % (os.path.basename(a["dst"]), "已填" if a["filled"] else "空白")
                     for a in archived)))
    return target, archived


# ===========================================================================
# 门禁 3 —— 双人标注的汇总与判定（**只汇总，不代判**）
# ===========================================================================
def cohen_kappa(pairs):
    """Cohen's κ（`PREREG_v1.md` §10 的校准门禁口径）。`pairs` 为 `(a, b)` 序列。"""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "p0": None, "pe": None, "kappa": None}
    labels = sorted({label for pair in pairs for label in pair})
    p0 = sum(1 for a, b in pairs if a == b) / float(n)
    pe = 0.0
    for label in labels:
        ca = sum(1 for a, _b in pairs if a == label) / float(n)
        cb = sum(1 for _a, b in pairs if b == label) / float(n)
        pe += ca * cb
    kappa = None if pe >= 1.0 else (p0 - pe) / (1.0 - pe)
    return {"n": n, "p0": p0, "pe": pe, "kappa":kappa}


def _normalize_pass(value):
    """把标注者填写的 `pass` 归一为 `True`／`False`；无法识别时返回 `None`。"""
    text = str(value or "").strip().lower()
    if text in ("yes", "y", "1", "true", "pass", "通过", "是"):
        return True
    if text in ("no", "n", "0", "false", "fail", "不通过", "否"):
        return False
    return None


def judge_qc(pairs, threshold):
    """门禁 3：双人**均为 `G3-PASS`** 的样本 / 抽样总数 ≥ `threshold`。

    `pairs` 为 `[(sample_id, pass_a, pass_b)]`；任一侧为 `None`（未填）视为**未通过**
    —— **禁止按「已填的算」凑通过率**（`F8`）。分母恒为 `len(pairs)`。
    """
    total = len(pairs)
    both = sum(1 for _sid, a, b in pairs if a is True and b is True)
    rate = (both / float(total)) if total else 0.0
    return {"total": total, "both_pass": both, "rate": rate,
            "threshold": threshold, "passed": bool(total) and rate >= threshold,
            "unfilled": sum(1 for _sid, a, b in pairs if a is None or b is None)}


def assert_f8(set_a, set_b, expected_ids):
    """`F8`：两张标注表的 `sample_id` 集合**完全一致**且等于抽样的样本集。"""
    expected = set(expected_ids)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    missing = sorted(expected - set_a)
    extra = sorted(set_a - expected)
    extra_b = sorted(set_b - expected)
    passed = not (only_a or only_b or missing or extra or extra_b)
    return {"passed": passed, "expected": len(expected),
            "annotator_a_only": only_a, "annotator_b_only": only_b,
            "missing_from_a": missing, "extra_in_a": extra, "extra_in_b": extra_b}


def rows_to_annotations(rows):
    """把一张标注表的原始行转成 `{sample_id: {verdict, pass, remark, annotator}}`。"""
    out = {}
    for row in rows:
        sid = (row.get("sample_id") or "").strip()
        if not sid:
            continue
        out[sid] = {"verdict": (row.get("verdict") or "").strip(),
                    "pass": _normalize_pass(row.get("pass")),
                    "remark": (row.get("remark") or "").strip(),
                    "annotator": (row.get("annotator") or "").strip()}
    return out


def load_annotations(path):
    """读一张标注表，返回 `{sample_id: {verdict, pass, remark, annotator}}`。"""
    return rows_to_annotations(read_csv(path))


def verify_worksheet_integrity(rows, g3_text_by_id):
    """复验待标注表的**内容完整性**（补正轮 `W3.5-r1`）。

    三条判据，**任一不满足即非零退出**（读表阶段、先于任何汇总与门禁判定）：

    1. **`text_sha256` 自洽**：`sha256(text) == text_sha256`。该列由
       `--make-worksheet` 写入；**不复验它等于没写** —— 表被替换／被截断／
       被手工改动后，标注者判的是另一段文本，而门禁会**静默通过**。
    2. **与权威 `G3.csv` 交叉**：同一 `sample_id` 的 `text` 必须与 `G3.csv` 的
       `text` **逐字符相同**（防「用上一轮的旧表配这一轮的 G3」）。
    3. `sample_id` 必须能在 `G3.csv` 中找到（防表与数据集错配）。

    `text_sha256` 列**缺失**即判失败：非 `--make-worksheet` 产出的表不是合法待标注表。
    """
    has_column = bool(rows) and ("text_sha256" in (rows[0] or {}))
    sha_mismatch, g3_mismatch, missing_in_g3, checked = [], [], [], 0
    for row in rows:
        sid = (row.get("sample_id") or "").strip()
        if not sid:
            continue
        if not has_column:
            continue
        checked += 1
        if sha256_text(row.get("text") or "") != (row.get("text_sha256") or ""):
            sha_mismatch.append(sid)
        if sid not in g3_text_by_id:
            missing_in_g3.append(sid)
        elif (row.get("text") or "") != g3_text_by_id[sid]:
            g3_mismatch.append(sid)
    passed = bool(has_column) and checked > 0 and not (
        sha_mismatch or g3_mismatch or missing_in_g3)
    return {"passed": passed, "checked": checked, "has_text_sha256_column": has_column,
            "sha_mismatch": sha_mismatch[:10], "sha_mismatch_count": len(sha_mismatch),
            "g3_mismatch": g3_mismatch[:10], "g3_mismatch_count": len(g3_mismatch),
            "missing_in_g3": missing_in_g3[:10], "missing_in_g3_count": len(missing_in_g3)}


def build_archive_rows(table_a, table_b, name_a, name_b, labels):
    """按 `PROJECT_CONTRACT.md` §3.1 生成 `g3_qc.csv`（**不含样本原文**，`g3_hard_negative.md` §7-8）。"""
    rows = []
    for sid in sorted(set(table_a) | set(table_b)):
        for annotator, table, default_name in ((name_a, table_a, "AN-A"),
                                               (name_b, table_b, "AN-B")):
            item = table.get(sid) or {}
            code = item.get("verdict") or ""
            passed = item.get("pass")
            if passed is None and code in labels:
                passed = (code == labels[0])          # `labels[0]` 即 `G3-PASS`
            rows.append({
                "sample_id": sid,
                "annotator": item.get("annotator") or default_name,
                "pass": "" if passed is None else ("yes" if passed else "no"),
                "remark": item.get("remark") or ("" if code else "（未填写）"),
            })
    return rows


# ===========================================================================
# 门禁 4 的复核
# ===========================================================================
def review_guard(guard_rows, threshold):
    """复核 `benign_guard_check.csv`：只按内容计数（`counted` 列），阈值 ≤5%。"""
    denominator = len(guard_rows)
    counted = sum(1 for r in guard_rows if str(r.get("counted") or "0").strip() == "1")
    rate = (counted / float(denominator)) if denominator else 0.0
    per_set = {}
    for row in guard_rows:
        bucket = per_set.setdefault(row.get("benign_set") or "?", {"n": 0, "counted": 0})
        bucket["n"] += 1
        if str(row.get("counted") or "0").strip() == "1":
            bucket["counted"] += 1
    return {"denominator": denominator, "counted": counted, "rate": rate,
            "threshold": threshold, "passed": denominator > 0 and rate <= threshold,
            "per_set": per_set}


# ===========================================================================
# 报告
# ===========================================================================
def _quoted(text):
    return "\n".join("> " + line if line else ">" for line in text.split("\n"))


def render_qc_md(ctx):
    cfg = ctx["cfg"]
    audit = cfg["audit"]
    out = []
    out.append("# G3 质检报告 v1.0（门禁 3 / 门禁 4 复核）\n")
    out.append("- 窗口：**W3.5**（阶段2 G1/G2/G3 良性样本构建，权限 A3）")
    out.append("- 日期：%s" % ctx["stamp"])
    out.append("- 脚本：`scripts/data/02_09_audit_g3.py`（无参运行 ＝ `make audit GATE=G3`）")
    out.append("- 依据：`PROJECT_CONTRACT.md` §3.1；`PREREG_v1.md` §5／§11.1；"
               "`configs/prereg/rubrics/g3_hard_negative.md`（**唯一判定依据**）\n")
    integrity = ctx.get("integrity") or {}
    if integrity:
        out.append("## 0. 待标注表完整性复验（补正轮 `W3.5-r1`）\n")
        out.append("读表**即**复验，先于 `F8`／通过率／门禁 4 复核，也先于写任何产物；"
                   "不通过则**不予汇总**并非零退出。\n")
        out.append("| 表 | 复验条数 | `text_sha256` 自洽 | 与 `G3.csv` 逐字符相同 | 结果 |")
        out.append("|---|---|---|---|---|")
        for name in ("a", "b"):
            item = integrity.get(name) or {}
            out.append("| %s | %d | %s | %s | %s |"
                       % (("`%s`" % (ctx.get("worksheet_paths") or {}).get(name, name)),
                          item.get("checked", 0),
                          "通过" if (item.get("has_text_sha256_column")
                                     and not item.get("sha_mismatch_count")) else "**不通过**",
                          "通过" if not (item.get("g3_mismatch_count")
                                         or item.get("missing_in_g3_count")) else "**不通过**",
                          "**通过**" if item.get("passed") else "**不通过**"))
        out.append("")
        out.append("**判据**：① `sha256(text) == text_sha256`（该列由 `--make-worksheet` 写入；"
                   "**不复验它等于没写**——表被替换／截断／手改后，标注者判的是另一段文本，"
                   "而门禁会**静默通过**）；② 同一 `sample_id` 的 `text` 与权威 `G3.csv` **逐字符相同**；"
                   "③ `sample_id` 必须存在于 `G3.csv`。`text_sha256` 列缺失即判失败。\n")
    out.append("## 1. 门禁 3 —— G3 质量门禁（通过率 ≥%.0f%%，双人独立）\n"
               % (float(audit["pass_threshold"]) * 100))
    out.append("| 项 | 值 |")
    out.append("|---|---|")
    out.append("| 抽样规模 | %d 条（`audit.qc_sample_size`）|" % ctx["qc"]["total"])
    out.append("| 抽样方式 | 确定性等距（按 `sample_id` 升序，步长 %s）|" % ctx["step"])
    out.append("| 待标注表 | `%s` |" % ctx["worksheet_path"])
    out.append("| 标注者 A | `%s` |" % ctx["annotators"][0])
    out.append("| 标注者 B | `%s` |" % ctx["annotators"][1])
    out.append("| 双人均 `G3-PASS` 的样本数 | **%d** |" % ctx["qc"]["both_pass"])
    out.append("| 通过率 | **%.4f**（阈值 ≥%.2f）|"
               % (ctx["qc"]["rate"], ctx["qc"]["threshold"]))
    out.append("| 未填写（任一侧） | %d |" % ctx["qc"]["unfilled"])
    out.append("| **结果** | **%s** |" % ("通过" if ctx["gate3_passed"] else "未通过"))
    out.append("")
    out.append("### 1.1 `F8` 两张标注表的一致性\n")
    out.append("```json")
    out.append(json.dumps(ctx["f8"], ensure_ascii=False, indent=2))
    out.append("```\n")
    out.append("**口径**：分母恒为抽样的 %d 条；任一侧未填一律计为**未通过**"
               "——**禁止**按「已填的算」凑通过率（`windows/W3.5_benign.md` §5 `F8`）。\n"
               % ctx["qc"]["total"])
    if ctx["qc"]["unfilled"]:
        out.append("> ⚠ **标注未完成**：有 %d 条样本至少一侧未填。**门禁 3 判为未通过**；"
                   "脚本**不代填、不代判**（`windows/W3.5_benign.md` §8-3）。\n"
                   % ctx["qc"]["unfilled"])
    out.append("### 1.2 不合格样本的处置\n")
    out.append("判为 `G3-NONSENSE`／`G3-LEAK`／`G3-EXEC` 的样本**须移出 G3 集**，"
               "**不得**以任何方式（含保留原标签、修改标签）留在 G3 集或良性集中"
               "（`g3_hard_negative.md` §7-4；`AGENTS.md` §6-5 的精神）。"
               "移出动作由 W3 决定后交 W3.6 重新划分（回退 2.5／2.6）。\n")
    if ctx["nonpass_ids"]:
        out.append("本次判为不通过的样本：\n")
        out.append("| sample_id | 标注者 | verdict |")
        out.append("|---|---|---|")
        for item in ctx["nonpass_ids"][:200]:
            out.append("| %s | %s | %s |" % (item["sample_id"], item["annotator"],
                                             item["verdict"]))
        out.append("")
    out.append("## 2. 门禁 4 复核 —— 良性集恶意检出率（≤5%）\n")
    guard = ctx["guard"]
    if guard is None:
        out.append("**未执行**（缺 `data/interim/benign_G1_G2_G3/benign_guard_check.csv`）。"
                   "须先跑 `python scripts/data/02_05_build_benign.py --guard-check`。\n")
    else:
        out.append("| 项 | 值 |")
        out.append("|---|---|")
        auto = sum(1 for r in ctx["guard_rows"] if (r.get("auto_label") or "") == "MI-M")
        out.append("| 检出数（`counted` 列，**`02_05` 已按人工复核值写入**）| **%d** |"
                   % guard["counted"])
        out.append("| 其中检测器机械命中（`auto_label=MI-M`，仅供审计） | %d |" % auto)
        out.append("| 分母 | %d |" % guard["denominator"])
        out.append("| 检出率 | **%.6f**（阈值 ≤%.2f）|" % (guard["rate"], guard["threshold"]))
        out.append("| **结果** | **%s** |" % ("通过" if guard["passed"] else "未通过"))
        out.append("")
        out.append("**计数口径声明**：依 `D-00-05` 与 `g3_hard_negative.md` §4 门禁映射表，"
                   "**只按内容计数**——还原后判 `MI-M` 的计入；判 `MI-W` 或 `MI-B` 的**不计入**"
                   "（含 `G3-LEAK` 中命中 `Q1(b)` 的 `MI-W` 分支者）。"
                   "**本脚本不自行从判词推 MI 标签**：`counted` 一律**只读** `02_05` 写出的列"
                   "（该列已按 W3 裁定的**人工复核值**回写，补正轮 r3）。\n")
        out.append("| 分集 | 条数 | 检出数 |")
        out.append("|---|---|---|")
        for key in sorted(guard["per_set"]):
            item = guard["per_set"][key]
            out.append("| %s | %d | %d |" % (key, item["n"], item["counted"]))
        out.append("")
    out.append("## 3. 转交项 3 —— 引用义务（**四条，逐条落实**）\n")
    out.append("### 3.1 显式引用 `W0-A1` 覆盖口径裁定\n")
    out.append("`PREREG_v1.md` §10 的 `W0-A1` 原文：\n")
    out.append(_quoted(W0_A1_TEXT))
    out.append("")
    out.append("`DECISIONS.md` `D-00-07` 的两项口径：\n")
    out.append("- **「代表性覆盖」**（`D-00-07` 事项一）：`W0-A1` 的「覆盖」指覆盖四份 rubric 的"
               "**判定对象类型**与边界 case 的**类型实例**，**不是**逐条覆盖全部 `case_id`"
               "（20 条在算术上不可能铺满 53 个 `case_id`）。")
    out.append("- **「分支级显著不同」**（`D-00-07` 事项二）：触发追加校准的判据取**分支级**"
               "（rubric §4 的判定分支，即四个终态标签），**不是** `case_id` 级；"
               "仅出现未覆盖的 `case_id` 而分支已覆盖 ⇒ **不触发**，但须在落点分布比对表中逐条登记，"
               "并声明该维度未校准。")
    out.append("")
    out.append("### 3.2 逐字保留校准报告 §9 第 4 条的「未校准」声明\n")
    out.append("出处：**%s**。**未删改、未省略、未转述**：\n" % CALIBRATION_S9_4_SOURCE)
    out.append(_quoted(CALIBRATION_S9_4_TEXT))
    out.append("")
    out.append("### 3.3 校准报告 §3.5 的 12 条未覆盖 `case_id` 登记\n")
    out.append("| rubric | 未覆盖 `case_id` |")
    out.append("|---|---|")
    for rubric_name in ("MI", "G3", "AJ", "DF"):
        items = (audit["calibration_uncovered"] or {}).get(rubric_name) or []
        out.append("| `%s` | %s |" % (rubric_name,
                                      "、".join("`%s`" % c for c in items) or "—（无）"))
    out.append("")
    total_uncovered = sum(len((audit["calibration_uncovered"] or {}).get(r) or [])
                          for r in ("MI", "G3", "AJ", "DF"))
    out.append("合计 **%d 条**，与校准报告 §3.5 的「合计 41/53、未覆盖 12 条」一致。"
               "**以上 12 条一律不得追补为「已覆盖」**（`windows/W3.5_benign.md` §4.4-3；"
               "`D-00-07` 事项一第 3 条）。\n" % total_uncovered)
    out.append("### 3.4 一致性证据范围声明\n")
    out.append("> 本次 G3 标注的一致性证据范围为「**阶段0 校准已覆盖的维度 + 本次落点分布比对表"
               "登记的维度**」。")
    out.append(">")
    out.append("> 具体地：`g3_hard_negative` 在阶段0 校准中覆盖 9/16 个 `case_id`"
               "（覆盖率 56.3%%），未覆盖的 7 条（%s）全部属于「编码层为业务必需形态」的良性类型；"
               "该维度**在本批校准中无一致性证据**（校准报告 §3.5、§9 第 4 条）。"
               % "、".join("`%s`" % c for c in (audit["calibration_uncovered"] or {}).get("G3")
                           or []))
    out.append(">")
    out.append("> 本次落点分布比对表登记的该维度实例数见 `reports/data/g3_coverage_v1.0_*.md` T2。")
    out.append("")
    out.append("## 4. 结论\n")
    if ctx["gate3_passed"] and ctx["gate4_ok"]:
        out.append("> **门禁 3 与门禁 4 均通过。** 可进入 W3.6 划分（`make freeze` 的硬前置满足）。")
    else:
        out.append("> **门禁未全通过。** 依 `AGENTS.md` §6-7，**不得**执行 `make freeze`、"
                   "**不得**开始微调。")
        if not ctx["gate3_passed"]:
            out.append("> - **门禁 3 未过**：通过率 %.4f < %.2f，或有未填写项（%d 条）。"
                       "回退目标 **2.5**（重新生成 G3），且不合格样本须**移出 G3 集**。"
                       % (ctx["qc"]["rate"], ctx["qc"]["threshold"], ctx["qc"]["unfilled"]))
        if not ctx["gate4_ok"]:
            out.append("> - **门禁 4 未过**：检出率 %s > %.2f。回退目标 **2.5**"
                       "（重新清洗过滤良性集）。"
                       % (("—" if guard is None else "%.6f" % guard["rate"]),
                          float(cfg["guard_check"]["threshold"])))
    out.append("")
    return "\n".join(out) + "\n"


# ===========================================================================
# CLI
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="02_09_audit_g3.py",
        description="W3.5 G3 质检审计：门禁 3（双人）+ 门禁 4 复核 + 落点分布比对表（转交项 2/3）")
    p.add_argument("--config", default="configs/data/benign.yaml")
    p.add_argument("--benign-dir", default="data/interim/benign_G1_G2_G3/")
    p.add_argument("--make-worksheet", action="store_true",
                   help="① 开窗：产出待标注表与落点分布比对表（必须在人工标注之前）")
    p.add_argument("--annotator-a", default=None, help="标注者 A 填毕的表")
    p.add_argument("--annotator-b", default=None, help="标注者 B 填毕的表")
    p.add_argument("--guard-csv", default=None, help="门禁 4 明细（缺省取 benign-dir 下的 benign_guard_check.csv）")
    p.add_argument("--report-dir", default="reports/data")
    p.add_argument("--log-dir", default="logs/data")
    p.add_argument("--archive-existing", action="store_true",
                   help="r6.7：重出工作表时，**先把现有表归档**（含已填表）再覆盖；"
                        "缺省时「已填」一律拒绝覆盖")
    p.add_argument("--no-log", action="store_true", help="不落盘日志（供单元测试使用）")
    return p


def _load_g3_rows(path, work_mod):
    rows = read_csv(path)
    for row in rows:
        row["_restored"] = None
        row["recoverable"] = True
    return rows


def _restore_for_coverage(rows, cfg, work_mod):
    """为落点推导准备「还原后」的文本（`Q1(b)` 用的观察对象）。

    回退轮 r2 起，合成样本的构造改为**宿主需求优先**：载体即宿主制品本身，
    编码层只包裹取值、**不包裹**业务语境 ⇒ 全文即「还原后的内容」，
    无需也不应再反解单层载荷（`r2.2.1` 的 ⓐ 要求宿主代码与取值同现于同一制品）。
    真实代码部分本就没有单层编码外壳，观察对象同样是原文。
    """
    for row in rows:
        row["recoverable"] = True
        row["_restored"] = row.get("text")
    return rows


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(newline="\n", encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    args = build_parser().parse_args(argv)
    cfg = build_mod.load_config(args.config)
    errors = build_mod.validate_config(cfg)
    if errors:
        for err in errors:
            print("[02_09] FATAL 配置非法：%s" % err)
        return EXIT_USAGE

    g3_path = os.path.join(args.benign_dir, cfg["output"]["csv"]["G3"])
    if not os.path.exists(g3_path):
        print("[02_09] FATAL 找不到 %s；请先跑 02_05_build_benign.py" % g3_path)
        return EXIT_USAGE

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_lines = []

    def log(text):
        log_lines.append(text)
        print(text, end="" if text.endswith("\n") else "\n", flush=True)

    g3_rows = _load_g3_rows(g3_path, build_mod)
    _restore_for_coverage(g3_rows, cfg, build_mod)
    audit = cfg["audit"]
    sampled, step = build_mod.equidistant_sample(g3_rows, "G3", int(audit["qc_sample_size"]))
    expected_ids = [r["sample_id"] for r in sampled]
    mal = build_mod.MaliciousFilter(cfg["malicious_filter"])

    worksheet_path = os.path.join(args.benign_dir, "g3_qc_worksheet.csv")
    path_a = args.annotator_a or os.path.join(args.benign_dir, "g3_qc_worksheet_AN-A.csv")
    path_b = args.annotator_b or os.path.join(args.benign_dir, "g3_qc_worksheet_AN-B.csv")

    if args.make_worksheet:
        # ---- r6.7 硬前置：拒绝覆盖已填表；否则先归档、后覆盖 ----
        existing = [p for p in (worksheet_path, path_a, path_b) if os.path.exists(p)]
        filled_tables = []
        for path in existing:
            filled, example = annotation_filled(read_csv(path))
            if filled:
                filled_tables.append((path, example))
        if filled_tables and not args.archive_existing:
            for path, example in filled_tables:
                print("[02_09] FATAL 待标注表已填，禁止覆盖：%s（例：%s）" % (path, example))
            print("[02_09] FATAL 该表已填，禁止覆盖；如需重出请先归档或显式指定新路径"
                  "（或加 `--archive-existing` 先归档再覆盖）——`r6.7`／`AGENTS.md` §6-4")
            return EXIT_USAGE
        if existing:
            try:
                archive_worksheets(existing, args.benign_dir, stamp, log)
            except Exception as exc:                     # noqa: BLE001 - 归档失败即不覆盖
                print("[02_09] FATAL 覆盖前归档失败（%s）⇒ **不予覆盖**" % type(exc).__name__)
                return EXIT_USAGE
        rows = build_worksheet_rows(sampled, cfg, mal)
        offenders = assert_worksheet_blank(rows)
        if offenders:
            print("[02_09] FATAL 待标注表的标注列非空（禁止脚本预填）：%s" % offenders[:5])
            return EXIT_USAGE
        write_csv(worksheet_path, rows, WORKSHEET_FIELDS)
        write_csv(path_a, rows, WORKSHEET_FIELDS)
        write_csv(path_b, rows, WORKSHEET_FIELDS)
        log("[02_09] wrote %s（%d rows，标注列全空）\n" % (_rel(worksheet_path), len(rows)))
        log("[02_09] wrote %s / %s（两位标注者各填一份）\n"
            % (_rel(path_a), _rel(path_b)))

        coverage = build_coverage(sampled, cfg, mal)
        coverage_path = os.path.join(args.report_dir, "g3_coverage_v1.0_%s.md" % stamp)
        os.makedirs(os.path.dirname(os.path.abspath(coverage_path)), exist_ok=True)
        with open(coverage_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_coverage_md({"cfg": cfg, "coverage": coverage, "stamp": stamp,
                                         "step": step, "g3_total": len(g3_rows),
                                         "worksheet_path": _rel(worksheet_path),
                                         "max_per_sample_rows": 100}))
        per_sample_path = os.path.join(args.benign_dir, "g3_coverage_per_sample.csv")
        write_csv(per_sample_path,
                  [{k: p[k] for k in ("sample_id", "case_id", "business_form", "source",
                                      "carrier_type", "encoding_type", "in_business_branch")}
                   for p in coverage["per_sample"]],
                  ["sample_id", "case_id", "business_form", "source", "carrier_type",
                   "encoding_type", "in_business_branch"])
        log("[02_09] wrote %s\n" % _rel(coverage_path))
        log("[02_09] wrote %s\n" % _rel(per_sample_path))
        log("[02_09] T2 分支实例数 = %d / %d（占比 %.4f）\n"
            % (coverage["t2"]["instances"], coverage["t2"]["total"], coverage["t2"]["ratio"]))
        log("[02_09] 判定 = **不触发追加校准**（依 `D-00-07` 事项二的分支级口径；"
            "详见比对表 §4）\n")
        if not args.no_log:
            os.makedirs(args.log_dir, exist_ok=True)
            log_path = os.path.join(args.log_dir, "02_09_audit_g3_%s.log" % stamp)
            with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("".join(log_lines))
        print("[02_09] RESULT=WORKSHEET_OK（待人工标注；脚本不代判）")
        return EXIT_OK

    # ---------------- ③ 汇总与判定 ----------------
    coverage_present = bool([f for f in os.listdir(args.report_dir)
                             if f.startswith("g3_coverage_v1.0")]) \
        if os.path.isdir(args.report_dir) else False
    if not coverage_present:
        print("[02_09] FATAL 未找到落点分布比对表（`reports/data/g3_coverage_v1.0_*.md`）——"
              "转交项 2 要求该表**先于**门禁 3 的人工标注产出；"
              "请先跑 `--make-worksheet`（`windows/W3.5_benign.md` §8-7）")
        return EXIT_USAGE

    # ---- 补正轮 `W3.5-r1`：**读表即复验**（`text_sha256` + 与 `G3.csv` 交叉）----
    # 先于 F8／通过率／门禁 4 复核，也先于写任何产物：表若不完整，一切判定无意义。
    raw_a = read_csv(path_a) if os.path.exists(path_a) else []
    raw_b = read_csv(path_b) if os.path.exists(path_b) else []
    g3_text_by_id = {r["sample_id"]: r["text"] for r in g3_rows}
    integrity = {"a": verify_worksheet_integrity(raw_a, g3_text_by_id),
                 "b": verify_worksheet_integrity(raw_b, g3_text_by_id)}
    log("[02_09] VERIFY 待标注表完整性复验：A checked=%d passed=%s；B checked=%d passed=%s\n"
        % (integrity["a"]["checked"], integrity["a"]["passed"],
           integrity["b"]["checked"], integrity["b"]["passed"]))
    bad = [name for name, item in integrity.items() if not item["passed"]]
    if bad:
        for name in bad:
            item = integrity[name]
            print("[02_09] FATAL 待标注表 %s 完整性复验不通过："
                  "缺 text_sha256 列=%s；sha 不符 %d 条 %s；与 G3.csv 不符 %d 条 %s；"
                  "G3.csv 中无此 sample_id %d 条 %s ⇒ **不予汇总、不写任何产物**；"
                  "请重跑 `--make-worksheet` 后由标注者重填"
                  % (name, not item["has_text_sha256_column"],
                     item["sha_mismatch_count"], item["sha_mismatch"],
                     item["g3_mismatch_count"], item["g3_mismatch"],
                     item["missing_in_g3_count"], item["missing_in_g3"]))
        return EXIT_USAGE

    table_a = rows_to_annotations(raw_a)
    table_b = rows_to_annotations(raw_b)
    log("[02_09] annotatorA=%s（%d rows） annotatorB=%s（%d rows）\n"
        % (_rel(path_a), len(table_a), _rel(path_b), len(table_b)))

    f8 = assert_f8(set(table_a), set(table_b), expected_ids)
    log("[02_09] ASSERT F8 %s — 两张标注表的 sample_id 集合一致且等于抽样集\n"
        % ("PASS" if f8["passed"] else "FAIL"))

    labels = list(audit["verdict_labels"])
    pairs = []
    nonpass = []
    for sid in expected_ids:
        a = table_a.get(sid) or {}
        b = table_b.get(sid) or {}
        pa = a.get("pass")
        pb = b.get("pass")
        if pa is None and a.get("verdict") in labels:
            pa = (a["verdict"] == labels[0])
        if pb is None and b.get("verdict") in labels:
            pb = (b["verdict"] == labels[0])
        pairs.append((sid, pa, pb))
        for annotator, table in (("AN-A", a), ("AN-B", b)):
            verdict = table.get("verdict") or ""
            if verdict and verdict != labels[0]:
                nonpass.append({"sample_id": sid, "annotator": annotator, "verdict": verdict})

    qc = judge_qc(pairs, float(audit["pass_threshold"]))
    log("[02_09] gate3 both_pass=%d / %d  rate=%.4f  threshold=%.2f  unfilled=%d\n"
        % (qc["both_pass"], qc["total"], qc["rate"], qc["threshold"], qc["unfilled"]))

    guard_path = args.guard_csv or os.path.join(args.benign_dir,
                                                cfg["output"]["guard_csv"])
    guard_rows = read_csv(guard_path) if os.path.exists(guard_path) else []
    guard = review_guard(guard_rows, float(cfg["guard_check"]["threshold"])) \
        if guard_rows else None
    if guard is not None:
        log("[02_09] gate4 counted=%d / %d  rate=%.6f  threshold=%.2f\n"
            % (guard["counted"], guard["denominator"], guard["rate"], guard["threshold"]))

    gate3_passed = bool(qc["passed"] and f8["passed"])
    gate4_ok = bool(guard is not None and guard["passed"])

    archive_rows = build_archive_rows(table_a, table_b, "AN-A", "AN-B", labels)
    archive_path = os.path.join(args.benign_dir, "g3_qc.csv")
    write_csv(archive_path, archive_rows, ARCHIVE_FIELDS)
    log("[02_09] wrote %s（%d rows；字段依 PROJECT_CONTRACT.md §3.1）\n"
        % (_rel(archive_path), len(archive_rows)))

    reported = guard or {"counted": 0, "denominator": 0, "rate": 0.0, "per_set": {},
                         "threshold": float(cfg["guard_check"]["threshold"]), "passed": False}
    report_path = os.path.join(args.report_dir, "g3_qc_v1.0_%s.md" % stamp)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_qc_md({"cfg": cfg, "stamp": stamp, "qc": qc, "f8": f8,
                               "integrity": integrity,
                               "worksheet_paths": {"a": _rel(path_a), "b": _rel(path_b)},
                               "step": step, "worksheet_path": _rel(worksheet_path),
                               "annotators": ("AN-A", "AN-B") if not (table_a or table_b)
                               else ((list(table_a.values()) or [{}])[0].get("annotator")
                                     or "AN-A",
                                     (list(table_b.values()) or [{}])[0].get("annotator")
                                     or "AN-B"),
                               "guard": guard, "guard_rows": guard_rows,
                               "gate3_passed": gate3_passed,
                               "gate4_ok": gate4_ok, "nonpass_ids": nonpass}))
    log("[02_09] wrote %s\n" % _rel(report_path))

    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, "02_09_audit_g3_%s.log" % stamp)
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(log_lines))
        print("[02_09] log=%s" % _rel(log_path))

    if not (gate3_passed and gate4_ok):
        reasons = []
        if not f8["passed"]:
            reasons.append("F8 标注表集合不一致")
        if not qc["passed"]:
            reasons.append("门禁 3 通过率 %.4f < %.2f（未填 %d 条）"
                           % (qc["rate"], qc["threshold"], qc["unfilled"]))
        if not gate4_ok:
            reasons.append("门禁 4 未过或未执行")
        print("[02_09] RESULT=GATE_FAIL（%s）" % "；".join(reasons))
        return EXIT_GATE
    print("[02_09] RESULT=GATE_OK（门禁 3 通过率 %.4f ≥ %.2f；门禁 4 检出率 %.6f ≤ %.2f）"
          % (qc["rate"], qc["threshold"], guard["rate"], guard["threshold"]))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

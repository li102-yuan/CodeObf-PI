"""tests/eval/conftest.py — eval 层测试的公共前置（**`#64`**）。

缘起（依 `D-80` 裁定三 / `#64` / `D-82`）
---------------------------------------
`scripts/eval/05_01_eval_detection.py` 与 `scripts/eval/05_08_stat_bootstrap.py` 的
`flagged_logger()` 原把日志目录**写死**为 `logs/eval/`，既无 `--log-dir` 也未给
`--self-check`／`--dry-run` 分流 ⇒ **任何**以子进程跑这两个脚本的场合（**尤其本层单元测试**，
以及 `05_08 --self-check` 内部的递归 `_run`）都会在**生产日志目录** `logs/eval/` 留痕。

该现象已复发三次：`W6.9` 交付时 37 份 → `D-80` 清理 9 份 → `D-81` 清理 8 份。

修法（三管齐下）
----------------
1. **写入端**：两脚本的 `resolve_log_dir()` 支持 `CODEOBF_LOG_DIR` 环境变量，并对
   `--self-check`／`--dry-run` 自动分流到 `logs/eval/tmp/`；
2. **测试端（本文件）**：会话启动时把 `CODEOBF_LOG_DIR` 指向 `logs/eval/tmp/`。
   由于子进程继承环境，本层**所有** spawn（含不带 `--dry-run` 的正式路径调用）都落 tmp；
3. **忽略规则**：`.gitignore` L23 `logs/*/tmp/` 已覆盖 `logs/eval/tmp/`（依 `D-15`）。

⇒ 本文件与写入端改动共同构成 `#64` 的闭环；**缺任一环，测试仍会污染 `logs/eval/`**。
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TMP_LOG_DIR = os.path.join(_ROOT, "logs", "eval", "tmp")


def _redirect_test_logs_to_tmp():
    """把测试派生的日志重定向到 `logs/eval/tmp/`（不入库、不留痕）。"""
    os.makedirs(_TMP_LOG_DIR, exist_ok=True)
    os.environ["CODEOBF_LOG_DIR"] = _TMP_LOG_DIR


# 模块导入即生效：pytest 收集 tests/eval 时先加载 conftest，早于任何 spawn。
_redirect_test_logs_to_tmp()

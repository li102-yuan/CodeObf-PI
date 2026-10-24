# -*- coding: utf-8 -*-
"""tests/env/test_01_04_models_manifest.py — 阶段1 模型清单交付物的单元测试

覆盖 ``windows/W2.2_model_hashes.md`` §6.3 四项：

1. ``configs/env/models.yaml`` 与 ``models/commit_hashes/*.json`` 的 revision 一致性校验函数；
2. ``revision`` 与 ``evidence_output_digest`` 的格式校验（40 位十六进制）；
3. ``planned_load`` 取值必须落在 §4.4 词表内（词表外取值报错）；
4. ``fetch_cmd`` 中若出现疑似 token（形如 ``hf_`` 前缀长串）则校验失败。

另含 §6.1 的判据抽样（清单键集合、`attn_impl` 词表、INDEX.md 一行一模型）。

约束：不访问网络；不访问 ``data/``；不加载模型；不改变环境。
校验函数均可用 fixture 数据直接驱动（§6.3「用 fixture 数据」）。

命名说明：编号 ``01_04`` 为**预留号**（对应已上报 W0 的候选脚本 ``scripts/env/01_04_fetch_models.py``），
本窗口未新增该脚本（W2.2 §4.1），故本测试以文件路径读取交付物，不 import 任何脚本。
"""
import json
import os
import re

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_YAML = os.path.join(ROOT, "configs", "env", "models.yaml")
HASH_DIR = os.path.join(ROOT, "models", "commit_hashes")
INDEX_MD = os.path.join(HASH_DIR, "INDEX.md")

# ---------------------------------------------------------------- 校验规则本体

#: §4.4 加载方式词表（全项目统一；词表外取值不合格）
LOAD_VOCAB = frozenset(
    [
        "4bit-nf4",
        "4bit-nf4-offload-cpu",
        "4bit-nf4-cpu",
        "bf16",
        "fp16",
        "fp32",
        "cpu-fp32",
        "api",
    ]
)

DEVICE_VOCAB = frozenset(["cuda", "cpu", "api"])

#: 词表中**禁止**出现的取值（§4.4 末段：不得出现 flash_attention_2 相关取值）
FORBIDDEN_LOAD_SUBSTRINGS = ("flash_attention_2", "flash-attn", "flashattn")

REV_RE = re.compile(r"^[0-9a-f]{40}$")
#: 疑似 HF token：hf_ 前缀 + 至少 20 位词字符
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")

RESOLVED = "resolved"
PENDING = "pending_w2_decision"
STATUS_VOCAB = frozenset([RESOLVED, PENDING])

#: 本窗口必须覆盖的 key 全集（8 个 HF 模型 + DeepSeek-V3 API + 1 个外部基线 B6）
#: 【2026-10-22 追加（W6.12，经用户授权代做、待 W0 追认）】B6 = `llama-prompt-guard-2-86m`
#: 为 `D-151` ④ B 档新增的**外部基线**（只评测不训练），权重在项目外盘位
#: `D:/CodeObfPI_hf_cache/`；r1（`7b7f848`）已登记 `commit_hashes` 但漏登记清单侧，致
#: `test_real_manifest_json_revision_consistency` 存量红一条，本行即补该缺口。
REQUIRED_KEYS = (
    "llamaguard3-1b",
    "llamaguard3-8b",
    "unplug-tiny",
    "qwen2.5-coder-1.5b",
    "qwen2.5-coder-3b",
    "codellama-7b",
    "mistral-7b",
    "qwen2.5-72b",
    "deepseek-v3-api",
    "llama-prompt-guard-2-86m",
)

MANDATORY_FIELDS = (
    "key",
    "role",
    "repo_id",
    "revision",
    "resolution_status",
    "local_status",
    "local_path",
    "local_sha_verified",
    "size_bytes",
    "license",
    "gated",
    "planned_load",
    "planned_device",
    "fetch_cmd",
    "notes",
)


class ManifestError(ValueError):
    """清单不合格。校验函数统一抛此异常，便于测试断言。"""


def validate_revision(rev, where="<unknown>"):
    """revision 必须是 40 位小写十六进制（§6.1 判据 2）。"""
    if not isinstance(rev, str) or not REV_RE.match(rev):
        raise ManifestError("%s: revision 非 40 位十六进制: %r" % (where, rev))
    return rev


def validate_planned_load(value, where="<unknown>"):
    """planned_load 必须落在 §4.4 词表内（§6.3-3）。"""
    if value not in LOAD_VOCAB:
        raise ManifestError(
            "%s: planned_load=%r 不在 §4.4 词表内 %s" % (where, value, sorted(LOAD_VOCAB))
        )
    for bad in FORBIDDEN_LOAD_SUBSTRINGS:
        if bad in str(value):
            raise ManifestError("%s: planned_load 含禁止取值 %r" % (where, bad))
    return value


def validate_fetch_cmd(cmd, where="<unknown>"):
    """fetch_cmd 非空，且不得内含疑似 token 明文（§6.3-4）。"""
    if not isinstance(cmd, str) or not cmd.strip():
        raise ManifestError("%s: fetch_cmd 为空" % where)
    m = TOKEN_RE.search(cmd)
    if m:
        raise ManifestError("%s: fetch_cmd 含疑似 token 明文（已遮蔽位置 %d）" % (where, m.start()))
    return cmd


def check_entry(entry, where=None):
    """单条模型记录的字段级校验。返回 entry。"""
    where = where or str(entry.get("key"))
    missing = [f for f in MANDATORY_FIELDS if f not in entry]
    if missing:
        raise ManifestError("%s: 缺少必填字段 %s" % (where, missing))

    if entry["resolution_status"] not in STATUS_VOCAB:
        raise ManifestError(
            "%s: resolution_status=%r 不在 %s" % (where, entry["resolution_status"], sorted(STATUS_VOCAB))
        )
    if entry["local_status"] not in ("downloaded", "hash_only"):
        raise ManifestError("%s: local_status=%r 非法" % (where, entry["local_status"]))

    validate_planned_load(entry["planned_load"], where)
    if entry["planned_device"] not in DEVICE_VOCAB:
        raise ManifestError("%s: planned_device=%r 非法" % (where, entry["planned_device"]))
    validate_fetch_cmd(entry["fetch_cmd"], where)

    if entry["resolution_status"] == RESOLVED:
        if entry["repo_id"] is None and entry["key"] != "deepseek-v3-api":
            raise ManifestError("%s: resolved 但 repo_id 为空" % where)
        if entry["key"] != "deepseek-v3-api":
            validate_revision(entry["revision"], where)
    else:
        # 未裁定项：revision 必须为空，且不得谎称已落地
        if entry["revision"] is not None or entry["repo_id"] is not None:
            raise ManifestError("%s: pending 项的 repo_id/revision 必须为 null" % where)
        if entry["local_status"] != "hash_only":
            raise ManifestError("%s: pending 项不得为 downloaded" % where)

    if entry["local_status"] == "downloaded":
        if not entry["local_path"] or entry["local_sha_verified"] is not True:
            raise ManifestError("%s: downloaded 但 local_path/local_sha_verified 不完整" % where)
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
            raise ManifestError("%s: downloaded 但 size_bytes 非正整数" % where)
    return entry


def check_revision_consistency(manifest, json_entries):
    """§6.3-1：models.yaml 与 commit_hashes/*.json 的 revision 必须逐条一致。

    ``manifest``：models.yaml 解析结果（dict）。
    ``json_entries``：{key: dict} 形式的逐模型锁文件内容。
    任一 key 的 revision 不一致即报错；json 中出现 yaml 未登记的 key 亦报错。
    """
    by_key = {m["key"]: m for m in manifest["models"]}
    for key, js in json_entries.items():
        if key not in by_key:
            raise ManifestError("json key=%s 未在 models.yaml 中登记" % key)
        # deepseek-v3-api.json 依 §4.2 **无** revision 字段（改用 api_model_id 等），取 None 参与比对，
        # 与 yaml 侧的 null 视为一致；若 yaml 侧有 sha 而 json 侧缺字段，则判为不一致。
        js_rev = js.get("revision", None)
        if js_rev != by_key[key]["revision"]:
            raise ManifestError(
                "key=%s: revision 不一致 yaml=%r json=%r"
                % (key, by_key[key]["revision"], js_rev)
            )
        if js.get("repo_id", None) != by_key[key]["repo_id"]:
            raise ManifestError(
                "key=%s: repo_id 不一致 yaml=%r json=%r"
                % (key, by_key[key]["repo_id"], js.get("repo_id", None))
            )
    return True


# ---------------------------------------------------------------- fixture 数据

GOOD_ENTRY = {
    "key": "demo-model",
    "role": ["detector_b3"],
    "repo_id": "org/demo-model",
    "revision": "a" * 40,
    "resolution_status": RESOLVED,
    "local_status": "hash_only",
    "local_path": None,
    "local_sha_verified": None,
    "size_bytes": None,
    "license": "apache-2.0",
    "gated": False,
    "planned_load": "4bit-nf4",
    "planned_device": "cuda",
    "fetch_cmd": "HF_ENDPOINT=https://hf-mirror.com python -c \"...revision='%s'...\"" % ("a" * 40),
    "notes": "fixture",
}


def _fixture_json(revision="a" * 40):
    js = dict(GOOD_ENTRY)
    js["revision"] = revision
    return js


# ---------------------------------------------------------------- 1. 一致性校验

def test_consistency_passes_on_matching_fixture():
    manifest = {"models": [dict(GOOD_ENTRY)]}
    assert check_revision_consistency(manifest, {"demo-model": _fixture_json()}) is True


def test_consistency_detects_revision_mismatch():
    manifest = {"models": [dict(GOOD_ENTRY)]}
    with pytest.raises(ManifestError) as ei:
        check_revision_consistency(manifest, {"demo-model": _fixture_json("b" * 40)})
    assert "revision 不一致" in str(ei.value)


def test_consistency_detects_repo_id_mismatch():
    manifest = {"models": [dict(GOOD_ENTRY)]}
    js = _fixture_json()
    js["repo_id"] = "other/repo"
    with pytest.raises(ManifestError):
        check_revision_consistency(manifest, {"demo-model": js})


def test_consistency_detects_unregistered_json_key():
    manifest = {"models": [dict(GOOD_ENTRY)]}
    with pytest.raises(ManifestError):
        check_revision_consistency(manifest, {"ghost": _fixture_json()})


def test_consistency_tolerates_json_without_revision_only_when_yaml_is_null():
    """API 类条目的 json 依 §4.2 无 revision 字段：yaml 侧亦为 null 时视为一致。"""
    api_entry = dict(GOOD_ENTRY)
    api_entry.update(key="api-demo", repo_id=None, revision=None)
    js = {"key": "api-demo", "repo_id": None}
    assert check_revision_consistency({"models": [api_entry]}, {"api-demo": js}) is True

    with pytest.raises(ManifestError):
        check_revision_consistency({"models": [dict(GOOD_ENTRY)]}, {"demo-model": {"repo_id": "org/demo-model"}})


# ---------------------------------------------------------------- 2. 格式校验

@pytest.mark.parametrize("rev", ["a" * 40, "0123456789abcdef" * 2 + "01234567", "f" * 40])
def test_revision_accepts_40_hex(rev):
    assert validate_revision(rev) == rev


@pytest.mark.parametrize(
    "rev",
    [
        "a" * 39,
        "a" * 41,
        "A" * 40,          # 大写不接受：登记值一律小写
        "g" * 40,          # 非十六进制字符
        "",
        None,
        "19b7d6701bea1f6d9f03ec7bcaeeea390dff43b",  # 39 位（少一位）
    ],
)
def test_revision_rejects_bad_shapes(rev):
    with pytest.raises(ManifestError):
        validate_revision(rev, "fixture")


def test_resolved_entry_requires_40_hex_revision():
    e = dict(GOOD_ENTRY)
    e["revision"] = "zz"
    with pytest.raises(ManifestError):
        check_entry(e)


def test_deepseek_api_entry_allows_null_revision_but_others_do_not():
    e = dict(GOOD_ENTRY)
    e["key"] = "deepseek-v3-api"
    e["repo_id"] = None
    e["revision"] = None
    assert check_entry(e)["key"] == "deepseek-v3-api"

    e2 = dict(GOOD_ENTRY)
    e2["repo_id"] = None
    e2["revision"] = None
    with pytest.raises(ManifestError):
        check_entry(e2)


def test_downloaded_entry_requires_path_and_verification():
    e = dict(GOOD_ENTRY)
    e.update(local_status="downloaded", local_path="models/hf_cache/x", local_sha_verified=True,
             size_bytes=123)
    assert check_entry(e)["local_sha_verified"] is True

    for broken in (
        {"local_path": None},
        {"local_sha_verified": False},
        {"size_bytes": None},
        {"size_bytes": 0},
    ):
        bad = dict(e)
        bad.update(broken)
        with pytest.raises(ManifestError):
            check_entry(bad)


def test_pending_entry_must_not_claim_landing():
    e = dict(GOOD_ENTRY)
    e.update(resolution_status=PENDING, repo_id=None, revision=None)
    assert check_entry(e)["resolution_status"] == PENDING

    e2 = dict(e)
    e2["local_status"] = "downloaded"
    e2["local_path"] = "models/hf_cache/x"
    e2["local_sha_verified"] = True
    e2["size_bytes"] = 1
    with pytest.raises(ManifestError):
        check_entry(e2)

    e3 = dict(e)
    e3["repo_id"] = "org/guessed"
    with pytest.raises(ManifestError):
        check_entry(e3)


# ---------------------------------------------------------------- 3. 词表

@pytest.mark.parametrize("value", sorted(LOAD_VOCAB))
def test_planned_load_vocab_accepted(value):
    assert validate_planned_load(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "4bit",
        "int8",
        "nf4",
        "4bit-nf4-offload",
        "flash_attention_2",
        "bf16,4bit-nf4",
        "",
        None,
    ],
)
def test_planned_load_outside_vocab_rejected(value):
    with pytest.raises(ManifestError):
        validate_planned_load(value, "fixture")


def test_forbidden_attn_value_rejected():
    with pytest.raises(ManifestError):
        validate_planned_load("flash_attention_2", "fixture")


# ---------------------------------------------------------------- 4. token 检测

def test_fetch_cmd_placeholder_token_is_accepted():
    cmd = "HF_TOKEN=$HF_TOKEN python -c \"snapshot_download(repo_id='o/r')\""
    assert validate_fetch_cmd(cmd) == cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "curl -H 'Authorization: Bearer hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345' x",
        "snapshot_download(token='hf_0123456789abcdefghijklmn')",
    ],
)
def test_fetch_cmd_with_literal_token_rejected(cmd):
    with pytest.raises(ManifestError) as ei:
        validate_fetch_cmd(cmd, "fixture")
    assert "疑似 token" in str(ei.value)


def test_fetch_cmd_empty_rejected():
    with pytest.raises(ManifestError):
        validate_fetch_cmd("   ", "fixture")


# ---------------------------------------------------------------- 5. 真实交付物

def _load_manifest():
    with open(MODELS_YAML, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json_entries():
    out = {}
    for name in sorted(os.listdir(HASH_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(HASH_DIR, name), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if name == "PENDING_W2.json":
            assert data.get("pending"), "PENDING_W2.json 必须含 pending 段"
            continue
        out[data["key"]] = data
    return out


def test_real_manifest_schema_and_keys():
    m = _load_manifest()
    assert m["schema"] == "codeobf-pi/models/v1"
    assert m["generated_by"] == "W2.2"
    keys = [e["key"] for e in m["models"]]
    assert keys == list(REQUIRED_KEYS), "模型 key 集合或顺序不符 §3.1 要求"
    for entry in m["models"]:
        check_entry(entry)


def test_real_manifest_json_revision_consistency():
    assert check_revision_consistency(_load_manifest(), _load_json_entries()) is True


def test_real_manifest_no_flash_attention_anywhere():
    text = open(MODELS_YAML, "r", encoding="utf-8").read()
    assert "flash_attention_2" not in text
    for entry in _load_manifest()["models"]:
        assert "flash_attention_2" not in str(entry["planned_load"])


def test_real_manifest_has_no_token_literal():
    text = open(MODELS_YAML, "r", encoding="utf-8").read()
    assert not TOKEN_RE.search(text), "models.yaml 出现疑似 token 明文"
    for name in sorted(os.listdir(HASH_DIR)):
        if name.endswith((".json", ".md")):
            body = open(os.path.join(HASH_DIR, name), "r", encoding="utf-8").read()
            assert not TOKEN_RE.search(body), "%s 出现疑似 token 明文" % name


def test_index_md_has_one_row_per_model():
    text = open(INDEX_MD, "r", encoding="utf-8").read()
    for key in REQUIRED_KEYS:
        assert "`%s`" % key in text, "INDEX.md 缺 key=%s 的行" % key


def test_real_manifest_fetch_cmd_has_no_placeholder():
    """r1：回填后 fetch_cmd 必须是可执行命令，不得含 `<TBD>` 类占位符（§14.2 判据 5 的缺口）。"""
    for entry in _load_manifest()["models"]:
        cmd = str(entry["fetch_cmd"])
        for bad in ("<TBD>", "TBD_40hex", "待裁定", "待 W2 裁定"):
            assert bad not in cmd, "key=%s: fetch_cmd 仍含占位符 %r" % (entry["key"], bad)
        if entry["planned_load"] == "api":
            # API 类条目（qwen2.5-72b、deepseek-v3-api）不下载权重 ⇒ fetch_cmd 不得是可执行的下载命令。
            # 判据取「不含 python -c 调用」而非「不含 snapshot_download 字样」——禁止句本身会提到该函数名。
            assert "python -c" not in cmd, "key=%s: API 条目不得给出可执行下载命令" % entry["key"]
            assert "HF_ENDPOINT" not in cmd, "key=%s: API 条目不得含下载端点" % entry["key"]
            if entry["key"] == "qwen2.5-72b":
                assert "禁止" in cmd, "72B 的 fetch_cmd 须写明禁止下载权重"
            continue
        if entry["fetch_source"] == "modelscope":
            # D-07 改道后，gated 两项的 fetch_cmd 指向归档下载器（logs/env/ 下的 .log 文件，可直接执行）；
            # 其复现锚点是「魔搭 repo_id + 文件级 sha256」，而非 HF_ENDPOINT。
            # 【2026-10-22 泛化（W6.12，经用户授权代做、待 W0 追认）】归档下载器名接受两类：
            # A 批的 `01_04_fetch_models_ms_downloader_*` 与外部基线 B6 专用的
            # `01_05_fetch_external_baseline_ms_downloader_*`（二者均为 scripts/env/ 下可复用下载器）。
            # 只放宽**下载器文件名**，不放宽魔搭 repo_id／落地目录／无 token 三项硬判据。
            assert ("01_04_fetch_models_ms_downloader_" in cmd
                    or "01_05_fetch_external_baseline_ms_downloader_" in cmd), (
                "key=%s: 魔搭条目须指向归档下载器" % entry["key"]
            )
            assert entry["modelscope_repo_id"] in cmd, "key=%s: fetch_cmd 缺魔搭 repo_id" % entry["key"]
            assert entry["local_path"] in cmd, "key=%s: fetch_cmd 缺落地目录" % entry["key"]
            assert "HF_TOKEN" not in cmd, "D-07 后魔搭条目不应再出现 HF_TOKEN"
            continue
        assert "HF_ENDPOINT=https://hf-mirror.com" in cmd, "key=%s: fetch_cmd 缺 HF_ENDPOINT" % entry["key"]
        assert "revision='%s'" % entry["revision"] in cmd, (
            "key=%s: fetch_cmd 内的 revision 与登记值不一致" % entry["key"]
        )


def test_real_downstream_and_judge_backup_are_instruct_variants():
    """r1：4 个下游 SLM 与备用裁判一律 Instruct 变体（W2 §14.4 未决项 1；不得混用 base）。"""
    targets = ("qwen2.5-coder-1.5b", "qwen2.5-coder-3b", "codellama-7b", "mistral-7b", "qwen2.5-72b")
    by_key = {e["key"]: e for e in _load_manifest()["models"]}
    for key in targets:
        entry = by_key[key]
        assert entry["resolution_status"] == RESOLVED, "key=%s 未回填" % key
        assert "Instruct" in entry["repo_id"], "key=%s: repo_id=%r 不是 Instruct 变体" % (key, entry["repo_id"])
        validate_revision(entry["revision"], key)
    # 一致性硬约束：4 个下游一律同一变体类别
    assert all("Instruct" in by_key[k]["repo_id"] for k in targets[:4])


def test_real_a_batch_downloaded_entries_declare_verified_sha():
    """§6.1 判据 4：A 批中已落地者必须 local_sha_verified=true，且路径与「来源」相符。

    r2 起有两条来源（`D-07`）：
      * `hf-mirror`（hf_hub 缓存布局）── 落地 snapshot 目录名 == 登记 revision；
      * `modelscope`（`D-07` 改道）── 目录名为 `modelscope--<org>--<name>`；
        完整性锚点是**文件级 sha256**（魔搭 revision 为 `master`，会漂移），不是目录名。
    """
    m = _load_manifest()
    for entry in m["models"]:
        if entry["local_status"] != "downloaded":
            continue
        assert entry["local_sha_verified"] is True, "key=%s" % entry["key"]
        assert isinstance(entry["size_bytes"], int) and entry["size_bytes"] > 0
        src = entry["fetch_source"]
        if src == "hf-mirror":
            assert entry["local_path"].rstrip("/").endswith(entry["revision"]), (
                "key=%s: hf-mirror 落地路径尾段 != revision" % entry["key"]
            )
        elif src == "modelscope":
            # 【2026-10-22 泛化（W6.12，经用户授权代做、待 W0 追认）】路径 ↔ repo 对应判据：
            # 目录名必须是 repo 派生的 `modelscope--<org>--<name>`；缓存根**默认项目内**
            # `models/hf_cache/`，但允许**已声明的项目外根**（B 批下游 SLM 与外部基线 B6 均在
            # `D:/CodeObfPI_hf_cache/`，双盘位布局；上行 hf-mirror 分支本就对项目外路径用
            # `endswith` 宽容判据，此处同制）。只放宽**根前缀**，不放宽 repo 派生后缀与
            # `integrity_anchor == file_sha256` 两项硬判据。
            suffix = "modelscope--" + entry["modelscope_repo_id"].replace("/", "--")
            lp = entry["local_path"].rstrip("/")
            assert (lp == "models/hf_cache/" + suffix
                    or lp.endswith("/" + suffix) or lp.endswith("\\" + suffix)), (
                "key=%s: 魔搭落地路径 %r 与 repo_id 不符（须为 models/hf_cache/%s 或以分隔符 + %r 结尾）"
                % (entry["key"], entry["local_path"], suffix, suffix)
            )
            assert entry["integrity_anchor"] == "file_sha256", (
                "key=%s: 魔搭来源的完整性锚点必须为 file_sha256（D-07 (c)）" % entry["key"]
            )
        else:
            raise AssertionError("key=%s: 未知 fetch_source=%r" % (entry["key"], src))


def test_real_modelscope_entries_have_three_way_provenance():
    """`D-07` (b)：魔搭来源的 json 须含三方 provenance（魔搭 sha256／HF revision／加载方式）。"""
    for key in ("llamaguard3-1b", "llamaguard3-8b"):
        with open(os.path.join(HASH_DIR, "%s.json" % key), "r", encoding="utf-8") as fh:
            js = json.load(fh)
        assert js["fetch_source"] == "modelscope", key
        assert js["integrity_anchor"] == "file_sha256", key

        prov = js["provenance"]
        ms = prov["1_modelscope"]
        assert ms["repo_id"].startswith("LLM-Research/"), "魔搭 repo_id 须为 Meta 官方引入组织"
        assert ms["revision_ref"] == "master" and ms["files"], key
        for f in ms["files"]:
            assert re.match(r"^[0-9a-f]{64}$", f["sha256"]), "key=%s path=%s: 文件 sha256 非 64 hex" % (key, f["path"])
            assert isinstance(f["size_bytes"], int) and f["size_bytes"] >= 0
            assert f["sha256_matches_listing"] is True, "key=%s path=%s" % (key, f["path"])
            assert not f["path"].startswith("original/"), "original/* 不应进入落地清单"
        assert sum(f["size_bytes"] for f in ms["files"]) == ms["total_bytes"], (
            "key=%s: provenance 文件字节和与 total_bytes 不一致" % key
        )

        hf = prov["2_hf_revision"]
        assert hf["repo_id"] == js["repo_id"] and hf["revision"] == js["revision"], (
            "key=%s: 三方 provenance 的 HF revision 必须与登记 revision 一致" % key
        )

        load = prov["3_load"]
        assert load["planned_load"] == js["planned_load"] and load["attn_impl"] == "eager", key


def test_real_downloaded_entries_have_provenance_file_on_disk():
    """历史校验：落地目录内须存在 `PROVENANCE.json`（魔搭来源），且其 total_bytes 与登记一致。"""
    m = _load_manifest()
    for entry in m["models"]:
        if entry["local_status"] != "downloaded" or entry["fetch_source"] != "modelscope":
            continue
        p = os.path.join(ROOT, entry["local_path"].replace("/", os.sep), "PROVENANCE.json")
        assert os.path.exists(p), "缺 PROVENANCE.json：%s" % p
        with open(p, "r", encoding="utf-8") as fh:
            prov = json.load(fh)
        assert prov["repo_id"] == entry["modelscope_repo_id"]
        assert prov["total_bytes"] + os.path.getsize(p) == entry["size_bytes"], (
            "key=%s: 测量根口径应为「模型文件 + PROVENANCE.json」" % entry["key"]
        )

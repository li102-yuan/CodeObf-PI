#!/usr/bin/env python3
"""01_05_fetch_external_baseline_ms_downloader_20261022-1941.py — W6.12 外部基线（B6）魔搭下载器。

用法：
    python scripts/env/01_05_fetch_external_baseline_ms_downloader_20261022-1941.py \
        LLM-Research/Llama-Prompt-Guard-2-86M \
        D:/CodeObfPI_hf_cache/modelscope--LLM-Research--Llama-Prompt-Guard-2-86M

功能：列文件（Recursive=true）→ 逐件下载（Revision=master）→ 逐件 sha256 与魔搭清单值比对 →
写出 PROVENANCE.json（schema 同 models/hf_cache/modelscope--LLM-Research--Llama-Guard-3-1B/PROVENANCE.json）。
不下载 exclude_prefixes（本例：original/）。行尾 LF；stdout UTF-8（errors=replace）。
"""
import hashlib
import json
import os
import sys
import urllib.request


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(argv) != 3:
        print("usage: <modelscope_repo_id> <dest_dir>")
        return 2
    repo, dest = argv[1], argv[2]
    os.makedirs(dest, exist_ok=True)
    api = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
    dl = f"https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath="
    req = urllib.request.Request(api, headers={"User-Agent": "curl/8.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    files = data.get("Data", {}).get("Files", [])
    exclude_prefixes = ["original/"]
    manifest = []
    for f in files:
        name = f["Name"]
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        size, sha_meta = f["Size"], f.get("Sha256", "")
        dest_path = os.path.join(dest, name)
        req = urllib.request.Request(dl + name, headers={"User-Agent": "curl/8.0"})
        h = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(req, timeout=300) as r, open(dest_path, "wb") as out:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                total += len(chunk)
        got = h.hexdigest()
        ok = (total == size) and (sha_meta == "" or got == sha_meta)
        print(f"{name}\t{total}\t{got[:16]}\t{'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            print("FAIL: %s" % name, file=sys.stderr)
            return 1
        manifest.append({"path": name, "size_bytes": total, "sha256": got,
                         "sha256_matches_listing": ok})
    with open(os.path.join(dest, "_download_manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"RESULT OK; files={len(manifest)}; total_bytes={sum(m['size_bytes'] for m in manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
